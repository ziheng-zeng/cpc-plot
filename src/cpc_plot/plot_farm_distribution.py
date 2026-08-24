from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def load_raw_channel_files(data_dir: Path, serial_number: str) -> pd.DataFrame:
    pattern = str(data_dir / f"5T_{serial_number}_Raw_*")
    files = sorted(glob.glob(pattern) + glob.glob(pattern + ".*"))
    if not files:
        raise FileNotFoundError(f"No raw files found for {serial_number} in {data_dir}.")

    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, header=None, sep=None, engine="python", comment="#")
        if df.shape[1] < 3:
            raise ValueError(f"{path} has fewer than three columns.")
        frames.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(df.iloc[:, 0], errors="coerce"),
                    "counts_cm3": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
                }
            )
        )

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset=["time"])
    )


def build_wide_dataframe(data_dir: Path, serial_to_channel: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for serial_number, channel in serial_to_channel.items():
        channel_df = load_raw_channel_files(data_dir, serial_number)
        frames.append(channel_df.rename(columns={"counts_cm3": channel}).set_index("time")[[channel]])
    return pd.concat(frames, axis=1).sort_index().replace(999999, np.nan)


def compute_dndln_dp(
    wide_df: pd.DataFrame,
    channel_d50_nm: dict[str, float],
) -> tuple[pd.DataFrame, np.ndarray]:
    channels = sorted(channel_d50_nm, key=channel_d50_nm.get)
    if len(channels) < 5:
        raise ValueError("At least five cumulative CPC channels are needed to form size bins.")

    d50_edges = np.array([channel_d50_nm[channel] for channel in channels])
    cumulative = wide_df[channels].to_numpy()
    delta_n = cumulative[:, :-1] - cumulative[:, 1:]
    dndln_dp = delta_n / np.diff(np.log(d50_edges))
    dndln_dp[dndln_dp <= 0] = np.nan

    labels = [f"{d50_edges[i]:.2f}-{d50_edges[i + 1]:.2f} nm" for i in range(len(d50_edges) - 1)]
    return pd.DataFrame(dndln_dp, index=wide_df.index, columns=labels), d50_edges


def plot_contour(dndln_dp: pd.DataFrame, d50_edges: np.ndarray, output: Path | None = None) -> None:
    t_num = mdates.date2num(dndln_dp.index.to_pydatetime())
    dt = np.diff(t_num)
    if len(dt) == 0:
        raise ValueError("Not enough time points for contour plot.")

    t_edges = np.empty(len(t_num) + 1)
    t_edges[1:-1] = (t_num[1:] + t_num[:-1]) / 2
    t_edges[0] = t_num[0] - dt[0] / 2
    t_edges[-1] = t_num[-1] + dt[-1] / 2

    fig, ax = plt.subplots(figsize=(12, 4.8))
    mesh = ax.pcolormesh(
        t_edges,
        d50_edges,
        dndln_dp.to_numpy().T,
        shading="auto",
        norm=LogNorm(),
        cmap="turbo",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Diameter (nm)")
    ax.set_title("CPC FARM dN/dlnDp")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.colorbar(mesh, ax=ax, label="dN/dlnDp (cm^-3)")
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()


def _parse_mapping(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        key, mapped_value = value.split("=", 1)
        mapping[key] = mapped_value
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CPC FARM dN/dlnDp contour plot.")
    parser.add_argument("data_dir", type=Path, help="Folder with 5T_SN*_Raw_* files.")
    parser.add_argument("--serial-channel", action="append", required=True, help="Mapping like SN213=CH1.")
    parser.add_argument("--d50", action="append", required=True, help="Mapping like CH1=1.6.")
    parser.add_argument("--output", type=Path, help="Optional PNG path to save instead of showing.")
    args = parser.parse_args()

    serial_to_channel = _parse_mapping(args.serial_channel)
    d50_by_channel = {channel: float(d50) for channel, d50 in _parse_mapping(args.d50).items()}
    wide_df = build_wide_dataframe(args.data_dir, serial_to_channel)
    dndln_dp, d50_edges = compute_dndln_dp(wide_df, d50_by_channel)
    plot_contour(dndln_dp, d50_edges, args.output)


if __name__ == "__main__":
    main()

