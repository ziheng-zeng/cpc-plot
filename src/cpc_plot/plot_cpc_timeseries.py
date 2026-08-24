from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def load_cpc_files(folder: Path, cpc_id: str) -> pd.DataFrame:
    """Load every CSV in a folder whose filename contains the CPC id."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(folder.glob("*.csv")):
        if cpc_id not in csv_path.name:
            continue
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        if "time" not in df.columns or "concentration" not in df.columns:
            raise ValueError(f"{csv_path} must include time and concentration columns.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
        frames.append(df[["time", "concentration"]].dropna())

    if not frames:
        raise FileNotFoundError(f"No CSV files containing '{cpc_id}' found in {folder}.")

    return pd.concat(frames, ignore_index=True).sort_values("time")


def plot_timeseries(folder: Path, cpc_ids: list[str], output: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for cpc_id in cpc_ids:
        data = load_cpc_files(folder, cpc_id)
        ax.plot(data["time"], data["concentration"], label=f"CPC {cpc_id}", linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration (ct/cc)")
    ax.set_title("CPC Concentration vs. Time")
    ax.legend()
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(y_formatter)

    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPC concentration CSV files.")
    parser.add_argument("folder", type=Path, help="Folder containing CPC CSV files.")
    parser.add_argument(
        "--cpc",
        dest="cpc_ids",
        action="append",
        required=True,
        help="CPC id to include, such as 3025 or 3772. Repeat for multiple CPCs.",
    )
    parser.add_argument("--output", type=Path, help="Optional PNG path to save instead of showing.")
    args = parser.parse_args()

    plot_timeseries(args.folder, args.cpc_ids, args.output)


if __name__ == "__main__":
    main()

