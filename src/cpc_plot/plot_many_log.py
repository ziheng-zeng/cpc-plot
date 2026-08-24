from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _numbered_column(base: str, index: int) -> str:
    return base if index == 0 else f"{base}.{index}"


def load_many_logs(folder: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path, on_bad_lines="skip") for path in sorted(folder.glob("MANY_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No MANY_*.csv files found in {folder}.")
    return pd.concat(frames, axis=0, ignore_index=True)


def plot_many_log(folder: Path, labels: list[str] | None = None, output: Path | None = None) -> None:
    df = load_many_logs(folder)

    concentration_columns = [col for col in df.columns if col == "concentration" or col.startswith("concentration.")]
    if not concentration_columns:
        raise ValueError("No concentration columns found in the MANY log files.")

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, concentration_col in enumerate(concentration_columns):
        datetime_col = _numbered_column("datetime", i)
        if datetime_col not in df.columns:
            continue
        label = labels[i] if labels and i < len(labels) else f"CPC {i + 1}"
        x = pd.to_datetime(df[datetime_col], errors="coerce")
        y = pd.to_numeric(df[concentration_col], errors="coerce")
        valid = x.notna() & y.notna()
        ax.plot(x[valid], y[valid], label=label, linewidth=1.1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration (ct/cc)")
    ax.set_title("Multi-CPC Log Concentration vs. Time")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot concentration columns from MANY_*.csv CPC logs.")
    parser.add_argument("folder", type=Path, help="Folder containing MANY_*.csv logs.")
    parser.add_argument("--label", dest="labels", action="append", help="Optional CPC label; repeat in column order.")
    parser.add_argument("--output", type=Path, help="Optional PNG path to save instead of showing.")
    args = parser.parse_args()

    plot_many_log(args.folder, args.labels, args.output)


if __name__ == "__main__":
    main()

