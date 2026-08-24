import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.size": 14,          # overall font size
    "axes.titlesize": 16,     # title
    "axes.labelsize": 15,     # x/y labels
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16
})


def load_files_for_sn_two_days(
    data_dir: str,
    sn: str,
    date_str: str,
    tz: str | None = None,
) -> pd.DataFrame:
    """
    Load ALL raw files for SN for (date-1) and (date),
    because some early/late points fall in adjacent files.

    File pattern: 5T_SN223_Raw_YYYYMMDD_HHMMSS*
    Uses column 0 as time, column 2 as counts (cm^-3).
    """
    target_date = pd.to_datetime(date_str)  # e.g. "20250927"
    prev_day_str = (target_date - pd.Timedelta(days=1)).strftime("%Y%m%d")

    patterns = [
        os.path.join(data_dir, f"5T_{sn}_Raw_{prev_day_str}_*"),
        os.path.join(data_dir, f"5T_{sn}_Raw_{date_str}_*"),
    ]

    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))

    files = sorted(files)
    print(f"\nSN {sn}: loading {len(files)} files (prev day + this day)")
    if not files:
        raise FileNotFoundError(
            f"No files found for {sn} for {prev_day_str} or {date_str}"
        )

    parts = []
    for fp in files:
        df = pd.read_csv(
            fp,
            header=None,
            sep=",",
            engine="python",
            usecols=[0, 2],          # col 0 = time, col 2 = counts
            skipinitialspace=True,
            comment="#",
        )
        t = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.round("1S")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        parts.append(pd.DataFrame({"time": t, "counts_cm3": y}))

    out = pd.concat(parts, ignore_index=True)
    out = (
        out.dropna(subset=["time"])
           .sort_values("time")
           .drop_duplicates(subset=["time"])
    )

    if tz is not None:
        if out["time"].dt.tz is None:
            out["time"] = out["time"].dt.tz_localize(tz)
        else:
            out["time"] = out["time"].dt.tz_convert(tz)

    print(f"  Total rows after merge: {len(out)}")
    return out


def build_wide_two_days(
    data_dir: str,
    sn_to_channel: dict,
    date_str: str,
    tz: str | None = None,
) -> pd.DataFrame:
    """
    Load two days of data for each SN and merge to a wide dataframe:
    - index = time (1 s grid)
    - columns = CH1..CH5 counts (cm^-3)
    DOES NOT touch 999999 (so raw plot shows overrange).
    """
    print("\n=== Building wide dataframe (two days per channel) ===")
    wide = None
    for sn, chn in sn_to_channel.items():
        df = load_files_for_sn_two_days(data_dir, sn, date_str, tz=tz)
        df = df.rename(columns={"counts_cm3": chn})
        # Put on 1-second grid for this channel
        df = df.set_index("time").resample("1S").mean()
        if wide is None:
            wide = df
        else:
            wide = wide.join(df, how="inner")  # keep times where all channels have data

    print("Raw wide_df head (still has 999999 if present):")
    print(wide.head())
    print("NaN fraction per channel:")
    print(wide.isna().mean())
    print(f"Wide shape (two days span): {wide.shape}")

    return wide


def crop_to_single_day(wide_df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Crop wide_df to exactly [date 00:00, date+1 00:00)
    """
    day_start = pd.to_datetime(date_str)
    day_end = day_start + pd.Timedelta(days=1)

    # Handle tz-aware index
    if wide_df.index.tz is not None:
        day_start = day_start.tz_localize(wide_df.index.tz)
        day_end = day_end.tz_localize(wide_df.index.tz)

    cropped = wide_df.loc[(wide_df.index >= day_start) & (wide_df.index < day_end)]
    print(f"\nCropped to single day {date_str}: shape {cropped.shape}")
    return cropped


def compute_deltaN_and_dNdlnDp(
    wide_df_filtered: pd.DataFrame,
    channel_d50_nm: dict,
):
    """
    From 5 cumulative channels in wide_df_filtered -> 4 ΔN bins and dN/dlnDp.
    wide_df_filtered should already have 999999 replaced by NaN.
    """
    print("\n=== Computing ΔN and dN/dlnDp ===")

    chans = [ch for ch in wide_df_filtered.columns if ch in channel_d50_nm]
    print("Channels in wide_df:", chans)
    if len(chans) != 5:
        raise ValueError("Need exactly 5 channels for 4 bins.")

    # Sort by D50 so bins are in correct order
    chans_sorted = sorted(chans, key=lambda ch: channel_d50_nm[ch])
    print("Channels sorted by D50:", chans_sorted)

    d50_array = np.array([channel_d50_nm[ch] for ch in chans_sorted])
    print("D50 array:", d50_array)

    cum = wide_df_filtered[chans_sorted].to_numpy()
    print("Cumulative matrix shape:", cum.shape)

    # ΔN = C_i - C_{i+1}
    dN = cum[:, :-1] - cum[:, 1:]
    print("ΔN matrix shape:", dN.shape)

    # ΔlnDp (natural log)
    ln_edges = np.log(d50_array)
    dlnDp = np.diff(ln_edges)
    print("ΔlnDp:", dlnDp)

    # dN/dlnDp
    dNdlnDp = dN / dlnDp
    neg_count = np.sum(dNdlnDp <= 0)
    print(f"Non-positive dN/dlnDp bins: {neg_count}")
    dNdlnDp[dNdlnDp <= 0] = np.nan   # 0 or negative -> blank

    dp_mid_nm = np.sqrt(d50_array[:-1] * d50_array[1:])
    print("Bin midpoints:", dp_mid_nm)

    bin_labels = [
        f"{d50_array[i]:.2f}-{d50_array[i+1]:.2f} nm"
        for i in range(len(d50_array) - 1)
    ]

    deltaN_df = pd.DataFrame(dN, index=wide_df_filtered.index, columns=bin_labels)
    dNdlnDp_df = pd.DataFrame(dNdlnDp, index=wide_df_filtered.index, columns=bin_labels)

    return deltaN_df, dNdlnDp_df, dp_mid_nm, d50_array


def _format_hour_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # every 2 hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


def plot_raw_counts(wide_df_raw: pd.DataFrame, date_str: str):
    """Raw cumulative counts (with 999999 still there)."""
    plt.figure(figsize=(12, 4))
    for ch in wide_df_raw.columns:
        plt.plot(wide_df_raw.index, wide_df_raw[ch], label=ch, linewidth=1)

    ax = plt.gca()
    _format_hour_axis(ax)

    plt.title(f"Raw cumulative counts (includes 999999) — {date_str}")
    plt.ylabel("Counts (cm$^{-3}$)")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_deltaN(deltaN_df: pd.DataFrame, date_str: str):
    plt.figure(figsize=(12, 4))
    for col in deltaN_df.columns:
        plt.plot(deltaN_df.index, deltaN_df[col], label=col, linewidth=1)

    ax = plt.gca()
    _format_hour_axis(ax)

    plt.title(f"ΔN per size bin (C$_i$ - C$_{{i+1}}$) — {date_str}")
    plt.ylabel("ΔN (cm$^{-3}$)")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dNdlnDp_timeseries(dNdlnDp_df: pd.DataFrame, date_str: str):
    plt.figure(figsize=(12, 4))
    for col in dNdlnDp_df.columns:
        plt.plot(dNdlnDp_df.index, dNdlnDp_df[col], label=col, linewidth=1)

    ax = plt.gca()
    _format_hour_axis(ax)

    plt.title(f"dN/dlnDp per size bin — {date_str}")
    plt.ylabel("dN/dlnDp (cm$^{-3}$)")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dNdlnDp_contour(dNdlnDp_df: pd.DataFrame, d50_array: np.ndarray, date_str: str):
    """
    Contour-style plot:
      x = time
      y = Dp (nm, log)
      color = dN/dlnDp
    """
    print("\n=== Plotting dN/dlnDp contour ===")

    # Time axis
    t_index = dNdlnDp_df.index
    t_num = mdates.date2num(t_index.to_pydatetime())

    dt = np.diff(t_num)
    if len(dt) == 0:
        raise ValueError("Not enough time points for contour plot.")
    t_edges = np.empty(len(t_num) + 1)
    t_edges[1:-1] = (t_num[1:] + t_num[:-1]) / 2
    t_edges[0] = t_num[0] - dt[0] / 2
    t_edges[-1] = t_num[-1] + dt[-1] / 2

    # Y edges are D50s (5 edges for 4 bins)
    dp_edges_nm = d50_array

    Z = dNdlnDp_df.to_numpy()  # shape (T, 4)

    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(
        t_edges,
        dp_edges_nm,
        Z.T,
        shading="auto",
        norm=LogNorm(vmin=1e3, vmax=1e5),
        cmap="turbo",
    )

    ax.set_yscale("log")
    ax.set_ylabel("Diameter (nm)")
    ax.set_xlabel("Local Time (US Eastern)")
    # no title if you want it clean for proposal

    _format_hour_axis(ax)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("dN/dlnDp (cm$^{-3}$)")

    plt.tight_layout()
    plt.show()

    print("=== Finished contour plot ===")


if __name__ == "__main__":
    # === USER CONFIG ===
    data_dir = r"C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/CPC FARM/5T_GUI_2022_11_01/data/raw"
    date_str = "20250927"   # the day you want to plot (00:00–24:00)

    # SN -> channel label
    sn_to_channel = {
        "SN213": "CH1",
        "SN212": "CH2",
        "SN210": "CH3",
        "SN223": "CH4",
        "SN208": "CH5",
    }

    # Calibrated D50s (EDIT to your real values)
    channel_d50_nm = {
        "CH1": 1.6,
        "CH2": 1.9,
        "CH3": 2.3,
        "CH4": 2.7,
        "CH5": 3.3,
    }

    tz = None  # or "US/Eastern"

    print("\n>>> CPC FARM SINGLE-DAY PIPELINE (dN/dlnDp) <<<")

    # 1) Load two days per channel, merge to wide df
    wide_df_raw_full = build_wide_two_days(
        data_dir=data_dir,
        sn_to_channel=sn_to_channel,
        date_str=date_str,
        tz=tz,
    )

    # 2) Crop to this single day (00:00–24:00)
    wide_df_raw = crop_to_single_day(wide_df_raw_full, date_str)

    # 3) Plot raw counts INCLUDING 999999
    plot_raw_counts(wide_df_raw, date_str)

    # 4) Replace 999999 -> NaN for processing
    wide_df = wide_df_raw.replace(999999, np.nan)

    # 5) Compute ΔN and dN/dlnDp (with ≤0 → NaN)
    deltaN_df, dNdlnDp_df, dp_mid_nm, d50_array = compute_deltaN_and_dNdlnDp(
        wide_df,
        channel_d50_nm=channel_d50_nm,
    )

    # 6) ΔN time series
    plot_deltaN(deltaN_df, date_str)

    # 7) dN/dlnDp time series
    plot_dNdlnDp_timeseries(dNdlnDp_df, date_str)

    # 8) dN/dlnDp contour (time vs Dp, color=dN/dlnDp)
    plot_dNdlnDp_contour(
        dNdlnDp_df,
        d50_array,
        date_str=date_str,
    )

    print("\n>>> DONE <<<")
