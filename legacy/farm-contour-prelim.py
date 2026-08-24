import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm


def load_all_for_sn(data_dir: str, sn: str, tz: str | None = None) -> pd.DataFrame:
    print(f"\n--- Loading files for SN {sn} ---")

    pattern = os.path.join(data_dir, f"5T_{sn}_Raw_*")
    files = sorted(sum([glob.glob(pattern + ext) for ext in ["", ".*"]], []))

    print(f"  Found {len(files)} files")

    if not files:
        raise FileNotFoundError(f"No files found for {sn} in {data_dir}")

    parts = []
    for i, fp in enumerate(files, start=1):
        if i % 100 == 0:
            print(f"    Processed {i}/{len(files)} files…")
        df = pd.read_csv(fp, header=None, sep=None, engine="python", comment="#")
        if df.shape[1] < 3:
            raise ValueError(f"{os.path.basename(fp)} has fewer than 3 columns.")
        t = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        parts.append(pd.DataFrame({"time": t, "counts_cm3": y}))

    out = pd.concat(parts, ignore_index=True)
    print(f"  Raw entries: {len(out)}")

    out = (
        out.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset=["time"])
    )

    print(f"  After cleaning: {len(out)} time points")

    if tz is not None:
        print(f"  Localizing/Converting timezone to {tz}")
        if out["time"].dt.tz is None:
            out["time"] = out["time"].dt.tz_localize(tz)
        else:
            out["time"] = out["time"].dt.tz_convert(tz)

    out["SN"] = sn

    print(f"--- Finished SN {sn} ---")

    return out


def build_wide_dataframe(
        data_dir: str,
        sn_to_channel: dict,
        tz: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
) -> pd.DataFrame:
    print("\n=== Building wide dataframe (all channels) ===")

    frames = []
    for sn, chn in sn_to_channel.items():
        print(f"Loading SN {sn} as channel {chn}")
        df = load_all_for_sn(data_dir, sn, tz=tz)

        # time filtering
        if start_time is not None:
            df = df[df["time"] >= pd.to_datetime(start_time)]
        if end_time is not None:
            df = df[df["time"] <= pd.to_datetime(end_time)]

        df = df.rename(columns={"counts_cm3": chn})
        df = df.set_index("time")[[chn]]
        frames.append(df)

    print("Concatenating channels...")
    wide = pd.concat(frames, axis=1).sort_index()

    print("Replacing sentinel 999999 with NaN…")
    wide = wide.replace(999999, np.nan)

    print(f"Wide dataframe shape: {wide.shape}")

    return wide


def compute_dNdlnDp(
        wide_df: pd.DataFrame,
        channel_d50_nm: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute approximate dN/dlnDp from cumulative CPC channels.

    wide_df columns: CH1..CH5 (cumulative counts, cm^-3)
    channel_d50_nm: mapping channel -> D50 (nm)
    Returns:
        dNdlnDp_df  (time x 4 bins)
        dp_mid_nm   (geometric midpoints of each bin)
        dp_edges_nm (the 5 D50s, used as bin edges for plotting)
    """
    print("\n=== Computing dN/dlnDp ===")

    # Use only channels that have D50s
    chans = [ch for ch in wide_df.columns if ch in channel_d50_nm]
    print(f"Channels available: {chans}")

    if len(chans) < 5:
        raise ValueError("Need 5 CPC channels with D50 values to form 4 bins.")

    # Sort channels by D50 (ascending)
    chans_sorted = sorted(chans, key=lambda ch: channel_d50_nm[ch])
    print(f"Channels sorted by D50: {chans_sorted}")

    dp_edges_nm = np.array([channel_d50_nm[ch] for ch in chans_sorted])
    print(f"D50 array (bin edges): {dp_edges_nm}")

    # Cumulative counts matrix
    cum = wide_df[chans_sorted].to_numpy()   # shape (T, 5)
    print(f"Cumulative matrix shape: {cum.shape}")

    # ΔN between adjacent cumulative channels
    dN = cum[:, :-1] - cum[:, 1:]            # shape (T, 4)
    print("Computed dN (differential counts).")
    print(f"dN shape: {dN.shape}")

    # ΔlnDp (natural log)
    ln_edges = np.log(dp_edges_nm)
    dlnDp = np.diff(ln_edges)                # length 4
    print(f"ΔlnDp per bin: {dlnDp}")

    # dN/dlnDp
    dNdlnDp = dN / dlnDp                     # shape (T, 4)
    print("Computed dN/dlnDp.")

    # Mask non-positive values
    neg_count = np.sum(dNdlnDp <= 0)
    print(f"Found {neg_count} non-positive bins → setting to NaN.")
    dNdlnDp[dNdlnDp <= 0] = np.nan

    # Geometric midpoints (for reference / 1D plots)
    dp_mid_nm = np.sqrt(dp_edges_nm[:-1] * dp_edges_nm[1:])
    print(f"Bin midpoints: {dp_mid_nm}")

    # Column labels
    bin_labels = [
        f"{dp_edges_nm[i]:.2f}-{dp_edges_nm[i + 1]:.2f} nm"
        for i in range(len(dp_edges_nm) - 1)
    ]

    dNdlnDp_df = pd.DataFrame(
        dNdlnDp,
        index=wide_df.index,
        columns=bin_labels,
    )

    print(f"dN/dlnDp dataframe shape: {dNdlnDp_df.shape}")
    print("=== Finished computing dN/dlnDp ===")

    return dNdlnDp_df, dp_mid_nm, dp_edges_nm


def plot_cpc_farm_contour(
        dNdlogDp_df: pd.DataFrame,
        dp_edges_nm: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str = "CPC FARM dN/dlnDp",
):
    print("\n=== Plotting contour ===")
    print(f"Dataframe shape: {dNdlogDp_df.shape}")

    t_index = dNdlogDp_df.index
    t_num = mdates.date2num(t_index.to_pydatetime())

    dt = np.diff(t_num)
    if len(dt) == 0:
        raise ValueError("Not enough time points for contour plot.")
    t_edges = np.empty(len(t_num) + 1)
    t_edges[1:-1] = (t_num[1:] + t_num[:-1]) / 2
    t_edges[0] = t_num[0] - dt[0] / 2
    t_edges[-1] = t_num[-1] + dt[-1] / 2

    Z = dNdlogDp_df.to_numpy()

    print("Starting pcolormesh…")

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(
        t_edges,
        dp_edges_nm,
        Z.T,
        norm=LogNorm(vmin=vmin, vmax=vmax) if vmin is not None else LogNorm(),
        shading="auto",
        cmap="turbo",  # if you want turbo here too
    )

    ax.set_yscale("log")
    ax.set_ylabel("Diameter (nm)")
    ax.set_xlabel("Local Time (US Eastern)")
    ax.set_title(title)

    fig.colorbar(pcm, ax=ax, label="dN/dlnDp (cm$^{-3}$)")  # <— updated label

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    print("=== Finished plotting ===")



if __name__ == "__main__":
    # === USER CONFIG ===
    data_dir = r"C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/CPC FARM/5T_GUI_2022_11_01/data/raw"

    sn_to_channel = {
        "SN213": "CH1",
        "SN212": "CH2",
        "SN210": "CH3",
        "SN223": "CH4",
        "SN208": "CH5",
    }

    channel_d50_nm = {
        "CH1": 1.6,
        "CH2": 1.9,
        "CH3": 2.3,
        "CH4": 2.7,
        "CH5": 3.3,
    }

    start_time = "2025-09-19 00:00"
    end_time = "2025-09-23 23:59"

    tz = None

    print("\n>>> STARTING CPC FARM PROCESSING <<<")

    wide_df = build_wide_dataframe(
        data_dir=data_dir,
        sn_to_channel=sn_to_channel,
        tz=tz,
        start_time=start_time,
        end_time=end_time,
    )

    dNdlnDp_df, dp_mid_nm, dp_edges_nm = compute_dNdlnDp(
        wide_df,
        channel_d50_nm=channel_d50_nm,
    )

    plot_cpc_farm_contour(
        dNdlnDp_df,
        dp_edges_nm=dp_edges_nm,
        title="CPC FARM dN/dlnDp — Sept 2025",
    )

    print("\n>>> ALL DONE <<<\n")
