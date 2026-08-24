# CPC Plot

Python plotting scripts for condensation particle counter (CPC) data. This repo focuses on offline analysis: reading logged CPC CSV files, plotting concentration time series, comparing instruments, and building CPC FARM size-distribution contour plots.

## What is included

- `src/cpc_plot/plot_cpc_timeseries.py` plots one or more CPC CSV files with `time` and `concentration` columns.
- `src/cpc_plot/plot_many_log.py` plots daily `MANY_*.csv` logs created by the real-time GUI/data-acquisition program.
- `src/cpc_plot/plot_farm_distribution.py` builds approximate `dN/dlnDp` size-distribution contour plots from CPC FARM raw channel files.
- `examples/data/sample-run-2024-02-08/` contains small 3025 and 3772 CPC sample CSVs.
- `examples/plots/` contains example output plots.
- `legacy/` keeps the original archive scripts for reference.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Plot the sample CPC run

```powershell
python -m cpc_plot.plot_cpc_timeseries examples\data\sample-run-2024-02-08 --cpc 3025 --cpc 3772 --output examples\plots\sample_run_2024-02-08.png
```

This reads both sample CSVs and writes a comparison plot to `examples/plots/sample_run_2024-02-08.png`.

## Plot GUI `MANY_*.csv` logs

```powershell
python -m cpc_plot.plot_many_log D:\path\to\gui\logs --label "TSI 3025 cc" --label "TSI 3025 tb" --output plots\many_cpc_timeseries.png
```

The GUI data logger writes one row per update and repeats the CPC fields for each instrument. This script finds `datetime`, `datetime.1`, `concentration`, `concentration.1`, etc. and plots each CPC on the same time axis.

## CPC FARM contour plot

```powershell
python -m cpc_plot.plot_farm_distribution D:\path\to\raw ^
  --serial-channel SN213=CH1 --serial-channel SN212=CH2 --serial-channel SN210=CH3 --serial-channel SN223=CH4 --serial-channel SN208=CH5 ^
  --d50 CH1=1.6 --d50 CH2=1.9 --d50 CH3=2.3 --d50 CH4=2.7 --d50 CH5=3.3 ^
  --output plots\farm_dndlnDp.png
```

The FARM workflow treats each CPC channel as a cumulative concentration channel, sorts channels by calibrated D50, computes adjacent `Delta N`, converts to `dN/dlnDp`, masks non-positive bins, and plots time vs. particle diameter.

## Notes

- The sample data in this repo is intentionally small. Point the scripts at the full archive folders for day-long or multi-day analysis.
- Sentinel over-range values such as `999999` are treated as missing values in the FARM size-distribution workflow.
- The plotting code is meant for CPC concentration data, not calibration fitting. Calibration scripts live outside this plotting-focused repo.
