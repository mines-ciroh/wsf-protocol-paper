"""Fig 2: example forecasts at Hungry Horse (USGS 12362500 / RFC HHWM8).

Panel A: RFC ESP daily hydrograph vs. climatology vs. observed streamflow.
Panel B: cumulative Apr-Jul volume (RFC ESP vs. NRCS vs. observed vs. normal),
with an inset boxplot comparing RFC ensemble spread to NRCS's quantile forecast.

The raw daily ESP ensemble traces are NOT part of the published benchmark
archive (only the aggregated seasonal-volume product is archived; see
provenance note below) -- they're a local-only cache extracted from the
shared /beegfs/sets/aw-ciroh/common/datasets/rfc_wsf/nwrfc_wsf/ dataset.
Override WSF_RAW_ESP_ROOT if your extracted cache lives elsewhere; see the
comment near RFC_ESP_CSV for how to extract additional site/year/date combos.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from wsf_plot_common import archive_root

# ---- USER CONFIGURATION ----
RFC_SITE_ID = "HHWM8W"
USGS_ID = "12362500"
NRNI_ID = "HGH5N"
NRCS_TRIPLET = "12362500:MT:USGS"

FORECAST_DATE = "0401"
YEAR_LIST = range(2017, 2018)

WSF_ARCHIVE_ROOT = archive_root()
# Local-only cache of raw daily ESP traces, extracted from
# /beegfs/sets/aw-ciroh/common/datasets/rfc_wsf/nwrfc_wsf/DS_espensemble.<year>.tgz
# (member path: rfc_data/ARCHIVE/chps_ensemble/<year>/<year><mmdd>/<RFC_SITE_ID>_SQIN.ESPF10.csv.gz).
# To add another site/year/date, extract the matching member from that tgz
# into WSF_RAW_ESP_ROOT with the same internal directory structure.
WSF_RAW_ESP_ROOT = Path(os.environ.get("WSF_RAW_ESP_ROOT", str(Path.home() / "wsf_raw_esp_cache")))

# The manuscript's Fig 2 case (Hungry Horse, Apr 1 2017) is written as figures/fig2.example_fcsts.png;
# any other site/year/date goes to figures/case_study/.
FIG2_CASE = ("12362500", "2017", "0401")
FIG_DIR = Path("./figures")
OUT_DIR = FIG_DIR / "case_study"
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def calculate_kge(observed, simulated):
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    if len(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def calculate_nse(observed, simulated):
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs, sim = observed[mask], simulated[mask]
    if len(obs) == 0:
        return np.nan
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (np.sum((obs - sim) ** 2) / denominator)


def calculate_rmse(observed, predicted):
    if np.isnan(observed) or np.isnan(predicted):
        return np.nan
    return np.sqrt((observed - predicted) ** 2)


def calculate_pbias(observed, predicted):
    if np.isnan(observed) or np.isnan(predicted) or observed == 0:
        return np.nan
    return ((predicted - observed) / observed) * 100


def kcfs_to_kaf(series, day_fraction=None):
    """Cumulative KAF from a daily-mean KCFS series. 1 KCFS-day = 1.98347 KAF.

    day_fraction weights each day by the share of the day actually present in
    the raw traces. The ESP files start mid-day (2017 begins at 18:00, so April 1
    carries one 6-hourly timestep where a full day carries four); resampling to a
    daily mean and then multiplying by a full day counted that partial day at
    ~4x its real volume. The observed cumulative it is compared against has no
    partial day, so the error was one-sided.
    """
    daily = series if day_fraction is None else series.mul(day_fraction, axis=0)
    return daily.cumsum() * 1.98347


headline_rows = []

for YEAR_INT in YEAR_LIST:
    YEAR = str(YEAR_INT)
    esp_csv = WSF_RAW_ESP_ROOT / "rfc_data" / "ARCHIVE" / "chps_ensemble" / YEAR / f"{YEAR}{FORECAST_DATE}" / f"{RFC_SITE_ID}_SQIN.ESPF10.csv"
    if not esp_csv.exists():
        esp_csv_gz = esp_csv.with_suffix(esp_csv.suffix + ".gz")
        if esp_csv_gz.exists():
            esp_csv = esp_csv_gz
        else:
            print(f"SKIP {YEAR}: no raw ESP file at {esp_csv} (or .gz) -- see WSF_RAW_ESP_ROOT note above.")
            continue

    FLOW_OBS_CSV = WSF_ARCHIVE_ROOT / "source_data" / "obs" / "processed_nrni_csvs" / f"{NRNI_ID}.csv"
    NRCS_VOL_CSV = WSF_ARCHIVE_ROOT / "source_data" / "nrcs" / "nrcs_fcst_obs.all_sites_yrs.csv"

    plot_start = f"{YEAR}-04-01"
    plot_end = f"{YEAR}-07-31"

    # === OBS (Panel 1) ===
    obs_df = pd.read_csv(FLOW_OBS_CSV, parse_dates=True, index_col=0, usecols=["datetime", "streamflow_cfs"])
    obs_df.index = pd.to_datetime(obs_df.index)
    obs_df_kcfs = obs_df / 1000
    obs_df_kcfs.rename(columns={"streamflow_cfs": "streamflow_kcfs"}, inplace=True)
    obs_df_kcfs_fcst_period = obs_df_kcfs.loc[plot_start:plot_end]
    if obs_df_kcfs_fcst_period.empty:
        print(
            f"WARNING: {FLOW_OBS_CSV.name} has no daily obs coverage for {plot_start}..{plot_end} "
            f"(record spans {obs_df.index.min().date()}..{obs_df.index.max().date()}). "
            "Panel A (hydrograph) and KGE/NSE will be blank/NaN until this file is updated with more recent NRNI data."
        )

    # === RFC ESP (Panel 1) ===
    df = pd.read_csv(esp_csv, skiprows=6, parse_dates=True, index_col=0)
    _resampled = df.loc[plot_start:plot_end].resample("D")
    _steps = _resampled.size()
    df = _resampled.mean()
    # Fraction of each day actually covered by raw timesteps (1.0 for full days).
    _full_day_steps = int(_steps.max()) if len(_steps) else 1
    day_fraction = (_steps / _full_day_steps).clip(upper=1.0)

    q10_rfc = df.quantile(0.10, axis=1)
    q50_rfc = df.quantile(0.50, axis=1)
    q90_rfc = df.quantile(0.90, axis=1)

    obs_df_kcfs["doy"] = obs_df_kcfs.index.dayofyear
    climo_mean = obs_df_kcfs.groupby("doy").mean()["streamflow_kcfs"]
    climo_on_period = pd.Series([climo_mean.get(d.dayofyear, np.nan) for d in df.index], index=df.index)

    common_idx = q50_rfc.index.intersection(obs_df_kcfs_fcst_period.index)
    obs_values = obs_df_kcfs_fcst_period.loc[common_idx, "streamflow_kcfs"].values
    rfc_values = q50_rfc.loc[common_idx].values
    kge_value = calculate_kge(obs_values, rfc_values)
    nse_value = calculate_nse(obs_values, rfc_values)

    # === NRCS single forecast (Panel 2) ===
    df_nrcs = pd.read_csv(NRCS_VOL_CSV)
    df_nrcs["publicationDate"] = pd.to_datetime(df_nrcs["publicationDate"])
    df_nrcs = df_nrcs[df_nrcs["stationTriplet"] == NRCS_TRIPLET]
    df_nrcs_sub = df_nrcs[
        (df_nrcs["publicationDate"].dt.strftime("%Y-%m-%d") == f"{YEAR}-04-01")
        & (df_nrcs["forecastPeriodStart_str"].str.endswith("-04-01"))
        & (df_nrcs["forecastPeriodEnd_str"].str.endswith("-07-31"))
    ].drop_duplicates()

    nrcs_10 = df_nrcs_sub["forecast_10"].values[0] if "forecast_10" in df_nrcs_sub.columns and not df_nrcs_sub.empty else np.nan
    nrcs_30 = df_nrcs_sub["forecast_30"].values[0] if "forecast_30" in df_nrcs_sub.columns and not df_nrcs_sub.empty else np.nan
    nrcs_med = df_nrcs_sub["forecast_50"].values[0] if "forecast_50" in df_nrcs_sub.columns and not df_nrcs_sub.empty else np.nan
    nrcs_70 = df_nrcs_sub["forecast_70"].values[0] if "forecast_70" in df_nrcs_sub.columns and not df_nrcs_sub.empty else np.nan
    nrcs_90 = df_nrcs_sub["forecast_90"].values[0] if "forecast_90" in df_nrcs_sub.columns and not df_nrcs_sub.empty else np.nan
    obs_amjj = df_nrcs_sub["forecast_period_actual_kaf"].values[0] if not df_nrcs_sub.empty else np.nan
    avg_amjj = df_nrcs_sub["periodNormal"].values[0] if not df_nrcs_sub.empty else np.nan

    # === Panel 2: cumulative volume ===
    rfc_cum = kcfs_to_kaf(df, day_fraction)
    rfc_q10 = rfc_cum.quantile(0.10, axis=1)
    rfc_q50 = rfc_cum.quantile(0.50, axis=1)
    rfc_q90 = rfc_cum.quantile(0.90, axis=1)
    rfc_final_values = rfc_cum.iloc[-1].values

    rmse_nrcs = calculate_rmse(obs_amjj, nrcs_med)
    rmse_rfc = calculate_rmse(obs_amjj, rfc_q50.iloc[-1])
    pbias_nrcs = calculate_pbias(obs_amjj, nrcs_med)
    pbias_rfc = calculate_pbias(obs_amjj, rfc_q50.iloc[-1])

    headline_rows.append(
        {
            "year": YEAR, "site": USGS_ID, "kge_rfc_esp": round(kge_value, 3), "nse_rfc_esp": round(nse_value, 3),
            "obs_amjj_kaf": round(obs_amjj, 1) if not np.isnan(obs_amjj) else np.nan,
            "nrcs_median_kaf": round(nrcs_med, 1) if not np.isnan(nrcs_med) else np.nan,
            "rfc_median_kaf": round(rfc_q50.iloc[-1], 1),
            "rmse_nrcs_kaf": round(rmse_nrcs, 1) if not np.isnan(rmse_nrcs) else np.nan,
            "rmse_rfc_kaf": round(rmse_rfc, 1),
            "pbias_nrcs_pct": round(pbias_nrcs, 1) if not np.isnan(pbias_nrcs) else np.nan,
            "pbias_rfc_pct": round(pbias_rfc, 1),
        }
    )

    # === Figure ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(right=0.78)

    ax1.fill_between(df.index, q10_rfc, q90_rfc, color="red", alpha=0.15)
    ax1.plot(df.index, q50_rfc, color="red", linewidth=2)
    ax1.plot(climo_on_period.index, climo_on_period.values, color="green", linestyle="--", linewidth=2)
    ax1.plot(obs_df_kcfs_fcst_period.index, obs_df_kcfs_fcst_period["streamflow_kcfs"], color="black", linewidth=2)
    ax1.set_ylabel("Flow (KCFS)", fontsize=14)
    ax1.set_title("A) ESP Timeseries", fontsize=14, loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.text(0.02, 0.98, f"KGE = {kge_value:.3f}\nNSE = {nse_value:.3f}", transform=ax1.transAxes, fontsize=10,
              va="top", ha="left", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    ax2.fill_between(rfc_cum.index, rfc_q10, rfc_q90, color="red", alpha=0.15)
    ax2.plot(rfc_cum.index, rfc_q50, color="red", linewidth=2)
    if not np.isnan(nrcs_10) and not np.isnan(nrcs_90):
        ax2.axhspan(nrcs_10, nrcs_90, color="orange", alpha=0.25)
    if not np.isnan(nrcs_med):
        ax2.axhline(nrcs_med, color="orange", linestyle="-", linewidth=2)
    if not np.isnan(obs_amjj):
        ax2.axhline(obs_amjj, color="black", linestyle="-", linewidth=2)
    if not np.isnan(avg_amjj):
        ax2.axhline(avg_amjj, color="green", linestyle="--", linewidth=2)

    ax2.set_ylabel("Cumulative Volume (KAF)", fontsize=14)
    ax2.set_title("B) Cumulative Apr-Jul Volume", fontsize=14, loc="left", fontweight="bold")
    ax2.grid(True)

    all_ax2_vals = list(rfc_final_values) + [v for v in [nrcs_10, nrcs_90, nrcs_med, obs_amjj, avg_amjj] if not np.isnan(v)]
    ax2.set_ylim(min(0, min(all_ax2_vals) * 0.95), max(all_ax2_vals) * 1.05)

    ax2.text(0.02, 0.98, f"RMSE / PBIAS:\nNRCS-50: {rmse_nrcs:.1f} KAF / {pbias_nrcs:.1f}%\nRFC-50:  {rmse_rfc:.1f} KAF / {pbias_rfc:.1f}%",
             transform=ax2.transAxes, fontsize=10, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    # Boxplot inset -- whis=(10,90) so both RFC (many members) and NRCS
    # (3-point approx) show the same 10-90th percentile range.
    ax2_inset = inset_axes(ax2, width="35%", height="35%", loc="lower right", bbox_to_anchor=(0, 0.05, 1, 1), bbox_transform=ax2.transAxes, borderpad=0.5)

    box_stats = []
    if len(rfc_final_values) > 0:
        box_stats.append({
            "label": "RFC", "whislo": np.nanquantile(rfc_final_values, 0.10), "q1": np.nanquantile(rfc_final_values, 0.30),
            "med": np.nanquantile(rfc_final_values, 0.50), "q3": np.nanquantile(rfc_final_values, 0.70),
            "whishi": np.nanquantile(rfc_final_values, 0.90), "fliers": [],
        })
    if all(~np.isnan([nrcs_10, nrcs_30, nrcs_med, nrcs_70, nrcs_90])):
        box_stats.append({"label": "NRCS", "whislo": nrcs_10, "q1": nrcs_30, "med": nrcs_med, "q3": nrcs_70, "whishi": nrcs_90, "fliers": []})

    if box_stats:
        bp = ax2_inset.bxp(box_stats, widths=0.6, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("red")
        bp["boxes"][0].set_alpha(0.5)
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor("orange")
            bp["boxes"][1].set_alpha(0.5)
        if not np.isnan(obs_amjj):
            ax2_inset.axhline(obs_amjj, color="black", linewidth=2, linestyle="-", zorder=10)

        all_values = [v for s in box_stats for v in (s["whislo"], s["q1"], s["med"], s["q3"], s["whishi"])]
        if not np.isnan(obs_amjj):
            all_values.append(obs_amjj)
        if all_values:
            ymin, ymax = min(all_values), max(all_values)
            yrange = ymax - ymin
            ax2_inset.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

        ax2_inset.set_ylabel("KAF", fontsize=9)
        ax2_inset.tick_params(labelsize=9)
        ax2_inset.set_title("Apr-Jul Volume", fontsize=10)
        ax2_inset.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2, label="Adj. Streamflow"),
        Line2D([0], [0], color="green", linewidth=2, linestyle="--", label="Climatology"),
        Line2D([0], [0], color="red", linewidth=2, label="RFC ESP-50"),
        Patch(facecolor="red", alpha=0.3, label="RFC ESP 10-90%"),
        Line2D([0], [0], color="orange", linewidth=2, label="NRCS Median"),
        Patch(facecolor="orange", alpha=0.3, label="NRCS 10-90%"),
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=True, fontsize=10, title="Legend", title_fontsize=10)

    date_fmt = mdates.DateFormatter("%b\n%Y")
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(date_fmt)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="both", labelsize=12)

    # No suptitle on the manuscript version -- the caption carries it.
    if (USGS_ID, YEAR, FORECAST_DATE) == FIG2_CASE:
        fig_path = FIG_DIR / "fig2.example_fcsts.png"
    else:
        plt.suptitle(f"April 1st, {YEAR} Forecasts at USGS {USGS_ID} (RFC: {RFC_SITE_ID})", fontsize=15)
        fig_path = OUT_DIR / f"usgs_{USGS_ID}.{YEAR}{FORECAST_DATE}.case_study.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()

if headline_rows:
    table_df = pd.DataFrame(headline_rows)
    table_path = TABLE_DIR / "fig2.headline_stats.csv"
    table_df.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")
    print(table_df.to_string(index=False))
