"""Calculate deterministic (nRMSE, PBIAS) and probabilistic (CRPS, CRPSS) forecast
skill statistics from the WSF benchmark archive.

Outputs:
    skill_stats/wsf.determ_stats.by_init_method.csv
    skill_stats/wsf.prob_stats.by_init_method.csv
    skill_stats/wsf.climo_stats.by_site.csv

Climatological RMSE/nRMSE uses the same leave-one-out ensemble as
climatological CRPS, and is normalized by the same evaluation-period mean
observation that every method's nRMSE uses, so the two are comparable and
cancel correctly in a skill score. CRPSS scores forecast and climatology at
the SAME quantile resolution (3 levels); see climo_quantiles_3. Archive path is resolved via the WSF_ARCHIVE_NC env var
(falling back to a documented default) and validated to exist before
loading. Any skipped records during the deterministic/CRPS loops are counted
and reported at the end of the run.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from properscoring import crps_ensemble

# ---- SETTINGS ----
SCRIPT_DIR = Path(__file__).resolve().parent

# The benchmark archive is not versioned in this repo. Point WSF_ARCHIVE_NC at
# it explicitly, or rely on the default below (matches the layout used during
# manuscript preparation: ~/wsf_archive_v1.0/proc_data/...).
DEFAULT_NC_PATH = (
    Path.home() / "wsf_archive_v1.0" / "proc_data" / "wsf_obs_with_all_forecasts.nc"
)
nc_path = Path(os.environ.get("WSF_ARCHIVE_NC", DEFAULT_NC_PATH))
if not nc_path.exists():
    sys.exit(
        f"WSF archive NetCDF not found at {nc_path}.\n"
        "Set the WSF_ARCHIVE_NC environment variable to the correct path."
    )

out_dir = SCRIPT_DIR / "skill_stats"
out_dir.mkdir(exist_ok=True)
out_deter_csv = out_dir / "wsf.determ_stats.by_init_method.csv"
out_prob_csv = out_dir / "wsf.prob_stats.by_init_method.csv"
climo_out_csv = out_dir / "wsf.climo_stats.by_site.csv"

start_year = 2004
end_year = 2023
EVAL_YEARS = np.arange(2004, 2024)  # Years to evaluate forecasts (20 years)
CLIMO_YEARS = np.arange(1994, 2024)  # Years for climatology (30 years: 1994-2023)

# Minimum forecast-obs pairs to compute a stat at all. This is a data-floor,
# not the paper's reporting threshold: figures apply their own, stricter
# MIN_N_PAIRS = 10 filter (see fig_scripts/) before plotting or running the
# random-walk significance test. Do not treat this constant as satisfying
# that requirement.
MIN_DATA_LEN = 5

# What to include
MMDD_list = ["0101", "0201", "0301", "0401"]
hcst_titles = {
    "0101": "January 1st",
    "0201": "February 1st",
    "0301": "March 1st",
    "0401": "April 1st",
}
MEDIAN_LABEL = "f50"

# ---- LOAD DATA ----
ds = xr.open_dataset(nc_path)
site_names = ds["site"].values
init_dates = ds["init_date"].values
method_names = ds["fcst_method"].values
percentile_labels = ds["percentile"].values
year_vals = ds["year"].values

# Percentile order is assumed elsewhere (CRPS section) to be f10/f30/f50/f70/f90;
# assert it here so a future archive rebuild with a different column order
# fails loudly instead of silently scoring the wrong quantile as f10/f90.
EXPECTED_PERCENTILES = np.array(["f10", "f30", "f50", "f70", "f90"])
if not np.array_equal(percentile_labels, EXPECTED_PERCENTILES):
    sys.exit(
        f"Archive percentile order {list(percentile_labels)} does not match "
        f"the expected order {list(EXPECTED_PERCENTILES)}. The CRPS section "
        "hardcodes indices for f10/f50/f90 based on this order."
    )

perc_idx = np.where(percentile_labels == MEDIAN_LABEL)[0][0]
f10_idx, f50_idx, f90_idx = (
    np.where(percentile_labels == "f10")[0][0],
    perc_idx,
    np.where(percentile_labels == "f90")[0][0],
)

init_dates_pd = pd.to_datetime(init_dates, errors="coerce")
mmdd_strs = init_dates_pd.strftime("%m%d")
year_strs = init_dates_pd.year

# --- Collect obs for the evaluation period (nRMSE normalization) and the
#     30-yr climatology period (climatology mean + LOO ensemble) ---
site_obs_means = {}
site_obs_dict_eval = {}  # {site: {"values": ..., "years": ...}}
site_obs_dict_climo = {}  # {site: {"values": ..., "years": ...}}

for site_idx, site in enumerate(site_names):
    obs_vals_eval, obs_years_eval = [], []
    for year in EVAL_YEARS:
        yidxs = np.where(year_vals == year)[0]
        if len(yidxs) == 0:
            continue
        obs_val = ds["obs"][site_idx, yidxs[0]].values
        if np.isfinite(obs_val):
            obs_vals_eval.append(obs_val)
            obs_years_eval.append(year)
    site_obs_means[site] = np.mean(obs_vals_eval) if obs_vals_eval else np.nan
    site_obs_dict_eval[site] = {
        "values": np.array(obs_vals_eval),
        "years": np.array(obs_years_eval),
    }

    obs_vals_climo, obs_years_climo = [], []
    for year in CLIMO_YEARS:
        yidxs = np.where(year_vals == year)[0]
        if len(yidxs) == 0:
            continue
        obs_val = ds["obs"][site_idx, yidxs[0]].values
        if np.isfinite(obs_val):
            obs_vals_climo.append(obs_val)
            obs_years_climo.append(year)
    site_obs_dict_climo[site] = {
        "values": np.array(obs_vals_climo),
        "years": np.array(obs_years_climo),
    }


def loo_climo_mean(obs_vals_climo, obs_years_climo, target_year):
    """30-yr climatology mean with target_year's own observation excluded,
    if present in the climatology record (mirrors the CRPS LOO ensemble)."""
    match = np.where(obs_years_climo == target_year)[0]
    ensemble = np.delete(obs_vals_climo, match[0]) if len(match) > 0 else obs_vals_climo
    return np.nanmean(ensemble) if len(ensemble) > 0 else np.nan


def climo_quantiles_3(climo_ensemble):
    """The climatology sample reduced to its 10th/50th/90th percentiles.

    CRPS computed by crps_ensemble carries a finite-ensemble discretization bias
    that shrinks as the member count grows. Scoring a 3-quantile forecast against
    a ~30-member climatology therefore compares two different discretizations, not
    two different skills: a forecast with skill IDENTICAL to climatology scores
    CRPSS ~= -0.09 rather than 0.0, and the penalty grows with forecast spread --
    so it falls hardest on the widest methods (the Rodeo entries, see fig.s3).
    Reducing climatology to the same three probability levels the forecasts are
    given puts both sides of the CRPSS ratio on equal footing.

    Probability levels are non-exceedance 10/50/90, matching where the archive's
    f90/f50/f10 exceedance quantiles sit (f10 = wet/high tail, f90 = dry/low).
    """
    return np.percentile(climo_ensemble, [10, 50, 90])


# ---- Climatology stats per site ----
print("Calculating climatology statistics with 30-year leave-one-out ensemble...")
climo_nrmse_list = []

for site in site_names:
    obs_vals_climo = site_obs_dict_climo[site]["values"]
    obs_years_climo = site_obs_dict_climo[site]["years"]
    obs_vals_eval = site_obs_dict_eval[site]["values"]
    obs_years_eval = site_obs_dict_eval[site]["years"]

    if len(obs_vals_climo) == 0:
        continue

    climo_val = np.nanmean(obs_vals_climo)  # 30-yr climatological mean (the climatology forecast itself)

    # nRMSE normalizer: the EVALUATION-period (2004-2023) mean observation, i.e.
    # exactly the same denominator every forecast method's nRMSE uses below.
    # Previously this used climo_val (the 30-yr 1994-2023 mean), which differs
    # from the eval mean by up to ~9% at some sites (Colville R 12409000). That
    # made climatology's nRMSE non-comparable to method nRMSE, and in particular
    # left a stray mean_climo/mean_eval factor in fig5's
    # nRMSE_SS = 1 - nRMSE/climo_nrmse, where the normalizers should cancel.
    eval_mean = site_obs_means.get(site, np.nan)

    # RMSE/nRMSE: each eval year is scored against a leave-one-out climatology
    # mean that excludes that year's own observation, matching the CRPS LOO
    # treatment below.
    if len(obs_vals_eval) > 0:
        sq_errs = [
            (loo_climo_mean(obs_vals_climo, obs_years_climo, yr) - obs_val) ** 2
            for obs_val, yr in zip(obs_vals_eval, obs_years_eval)
        ]
        rmse_climo = round(np.sqrt(np.nanmean(sq_errs)), 3)
        nrmse_climo = (
            round(rmse_climo / eval_mean * 100, 3)
            if np.isfinite(eval_mean) and eval_mean != 0
            else np.nan
        )
    else:
        rmse_climo = np.nan
        nrmse_climo = np.nan

    # CRPS: leave-one-out ensemble from the 30-year climatology period.
    # Reported at both resolutions -- see climo_quantiles_3() for why the
    # 3-quantile version is the one CRPSS uses.
    crps_climo_vals, crps_climo3_vals = [], []
    for i, obs_val in enumerate(obs_vals_climo):
        climo_ensemble = np.delete(obs_vals_climo, i)
        if len(climo_ensemble) > 0:
            crps_climo_vals.append(crps_ensemble(obs_val, climo_ensemble))
            crps_climo3_vals.append(crps_ensemble(obs_val, climo_quantiles_3(climo_ensemble)))
    crps_climo = np.nanmean(crps_climo_vals) if crps_climo_vals else np.nan
    crps_climo3 = np.nanmean(crps_climo3_vals) if crps_climo3_vals else np.nan

    climo_nrmse_list.append(
        {
            "site": site,
            "climo_mean": round(climo_val, 3),
            "climo_rmse": rmse_climo,
            "climo_nrmse": nrmse_climo,
            "eval_mean": round(eval_mean, 3) if np.isfinite(eval_mean) else np.nan,
            "climo_crps": round(crps_climo, 3) if np.isfinite(crps_climo) else np.nan,
            "climo_crps_3q": round(crps_climo3, 3) if np.isfinite(crps_climo3) else np.nan,
            "n_years_climo": len(obs_vals_climo),
            "n_years_eval": len(obs_vals_eval),
        }
    )

climo_df = pd.DataFrame(climo_nrmse_list)
climo_df.to_csv(climo_out_csv, index=False)
print(f"Saved: {climo_out_csv}")

# ---- CALCULATE DETERMINISTIC SKILL METRICS ----
print("Calculating deterministic skill metrics...")
skill_rows = []
n_skipped_unexpected = 0

for mmdd in MMDD_list:
    mmdd_mask = mmdd_strs == mmdd
    init_date_idxs = np.where(mmdd_mask)[0]
    years_for_this_mmdd = year_strs[init_date_idxs]

    for method_idx, method in enumerate(method_names):
        fcst_vals = ds["wsf"].isel(percentile=perc_idx, fcst_method=method_idx).values[:, init_date_idxs]
        for site_idx, site in enumerate(site_names):
            for date_idx, init_date_idx in enumerate(init_date_idxs):
                fcst_year = years_for_this_mmdd[date_idx]
                if fcst_year < start_year or fcst_year > end_year:
                    continue
                obs_year_idx = np.where(year_vals == fcst_year)[0]
                if len(obs_year_idx) == 0:
                    continue
                try:
                    obs_val = ds["obs"][site_idx, obs_year_idx[0]].values
                    fcst_val = fcst_vals[site_idx, date_idx]
                except (IndexError, KeyError):
                    n_skipped_unexpected += 1
                    continue
                if np.isfinite(obs_val) and np.isfinite(fcst_val):
                    skill_rows.append(
                        {
                            "site": site,
                            "method": method,
                            "mmdd": mmdd,
                            "hcst_title": hcst_titles[mmdd],
                            "year": fcst_year,
                            "obs": obs_val,
                            "fcst": fcst_val,
                        }
                    )

if n_skipped_unexpected:
    print(f"WARNING: skipped {n_skipped_unexpected} deterministic records due to indexing errors")

df = pd.DataFrame(skill_rows)

# ---- AGGREGATE TO DETERMINISTIC STATS ----
results = []
for site in site_names:
    for method in method_names:
        for mmdd in MMDD_list:
            group = df[(df["site"] == site) & (df["method"] == method) & (df["mmdd"] == mmdd)]
            mean_obs = site_obs_means.get(site, np.nan)

            if len(group) < MIN_DATA_LEN or mean_obs == 0 or np.isnan(mean_obs):
                results.append(
                    {
                        "site": site,
                        "method": method,
                        "init_date": mmdd,
                        "hcst_title": hcst_titles[mmdd],
                        "nRMSE": np.nan,
                        "RMSE": np.nan,
                        "PBIAS": np.nan,
                        "climo_rmse_matched": np.nan,
                        "nRMSE_SS": np.nan,
                        "n_pairs": len(group),
                    }
                )
                continue

            obs = group["obs"].astype(float).values
            fcst = group["fcst"].astype(float).values
            rmse = round(np.sqrt(np.nanmean((fcst - obs) ** 2)), 3)
            nrmse = round(rmse / mean_obs * 100, 3)

            # Year-matched climatology reference. The per-site climo_rmse in
            # skill_stats/wsf.climo_stats.by_site.csv is computed over ALL eval
            # years with an observation (20 at every site), but a method's own
            # RMSE covers only the years IT has forecasts for -- as few as 8 for
            # sparse RFC records. Dividing one by the other compares different
            # year sets, which is the same defect as the normalizer mismatch
            # fixed above, in the numerator instead of the denominator. Here the
            # reference is rebuilt over exactly this cell's paired years, so
            # nRMSE_SS is like-for-like (mirroring how CRPSS already accumulates
            # its climatology reference per forecast-obs pair).
            grp_years = group["year"].astype(int).values
            climo_pred = np.array([
                loo_climo_mean(
                    site_obs_dict_climo[site]["values"],
                    site_obs_dict_climo[site]["years"],
                    yr,
                )
                for yr in grp_years
            ])
            climo_rmse_matched = np.sqrt(np.nanmean((climo_pred - obs) ** 2))
            nrmse_ss = (
                round(1 - rmse / climo_rmse_matched, 4)
                if np.isfinite(climo_rmse_matched) and climo_rmse_matched > 0
                else np.nan
            )
            obs_sum = np.nansum(obs)
            pbias = round(100 * (np.nansum(fcst - obs) / obs_sum), 3) if obs_sum != 0 else np.nan
            results.append(
                {
                    "site": site,
                    "method": method,
                    "init_date": mmdd,
                    "hcst_title": hcst_titles[mmdd],
                    "nRMSE": nrmse,
                    "RMSE": rmse,
                    "PBIAS": pbias,
                    "climo_rmse_matched": round(climo_rmse_matched, 3),
                    "nRMSE_SS": nrmse_ss,
                    "n_pairs": len(obs),
                }
            )

stats_df = pd.DataFrame(results)
stats_df.to_csv(out_deter_csv, index=False)
print(f"Saved: {out_deter_csv}")

# ---- CALCULATE PROBABILISTIC SKILL METRICS (CRPS) ----
print("Calculating probabilistic skill metrics (CRPS) with leave-one-out climatology...")
crps_rows = []
n_skipped_unexpected = 0

for mmdd in MMDD_list:
    mmdd_mask = mmdd_strs == mmdd
    init_date_idxs = np.where(mmdd_mask)[0]
    years_for_this_mmdd = year_strs[init_date_idxs]

    for method_idx, method in enumerate(method_names):
        fcst_quantiles_all = ds["wsf"].isel(fcst_method=method_idx).values[:, init_date_idxs, :]

        for site_idx, site in enumerate(site_names):
            site_obs_vals_climo = site_obs_dict_climo[site]["values"]
            site_obs_years_climo = site_obs_dict_climo[site]["years"]

            for date_idx, init_date_idx in enumerate(init_date_idxs):
                fcst_year = years_for_this_mmdd[date_idx]
                if fcst_year < start_year or fcst_year > end_year:
                    continue
                obs_year_idx = np.where(year_vals == fcst_year)[0]
                if len(obs_year_idx) == 0:
                    continue

                try:
                    obs_val = ds["obs"][site_idx, obs_year_idx[0]].values
                    fcst_quantiles = fcst_quantiles_all[site_idx, date_idx, :]
                except (IndexError, KeyError):
                    n_skipped_unexpected += 1
                    continue

                # CRPS is computed from exactly f10/f50/f90 for every method, never
                # a method's full native quantile set. Methods differ in how many
                # quantiles they natively provide (NRCS/RFC: 5, CA DWR/Rodeo: 3);
                # feeding a coarser or finer quantile set into crps_ensemble changes
                # the finite-sample discretization bias, not just the estimate's
                # precision, so an uneven quantile count across methods biases CRPS/
                # CRPSS comparisons in favor of finer-resolution methods regardless
                # of true forecast skill. f10/f50/f90 is the common denominator
                # every method has.
                required_quantiles = fcst_quantiles[[f10_idx, f50_idx, f90_idx]]
                if not (np.isfinite(obs_val) and np.all(np.isfinite(required_quantiles))):
                    continue

                crps_forecast = crps_ensemble(obs_val, required_quantiles)

                year_match = np.where(site_obs_years_climo == fcst_year)[0]
                if len(year_match) > 0:
                    climo_ensemble = np.delete(site_obs_vals_climo, year_match[0])
                else:
                    climo_ensemble = site_obs_vals_climo

                if len(climo_ensemble) == 0:
                    continue

                # Matched-resolution reference: climatology reduced to the same
                # three probability levels the forecast is given. This is the
                # denominator CRPSS uses. The full-ensemble version is retained
                # alongside it so the effect of the change stays auditable.
                crps_climatology = crps_ensemble(obs_val, climo_quantiles_3(climo_ensemble))
                crps_climatology_full = crps_ensemble(obs_val, climo_ensemble)
                crps_rows.append(
                    {
                        "site": site,
                        "method": method,
                        "mmdd": mmdd,
                        "hcst_title": hcst_titles[mmdd],
                        "year": fcst_year,
                        "obs": obs_val,
                        "crps_forecast": crps_forecast,
                        "crps_climatology": crps_climatology,
                        "crps_climatology_full_ens": crps_climatology_full,
                        "n_quantiles_used": 3,  # always f10/f50/f90, see note above
                        "n_climo_members": len(climo_ensemble),
                    }
                )

if n_skipped_unexpected:
    print(f"WARNING: skipped {n_skipped_unexpected} probabilistic records due to indexing errors")

crps_df = pd.DataFrame(crps_rows)

# ---- AGGREGATE TO PROBABILISTIC STATS ----
print("Aggregating CRPS statistics and calculating CRPSS...")
crps_results = []

for site in site_names:
    for method in method_names:
        for mmdd in MMDD_list:
            group = crps_df[(crps_df["site"] == site) & (crps_df["method"] == method) & (crps_df["mmdd"] == mmdd)]
            mean_obs = site_obs_means.get(site, np.nan)

            if len(group) < MIN_DATA_LEN or mean_obs == 0 or np.isnan(mean_obs):
                crps_results.append(
                    {
                        "site": site,
                        "method": method,
                        "init_date": mmdd,
                        "hcst_title": hcst_titles[mmdd],
                        "CRPS": np.nan,
                        "nCRPS": np.nan,
                        "CRPSS": np.nan,
                        "CRPSS_full_climo_ens": np.nan,
                        "n_pairs": len(group),
                    }
                )
                continue

            crps_forecast_vals = group["crps_forecast"].astype(float).values
            crps_climatology_vals = group["crps_climatology"].astype(float).values
            crps_climatology_full_vals = group["crps_climatology_full_ens"].astype(float).values

            mean_crps_forecast = np.nanmean(crps_forecast_vals)
            mean_crps_climatology = np.nanmean(crps_climatology_vals)
            mean_crps_climatology_full = np.nanmean(crps_climatology_full_vals)

            mean_crps = round(mean_crps_forecast, 3)
            ncrps = round(mean_crps / mean_obs * 100, 3)

            if not np.isnan(mean_crps_climatology) and mean_crps_climatology > 0:
                crpss = round(1 - (mean_crps_forecast / mean_crps_climatology), 3)
            else:
                crpss = np.nan

            # Pre-2026-09 behaviour: forecast scored on 3 quantiles against a
            # ~30-member climatology ensemble. Kept for comparison only; it
            # mixes two discretizations and is biased low (see climo_quantiles_3).
            if not np.isnan(mean_crps_climatology_full) and mean_crps_climatology_full > 0:
                crpss_full = round(1 - (mean_crps_forecast / mean_crps_climatology_full), 3)
            else:
                crpss_full = np.nan

            crps_results.append(
                {
                    "site": site,
                    "method": method,
                    "init_date": mmdd,
                    "hcst_title": hcst_titles[mmdd],
                    "CRPS": mean_crps,
                    "nCRPS": ncrps,
                    "CRPSS": crpss,
                    "CRPSS_full_climo_ens": crpss_full,
                    "n_pairs": len(crps_forecast_vals),
                }
            )

crps_stats_df = pd.DataFrame(crps_results)
crps_stats_df.to_csv(out_prob_csv, index=False)
print(f"Saved: {out_prob_csv}")

print("\n=== Summary ===")
print(f"Deterministic stats: {len(stats_df)} rows")
print(f"Probabilistic stats (CRPS): {len(crps_stats_df)} rows")
print(f"Climatology calculated with leave-one-out ensemble for {len(climo_df)} sites")

print("\nEarliest valid forecast-obs pair for each method (across all sites/init dates):")
for method in sorted(df["method"].unique()):
    obs_float = df["obs"].astype(float)
    fcst_float = df["fcst"].astype(float)
    sub = df[(df["method"] == method) & np.isfinite(obs_float) & np.isfinite(fcst_float)]
    if len(sub) == 0:
        print(f"{method}: No valid pairs")
        continue
    min_idx = sub["year"].idxmin()
    row = sub.loc[min_idx]
    print(f"{method:20s}  Year: {int(row['year'])},  Site: {row['site']},  Init MMDD: {row['mmdd']}")
