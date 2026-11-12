"""Random-walk sign test (DelSole & Tippett 2016) of each WSF Rodeo method vs.
its site's primary statistical benchmark (NRCS, or CA DWR at the two CA sites).

Writes no figure. Outputs:
  - ../calc_stats/skill_stats/random_walk_sign_test.results.csv: final net wins,
    N and two-sided binomial p for every site x method x init (Jan-Apr and all
    inits pooled). Read by fig7.random_walk_combined.py and fig.s4.random_walk_map.py.
  - tables/figs7_8.headline_stats.csv: significance counts by init and pooled,
    using the +/-2 sqrt(N) envelope and the N >= MIN_N_PAIRS floor (Table 1).
    Source of the manuscript's "26 of 125 (20.8%) significant; 6 favor Rodeo".

The name is kept from when this script also drew the submitted Figs 7 and 8.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binomtest

from wsf_plot_common import (
    archive_nc_path,
    START_YEAR as start_year,
    END_YEAR as end_year,
    MIN_N_PAIRS,
    INIT_MMDD_LIST,
    INIT_LABELS_SHORT as _INIT_LABELS_SHORT_MAP,
    WSF_RODEO_METHODS as WSF_METHODS,
)

# -------- SETTINGS --------
nc_path = archive_nc_path()

TABLE_DIR = Path("./tables")
TABLE_DIR.mkdir(exist_ok=True)
RESULTS_CSV = Path("../calc_stats/skill_stats/random_walk_sign_test.results.csv")

NRCS_METHOD = "nrcs"
CA_DWR_METHOD = "ca_dwr"
sites_use_cadwr = {"11251000", "11446220"}
PCTILE = "f50"
INIT_LABELS_SHORT = [_INIT_LABELS_SHORT_MAP[mmdd] for mmdd in INIT_MMDD_LIST]

# -------- LOAD DATA --------
ds = xr.open_dataset(nc_path)
site_names = ds["site"].values
init_dates = ds["init_date"].values
percentile_labels = ds["percentile"].values
method_names = ds["fcst_method"].values
year_vals = ds["year"].values

p50_idx = np.where(percentile_labels == PCTILE)[0][0]
mmdd_dates = pd.to_datetime(init_dates, errors="coerce")
all_mmdd_str = mmdd_dates.strftime("%m%d")


def ref_method_for_site(site):
    return CA_DWR_METHOD if site in sites_use_cadwr else NRCS_METHOD


def obs_ref_for_indices(basin_idx, ref_idx, idxs):
    """Collect obs and reference-method f50 arrays for a set of init_date indices."""
    obs_list, ref_list = [], []
    for init_idx in idxs:
        year = mmdd_dates[init_idx].year
        if not (start_year <= year <= end_year):
            continue
        obs_year_idx = np.where(year_vals == year)[0]
        if len(obs_year_idx) == 0:
            continue
        obs_val = ds["obs"][basin_idx, obs_year_idx[0]].values
        ref_val = ds["wsf"].isel(site=basin_idx, init_date=init_idx, percentile=p50_idx, fcst_method=ref_idx).values
        obs_list.append(obs_val)
        ref_list.append(ref_val)
    return np.array(obs_list), np.array(ref_list)


def cand_for_indices(basin_idx, wsf_idx, idxs):
    cand_list = []
    for init_idx in idxs:
        year = mmdd_dates[init_idx].year
        if not (start_year <= year <= end_year):
            continue
        cand_list.append(ds["wsf"].isel(site=basin_idx, init_date=init_idx, percentile=p50_idx, fcst_method=wsf_idx).values)
    return np.array(cand_list)


def sign_test(obs_arr, ref_arr, cand_arr):
    """DelSole & Tippett (2016) random-walk sign test on squared f50 error.
    Returns (walk, N, pval); walk/pval are None/nan if N < 2 (ties removed)."""
    mask = np.isfinite(obs_arr) & np.isfinite(ref_arr) & np.isfinite(cand_arr)
    obs, ref, cand = obs_arr[mask], ref_arr[mask], cand_arr[mask]
    wins = np.where((cand - obs) ** 2 < (ref - obs) ** 2, 1, np.where((cand - obs) ** 2 > (ref - obs) ** 2, -1, 0))
    wins = wins[wins != 0]  # remove ties
    N = len(wins)
    if N < 2:
        return None, 0, np.nan
    walk = np.cumsum(wins)
    K = int(np.sum(wins == 1))
    # Exact two-sided binomial test. The previous doubled-CDF form,
    # 2 * binom.cdf(min(K, N-K), N, 0.5), double-counts the central mass
    # whenever K == N - K and returned p > 1 (up to 1.31) for 71 of 650 comparisons.
    pval = binomtest(K, N, 0.5).pvalue
    return walk, N, pval


# -------- Final net wins per site x method x init (and pooled) --------
print("Computing random-walk results...")
rw_results = []
for basin_idx, site in enumerate(site_names):
    ref_idx = np.where(method_names == ref_method_for_site(site))[0][0]
    groups = [(label, np.where(all_mmdd_str == mmdd)[0]) for mmdd, label in zip(INIT_MMDD_LIST, INIT_LABELS_SHORT)]
    groups.append(("All", np.where(np.isin(all_mmdd_str, INIT_MMDD_LIST))[0]))
    for init_label, idxs in groups:
        obs_arr, ref_arr = obs_ref_for_indices(basin_idx, ref_idx, idxs)
        for wsf_method in WSF_METHODS:
            wsf_idx = np.where(method_names == wsf_method)[0][0]
            walk, N, pval = sign_test(obs_arr, ref_arr, cand_for_indices(basin_idx, wsf_idx, idxs))
            skill_wins_final = int(walk[-1]) if walk is not None else np.nan
            rw_results.append({"site": site, "init": init_label, "method": wsf_method, "N": N,
                               "skill_wins_final": skill_wins_final, "pval": round(pval, 4)})

rw_results_df = pd.DataFrame(rw_results)
rw_results_df.to_csv(RESULTS_CSV, index=False)
print(f"Saved: {RESULTS_CSV}")

# ---- Headline-stats table: significance counts by init (and pooled) ----
# Filter to the protocol's stated minimum sample size (Table 1), matching
# the MIN_N_PAIRS filter used in fig4/fig5/fig.s1/fig.s3/fig.s4.
df_plot = rw_results_df[rw_results_df["N"] >= MIN_N_PAIRS]
df_individual = df_plot[df_plot["init"] != "All"]
df_all = df_plot[df_plot["init"] == "All"]

summary_rows = []
for init in list(sorted(df_individual["init"].unique())) + ["All"]:
    d = df_all if init == "All" else df_individual[df_individual["init"] == init]
    # Each comparison is tested against its own +/-2 sqrt(N), N = pairs it actually has
    # (the test as stated; N runs 14-20 per init and 35-80 pooled). A fixed 2 sqrt(20) /
    # 2 sqrt(80) held short-record sites to a stricter bar than their own envelope.
    threshold = 2 * np.sqrt(d["N"])
    positive_sig = d[d["skill_wins_final"] > threshold]
    negative_sig = d[d["skill_wins_final"] < -threshold]
    total_sig = len(positive_sig) + len(negative_sig)
    total = len(d)
    summary_rows.append(
        {
            "init": init,
            "threshold_abs_wins_range": f"{threshold.min():.2f}-{threshold.max():.2f}",
            "n_comparisons": total,
            "n_significant": total_sig,
            "pct_significant": round(total_sig / total * 100, 1) if total else np.nan,
            "n_favor_rodeo": len(positive_sig),
            "n_favor_benchmark": len(negative_sig),
        }
    )
summary_df = pd.DataFrame(summary_rows)
summary_path = TABLE_DIR / "figs7_8.headline_stats.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")
print(summary_df.to_string(index=False))
