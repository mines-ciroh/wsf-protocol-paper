"""Fig 7: random-walk sign test, two example sites + all sites.

Replaces the submitted Fig 7 (2x2 Sweetwater walks) and Fig 8 (cross-site
boxplots). figs7_8.random_walks.py now only computes the sign-test results CSV
and headline table this figure reads. All four initializations pooled (N <= 80 per site-method), each WSF Rodeo
method vs. the site's statistical benchmark (NRCS; CA DWR at the two CA DWR
sites), WY2004-2023 -- the same test and inputs as figs7_8.random_walks.py.

  A) Yampa R nr Maybell (09251000): two methods finish significantly worse than
     NRCS; a third finishes at +17, just inside the envelope.
  B) Stehekin R at Stehekin (12451000): one method finishes significantly better.
  C) Every site-method comparison with N >= MIN_N_PAIRS: final cumulative wins
     scaled by sqrt(N), so one +/-2 band (the +/-2 sqrt(N) envelope drawn in
     A-B) applies to all of them despite N varying by site. A and B's sites are
     marked.

Significance here is the envelope rule drawn in the figure (|final| > 2 sqrt(N)).
The exact binomial p-value (p < 0.05) differs from it for borderline cases; the
headline table reports both counts so the text can match whichever rule it states.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import binomtest

from wsf_plot_common import (
    archive_nc_path,
    START_YEAR,
    END_YEAR,
    MIN_N_PAIRS,
    INIT_MMDD_LIST,
    WSF_RODEO_METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
)

EXAMPLES = [("09251000", "A) Yampa River near Maybell, CO"), ("12451000", "B) Stehekin River at Stehekin, WA")]
CA_DWR_SITES = {"11251000", "11446220"}
SHORT = {m: f"Rodeo {i}" for i, m in enumerate(WSF_RODEO_METHODS, start=1)}

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# ---- Data: f50 at the four initializations, WY2004-2023, in archive (chronological) order ----
ds = xr.open_dataset(archive_nc_path())
dates = pd.to_datetime(ds["init_date"].values)
sel = np.isin(dates.strftime("%m%d"), INIT_MMDD_LIST) & (dates.year >= START_YEAR) & (dates.year <= END_YEAR)
ev_dates = dates[sel]
f50 = ds["wsf"].sel(percentile="f50").isel(init_date=np.where(sel)[0])
obs_by_year = {str(s): pd.Series(ds["obs"].sel(site=s).values, index=ds["year"].values) for s in ds["site"].values}


def walk_for(site, method):
    """Sign-test walk as in figs7_8.random_walks.sign_test: +1 when the Rodeo f50 has the
    smaller squared error, ties dropped. Returns (event years, walk, N, p)."""
    ref = "ca_dwr" if site in CA_DWR_SITES else "nrcs"
    o = obs_by_year[site].reindex(ev_dates.year).values
    r = f50.sel(site=site, fcst_method=ref).values
    c = f50.sel(site=site, fcst_method=method).values
    ok = np.isfinite(o) & np.isfinite(r) & np.isfinite(c)
    step = np.sign((r[ok] - o[ok]) ** 2 - (c[ok] - o[ok]) ** 2)
    yrs = ev_dates.year.values[ok]
    keep = step != 0
    step, yrs = step[keep], yrs[keep]
    n = len(step)
    p = binomtest(int((step > 0).sum()), n, 0.5).pvalue if n else np.nan
    return yrs, np.cumsum(step), n, p


rows = []
for site in [str(s) for s in ds["site"].values]:
    for m in WSF_RODEO_METHODS:
        _, walk, n, p = walk_for(site, m)
        if n >= MIN_N_PAIRS:
            rows.append({"site": site, "method": m, "N": n, "final": int(walk[-1]), "p": p})
res = pd.DataFrame(rows)
res["z"] = res["final"] / np.sqrt(res["N"])
res["sig_envelope"] = res["z"].abs() > 2
res["sig_p05"] = res["p"] < 0.05

# Cross-check against the committed pipeline output.
committed = pd.read_csv("../calc_stats/skill_stats/random_walk_sign_test.results.csv", dtype={"site": str})
committed = committed[(committed["init"] == "All") & (committed["N"] >= MIN_N_PAIRS)]
chk = res.merge(committed, on=["site", "method"], suffixes=("", "_c"))
if len(chk) != len(res) or (chk["final"] != chk["skill_wins_final"]).any() or (chk["N"] != chk["N_c"]).any():
    raise SystemExit("Pooled walks do not reproduce random_walk_sign_test.results.csv")

summary = pd.DataFrame([{
    "n_comparisons": len(res),
    "sig_envelope": int(res["sig_envelope"].sum()),
    "sig_envelope_favor_rodeo": int((res["sig_envelope"] & (res["final"] > 0)).sum()),
    "sig_envelope_favor_benchmark": int((res["sig_envelope"] & (res["final"] < 0)).sum()),
    "sig_p05": int(res["sig_p05"].sum()),
    "sig_p05_favor_rodeo": int((res["sig_p05"] & (res["final"] > 0)).sum()),
    "sig_p05_favor_benchmark": int((res["sig_p05"] & (res["final"] < 0)).sum()),
    "rule_disagreements": ";".join(f"{r.site}/{SHORT[r.method]}(final={r.final},N={r.N},p={r.p:.3f})"
                                   for r in res[res["sig_envelope"] != res["sig_p05"]].itertuples()),
}])
summary.to_csv(TABLE_DIR / "fig7.random_walk_combined.headline_stats.csv", index=False)
print(summary.T.to_string())

# ---- Plot ----
fig = plt.figure(figsize=(16, 11.5))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.34, wspace=0.12)
ax_ex = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
axC = fig.add_subplot(gs[1, :])

for k, (ax, (site, title)) in enumerate(zip(ax_ex, EXAMPLES)):
    n_max = 0
    for m in WSF_RODEO_METHODS:
        yrs, walk, n, _ = walk_for(site, m)
        n_max = max(n_max, n)
        ax.plot(np.arange(n + 1), np.r_[0, walk], color=METHOD_COLORS[m], lw=2.2, label=SHORT[m])
        ax.annotate(f"{int(walk[-1]):+d}", (n, walk[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=14, color=METHOD_COLORS[m], fontweight="bold")
    bx = np.arange(n_max + 1)
    ax.fill_between(bx, -2 * np.sqrt(bx), 2 * np.sqrt(bx), color="gray", alpha=0.18, lw=0)
    ax.axhline(0, color="k", ls="--", lw=1)
    # Water-year ticks at each year's first event
    yrs = walk_for(site, WSF_RODEO_METHODS[0])[0]
    first_ev = {y: int(np.argmax(yrs == y)) + 1 for y in set(yrs)}
    tick_years = list(range(START_YEAR, 2026, 3))
    tick_pos = [first_ev[y] if y in first_ev else first_ev[max(first_ev)] + 4 * (y - max(first_ev))
                for y in tick_years]
    ax.set_xticks(tick_pos, [str(y) for y in tick_years])
    ax.set_xlim(0, max(n_max + 7, tick_pos[-1] + 1))
    ax.set_ylim(-30, 30)
    ax.set_title(title, fontsize=17, fontweight="bold", loc="left")
    ax.set_xlabel("Forecast (Jan-Apr initializations, by water year)", fontsize=16)
    if k == 0:
        ax.set_ylabel("Net skill wins vs. benchmark", fontsize=17, fontweight="bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(alpha=0.3)

# Panel C: all comparisons, final wins / sqrt(N)
rng = np.random.default_rng(0)
axC.axhspan(-2, 2, color="gray", alpha=0.18, lw=0)
axC.axhline(0, color="k", ls="--", lw=1)
for i, m in enumerate(WSF_RODEO_METHODS):
    d = res[res["method"] == m]
    x = i + rng.uniform(-0.18, 0.18, len(d))
    sig = d["sig_envelope"].values
    axC.scatter(x[~sig], d["z"].values[~sig], s=80, facecolor="white", edgecolor=METHOD_COLORS[m], lw=1.6, zorder=3)
    axC.scatter(x[sig], d["z"].values[sig], s=100, color=METHOD_COLORS[m], edgecolor="black", lw=0.8, zorder=4)
    for (site, title), mk in zip(EXAMPLES, ["A", "B"]):
        hit = (d["site"] == site).values
        if hit.any():
            xi, zi, si = x[hit][0], d["z"].values[hit][0], bool(sig[hit][0])
            # Example sites: thick black ring, keeping the point's own fill (filled = outside range)
            axC.scatter([xi], [zi], s=190, facecolor=METHOD_COLORS[m] if si else "white",
                        edgecolor="black", lw=2.6, zorder=6)
            axC.annotate(mk, (xi, zi), xytext=(12, 0), textcoords="offset points", va="center", fontsize=18,
                         fontweight="bold", zorder=7, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
axC.set_xticks(range(len(WSF_RODEO_METHODS)), [METHOD_LABELS[m] for m in WSF_RODEO_METHODS], fontsize=16)
axC.set_xlim(-0.6, len(WSF_RODEO_METHODS) - 0.4)
axC.set_ylim(-4.0, 4.0)
axC.set_ylabel("Final wins / √N", fontsize=17, fontweight="bold")
axC.tick_params(axis="y", labelsize=15)
axC.grid(axis="y", alpha=0.3)
axC.set_title(f"C) All {len(res)} site-method comparisons",
              fontsize=17, fontweight="bold", loc="left")

handles = [Line2D([], [], color=METHOD_COLORS[m], lw=2.6, label=SHORT[m]) for m in WSF_RODEO_METHODS]
handles2 = [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.18, label="±2√N (95% range under equal skill)"),
            Line2D([], [], marker="o", ls="", markerfacecolor="gray", markeredgecolor="black", ms=10,
                   label="outside range"),
            Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="gray", ms=8,
                   label="inside range")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.995),
           columnspacing=1.6, handlelength=2.0)
fig.legend(handles=handles2, loc="upper center", ncol=3, fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.958),
           columnspacing=1.6, handlelength=2.0)

fig_path = OUT_DIR / "fig7.random_walk_combined.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
