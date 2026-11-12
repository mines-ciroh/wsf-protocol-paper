"""Fig S.5: number of published water supply forecasting studies in the western
US per year, with a linear trend.

Input: wsf_pubs_counts_by_year.csv, the per-year record counts from a Web of
Science search (query and search details in readme.txt). Only the counts are
distributed here; the bibliographic records themselves are not.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "wsf_pubs_counts_by_year.csv"
OUTPUT_FIG = HERE / "wsf_study_counts_with_trend.png"

MIN_YEAR, MAX_YEAR = 1950, 2025

counts = pd.read_csv(INPUT_FILE)
counts = counts[(counts["publication_year"] >= MIN_YEAR) & (counts["publication_year"] <= MAX_YEAR)]
years = counts["publication_year"].values
n_studies = counts["n_studies"].values

# Linear trend: n_studies = slope * year + intercept
slope, intercept = np.polyfit(years, n_studies, 1)
trend = slope * years + intercept
ss_tot = np.sum((n_studies - n_studies.mean()) ** 2)
r2 = 1 - np.sum((n_studies - trend) ** 2) / ss_tot if ss_tot > 0 else np.nan
print(f"Records {MIN_YEAR}-{MAX_YEAR}: {n_studies.sum()}; trend slope = {slope:.3f} studies/yr, R² = {r2:.3f}")

plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.bar(years, n_studies, label="Studies per year")
ax.plot(years, trend, linewidth=2.5, color="orange",
        label=f"Linear trend (slope={slope:.2f} studies/yr, $R^2$={r2:.2f})")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xlabel("Publication Year", fontsize=12)
ax.set_ylabel("Number of Published Studies", fontsize=12)
ax.set_title("Water Supply Forecasting Studies in the Western U.S. Over Time", fontsize=16)
ax.set_xticks(np.arange(years.min(), years.max() + 1, 5 if (years.max() - years.min()) > 10 else 1))
plt.xticks(rotation=45, ha="right")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, facecolor="white")
print(f"Saved: {OUTPUT_FIG}")
