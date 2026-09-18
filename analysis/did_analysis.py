"""
Baseline Event Study — Full Panel (13 forums)
Equation 1: grievance_it = αi + γt + β1·NFP_t + β2·CPI_t + εit
Two-way FE, clustered SE by user
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

BASE_DIR = "/Users/"

TREAT_FORUMS   = {"stormfront2025", "whitenations2021", "incelsnet2021",
                  "onionfarms2025", "vanguard", "mgtow", "goingyourownway"}
CONTROL_FORUMS = {"lookism", "looksmax", "rooshv", "pickupartist", "lookstheory"}

# ── Load panel ────────────────────────────────────────────────────────────────
print("Loading panel...")
uq = pd.read_csv(os.path.join(BASE_DIR, "user_quarter_panel_v4.csv"))
uq["year"]    = uq["year"].astype(int)
uq["quarter"] = uq["quarter"].astype(int)
uq["yearq"]   = uq["year"].astype(str) + "Q" + uq["quarter"].astype(str)
uq["forum_type"] = uq["forum_name"].apply(
    lambda x: "treatment" if x in TREAT_FORUMS else "control")

print(f"Total user-quarter rows: {len(uq):,}")
print(f"\nBy forum type:")
print(uq.groupby(["forum_type","forum_name"])["user_id"].nunique())

# ── Load surprise series ──────────────────────────────────────────────────────
print("\nLoading surprise series...")
nfp = pd.read_csv(os.path.join(BASE_DIR, "nfp_surprise_series.csv"))
cpi = pd.read_csv(os.path.join(BASE_DIR, "cpi_surprise_series.csv"))
for df in [nfp, cpi]:
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["year"]    = df["release_date"].dt.year
    df["quarter"] = df["release_date"].dt.quarter

nfp_q = (nfp.sort_values("release_date").groupby(["year","quarter"]).first()
             .reset_index()[["year","quarter","nfp_surprise_std",
                             "nfp_large_surprise","nfp_negative_surprise"]]
             .dropna(subset=["nfp_surprise_std"]))
cpi_q = (cpi.sort_values("release_date").groupby(["year","quarter"]).first()
             .reset_index()[["year","quarter","cpi_surprise_std"]]
             .dropna(subset=["cpi_surprise_std"]))
uq["year"]    = uq["year"].astype(int)
uq["quarter"] = uq["quarter"].astype(int)
# Drop existing surprise columns to avoid _x _y conflicts
cols_to_drop = [c for c in uq.columns if c in [
    'nfp_surprise_std', 'cpi_surprise_std', 'nfp_large_surprise',
    'cpi_large_surprise', 'nfp_negative_surprise', 'cpi_positive_surprise']]
uq = uq.drop(columns=cols_to_drop, errors='ignore')
uq = uq.merge(nfp_q, on=["year","quarter"], how="left")
uq = uq.merge(cpi_q, on=["year","quarter"], how="left")
print(uq.columns.tolist())
uq["nfp_surprise_std"] = uq["nfp_surprise_std"].fillna(0)
uq["cpi_surprise_std"] = uq["cpi_surprise_std"].fillna(0)

# ── Two-way FE demeaning ──────────────────────────────────────────────────────
print("Demeaning...")
uq["user_fe"]  = uq["user_id"].astype(str) + "_" + uq["forum_name"]
uq["yearq_fe"] = uq["yearq"]

p1  = uq["mean_grievance_score"].quantile(0.01)
p99 = uq["mean_grievance_score"].quantile(0.99)
uq["grievance_w"] = uq["mean_grievance_score"].clip(p1, p99)

for col in ["grievance_w","frac_grievance",
            "nfp_surprise_std","cpi_surprise_std"]:
    um = uq.groupby("user_fe")[col].transform("mean")
    uq[f"{col}_dm"] = uq[col] - um
    wm = uq.groupby("yearq_fe")[f"{col}_dm"].transform("mean")
    uq[f"{col}_dm2"] = uq[f"{col}_dm"] - wm

# ── Baseline regression — full panel ─────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE — Full Panel (13 forums)")
print("="*60)

model_full = smf.ols(
    "grievance_w_dm2 ~ nfp_surprise_std_dm2 + cpi_surprise_std_dm2 - 1",
    data=uq.dropna(subset=["grievance_w_dm2"])
).fit(cov_type="cluster",
      cov_kwds={"groups": uq.dropna(subset=["grievance_w_dm2"])["user_fe"]})

print(model_full.summary().tables[1])
print(f"\nβ_NFP = {model_full.params['nfp_surprise_std_dm2']:.4f} "
      f"(SE={model_full.bse['nfp_surprise_std_dm2']:.4f}, "
      f"p={model_full.pvalues['nfp_surprise_std_dm2']:.4f})")
print(f"β_CPI = {model_full.params['cpi_surprise_std_dm2']:.4f} "
      f"(SE={model_full.bse['cpi_surprise_std_dm2']:.4f}, "
      f"p={model_full.pvalues['cpi_surprise_std_dm2']:.4f})")
print(f"N = {int(model_full.nobs):,} | R² = {model_full.rsquared:.4f}")

# ── Treatment forums only ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TREATMENT FORUMS ONLY")
print("="*60)

uq_treat = uq[uq["forum_type"]=="treatment"].copy()
for col in ["grievance_w","nfp_surprise_std","cpi_surprise_std"]:
    um = uq_treat.groupby("user_fe")[col].transform("mean")
    uq_treat[f"{col}_dm"] = uq_treat[col] - um
    wm = uq_treat.groupby("yearq_fe")[f"{col}_dm"].transform("mean")
    uq_treat[f"{col}_dm2"] = uq_treat[f"{col}_dm"] - wm

model_treat = smf.ols(
    "grievance_w_dm2 ~ nfp_surprise_std_dm2 + cpi_surprise_std_dm2 - 1",
    data=uq_treat.dropna(subset=["grievance_w_dm2"])
).fit(cov_type="cluster",
      cov_kwds={"groups": uq_treat.dropna(subset=["grievance_w_dm2"])["user_fe"]})

print(f"β_NFP = {model_treat.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_treat.pvalues['nfp_surprise_std_dm2']:.4f})")
print(f"β_CPI = {model_treat.params['cpi_surprise_std_dm2']:.4f} "
      f"(p={model_treat.pvalues['cpi_surprise_std_dm2']:.4f})")
print(f"N = {int(model_treat.nobs):,}")

# ── Control forums only ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONTROL FORUMS ONLY")
print("="*60)

uq_ctrl = uq[uq["forum_type"]=="control"].copy()
for col in ["grievance_w","nfp_surprise_std","cpi_surprise_std"]:
    um = uq_ctrl.groupby("user_fe")[col].transform("mean")
    uq_ctrl[f"{col}_dm"] = uq_ctrl[col] - um
    wm = uq_ctrl.groupby("yearq_fe")[f"{col}_dm"].transform("mean")
    uq_ctrl[f"{col}_dm2"] = uq_ctrl[f"{col}_dm"] - wm

model_ctrl = smf.ols(
    "grievance_w_dm2 ~ nfp_surprise_std_dm2 + cpi_surprise_std_dm2 - 1",
    data=uq_ctrl.dropna(subset=["grievance_w_dm2"])
).fit(cov_type="cluster",
      cov_kwds={"groups": uq_ctrl.dropna(subset=["grievance_w_dm2"])["user_fe"]})

print(f"β_NFP = {model_ctrl.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_ctrl.pvalues['nfp_surprise_std_dm2']:.4f})")
print(f"β_CPI = {model_ctrl.params['cpi_surprise_std_dm2']:.4f} "
      f"(p={model_ctrl.pvalues['cpi_surprise_std_dm2']:.4f})")
print(f"N = {int(model_ctrl.nobs):,}")

# ── By forum ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BY FORUM")
print("="*60)

for forum in sorted(uq["forum_name"].unique()):
    sub = uq[uq["forum_name"]==forum].copy()
    if len(sub) < 100:
        print(f"{forum}: too few observations ({len(sub)})")
        continue
    for col in ["grievance_w","nfp_surprise_std","cpi_surprise_std"]:
        um = sub.groupby("user_fe")[col].transform("mean")
        sub[f"{col}_dm"] = sub[col] - um
        wm = sub.groupby("yearq_fe")[f"{col}_dm"].transform("mean")
        sub[f"{col}_dm2"] = sub[f"{col}_dm"] - wm
    try:
        m = smf.ols(
            "grievance_w_dm2 ~ nfp_surprise_std_dm2 + cpi_surprise_std_dm2 - 1",
            data=sub.dropna(subset=["grievance_w_dm2"])
        ).fit(cov_type="cluster",
              cov_kwds={"groups": sub.dropna(subset=["grievance_w_dm2"])["user_fe"]})
        ftype = "T" if forum in TREAT_FORUMS else "C"
        print(f"[{ftype}] {forum} (n={len(sub):,}): "
              f"β_NFP={m.params['nfp_surprise_std_dm2']:.4f} "
              f"(p={m.pvalues['nfp_surprise_std_dm2']:.3f}) | "
              f"β_CPI={m.params['cpi_surprise_std_dm2']:.4f} "
              f"(p={m.pvalues['cpi_surprise_std_dm2']:.3f})")
    except Exception as e:
        print(f"{forum}: ERROR — {e}")

# ── Robustness: binary flag ───────────────────────────────────────────────────
print("\n" + "="*60)
print("ROBUSTNESS — Binary grievance flag")
print("="*60)

model_bin = smf.ols(
    "frac_grievance_dm2 ~ nfp_surprise_std_dm2 + cpi_surprise_std_dm2 - 1",
    data=uq.dropna(subset=["frac_grievance_dm2"])
).fit(cov_type="cluster",
      cov_kwds={"groups": uq.dropna(subset=["frac_grievance_dm2"])["user_fe"]})

print(f"β_NFP = {model_bin.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_bin.pvalues['nfp_surprise_std_dm2']:.4f})")
print(f"β_CPI = {model_bin.params['cpi_surprise_std_dm2']:.4f} "
      f"(p={model_bin.pvalues['cpi_surprise_std_dm2']:.4f})")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"Full panel (13 forums):  β_NFP={model_full.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_full.pvalues['nfp_surprise_std_dm2']:.4f}), N={int(model_full.nobs):,}")
print(f"Treatment only:          β_NFP={model_treat.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_treat.pvalues['nfp_surprise_std_dm2']:.4f}), N={int(model_treat.nobs):,}")
print(f"Control only:            β_NFP={model_ctrl.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_ctrl.pvalues['nfp_surprise_std_dm2']:.4f}), N={int(model_ctrl.nobs):,}")
print(f"Binary robustness:       β_NFP={model_bin.params['nfp_surprise_std_dm2']:.4f} "
      f"(p={model_bin.pvalues['nfp_surprise_std_dm2']:.4f})")
print("\nDone.")
