"""
========================================================================================
Thermal Dimorphism — Validation Module
========================================================================================
Validates thermal_dimorphism.py against the ASHRAE Global Thermal Comfort
Database II (CBE-II).
Source  : https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II
CSV     : decompressed_data.csv (109 033 rows, v2.1)

COLUMN MAPPING (CBE-II):
  gender            : 'female' / 'male'
  ta / top / tr     : air / operative / mean radiant temperature (°C)
  rh / vel          : relative humidity (%) / air velocity (m/s)
  met / clo         : metabolic rate (met) / clothing insulation (clo)
  thermal_sensation : ASHRAE vote (−3 to +3)
  season            : 'winter' / 'spring' / 'summer' / 'autumn' / 'fall'
  koppen_geiger     : Köppen climate classification (optional — for EU filter)
========================================================================================
"""

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

VERSION = "v15.0"

# CBE-II database search paths (extend if needed)
POSSIBLE_PATHS = [
    "decompressed_data.csv",
    Path.home() / "Desktop" / "decompressed_data.csv",
    Path.home() / "Downloads" / "decompressed_data.csv",
    Path.home() / "Desktop" / "ashrae-db-II" / "decompressed_data.csv",
    Path.home() / "Documents" / "ashrae-db-II" / "decompressed_data.csv",
]

# Karjalainen 2012 reference
# (exact: 0.943 − 0.812 from Table 3, Karjalainen 2012, Indoor Air 22(2))
KARJALAINEN_DELTA_CLO = 0.131

# Biophysical constants — must stay in sync with thermal_dimorphism.py
H_EFF     = 10.105           # W/m²K  ISO 7730/Fanger
PA_REF    = 1.2              # kPa    standard office humidity
KCAL_TO_W = 4184.0 / 86400.0

# SWAN logistic constants
T_SKIN_N_M      = 33.5
T_SKIN_N_F_BASE = 33.2
T_SKIN_LUTEAL_D = -0.3
AGE_MEDIAN_SWAN = 52.54
KAPPA_SWAN      = 1.1
AGE_DEFAULT     = 42.6       # EU active workforce mean (Eurostat lfsa_egan 2022)

# Sigmoid fat fraction parameters
SIGMOID_PARAMS = {
    "male":   {"pmin": 0.03, "pmax": 0.55, "k": 0.15, "B0": 24.5},
    "female": {"pmin": 0.12, "pmax": 0.65, "k": 0.14, "B0": 24.0},
}

# Köppen codes for Western European temperate zones (Cfb/Dfb)
KOPPEN_EU_TEMPERATE = {"Cfb", "Dfb", "Cfc", "Dfc"}

# =============================================================================
# 2. DATA LOADING & CLEANING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    print(f"  Loading : {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters CBE-II to office-like sedentary observations.
    Keeps only rows with valid gender, temperature, and thermal sensation.
    """
    df = df.dropna(subset=["gender", "ta", "thermal_sensation"]).copy()
    df["gender"] = df["gender"].str.strip().str.lower()
    df = df[df["gender"].isin(["female", "male"])]

    for col in ["ta", "top", "tr", "rh", "vel", "met", "clo",
                "thermal_sensation", "age", "ht", "wt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["t_comfort"] = df["top"].fillna(df["ta"]) if "top" in df.columns else df["ta"]
    df = df[(df["t_comfort"] >= 16) & (df["t_comfort"] <= 34)]
    df = df[(df["thermal_sensation"] >= -3) & (df["thermal_sensation"] <= 3)]

    if "met" in df.columns:
        df = df[df["met"].between(0.8, 1.8) | df["met"].isna()]
    if "clo" in df.columns:
        df = df[df["clo"].between(0.3, 1.5) | df["clo"].isna()]

    print(f"  Cleaned : {df.shape[0]:,} rows ({len(df[df['gender']=='female']):,} F, "
          f"{len(df[df['gender']=='male']):,} M)")
    return df


def filter_european_temperate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restricts to Western European temperate zones (Köppen Cfb/Dfb).
    Used for the geographic robustness check (§4.6) — eliminates the
    hemispheric aggregation artefact causing spring/autumn ΔT inversions.
    Falls back to full dataset if Köppen column is unavailable.
    """
    koppen_col = next((c for c in df.columns
                       if "koppen" in c.lower() or "climate" in c.lower()), None)
    if koppen_col is None:
        print("  [WARN] No Köppen column found — geographic filter unavailable.")
        return df
    mask = df[koppen_col].str[:3].isin(KOPPEN_EU_TEMPERATE)
    filtered = df[mask].copy()
    print(f"  EU Cfb/Dfb filter : {len(filtered):,} rows "
          f"({mask.sum()/len(df)*100:.1f}% of full dataset)")
    return filtered

# =============================================================================
# 3. T_NEUTRAL ESTIMATION (bootstrap OLS)
# =============================================================================

def compute_t_neutral(df: pd.DataFrame, temp_col: str = "t_comfort",
                      n_boot: int = 1000, seed: int = 7730) -> dict:
    """
    Estimates T_neutral per sex via OLS regression (TSV = m·T + b → T* = −b/m).
    Bootstrap CI from 1000 resamples.
    Also tests whether T_neutral_F ≠ T_neutral_M via: (i) a Welch t-test on the
    bootstrap distributions of the intercepts (precision of the means, large t),
    and (ii) the statistically appropriate bootstrap of the *difference* in
    intercepts (reported in the manuscript), corroborated by a pooled OLS
    sex×temperature interaction.
    """
    results = {}
    t_neutral_by_sex = {}

    for gender in ["male", "female"]:
        sub = df[df["gender"] == gender].dropna(subset=[temp_col, "thermal_sensation"])
        N = len(sub)
        if N < 100:
            results[gender] = {"N": N, "error": "insufficient data"}
            continue

        x, y = sub[temp_col].values, sub["thermal_sensation"].values
        m, b  = np.polyfit(x, y, 1)
        t_neu = -b / m if abs(m) > 1e-6 else np.nan

        rng   = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(N, N, replace=True)
            c = np.polyfit(x[idx], y[idx], 1)
            if abs(c[0]) > 1e-6:
                boots.append(-c[1] / c[0])
        boots = np.array(boots)
        t_neutral_by_sex[gender] = boots

        results[gender] = {
            "N":            int(N),
            "T_neutral":    round(float(t_neu), 3),
            "CI_95_lo":     round(float(np.percentile(boots, 2.5)),  3),
            "CI_95_hi":     round(float(np.percentile(boots, 97.5)), 3),
            "SE_bootstrap": round(float(boots.std()), 4),
            "regression_m": round(float(m), 5),
            "regression_b": round(float(b), 4),
            "mean_vote":    round(float(y.mean()), 4),
            "mean_temp":    round(float(x.mean()), 3),
        }

    # Welch t-test on bootstrap distributions
    if "male" in t_neutral_by_sex and "female" in t_neutral_by_sex:
        wt, wp = stats.ttest_ind(t_neutral_by_sex["female"],
                                  t_neutral_by_sex["male"], equal_var=False)
        results["welch_bootstrap"] = {
            "t_stat": round(float(wt), 4), "p_value": float(wp),
            "significant_0.05": bool(wp < 0.05),
        }

    # ------------------------------------------------------------------
    # Proper inter-sex test of the neutral-temperature gap.
    # The Welch above operates on the *bootstrap distributions* of the
    # intercepts, so its t reflects the precision of the means, not a
    # per-individual effect. The statistically appropriate test is a
    # bootstrap of the DIFFERENCE in regression intercepts (manuscript).
    # ------------------------------------------------------------------
    m_sub = df[df["gender"] == "male"].dropna(subset=[temp_col, "thermal_sensation"])
    f_sub = df[df["gender"] == "female"].dropna(subset=[temp_col, "thermal_sensation"])
    if len(m_sub) >= 100 and len(f_sub) >= 100:
        xm, ym = m_sub[temp_col].values, m_sub["thermal_sensation"].values
        xf, yf = f_sub[temp_col].values, f_sub["thermal_sensation"].values
        cm0, cf0 = np.polyfit(xm, ym, 1), np.polyfit(xf, yf, 1)
        delta_obs = (-cf0[1] / cf0[0]) - (-cm0[1] / cm0[0])

        rng_d = np.random.default_rng(seed)
        B = 2000
        diffs = np.empty(B)
        for i in range(B):
            im  = rng_d.integers(0, len(xm), len(xm))
            iff = rng_d.integers(0, len(xf), len(xf))
            cm = np.polyfit(xm[im], ym[im], 1)
            cf = np.polyfit(xf[iff], yf[iff], 1)
            diffs[i] = (-cf[1] / cf[0]) - (-cm[1] / cm[0])
        se = float(diffs.std(ddof=1))
        z  = float(delta_obs / se) if se > 0 else float("nan")
        results["welch_difference"] = {
            "delta_T":  round(float(delta_obs), 4),
            "SE":       round(se, 4),
            "z_stat":   round(z, 3),
            "p_value":  float(2 * stats.norm.sf(abs(z))),
            "CI_95_lo": round(float(np.percentile(diffs, 2.5)), 3),
            "CI_95_hi": round(float(np.percentile(diffs, 97.5)), 3),
            "n_boot":   B,
        }

        # Pooled OLS corroboration: TSV ~ Tc * Female (centred temperature)
        x_all = np.concatenate([xm, xf]); y_all = np.concatenate([ym, yf])
        fem   = np.concatenate([np.zeros(len(xm)), np.ones(len(xf))])
        tc    = x_all - x_all.mean()
        X     = np.column_stack([np.ones_like(tc), tc, fem, tc * fem])
        beta, *_ = np.linalg.lstsq(X, y_all, rcond=None)
        resid = y_all - X @ beta
        dof   = len(y_all) - X.shape[1]
        sigma2 = float(resid @ resid) / dof
        se_b  = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        results["pooled_regression"] = {
            "interaction_coef": round(float(beta[3]), 5),
            "interaction_t":    round(float(beta[3] / se_b[3]), 2),
            "interaction_p":    float(2 * stats.t.sf(abs(beta[3] / se_b[3]), dof)),
            "female_coef":      round(float(beta[2]), 5),
            "female_p":         float(2 * stats.t.sf(abs(beta[2] / se_b[2]), dof)),
            "dof":              int(dof),
        }

    return results


def compute_t_neutral_by_season(df: pd.DataFrame) -> dict:
    """T_neutral per sex for each available season."""
    if "season" not in df.columns:
        return {}
    season_map = {"winter": "Winter", "spring": "Spring",
                  "summer": "Summer", "autumn": "Autumn", "fall": "Autumn"}
    df = df.copy()
    df["season_std"] = df["season"].str.lower().map(season_map)
    return {
        s: compute_t_neutral(df[df["season_std"] == s])
        for s in ["Winter", "Spring", "Summer", "Autumn"]
        if (df["season_std"] == s).sum() >= 200
    }


def compute_t_neutral_by_age(df: pd.DataFrame,
                              brackets: list = None) -> list[dict]:
    """
    T_neutral stratified by decade — isolates age effects in observed data.
    Complements the simulated age stratification (§4.2).
    """
    if "age" not in df.columns:
        return []
    if brackets is None:
        brackets = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 75)]
    results = []
    for lo, hi in brackets:
        sub = df[(df["age"] >= lo) & (df["age"] < hi)]
        if len(sub) < 200:
            continue
        tn = compute_t_neutral(sub, n_boot=500)
        entry = {"bracket": f"{lo}-{hi}", "N": len(sub)}
        for g in ["male", "female"]:
            if g in tn and "T_neutral" in tn[g]:
                entry[f"T_neutral_{g}"] = tn[g]["T_neutral"]
                entry[f"CI_lo_{g}"]     = tn[g]["CI_95_lo"]
                entry[f"CI_hi_{g}"]     = tn[g]["CI_95_hi"]
        if "T_neutral_female" in entry and "T_neutral_male" in entry:
            entry["delta_T_observed"] = round(
                entry["T_neutral_female"] - entry["T_neutral_male"], 3)
        results.append(entry)
    return results

# =============================================================================
# 4. INDIVIDUAL PHYSIOLOGICAL PREDICTION SOLVER
# =============================================================================

def predict_t_opt_individual(row) -> float:
    """
    Row-level T_opt predictor for DataFrame.apply().
    Mirrors the biophysical solver of thermal_dimorphism.py
    using measured ht/wt/age/clo from CBE-II observations.
    """
    h   = row.get("ht",  np.nan)
    w   = row.get("wt",  np.nan)
    age = row.get("age", np.nan)
    if pd.isna(h) or pd.isna(w) or h <= 0 or w <= 0:
        return np.nan
    if pd.isna(age):
        age = AGE_DEFAULT

    sex = row.get("gender", "male")
    s   = 5.0 if sex == "male" else -161.0
    clo = row.get("clo", 0.7)
    if pd.isna(clo):
        clo = 0.7

    p   = SIGMOID_PARAMS[sex]
    bsa = 0.007184 * (w**0.425) * ((h * 100)**0.725)
    met_val = row.get("met", 1.2)
    if pd.isna(met_val):
        met_val = 1.2

    q    = (10.0*w + 6.25*(h*100) - 5.0*age + s) * KCAL_TO_W * (met_val + 0.25)
    flux = q / bsa

    if sex == "male":
        t_sk = T_SKIN_N_M
    else:
        t_pre  = 0.5*T_SKIN_N_F_BASE + 0.5*(T_SKIN_N_F_BASE + T_SKIN_LUTEAL_D)
        t_post = T_SKIN_N_F_BASE
        t_sk   = t_pre + (t_post - t_pre) / (
                     1.0 + np.exp(-KAPPA_SWAN * (age - AGE_MEDIAN_SWAN)))

    e_diff    = max(0.0, 3.05e-3 * (5733.0 - 6.99*flux - PA_REF*1000.0))
    flux_evap = e_diff + max(0.0, 0.42*(flux - 58.15))
    r_clo     = clo * 0.155

    ta = 23.0
    for _ in range(10):
        alpha = 0.0014*(34.0 - ta) + 0.0173*(5.87 - PA_REF)
        fd    = max(0.0, flux - alpha*flux - flux_evap)
        ta_n  = max(10.0, min(t_sk - fd*r_clo - fd/H_EFF, 40.0))
        if abs(ta_n - ta) < 1e-4:
            return ta_n
        ta = ta_n
    return ta


def validate_individual_predictions(df: pd.DataFrame) -> dict:
    """
    Validates T_opt_predicted vs T_neutral observed (TSV ≈ 0 subset).
    Metrics: MAE, RMSE, Bias + bootstrap CI, Pearson r + p-value, R².
    """
    sub = df.dropna(subset=["ht", "wt", "thermal_sensation", "t_comfort"]).copy()
    sub = sub[(sub["ht"] > 1.0) & (sub["ht"] < 2.5) &
              (sub["wt"] > 30)  & (sub["wt"] < 200)]
    if len(sub) < 50:
        return {"N_with_ht_wt": len(sub), "status": "insufficient_data"}

    sub["t_opt_predicted"] = sub.apply(predict_t_opt_individual, axis=1)
    sub = sub.dropna(subset=["t_opt_predicted"])
    neutral = sub[sub["thermal_sensation"].abs() <= 0.5]
    status  = "few_neutral_obs" if len(neutral) < 20 else "ok"

    if len(neutral) == 0:
        return {"N_with_ht_wt": len(sub), "N_neutral_obs": 0, "status": status}

    errors = neutral["t_opt_predicted"] - neutral["t_comfort"]
    mae    = float(errors.abs().mean())
    rmse   = float(np.sqrt((errors**2).mean()))
    bias   = float(errors.mean())

    ss_res = float((errors**2).sum())
    ss_tot = float(((neutral["t_comfort"] - neutral["t_comfort"].mean())**2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng   = np.random.default_rng(7730)
    boots = [np.mean(rng.choice(errors.values, len(errors), replace=True))
             for _ in range(1000)]
    bias_lo, bias_hi = np.percentile(boots, [2.5, 97.5])

    r_stat, p_pearson = stats.pearsonr(neutral["t_opt_predicted"],
                                        neutral["thermal_sensation"])
    return {
        "N_with_ht_wt":   int(len(sub)),
        "N_neutral_obs":  int(len(neutral)),
        "status":         status,
        "MAE_C":          round(mae,  3),
        "RMSE_C":         round(rmse, 3),
        "bias_C":         round(bias, 3),
        "bias_ci_lo":     round(float(bias_lo), 3),
        "bias_ci_hi":     round(float(bias_hi), 3),
        "R2":             round(r2,   4),
        "corr_topt_vote": round(float(r_stat), 4),
        "p_pearson":      float(p_pearson),
    }

# =============================================================================
# 5. CLOTHING STATISTICS
# =============================================================================

def clothing_stats_by_sex(df: pd.DataFrame) -> dict:
    """Clo summary statistics by sex + ΔClo vs Karjalainen reference."""
    results = {}
    for gender in ["male", "female"]:
        sub = df[df["gender"] == gender].dropna(subset=["clo"])
        if len(sub) < 100:
            continue
        results[gender] = {
            "N":    int(len(sub)),
            "mean": round(float(sub["clo"].mean()),         3),
            "sd":   round(float(sub["clo"].std()),          3),
            "p25":  round(float(sub["clo"].quantile(0.25)), 3),
            "p50":  round(float(sub["clo"].median()),       3),
            "p75":  round(float(sub["clo"].quantile(0.75)), 3),
        }
    if "male" in results and "female" in results:
        delta = round(results["male"]["mean"] - results["female"]["mean"], 3)
        results["delta_clo_observed"]  = delta
        results["karjalainen_assumed"] = KARJALAINEN_DELTA_CLO
        results["ratio_pct"] = round(abs(delta / KARJALAINEN_DELTA_CLO) * 100, 1)
    return results

# =============================================================================
# 6. EXPORT HELPERS
# =============================================================================

def get_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _fmt_p(p) -> str:
    if p is None:
        return "N/A"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

# =============================================================================
# 7. MAIN EXECUTION PIPELINE
# =============================================================================

if __name__ == "__main__":
    SEP = "=" * 78

    # ── Database discovery ───────────────────────────────────────────────────
    cbe_path = next((str(p) for p in POSSIBLE_PATHS if Path(p).exists()), None)
    if cbe_path is None:
        print("  [ERROR] CBE-II database not found. Execution halted.")
        print("  Searched:", [str(p) for p in POSSIBLE_PATHS])
        raise SystemExit(1)

    print(SEP)
    print(f"  Thermal Dimorphism — Validation Module {VERSION}")
    print(f"  Database : {cbe_path}")
    print(SEP)

    df_raw = load_data(cbe_path)
    df     = clean_data(df_raw)

    # ── STEP 1 — DATA AUDIT ─────────────────────────────────────────────────
    print(f"\n── STEP 1 — DATA AUDIT " + "─"*52)
    gc = df["gender"].value_counts()
    print(f"  Total usable rows : {len(df):,}")
    for g, n in gc.items():
        print(f"  {g:8s}          : {n:,} ({n/len(df)*100:.1f}%)")

    # ── STEP 2 — T_NEUTRAL OVERALL ──────────────────────────────────────────
    print(f"\n── STEP 2 — T_NEUTRAL BY SEX (overall) " + "─"*35)
    tn = compute_t_neutral(df)
    for g in ["male", "female"]:
        if g not in tn or "error" in tn[g]:
            continue
        r = tn[g]
        print(f"\n  {g.upper()} (N={r['N']:,})")
        print(f"    T_neutral : {r['T_neutral']:.3f}°C "
              f"[95% CI: {r['CI_95_lo']:.2f}–{r['CI_95_hi']:.2f}]  "
              f"SE={r['SE_bootstrap']:.4f}°C")

    if "male" in tn and "female" in tn:
        delta_obs = tn["female"]["T_neutral"] - tn["male"]["T_neutral"]
        print(f"\n  ΔT observed (F−M) : {delta_obs:+.3f}°C")
        if "welch_bootstrap" in tn:
            wb = tn["welch_bootstrap"]
            print(f"  Welch (bootstrap) : t={wb['t_stat']:.3f}  "
                  f"p={_fmt_p(wb['p_value'])}  "
                  f"significant={'✓' if wb['significant_0.05'] else '✗'}")
        if "welch_difference" in tn:
            wd = tn["welch_difference"]
            print(f"  Gap significance  : ΔT={wd['delta_T']:+.3f}°C  "
                  f"z={wd['z_stat']:.2f}  p={_fmt_p(wd['p_value'])}  "
                  f"[95% CI: {wd['CI_95_lo']:+.2f}, {wd['CI_95_hi']:+.2f}]  "
                  f"(bootstrap of the difference, B={wd['n_boot']})")
        if "pooled_regression" in tn:
            pr = tn["pooled_regression"]
            print(f"  Pooled regression : sex×T interaction p={_fmt_p(pr['interaction_p'])}  "
                  f"(t={pr['interaction_t']:.2f})")

    # ── STEP 2.5 — LEVENE TEST ───────────────────────────────────────────────
    print(f"\n── STEP 2.5 — LEVENE TEST (neutral subset) " + "─"*31)
    neut_m = df[(df["gender"]=="male")   & (df["thermal_sensation"].abs()<=0.5)]["t_comfort"].dropna()
    neut_f = df[(df["gender"]=="female") & (df["thermal_sensation"].abs()<=0.5)]["t_comfort"].dropna()
    if len(neut_m) > 0 and len(neut_f) > 0:
        W, pW = stats.levene(neut_m, neut_f)
        print(f"  Levene W={W:.3f}  p={_fmt_p(pW)}  "
              f"(N_M={len(neut_m):,}  N_F={len(neut_f):,})")

    # ── STEP 3 — SEASONAL T_NEUTRAL ─────────────────────────────────────────
    print(f"\n── STEP 3 — T_NEUTRAL BY SEASON " + "─"*43)
    tn_seasonal = compute_t_neutral_by_season(df)
    if tn_seasonal:
        print(f"  {'Season':<8} │ {'T_neut_M':>10} │ {'T_neut_F':>10} │ {'ΔT_obs':>8}")
        print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}")
        for s in ["Winter", "Spring", "Summer", "Autumn"]:
            if s not in tn_seasonal:
                continue
            rs = tn_seasonal[s]
            if "male" not in rs or "female" not in rs:
                continue
            tm  = rs["male"]["T_neutral"]
            tf  = rs["female"]["T_neutral"]
            dt  = tf - tm
            print(f"  {s:<8} │ {tm:>10.3f} │ {tf:>10.3f} │ {dt:>+8.3f}")
    else:
        print("  No season column found in the dataset.")

    # ── STEP 3b — GEOGRAPHIC FILTER (EU Cfb/Dfb) ────────────────────────────
    print(f"\n── STEP 3b — GEOGRAPHIC FILTER (EU Köppen Cfb/Dfb) " + "─"*23)
    df_eu = filter_european_temperate(df)
    if len(df_eu) < len(df):
        tn_eu          = compute_t_neutral(df_eu)
        tn_eu_seasonal = compute_t_neutral_by_season(df_eu)
        if "male" in tn_eu and "female" in tn_eu:
            dobs_eu = tn_eu["female"]["T_neutral"] - tn_eu["male"]["T_neutral"]
            print(f"  ΔT observed EU-only : {dobs_eu:+.3f}°C "
                  f"(vs global {delta_obs:+.3f}°C)")
        if tn_eu_seasonal:
            print(f"  {'Season':<8} │ {'ΔT_obs_EU':>12}")
            for s in ["Winter", "Spring", "Summer", "Autumn"]:
                if s not in tn_eu_seasonal:
                    continue
                rs = tn_eu_seasonal[s]
                if "male" not in rs or "female" not in rs:
                    continue
                dt   = rs["female"]["T_neutral"] - rs["male"]["T_neutral"]
                sign = "✓ positive" if dt > 0 else "✗ INVERSION"
                print(f"  {s:<8} │ {dt:>+12.3f}°C  {sign}")

    # ── STEP 3c — AGE-STRATIFIED T_NEUTRAL ──────────────────────────────────
    print(f"\n── STEP 3c — AGE-STRATIFIED T_NEUTRAL (CBE observed) " + "─"*21)
    age_results = compute_t_neutral_by_age(df)
    if age_results:
        print(f"  {'Bracket':<8} {'N':>7} {'T_neu_F':>9} {'T_neu_M':>9} {'ΔT_obs':>8}")
        for r in age_results:
            tf = r.get("T_neutral_female", float("nan"))
            tm = r.get("T_neutral_male",   float("nan"))
            dt = r.get("delta_T_observed", float("nan"))
            print(f"  {r['bracket']:<8} {r['N']:>7,} {tf:>9.3f} {tm:>9.3f} {dt:>+8.3f}°C")
    else:
        print("  No age column found in the dataset.")

    # ── STEP 4 — CLO STATISTICS ──────────────────────────────────────────────
    print(f"\n── STEP 4 — CLO STATISTICS " + "─"*48)
    clo = clothing_stats_by_sex(df)
    for g in ["male", "female"]:
        if g in clo:
            r = clo[g]
            print(f"  {g:8s}: mean={r['mean']:.3f}  SD={r['sd']:.3f}  "
                  f"[P25={r['p25']:.3f}  P50={r['p50']:.3f}  P75={r['p75']:.3f}]  N={r['N']:,}")
    if "delta_clo_observed" in clo:
        print(f"\n  ΔClo observed (M−F) : {clo['delta_clo_observed']:+.3f}")
        print(f"  ΔClo Karjalainen    : {clo['karjalainen_assumed']:+.3f}")
        print(f"  Ratio               : {clo['ratio_pct']:.0f}% of assumed "
              f"({'⚠ low — behavioural convergence' if clo['ratio_pct'] < 20 else 'consistent'})")

    # ── STEP 5 — INDIVIDUAL T_opt PREDICTIONS ────────────────────────────────
    print(f"\n── STEP 5 — INDIVIDUAL T_opt PREDICTIONS " + "─"*33)
    ind = validate_individual_predictions(df)
    print(f"  N with ht/wt   : {ind.get('N_with_ht_wt', 0):,}")
    print(f"  N neutral obs  : {ind.get('N_neutral_obs', 0):,}")
    print(f"  MAE            : {ind.get('MAE_C')}°C")
    print(f"  RMSE           : {ind.get('RMSE_C')}°C")
    print(f"  R²             : {ind.get('R2')}")
    bc, bl, bh = ind.get("bias_C"), ind.get("bias_ci_lo"), ind.get("bias_ci_hi")
    print(f"  Bias           : {bc:+.3f}°C  [IC 95% bootstrap: {bl:+.3f}; {bh:+.3f}°C]")
    p_p = ind.get("p_pearson")
    print(f"  Corr(T_opt,V)  : {ind.get('corr_topt_vote')}  "
          f"(p={_fmt_p(p_p)})  [neutral obs only, N={ind.get('N_neutral_obs',0):,}]")

    # ── EXPORT JSON ──────────────────────────────────────────────────────────
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    script_hash = get_file_hash(__file__)

    export = {
        "metadata": {
            "module":        f"Validation_{VERSION}",
            "timestamp":     datetime.now().isoformat(),
            "script_sha256": script_hash,
            "cbe_n_rows":    len(df),
        },
        "t_neutral_overall":    {g: {k: v for k, v in r.items()}
                                  for g, r in tn.items()},
        "t_neutral_seasonal":   tn_seasonal,
        "t_neutral_by_age":     age_results,
        "individual_validation": ind,
        "clothing_stats":       clo,
    }
    json_path = out / f"validation_{VERSION.replace('.', '_')}_results.json"
    json_path.write_text(json.dumps(export, indent=4, default=str))

    print(f"\n[SECURITÉ] SHA-256 : {script_hash}")
    print(f"[OUTPUT]   JSON    → {json_path}")
    print(SEP)
