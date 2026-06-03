"""
=============================================================================
Thermal Dimorphism Simulation
=============================================================================
PROJECT  : Sexual Dimorphism in Thermal Comfort: A Reassessment of ISO 7730
SCENARIO : Office / Sedentary — Multi-season, EU active workforce
VERSION  : 15.0 — Full architectural refactor (2026-05-30)

=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json, csv, hashlib, uuid
from scipy.stats import ks_2samp, ttest_ind
from scipy.optimize import brentq

# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================

H_EFF     = 10.105   # W/m²K  — ISO 7730 / Fanger 1970
PA_REF    = 1.2      # kPa    — standard office humidity

# Neutral skin temperature (sex-dimorphic)
T_SKIN_N_M      = 33.5    # °C  — Hardy & DuBois 1938, males
T_SKIN_N_F_BASE = 33.2    # °C  — female baseline, calibrated (Lan 2008; Vellei 2025)
T_SKIN_LUTEAL_D = -0.3    # °C  — progestérone lutéale, Charkoudian 2014
AGE_MEDIAN_SWAN = 52.54   # yr  — median menopausal age, Gold et al. 2013
KAPPA_SWAN      = 1.1     # logistic slope, 5-yr perimenopause window

CLO_UNDERWEAR_F = 0.05    # clo — underwear correction (Smallcombe et al. 2021; ISO 9920)

MET_BASE  = 1.2
MET_CV    = 0.15          # Garby & Astrup 1987
KCAL_TO_W = 4184.0 / 86400.0

# Simulation parameters
N_GEN      = 200_000
BASE_SEED  = 7730

# ── Overrides via environment variables (run_pipeline.py --quick) ─────────
import os as _os
if _os.environ.get("TD_QUICK_MODE"):
    N_GEN = int(_os.environ.get("TD_N_GEN", 20_000))
del _os
H_BOUNDS   = (1.20, 2.30)
BMI_BOUNDS = (12.0, 58.0)
GTB_RESOLUTION = 0.25     # °C — BACnet/KNX standard addressing step

# Anthropometric distributions — NCD-RisC 2016 (Lancet 387:1377)
# Western European working-age population, zones Cfb/Dfb
# M: 26.5 kg/m² (FR 26.4, DE 27.0, NL 25.9, BE 26.2, SE 26.7)
# F: 25.5 kg/m² (FR 25.2, DE 25.6, NL 25.0, BE 24.7, SE 24.9)
BODY_PARAMS = {
    "Male": {
        "phi_min": 0.03, "phi_max": 0.55, "k": 0.15, "B0": 24.5,
        "height_mean": 1.75, "height_sd": 0.07,
        "bmi_mean": 26.5,    "bmi_sd": 3.5,
        "mifflin_s": 5.0,
    },
    "Female": {
        "phi_min": 0.12, "phi_max": 0.65, "k": 0.14, "B0": 24.0,
        "height_mean": 1.63, "height_sd": 0.06,
        "bmi_mean": 25.5,    "bmi_sd": 4.0,
        "mifflin_s": -161.0,
    },
}

# Seasonal Clo — Karjalainen 2012, Table 3 + underwear correction
SEASONAL_CLO = {
    "Winter": {"Male": (0.990, 0.14), "Female": (0.850 + CLO_UNDERWEAR_F, 0.21)},
    "Spring": {"Male": (0.950, 0.14), "Female": (0.777 + CLO_UNDERWEAR_F, 0.21)},
    "Summer": {"Male": (0.890, 0.14), "Female": (0.663 + CLO_UNDERWEAR_F, 0.21)},
    "Autumn": {"Male": (0.940, 0.14), "Female": (0.757 + CLO_UNDERWEAR_F, 0.21)},
}

# Age distribution — Eurostat lfsa_egan (2022), EU active workforce
AGE_BRACKETS = [(20, 30, 0.18), (30, 40, 0.25), (40, 50, 0.26),
                (50, 60, 0.23), (60, 65, 0.08)]

# Parametric uncertainties for quadrature (§2.10)
UNCERTAINTY_PARAMS = {
    "s_F":    (-161.0,  15.0,  "s female [kcal/d] — Frankenfield 2005"),
    "s_M":    (   5.0,  15.0,  "s male [kcal/d] — Frankenfield 2005"),
    "clo_s1": (   0.7,   0.07,        "Clo S1 [±10 %, ISO 9920]"),
    "h_eff":  (H_EFF,  H_EFF * 0.05, "h_eff [±5 %, ISO 7730]"),
    # σ = SE of published mean, NOT SWAN temporal amplitude (already in SWAN model)
    "tskin_f": (T_SKIN_N_F_BASE, 0.05, "T_skin,n F [SE ±0.05°C]"),
    "tskin_m": (T_SKIN_N_M,      0.05, "T_skin,n M [SE ±0.05°C]"),
}

# =============================================================================
# 2. BIOPHYSICAL FUNCTIONS
# =============================================================================

def sigmoid_fat(bmi: np.ndarray, bp: dict) -> np.ndarray:
    """Fat fraction — logistic adaptation of Deurenberg 1991 + Fujii 2021."""
    return bp["phi_min"] + (bp["phi_max"] - bp["phi_min"]) / (
        1.0 + np.exp(-bp["k"] * (bmi - bp["B0"])))


def bsa_dubois(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Body Surface Area — DuBois & DuBois 1916 (ISO 7730 standard)."""
    return 0.007184 * (w ** 0.425) * ((h * 100.0) ** 0.725)


def alpha_resp(ta: np.ndarray, pa: float = PA_REF) -> np.ndarray:
    """Respiratory heat loss fraction — ISO 7730 Eq. C.3-C.4."""
    return 0.0014 * (34.0 - ta) + 0.0173 * (5.87 - pa)


def t_skin_neutral(sex: str, ages: np.ndarray,
                   f_base: float = T_SKIN_N_F_BASE,
                   m_val:  float = T_SKIN_N_M) -> np.ndarray:
    """
    Neutral skin temperature — SWAN logistic model.
    Optional f_base / m_val allow parametric override for sensitivity analysis.
    """
    if sex == "Male":
        return np.full(len(ages), m_val)
    t_pre  = 0.5 * f_base + 0.5 * (f_base + T_SKIN_LUTEAL_D)   # pre-meno
    t_post = f_base                                               # post-meno
    return t_pre + (t_post - t_pre) / (
        1.0 + np.exp(-KAPPA_SWAN * (ages - AGE_MEDIAN_SWAN)))


def t_opt_from_flux(flux: np.ndarray, r_clo: np.ndarray,
                    t_skin_n: np.ndarray, sex: str = "Male",
                    h_eff_val: float = H_EFF,
                    delta_SD: float = 0.0, gamma: float = 0.30,
                    delta_CC: float = 0.0, r_clo_m: np.ndarray = None,
                    ta_est: float = 23.0, pa: float = PA_REF,
                    max_iter: int = 10, tol: float = 1e-4) -> np.ndarray:
    """
    Optimal ambient temperature [°C] — ISO 7730 dry-flux iterative solver.
    h_eff_val: convective coefficient override for sensitivity / quadrature.
    delta_SD / delta_CC: biophysical cursors (sudomotor / clothing convergence).
    """
    ta = np.full(len(flux), ta_est)
    e_diff = 3.05e-3 * (5733.0 - 6.99 * flux - pa * 1000.0)
    e_sw   = np.maximum(0.0, 0.42 * (flux - 58.15) *
                        (1.0 - delta_SD * gamma if sex == "Female" else 1.0))
    flux_evap = e_diff + e_sw
    r_clo_eff = (r_clo + delta_CC * (r_clo_m - r_clo)
                 if sex == "Female" and r_clo_m is not None and delta_CC > 0
                 else r_clo)
    for _ in range(max_iter):
        flux_dry = np.maximum(0.0, flux - alpha_resp(ta, pa) * flux - flux_evap)
        ta_new   = np.clip(t_skin_n - flux_dry * r_clo_eff - flux_dry / h_eff_val,
                           10.0, 40.0)
        if np.max(np.abs(ta_new - ta)) < tol:
            return ta_new
        ta = ta_new
    return ta

# =============================================================================
# 3. POPULATION SAMPLING
# =============================================================================

def sample_ages(n: int, rng: np.random.Generator) -> np.ndarray:
    """EU active workforce age distribution — Eurostat lfsa_egan 2022."""
    ages, idx = np.empty(n), 0
    for lo, hi, prop in AGE_BRACKETS:
        nb = min(int(round(n * prop)), n - idx)
        ages[idx:idx+nb] = rng.uniform(lo, hi, nb)
        idx += nb
    if idx < n:
        ages[idx:] = rng.uniform(20, 65, n - idx)
    rng.shuffle(ages)
    return np.clip(ages, 20.0, 65.0)


def simulate_population(sex: str, season: str, n: int, seed: int,
                        bmr_model: str = "mifflin",
                        delta_SD: float = 0.0, delta_CC: float = 0.0,
                        r_clo_m_ref: np.ndarray = None) -> dict:
    """Generates stochastic micro-cohort with log-normal BMI distribution."""
    rng = np.random.default_rng(seed)
    bp  = BODY_PARAMS[sex]
    clo_mean, clo_sd = SEASONAL_CLO[season][sex]
    m, v = bp["bmi_mean"], bp["bmi_sd"] ** 2
    bmi  = np.clip(rng.lognormal(np.log(m**2 / np.sqrt(v + m**2)),
                                  np.sqrt(np.log(1.0 + v / m**2)), n), *BMI_BOUNDS)
    h    = np.clip(rng.normal(bp["height_mean"], bp["height_sd"], n), *H_BOUNDS)
    w    = bmi * h ** 2
    fat  = sigmoid_fat(bmi, bp)
    lbm  = w * (1.0 - fat)
    ages = sample_ages(n, rng)
    if bmr_model == "mifflin":
        bmr = 10.0 * w + 6.25 * (h * 100.0) - 5.0 * ages + bp["mifflin_s"]
    elif bmr_model == "katch_mcardle":
        bmr = 370.0 + 21.6 * lbm
    else:
        raise ValueError(f"Unknown BMR model: {bmr_model}")
    met  = np.clip(rng.normal(MET_BASE, MET_BASE * MET_CV, n), 0.8, 1.8)
    q    = bmr * KCAL_TO_W * (met + 0.25)
    bsa  = bsa_dubois(w, h)
    flux = q / bsa
    clo_ind = (np.clip(rng.normal(clo_mean, clo_sd, n), 0.3, 1.5)
               if clo_sd > 0 else np.full(n, clo_mean))
    r_clo = clo_ind * 0.155
    t_sk_n = t_skin_neutral(sex, ages)
    t_opt  = t_opt_from_flux(flux, r_clo, t_sk_n, sex=sex,
                              delta_SD=delta_SD, delta_CC=delta_CC,
                              r_clo_m=r_clo_m_ref)
    return {"t_opt": t_opt, "flux_surface": flux, "flux_lbm": q / lbm,
            "clo": clo_ind, "age": ages, "t_skin_n": t_sk_n,
            "lbm": lbm, "bmi": bmi, "r_clo": r_clo}

# =============================================================================
# 4. STATISTICS & OUTPUT GENERATORS
# =============================================================================

def group_stats(pop_f: dict, pop_m: dict) -> dict:
    """Seasonal group statistics — includes Welch t-test and Cohen's d."""
    t_f, t_m = pop_f["t_opt"], pop_m["t_opt"]
    pooled   = np.sqrt((t_f.std()**2 + t_m.std()**2) / 2.0)
    ks_stat, ks_pval = ks_2samp(t_f, t_m)
    # Welch t-test (unequal variances) — appropriate for large sex-dimorphic cohorts
    welch_t, welch_p = ttest_ind(t_f, t_m, equal_var=False)
    return {
        "t_opt_mean_f": float(t_f.mean()),  "t_opt_sd_f": float(t_f.std()),
        "t_opt_mean_m": float(t_m.mean()),  "t_opt_sd_m": float(t_m.std()),
        "delta_t_opt":  float(t_f.mean() - t_m.mean()),
        "cohens_d":     float((t_f.mean() - t_m.mean()) / pooled),
        "ks_statistic": float(ks_stat),     "ks_pvalue":    float(ks_pval),
        "welch_t":      float(welch_t),     "welch_p":      float(welch_p),
        "mean_flux_f":  float(pop_f["flux_surface"].mean()),
        "mean_flux_m":  float(pop_m["flux_surface"].mean()),
        "mean_clo_f":   float(pop_f["clo"].mean()),
        "mean_clo_m":   float(pop_m["clo"].mean()),
        "t_skin_n_f":   float(pop_f["t_skin_n"].mean()),
        "t_skin_n_m":   float(pop_m["t_skin_n"].mean()),
        "lbm_flux_f":   float(pop_f["flux_lbm"].mean()),
        "lbm_flux_m":   float(pop_m["flux_lbm"].mean()),
        "lbm_paradox":  bool(pop_f["flux_lbm"].mean() > pop_m["flux_lbm"].mean()),
    }



def welch_mean_diff_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Welch 95% CI for (mean_a − mean_b) — Satterthwaite degrees of freedom.
    Complements ttest_ind: gives explicit [ci_lo, ci_hi] for the mean difference.
    """
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff   = a.mean() - b.mean()
    se     = np.sqrt(va / na + vb / nb)
    df     = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    from scipy.stats import t as t_dist
    tc = t_dist.ppf(1 - alpha / 2, df)
    _, p = ttest_ind(a, b, equal_var=False)
    return {"diff": float(diff), "ci_lo": float(diff - tc*se),
            "ci_hi": float(diff + tc*se), "se": float(se), "p": float(p)}


def bootstrap_delta_T(t_opt_f: np.ndarray, t_opt_m: np.ndarray,
                      n_boot: int = 1000, n_sample: int = 5000,
                      seed: int = BASE_SEED) -> dict:
    """
    Bootstrap CI for mean ΔT = T_opt_F − T_opt_M.
    Resamples from the existing population arrays (fast — no re-simulation).
    Returns CI_lo, CI_hi, SE, and the full bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    n = min(len(t_opt_f), len(t_opt_m))
    deltas = np.array([
        t_opt_f[rng.choice(n, n_sample, replace=True)].mean() -
        t_opt_m[rng.choice(n, n_sample, replace=True)].mean()
        for _ in range(n_boot)
    ])
    return {
        "mean":   float(deltas.mean()),
        "se":     float(deltas.std()),
        "ci_lo":  float(np.percentile(deltas, 2.5)),
        "ci_hi":  float(np.percentile(deltas, 97.5)),
        "n_boot": n_boot, "n_sample": n_sample,
    }



def compute_prediction_interval(t_opt_f: np.ndarray, t_opt_m: np.ndarray,
                                 alpha: float = 0.05) -> dict:
    """
    Distinguishes IC of the mean from Prediction Interval for individuals (§3.4).

    IC_mean  : precision of the estimated mean ΔT — very tight for large N
               ΔT_mean ± z × √(σ²_F/N_F + σ²_M/N_M)

    PI_indiv : range covering 95% of individual (T_opt_F_i − T_opt_M_j) pairs
               ΔT_mean ± z × √(σ²_F + σ²_M)   [Normal approx, independent draws]

    The PI is much wider than the IC — relevant for GTB design margin.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    mu = t_opt_f.mean() - t_opt_m.mean()
    se_mean  = np.sqrt(t_opt_f.var() / len(t_opt_f) + t_opt_m.var() / len(t_opt_m))
    se_indiv = np.sqrt(t_opt_f.var() + t_opt_m.var())
    return {
        "delta_mean":  float(mu),
        "ic_mean_lo":  float(mu - z * se_mean),
        "ic_mean_hi":  float(mu + z * se_mean),
        "se_mean":     float(se_mean),
        "pi_lo":       float(mu - z * se_indiv),
        "pi_hi":       float(mu + z * se_indiv),
        "se_indiv":    float(se_indiv),
    }


def stratify_by_age(pop_f: dict, pop_m: dict) -> list[dict]:
    """
    ΔT_opt stratified by Eurostat age brackets.
    Isolates SWAN transition effect: ΔT should increase in the 50-60 bracket.
    """
    results = []
    for lo, hi, _ in AGE_BRACKETS:
        mask_f = (pop_f["age"] >= lo) & (pop_f["age"] < hi)
        mask_m = (pop_m["age"] >= lo) & (pop_m["age"] < hi)
        tf = pop_f["t_opt"][mask_f]
        tm = pop_m["t_opt"][mask_m]
        if len(tf) < 50 or len(tm) < 50:
            continue
        pooled = np.sqrt((tf.std()**2 + tm.std()**2) / 2.0)
        _, wp = ttest_ind(tf, tm, equal_var=False)
        results.append({
            "bracket":  f"{lo}-{hi}",
            "n_f":      int(mask_f.sum()),
            "n_m":      int(mask_m.sum()),
            "T_opt_f":  round(float(tf.mean()), 3),
            "T_opt_m":  round(float(tm.mean()), 3),
            "delta_T":  round(float(tf.mean() - tm.mean()), 3),
            "cohens_d": round(float((tf.mean() - tm.mean()) / pooled), 3),
            "welch_p":  float(wp),
            "t_skin_f": round(float(pop_f["t_skin_n"][mask_f].mean()), 3),
        })
    return results


def stratify_by_bmi(pop_f: dict, pop_m: dict) -> list[dict]:
    """
    ΔT_opt stratified by BMI category (WHO classification).
    Quantifies the BMI × sex interaction on thermal optimum.
    """
    categories = [
        ("Normal (18.5–25)", 18.5, 25.0),
        ("Surpoids (25–30)", 25.0, 30.0),
        ("Obèse (≥30)",      30.0, 99.0),
    ]
    results = []
    for label, lo, hi in categories:
        mask_f = (pop_f["bmi"] >= lo) & (pop_f["bmi"] < hi)
        mask_m = (pop_m["bmi"] >= lo) & (pop_m["bmi"] < hi)
        tf = pop_f["t_opt"][mask_f]
        tm = pop_m["t_opt"][mask_m]
        if len(tf) < 50 or len(tm) < 50:
            continue
        pooled = np.sqrt((tf.std()**2 + tm.std()**2) / 2.0)
        _, wp = ttest_ind(tf, tm, equal_var=False)
        results.append({
            "category": label,
            "n_f":      int(mask_f.sum()),
            "n_m":      int(mask_m.sum()),
            "bmi_f":    round(float(pop_f["bmi"][mask_f].mean()), 2),
            "bmi_m":    round(float(pop_m["bmi"][mask_m].mean()), 2),
            "T_opt_f":  round(float(tf.mean()), 3),
            "T_opt_m":  round(float(tm.mean()), 3),
            "delta_T":  round(float(tf.mean() - tm.mean()), 3),
            "cohens_d": round(float((tf.mean() - tm.mean()) / pooled), 3),
            "welch_p":  float(wp),
        })
    return results



def build_gtb_continuous(season_pops: dict, results_s1: dict = None,
                          n_points: int = 200,
                          resolution: float = GTB_RESOLUTION) -> dict:
    """
    Continuous T_opt(p_f) curve — 200 points for smooth matplotlib rendering.
    Returns dict of numpy arrays keyed by season + 'p_f'.
    Each season has both exact (float) and operational (rounded to resolution) values.
    """
    p_f  = np.linspace(0.0, 1.0, n_points)
    out  = {"p_f": p_f}
    for s, pops in season_pops.items():
        mm, mf = pops["Male"]["t_opt"].mean(), pops["Female"]["t_opt"].mean()
        out[s]       = (1 - p_f) * mm + p_f * mf
        out[f"{s}_op"] = np.round(out[s] / resolution) * resolution
    if results_s1 is not None:
        mm1 = results_s1["Male"]["t_opt"].mean()
        mf1 = results_s1["Female"]["t_opt"].mean()
        out["S1"]    = (1 - p_f) * mm1 + p_f * mf1
        out["S1_op"] = np.round(out["S1"] / resolution) * resolution
    return out


def export_gtb_continuous_csv(curve: dict, out_dir: Path) -> None:
    """CSV of continuous T_opt vs p_f — import directly into Excel / matplotlib."""
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fieldnames = ["p_f_pct"] + [f"{s}_exact" for s in seasons] +                  [f"{s}_op" for s in seasons]
    if "S1" in curve:
        fieldnames += ["S1_exact", "S1_op"]
    with open(out_dir / "gtb_continuous_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(curve["p_f"]):
            row = {"p_f_pct": round(p * 100, 1)}
            for s in seasons:
                if s in curve:
                    row[f"{s}_exact"] = round(float(curve[s][i]), 3)
                    row[f"{s}_op"]    = round(float(curve[f"{s}_op"][i]), 2)
            if "S1" in curve:
                row["S1_exact"] = round(float(curve["S1"][i]), 3)
                row["S1_op"]    = round(float(curve["S1_op"][i]), 2)
            writer.writerow(row)


def build_gtb_lookup(season_pops: dict, results_s1: dict = None,
                     resolution: float = GTB_RESOLUTION) -> pd.DataFrame:
    """GTB lookup table T_opt(p_f, season). Resolution in °C (default 0.25)."""
    def round_res(v): return round(round(v / resolution) * resolution, 4)
    rows = []
    for p_f in np.linspace(0.0, 1.0, 21):
        row = {"p_female": p_f, "p_female_pct": int(round(p_f * 100))}
        for season, pops in season_pops.items():
            v = (1 - p_f) * pops["Male"]["t_opt"].mean() + p_f * pops["Female"]["t_opt"].mean()
            row[season] = round(v, 2)
            row[f"{season}_op"] = round_res(v)
        if results_s1 is not None:
            v_s1 = (1 - p_f) * results_s1["Male"]["t_opt"].mean() + p_f * results_s1["Female"]["t_opt"].mean()
            row["S1"] = round(v_s1, 2)
            row["S1_op"] = round_res(v_s1)
        rows.append(row)
    return pd.DataFrame(rows)


def percentile_table(pop_f: dict, pop_m: dict, p_f: float = 0.5) -> dict:
    """P5/P25/P50/P75/P95 mixed-cohort lookup for HVAC design loads."""
    n = len(pop_m["t_opt"])
    rng = np.random.default_rng(BASE_SEED)
    n_f, n_m = int(round(n * p_f)), n - int(round(n * p_f))
    mixed = np.concatenate([pop_m["t_opt"][rng.choice(n, n_m, replace=False)],
                             pop_f["t_opt"][rng.choice(n, n_f, replace=False)]])
    ps = np.percentile(mixed, [5, 25, 50, 75, 95])
    return {f"P{p}": round(float(v), 2) for p, v in zip([5, 25, 50, 75, 95], ps)}


def solve_residual_clo(pop: dict, target_t_opt: float, sex: str = "Female") -> dict:
    """Reverse solver: ΔClo needed to align T_opt with CBE target (Brent method)."""
    def objective(dc):
        return float(t_opt_from_flux(pop["flux_surface"],
                                     pop["r_clo"] + dc * 0.155,
                                     pop["t_skin_n"], sex=sex).mean() - target_t_opt)
    try:
        dc = brentq(objective, -0.5, 0.5, xtol=1e-4)
        if   abs(dc) <= 0.05: diag = "COHÉRENT (≤0.05 clo) — compatible CBE II."
        elif abs(dc) >= 0.10: diag = "STRUCTURAL FAILURE (≥0.10 clo) — irreducible bias."
        else:                  diag = "ZONE GRISE (0.05–0.10 clo) — incertitude résiduelle."
        return {"delta_clo": dc, "diagnosis": diag}
    except ValueError:
        return {"delta_clo": float("nan"), "diagnosis": "ERREUR SOLVEUR: pas de convergence."}


def simulate_s2_delta_T(season: str, s_F=-161.0, s_M=5.0,
                         h_eff_val=H_EFF,
                         tskin_f=T_SKIN_N_F_BASE, tskin_m=T_SKIN_N_M,
                         n=N_GEN, seed=BASE_SEED) -> float:
    """
    ΔT_S2(season) with parameter overrides — for seasonal parametric quadrature.
    Uses seasonal Clo values (asymmetric, from SEASONAL_CLO).
    """
    rng = np.random.default_rng(seed)
    means = {}
    for sex in ["Male", "Female"]:
        bp   = BODY_PARAMS[sex]
        s_v  = s_F if sex == "Female" else s_M
        clo_mean, clo_sd = SEASONAL_CLO[season][sex]
        m, v = bp["bmi_mean"], bp["bmi_sd"]**2
        bmi  = np.clip(rng.lognormal(np.log(m**2/np.sqrt(v+m**2)),
                                      np.sqrt(np.log(1+v/m**2)), n), *BMI_BOUNDS)
        h    = np.clip(rng.normal(bp["height_mean"], bp["height_sd"], n), *H_BOUNDS)
        w    = bmi * h**2
        ages = sample_ages(n, rng)
        bmr  = 10.0*w + 6.25*(h*100.0) - 5.0*ages + s_v
        met  = np.clip(rng.normal(MET_BASE, MET_BASE*MET_CV, n), 0.8, 1.8)
        flux = bmr * KCAL_TO_W * (met + 0.25) / bsa_dubois(w, h)
        clo_ind = (np.clip(rng.normal(clo_mean, clo_sd, n), 0.3, 1.5)
                   if clo_sd > 0 else np.full(n, clo_mean))
        t_sk = t_skin_neutral(sex, ages, f_base=tskin_f, m_val=tskin_m)
        t_opt = t_opt_from_flux(flux, clo_ind * 0.155, t_sk,
                                 sex=sex, h_eff_val=h_eff_val)
        means[sex] = float(t_opt.mean())
    return means["Female"] - means["Male"]


def compute_s2_seasonal_parametric_quadrature(season: str,
                                               verbose: bool = False) -> dict:
    """
    Full parametric quadrature for S2 seasonal ΔT (not just S1).
    Same 6 parameters as S1 quadrature; Clo values are seasonal (asymmetric).
    """
    from scipy.stats import norm
    z = norm.ppf(0.975)
    base_vals = {k: v[0] for k, v in UNCERTAINTY_PARAMS.items()}
    delta_base = simulate_s2_delta_T(season, n=N_GEN, seed=BASE_SEED)
    contribs = {}
    for pname, (base_val, sigma_i, label) in UNCERTAINTY_PARAMS.items():
        step = sigma_i / 2.0
        kw_hi = {**base_vals, pname: base_val + step}
        kw_lo = {**base_vals, pname: base_val - step}
        def _run(kw):
            return simulate_s2_delta_T(
                season, s_F=kw["s_F"], s_M=kw["s_M"],
                h_eff_val=kw["h_eff"],
                tskin_f=kw["tskin_f"], tskin_m=kw["tskin_m"],
                n=N_GEN, seed=BASE_SEED)
        deriv   = (_run(kw_hi) - _run(kw_lo)) / (2.0 * step)
        contrib = deriv * sigma_i
        contribs[pname] = {"sigma": sigma_i, "deriv": deriv, "contrib": contrib}
    sigma_total = float(np.sqrt(sum(c["contrib"]**2 for c in contribs.values())))
    ci_lo = delta_base - z * sigma_total
    ci_hi = delta_base + z * sigma_total
    if verbose:
        print(f"  {season:<8}: ΔT={delta_base:+.3f}°C  σ={sigma_total:.4f}°C  "
              f"[IC95%: {ci_lo:+.3f};{ci_hi:+.3f}°C]")
    return {"season": season, "delta_base": delta_base,
            "sigma_total": sigma_total, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "contributions": contribs}


def compute_cross_sensitivity_2nd_order(n: int = 20_000,
                                         seed: int = BASE_SEED) -> dict:
    """
    2nd-order cross-sensitivity: ∂²(ΔT_S1)/∂x_i∂x_j via centered finite differences.
    Formula: [f(+i,+j) - f(+i,-j) - f(-i,+j) + f(-i,-j)] / (4δ_i δ_j)

    Only the dominant pairs are computed (s_F×T_skin_F, s_F×Clo, s_M×T_skin_M).
    Higher N_GEN is used in main simulation; smaller n here to keep it fast.
    """
    base = {k: v[0] for k, v in UNCERTAINTY_PARAMS.items()}
    f0   = simulate_s1_delta_T(n=n, seed=seed)
    pairs = [("s_F", "tskin_f"), ("s_F", "clo_s1"), ("s_M", "tskin_m"),
             ("s_F", "s_M"),     ("tskin_f", "tskin_m")]
    results = {}
    for pi, pj in pairs:
        di = UNCERTAINTY_PARAMS[pi][1] * 0.5
        dj = UNCERTAINTY_PARAMS[pj][1] * 0.5
        def _f(**kw):
            return simulate_s1_delta_T(s_F=kw["s_F"], s_M=kw["s_M"],
                                        clo_s1=kw["clo_s1"], h_eff_val=kw["h_eff"],
                                        tskin_f=kw["tskin_f"], tskin_m=kw["tskin_m"],
                                        n=n, seed=seed)
        fpp = _f(**{**base, pi: base[pi]+di, pj: base[pj]+dj})
        fpm = _f(**{**base, pi: base[pi]+di, pj: base[pj]-dj})
        fmp = _f(**{**base, pi: base[pi]-di, pj: base[pj]+dj})
        fmm = _f(**{**base, pi: base[pi]-di, pj: base[pj]-dj})
        d2  = (fpp - fpm - fmp + fmm) / (4 * di * dj)
        si, sj = UNCERTAINTY_PARAMS[pi][1], UNCERTAINTY_PARAMS[pj][1]
        results[f"{pi}×{pj}"] = {
            "d2f":         round(d2, 6),
            "contrib_2nd": round(0.5 * d2 * si * sj, 6),  # Taylor term
            "sigma_i": si, "sigma_j": sj,
        }
    return results


# =============================================================================
# 5. QUADRATURE — PARAMETRIC UNCERTAINTY PROPAGATION
# =============================================================================

def simulate_s1_delta_T(s_F=-161.0, s_M=5.0, clo_s1=0.7,
                        h_eff_val=H_EFF, tskin_f=T_SKIN_N_F_BASE,
                        tskin_m=T_SKIN_N_M, n=N_GEN, seed=BASE_SEED) -> float:
    """ΔT_S1 = T_opt_F − T_opt_M with parameter overrides (for quadrature)."""
    rng = np.random.default_rng(seed)
    means = {}
    for sex in ["Male", "Female"]:
        bp    = BODY_PARAMS[sex]
        s_val = s_F if sex == "Female" else s_M
        m, v  = bp["bmi_mean"], bp["bmi_sd"]**2
        bmi   = np.clip(rng.lognormal(np.log(m**2/np.sqrt(v+m**2)),
                                       np.sqrt(np.log(1+v/m**2)), n), *BMI_BOUNDS)
        h     = np.clip(rng.normal(bp["height_mean"], bp["height_sd"], n), *H_BOUNDS)
        w     = bmi * h**2
        ages  = sample_ages(n, rng)
        bmr   = 10.0*w + 6.25*(h*100.0) - 5.0*ages + s_val
        met   = np.clip(rng.normal(MET_BASE, MET_BASE*MET_CV, n), 0.8, 1.8)
        flux  = bmr * KCAL_TO_W * (met + 0.25) / bsa_dubois(w, h)
        t_sk  = t_skin_neutral(sex, ages, f_base=tskin_f, m_val=tskin_m)
        t_opt = t_opt_from_flux(flux, np.full(n, clo_s1*0.155), t_sk,
                                 sex=sex, h_eff_val=h_eff_val)
        means[sex] = float(t_opt.mean())
    return means["Female"] - means["Male"]


def compute_quadrature_ci(verbose=True, n=N_GEN, seed=BASE_SEED,
                          alpha=0.05) -> dict:
    """
    Parametric uncertainty propagation via centered finite differences + RSS.
    σ_total = √Σ(∂ΔT/∂x_i × σ_i)²
    IC_(1-α) = ΔT_S1 ± z × σ_total
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    SEP = "─" * 74
    base_vals = {k: v[0] for k, v in UNCERTAINTY_PARAMS.items()}
    delta_base = simulate_s1_delta_T(n=n, seed=seed)
    contribs = {}
    if verbose:
        print(f"\n{SEP}")
        print(f"  Propagation d'incertitude paramétrique — ΔT_S1 (§3.4)")
        print(f"  N={n:,}  seed={seed}  IC={(1-alpha)*100:.0f}%")
        print(f"  ΔT_S1 central : {delta_base:+.4f} °C")
        print(f"  {'Param':<12} {'Base':>8} {'σ_i':>7}  {'∂ΔT/∂x':>10}  {'contrib':>10}  Source")
        print(f"  {SEP}")
    for pname, (base_val, sigma_i, label) in UNCERTAINTY_PARAMS.items():
        step = sigma_i / 2.0
        kw_hi = {**base_vals, pname: base_val + step}
        kw_lo = {**base_vals, pname: base_val - step}
        def _run(kw):
            return simulate_s1_delta_T(s_F=kw["s_F"], s_M=kw["s_M"],
                                       clo_s1=kw["clo_s1"], h_eff_val=kw["h_eff"],
                                       tskin_f=kw["tskin_f"], tskin_m=kw["tskin_m"],
                                       n=n, seed=seed)
        deriv = (_run(kw_hi) - _run(kw_lo)) / (2.0 * step)
        contrib = deriv * sigma_i
        contribs[pname] = {"label": label, "base": base_val, "sigma": sigma_i,
                            "deriv": deriv, "contrib": contrib}
        if verbose:
            print(f"  {pname:<12} {base_val:>8.3f} {sigma_i:>7.3f}  "
                  f"{deriv:>+10.5f}  {abs(contrib):>10.4f}°C  {label}")
    sigma_total = float(np.sqrt(sum(c["contrib"]**2 for c in contribs.values())))
    ci_lo, ci_hi = delta_base - z*sigma_total, delta_base + z*sigma_total
    if verbose:
        print(f"  {SEP}")
        print(f"  σ_total RSS : {sigma_total:.4f} °C")
        print(f"  ΔT_S1 = {delta_base:+.3f}°C  "
              f"[IC {(1-alpha)*100:.0f}% : {ci_lo:+.3f} ; {ci_hi:+.3f}°C]")
    return {"delta_base": delta_base, "sigma_total": sigma_total,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "z": z,
            "contributions": contribs}


def compute_s2_seasonal_quadrature(season: str, season_pops: dict,
                                   n_boot: int = 500) -> dict:
    """
    Bootstrap CI for S2 seasonal ΔT — resamples from existing cohort.
    Faster than parametric quadrature; gives MC uncertainty for each season.
    """
    pop_f = season_pops[season]["Female"]["t_opt"]
    pop_m = season_pops[season]["Male"]["t_opt"]
    return bootstrap_delta_T(pop_f, pop_m, n_boot=n_boot, n_sample=5000)

# =============================================================================
# 6. EXPORT HELPERS
# =============================================================================

def get_script_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def export_gtb_csv(lookup: pd.DataFrame, out_dir: Path) -> None:
    """CSV export of GTB table — operational values for BACnet/KNX import."""
    cols = ["p_female_pct"] + [f"{s}_op" for s in ["Winter","Spring","Summer","Autumn"]] + ["S1_op"]
    cols = [c for c in cols if c in lookup.columns]
    lookup[cols].to_csv(out_dir / "gtb_lookup_operational.csv", index=False)


def export_tornado_csv(quad_result: dict, out_dir: Path) -> None:
    """CSV export of Tornado diagram data — bar widths for matplotlib."""
    rows = []
    for pname, c in quad_result["contributions"].items():
        rows.append({"parameter": pname, "label": c["label"],
                     "base": c["base"], "sigma": c["sigma"],
                     "deriv": round(c["deriv"], 6),
                     "contrib_abs": round(abs(c["contrib"]), 4),
                     "contrib_pct_variance": round(
                         100 * c["contrib"]**2 /
                         sum(x["contrib"]**2 for x in quad_result["contributions"].values()), 1)
                     })
    rows.sort(key=lambda r: r["contrib_abs"], reverse=True)
    with open(out_dir / "tornado_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# =============================================================================
# COMFORT FRACTION & DISCOMFORT CURVE
# =============================================================================
# Method: comfort band ±Δ around each individual T_opt.
# For each setpoint T_a, an occupant is classified as uncomfortable if:
#   "too cold" : T_opt_i > T_a + Δ  (optimum warmer than setpoint)
#   "too warm" : T_opt_i < T_a − Δ  (optimum cooler than setpoint)
# Δ = 1.5°C ≈ ISO 7730 Category B (|PMV| ≤ 0.5, PPD < 10%)
# Δ = 2.5°C ≈ ISO 7730 Category C (|PMV| ≤ 0.85, PPD < 20%)
#
# Model consistency: T_opt is defined as the zero of the biophysical
# heat balance — this approach is therefore more consistent than the
# standard ISO 7730 PMV formula (which uses a slightly different radiation term).
# =============================================================================

COMFORT_DELTA_B = 1.5   # °C — bande de confort ISO 7730 Cat. B
COMFORT_DELTA_C = 2.5   # °C — bande de confort ISO 7730 Cat. C


def compute_discomfort_curve(season_pops: dict,
                              ta_range: np.ndarray = None,
                              delta: float = COMFORT_DELTA_B,
                              n_sample: int = 20_000,
                              seed: int = BASE_SEED) -> list:
    """
    Courbe "% inconfort vs consigne CVC" par sexe et saison.

    Pour chaque setpoint T_a ∈ ta_range :
      - pct_too_cold : T_opt_i > T_a + delta (personne veut plus chaud)
      - pct_too_warm : T_opt_i < T_a - delta (personne veut plus froid)
      - pct_comfortable = 100 - pct_too_cold - pct_too_warm

    Retourne une liste de dicts (saison × sexe × consigne).
    """
    if ta_range is None:
        ta_range = np.arange(20.0, 26.25, 0.25)

    rng  = np.random.default_rng(seed)
    rows = []

    for season, pops in season_pops.items():
        for sex, pop in pops.items():
            n   = len(pop["t_opt"])
            idx = rng.choice(n, min(n_sample, n), replace=False)
            t   = pop["t_opt"][idx]

            for ta in ta_range:
                ta = round(float(ta), 2)
                cold = float((t > ta + delta).mean() * 100)
                warm = float((t < ta - delta).mean() * 100)
                rows.append({
                    "season":          season,
                    "sex":             sex,
                    "ta":              ta,
                    "pct_too_cold":    round(cold, 2),
                    "pct_too_warm":    round(warm, 2),
                    "pct_uncomfortable": round(cold + warm, 2),
                    "pct_comfortable": round(100 - cold - warm, 2),
                    "delta_comfort":   delta,
                })
    return rows


def find_equitable_setpoint(discomfort_rows: list,
                             season: str = "Summer") -> dict:
    """
    Trouve le setpoint qui minimise l'inconfort total (H+F) pour une saison,
    et le setpoint qui égalise l'inconfort entre hommes et femmes.

    Retourne un dict avec les deux setpoints et les métriques associées.
    """
    sub = [r for r in discomfort_rows if r["season"] == season]
    ta_vals = sorted({r["ta"] for r in sub})
    results = {}

    for ta in ta_vals:
        m_row = next((r for r in sub if r["sex"] == "Male"   and r["ta"] == ta), None)
        f_row = next((r for r in sub if r["sex"] == "Female" and r["ta"] == ta), None)
        if m_row and f_row:
            total = m_row["pct_uncomfortable"] + f_row["pct_uncomfortable"]
            gap   = abs(f_row["pct_uncomfortable"] - m_row["pct_uncomfortable"])
            results[ta] = {"total": total, "gap": gap,
                            "m_pct": m_row["pct_uncomfortable"],
                            "f_pct": f_row["pct_uncomfortable"]}

    ta_min_total = min(results, key=lambda t: results[t]["total"])
    ta_min_gap   = min(results, key=lambda t: results[t]["gap"])

    return {
        "season":              season,
        "setpoint_min_total":  ta_min_total,
        "metrics_min_total":   results[ta_min_total],
        "setpoint_min_gap":    ta_min_gap,
        "metrics_min_gap":     results[ta_min_gap],
    }


def export_discomfort_csv(rows: list, out_dir) -> None:
    """CSV de la courbe d'inconfort — prêt pour matplotlib/Excel."""
    if not rows:
        return
    path = Path(out_dir) / "discomfort_curve.csv"
    with open(path, "w", newline="") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_discomfort_table(rows: list, season: str,
                            ta_keys: list = None) -> None:
    """Affiche la table d'inconfort pour les consignes clés."""
    if ta_keys is None:
        ta_keys = [22.0, 23.0, 24.0, 24.25, 25.0]
    sub = [r for r in rows if r["season"] == season]
    if not sub:
        return
    print(f"\n── DISCOMFORT CURVE — {season} (Δ={sub[0]['delta_comfort']}°C) " + "─"*25)
    print(f"  {'Ta':>5} │ {'Sexe':>6} │ {'%Froid':>7} │ {'%Chaud':>7} │ "
          f"{'%Inconf':>8} │ {'%Confort':>9}")
    for ta in ta_keys:
        ta = round(ta, 2)
        for r in sorted(sub, key=lambda x: x["sex"]):
            if abs(r["ta"] - ta) < 0.01:
                print(f"  {ta:>5.2f} │ {r['sex']:>6} │ "
                      f"{r['pct_too_cold']:>7.1f}% │ {r['pct_too_warm']:>7.1f}% │ "
                      f"{r['pct_uncomfortable']:>8.1f}% │ {r['pct_comfortable']:>9.1f}%")



# =============================================================================
# 7. MAIN EXECUTION PIPELINE
# =============================================================================

if __name__ == "__main__":
    SEP = "=" * 78
    run_id      = str(uuid.uuid4())
    script_hash = get_script_hash(__file__)
    out         = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)

    print(SEP)
    print(f"  Thermal Dimorphism Simulation — v15.0")
    print(f"  N_GEN={N_GEN:,}  |  seed={BASE_SEED}  |  Met={MET_BASE}±{MET_CV*100:.0f}%CV")
    print(f"  IMC M={BODY_PARAMS['Male']['bmi_mean']} / F={BODY_PARAMS['Female']['bmi_mean']} kg/m²  [NCD-RisC 2016]")
    print(SEP)

    # ── 7.1  SEASONAL SIMULATION (S2 — asymmetric Clo) ──────────────────────
    season_results, season_pops = {}, {}
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        pop_m = simulate_population("Male",   season, N_GEN, BASE_SEED)
        pop_f = simulate_population("Female", season, N_GEN, BASE_SEED+1,
                                    r_clo_m_ref=pop_m["r_clo"])
        s = group_stats(pop_f, pop_m)
        season_results[season] = s
        season_pops[season]    = {"Male": pop_m, "Female": pop_f}
        print(f"\n── {season.upper()} " + "─"*50)
        print(f"  T_skin_N (F/M)  : {s['t_skin_n_f']:.3f} / {s['t_skin_n_m']:.3f} °C")
        print(f"  Clo (F/M)       : {s['mean_clo_f']:.3f} / {s['mean_clo_m']:.3f}")
        print(f"  Flux (F/M)      : {s['mean_flux_f']:.2f} / {s['mean_flux_m']:.2f} W/m²")
        print(f"  T_opt (F/M)     : {s['t_opt_mean_f']:.3f} / {s['t_opt_mean_m']:.3f} °C")
        wci = welch_mean_diff_ci(pop_f["t_opt"], pop_m["t_opt"])
        print(f"  ΔT_opt          : {s['delta_t_opt']:+.3f}°C  "
              f"(d={s['cohens_d']:.3f}  Welch p={s['welch_p']:.2e}  "
              f"IC_Welch=[{wci['ci_lo']:+.3f};{wci['ci_hi']:+.3f}°C])")

    # ── 7.2  S1 — PURE BIOPHYSICAL (Clo=0.7 identical, lognormal BMI) ──────
    print(f"\n── S1 — PURE BIOPHYSICAL (Clo=0.7 identical) " + "─"*30)
    results_s1 = {}
    for sex in ["Male", "Female"]:
        rng_s1 = np.random.default_rng(BASE_SEED if sex == "Male" else BASE_SEED+1)
        bp = BODY_PARAMS[sex]
        m, v = bp["bmi_mean"], bp["bmi_sd"]**2
        bmi  = np.clip(rng_s1.lognormal(np.log(m**2/np.sqrt(v+m**2)),
                                         np.sqrt(np.log(1+v/m**2)), N_GEN), *BMI_BOUNDS)
        h    = np.clip(rng_s1.normal(bp["height_mean"], bp["height_sd"], N_GEN), *H_BOUNDS)
        w    = bmi * h**2
        fat  = sigmoid_fat(bmi, bp)
        ages = sample_ages(N_GEN, rng_s1)
        bmr  = 10.0*w + 6.25*(h*100.0) - 5.0*ages + bp["mifflin_s"]
        met  = np.clip(rng_s1.normal(MET_BASE, MET_BASE*MET_CV, N_GEN), 0.8, 1.8)
        q    = bmr * KCAL_TO_W * (met + 0.25)
        flux = q / bsa_dubois(w, h)
        r_clo = np.full(N_GEN, 0.7 * 0.155)
        t_sk  = t_skin_neutral(sex, ages)
        t_opt = t_opt_from_flux(flux, r_clo, t_sk, sex=sex)
        results_s1[sex] = {"t_opt": t_opt, "flux": flux, "lbm_flux": q/(w*(1-fat))}
    s1_delta = results_s1["Female"]["t_opt"].mean() - results_s1["Male"]["t_opt"].mean()
    boot_s1  = bootstrap_delta_T(results_s1["Female"]["t_opt"],
                                  results_s1["Male"]["t_opt"],
                                  n_boot=1000, n_sample=5000)
    print(f"  T_opt M  : {results_s1['Male']['t_opt'].mean():.3f}°C")
    print(f"  T_opt F  : {results_s1['Female']['t_opt'].mean():.3f}°C")
    pi_s1 = compute_prediction_interval(results_s1["Female"]["t_opt"],
                                          results_s1["Male"]["t_opt"])
    print(f"  ΔT S1    : {s1_delta:+.3f}°C  "
          f"[IC_mean: {pi_s1['ic_mean_lo']:+.3f};{pi_s1['ic_mean_hi']:+.3f}°C  "
          f"PI_indiv: {pi_s1['pi_lo']:+.3f};{pi_s1['pi_hi']:+.3f}°C]")

    # ── 7.3  GTB LOOKUP TABLE ────────────────────────────────────────────────
    print(f"\n── GTB LOOKUP TABLE — T_opt(p_f, season) [°C] " + "─"*27)
    lookup = build_gtb_lookup(season_pops, results_s1)
    for _, row in lookup.iterrows():
        note = "  ← parity ★" if row["p_female"] == 0.5 else ""
        s1_val = f" │ S1: {row['S1']:.2f}" if "S1" in row else ""
        print(f"  {row['p_female_pct']:>4}% │ "
              f"{row['Winter']:.2f} │ {row['Spring']:.2f} │ "
              f"{row['Summer']:.2f} │ {row['Autumn']:.2f}{s1_val}{note}")
    export_gtb_csv(lookup, out)
    gtb_curve = build_gtb_continuous(season_pops, results_s1)
    export_gtb_continuous_csv(gtb_curve, out)

    # ── 7.4  TABLE 8 — PERCENTILES ───────────────────────────────────────────
    print(f"\n── TABLE 8 : PERCENTILES DE DIMENSIONNEMENT (p_f=50%) " + "─"*20)
    def print_pct(name, arr):
        ps = np.percentile(arr, [5, 25, 50, 75, 95])
        print(f"  {name:<15} │ P5:{ps[0]:.2f} │ P25:{ps[1]:.2f} │ "
              f"P50:{ps[2]:.2f} │ P75:{ps[3]:.2f} │ P95:{ps[4]:.2f}")
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        comb = np.concatenate([season_pops[season]["Male"]["t_opt"],
                               season_pops[season]["Female"]["t_opt"]])
        print_pct(season, comb)
    print(f"  {'-'*65}")
    print_pct("S1 (Clo=0.7)",
              np.concatenate([results_s1["Male"]["t_opt"], results_s1["Female"]["t_opt"]]))
    print(f"  {'-'*65}")
    comb_annual = np.concatenate(
        [season_pops[s][sx]["t_opt"] for s in season_pops for sx in ["Male","Female"]])
    print_pct("Annuel (Exact)", comb_annual)
    def r025(v): return round(round(v/GTB_RESOLUTION)*GTB_RESOLUTION, 2)
    ps_a = np.percentile(comb_annual, [5,25,50,75,95])
    print(f"  {'Annuel (Opér.)':<15} │ "
          f"P5:{r025(ps_a[0]):.2f} │ P25:{r025(ps_a[1]):.2f} │ "
          f"P50:{r025(ps_a[2]):.2f} │ P75:{r025(ps_a[3]):.2f} │ P95:{r025(ps_a[4]):.2f}")

    # ── 7.5  VALIDATION vs CBE DATABASE ─────────────────────────────────────
    print(f"\n── VALIDATION vs CBE DATABASE " + "─"*44)
    annual_m = np.mean([s["t_opt_mean_m"] for s in season_results.values()])
    annual_f = np.mean([s["t_opt_mean_f"] for s in season_results.values()])
    print(f"  T_opt_M annual : {annual_m:.3f}°C  |  CBE: 24.11°C  |  err: {annual_m-24.11:+.3f}°C")
    print(f"  T_opt_F annual : {annual_f:.3f}°C  |  CBE: 24.40°C  |  err: {annual_f-24.40:+.3f}°C")
    print(f"  ΔT observed (CBE) : {24.40-24.11:+.3f}°C  |  ΔT model annual: "
          f"{annual_f-annual_m:+.3f}°C")

    # ── 7.6  ANALYTICAL DECOMPOSITION OF ΔT_S1 ──────────────────────────────
    print(f"\n── ANALYTICAL DECOMPOSITION OF ΔT_S1 " + "─"*37)
    t_opt_m_s1 = results_s1["Male"]["t_opt"]
    # (a) BMR effect only: female flux + forced male T_skin
    t_bmr = t_opt_from_flux(results_s1["Female"]["flux"],
                              np.full(N_GEN, 0.7*0.155),
                              np.full(N_GEN, T_SKIN_N_M), sex="Female")
    delta_bmr = t_bmr.mean() - t_opt_m_s1.mean()
    # (b) SWAN effect only: male flux + female T_skin
    rng_d = np.random.default_rng(BASE_SEED+1)
    t_swan = t_opt_from_flux(results_s1["Male"]["flux"],
                              np.full(N_GEN, 0.7*0.155),
                              t_skin_neutral("Female", sample_ages(N_GEN, rng_d)), sex="Female")
    delta_swan = t_swan.mean() - t_opt_m_s1.mean()
    print(f"  (a) Isolated BMR Effect (Forced T_skin)  : {delta_bmr:+.3f}°C")
    print(f"  (b) Isolated SWAN Effect (Forced Flux)    : {delta_swan:+.3f}°C")
    print(f"  Summed Linear Effects                      : {delta_bmr+delta_swan:+.3f}°C")
    print(f"  Simulated Joint S1 ΔT                      : {s1_delta:+.3f}°C")
    print(f"  Coupling Interaction Residue               : {s1_delta-(delta_bmr+delta_swan):+.4f}°C")

    # ── 7.7  CRASH-TEST: LBM PARADOX ────────────────────────────────────────
    print(f"\n── CRASH-TEST: LBM PARADOX METABOLIC BENCHMARK " + "─"*26)
    pop_m_k = simulate_population("Male",   "Spring", N_GEN, BASE_SEED,   bmr_model="katch_mcardle")
    pop_f_k = simulate_population("Female", "Spring", N_GEN, BASE_SEED+1, bmr_model="katch_mcardle")
    sk = group_stats(pop_f_k, pop_m_k)
    sm = season_results["Spring"]
    print(f"  Mifflin  — F LBM Flux: {sm['lbm_flux_f']:.2f} W/kg | M: {sm['lbm_flux_m']:.2f} W/kg | Paradox: {sm['lbm_paradox']}")
    print(f"  Katch-Mc — F LBM Flux: {sk['lbm_flux_f']:.2f} W/kg | M: {sk['lbm_flux_m']:.2f} W/kg | Paradox: {sk['lbm_paradox']}")

    # ── 7.8  QUADRATURE — PARAMETRIC UNCERTAINTY ────────────────────────────
    quad_result = compute_quadrature_ci(verbose=True, n=N_GEN, seed=BASE_SEED)
    export_tornado_csv(quad_result, out)

    # ── 7.9  BOOTSTRAP CI — S2 SEASONAL ─────────────────────────────────────
    print(f"\n── BOOTSTRAP CI — S2 SEASONAL (ΔT, n_boot=1000, n_sample=5000) " + "─"*10)
    seasonal_boots = {}
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        b = compute_s2_seasonal_quadrature(season, season_pops, n_boot=1000)
        seasonal_boots[season] = b
        print(f"  {season:<8} : ΔT={season_results[season]['delta_t_opt']:+.3f}°C  "
              f"[IC 95%: {b['ci_lo']:+.3f} ; {b['ci_hi']:+.3f}°C]  SE={b['se']:.4f}°C")

    # ── 7.9b  PARAMETRIC S2 SEASONAL QUADRATURE ────────────────────────────
    print(f"\n── QUADRATURE S2 PARAMÉTRIQUE PAR SAISON " + "─"*32)
    s2_quad_results = {}
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        r = compute_s2_seasonal_parametric_quadrature(season, verbose=True)
        s2_quad_results[season] = r

    # ── 7.9c  2nd-ORDER CROSS-SENSITIVITY ──────────────────────────────────
    print(f"\n── SENSIBILITÉ CROISÉE 2e ORDRE (∂²ΔT_S1/∂x_i∂x_j, n=20k) " + "─"*11)
    cross = compute_cross_sensitivity_2nd_order(n=20_000, seed=BASE_SEED)
    print(f"  {'Paire':<22} {'∂²ΔT':>10}  {'contrib 2nd':>13}  (Taylor ½σ_i σ_j ∂²f)")
    for pair, r in cross.items():
        print(f"  {pair:<22} {r['d2f']:>+10.5f}  {r['contrib_2nd']:>+13.6f}°C")
    max_pair = max(cross, key=lambda k: abs(cross[k]['contrib_2nd']))
    print(f"  → Terme croisé dominant : {max_pair}  (={cross[max_pair]['contrib_2nd']:+.6f}°C)")
    print(f"    (Si petit devant σ_total={quad_result['sigma_total']:.4f}°C → hypothèse linéaire validée)")

    # ── 7.10  AGE-STRATIFIED ANALYSIS ────────────────────────────────────────
    print(f"\n── AGE-STRATIFIED ΔT (S2 Summer — SWAN effect) " + "─"*25)
    print(f"  {'Bracket':<8} {'N_F':>6} {'N_M':>6} {'T_opt_F':>8} {'T_opt_M':>8} {'ΔT':>7} {'d':>6} {'p':>10} {'T_skin_F':>9}")
    for r in stratify_by_age(season_pops["Summer"]["Female"],
                              season_pops["Summer"]["Male"]):
        print(f"  {r['bracket']:<8} {r['n_f']:>6,} {r['n_m']:>6,} "
              f"{r['T_opt_f']:>8.3f} {r['T_opt_m']:>8.3f} "
              f"{r['delta_T']:>+7.3f} {r['cohens_d']:>6.3f} "
              f"{r['welch_p']:>10.2e} {r['t_skin_f']:>9.3f}")

    # ── 7.11  BMI-STRATIFIED ANALYSIS ────────────────────────────────────────
    print(f"\n── BMI-STRATIFIED ΔT (S2 Annual — all seasons pooled) " + "─"*18)
    # Pool all seasons for annual BMI stratification
    pop_f_annual = {
        "t_opt": np.concatenate([season_pops[s]["Female"]["t_opt"] for s in season_pops]),
        "bmi":   np.concatenate([season_pops[s]["Female"]["bmi"]   for s in season_pops]),
    }
    pop_m_annual = {
        "t_opt": np.concatenate([season_pops[s]["Male"]["t_opt"] for s in season_pops]),
        "bmi":   np.concatenate([season_pops[s]["Male"]["bmi"]   for s in season_pops]),
    }
    print(f"  {'Catégorie':<22} {'N_F':>7} {'N_M':>7} {'IMC_F':>6} {'IMC_M':>6} "
          f"{'T_opt_F':>8} {'T_opt_M':>8} {'ΔT':>7} {'d':>6}")
    for r in stratify_by_bmi(pop_f_annual, pop_m_annual):
        print(f"  {r['category']:<22} {r['n_f']:>7,} {r['n_m']:>7,} "
              f"{r['bmi_f']:>6.1f} {r['bmi_m']:>6.1f} "
              f"{r['T_opt_f']:>8.3f} {r['T_opt_m']:>8.3f} "
              f"{r['delta_T']:>+7.3f} {r['cohens_d']:>6.3f}")

    # ── 7.11b  DISCOMFORT CURVE (% inconfort vs setpoint) ──────────────────
    print(f"\n── DISCOMFORT CURVE — % inconfort par sexe vs consigne " + "─"*19)
    disc_rows = compute_discomfort_curve(season_pops, n_sample=20_000)
    print_discomfort_table(disc_rows, "Summer",
                            ta_keys=[22.0, 23.0, 24.0, 24.25, 25.0])
    eq = find_equitable_setpoint(disc_rows, season="Summer")
    print(f"\n  Setpoint min-inconf total : {eq['setpoint_min_total']:.2f}°C  "
          f"(M={eq['metrics_min_total']['m_pct']:.1f}%  "
          f"F={eq['metrics_min_total']['f_pct']:.1f}%)")
    print(f"  Setpoint équité (min gap) : {eq['setpoint_min_gap']:.2f}°C  "
          f"(gap={eq['metrics_min_gap']['gap']:.1f} pp)")
    export_discomfort_csv(disc_rows, out)

    # ── 7.12  RESIDUAL SOLVER ────────────────────────────────────────────────
    print(f"\n── SOLVEUR RÉSIDUEL : ÉQUATION À REBOURS (Cible CBE II) " + "─"*17)
    macro_f = {
        "flux_surface": np.concatenate([season_pops[s]["Female"]["flux_surface"] for s in season_pops]),
        "t_skin_n":     np.concatenate([season_pops[s]["Female"]["t_skin_n"]     for s in season_pops]),
        "r_clo":        np.concatenate([season_pops[s]["Female"]["r_clo"]        for s in season_pops]),
    }
    res_clo = solve_residual_clo(macro_f, 24.40)
    print(f"  T_opt cible (CBE II) : 24.40°C")
    print(f"  Biais initial        : {annual_f - 24.40:+.2f}°C")
    print(f"  ΔClo requis          : {res_clo['delta_clo']:+.3f} clo")
    print(f"  Diagnostic           : {res_clo['diagnosis']}")

    # ── 7.13  NORMALITY CHECK ────────────────────────────────────────────────
    ks_s = season_results["Summer"]
    print(f"\n── KS TEST T_opt simulée (Summer) ──────────────────────────────────")
    print(f"  KS D = {ks_s['ks_statistic']:.3f}  p = {ks_s['ks_pvalue']:.2e}")

    # ── 7.14  EXPORT JSON ────────────────────────────────────────────────────
    print(f"\n[SECURITÉ] Run ID    : {run_id}")
    print(f"[SECURITÉ] SHA-256   : {script_hash}")

    export = {
        "metadata": {
            "model": "thermal_dimorphism_v15.0",
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id, "script_sha256": script_hash,
            "N_GEN": N_GEN, "seed": BASE_SEED,
            "bmi_source": "NCD-RisC 2016 (Lancet 387:1377)",
            "bmi_male": BODY_PARAMS["Male"]["bmi_mean"],
            "bmi_female": BODY_PARAMS["Female"]["bmi_mean"],
        },
        "seasonal_results": {
            season: {
                "t_opt_mean_m": round(s["t_opt_mean_m"], 4),
                "t_opt_mean_f": round(s["t_opt_mean_f"], 4),
                "delta_t_opt":  round(s["delta_t_opt"],  4),
                "cohens_d":     round(s["cohens_d"],     4),
                "welch_p":      s["welch_p"],
                "bootstrap_ci": seasonal_boots.get(season, {}),
            }
            for season, s in season_results.items()
        },
        "s1": {
            "delta_t":     round(s1_delta, 4),
            "bootstrap_ci": boot_s1,
            "bmr_effect":  round(delta_bmr, 4),
            "swan_effect": round(delta_swan, 4),
            "coupling_residue": round(s1_delta - (delta_bmr + delta_swan), 6),
        },
        "quadrature": {
            "delta_t_central":  round(quad_result["delta_base"], 4),
            "sigma_total":      round(quad_result["sigma_total"], 4),
            "ic_95_lo":         round(quad_result["ci_lo"], 4),
            "ic_95_hi":         round(quad_result["ci_hi"], 4),
            "contributions": {
                k: {"sigma": round(v["sigma"], 4),
                    "deriv": round(v["deriv"], 6),
                    "contrib": round(v["contrib"], 4)}
                for k, v in quad_result["contributions"].items()
            },
        },
        "validation": {
            "cbe_t_neutral_m": 24.11, "cbe_t_neutral_f": 24.40,
            "model_annual_m":  round(annual_m, 3),
            "model_annual_f":  round(annual_f, 3),
            "error_m":         round(annual_m - 24.11, 3),
            "error_f":         round(annual_f - 24.40, 3),
            "required_delta_clo": round(res_clo["delta_clo"], 4),
            "clo_diagnosis":      res_clo["diagnosis"],
        },
        "percentiles_annual_s2": {
            f"P{p}": round(float(np.percentile(comb_annual, p)), 2)
            for p in [5, 25, 50, 75, 95]
        },
    }
    json_path = out / "thermal_dimorphism.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=4)

    print(f"\n[OUTPUT] JSON    → {json_path}")
    print(f"[OUTPUT] GTB CSV → {out / 'gtb_lookup_operational.csv'}")
    print(f"[OUTPUT] Tornado → {out / 'tornado_data.csv'}")
    print(f"[OUTPUT] Discomfort → {out / 'discomfort_curve.csv'}")
    print(SEP)
