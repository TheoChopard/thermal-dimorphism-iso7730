import numpy as np

# ── ISO thermal constants ─────────────────────────────────────────────────────
H_EFF     = 10.105          # W/m²K  — ISO 7730 / Fanger 1970
PA_REF    = 1.2             # kPa    — standard office humidity
KCAL_TO_W = 4184.0 / 86400.0

# ── Neutral skin temperature — sex dimorphism ─────────────────────────────────
T_SKIN_N_M      = 33.5      # °C  — Hardy & DuBois 1938 (males)
T_SKIN_N_F_BASE = 33.2      # °C  — female baseline, calibrated (Lan 2008; Vellei 2025)
T_SKIN_LUTEAL_D = -0.3      # °C  — luteal phase, Charkoudian 2014
AGE_MEDIAN_SWAN = 52.54     # years — menopause median, SWAN study, Gold 2013
KAPPA_SWAN      = 1.1       # logistic slope (5-year perimenopausal window)

# ── Clothing ──────────────────────────────────────────────────────────────────
CLO_UNDERWEAR_F       = 0.05    # clo — underwear correction (Smallcombe 2021; ISO 9920)
KARJALAINEN_DELTA_CLO = 0.131   # clo — Table 3, Karjalainen 2012 (0.943−0.812)

# ── Metabolism ────────────────────────────────────────────────────────────────
MET_BASE    = 1.2
MET_CV      = 0.15              # Garby & Astrup 1987
AGE_DEFAULT = 42.6              # years — EU workforce mean, Eurostat lfsa_egan 2022

# ── Simulation ────────────────────────────────────────────────────────────────
N_GEN          = 200_000
BASE_SEED      = 7730
H_BOUNDS       = (1.20, 2.30)
BMI_BOUNDS     = (12.0, 58.0)
GTB_RESOLUTION = 0.25           # °C — standard BACnet/KNX addressing step

# ── Anthropometry — NCD-RisC 2016 (Lancet 387:1377) ──────────────────────────
# Western Europe, Cfb/Dfb climate zones, active population
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

# Lowercase aliases for Validation.py (pandas gender = "male"/"female")
SIGMOID_PARAMS = {
    "male":   {k: BODY_PARAMS["Male"][k]   for k in ("phi_min","phi_max","k","B0")},
    "female": {k: BODY_PARAMS["Female"][k] for k in ("phi_min","phi_max","k","B0")},
}

# ── Seasonal clothing — Karjalainen 2012, Table 3 ─────────────────────────────
SEASONAL_CLO = {
    "Winter": {"Male": (0.990, 0.14), "Female": (0.850 + CLO_UNDERWEAR_F, 0.21)},
    "Spring": {"Male": (0.950, 0.14), "Female": (0.777 + CLO_UNDERWEAR_F, 0.21)},
    "Summer": {"Male": (0.890, 0.14), "Female": (0.663 + CLO_UNDERWEAR_F, 0.21)},
    "Autumn": {"Male": (0.940, 0.14), "Female": (0.757 + CLO_UNDERWEAR_F, 0.21)},
}

# ── EU age distribution — Eurostat lfsa_egan 2022 ────────────────────────────
AGE_BRACKETS = [
    (20, 30, 0.18), (30, 40, 0.25), (40, 50, 0.26),
    (50, 60, 0.23), (60, 65, 0.08)
]

# ── Parametric uncertainties for quadrature (§2.10) ──────────────────────────
UNCERTAINTY_PARAMS = {
    "s_F":     (-161.0,  15.0,         "Female Mifflin intercept [kcal/d] — Frankenfield 2005"),
    "s_M":     (   5.0,  15.0,         "Male Mifflin intercept [kcal/d] — Frankenfield 2005"),
    "clo_s1":  (   0.7,   0.07,        "Clo S1 [±10%, ISO 9920]"),
    "h_eff":   (H_EFF,  H_EFF * 0.05, "h_eff [±5%, ISO 7730]"),
    "tskin_f": (T_SKIN_N_F_BASE, 0.05, "T_skin,n F [SE ±0.05°C]"),
    "tskin_m": (T_SKIN_N_M,      0.05, "T_skin,n M [SE ±0.05°C]"),
}

# ── Geography ─────────────────────────────────────────────────────────────────
KOPPEN_EU_TEMPERATE = {"Cfb", "Dfb", "Cfc", "Dfc"}

# ── PMV/PPD — ISO 7730:2025 thresholds ───────────────────────────────────────
PMV_UNCOMFORTABLE_LO = -0.85   # PMV < -0.85 → too cold  (PPD > 20%)
PMV_UNCOMFORTABLE_HI = +0.85   # PMV > +0.85 → too warm  (PPD > 20%)
PMV_COLD_ISO         = -0.5    # PMV < -0.5  → ISO 7730 Category B/C
PMV_HOT_ISO          = +0.5    # PMV > +0.5  → ISO 7730 Category B/C
