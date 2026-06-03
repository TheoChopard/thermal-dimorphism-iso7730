# Thermal Dimorphism — Pipeline v15.0

**Project**: Sex Differences in Thermal Optimum: A Biophysical Reassessment of ISO 7730  
**Scenario**: Office / Sedentary — Multi-season, EU active population  
**Reference SHA-256**: `b3fd0c5e96fad280336547b91c7fc26482bf3c7d70ec2b0eab096e0f01b5b5d1`

---

## File structure

```
thermal-dimorphism-iso7730/
├── model_constants.py        Reference list of biophysical constants (documentation)
├── thermal_dimorphism.py     Monte-Carlo simulation module
├── Validation.py             CBE-II empirical validation module
├── run_pipeline.py           CLI orchestrator
├── launch_simulation.command Double-click shortcut (macOS)
└── output/                   Auto-generated at each run
    ├── thermal_dimorphism.json
    ├── gtb_lookup_operational.csv
    ├── gtb_continuous_curve.csv
    ├── tornado_data.csv
    └── discomfort_curve.csv
```

---

## Quick start

```bash
cd thermal-dimorphism-iso7730

python3 run_pipeline.py --simulate          # full simulation (~5s)
python3 run_pipeline.py --validate          # CBE validation only (~3min)
python3 run_pipeline.py --all               # simulation + validation
python3 run_pipeline.py --simulate --quick  # quick test N_GEN=20k (~1s)
```

Or double-click `launch_simulation.command` from Finder.

---

## Module descriptions

### `model_constants.py`
Single-file reference listing the biophysical constants and their literature
sources (table below). For transparency the values are documented here in one
place; the simulation and validation modules define their constants **inline**,
so this file is **not imported at runtime**. To change a parameter, edit the
corresponding definition in `thermal_dimorphism.py` (and `Validation.py`).

| Constant | Value | Source |
|----------|-------|--------|
| Male BMI | 26.5 kg/m² | NCD-RisC 2016, Lancet 387:1377 |
| Female BMI | 25.5 kg/m² | NCD-RisC 2016, Lancet 387:1377 |
| H_EFF | 10.105 W/m²K | ISO 7730 / Fanger 1970 |
| T_skin_N_M | 33.5°C | Hardy & DuBois 1938 |
| T_skin_N_F | 33.2°C | Iampietro et al. 1960 |
| Luteal ΔT | −0.3°C | Charkoudian 2014 |
| SWAN median age | 52.54 years | Gold et al. 2013 |
| Clo ΔKarjalainen | 0.131 | Karjalainen 2012, Table 3 |

> *T_skin_N_F (33.2 °C): the code comment cites Iampietro et al. 1960 as the
> historical female baseline; the manuscript (§2.5) refines this to a calibrated
> value (Lan 2008; Vellei 2025) within the SWAN logistic (Gold et al. 2013).*

---

### `thermal_dimorphism.py`
Main Monte-Carlo simulation. N_GEN = 200,000 agents.

**16 active features:**

| # | Feature | Manuscript section |
|---|---------|-------------------|
| 1 | Seasonal S2 simulation (4 seasons) | §3.3, Table 5 |
| 2 | Scenario S1 — pure biophysical baseline | §3.2 |
| 3 | GTB table T_opt(p_f, season) | §3.6 |
| 4 | Continuous GTB curve (200 points) | §3.6 |
| 5 | Sizing percentiles P5–P95 | §3.7, Table 8 |
| 6 | Validation vs CBE-II (annual errors) | §3.5 |
| 7 | Analytical decomposition ΔT_S1 (BMR / SWAN) | §4.2 |
| 8 | Crash-test LBM paradox (Katch-McArdle) | §3.1 |
| 9 | Parametric quadrature S1 (6 parameters) | §3.4 |
| 10 | Parametric S2 seasonal quadrature | §3.4 |
| 11 | 2nd-order cross-sensitivity | §3.4 |
| 12 | Bootstrap CI seasonal ΔT (N=1,000) | §3.4 |
| 13 | Mean CI vs individual PI | §3.4 |
| 14 | Age stratification (Eurostat brackets) | §4.2 |
| 15 | BMI stratification (Normal/Overweight/Obese) | §4.1 |
| 16 | Discomfort curve % vs HVAC setpoint | §4.7 |

**Key results v15.0:**

| Quantity | Value |
|----------|-------|
| ΔT_S1 | +0.621°C |
| Mean CI S1 | [+0.614; +0.628°C] |
| Individual PI S1 | [−2.522; +3.764°C] |
| Quadrature CI 95% | [+0.325; +0.926°C] |
| ΔT S2 Winter | +1.325°C |
| ΔT S2 Spring | +1.472°C |
| ΔT S2 Summer | +1.713°C |
| ΔT S2 Autumn | +1.518°C |
| GTB setpoint S2 annual parity | 24.25°C |
| GTB setpoint S1 parity | 25.50°C |
| Equity setpoint (min gap, summer) | 25.00°C |
| % female uncomfortable at 23°C (summer) | 76.1% |
| % male uncomfortable at 23°C (summer) | 37.1% |
| Isolated BMR effect | +1.027°C |
| Isolated SWAN effect | −0.405°C |
| Dominant cross-term s_F×Clo | −0.0026°C (<2% σ) |
| Required ΔClo (CBE backward solver) | +0.129 clo |

---

### `Validation.py`
Empirical validation against the ASHRAE Global Thermal Comfort Database II  
(109,033 observations; 59,927 after cleaning).  
**Requires**: `decompressed_data.csv` in the same folder or on the Desktop.

**6 analysis blocks:**

| Block | Content |
|-------|---------|
| STEP 1 | Data audit — total N, M/F ratio |
| STEP 2 | T_neutral by sex — OLS + 1,000-iteration bootstrap + Welch |
| STEP 2.5 | Levene test on neutral subset |
| STEP 3 | Seasonal T_neutral + geographic filter (Köppen Cfb/Dfb) |
| STEP 3c | T_neutral stratified by age decade |
| STEP 4 | Clo statistics by sex vs Karjalainen reference |
| STEP 5 | Individual T_opt predictions — MAE, RMSE, R², bias, r |

**Key CBE-II results:**

| Quantity | Value |
|----------|-------|
| Male T_neutral | 24.110°C [24.00–24.22] |
| Female T_neutral | 24.400°C [24.30–24.50] |
| Observed ΔT (F−M) | +0.290°C |
| Inter-sex gap significance | z = 3.79, p ≈ 1.5×10⁻⁴ (see note) |
| Observed ΔClo (M−F) | +0.013 (10% of Karjalainen assumption) |
| 40–50 age inversion | ΔT_obs = −0.621°C (perimenopause) |
| Individual R² | −0.112 (expected: T_opt ≠ T_imposed) |
| r(T_opt, vote) | 0.054 (p=2.5×10⁻⁷) |

> **Note.** STEP 2's `welch_bootstrap` applies Welch's t-test to the two
> *bootstrap distributions* of the regression intercepts and prints t = 121.7;
> because those distributions reflect the sampling precision of the means, that
> figure indexes the robustness of the mean gap, **not** a per-individual effect.
> The value reported above (and in the manuscript) is the statistically
> appropriate test — a bootstrap of the *difference* in intercepts —
> **z = 3.79, p ≈ 1.5×10⁻⁴, 95% CI [+0.14, +0.44] °C**, corroborated by a pooled
> regression (sex×temperature interaction p ≈ 1×10⁻⁶).

---

## Outputs

| File | Format | Content |
|------|--------|---------|
| `thermal_dimorphism.json` | JSON | All numerical results + metadata + SHA-256 |
| `gtb_lookup_operational.csv` | CSV | GTB table 21×6, rounded to 0.25°C (BACnet/KNX) |
| `gtb_continuous_curve.csv` | CSV | 200-point T_opt vs p_f curve for matplotlib |
| `tornado_data.csv` | CSV | Sorted quadrature contributions — Figure 5 |
| `discomfort_curve.csv` | CSV | % discomfort F/M by season × setpoint |
| `validation_v15_0_results.json` | JSON | CBE validation results + SHA-256 |

---

## Traceability

The script SHA-256 is computed at every run and included in the output JSON.  
Every numerical result is associated with a unique Run ID (UUID v4).

```
Stable SHA-256 : b3fd0c5e96fad280336547b91c7fc26482bf3c7d70ec2b0eab096e0f01b5b5d1
Seed           : 7730
N_GEN          : 200,000
```

---

## Python dependencies

```
numpy       scipy       pandas      hashlib (stdlib)
uuid (stdlib)           json (stdlib)           csv (stdlib)
```

Install if missing:
```bash
pip3 install numpy scipy pandas
```

---

## Correspondence with the manuscript

| Section | Key value | Code source |
|---------|-----------|-------------|
| §3.2 | ΔT_S1 = +0.621°C | `s1_delta` |
| §3.3 Table 5 | Seasonal ΔT 1.325/1.472/1.713/1.518°C | `season_results` |
| §3.4 | Quadrature CI [+0.325; +0.926°C] | `quad_result` |
| §3.5 | Female bias = +0.695°C, ΔClo = +0.129 | `res_clo` |
| §3.6 | GTB parity S2 = 24.25°C, S1 = 25.50°C | `lookup` |
| §4.2 | BMR = +1.027°C, SWAN = −0.405°C | decomposition |
| §4.3 | Required ΔClo = +0.129 clo | `res_clo` |
| §4.7 | Summer equity setpoint = 25.0°C | `discomfort_curve` |
| §4.9 | Individual PI [−2.52; +3.76°C] | `pi_s1` |

---

## Troubleshooting & macOS Notes

**Command Not Found (`python`)**
Modern macOS environments do not alias `python` to Python 3 by default. Ensure you run the orchestrator using `python3` (or the absolute path like `/usr/local/bin/python3`) as written in the Quick Start section.

**macOS Execution Policies (Gatekeeper)**
If you use the `launch_simulation.command` shortcut on macOS, the system might block its execution due to default security policies for uncertified or downloaded scripts. You can bypass this using either the graphical interface or the terminal.

*Option A: GUI Method (Recommended)*
1. Right-click (or Control-click) on `launch_simulation.command` in Finder.
2. Select **Open** from the context menu.
3. Click **Open** again in the security dialog. *(macOS will remember this authorization for future runs).*

*Option B: Terminal Method (Advanced)*
If the script fails to run due to permission errors or persistent Gatekeeper blocks, you can manually unlock it via the terminal. Navigate to the project folder and execute the following commands:

```bash
# 1. Grant execution permissions to the file
chmod +x launch_simulation.command

# 2. Remove the macOS quarantine attribute
xattr -d com.apple.quarantine launch_simulation.command