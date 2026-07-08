# Li-ion Battery Degradation Pipeline (MIT/Kaggle Cycle-Life Dataset)

An ML pipeline that estimates a battery's State of Health (SOH), Remaining
Useful Life (RUL), and replacement need from behavioral cycling data, with
anomaly detection and unsupervised degradation-stage clustering, wrapped in
an LLM agent for natural-language queries.

![Architecture](assets/architecture_diagram.svg)

## Results

![Results summary](assets/results_summary.png)

| Component | Metric | Result | Notes |
|---|---|---|---|
| SOH regressor | R² (GroupKFold) | **0.893** | Gradient Boosting |
| RUL regressor | R² (GroupKFold) | **0.616** | Gradient Boosting; noisier at low-RUL, exactly where it matters most |
| Replacement classifier | F1 / Precision / Recall | **0.78 / 0.68 / 0.93** | Random Forest; deliberately recall-favoring (0.67% positive class) |
| Anomaly detector | Known-glitch catch rate / false-positive rate | **100% / 4.2%** | Isolation Forest + hard ratio rule, ratio-normalized features |
| Clustering | Silhouette (k=3) | **0.72** | Healthy (0.975) -> Transitional (0.886) -> Critical (0.813) mean SOH |

All regression/classification results validated with **GroupKFold grouped by
`battery_id`** -- never a random split, so no battery's cycles leak across
train/test.

## Project structure

```
source/
  loader.py     -- raw CSV load + cleaning (drops cycle-1 sensor artifact)
  features.py   -- leak-free feature engineering + label computation
  models.py     -- trains/evaluates all 5 components, agent-facing predict functions
  pipeline.py   -- integration layer: battery baseline + run_battery_pipeline()
agent/
  agent.py         -- Claude-backed natural-language agent
  agent_gemini.py  -- Gemini-backed equivalent
models/          -- saved .pkl model bundles (scaler + model + feature list)
data/
  raw/           -- source CSV
  processed/     -- engineered feature table
notebooks/       -- 01-06, EDA through full pipeline integration
```

## Key design decisions (and why)

- **Never feed raw `QD`/`QC` (capacity) into any model.** The original
  tool-generated pipeline this project started from defined `target_soh` as
  exactly `QD/Nominal_QD_Cap` and then fed both `QD` and `Nominal_QD_Cap`
  back in as features -- the model was trivially reconstructing its own
  target, not learning degradation (verified R2=99.95% was 100% leakage
  artifact). Only indirect behavioral signals (`IR`, `chargetime`,
  temperatures) are used as inputs.
- **Ratio-normalize `IR` and `chargetime` against each battery's own
  early-cycle baseline**, not absolute values. Different fast-charging
  protocols produce genuinely different absolute scales -- comparing raw
  values across protocols conflates "which protocol" with "how degraded."
  Verified this substantially improves signal: `IR_ratio` correlates with
  `soh` at -0.745 vs raw `IR`'s -0.450.
- **RUL computed directly from `cycle_life - cycle`**, not the more
  "sophisticated" average-degradation-rate formula from the original
  pipeline. That formula divides by near-zero early-cycle fade rate,
  producing artifact values (found: 5,932 rows hard-capped at a sentinel
  15000) -- and misbehaves on any realistic fast-then-slow fade curve, not
  just the fresh-battery edge case it was built to guard against.
- **Clustering uses only `IR_ratio`/`chargetime_ratio`**, not the full
  feature set. An earlier version included protocol parameters and
  temperature, which caused K-means to cluster by *charging protocol*
  instead of degradation stage (every protocol group spans all health
  levels, so clusters came back with nearly identical mean SOH -- a real
  bug, found and fixed).
- **Two independent anomaly guardrails, not one.** A MAD-based statistical
  check flags corrupted `QD` readings in `features.py`; Isolation Forest +
  a hard ratio-threshold rule flags `IR`/`chargetime` anomalies in
  `models.py`. Neither sees what the other catches -- documented explicitly
  rather than assumed to be one unified system.

## Known limitations

- **RUL accuracy degrades at low true-RUL** (see notebook 03) -- exactly
  where a replacement decision needs it to be reliable. Treat RUL as a
  rough signal, not a maintenance-scheduling number.
- **Replacement classifier trades precision for recall on purpose**
  (catches 93% of real cases, ~32% false-alarm rate) -- appropriate for
  this use case, but worth tuning the decision threshold if your
  false-alarm tolerance differs.
- **"Latest recorded cycle" means something different for historical vs.
  live data** (see notebook 06) -- for a completed historical battery, the
  last recorded cycle is definitionally near its own end-of-life, which
  biases any bulk analysis using `run_battery_pipeline()`'s default "latest
  cycle" behavior toward Critical/anomalous results. Not a problem for
  genuinely live-monitored batteries, where the latest reading really is
  the current state.
- **Anomaly detector cannot see capacity-based anomalies** -- it only
  covers `IR`/`chargetime`/temperature; `QD` glitches are caught upstream
  by a separate mechanism.

## Setup

```bash
pip install pandas numpy scikit-learn joblib
# for the agent layer, one of:
pip install anthropic       # agent.py
pip install google-genai    # agent_gemini.py
```

Set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` before running an agent script.

## Running

```bash
# rebuild features from raw data
python3 source/features.py data/raw/<file>.csv

# train all 5 models
python3 source/models.py

# query a battery via the agent
python3 agent/agent.py <battery_id> "Should I replace this battery soon?"
```

## Notebooks

| # | Notebook | Covers |
|---|---|---|
| 01 | `01_eda.ipynb` | Raw data, cleaning, capacity fade curves, data-quality flags |
| 02 | `02_feature_engineering.ipynb` | Ratio-normalization, quantitative proof it helps |
| 03 | `03_regressors.ipynb` | SOH + RUL: algorithm comparison, feature importance, held-out predictions |
| 04 | `04_anomaly_clustering.ipynb` | Both unsupervised components, including the bugs found and fixed |
| 05 | `05_replacement_classifier.ipynb` | Class imbalance, F1 vs accuracy, precision/recall tradeoff |
| 06 | `06_pipeline_integration.ipynb` | All 5 components together, real disagreement cases |
