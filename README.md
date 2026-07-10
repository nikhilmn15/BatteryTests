# Li-ion Battery Degradation Pipeline

## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Metrics](#metrics)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Sample Output](#sample-output)
- [Challenges Faced](#challenges-faced)
- [Achievements](#achievements)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Introduction

This project estimates a lithium-ion battery's **State of Health (SOH)**,
**Remaining Useful Life (RUL)**, and whether it **needs replacement**, using
only indirect behavioral signals from its charge/discharge cycling data --
never the raw capacity readings that would make the problem trivial. It also
includes an **anomaly detector** (flags corrupted or out-of-distribution
cycles) and **unsupervised clustering** (groups cycles into Healthy /
Transitional / Critical degradation stages), all wrapped in an **LLM agent**
that answers natural-language questions about a battery's condition.

The project went through two full iterations: an initial NASA PCoE
battery-dataset version, then a rebuild on a larger 140-battery MIT/Kaggle
fast-charging dataset once the pipeline architecture was validated. Several
real bugs -- data leakage, a flawed RUL formula, a clustering bug -- were
found and fixed along the way; see [Challenges Faced](#challenges-faced).

## Architecture

![Architecture](assets/architecture_diagram.svg)

Raw cycle data flows through a feature pipeline into two guardrail/insight
components (anomaly detection, clustering) and three supervised predictors
(SOH, RUL, replacement), all of which feed into an LLM agent that combines
them into one natural-language answer.

## Dataset

**Source:** [Kaggle -- Lithium-Ion Battery Cycle Life Time Series Dataset](https://www.kaggle.com/datasets/solitaryseeker/lithium-ion-battery-cycle-life-time-series-dataset)
(originally from the MIT/Stanford/Toyota fast-charging battery aging study).

- **140 batteries**, 114,598 raw cycles total
- Multiple fast-charging protocols (single-step and two-step CC-CC-CV),
  producing genuinely different absolute charge-time/resistance scales per
  battery -- a key reason features are ratio-normalized (see
  [Challenges Faced](#challenges-faced))
- Per-cycle fields: `IR` (internal resistance), `QC`/`QD` (charge/discharge
  capacity), `Tavg`/`Tmin`/`Tmax`, `chargetime`, plus per-battery protocol
  parameters `C1`/`Q1`/`C2` and the recorded `cycle_life`
- Cycle life ranges from 148 to 2,237 cycles across the fleet -- large,
  genuine heterogeneity, not noise

## Models Used

| Component | Algorithms compared | Winner |
|---|---|---|
| SOH regressor | Linear Regression, Random Forest, Gradient Boosting | Gradient Boosting |
| RUL regressor | Linear Regression, Random Forest, Gradient Boosting | Gradient Boosting |
| Replacement classifier | Logistic Regression, Random Forest, Gradient Boosting | Random Forest |
| Anomaly detector | Isolation Forest + hard ratio-threshold rule | -- |
| Clustering | K-means (k selected via elbow + silhouette) | k=3 |

SVM/SVR were deliberately excluded -- too slow at this row count (114k+
rows x 5-fold GroupKFold) for the marginal benefit seen during development.

## Metrics

![Results summary](assets/results_summary.png)

| Component | Metric | Result |
|---|---|---|
| SOH regressor | R2 (GroupKFold) | **0.893** |
| RUL regressor | R2 (GroupKFold) | **0.616** |
| Replacement classifier | F1 / Precision / Recall | **0.78 / 0.68 / 0.93** |
| Anomaly detector | Known-glitch catch rate / false-positive rate | **100% / 4.2%** |
| Clustering | Silhouette (k=3) | **0.72** |

All supervised results validated with **GroupKFold grouped by
`battery_id`** -- a battery's cycles never appear in both train and test,
which a random split would allow (cycle 500 and 501 of the same battery are
nearly identical -- that's leakage, not learning).

## Project Structure

```
V2/
  source/
    loader.py     -- raw CSV load + cleaning
    features.py   -- leak-free feature engineering + label computation
    models.py     -- trains/evaluates all 5 components, agent-facing predict functions
    pipeline.py   -- integration layer: battery baseline + run_battery_pipeline()
  models/          -- saved .pkl model bundles (scaler + model + feature list)
agent/
  agent.py         -- Claude-backed natural-language agent
  agent_gemini.py  -- Gemini-backed equivalent
data/
  raw/             -- source CSV
  processed/       -- engineered feature table
notebooks/         -- 01-06, EDA through full pipeline integration
assets/            -- architecture diagram, results chart
```

## How to Run

```bash
pip install pandas numpy scikit-learn joblib
pip install anthropic       # for agent.py
pip install google-genai    # for agent_gemini.py
```

```bash
# rebuild features from raw data
python3 V2/source/features.py data/raw/"Lithium-Ion Battery Cycle Life.csv"

# train all 5 models
python3 V2/source/models.py

# query a battery through the agent (needs ANTHROPIC_API_KEY set)
python3 agent/agent.py b1c5 "Should I replace this battery soon?"
```

## Sample Output

```
$ py agent/agent.py b1c5 "What is the condition of the battery?"

--- Raw pipeline output for b1c5 ---
  predicted_soh: 0.8187
  predicted_rul_cycles: 11.3
  needs_replacement: False
  replacement_probability: 0.04
  degradation_stage: Critical
  is_anomalous: True
  anomaly_top_reasons: [{'feature': 'chargetime_ratio', 'z_score': 5.01}, ...]

--- Agent's answer ---
The readings for battery b1c5 at cycle 1073 are anomalous. This indicates
that the other predictions may be less trustworthy for this specific
reading. The anomaly is primarily due to an unusual chargetime ratio
(z-score 5.01), IR ratio (z-score 3.9), and Tmax (z-score 1.18)...

The degradation stage is categorized as "Critical". While the degradation
stage is "Critical", the supervised model does not recommend replacement.
This is not a contradiction, as degradation stage reflects unsupervised
behavioral grouping, whereas the replacement recommendation is a
supervised, threshold-based decision.
```

## Challenges Faced

- **Data leakage in an early version of the pipeline.** `target_soh` was
  defined as exactly `QD / Nominal_QD_Cap`, and both `QD` and
  `Nominal_QD_Cap` were then fed back in as model features -- the model was
  reconstructing its own answer, not learning degradation. Reported
  R2=99.95%; the real, leak-free number is 0.893. Even after removing the
  obvious leaks, engineered columns like `thermal_efficiency_index =
  QD/Tavg` still smuggled in raw capacity and had to be dropped too.
- **A flawed RUL formula.** The original approach computed RUL as
  `(soh - 0.80) / average_degradation_rate_since_cycle_1`. Near-zero early
  fade rate causes this to explode -- 5,932 rows had to be hard-capped at a
  sentinel value of 15000. Worse, the formula misbehaves on any realistic
  fast-then-slow fade curve (not just the fresh-battery edge case), because
  it uses an *average* rate rather than a local one -- verified it can
  produce RUL that *increases* over time on a synthetic but physically
  realistic degradation curve. Replaced with a direct `cycle_life - cycle`
  calculation.
- **A clustering bug caused by protocol-dominated features.** An early
  version clustered on the full feature set (including protocol parameters
  and temperature) and produced three clusters with nearly identical mean
  SOH (0.966/0.961/0.960) -- it was grouping by *charging protocol*, not
  degradation stage, since every protocol group spans all health levels.
  Fixed by restricting clustering to only `IR_ratio`/`chargetime_ratio`.
- **A handful of sensor glitches distorted k-selection entirely.** 14 rows
  with `chargetime` readings 49-96x their real baseline created a
  trivially-separable "glitch vs everyone" cluster that hijacked the
  silhouette-score-based choice of k. Filtering these out first (via a
  dedicated `is_ratio_anomaly` flag) revealed the real, meaningful 3-stage
  structure underneath.
- **The anomaly detector initially performed worse than random.** Raw
  `IR`/`chargetime`/protocol features caused Isolation Forest to flag rare
  protocol combinations more often than actual behavioral anomalies.
  Switching to the same ratio-normalized features used elsewhere raised the
  known-glitch catch rate from 78.6% to 100%.
- **Cross-environment path/dependency mismatches** when moving the
  pipeline from a development sandbox to a real local project structure
  (different folder layout, renamed files, Windows paths) -- required
  explicit fixes to import paths and data/model file locations.

## Achievements

- Full 5-component pipeline (SOH, RUL, replacement, anomaly detection,
  clustering) trained, validated, and reproducible from raw data to saved
  models.
- Two working LLM agent integrations (Claude and Gemini) that correctly
  reason about anomaly caveats and explain apparent disagreements between
  components rather than just reporting raw numbers.
- Six documentation notebooks covering the full pipeline, each with
  real, pre-run outputs rather than just code.
- Every major result independently stress-tested: synthetic input testing,
  real held-out battery testing, and a properly isolated leakage
  investigation that avoided a false "detector is broken" conclusion.

## Tech Stack

- **Data/ML:** Python, pandas, NumPy, scikit-learn, joblib
- **Models:** Linear/Logistic Regression, Random Forest, Gradient Boosting,
  Isolation Forest, K-means
- **Agent layer:** Anthropic Claude API (`anthropic`), Google Gemini API
  (`google-genai`)
- **Visualization:** matplotlib
- **Notebooks:** Jupyter (`.ipynb`)

## References

- Dataset: [Kaggle -- Lithium-Ion Battery Cycle Life Time Series Dataset](https://www.kaggle.com/datasets/solitaryseeker/lithium-ion-battery-cycle-life-time-series-dataset)
- Original data source: Severson et al., *Data-driven prediction of battery
  cycle life before capacity degradation*, Nature Energy (2019)
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [Anthropic API documentation](https://docs.claude.com)
- [Google Gemini API documentation](https://ai.google.dev/gemini-api/docs)
