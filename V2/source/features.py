"""
features.py (MIT battery cycle-life dataset)
----------------------------------------------
Same design principles as the NASA pipeline's features.py:
  - Raw capacity (QD/QC) is used ONLY to compute labels, never as a model
    input -- avoids the exact leakage found in the original tool-generated
    pipeline (target_soh was defined as QD/Nominal_QD_Cap, and both QD and
    Nominal_QD_Cap were then fed back in as features).
  - RUL is computed via direct threshold logic (cycle_life - cycle, since
    this dataset's own cycle_life field IS the recorded EOL cycle), NOT the
    "average degradation rate since cycle 1" formula from the original
    pipeline -- that formula misbehaves on realistic fast-then-slow fade
    curves (produces INCREASING "RUL" over time), not just the fresh-battery
    edge case it was designed to guard against.
  - Absolute-scale features (IR, chargetime) are protocol/battery-dependent
    in magnitude -- ratio-normalized against each battery's own early-life
    baseline, same fix that mattered throughout the NASA pipeline.

Run order: clean_data() -> build_feature_table() -> add_labels()
"""

import numpy as np
import pandas as pd

RATIO_COLS = ["IR", "chargetime"]
EOL_SOH_THRESHOLD = 0.80  # standard convention for this dataset/benchmark


def _flag_anomalies(s: pd.Series, window: int = 5, mad_thresh: float = 5) -> pd.Series:
    """Robust local outlier flag (median + MAD), same logic used throughout
    the NASA pipeline -- catches isolated bad readings without being thrown
    off by the outliers it's trying to catch."""
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median().replace(0, np.nan)
    return ((s - med).abs() / (1.4826 * mad)) > mad_thresh


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

    # --- flag corrupted QD readings before using them for anything ---
    df["is_capacity_anomaly"] = df.groupby("battery_id")["QD"].transform(_flag_anomalies).fillna(False)
    df["is_capacity_anomaly"] = df["is_capacity_anomaly"] | (df["QD"] <= 0)

    # --- rated capacity: max of first 5 VALID cycles (deployment-realistic --
    # a real monitoring system wouldn't know the battery's lifetime-peak
    # capacity in advance; only its own early-life behavior) ---
    def _rated(group):
        valid = group.loc[~group["is_capacity_anomaly"], "QD"]
        window = valid.iloc[:5] if len(valid) >= 1 else group["QD"].iloc[:5]
        return window.max()
    df["capacity_rated"] = df["battery_id"].map(df.groupby("battery_id", group_keys=False).apply(_rated))

    # --- ratio-normalize scale-dependent features against battery's own baseline ---
    for col in RATIO_COLS:
        baseline = df.groupby("battery_id")[col].transform(lambda s: s.iloc[:5].median())
        df[f"{col}_ratio"] = df[col] / baseline

    # Occasional corrupted single-cycle readings (e.g. a chargetime logged at
    # 50-96x a battery's own baseline -- a sensor glitch, not real degradation)
    # distort everything downstream that uses these ratios, especially
    # clustering (a handful of glitch rows can hijack k-selection entirely by
    # forming a trivially-separable "glitch vs everyone" cluster). Real
    # degradation ratios for this dataset stay well under 5x baseline.
    df["is_ratio_anomaly"] = False
    for col in RATIO_COLS:
        df["is_ratio_anomaly"] = df["is_ratio_anomaly"] | (df[f"{col}_ratio"] > 5) | (df[f"{col}_ratio"] < 0)

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soh"] = df["QD"] / df["capacity_rated"]

    # RUL: direct, no division-by-rate formula -- this dataset's cycle_life
    # field already IS the recorded EOL cycle (verified: max(cycle) per
    # battery == cycle_life - 1 consistently).
    df["rul_cycles"] = (df["cycle_life"] - df["cycle"]).clip(lower=0)

    df["needs_replacement"] = (df["soh"] < EOL_SOH_THRESHOLD).astype(int)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from loader import load_raw, clean_data

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Lithium-Ion Battery Cycle Life.csv"
    df = add_labels(build_feature_table(clean_data(load_raw(path))))
    print(df[["battery_id", "cycle", "QD", "soh", "rul_cycles", "needs_replacement"]].head(8))
    df.to_csv("data/processed/ProcessedV2.csv", index=False)
    print(f"\nSaved {len(df)} rows, {df['battery_id'].nunique()} batteries")