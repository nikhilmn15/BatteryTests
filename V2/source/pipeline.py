"""
Extremely important as we excluded early life data from our train sets because of the blowing up of
RUL when being trained with it as evident in V1 where it was capped at 15k. However actual implementation now needs that 
baseline early life cycles which is why this isnt just a short pipeline fucntion

run_battery_pipeline() is what agent.py calls and this pipeline makes sure all 5 get called at the same time
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import models


def get_battery_baseline(battery_id: str, df: pd.DataFrame) -> dict:
    """Median IR/chargetime over a battery's first 5 (non-anomalous) cycles and
    the reference point every ratio feature is computed against."""
    b = df[df["battery_id"] == battery_id].sort_values("cycle")
    if b.empty:
        raise ValueError(f"No data found for battery_id={battery_id}")
    valid = b[~b["is_capacity_anomaly"]] if "is_capacity_anomaly" in b.columns else b
    valid = valid if len(valid) >= 1 else b
    return {
        "IR": float(valid["IR"].iloc[:5].median()),
        "chargetime": float(valid["chargetime"].iloc[:5].median()),
    }


def get_latest_cycle_features(battery_id: str, df: pd.DataFrame) -> dict:
    """Pulls the most recent recorded cycle's raw readings for a battery and
    the 'current state' the agent will estimate SOH/RUL/etc. from."""
    b = df[df["battery_id"] == battery_id].sort_values("cycle")
    if b.empty:
        raise ValueError(f"No data found for battery_id={battery_id}")
    latest = b.iloc[-1]
    return {
        "IR": float(latest["IR"]), "chargetime": float(latest["chargetime"]),
        "Tavg": float(latest["Tavg"]), "Tmin": float(latest["Tmin"]), "Tmax": float(latest["Tmax"]),
        "C1": float(latest["C1"]), "Q1": float(latest["Q1"]), "C2": float(latest["C2"]),
    }, int(latest["cycle"])


def run_battery_pipeline(battery_id: str, df: pd.DataFrame) -> dict:
    """Runs all the 5 models cuz I want the agent to decide what to mention while still running all of it"""
    baseline = get_battery_baseline(battery_id, df)
    raw_features, cycle_num = get_latest_cycle_features(battery_id, df)

    soh = models.predict_soh(raw_features, baseline)
    rul = models.predict_rul(raw_features, baseline)
    replacement = models.recommend_replacement(raw_features, baseline)
    anomaly = models.detect_anomaly(raw_features, baseline)
    stage = models.get_degradation_stage(raw_features, baseline)

    return {
        "Battery ID": battery_id,
        "Cycles Run": cycle_num,
        "SOH": round(soh, 4),
        "RUL Cycles": round(rul, 1),
        "Replacement Condition": replacement["needs_replacement"],
        "Probability of Replacement": round(replacement["confidence"], 3) if replacement["confidence"] is not None else None,
        "Extent of Degradation": stage,
        "Anomalous Behaviour": anomaly["is_anomalous"],
        "Anomaly Score": round(anomaly["anomaly_score"], 4),
        "Rule Violations": anomaly["flagged_by_rule"],
        "Variable Anomalies": anomaly["top_reasons"],
    }


if __name__ == "__main__":
    import sys as _sys
    battery_id = _sys.argv[1] if len(_sys.argv) > 1 else None
    df = models.load_clean_data()
    if battery_id is None:
        battery_id = df["battery_id"].iloc[0]
        print(f"(no battery_id given, using {battery_id})")
    result = run_battery_pipeline(battery_id, df)
    for k, v in result.items():
        print(f"{k}: {v}")