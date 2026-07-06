"""
models.py (MIT battery cycle-life dataset)
---------------------------------------------
Same 5-component architecture as the NASA pipeline:
    1. SOH regressor          -- predicts capacity ratio
    2. RUL regressor          -- predicts cycles remaining until EOL
    3. Replacement classifier -- predicts replace vs keep
    4. Anomaly detector       -- flags out-of-distribution cycles (guardrail)
    5. K-means clustering     -- degradation stage label (standalone insight)

Validation: GroupKFold (grouped by battery_id) -- NEVER random split, so no
battery's cycles leak across train/test. SVM/SVR excluded (too slow at this
row count for the marginal benefit seen on the NASA dataset).

Run: python3 models.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier,
                               IsolationForest)
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "ProcessedV2.csv"

# Ratio-normalized (protocol/battery-scale-independent) + protocol params.
# NOTE: QD, QC, Nominal_QD_Cap, and anything derived from them (e.g. the
# original pipeline's thermal_efficiency_index = QD/Tavg) are deliberately
# EXCLUDED -- see features.py docstring for why.
MODEL_FEATURES = ["IR_ratio", "chargetime_ratio", "Tavg", "Tmin", "Tmax", "C1", "Q1", "C2"]

# Clustering deliberately uses ONLY the two true degradation-ratio signals --
# NOT Tavg/Tmin/Tmax/C1/Q1/C2. Those are protocol/test-condition features:
# every protocol group contains batteries across all health levels, so
# clustering on them groups by PROTOCOL, not degradation stage (confirmed
# bug: initial run produced clusters with nearly identical mean_soh --
# 0.966/0.961/0.960 -- because it was splitting by charging protocol, not
# health). Same fix applied to the NASA pipeline's clustering for the same
# reason (ambient_temperature/protocol dummies excluded there too).
CLUSTER_FEATURES = ["IR_ratio", "chargetime_ratio"]

# Anomaly detector uses raw (unratio'd) values too -- absolute-scale weirdness
# is itself a valid anomaly signal, not just relative degradation weirdness.
BEHAVIORAL_FEATURES = ["IR", "Tavg", "Tmin", "Tmax", "chargetime", "C1", "Q1", "C2"]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_model_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[(~df["is_capacity_anomaly"]) & (~df["is_ratio_anomaly"])].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=MODEL_FEATURES + ["soh"])
    return clean


# ---------------------------------------------------------------------
# GroupKFold evaluation harness
# ---------------------------------------------------------------------

def evaluate_regression(model_ctor, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(X[tr]), scaler.transform(X[te])
        m = model_ctor()
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
        r2s.append(r2_score(y[te], p))
    return {"rmse": np.mean(rmses), "mae": np.mean(maes), "r2": np.mean(r2s)}


def evaluate_classification(model_ctor, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    f1s, accs = [], []
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(X[tr]), scaler.transform(X[te])
        m = model_ctor()
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        f1s.append(f1_score(y[te], p, zero_division=0))
        accs.append(accuracy_score(y[te], p))
    return {"f1": np.mean(f1s), "accuracy": np.mean(accs)}


# ---------------------------------------------------------------------
# 1. SOH regressor
# ---------------------------------------------------------------------

SOH_CANDIDATES = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}


def train_soh(df: pd.DataFrame):
    data = get_model_data(df)
    X, y, groups = data[MODEL_FEATURES].values, data["soh"].values, data["battery_id"].values

    print("\n=== SOH regressor (GroupKFold, 5 folds) ===")
    results = {}
    for name, ctor in SOH_CANDIDATES.items():
        res = evaluate_regression(ctor, X, y, groups)
        results[name] = res
        print(f"{name:18s} RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  R2={res['r2']:.4f}")

    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"Best: {best}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = SOH_CANDIDATES[best]()
    model.fit(Xs, y)
    joblib.dump({"model": model, "scaler": scaler, "algorithm": best, "features": MODEL_FEATURES},
                MODELS_DIR / "soh.pkl")
    return results, best


# ---------------------------------------------------------------------
# 2. RUL regressor
# ---------------------------------------------------------------------

RUL_CANDIDATES = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}


def train_rul(df: pd.DataFrame):
    data = get_model_data(df)
    X, y, groups = data[MODEL_FEATURES].values, data["rul_cycles"].values, data["battery_id"].values

    print(f"\n=== RUL regressor (GroupKFold, 5 folds, n={len(data)}) ===")
    results = {}
    for name, ctor in RUL_CANDIDATES.items():
        res = evaluate_regression(ctor, X, y, groups)
        results[name] = res
        print(f"{name:18s} RMSE={res['rmse']:.2f}  MAE={res['mae']:.2f}  R2={res['r2']:.4f}")

    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"Best: {best}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RUL_CANDIDATES[best]()
    model.fit(Xs, y)
    joblib.dump({"model": model, "scaler": scaler, "algorithm": best, "features": MODEL_FEATURES},
                MODELS_DIR / "rul.pkl")
    return results, best


# ---------------------------------------------------------------------
# 3. Replacement classifier
# ---------------------------------------------------------------------

REPLACEMENT_CANDIDATES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                                    class_weight="balanced", n_jobs=-1),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
}


def train_replacement(df: pd.DataFrame):
    data = get_model_data(df)
    X = data[MODEL_FEATURES].values
    y = data["needs_replacement"].values
    groups = data["battery_id"].values

    print(f"\n=== Replacement classifier (GroupKFold, 5 folds) -- class balance: {y.mean():.3f} positive ===")
    results = {}
    for name, ctor in REPLACEMENT_CANDIDATES.items():
        res = evaluate_classification(ctor, X, y, groups)
        results[name] = res
        print(f"{name:18s} F1={res['f1']:.4f}  Accuracy={res['accuracy']:.4f}")

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"Best: {best}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = REPLACEMENT_CANDIDATES[best]()
    model.fit(Xs, y)
    joblib.dump({"model": model, "scaler": scaler, "algorithm": best, "features": MODEL_FEATURES},
                MODELS_DIR / "replacement.pkl")
    return results, best


# ---------------------------------------------------------------------
# 4. Anomaly detector (guardrail)
# ---------------------------------------------------------------------

def train_anomaly_detector(df: pd.DataFrame):
    data = get_model_data(df)
    X = data[BEHAVIORAL_FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    iso.fit(Xs)

    feature_stats = {c: {"mean": float(data[c].mean()), "std": float(data[c].std())} for c in BEHAVIORAL_FEATURES}
    flagged = iso.predict(Xs) == -1
    print(f"\n=== Anomaly detector (Isolation Forest) ===")
    print(f"Trained on {len(X)} cycles. Flagged {flagged.sum()} ({flagged.mean()*100:.1f}%).")

    joblib.dump({"model": iso, "scaler": scaler, "features": BEHAVIORAL_FEATURES, "feature_stats": feature_stats},
                MODELS_DIR / "anomaly.pkl")
    return iso


# ---------------------------------------------------------------------
# 5. K-means degradation-stage clustering (standalone insight)
# ---------------------------------------------------------------------

def train_clustering(df: pd.DataFrame):
    data = get_model_data(df)
    X = data[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    data = data.copy()
    data["cluster"] = labels

    order = data.groupby("cluster")["soh"].mean().sort_values(ascending=False).index.tolist()
    stage_names = ["Healthy", "Transitional", "Critical"]
    label_map = {cid: stage_names[i] for i, cid in enumerate(order)}

    print("\n=== K-means degradation stage clustering ===")
    summary = data.groupby("cluster").agg(n=("cluster", "size"), mean_soh=("soh", "mean")).round(3)
    summary["stage"] = summary.index.map(label_map)
    print(summary.sort_values("mean_soh", ascending=False))

    joblib.dump({"model": km, "scaler": scaler, "label_map": label_map, "features": CLUSTER_FEATURES},
                MODELS_DIR / "cluster.pkl")
    return km, label_map


# ---------------------------------------------------------------------
# Agent-facing inference functions
# ---------------------------------------------------------------------

def _to_vector(features: dict, columns: list) -> np.ndarray:
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return np.array([[features[c] for c in columns]])


def _build_ratio_features(raw: dict, battery_baseline: dict) -> dict:
    """raw: this cycle's IR/chargetime/Tavg/Tmin/Tmax/C1/Q1/C2.
    battery_baseline: {'IR': <median of this battery's first 5 cycles>,
                        'chargetime': <same>}"""
    return {
        "IR_ratio": raw["IR"] / battery_baseline["IR"],
        "chargetime_ratio": raw["chargetime"] / battery_baseline["chargetime"],
        "Tavg": raw["Tavg"], "Tmin": raw["Tmin"], "Tmax": raw["Tmax"],
        "C1": raw["C1"], "Q1": raw["Q1"], "C2": raw["C2"],
    }


def predict_soh(raw_features: dict, battery_baseline: dict) -> float:
    bundle = joblib.load(MODELS_DIR / "soh_model.pkl")
    feats = _build_ratio_features(raw_features, battery_baseline)
    X = bundle["scaler"].transform(_to_vector(feats, bundle["features"]))
    return float(bundle["model"].predict(X)[0])


def predict_rul(raw_features: dict, battery_baseline: dict) -> float:
    bundle = joblib.load(MODELS_DIR / "rul_model.pkl")
    feats = _build_ratio_features(raw_features, battery_baseline)
    X = bundle["scaler"].transform(_to_vector(feats, bundle["features"]))
    return max(0.0, float(bundle["model"].predict(X)[0]))


def recommend_replacement(raw_features: dict, battery_baseline: dict) -> dict:
    bundle = joblib.load(MODELS_DIR / "replacement_model.pkl")
    feats = _build_ratio_features(raw_features, battery_baseline)
    X = bundle["scaler"].transform(_to_vector(feats, bundle["features"]))
    pred = bundle["model"].predict(X)[0]
    proba = bundle["model"].predict_proba(X)[0][1] if hasattr(bundle["model"], "predict_proba") else None
    return {"needs_replacement": bool(pred), "confidence": float(proba) if proba is not None else None}


def detect_anomaly(raw_features: dict) -> dict:
    bundle = joblib.load(MODELS_DIR / "anomaly_model.pkl")
    X = bundle["scaler"].transform(_to_vector(raw_features, bundle["features"]))
    score = float(bundle["model"].decision_function(X)[0])
    is_anom = bool(bundle["model"].predict(X)[0] == -1)
    stats = bundle["feature_stats"]
    z = {c: abs((raw_features[c] - stats[c]["mean"]) / stats[c]["std"]) if stats[c]["std"] > 0 else 0.0
         for c in bundle["features"]}
    top = sorted(z.items(), key=lambda kv: -kv[1])[:3]
    return {"is_anomalous": is_anom, "anomaly_score": score,
            "top_reasons": [{"feature": f, "z_score": round(v, 2)} for f, v in top]}


def get_degradation_stage(raw_features: dict, battery_baseline: dict) -> str:
    bundle = joblib.load(MODELS_DIR / "cluster_model.pkl")
    feats = {
        "IR_ratio": raw_features["IR"] / battery_baseline["IR"],
        "chargetime_ratio": raw_features["chargetime"] / battery_baseline["chargetime"],
    }
    X = bundle["scaler"].transform(_to_vector(feats, bundle["features"]))
    cid = int(bundle["model"].predict(X)[0])
    return bundle["label_map"][cid]

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df = load_clean_data()

    soh_results, soh_best = train_soh(df)
    rul_results, rul_best = train_rul(df)
    repl_results, repl_best = train_replacement(df)
    train_anomaly_detector(df)
    train_clustering(df)

    print("\n=== Summary: best algorithm per task ===")
    print(f"SOH regressor:          {soh_best}")
    print(f"RUL regressor:          {rul_best}")
    print(f"Replacement classifier: {repl_best}")
    print(f"Anomaly detector:       IsolationForest")
    print(f"Clustering:             KMeans (k=3)")
    print(f"\nAll models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()