import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Preprocessing, Unsupervised Segmentation & Scaling
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Candidate Estimators
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Validation Metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

def run_model_tournament(X_train, X_test, y_train, y_test, candidates_dict, task_name="Task"):
    """
    Trains multiple candidate estimators for a target task, prints an evaluation 
    matrix, and automatically extracts the optimal champion model.
    """
    best_score = -float('inf')
    champion_name = None
    champion_instance = None
    champion_metrics = {}
    
    is_regression = not np.all(np.equal(np.mod(y_train, 1), 0)) or len(np.unique(y_train)) > 2
    metric_to_optimize = "R²" if is_regression else "F1-Score"
    
    print(f"\n⚡ Starting Competitive Algorithm Tournament for [{task_name.upper()}] ⚡")
    print("-" * 75)
    
    for name, model in candidates_dict.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        if is_regression:
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f" -> {name:<28} | Validation R²: {r2:.4f} | RMSE: {rmse:.5f}")
            score = r2
            metrics = {"R2": r2, "RMSE": rmse}
        else:
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, zero_division=0)
            print(f" -> {name:<28} | Validation Acc: {acc:.4f} | F1-Score: {f1:.4f}")
            score = f1
            metrics = {"Accuracy": acc, "F1-Score": f1}
            
        if score > best_score:
            best_score = score
            champion_name = name
            champion_instance = model
            champion_metrics = metrics
            
    print(f"🏆 CHAMPION FOR {task_name.upper()}: {champion_name} (Optimized via {metric_to_optimize})")
    print("-" * 75)
    return champion_instance, champion_name, champion_metrics


def run_battery_ml_pipeline():
    # Structural Path Targets
    PROCESSED_DIR = Path(r"C:\Users\Nikhil\OneDrive\Desktop\BatteryTests\data\processed")
    MODEL_SAVE_DIR = Path(r"C:\Users\Nikhil\OneDrive\Desktop\BatteryTests\models")
    MAIN_FILE = PROCESSED_DIR / "processed_Lithium-Ion Battery Cycle Life.csv"
    
    # Secure storage verification block
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MAIN_FILE.exists():
        print(f"❌ Execution Failure: Transformed asset matrix file not located at: {MAIN_FILE.resolve()}")
        print("Please run features.py first to establish the processed data baseline.")
        return

    print("┌── Ingesting fully engineered Approach B dataset...")
    df = pd.read_csv(MAIN_FILE)
    
    engineered_targets = ['target_soh', 'target_rul_cycles', 'target_replace']
    unwanted_cols = ['battery_id', 'cycle_life', 'cycle']
    feature_cols = [col for col in df.columns if col not in engineered_targets and col not in unwanted_cols]
    
    X = df[feature_cols]
    y_soh = df['target_soh']
    y_rul = df['target_rul_cycles']
    y_replace = df['target_replace']
    groups = df['battery_id'].values
    
    # 1. GroupShuffleSplit to isolate complete asset lifecycles
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, groups=groups))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_soh_train, y_soh_test = y_soh.iloc[train_idx], y_soh.iloc[test_idx]
    y_rul_train, y_rul_test = y_rul.iloc[train_idx], y_rul.iloc[test_idx]
    y_replace_train, y_replace_test = y_replace.iloc[train_idx], y_replace.iloc[test_idx]
    
    print(f"├── Zero-Leakage Split Secure. Train Size: {X_train.shape[0]} | Test Size: {X_test.shape[0]}")
    
    # 2. Scaling Boundary Transformer
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    # 3. Unsupervised Profiler: K-Means
    print("├── Initializing Unsupervised K-Means clustering segmentation...")
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled_df)
    test_clusters = kmeans.predict(X_test_scaled_df)
    
    for cluster_idx in range(3):
        X_train_scaled_df[f'cluster_flag_{cluster_idx}'] = (train_clusters == cluster_idx).astype(int)
        X_test_scaled_df[f'cluster_flag_{cluster_idx}'] = (test_clusters == cluster_idx).astype(int)

    # 4. Unsupervised Profiler: Isolation Forest Anomaly Detector
    print("├── Deploying Isolation Forest Anomaly Detection Engine...")
    iso_forest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_scaled_df)
    test_anomaly_flags = (iso_forest.predict(X_test_scaled_df) == -1).astype(int)

    # 5. Supervised Model Tournaments
    soh_candidates = {
        "Linear Ridge Regressor": Ridge(alpha=1.0),
        "Decision Tree Estimator": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting Trees": GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=6, random_state=42)
    }
    best_soh_model, soh_name, soh_metrics = run_model_tournament(
        X_train_scaled_df, X_test_scaled_df, y_soh_train, y_soh_test, soh_candidates, task_name="State of Health (SOH)"
    )
    
    rul_candidates = {
        "Linear Ridge Regressor": Ridge(alpha=10.0),
        "Decision Tree Estimator": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=30, max_depth=14, random_state=42, n_jobs=-1),
        "Gradient Boosting Trees": GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=7, random_state=42)
    }
    best_rul_model, rul_name, rul_metrics = run_model_tournament(
        X_train_scaled_df, X_test_scaled_df, y_rul_train, y_rul_test, rul_candidates, task_name="Remaining Useful Life (RUL)"
    )
    
    replace_candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=30, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=30, random_state=42)
    }
    best_replace_model, replace_name, replace_metrics = run_model_tournament(
        X_train_scaled_df, X_test_scaled_df, y_replace_train, y_replace_test, replace_candidates, task_name="Replacement Route Arbitration"
    )
    
    # ------------------------------------------------------------------------
    # STEP 6: MODEL EXPORT REGISTRY PIPELINE (SAVE BLOCKS)
    # ------------------------------------------------------------------------
    print(f"\n📦 Serializing pipeline states and components to: {MODEL_SAVE_DIR.resolve()}")
    print("-" * 75)
    
    artifacts_to_save = {
        "scaler.pkl": scaler,
        "kmeans_segmenter.pkl": kmeans,
        "anomaly_isolation_forest.pkl": iso_forest,
        "champion_soh_regressor.pkl": best_soh_model,
        "champion_rul_regressor.pkl": best_rul_model,
        "champion_replace_classifier.pkl": best_replace_model
    }
    
    for filename, artifact in artifacts_to_save.items():
        save_path = MODEL_SAVE_DIR / filename
        with open(save_path, 'wb') as f:
            pickle.dump(artifact, f)
        print(f" -> Successfully exported component: {filename}")

    print("-" * 75)
    print("\n" + "="*85)
    print("                      FINAL CHAMPION MODEL DEPLOYMENT MATRIX                   ")
    print("="*85)
    print(f" 🎯 1. SOH ESTIMATOR    : {soh_name:<25} | Target Validation R²    : {soh_metrics['R2']:.2%}")
    print(f" 🎯 2. RUL ESTIMATOR    : {rul_name:<25} | Target Validation R²    : {rul_metrics['R2']:.2%}")
    print(f" 🎯 3. ROUTE ARBITER    : {replace_name:<25} | Target Validation F1    : {replace_metrics['F1-Score']:.2%}")
    print(f" 🎯 4. ANOMALY ENGINE   : Isolation Forest Engine   | Total Anomalies Flagged : {np.sum(test_anomaly_flags)} rows")
    print("="*85)

if __name__ == "__main__":
    run_battery_ml_pipeline()