import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model import _get_model_dir

MODELS_DIR = _get_model_dir()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ROOT_DIR / "sot_dataset_ready.csv"

def train_meta_calibrator():
    print("Starting Meta-Calibrator Training Pipeline...")
    
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Run generate_sot_dataset.py first.")
        return

    # Load SOT Dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} market-level samples.")

    # FEATURES: distance_to_strike_bps, avg_TTR_minutes, avg_P_model, avg_live_edge, is_coinflip
    FEATURES = ["distance_to_strike_bps", "avg_TTR_minutes", "avg_P_model", "avg_live_edge", "is_coinflip"]
    TARGET = "target_win"

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # 1. Chronological Split (Assume SOT is already somewhat ordered or metadata exists)
    # Since we don't have timestamps in SOT, and it's small, we'll do a 70/30 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}")

    # 2. Optimized LGBM (Stage 2)
    # Using 'binary' objective for Log Loss optimization
    lgbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": 5
    }

    print("Fitting Meta-LGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgbm_model = lgb.train(
        lgbm_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # 3. Logistic Regression Ensemble Component
    print("Fitting Meta-LogisticRegression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logreg = LogisticRegression(class_weight="balanced", random_state=42)
    logreg.fit(X_train_scaled, y_train)

    # 4. Ensemble Predictions (0.7 LGBM + 0.3 LogReg)
    lgbm_val_probs = lgbm_model.predict(X_val)
    logreg_val_probs = logreg.predict_proba(X_val_scaled)[:, 1]
    
    ensemble_val_probs = 0.7 * lgbm_val_probs + 0.3 * logreg_val_probs

    # 5. Isotonic Calibration
    print("Applying Isotonic Calibration...")
    # To use IsotonicRegression, we fit on the ensemble probabilities vs actual outcomes
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(ensemble_val_probs, y_val)

    # 6. Evaluation
    final_probs = calibrator.transform(ensemble_val_probs)
    
    brier = brier_score_loss(y_val, final_probs)
    lloss = log_loss(y_val, final_probs)
    
    print("\n" + "="*40)
    print("META-CALIBRATOR TRAINING SUMMARY")
    print("="*40)
    print(f"Brier Score:   {brier:.6f}")
    print(f"Log Loss:      {lloss:.6f}")
    
    # Calibration Curve Summary
    prob_true, prob_pred = calibration_curve(y_val, final_probs, n_bins=5)
    print("\nCalibration Curve (Empirical vs Predicted):")
    for t, p in zip(prob_true, prob_pred):
        print(f"  Bin Approx: Predicted={p:.2f} -> Realized={t:.2f}")

    # 7. Serialization (Overwrite existing if any, strictly artifacts for this model)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # We save these with 'meta_' prefix to distinguish from base model
    artifacts = {
        "meta_lgbm.pkl": lgbm_model,
        "meta_logreg.pkl": logreg,
        "meta_scaler.pkl": scaler,
        "meta_calibrator_isotonic.pkl": calibrator
    }

    for name, obj in artifacts.items():
        path = MODELS_DIR / name
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    
    print(f"\nMeta-Calibrator artifacts saved to {MODELS_DIR}")
    print("Directives achieved: Optimized for Log Loss & Brier Score with Isotonic Scaling.")

if __name__ == "__main__":
    train_meta_calibrator()
