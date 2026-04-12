import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report

def train():
    print("Loading data...")
    df = pd.read_csv("dataset/market_data_2sec_weekly5_with_resolutions.csv")
    
    print(f"Initial rows: {len(df)}")
    
    # --- Feature Engineering ---
    df['TTR_minutes'] = (300 - df['elapsed']) / 60.0
    df['gap_absolute'] = df['btc_gap']
    df['gap_percentage'] = (df['btc_gap'] / df['btc_strike']) * 100.0
    df['spread_YES'] = df['ask_YES'] - df['bid_YES']
    df['mid_YES'] = (df['ask_YES'] + df['bid_YES']) / 2.0
    
    # Target Mapping Binary
    df['target'] = (df['winner'].str.upper() == 'UP').astype(int)
    
    # --- Filtering Garbage Data ---
    df = df[df['elapsed'] >= 0]
    df = df[df['elapsed'] <= 300]
    df = df[df['spread_YES'] >= 0]
    df = df[df['spread_YES'] <= 0.20]  # Filter out extremely wide spreads
    df = df[df['ask_YES'] > 0]
    df = df[df['bid_YES'] > 0]
    df = df[df['mid_YES'] > 0]
    df = df[df['mid_YES'] < 1]
    
    df = df.dropna(subset=['TTR_minutes', 'gap_absolute', 'gap_percentage', 'spread_YES', 'mid_YES', 'target', 'timestamp_log'])
    
    print(f"Rows after filtering: {len(df)}")
    
    # --- Chronological Split Series ---
    # We use 70% Train, 15% Validation (for calibration), 15% Test
    df = df.sort_values('timestamp_log')
    split_1 = int(len(df) * 0.70)
    split_2 = int(len(df) * 0.85)

    features = ['TTR_minutes', 'gap_absolute', 'gap_percentage', 'spread_YES', 'mid_YES']

    X_train = df.iloc[:split_1][features]
    y_train = df.iloc[:split_1]['target']
    
    X_val = df.iloc[split_1:split_2][features]
    y_val = df.iloc[split_1:split_2]['target']
    
    X_test = df.iloc[split_2:][features]
    y_test = df.iloc[split_2:]['target']
    
    print("Training LGBMClassifier...")
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    
    print("Applying Isotonic Regression Calibration on Holdout Validation Set...")
    val_preds_raw = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_preds_raw, y_val)
    
    # --- Evaluation on Hidden Test Set ---
    test_preds_raw = model.predict_proba(X_test)[:, 1]
    test_preds_cal = calibrator.transform(test_preds_raw)
    test_preds_binary = (test_preds_cal >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, test_preds_cal)
    brier = brier_score_loss(y_test, test_preds_cal)
    
    print("\n==================================")
    print("      META-BRAIN V2: METRICS      ")
    print("==================================")
    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"Brier Score   : {brier:.4f}")
    print("----------------------------------")
    print("Classification Report:")
    print(classification_report(y_test, test_preds_binary))
    print("==================================\n")
    
    # --- Export ---
    os.makedirs('models', exist_ok=True)
    with open('models/meta_v2_lgbm.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/meta_v2_calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)
        
    print("Artifacts successfully generated:")
    print("- models/meta_v2_lgbm.pkl")
    print("- models/meta_v2_calibrator.pkl")

if __name__ == "__main__":
    train()
