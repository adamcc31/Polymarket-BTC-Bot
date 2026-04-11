import pandas as pd
import numpy as np
import pickle
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_manager import ConfigManager
from src.model import ModelEnsemble, _get_model_dir

def run_meta_replay():
    print("Starting Meta-Replay Backtest (SOT-Driven)...")
    
    # 1. Load Config
    config = ConfigManager()
    taker_edge_threshold = config.get("execution.taker_edge_threshold", 0.04)
    print(f"Using taker_edge_threshold: {taker_edge_threshold}")
    
    # 2. Load Model
    model = ModelEnsemble(config)
    if not model.load_latest():
        print("Error: Could not load latest ModelEnsemble artifacts.")
        return
    
    if not model._has_meta_brain:
        print("Error: Meta-Brain artifacts not found. Cannot proceed with stacking backtest.")
        return
    
    print(f"Model loaded successfully (Version: {model.version})")
    
    # 3. Load SOT Dataset
    sot_path = ROOT_DIR / "sot_dataset_ready.csv"
    if not sot_path.exists():
        print(f"Error: SOT dataset not found at {sot_path}")
        return
        
    df_sot = pd.read_csv(sot_path)
    total_opps = len(df_sot)
    print(f"Loaded {total_opps} market-level opportunities from SOT.")
    
    # 4. Simulation
    approved_trades = 0
    coinflip_rejections = 0
    edge_rejections = 0
    
    total_pnl = 0.0
    wins = 0
    
    results = []
    
    for _, row in df_sot.iterrows():
        # Map SOT to Meta features
        raw_p = row['avg_P_model']
        ttr_min = row['avg_TTR_minutes']
        live_edge = row['avg_live_edge']
        dist_bps = row['distance_to_strike_bps']
        
        # Approximate clob_ask (base for edge calculation)
        clob_ask = raw_p - live_edge
        
        # Construct Meta-Feature Vector (matches retrain_meta_calibrator.py order)
        # [dist_bps, TTR_minutes, raw_p, live_edge, is_coinflip]
        is_coinflip_flag = 1.0 if 0.40 <= raw_p <= 0.60 else 0.0
        
        meta_X = np.array([
            dist_bps,
            ttr_min,
            raw_p,
            live_edge,
            is_coinflip_flag
        ]).reshape(1, -1)
        
        # Meta-Inference (replicating ModelEnsemble.predict's Stage 2)
        meta_lgbm_prob = model._meta_lgbm.predict(meta_X)[0]
        
        meta_X_scaled = model._meta_scaler.transform(meta_X)
        meta_logreg_prob = model._meta_logreg.predict_proba(meta_X_scaled)[0, 1]
        
        # Meta-Ensemble (0.7 / 0.3)
        meta_p = (0.7 * meta_lgbm_prob + 0.3 * meta_logreg_prob)
        
        # Meta-Isotonic Calibration
        final_p = model._meta_calibrator_isotonic.transform([meta_p])[0]
        final_p = max(0.0, min(1.0, final_p))
        
        # --- EXECUTION LOGIC ---
        
        # 1. Coinflip Rejection
        if 0.40 <= final_p <= 0.60:
            coinflip_rejections += 1
            continue
            
        # 2. Edge Rejection
        meta_edge = abs(final_p - clob_ask)
        if meta_edge < taker_edge_threshold:
            edge_rejections += 1
            continue
            
        # TRADE APPROVED
        approved_trades += 1
        total_pnl += row['net_pnl_usd']
        if row['target_win'] == 1:
            wins += 1
            
        results.append({
            'final_p': final_p,
            'meta_edge': meta_edge,
            'ttr': ttr_min,
            'pnl': row['net_pnl_usd']
        })
        
    # 5. Reporting
    exec_rate = (approved_trades / total_opps) * 100 if total_opps > 0 else 0
    win_rate = (wins / approved_trades) * 100 if approved_trades > 0 else 0
    
    avg_ttr = np.mean([r['ttr'] for r in results]) if results else 0
    avg_meta_edge = np.mean([r['meta_edge'] for r in results]) if results else 0
    
    print("\n========================================")
    print("META-REPLAY BACKTEST REPORT")
    print("========================================")
    print(f"Total Opportunities:      {total_opps}")
    print(f"Execution Rate:           {exec_rate:.2f}% ({approved_trades}/{total_opps})")
    print(f"Coinflip Rejections:       {coinflip_rejections}")
    print(f"Edge Filter Rejections:    {edge_rejections}")
    print("----------------------------------------")
    print(f"AVG TTR at Entry:         {avg_ttr:.2f} minutes")
    print(f"AVG Meta-Edge Captured:    {avg_meta_edge:.4f}")
    print("----------------------------------------")
    print(f"Simulated Total PnL:       ${total_pnl:,.2f}")
    print(f"Simulated Win Rate:        {win_rate:.2f}%")
    
    if total_pnl > 0:
        print(f"Status:                    SUCCESS - Meta-Brain Outperforms Base Thresholds.")
    else:
        print(f"Status:                    CAUTION - Calibration suggests neutral bias.")
    print("========================================\n")

if __name__ == "__main__":
    run_meta_replay()
