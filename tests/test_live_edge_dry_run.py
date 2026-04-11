import pandas as pd
import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from src.config_manager import ConfigManager

def run_replay(csv_path: str):
    print(f"--- Replay Test: Live Edge Verification ---")
    print(f"Target CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Initialize ConfigManager
    config = ConfigManager.get_instance()
    
    # Extract Gate Parameters
    max_buy_price = float(config.get("risk.max_buy_price", 0.75))
    edge_tolerance = float(config.get("risk.live_edge_tolerance", 0.05))
    margin_of_safety = float(config.get("signal.margin_of_safety", 0.02))

    print(f"Config Parameters (Gates):")
    print(f"  max_buy_price: {max_buy_price}")
    print(f"  edge_tolerance: {edge_tolerance}")
    print(f"  margin_of_safety: {margin_of_safety}")
    print("-" * 50)

    total_rows = len(df)
    passed = 0
    failed = 0
    aborted_price = 0
    aborted_tolerance = 0
    aborted_edge = 0

    for idx, row in df.iterrows():
        trade_id = str(row.get("trade_id", f"row_{idx}"))[:8]
        real_best_ask = float(row.get("entry_price_usdc", 0.0))
        
        # Fallback for older CSV formats
        signal_type = row.get("signal_type", "BUY_YES")
        if "synthetic_edge" in row:
            synthetic_edge = float(row.get("synthetic_edge", 0.0))
            live_edge = float(row.get("live_edge", 0.0))
        else:
            # Re-derive from legacy columns
            synthetic_edge = float(row.get("edge_yes" if signal_type == "BUY_YES" else "edge_no", 0.0))
            # Assume live edge matches synthetic for legacy replay unless we have other data
            live_edge = synthetic_edge

        edge_deviation = abs(synthetic_edge - live_edge)

        print(f"[{idx+1}/{total_rows}] Trade {trade_id}:")
        print(f"  - Price: {real_best_ask} vs Cap {max_buy_price}")
        print(f"  - SynEdge: {round(synthetic_edge, 4)} | LiveEdge: {round(live_edge, 4)}")
        print(f"  - Deviation: {round(edge_deviation, 4)}")

        # Verification Logic (Mirroring main.py STEP 2)
        is_aborted = False
        reason = None

        if real_best_ask > max_buy_price:
            is_aborted = True
            reason = "PRICE_EXCEEDS_MAX_CAP"
            aborted_price += 1
        elif edge_deviation > edge_tolerance:
            is_aborted = True
            reason = "EDGE_DEVIATION_TOO_HIGH"
            aborted_tolerance += 1
        elif live_edge <= margin_of_safety:
            is_aborted = True
            reason = "LIVE_EDGE_NEGATIVE"
            aborted_edge += 1

        if is_aborted:
            print(f"  >>> RESULT: [REJECTED] Reason: {reason}")
            failed += 1
        else:
            print(f"  >>> RESULT: [PASSED]")
            passed += 1
        print()

    print("-" * 50)
    print(f"REPLAY SUMMARY:")
    print(f"  Total Trades Evaluated: {total_rows}")
    print(f"  Passed (Valid Edge):    {passed}")
    print(f"  Failed (Would Abort):   {failed}")
    if failed > 0:
        print(f"    - Price Cap Aborts: {aborted_price}")
        print(f"    - Tolerance Aborts: {aborted_tolerance}")
        print(f"    - Edge Strength Aborts: {aborted_edge}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay Polymarket trades against Live Edge gates.")
    parser.add_argument("csv_path", help="Path to trades.csv")
    args = parser.parse_args()
    
    run_replay(args.csv_path)
