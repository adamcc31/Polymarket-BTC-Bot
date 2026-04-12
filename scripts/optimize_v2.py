"""
optimize_v2.py — Grid Search Optimizer for Meta-Brain v2.0 execution thresholds.

Sweeps margin_of_safety, max_live_edge, max_buy_price, and max_ttr_minutes
across the trade_12042026.csv dataset to find the configuration that maximizes
PnL while maintaining 60-75% Win Rate with >= 8 trades.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModelEnsemble
from src.config_manager import ConfigManager
from src.schemas import FeatureMetadata
from src.feature_engine import FEATURE_NAMES


def run_grid_search():
    print("=" * 60)
    print(" META-BRAIN V2.0 — GRID SEARCH THRESHOLD OPTIMIZER")
    print("=" * 60)

    # ── Load Model ──
    config = ConfigManager()
    model = ModelEnsemble(config)
    model.load_latest()
    assert model._has_meta_v2, "Meta-Brain v2 not loaded!"

    # ── Load Trade Data ──
    df = pd.read_csv("dataset/trade_12042026.csv")
    print(f"Loaded {len(df)} trades from dataset.")

    # ── Pre-compute V2 predictions for every row ──
    print("Pre-computing Meta-Brain v2 inference for all rows...")
    v2_probs = []
    for _, row in df.iterrows():
        raw_p = row['P_model']
        live_edge = row['live_edge']
        clob_ask = raw_p - live_edge
        spread_yes = 0.04
        mid_yes = clob_ask - (spread_yes / 2)

        feature_vector = np.zeros(len(FEATURE_NAMES))
        if 'clob_yes_mid' in FEATURE_NAMES:
            feature_vector[FEATURE_NAMES.index('clob_yes_mid')] = mid_yes
        if 'clob_yes_spread' in FEATURE_NAMES:
            feature_vector[FEATURE_NAMES.index('clob_yes_spread')] = spread_yes

        metadata = FeatureMetadata(
            timestamp=datetime.now(timezone.utc),
            bar_close_time=datetime.now(timezone.utc),
            market_id=row['market_id'],
            strike_price=row['strike_price'],
            current_btc_price=row.get('btc_price_at_trigger', row['strike_price']),
            TTR_minutes=row['TTR_minutes'],
            TTR_phase="ENTRY_WINDOW",
            clob_ask=clob_ask,
            compute_lag_ms=10.0
        )
        final_p = model.predict(feature_vector, metadata)
        v2_probs.append(final_p)

    df['v2_prob'] = v2_probs
    print(f"V2 Predictions: min={min(v2_probs):.4f}, max={max(v2_probs):.4f}, mean={np.mean(v2_probs):.4f}")

    # ── Grid Definition ──
    margin_of_safety_grid = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    max_live_edge_grid = [0.12, 0.15, 0.18, 0.22, 0.30]
    max_buy_price_grid = [0.55, 0.60, 0.65, 0.70]
    max_ttr_grid = [3.5, 4.0, 4.5, 5.0]

    combos = list(product(
        margin_of_safety_grid, max_live_edge_grid, max_buy_price_grid, max_ttr_grid
    ))
    print(f"Total grid combinations: {len(combos)}")

    # ── Grid Sweep ──
    results = []

    for mos, mle, mbp, mttr in combos:
        wins = 0
        losses = 0
        total_pnl = 0.0
        starting_capital = 30.0
        current_capital = starting_capital
        max_capital = starting_capital
        max_dd = 0.0

        for _, row in df.iterrows():
            final_p = row['v2_prob']
            ttr = row['TTR_minutes']
            raw_p = row['P_model']
            live_edge_orig = row['live_edge']
            clob_ask = raw_p - live_edge_orig
            entry_price = row['entry_price_usdc']
            pnl = row['pnl_usd']

            # TTR filter
            if ttr > mttr:
                continue

            # Edge calculation against v2 probability
            v2_edge = abs(final_p - clob_ask)

            # Margin of safety: edge must exceed threshold
            if v2_edge < mos:
                continue

            # Anti-hallucination: reject unrealistically high edge
            if v2_edge > mle:
                continue

            # Max buy price cap
            if entry_price > mbp:
                continue

            # Coinflip rejection
            if 0.40 <= final_p <= 0.60:
                continue

            # Trade passes all gates
            if pd.isna(pnl):
                continue

            if row['outcome'] == 'WIN':
                wins += 1
            else:
                losses += 1

            total_pnl += pnl
            current_capital += pnl
            if current_capital > max_capital:
                max_capital = current_capital
            dd = max_capital - current_capital
            if dd > max_dd:
                max_dd = dd

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        results.append({
            'margin_of_safety': mos,
            'max_live_edge': mle,
            'max_buy_price': mbp,
            'max_ttr': mttr,
            'trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_dd,
            'final_capital': starting_capital + total_pnl,
        })

    # ── Filter and Rank ──
    df_results = pd.DataFrame(results)

    # Filter: 60-75% WR + min 8 trades
    qualified = df_results[
        (df_results['win_rate'] >= 60.0) &
        (df_results['win_rate'] <= 75.0) &
        (df_results['trades'] >= 8)
    ].sort_values('total_pnl', ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f" GRID SEARCH COMPLETE")
    print(f" Total Configs Tested: {len(df_results)}")
    print(f" Qualified Configs (60-75% WR, >=8 trades): {len(qualified)}")
    print(f"{'=' * 60}")

    if len(qualified) == 0:
        # Relax constraints progressively
        print("\n[!] No configs met strict criteria. Relaxing to 55-80% WR, >=5 trades...")
        qualified = df_results[
            (df_results['win_rate'] >= 55.0) &
            (df_results['win_rate'] <= 80.0) &
            (df_results['trades'] >= 5)
        ].sort_values('total_pnl', ascending=False).reset_index(drop=True)

        if len(qualified) == 0:
            print("\n[!] Still no configs. Relaxing to >50% WR, >=3 trades...")
            qualified = df_results[
                (df_results['win_rate'] > 50.0) &
                (df_results['trades'] >= 3)
            ].sort_values('total_pnl', ascending=False).reset_index(drop=True)

    if len(qualified) == 0:
        print("\n[X] FATAL: No profitable configuration found. Dataset may be too small or adversarial.")
        # Still show top 5 by PnL regardless
        top_all = df_results.sort_values('total_pnl', ascending=False).head(5)
        print("\nTop 5 by PnL (no WR filter):")
        for i, r in top_all.iterrows():
            print(f"  [{i+1}] MoS={r['margin_of_safety']:.2f} MLE={r['max_live_edge']:.2f} "
                  f"MBP={r['max_buy_price']:.2f} TTR<={r['max_ttr']:.1f} | "
                  f"Trades={r['trades']}, WR={r['win_rate']:.1f}%, "
                  f"PnL=${r['total_pnl']:.2f}, DD=${r['max_drawdown']:.2f}")
        return

    # Show Top 3
    top_n = min(3, len(qualified))
    print(f"\n{'-' * 60}")
    print(f" TOP {top_n} CONFIGURATIONS")
    print(f"{'-' * 60}")
    for rank in range(top_n):
        r = qualified.iloc[rank]
        tag = "(** BEST **)" if rank == 0 else ""
        print(f"\n  +== RANK #{rank+1} {tag}")
        print(f"  | margin_of_safety  = {r['margin_of_safety']:.2f}")
        print(f"  | max_live_edge     = {r['max_live_edge']:.2f}")
        print(f"  | max_buy_price     = {r['max_buy_price']:.2f}")
        print(f"  | max_ttr_minutes   = {r['max_ttr']:.1f}")
        print(f"  +--------------------------------------")
        print(f"  | Trades: {int(r['trades'])}  (W:{int(r['wins'])} / L:{int(r['losses'])})")
        print(f"  | Win Rate:        {r['win_rate']:.2f}%")
        print(f"  | Total PnL:       ${r['total_pnl']:.2f}")
        print(f"  | Final Capital:   ${r['final_capital']:.2f}")
        print(f"  | Max Drawdown:    ${r['max_drawdown']:.2f}")
        print(f"  +======================================")

    # -- Auto-inject Rank #1 into config.json --
    best = qualified.iloc[0]
    config_path = Path("config/config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg["signal"]["margin_of_safety"] = best["margin_of_safety"]
    cfg["risk"]["max_live_edge"] = best["max_live_edge"]
    cfg["risk"]["max_buy_price"] = best["max_buy_price"]
    cfg["signal"]["min_ttr_minutes"] = best["max_ttr"]

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f" [OK] RANK #1 INJECTED INTO config/config.json")
    print(f"    signal.margin_of_safety = {best['margin_of_safety']:.2f}")
    print(f"    risk.max_live_edge      = {best['max_live_edge']:.2f}")
    print(f"    risk.max_buy_price      = {best['max_buy_price']:.2f}")
    print(f"    signal.min_ttr_minutes  = {best['max_ttr']:.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_grid_search()
