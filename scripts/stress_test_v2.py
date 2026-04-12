"""
stress_test_v2.py -- Macro Stress Test for Meta-Brain v2.0 + God Mode Config.

Simulates the EXACT execution logic across 113,000+ rows of 2-second
microstructure data to validate that the optimized thresholds are not
overfitted to the 40-trade sample.

Rules:
  - TTR window: [1.5, 4.8] minutes
  - Edge window: [margin_of_safety, max_live_edge] = [0.15, 0.30]
  - Max buy price: 0.65
  - One trade per slug (first valid signal locks the position)
  - PnL: WIN = (1/entry_price - 1) * bet_size, LOSS = -bet_size
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModelEnsemble
from src.config_manager import ConfigManager
from src.feature_engine import FEATURE_NAMES


def run_stress_test():
    print("=" * 65)
    print(" META-BRAIN V2.0 -- 113K MACRO STRESS TEST")
    print("=" * 65)

    # -- Config Thresholds (God Mode) --
    MIN_TTR = 1.5
    MAX_TTR = 4.8
    MARGIN_OF_SAFETY = 0.15
    MAX_LIVE_EDGE = 0.30
    MAX_BUY_PRICE = 0.65
    COINFLIP_LOW = 0.40
    COINFLIP_HIGH = 0.60
    BET_SIZE = 1.0  # Fixed $1 per trade for normalized comparison

    # -- Load Model --
    config = ConfigManager()
    model = ModelEnsemble(config)
    model.load_latest()
    assert model._has_meta_v2, "Meta-Brain v2 not loaded!"
    print(f"Model loaded: v2={model._has_meta_v2}")

    # -- Load Dataset --
    csv_path = "dataset/market_data_2sec_weekly5_with_resolutions.csv"
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} rows")

    # -- Feature Engineering (same as training) --
    df['TTR_minutes'] = (300 - df['elapsed']) / 60.0
    df['gap_absolute'] = df['btc_gap']
    df['gap_percentage'] = (df['btc_gap'] / df['btc_strike']) * 100.0
    df['spread_YES'] = df['ask_YES'] - df['bid_YES']
    df['mid_YES'] = (df['ask_YES'] + df['bid_YES']) / 2.0
    df['target'] = (df['winner'].str.upper() == 'UP').astype(int)

    # -- Filter garbage --
    df = df[df['elapsed'] >= 0]
    df = df[df['elapsed'] <= 300]
    df = df[df['spread_YES'] >= 0]
    df = df[df['spread_YES'] <= 0.20]
    df = df[df['ask_YES'] > 0]
    df = df[df['bid_YES'] > 0]
    df = df.dropna(subset=['TTR_minutes', 'gap_absolute', 'gap_percentage',
                           'spread_YES', 'mid_YES', 'target'])

    # Sort chronologically
    df = df.sort_values('timestamp_log').reset_index(drop=True)
    print(f"Rows after filtering: {len(df)}")

    # -- Pre-compute v2 probabilities in batch (vectorized via model) --
    print("Pre-computing Meta-Brain v2 inference for all rows (batch)...")

    META_V2_FEATURES = ['TTR_minutes', 'gap_absolute', 'gap_percentage',
                        'spread_YES', 'mid_YES']
    v2_input = df[META_V2_FEATURES].values

    # Direct batch inference through the LGBM model (bypass the per-row predict())
    raw_probs = model._meta_v2_lgbm.predict_proba(
        pd.DataFrame(v2_input, columns=META_V2_FEATURES)
    )[:, 1]
    cal_probs = model._meta_v2_calibrator.transform(raw_probs)
    cal_probs = np.clip(cal_probs, 0.0, 1.0)
    df['v2_prob'] = cal_probs

    print(f"V2 Probs: min={cal_probs.min():.4f}, max={cal_probs.max():.4f}, "
          f"mean={cal_probs.mean():.4f}")

    # -- Simulation --
    print("\nRunning simulation with God Mode config...")
    traded_slugs = set()
    trades = []

    for _, row in df.iterrows():
        slug = row['slug']
        ttr = row['TTR_minutes']
        v2_p = row['v2_prob']

        # One trade per slug
        if slug in traded_slugs:
            continue

        # TTR window
        if ttr < MIN_TTR or ttr > MAX_TTR:
            continue

        # Coinflip rejection
        if COINFLIP_LOW <= v2_p <= COINFLIP_HIGH:
            continue

        # Determine direction and ask price
        if v2_p > 0.5:
            # Predict UP -> buy YES
            ask_price = row['ask_YES']
            edge = v2_p - ask_price
            actual_win = (row['target'] == 1)
        else:
            # Predict DOWN -> buy NO
            ask_price = row['ask_NO']
            edge = (1.0 - v2_p) - ask_price
            actual_win = (row['target'] == 0)

        # Edge gate
        if edge < MARGIN_OF_SAFETY:
            continue
        if edge > MAX_LIVE_EDGE:
            continue

        # Max buy price
        if ask_price > MAX_BUY_PRICE:
            continue

        # TRADE EXECUTED
        traded_slugs.add(slug)

        if actual_win:
            pnl = (1.0 / ask_price - 1.0) * BET_SIZE  # payout is $1/ask, cost is $1
        else:
            pnl = -BET_SIZE

        trades.append({
            'slug': slug,
            'ttr': ttr,
            'v2_prob': v2_p,
            'ask_price': ask_price,
            'edge': edge,
            'direction': 'UP' if v2_p > 0.5 else 'DOWN',
            'actual_winner': row['winner'],
            'outcome': 'WIN' if actual_win else 'LOSS',
            'pnl': pnl,
        })

    # -- Metrics --
    df_trades = pd.DataFrame(trades)

    if len(df_trades) == 0:
        print("\n[X] FATAL: Zero trades executed. Thresholds are too tight.")
        return

    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['outcome'] == 'WIN'])
    losses = len(df_trades[df_trades['outcome'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100

    total_pnl = df_trades['pnl'].sum()
    cumulative = df_trades['pnl'].cumsum()
    max_capital = cumulative.cummax()
    drawdowns = max_capital - cumulative
    max_dd = drawdowns.max()

    avg_edge = df_trades['edge'].mean()
    avg_ttr = df_trades['ttr'].mean()
    avg_ask = df_trades['ask_price'].mean()

    profit_factor = 0.0
    gross_win = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss

    # Direction breakdown
    up_trades = df_trades[df_trades['direction'] == 'UP']
    down_trades = df_trades[df_trades['direction'] == 'DOWN']
    up_wr = (len(up_trades[up_trades['outcome'] == 'WIN']) / len(up_trades) * 100) if len(up_trades) > 0 else 0
    down_wr = (len(down_trades[down_trades['outcome'] == 'WIN']) / len(down_trades) * 100) if len(down_trades) > 0 else 0

    print(f"\n{'=' * 65}")
    print(f" MACRO STRESS TEST RESULTS -- META-BRAIN V2.0 + GOD MODE")
    print(f"{'=' * 65}")
    print(f" Dataset Rows Scanned   : {len(df):,}")
    print(f" Unique Slugs (Markets) : {df['slug'].nunique():,}")
    print(f"{'=' * 65}")
    print(f" Total Markets Traded   : {total_trades}")
    print(f" Wins                   : {wins}")
    print(f" Losses                 : {losses}")
    print(f" Win Rate               : {win_rate:.2f}%")
    print(f"{'-' * 65}")
    print(f" Simulated PnL ($1/trade): ${total_pnl:.2f}")
    print(f" Profit Factor          : {profit_factor:.2f}")
    print(f" Max Drawdown           : ${max_dd:.2f}")
    print(f"{'-' * 65}")
    print(f" Avg Edge at Entry      : {avg_edge:.4f}")
    print(f" Avg TTR at Entry       : {avg_ttr:.2f} min")
    print(f" Avg Ask at Entry       : {avg_ask:.4f}")
    print(f"{'-' * 65}")
    print(f" Direction Breakdown:")
    print(f"   UP  trades: {len(up_trades)} (WR: {up_wr:.1f}%)")
    print(f"   DOWN trades: {len(down_trades)} (WR: {down_wr:.1f}%)")
    print(f"{'=' * 65}")

    if win_rate >= 60.0:
        print(f"\n [PASS] Config survives macro stress test. DEPLOY AUTHORIZED.")
    elif win_rate >= 55.0:
        print(f"\n [MARGINAL] Config is borderline. Consider tightening thresholds.")
    else:
        print(f"\n [FAIL] Config does not survive stress test. DO NOT DEPLOY.")


if __name__ == "__main__":
    run_stress_test()
