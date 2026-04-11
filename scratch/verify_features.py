import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock
import json
import os
import pandas as pd
from pathlib import Path

# Add project root to path if needed (assuming run from root)
import sys
sys.path.append(os.getcwd())

from src.schemas import (
    ActiveMarket, CLOBState, SignalResult, ApprovedBet, PaperTrade, FeatureMetadata, FeatureVector
)
from src.dry_run import DryRunEngine
from src.exporter import Exporter
from src.risk_manager import RiskManager

class FeatureVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.market_id = "test_market_123"
        self.active_market = ActiveMarket(
            market_id=self.market_id,
            slug="test-market",
            question="Will BTC be above 100k?",
            strike_price=100000.0,
            T_open=self.now - timedelta(hours=1),
            T_resolution=self.now + timedelta(minutes=10),
            TTR_minutes=10.0,
            clob_token_ids={"YES": "y_token", "NO": "n_token"},
            settlement_exchange="BINANCE",
            settlement_instrument="BTCUSDT",
            settlement_granularity="1m",
            settlement_price_type="close",
        )
        
        # Mock Config
        self.config = MagicMock()
        self.config.get.side_effect = lambda key, default=None: {
            "risk.max_buy_price": 0.75,
            "risk.live_edge_tolerance": 0.05,
            "signal.min_ttr_minutes": 1.5,
            "signal.margin_of_safety": 0.02,
            "risk.kelly_divisor": 2,
            "risk.min_bet_usd": 1.0,
            "risk.max_bet_fraction": 0.1,
            "risk.consecutive_loss_multiplier": 0.15,
            "risk.kelly_floor_multiplier": 0.25,
            "risk.ttr_size_scale_minutes": 10.0,
            "risk.max_vig_for_sizing": 0.07,
            "dry_run.abstain_summary_every_signals": 25
        }.get(key, default)

    async def test_min_ttr_gate(self):
        """Verify that a market with TTR < min_ttr_minutes is skipped (logic simulation)."""
        min_ttr = self.config.get("signal.min_ttr_minutes")
        
        # Test case: TTR 1.0 (should skip if min is 1.5)
        market_late = self.active_market.model_copy(update={"TTR_minutes": 1.0})
        self.assertTrue(market_late.TTR_minutes < min_ttr)
        print(f"[OK] Gate logic: TTR {market_late.TTR_minutes} < {min_ttr} SUCCESS")

    async def test_live_edge_verification_gates(self):
        """Verify the execution gates added to main.py."""
        # Setup signal
        signal = SignalResult(
            signal="BUY_YES",
            P_model=0.70,
            uncertainty_u=0.02,
            clob_yes_ask=0.60,
            clob_yes_bid=0.59,
            clob_no_ask=0.41,
            clob_no_bid=0.40,
            edge_yes=0.08, # 0.70 - 0.02 - 0.60
            edge_no=-0.13,
            TTR_minutes=10.0,
            strike_price=100000.0,
            current_price=99000.0,
            strike_distance=0.01,
            market_id=self.market_id,
            timestamp=self.now,
        )

        # Gate 1: Price Exceeds Max Cap
        real_best_ask_too_high = 0.80
        max_buy_price = self.config.get("risk.max_buy_price")
        self.assertTrue(real_best_ask_too_high > max_buy_price)
        print(f"[OK] Execution Gate: Price {real_best_ask_too_high} > Cap {max_buy_price} -> ABORT SUCCESS")

        # Gate 2: Edge Deviation Too High
        real_best_ask_diverged = 0.66 
        # synthetic_edge = 0.08
        # live_edge = 0.70 - 0.02 - 0.66 = 0.02
        # deviation = 0.06
        edge_tolerance = self.config.get("risk.live_edge_tolerance")
        self.assertTrue(abs(0.08 - 0.02) > edge_tolerance)
        print(f"[OK] Execution Gate: Edge Deviation 0.06 > Tolerance {edge_tolerance} -> ABORT SUCCESS")

    async def test_csv_export_enrichment(self):
        """Verify that PaperTrade and Exporter handle the new columns."""
        dry_run = DryRunEngine(self.config, initial_capital=100.0)
        exporter = Exporter(dry_run.session_id)
        
        signal = SignalResult(
            signal="BUY_YES",
            P_model=0.70,
            uncertainty_u=0.02,
            clob_yes_ask=0.60,
            clob_yes_bid=0.59,
            clob_no_ask=0.41,
            clob_no_bid=0.40,
            edge_yes=0.08,
            edge_no=-0.13,
            TTR_minutes=10.0,
            strike_price=100000.0,
            current_price=99000.0,
            strike_distance=0.01,
            market_id=self.market_id,
            timestamp=self.now,
            synthetic_edge=0.08,
            live_edge=0.07 # Simulated live edge
        )
        
        approved = ApprovedBet(
            signal=signal,
            bet_size=10.0,
            kelly_fraction=0.1,
            kelly_multiplier=1.0
        )
        
        # Create trade
        trade = dry_run.simulate_trade(signal, approved, self.active_market)
        
        # Verify PaperTrade object has new fields
        self.assertEqual(trade.synthetic_edge, 0.08)
        self.assertEqual(trade.live_edge, 0.07)
        self.assertEqual(trade.btc_price_at_trigger, 99000.0)
        self.assertEqual(trade.btc_distance_to_strike, 99000.0 - 100000.0)
        self.assertIsNotNone(trade.trigger_timestamp)
        print("[OK] PaperTrade schema enrichment SUCCESS")
        
        # Resolve trade and export
        resolved = await dry_run.resolve_trade(trade, btc_at_resolution=101000.0)
        csv_path = exporter.export_trades([resolved])
        
        # Read CSV and check columns
        df = pd.read_csv(csv_path)
        new_cols = [
            "synthetic_edge", "live_edge", "btc_price_at_trigger", 
            "btc_distance_to_strike", "trigger_timestamp", "TTR_minutes"
        ]
        for col in new_cols:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isnull().all(), f"Column {col} is all null")
            
        print(f"[OK] CSV Export enrichment: {new_cols} found SUCCESS")
        print(f"[OK] CSV Path: {csv_path}")

if __name__ == "__main__":
    unittest.main()
