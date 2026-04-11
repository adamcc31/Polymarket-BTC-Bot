import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta, timezone
from collections import Counter

# Add workspace to path
sys.path.append(os.getcwd())

from main import TradingBot
from src.schemas import SignalResult, ActiveMarket, FeatureVector, FeatureMetadata, CLOBState

# Configure logging to see the epoch_post_mortem output
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("verify_aggregator")

class DummyConfig:
    def __init__(self, overrides=None):
        self._overrides = overrides or {}
    def get(self, key, default=None):
        return self._overrides.get(key, default)
    def stop(self):
        pass

async def verify():
    print("Starting Aggregator Verification...")
    
    # Setup bot with dummy config
    cfg = DummyConfig({
        "telegram.heartbeat_minutes": 15.0,
        "clob.poll_interval_seconds": 5,
        "dry_run.max_duration_hours": 48
    })
    
    # Mocking components to avoid real network/IO calls
    bot = TradingBot(mode="dry-run")
    bot._config = cfg
    
    # Create a dummy market that resolves in 5 seconds
    now = datetime.now(timezone.utc)
    res_time = now + timedelta(seconds=5)
    market = ActiveMarket(
        market_id="test_market_agg",
        question="Test Market?",
        strike_price=50000.0,
        T_open=now - timedelta(minutes=5),
        T_resolution=res_time,
        TTR_minutes=5.0,
        clob_token_ids={"YES": "0x1", "NO": "0x2"},
        settlement_exchange="BINANCE",
        settlement_instrument="BTCUSDT",
        settlement_granularity="1m",
        settlement_price_type="close"
    )
    
    # Create signals
    # 1. Abstain: NO_EDGE
    signal1 = SignalResult(
        signal="ABSTAIN",
        abstain_reason="NO_EDGE",
        P_model=0.5,
        uncertainty_u=0.02,
        edge_yes=0.01,
        edge_no=0.01,
        clob_yes_bid=0.48, clob_yes_ask=0.49,
        clob_no_bid=0.51, clob_no_ask=0.52,
        TTR_minutes=5.0,
        strike_price=50000.0,
        current_price=50000.0,
        strike_distance=0.0,
        market_id="test_market_agg",
        timestamp=now
    )
    
    # 2. Abstain: REGIME_BLOCK, High Edge
    signal2 = SignalResult(
        signal="ABSTAIN",
        abstain_reason="REGIME_BLOCK",
        P_model=0.5,
        uncertainty_u=0.02,
        edge_yes=0.15, # High edge but blocked by regime
        edge_no=-0.15,
        clob_yes_bid=0.33, clob_yes_ask=0.35,
        clob_no_bid=0.65, clob_no_ask=0.67,
        TTR_minutes=4.0,
        strike_price=50000.0,
        current_price=50000.0,
        strike_distance=0.0,
        market_id="test_market_agg",
        timestamp=now + timedelta(seconds=1)
    )

    # Simulate _on_bar_close aggregation logic manually to avoid full pipeline
    # We call _schedule_post_mortem through the bot logic
    
    print("\n--- Simulating 1st Abstention (Triggering Watcher) ---")
    # Manually trigger the logic that would be in _on_bar_close
    m_id = market.market_id
    if m_id not in bot._post_mortem_tracker:
        bot._post_mortem_tracker[m_id] = {
            "evals": 0,
            "reasons": Counter(),
            "max_edge": 0.0
        }
        asyncio.create_task(
            bot._schedule_post_mortem(market),
            name=f"pm_test"
        )
    
    stats = bot._post_mortem_tracker[m_id]
    stats["evals"] += 1
    stats["reasons"][signal1.abstain_reason] += 1
    stats["max_edge"] = max(stats["max_edge"], max(signal1.edge_yes, signal1.edge_no))
    
    print(f"Tracker State: {bot._post_mortem_tracker[m_id]}")

    print("\n--- Simulating 2nd Abstention ---")
    stats["evals"] += 1
    stats["reasons"][signal2.abstain_reason] += 1
    stats["max_edge"] = max(stats["max_edge"], max(signal2.edge_yes, signal2.edge_no))
    
    print(f"Tracker State: {bot._post_mortem_tracker[m_id]}")
    
    print(f"\nWaiting for market resolution (T_res = {res_time.isoformat()})...")
    # T_resolution is in 5 seconds. _schedule_post_mortem waits res_time - now + 5s.
    # Total wait: ~10 seconds.
    
    await asyncio.sleep(12)
    
    print("\n--- Final Verification ---")
    if m_id not in bot._post_mortem_tracker:
        print("SUCCESS: Tracker cleaned up after resolution.")
    else:
        print("FAILURE: Tracker still contains market data.")
        print(bot._post_mortem_tracker)

if __name__ == "__main__":
    asyncio.run(verify())
