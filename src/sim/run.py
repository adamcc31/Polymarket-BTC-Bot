"""
run.py — deterministic backtest/simulation harness.

CLI:
  python -m src.sim.run --market-id <id> --limit 500
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from src.config_manager import ConfigManager
from src.dry_run import DryRunEngine
from src.fair_probability import FairProbabilityEngine
from src.risk_manager import RiskManager
from src.schemas import (
    ActiveMarket,
    CLOBState,
    FeatureMetadata,
    FeatureVector,
)
from src.signal_generator import SignalGenerator


FEATURE_NAMES = [
    "OBI",
    "TFM_normalized",
    "VAM",
    "RV",
    "vol_percentile",
    "depth_ratio",
    "price_vs_ema20",
    "binance_spread_bps",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "TTR_normalized",
    "TTR_sin",
    "TTR_cos",
    "strike_distance_pct",
    "contest_urgency",
    "ttr_x_obi",
    "ttr_x_tfm",
    "ttr_x_strike",
    "clob_yes_mid",
    "clob_yes_spread",
    "clob_no_spread",
    "market_vig",
]


class FakeBinanceFeed:
    def __init__(self, latest_price: float) -> None:
        self.latest_price = latest_price


class ConfigOverride:
    """In-memory config overrides without persisting to config/config.json."""

    def __init__(self, base_cfg: ConfigManager, overrides: Dict[str, float]) -> None:
        self._base = base_cfg
        self._overrides = overrides

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base.get(key, default)


def _ms_to_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def _ttr_phase(ttr_minutes: float, cfg: ConfigManager) -> str:
    ttr_min = float(cfg.get("signal.ttr_min_minutes", 5.0))
    ttr_max = float(cfg.get("signal.ttr_max_minutes", 12.0))
    if ttr_minutes > ttr_max:
        return "EARLY"
    if ttr_minutes >= ttr_min:
        return "ENTRY_WINDOW"
    return "LATE"


def _reconstruct_clob_state(row: Dict[str, Any], market_id: str, cfg: ConfigManager, ts: datetime) -> CLOBState:
    yes_mid = float(row["clob_yes_mid"])
    yes_spread = float(row["clob_yes_spread"])
    no_spread = float(row["clob_no_spread"])
    market_vig = float(row["market_vig"])

    yes_bid = yes_mid - yes_spread / 2.0
    yes_ask = yes_mid + yes_spread / 2.0

    # Use vig definition: yes_ask + no_ask - 1.0 = market_vig
    no_ask = 1.0 - market_vig - yes_ask
    no_bid = no_ask - no_spread

    # Clamp into [0,1] for safety.
    yes_bid = max(0.0, min(1.0, yes_bid))
    yes_ask = max(0.0, min(1.0, yes_ask))
    no_bid = max(0.0, min(1.0, no_bid))
    no_ask = max(0.0, min(1.0, no_ask))

    max_vig = float(cfg.get("clob.max_market_vig", 0.07))
    is_liquid = market_vig <= max_vig

    # Depth is not present in the processed dataset; set large depth to avoid
    # accidentally blocking on depth in the signal generator.
    yes_depth_usd = 1e6
    no_depth_usd = 1e6

    return CLOBState(
        market_id=market_id,
        timestamp=ts,
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask,
        no_bid=no_bid,
        yes_depth_usd=yes_depth_usd,
        no_depth_usd=no_depth_usd,
        market_vig=market_vig,
        is_liquid=is_liquid,
        is_stale=False,
    )


def _build_feature_vector(row: Dict[str, Any], now_ts: datetime, cfg: ConfigManager) -> FeatureVector:
    values = []
    for name in FEATURE_NAMES:
        v = row.get(name, 0.0)
        # Polars may return NaNs; signal generator expects floats.
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = 0.0
        values.append(float(v))

    ttr_minutes = float(row["TTR_minutes"])
    md = FeatureMetadata(
        timestamp=now_ts,
        bar_close_time=now_ts,
        market_id=str(row["market_id"]),
        strike_price=float(row["strike_price"]),
        current_btc_price=float(row["btc_at_signal"]),
        TTR_minutes=ttr_minutes,
        TTR_phase=_ttr_phase(ttr_minutes, cfg),
        compute_lag_ms=0.0,
    )

    return FeatureVector(values=values, feature_names=FEATURE_NAMES, metadata=md)


def _build_active_market(row: Dict[str, Any], cfg: ConfigManager) -> ActiveMarket:
    t_resolution = _ms_to_dt(row["t_resolution_ms"])
    ttr_minutes = float(row["TTR_minutes"])
    # Approximate open time for feature normalization (not critical for signal generator).
    t_open = t_resolution - timedelta(minutes=max(15.0, ttr_minutes + 15.0))

    return ActiveMarket(
        market_id=str(row["market_id"]),
        question="SIM",
        strike_price=float(row["strike_price"]),
        T_open=t_open,
        T_resolution=t_resolution,
        TTR_minutes=ttr_minutes,
        clob_token_ids={"YES": "", "NO": ""},
        settlement_exchange="BINANCE",
        settlement_instrument="BTCUSDT",
        settlement_granularity="1m",
        settlement_price_type="close",
        resolution_source="Binance",
    )


async def run_backtest(
    market_id: Optional[str],
    limit: Optional[int],
    ttr_min: Optional[float],
    ttr_max: Optional[float],
    model_version: str = "fair_prob_sim_v1",
) -> None:
    cfg = ConfigManager.get_instance()
    overrides: Dict[str, float] = {}
    if ttr_min is not None:
        overrides["signal.ttr_min_minutes"] = float(ttr_min)
    if ttr_max is not None:
        overrides["signal.ttr_max_minutes"] = float(ttr_max)

    effective_cfg: Any = ConfigOverride(cfg, overrides) if overrides else cfg

    df = pl.read_parquet("data/processed/merged_training_features.parquet")
    if market_id:
        df = df.filter(pl.col("market_id") == str(market_id))
    if limit:
        df = df.head(int(limit))

    if df.height == 0:
        print("No rows after filtering.")
        return

    df = df.sort("signal_timestamp_ms")

    dry = DryRunEngine(effective_cfg, initial_capital=float(effective_cfg.get("sim.initial_capital", 100.0)))
    risk = RiskManager(effective_cfg)
    fair_engine = FairProbabilityEngine(effective_cfg)
    signal_gen = SignalGenerator(effective_cfg)

    pending: List[Tuple[Any, datetime, float]] = []
    # pending items: (PaperTrade, T_resolution_dt, btc_at_resolution)
    signal_counts: Dict[str, int] = {"BUY_YES": 0, "BUY_NO": 0, "ABSTAIN": 0}
    abstain_reason_counts: Dict[str, int] = {}

    for row in df.iter_rows(named=True):
        sig_ts = _ms_to_dt(row["signal_timestamp_ms"])
        active_market = _build_active_market(row, effective_cfg)

        # Resolve any trades that matured before this signal.
        pending.sort(key=lambda x: x[1])
        while pending and pending[0][1] <= sig_ts:
            trade, _, btc_at_resolution = pending.pop(0)
            resolved = await dry.resolve_trade(trade, btc_at_resolution)
            await risk.on_trade_resolved(resolved.pnl_usd or 0.0)

        dry.increment_bars()

        # Circuit: if signal happened while a position is open, risk manager will reject.
        # Still record the signal for calibration/prediction tracking.
        clob_state = _reconstruct_clob_state(row, active_market.market_id, effective_cfg, sig_ts)
        fv = _build_feature_vector(row, sig_ts, effective_cfg)

        sigma_override = float(row["RV"]) if row.get("RV") is not None else None
        if isinstance(sigma_override, float) and math.isnan(sigma_override):
            sigma_override = None
        if sigma_override is None:
            sigma_override = float(cfg.get("fair_prob.sigma_default_ann", 0.30))

        fake_feed = FakeBinanceFeed(latest_price=float(row["btc_at_signal"]))
        fair = fair_engine.compute(
            binance_feed=fake_feed,  # sigma is overridden from dataset RV
            active_market=active_market,
            clob_state=clob_state,
            sigma_ann_override=sigma_override,
            data_confidence_override=1.0,
            as_of_time=sig_ts,
        )

        signal = signal_gen.evaluate(
            P_model=fair.q_fair,
            uncertainty_u=fair.uncertainty_u,
            clob_state=clob_state,
            active_market=active_market,
            feature_vector=fv,
        )
        dry.record_signal(signal)
        signal_counts[signal.signal] = signal_counts.get(signal.signal, 0) + 1
        if signal.signal == "ABSTAIN":
            r = signal.abstain_reason or "NO_EDGE"
            abstain_reason_counts[r] = abstain_reason_counts.get(r, 0) + 1

        if signal.signal == "ABSTAIN":
            continue

        approved = await risk.approve(signal, dry.capital)
        # ApprovedBet / RejectedBet are Pydantic models; compare by presence of bet_size.
        if not hasattr(approved, "bet_size"):
            continue

        trade = dry.simulate_trade(signal, approved, active_market)
        btc_at_res = float(row["btc_at_resolution"])
        pending.append((trade, active_market.T_resolution, btc_at_res))

    # Resolve remaining pending trades at end.
    if pending:
        pending.sort(key=lambda x: x[1])
        for trade, _, btc_at_resolution in pending:
            resolved = await dry.resolve_trade(trade, btc_at_resolution)
            await risk.on_trade_resolved(resolved.pnl_usd or 0.0)

    metrics = dry.compute_session_metrics(model_version=model_version)
    print("=== Backtest Summary ===")
    print(f"session_id={metrics.session_id}")
    print(f"trades_executed={metrics.trades_executed}")
    print(f"signal_counts={signal_counts}")
    print(f"abstain_reason_counts={abstain_reason_counts}")
    print(f"win_rate={metrics.win_rate}")
    print(f"total_pnl_usd={metrics.total_pnl_usd}")
    print(f"max_drawdown={metrics.max_drawdown}")
    print(f"profit_factor={metrics.profit_factor}")
    print(f"brier_score={metrics.brier_score}")
    print(f"pass_fail={metrics.pass_fail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--ttr-min", type=float, default=None)
    parser.add_argument("--ttr-max", type=float, default=None)
    args = parser.parse_args()

    asyncio.run(
        run_backtest(
            market_id=args.market_id,
            limit=args.limit,
            ttr_min=args.ttr_min,
            ttr_max=args.ttr_max,
        )
    )


if __name__ == "__main__":
    main()

