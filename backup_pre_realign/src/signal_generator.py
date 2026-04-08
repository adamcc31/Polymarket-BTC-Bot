"""
signal_generator.py — 6-step signal generation with hard gates.

Decision Flow:
  STEP 1: TTR Gate (ENTRY_WINDOW only)
  STEP 2: Regime Filter (vol percentile + spread + depth)
  STEP 3: Liquidity Filter (CLOB depth + vig + staleness)
  STEP 4: Mispricing Calculation (edge_yes, edge_no)
  STEP 5: Margin of Safety Check (edge > threshold)
  STEP 6: Final Signal Selection (BUY_YES / BUY_NO / ABSTAIN)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import structlog

from src.config_manager import ConfigManager
from src.schemas import (
    ActiveMarket,
    CLOBState,
    FeatureMetadata,
    FeatureVector,
    SignalResult,
)

logger = structlog.get_logger(__name__)


class SignalGenerator:
    """
    Evaluates model output against hard gates to produce trading signals.

    Expected abstain rate: ~60-75%. High selectivity is the edge, not a weakness.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    def evaluate(
        self,
        P_model: float,
        clob_state: CLOBState,
        active_market: ActiveMarket,
        feature_vector: FeatureVector,
    ) -> SignalResult:
        """
        Run full 6-step signal evaluation pipeline.

        Returns SignalResult with signal direction + full diagnostics.
        """
        now = datetime.now(timezone.utc)
        metadata = feature_vector.metadata

        # Build feature dict for logging
        features_dict = dict(zip(feature_vector.feature_names, feature_vector.values))

        # Common fields
        base = {
            "P_model": P_model,
            "clob_yes_ask": clob_state.yes_ask,
            "clob_no_ask": clob_state.no_ask,
            "TTR_minutes": metadata.TTR_minutes,
            "strike_price": active_market.strike_price,
            "current_price": metadata.current_btc_price,
            "strike_distance": features_dict.get("strike_distance_pct", 0.0),
            "market_id": active_market.market_id,
            "timestamp": now,
            "features": features_dict,
        }

        # ── STEP 1: TTR GATE ─────────────────────────────────
        ttr_min = self._config.get("signal.ttr_min_minutes", 5.0)
        ttr_max = self._config.get("signal.ttr_max_minutes", 12.0)
        ttr = metadata.TTR_minutes

        if ttr < ttr_min or ttr > ttr_max:
            edge_yes = P_model - clob_state.yes_ask
            edge_no = (1.0 - P_model) - clob_state.no_ask
            logger.info(
                "signal_abstain_ttr",
                TTR_minutes=round(ttr, 2),
                phase=metadata.TTR_phase,
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="TTR_PHASE",
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 2: REGIME FILTER ────────────────────────────
        vol_lower = self._config.get("regime.vol_lower_threshold", 0.15)
        vol_upper = self._config.get("regime.vol_upper_threshold", 0.80)
        spread_max = self._config.get("regime.binance_spread_max_bps", 5.0)
        depth_min = self._config.get("regime.binance_min_depth_btc", 0.5)

        vol_pct = features_dict.get("vol_percentile", 0.5)
        spread_bps = features_dict.get("binance_spread_bps", 2.0)
        # Depth check uses feature_engine's depth_ratio as proxy
        depth_ratio = features_dict.get("depth_ratio", 1.0)

        regime_vol_ok = vol_lower < vol_pct < vol_upper
        spread_ok = spread_bps < spread_max
        # Depth check: simplified — top5 bid depth as ratio proxy
        depth_ok = True  # Will be properly checked via binance_feed in orchestrator

        regime_go = regime_vol_ok and spread_ok and depth_ok

        if not regime_go:
            edge_yes = P_model - clob_state.yes_ask
            edge_no = (1.0 - P_model) - clob_state.no_ask
            logger.info(
                "signal_abstain_regime",
                vol_pct=round(vol_pct, 3),
                spread_bps=round(spread_bps, 2),
                regime_vol_ok=regime_vol_ok,
                spread_ok=spread_ok,
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="REGIME_BLOCK",
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 3: LIQUIDITY FILTER ─────────────────────────
        if not clob_state.is_liquid or clob_state.is_stale:
            edge_yes = P_model - clob_state.yes_ask
            edge_no = (1.0 - P_model) - clob_state.no_ask
            logger.info(
                "signal_abstain_liquidity",
                is_liquid=clob_state.is_liquid,
                is_stale=clob_state.is_stale,
                yes_depth=round(clob_state.yes_depth_usd, 2),
                no_depth=round(clob_state.no_depth_usd, 2),
                vig=round(clob_state.market_vig, 4),
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="LIQUIDITY_BLOCK",
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 4: MISPRICING CALCULATION ───────────────────
        edge_yes = P_model - clob_state.yes_ask
        edge_no = (1.0 - P_model) - clob_state.no_ask

        # ── STEP 5: MARGIN OF SAFETY CHECK ───────────────────
        margin = self._config.get("signal.margin_of_safety", 0.05)

        if max(edge_yes, edge_no) <= margin:
            logger.info(
                "signal_abstain_no_edge",
                edge_yes=round(edge_yes, 4),
                edge_no=round(edge_no, 4),
                margin_of_safety=margin,
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="NO_EDGE",
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 6: FINAL SIGNAL SELECTION ───────────────────
        if edge_yes > margin and edge_no > margin:
            # Both edges positive — pick larger
            signal = "BUY_YES" if edge_yes >= edge_no else "BUY_NO"
        elif edge_yes > margin:
            signal = "BUY_YES"
        else:
            signal = "BUY_NO"

        logger.info(
            "signal_generated",
            signal=signal,
            P_model=round(P_model, 4),
            edge_yes=round(edge_yes, 4),
            edge_no=round(edge_no, 4),
            TTR_minutes=round(ttr, 2),
            market_id=active_market.market_id,
        )

        return SignalResult(
            signal=signal,
            abstain_reason=None,
            edge_yes=edge_yes,
            edge_no=edge_no,
            **base,
        )
