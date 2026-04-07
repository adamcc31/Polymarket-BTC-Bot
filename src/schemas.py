"""
schemas.py — All Pydantic v2 models serving as inter-module API contracts.

Every boundary between modules communicates via typed Pydantic models.
No raw dicts are passed across module boundaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Market Discovery
# ============================================================

class ActiveMarket(BaseModel):
    """Represents a currently active Polymarket market."""

    market_id: str = Field(..., description="Polymarket condition_id")
    question: str = Field(..., description="Market question text for verification")
    strike_price: float = Field(
        ...,
        description=(
            "Static strike price from market metadata (parsed from question "
            "text or Gamma API). NOT Binance price at T_open."
        ),
    )
    T_open: datetime = Field(..., description="Market open time (UTC)")
    T_resolution: datetime = Field(..., description="Resolution time (UTC)")
    TTR_minutes: float = Field(..., description="Time-to-resolution in minutes at discovery")
    clob_token_ids: Dict[str, str] = Field(
        ..., description='{"YES": "0x...", "NO": "0x..."}'
    )
    resolution_source: Optional[str] = Field(
        default=None,
        description=(
            "Oracle source for resolution (e.g. 'Pyth', 'Coinbase', 'CoinGecko'). "
            "Parsed from market rules. Used to assess basis risk."
        ),
    )

    @field_validator("strike_price")
    @classmethod
    def strike_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"strike_price must be positive, got {v}")
        return v


# ============================================================
# CLOB Feed
# ============================================================

class CLOBState(BaseModel):
    """Snapshot of Polymarket CLOB orderbook state."""

    market_id: str
    timestamp: datetime
    yes_ask: float = Field(..., ge=0.0, le=1.0, description="Best ask YES [0,1]")
    yes_bid: float = Field(..., ge=0.0, le=1.0, description="Best bid YES [0,1]")
    no_ask: float = Field(..., ge=0.0, le=1.0, description="Best ask NO [0,1]")
    no_bid: float = Field(..., ge=0.0, le=1.0, description="Best bid NO [0,1]")
    yes_depth_usd: float = Field(..., ge=0.0, description="Total USDC depth YES within 3% of ask")
    no_depth_usd: float = Field(..., ge=0.0, description="Total USDC depth NO within 3% of ask")
    market_vig: float = Field(..., description="yes_ask + no_ask - 1.0")
    is_liquid: bool = Field(..., description="Meets minimum depth & vig requirements")
    is_stale: bool = Field(default=False, description="Data older than STALE_TIMEOUT_S")


# ============================================================
# Feature Engine
# ============================================================

class FeatureVector(BaseModel):
    """24-feature vector output from feature engine."""

    values: List[float] = Field(..., min_length=24, max_length=24)
    feature_names: List[str] = Field(..., min_length=24, max_length=24)
    metadata: FeatureMetadata

    class Config:
        arbitrary_types_allowed = True


class FeatureMetadata(BaseModel):
    """Metadata attached to every feature vector."""

    timestamp: datetime
    bar_close_time: datetime
    market_id: str
    strike_price: float
    current_btc_price: float
    TTR_minutes: float
    TTR_phase: Literal["EARLY", "ENTRY_WINDOW", "LATE"]
    compute_lag_ms: float = Field(default=0.0, description="Feature computation latency")


# ============================================================
# Signal Generator
# ============================================================

class SignalResult(BaseModel):
    """Output of signal evaluation pipeline."""

    signal: Literal["BUY_YES", "BUY_NO", "ABSTAIN"]
    abstain_reason: Optional[
        Literal["TTR_PHASE", "REGIME_BLOCK", "LIQUIDITY_BLOCK", "NO_EDGE"]
    ] = None
    P_model: float = Field(..., ge=0.0, le=1.0)
    edge_yes: float
    edge_no: float
    clob_yes_ask: float
    clob_no_ask: float
    TTR_minutes: float
    strike_price: float
    current_price: float
    strike_distance: float
    market_id: str
    timestamp: datetime
    features: Dict[str, float] = Field(
        default_factory=dict, description="Full feature snapshot for logging"
    )


# ============================================================
# Risk Manager
# ============================================================

class ApprovedBet(BaseModel):
    """Trade approved by risk manager."""

    signal: SignalResult
    bet_size: float = Field(..., gt=0.0, description="USDC amount to bet")
    kelly_fraction: float = Field(..., ge=0.0)
    kelly_multiplier: float = Field(..., ge=0.0, le=1.0)


class RejectedBet(BaseModel):
    """Trade rejected by risk manager."""

    signal: SignalResult
    reason: Literal[
        "DAILY_LOSS_LIMIT_HIT",
        "SESSION_LOSS_LIMIT_HIT",
        "MAX_POSITIONS_REACHED",
        "CAPITAL_BELOW_FLOOR",
        "BET_BELOW_MINIMUM",
    ]


# ============================================================
# Trade Outcome
# ============================================================

class TradeOutcome(BaseModel):
    """Resolution outcome of a completed trade."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    outcome: Literal["WIN", "LOSS"]
    btc_at_resolution: float
    pnl_usd: float
    pnl_pct_capital: float = 0.0


class PaperTrade(BaseModel):
    """Paper trade record for dry run engine."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    market_id: str
    signal_type: Literal["BUY_YES", "BUY_NO"]
    entry_price: float
    bet_size: float
    strike_price: float
    T_resolution: datetime
    TTR_at_entry: float
    P_model: float
    edge_yes: float
    edge_no: float
    kelly_fraction: float
    kelly_multiplier: float
    capital_before: float

    # Filled post-resolution
    btc_at_resolution: Optional[float] = None
    outcome: Optional[Literal["WIN", "LOSS", "PENDING"]] = "PENDING"
    pnl_usd: Optional[float] = None
    pnl_pct_capital: Optional[float] = None
    capital_after: Optional[float] = None
    timestamp_signal: datetime = Field(default_factory=datetime.utcnow)
    timestamp_resolution: Optional[datetime] = None


# ============================================================
# Session Metrics & Performance
# ============================================================

class SessionMetrics(BaseModel):
    """Aggregated metrics for a trading session."""

    session_id: str
    date_utc: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    mode: Literal["DRY", "LIVE"]
    total_bars_processed: int = 0
    total_signals_evaluated: int = 0
    signals_abstained: int = 0
    abstain_regime: int = 0
    abstain_liquidity: int = 0
    abstain_ttr: int = 0
    abstain_no_edge: int = 0
    trades_executed: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_usd: float = 0.0
    total_pnl_pct_capital: float = 0.0
    max_drawdown: float = 0.0
    sharpe_rolling: float = 0.0
    sortino_rolling: float = 0.0
    brier_score: float = 0.0
    mean_edge_yes_traded: float = 0.0
    mean_edge_no_traded: float = 0.0
    mean_ttr_at_signal: float = 0.0
    mean_strike_distance: float = 0.0
    capital_start: float = 0.0
    capital_end: float = 0.0
    dry_run_score: float = 0.0
    pass_fail: Optional[Literal["PASS", "FAIL"]] = None
    model_version: str = ""
    ws_drop_rate_pct: float = 0.0
    clob_stale_events: int = 0
    consecutive_losses: int = 0


# ============================================================
# Observability
# ============================================================

class WSHealthMetrics(BaseModel):
    """WebSocket connection health metrics."""

    messages_received: int = 0
    messages_expected: int = 0
    messages_dropped: int = 0
    drop_rate: float = 0.0
    reconnect_count: int = 0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    queue_depth_max: int = 0
    last_message_timestamp: Optional[datetime] = None


class StalenessReport(BaseModel):
    """Data staleness status across all feeds."""

    binance_stale: bool = False
    clob_stale: bool = False
    market_stale: bool = False

    @property
    def any_stale(self) -> bool:
        return self.binance_stale or self.clob_stale or self.market_stale


# ============================================================
# Execution
# ============================================================

class FillResult(BaseModel):
    """Result of order fill monitoring."""

    status: Literal["FILLED", "FAILED", "TIMEOUT_CANCELLED"]
    fill_price: Optional[float] = None
    reason: Optional[str] = None
    order_id: Optional[str] = None


class OrderRejected(BaseModel):
    """Pre-order validation rejection."""

    reason: str
