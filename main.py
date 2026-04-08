"""
main.py — Entry point for Polymarket Mispricing Detection Bot.

Usage:
  python main.py --mode dry-run              # Paper trading (default)
  python main.py --mode live --confirm-live   # Live trading (triple-gated)
  python main.py --config show               # Show current config
  python main.py --config set KEY VALUE      # Hot-update config
  python main.py --rollback-model            # Rollback to previous model
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import structlog
from dotenv import load_dotenv
from rich.live import Live

# Load .env before anything else
load_dotenv()

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if os.getenv("ENVIRONMENT", "development") == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(
            __import__("logging"),
            os.getenv("LOG_LEVEL", "INFO").upper(),
            20,
        )
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


from src.binance_feed import BinanceFeed
from src.cli import build_dashboard
from src.clob_feed import CLOBFeed
from src.config_manager import ConfigManager
from src.database import DatabaseManager
from src.dry_run import DryRunEngine
from src.execution import ExecutionClient
from src.exporter import Exporter
from src.feature_engine import FeatureEngine
from src.market_discovery import MarketDiscovery
from src.model import ModelEnsemble
from src.risk_manager import RiskManager
from src.signal_generator import SignalGenerator


class TradingBot:
    """
    Main orchestrator — wires all modules together.

    Lifecycle:
      1. Initialize all components
      2. Bootstrap historical data
      3. Start WebSocket feeds + market discovery
      4. On each bar close: features → model → signal → risk → trade
      5. On market resolution: settle trades, discover next market
    """

    def __init__(self, mode: str = "dry-run", confirm_live: bool = False) -> None:
        self._mode = mode
        self._confirm_live = confirm_live
        self._running = False

        # Initialize components
        self._config = ConfigManager.get_instance()
        self._db = DatabaseManager()
        self._binance = BinanceFeed(self._config)
        self._discovery = MarketDiscovery(self._config)
        self._clob = CLOBFeed(self._config)
        self._feature_engine = FeatureEngine(self._config)
        self._model = ModelEnsemble(self._config)
        self._signal_gen = SignalGenerator(self._config)
        self._risk_mgr = RiskManager(self._config)
        self._execution = ExecutionClient(self._config)
        self._exporter: Exporter | None = None

        # Dry run / live engine
        initial_capital = 100.0 if mode == "dry-run" else 50.0
        self._dry_run = DryRunEngine(self._config, initial_capital=initial_capital)
        self._exporter = Exporter(self._dry_run.session_id)

        # Dashboard state
        self._latest_signal = None
        self._latest_metrics = None

    async def start(self) -> None:
        """Start all subsystems and enter main loop."""
        logger.info(
            "bot_starting",
            mode=self._mode,
            session_id=self._dry_run.session_id,
        )

        self._running = True

        # Initialize database
        await self._db.init_db()

        # Load model
        if not self._model.load_latest():
            logger.warning(
                "no_model_loaded_running_in_data_collection_mode",
                info="Bot will run but all signals will ABSTAIN (P_model=0.5)",
            )

        # Bootstrap historical data
        bars_loaded = await self._binance.bootstrap_historical(limit=500)
        logger.info("bootstrap_complete", bars=bars_loaded)

        # Live mode gate
        if self._mode == "live":
            if not self._execution.confirm_live(cli_flag=self._confirm_live):
                logger.error("live_mode_not_confirmed_falling_back_to_dry_run")
                self._mode = "dry-run"

        # Register bar close callback
        self._binance.set_on_bar_close(self._on_bar_close)

        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._binance.start(), name="binance_feed"),
            asyncio.create_task(self._discovery.start(), name="market_discovery"),
            asyncio.create_task(self._run_clob_loop(), name="clob_feed"),
            asyncio.create_task(self._run_dashboard(), name="dashboard"),
        ]

        # Wait for shutdown
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("bot_shutting_down")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        await self._binance.stop()
        await self._discovery.stop()
        await self._clob.stop()

        # Export session data
        metrics = self._dry_run.compute_session_metrics(self._model.version)
        self._exporter.export_session(
            trades=self._dry_run._resolved_trades,
            metrics=metrics,
            initial_capital=self._dry_run._initial_capital,
        )

        await self._db.close()
        self._config.stop()
        logger.info("bot_stopped", session=self._dry_run.session_id)

    # ── Core Trading Loop ─────────────────────────────────────

    async def _on_bar_close(self, bar: dict) -> None:
        """
        Called on each 15-minute bar close.
        Full pipeline: features → model → signal → risk → trade.
        """
        self._dry_run.increment_bars()

        # Check if we have an active market
        if not self._discovery.is_market_active:
            return

        market = self._discovery.active_market
        await self._discovery.refresh_ttr()

        # ── Bar-close rotation check ──────────────────────────
        # Aligned here (not on an independent timer) so market switches never
        # interrupt a Z-score computation mid-window.
        rotated = await self._discovery.check_and_rotate()
        if rotated:
            # Discard stale CLOB cache — next poll will fetch fresh data
            self._clob._cached_state = None
            market = self._discovery.active_market
            logger.info(
                "bar_close_rotation_applied",
                new_market_id=market.market_id if market else None,
            )
            return  # Skip this bar's signal; let next bar compute on new market

        # Check data staleness
        if self._binance.is_stale:
            logger.warning("binance_data_stale_skipping_signal")
            return

        clob_state = self._clob.clob_state
        if not clob_state:
            logger.warning("no_clob_data_skipping_signal")
            return

        # Record CLOB snapshot
        self._exporter.record_clob_snapshot(clob_state, market.TTR_minutes)

        # ── Feature Computation ───────────────────────────────
        fv = self._feature_engine.compute(self._binance, market, clob_state)
        if fv is None:
            return

        # ── Model Prediction ──────────────────────────────────
        import numpy as np
        P_model = self._model.predict(np.array(fv.values))

        # ── Signal Generation ─────────────────────────────────
        signal = self._signal_gen.evaluate(P_model, clob_state, market, fv)
        self._latest_signal = signal
        self._dry_run.record_signal(signal)

        if signal.signal == "ABSTAIN":
            return

        # ── Risk Management ───────────────────────────────────
        result = await self._risk_mgr.approve(signal, self._dry_run.capital)

        from src.schemas import ApprovedBet, RejectedBet

        if isinstance(result, RejectedBet):
            logger.info("trade_rejected", reason=result.reason)
            return

        approved = result
        assert isinstance(approved, ApprovedBet)

        # ── Execute Trade (Dry Run) ───────────────────────────
        if self._mode == "dry-run":
            trade = self._dry_run.simulate_trade(signal, approved, market)

            # Schedule resolution
            asyncio.create_task(
                self._schedule_resolution(trade, market),
                name=f"resolve_{trade.trade_id[:8]}",
            )
        else:
            # Live mode execution
            fill_result = await self._execution.place_order(approved, market)
            logger.info("live_order_result", status=fill_result.status if hasattr(fill_result, 'status') else 'rejected')

        # Check abort conditions
        abort = self._dry_run.check_abort_conditions()
        if abort:
            logger.critical("session_abort", reason=abort)
            self._running = False

        # Update metrics
        self._latest_metrics = self._dry_run.compute_session_metrics(
            self._model.version
        )

    async def _schedule_resolution(self, trade, market) -> None:
        """Wait for market resolution and settle trade."""
        now = datetime.now(timezone.utc)
        wait_seconds = (market.T_resolution - now).total_seconds()

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds + 5)  # +5s buffer for price settlement

        # Get BTC price at resolution
        price = self._binance.latest_price
        if price is None:
            # Fallback: fetch from REST
            klines = await self._binance.fetch_rest_klines(limit=1)
            if klines:
                price = klines[-1]["close"]
            else:
                logger.error("cannot_resolve_trade_no_price")
                return

        resolved = await self._dry_run.resolve_trade(trade, price)
        await self._risk_mgr.on_trade_resolved(resolved.pnl_usd or 0)

    # ── CLOB Polling Loop ─────────────────────────────────────

    async def _run_clob_loop(self) -> None:
        """
        Poll CLOB data when market is active.

        Circuit breaker: if CLOBFeed accumulates max_consecutive_404 errors,
        the market has almost certainly expired. We call force_rediscover() to
        immediately restart the discovery state machine, then reset the breaker
        so it is ready for the next market cycle.
        """
        while self._running:
            if self._discovery.is_market_active:
                market = self._discovery.active_market
                try:
                    state = await self._clob.fetch_clob_snapshot(market)
                    if state:
                        self._clob._cached_state = state
                        self._clob._last_fetch_time = __import__("time").time()
                except Exception as e:
                    logger.error("clob_loop_error", error=str(e))

                # ── Circuit breaker check ─────────────────────
                if self._clob.circuit_breaker_tripped:
                    logger.warning(
                        "clob_circuit_breaker_triggering_rediscover",
                        market_id=market.market_id if market else None,
                    )
                    self._discovery.force_rediscover()
                    self._clob.reset_circuit_breaker()

            poll_interval = self._config.get("clob.poll_interval_seconds", 5)
            await asyncio.sleep(poll_interval)

    # ── Dashboard ─────────────────────────────────────────────

    async def _run_dashboard(self) -> None:
        """Update Rich dashboard every 5 seconds."""
        try:
            with Live(
                build_dashboard(),
                refresh_per_second=0.2,
                console=__import__("rich.console", fromlist=["Console"]).Console(),
            ) as live:
                while self._running:
                    dashboard = build_dashboard(
                        market=self._discovery.active_market,
                        clob=self._clob.clob_state,
                        signal=self._latest_signal,
                        metrics=self._latest_metrics,
                        ws_health=self._binance.health,
                        current_price=self._binance.latest_price,
                        mode="DRY RUN" if self._mode == "dry-run" else "LIVE",
                        session_id=self._dry_run.session_id,
                    )
                    live.update(dashboard)
                    await asyncio.sleep(5)
        except Exception as e:
            # Dashboard failure should not crash the bot
            logger.warning("dashboard_error", error=str(e))


# ============================================================
# CLI Entry Point
# ============================================================


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["dry-run", "live"]),
    default="dry-run",
    help="Trading mode",
)
@click.option(
    "--confirm-live",
    is_flag=True,
    default=False,
    help="Confirm live trading (required with --mode live)",
)
@click.option(
    "--config",
    "config_cmd",
    type=click.Choice(["show", "set", "get"]),
    default=None,
    help="Config management command",
)
@click.option("--key", default=None, help="Config key (for set/get)")
@click.option("--value", default=None, help="Config value (for set)")
@click.option(
    "--rollback-model",
    is_flag=True,
    default=False,
    help="Rollback to previous model version",
)
def main(
    mode: str,
    confirm_live: bool,
    config_cmd: str | None,
    key: str | None,
    value: str | None,
    rollback_model: bool,
) -> None:
    """Polymarket Bitcoin Up/Down — Probability Mispricing Detection Bot."""

    # Config commands (non-trading)
    if config_cmd:
        cfg = ConfigManager.get_instance()
        if config_cmd == "show":
            import json
            click.echo(json.dumps(cfg.all(), indent=2))
        elif config_cmd == "get" and key:
            click.echo(f"{key} = {cfg.get(key)}")
        elif config_cmd == "set" and key and value:
            # Auto-convert types
            try:
                typed_value = float(value)
            except ValueError:
                typed_value = value
            cfg.set(key, typed_value)
            click.echo(f"Set {key} = {typed_value}")
        cfg.stop()
        return

    # Model rollback
    if rollback_model:
        cfg = ConfigManager.get_instance()
        model = ModelEnsemble(cfg)
        if model.rollback():
            click.echo("✓ Model rolled back successfully")
        else:
            click.echo("✗ Rollback failed — no previous version available")
        cfg.stop()
        return

    # Trading mode
    click.echo(f"\n🚀 Starting Polymarket Bot — Mode: {mode.upper()}\n")

    bot = TradingBot(mode=mode, confirm_live=confirm_live)

    # Graceful shutdown handler
    def handle_shutdown(sig, frame):
        click.echo("\n\n⏹  Shutting down gracefully...")
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.create_task(bot.stop())
        )

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        click.echo("\nBot stopped.")


if __name__ == "__main__":
    main()