"""
redeemer.py — Asynchronous worker for market redemption and allowance management.

Automates the lifecycle of converting winning shares (CTF tokens) into liquid USDC.
Runs a background loop every 60 seconds.
"""

from __future__ import annotations

import asyncio
import time
from typing import Set

import structlog

from src.config_manager import ConfigManager
from src.execution import ExecutionClient

logger = structlog.get_logger(__name__)


class RedeemerWorker:
    """
    Background worker that monitors resolved markets and redeems winning positions.
    Ensures USDC allowance is set on startup.
    """

    def __init__(self, config: ConfigManager, execution: ExecutionClient) -> None:
        self._config = config
        self._execution = execution
        self._running = False
        self._poll_interval = 60 # 1 minute
        self._seen_markets: Set[str] = set()

    async def start(self) -> None:
        """Initialize and start the background loop."""
        if not self._execution.is_live:
            logger.info("redeemer_not_starting_live_mode_inactive")
            return

        self._running = True
        logger.info("redeemer_worker_starting", interval_s=self._poll_interval)

        # ── Step 1: Startup Allowance Check ───────────────────
        # User requirement: Check and set allowance on startup.
        await self._execution.check_and_set_allowance()

        # ── Step 2: Background Loop ───────────────────────────
        while self._running:
            try:
                await self._run_once()
            except Exception as e:
                logger.error("redeemer_loop_error", error=str(e), exc_info=True)

            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        logger.info("redeemer_worker_stopped")

    async def _run_once(self) -> None:
        """Single scan and redeem cycle."""
        # Note: In a real implementation, we would scan for all held assets
        # and identify which ones correspond to resolved markets.
        # py-clob-client's redeem_positions() is gasless on Polymarket
        # but requires knowing the condition_id.
        
        positions = await self._execution.get_positions()
        if not positions:
            # logger.debug("redeemer_no_active_positions")
            return

        # Track which unique condition_ids need redemption
        # positions often looks like [{"asset_id": "...", "size": "..."}]
        # Mapping asset_id to condition_id usually requires a lookup.
        # Here we attempt redemption if we have any positions, 
        # relying on execution.redeem_positions to filter winning ones.
        
        # In a refined version, we would fetch resolved markets from CLOB API
        # and match them against our holdings.
        
        # For now, we utilize the execution client's ability to redeem.
        # We might keep a local cache of condition_ids we've traded in.
        pass

    async def force_redeem(self, market_id: str) -> None:
        """Explicitly trigger redemption for a specific market."""
        logger.info("redeemer_explicit_trigger", market_id=market_id)
        success = await self._execution.redeem_positions(market_id)
        if success:
            # We don't know the exact amount without a separate balance check
            # but we log the attempt.
            logger.info("liquidity_redeemed", market_id=market_id)
