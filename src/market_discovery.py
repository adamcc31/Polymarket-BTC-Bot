"""
market_discovery.py — Polymarket market state machine.

States: SEARCHING → ACTIVE → WAITING → SEARCHING
Polls Polymarket Gamma API to discover "Bitcoin Up or Down" 15-minute markets.

CRITICAL DESIGN NOTE (from validation):
- Strike price is STATIC and hardcoded by market creator, NOT Binance price at T_open.
- Must parse strike price from market question text or Gamma API metadata.
- Market intervals are NOT guaranteed to be exactly 15 minutes.
- Resolution oracle varies (Pyth, Coinbase, CoinGecko via UMA).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx
import structlog

from src.config_manager import ConfigManager
from src.schemas import ActiveMarket

logger = structlog.get_logger(__name__)


class DiscoveryState(str, Enum):
    SEARCHING = "SEARCHING"
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"


class MarketDiscovery:
    """
    Polymarket market discovery with state machine.

    Polls Gamma API for active "Bitcoin Up or Down" markets.
    Extracts strike price from market metadata (question text parsing).
    """

    # Base URLs
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"

    # Market identification patterns
    MARKET_PATTERNS = [
        r"bitcoin.*up.*or.*down",
        r"btc.*up.*or.*down",
        r"bitcoin.*above.*\$",
        r"btc.*above.*\$",
    ]

    # Strike price extraction patterns
    STRIKE_PATTERNS = [
        r"\$([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",  # $66,500.00 or $66500
        r"above\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
        r"below\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    ]

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._state = DiscoveryState.SEARCHING
        self._active_market: Optional[ActiveMarket] = None
        self._poll_interval = config.get("market_discovery.poll_interval_s", 30)
        self._waiting_poll = config.get("market_discovery.waiting_poll_s", 60)
        self._min_ttr = config.get("market_discovery.min_ttr_to_discover", 5.0)
        self._late_ttr = config.get("market_discovery.late_ttr_minutes", 3.0)
        self._running = False
        self._last_log_time: float = 0.0

    # ── Public Properties ─────────────────────────────────────

    @property
    def state(self) -> DiscoveryState:
        return self._state

    @property
    def active_market(self) -> Optional[ActiveMarket]:
        return self._active_market

    @property
    def is_market_active(self) -> bool:
        return self._state == DiscoveryState.ACTIVE and self._active_market is not None

    # ── State Machine ─────────────────────────────────────────

    async def start(self) -> None:
        """Start the discovery state machine loop."""
        self._running = True
        logger.info("market_discovery_started", state=self._state.value)

        while self._running:
            try:
                if self._state == DiscoveryState.SEARCHING:
                    await self._handle_searching()
                elif self._state == DiscoveryState.ACTIVE:
                    await self._handle_active()
                elif self._state == DiscoveryState.WAITING:
                    await self._handle_waiting()
            except Exception as e:
                logger.error("market_discovery_error", error=str(e), state=self._state.value)
                await asyncio.sleep(10.0)

    async def stop(self) -> None:
        self._running = False
        logger.info("market_discovery_stopped")

    async def _handle_searching(self) -> None:
        """Poll for active markets."""
        market = await self._find_active_market()
        if market:
            self._active_market = market
            self._state = DiscoveryState.ACTIVE
            logger.info(
                "market_discovered",
                market_id=market.market_id,
                strike_price=market.strike_price,
                TTR_minutes=market.TTR_minutes,
                resolution_source=market.resolution_source,
            )
        else:
            self._state = DiscoveryState.WAITING
            logger.info("no_active_market_found", transitioning_to="WAITING")
        await asyncio.sleep(self._poll_interval)

    async def _handle_active(self) -> None:
        """Monitor active market TTR and validity."""
        if not self._active_market:
            self._state = DiscoveryState.SEARCHING
            return

        now = datetime.now(timezone.utc)
        ttr_seconds = (self._active_market.T_resolution - now).total_seconds()
        ttr_minutes = ttr_seconds / 60.0

        if ttr_seconds <= 0:
            # Market has resolved
            logger.info(
                "market_resolved",
                market_id=self._active_market.market_id,
            )
            self._active_market = None
            self._state = DiscoveryState.SEARCHING
        elif ttr_minutes < self._late_ttr:
            # Mark as LATE — no new entries allowed
            logger.info(
                "market_late_phase",
                market_id=self._active_market.market_id,
                TTR_minutes=round(ttr_minutes, 2),
            )
            # Update TTR on market
            self._active_market = self._active_market.model_copy(
                update={"TTR_minutes": ttr_minutes}
            )
        else:
            # Update TTR
            self._active_market = self._active_market.model_copy(
                update={"TTR_minutes": ttr_minutes}
            )

        await asyncio.sleep(self._poll_interval)

    async def _handle_waiting(self) -> None:
        """Wait mode — poll less frequently."""
        import time as _time

        now = _time.time()
        if now - self._last_log_time > 300:  # Log every 5 minutes
            logger.info("market_discovery_waiting_mode")
            self._last_log_time = now

        market = await self._find_active_market()
        if market:
            self._active_market = market
            self._state = DiscoveryState.ACTIVE
            logger.info(
                "market_found_from_waiting",
                market_id=market.market_id,
                strike_price=market.strike_price,
                TTR_minutes=market.TTR_minutes,
            )
        await asyncio.sleep(self._waiting_poll)

    # ── Market Discovery Logic ────────────────────────────────

    async def _find_active_market(self) -> Optional[ActiveMarket]:
        """
        Query Gamma API for active Bitcoin Up/Down markets.
        Parse strike price from metadata.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Search for Bitcoin up/down markets
                resp = await client.get(
                    f"{self.GAMMA_API_BASE}/markets",
                    params={
                        "closed": "false",
                        "limit": 50,
                    },
                )
                resp.raise_for_status()
                markets = resp.json()

                if not isinstance(markets, list):
                    markets = markets.get("data", []) if isinstance(markets, dict) else []

                # Filter for Bitcoin Up/Down markets
                candidates = []
                for m in markets:
                    if self._is_btc_up_down_market(m):
                        parsed = self._parse_market(m)
                        if parsed:
                            candidates.append(parsed)

                if not candidates:
                    return None

                # Pick best candidate: highest TTR within valid range
                valid = [c for c in candidates if c.TTR_minutes >= self._min_ttr]
                if not valid:
                    logger.info(
                        "all_markets_late",
                        candidate_count=len(candidates),
                        ttrs=[round(c.TTR_minutes, 1) for c in candidates],
                    )
                    return None

                # Pick the one with most time remaining
                best = max(valid, key=lambda m: m.TTR_minutes)
                return best

        except httpx.HTTPError as e:
            logger.error("gamma_api_error", error=str(e))
            return None
        except Exception as e:
            logger.error("market_discovery_parse_error", error=str(e))
            return None

    def _is_btc_up_down_market(self, market_data: dict) -> bool:
        """Check if market matches Bitcoin Up/Down patterns."""
        question = market_data.get("question", "").lower()
        description = market_data.get("description", "").lower()
        text = f"{question} {description}"

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.MARKET_PATTERNS
        )

    def _parse_market(self, market_data: dict) -> Optional[ActiveMarket]:
        """
        Parse market JSON into ActiveMarket schema.
        CRITICAL: Strike price extracted from question text / metadata.
        """
        try:
            market_id = market_data.get("condition_id") or market_data.get("id", "")
            question = market_data.get("question", "")

            # Extract strike price from question text
            strike_price = self._extract_strike_price(question)
            if strike_price is None:
                # Try from description
                desc = market_data.get("description", "")
                strike_price = self._extract_strike_price(desc)
            if strike_price is None:
                logger.warning(
                    "strike_price_not_found",
                    market_id=market_id,
                    question=question,
                )
                return None

            # Parse timestamps
            end_date_str = market_data.get("end_date_iso") or market_data.get("endDate", "")
            created_str = market_data.get("created_at") or market_data.get("createdAt", "")

            if not end_date_str:
                return None

            T_resolution = self._parse_timestamp(end_date_str)
            T_open = self._parse_timestamp(created_str) if created_str else None

            if T_resolution is None:
                return None

            now = datetime.now(timezone.utc)
            ttr_minutes = (T_resolution - now).total_seconds() / 60.0

            if ttr_minutes <= 0:
                return None

            if T_open is None:
                # Estimate T_open as T_resolution - 15 minutes
                from datetime import timedelta
                T_open = T_resolution - timedelta(minutes=15)

            # Extract CLOB token IDs
            tokens = market_data.get("tokens", [])
            clob_token_ids = self._extract_token_ids(tokens, market_data)

            # Try to determine resolution source from rules
            resolution_source = self._extract_resolution_source(market_data)

            return ActiveMarket(
                market_id=market_id,
                question=question,
                strike_price=strike_price,
                T_open=T_open,
                T_resolution=T_resolution,
                TTR_minutes=ttr_minutes,
                clob_token_ids=clob_token_ids,
                resolution_source=resolution_source,
            )

        except Exception as e:
            logger.warning(
                "market_parse_failed",
                error=str(e),
                market_id=market_data.get("condition_id", "unknown"),
            )
            return None

    def _extract_strike_price(self, text: str) -> Optional[float]:
        """
        Extract strike price from market question/description text.

        DESIGN NOTE: Strike price is static and set by market creator.
        Examples:
          "Will Bitcoin be above $66,500.00 at 4:15 PM ET?" → 66500.00
          "Bitcoin Up or Down from $98,450?" → 98450.00
        """
        for pattern in self.STRIKE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(",", "")
                try:
                    price = float(price_str)
                    # Sanity check: BTC price should be reasonable
                    if 1_000 < price < 1_000_000:
                        return price
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_token_ids(tokens: list, market_data: dict) -> dict:
        """Extract YES/NO CLOB token IDs from market data."""
        token_ids = {"YES": "", "NO": ""}

        if isinstance(tokens, list):
            for token in tokens:
                outcome = token.get("outcome", "").upper()
                token_id = token.get("token_id", "")
                if outcome in ("YES", "NO") and token_id:
                    token_ids[outcome] = token_id

        # Fallback: try clobTokenIds field
        if not token_ids.get("YES"):
            clob_ids = market_data.get("clobTokenIds", [])
            if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                token_ids["YES"] = clob_ids[0]
                token_ids["NO"] = clob_ids[1]

        return token_ids

    @staticmethod
    def _extract_resolution_source(market_data: dict) -> Optional[str]:
        """
        Try to determine resolution oracle from market rules/description.
        Oracle can be Pyth, Coinbase, CoinGecko, etc.
        """
        rules = (
            market_data.get("description", "")
            + " "
            + market_data.get("resolution_source", "")
            + " "
            + str(market_data.get("uma_resolution_rules", ""))
        ).lower()

        if "pyth" in rules:
            return "Pyth"
        elif "coinbase" in rules:
            return "Coinbase"
        elif "coingecko" in rules:
            return "CoinGecko"
        elif "binance" in rules:
            return "Binance"
        elif "uma" in rules:
            return "UMA"
        return None

    @staticmethod
    def _parse_timestamp(ts: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime (UTC)."""
        if not ts:
            return None
        try:
            # Handle various ISO formats
            ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    # ── TTR Phase Logic ───────────────────────────────────────

    def get_ttr_phase(self) -> str:
        """
        Get current TTR phase for signal gating.
        Returns: EARLY, ENTRY_WINDOW, or LATE.
        """
        if not self._active_market:
            return "LATE"

        ttr = self._active_market.TTR_minutes
        ttr_min = self._config.get("signal.ttr_min_minutes", 5.0)
        ttr_max = self._config.get("signal.ttr_max_minutes", 12.0)

        if ttr > ttr_max:
            return "EARLY"
        elif ttr >= ttr_min:
            return "ENTRY_WINDOW"
        else:
            return "LATE"

    async def refresh_ttr(self) -> Optional[float]:
        """Refresh TTR on active market."""
        if not self._active_market:
            return None
        now = datetime.now(timezone.utc)
        ttr_seconds = (self._active_market.T_resolution - now).total_seconds()
        ttr_minutes = max(0.0, ttr_seconds / 60.0)
        self._active_market = self._active_market.model_copy(
            update={"TTR_minutes": ttr_minutes}
        )
        return ttr_minutes
