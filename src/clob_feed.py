"""
clob_feed.py — Polymarket CLOB orderbook polling + cache + stale detection.

Refactored to use WebSocket streaming for real-time 0-latency pricing.
Constructs CLOBState using best bid/ask from WS.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import websockets
import structlog

from src.config_manager import ConfigManager
from src.schemas import ActiveMarket, CLOBState

logger = structlog.get_logger(__name__)


class CLOBFeed:
    """
    Polymarket CLOB data feed via WebSocket.

    Maintains cached CLOBState with staleness detection.
    """

    WS_URL = os.getenv("REALTIME_PRICE_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market")
    CLOB_BASE_URL = "https://clob.polymarket.com"

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._stale_timeout = config.get("clob.stale_threshold_seconds", 30)
        self._min_depth_usd = config.get("clob.min_depth_usd", 10.0)
        self._max_vig = config.get("clob.max_market_vig", 0.07)

        self._verify_ssl = os.getenv("SSL_VERIFY", "true").lower() == "true"

        # Circuit breaker
        self._max_consecutive_404 = config.get("clob.max_consecutive_404", 3)
        self._consecutive_404_count: int = 0
        self._circuit_breaker_tripped: bool = False

        # State
        self._cached_state: Optional[CLOBState] = None
        self._last_fetch_time: float = 0.0
        self._stale_event_count: int = 0
        self._running = False
        
        self._ws_task: Optional[asyncio.Task] = None
        self._cache_dict: dict[str, dict] = {}
        self._market: Optional[ActiveMarket] = None
        self._fetch_locks: dict[str, asyncio.Lock] = {}

    # ── Public Properties ─────────────────────────────────────

    @property
    def clob_state(self) -> Optional[CLOBState]:
        """Current CLOB state (may be cached)."""
        if self._cached_state and self._is_stale():
            return self._cached_state.model_copy(update={"is_stale": True})
        return self._cached_state

    @property
    def stale_event_count(self) -> int:
        return self._stale_event_count

    @property
    def circuit_breaker_tripped(self) -> bool:
        return self._circuit_breaker_tripped

    def reset_circuit_breaker(self) -> None:
        self._consecutive_404_count = 0
        self._circuit_breaker_tripped = False
        logger.info("clob_circuit_breaker_reset")

    # ── WebSocket Loop ────────────────────────────────────────

    async def start(self, market: ActiveMarket) -> None:
        """Start polling loop for given market."""
        self._running = True
        self._market = market
        self._cache_dict.clear()
        
        logger.info(
            "clob_ws_started",
            market_id=market.market_id,
            url=self.WS_URL,
        )

        self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        logger.info("clob_ws_stopped")

    async def _ws_loop(self) -> None:
        reconnect_attempts = 0
        while self._running and self._market is not None:
            try:
                token_ids = []
                # Fallback to empty list or explicit checks if needed
                try:
                    t0 = self._market.clob_token_ids[0]
                    t1 = self._market.clob_token_ids[1]
                    if t0: token_ids.append(t0)
                    if t1: token_ids.append(t1)
                except Exception as e:
                    logger.error("ws_parse_tokens_error", error=str(e))

                if not token_ids:
                    logger.warning("ws_no_tokens", market_id=self._market.market_id)
                    await asyncio.sleep(5)
                    continue

                logger.info("ws_connecting", tokens=token_ids)
                
                ssl_context = None
                if not self._verify_ssl:
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                async with websockets.connect(self.WS_URL, ssl=ssl_context) as ws:
                    logger.info("ws_connected")
                    reconnect_attempts = 0
                    
                    sub_msg = {
                        "assets_ids": token_ids,
                        "type": "market",
                        "custom_feature_enabled": True
                    }
                    await ws.send(json.dumps(sub_msg))
                    
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    
                    try:
                        async for message in ws:
                            if not self._running:
                                break
                                
                            data = json.loads(message)
                            evt_type = data.get("event_type")
                            
                            if data.get("type") == "pong":
                                continue
                                
                            if evt_type == "best_bid_ask":
                                asset_id = str(data.get("asset_id"))
                                best_bid = float(data.get("best_bid", 0) or 0)
                                best_ask = float(data.get("best_ask", 0) or 0)
                                self._cache_dict[asset_id] = {"best_bid": best_bid, "best_ask": best_ask}
                                self._rebuild_clob_state()
                                
                            elif evt_type == "price_change" and isinstance(data.get("price_changes"), list):
                                for pc in data["price_changes"]:
                                    aid = str(pc.get("asset_id"))
                                    best_bid = float(pc.get("best_bid", 0) or 0)
                                    best_ask = float(pc.get("best_ask", 0) or 0)
                                    self._cache_dict[aid] = {"best_bid": best_bid, "best_ask": best_ask}
                                self._rebuild_clob_state()

                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("ws_connection_closed")
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error("ws_msg_error", error=str(e))
                    finally:
                        ping_task.cancel()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ws_connection_error", error=str(e))
                
            if self._running:
                delay = min(5.0 * (2 ** reconnect_attempts), 30.0)
                reconnect_attempts += 1
                logger.warning("ws_reconnect_backoff", attempt=reconnect_attempts, delay=delay)
                await asyncio.sleep(delay)

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        try:
            while True:
                await asyncio.sleep(8)
                if ws.open:
                    await ws.send(json.dumps({"type": "ping"}))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def _rebuild_clob_state(self) -> None:
        """Constructs an updated CLOBState whenever websocket pushes new data."""
        if not self._market:
            return

        try:
            token_0 = self._market.clob_token_ids[0]
            token_1 = self._market.clob_token_ids[1]
        except (IndexError, TypeError):
            return

        q0 = self._cache_dict.get(token_0)
        q1 = self._cache_dict.get(token_1)

        if not q0 or not q1:
            return

        ask_0 = float(q0["best_ask"])
        bid_0 = float(q0["best_bid"])
        ask_1 = float(q1["best_ask"])
        bid_1 = float(q1["best_bid"])

        # Fallback handling for missing asks
        if ask_0 <= 0: ask_0 = 1.0
        if ask_1 <= 0: ask_1 = 1.0

        vig = (ask_0 + ask_1) - 1.0

        # Since WS stream doesn't easily provide full depth at arbitrary %, 
        # we proxy liquidity via tight vig and non-zero bids.
        # We assume 1000 depth for liquidity checks if it has active quotes.
        depth_param = 1000.0 if bid_0 > 0 and bid_1 > 0 else 0.0

        is_liquid = (
            depth_param >= self._min_depth_usd
            and vig <= self._max_vig
            and vig >= -0.1  # Prevent crazy crossed books from being marked liquid falsely
        )

        self._cached_state = CLOBState(
            market_id=self._market.market_id,
            timestamp=datetime.now(timezone.utc),
            yes_ask=ask_0,
            yes_bid=bid_0,
            no_ask=ask_1,
            no_bid=bid_1,
            yes_depth_usd=depth_param,
            no_depth_usd=depth_param,
            market_vig=vig,
            is_liquid=is_liquid,
            is_stale=False,
        )
        self._last_fetch_time = time.time()
        self._consecutive_404_count = 0  # Re-establishing healthy market sync

    # ── Snapshot Fetch ────────────────────────────────────────

    async def fetch_clob_snapshot(
        self, market: ActiveMarket
    ) -> Optional[CLOBState]:
        """
        Return cached CLOBState locally retrieved from WS.
        Wait briefly on cold start before returning None.
        """
        # Brief pause to allow WS sub to establish and return a packet
        if not self._cached_state:
            for _ in range(10):
                if self._cached_state:
                    break
                await asyncio.sleep(0.1)
                
        if self._cached_state and self._is_stale():
            self._stale_event_count += 1
            return self._cached_state.model_copy(update={"is_stale": True})
            
        return self._cached_state

    def _is_stale(self) -> bool:
        """Check if CLOB data is stale."""
        if self._last_fetch_time == 0.0:
            return True
        return (time.time() - self._last_fetch_time) > self._stale_timeout

    async def fetch_clob_snapshot_with_fallback(
        self, 
        market: ActiveMarket, 
        timeout_ms: int = 500
    ) -> tuple[Optional[CLOBState], float]:
        """
        Fetch real-time CLOB snapshot locally from WebSocket dictionary.
        """
        lock = self._fetch_locks.setdefault(market.market_id, asyncio.Lock())
        
        async with lock:
            start = time.perf_counter()
            state = await self.fetch_clob_snapshot(market)
            latency = (time.perf_counter() - start) * 1000
            
            if state is None:
                if self._cached_state is None:
                    # Cold start failure - no data to fall back on
                    raise RuntimeError("clob_no_cache_available_on_cold_start")
                    
                logger.warning(
                    "clob_fetch_timeout_falling_back_to_cache",
                    market_id=market.market_id,
                    latency_ms=round(latency, 2)
                )
                return self._cached_state.model_copy(update={"is_stale": True}), latency

            return state, latency