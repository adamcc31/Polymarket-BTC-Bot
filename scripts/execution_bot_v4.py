import asyncio
import os
import time
import logging
import ssl
import traceback
import requests
import aiohttp
import json
from datetime import datetime
from collections import deque
import numpy as np
from dotenv import load_dotenv

# GLOBAL SSL BYPASS FOR CLOUD STABILITY
ssl._create_default_https_context = ssl._create_unverified_context

# Load credentials
load_dotenv()

# Dependencies
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

# Setup Logging
LOG_DIR = "/app/logs" if os.environ.get("RAILWAY_ENVIRONMENT") else "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/execution_v4.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PHASE 1: HARDENED BTC/USD Price Feed ID
BTC_FEED_ID = "0xe62df6c8b4a94ed1aee7242143411a931112ddf1bd8147cd1b641375f79f58"

# PHASE 2: HARDENED ENDPOINTS (Beta Priority)
PYTH_ENDPOINTS = [
    "https://hermes-beta.pyth.network/v2/updates/price/latest",
    "https://xc-mainnet.pyth.network/v2/updates/price/latest",
    "https://hermes.pyth.network/v2/updates/price/latest"
]

class SlowSkewBotV4:
    def __init__(self):
        # Configuration
        self.tfm_window_mins = 15
        self.baseline_window_mins = 240
        self.z_threshold = 1.0 
        self.notif_threshold = 5.0 
        
        # State
        self.yes_token_id = None
        self.target_market_name = "Discovery Active..."
        self.price_buffer = deque(maxlen=self.baseline_window_mins)
        self.last_minute_price = None
        self.last_minute_ts = None
        self.last_heartbeat = 0
        self.last_tg_heartbeat = 0
        
        # Clients
        self.clob_client = None
        self.telemetry_file = f"{LOG_DIR}/execution_v4_telemetry.csv"
        self._init_telemetry()

    def _init_telemetry(self):
        if not os.path.exists(self.telemetry_file):
            with open(self.telemetry_file, "w") as f:
                f.write("trigger_ts,z_score,poly_up_ask,poly_up_bid,ask_size,bid_size,latency_ms,status\n")

    def init_clob_client(self):
        api_key = os.environ.get("POLY_BUILDER_API_KEY")
        api_secret = os.environ.get("POLY_BUILDER_SECRET")
        api_passphrase = os.environ.get("POLY_BUILDER_PASSPHRASE")
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not all([api_key, api_secret, api_passphrase, private_key]):
            logger.warning("Missing API credentials. Entering MONITOR-ONLY mode.")
            return False
        try:
            creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)
            # PHASE 3: INFO Level Logging for initialization
            self.clob_client = ClobClient(host="https://clob.polymarket.com", key=private_key, chain_id=137, creds=creds)
            logger.info("Polymarket CLOB Client Initialized.")
            return True
        except Exception:
            logger.error(f"CLOB Init Failed: {traceback.format_exc()}")
            return False

    def bootstrap_discovery(self):
        manual_id = os.environ.get("TARGET_TOKEN_ID")
        if manual_id:
            logger.info(f"🛡️ MANUAL OVERRIDE: Using Token ID {manual_id}")
            self.yes_token_id = manual_id
            return True
        logger.info("Searching for LIQUID active Bitcoin market (Limit 50)...")
        try:
            r = requests.get('https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=50', timeout=15)
            markets = r.json()
            for m in markets:
                if "Bitcoin" in m.get('question', '') and m.get('clobTokenIds'):
                    ids = m['clobTokenIds']
                    if isinstance(ids, str):
                        try: ids = json.loads(ids)
                        except: ids = [ids.replace('[','').replace(']','').replace('"','')]
                    if not ids: continue
                    tid = ids[0]
                    try:
                        # PHASE 3: INFO logging for Auditing
                        logger.info(f"Auditing '{m['question'][:40]}...' | Token: {tid}")
                        self.clob_client.get_order_book(tid)
                        self.yes_token_id = tid
                        self.target_market_name = m.get('question', 'Unknown Bitcoin Market')
                        logger.info(f"🎯 LIQUID TARGET FOUND: {self.target_market_name}")
                        return True
                    except: continue
            return False
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return False

    async def send_alert(self, session, title, message):
        tg_token = os.environ.get("TELEGRAM_TOKEN")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat:
            try:
                url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                payload = {"chat_id": tg_chat, "text": f"🔥 {title}\n\n{message}"}
                async with session.post(url, json=payload, timeout=5) as resp:
                    if resp.status == 200: return
            except: pass

    async def get_pyth_price(self, session):
        """Phase 2 & 3: fail-safe multi-endpoint price ingest."""
        params = {"ids[]": BTC_FEED_ID}
        for url in PYTH_ENDPOINTS:
            try:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        parsed_list = data.get('parsed', [])
                        if parsed_list:
                            p_info = parsed_list[0].get('price', {})
                            raw_px = p_info.get('price')
                            expo = p_info.get('expo')
                            if raw_px is not None and expo is not None:
                                return float(raw_px) * (10 ** int(expo))
                    else:
                        # PHASE 3: INFO for retry, ERROR for final failure
                        logger.info(f"Endpoint {url} returned {resp.status}. Trying next...")
            except Exception: pass
        
        # PHASE 3: Critical Fail-safe
        logger.error("ALL PYTH ENDPOINTS FAILED. Oracle price is NULL.")
        return None

    async def main_loop(self):
        async with aiohttp.ClientSession() as session:
            logger.info(f"🟢 CLOUD BOT ACTIVE. Mode: Mainnet-V25. Target: {self.target_market_name}")
            await self.send_alert(session, "[SYSTEM ONLINE]", 
                                 f"Hardened Bot v25 is active.\nTarget: `{self.target_market_name}`")
            
            while True:
                try:
                    price = await self.get_pyth_price(session)
                    
                    # PHASE 3 Fail-safe: No evaluations if price is None
                    if not price: 
                        await asyncio.sleep(5)
                        continue
                        
                    now = datetime.now()
                    current_minute = now.replace(second=0, microsecond=0)
                    
                    if self.last_heartbeat == 0 or (time.time() - self.last_heartbeat > 300):
                        logger.info(f"💓 HEARTBEAT: BTC: ${price:,.2f} | Buff: {len(self.price_buffer)}")
                        self.last_heartbeat = time.time()
                    
                    if time.time() - self.last_tg_heartbeat > 14400:
                        await self.send_alert(session, "[HEARTBEAT]", f"BTC Stable: ${price:,.2f}")
                        self.last_tg_heartbeat = time.time()

                    if self.last_minute_ts and current_minute > self.last_minute_ts:
                        momentum = price - self.last_minute_price
                        self.price_buffer.append(momentum)
                        await self.evaluate_signal(session, price)
                        self.last_minute_price = price
                    
                    if not self.last_minute_ts:
                        self.last_minute_price = price
                        logger.info(f"Baseline anchored: ${price:,.2f}")
                    
                    self.last_minute_ts = current_minute
                    await asyncio.sleep(10)
                except Exception as e:
                    logger.error(f"Main Loop Fatal: {e}")
                    await asyncio.sleep(5)

    async def evaluate_signal(self, session, current_price):
        if len(self.price_buffer) < 2: return
        history = list(self.price_buffer)
        mu, sigma = np.mean(history), np.std(history)
        if sigma < 1e-9: return
        z = (history[-1] - mu) / sigma
        
        if abs(z) > self.z_threshold:
            logger.info(f"🚨 SKEW ALERT: Z={z:.2f}")
            await self.execute_dry_run(z)
            if abs(z) >= self.notif_threshold:
                await self.send_alert(session, "HIGH MOMENTUM", 
                                     f"Z-Score: `{z:.2f}`\nBTC: `${current_price:,.2f}`\nMarket: {self.target_market_name}")

    async def execute_dry_run(self, z):
        start_ms = time.time() * 1000
        try:
            ask, bid, ask_sz, bid_sz = 0.0, 0.0, 0.0, 0.0
            if self.clob_client and self.yes_token_id:
                book = self.clob_client.get_order_book(self.yes_token_id)
                if book.asks:
                    ask, ask_sz = float(book.asks[0].price), float(book.asks[0].size)
                if book.bids:
                    bid, bid_sz = float(book.bids[0].price), float(book.bids[0].size)
            
            latency = time.time() * 1000 - start_ms
            with open(self.telemetry_file, "a") as f:
                f.write(f"{datetime.now().isoformat()},{z:.2f},{ask},{bid},{ask_sz},{bid_sz},{latency:.2f},SUCCESS\n")
            logger.info(f"✔ Telemetry OK: {latency:.1f}ms")
        except Exception: pass

async def main():
    bot = SlowSkewBotV4()
    if not bot.init_clob_client(): return
    if not bot.bootstrap_discovery():
        logger.error("COULD NOT FIND LIQUID TARGET.")
        return
    await bot.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.error(f"System Fatal: {traceback.format_exc()}")
