import asyncio
import os
import time
import logging
import ssl
import traceback
import requests
import aiohttp
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
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/execution_v4.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PYTH CONFIG: BTC/USD Price Feed ID
PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest?ids[]=ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace"

class SlowSkewBotV4:
    def __init__(self):
        # Configuration
        self.tfm_window_mins = 15
        self.baseline_window_mins = 240
        self.z_threshold = 1.0 # Detection threshold
        
        # State
        self.yes_token_id = None
        self.target_market_slug = None
        self.price_buffer = deque(maxlen=self.baseline_window_mins)
        self.last_minute_price = None
        self.last_minute_ts = None
        
        # Clients
        self.clob_client = None
        self.telemetry_file = "logs/execution_v4_telemetry.csv"
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
            logger.warning("Missing 'POLY_BUILDER_*' credentials. MONITOR ONLY.")
            return False
            
        try:
            creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)
            logger.info("Initializing Polymarket CLOB Client...")
            self.clob_client = ClobClient(host="https://clob.polymarket.com", key=private_key, chain_id=137, creds=creds)
            logger.info("Polymarket CLOB Client Initialized.")
            return True
        except Exception:
            logger.error(f"CLOB Init Failed: {traceback.format_exc()}")
            return False

    def bootstrap_discovery(self):
        """Hardened Discovery: Verify CLOB Orderbook before selecting target."""
        logger.info("Searching for LIQUID active Bitcoin market...")
        try:
            r = requests.get('https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=20', timeout=10)
            markets = r.json()
            for m in markets:
                # 1. Topic Filter
                if "Bitcoin" in m.get('question', '') and m.get('clobTokenIds'):
                    tid = m['clobTokenIds'][0]
                    # 2. LIQUIDITY AUDIT: Verify CLOB existence
                    try:
                        logger.info(f"Auditing '{m['question'][:40]}...' | Token: {tid[:10]}")
                        self.clob_client.get_order_book(tid)
                        self.yes_token_id = tid
                        self.target_market_slug = m.get('market_slug')
                        logger.info(f"🎯 LIQUID TARGET FOUND: {m['question']}")
                        return True
                    except:
                        logger.warning(f"Skipping market {m.get('market_slug')} (No CLOB orderbook)")
                        continue
            return False
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return False

    async def get_pyth_price(self, session):
        """Bypass Geographic restrictions using Decantralized Pyth Data."""
        try:
            async with session.get(PYTH_HERMES_URL, timeout=5) as resp:
                data = await resp.json()
                price_info = data['parsed'][0]['price']
                price = float(price_info['price']) * (10 ** price_info['expo'])
                return price
        except Exception as e:
            logger.error(f"Pyth Feed Error: {e}")
            return None

    async def main_loop(self):
        async with aiohttp.ClientSession() as session:
            logger.info("🟢 CLOUD BOT ACTIVE (Source: Pyth Network).")
            while True:
                try:
                    price = await self.get_pyth_price(session)
                    if not price: 
                        await asyncio.sleep(1)
                        continue
                        
                    now = datetime.now()
                    current_minute = now.replace(second=0, microsecond=0)
                    
                    if self.last_minute_ts and current_minute > self.last_minute_ts:
                        # Minute Concluded: Calculate Momentum (Price Change)
                        momentum = price - self.last_minute_price
                        self.price_buffer.append(momentum)
                        await self.evaluate_signal()
                        self.last_minute_price = price
                    
                    if not self.last_minute_ts:
                        self.last_minute_price = price
                    self.last_minute_ts = current_minute
                    
                    await asyncio.sleep(10) # 10s Sampling
                except Exception as e:
                    logger.error(f"Main loop crash: {e}")
                    await asyncio.sleep(5)

    async def evaluate_signal(self):
        if len(self.price_buffer) < 30: return # Wait for baseline
        
        history = list(self.price_buffer)
        mu, sigma = np.mean(history), np.std(history)
        if sigma < 1e-9: return
        
        current_mom = history[-1]
        z = (current_mom - mu) / sigma
        
        if abs(z) > self.z_threshold:
            logger.info(f"🚨 CLOUD SKEW ALERT: Z={z:.2f}. Probing Orderbook...")
            await self.execute_dry_run(z)

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
            logger.info(f"✔ Telemetry OK: {latency:.1f}ms latency | {ask_sz} Size")
        except Exception:
            logger.error(f"Telemetry error: {traceback.format_exc()}")

async def main():
    bot = SlowSkewBotV4()
    
    # 🔗 CLOUD HANDSHAKE
    clob_ok = bot.init_clob_client()
    if not clob_ok: return
    
    # 🎯 SMART DISCOVERY (Liquid Only)
    if not bot.bootstrap_discovery():
        logger.error("COULD NOT FIND LIQUID TARGET. Exiting.")
        return
        
    # Start Cloud Monitoring
    await bot.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown.")
    except Exception:
        logger.error(f"System Fatal: {traceback.format_exc()}")
