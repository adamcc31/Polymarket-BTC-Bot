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

# PYTH CONFIG: BTC/USD Price Feed ID
PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest?ids[]=ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace"

class SlowSkewBotV4:
    def __init__(self):
        # Configuration
        self.tfm_window_mins = 15
        self.baseline_window_mins = 240
        self.z_threshold = 1.0 # Detection threshold
        self.notif_threshold = 5.0 # High-alpha alert threshold
        
        # State
        self.yes_token_id = None
        self.target_market_slug = None
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
        manual_id = os.environ.get("TARGET_TOKEN_ID")
        if manual_id:
            logger.info(f"🛡️ MANUAL OVERRIDE: Using Token ID {manual_id}")
            self.yes_token_id = manual_id
            return True

        if os.path.exists('tmp_tokens.txt'):
            try:
                with open('tmp_tokens.txt', 'r') as f:
                    self.yes_token_id = f.readline().strip().replace('[','').replace(']','').replace('"','').replace(',','')
                    logger.info(f"Loaded local target: {self.yes_token_id}")
                    return True
            except: pass

        logger.info("Searching for LIQUID active Bitcoin market (Limit 50)...")
        try:
            r = requests.get('https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=50', timeout=10)
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
                        logger.info(f"Auditing '{m['question'][:40]}...' | Token: {tid}")
                        self.clob_client.get_order_book(tid)
                        self.yes_token_id = tid
                        self.target_market_slug = m.get('market_slug')
                        logger.info(f"🎯 LIQUID TARGET FOUND: {m['question']}")
                        return True
                    except: continue
            return False
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return False

    async def send_alert(self, session, title, message):
        # 1. TELEGRAM NATIVE
        tg_token = os.environ.get("TELEGRAM_TOKEN")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat:
            try:
                url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                payload = {"chat_id": tg_chat, "text": f"🔥 {title}\n\n{message}", "parse_mode": "Markdown"}
                async with session.post(url, json=payload, timeout=5) as resp:
                    if resp.status == 200: return
            except Exception as e:
                logger.error(f"Telegram failed: {e}")

        # 2. WEBHOOK FALLBACK (Discord)
        webhook_url = os.environ.get("NOTIF_WEBHOOK_URL")
        if not webhook_url: return
        try:
            payload = {"content": f"🔥 **{title}**\n{message}"}
            async with session.post(webhook_url, json=payload, timeout=5) as resp:
                if resp.status >= 300:
                    logger.warning(f"Webhook failed: {resp.status}")
        except Exception as e:
            logger.error(f"Notification error: {e}")

    async def get_pyth_price(self, session):
        try:
            async with session.get(PYTH_HERMES_URL, timeout=5) as resp:
                data = await resp.json()
                price_info = data['parsed'][0]['price']
                price = float(price_info['price']) * (10 ** price_info['expo'])
                return price
        except Exception: return None

    async def main_loop(self):
        async with aiohttp.ClientSession() as session:
            logger.info("🟢 CLOUD BOT ACTIVE (Source: Pyth Network).")
            # INITIAL ONLINE NOTIFICATION
            await self.send_alert(session, "[SYSTEM ONLINE]", 
                                 f"Slow Skew Bot v4 has started successfully.\nTarget: {self.target_market_slug or 'Discovery Active'}")
            
            while True:
                try:
                    price = await self.get_pyth_price(session)
                    if not price: 
                        await asyncio.sleep(1)
                        continue
                        
                    now = datetime.now()
                    current_minute = now.replace(second=0, microsecond=0)
                    
                    # 💓 CONSOLE HEARTBEAT (Every 5 Minutes)
                    if time.time() - self.last_heartbeat > 300:
                        logger.info(f"💓 HEARTBEAT: Monitoring {self.target_market_slug or 'Active Target'}. BTC: ${price:,.2f} | Samples: {len(self.price_buffer)}")
                        self.last_heartbeat = time.time()
                    
                    # 📱 TELEGRAM HEARTBEAT (Every 4 Hours to avoid spam)
                    if time.time() - self.last_tg_heartbeat > 14400:
                        await self.send_alert(session, "[HEARTBEAT]", 
                                             f"System is healthy.\nBTC: ${price:,.2f}\nZ-Score Samples: {len(self.price_buffer)}")
                        self.last_tg_heartbeat = time.time()

                    if self.last_minute_ts and current_minute > self.last_minute_ts:
                        momentum = price - self.last_minute_price
                        self.price_buffer.append(momentum)
                        await self.evaluate_signal(session, price)
                        self.last_minute_price = price
                    
                    if not self.last_minute_ts:
                        self.last_minute_price = price
                    self.last_minute_ts = current_minute
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    await asyncio.sleep(5)

    async def evaluate_signal(self, session, current_price):
        if len(self.price_buffer) < 5: return
        history = list(self.price_buffer)
        mu, sigma = np.mean(history), np.std(history)
        if sigma < 1e-9: return
        z = (history[-1] - mu) / sigma
        
        if abs(z) > self.z_threshold:
            logger.info(f"🚨 SKEW ALERT: Z={z:.2f}. Logging Telemetry...")
            await self.execute_dry_run(z)
            
            if abs(z) >= self.notif_threshold:
                await self.send_alert(session, "HIGH SKEW ANOMALY DETECTED", 
                                     f"Z-Score: `{z:.2f}`\nBTC Price: `${current_price:,.2f}`\nTarget: {self.target_market_slug}\nStatus: **Dry-Run Mode**")

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
            logger.warning(f"Telemetry write failed: {traceback.format_exc()}")

async def main():
    bot = SlowSkewBotV4()
    clob_ok = bot.init_clob_client()
    if not clob_ok: return
    if not bot.bootstrap_discovery():
        logger.error("COULD NOT FIND LIQUID TARGET. Exiting.")
        return
    await bot.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.error(f"System Fatal: {traceback.format_exc()}")
