import asyncio
import os
import time
import sys
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

class InfoFilter(logging.Filter):
    def filter(self, record): return record.levelno < logging.ERROR

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(log_formatter)
stdout_handler.addFilter(InfoFilter())
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler(f"{LOG_DIR}/execution_v4.log", encoding='utf-8')
file_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, stderr_handler, file_handler])
logger = logging.getLogger(__name__)

# CONFIG: HARDENED BTC/USD Price Feed ID
BTC_FEED_ID = "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
PYTH_HERMES_BASE = "https://hermes.pyth.network/v2/updates/price/latest"
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

class SlowSkewBotV4:
    def __init__(self):
        # Configuration
        self.tfm_window_mins = 15
        self.baseline_window_mins = 240
        self.z_threshold = 1.0 
        self.notif_threshold = 5.0 
        
        # State
        self.yes_token_id = None
        self.target_market_name = "Target Discovery Active..."
        self.price_buffer = deque(maxlen=self.baseline_window_mins)
        self.last_minute_price = None
        self.last_minute_ts = None
        self.last_heartbeat = 0
        self.last_tg_heartbeat = 0
        self.last_telemetry_export = time.time()
        self.is_redundant_mode = False
        self.pyth_fail_count = 0
        
        # Clients
        self.clob_client = None
        self.telemetry_file = f"{LOG_DIR}/execution_v4_telemetry.csv"
        self._init_telemetry()

    def _init_telemetry(self):
        if not os.path.exists(self.telemetry_file):
            with open(self.telemetry_file, "w") as f:
                f.write("trigger_ts,z_score,poly_up_ask,poly_up_bid,ask_size,bid_size,latency_ms,status,source\n")

    def init_clob_client(self):
        api_key = os.environ.get("POLY_BUILDER_API_KEY")
        api_secret = os.environ.get("POLY_BUILDER_SECRET")
        api_passphrase = os.environ.get("POLY_BUILDER_PASSPHRASE")
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not all([api_key, api_secret, api_passphrase, private_key]):
            return False
        try:
            creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)
            self.clob_client = ClobClient(host="https://clob.polymarket.com", key=private_key, chain_id=137, creds=creds)
            logger.info("Polymarket CLOB Client Initialized.")
            return True
        except Exception:
            logger.error(f"CLOB Init Failed: {traceback.format_exc()}")
            return False

    def bootstrap_discovery(self):
        # [REVERTED TO v28 STABLE LOGIC]
        manual_id = os.environ.get("TARGET_TOKEN_ID")
        if manual_id:
            logger.info(f"🛡️ MANUAL OVERRIDE: Using Token ID {manual_id}")
            self.yes_token_id = manual_id
            return True

        logger.info("Searching for Active Bitcoin target...")
        try:
            r = requests.get('https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=50', timeout=15)
            markets = r.json()
            for m in markets:
                # v28 question pattern check
                if ("Bitcoin" in m.get('question', '') or "BTC" in m.get('question', '')) and m.get('clobTokenIds'):
                    ids = m['clobTokenIds']
                    if isinstance(ids, str):
                        try: ids = json.loads(ids)
                        except: ids = [ids.replace('[','').replace(']','').replace('"','').replace(',','')]
                    if not ids: continue
                    tid = ids[0]
                    try:
                        logger.info(f"Auditing '{m['question'][:40]}...' | Token: {tid}")
                        self.clob_client.get_order_book(tid)
                        self.yes_token_id = tid
                        self.target_market_name = m.get('question', 'Unknown Bitcoin Market')
                        logger.info(f"🎯 LIQUID TARGET FOUND: {self.target_market_name}")
                        return True
                    except: continue
            return False
        except Exception as e:
            logger.error(f"Discovery error: {e}")
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

    async def push_telemetry_file(self, session):
        """[MODUL TELEMETRY v29]"""
        tg_token = os.environ.get("TELEGRAM_TOKEN")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if not (tg_token and tg_chat) or not os.path.exists(self.telemetry_file):
            return
        try:
            logger.info("📤 Pushing telemetry CSV to Telegram...")
            url = f"https://api.telegram.org/bot{tg_token}/sendDocument"
            data = aiohttp.FormData()
            data.add_field('chat_id', tg_chat)
            data.add_field('caption', f"📊 Telemetry Export: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            with open(self.telemetry_file, 'rb') as f:
                data.add_field('document', f, filename=os.path.basename(self.telemetry_file))
                async with session.post(url, data=data, timeout=30) as resp:
                    if resp.status == 200:
                        logger.info("✅ Telemetry CSV sent successfully.")
                        return True
        except Exception as e:
            logger.error(f"Telemetry push failed: {e}")
        return False

    async def get_binance_price(self, session):
        try:
            async with session.get(BINANCE_API_URL, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get('price', 0))
                return None
        except Exception: return None

    async def get_pyth_price(self, session):
        # [REVERTED TO v28 STABLE LOGIC]
        params = {"ids[]": [BTC_FEED_ID]}
        try:
            async with session.get(PYTH_HERMES_BASE, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.pyth_fail_count = 0
                    self.is_redundant_mode = False
                    parsed = data.get('parsed', []) if isinstance(data, dict) else []
                    if parsed:
                        p_info = parsed[0].get('price', {})
                        raw_px = p_info.get('price')
                        expo = p_info.get('expo')
                        if raw_px is not None and expo is not None:
                            return float(raw_px) * (10 ** int(expo))
                else:
                    logger.warning(f"Pyth Primary failed ({resp.status}). Activating Binance Fallback...")
        except Exception: pass
        self.pyth_fail_count += 1
        self.is_redundant_mode = True
        return await self.get_binance_price(session)

    async def main_loop(self):
        async with aiohttp.ClientSession() as session:
            logger.info("🟢 CLOUD BOT ACTIVE (v30: Stabilized + Telemetry).")
            await self.send_alert(session, "[SYSTEM RESTORED]", 
                                 f"Bot returned to v28 Discovery logic.\nTarget: `{self.target_market_name}`")
            
            while True:
                try:
                    price = await self.get_pyth_price(session)
                    if not price: 
                        await asyncio.sleep(5)
                        continue
                        
                    now = datetime.now()
                    current_minute = now.replace(second=0, microsecond=0)
                    
                    if self.last_heartbeat == 0 or (time.time() - self.last_heartbeat > 300):
                        source = "Binance (Backup)" if self.is_redundant_mode else "Pyth (Primary)"
                        logger.info(f"💓 HEARTBEAT | Source: {source} | BTC: ${price:,.2f}")
                        self.last_heartbeat = time.time()
                    
                    if time.time() - self.last_tg_heartbeat > 14400:
                        await self.send_alert(session, "[HEARTBEAT]", f"BTC Stable: ${price:,.2f}")
                        self.last_tg_heartbeat = time.time()

                    # Telemetry push (5h)
                    if time.time() - self.last_telemetry_export > 18000:
                        if await self.push_telemetry_file(session):
                            self.last_telemetry_export = time.time()

                    if self.last_minute_ts and current_minute > self.last_minute_ts:
                        momentum = price - self.last_minute_price
                        self.price_buffer.append(momentum)
                        await self.evaluate_signal(session, price)
                        self.last_minute_price = price
                    
                    if not self.last_minute_ts:
                        self.last_minute_price = price
                        logger.info(f"Baseline set: ${price:,.2f}")
                    
                    self.last_minute_ts = current_minute
                    backoff_delay = min(self.pyth_fail_count * 15, 60) if self.is_redundant_mode else 0
                    await asyncio.sleep(15 + backoff_delay)
                except Exception as e:
                    logger.error(f"Fatal Loop: {e}")
                    await asyncio.sleep(15)

    async def evaluate_signal(self, session, current_price):
        if len(self.price_buffer) < 2: return
        history = list(self.price_buffer)
        margin_mult = 1.001 if self.is_redundant_mode else 1.0
        mu, sigma = np.mean(history), np.std(history)
        if sigma < 1e-9: return
        z = (history[-1] - mu) / sigma
        eff_threshold = self.z_threshold * margin_mult
        
        if abs(z) > eff_threshold:
            mode = "REDUNDANT" if self.is_redundant_mode else "PRIMARY"
            logger.info(f"🚨 SKEW ALERT [{mode}]: Z={z:.2f}")
            await self.execute_dry_run(z)
            if abs(z) >= self.notif_threshold:
                await self.send_alert(session, f"HIGH MOMENTUM ({mode})", 
                                     f"Z-Score: `{z:.2f}`\nBTC: `${current_price:,.2f}`")

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
                f.write(f"{datetime.now().isoformat()},{z:.2f},{ask},{bid},{ask_sz},{bid_sz},{latency:.2f},SUCCESS,{'REDUNDANT' if self.is_redundant_mode else 'PRIMARY'}\n")
            logger.info(f"✔ Telemetry OK: {latency:.1f}ms")
        except Exception: pass

async def main():
    bot = SlowSkewBotV4()
    if not bot.init_clob_client(): return
    if not bot.bootstrap_discovery():
        logger.error("COULD NOT FIND TARGET.")
        return
    await bot.main_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.error(f"Terminal Fatal: {traceback.format_exc()}")
