# FULL TRD — Polymarket Bitcoin Up or Down (15 Menit)
## Probability Mispricing Detection Bot

**Versi Dokumen:** 1.0 — Production-Ready TRD  
**Dikonversi dari:** PRE-TRD v2.0 (Revised)  
**Status:** Final — Siap Implementasi  
**Tanggal:** 2025  
**Bahasa:** Indonesia  

---

## Daftar Isi

1. [Executive Summary](#1-executive-summary)
2. [System Overview & Architecture](#2-system-overview--architecture)
3. [Market & Contract Specification](#3-market--contract-specification)
4. [Data Sources & Ingestion Layer](#4-data-sources--ingestion-layer)
5. [Feature Engineering Specification](#5-feature-engineering-specification)
6. [Machine Learning Model Design](#6-machine-learning-model-design)
7. [Signal Generation Logic](#7-signal-generation-logic)
8. [Risk Management System](#8-risk-management-system)
9. [Execution System](#9-execution-system)
10. [Dry Run & Validation Framework](#10-dry-run--validation-framework)
11. [Performance Metrics & Evaluation](#11-performance-metrics--evaluation)
12. [Database Design](#12-database-design)
13. [Observability & Monitoring](#13-observability--monitoring)
14. [Configuration Management](#14-configuration-management)
15. [Deployment Architecture](#15-deployment-architecture)
16. [Security Considerations](#16-security-considerations)
17. [Failure Modes & Edge Cases](#17-failure-modes--edge-cases)
18. [Implementation Roadmap](#18-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Ringkasan Sistem

Sistem ini adalah **probability mispricing detection engine** yang beroperasi di pasar prediksi Polymarket — spesifiknya pada market *"Bitcoin Up or Down — 15 Minutes"*. Bot ini bukan sistem prediksi arah harga Bitcoin; ia adalah sistem yang mendeteksi **gap antara probabilitas yang diperkirakan model internal dan harga yang ditawarkan oleh CLOB (Central Limit Order Book) Polymarket** — kemudian mengeksploitasi gap tersebut sebagai peluang trading bernilai positif secara expected value.

Sistem dibangun menggunakan full-Python stack (Python 3.11 + asyncio) dan dapat di-deploy secara lokal maupun di Railway tanpa modifikasi kode.

### 1.2 Objective Utama

**Utama:** Deteksi dan eksploitasi mispricing probabilitas — bukan prediksi arah pergerakan harga BTC.

Definisi operasional mispricing:

```
edge_yes = P_model - clob_yes_ask  →  jika > 0: YES underpriced relatif terhadap model
edge_no  = (1 - P_model) - clob_no_ask  →  jika > 0: NO underpriced relatif terhadap model
```

Sistem hanya memasang taruhan ketika salah satu edge melebihi `MARGIN_OF_SAFETY` (default: 5 poin persentase / 0.05), seluruh hard gate terpenuhi, dan likuiditas CLOB mencukupi.

### 1.3 Core Edge yang Dimanfaatkan

Edge sistem ini bukan berasal dari keunggulan prediksi arah harga, melainkan dari tiga sumber struktural:

**Edge 1 — Informational:** Model mengintegrasikan sinyal microstructure (Order Book Imbalance, Trade Flow Momentum) secara kuantitatif yang tidak selalu tercermin dalam harga CLOB Polymarket.

**Edge 2 — Kontekstual:** Model menggabungkan Time-to-Resolution (TTR) dan Strike Distance sebagai fitur eksplisit. Ini memungkinkan valuasi yang lebih presisi dibandingkan trader yang menggunakan heuristik sederhana.

**Edge 3 — Disciplined Abstention:** Sistem di-desain untuk *tidak* berdagang sebagian besar waktu (expected abstain rate ~60–75%). Selektivitas ini adalah sumber edge, bukan kelemahan. Sistem hanya masuk ketika mispricing dan kondisi pasar memenuhi semua kriteria secara simultan.

### 1.4 Hard Constraints Non-Negotiable

Empat constraint berikut adalah **acceptance criteria sistem**, bukan catatan opsional:

| # | Constraint | Implikasi |
|---|---|---|
| C1 | Verifikasi mekanisme kontrak Polymarket sebelum development | Seluruh formula bergantung pada asumsi ini |
| C2 | Market Availability tidak dijamin — bot harus graceful degrade | Market Discovery wajib robust |
| C3 | Minimum 4 minggu data collection sebelum training | Timeline tidak bisa dipercepat |
| C4 | Modal minimum operasional $50 ($100 direkomendasikan) | $10 hanya untuk smoke test go-live |

---

## 2. System Overview & Architecture

### 2.1 Arsitektur Sistem (Deskriptif)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                        │
│                                                                       │
│   ┌───────────────────┐           ┌───────────────────────────────┐  │
│   │  Binance Exchange │           │        Polymarket Platform     │  │
│   │  ─────────────── │           │  ──────────────────────────── │  │
│   │  WS: depth20      │           │  REST: /markets (Discovery)    │  │
│   │  WS: aggTrade     │           │  REST: /book (CLOB snapshot)   │  │
│   │  WS: kline_15m    │           │  SDK:  py-clob-client (orders) │  │
│   └────────┬──────────┘           └──────────────┬────────────────┘  │
└────────────┼─────────────────────────────────────┼───────────────────┘
             │                                     │
             ▼                                     ▼
┌────────────────────────┐             ┌───────────────────────────────┐
│   binance_feed.py      │             │   market_discovery.py         │
│   ───────────────────  │             │   ─────────────────────────── │
│   WS conn + reconnect  │             │   Poll markets API            │
│   Circular buffer 500  │             │   Validate TTR >= 5 min       │
│   Bar OHLCV + OB snaps │             │   Return active_market object │
│   Observability metrics│             │   Hot-reload on market rotate │
└────────────┬───────────┘             └──────────────┬────────────────┘
             │                                        │
             │                         ┌──────────────▼────────────────┐
             │                         │   clob_feed.py                │
             │                         │   ─────────────────────────── │
             │                         │   Poll CLOB setiap 5 detik    │
             │                         │   YES/NO ask/bid + depth      │
             │                         │   Stale detection (>30s)      │
             │                         └──────────────┬────────────────┘
             │                                        │
             └──────────────────┬─────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │            feature_engine.py             │
             │  ───────────────────────────────────────  │
             │  24 features: Microstructure + TTR +     │
             │  Strike Distance + CLOB + Interaction    │
             │  Anti-lookahead: shift(1) before rolling │
             │  Output: numpy array (1, 24) + metadata  │
             └──────────────────┬───────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │               model.py                   │
             │  ───────────────────────────────────────  │
             │  LightGBM (0.7) + LogReg (0.3) ensemble  │
             │  P(BTC_T_res > strike | features, TTR)   │
             │  Output: P_model ∈ [0, 1]                │
             └──────────────────┬───────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │          signal_generator.py             │
             │  ───────────────────────────────────────  │
             │  Hard Gates: TTR / Regime / Liquidity    │
             │  Mispricing: edge_yes, edge_no           │
             │  Output: SignalResult (BUY_YES/NO/ABSTAIN)│
             └──────────────────┬───────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │            risk_manager.py               │
             │  ───────────────────────────────────────  │
             │  Half-Kelly sizing (CLOB odds)           │
             │  Dynamic multiplier (consecutive loss)   │
             │  Hard limits enforcement                 │
             │  Output: ApprovedBet | RejectedBet       │
             └──────────────────┬───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
       ┌─────────────────────┐   ┌──────────────────────┐
       │    dry_run.py       │   │    execution.py      │
       │  ─────────────────  │   │  ─────────────────── │
       │  Paper simulation   │   │  Live CLOB orders    │
       │  Real-time data     │   │  py-clob-client SDK  │
       │  PASS/FAIL metrics  │   │  LIVE_MODE gate      │
       └──────────┬──────────┘   └──────────┬───────────┘
                  │                          │
                  └─────────────┬────────────┘
                                ▼
             ┌──────────────────────────────────────────┐
             │  database.py + exporter.py + cli.py      │
             │  ───────────────────────────────────────  │
             │  SQLite (local) / PostgreSQL (Railway)   │
             │  CSV + JSON exports per sesi             │
             │  Rich terminal dashboard                 │
             └──────────────────────────────────────────┘
```

### 2.2 Data Flow End-to-End

```
Setiap bar 15-menit (dan tick real-time di antara bar):

[T] Binance WS menerima kline_close event (bar selesai)
 │
 ├─► [T+0ms]   BinanceFeed update circular buffer (OHLCV row, OB snapshot)
 ├─► [T+50ms]  MarketDiscovery konfirmasi active_market masih valid
 ├─► [T+100ms] CLOBFeed.get_clob_snapshot() — ambil data terbaru (cached 5s)
 ├─► [T+150ms] FeatureEngine.compute(bar, active_market, clob_state)
 │              → return FeatureVector(24 features) + metadata
 ├─► [T+200ms] Model.predict(feature_vector) → P_model
 ├─► [T+250ms] SignalGenerator.evaluate(P_model, clob_state, market, features)
 │              → return SignalResult
 ├─► [T+280ms] RiskManager.approve(signal, capital, trade_history)
 │              → return ApprovedBet | RejectedBet
 ├─► [T+300ms] DryRunEngine / ExecutionClient process bet
 └─► [T+350ms] Database.insert(signal, trade); CLI.refresh()

Total target latency: < 500ms per bar (non-negotiable untuk integritas fitur)
```

### 2.3 Komponen Utama dan Interaksi Antar Modul

| Modul | Input | Output | Dependency |
|---|---|---|---|
| `market_discovery.py` | Polymarket REST API | `ActiveMarket` object | httpx, config |
| `binance_feed.py` | Binance WebSocket | OHLCV buffer, OB snapshots, callbacks | websockets, ccxt |
| `clob_feed.py` | Polymarket REST CLOB | `CLOBState` object | httpx, active_market |
| `feature_engine.py` | OHLCV + OB + CLOBState + ActiveMarket | `FeatureVector(1,24)` + metadata | numpy, pandas |
| `model.py` | `FeatureVector` | `P_model` float | lightgbm, sklearn |
| `signal_generator.py` | `P_model` + `CLOBState` + metadata | `SignalResult` | config |
| `risk_manager.py` | `SignalResult` + capital + history | `ApprovedBet` / `RejectedBet` | config |
| `dry_run.py` | `ApprovedBet` + real-time outcome | session metrics, PASS/FAIL | database |
| `execution.py` | `ApprovedBet` | CLOB order placement | py-clob-client |
| `database.py` | semua modul | persistence layer | SQLAlchemy |
| `exporter.py` | database | CSV + JSON files | pandas |
| `cli.py` | semua modul | Rich terminal dashboard | rich, click |

---

## 3. Market & Contract Specification

### 3.1 Identifikasi Market Target

```
Platform     : Polymarket (https://polymarket.com)
Market name  : "Bitcoin Up or Down — 15 Minutes"
Market type  : Binary outcome (YES / NO)
Instrumen    : USDC-settled
Payout       : $1.00 per share jika outcome YES terjadi (untuk YES buyer)
               $1.00 per share jika outcome NO terjadi (untuk NO buyer)
```

### 3.2 Mekanisme Resolusi Kontrak

```
T_open        : Waktu market dibuka — strike_price di-capture pada momen ini
T_resolution  : T_open + 15 menit

Pertanyaan kontrak:
  "Apakah harga BTC lebih tinggi dari strike_price pada T_resolution?"

Outcome:
  YES menang jika: BTC_price(T_resolution) > strike_price
  NO  menang jika: BTC_price(T_resolution) ≤ strike_price

Sumber harga resolusi:
  REQUIRES VALIDATION: Konfirmasi via Polymarket API documentation apakah:
    (a) Spot price dari oracle tertentu (Chainlink? Pyth?)
    (b) TWAP dari interval tertentu
    (c) Binance last trade price
  Implikasi: jika bukan spot price real-time Binance, formula VAM dan
  strike_distance_pct harus disesuaikan terhadap oracle yang digunakan.

Strike price definition:
  REQUIRES VALIDATION: Konfirmasi bahwa strike_price = harga BTC tepat saat T_open.
  Bisa jadi diambil dari oracle price saat market creation, bukan Binance last price.
```

### 3.3 Market Rotation

```
Rotasi      : Setiap 15 menit, satu market baru dibuka setelah market sebelumnya resolve
Market ID   : Berubah setiap siklus (condition_id berbeda per market)
Gap         : REQUIRES VALIDATION — konfirmasi apakah ada jeda antara
              resolusi market lama dan pembukaan market baru

Implikasi arsitektur:
  1. Tidak boleh hardcode market ID
  2. Market Discovery module harus polling aktif
  3. Bot harus bisa hot-reload active_market tanpa restart
  4. Harus handle kondisi: tidak ada market aktif (WAITING mode)
```

### 3.4 Implikasi Terhadap Desain Sistem

| Karakteristik Kontrak | Implikasi Desain |
|---|---|
| Strike price tetap saat T_open | `strike_price` adalah input wajib model — bukan current price |
| TTR menentukan uncertainty window | TTR_normalized adalah fitur terpenting (v2.0) |
| Binary outcome $0 atau $1 | Semua Kelly formula menggunakan binary payout |
| CLOB: share = [0,1] probabilitas | `clob_yes_ask` adalah implied probability yang bisa dibandingkan langsung dengan P_model |
| Market rotate setiap 15 menit | Butuh Market Discovery + hot-reload aktif |

### 3.5 Verifikasi Wajib Sebelum Development Dimulai

```
[ ] Konfirmasi oracle harga resolusi (endpoint dan metodologi)
[ ] Konfirmasi strike_price = harga saat T_open atau ada mekanisme lain
[ ] Konfirmasi interval market baru: benar-benar setiap 15 menit atau adaptive?
[ ] Konfirmasi ada/tidaknya gap antara resolusi dan pembukaan market baru
[ ] Observasi manual 5–10 market cycles untuk validasi asumsi
[ ] Konfirmasi ketersediaan historical market data (untuk label construction training)
```

---

## 4. Data Sources & Ingestion Layer

### 4.1 Binance Data Feed (`binance_feed.py`)

#### 4.1.1 WebSocket Streams

```python
# Streams yang di-subscribe (concurrent, asyncio)
STREAMS = [
    "btcusdt@depth20@100ms",   # Top-20 orderbook, update 100ms
    "btcusdt@aggTrade",         # Aggregated trades real-time
    "btcusdt@kline_15m",        # 15-menit OHLCV bar close event
]

WS_BASE_URL = "wss://stream.binance.com:9443/stream"
```

#### 4.1.2 REST Fallback

```python
# REST digunakan untuk:
#   1. Historical OHLCV bootstrap (500 bar saat startup)
#   2. Gap-fill jika WS disconnect > 30 detik

REST_ENDPOINT = "GET /api/v3/klines"
PARAMS = {
    "symbol"   : "BTCUSDT",
    "interval" : "15m",
    "limit"    : 500  # max per request
}
```

#### 4.1.3 In-Memory Storage

```python
# Circular buffer — thread-safe deque
BUFFER_CAPACITY = 500  # bars OHLCV + OB snapshots

# Schema per entry OHLCV:
{
    "open_time": int,      # millisecond epoch UTC
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
    "close_time": int
}

# Schema per entry OB snapshot (setiap 1 detik dari depth20):
{
    "timestamp": datetime,
    "bids": [[price, qty], ...],  # top 20 level
    "asks": [[price, qty], ...],  # top 20 level
}
```

#### 4.1.4 Data Validation Rules

```python
# Reject bar jika:
assert volume > 0,           "Zero-volume bar rejected"
assert high >= low,          "Inconsistent OHLC: high < low"
assert open <= high,         "Inconsistent OHLC: open > high"
assert open >= low,          "Inconsistent OHLC: open < low"
assert close <= high,        "Inconsistent OHLC: close > high"
assert close >= low,         "Inconsistent OHLC: close < low"
assert open_time < close_time, "Invalid timestamp ordering"
```

#### 4.1.5 Reconnection Strategy

```python
# Exponential backoff reconnection
MAX_RETRIES       = 10
INITIAL_DELAY_S   = 1
BACKOFF_MULTIPLIER = 2
MAX_DELAY_S       = 60

# Gap detection
STALE_THRESHOLD_S = 30  # jika disconnect > 30s → mark data as STALE
# Selama STALE: blokir semua signal generation, log WARNING
```

#### 4.1.6 Observability Metrics

```python
# Metrics yang harus di-track per sesi:
ws_message_drop_rate = dropped_messages / total_expected_messages
# Detected via: sequence number gap detection pada stream aggTrade

ws_latency_ms_p99 = percentile_99(time_received - exchange_timestamp)
queue_depth       = len(unprocessed_message_queue)

# Alert threshold:
ALERT_DROP_RATE   = 0.001  # 0.1%
ALERT_LATENCY_MS  = 2000   # 2 detik
```

### 4.2 Polymarket CLOB Feed (`clob_feed.py`)

#### 4.2.1 Polling Strategy

```python
# Polymarket CLOB saat ini REST-only (bukan WebSocket)
POLL_INTERVAL_S  = 5     # setiap 5 detik
STALE_TIMEOUT_S  = 30    # jika data belum diperbarui > 30s → flag STALE
CACHE_EXPIRE_S   = 10    # cache snapshot expire setelah 10 detik

CLOB_ENDPOINT = "GET https://clob.polymarket.com/book"
PARAMS        = {"token_id": clob_token_id_yes}  # fetch YES orderbook
# Fetch NO book terpisah dengan token_id NO
```

#### 4.2.2 CLOBState Schema (Pydantic Model)

```python
class CLOBState(BaseModel):
    market_id      : str
    timestamp      : datetime
    yes_ask        : float    # best ask YES, range [0, 1]
    yes_bid        : float    # best bid YES, range [0, 1]
    no_ask         : float    # best ask NO,  range [0, 1]
    no_bid         : float    # best bid NO,  range [0, 1]
    yes_depth_usd  : float    # total USDC depth YES dalam 3% dari ask
    no_depth_usd   : float    # total USDC depth NO  dalam 3% dari ask
    market_vig     : float    # yes_ask + no_ask - 1.0
    is_liquid      : bool     # yes_depth >= MIN && no_depth >= MIN && vig <= MAX
    is_stale       : bool     # timestamp > STALE_TIMEOUT_S yang lalu
```

#### 4.2.3 Error Handling & Fallback

```python
# Jika fetch gagal:
#   1. Gunakan cached snapshot terakhir + set is_stale = True
#   2. Jika stale > STALE_TIMEOUT_S → blokir semua signal (LIQUIDITY_BLOCK)
#   3. Log error dengan konteks penuh (response code, latency, market_id)

# Retry policy:
MAX_RETRIES      = 3
BACKOFF_DELAYS   = [1, 2, 4]  # exponential backoff dalam detik
```

### 4.3 Market Discovery (`market_discovery.py`)

#### 4.3.1 Discovery Logic

```python
DISCOVERY_ENDPOINT   = "GET https://clob.polymarket.com/markets"
POLL_INTERVAL_S      = 30    # normal poll
WAITING_POLL_S       = 60    # poll saat tidak ada market aktif
MIN_TTR_TO_DISCOVER  = 5.0   # menit — jangan masuk market yang tersisa < 5 menit

# Filter market:
#   1. Nama mengandung: "Bitcoin Up or Down" atau "BTC Up or Down"
#   2. Status: ACTIVE (belum resolve)
#   3. TTR >= MIN_TTR_TO_DISCOVER
#   4. Memiliki YES dan NO token_id di CLOB
```

#### 4.3.2 ActiveMarket Schema (Pydantic Model)

```python
class ActiveMarket(BaseModel):
    market_id       : str      # Polymarket condition_id
    question        : str      # text pertanyaan (untuk verifikasi)
    strike_price    : float    # harga BTC saat T_open
    T_open          : datetime # waktu market dibuka (UTC)
    T_resolution    : datetime # waktu resolusi (UTC)
    TTR_minutes     : float    # dihitung saat discovery
    clob_token_ids  : dict     # {"YES": "0x...", "NO": "0x..."}
```

#### 4.3.3 State Machine Market Discovery

```
States: SEARCHING → ACTIVE → WAITING → SEARCHING

SEARCHING: Polling setiap 30 detik, mencari market aktif
  → Jika market ditemukan & TTR >= 5 min: transisi ke ACTIVE
  → Jika tidak ditemukan: transisi ke WAITING

ACTIVE: Market sedang dipantau, bot aktif trading
  → Cek TTR setiap 30 detik
  → Jika T_resolution sudah lewat: transisi ke SEARCHING
  → Jika TTR < 3 menit: flag LATE (tidak boleh entry baru)

WAITING: Tidak ada market aktif
  → Polling setiap 60 detik
  → Log INFO setiap 5 menit (deklarasi WAITING mode)
  → Jika market ditemukan: transisi ke SEARCHING
```

### 4.4 Data Validation & Fallback Strategy Keseluruhan

| Kondisi | Behavior | Signal State |
|---|---|---|
| Binance WS nominal | Normal | Sesuai gate lain |
| Binance WS STALE (>30s) | Block all signals | ABSTAIN |
| CLOB STALE (>30s) | Block all signals | LIQUIDITY_BLOCK |
| CLOB fetch error < 30s | Gunakan cache, flag stale | Normal |
| Market ID berubah | Hot-reload, update CLOB token_ids | Reset TTR phase |
| Tidak ada market aktif | WAITING mode | ABSTAIN |
| Binance REST fallback aktif | Log WARNING, continue | Normal setelah fill |

---

## 5. Feature Engineering Specification

**Prinsip Anti-Lookahead (KRITIS):**  
Semua rolling computation **wajib** menggunakan `shift(1)` sebelum kalkulasi window, sehingga nilai di waktu `t` hanya menggunakan data dari `[t-window, t-1]`. Pelanggaran terhadap constraint ini akan menghasilkan data snooping bias yang merusak validitas seluruh backtest.

### 5.1 Microstructure Features (dari Binance)

#### Feature 01 — Order Book Imbalance (OBI)

```
Definisi matematis:
  OBI = (Σ bid_qty[1:N] - Σ ask_qty[1:N]) / (Σ bid_qty[1:N] + Σ ask_qty[1:N])

Input data   : Top-5 orderbook level dari Binance depth20 stream
Range nilai  : [-1.0, +1.0]
Interpretasi :
  OBI > +0.2  → tekanan beli dominan, bullish microstructure
  OBI < -0.2  → tekanan jual dominan, bearish microstructure
  [-0.2, +0.2] → balanced/neutral

Constraint:
  - Snapshot diambil setiap 1 detik
  - Level N = 5 (top 5 bids dan asks)
  - Epsilon: tidak diperlukan (denominator tidak bisa nol jika ada orderbook)
```

#### Feature 02 — Trade Flow Momentum (TFM)

```
Definisi matematis:
  TFM_raw        = Σ taker_buy_volume[60s window] - Σ taker_sell_volume[60s window]
  TFM_normalized = TFM_raw / (std(TFM_raw, window=20_periods) + ε)
  ε = 1e-8 (zero-division guard)

Input data   : Binance aggTrade stream (buy-initiated vs sell-initiated)
Range nilai  : Tidak terbatas — normalized z-score
Interpretasi :
  |TFM_normalized| > 1.5 → sinyal bermakna secara statistik
  TFM_normalized > 1.5   → net buying pressure signifikan
  TFM_normalized < -1.5  → net selling pressure signifikan

Constraint:
  - Window aggregasi raw: 60 detik (bukan bar-based)
  - Window normalisasi: 20 periode
  - Anti-lookahead: gunakan shift(1) sebelum rolling std
```

#### Feature 03 — Volatility-Adjusted Momentum (VAM)

```
Definisi matematis:
  close_returns = ln(close[t] / close[t-1])
  realized_vol  = std(close_returns, window=12)  # 12 bars × 15 menit = 3 jam
  VAM           = close_returns / (realized_vol + ε)

Input data   : OHLCV close prices (15-menit bars)
Range nilai  : Tidak terbatas — z-score-like
Interpretasi :
  Magnitude pergerakan bar terkini relatif terhadap "noise normal" 3 jam terakhir
  |VAM| > 2.0 → pergerakan outlier secara historis

Constraint:
  - Window 12 bars = 3 jam data terbaru
  - Anti-lookahead: realized_vol computed dari [t-12, t-1]
```

#### Feature 04 — Realized Volatility (RV)

```
Definisi matematis:
  close_returns = ln(close[t] / close[t-1])
  RV = std(close_returns, window=12) × √(252 × 96)
  # 96 bars per hari di 15-menit, 252 hari trading per tahun

Input data   : OHLCV close prices
Range nilai  : [0, ∞), annualized, typical BTC: 0.5 – 2.0 (50%–200% annualized)
Interpretasi :
  RV rendah  → volatilitas tenang, OBI dan TFM lebih prediktif
  RV tinggi  → volatilitas ekstrem, model cenderung kurang reliable

Constraint:
  - Window 12 bars (3 jam)
  - Anti-lookahead: window computed dari [t-12, t-1]
  - Digunakan juga untuk Regime Filter (volatility percentile)
```

#### Feature 05 — Volatility Percentile

```
Definisi matematis:
  vol_percentile = rolling_rank(RV[t], window=500) / 500

Input data   : RV series (feature 04)
Range nilai  : [0.0, 1.0]
Interpretasi :
  0.15 → RV sedang di 15th percentile dari 500 bar terakhir
  0.80 → RV sedang di 80th percentile (volatilitas tinggi)

Constraint:
  - Window 500 bar = ~5 hari data
  - Anti-lookahead: rank dihitung dari [t-500, t-1]
  - Digunakan sebagai hard gate dalam Regime Filter
```

#### Feature 06 — Order Book Depth Ratio

```
Definisi matematis:
  depth_ratio = Σ bid_size[level 1..3] / (Σ ask_size[level 1..3] + ε)

Input data   : Binance orderbook top-3 levels
Range nilai  : (0, ∞), typical: [0.5, 2.0]
Interpretasi :
  > 1.0 → bid side lebih tebal (demand > supply di level top)
  < 1.0 → ask side lebih tebal (supply > demand)

Constraint:
  - Level 1–3 dari depth20 snapshot
  - ε = 1e-8
```

#### Feature 07 — Price vs EMA20

```
Definisi matematis:
  price_vs_ema20 = (close[t] - EMA(close, period=20)) / (close[t] + ε)

Input data   : OHLCV close prices
Range nilai  : Kecil, typical: [-0.02, +0.02]
Interpretasi :
  Positif → harga di atas rata-rata 20 bar terakhir (uptrend)
  Negatif → harga di bawah EMA (downtrend)

Constraint:
  - EMA dihitung dari [t-20, t-1]
  - Menggunakan EMA pandas (ewm dengan adjust=False)
```

#### Feature 08 — Binance Spread (bps)

```
Definisi matematis:
  binance_spread_bps = (best_ask - best_bid) / mid_price × 10000

Input data   : Binance depth20 top level
Range nilai  : [0, ∞), normal BTC/USDT: 1–3 bps
Interpretasi :
  < 5 bps → likuiditas normal, OBI data reliable
  > 5 bps → spread melebar, OBI/TFM kurang reliable → Regime Gate aktif

Constraint:
  - Digunakan sebagai hard gate dalam Regime Filter (bukan hanya fitur model)
```

### 5.2 Temporal Cyclical Features

#### Features 09–12 — Time Encoding

```
Definisi matematis:
  hour_sin = sin(2π × utc_hour / 24)
  hour_cos = cos(2π × utc_hour / 24)
  dow_sin  = sin(2π × day_of_week / 7)   # day_of_week: 0 = Senin
  dow_cos  = cos(2π × day_of_week / 7)

Input data   : datetime.utcnow() saat bar close
Range nilai  : [-1.0, +1.0] masing-masing
Interpretasi :
  Encoding cyclical: pair sin+cos bersama-sama merepresentasikan posisi dalam siklus
  Tanpa ini, model tidak bisa membedakan 23:00 dan 01:00 (keduanya "dekat midnight")

Constraint:
  - Gunakan UTC selalu — Binance menggunakan UTC
  - day_of_week: 0 = Senin, 6 = Minggu (konsisten dengan Python datetime.weekday())
```

### 5.3 Contextual Market Features (TTR & Strike)

#### Feature 13 — TTR Normalized

```
Definisi matematis:
  TTR_seconds    = (T_resolution - datetime.utcnow()).total_seconds()
  TTR_minutes    = TTR_seconds / 60.0
  TTR_normalized = max(0.0, min(1.0, TTR_minutes / 15.0))

Input data   : ActiveMarket.T_resolution + datetime.utcnow()
Range nilai  : [0.0, 1.0]
Interpretasi :
  1.0 = market baru dibuka (15 menit tersisa)
  0.5 = 7.5 menit tersisa (tengah market)
  0.0 = tepat di momen resolusi

Constraint:
  - Nilai tidak pernah negatif (max(0.0, ...))
  - Dihitung fresh setiap bar — bukan di-cache
```

#### Features 14–15 — TTR Cyclical Encoding

```
Definisi matematis:
  TTR_sin = sin(π × TTR_normalized)
  TTR_cos = cos(π × TTR_normalized)

Input data   : TTR_normalized (feature 13)
Range nilai  : [0.0, 1.0] untuk TTR_sin; [-1.0, 1.0] untuk TTR_cos
Interpretasi :
  TTR_sin: puncak (1.0) saat TTR = 0.5 (7.5 menit tersisa)
  TTR_cos: membedakan fase awal vs akhir (meskipun TTR_sin sama)
  Bersama-sama: model bisa belajar non-linearity dalam lifecycle market

Constraint:
  - Gunakan π (bukan 2π) karena TTR hanya berjalan dari 1.0 ke 0.0 (setengah siklus)
```

#### Feature 16 — Strike Distance (%)

```
Definisi matematis:
  strike_distance_pct = (current_btc_price - strike_price) / strike_price × 100

Input data   : Binance current price + ActiveMarket.strike_price
Range nilai  : Tidak terbatas, typical dalam 15-menit window: [-2.0, +2.0]
Interpretasi :
  > +0.5%  → harga sudah jelas di atas strike (YES territory, model kurang tambah nilai)
  < -0.5%  → harga jelas di bawah strike (NO territory)
  [-0.5, +0.5] → "contest zone" — di sini model memberikan nilai tambah terbesar

Constraint:
  - strike_price HARUS dari ActiveMarket.strike_price (T_open), BUKAN current price
  - Kesalahan menggunakan current price sebagai strike akan merusak seluruh feature ini
```

#### Feature 17 — Contest Urgency

```
Definisi matematis:
  contest_urgency = |strike_distance_pct| × (1 - TTR_normalized)

Input data   : Features 16 dan 13
Range nilai  : [0, ∞), typical: [0, 2]
Interpretasi :
  Nilai tinggi → harga dekat ke strike DAN waktu menuju resolusi hampir habis
  → kondisi paling kritis untuk akurasi model

Constraint:
  - Interaction feature yang didesain sebagai sinyal eksplisit untuk kondisi kritis
```

### 5.4 Interaction Features

#### Features 18–20 — TTR × Microstructure Interactions

```
Definisi matematis:
  ttr_x_obi    = TTR_normalized × OBI
  ttr_x_tfm    = TTR_normalized × TFM_normalized
  ttr_x_strike = TTR_normalized × strike_distance_pct

Input data   : Features 13, 01, 02, 16

Interpretasi :
  ttr_x_obi: OBI lebih prediktif ketika TTR besar (masih banyak waktu untuk pergerakan)
  ttr_x_tfm: TFM memberikan sinyal lebih kuat di awal lifecycle market
  ttr_x_strike: Interaksi TTR dan strike distance untuk menangkap dynamics akhir market

Rasional:
  LightGBM secara otomatis menemukan interaksi via tree splitting.
  Explicit interaction features memberikan sinyal langsung yang mempercepatlearning
  dan mempermudah interpretasi feature importance.

Constraint:
  - Dihitung setelah features komponen tersedia
  - Urutan komputasi: microstructure → TTR → strike → interactions
```

### 5.5 CLOB Features (dari Polymarket)

#### Features 21–24 — CLOB Polymarket

```
Feature 21 — CLOB YES Mid
  Definisi: clob_yes_mid = (clob_yes_ask + clob_yes_bid) / 2
  Range   : [0.0, 1.0]
  Makna   : Implied probabilitas YES dari Polymarket CLOB (konsensus pasar)
  Catatan : Digunakan sebagai FITUR (bukan hanya untuk edge calculation)
            Model dapat belajar kapan P_model harus deviate dari CLOB mid

Feature 22 — CLOB YES Spread
  Definisi: clob_yes_spread = clob_yes_ask - clob_yes_bid
  Range   : [0.0, 1.0], typical: 0.01–0.05
  Makna   : Lebar spread YES — proxy ketidakpastian dan likuiditas
  Catatan : Spread lebar → liquidity rendah → execution risk tinggi

Feature 23 — CLOB NO Spread
  Definisi: clob_no_spread = clob_no_ask - clob_no_bid
  Range   : [0.0, 1.0]
  Makna   : Sama seperti feature 22 untuk sisi NO

Feature 24 — Market Vig
  Definisi: market_vig = clob_yes_ask + clob_no_ask - 1.0
  Range   : [0.0, 0.2], normal: 0.02–0.05 (2–5%)
  Makna   : Total "rake" Polymarket. Vig tinggi → market kurang efisien
  Hard gate: market_vig > 0.07 → ABSTAIN (tidak trade)
```

### 5.6 Feature List & Output Format

```python
FEATURE_LIST = [
    "OBI",                  # 01
    "TFM_normalized",       # 02
    "VAM",                  # 03
    "RV",                   # 04
    "vol_percentile",       # 05
    "depth_ratio",          # 06
    "price_vs_ema20",       # 07
    "binance_spread_bps",   # 08
    "hour_sin",             # 09
    "hour_cos",             # 10
    "dow_sin",              # 11
    "dow_cos",              # 12
    "TTR_normalized",       # 13
    "TTR_sin",              # 14
    "TTR_cos",              # 15
    "strike_distance_pct",  # 16
    "contest_urgency",      # 17
    "ttr_x_obi",            # 18
    "ttr_x_tfm",            # 19
    "ttr_x_strike",         # 20
    "clob_yes_mid",         # 21
    "clob_yes_spread",      # 22
    "clob_no_spread",       # 23
    "market_vig",           # 24
]
# Total: 24 features
# Shape output: numpy.ndarray (1, 24) — single inference row

# Feature list disimpan di: config/feature_list.json
# Model WAJIB dilatih dengan urutan persis ini
# Perubahan urutan = retrain wajib
```

### 5.7 Rolling Z-Score Normalisasi (Anti-Lookahead)

```python
def z_score_safe(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Z-score normalisasi yang aman dari lookahead bias.
    KRITIS: JANGAN gunakan series[t] dalam kalkulasi window untuk series[t].
    """
    # shift(1) memastikan rolling hanya menggunakan data sebelum t
    rolling_mean = series.shift(1).rolling(window=window, min_periods=20).mean()
    rolling_std  = series.shift(1).rolling(window=window, min_periods=20).std()
    return (series - rolling_mean) / (rolling_std + 1e-8)
```

---

## 6. Machine Learning Model Design

### 6.1 Problem Formulation

**Target variable:**  
```
y = P(BTC_price(T_resolution) > strike_price | features, TTR_normalized)
Label: binary
  1 → BTC_at_T_resolution > strike_price  (YES outcome)
  0 → BTC_at_T_resolution ≤ strike_price  (NO outcome)
```

**Bukan:** `P(BTC naik dari harga sekarang dalam N menit)`.  
Strike price adalah reference point yang tetap per market cycle, bukan rolling current price.

**Konsekuensi untuk label construction:**
- Setiap training sample memerlukan: `(timestamp_signal, strike_price, BTC_at_T_resolution)`
- `strike_price` diambil dari `T_open` market, bukan dari `timestamp_signal`
- `BTC_at_T_resolution` diambil dari Binance kline close pada timestamp terdekat ke `T_resolution`

### 6.2 Model Architecture

```
Primary Model : LightGBM Classifier
Secondary     : Logistic Regression (baseline — deteksi overfit)
Ensemble      : weighted_avg = 0.7 × LGBM_prob + 0.3 × LogReg_prob

Rasional ensemble:
  - LightGBM: menangkap non-linearitas dan interaksi fitur secara otomatis
  - LogReg: memberikan regularisasi implisit — jika LightGBM >> LogReg,
    indikasi overfit yang harus diselidiki
  - Bobot 70/30: LightGBM dominan, LogReg sebagai stabilizer
```

#### 6.2.1 LightGBM Hyperparameter Space (untuk Optuna)

```python
LGBM_PARAM_SPACE = {
    "n_estimators"      : (100, 1000),
    "max_depth"         : (3, 8),
    "num_leaves"        : (15, 127),
    "learning_rate"     : (0.01, 0.1),
    "feature_fraction"  : (0.6, 1.0),
    "bagging_fraction"  : (0.6, 1.0),
    "bagging_freq"      : (1, 10),
    "min_child_samples" : (10, 100),
    "reg_alpha"         : (0.0, 1.0),    # L1
    "reg_lambda"        : (0.0, 1.0),    # L2
    "class_weight"      : "balanced",    # untuk handle imbalanced classes
}

OPTUNA_N_TRIALS = 50
CV_STRATEGY     = TimeSeriesSplit(n_splits=5)
```

#### 6.2.2 Logistic Regression Config

```python
LOGREG_CONFIG = {
    "solver"    : "lbfgs",
    "max_iter"  : 1000,
    "C"         : 0.1,           # regularisasi (tunable)
    "class_weight": "balanced",
}
# StandardScaler wajib sebelum LogReg (LightGBM tidak perlu scaling)
```

### 6.3 Training Pipeline

#### 6.3.1 Data Requirements

```
Minimum data historis:
  - Binance OHLCV 15-menit: 3 bulan terakhir (≥ 8,640 bars)
  - Polymarket market history: strike_price + BTC_at_T_resolution per market
    REQUIRES VALIDATION: ketersediaan data historis via API
  - Binance aggTrade stream: untuk reconstruct TFM historis
  - Binance orderbook snapshots: minimum 4 minggu (dikumpulkan live)

Estimasi total data: 2–3 GB untuk 3 bulan penuh
```

#### 6.3.2 Train / Validation / Test Split

```python
# WAJIB: Chronological split, TIDAK random
# Random split menyebabkan data leakage temporal

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# Contoh untuk 8,640 bars:
# train  : bar 1      – 5,184
# val    : bar 5,185  – 6,912
# test   : bar 6,913  – 8,640

# Penggunaan:
# train  → fit model + hyperparameter tuning (via Optuna CV)
# val    → early stopping untuk LightGBM
# test   → evaluasi final (brier score, AUC, win_rate_oos)
# JANGAN touch test set sampai model final dipilih
```

#### 6.3.3 Training Script (`scripts/train_model.py`)

```python
"""
Usage: python scripts/train_model.py --data-path ./data/processed/training.parquet

Output:
  models/model_v{YYYYMMDD_HHMMSS}.pkl    → LightGBM model
  models/logreg_v{YYYYMMDD_HHMMSS}.pkl   → LogReg model
  models/scaler_v{YYYYMMDD_HHMMSS}.pkl   → StandardScaler (untuk LogReg)
  models/training_metrics_{version}.json → evaluasi metrics
  config/feature_list.json               → urutan fitur (TIDAK boleh diubah manual)
"""

def train_pipeline(data_path: str) -> TrainingResult:
    df = load_and_validate(data_path)
    X, y = build_features(df)                    # anti-lookahead guaranteed
    X_train, X_val, X_test = chronological_split(X, y)
    
    # Optuna hyperparameter search
    lgbm_best = run_optuna(X_train, y_train, n_trials=50)
    
    # Fit final models
    lgbm_model  = fit_lightgbm(X_train, y_train, X_val, y_val, lgbm_best)
    logreg_model = fit_logreg(X_train_scaled, y_train)
    
    # Evaluate ensemble on test set
    metrics = evaluate_ensemble(lgbm_model, logreg_model, X_test, y_test)
    
    # Save artifacts
    save_models(lgbm_model, logreg_model, scaler, version_tag)
    save_metrics(metrics, version_tag)
    
    return TrainingResult(metrics=metrics, version=version_tag)
```

### 6.4 Model Persistence & Versioning

```python
MODEL_DIR      = "models/"  # di Railway: /app/data/models/
KEEP_N_VERSIONS = 3  # simpan 3 versi terakhir untuk rollback

# File per versi:
f"model_lgbm_v{YYYYMMDD_HHMMSS}.pkl"
f"model_logreg_v{YYYYMMDD_HHMMSS}.pkl"
f"scaler_v{YYYYMMDD_HHMMSS}.pkl"
f"training_metrics_{version}.json"  # brier, AUC, win_rate_oos, timestamp

# Rollback: load versi sebelumnya dari MODEL_DIR
# CLI command: python main.py --rollback-model
```

### 6.5 Validation Strategy (Time-Series Aware)

```python
# TimeSeriesSplit untuk Optuna CV — tidak ada data leakage
CV = TimeSeriesSplit(n_splits=5, gap=96)  # gap=96 bars = 1 hari buffer

# Metric utama untuk tuning: Brier Score (bukan hanya AUC)
# Brier Score mengukur kalibrasi probabilitas, bukan hanya ranking
# Target: brier_score_val < 0.24

# Calibration check (post-training):
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted = calibration_curve(y_test, P_model_test, n_bins=10)
# Jika curve jauh dari diagonal → model perlu Platt Scaling atau Isotonic Regression
```

### 6.6 Retraining Triggers

```python
RETRAIN_TRIGGERS = {
    "rolling_win_rate"  : {
        "threshold" : 0.50,
        "window"    : "last 30 trades",
        "condition" : "2 consecutive sessions below threshold"
    },
    "brier_score_rolling": {
        "threshold" : 0.245,
        "window"    : "last 50 predictions",
        "condition" : "exceed threshold"
    },
}

# Retrain menggunakan sliding window: 3 bulan terakhir (bukan akumulasi)
# Tujuan: mencegah model terlalu terikat pada kondisi historis jauh

# Manual trigger:
# python main.py --retrain

# Auto trigger: diimplementasi di dry_run.py post-session evaluation
```

---

## 7. Signal Generation Logic

### 7.1 Flow Keputusan Step-by-Step

```
INPUT: FeatureVector + P_model + CLOBState + ActiveMarket metadata

STEP 1: TTR GATE
  Cek TTR_phase == "ENTRY_WINDOW" (5 ≤ TTR_minutes ≤ 12)
  → FAIL: ABSTAIN (reason = "TTR_PHASE")
  → PASS: lanjut ke STEP 2

STEP 2: REGIME FILTER
  Cek REGIME_GO (lihat section 7.3)
  → FAIL: ABSTAIN (reason = "REGIME_BLOCK")
  → PASS: lanjut ke STEP 3

STEP 3: LIQUIDITY FILTER
  Cek CLOB_LIQUID (lihat section 7.4)
  → FAIL: ABSTAIN (reason = "LIQUIDITY_BLOCK")
  → PASS: lanjut ke STEP 4

STEP 4: MISPRICING CALCULATION
  edge_yes = P_model - clob_yes_ask
  edge_no  = (1 - P_model) - clob_no_ask
  (Lihat section 7.5)

STEP 5: MARGIN OF SAFETY CHECK
  Cek apakah max(edge_yes, edge_no) > MARGIN_OF_SAFETY (default: 0.05)
  → FAIL (keduanya ≤ MARGIN_OF_SAFETY): ABSTAIN (reason = "NO_EDGE")
  → PASS: lanjut ke STEP 6

STEP 6: FINAL SIGNAL SELECTION
  Jika edge_yes > MARGIN_OF_SAFETY DAN edge_no > MARGIN_OF_SAFETY:
      → pilih yang terbesar: "BUY_YES" jika edge_yes ≥ edge_no, else "BUY_NO"
  Jika hanya edge_yes > MARGIN_OF_SAFETY:
      → "BUY_YES"
  Jika hanya edge_no > MARGIN_OF_SAFETY:
      → "BUY_NO"

OUTPUT: SignalResult (lihat section 7.6)
```

### 7.2 TTR Gate

```python
# Config-driven threshold (bukan hardcoded)
TTR_MIN_MINUTES = config.get("signal.ttr_min_minutes", 5.0)
TTR_MAX_MINUTES = config.get("signal.ttr_max_minutes", 12.0)

TTR_phase = (
    "ENTRY_WINDOW" if TTR_MIN_MINUTES <= TTR_minutes <= TTR_MAX_MINUTES
    else "EARLY"   if TTR_minutes > TTR_MAX_MINUTES
    else "LATE"
)

TTR_GATE = (TTR_phase == "ENTRY_WINDOW")

# Rationale:
# < 5 menit: spread CLOB melebar, edge terkompres, fill uncertainty tinggi
# > 12 menit: terlalu banyak uncertainty tersisa, probabilitas estimasi kurang akurat
```

### 7.3 Regime Filter (Hard Gate)

```python
VOL_LOWER     = config.get("regime.vol_lower_threshold", 0.15)
VOL_UPPER     = config.get("regime.vol_upper_threshold", 0.80)
SPREAD_MAX_BPS = config.get("regime.binance_spread_max_bps", 5.0)
DEPTH_MIN_BTC  = config.get("regime.binance_min_depth_btc", 0.5)

# Volatility regime check
REGIME_VOL_OK = VOL_LOWER < vol_percentile < VOL_UPPER
# vol_percentile < VOL_LOWER → volatilitas terlalu rendah (illiquid, wide spread)
# vol_percentile > VOL_UPPER → volatilitas ekstrem (model overconfident, avoid)

# Binance data quality checks
BINANCE_SPREAD_OK = binance_spread_bps < SPREAD_MAX_BPS
BINANCE_DEPTH_OK  = binance_top5_bid_btc > DEPTH_MIN_BTC

REGIME_GO = REGIME_VOL_OK AND BINANCE_SPREAD_OK AND BINANCE_DEPTH_OK

# Tuning cadence (data-driven, bukan intuisi):
# Setelah setiap 5 sesi dry run, evaluasi:
#   - signal_count < 5/session → pertimbangkan naikkan VOL_UPPER ke 0.85
#   - win_rate di vol_percentile [0.75-0.80] < 0.50 → turunkan VOL_UPPER ke 0.75
```

### 7.4 Liquidity Filter (Hard Gate)

```python
MIN_CLOB_DEPTH_USD = config.get("clob.min_depth_usd", 10.0)
MAX_MARKET_VIG     = config.get("clob.max_market_vig", 0.07)

CLOB_LIQUID = (
    clob_state.yes_depth_usd >= MIN_CLOB_DEPTH_USD AND
    clob_state.no_depth_usd  >= MIN_CLOB_DEPTH_USD AND
    clob_state.market_vig    <= MAX_MARKET_VIG      AND
    NOT clob_state.is_stale
)
```

### 7.5 Mispricing Calculation

```python
# Core mispricing formulas
edge_yes = P_model - clob_yes_ask
# Interpretasi: jika P_model = 0.62 dan clob_yes_ask = 0.54
#               edge_yes = 0.08 → YES underpriced 8 poin persentase

edge_no  = (1 - P_model) - clob_no_ask
# Interpretasi: jika P_model = 0.38 dan clob_no_ask = 0.44
#               edge_no = 0.18 → NO underpriced 18 poin persentase

# Penting: edge_yes dan edge_no bisa keduanya negatif (tidak ada peluang)
#          atau keduanya positif (pilih yang lebih besar)
```

### 7.6 SignalResult Schema (Pydantic)

```python
class SignalResult(BaseModel):
    signal          : Literal["BUY_YES", "BUY_NO", "ABSTAIN"]
    abstain_reason  : Optional[Literal[
                        "TTR_PHASE", "REGIME_BLOCK",
                        "LIQUIDITY_BLOCK", "NO_EDGE"]]
    P_model         : float
    edge_yes        : float
    edge_no         : float
    clob_yes_ask    : float
    clob_no_ask     : float
    TTR_minutes     : float
    strike_price    : float
    current_price   : float
    strike_distance : float
    market_id       : str
    timestamp       : datetime
    features        : Dict[str, float]  # full feature snapshot untuk logging
```

---

## 8. Risk Management System

### 8.1 Position Sizing — Half-Kelly dengan CLOB Odds

#### 8.1.1 Decimal Odds dari CLOB

```python
# Odds aktual berdasarkan harga CLOB (bukan asumsi 50/50)
b_yes = (1.0 - clob_yes_ask) / clob_yes_ask
# Contoh: clob_yes_ask = 0.54 → b_yes = 0.46/0.54 = 0.852
# Interpretasi: untuk setiap $0.54 diinvestasikan, menang = $0.46 profit

b_no  = (1.0 - clob_no_ask) / clob_no_ask
```

#### 8.1.2 Full Kelly Fraction

```python
full_kelly_yes = (P_model * b_yes - (1 - P_model)) / b_yes
full_kelly_no  = ((1 - P_model) * b_no - P_model) / b_no

# Contoh numerik:
# P_model = 0.62, clob_yes_ask = 0.54, b_yes = 0.852
# full_kelly = (0.62 * 0.852 - 0.38) / 0.852
#            = (0.528 - 0.38) / 0.852 = 0.174 (17.4% of capital)
```

#### 8.1.3 Half-Kelly (Operasional Default)

```python
KELLY_DIVISOR = config.get("risk.kelly_divisor", 2)

half_kelly_yes = max(0.0, full_kelly_yes / KELLY_DIVISOR)
half_kelly_no  = max(0.0, full_kelly_no  / KELLY_DIVISOR)

# Contoh: full_kelly = 0.174 → half_kelly = 0.087 (8.7% of capital)
# Alasan Half-Kelly: safety margin terhadap model estimation error
# Model tidak sempurna → full kelly terlalu agresif
```

#### 8.1.4 Dynamic Kelly Multiplier

```python
consecutive_losses = count_consecutive_losses(recent_trades)
MULTIPLIER_DECAY   = config.get("risk.consecutive_loss_multiplier", 0.15)
KELLY_FLOOR        = config.get("risk.kelly_floor_multiplier", 0.25)

kelly_multiplier = max(KELLY_FLOOR, 1.0 - consecutive_losses * MULTIPLIER_DECAY)
# Tabel:
# 0 losses  → 1.00 × kelly  (100%)
# 1 loss    → 0.85 × kelly  (85%)
# 2 losses  → 0.70 × kelly  (70%)
# 3 losses  → 0.55 × kelly  (55%)
# 4 losses  → 0.40 × kelly  (40%)
# 5+ losses → 0.25 × kelly  (25% floor — tidak pernah di bawah ini)
```

#### 8.1.5 Bet Size Final dengan Constraints

```python
MIN_BET_USD       = config.get("risk.min_bet_usd", 1.00)       # Polymarket minimum
MAX_BET_FRACTION  = config.get("risk.max_bet_fraction", 0.10)   # ceiling 10% capital

kelly_fraction = half_kelly_yes if signal == "BUY_YES" else half_kelly_no
raw_bet        = capital * kelly_fraction * kelly_multiplier
bet_size       = max(raw_bet, MIN_BET_USD)
bet_size       = min(bet_size, capital * MAX_BET_FRACTION)
bet_size       = round(bet_size, 2)  # USDC 2-decimal precision
```

### 8.2 Hard Limits

```python
DAILY_LOSS_LIMIT     = config.get("risk.daily_loss_limit_pct", 0.05)   # 5% capital
SESSION_LOSS_LIMIT   = config.get("risk.session_loss_limit_pct", 0.03) # 3% capital
MAX_OPEN_POSITIONS   = 1   # tidak boleh punya 2 posisi open bersamaan
MIN_CAPITAL_FLOOR    = 5.0  # stop jika capital < $5 (di bawah viable bet size)
```

### 8.3 Trade Approval Logic

```python
class RiskManager:
    def approve(self, signal: SignalResult, capital: float,
                trade_history: List[Trade]) -> Union[ApprovedBet, RejectedBet]:
        
        # Hard limit checks
        if self.daily_pnl < -DAILY_LOSS_LIMIT * capital:
            return RejectedBet(reason="DAILY_LOSS_LIMIT_HIT")
        
        if self.session_pnl < -SESSION_LOSS_LIMIT * capital:
            return RejectedBet(reason="SESSION_LOSS_LIMIT_HIT")
        
        if self.open_positions >= MAX_OPEN_POSITIONS:
            return RejectedBet(reason="MAX_POSITIONS_REACHED")
        
        if capital < MIN_CAPITAL_FLOOR:
            return RejectedBet(reason="CAPITAL_BELOW_FLOOR")
        
        # Bet sizing
        bet_size = self.compute_bet_size(signal, capital, trade_history)
        
        if bet_size < MIN_BET_USD:
            return RejectedBet(reason="BET_BELOW_MINIMUM")
        
        return ApprovedBet(
            signal=signal,
            bet_size=bet_size,
            kelly_fraction=kelly_fraction,
            kelly_multiplier=kelly_multiplier,
        )
```

---

## 9. Execution System

### 9.1 LIVE_MODE Gate

```python
# Default: LIVE_MODE = False (hardcoded dalam kode, bukan hanya config)
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"

# Enable live trading:
# 1. Set environment variable: LIVE_MODE=true
# 2. Tambahkan CLI flag: python main.py --mode live --confirm-live
# 3. Konfirmasi prompt eksplisit:
#    "Anda akan mengeksekusi order NYATA dengan uang sungguhan."
#    "Ketik 'CONFIRM-LIVE-TRADING' untuk melanjutkan: "

# Double-gate requirement: KEDUA kondisi harus terpenuhi:
# (LIVE_MODE env == "true") AND (CLI flag --confirm-live) AND (interactive confirmation)
```

### 9.2 Order Placement Logic

```python
class ExecutionClient:
    def place_order(self, approved_bet: ApprovedBet, active_market: ActiveMarket):
        
        # Pre-order validation
        if active_market.TTR_minutes < 3.0:
            return OrderRejected(reason="TTR_TOO_LOW_FOR_FILL")
        
        # Tentukan token_id berdasarkan signal
        token_id = active_market.clob_token_ids["YES" if signal == "BUY_YES" else "NO"]
        
        # Harga order: ask + slippage buffer
        order_price = clob_ask + 0.002  # 0.2 poin persentase buffer
        
        # Place limit order via py-clob-client
        order = self.clob_client.create_and_post_order(
            token_id    = token_id,
            price       = order_price,
            size        = approved_bet.bet_size,
            side        = "BUY",
            order_type  = "GTC",  # Good Till Cancelled
        )
        return order
```

### 9.3 Fill Monitoring & Timeout

```python
ORDER_POLL_INTERVAL_S = 5
ORDER_TIMEOUT_S       = 60  # cancel jika tidak ter-fill dalam 60 detik

async def monitor_fill(self, order_id: str) -> FillResult:
    start_time = time.time()
    
    while (time.time() - start_time) < ORDER_TIMEOUT_S:
        order_status = await self.clob_client.get_order(order_id)
        
        if order_status.status == "FILLED":
            return FillResult(status="FILLED", fill_price=order_status.avg_price)
        
        elif order_status.status in ["CANCELLED", "REJECTED"]:
            return FillResult(status="FAILED", reason=order_status.status)
        
        await asyncio.sleep(ORDER_POLL_INTERVAL_S)
    
    # Timeout: cancel order
    await self.clob_client.cancel_order(order_id)
    return FillResult(status="TIMEOUT_CANCELLED")
```

### 9.4 Market Rotation Handling

```python
# Saat market lama resolve:
# 1. Execution client mendeteksi T_resolution sudah lewat
# 2. Trigger Market Discovery untuk mencari market baru
# 3. Update active_market secara atomik (tidak boleh ada race condition)
# 4. Reset TTR state, strike_price, CLOB token_ids
# 5. TIDAK boleh open posisi baru sampai active_market sudah diperbarui

# Diagram state:
# [Market A ACTIVE] → T_resolution reached →
# [MARKET_RESOLVING] → Discovery finds Market B →
# [Market B ACTIVE]

# Gap handling: jika Discovery tidak menemukan market B dalam 5 menit:
# → Log WARNING: "No new market found after resolution"
# → Masuk WAITING mode, poll setiap 60 detik
```

### 9.5 Post-Resolution: Claim Winnings

```python
async def claim_winnings(self, trade: Trade, active_market: ActiveMarket):
    """
    Setelah T_resolution, claim payout jika trade menang.
    Dipanggil otomatis oleh dry_run.py dan execution.py.
    """
    # Ambil BTC price at resolution dari Binance
    btc_at_resolution = await self.get_btc_price_at_resolution(
        active_market.T_resolution
    )
    
    # Tentukan outcome
    outcome = "WIN" if (
        trade.signal_type == "BUY_YES" and btc_at_resolution > active_market.strike_price
        or
        trade.signal_type == "BUY_NO"  and btc_at_resolution <= active_market.strike_price
    ) else "LOSS"
    
    if outcome == "WIN" and LIVE_MODE:
        await self.clob_client.redeem_positions(trade.market_id)
    
    return TradeOutcome(
        trade_id         = trade.trade_id,
        outcome          = outcome,
        btc_at_resolution = btc_at_resolution,
        pnl_usd          = self.compute_pnl(trade, outcome),
    )
```

---

## 10. Dry Run & Validation Framework

### 10.1 Definisi Sesi Dry Run

```
Mode            : Data real-time Binance + CLOB Polymarket, TANPA uang nyata
Durasi per sesi : 5–8 jam (sesuai ketersediaan operasional)
Frekuensi       : Setiap hari minimum selama 2 minggu
Modal simulasi  : $100.00 USDC (virtual, tidak masuk Polymarket)
Go-live review  : Setelah minimum 10 sesi terkumpul
```

### 10.2 Paper Trade Mechanics

```python
def simulate_trade(self, signal: SignalResult, approved_bet: ApprovedBet,
                   active_market: ActiveMarket) -> PaperTrade:
    
    entry_price = (signal.clob_yes_ask if signal.signal == "BUY_YES"
                   else signal.clob_no_ask)
    
    # Catat entry
    paper_trade = PaperTrade(
        market_id       = active_market.market_id,
        signal_type     = signal.signal,
        entry_price     = entry_price,       # worst-case fill (ask price)
        bet_size        = approved_bet.bet_size,
        strike_price    = active_market.strike_price,
        T_resolution    = active_market.T_resolution,
        TTR_at_entry    = signal.TTR_minutes,
    )
    
    # Tunggu T_resolution, ambil actual BTC price
    # [async: await T_resolution]
    btc_at_res = get_binance_price_at(active_market.T_resolution)
    
    # Compute outcome
    won = (signal.signal == "BUY_YES" and btc_at_res > active_market.strike_price) or \
          (signal.signal == "BUY_NO"  and btc_at_res <= active_market.strike_price)
    
    pnl = approved_bet.bet_size * ((1.0 / entry_price) - 1) if won else -approved_bet.bet_size
    
    return paper_trade.update(outcome="WIN" if won else "LOSS", pnl_usd=pnl)
```

### 10.3 PASS / FAIL Criteria

```
HARD GATES (semua harus terpenuhi — satu gagal = sesi FAIL otomatis):

  1. min_trades_executed   >= 10 per sesi
  2. win_rate_rolling_50   >= 0.53
  3. max_drawdown_session  > -0.15   (drawdown tidak melebihi -15%)
  4. profit_factor         >= 1.10

SOFT SCORE KOMPOSIT       >= 0.70

SYARAT DATA QUALITY (non-negotiable):
  - ws_message_drop_rate   < 0.5%    (data quality gate)
  - clob_stale_events      == 0      (CLOB data harus fresh seluruh sesi)
```

### 10.4 Dry Run Score Komposit

```python
def compute_dry_run_score(metrics: SessionMetrics) -> float:
    def normalize(x, lower, upper):
        return max(0.0, min(1.0, (x - lower) / (upper - lower)))
    
    score = (
        0.35 * normalize(metrics.win_rate_rolling,  lower=0.50, upper=0.62) +
        0.25 * normalize(metrics.expectancy,         lower=-0.01, upper=0.05) +
        0.20 * normalize(metrics.sharpe_rolling,     lower=0.0,  upper=2.0)  +
        0.20 * normalize(1.0 + metrics.max_drawdown, lower=0.80, upper=1.0)
    )
    
    return score  # PASS if >= 0.70
```

### 10.5 Go-Live Decision Criteria

```
BOLEH go-live jika SEMUA kondisi terpenuhi:
  1. 5 sesi PASS berturut-turut, ATAU 8 dari 10 sesi terakhir PASS
  2. Total trades dry run >= 100
  3. Tidak ada sesi dengan drawdown > -20% dalam seluruh history
  4. brier_score_cumulative < 0.24 (model masih well-calibrated)
  5. mean_edge_traded > 0.04 (rata-rata edge saat entry > 4 poin persentase)

KONFIRMASI MANUAL WAJIB sebelum go-live:
  - Review performance.json dari semua sesi (manual)
  - Konfirmasi market_id yang akan digunakan sudah tervalidasi
  - Konfirmasi LIVE_MODE = true via CLI explicit confirmation
  - Verifikasi saldo USDC di Polymarket wallet mencukupi
```

### 10.6 Abort Conditions

```
STOP DRY RUN — evaluasi ulang sebelum melanjutkan:

  A. 3 sesi FAIL berturut-turut
     → Action: review CLOB integration + model calibration + regime thresholds

  B. win_rate_cumulative < 0.48 setelah 50 total trades
     → Action: STOP, trigger retrain model

  C. consecutive_losses >= 6 dalam satu sesi
     → Action: PAUSE sesi, investigasi kondisi pasar dan CLOB liquidity

  D. ws_message_drop_rate > 0.5% per sesi
     → Action: PAUSE, investigasi koneksi (Railway network / Binance WS)

  E. clob_stale_events > 2 per sesi
     → Action: PAUSE, investigasi CLOB feed polling logic
```

---

## 11. Performance Metrics & Evaluation

### 11.1 Sharpe Ratio (Per-Trade Basis)

```python
# Basis: per-trade returns, bukan per-hari (karena frekuensi rendah per hari)
per_trade_returns = [pnl_i / bet_size_i for trade_i in session_trades]
N_annual          = 96 * 252 * avg_signal_rate  # estimasi trades per tahun

sharpe_rolling = (
    mean(per_trade_returns) /
    (std(per_trade_returns) + 1e-8)
) * sqrt(N_annual)

# Rolling window: 30 trades terakhir
# Target: sharpe_rolling > 1.0
```

### 11.2 Sortino Ratio

```python
downside_returns = [r for r in per_trade_returns if r < 0]
sortino = (
    mean(per_trade_returns) /
    (std(downside_returns) + 1e-8)
) * sqrt(N_annual)

# Target: sortino > 1.5
```

### 11.3 Maximum Drawdown

```python
equity_curve = np.cumsum(pnl_sequence) + capital_start
peak         = np.maximum.accumulate(equity_curve)
drawdown     = (equity_curve - peak) / peak
max_dd       = np.min(drawdown)  # negatif nilai → misal -0.12 = -12%

# Target: max_dd > -0.20 (tidak melebihi -20%)
# Hard gate: max_dd > -0.15 untuk PASS (per sesi)
```

### 11.4 Calmar Ratio

```python
# Annualized return / |max drawdown|
calmar = annualized_return / abs(max_dd)
# Target: calmar > 0.5
```

### 11.5 Win Rate & Profit Factor

```python
# Rolling win rate (50-trade window)
win_rate_rolling = rolling_sum(is_win, window=50) / 50
# Target: >= 0.53

# Profit factor
gross_profit = sum(pnl for pnl in pnl_list if pnl > 0)
gross_loss   = abs(sum(pnl for pnl in pnl_list if pnl < 0))
profit_factor = gross_profit / (gross_loss + 1e-8)
# Target: >= 1.10
```

### 11.6 Brier Score (Model Calibration)

```python
brier = mean((P_model_i - actual_outcome_i) ** 2 for all predictions)
# actual_outcome_i: 1 jika YES menang, 0 jika NO menang
# Range: [0, 1] — random model = 0.25, perfect = 0.0
# Target: brier < 0.24 (lebih baik dari random)

# Interpretasi:
# brier = 0.20 → model cukup well-calibrated
# brier = 0.24 → mendekati random — trigger retrain
# brier > 0.24 → model degradasi — wajib retrain sebelum lanjut
```

### 11.7 Edge Accuracy

```python
# Korelasi antara edge prediksi dan outcome aktual
edge_accuracy = np.corrcoef(
    [edge_yes_i for trade in BUY_YES_trades],
    [actual_outcome_i for trade in BUY_YES_trades]
)[0, 1]
# Positif → edge prediksi ber-korelasi dengan kemenangan (edge memang ada)
# ≈ 0     → tidak ada korelasi (edge tidak terbukti)
```

### 11.8 Rangkuman Metric Targets

| Metric | Target | Hard Gate (Dry Run PASS) |
|---|---|---|
| Win Rate (rolling 50) | ≥ 0.55 | ≥ 0.53 |
| Profit Factor | ≥ 1.20 | ≥ 1.10 |
| Sharpe Ratio | > 1.0 | - |
| Sortino Ratio | > 1.5 | - |
| Max Drawdown (session) | > -0.10 | > -0.15 |
| Brier Score | < 0.22 | < 0.24 (go-live) |
| Dry Run Score | - | ≥ 0.70 |
| Mean Edge at Entry | > 0.06 | > 0.04 (go-live) |
| WS Drop Rate | < 0.1% | < 0.5% |
| CLOB Stale Events | 0 | 0 |

---

## 12. Database Design

### 12.1 Schema Overview

```
Engine:
  Local   : SQLite (aiosqlite driver, file: /app/data/trading.db)
  Railway : PostgreSQL (asyncpg driver, via DATABASE_URL env var)
  ORM     : SQLAlchemy 2.x (async-compatible)
  Switch  : otomatis berdasarkan DATABASE_URL — jika set → PostgreSQL
```

### 12.2 Tabel `markets`

```sql
CREATE TABLE markets (
    id              TEXT PRIMARY KEY,       -- condition_id Polymarket
    question        TEXT NOT NULL,
    strike_price    REAL NOT NULL,
    t_open          TIMESTAMP NOT NULL,
    t_resolution    TIMESTAMP NOT NULL,
    clob_token_yes  TEXT,
    clob_token_no   TEXT,
    status          TEXT DEFAULT 'DISCOVERED',  -- DISCOVERED | ACTIVE | RESOLVED
    btc_at_resolution REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_markets_t_resolution ON markets(t_resolution);
CREATE INDEX idx_markets_status       ON markets(status);
```

### 12.3 Tabel `signals`

```sql
CREATE TABLE signals (
    signal_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL,
    market_id       TEXT REFERENCES markets(id),
    timestamp_utc   TIMESTAMP NOT NULL,
    signal_type     TEXT NOT NULL,        -- BUY_YES | BUY_NO | ABSTAIN
    abstain_reason  TEXT,                 -- REGIME_BLOCK | LIQUIDITY_BLOCK | TTR_PHASE | NO_EDGE
    p_model         REAL,
    edge_yes        REAL,
    edge_no         REAL,
    clob_yes_ask    REAL,
    clob_no_ask     REAL,
    ttr_minutes     REAL,
    strike_distance REAL,
    vol_percentile  REAL,
    obi_value       REAL,
    tfm_norm        REAL,
    market_vig      REAL,
    model_version   TEXT,
    mode            TEXT NOT NULL         -- DRY | LIVE
);

CREATE INDEX idx_signals_timestamp  ON signals(timestamp_utc, market_id);
CREATE INDEX idx_signals_session    ON signals(session_id);
CREATE INDEX idx_signals_signal_type ON signals(signal_type);
```

### 12.4 Tabel `trades`

```sql
CREATE TABLE trades (
    trade_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          TEXT NOT NULL,
    signal_id           UUID REFERENCES signals(signal_id),
    market_id           TEXT REFERENCES markets(id),
    timestamp_signal    TIMESTAMP NOT NULL,
    timestamp_entry     TIMESTAMP NOT NULL,
    timestamp_resolution TIMESTAMP,
    signal_type         TEXT NOT NULL,    -- BUY_YES | BUY_NO
    p_model             REAL NOT NULL,
    clob_yes_ask        REAL,
    clob_no_ask         REAL,
    edge_yes            REAL,
    edge_no             REAL,
    entry_price         REAL NOT NULL,
    bet_size_usd        REAL NOT NULL,
    kelly_fraction      REAL,
    kelly_multiplier    REAL,
    strike_price        REAL NOT NULL,
    btc_at_signal       REAL,
    strike_distance_pct REAL,
    ttr_minutes         REAL,
    btc_at_resolution   REAL,
    outcome             TEXT,             -- WIN | LOSS | PENDING
    pnl_usd             REAL,
    pnl_pct_capital     REAL,
    capital_before      REAL,
    capital_after       REAL,
    vol_percentile      REAL,
    obi_at_signal       REAL,
    tfm_norm_at_signal  REAL,
    market_vig          REAL,
    model_version       TEXT,
    mode                TEXT NOT NULL     -- DRY | LIVE
);

CREATE INDEX idx_trades_session       ON trades(session_id, timestamp_signal);
CREATE INDEX idx_trades_market        ON trades(market_id);
CREATE INDEX idx_trades_outcome       ON trades(outcome);
CREATE INDEX idx_trades_mode_session  ON trades(mode, session_id);
```

### 12.5 Tabel `performance`

```sql
CREATE TABLE performance (
    session_id              TEXT PRIMARY KEY,
    date_utc                DATE NOT NULL,
    start_time              TIMESTAMP,
    end_time                TIMESTAMP,
    duration_hours          REAL,
    mode                    TEXT NOT NULL,
    total_bars_processed    INTEGER DEFAULT 0,
    total_signals_evaluated INTEGER DEFAULT 0,
    signals_abstained       INTEGER DEFAULT 0,
    abstain_regime          INTEGER DEFAULT 0,
    abstain_liquidity       INTEGER DEFAULT 0,
    abstain_ttr             INTEGER DEFAULT 0,
    abstain_no_edge         INTEGER DEFAULT 0,
    trades_executed         INTEGER DEFAULT 0,
    win_count               INTEGER DEFAULT 0,
    loss_count              INTEGER DEFAULT 0,
    win_rate                REAL,
    profit_factor           REAL,
    total_pnl_usd           REAL,
    total_pnl_pct_capital   REAL,
    max_drawdown            REAL,
    sharpe_rolling          REAL,
    sortino_rolling         REAL,
    brier_score             REAL,
    mean_edge_yes_traded    REAL,
    mean_edge_no_traded     REAL,
    mean_ttr_at_signal      REAL,
    mean_strike_distance    REAL,
    capital_start           REAL,
    capital_end             REAL,
    dry_run_score           REAL,
    pass_fail               TEXT,         -- PASS | FAIL
    model_version           TEXT,
    ws_drop_rate_pct        REAL,
    clob_stale_events       INTEGER DEFAULT 0,
    notes                   TEXT
);

CREATE INDEX idx_perf_date    ON performance(date_utc);
CREATE INDEX idx_perf_passfail ON performance(pass_fail);
```

### 12.6 Tabel `model_versions`

```sql
CREATE TABLE model_versions (
    version_id       TEXT PRIMARY KEY,    -- e.g., "model_v20250115_143022"
    created_at       TIMESTAMP NOT NULL,
    brier_score_oos  REAL,
    auc_oos          REAL,
    win_rate_oos     REAL,
    training_samples INTEGER,
    retrain_trigger  TEXT,                -- MANUAL | AUTO_WIN_RATE | AUTO_BRIER
    file_path_lgbm   TEXT,
    file_path_logreg TEXT,
    file_path_scaler TEXT,
    is_active        BOOLEAN DEFAULT FALSE
);
```

### 12.7 Tabel `system_health`

```sql
CREATE TABLE system_health (
    id                  SERIAL PRIMARY KEY,
    session_id          TEXT,
    timestamp_utc       TIMESTAMP NOT NULL,
    ws_drop_rate        REAL,
    ws_latency_p99_ms   REAL,
    clob_fetch_latency_ms REAL,
    clob_is_stale       BOOLEAN,
    queue_depth         INTEGER,
    market_id_active    TEXT,
    notes               TEXT
);

CREATE INDEX idx_health_session ON system_health(session_id, timestamp_utc);
```

---

## 13. Observability & Monitoring

### 13.1 WebSocket Health Metrics

```python
# Dikumpulkan per sesi oleh BinanceFeed
class WSHealthMetrics:
    messages_received       : int
    messages_expected       : int   # estimated via sequence number
    messages_dropped        : int
    drop_rate               : float  # = dropped / expected
    reconnect_count         : int
    latency_p50_ms          : float
    latency_p99_ms          : float
    queue_depth_max         : int
    last_message_timestamp  : datetime

ALERT_THRESHOLDS = {
    "drop_rate"      : 0.001,  # > 0.1% → log WARNING
    "latency_p99_ms" : 2000,   # > 2s → log WARNING
    "reconnect_count": 3,      # > 3 per sesi → log ERROR
    "queue_depth"    : 100,    # > 100 messages pending → log WARNING
}
```

### 13.2 Data Latency Metrics

```python
# Feature compute lag (bar_close_time → features_ready)
feature_compute_lag_ms = time_features_ready - bar_close_time
# Target: < 500ms

# CLOB fetch latency
clob_fetch_latency_ms = time_clob_fetched - time_request_sent
# Target: < 2000ms

# End-to-end signal latency
e2e_signal_latency_ms = time_signal_emitted - bar_close_time
# Target: < 500ms
```

### 13.3 System Alerts

| Alert | Threshold | Level | Action |
|---|---|---|---|
| WS drop rate tinggi | > 0.1% | WARNING | Log, monitor |
| WS drop rate kritis | > 0.5% | ERROR | PAUSE trading, investigasi |
| WS latency tinggi | > 2000ms | WARNING | Log, monitor |
| WS reconnect berulang | > 3 per sesi | ERROR | Log, investigate koneksi |
| CLOB stale | > 30s tanpa update | ERROR | Block signals, log |
| Feature lag | > 500ms | WARNING | Log |
| Daily loss limit | pnl < -5% capital | CRITICAL | STOP all trading |
| Session loss limit | pnl < -3% capital | ERROR | PAUSE 30 menit |
| Brier score degradasi | > 0.245 | WARNING | Schedule retrain |
| Win rate turun | < 0.50 (2 sesi) | ERROR | Auto-trigger retrain |

### 13.4 Logging Strategy

```python
# Framework: structlog (JSON format untuk Railway)
import structlog

logger = structlog.get_logger()

# Log levels dan konteks:

# DEBUG: feature values per bar (hanya di development)
logger.debug("features_computed", bar_time=t, OBI=obi, TFM=tfm, ...)

# INFO: setiap signal (termasuk ABSTAIN)
logger.info("signal_generated",
    signal=signal.signal,
    abstain_reason=signal.abstain_reason,
    P_model=signal.P_model,
    edge_yes=signal.edge_yes,
    TTR_minutes=signal.TTR_minutes,
    market_id=signal.market_id,
)

# INFO: setiap trade
logger.info("trade_executed",
    trade_id=str(trade.trade_id),
    signal_type=trade.signal_type,
    bet_size=trade.bet_size_usd,
    entry_price=trade.entry_price,
    kelly_fraction=trade.kelly_fraction,
)

# WARNING: setiap kondisi yang memerlukan monitoring
logger.warning("ws_drop_rate_alert",
    drop_rate=metrics.drop_rate,
    threshold=0.001,
)

# ERROR: kondisi yang memblokir trading
logger.error("clob_stale",
    last_update=clob_state.timestamp,
    staleness_seconds=staleness,
)

# CRITICAL: kondisi yang memerlukan intervensi segera
logger.critical("daily_loss_limit_hit",
    daily_pnl_pct=daily_pnl_pct,
    limit=-0.05,
)

# Log rotation: per sesi, simpan 30 sesi terakhir di /app/data/logs/
```

### 13.5 CLI Dashboard Real-Time

```
Framework: Rich (Python)
Refresh rate: setiap 5 detik

┌─────────────── MARKET STATUS ──────────────────┐
│ Market ID: 0xabc123...                          │
│ TTR: 8.4 menit | Phase: ■ ENTRY_WINDOW          │
│ Strike: $98,450.00 | Current: $98,623.00        │
│ Strike Distance: +0.18% (Contest Zone)          │
└─────────────────────────────────────────────────┘
┌────────────────── CLOB ─────────────────────────┐
│ YES Ask: 0.542 | YES Bid: 0.528                 │
│ NO  Ask: 0.470 | NO  Bid: 0.455                 │
│ Market Vig: 0.012 | Status: ✓ LIQUID            │
└─────────────────────────────────────────────────┘
┌───────────────── MODEL ─────────────────────────┐
│ P_model: 0.614                                  │
│ edge_YES: +0.072 ■ SIGNAL | edge_NO: -0.074     │
│ Signal: → BUY_YES                               │
└─────────────────────────────────────────────────┘
┌──────────────── SESSION P&L ────────────────────┐
│ Session PnL: +$2.87 (+2.87%) | Capital: $102.87 │
│ Win Rate (50): 62.5% | Trades: 8 (W:5, L:3)     │
│ Dry Run Score: 0.76 | Status: ✓ PASS (cumul.)   │
└─────────────────────────────────────────────────┘
┌─────────────── SYSTEM HEALTH ───────────────────┐
│ WS Drop Rate: 0.004% ✓ | Latency P99: 124ms ✓  │
│ CLOB Feed: ✓ FRESH (1.2s ago)                   │
│ Mode: DRY RUN | Session: 2025-01-15_001         │
└─────────────────────────────────────────────────┘
```

---

## 14. Configuration Management

### 14.1 `config.json` — Semua Parameter Tunable

```json
{
  "_comment": "Edit via CLI atau langsung. Tidak perlu restart container.",

  "regime": {
    "vol_lower_threshold"   : 0.15,
    "vol_upper_threshold"   : 0.80,
    "binance_spread_max_bps": 5.0,
    "binance_min_depth_btc" : 0.5
  },

  "signal": {
    "margin_of_safety" : 0.05,
    "ttr_min_minutes"  : 5.0,
    "ttr_max_minutes"  : 12.0
  },

  "clob": {
    "min_depth_usd"          : 10.0,
    "max_market_vig"         : 0.07,
    "stale_threshold_seconds": 30,
    "poll_interval_seconds"  : 5
  },

  "risk": {
    "kelly_divisor"              : 2,
    "max_bet_fraction"           : 0.10,
    "min_bet_usd"                : 1.00,
    "daily_loss_limit_pct"       : 0.05,
    "session_loss_limit_pct"     : 0.03,
    "consecutive_loss_multiplier": 0.15,
    "kelly_floor_multiplier"     : 0.25
  },

  "dry_run": {
    "pass_win_rate"            : 0.53,
    "pass_profit_factor"       : 1.10,
    "pass_dry_run_score"       : 0.70,
    "pass_max_drawdown"        : -0.15,
    "min_trades_per_session"   : 10,
    "go_live_consecutive_pass" : 5,
    "go_live_min_total_trades" : 100,
    "abort_consecutive_fail"   : 3,
    "abort_win_rate_threshold" : 0.48,
    "abort_consecutive_losses" : 6
  },

  "model": {
    "retrain_win_rate_trigger"       : 0.50,
    "retrain_brier_trigger"          : 0.245,
    "retrain_consecutive_sessions"   : 2,
    "ensemble_lgbm_weight"           : 0.70,
    "ensemble_logreg_weight"         : 0.30
  },

  "observability": {
    "ws_drop_rate_warning_threshold" : 0.001,
    "ws_drop_rate_error_threshold"   : 0.005,
    "ws_latency_warning_ms"          : 2000,
    "clob_poll_interval_seconds"     : 5,
    "feature_compute_lag_warning_ms" : 500
  }
}
```

### 14.2 Update Config Tanpa Redeploy

```python
# Config di-load pada startup DAN di-watch untuk perubahan
class ConfigManager:
    def __init__(self, path: str = "config/config.json"):
        self.path = path
        self._config = self._load()
        self._watch_thread = self._start_file_watcher()
    
    def get(self, key: str, default=None):
        """Key format: 'regime.vol_upper_threshold'"""
        parts = key.split(".")
        val = self._config
        for p in parts:
            val = val.get(p, {})
        return val if val != {} else default
    
    def _load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)
    
    def _start_file_watcher(self):
        """Watch config file untuk perubahan, reload otomatis"""
        # Menggunakan watchdog library atau polling setiap 30 detik
        pass

# CLI update:
# python main.py --config set regime.vol_upper_threshold 0.75
# python main.py --config get signal.margin_of_safety
# python main.py --config show   # tampilkan semua parameter aktif
```

### 14.3 Dependency Antar Config

| Parameter | Dependent Modules | Perubahan Efek |
|---|---|---|
| `regime.vol_upper_threshold` | signal_generator (regime gate) | Immediate, next bar |
| `signal.margin_of_safety` | signal_generator (edge threshold) | Immediate, next signal |
| `clob.min_depth_usd` | signal_generator (liquidity gate) | Immediate |
| `risk.max_bet_fraction` | risk_manager | Immediate, next trade |
| `risk.kelly_divisor` | risk_manager | Immediate, next trade |
| `model.*` | dry_run post-session evaluation | Post-session |

### 14.4 Environment Variables (.env — Secret Management)

```bash
# .env — TIDAK pernah di-commit ke git
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx

POLYMARKET_API_KEY=xxx
POLYMARKET_API_SECRET=xxx
POLYMARKET_PRIVATE_KEY=xxx   # untuk EIP-712 order signing

DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
             # Jika tidak di-set → SQLite digunakan

LIVE_MODE=false              # default false — hanya true setelah manual konfirmasi
LOG_LEVEL=INFO               # DEBUG | INFO | WARNING | ERROR
ENVIRONMENT=development      # development | production
```

---

## 15. Deployment Architecture

### 15.1 Local Development

```
Environment : Python 3.11 + virtualenv
Storage     : ./data/ (OHLCV, orderbook snapshots)
              ./models/ (model artifacts)
              ./exports/ (session exports)
Database    : SQLite (auto-created di ./data/trading.db)
Run command : python main.py --mode dry-run

Hardware minimum:
  RAM     : 4 GB (8 GB direkomendasikan)
  Storage : 5 GB free
  CPU     : modern (LightGBM training: 5–15 menit)
  OS      : Linux / macOS / Windows dengan WSL2
```

### 15.2 Railway Deployment

```
Platform    : Railway (https://railway.app)
Builder     : Dockerfile
Restart     : ON_FAILURE, max 3 retries

Plan        : Hobby ($5/bulan) untuk dry run
              Pro  ($20/bulan) jika RAM > 512 MB
RAM         : 512 MB minimum, 1 GB direkomendasikan
Estimasi    : $8–15/bulan total (termasuk Volume + PostgreSQL)

KRITIS — Persistent Volume:
  - Railway TIDAK persist default filesystem
  - Semua data WAJIB di /app/data (Volume yang di-mount)
  - Setup Volume via Railway dashboard SEBELUM deploy pertama
  - Mount path: /app/data

PostgreSQL  : Railway add-on PostgreSQL
  - DATABASE_URL di-inject otomatis sebagai environment variable
  - Backup: Railway auto-backup (enable di dashboard)
```

### 15.3 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LIVE_MODE=false

# Create required directories
RUN mkdir -p /app/data/models /app/data/exports /app/data/raw /app/data/logs

# Default command: dry run mode
CMD ["python", "main.py", "--mode", "dry-run"]
```

### 15.4 railway.toml

```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "python main.py --mode dry-run"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[[volumes]]
mountPath = "/app/data"

[environments.production]
startCommand = "python main.py --mode live --confirm-live"
```

### 15.5 Scaling Constraints

```
Current architecture: single-process, single-market
  - Tidak di-design untuk horizontal scaling
  - Satu bot per Railway service

Limitasi yang diketahui:
  - Railway shared CPU → LightGBM training lambat (lakukan offline, upload artifact)
  - Railway restart → cold start ~30 detik (data collection gap)
  - asyncio GIL: acceptable untuk I/O-bound (WS + HTTP polling)
    Jika ws_drop_rate > 0.1% secara konsisten → pertimbangkan migrasi ke Go/Rust
    untuk WS layer (tidak dalam scope MVP)

REQUIRES VALIDATION:
  - Cek Railway network latency ke Binance WS (geographically variable)
  - Cek Railway outbound IP untuk Binance API whitelist
```

### 15.6 Storage Strategy

```
/app/data/                    (Railway Volume — persistent)
├── trading.db                # SQLite (atau DATABASE_URL → PostgreSQL)
├── models/
│   ├── model_lgbm_v*.pkl     # LightGBM artifacts (simpan 3 versi)
│   ├── model_logreg_v*.pkl
│   ├── scaler_v*.pkl
│   └── training_metrics_*.json
├── raw/
│   ├── ohlcv_15m.parquet     # Binance OHLCV
│   ├── orderbook_snapshots/  # Orderbook collector output
│   └── aggTrades/            # AggTrade data untuk TFM reconstruction
├── exports/
│   └── YYYY-MM-DD_session_id/
│       ├── trades.csv
│       ├── signals_all.csv
│       ├── performance.json
│       ├── features_log.csv
│       ├── clob_log.csv
│       ├── equity_curve.csv
│       └── ws_health.csv
└── logs/
    └── session_*.log         # structured JSON logs per sesi
```

---

## 16. Security Considerations

### 16.1 API Key Management

```
PRINSIP: Zero secrets in code or config files yang di-commit ke git.

Rules:
  1. Semua credentials di .env (ada di .gitignore)
  2. .env.example berisi template tanpa nilai sensitif
  3. Di Railway: inject sebagai Environment Variables via dashboard
  4. Binance API Key: READ-ONLY permission (tidak perlu trading permission)
  5. Polymarket PRIVATE_KEY: hanya digunakan untuk EIP-712 order signing
     Simpan dengan enkripsi tambahan jika memungkinkan

Code enforcement:
  # TIDAK PERNAH hardcode API keys dalam kode
  API_KEY = os.getenv("BINANCE_API_KEY")  # ✓ benar
  API_KEY = "xxxxxapikey"                 # ✗ SALAH — reject via code review
```

### 16.2 Credential Isolation

```
File isolation:
  .gitignore WAJIB include:
    .env
    *.env.*
    !.env.example  # template boleh di-commit

Git history:
  Jika credentials pernah ter-commit → rotate semua keys segera
  Gunakan: git filter-branch atau BFG Repo Cleaner untuk hapus dari history

Runtime:
  Log statements TIDAK boleh print credentials (audit secara berkala)
  structlog context tidak boleh include API_KEY atau PRIVATE_KEY
```

### 16.3 Attack Surface Analysis

#### CLOB Manipulation Risk

```
Risiko: Pihak ketiga memanipulasi harga CLOB Polymarket untuk
        memancing bot entry di posisi yang merugikan

Mitigasi:
  1. Minimum depth requirement ($10 minimum) mencegah thin-book manipulation
  2. Market vig cap (7%) sebagai proxy efisiensi — vig tinggi abnormal → skip
  3. Margin of safety (5%) memberikan buffer terhadap manipulasi kecil
  4. Bot tidak pernah menjadi "price maker" — hanya taker

Monitoring:
  - Log setiap sinyal dengan CLOB snapshot lengkap
  - Analisis post-sesi: apakah CLOB prices sebelum entry berkorelasi dengan losses?
```

#### Data Poisoning Risk

```
Risiko: Data yang diterima dari Binance WS atau REST dimanipulasi
        (sangat unlikely untuk exchange tier-1 seperti Binance)

Mitigasi:
  1. Validasi OHLC consistency pada setiap bar
  2. Reject bar dengan volume = 0
  3. Sanity check: harga BTC tidak berubah > 10% dalam satu bar 15-menit
     (jika ya → treat sebagai data anomali, block signal)
  4. Cross-reference: jika Binance WS dan REST REST sangat berbeda → alert

REQUIRES VALIDATION:
  - Apakah py-clob-client melakukan TLS verification? (seharusnya ya)
  - Apakah ada signature verification untuk Polymarket data?
```

#### Execution Risk

```
Risiko: Order ditempatkan di market yang salah atau dengan size yang salah

Mitigasi:
  1. Pre-order validation: verifikasi market_id masih aktif
  2. Log setiap order dengan full context sebelum submit
  3. MAX_BET_FRACTION = 10% sebagai circuit breaker
  4. LIVE_MODE double-gate mencegah accidental live trading
  5. Dry run mode tidak pernah memanggil CLOB order endpoint
```

---

## 17. Failure Modes & Edge Cases

### 17.1 API Failure Handling

| Failure | Detection | Response | Recovery |
|---|---|---|---|
| Binance WS disconnect | WS close event / ping timeout | Reconnect dengan exponential backoff | Resume setelah connect, flag gap |
| Binance WS STALE > 30s | Timestamp gap detection | Block signals, log ERROR | Auto-resume saat data fresh |
| Binance REST timeout | httpx timeout exception | Retry 3x, fallback ke cached buffer | Use buffer jika < 500 bars hilang |
| CLOB REST timeout | httpx timeout exception | Retry 3x, gunakan cache | Flag stale, block jika > 30s |
| Polymarket API down | Connection refused / 5xx | WAITING mode | Poll setiap 60 detik |

### 17.2 Data Stale Conditions

```python
class DataStalenessChecker:
    def check_all(self) -> StalenessReport:
        binance_stale = (datetime.utcnow() - self.last_bar_time).seconds > 30
        clob_stale    = self.clob_state.is_stale
        market_stale  = (self.active_market is None or
                         self.active_market.TTR_minutes < 0)
        
        any_stale = binance_stale or clob_stale or market_stale
        
        if any_stale:
            # Block ALL signal generation
            # Log with specific stale source
            pass
        
        return StalenessReport(
            binance_stale = binance_stale,
            clob_stale    = clob_stale,
            market_stale  = market_stale,
        )
```

### 17.3 Market Tidak Tersedia

```
Skenario yang harus dihandle:
  A. Tidak ada market aktif ditemukan
     → WAITING mode, poll setiap 60 detik
     → Log INFO setiap 5 menit (tidak spam)

  B. Market aktif tapi TTR < 5 menit saat discovery
     → Skip market ini, tunggu market berikutnya
     → TTR_phase = "LATE", tidak ada entry

  C. Market dalam maintenance / closed
     → Sama dengan A (WAITING mode)

  D. Market ID berubah (rotasi normal)
     → Hot-reload: update active_market, reset state
     → TIDAK perlu restart bot

  E. Market ditemukan tapi CLOB tidak liquid
     → Enter WAITING mode untuk CLOB (jangan tunggu TTR habis)
     → Re-check CLOB setiap 30 detik

Test case WAJIB (dalam test suite):
  - test_market_not_found() → bot masuk WAITING, tidak crash
  - test_market_rotate()    → bot hot-reload market baru
  - test_all_markets_late() → semua TTR < 5 menit, semua ABSTAIN
```

### 17.4 Liquidity Collapse

```
Definisi: depth_yes < MIN_CLOB_DEPTH_USD atau depth_no < MIN_CLOB_DEPTH_USD
          atau market_vig > MAX_MARKET_VIG

Response:
  1. Semua signal → ABSTAIN (LIQUIDITY_BLOCK)
  2. Log WARNING dengan CLOB snapshot
  3. Jangan re-enter sampai kondisi liquid kembali

Monitoring:
  - clob_log.csv mencatat setiap snapshot (termasuk yang illiquid)
  - Analisis post-sesi: berapa % waktu market illiquid?
  - Jika illiquid > 30% waktu → pertimbangkan revisi MIN_CLOB_DEPTH_USD
```

### 17.5 Model Degradation

```
Deteksi:
  - brier_score_rolling > 0.245 (auto-trigger retrain)
  - win_rate_rolling < 0.50 selama 2 sesi berturut-turut (auto-trigger retrain)

Response gradual (urutan eskalasi):
  Level 1 (soft warning): log WARNING, reduce bet size via kelly_multiplier
  Level 2 (hard warning): brier > 0.245 → schedule retrain di akhir sesi
  Level 3 (critical): 2 sesi consecutive win_rate < 0.50 → auto-retrain immediately

Rollback procedure:
  1. Stop current model inference
  2. Load model versi sebelumnya (simpan 3 versi)
  3. Evaluate rollback model pada data terbaru (brier score check)
  4. Jika rollback model juga degraded → STOP trading, investigasi

Distribution shift detection:
  REQUIRES VALIDATION: implementasikan Population Stability Index (PSI)
  untuk deteksi feature distribution drift antara training dan inference
```

### 17.6 Concurrent Position Bug (Race Condition)

```
Risiko: Dua signal di-process bersamaan, dua posisi ter-open

Prevention:
  - MAX_OPEN_POSITIONS = 1 di risk_manager
  - asyncio.Lock() pada position tracking state
  - Atomic check-and-set untuk open_positions counter

```python
position_lock = asyncio.Lock()

async def approve_trade(self, signal):
    async with self.position_lock:
        if self.open_positions >= 1:
            return RejectedBet(reason="MAX_POSITIONS_REACHED")
        self.open_positions += 1
        return ApprovedBet(...)
```

---

## 18. Implementation Roadmap

### 18.1 Fase 0 — Prerequisite & Setup (1–2 minggu)

**Deliverable:** Development environment + API verification siap

| Task | Estimasi | Output |
|---|---|---|
| Setup Python env, install dependencies | 2 jam | requirements.txt verified |
| Buat Binance READ-ONLY API key | 30 menit | API key + whitelisting Railway IP |
| Buat akun Polymarket + deposit minimal | 1 jam | Akun aktif |
| **KRITIS: Observasi manual 10 market cycles** | 2–3 jam | Contract spec verified |
| **KRITIS: Konfirmasi oracle resolusi Polymarket** | 2 jam | Oracle confirmed |
| **KRITIS: Cek ketersediaan historical market data** | 1 jam | Data availability confirmed |
| Setup Railway project + Volume + PostgreSQL | 1 jam | Railway deployment ready |
| Setup git repo + .gitignore + .env.example | 30 menit | Repository ready |

**Gate: Semua KRITIS tasks harus selesai sebelum development dimulai.**

### 18.2 Fase 1 — Data Collection Infrastructure (4+ minggu)

**TIDAK BISA DIPERPENDEK.** OBI dan TFM memerlukan data historis yang hanya tersedia via live collection.

| Task | Estimasi | Output |
|---|---|---|
| `binance_feed.py` WS + REST + buffer | 8 jam | Binance data stream verified |
| `scripts/collect_binance_data.py` (OHLCV + aggTrades) | 4 jam | 3 bulan historical data |
| `scripts/collect_orderbook.py` (24/7 collector) | 6 jam | Orderbook snapshots per 5 detik |
| `scripts/collect_polymarket.py` (market history) | 6 jam | Strike prices + resolutions |
| `database.py` schema + migrations | 4 jam | Database fully operational |
| Unit tests untuk data collection (coverage ≥ 80%) | 4 jam | Test suite green |
| **Deploy ke Railway, jalankan collector 24/7** | 2 jam | 4 minggu data collection |

**Gate: 4 minggu live data collection selesai → baru bisa lanjut ke Fase 2**

### 18.3 Fase 2 — Core System MVP (2–3 minggu)

**Dependency:** Fase 1 selesai + minimum 4 minggu orderbook data

| Task | Estimasi | Dependency |
|---|---|---|
| `market_discovery.py` + state machine | 8 jam | Polymarket API verified |
| `clob_feed.py` polling + cache + stale detection | 6 jam | market_discovery |
| `feature_engine.py` 24 features + anti-lookahead | 12 jam | binance_feed + clob_feed |
| `model.py` training pipeline + Optuna | 10 jam | feature_engine + data |
| Model training run pertama + calibration check | 4 jam | Training data ready |
| `signal_generator.py` full gate logic | 8 jam | model + feature_engine + clob_feed |
| `risk_manager.py` Kelly + limits | 6 jam | signal_generator |
| `dry_run.py` paper trading + PASS/FAIL | 8 jam | risk_manager |
| `exporter.py` CSV + JSON | 4 jam | database |
| `cli.py` Rich dashboard | 6 jam | semua modul |
| Integration test end-to-end | 8 jam | semua modul |
| Unit tests semua modul (coverage ≥ 80%) | 16 jam | semua modul |

### 18.4 Fase 3 — Dry Run (2+ minggu)

**Dependency:** Fase 2 MVP berjalan stabil

| Task | Estimasi | Output |
|---|---|---|
| Deploy Fase 2 ke Railway | 2 jam | System live di Railway |
| Sesi dry run harian (5–8 jam/hari, 14+ hari) | 70–112 jam | Minimum 10 sesi data |
| Analisis performance.json setelah setiap sesi | 1 jam/sesi | Tuning insights |
| Config tuning berdasarkan dry run data | 4 jam total | Optimized config.json |
| Go-live decision review (setelah 10+ sesi) | 2 jam | PASS/FAIL decision |

### 18.5 Fase 4 — Execution Layer & Go-Live (1 minggu)

**Gate: 5 sesi PASS berturut-turut + 100 total dry run trades + manual review**

| Task | Estimasi | Output |
|---|---|---|
| `execution.py` live order placement | 10 jam | Live trading capable |
| Integration test execution dalam dry mode | 4 jam | Execution verified |
| Security audit API key handling | 2 jam | Security checklist PASS |
| Smoke test live dengan $10 (1–2 trades) | 2 jam | Go-live confirmed |
| Scale ke operasional capital ($50–$100) | 30 menit | Full live operation |

### 18.6 Fase 5 — Production Hardening (ongoing)

| Task | Timeline | Trigger |
|---|---|---|
| Retrain model (pertama) | 8–12 minggu setelah go-live | Scheduled atau auto-trigger |
| PSI distribution drift monitoring | +2 minggu dari go-live | - |
| Regime threshold re-tuning | Setelah 30+ sesi | Data-driven, bukan intuisi |
| Multi-market expansion (jika ada market BTC lain) | +3 bulan | After stable operation |

### 18.7 Dependency Graph Antar Modul

```
Phase 0: API Setup, Contract Verification
  ↓
Phase 1: binance_feed → database → collect_scripts (4 minggu parallel)
  ↓                                      ↓
Phase 2: market_discovery → clob_feed → feature_engine → model
                                                    ↓
                            signal_generator ← model
                                    ↓
                            risk_manager → dry_run → exporter → cli
  ↓
Phase 3: Dry Run (iterate + tune)
  ↓
Phase 4: execution.py → smoke_test → live
  ↓
Phase 5: ongoing retrain + monitoring
```

### 18.8 API Contracts Antar Modul (Pydantic Schemas)

```python
# Semua modul berkomunikasi via typed Pydantic models
# Tidak ada dict yang di-pass antar boundary modul

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal, Dict

class ActiveMarket(BaseModel): ...     # market_discovery → semua modul
class CLOBState(BaseModel): ...        # clob_feed → signal_generator
class FeatureVector(BaseModel): ...    # feature_engine → model, signal_generator
class SignalResult(BaseModel): ...     # signal_generator → risk_manager, dry_run
class ApprovedBet(BaseModel): ...      # risk_manager → dry_run, execution
class RejectedBet(BaseModel): ...      # risk_manager → dry_run (logging only)
class TradeOutcome(BaseModel): ...     # dry_run / execution → database
class SessionMetrics(BaseModel): ...   # dry_run → exporter, cli
```

### 18.9 Rollback Plan (Jika Dry Run Tidak Mencapai Threshold)

```
Skenario: 3 minggu dry run, belum mencapai 5 sesi PASS berturut-turut

Investigasi terstruktur:
  Step 1: Cek win_rate breakdown by TTR_phase
          → Apakah filter TTR sudah optimal?
  Step 2: Cek win_rate breakdown by vol_percentile bucket
          → Apakah regime filter terlalu longgar/ketat?
  Step 3: Cek brier score — apakah model well-calibrated?
          → Jika brier > 0.24: RETRAIN dengan data terbaru
  Step 4: Cek edge_yes accuracy vs outcome
          → Apakah edge prediksi berkorelasi dengan kemenangan?
  Step 5: Cek CLOB data kualitas (clob_stale_events, vig distribusi)
          → Apakah ada data quality issue?

Keputusan setelah 3 minggu tanpa target tercapai:
  Option A: Retrain model dengan lebih banyak data + tune hyperparameters
  Option B: Revisi regime threshold (data-driven berdasarkan dry run analytics)
  Option C: Revisi MARGIN_OF_SAFETY (naikkan ke 0.07 untuk selektivitas lebih tinggi)
  Option D: Tunggu 2 minggu tambahan data collection sebelum retrain

  TIDAK ada opsi untuk go-live tanpa threshold tercapai.
  Modal $10 smoke test tidak menggantikan kewajiban dry run gate.
```

---

## Appendix A — Dependencies & Versions

```
requirements.txt:

# Data & Exchange
ccxt==4.2.x
websockets==12.x
httpx==0.27.x

# Polymarket
py-clob-client==0.17.x

# ML
lightgbm==4.x
scikit-learn==1.4.x
optuna==3.x
pandas==2.x
numpy==1.26.x

# Database
sqlalchemy==2.x
aiosqlite
asyncpg

# Application
python-dotenv
rich==13.x
pydantic==2.x
structlog
click
watchdog  # untuk config file watching

# Testing
pytest
pytest-asyncio
pytest-mock

# Utilities
pyarrow  # untuk parquet file I/O
```

---

## Appendix B — Export Data Schema (Contract)

Schema ini adalah **immutable contract** antara dry_run engine dan analisis model berikutnya. Perubahan hanya boleh dilakukan dengan versioning.

### trades.csv

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `trade_id` | UUID | Primary key |
| `session_id` | string | Identifier sesi |
| `market_id` | string | Polymarket condition_id |
| `timestamp_signal` | datetime UTC | Waktu signal di-generate |
| `timestamp_entry` | datetime UTC | Waktu trade "dieksekusi" |
| `timestamp_resolution` | datetime UTC | T_resolution market |
| `signal_type` | BUY_YES / BUY_NO | Arah taruhan |
| `P_model` | float [0,1] | Output model ensemble |
| `clob_yes_ask_at_signal` | float | Harga YES ask saat signal |
| `clob_no_ask_at_signal` | float | Harga NO ask saat signal |
| `edge_yes` | float | P_model − clob_yes_ask |
| `edge_no` | float | (1−P_model) − clob_no_ask |
| `entry_price_usdc` | float | Harga per share dibayar |
| `bet_size_usd` | float | Total USDC diinvestasikan |
| `kelly_fraction` | float | Half-Kelly fraction |
| `kelly_multiplier` | float | Dynamic multiplier |
| `strike_price` | float | Harga BTC saat T_open |
| `btc_at_signal` | float | Harga BTC saat signal |
| `strike_distance_pct` | float | % distance dari strike |
| `TTR_minutes_at_signal` | float | Menit tersisa saat signal |
| `btc_at_resolution` | float | Harga BTC saat T_resolution |
| `outcome` | WIN / LOSS | Hasil aktual |
| `pnl_usd` | float | Profit/loss USD |
| `pnl_pct_capital` | float | PnL sebagai % capital |
| `capital_before` | float | Modal sebelum trade |
| `capital_after` | float | Modal setelah trade |
| `vol_percentile_at_signal` | float | Vol percentile [0,1] |
| `OBI_at_signal` | float | Order Book Imbalance |
| `TFM_norm_at_signal` | float | TFM normalized |
| `market_vig_at_signal` | float | Total vig Polymarket |
| `model_version` | string | Versi model aktif |
| `mode` | DRY / LIVE | Mode trading |

### signals_all.csv

Sama dengan trades.csv, ditambah:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `abstain_reason` | enum / null | REGIME_BLOCK / LIQUIDITY_BLOCK / TTR_PHASE / NO_EDGE |
| `all_edges` | json | `{"edge_yes": x, "edge_no": y}` untuk semua evaluated signals |

### clob_log.csv

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `timestamp` | datetime UTC | Waktu snapshot |
| `market_id` | string | Market aktif |
| `TTR_minutes` | float | Menit tersisa |
| `yes_ask` | float | Best ask YES |
| `yes_bid` | float | Best bid YES |
| `no_ask` | float | Best ask NO |
| `no_bid` | float | Best bid NO |
| `yes_depth_usd` | float | YES liquidity dalam range 3% |
| `no_depth_usd` | float | NO liquidity dalam range 3% |
| `market_vig` | float | yes_ask + no_ask − 1 |
| `is_liquid` | bool | Memenuhi min depth |

---

## Appendix C — REQUIRES VALIDATION Checklist

```
Sebelum development dimulai, semua item ini harus diverifikasi:

CONTRACT SPECIFICATION:
  [ ] Konfirmasi oracle resolusi harga: Chainlink? Pyth? Binance spot?
  [ ] Konfirmasi strike_price = harga BTC tepat saat T_open atau mechanism lain
  [ ] Konfirmasi interval market: apakah benar-benar setiap 15 menit?
  [ ] Konfirmasi ada/tidaknya gap antara market lama resolve dan baru buka
  [ ] Observasi manual 10 market cycles untuk validasi asumsi

DATA AVAILABILITY:
  [ ] Ketersediaan historical market data (strike_price + BTC_at_resolution)
  [ ] Format dan schema historical market data dari Polymarket API
  [ ] Apakah aggTrade historical data dari Binance Vision cukup untuk TFM reconstruction?

API & INFRASTRUCTURE:
  [ ] Railway network latency ke Binance WS (cek dari Railway environment)
  [ ] Binance API whitelist: konfirmasi Railway outbound IP static atau dynamic
  [ ] py-clob-client 0.17.x: konfirmasi TLS verification aktif
  [ ] Polymarket CLOB API rate limits (request per detik)
  [ ] KYC requirement untuk Polymarket (region Indonesia?)

MODEL:
  [ ] Population Stability Index (PSI) implementation untuk feature drift
  [ ] Platt Scaling / Isotonic Regression untuk model calibration jika diperlukan
```

---

*Dokumen ini adalah FULL TRD v1.0 yang dikonversi dari PRE-TRD v2.0.*  
*Setiap item yang ditandai `REQUIRES VALIDATION` harus diselesaikan sebelum development dimulai.*  
*Semua threshold dan parameter di-define dalam config.json dan dapat di-update tanpa redeploy.*