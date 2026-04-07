# PRE-TRD: Polymarket "Bitcoin Up or Down — 15 Minutes"
## Probability Mispricing Detection Bot

**Versi Dokumen:** 2.0 — Revised  
**Status:** Pre-TRD Final — Siap dikonversi ke TRD Formal  
**Tanggal Revisi:** 2025  
**Revisi Utama dari v1.0:**
- Problem framing diubah dari direction prediction ke probability mispricing detection
- Target market dispesifikkan: Polymarket "Bitcoin Up or Down — 15 Minutes"
- Model target variable diubah ke conditional probability dengan TTR sebagai interaction feature
- Signal Generator diintegrasikan dengan CLOB Polymarket real-time
- Modul Market Discovery dan CLOB Feed ditambahkan
- Regime Filter dipertahankan sebagai hard gate dengan threshold tunable
- Bagian PnL Growth Scenarios dihapus (bukan konten TRD)
- Python asyncio dipertahankan untuk data ingestion, observability metrics ditambahkan

---

## Verdict Operasional

Sistem ini layak dieksekusi dengan satu reframing fundamental dari versi sebelumnya: ini bukan sekadar "bot prediksi arah BTC", melainkan **engine pendeteksi mispricing probabilitas** di pasar prediksi Polymarket. Perbedaan ini bukan semantik — ia mendefinisikan ulang kondisi entry, objective function model, dan sumber edge yang sesungguhnya.

Edge tidak berasal dari kemampuan memprediksi arah harga lebih akurat dari pasar. Edge berasal dari **deteksi gap antara probabilitas model dan harga yang ditawarkan CLOB Polymarket**. Ketika model menilai peluang "UP" di 62% tetapi CLOB menjual YES di harga $0.54, terdapat mispricing 8 poin persentase yang bisa dieksploitasi — terlepas dari apakah prediksi arah itu sendiri sempurna.

Arsitektur full-Python MVP dapat berjalan di lokal maupun Railway tanpa modifikasi kode.

---

## Bagian 0 — Definisi Target Market & Mekanisme Kontrak

Ini adalah prerequisite zero. Semua keputusan arsitektur berikutnya bergantung pada pemahaman mekanisme kontrak yang tepat.

### Spesifikasi Market Target

```
Platform     : Polymarket (https://polymarket.com)
Market name  : "Bitcoin Up or Down — 15 Minutes"
Market type  : Binary outcome (YES/NO)
Instrumen    : USDC-settled, setiap share bernilai $1.00 saat menang

Mekanisme resolusi:
  - Market dibuka pada waktu T_open dengan strike_price = harga BTC saat T_open
  - Pertanyaan: "Apakah harga BTC lebih tinggi dari strike_price pada T_open + 15 menit?"
  - YES menang jika: BTC_price(T_open + 15 min) > strike_price
  - NO menang  jika: BTC_price(T_open + 15 min) <= strike_price
  - Sumber harga resolusi: ditentukan oleh Polymarket (verifikasi via API docs)

Market rotation:
  - Setiap 15 menit satu market baru dibuka (atau setelah market sebelumnya resolve)
  - Bot harus mendeteksi market aktif, bukan hardcode market ID tunggal
  - Market ID berubah setiap siklus → wajib implementasi Market Discovery

VERIFIKASI WAJIB sebelum development:
  [ ] Konfirmasi sumber harga resolusi via Polymarket API documentation
  [ ] Konfirmasi apakah strike = harga tepat saat T_open atau ada mekanisme lain
  [ ] Konfirmasi interval market baru (apakah benar-benar setiap 15 menit atau on-demand)
  [ ] Cek apakah ada gap antara resolusi market lama dan pembukaan market baru
```

### Implikasi Arsitektur dari Mekanisme Kontrak

```
1. Model tidak memprediksi "naik atau turun dari sekarang"
   Melainkan: "naik atau turun dari strike_price pada saat T_resolution"

2. Strike price adalah parameter input wajib bagi model
   Bukan current price — karena yang relevan adalah jarak harga saat ini ke strike

3. TTR (Time-to-Resolution) menentukan volatility window yang relevan
   TTR = 14 menit sisa → model membutuhkan 14 menit volatility forecast
   TTR = 3 menit sisa  → price sudah sangat dekat ke outcome, model kurang relevan

4. Window optimal untuk entry: TTR antara 5-12 menit
   Terlalu awal (>12 min): terlalu banyak uncertainty tersisa
   Terlalu terlambat (<5 min): spread CLOB melebar, edge terkompres
```

---

## Bagian 1 — Semua Formula Inti

### A. Feature Engineering Formulas

#### A.1 Fitur Microstructure (dari Binance)

```python
# 1. ORDER BOOK IMBALANCE (OBI)
# Input: top-N level bid/ask dari orderbook snapshot BTC/USDT Binance
OBI = (bid_volume_sum - ask_volume_sum) / (bid_volume_sum + ask_volume_sum)
# Range: [-1.0, +1.0]
# OBI > +0.2  → tekanan beli, bullish microstructure
# OBI < -0.2  → tekanan jual, bearish microstructure
# Level: top 5 orderbook levels, snapshot setiap 1 detik

# 2. TRADE FLOW MOMENTUM (TFM)
# Input: stream taker trades (buy-initiated vs sell-initiated)
TFM_raw        = rolling_sum(taker_buy_volume, window=60s) \
               - rolling_sum(taker_sell_volume, window=60s)
TFM_normalized = TFM_raw / (rolling_std(TFM_raw, window=20_periods) + epsilon)
# epsilon = 1e-8 (zero-division guard)
# |TFM_normalized| > 1.5 = sinyal bermakna secara statistik

# 3. VOLATILITY-ADJUSTED MOMENTUM (VAM)
close_returns = log(close[t] / close[t-1])
realized_vol  = rolling_std(close_returns, window=12)  # 12 bars = 3 jam di 15m
VAM           = close_returns / (realized_vol + epsilon)
# VAM mengukur magnitude pergerakan relatif terhadap "noise normal" pasar

# 4. REALIZED VOLATILITY (per-bar, annualized)
RV = rolling_std(close_returns, window=12) * sqrt(252 * 96)
# 96 bars per hari di 15-menit, 252 hari trading per tahun

# 5. ORDER BOOK DEPTH RATIO
depth_ratio = sum(bid_size_levels_1_to_3) / (sum(ask_size_levels_1_to_3) + epsilon)
# > 1.0 → bid side lebih tebal (bullish)
# < 1.0 → ask side lebih tebal (bearish)

# 6. ROLLING Z-SCORE — normalisasi anti-lookahead (WAJIB)
z_score(feature, t) = (
    (feature[t] - rolling_mean(feature, window=100)[t])
    / (rolling_std(feature, window=100)[t] + epsilon)
)
# KRITIS: window di-compute HANYA dari data sebelum t
# Implementasi: shift(1) sebelum rolling calculation, TIDAK menggunakan data[t] dalam window
```

#### A.2 Fitur Kontekstual Market (NEW — dari Polymarket + perhitungan)

```python
# 7. TIME-TO-RESOLUTION (TTR) — fitur paling penting v2.0
T_resolution   = active_market.end_datetime_utc      # dari Polymarket API
TTR_seconds    = (T_resolution - datetime.utcnow()).total_seconds()
TTR_minutes    = TTR_seconds / 60.0
TTR_normalized = max(0.0, min(1.0, TTR_minutes / 15.0))
# [0.0, 1.0]: 1.0 = baru dibuka, 0.0 = tepat di momen resolusi

# Cyclical encoding TTR (agar model belajar non-linearity)
TTR_sin = sin(pi * TTR_normalized)  # puncak saat TTR = 0.5 (7.5 menit sisa)
TTR_cos = cos(pi * TTR_normalized)  # membedakan fase awal vs akhir

# Phase classification (digunakan sebagai hard gate, bukan fitur model)
TTR_phase = "ENTRY_WINDOW" if 5 <= TTR_minutes <= 12 else \
            "EARLY"        if TTR_minutes > 12         else \
            "LATE"         # LATE: NO new positions allowed

# 8. STRIKE DISTANCE (NEW)
strike_price         = active_market.initial_btc_price  # harga BTC saat T_open
strike_distance_pct  = (current_btc_price - strike_price) / strike_price * 100
# > +0.5%  → harga sudah jelas di atas strike (YES territory)
# < -0.5%  → harga sudah jelas di bawah strike (NO territory)
# [-0.5, +0.5] → "contest zone" — di sini model memberikan nilai tambah terbesar

# 9. STRIKE DISTANCE × TTR INTERACTION
# Model edge tertinggi: contest zone + TTR di mid-range
contest_urgency = abs(strike_distance_pct) * (1 - TTR_normalized)
# Semakin kecil jarak ke strike DAN semakin dekat ke resolusi → semakin kritis

# 10. TTR INTERACTION FEATURES (cross-features untuk LightGBM)
ttr_x_obi       = TTR_normalized * OBI
ttr_x_tfm       = TTR_normalized * TFM_normalized
ttr_x_strike    = TTR_normalized * strike_distance_pct
# Memungkinkan model belajar: "OBI lebih prediktif ketika TTR masih besar"

# 11. TIME ENCODING (cyclical — untuk pola intraday)
hour_sin = sin(2 * pi * utc_hour / 24)
hour_cos = cos(2 * pi * utc_hour / 24)
dow_sin  = sin(2 * pi * day_of_week / 7)
dow_cos  = cos(2 * pi * day_of_week / 7)

# 12. HARGA vs EMA (posisi harga dalam tren)
price_vs_ema20 = (close - EMA(close, 20)) / (close + epsilon)
```

#### A.3 Fitur CLOB Polymarket (NEW)

```python
# 13. CLOB IMPLIED PROBABILITY
clob_yes_ask    = polymarket_clob.best_ask("YES")   # harga termurah untuk beli YES [0,1]
clob_yes_bid    = polymarket_clob.best_bid("YES")   # harga tertinggi yang mau beli YES [0,1]
clob_no_ask     = polymarket_clob.best_ask("NO")    # harga termurah untuk beli NO [0,1]
clob_no_bid     = polymarket_clob.best_bid("NO")

clob_yes_mid    = (clob_yes_ask + clob_yes_bid) / 2  # mid-market YES probability
clob_no_mid     = (clob_no_ask  + clob_no_bid)  / 2  # mid-market NO probability

# Market vig (overhead Polymarket)
market_vig      = (clob_yes_ask + clob_no_ask) - 1.0
# Normal range: 0.02–0.05 (2–5%). Jika > 0.07 → market tidak efisien, jangan trade

# 14. CLOB SPREAD (sebagai fitur ketidakpastian)
clob_yes_spread = clob_yes_ask - clob_yes_bid
clob_no_spread  = clob_no_ask  - clob_no_bid
# Spread lebar → liquidity rendah → execution risk tinggi

# 15. CLOB DEPTH (likuiditas yang tersedia dalam range 3% dari ask)
clob_yes_depth  = sum_volume_within_range(YES, range_pct=0.03)
clob_no_depth   = sum_volume_within_range(NO,  range_pct=0.03)
# Minimum depth yang diperlukan: cukup untuk fill bet_size yang direncanakan
```

---

### B. Probability Arbitrage Detection (Pengganti "Confidence Threshold")

Ini adalah perubahan paling fundamental dari v1.0. Sistem tidak lagi bertanya "seberapa confident model?", melainkan "seberapa besar selisih antara penilaian model dan harga yang ditawarkan pasar?"

```python
# PARAMETER (tunable hyperparameters, bukan hardcoded)
MARGIN_OF_SAFETY    = config.get("margin_of_safety", 0.05)    # default 5 poin persentase
MIN_CLOB_DEPTH_USD  = config.get("min_clob_depth", 10.0)       # minimum $10 liquidity
MAX_MARKET_VIG      = config.get("max_market_vig", 0.07)        # reject jika vig > 7%

# LIQUIDITY GATE (hard gate — tidak negotiable)
CLOB_LIQUID = (
    clob_yes_depth >= MIN_CLOB_DEPTH_USD AND
    clob_no_depth  >= MIN_CLOB_DEPTH_USD AND
    market_vig     <= MAX_MARKET_VIG
)

# TTR GATE (hard gate — jangan entry terlalu awal atau terlalu terlambat)
TTR_GATE = (TTR_phase == "ENTRY_WINDOW")  # 5 ≤ TTR_minutes ≤ 12

# MISPRICING DETECTION
edge_yes = P_model - clob_yes_ask          # > 0 → model melihat YES underpriced
edge_no  = (1 - P_model) - clob_no_ask    # > 0 → model melihat NO underpriced

# SIGNAL GENERATION
SIGNAL_BUY_YES = (edge_yes > MARGIN_OF_SAFETY) AND CLOB_LIQUID AND TTR_GATE
SIGNAL_BUY_NO  = (edge_no  > MARGIN_OF_SAFETY) AND CLOB_LIQUID AND TTR_GATE
SIGNAL_ABSTAIN = not (SIGNAL_BUY_YES or SIGNAL_BUY_NO)

# SINYAL TERBAIK (jika keduanya eligible, pilih edge terbesar)
if SIGNAL_BUY_YES and SIGNAL_BUY_NO:
    final_signal = "BUY_YES" if edge_yes >= edge_no else "BUY_NO"
elif SIGNAL_BUY_YES:
    final_signal = "BUY_YES"
elif SIGNAL_BUY_NO:
    final_signal = "BUY_NO"
else:
    final_signal = "ABSTAIN"
```

---

### C. Regime Filter (Hard Gate, Threshold Tunable)

Regime filter dipertahankan sebagai hard gate. Alasan: model LightGBM paling tidak reliable di regime volatility ekstrem — justru di sanalah model cenderung mengeluarkan confidence tinggi yang salah (overconfidence under distribution shift). Threshold disimpan di config, bukan hardcoded, sehingga bisa di-tune berdasarkan data dry run.

```python
# VOLATILITY REGIME GATE
# Threshold disimpan di config.json — bisa di-update tanpa ubah kode
VOL_LOWER = config.get("regime_vol_lower", 0.15)  # default: hindari 15% terendah
VOL_UPPER = config.get("regime_vol_upper", 0.80)  # default: hindari 20% tertinggi

vol_percentile = rolling_rank(RV, window=500) / 500  # [0,1]
REGIME_VOL_OK  = (vol_percentile > VOL_LOWER) AND (vol_percentile < VOL_UPPER)

# BINANCE SPREAD GATE (untuk kualitas data OBI/TFM)
binance_spread_bps = (ask_price - bid_price) / mid_price * 10000
BINANCE_SPREAD_OK  = binance_spread_bps < 5  # max 5 bps

# BINANCE DEPTH GATE
binance_top5_bid = sum(bid_size for bids[:5])
BINANCE_DEPTH_OK = binance_top5_bid > 0.5  # minimum 0.5 BTC di top 5 levels

# FINAL REGIME GATE
REGIME_GO = REGIME_VOL_OK AND BINANCE_SPREAD_OK AND BINANCE_DEPTH_OK

# TUNING CADENCE
# Setelah setiap 5 sesi dry run, evaluasi:
# - Jika signal_count terlalu rendah (<5/session) → pertimbangkan turunkan VOL_UPPER ke 0.85
# - Jika win_rate di vol_percentile [0.75-0.80] rendah (<0.50) → turunkan ke 0.75
# - Jangan turunkan berdasarkan intuisi — berdasarkan data dry run saja
```

---

### D. Position Sizing (Kelly untuk Non-Even-Money Bet)

Ini adalah revisi penting dari v1.0: Kelly dihitung menggunakan odds aktual dari CLOB, bukan asumsi even-money 50/50.

```python
# ODDS AKTUAL DARI CLOB
# Jika beli YES di harga clob_yes_ask:
#   - Jika menang: profit = (1.0 - clob_yes_ask) per share
#   - Jika kalah:  loss   = clob_yes_ask per share
b_yes = (1.0 - clob_yes_ask) / clob_yes_ask  # decimal odds untuk YES

# Jika beli NO di harga clob_no_ask:
b_no  = (1.0 - clob_no_ask) / clob_no_ask   # decimal odds untuk NO

# FULL KELLY
full_kelly_yes = (P_model * b_yes - (1 - P_model)) / b_yes
full_kelly_no  = ((1 - P_model) * b_no - P_model) / b_no

# HALF-KELLY (selalu gunakan ini — safety margin terhadap model error)
half_kelly_yes = max(0.0, full_kelly_yes / 2)
half_kelly_no  = max(0.0, full_kelly_no  / 2)

# CONTOH NUMERIK
# P_model = 0.62, clob_yes_ask = 0.54
# b_yes = (1 - 0.54) / 0.54 = 0.852
# full_kelly = (0.62 * 0.852 - 0.38) / 0.852 = (0.528 - 0.38) / 0.852 = 0.174
# half_kelly = 0.087 → bet 8.7% dari capital

# BET SIZE DENGAN CONSTRAINTS
kelly_fraction = half_kelly_yes if final_signal == "BUY_YES" else half_kelly_no
raw_bet        = capital * kelly_fraction
bet_size       = max(raw_bet, MIN_BET)    # floor: $1.00 (Polymarket minimum)
bet_size       = min(bet_size, capital * MAX_BET_FRACTION)  # ceiling: 10% capital
bet_size       = round(bet_size, 2)       # USDC 2-decimal precision

# DYNAMIC KELLY MULTIPLIER (protective scale-down saat losing streak)
consecutive_losses = count_consecutive_losses(recent_trades)
kelly_multiplier   = max(0.25, 1.0 - (consecutive_losses * 0.15))
# 0 losses → 100% of kelly  |  3 losses → 55%  |  5 losses → 25% floor
adjusted_bet = round(bet_size * kelly_multiplier, 2)
```

---

### E. Expected Value (Menggunakan CLOB Odds)

```python
# EV UNTUK BUY YES
EV_yes = P_model * (1.0 - clob_yes_ask) - (1 - P_model) * clob_yes_ask
       = P_model - clob_yes_ask
# Interpretasi: EV_yes = edge_yes (dari probability arbitrage section)
# Contoh: P_model=0.62, clob_yes_ask=0.54 → EV_yes = 0.08 per share (8 sen per $0.54 bet)

# EV UNTUK BUY NO
EV_no  = (1 - P_model) * (1.0 - clob_no_ask) - P_model * clob_no_ask
       = (1 - P_model) - clob_no_ask

# EV PER DOLLAR YANG DIINVESTASIKAN (untuk perbandingan antar bet)
EV_yes_pct = EV_yes / clob_yes_ask
EV_no_pct  = EV_no  / clob_no_ask

# EXPECTANCY (agregat per sesi)
expectancy_session = mean([ev_per_trade_1, ev_per_trade_2, ...]) / mean(bet_sizes)
```

---

### F. Performance Metrics

```python
# SHARPE RATIO (rolling 30-trade window, basis per-trade bukan per-hari)
per_trade_returns = [pnl_i / bet_size_i for each trade i]
sharpe_rolling    = mean(per_trade_returns) / std(per_trade_returns) * sqrt(N_trades_annualized)
# Target: sharpe_rolling > 1.0

# SORTINO RATIO
downside_returns = [r for r in per_trade_returns if r < 0]
sortino          = mean(per_trade_returns) / std(downside_returns) * sqrt(N_trades_annualized)

# MAXIMUM DRAWDOWN
equity_curve = cumsum(pnl_sequence) + capital_start
peak         = cummax(equity_curve)
drawdown     = (equity_curve - peak) / peak
max_dd       = min(drawdown)
# Target: max_dd > -0.20 (drawdown tidak melebihi -20%)

# CALMAR RATIO
calmar = annualized_return / abs(max_dd)
# Target: calmar > 0.5

# WIN RATE (rolling 50-trade window)
win_rate_rolling = rolling_sum(is_win, window=50) / 50
# Target post-filter: >= 0.53

# PROFIT FACTOR
profit_factor = sum(pnl for wins) / abs(sum(pnl for losses))
# Target: >= 1.10

# BRIER SCORE (model calibration — akurasi probabilitas, bukan hanya arah)
brier = mean((P_model_i - actual_outcome_i)^2)
# Target: brier < 0.24 (random model = 0.25)
# KRITIS: model yang well-calibrated tidak hanya prediksi arah, tapi
# juga memberikan probabilitas yang mendekati frekuensi aktual

# MISPRICING EDGE ACCURACY
edge_accuracy = correlation(edge_yes_predicted, actual_outcome_binary)
# Mengukur apakah sinyal mispricing kita benar-benar menghasilkan alpha

# DRY RUN SCORE KOMPOSIT
dry_run_score = (
    0.35 * normalize(win_rate_rolling,  lower=0.50, upper=0.62) +
    0.25 * normalize(expectancy,        lower=-0.01, upper=0.05) +
    0.20 * normalize(sharpe_rolling,    lower=0.0,  upper=2.0)  +
    0.20 * normalize(1.0 + max_dd,      lower=0.80, upper=1.0)
)
# Fungsi normalize: (x - lower) / (upper - lower), clipped ke [0, 1]
# PASS jika: dry_run_score >= 0.70 AND win_rate_rolling >= 0.53 AND profit_factor >= 1.10

# WEBSOCKET OBSERVABILITY (NEW — untuk validasi kualitas data)
ws_message_drop_rate  = dropped_messages / total_expected_messages  # target: < 0.1%
ws_reconnect_count    = count of reconnections per session            # alert if > 3
feature_compute_lag   = mean(time_to_compute_features - bar_close_time)  # target: < 500ms
```

---

## Bagian 2 — Komponen yang Harus Dibangun

### Modul 0: Market Discovery (`market_discovery.py`) — NEW

```
Tanggung jawab:
- Poll Polymarket API untuk menemukan market "Bitcoin Up or Down — 15 Minutes" yang aktif
- Validasi market memiliki waktu tersisa yang cukup (TTR_minutes >= 5)
- Validasi CLOB market memiliki likuiditas minimum
- Return active_market object dengan: market_id, T_resolution, strike_price, TTR
- Detect dan handle transisi: ketika market lama resolve, temukan market baru
- Handle edge case: jika tidak ada market aktif, masuk ke WAITING mode

Market object schema:
  {
    market_id      : str,       # Polymarket condition_id
    question       : str,       # verifikasi ini memang "Up or Down" market
    strike_price   : float,     # harga BTC saat T_open
    T_open         : datetime,  # kapan market dibuka
    T_resolution   : datetime,  # kapan market resolve
    TTR_minutes    : float,     # sisa waktu saat ini
    clob_token_ids : dict,      # {"YES": "0x...", "NO": "0x..."}
  }

Polling interval: setiap 30 detik untuk cek market state
Endpoint: GET https://clob.polymarket.com/markets (filter by tag/keyword)

Error handling:
  - API timeout    → retry 3x dengan exponential backoff (1s, 2s, 4s)
  - No market found → WAITING mode, re-poll setiap 60 detik
  - Market ID changed → hot-reload tanpa restart bot
```

---

### Modul 1: Binance Data Feed (`binance_feed.py`)

```
Tanggung jawab:
- WebSocket connection ke Binance (orderbook depth + aggTrade stream)
- REST fallback untuk OHLCV historical bars
- Reconnect logic dengan exponential backoff
- In-memory circular buffer: 500 bar terakhir OHLCV + orderbook snapshots
- WebSocket observability metrics (NEW)

Streams yang di-subscribe:
  - btcusdt@depth20@100ms   (orderbook top-20, update setiap 100ms)
  - btcusdt@aggTrade         (aggregated trades real-time)
  - btcusdt@kline_15m        (15-menit OHLCV bars)

Input  : Binance WebSocket streams
Output : DataFrame real-time + event callbacks ke Feature Engine

Critical behaviors:
  - Timestamp normalization ke UTC (Binance menggunakan millisecond epoch)
  - Validasi bar: reject jika volume = 0 atau high < low atau OHLC inconsistent
  - Gap detection: jika disconnect > 30 detik, tandai bars sebagai STALE
    dan blokir signal generation sampai data segar kembali

Observability metrics (NEW — untuk monitoring GIL impact):
  - ws_message_drop_rate : track via sequence number gap detection
  - ws_latency_ms        : timestamp received vs exchange timestamp
  - queue_depth          : panjang antrian pesan yang belum diproses
  Alert threshold: drop_rate > 0.1% ATAU latency > 2000ms → log WARNING

Library: websockets==12.x dengan asyncio (asyncio adalah I/O-bound,
  tidak terpengaruh GIL secara material untuk interval 15-menit.
  Migration ke Go hanya jika ws_message_drop_rate > 0.1% di production.)
```

---

### Modul 1b: Polymarket CLOB Feed (`clob_feed.py`) — NEW

```
Tanggung jawab:
- Poll atau subscribe ke CLOB Polymarket untuk market aktif
- Track YES/NO ask/bid prices dan depth secara real-time
- Deteksi kondisi liquidity insufficient → trigger ABSTAIN
- Expose interface: get_clob_snapshot() → CLOBState object

CLOBState schema:
  {
    market_id        : str,
    timestamp        : datetime,
    yes_ask          : float,    # best ask untuk YES [0,1]
    yes_bid          : float,    # best bid untuk YES [0,1]
    no_ask           : float,    # best ask untuk NO [0,1]
    no_bid           : float,    # best bid untuk NO [0,1]
    yes_depth_usd    : float,    # likuiditas YES dalam $USD (dalam range 3% dari ask)
    no_depth_usd     : float,    # likuiditas NO dalam $USD
    market_vig       : float,    # yes_ask + no_ask - 1.0
    is_liquid        : bool,     # apakah memenuhi MIN_CLOB_DEPTH_USD
  }

Polling interval: setiap 5 detik (CLOB Polymarket biasanya REST-only, bukan WS)
Endpoint: GET https://clob.polymarket.com/book?token_id={token_id}

Caching: simpan snapshot terakhir, expire setelah 10 detik
Error: jika fetch gagal → gunakan cached snapshot dengan flag STALE,
  jika stale > 30 detik → blokir semua signal (data tidak dipercaya)

Library: httpx (async HTTP client) untuk non-blocking polling
```

---

### Modul 2: Feature Engine (`feature_engine.py`)

```
Tanggung jawab:
- Menerima: bar data dari Binance Feed + TTR dari Market Discovery
           + CLOB snapshot dari CLOB Feed
- Compute semua features tanpa lookahead
- Rolling window management: shift(1) wajib sebelum setiap rolling calc
- Output normalized feature vector untuk model inference

Feature list lengkap (urutan ini adalah feature_list.json):
  # Microstructure (dari Binance)
  01. OBI                     # Order Book Imbalance
  02. TFM_normalized          # Trade Flow Momentum normalized
  03. VAM                     # Volatility-Adjusted Momentum
  04. RV                      # Realized Volatility
  05. vol_percentile          # rolling rank RV dalam 500 bars
  06. depth_ratio             # bid/ask depth ratio top 3
  07. price_vs_ema20          # harga vs EMA 20
  08. binance_spread_bps      # Binance spread dalam basis points

  # Temporal cyclical
  09. hour_sin
  10. hour_cos
  11. dow_sin
  12. dow_cos

  # Kontekstual market (NEW)
  13. TTR_normalized          # Time-to-Resolution [0,1]
  14. TTR_sin                 # cyclical TTR
  15. TTR_cos                 # cyclical TTR
  16. strike_distance_pct     # (current_price - strike) / strike * 100
  17. contest_urgency         # |strike_distance| * (1 - TTR_normalized)

  # Interaction features (NEW)
  18. ttr_x_obi               # TTR × OBI
  19. ttr_x_tfm               # TTR × TFM_normalized
  20. ttr_x_strike            # TTR × strike_distance_pct

  # CLOB features (NEW)
  21. clob_yes_mid            # mid-market YES probability
  22. clob_yes_spread         # YES ask-bid spread
  23. clob_no_spread          # NO ask-bid spread
  24. market_vig              # total Polymarket overhead

Output struct:
  {
    "features"    : numpy.ndarray shape (1, 24),
    "timestamp"   : datetime,
    "TTR_minutes" : float,
    "TTR_phase"   : str,          # ENTRY_WINDOW / EARLY / LATE
    "regime_ok"   : bool,
    "clob_liquid" : bool,
    "raw_values"  : dict,         # untuk logging & export (unscaled)
  }
```

---

### Modul 3: ML Model (`model.py`)

```
Model arsitektur (MVP):
  Primary  : LightGBM Classifier
  Secondary: Logistic Regression (baseline — untuk deteksi overfit)
  Ensemble : weighted average = 0.7 × LGBM + 0.3 × LogReg

TARGET VARIABLE (REVISED dari v1.0):
  BUKAN  : P(BTC_price > entry_price_now in 15 min)
  ADALAH : P(BTC_price(T_res) > strike_price | features, TTR_normalized)
  Label  : binary, 1 jika BTC_at_T_res > strike_price, 0 jika tidak

  Konsekuensi untuk label construction:
    - Setiap training sample harus memiliki: T_signal, strike_price, BTC_at_T_res
    - T_signal ≠ T_open market — bisa kapan saja dalam ENTRY_WINDOW
    - Pastikan strike_price diambil dari T_open market, BUKAN dari T_signal

TTR SEBAGAI INTERACTION FEATURE (bukan fitur biasa):
  - TTR_normalized masuk sebagai fitur individual (feature 13)
  - Plus 3 cross-features (18, 19, 20) yang explicit
  - LightGBM akan menemukan interaksi lebih lanjut via tree splitting
  - Ini memungkinkan model belajar: "OBI prediktif ketika TTR besar,
    tapi strike_distance lebih dominan ketika TTR kecil"

Training requirements:
  - Minimum: 3 bulan data historis (≥ 8,640 bars @ 15-menit dari Binance)
  - PLUS: Polymarket market data historis untuk mendapatkan strike_price per market
    (ini yang akan membatasi training data — verifikasi ketersediaan historical market data)
  - Split: 60% train / 20% validation / 20% test — CHRONOLOGICAL, TIDAK random
  - Hyperparameter tuning: Optuna, 50 trials, TimeSeriesSplit(n_splits=5) CV

Model persistence:
  - Files: model_v{YYYYMMDD_HHMMSS}.pkl + scaler.pkl + feature_list.json
  - Simpan 3 versi terakhir untuk rollback
  - Setiap versi disertai training_metrics.json (brier, AUC, win_rate_oos)

Retrain trigger (otomatis):
  - rolling_win_rate_30 < 0.50 selama 2 sesi berturut-turut
  - brier_score_rolling > 0.245 (mendekati random)
  - Retrain menggunakan data terbaru (sliding window, 3 bulan terakhir)
```

---

### Modul 4: Signal Generator (`signal_generator.py`)

```
Tanggung jawab:
- Orkestrasi: menerima output Feature Engine + Model + CLOB Feed
- Implementasi probability arbitrage detection (Bagian 1B)
- Emit signal atau ABSTAIN dengan full context

Decision flow:
  1. Cek TTR_phase == "ENTRY_WINDOW" → jika tidak, ABSTAIN
  2. Cek REGIME_GO → jika tidak, ABSTAIN (REGIME_BLOCK)
  3. Cek CLOB_LIQUID → jika tidak, ABSTAIN (LIQUIDITY_BLOCK)
  4. Run model inference → dapatkan P_model
  5. Hitung edge_yes = P_model - clob_yes_ask
  6. Hitung edge_no  = (1 - P_model) - clob_no_ask
  7. Cek apakah salah satu edge > MARGIN_OF_SAFETY → emit signal
  8. Jika keduanya di bawah threshold → ABSTAIN (NO_EDGE)

Output struct (SignalResult):
  {
    signal          : "BUY_YES" | "BUY_NO" | "ABSTAIN",
    abstain_reason  : "REGIME_BLOCK" | "LIQUIDITY_BLOCK" |
                      "TTR_PHASE" | "NO_EDGE" | None,
    P_model         : float,
    edge_yes        : float,
    edge_no         : float,
    clob_yes_ask    : float,
    clob_no_ask     : float,
    TTR_minutes     : float,
    strike_price    : float,
    current_price   : float,
    strike_distance : float,
    market_id       : str,
    timestamp       : datetime,
    features        : dict,       # full feature snapshot untuk logging
  }
```

---

### Modul 5: Risk Manager (`risk_manager.py`)

```
Tanggung jawab:
- Compute bet size via half-Kelly (menggunakan CLOB odds aktual)
- Apply dynamic Kelly multiplier berdasarkan losing streak
- Enforce hard limits (daily loss limit, session loss limit)
- Gate live execution: approve atau reject bet

Input  : SignalResult + capital + recent_trades_history
Output : ApprovedBet object atau RejectedBet dengan alasan

Hard limits:
  - daily_loss_limit    : jika daily_pnl < -5% capital → STOP semua trading hari itu
  - session_loss_limit  : jika session_pnl < -3% capital → PAUSE 30 menit
  - max_open_positions  : 1 (tidak boleh punya 2 posisi open bersamaan)
  - min_capital_floor   : jika capital < $5 → STOP (di bawah viable bet size)

Bet size computation:
  - Panggil kelly_fraction sesuai rumus Bagian 1D
  - Apply dynamic multiplier
  - Apply floor/ceiling constraints
  - Return final bet_size dalam USDC
```

---

### Modul 6: Dry Run Engine (`dry_run.py`)

```
Tanggung jawab:
- Simulasi eksekusi tanpa uang nyata menggunakan data real-time
- Catat setiap signal (termasuk ABSTAIN) dan outcome aktual
- Compute performance metrics per sesi
- Tentukan PASS/FAIL

Paper trade mechanics (sesuai mekanisme Polymarket):
  - Pada signal BUY_YES/BUY_NO:
    * Catat entry: market_id, signal, entry_price (clob_yes_ask atau clob_no_ask),
                   bet_size, strike_price, T_resolution, TTR_at_entry
    * Tunggu hingga T_resolution
    * Ambil BTC_price_at_resolution dari Binance (kline close terdekat T_resolution)
    * Tentukan outcome: WIN jika arah prediksi benar, LOSS jika salah
  
  - Simulasi fill: asumsikan fill di clob_ask (worst case — no slippage improvement)
  - Simulasi payout: jika WIN → profit = bet_size * ((1/entry_price) - 1)
                     jika LOSS → loss = bet_size

PASS criteria (SEMUA harus terpenuhi simultan):
  Hard gates (1 gagal → session FAIL):
    1. min_trades_executed   >= 10 per sesi
    2. win_rate_rolling_50   >= 0.53
    3. max_drawdown_session  > -0.15
    4. profit_factor         >= 1.10

  Soft score komposit        >= 0.70 (formula di Bagian 1F)

Go-live trigger:
  - 5 sesi PASS berturut-turut ATAU 8 dari 10 sesi terakhir PASS
  - Minimum total 100 trades completed di dry run
  - Tidak ada sesi dengan drawdown > -20% dalam seluruh history

Abort triggers (STOP langsung, evaluasi sebelum lanjut):
  - 3 sesi FAIL berturut-turut → review model + CLOB integration
  - win_rate_cumulative < 0.48 setelah 50 trades → STOP, retrain model
  - Consecutive losses >= 6 → PAUSE sesi, cek kondisi pasar dan CLOB liquidity
  - ws_message_drop_rate > 0.5% selama sesi → PAUSE, investigasi koneksi
```

---

### Modul 7: Execution Client (`execution.py`)

```
Tanggung jawab:
- Interface ke Polymarket CLOB API untuk live trading
- HANYA aktif setelah dry run PASS (LIVE_MODE = False by default)
- Place limit order, monitor fill, track open positions, claim winnings

LIVE_MODE gate:
  - Default: LIVE_MODE = False (hardcoded di config)
  - Enable via: python main.py --mode live --confirm-live
  - Memerlukan explicit confirmation prompt di terminal ("ketik YES untuk konfirmasi")

Market rotation handling (NEW):
  - Saat market aktif resolve, otomatis call Market Discovery untuk market berikutnya
  - Tidak boleh open posisi baru jika Market Discovery belum return active_market
  - Jika tidak ada market aktif dalam 5 menit → log WARNING, masuk WAITING mode

Order flow:
  1. Terima ApprovedBet dari Risk Manager
  2. Verifikasi market_id masih aktif (TTR > 3 menit tersisa)
  3. Place limit order di CLOB: price = clob_ask + 0.002 (slight slippage buffer)
  4. Monitor fill status (polling setiap 5 detik, timeout setiap 60 detik)
  5. Jika tidak ter-fill dalam 60 detik → cancel order
  6. Setelah resolution → check winnings, claim jika ada

Library: py-clob-client (Polymarket official Python SDK)
Auth   : API_KEY + API_SECRET + PRIVATE_KEY (dari .env, tidak pernah di kode)
```

---

### Modul 8: Database Layer (`database.py`)

```
Engine:
  - Local    : SQLite (zero-config, file-based)
  - Railway  : PostgreSQL (via DATABASE_URL environment variable)
  - Abstraksi: SQLAlchemy ORM — auto-switch berdasarkan DATABASE_URL

Tables:
  markets         : history market Polymarket yang ditemukan dan dipantau
  signals         : SEMUA signals yang di-generate (termasuk ABSTAIN)
  trades          : semua paper/live trades yang executed
  outcomes        : hasil aktual setelah resolution
  performance     : aggregated metrics per sesi
  model_versions  : metadata model yang pernah dipakai
  system_health   : WebSocket drop rates, latency metrics per sesi

Index wajib:
  signals(timestamp, market_id)
  trades(session_id, timestamp)
  performance(session_id)
```

---

### Modul 9: Export Module (`exporter.py`)

```
Format   : CSV + JSON
Trigger  : otomatis di akhir setiap sesi + manual via CLI
Output   : ./exports/{YYYY-MM-DD}_{session_id}/

Files:
  trades.csv          : semua executed trades (schema: Bagian 4)
  signals_all.csv     : termasuk ABSTAIN dengan alasan abstain
  performance.json    : aggregated session metrics
  features_log.csv    : feature values per bar (untuk model analysis)
  clob_log.csv        : CLOB snapshots yang diobservasi (YES/NO prices per bar)
  equity_curve.csv    : timestamp + capital_value (paper/live)
  ws_health.csv       : WebSocket drop rates dan latency per interval
```

---

### Modul 10: CLI Dashboard (`cli.py`)

```
Framework: Rich (Python)

Display real-time (refresh setiap 5 detik):
  Panel 1 — Market Status:
    - Market ID aktif + TTR tersisa + strike_price
    - BTC current price + strike_distance_pct
    - TTR phase: EARLY / ENTRY_WINDOW / LATE

  Panel 2 — CLOB:
    - YES ask/bid + implied probability
    - NO ask/bid  + implied probability
    - Market vig + liquidity status

  Panel 3 — Model:
    - P_model (latest)
    - edge_yes + edge_no
    - Latest signal + abstain reason (jika ABSTAIN)

  Panel 4 — Session PnL:
    - Session P&L (paper/live)
    - Rolling win rate (last 50 trades)
    - Dry run score + PASS/FAIL status

  Panel 5 — System Health:
    - WS drop rate + latency
    - CLOB feed status (FRESH/STALE)

Commands:
  python main.py --mode dry-run           # mulai sesi dry run
  python main.py --mode live --confirm-live  # live (explicit confirm)
  python main.py --export                 # export sesi terakhir
  python main.py --report                 # lihat aggregated performance
  python main.py --retrain                # trigger model retrain manual
  python main.py --config set vol_upper 0.75  # update regime threshold
```

---

## Bagian 3 — Preparation Checklist

### A. API Keys & Akses

```
[ ] Binance API Key (READ-ONLY)
    URL: https://www.binance.com/en/my/settings/api-management
    Permissions: "Read Info" saja — tidak perlu trading permission
    IP whitelist: Railway VPS IP (update setelah deploy)

[ ] Polymarket CLOB API Access
    URL: https://docs.polymarket.com/#clob-api
    Langkah:
      1. Buat akun Polymarket, complete KYC (jika required by region)
      2. Deposit USDC ke Polymarket wallet (minimum sesuai modal)
      3. Generate API credentials via Polymarket dashboard
      4. Catat: API_KEY, API_SECRET, PRIVATE_KEY (untuk order signing)
    Note: Polymarket menggunakan EIP-712 signature untuk order placement
          py-clob-client handles ini secara otomatis

[ ] Polymarket Historical Market Data (VERIFIKASI)
    - Konfirmasi: apakah ada historical data untuk "Bitcoin Up or Down - 15 Minutes" markets?
    - Endpoint: GET https://clob.polymarket.com/markets (filter by slug/keyword)
    - Jika historical market data tidak tersedia via API → perlu manual collection
      selama 4-8 minggu sebelum training untuk mendapatkan strike_price history

[ ] CoinGlass atau Binance Funding Rate (opsional, untuk future feature)
    - Free public endpoint: https://fapi.binance.com/fapi/v1/fundingRate
```

---

### B. Environment & Dependencies

```
Python version: 3.11+ (wajib — Railway default support)

requirements.txt (final):
  # Data & Exchange
  ccxt==4.2.x             # Binance REST API + historical data
  websockets==12.x        # Binance WebSocket streams
  httpx==0.27.x           # async HTTP untuk CLOB Polymarket polling

  # Polymarket
  py-clob-client==0.17.x  # Official Polymarket Python SDK

  # ML & Data Processing
  lightgbm==4.x           # primary ML model
  scikit-learn==1.4.x     # LogReg baseline + preprocessing
  optuna==3.x             # hyperparameter optimization
  pandas==2.x             # data manipulation
  numpy==1.26.x           # numerical computing

  # Database
  sqlalchemy==2.x         # ORM abstraction
  aiosqlite               # async SQLite driver
  asyncpg                 # async PostgreSQL driver (Railway)

  # Application
  python-dotenv           # .env management
  rich==13.x              # terminal UI
  pydantic==2.x           # data validation + schemas
  structlog               # structured logging (JSON format untuk Railway)
  click                   # CLI argument parsing

  # Testing
  pytest
  pytest-asyncio
  pytest-mock
```

---

### C. Data yang Harus Dikumpulkan Sebelum Training

```
FASE PRE-TRAINING (mulai sesegera mungkin — tidak bisa dipercepat):

1. Binance OHLCV 15-menit: 3 bulan terakhir
   - Sumber: Binance REST API, endpoint /api/v3/klines
   - Format: [open_time, open, high, low, close, volume, ...]
   - Script: scripts/collect_data.py --symbol BTCUSDT --interval 15m --months 3

2. Binance Trade Stream (historical):
   - Sumber: https://data.binance.vision (free)
   - File: BTCUSDT-aggTrades-{YEAR}-{MONTH}.zip
   - Digunakan untuk: reconstruct TFM feature secara historis
   - Estimasi size: ~500MB per bulan

3. Binance Orderbook Snapshots:
   - Tidak tersedia secara historis gratis
   - Solusi: jalankan data_collector.py 24/7 selama 4 minggu SEBELUM training
   - Output: snapshot top-10 orderbook setiap 5 detik → ~200MB per minggu

4. Polymarket Market History (KRITIS — verifikasi ketersediaan):
   - Butuhkan: strike_price per market dan BTC_price_at_resolution per market
   - Endpoint: GET https://clob.polymarket.com/markets (historical markets)
   - Jika data tidak lengkap → perlu deploy early, collect live data dulu

ESTIMASI TOTAL STORAGE DATA: 2-3 GB untuk 3 bulan penuh
ESTIMASI WAKTU PENGUMPULAN SEBELUM TRAINING BISA DIMULAI: 4 minggu minimum
```

---

### D. Hardware Requirements

```
Development (lokal):
  RAM     : minimum 4GB, rekomendasi 8GB
  Storage : 5GB free (data + model + exports)
  CPU     : modern CPU (training LightGBM: 5-15 menit)
  OS      : Linux / macOS / Windows dengan WSL2

VPS Railway:
  Plan              : Hobby ($5/bulan) untuk dry run
                      Pro ($20/bulan) jika butuh lebih dari 512MB RAM
  RAM               : 512MB minimum, 1GB direkomendasikan
  CPU               : shared (sufficient untuk inference + polling)
  Persistent Volume : $0.25/GB/month (mount di /app/data)
  Estimasi total    : $8–15/bulan

KRITIS untuk Railway:
  - Railway TIDAK persist filesystem default → semua data WAJIB ke /app/data (Volume)
  - Model files (.pkl), database (.db / PostgreSQL), dan exports harus di /app/data
  - Setup Railway Volume via dashboard SEBELUM deploy pertama
  - Environment variables: DATABASE_URL, BINANCE_API_KEY, POLYMARKET_API_KEY,
    POLYMARKET_API_SECRET, POLYMARKET_PRIVATE_KEY, LIVE_MODE=false
```

---

### E. Struktur File Proyek

```
btc-updown-bot/
├── .env                         # secrets — JANGAN commit ke git
├── .env.example                 # template tanpa nilai sensitif
├── .gitignore
├── requirements.txt
├── Dockerfile
├── railway.toml
├── config.json                  # tunable hyperparameters (bukan secrets)
├── main.py                      # entry point
├── cli.py                       # CLI commands
├── src/
│   ├── __init__.py
│   ├── market_discovery.py      # (NEW) Polymarket market finder
│   ├── binance_feed.py          # (RENAMED) Binance WS + REST
│   ├── clob_feed.py             # (NEW) Polymarket CLOB polling
│   ├── feature_engine.py        # (REVISED) includes TTR + CLOB features
│   ├── model.py                 # (REVISED) new target variable
│   ├── signal_generator.py      # (REVISED) probability arbitrage
│   ├── risk_manager.py          # (REVISED) CLOB-based Kelly
│   ├── dry_run.py               # (REVISED) Polymarket-aware mechanics
│   ├── execution.py             # (REVISED) market rotation handling
│   ├── database.py
│   └── exporter.py
├── scripts/
│   ├── collect_binance_data.py  # download OHLCV + aggTrades historical
│   ├── collect_orderbook.py     # 24/7 orderbook snapshot collector
│   ├── collect_polymarket.py    # (NEW) Polymarket market history collector
│   ├── train_model.py           # standalone training script
│   └── backtest.py              # offline backtest dengan dry run logic
├── models/                      # model artifacts (di /app/data di Railway)
├── data/                        # raw + processed data
├── exports/                     # session exports
├── config/
│   └── config.json              # regime thresholds, margin_of_safety, dll
└── tests/
    ├── test_market_discovery.py
    ├── test_clob_feed.py
    ├── test_feature_engine.py
    ├── test_signal_generator.py
    ├── test_risk_manager.py
    └── test_dry_run.py
```

---

### F. config.json (Tunable Hyperparameters)

```json
{
  "_comment": "Hyperparameters ini tunable berdasarkan dry run data. Edit via CLI atau langsung.",
  "regime": {
    "vol_lower_threshold": 0.15,
    "vol_upper_threshold": 0.80,
    "binance_spread_max_bps": 5.0,
    "binance_min_depth_btc": 0.5
  },
  "signal": {
    "margin_of_safety": 0.05,
    "ttr_min_minutes": 5.0,
    "ttr_max_minutes": 12.0
  },
  "clob": {
    "min_depth_usd": 10.0,
    "max_market_vig": 0.07,
    "stale_threshold_seconds": 30
  },
  "risk": {
    "kelly_divisor": 2,
    "max_bet_fraction": 0.10,
    "min_bet_usd": 1.00,
    "daily_loss_limit_pct": 0.05,
    "session_loss_limit_pct": 0.03,
    "consecutive_loss_multiplier": 0.15,
    "kelly_floor_multiplier": 0.25
  },
  "dry_run": {
    "pass_win_rate": 0.53,
    "pass_profit_factor": 1.10,
    "pass_dry_run_score": 0.70,
    "pass_max_drawdown": -0.15,
    "min_trades_per_session": 10,
    "go_live_consecutive_pass": 5,
    "go_live_min_total_trades": 100,
    "abort_consecutive_fail": 3,
    "abort_win_rate_threshold": 0.48,
    "abort_consecutive_losses": 6
  },
  "model": {
    "retrain_win_rate_trigger": 0.50,
    "retrain_brier_trigger": 0.245,
    "retrain_consecutive_sessions": 2
  },
  "observability": {
    "ws_drop_rate_alert_threshold": 0.001,
    "ws_latency_alert_ms": 2000
  }
}
```

---

### G. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LIVE_MODE=false

RUN mkdir -p /app/data/models /app/data/exports /app/data/raw

CMD ["python", "main.py", "--mode", "dry-run"]
```

---

### H. railway.toml

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

---

## Bagian 4 — Export Data Schema

Schema ini adalah contract antara dry run engine dan proses optimasi model berikutnya. Setiap kolom harus konsisten antara dry run dan live trading untuk memastikan perbandingan apples-to-apples.

### trades.csv

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `trade_id` | UUID | primary key |
| `session_id` | string | identifier sesi trading |
| `market_id` | string | Polymarket condition_id |
| `timestamp_signal` | datetime UTC | kapan SignalResult di-generate |
| `timestamp_entry` | datetime UTC | kapan trade "dieksekusi" (paper/live) |
| `timestamp_resolution` | datetime UTC | T_resolution market |
| `signal_type` | enum: BUY_YES/BUY_NO | arah taruhan |
| `P_model` | float [0,1] | output model LightGBM |
| `clob_yes_ask_at_signal` | float | harga YES ask saat signal |
| `clob_no_ask_at_signal` | float | harga NO ask saat signal |
| `edge_yes` | float | P_model - clob_yes_ask |
| `edge_no` | float | (1-P_model) - clob_no_ask |
| `entry_price_usdc` | float | harga per share yang dibayar |
| `bet_size_usd` | float | total USDC yang diinvestasikan |
| `kelly_fraction` | float | half-Kelly fraction yang diterapkan |
| `kelly_multiplier` | float | dynamic multiplier (consecutive loss) |
| `strike_price` | float | harga BTC saat market T_open |
| `btc_at_signal` | float | harga BTC saat signal |
| `strike_distance_pct` | float | (btc_at_signal - strike) / strike * 100 |
| `TTR_minutes_at_signal` | float | menit tersisa saat signal |
| `btc_at_resolution` | float | harga BTC saat T_resolution |
| `outcome` | enum: WIN/LOSS | hasil aktual |
| `pnl_usd` | float | profit/loss dalam USD |
| `pnl_pct_capital` | float | PnL sebagai % of capital saat entry |
| `capital_before` | float | modal sebelum trade |
| `capital_after` | float | modal setelah trade |
| `vol_percentile_at_signal` | float | volatility percentile [0,1] |
| `OBI_at_signal` | float | Order Book Imbalance saat signal |
| `TFM_norm_at_signal` | float | TFM normalized saat signal |
| `market_vig_at_signal` | float | total vig Polymarket |
| `model_version` | string | versi model (nama file .pkl) |
| `mode` | enum: DRY/LIVE | paper atau real money |

---

### signals_all.csv (termasuk ABSTAIN)

Sama dengan trades.csv PLUS kolom tambahan:
| `abstain_reason` | enum | REGIME_BLOCK / LIQUIDITY_BLOCK / TTR_PHASE / NO_EDGE / null |
| `all_signal_types` | json | {"BUY_YES": edge_yes, "BUY_NO": edge_no} |

---

### performance.json (per sesi)

```json
{
  "session_id": "2025-01-15_001",
  "date": "2025-01-15",
  "duration_hours": 6.5,
  "mode": "DRY",
  "total_bars_processed": 26,
  "total_signals_evaluated": 26,
  "signals_abstained": 18,
  "abstain_breakdown": {
    "REGIME_BLOCK": 5,
    "LIQUIDITY_BLOCK": 3,
    "TTR_PHASE": 4,
    "NO_EDGE": 6
  },
  "trades_executed": 8,
  "win_count": 5,
  "loss_count": 3,
  "win_rate": 0.625,
  "profit_factor": 1.48,
  "total_pnl_usd": 2.87,
  "total_pnl_pct_capital": 0.0287,
  "max_drawdown_session": -0.032,
  "sharpe_rolling": 1.24,
  "sortino_rolling": 1.87,
  "brier_score": 0.219,
  "mean_edge_yes_traded": 0.072,
  "mean_edge_no_traded": 0.0,
  "mean_TTR_at_signal_minutes": 8.4,
  "mean_strike_distance_at_signal_pct": 0.18,
  "capital_start": 100.00,
  "capital_end": 102.87,
  "dry_run_score": 0.76,
  "pass_fail": "PASS",
  "model_version": "model_v20250115_143022",
  "regime_threshold_vol_upper": 0.80,
  "margin_of_safety": 0.05,
  "ws_drop_rate_pct": 0.004,
  "clob_stale_events": 0,
  "notes": ""
}
```

---

### clob_log.csv (NEW)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `timestamp` | datetime UTC | waktu snapshot |
| `market_id` | string | market aktif |
| `TTR_minutes` | float | menit tersisa |
| `yes_ask` | float | best ask YES |
| `yes_bid` | float | best bid YES |
| `no_ask` | float | best ask NO |
| `no_bid` | float | best bid NO |
| `yes_depth_usd` | float | YES liquidity dalam range 3% |
| `no_depth_usd` | float | NO liquidity dalam range 3% |
| `market_vig` | float | yes_ask + no_ask - 1 |
| `is_liquid` | bool | apakah memenuhi min depth |

---

## Bagian 5 — Dry Run Framework

### Definisi Sesi Dry Run

```
Durasi per sesi : 5–8 jam (sesuai ketersediaan operasional)
Frekuensi       : setiap hari selama minimum 2 minggu
Mode            : data real-time Binance + CLOB Polymarket, tanpa uang nyata
Evaluasi        : PASS/FAIL per sesi dihitung di akhir sesi otomatis
Go-live review  : setelah 10 sesi terkumpul
```

### PASS Criteria (semua harus terpenuhi)

```
Hard gates — jika 1 gagal → sesi FAIL otomatis:
  1. min_trades_executed   >= 10
  2. win_rate_rolling_50   >= 0.53
  3. max_drawdown_session  > -0.15   (drawdown tidak melebihi -15%)
  4. profit_factor         >= 1.10

Soft score komposit        >= 0.70
  (formula: Bagian 1F — DRY RUN SCORE KOMPOSIT)

Syarat tambahan (non-negotiable):
  - ws_message_drop_rate   < 0.5%    (data quality gate)
  - clob_stale_events      == 0      (CLOB data harus fresh selama sesi)
```

### Go-Live Decision Criteria

```
BOLEH go-live jika:
  1. 5 sesi PASS berturut-turut, ATAU 8 dari 10 sesi terakhir PASS
  2. Total trades dry run >= 100
  3. Tidak ada sesi dengan drawdown > -20% dalam seluruh history
  4. brier_score_cumulative < 0.24 (model masih well-calibrated)
  5. mean_edge_traded > 0.04 (rata-rata edge saat entry > 4 poin persentase)

WAJIB konfirmasi manual sebelum go-live:
  - Review performance.json dari semua sesi
  - Konfirmasi market_id yang akan digunakan sudah tervalidasi
  - Konfirmasi LIVE_MODE = true via CLI explicit confirmation
```

### Abort Triggers

```
Langsung STOP dry run, evaluasi ulang sebelum lanjut:
  - 3 sesi FAIL berturut-turut → review CLOB integration + model
  - win_rate_cumulative < 0.48 setelah 50 trades → STOP, retrain model
  - consecutive_losses >= 6 → PAUSE sesi, investigasi kondisi pasar
  - ws_drop_rate > 0.5% per sesi → PAUSE, investigasi koneksi
  - clob_stale_events > 2 per sesi → PAUSE, investigasi CLOB feed
```

---

## Bagian 6 — Catatan Kritis untuk TRD

Empat hal berikut adalah hard constraints yang harus masuk ke TRD sebagai acceptance criteria, bukan sebagai catatan opsional.

**Constraint 1 — Verifikasi mekanisme kontrak sebelum development dimulai.**
Seluruh arsitektur bergantung pada asumsi: strike_price = harga BTC saat T_open, dan resolusi berdasarkan harga BTC pada T_open + 15 menit. Jika Polymarket menggunakan mekanisme yang berbeda (misal: TWAP, atau menggunakan oracle eksternal dengan delay), semua formula di Bagian 1 harus direvisi. Verifikasi wajib dilakukan via Polymarket API documentation DAN observasi langsung 5-10 market sebelum sebuah baris kode ditulis.

**Constraint 2 — Polymarket market availability tidak dijamin.**
"Bitcoin Up or Down — 15 Minutes" bisa tidak tersedia setiap saat. Market Discovery module harus menangani skenario: market tidak ditemukan, market sedang closed, market dalam maintenance. Bot harus graceful degrade ke WAITING mode, bukan crash. Ini harus masuk sebagai test case eksplisit dalam test suite.

**Constraint 3 — Data collection phase tidak bisa diperpendek.**
Minimum 4 minggu orderbook snapshot collection sebelum training bisa dimulai. Ini adalah non-negotiable karena OBI dan TFM adalah fitur paling prediktif dalam sistem, dan keduanya membutuhkan data historis yang belum tersedia secara gratis dari sumber manapun. Development timeline harus memfaktorkan ini.

**Constraint 4 — $10 secara matematis undercapitalized untuk sistem ini.**
Minimum bet Polymarket $1, dan dengan Kelly fraction ~5-8%, capital minimum yang proper adalah $12.5–20 hanya untuk 1 bet. Untuk portfolio yang bisa dikelola dengan risk management yang benar, minimum adalah $50, direkomendasikan $100. Modal $10 dapat digunakan sebagai smoke test go-live, bukan sebagai modal operasional yang serius.

---

## Ringkasan Pre-TRD v2.0

```
Sistem        : Polymarket Bitcoin Up or Down Probability Mispricing Bot
Target market : "Bitcoin Up or Down — 15 Minutes" di Polymarket
Problem frame : Probability mispricing detection (bukan direction prediction)
Model target  : P(BTC(T_res) > strike | features, TTR) — conditional probability
Stack         : Python 3.11, LightGBM, asyncio, SQLAlchemy, Rich CLI
Deploy        : Local + Railway (Dockerfile + Volume + PostgreSQL)
Session       : 5–8 jam/hari, dry run minimum 2 minggu sebelum live
Capital min   : $50 (operasional) — $10 hanya untuk smoke test
Go-live gate  : 5 sesi PASS berturut + 100 total dry run trades + manual confirm
Retrain       : Trigger otomatis jika rolling win_rate < 0.50 selama 2 sesi
Export        : CSV + JSON otomatis per sesi + clob_log.csv untuk analisis
Key edge      : Gap antara P_model dan CLOB ask price (minimum 5 poin persentase)
Key risk      : Market availability, CLOB liquidity, model calibration decay

Section yang perlu ditambahkan saat konversi ke TRD formal:
  - Sprint timeline dengan estimasi jam per modul
  - API contract antar modul (input/output schemas via Pydantic)
  - Acceptance criteria per modul (unit test coverage minimum 80%)
  - Rollback plan jika dry run tidak mencapai threshold setelah 3 minggu
```