# Polymarket BTC Trading Bot: Mispricing & Arbitrage Engine

![Version](https://img.shields.io/badge/version-1.2.0--production-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Status](https://img.shields.io/badge/status-active-success)

Dokumentasi teknis ini menyediakan gambaran mendalam tentang arsitektur, strategi, dan operasional bot trading otomatis yang dirancang khusus untuk mengeksploitasi inefisiensi harga pada market prediksi Bitcoin (BTC) di Polymarket.

---

## 1. Overview

Proyek ini adalah sistem trading frekuensi menengah (mid-frequency) dengan **alpha tunggal**: *edge* antara **fair probability** (diukur terhadap payoff biner Polymarket) versus harga order book CLOB.

Dalam sistem produksi, **Binance menjadi satu-satunya sumber untuk fair probability dan label settlement** (menggunakan **Binance 1-minute candle**). Oracle non-Binance hanya diperlakukan sebagai *basis-risk* (uncertainty inflation atau abstain sesuai policy).

**Value Proposition Utama:**
- **Deteksi Mispricing:** Menggunakan model probabilistik untuk mengidentifikasi kapan harga "Yes" atau "No" di Polymarket tidak selaras dengan dinamika harga BTC yang sebenarnya.
- **Otomatisasi Penuh:** Dari penemuan market baru hingga eksekusi order dan penyelesaian (settlement) tanpa intervensi manual.
- **Resiliensi Cloud:** Dirancang untuk berjalan 24/7 di infrastruktur Railway dengan sistem fail-safe yang ketat.

---

## 2. Arsitektur Sistem

Sistem dibangun dengan arsitektur modular berbasis event-driven menggunakan Python `asyncio`.

### Komponen Utama:
1.  **Data Ingestion Layer:**
    *   **Binance Feed:** Sinkronisasi harga BTC secara real-time via WebSocket dengan fallback REST API.
    *   **CLOB Feed:** Polling snapshot Order Book dari Polymarket Central Limit Order Book (CLOB).
2.  **Signal Engine:**
    *   **Feature Engine:** Komputasi 24 fitur teknis (Microstructure, Temporal, Contextual).
    *   **FairProbabilityEngine:** Menghasilkan `q_fair = P(S_T >= K)` dengan pendekatan digital-option (lognormal + Phi(d2)).
    *   **SignalGenerator V2:** Menentukan `BUY_YES` / `BUY_NO` / `ABSTAIN` berdasarkan **edge setelah uncertainty & biaya** + **no-trade zone**.
3.  **Execution Engine:** Sinkronisasi order biner dengan manajemen slippage dan retry logic.
4.  **Monitoring System:** Dashboard berbasis terminal (Rich) dan logging terstruktur (Structlog) untuk observabilitas maksimal.

---

## 3. Cara Kerja Bot (Lifecycle)

Bot beroperasi dalam loop kontinu dengan tahapan sebagai berikut:

1.  **Market Discovery:** Bot secara berkala memindai API Polymarket untuk mencari event BTC "Price Action" yang aktif.
2.  **Bootstrap:** Mengunduh data historis OHLCV untuk mengisi buffer memori (rolling window).
3.  **Real-time Monitoring:** Sinkronisasi WebSocket dimulai. Setiap penutupan bar (misal: 15 menit), siklus inferensi dipicu.
4.  **Signal Generation:** Fitur dihitung, `FairProbabilityEngine` menghitung `q_fair`, lalu `SignalGenerator V2` menghasilkan sinyal `BUY_YES` / `BUY_NO` / `ABSTAIN` berbasis **edge setelah uncertainty** dan **no-trade zone**.
5.  **Risk Gates:** Sinyal harus melewati filter Risk Manager (balance check, exposure limit, volatility filter).
6.  **Execution:** Order dikirim ke Polymarket CLOB.
7.  **Settlement:** Bot memantau waktu resolusi market untuk mencatat performa (P&L) akhir.

---

## 4. Strategi Trading (Edge vs Fair Probability)

Strategi inti sekarang hanya memiliki **alpha**: *edge* antara **fair probability** (`q_fair`) dan harga order book **CLOB** pada payoff biner Polymarket.

### Inti Perhitungan Edge
Untuk token **YES**:

- `edge_yes_raw = q_fair - yes_ask`
- `edge_yes = edge_yes_raw - uncertainty_u`

Untuk token **NO**:

- `edge_no_raw = (1 - q_fair) - no_ask`
- `edge_no = edge_no_raw - uncertainty_u`

Keputusan entry:
- `BUY_YES` jika `edge_yes` melebihi `signal.margin_of_safety`
- `BUY_NO` jika `edge_no` melebihi `signal.margin_of_safety`
- `ABSTAIN` jika tidak ada edge yang cukup

### No-trade Zone
Untuk menghindari churn (terlalu sering masuk di sekitar “wajar”), bot abstain jika:
- `abs(q_fair - mid_yes)` berada dalam `no_trade_deadband + uncertainty_u`.

### Basis-risk Policy (Non-Binance settlement)
`ActiveMarket` membawa metadata settlement (`settlement_exchange`, `settlement_granularity`).
Jika settlement bukan **Binance 1m**, bot:
- dapat **hard-abstain** dekat resolusi (basis-risk block), atau
- dapat tetap trade dengan **uncertainty inflation** sesuai konfigurasi.

---

## 5. Settlement Alignment (Binance 1m)

Akusisi data probabilitas harus **sejajar** dengan mekanisme settlement kontrak biner.

- **Binance 1-minute candle** adalah sumber label settlement untuk evaluasi P&L dan perhitungan `q_fair`.
- **Polymarket CLOB** tetap menjadi sumber harga entri (YES/NO asks/bids).

Jika market tidak menggunakan Binance-1m untuk settlement, sistem memperlakukan hal tersebut sebagai **basis-risk**: `q_fair` tetap dihitung, tetapi bot memakai uncertainty inflation atau abstain sesuai policy.

---

## 6. Market Discovery Engine

Bot tidak terbatas pada satu market statis. Engine penemuan market bekerja secara otomatis:

- **Filter Market:** Mencari market dengan metadata `group_id` yang berkaitan dengan Bitcoin.
- **Kriteria Seleksi:**
    - **Volume:** Minimal trade volume > $1,000.
    - **Likuiditas:** Spread antara bid-ask di Polymarket tidak boleh lebih dari 5%.
    - **Expiry:** Fokus pada market "15-minute", "Daily", atau "Weekly" yang mendekati waktu resolusi.
- **Rotasi Otomatis:** Jika market saat ini berakhir, bot secara instan melakukan *cold-start* pada market baru yang paling likuid.

---

## 7. Risk Management & Filtering

Setiap trade harus melewati **Circuit Breaker** sistem:

1.  **Conservative Win Probabilities:** Risk sizing memakai probabilitas konservatif berbasis uncertainty buffer:
    - YES: `q_minus = q_fair - uncertainty_u`
    - NO: `q_minus = (1 - q_fair) - uncertainty_u`
2.  **Time/Liquidity-aware Sizing:** ukuran bet diperkecil saat `TTR` makin dekat dan ketika book lebih “mahal/agresif” (proxy via market vig).
3.  **Max Exposure & Kelly Clipping:** bet size dibatasi oleh `risk.max_bet_fraction` dan skema Kelly ter-fraksi/divisor.
4.  **Stop Condition:** abort sesi jika melewati batas kerugian harian/sesi atau consecutive loss (session circuit breaker).

---

## 8. Execution Engine

Eksekusi CLOB memprioritaskan pengendalian biaya dan probabilitas fill:

- **Maker vs taker-like:** ketika edge kecil bot memasang limit pasif (di dalam spread); ketika edge besar bot melakukan cross lebih agresif.
- **Spread-aware pricing:** limit harga dihitung dari `clob_bid/clob_ask` + buffer konfigurasi.
- **Partial-fill accounting:** monitoring fill mengenali `PARTIALLY_FILLED` dan menyetel bet sizing serta settlement berbasis fill.
- **Settlement correctness:** PnL/risiko diselesaikan dengan Binance 1m price provider (bukan last mid-trade).

---

## 9. Monitoring & Logging

Observabilitas adalah prioritas utama untuk sistem produksi:

- **Logging:** Menggunakan format JSON via `structlog` yang dikirim ke sistem monitoring terpusat (Railways Logs).
- **Dashboard:** UI terminal real-time yang menampilkan:
    - Status WebSocket (Active/Stale).
    - Skor Z-Score terbaru.
    - Posisi aktif dan sisa waktu (TTR).
    - Equity Curve (Real-time P&L).

---

## 10. Deployment

Secara default, bot dideploy menggunakan Docker di infrastruktur Railway.

### Persyaratan Sistem:
- **Runtime:** Python 3.10 atau lebih tinggi.
- **Environment Variables:**
    - `POLYGON_PRIVATE_KEY`: Untuk signing transaksi.
    - `BINANCE_API_KEY`: Feed data (optional untuk public stream).
    - `ENVIRONMENT`: `production` atau `development`.

### Cara Menjalankan:
```bash
# Instalasi dependensi
pip install -r requirements.txt

# Menjalankan bot dalam mode Dry Run (Simulasi)
python main.py --mode dry-run

# Menjalankan bot dalam mode Live Trading
python main.py --mode live --confirm-live
```

### Go-live Preflight Gate (operational safety)
Saat `--mode live`, bot tidak langsung mengeksekusi order live. Bot menjalankan **shadow dry-run** sampai dry-run memenuhi gate konfigurasi:
- `dry_run.go_live_min_total_trades`
- `dry_run.pass_dry_run_score` (via `pass_fail == PASS`)
- `dry_run.go_live_consecutive_pass`

Setelah gate lolos barulah eksekusi beralih ke live (tetap settlement-aligned dengan Binance 1m).

---

## 11. Limitasi & Risiko

- **Execution Risk:** Polymarket menggunakan order book yang relatif tipis dibandingkan exchange tradisional; trade besar dapat memicu slippage signifikan.
- **API Rate Limits:** Cloud polling yang terlalu agresif dapat menyebabkan IP blocking.
- **Smart Contract Risk:** Ketergantungan pada reliabilitas kontrak UMA (oracle) untuk resolusi market.

---

## 12. Future Improvements

Rencana pengembangan jangka panjang meliputi:
- **ML Integration:** Mengganti heuristic signal dengan model Deep Learning (LSTM/Transformer) untuk prediksi deret waktu.
- **Latency Optimization:** Implementasi parser WebSocket menggunakan C++ untuk meminimalkan lag internal.
- **Cross-Market Arbitrage:** Melakukan arbitrase antar market prediksi yang berkorelasi (misal: BTC vs ETH price action).

---

*Dibuat dengan presisi oleh Senior Quant Engineering Team.*
