# Polymarket BTC Trading Bot: Latency Arbitrage & Slow Skew Engine

![Version](https://img.shields.io/badge/version-2.0.0--production-blue)
![Python](https://img.shields.io/badge/python-3.11-green)
![Status](https://img.shields.io/badge/status-active-success)

Dokumentasi teknis ini menyediakan gambaran mendalam tentang arsitektur, strategi, dan operasional bot trading otomatis yang dirancang khusus untuk mengeksploitasi inefisiensi latensi harga (*Slow Skew Arbitrage*) pada market prediksi Bitcoin (BTC) di Polymarket.

---

## 1. Overview & Paradigma Strategi

Proyek ini **BUKAN** bot peramal arah harga (directional trading/prediction), melainkan sistem **Arbitrase Latensi (Slow Skew Arbitrage)**.

**Cara Kerja Edge Bot:**
Alih-alih menebak apakah Bitcoin akan naik ke $72,000 dalam 4 jam ke depan, bot membandingkan **Spot Harga Binance Detik Ini** dengan **Harga Order Book Polymarket (CLOB)**. 
Polymarket (CLOB) sering beroperasi lebih lambat merespons gejolak volatilitas spot dibandingkan Binance. Saat BTC melonjak tajam, rumus *Black-Scholes* di otak bot seketika mendeteksi *Fair Probability* yang baru. Jika harga tiket (shares) di Polymarket masih tertinggal (murah), bot akan langsung memborongnya sebelum *Market Maker* manusia/algoritmik Polymarket sempat mengkoreksinya.

**Value Proposition Utama:**
- **Deteksi Slow-Skew:** Menangkap *edge* saat Market Maker lambat merevisi probabilitas seiring dengan perubahan harga spot mendadak.
- **Serverless & Lightweight:** Tanpa *Database* berat. Seluruh metrik diekspor via Flat Files (.CSV) dan dikirim seketika melalui Telegram.
- **Resiliensi Railway Cloud:** Dirancang untuk berjalan 24/7 di VPS/Cloud dengan mekanisme fail-safe ketat terhadap putusnya websocket.

---

## 2. Arsitektur Sistem

Sistem direkayasa ulang menjadi sangat modular (Serverless-Oriented) berbasis event-driven menggunakan Python `asyncio`.

1.  **Data Ingestion Layer:**
    *   **Binance Feed:** Sinkronisasi harga BTC secara real-time via WebSocket berkecepatan tinggi.
    *   **CLOB Feed:** Polling snapshot Order Book dari Polymarket Clob-API.
2.  **Engine Keputusan (The Engine):**
    *   **FairProbabilityEngine:** Menghasilkan `q_fair = P(S_T >= K)` secara instan memakai model *digital-option* (Black-Scholes).
    *   **SignalGenerator:** Mencari ketimpangan antara `q_fair` dan `clob_ask` Polymarket. Akan memutuskan `BUY_YES` / `BUY_NO` / `ABSTAIN`.
3.  **Reporter & Observabilitas:**
    *   **Exporter:** Setiap *slice* dari transaksi akan dicatat dalam rentetan `trades.csv`.
    *   **Telegram Notifier:** Setiap 6 jam, bot bangun mengkomputasi laba bersih (PnL) & Win-rate, kemudian **mengirim `trades.csv` secara otomatis** ke Telegram grup/channel milik Anda.

*(Catatan: Proyek ini TIDAK LAGI menggunakan SQLite atau PostgreSQL untuk memudahkan deployment cloud yang efisien).*

---

## 3. Market Discovery Engine (Automatisasi Rotasi)

Anda tidak perlu menyuapi bot dengan ID market secara manual.
Modul `market_discovery.py` memastikan bot otomatis rotasi dan mengunci ke market ter-likuid (`active_market`):

- **Auto-Search:** Memindai sub-market yang berkaitan dengan struktur: "Will BTC be above $X on [Date]".
- **Auto-Rotate:** Jika satu market selesai resolusi atau mendekati saat *closing time*, bot langsung mencari event 1-12 jam berikutnya tanpa perlu di-*restart*.
- **Vig Safety Filter:** Bot menolak masuk ke sub-market jika biaya spread (Vig) mencapai margin mematikan (default disetel aman di 7%).

---

## 4. Peluang Kemenangan & Manajemen Risiko

Setiap trade harus melewati **Circuit Breaker** ketat:

- **Margin of Safety:** Tidak akan masuk ke market bila ketimpangan *Fair Prob* versus *Polymarket Price* (Edge) kurang dari batas persentase krusial.
- **Liquidity Reject:** Jika Market Maker Polymarket sengaja melebarkan selisih Bid-Ask (>7%), bot mendeteksi bahaya dan `LIQUIDITY_BLOCK` (Abstain).
- **Dry-Run Live Gate:** *Sistem tidak merelakan uang sungguhan hingga simulasi Dry-Run minimal telah melewati 10 trade simulasi dengan Win-Rate positif*.

---

## 5. Deployment Serverless & Telemetry (Railway)

Dirancang murni untuk *Dockerized Runtime* (mis: Railway.app):

### Konfigurasi `.env` Wajib
```env
# Credentials Opsional API
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx

# Polygon Private Key untuk tanda tangan eksekusi L2 (Polymarket)
POLYMARKET_PRIVATE_KEY=0x...

# Konfigurasi Telegram (Wajib agar CSV dikirim otomatis tiap 6 jam)
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

### Script Eksekusi Server
```bash
# Instalasi (termasuk Polars ringan, httpx, & pydantic)
pip install -r requirements.txt

# Menjalankan bot (mode Dry Run, Default)
python main.py --mode dry-run

# Mode Live Eksekusi
python main.py --mode live --confirm-live
```

---

## 6. Riset Masa Depan (The 13 Million Ticks Edge)

Di dalam repositori ini terdapat dataset emas **`dataset/btc_5m_hf_ticks.parquet`** yang berisi ~13,300,000 baris rekam jejak latensi nyata Polymarket (resolusi 100ms).
Penyempurnaan masa depan akan difokuskan untuk menyesuaikan formula *Spread Buffer* secara mutlak memanfaatkan data High-Frequency Ticks ini, memantapkan seberapa banyak milidetik tepatnya *Market Maker* Polymarket bisa dikalahkan.

---
*Dibuat dengan presisi oleh Senior Quant Engineering Team.*
