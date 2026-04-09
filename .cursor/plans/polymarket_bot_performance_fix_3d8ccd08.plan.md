---
name: Polymarket Bot Performance Fix
overview: "Menyusun rencana perbaikan untuk tiga isu inti: market selection tidak rasional, ABSTAIN terlalu tinggi, dan backtesting yang belum representatif. Rencana fokus pada penyelarasan horizon prediksi dengan pemilihan market, observability alasan ABSTAIN, serta pipeline evaluasi historis yang bisa dipakai untuk tuning parameter."
todos:
  - id: audit-selection-ttr
    content: Audit dan selaraskan objective market discovery dengan gate TTR agar tidak memilih market yang pasti ABSTAIN.
    status: completed
  - id: implement-rational-scoring
    content: Tambahkan scoring strike/horizon rationality pada discovery + guardrails saat rotation.
    status: completed
  - id: improve-abstain-observability
    content: Lengkapi logging semua abstain path dan agregasi reason distribution.
    status: completed
  - id: add-rotation-lock
    content: Tambahkan rotation lock (minimum dwell time + no-rotate saat market aktif masih dalam valid-entry window).
    status: completed
  - id: run-mini-dryrun-12h
    content: Jalankan mini dry-run 12 jam untuk validasi bahwa fix 1+2+3 menghasilkan trades dan reason ABSTAIN lebih sehat.
    status: completed
  - id: align-signal-source
    content: Tegaskan sumber probabilitas (fair-prob vs model) di orchestrator agar strategi sesuai ekspektasi.
    status: completed
  - id: run-backtest-matrix
    content: Jalankan matriks backtest baseline vs improved dan laporkan metrik performa utama.
    status: completed
isProject: false
---

# Rencana Perbaikan Bot Polymarket (Market Selection, Abstain, Backtest)

## Temuan Utama dari Run 12 Jam

- Dari `logs/logs.1775699859251.json`, rentang run sekitar 11.62 jam dengan event kunci: `market_discovered=1`, `market_rotated=3`, `bar_close_rotation_applied=3`, `signal_abstain_ttr=43`, `signal_generated=0`.
- Discovery awal sempat menemukan strike wajar (`strike_price=72000`), lalu rotasi pindah ke market ber-TTR sangat panjang (`new_ttr_minutes` sekitar 31k–32k), yang kemudian hampir selalu terkena gate TTR.
- Jalur sinyal saat ini di `main.py` menggunakan fair probability (`q_fair`) bukan output model ensemble, sehingga ekspektasi “prediksi 1–12 jam lalu pilih market spesifik horizon” belum ter-implementasi di orchestrator.

## Akar Masalah

- **Mismatch objective**: scoring discovery menargetkan kedekatan TTR ke `target_ttr_minutes` (default 120) dan likuiditas, bukan kedekatan strike ke harga spot + horizon prediksi.
- **Hard gate TTR terlalu sempit terhadap market yang dipilih**: sinyal hanya valid pada `signal.ttr_min_minutes..signal.ttr_max_minutes` (default 5–12), sedangkan market aktif sering ber-TTR ratusan hingga puluhan ribu menit.
- **Rotation behavior mengunci mismatch**: rotasi memilih score lebih tinggi meski mendorong ke TTR ekstrem; setelah rotasi, bar berjalan tanpa peluang signal.
- **Observability abstain belum lengkap**: beberapa jalur ABSTAIN tidak punya event log khusus (mis. `NO_TRADE_ZONE`, `BASIS_RISK_BLOCK`), sehingga root-cause agregat sulit dipantau dari Railway log saja.
- **Backtest belum representatif live**: simulator memaksa depth besar, settlement diasumsikan Binance 1m, dan belum menguji kebijakan market selection yang benar-benar dipakai saat live.

## Rencana Implementasi

### 1) Ubah Objective Market Selection agar Rasional terhadap Spot + Horizon

- File utama: [src/market_discovery.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/market_discovery.py)
- Tambahkan komponen skor baru:
  - `strike_distance_score` terhadap spot Binance saat ini (lebih dekat ke proyeksi target lebih baik).
  - `horizon_alignment_score` terhadap horizon prediksi target (mis. 1h, 4h, 8h, 12h) alih-alih hanya target TTR statis.
  - Penalti keras untuk strike absurd (contoh > X% dari spot untuk horizon pendek).
- Ubah rotasi agar hanya terjadi jika:
  - Market baru lebih baik secara total score **dan**
  - Lulus batas minimum rasionalitas strike-to-spot **dan**
  - Tidak memperburuk horizon fit secara ekstrem.

### 2) Sinkronkan Gate TTR dengan Desain Strategi Horizon

- File utama: [src/signal_generator.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/signal_generator.py), [config/config.json](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/config/config.json)
- Ganti model gate dari fixed 5–12 menit menjadi dynamic policy berbasis jenis market/horizon:
  - Contoh: untuk market 4 jam, gunakan entry window yang relevan (mis. 30–240 menit) bukan 5–12.
- Pastikan kebijakan TTR untuk signal konsisten dengan scoring discovery agar bot tidak memilih market yang sejak awal pasti di-abstain.

### 3) Turunkan Blind Spot ABSTAIN lewat Logging Terstruktur

- File utama: [src/signal_generator.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/signal_generator.py), [src/dry_run.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/dry_run.py)
- Tambah event log eksplisit untuk semua jalur abstain (`NO_TRADE_ZONE`, `BASIS_RISK_BLOCK`, dan jalur early return lain).
- Tambah ringkasan periodik reason distribution agar bisa cepat menjawab: apakah abstain dominan karena TTR, no-edge, regime, likuiditas, atau basis risk.

### 4) Tambahkan Rotation Lock agar Rotasi Tidak Chaotic

- File utama: [src/market_discovery.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/market_discovery.py), [main.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/main.py), [config/config.json](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/config/config.json)
- Tambahkan mekanisme lock rotasi:
  - `rotation.min_dwell_minutes`: setelah market menjadi aktif, rotasi dilarang sebelum dwell minimum tercapai.
  - `rotation.freeze_when_in_entry_window`: jika market saat ini masih dalam valid-entry window, rotasi diblok.
  - `rotation.cooldown_after_trade_minutes`: opsional, cegah rotasi segera setelah eksekusi trade.
- Tambahkan log reason saat rotasi dibatalkan (mis. `rotation_locked_dwell`, `rotation_locked_entry_window`) agar audit behavior mudah.

### 5) Mini Dry-Run 12 Jam sebagai Milestone Validasi

- Jalankan dry-run 12 jam setelah Item 1+2+3+4 selesai.
- Kriteria lulus milestone awal:
  - Muncul `signal_generated` dan trade nyata (tidak 0 trade).
  - Proporsi `TTR_PHASE` turun signifikan dibanding baseline run sebelumnya.
  - Tidak ada rotasi liar ke market TTR ekstrem ketika market aktif masih valid untuk entry.
- Output evaluasi minimum:
  - jumlah signal/trade,
  - breakdown abstain reason,
  - contoh rotasi yang diterima vs diblok.

### 6) Rapikan Jalur Signal Generation agar Sesuai Ekspektasi Model

- File utama: [main.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/main.py), [src/model.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/model.py)
- Putuskan satu sumber probabilitas utama:
  - Opsi A: tetap fair-prob engine sebagai alpha utama.
  - Opsi B: aktifkan model ensemble sebagai prior, lalu kombinasikan dengan fair-prob untuk kalibrasi.
- Implementasikan secara eksplisit agar klaim strategi dan behavior live konsisten.

### 7) Bangun Backtest yang Meniru Kondisi Live

- File utama: [src/sim/run.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/src/sim/run.py), [scripts/build_dataset.py](Z:/01 ADAM/00 DOKUMENTASI PROJECT/polymarket/MADE IN ABYSS/scripts/build_dataset.py)
- Tambah eksperimen baseline terstandar:
  - Baseline current config (untuk pembanding).
  - Config setelah alignment discovery+TTR.
  - Sensitivity test parameter kunci (margin, no-trade deadband, TTR window, liquidity gates).
- Laporkan metrik inti per eksperimen:
  - signal rate, abstain rate + breakdown reason,
  - trades executed, win rate, expected value/expectancy,
  - total pnl, max drawdown, profit factor, brier score.
- Catat limitasi simulasi (depth sintetis, settlement assumption) dalam laporan agar tidak overclaim.

## Urutan Prioritas Eksekusi (Revisi)

- **Prioritas 1 (blocker utama):** kerjakan Item 1 + 2 bersamaan.
- **Prioritas 2:** kerjakan Item 3 bersamaan dengan Prioritas 1 agar iterasi berikutnya terukur.
- **Prioritas 3:** implementasikan Item 4 (rotation lock).
- **Prioritas 4:** jalankan Item 5 (mini dry-run 12 jam) sebagai checkpoint wajib sebelum langkah strategis berikutnya.
- **Prioritas 5:** kerjakan Item 6 (keputusan signal source) setelah ada trade data baru.
- **Prioritas 6:** kerjakan Item 7 (backtest representatif) setelah sistem stabil.

## Kriteria Sukses

- Market terpilih berada pada strike dan horizon yang konsisten dengan konteks spot/horizon target.
- Abstain tetap selektif namun tidak didominasi mismatch TTR sistemik.
- Rotasi tidak terjadi saat market aktif masih berada dalam valid-entry window dan belum melewati minimum dwell.
- Mini dry-run 12 jam pasca-fix menghasilkan trade (bukan nol trade) dengan distribusi reason ABSTAIN yang bisa dijelaskan.
- Backtest menghasilkan perbandingan baseline vs improved yang reproducible dan bisa dijadikan dasar tuning selanjutnya.

