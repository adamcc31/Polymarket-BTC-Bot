import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import sys
import math
from pathlib import Path
from datetime import datetime, timezone

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.feature_engine import FEATURE_NAMES
from src.model import _get_model_dir

MODELS_DIR = _get_model_dir()
INPUT_PARQUET = ROOT_DIR / "dataset" / "btc_5m_hf_ticks.parquet"

def train_base_model_hf():
    print(f"Loading HF dataset (Lazy): {INPUT_PARQUET}")
    
    # 1. Load Parquet with Polars (Lazy)
    q = pl.scan_parquet(INPUT_PARQUET)
    
    # Drop irrelevant columns early to save memory
    q = q.select([
        "event_slug", "datetime", "binance_price", "poly_up_bid", "poly_up_ask", 
        "up_mid", "down_mid", "up_spread", "down_spread", "expires"
    ])

    # 2. Extract Event Groups (Strike & Outcome)
    print("Determining market outcomes...")
    events_q = q.group_by("event_slug").agg([
        pl.col("binance_price").first().alias("strike"),
        pl.col("binance_price").last().alias("settlement"),
    ])
    
    events_q = events_q.with_columns(
        (pl.col("settlement") > pl.col("strike")).cast(pl.Int8).alias("label")
    )
    
    # 3. Join back to ticks
    print("Joining and computing features...")
    q = q.join(events_q.select(["event_slug", "strike", "label"]), on="event_slug")
    
    # Sort for rolling computations
    q = q.sort("datetime")

    # Basic price physics
    q = q.with_columns([
        (pl.col("binance_price").log().diff().over("event_slug").fill_null(0.0)).alias("log_return")
    ])
    
    q = q.with_columns([
        pl.col("log_return").rolling_std(window_size=100).over("event_slug").fill_null(0.001).alias("rv"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("dow")
    ])
    
    print("Collecting results (this may take time)...")
    df = q.collect()
    print(f"Collected {len(df)} processed records.")
    
    # ── TTR Features ────────────────────────────────────────────
    # expires was str in parquet, convert to datetime
    df = df.with_columns([
        pl.col("expires").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC").alias("expires_dt")
    ])
    
    # ── Assemble Features ───────────────────────────────────────
    # Features produced must match FEATURE_NAMES in src.feature_engine
    # Current list from config: VAM, RV, vol_percentile, depth_ratio, price_vs_ema20, 
    # binance_spread_bps, hour_sin/cos, dow_sin/cos, TTR_normalized, TTR_sin, TTR_cos, 
    # strike_distance_pct, contest_urgency, ttr_x_obi, ttr_x_tfm, ttr_x_strike, 
    # clob_yes_mid, clob_yes_spread, clob_no_spread, market_vig
    
    print("Finalizing feature matrix (Polars-native)...")
    
    # Pre-calculate rolling rank in Polars
    df = df.with_columns([
        (pl.col("rv").rolling_rank(window_size=500) / 500.0).alias("vol_percentile"),
        (pl.col("binance_price") - pl.col("strike")).alias("dist_raw")
    ])

    final_df = df.select([
        (pl.col("log_return") / (pl.col("rv") + 1e-8)).alias("VAM"),
        (pl.col("rv") * math.sqrt(252 * 24 * 60 * 60)).alias("RV"),
        pl.col("vol_percentile"),
        (pl.col("poly_up_bid") / pl.col("poly_up_ask")).fill_null(1.0).alias("depth_ratio"),
        pl.lit(0.0).alias("price_vs_ema20"),
        ((pl.col("poly_up_ask") - pl.col("poly_up_bid")) * 10000 / 0.5).alias("binance_spread_bps"),
        (pl.col("hour") * 2 * math.pi / 24.0).sin().alias("hour_sin"),
        (pl.col("hour") * 2 * math.pi / 24.0).cos().alias("hour_cos"),
        (pl.col("dow") * 2 * math.pi / 7.0).sin().alias("dow_sin"),
        (pl.col("dow") * 2 * math.pi / 7.0).cos().alias("dow_cos"),
        pl.lit(0.5).alias("TTR_normalized"),
        pl.lit(1.0).alias("TTR_sin"),
        pl.lit(0.0).alias("TTR_cos"),
        (pl.col("dist_raw") / (pl.col("strike") + 1e-8) * 100.0).alias("strike_distance_pct"),
        pl.lit(0.0).alias("contest_urgency"),
        pl.lit(0.0).alias("ttr_x_obi"),
        pl.lit(0.0).alias("ttr_x_tfm"),
        pl.lit(0.0).alias("ttr_x_strike"),
        pl.col("up_mid").alias("clob_yes_mid"),
        pl.col("up_spread").alias("clob_yes_spread"),
        pl.col("down_spread").alias("clob_yes_spread_unused").alias("clob_no_spread"), # Fixed mapping from p_df
        (pl.col("up_mid") + pl.col("down_mid") - 1.0).alias("market_vig"),
        pl.col("label")
    ])

    # Convert to numpy for LGBM (Polars to numpy doesn't require pyarrow)
    X_np = final_df.select(FEATURE_NAMES).to_numpy()
    y_np = final_df.select("label").to_numpy().flatten()

    print(f"Training on {len(X_np)} samples with {len(FEATURE_NAMES)} features.")

    # 5. Train LGBM
    # Parameters for HFT ticks (higher depth, lower learning rate)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": 7
    }

    train_data = lgb.Dataset(X_np, label=y_np)
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # 6. Save Artifacts
    # We save as foundational version tag
    version_tag = f"vHF_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"model_lgbm_{version_tag}.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    # Symlink/copy to base name for easy loading
    latest_path = MODELS_DIR / "base_model_v1.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(model, f)

    print(f"SUCCESS: Base model saved to {latest_path}")
    print(f"Version: {version_tag}")

if __name__ == "__main__":
    train_base_model_hf()
