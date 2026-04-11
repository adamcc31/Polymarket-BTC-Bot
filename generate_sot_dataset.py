import pandas as pd
import numpy as np
import aiohttp
import asyncio
import time
from typing import Dict, List

# Configuration
INPUT_CSV = "dataset/dry_run_11042026.csv"
OUTPUT_CSV = "sot_dataset_ready.csv"
API_BASE_URL = "https://clob.polymarket.com/markets"
CONCURRENCY_LIMIT = 10

async def fetch_market_outcome(session: aiohttp.ClientSession, market_id: str, semaphore: asyncio.Semaphore) -> Dict:
    """Fetch official market outcome from Polymarket CLOB API."""
    async with semaphore:
        url = f"{API_BASE_URL}/{market_id}"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    tokens = data.get("tokens", [])
                    # Winning index is the index of the token where winner is True
                    for i, token in enumerate(tokens):
                        if token.get("winner") is True:
                            return {"market_id": market_id, "winning_index": i}
                    return {"market_id": market_id, "winning_index": None} # Not resolved?
                else:
                    print(f"Error fetching {market_id}: status {response.status}")
                    return {"market_id": market_id, "winning_index": None}
        except Exception as e:
            print(f"Exception for {market_id}: {e}")
            return {"market_id": market_id, "winning_index": None}

async def reconcile_outcomes(market_ids: List[str]) -> Dict[str, int]:
    """Batch fetch outcomes for all unique market IDs."""
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_market_outcome(session, mid, semaphore) for mid in market_ids]
        results = await asyncio.gather(*tasks)
    
    return {r["market_id"]: r["winning_index"] for r in results if r["winning_index"] is not None}

def main():
    print("Starting SOT Dataset Generation Pipeline...")
    start_time = time.time()

    # STEP 1: Ingestion
    print(f"Loading raw data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    original_row_count = len(df)
    
    unique_markets = df["market_id"].unique().tolist()
    print(f"Found {len(unique_markets)} unique markets to reconcile.")

    # STEP 0: Oracle Truth Reconciliation
    print("Fetching official outcomes from Polymarket API...")
    loop = asyncio.get_event_loop()
    outcomes_map = loop.run_until_complete(reconcile_outcomes(unique_markets))
    
    print(f"Successfully reconciled {len(outcomes_map)}/{len(unique_markets)} markets.")

    # Apply corrections
    def get_corrected_outcome(row):
        mid = row["market_id"]
        if mid not in outcomes_map:
            return row["outcome"] # Fallback if API failed
        
        winning_idx = outcomes_map[mid]
        # Mapping: BUY_YES -> Index 0, BUY_NO -> Index 1
        signal_idx = 0 if "YES" in row["signal_type"].upper() or "UP" in row["signal_type"].upper() else 1
        return "WIN" if signal_idx == winning_idx else "LOSS"

    df["outcome"] = df.apply(get_corrected_outcome, axis=1)

    # Recalculate PnL based on Oracle truth
    def recalculate_pnl(row):
        bet = row["bet_size_usd"]
        price = row["entry_price_usdc"]
        if row["outcome"] == "WIN":
            # PnL = (Bet / Price) - Bet
            return (bet / price) - bet
        else:
            return -bet

    df["pnl_usd"] = df.apply(recalculate_pnl, axis=1)

    # STEP 1: Slippage & Liquidity Penalty Injection
    print("Injecting liquidity slippage penalties...")
    # implied_slippage = min((bet_size_usd / 50.0) * 0.015, 0.10)
    df["implied_slippage"] = np.minimum((df["bet_size_usd"] / 50.0) * 0.015, 0.10)
    # Adjust PnL: pnl = pnl - (bet * slippage)
    df["pnl_usd"] = df["pnl_usd"] - (df["bet_size_usd"] * df["implied_slippage"])

    # STEP 2: Market-Level Aggregation (VWAP)
    print("Aggregating ticks into market-level predictive rows...")
    
    # Pre-calculate components for VWAP: sum(price * size) / sum(size)
    df["weighted_price"] = df["entry_price_usdc"] * df["bet_size_usd"]
    
    agg_funcs = {
        "bet_size_usd": "sum",
        "pnl_usd": "sum",
        "weighted_price": "sum",
        "P_model": "mean",
        "live_edge": "mean",
        "TTR_minutes": "mean",
        "btc_price_at_trigger": "first", # Needed for distance normalization
        "btc_distance_to_strike": "last",
        "outcome": "first" # Asserted to be same per group anyway
    }

    # Signal type acts as a unique decision ID per market
    df_agg = df.groupby(["market_id", "signal_type"]).agg(agg_funcs).reset_index()
    
    # Finalize VWAP
    df_agg["vwap_entry_price"] = df_agg["weighted_price"] / df_agg["bet_size_usd"]
    
    # Rename columns for clarity
    df_agg = df_agg.rename(columns={
        "bet_size_usd": "total_bet_usd",
        "pnl_usd": "net_pnl_usd",
        "P_model": "avg_P_model",
        "live_edge": "avg_live_edge",
        "TTR_minutes": "avg_TTR_minutes"
    })

    # STEP 3: Feature Engineering & Normalization
    print("Engineering features and target variables...")
    
    # target_win
    df_agg["target_win"] = (df_agg["outcome"] == "WIN").astype(int)
    
    # distance_to_strike_bps: (dist / trigger_price) * 10000
    df_agg["distance_to_strike_bps"] = (df_agg["btc_distance_to_strike"] / df_agg["btc_price_at_trigger"]) * 10000
    
    # is_coinflip: P_model in [0.4, 0.6]
    df_agg["is_coinflip"] = (df_agg["avg_P_model"] >= 0.40) & (df_agg["avg_P_model"] <= 0.60)

    # Final Cleanup
    final_columns = [
        "market_id", "signal_type", "total_bet_usd", "net_pnl_usd", 
        "vwap_entry_price", "avg_P_model", "avg_live_edge", 
        "avg_TTR_minutes", "distance_to_strike_bps", "is_coinflip", "target_win"
    ]
    df_final = df_agg[final_columns]
    
    # STEP 4: Output
    df_final.to_csv(OUTPUT_CSV, index=False)
    aggregated_row_count = len(df_final)

    # Summary Statistics
    print("\n" + "="*40)
    print("SOT GENERATION SUMMARY")
    print("="*40)
    print(f"Original Row Count (Ticks):    {original_row_count}")
    print(f"Aggregated Row Count (Markets): {aggregated_row_count}")
    print(f"Compression Ratio:              {original_row_count/aggregated_row_count:.2f}x")
    print("-" * 20)
    
    orig_win_rate = (df["outcome"] == "WIN").mean()
    new_win_rate = df_final["target_win"].mean()
    print(f"Raw Win Rate:                   {orig_win_rate:.2%}")
    print(f"SOT Win Rate (Post-Penalty):    {new_win_rate:.2%}")
    print(f"Net Realized PnL:               ${df_final['net_pnl_usd'].sum():.2f}")
    
    end_time = time.time()
    print(f"SOT Dataset saved to {OUTPUT_CSV}")
    print(f"Finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
