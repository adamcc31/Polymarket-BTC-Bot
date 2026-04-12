import sys
import os
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModelEnsemble
from src.config_manager import ConfigManager
from src.schemas import FeatureMetadata
from src.feature_engine import FEATURE_NAMES

def run_test():
    print("Initializing ModelEnsemble...")
    config = ConfigManager()
    model = ModelEnsemble(config)
    
    print("Loading latest model...")
    loaded = model.load_latest()
    print(f"Model Stack Loaded: {loaded}")
    print(f"Is Meta-Brain v2 Active: {model._has_meta_v2}")
    
    if not model._has_meta_v2:
        print("ERROR: Meta-Brain v2 could not be loaded. Please check data/models/ directory.")
        sys.exit(1)
        
    print(f"Constructing mock feature vector (Length: {len(FEATURE_NAMES)})...")
    feature_vector = np.zeros(len(FEATURE_NAMES))
    
    clob_yes_mid_idx = FEATURE_NAMES.index('clob_yes_mid')
    clob_yes_spread_idx = FEATURE_NAMES.index('clob_yes_spread')
    
    feature_vector[clob_yes_mid_idx] = 0.55  # mid_YES
    feature_vector[clob_yes_spread_idx] = 0.05  # spread_YES
    
    print("Constructing mock metadata...")
    metadata = FeatureMetadata(
        timestamp=datetime.now(timezone.utc),
        bar_close_time=datetime.now(timezone.utc),
        market_id="test_market_1",
        strike_price=66180.25,
        current_btc_price=66224.41,
        TTR_minutes=1.68,
        TTR_phase="ENTRY_WINDOW",
        clob_ask=0.575,
        compute_lag_ms=10.0
    )
    
    print("Running inference through predict()...")
    p_yes = model.predict(feature_vector, metadata)
    
    print(f"\n=======================")
    print(f"Meta-Brain v2 Output: {p_yes:.4f}")
    print(f"=======================\n")
    print("SUCCESS: Inference mathematically flowed through new V2 models without crashing!")

if __name__ == "__main__":
    run_test()
