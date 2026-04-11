import sys
import os
from pathlib import Path
import numpy as np
import pickle
from datetime import datetime, timezone

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model import ModelEnsemble
from src.config_manager import ConfigManager
from src.schemas import FeatureMetadata

def test_stacking_inference():
    print("--- Testing Stacking Ensemble Inference ---")
    config = ConfigManager.get_instance()
    model = ModelEnsemble(config)
    
    # Ensure models directory exists for the test
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Try to load models
    print("Loading models...")
    success = model.load_latest()
    
    if not success:
        print("INFO: No base model found in models/. Testing with neutral fallback.")
        # We'll skip the real prediction logic if no model at all, but we've verified the code loads.
    else:
        print(f"Base Model version: {model.version}")
        print(f"Meta-Brain Status: {'LOADED' if model._has_meta_brain else 'NOT FOUND (Fallback Mode)'}")

    # 2. Mock Feature Vector (24 features)
    fv = np.random.rand(24)
    
    # 3. Mock Metadata
    metadata = FeatureMetadata(
        timestamp=datetime.now(timezone.utc),
        bar_close_time=datetime.now(timezone.utc),
        market_id="test_market",
        strike_price=60000.0,
        current_btc_price=60500.0,
        TTR_minutes=4.5,
        TTR_phase="ENTRY_WINDOW",
        clob_ask=0.52,
        compute_lag_ms=10.0
    )

    # 4. Perform Prediction
    print("Executing predict()...")
    prob = model.predict(fv, metadata=metadata)
    
    print("\n--- Inference Result ---")
    print(f"Input Strike:    {metadata.strike_price}")
    print(f"Input Price:     {metadata.current_btc_price}")
    print(f"Output P(YES):   {prob:.6f}")
    
    assert 0.0 <= prob <= 1.0, "Probability out of bounds!"
    print("\nVerification Successful: Inference pipeline is functional.")

if __name__ == "__main__":
    test_stacking_inference()
