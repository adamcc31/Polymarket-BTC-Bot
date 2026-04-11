import sys
import os
import asyncio
import logging
from datetime import datetime, timezone

# 1. Setup path to use the patched version in scratch
project_root = os.getcwd()
patch_dir = os.path.join(project_root, "scratch", "feature_live_mode_patches")
sys.path.insert(0, patch_dir)
sys.path.insert(1, project_root)

# Verify we're loading the patched version
import market_discovery
print(f"Loaded market_discovery from: {market_discovery.__file__}")

from src.config_manager import ConfigManager
import structlog

# Setup structlog to match the bot's logging style
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

async def verify_loop():
    config = ConfigManager.get_instance()
    
    # Ensure dynamic 5m discovery is enabled in config for this test
    # We don't want to modify the config file on disk, so we use a mock if possible
    # but ConfigManager is a singleton. Let's see if we can just set values.
    config._config["market_discovery"]["dynamic_5m_event_slugs"] = ["btc-updown-5m"]
    config._config["market_discovery"]["poll_interval_s"] = 10
    
    discovery = market_discovery.MarketDiscovery(config)
    
    print("Starting discovery loop... Waiting for a dynamic 5m market to be found.")
    print("Example log sought: dynamic_5m_candidate_found")
    
    found = False
    for i in range(100): # Run for max ~15 minutes
        print(f"Polling iteration {i+1} at {datetime.now(timezone.utc).isoformat()}")
        
        # We call the private method directly or the candidate query
        candidates = await discovery._query_candidates()
        
        if any(c.get("source", "").startswith("event:") for c in candidates):
            print("\nSUCCESS: Found dynamic 5m candidate!")
            found = True
            break
            
        await asyncio.sleep(10)
    
    if not found:
        print("\nFAILED: No dynamic 5m market found after 100 iterations.")

if __name__ == "__main__":
    asyncio.run(verify_loop())
