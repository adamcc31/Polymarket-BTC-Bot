import threading
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import structlog # type: ignore
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class ShadowTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, Dict[str, Any]] = {}

    def on_candidate_found(
        self,
        timestamp: str,
        slug: str,
        market_id: str,
        strike_price: Optional[float],
        ttr_minutes: float,
        yes_prob: Optional[float],
    ) -> None:
        with self._lock:
            if market_id not in self._records:
                self._records[market_id] = {
                    "timestamp_found": timestamp,
                    "slug": slug,
                    "market_id": market_id,
                    "strike_price": strike_price,
                    "ttr_minutes_found": ttr_minutes,
                    "market_yes_prob": yes_prob,
                    "status": "EVALUATING",
                    "abstain_reason": None,
                    "live_edge": None,
                    "execution_price": None,
                }

    def on_trade_aborted(
        self,
        market_id: str,
        reason: Optional[str] = None,
        live_edge: Optional[float] = None,
    ) -> None:
        with self._lock:
            record = self._records.get(market_id)
            if not record:
                return
            
            if record["status"] != "EXECUTED":
                record["status"] = "ABSTAINED"
                if reason:
                    record["abstain_reason"] = reason
                if live_edge is not None:
                    record["live_edge"] = live_edge

    def on_trade_executed(
        self,
        market_id: str,
        entry_price: float,
    ) -> None:
        with self._lock:
            record = self._records.get(market_id)
            if not record:
                return
                
            record["status"] = "EXECUTED"
            record["execution_price"] = entry_price

    def export_to_csv(self, export_dir: Path) -> Optional[Path]:
        """Export accumulated shadow book records to a CSV file."""
        with self._lock:
            if not self._records:
                return None
                
            rows = [rec for rec in self._records.values() if rec.get("slug")]
            
        if not rows:
            return None
            
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            path = export_dir / "shadow_book_report.csv"
            
            fields = [
                "timestamp_found", "slug", "market_id", "strike_price", 
                "ttr_minutes_found", "market_yes_prob", "status", 
                "abstain_reason", "live_edge", "execution_price"
            ]
            
            df = pd.DataFrame(rows, columns=fields)
            df.to_csv(path, index=False)
            logger.info("shadow_book_exported", path=str(path), records=len(rows))
            return path
        except Exception as e:
            logger.error("shadow_book_export_failed", error=str(e))
            return None

    def clear_exported(self) -> None:
        """Clear memory safely."""
        with self._lock:
            self._records.clear()
            logger.info("shadow_book_memory_cleared")

shadow_tracker = ShadowTracker()
