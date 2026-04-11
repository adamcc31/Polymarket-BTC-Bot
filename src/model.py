"""
model.py — LightGBM + Logistic Regression ensemble with calibration.

Architecture:
  Primary   : LightGBM Classifier
  Secondary : Logistic Regression (baseline, overfit detection)
  Ensemble  : P_model = 0.7 * LGBM_calibrated + 0.3 * LogReg_prob

CRITICAL (from validation):
  LightGBM outputs are NOT well-calibrated probabilities by default.
  Isotonic Regression calibration is MANDATORY for Kelly sizing accuracy.
  Without calibration, edge calculations and Kelly fractions will be destructive.
"""

from __future__ import annotations

import glob
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.config_manager import ConfigManager

logger = structlog.get_logger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"
_DATA_DIR = Path(__file__).parent.parent / "data" / "models"


def _get_model_dir() -> Path:
    """Get model directory — fallback to data/models for Railway."""
    for d in [_DATA_DIR, _MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR if _DATA_DIR.exists() else _MODELS_DIR


class ModelEnsemble:
    """
    LightGBM + LogReg ensemble with Isotonic Regression calibration.

    The ensemble outputs calibrated P(YES outcome) in [0, 1].
    Calibration is performed post-training using isotonic regression
    because tree-based models produce biased probability estimates.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        
        # --- Base Model (24 features) ---
        self._lgbm_model = None
        self._logreg_model = None
        self._scaler = None
        self._calibrator = None
        
        # --- Meta-Brain (5 features) ---
        self._meta_lgbm = None
        self._meta_logreg = None
        self._meta_scaler = None
        self._meta_calibrator_isotonic = None

        self._lgbm_weight = config.get("model.ensemble_lgbm_weight", 0.70)
        self._logreg_weight = config.get("model.ensemble_logreg_weight", 0.30)
        self._calibration_method = config.get("model.calibration_method", "isotonic")
        self._version: Optional[str] = None
        self._is_loaded = False
        self._has_meta_brain = False

    # ── Public Interface ──────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def version(self) -> str:
        return self._version or "none"

    def predict(self, feature_vector: np.ndarray, metadata: Any = None) -> float:
        """
        Predict P(YES outcome) using STACKED ENSEMBLE architecture.

        Stage 1: Base Model (24 features) -> raw_p
        Stage 2: Meta-Brain (5 features) -> final_p

        Args:
            feature_vector: numpy array of shape (1, 24)
            metadata: FeatureMetadata object for context (Stage 2 features)

        Returns:
            P_model: calibrated probability ∈ [0, 1]
        """
        if not self._is_loaded:
            logger.warning("model_not_loaded_returning_neutral")
            return 0.5

        try:
            X = np.array(feature_vector).reshape(1, -1)

            # --- STAGE 1: Base Model Inference ---
            # LightGBM prediction
            if self._calibrator is not None:
                lgbm_prob = self._calibrator.predict_proba(X)[0, 1]
            else:
                lgbm_prob = self._lgbm_model.predict_proba(X)[0, 1]

            # LogReg prediction
            if self._logreg_model is not None and self._scaler is not None:
                X_scaled = self._scaler.transform(X)
                logreg_prob = self._logreg_model.predict_proba(X_scaled)[0, 1]
            else:
                logreg_prob = lgbm_prob  # Fallback

            # Weighted base probability (raw_p)
            raw_p = (self._lgbm_weight * lgbm_prob + self._logreg_weight * logreg_prob)
            raw_p = max(0.0, min(1.0, raw_p))

            # If Meta-Brain is not available or metadata missing, return Base Model output (backwards compatibility)
            if not self._has_meta_brain or metadata is None:
                return float(raw_p)

            # --- STAGE 2: Meta-Brain Inference ---
            try:
                # 1. distance_to_strike_bps: 1% = 100 BPS
                dist_bps = (metadata.current_btc_price - metadata.strike_price) / metadata.strike_price * 10000.0
                
                # 2. is_coinflip: raw_p in [0.4, 0.6]
                is_coinflip = 1.0 if 0.40 <= raw_p <= 0.60 else 0.0
                
                # 3. live_edge = raw_p - current_ask
                # We use clob_ask from metadata which main.py should populate via feature_engine
                clob_ask = getattr(metadata, "clob_ask", 0.5)
                live_edge = raw_p - clob_ask

                # Construct Meta-Feature Vector (matches retrain_meta_calibrator.py FEATURES)
                meta_X = np.array([
                    dist_bps,
                    metadata.TTR_minutes,
                    raw_p,
                    live_edge,
                    is_coinflip
                ]).reshape(1, -1)

                # Meta-Ensemble Predictions
                meta_lgbm_prob = self._meta_lgbm.predict(meta_X)[0]
                
                meta_X_scaled = self._meta_scaler.transform(meta_X)
                meta_logreg_prob = self._meta_logreg.predict_proba(meta_X_scaled)[0, 1]
                
                meta_p = (0.7 * meta_lgbm_prob + 0.3 * meta_logreg_prob)
                
                # Meta-Isotonic Calibration
                final_p = self._meta_calibrator_isotonic.transform([meta_p])[0]
                final_p = max(0.0, min(1.0, final_p))

                logger.debug(
                    "stacked_inference_complete",
                    raw_p=round(raw_p, 4),
                    final_p=round(final_p, 4),
                    dist_bps=round(dist_bps, 1)
                )

                return float(final_p)

            except Exception as meta_e:
                logger.error("meta_inference_failed_using_base", error=str(meta_e))
                return float(raw_p)

        except Exception as e:
            logger.error("stacked_predict_error", error=str(e), exc_info=True)
            return 0.5

    # ── Model Loading ─────────────────────────────────────────

    def load_latest(self) -> bool:
        """Load the latest model version from disk."""
        model_dir = _get_model_dir()

        # Find latest LGBM model file
        lgbm_files = sorted(
            glob.glob(str(model_dir / "model_lgbm_v*.pkl")),
            reverse=True,
        )

        if not lgbm_files:
            logger.warning("no_model_files_found", dir=str(model_dir))
            return False

        latest_lgbm = Path(lgbm_files[0])
        version = latest_lgbm.stem.replace("model_lgbm_", "")

        return self.load_version(version)

    def load_version(self, version: str) -> bool:
        """Load a specific model version."""
        model_dir = _get_model_dir()

        lgbm_path = model_dir / f"model_lgbm_{version}.pkl"
        logreg_path = model_dir / f"model_logreg_{version}.pkl"
        scaler_path = model_dir / f"scaler_{version}.pkl"
        calibrator_path = model_dir / f"calibrator_{version}.pkl"
        
        # Meta-Brain paths (using specific names from training script)
        meta_lgbm_path = model_dir / "meta_lgbm.pkl"
        meta_logreg_path = model_dir / "meta_logreg.pkl"
        meta_scaler_path = model_dir / "meta_scaler.pkl"
        meta_calibrator_path = model_dir / "meta_calibrator_isotonic.pkl"

        try:
            # 1. Load Base Model (required)
            if not lgbm_path.exists():
                logger.error("lgbm_model_not_found", path=str(lgbm_path))
                return False

            with open(lgbm_path, "rb") as f:
                self._lgbm_model = pickle.load(f)

            if logreg_path.exists():
                with open(logreg_path, "rb") as f:
                    self._logreg_model = pickle.load(f)

            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)

            if calibrator_path.exists():
                with open(calibrator_path, "rb") as f:
                    self._calibrator = pickle.load(f)

            # 2. Load Meta-Brain (optional)
            if all(p.exists() for p in [meta_lgbm_path, meta_logreg_path, meta_scaler_path, meta_calibrator_path]):
                with open(meta_lgbm_path, "rb") as f:
                    self._meta_lgbm = pickle.load(f)
                with open(meta_logreg_path, "rb") as f:
                    self._meta_logreg = pickle.load(f)
                with open(meta_scaler_path, "rb") as f:
                    self._meta_scaler = pickle.load(f)
                with open(meta_calibrator_path, "rb") as f:
                    self._meta_calibrator_isotonic = pickle.load(f)
                
                self._has_meta_brain = True
                logger.info("meta_brain_loaded_stacking_enabled")
            else:
                logger.warning("meta_brain_not_found_stacking_disabled")

            self._version = version
            self._is_loaded = True

            logger.info(
                "model_loaded",
                version=version,
                has_logreg=self._logreg_model is not None,
                has_calibrator=self._calibrator is not None,
            )
            return True

        except Exception as e:
            logger.error("model_load_error", version=version, error=str(e))
            return False

    # ── Model Saving ──────────────────────────────────────────

    @staticmethod
    def save_models(
        lgbm_model: Any,
        logreg_model: Any,
        scaler: Any,
        calibrator: Any,
        version_tag: str,
        metrics: Optional[Dict] = None,
    ) -> Path:
        """Save model artifacts to disk."""
        model_dir = _get_model_dir()

        with open(model_dir / f"model_lgbm_{version_tag}.pkl", "wb") as f:
            pickle.dump(lgbm_model, f)

        with open(model_dir / f"model_logreg_{version_tag}.pkl", "wb") as f:
            pickle.dump(logreg_model, f)

        with open(model_dir / f"scaler_{version_tag}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        if calibrator is not None:
            with open(model_dir / f"calibrator_{version_tag}.pkl", "wb") as f:
                pickle.dump(calibrator, f)

        if metrics:
            import json
            with open(model_dir / f"training_metrics_{version_tag}.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)

        logger.info("models_saved", version=version_tag, dir=str(model_dir))

        # Cleanup: keep only N latest versions
        _cleanup_old_versions(model_dir, keep_n=3)

        return model_dir

    # ── Rollback ──────────────────────────────────────────────

    def rollback(self) -> bool:
        """Rollback to previous model version."""
        model_dir = _get_model_dir()
        lgbm_files = sorted(
            glob.glob(str(model_dir / "model_lgbm_v*.pkl")),
            reverse=True,
        )

        if len(lgbm_files) < 2:
            logger.error("no_previous_version_for_rollback")
            return False

        prev_file = Path(lgbm_files[1])
        prev_version = prev_file.stem.replace("model_lgbm_", "")
        logger.info("rolling_back_model", to_version=prev_version)
        return self.load_version(prev_version)

    # ── Overfit Detection ─────────────────────────────────────

    def check_overfit_divergence(self, X: np.ndarray) -> Optional[float]:
        """
        Check divergence between LGBM and LogReg predictions.
        Large divergence suggests LGBM may be overfitting.
        """
        if not self._is_loaded or self._logreg_model is None:
            return None

        try:
            lgbm_probs = self._lgbm_model.predict_proba(X)[:, 1]
            X_scaled = self._scaler.transform(X)
            logreg_probs = self._logreg_model.predict_proba(X_scaled)[:, 1]
            mae = float(np.mean(np.abs(lgbm_probs - logreg_probs)))
            return mae
        except Exception:
            return None


def _cleanup_old_versions(model_dir: Path, keep_n: int = 3) -> None:
    """Keep only the N latest model versions, delete older ones."""
    versions = set()
    for f in model_dir.glob("model_lgbm_v*.pkl"):
        version = f.stem.replace("model_lgbm_", "")
        versions.add(version)

    sorted_versions = sorted(versions, reverse=True)
    to_delete = sorted_versions[keep_n:]

    for v in to_delete:
        for pattern in [
            f"model_lgbm_{v}.pkl",
            f"model_logreg_{v}.pkl",
            f"scaler_{v}.pkl",
            f"calibrator_{v}.pkl",
            f"training_metrics_{v}.json",
        ]:
            path = model_dir / pattern
            if path.exists():
                path.unlink()
                logger.info("old_model_deleted", file=str(path))
