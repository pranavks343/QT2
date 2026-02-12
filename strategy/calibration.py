"""
SNIPER FRAMEWORK - META-LABEL CALIBRATION
Platt scaling calibration for meta-label confidence scores.

Transforms raw composite scores (0-1) into calibrated probabilities
that accurately reflect true win rates.

Example:
  - Raw score = 0.70 → Calibrated probability = 0.63
  - This means: historically, signals with score ~0.70 won 63% of the time

Uses sklearn's CalibratedClassifierCV with sigmoid method (Platt scaling).

Gracefully falls back to raw scores if:
  - sklearn not installed
  - Calibrator not yet trained
  - Insufficient training data
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Optional, Union

# Check for sklearn availability
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not installed. Install with: pip install scikit-learn\n"
        "Meta-label scores will not be calibrated."
    )


class MetaLabelCalibrator:
    """
    Calibrates meta-label confidence scores using Platt scaling.
    
    Usage:
        calibrator = MetaLabelCalibrator()
        calibrator.fit(signal_scores, outcomes)  # Train on historical data
        calibrated_prob = calibrator.calibrate(0.68)  # Transform score
        calibrator.save("models/calibrator.pkl")
    """
    
    def __init__(self, method: str = "sigmoid"):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('sigmoid' for Platt scaling)
        """
        self.method = method
        self.is_fitted = False
        self.calibrator: Optional[CalibratedClassifierCV] = None
        
        if not SKLEARN_AVAILABLE:
            self.calibrator = None
            return
        
        # Base classifier for Platt scaling
        base_clf = LogisticRegression(random_state=42, max_iter=1000)
        
        # Calibrated classifier wrapper
        self.calibrator = CalibratedClassifierCV(
            base_clf,
            method=method,
            cv=5,  # 5-fold cross-validation
        )
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        available = "available" if SKLEARN_AVAILABLE else "not available"
        return f"MetaLabelCalibrator(method={self.method}, status={status}, sklearn={available})"
    
    def fit(self, signals_df: pd.DataFrame, outcomes: Union[pd.Series, np.ndarray]) -> "MetaLabelCalibrator":
        """
        Train calibrator on historical signal scores vs actual outcomes.
        
        Args:
            signals_df: DataFrame with 'ml_score' column (raw composite scores)
            outcomes: Binary outcomes (1=win, 0=loss) or string ("win"/"loss")
        
        Returns:
            Self (for chaining)
        """
        if not SKLEARN_AVAILABLE or self.calibrator is None:
            print("  ⚠️  Calibration training skipped (sklearn not available)")
            return self
        
        # Extract scores
        if isinstance(signals_df, pd.DataFrame):
            if "ml_score" not in signals_df.columns:
                raise ValueError("signals_df must have 'ml_score' column")
            X = signals_df["ml_score"].values.reshape(-1, 1)
        else:
            X = np.array(signals_df).reshape(-1, 1)
        
        # Convert outcomes to binary (1=win, 0=loss)
        if isinstance(outcomes, pd.Series):
            y = (outcomes == "win").astype(int).values
        elif isinstance(outcomes, np.ndarray):
            if outcomes.dtype == object or outcomes.dtype.kind in ['U', 'S']:
                y = (outcomes == "win").astype(int)
            else:
                y = outcomes.astype(int)
        else:
            y = np.array(outcomes).astype(int)
        
        # Filter out NaN/invalid scores
        valid_mask = ~np.isnan(X.flatten()) & (X.flatten() > 0)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 50:
            print(f"  ⚠️  Insufficient data for calibration: {len(X)} samples (need 50+)")
            return self
        
        try:
            # Train calibrator
            self.calibrator.fit(X, y)
            self.is_fitted = True
            
            # Compute calibration quality metrics
            y_pred = self.calibrator.predict_proba(X)[:, 1]
            mae = np.mean(np.abs(y_pred - y))
            
            print(f"  ✓ Calibrator trained on {len(X)} samples")
            print(f"    Mean Absolute Error: {mae:.4f}")
            
        except Exception as e:
            print(f"  ⚠️  Calibration training failed: {e}")
            self.is_fitted = False
        
        return self
    
    def calibrate(self, ml_score: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
        """
        Transform raw score(s) into calibrated probability.
        
        Args:
            ml_score: Raw composite score (0-1) or array of scores
        
        Returns:
            Calibrated probability (0-1) or array of probabilities
        """
        if not SKLEARN_AVAILABLE or self.calibrator is None or not self.is_fitted:
            # Fallback: return raw score
            return ml_score
        
        # Handle single value
        if isinstance(ml_score, (float, int)):
            if ml_score <= 0 or np.isnan(ml_score):
                return 0.0
            X = np.array([[ml_score]])
            try:
                return float(self.calibrator.predict_proba(X)[0, 1])
            except Exception:
                return float(ml_score)
        
        # Handle array/series
        if isinstance(ml_score, pd.Series):
            scores = ml_score.values
        else:
            scores = np.array(ml_score)
        
        # Filter valid scores
        valid_mask = (scores > 0) & ~np.isnan(scores)
        result = scores.copy()
        
        if valid_mask.sum() == 0:
            return result
        
        try:
            X = scores[valid_mask].reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(X)[:, 1]
            result[valid_mask] = calibrated
        except Exception:
            pass  # Keep original scores on error
        
        return result
    
    def reliability_diagram(self, signals_df: pd.DataFrame, outcomes: Union[pd.Series, np.ndarray],
                           n_bins: int = 10) -> pd.DataFrame:
        """
        Generate reliability diagram data: predicted vs actual win rates.
        
        Args:
            signals_df: DataFrame with 'ml_score' column
            outcomes: Binary outcomes (1=win, 0=loss)
            n_bins: Number of probability bins
        
        Returns:
            DataFrame with columns: bin_center, predicted_prob, actual_win_rate, count
        """
        # Extract scores and outcomes
        if isinstance(signals_df, pd.DataFrame):
            scores = signals_df["ml_score"].values
        else:
            scores = np.array(signals_df)
        
        if isinstance(outcomes, pd.Series):
            y = (outcomes == "win").astype(int).values
        else:
            y = np.array(outcomes).astype(int)
        
        # Filter valid
        valid_mask = (scores > 0) & ~np.isnan(scores)
        scores = scores[valid_mask]
        y = y[valid_mask]
        
        if len(scores) == 0:
            return pd.DataFrame()
        
        # Calibrate scores if fitted
        if self.is_fitted and SKLEARN_AVAILABLE:
            calibrated = self.calibrate(scores)
        else:
            calibrated = scores
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        result = []
        for i in range(n_bins):
            mask = (calibrated >= bins[i]) & (calibrated < bins[i + 1])
            if mask.sum() > 0:
                result.append({
                    "bin_center": bin_centers[i],
                    "predicted_prob": calibrated[mask].mean(),
                    "actual_win_rate": y[mask].mean(),
                    "count": mask.sum(),
                })
        
        return pd.DataFrame(result)
    
    def save(self, path: str) -> None:
        """Save trained calibrator to disk."""
        if not SKLEARN_AVAILABLE or self.calibrator is None or not self.is_fitted:
            print("  ⚠️  Cannot save: calibrator not fitted or sklearn not available")
            return
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump({
                "calibrator": self.calibrator,
                "method": self.method,
                "is_fitted": self.is_fitted,
            }, path)
            print(f"  ✓ Calibrator saved to {path}")
        except Exception as e:
            print(f"  ✗ Failed to save calibrator: {e}")
    
    @classmethod
    def load(cls, path: str) -> "MetaLabelCalibrator":
        """Load trained calibrator from disk."""
        if not SKLEARN_AVAILABLE:
            print("  ⚠️  Cannot load: sklearn not available")
            return cls()
        
        try:
            data = joblib.load(path)
            calibrator = cls(method=data["method"])
            calibrator.calibrator = data["calibrator"]
            calibrator.is_fitted = data["is_fitted"]
            print(f"  ✓ Calibrator loaded from {path}")
            return calibrator
        except Exception as e:
            print(f"  ⚠️  Failed to load calibrator: {e}")
            print("  → Creating new calibrator")
            return cls()


# Singleton instance
_calibrator = None

def get_calibrator(force_new: bool = False) -> MetaLabelCalibrator:
    """
    Get or create the global calibrator instance.
    
    Args:
        force_new: Create a new calibrator instead of using cached one
    
    Returns:
        MetaLabelCalibrator instance
    """
    global _calibrator
    
    if _calibrator is None or force_new:
        # Try to load from disk first
        model_path = "models/calibrator.pkl"
        if os.path.exists(model_path) and not force_new:
            _calibrator = MetaLabelCalibrator.load(model_path)
        else:
            _calibrator = MetaLabelCalibrator()
    
    return _calibrator
