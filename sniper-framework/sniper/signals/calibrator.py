"""Platt scaling for probability calibration."""

from typing import Any


class Calibrator:
    """
    Platt scaling stub for probability calibration.
    Maps raw scores to calibrated probabilities.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, scores: list[float], labels: list[int]) -> None:
        """Fit calibrator on (score, label) pairs."""
        self._fitted = len(scores) > 0

    def calibrate(self, score: float) -> float:
        """Map raw score to calibrated probability."""
        return max(0.0, min(1.0, score))
