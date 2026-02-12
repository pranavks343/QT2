"""HMM-based regime detection."""

from typing import Any

import numpy as np
import pandas as pd

from sniper.regime.simple_detector import SimpleRegimeDetector


class HMMRegimeDetector:
    """
    HMM-based regime detection. Uses hmmlearn if available.
    Falls back to SimpleRegimeDetector if hmmlearn not installed.
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self._model = None
        self._fallback = SimpleRegimeDetector()
        self._last_regime: str = "normal"
        self._returns_buffer: list[float] = []
        self._min_samples = 50
        self._prev_close: float | None = None

        try:
            from hmmlearn import hmm
            self._hmm = hmm
            self._has_hmm = True
        except ImportError:
            self._has_hmm = False

    def detect(self, data: pd.Series | pd.DataFrame) -> str:
        """
        Detect regime. Returns 'low_vol', 'normal', or 'high_vol'.
        """
        close = self._extract_close(data)
        if close is None:
            return self._last_regime

        if self._prev_close is not None:
            ret = np.log(close / self._prev_close)
            self._returns_buffer.append(ret)
        self._prev_close = close

        if len(self._returns_buffer) > 500:
            self._returns_buffer = self._returns_buffer[-500:]

        if not self._has_hmm or len(self._returns_buffer) < self._min_samples:
            bar = data.iloc[-1] if isinstance(data, pd.DataFrame) else data
            self._last_regime = self._fallback.detect(bar)
            return self._last_regime

        X = np.array(self._returns_buffer).reshape(-1, 1)
        try:
            if self._model is None:
                self._model = self._hmm.GaussianHMM(n_components=self.n_states, covariance_type="full")
            self._model.fit(X)
            states = self._model.predict(X)
            current_state = int(states[-1])
            stds = np.sqrt(self._model.covars_.flatten())
            if current_state < len(stds):
                vol = abs(stds[current_state])
                vol_median = np.median(stds)
                if vol < vol_median * 0.7:
                    self._last_regime = "low_vol"
                elif vol > vol_median * 1.5:
                    self._last_regime = "high_vol"
                else:
                    self._last_regime = "normal"
            else:
                self._last_regime = "normal"
        except Exception:
            self._last_regime = self._fallback.get_regime()

        return self._last_regime

    def _extract_close(self, data: pd.Series | pd.DataFrame) -> float | None:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return None
            if "close" in data.columns:
                return float(data["close"].iloc[-1])
            return float(data.iloc[-1, 0])
        if isinstance(data, pd.Series):
            return float(data.get("close", 0))
        return None

    def get_regime(self) -> str:
        """Return last detected regime."""
        return self._last_regime

    def reset(self) -> None:
        """Reset detector state."""
        self._model = None
        self._returns_buffer = []
        self._prev_close = None
        self._fallback.reset()
        self._last_regime = "normal"
