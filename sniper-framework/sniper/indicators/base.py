"""Base indicator class."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Indicator(ABC):
    """
    Base indicator class. All indicators inherit from this.
    Indicators can depend on other indicators (chaining).
    """

    def __init__(self, name: str):
        self.name = name
        self.values: list[float] = []
        self.dependencies: list["Indicator"] = []

    @abstractmethod
    def compute(self, data: pd.Series) -> float:
        """Compute indicator value for one bar."""
        pass

    def update(self, data: pd.Series) -> float:
        """Update indicator with new bar. Returns computed value."""
        value = self.compute(data)
        self.values.append(value)
        return value

    def __getitem__(self, idx: int) -> float:
        """Allow indexing: indicator[-1] = most recent value."""
        return self.values[idx]

    def add_dependency(self, indicator: "Indicator") -> None:
        """Add another indicator as dependency."""
        self.dependencies.append(indicator)

    def reset(self) -> None:
        """Reset indicator state."""
        self.values = []
