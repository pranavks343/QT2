"""Deflated Sharpe Ratio - adjust for multiple testing."""

from typing import Any


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int = 1,
    n_observations: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado).
    Adjusts Sharpe for multiple strategy trials.
    """
    if n_trials <= 1:
        return sharpe
    import math
    euler = 0.5772
    var_sr = (1 + 0.5 * sharpe**2) / (n_observations - 1)
    adj = math.sqrt(var_sr * 2 * math.log(n_trials)) - euler * math.sqrt(var_sr * 2 * math.log(n_trials)) / (2 * math.log(n_trials))
    return sharpe - adj
