"""Historical stress scenarios."""

SCENARIOS = {
    "2008_crash": {"name": "2008 Financial Crisis", "drawdown": 0.50},
    "covid_2020": {"name": "COVID-19 March 2020", "drawdown": 0.35},
    "flash_crash": {"name": "Flash Crash", "drawdown": 0.15},
}


def get_scenario(name: str) -> dict:
    """Get scenario by name."""
    return SCENARIOS.get(name, {})


def list_scenarios() -> list[str]:
    """List scenario names."""
    return list(SCENARIOS.keys())
