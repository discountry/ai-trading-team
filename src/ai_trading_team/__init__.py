"""AI Trading Team - Multi-agent cryptocurrency trading bot."""

__version__ = "0.1.0"

# Re-export submodules for convenient access
from ai_trading_team import agent, audit, core, data, execution, indicators, risk, strategy, ui

__all__ = [
    "__version__",
    "agent",
    "audit",
    "core",
    "data",
    "execution",
    "indicators",
    "risk",
    "strategy",
    "ui",
]
