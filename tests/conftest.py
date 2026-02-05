"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing indicators."""
    return [
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000.0},
        {"open": 104.0, "high": 106.0, "low": 102.0, "close": 103.0, "volume": 1200.0},
        {"open": 103.0, "high": 107.0, "low": 101.0, "close": 106.0, "volume": 1100.0},
    ]
