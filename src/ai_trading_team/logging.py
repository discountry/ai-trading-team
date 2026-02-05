"""Logging configuration for AI Trading Team."""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_to_file: Whether to also log to file (default: True)

    Returns:
        Root logger for the application
    """
    # Disable propagation to root logger to avoid duplicate logs
    logger = logging.getLogger("ai_trading_team")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"trading_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress duplicate logs from third-party libraries using our logger
    # by setting root logger to WARNING
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'ai_trading_team.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"ai_trading_team.{name}")
