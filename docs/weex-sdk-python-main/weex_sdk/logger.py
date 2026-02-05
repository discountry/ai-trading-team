"""Logging configuration for Weex SDK."""

import logging
import sys
from typing import Optional

# Create logger for weex_sdk
logger = logging.getLogger("weex_sdk")

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    stream: Optional[object] = None,
) -> logging.Logger:
    """Setup logger configuration.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom log format string
        date_format: Custom date format string
        stream: Output stream (default: sys.stderr)

    Returns:
        Configured logger instance
    """
    if logger.handlers:
        # Logger already configured
        return logger

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(
        format_string or DEFAULT_LOG_FORMAT,
        datefmt=date_format or DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name (default: 'weex_sdk')

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"weex_sdk.{name}")
    return logger


# Initialize default logger
setup_logger()
