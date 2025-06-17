"""Logging utilities for QuantMind."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Logger name (defaults to 'quantmind')
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    if name is None:
        name = "quantmind"

    if format_string is None:
        format_string = "[%(asctime)s %(levelname)s %(name)s] %(message)s"

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            format_string, datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with QuantMind formatting.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    # Ensure base logger is set up
    setup_logger()

    # Return child logger
    return logging.getLogger(f'quantmind.{name.split(".")[-1]}')


# Set up the base logger on import
_base_logger = setup_logger()
