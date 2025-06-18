"""Logging utilities for QuantMind."""

import logging
import os
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    # Bold versions for better visibility
    BOLD_COLORS = {
        "DEBUG": "\033[1;36m",  # Bold Cyan
        "INFO": "\033[1;32m",  # Bold Green
        "WARNING": "\033[1;33m",  # Bold Yellow
        "ERROR": "\033[1;31m",  # Bold Red
        "CRITICAL": "\033[1;35m",  # Bold Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def __init__(self, *args, use_color: bool = None, **kwargs):
        """Initialize the colored formatter.

        Args:
            *args: Positional arguments passed to parent Formatter
            use_color: Whether to use colors. If None, auto-detect based on terminal.
            **kwargs: Keyword arguments passed to parent Formatter
        """
        super().__init__(*args, **kwargs)

        if use_color is None:
            # Auto-detect color support
            self.use_color = (
                hasattr(sys.stderr, "isatty")
                and sys.stderr.isatty()
                and os.environ.get("TERM") != "dumb"
                and os.environ.get("NO_COLOR") is None
            )
        else:
            self.use_color = use_color

    def format(self, record):
        """Format the log record with colors."""
        if not self.use_color:
            return super().format(record)

        # Get the base formatted message
        formatted = super().format(record)

        # Choose color scheme based on level
        if record.levelname in ["ERROR", "CRITICAL"]:
            colors = self.BOLD_COLORS
        else:
            colors = self.COLORS

        # Apply colors to level name and message
        color = colors.get(record.levelname, colors["RESET"])
        reset = colors["RESET"]

        # Color the entire message for errors/critical, just the level for others
        if record.levelname in ["ERROR", "CRITICAL"]:
            return f"{color}{formatted}{reset}"
        else:
            # Replace the level name with colored version
            level_colored = f"{color}{record.levelname}{reset}"
            return formatted.replace(record.levelname, level_colored, 1)


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_color: Optional[bool] = None,
    file_output: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with consistent formatting and optional colors.

    Args:
        name: Logger name (defaults to 'quantmind')
        level: Logging level
        format_string: Custom format string
        use_color: Whether to use colors (auto-detect if None)
        file_output: Optional file path for file logging

    Returns:
        Configured logger instance
    """
    if name is None:
        name = "quantmind"

    if format_string is None:
        format_string = "[%(asctime)s %(levelname)-8s %(name)s] %(message)s"

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create colored formatter for console
        console_formatter = ColoredFormatter(
            format_string, datefmt="%Y-%m-%d %H:%M:%S", use_color=use_color
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Add file handler if requested
        if file_output:
            file_handler = logging.FileHandler(file_output)
            file_handler.setLevel(level)

            # File output should not have colors
            file_formatter = logging.Formatter(
                format_string, datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def get_logger(
    name: str,
    level: Optional[int] = None,
    use_color: Optional[bool] = None,
    file_output: Optional[str] = None,
) -> logging.Logger:
    """Get a logger instance with QuantMind formatting and optional colors.

    Args:
        name: Logger name (usually __name__)
        level: Optional logging level override
        use_color: Whether to use colors (auto-detect if None)
        file_output: Optional file path for file logging

    Returns:
        Logger instance
    """
    # Extract module name from full path
    module_name = name.split(".")[-1] if "." in name else name
    logger_name = f"quantmind.{module_name}"

    # Check if logger already exists and is configured
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    # Set up new logger with specified options
    return setup_logger(
        name=logger_name,
        level=level or logging.INFO,
        use_color=use_color,
        file_output=file_output,
    )


def configure_logging(
    level: int = logging.INFO,
    use_color: Optional[bool] = None,
    file_output: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure global logging settings for QuantMind.

    Args:
        level: Global logging level
        use_color: Whether to use colors (auto-detect if None)
        file_output: Optional file path for file logging
        format_string: Custom format string
    """
    # Set up the root quantmind logger
    setup_logger(
        name="quantmind",
        level=level,
        format_string=format_string,
        use_color=use_color,
        file_output=file_output,
    )


def create_demo_logger() -> logging.Logger:
    """Create a demo logger to show color capabilities.

    Returns:
        Demo logger instance
    """
    demo_logger = get_logger("demo", level=logging.DEBUG, use_color=True)

    # Log messages at different levels to demonstrate colors
    demo_logger.debug("This is a DEBUG message (cyan)")
    demo_logger.info("This is an INFO message (green)")
    demo_logger.warning("This is a WARNING message (yellow)")
    demo_logger.error("This is an ERROR message (red)")
    demo_logger.critical("This is a CRITICAL message (magenta)")

    return demo_logger


# Set up the base logger on import with color support
setup_logger("quantmind", use_color=True)
