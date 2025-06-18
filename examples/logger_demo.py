#!/usr/bin/env python3
"""Demonstration of QuantMind's colored logging functionality."""

import logging
import time

from quantmind.utils.logger import (
    get_logger,
    configure_logging,
    create_demo_logger,
)


def basic_logging_demo():
    """Demonstrate basic colored logging."""
    print("=== Basic Colored Logging Demo ===")

    # Get a logger for this module
    logger = get_logger(__name__)

    # Log messages at different levels
    logger.debug("Debug message - usually for development")
    logger.info("Info message - general information")
    logger.warning("Warning message - something to pay attention to")
    logger.error("Error message - something went wrong")
    logger.critical("Critical message - severe error!")

    print()


def module_specific_logging():
    """Demonstrate module-specific logging."""
    print("=== Module-Specific Logging ===")

    # Create loggers for different modules
    arxiv_logger = get_logger("quantmind.sources.arxiv_source")
    parser_logger = get_logger("quantmind.parsers.pdf_parser")
    workflow_logger = get_logger("quantmind.workflow.agent")

    # Each logger maintains its own identity
    arxiv_logger.info("ArXiv source: Found 10 papers")
    parser_logger.warning("PDF parser: Document formatting is unusual")
    workflow_logger.error("Workflow agent: Task failed with timeout")

    print()


def configuration_demo():
    """Demonstrate different logging configurations."""
    print("=== Configuration Options Demo ===")

    # Configure global logging
    print("1. Debug level with colors:")
    debug_logger = get_logger("debug_demo", level=logging.DEBUG, use_color=True)
    debug_logger.debug("This debug message is now visible")
    debug_logger.info("Info message with debug level")

    print("\n2. No colors (simulating non-TTY environment):")
    no_color_logger = get_logger("no_color_demo", use_color=False)
    no_color_logger.info("This message has no colors")
    no_color_logger.error("This error message also has no colors")

    print("\n3. Custom format with file output:")
    import tempfile
    import os

    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False
    )
    temp_file.close()

    try:
        file_logger = get_logger("file_demo", file_output=temp_file.name)
        file_logger.info("This message goes to both console and file")
        file_logger.warning("File logging is useful for production")

        # Show file contents
        with open(temp_file.name, "r") as f:
            file_contents = f.read()
        print(f"\nFile contents:\n{file_contents}")

    finally:
        os.unlink(temp_file.name)

    print()


def real_world_example():
    """Demonstrate real-world usage scenario."""
    print("=== Real-World Example: ArXiv Source ===")

    # Simulate ArXiv source operations
    logger = get_logger("quantmind.sources.arxiv")

    logger.info("Initializing ArXiv source with configuration")
    time.sleep(0.5)

    logger.info("Searching for papers: 'machine learning finance'")
    time.sleep(0.5)

    logger.warning("Rate limiting: Waiting 1 second between requests")
    time.sleep(0.5)

    logger.info("Found 15 papers matching query")
    time.sleep(0.5)

    logger.info("Downloading PDF: paper_2301.12345.pdf")
    time.sleep(0.5)

    logger.error("Failed to download PDF: Network timeout")
    time.sleep(0.5)

    logger.info("Retrying download with exponential backoff")
    time.sleep(0.5)

    logger.info("Successfully downloaded 14/15 papers")

    print()


def environment_awareness_demo():
    """Demonstrate environment-aware color detection."""
    print("=== Environment Awareness Demo ===")

    import os
    import sys

    # Show current environment detection
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    term_type = os.environ.get("TERM", "unknown")
    no_color = os.environ.get("NO_COLOR")

    print(f"Terminal detection:")
    print(f"  - Is TTY: {is_tty}")
    print(f"  - TERM environment: {term_type}")
    print(f"  - NO_COLOR set: {no_color is not None}")

    # Auto-detected logger
    auto_logger = get_logger("auto_detect")
    print(f"\nAuto-detected color usage:")
    auto_logger.info("This message uses auto-detected color settings")

    # Test with NO_COLOR environment variable
    print(f"\nTesting NO_COLOR environment variable:")
    os.environ["NO_COLOR"] = "1"
    try:
        no_color_auto = get_logger("no_color_auto")
        no_color_auto.info("This should have no colors due to NO_COLOR=1")
    finally:
        del os.environ["NO_COLOR"]

    print()


def performance_demo():
    """Demonstrate logging performance with colors."""
    print("=== Performance Demo ===")

    import time

    # Test performance with colors
    colored_logger = get_logger("perf_colored", use_color=True)
    plain_logger = get_logger("perf_plain", use_color=False)

    # Measure colored logging time
    start_time = time.time()
    for i in range(100):
        colored_logger.info(f"Colored log message {i}")
    colored_time = time.time() - start_time

    # Measure plain logging time
    start_time = time.time()
    for i in range(100):
        plain_logger.info(f"Plain log message {i}")
    plain_time = time.time() - start_time

    print(f"Performance comparison (100 messages):")
    print(f"  - Colored logging: {colored_time:.4f} seconds")
    print(f"  - Plain logging: {plain_time:.4f} seconds")
    print(
        f"  - Overhead: {((colored_time - plain_time) / plain_time * 100):.2f}%"
    )

    print()


def main():
    """Run all logging demonstrations."""
    print("QuantMind Colored Logging Demonstration")
    print("=" * 50)
    print()

    # Configure global logging for demos
    configure_logging(level=logging.DEBUG, use_color=True)

    demos = [
        basic_logging_demo,
        module_specific_logging,
        configuration_demo,
        real_world_example,
        environment_awareness_demo,
        # performance_demo,  # Commented out to reduce output
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Demo failed: {e}")

    print("=== Final Demo: All Log Levels ===")
    create_demo_logger()

    print("\n" + "=" * 50)
    print("Logging demonstration complete!")
    print("The colored output should help distinguish different log levels.")


if __name__ == "__main__":
    main()
