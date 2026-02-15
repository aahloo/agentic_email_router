"""
Simple Logging Utilities for Agentic Workflows

Minimal logging infrastructure providing dual console + file output with timestamps.
This is a simplified alternative to workflow_utils.py - 90% less code, adequate for
most use cases.

Key Features:
- Dual logging: Console + file with identical output
- Timestamped log files for tracking multiple runs
- Pure Python stdlib, no external dependencies

What's NOT included (vs workflow_utils.py):
- No retry logic (no safe_api_call decorator)
- No execution timing (no WorkflowTimer class)
- Same log level for console and file (INFO)

Usage Example:
    from simple_logging import setup_simple_logging

    logger = setup_simple_logging()
    logger.info("Workflow started")
    logger.error("Something went wrong")
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_simple_logging(log_dir="logs"):
    """
    Set up minimal dual logging (console + file) for workflow execution.

    Creates timestamped log files in the specified directory and configures
    both console and file handlers with identical formatting and log level.

    Args:
        log_dir: Directory to store log files (created if doesn't exist)

    Returns:
        Logger instance configured with both console and file handlers

    Example:
        >>> logger = setup_simple_logging()
        >>> logger.info("Workflow started")
        INFO     | Workflow started

    Note:
        - Both console and file use INFO level
        - Log filename format: workflow_simple_YYYYMMDD_HHMMSS.log
        - No retry logic or timing - use workflow_utils.py for production features
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"workflow_simple_{timestamp}.log"

    # Create logger
    logger = logging.getLogger('agentic_workflow_simple')
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create shared formatter (same for console and file)
    log_format = logging.Formatter('%(levelname)-8s | %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Log initialization
    logger.info(f"Logging initialized: {log_file}")

    return logger
