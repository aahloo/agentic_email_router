"""
Workflow Utilities for Production-Ready Agentic Workflows

This module provides infrastructure for error handling, logging, and execution tracking
in AI agent workflows. All components are designed to be reusable and optional, ensuring
100% backwards compatibility with existing code.

Components:
- setup_logging(): Configure dual console + file logging
- safe_api_call(): Decorator for retry logic with exponential backoff
- WorkflowTimer: Track execution time for workflow steps

Usage Example:
    from workflow_utils import setup_logging, WorkflowTimer

    # Initialize logging
    logger = setup_logging()
    logger.info("Workflow started")

    # Track execution time
    timer = WorkflowTimer()
    timer.start()
    # ... do work ...
    timer.end()
    print(timer.get_summary())
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any
import os


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up dual logging (console + file) for workflow execution.

    Creates timestamped log files in the specified directory and configures
    both console and file handlers with appropriate log levels.

    Args:
        log_dir: Directory to store log files (created if doesn't exist)
        console_level: Log level for console output (default: INFO)
        file_level: Log level for file output (default: DEBUG)

    Returns:
        Logger instance configured with both handlers

    Example:
        >>> logger = setup_logging()
        >>> logger.info("Workflow started")
        >>> logger.debug("Detailed debug information")

    Note:
        - Console shows INFO+ messages (user-friendly progress)
        - File contains DEBUG+ messages (comprehensive debugging)
        - Log filename format: workflow_YYYYMMDD_HHMMSS.log
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"workflow_{timestamp}.log"

    # Create logger
    logger = logging.getLogger('agentic_workflow')
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (INFO level - user-friendly)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (DEBUG level - comprehensive)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Log initialization
    logger.info(f"Logging initialized: {log_file}")
    logger.debug(f"Console level: {logging.getLevelName(console_level)}, File level: {logging.getLevelName(file_level)}")

    return logger


def safe_api_call(
    max_retries: int = 3,
    backoff_base: int = 2,
    fallback_value: Any = None,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator for API calls with automatic retry logic and exponential backoff.

    Wraps functions that make API calls, automatically retrying on failures
    with increasing wait times. Handles common API errors gracefully.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base for exponential backoff calculation (default: 2)
        fallback_value: Value to return if all retries fail (default: None)
        logger: Optional logger instance for detailed logging

    Returns:
        Decorated function with retry logic

    Example:
        >>> @safe_api_call(max_retries=3)
        >>> def call_openai_api():
        >>>     return client.chat.completions.create(...)
        >>>
        >>> result = call_openai_api()  # Automatically retries on failure

    Retry Logic:
        - Attempt 1: Immediate
        - Attempt 2: Wait 2^1 = 2 seconds
        - Attempt 3: Wait 2^2 = 4 seconds
        - If all fail: Return fallback_value

    Error Handling:
        - Retries on: API connection errors, timeouts, rate limits
        - Propagates: Authentication errors, invalid requests
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger('agentic_workflow.safe_api_call')

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        log.debug(f"{func.__name__}() retry attempt {attempt + 1}/{max_retries}")

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        log.info(f"{func.__name__}() succeeded on retry {attempt + 1}")

                    return result

                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # Check if this is a retryable error
                    retryable_errors = [
                        'RateLimitError',
                        'APIConnectionError',
                        'Timeout',
                        'APITimeoutError',
                        'ConnectionError',
                        'HTTPError'
                    ]

                    is_retryable = any(err in error_type for err in retryable_errors)

                    if attempt < max_retries - 1 and is_retryable:
                        # Calculate backoff time
                        wait_time = backoff_base ** attempt
                        log.warning(
                            f"{func.__name__}() failed with {error_type}: {error_msg}. "
                            f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        # Max retries reached or non-retryable error
                        if is_retryable:
                            log.error(
                                f"{func.__name__}() failed after {max_retries} attempts. "
                                f"Last error: {error_type}: {error_msg}"
                            )
                        else:
                            log.error(
                                f"{func.__name__}() failed with non-retryable error: "
                                f"{error_type}: {error_msg}"
                            )

                        # Return fallback value or re-raise
                        if fallback_value is not None:
                            log.warning(f"Returning fallback value: {fallback_value}")
                            return fallback_value
                        else:
                            raise

            # Shouldn't reach here, but just in case
            log.error(f"{func.__name__}() exhausted retries without success")
            if fallback_value is not None:
                return fallback_value
            return None

        return wrapper
    return decorator


class WorkflowTimer:
    """
    Track execution time for workflow and individual steps.

    Provides simple timing functionality to measure workflow performance
    and identify bottlenecks. Can track overall workflow time and individual
    step durations.

    Attributes:
        start_time: Workflow start timestamp (None if not started)
        end_time: Workflow end timestamp (None if not ended)
        steps: List of recorded step timings

    Example:
        >>> timer = WorkflowTimer()
        >>> timer.start()
        >>>
        >>> timer.record_step("action_planning", 2.5)
        >>> timer.record_step("routing", 15.3)
        >>>
        >>> timer.end()
        >>> print(timer.get_summary())

        Workflow Execution Time: 17.8 seconds
        Steps:
          - action_planning: 2.5s
          - routing: 15.3s
    """

    def __init__(self):
        """Initialize WorkflowTimer with no timing data."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.steps: list = []

    def start(self) -> None:
        """
        Start the workflow timer.

        Records the current timestamp as the workflow start time.
        """
        self.start_time = time.time()

    def end(self) -> None:
        """
        End the workflow timer.

        Records the current timestamp as the workflow end time.
        Requires start() to have been called first.

        Raises:
            RuntimeError: If start() was not called before end()
        """
        if self.start_time is None:
            raise RuntimeError("WorkflowTimer.start() must be called before end()")
        self.end_time = time.time()

    def record_step(self, step_name: str, duration: float) -> None:
        """
        Record timing for a completed workflow step.

        Args:
            step_name: Name/description of the step
            duration: Duration in seconds

        Example:
            >>> timer.record_step("agent_routing", 5.2)
        """
        self.steps.append({
            'name': step_name,
            'duration': duration
        })

    def get_duration(self) -> Optional[float]:
        """
        Get total workflow duration in seconds.

        Returns:
            Duration in seconds, or None if workflow not completed
        """
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "2m 15s" or "45.2s")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"

    def get_summary(self) -> str:
        """
        Get formatted summary of workflow timing.

        Returns:
            Multi-line string with overall duration and step breakdowns

        Example output:
            Workflow Execution Time: 25.3 seconds

            Step Breakdown:
              - action_planning: 2.5s
              - routing_step_1: 8.1s
              - routing_step_2: 7.9s
              - routing_step_3: 6.8s
        """
        lines = []

        # Overall duration
        duration = self.get_duration()
        if duration is not None:
            lines.append(f"Workflow Execution Time: {self.format_duration(duration)}")
        else:
            lines.append("Workflow Execution Time: Not completed")

        # Step breakdown
        if self.steps:
            lines.append("")
            lines.append("Step Breakdown:")
            for step in self.steps:
                lines.append(f"  - {step['name']}: {self.format_duration(step['duration'])}")

        return "\n".join(lines)


# Module-level logger instance (optional convenience)
_default_logger: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    """
    Get or create the default workflow logger.

    Returns:
        Default logger instance (creates if doesn't exist)
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
