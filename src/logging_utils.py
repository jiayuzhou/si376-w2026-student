"""Logging utilities for the FEVER course project.

Simple logging helpers for students to debug their code and track progress.

For beginners: **Logging** is like adding print() statements, but much more powerful!
Instead of scattered print() calls, logging:
- Lets you control verbosity (DEBUG vs INFO vs ERROR)
- Saves output to files for later analysis
- Adds timestamps automatically
- Color-codes messages by importance
- Can be turned on/off without changing code

Think of logging as "professional print()" - it's how real software tracks what's happening.
"""

# ============================================================
# Imports
# ============================================================
from __future__ import annotations

import logging  # Python's built-in logging module
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import PATHS


# ============================================================
# Colored Formatter
# ============================================================
# For beginners: This class makes log messages colorful in the terminal

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output.

    For beginners: This class adds colors to log messages in the terminal.
    - DEBUG messages are cyan
    - INFO messages are green
    - WARNING messages are yellow
    - ERROR messages are red
    - CRITICAL messages are magenta

    Why colors?
    - Quick visual scanning (red = problem! green = all good)
    - Easier to spot errors in long output

    What are ANSI codes?
    - ANSI codes are special sequences that control terminal colors
    - Example: "\\033[31m" turns text red, "\\033[0m" resets to normal
    - These work on Linux/Mac terminals and Windows Terminal (not old cmd.exe)

    Inheritance:
    - ColoredFormatter "extends" logging.Formatter (parent class)
    - We override the format() method to add colors
    - super() calls the parent class's format() method
    """

    # ANSI color codes for different log levels
    # For beginners: Dictionary mapping log level name → color code
    # \\033[ starts an ANSI code, numbers are colors, m ends the code
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan (technical details)
        "INFO": "\033[32m",       # Green (normal progress)
        "WARNING": "\033[33m",    # Yellow (potential issues)
        "ERROR": "\033[31m",      # Red (errors!)
        "CRITICAL": "\033[35m",   # Magenta (critical failures!)
    }
    # Reset code turns off colors
    RESET = "\033[0m"

    def format(self, record):
        """Format a log record with colors.

        For beginners: This method is called automatically by Python's logging
        system whenever a log message is created. We intercept it to add colors.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format (contains message, level, timestamp, etc.)

        Returns
        -------
        str
            Formatted log message with color codes
        """
        # Get color for this log level
        # For beginners: .get(key, default) returns COLORS[key] if exists, else RESET
        # Example: record.levelname="ERROR" → log_color="\\033[31m" (red)
        log_color = self.COLORS.get(record.levelname, self.RESET)

        # Add color to level name
        # For beginners: f"{log_color}{record.levelname:8s}{self.RESET}"
        # - record.levelname: "INFO", "ERROR", etc.
        # - :8s: Pad to 8 characters (so "INFO    " aligns with "WARNING ")
        # - Wrapped in color codes: "\\033[32mINFO    \\033[0m"
        record.levelname = f"{log_color}{record.levelname:8s}{self.RESET}"

        # Call parent class's format() to do the actual formatting
        # For beginners: super() refers to the parent class (logging.Formatter)
        # We've modified record.levelname to add colors, now let parent do the rest
        return super().format(record)


# ============================================================
# Logger Setup
# ============================================================
# For beginners: This function creates a logger configured for our project

def setup_logger(
    name: str = "fever",
    level: str = "INFO",
    log_to_file: bool = False,
    log_file: Optional[Path] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """Set up a logger with console and optional file output.

    For beginners: This function creates a "logger" - an object you use to
    log messages. Think of it as setting up a reporting system:
    - Where do messages go? (console, file, or both)
    - What gets logged? (only errors, or everything including debug info)
    - How are messages formatted? (with colors, timestamps, etc.)

    What are logging levels?
    - **DEBUG**: Detailed info for diagnosing problems (verbose!)
    - **INFO**: General progress messages (default)
    - **WARNING**: Something unexpected but not critical
    - **ERROR**: Something failed
    - **CRITICAL**: Major failure, system might crash

    Rule: If level=INFO, you see INFO/WARNING/ERROR/CRITICAL (but not DEBUG)

    Parameters
    ----------
    name : str, default="fever"
        Logger name (useful if you have multiple loggers)
    level : str, default="INFO"
        Minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_to_file : bool, default=False
        Also save logs to a file (in addition to console)
    log_file : Path, optional
        File path for logs (default: outputs/fever_YYYYMMDD_HHMMSS.log)
    use_colors : bool, default=True
        Use colored output in terminal

    Returns
    -------
    logging.Logger
        Configured logger instance

    Example
    -------
    >>> from src.logging_utils import setup_logger
    >>> logger = setup_logger(level="DEBUG")
    >>> logger.info("Starting experiment...")
    INFO     | Starting experiment...
    >>> logger.debug("Loaded 100 samples")
    DEBUG    | Loaded 100 samples
    >>> logger.error("Something went wrong!")
    ERROR    | Something went wrong!
    """
    # Create or get logger
    # For beginners: logging.getLogger(name) creates a new logger or retrieves existing one
    # Multiple calls with same name return the SAME logger object
    logger = logging.getLogger(name)

    # Clear any existing handlers
    # For beginners: Handlers control where log messages go (console, file, etc.)
    # We clear them to avoid duplicate messages if setup_logger is called twice
    logger.handlers.clear()

    # Set logging level
    # For beginners: getattr(logging, "INFO") gets the logging.INFO constant
    # - level.upper(): Convert "info" → "INFO"
    # - getattr(logging, "INFO"): Get logging.INFO (which is an integer like 20)
    # This allows users to pass level as string "INFO" instead of logging.INFO
    logger.setLevel(getattr(logging, level.upper()))

    # ====== Console Handler ======
    # For beginners: Handler = destination for log messages
    # StreamHandler(sys.stdout) sends messages to terminal (standard output)

    console_handler = logging.StreamHandler(sys.stdout)

    # Set console level
    # For beginners: Handler level can be different from logger level
    # We set them the same here
    console_handler.setLevel(getattr(logging, level.upper()))

    # Choose formatter based on color support
    # For beginners: Formatter = how messages are displayed (with/without timestamp, etc.)
    # - sys.stdout.isatty(): True if output is a terminal (not redirected to file)
    # - Only use colors if terminal supports them
    if use_colors and sys.stdout.isatty():
        # Format: "INFO     | Message here"
        # For beginners: %(levelname)s is replaced with log level (INFO, ERROR, etc.)
        # %(message)s is replaced with the actual log message
        console_format = "%(levelname)s | %(message)s"

        # Use ColoredFormatter to add colors
        console_formatter = ColoredFormatter(console_format)
    else:
        # No colors (for file redirection or old terminals)
        # For beginners: %-8s left-aligns level name to 8 characters
        console_format = "%(levelname)-8s | %(message)s"
        console_formatter = logging.Formatter(console_format)

    # Attach formatter to handler
    # For beginners: Tell console_handler to use this formatter
    console_handler.setFormatter(console_formatter)

    # Attach handler to logger
    # For beginners: Tell logger to send messages to console_handler
    logger.addHandler(console_handler)

    # ====== File Handler (Optional) ======
    # For beginners: Optionally also save logs to a file

    if log_to_file:
        # Determine log file path
        # For beginners: If no path provided, generate one with timestamp
        if log_file is None:
            # Get current timestamp
            # For beginners: datetime.now() gets current date/time
            # .strftime() formats it as string: "20250115_143022"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create file path
            # For beginners: PATHS.outputs_dir / "fever_20250115_143022.log"
            log_file = PATHS.outputs_dir / f"fever_{timestamp}.log"

        # Create outputs directory if it doesn't exist
        # For beginners: .mkdir() creates directory
        # - exist_ok=True: Don't error if directory already exists
        # - parents=True: Create parent directories too
        PATHS.outputs_dir.mkdir(exist_ok=True, parents=True)

        # Create file handler
        # For beginners: FileHandler writes log messages to a file
        file_handler = logging.FileHandler(log_file)

        # Always log DEBUG+ to file (even if console only shows INFO+)
        # For beginners: Files can store more detail than we want to see on screen
        file_handler.setLevel(logging.DEBUG)

        # File format (more detailed than console)
        # For beginners: File logs include timestamp and logger name
        # - %(asctime)s: Timestamp (e.g., "2025-01-15 14:30:22")
        # - %(name)s: Logger name (e.g., "fever")
        # - %(levelname)-8s: Log level (e.g., "INFO    ")
        # - %(message)s: The actual message
        file_format = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"

        # Create formatter with custom date format
        # For beginners: datefmt controls how %(asctime)s is formatted
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

        # Attach formatter to handler
        file_handler.setFormatter(file_formatter)

        # Attach handler to logger
        # For beginners: Now logger sends messages to BOTH console and file
        logger.addHandler(file_handler)

        # Log where we're saving to
        logger.info(f"Logging to file: {log_file}")

    return logger


# ============================================================
# Progress Logger
# ============================================================
# For beginners: Class for tracking long-running tasks (training, evaluation, etc.)

class ProgressLogger:
    """Simple progress logger for tracking experiment progress.

    For beginners: Use this when you have a long loop and want to track progress.
    Instead of printing every iteration (spammy!), this:
    - Logs every 10% of progress
    - Logs every 30 seconds (if progress is slow)
    - Tracks elapsed time
    - Shows metrics (loss, accuracy, etc.)

    Example use case:
    - Training a model for 1000 epochs → log every 100 epochs
    - Processing 10000 examples → log every 1000 examples

    Attributes
    ----------
    task_name : str
        Name of the task (e.g., "Training model")
    total : int, optional
        Total number of steps (if known)
    logger : logging.Logger
        Logger to use for output
    start_time : datetime
        When task started
    last_log_time : datetime
        When we last logged (for 30-second rule)

    Example
    -------
    >>> progress = ProgressLogger("Training model", total=100)
    >>> for i in range(100):
    ...     # Train one epoch
    ...     loss = train_epoch()
    ...     # Update progress
    ...     progress.update(i, loss=loss)
    >>> progress.finish(final_loss=0.05)
    INFO     | Starting: Training model
    INFO     | Progress: 10/100 (10.0%) | loss=0.5000 | elapsed=5.2s
    INFO     | Progress: 20/100 (20.0%) | loss=0.4000 | elapsed=10.1s
    ...
    INFO     | Finished: Training model | final_loss=0.0500 | total_time=50.3s
    """

    def __init__(
        self,
        task_name: str,
        total: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize progress logger.

        For beginners: Sets up tracking for a task.

        Parameters
        ----------
        task_name : str
            Name of task (shown in log messages)
        total : int, optional
            Total number of steps (for percentage calculation)
        logger : logging.Logger, optional
            Logger to use (default: "fever" logger)
        """
        # Store task info
        # For beginners: Save task name and total for later use
        self.task_name = task_name
        self.total = total

        # Get or create logger
        # For beginners: logger or logging.getLogger("fever")
        # - If logger is provided, use it
        # - Otherwise (logger is None), get default "fever" logger
        self.logger = logger or logging.getLogger("fever")

        # Track timing
        # For beginners: Record when task started
        self.start_time = datetime.now()
        self.last_log_time = self.start_time

        # Log start
        self.logger.info(f"Starting: {task_name}")

    def update(self, current: int, **metrics):
        """Update progress with current step and optional metrics.

        For beginners: Call this in your loop to report progress.
        Only logs periodically (not every call) to avoid spam.

        Parameters
        ----------
        current : int
            Current step number
        **metrics : keyword arguments
            Optional metrics to log (e.g., loss=0.5, acc=0.9)

        Example
        -------
        >>> progress.update(50, loss=0.3, accuracy=0.85)
        INFO     | Progress: 50/100 (50.0%) | loss=0.3000, accuracy=0.8500 | elapsed=25.1s
        """
        # Get current time
        # For beginners: datetime.now() gets current date/time
        now = datetime.now()

        # Calculate elapsed time
        # For beginners: (now - self.start_time) is a timedelta object
        # .total_seconds() converts to seconds (float)
        elapsed = (now - self.start_time).total_seconds()

        # Decide whether to log
        # For beginners: We don't log EVERY update (too spammy)
        # Log if: (1) progress reached 10% milestone, OR (2) 30 seconds passed
        should_log = False

        # Rule 1: Log every 10%
        # For beginners: If total=100, log at 10, 20, 30, ..., 90
        # - self.total // 10: Integer division (100 // 10 = 10)
        # - max(1, ...): Avoid division by zero for small totals
        # - current % ...: Modulo checks if current is divisible
        if self.total and current % max(1, self.total // 10) == 0:
            should_log = True

        # Rule 2: Log every 30 seconds
        # For beginners: If loop is slow, log at least every 30 seconds
        # (now - self.last_log_time).total_seconds() = seconds since last log
        elif (now - self.last_log_time).total_seconds() > 30:
            should_log = True

        # Log if criteria met
        if should_log:
            # Build log message
            # For beginners: msg_parts is a list we'll join with " | "
            # Start with progress
            msg_parts = [f"Progress: {current}"]

            # Add percentage if total is known
            # For beginners: If total=100 and current=50, add " (50.0%)"
            if self.total:
                # Calculate percentage
                pct = (current / self.total) * 100
                # Append to first part
                msg_parts[0] += f"/{self.total} ({pct:.1f}%)"

            # Add metrics if provided
            # For beginners: **metrics captures keyword arguments as dictionary
            # Example: update(50, loss=0.3) → metrics={"loss": 0.3}
            if metrics:
                # Format metrics
                # For beginners: List comprehension creates "key=value" strings
                # - If value is float, format as "key=0.1234"
                # - Otherwise, format as "key=value"
                metrics_str = ", ".join([
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()
                ])
                msg_parts.append(metrics_str)

            # Add elapsed time
            msg_parts.append(f"elapsed={elapsed:.1f}s")

            # Join and log
            # For beginners: " | ".join() creates "Progress: 50/100 | loss=0.3 | elapsed=25.1s"
            self.logger.info(" | ".join(msg_parts))

            # Update last log time
            # For beginners: Remember when we logged, for 30-second rule
            self.last_log_time = now

    def finish(self, **final_metrics):
        """Mark task as finished and log final metrics.

        For beginners: Call this when task is complete to log final summary.

        Parameters
        ----------
        **final_metrics : keyword arguments
            Final metrics to log (e.g., final_loss=0.05, final_acc=0.95)

        Example
        -------
        >>> progress.finish(final_accuracy=0.92)
        INFO     | Finished: Training model | final_accuracy=0.9200 | total_time=120.5s
        """
        # Calculate total elapsed time
        # For beginners: How long did entire task take?
        elapsed = (datetime.now() - self.start_time).total_seconds()

        # Build finish message
        # For beginners: Similar to update(), but says "Finished"
        msg_parts = [f"Finished: {self.task_name}"]

        # Add final metrics
        # For beginners: Same formatting as update()
        if final_metrics:
            metrics_str = ", ".join([
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in final_metrics.items()
            ])
            msg_parts.append(metrics_str)

        # Add total time
        msg_parts.append(f"total_time={elapsed:.1f}s")

        # Log finish
        self.logger.info(" | ".join(msg_parts))


# ============================================================
# Data Logging Utilities
# ============================================================
# For beginners: Helper functions for logging pandas DataFrames and models

def log_dataframe_info(df, name: str = "DataFrame", logger: Optional[logging.Logger] = None):
    """Log useful information about a pandas DataFrame.

    For beginners: This function logs a summary of a DataFrame - useful for
    debugging data loading and preprocessing. Shows:
    - Shape (rows, columns)
    - Label distribution (if "label" column exists)
    - Column names and data types
    - Missing values (if any)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to log info about
    name : str, default="DataFrame"
        Name to use in log messages (e.g., "Training data")
    logger : logging.Logger, optional
        Logger to use (default: "fever" logger)

    Example
    -------
    >>> import pandas as pd
    >>> from src.logging_utils import log_dataframe_info
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> log_dataframe_info(df, "My Data")
    INFO     | My Data shape: (3, 2) (3 rows, 2 cols)
    DEBUG    | My Data columns: ['a', 'b']
    DEBUG    | My Data dtypes:
    a    int64
    b    int64
    dtype: object
    """
    # Get or create logger
    # For beginners: Use provided logger or default "fever" logger
    logger = logger or logging.getLogger("fever")

    # Log shape
    # For beginners: df.shape is a tuple (rows, cols)
    # df.shape[0] = number of rows, df.shape[1] = number of columns
    logger.info(f"{name} shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Log label distribution if "label" column exists
    # For beginners: .value_counts() counts occurrences of each unique value
    # Useful for checking class balance (SUPPORTS, REFUTES, NEI)
    if "label" in df.columns:
        # Count labels
        # For beginners: .value_counts() returns Series with counts
        # .to_dict() converts to dictionary for cleaner display
        label_counts = df["label"].value_counts()
        logger.info(f"{name} labels: {label_counts.to_dict()}")

    # Log column names (DEBUG level)
    # For beginners: df.columns is Index object, .tolist() converts to list
    logger.debug(f"{name} columns: {df.columns.tolist()}")

    # Log data types (DEBUG level)
    # For beginners: df.dtypes shows type of each column (int64, object, float64, etc.)
    logger.debug(f"{name} dtypes:\n{df.dtypes}")

    # Check for missing values
    # For beginners: df.isnull() returns DataFrame of True/False
    # .sum() counts True values per column
    # missing is a Series with count of missing values per column
    missing = df.isnull().sum()

    # Warn if any missing values
    # For beginners: missing.any() returns True if ANY column has missing values
    if missing.any():
        # Filter to only columns with missing values
        # For beginners: missing[missing > 0] keeps only columns where count > 0
        logger.warning(f"{name} missing values:\n{missing[missing > 0]}")


def log_model_info(model, logger: Optional[logging.Logger] = None):
    """Log information about a model (scikit-learn or transformers).

    For beginners: This function logs model information - useful for tracking
    what models you're using. Shows:
    - Model type (LogisticRegression, DistilBertForSequenceClassification, etc.)
    - Number of parameters (for transformer models)
    - Hyperparameters (for sklearn models)

    Parameters
    ----------
    model : model instance
        Model to log info about (sklearn or transformers)
    logger : logging.Logger, optional
        Logger to use (default: "fever" logger)

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from src.logging_utils import log_model_info
    >>> model = LogisticRegression(C=1.0)
    >>> log_model_info(model)
    INFO     | Model type: LogisticRegression
    DEBUG    | Model params: {'C': 1.0, 'class_weight': None, ...}
    """
    # Get or create logger
    logger = logger or logging.getLogger("fever")

    # Get model type name
    # For beginners: type(model).__name__ gets class name as string
    # Example: LogisticRegression object → "LogisticRegression"
    model_type = type(model).__name__
    logger.info(f"Model type: {model_type}")

    # Try to get parameter count for transformers models
    # For beginners: Transformers models have .num_parameters() method
    # sklearn models don't, so we use try/except
    try:
        # Check if model has num_parameters method
        # For beginners: hasattr(obj, "method") returns True if method exists
        if hasattr(model, "num_parameters"):
            # Get parameter count
            # For beginners: Transformers models have millions of parameters
            params = model.num_parameters()

            # Log with thousands separator
            # For beginners: :, adds commas (e.g., 66,955,010)
            logger.info(f"Model parameters: {params:,}")

    # Catch any errors
    # For beginners: Bare except catches all errors (usually bad practice,
    # but OK here since this is optional logging)
    except:
        pass

    # Log hyperparameters for sklearn models
    # For beginners: sklearn models have .get_params() method
    if hasattr(model, "get_params"):
        try:
            # Get parameters
            # For beginners: .get_params() returns dict of hyperparameters
            # Example: {"C": 1.0, "max_iter": 100, ...}
            params = model.get_params()

            # Log at DEBUG level (verbose)
            logger.debug(f"Model params: {params}")

        # Catch errors
        except:
            pass


# ============================================================
# Default Logger
# ============================================================
# For beginners: Create a default logger for convenience

# Create default logger
# For beginners: This creates a logger you can import and use immediately
# without calling setup_logger() yourself
#
# Usage:
#   from src.logging_utils import default_logger
#   default_logger.info("Hello!")
#
# vs:
#   from src.logging_utils import setup_logger
#   logger = setup_logger()
#   logger.info("Hello!")
default_logger = setup_logger()
