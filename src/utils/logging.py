"""
Centralised logging and console output for the project.

Provides a lazy-initialised logger singleton via get_logger(). On first call,
setup_logging() configures two handlers: a file handler (always) and a console
handler (RichHandler if rich is installed, plain StreamHandler otherwise).

Usage in any module:
    from src.utils.logging import get_logger

    logger = get_logger()
    logger.info(...)
"""

import logging
import os
import sys

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from src.config.settings import (
    LOG_FILE, PATH_OUT_LOGS,
    LOG_ROOT_LEVEL, LOG_FILE_LEVEL, LOG_CONSOLE_LEVEL,
    PROJECT_NAME
)

# ---------------------------------------------------------------------------
# logger — starts as None. Not ready until setup_logging() runs via
# get_logger(). Use get_logger() rather than importing this directly.
# ---------------------------------------------------------------------------
logger = None


def setup_logging():
    """
    Initialise and return the project logger with two handlers.

    Do not call directly — get_logger() calls this once lazily on first
    access and caches the result in the module-level logger sentinel.

    Handlers
    --------
    FileHandler     : Plain text, full timestamps, DEBUG+ to LOG_FILE.
                      Persists across notebook restarts.
    Console handler : RichHandler if rich is installed, StreamHandler otherwise.
                      INFO+ to stdout.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(PATH_OUT_LOGS, exist_ok=True)

    _logger = logging.getLogger(PROJECT_NAME)
    _logger.setLevel(getattr(logging, LOG_ROOT_LEVEL))

    # Prevent duplicate emissions when root logger has handlers (common in Jupyter)
    _logger.propagate = False

    # Clear stale handlers — guards against double-registration on notebook re-runs
    if _logger.hasHandlers():
        _logger.handlers.clear()

    # File handler — plain text, all levels including DEBUG, full context
    file_handler = logging.FileHandler(str(LOG_FILE))
    file_handler.setLevel(getattr(logging, LOG_FILE_LEVEL))
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s][%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler — rich if available, plain StreamHandler otherwise
    if _RICH_AVAILABLE:
        console_handler = RichHandler(
            level=getattr(logging, LOG_CONSOLE_LEVEL),
            console=Console(),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_CONSOLE_LEVEL))
        console_handler.setFormatter(logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    _logger.debug("Logging system initialised")
    return _logger


def get_logger():
    """
    Return the project logger, initialising it on first call.

    Implements a simple singleton — setup_logging() runs exactly once per
    session regardless of how many modules call get_logger().

    Returns
    -------
    logging.Logger
    """
    global logger
    if logger is None:
        logger = setup_logging()
    return logger
