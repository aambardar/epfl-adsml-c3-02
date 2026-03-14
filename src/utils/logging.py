"""
Centralised logging and console output for the project.

Provides two shared singletons with different access patterns reflecting
their different initialisation needs:

    console              → rich.Console, ready immediately, import directly.
                           Use for structured output: tables, rules, panels.

    logger = get_logger()→ standard Python logger, lazy-initialised on first
                           call. Dual-output:
                             - file   : plain text, DEBUG+, full timestamps
                             - console: RichHandler, INFO+, coloured output

Usage in any module:
    from src.utils.logging import get_logger, console

    logger = get_logger()
    console.print(...)       # no getter needed — console is always ready
    logger.info(...)         # routed to both file and rich console handler
"""

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

from src.config.settings import (
    LOG_FILE, PATH_OUT_LOGS,
    LOG_ROOT_LEVEL, LOG_FILE_LEVEL, LOG_CONSOLE_LEVEL,
    PROJECT_NAME
)

# ---------------------------------------------------------------------------
# console — instantiated immediately, safe to import and use directly.
# Shared across all modules so all rich output goes to the same stream.
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# logger — starts as None. Not ready until setup_logging() runs via
# get_logger(). Use get_logger() rather than importing this directly.
# ---------------------------------------------------------------------------
logger = None


def setup_logging():
    """
    Initialise and return the project logger with two handlers.

    Do not call directly — get_logger() calls this once lazily on first
    access and caches the result in the module-level `logger` sentinel.

    Handlers
    --------
    FileHandler   : Plain text, full timestamps, DEBUG+ → LOG_FILE.
                    Persists across notebook restarts.
    RichHandler   : Colour-coded levels, INFO+ → console/notebook output.
                    Uses the shared module-level Console instance so all
                    rich output (logger + console.print) goes to one stream.

    Returns
    -------
    logging.Logger
    """
    # Ensure the logs output directory exists before opening the file handler
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

    # Console handler — replaces plain StreamHandler with rich-formatted output
    rich_handler = RichHandler(
        level=getattr(logging, LOG_CONSOLE_LEVEL),
        console=console,        # shared Console instance — one output stream
        show_time=True,
        show_path=False,        # hides file:line — keeps output concise
        markup=True,            # enables [bold], [red] etc. in log messages
        rich_tracebacks=True,   # renders exceptions with rich formatting
    )

    _logger.addHandler(file_handler)
    _logger.addHandler(rich_handler)

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
