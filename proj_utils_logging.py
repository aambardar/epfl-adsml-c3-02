import logging
import os
from proj_configs import LOG_FILE, PATH_OUT_LOGS, LOG_ROOT_LEVEL, LOG_FILE_LEVEL, LOG_CONSOLE_LEVEL, PROJECT_NAME


def setup_logging():
    """
    Sets up logging configuration with both file and console handlers.
    File handler: Logs everything to a file with timestamps
    Console handler: Shows logs on the console with a simpler format
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(PATH_OUT_LOGS), exist_ok=True)

    # Create logger
    logger = logging.getLogger(PROJECT_NAME)
    logger.setLevel(getattr(logging, LOG_ROOT_LEVEL))

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s][%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '[%(levelname)s][%(funcName)s] %(message)s'
    )

    # Create and configure file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_FILE_LEVEL))
    file_handler.setFormatter(file_formatter)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_CONSOLE_LEVEL))
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial message to verify setup
    logger.debug("Logging system initialised")

    return logger


# Create a global logger instance
logger = None


def get_logger():
    """
    Returns the global logger instance, creating it if necessary.
    """
    global logger
    if logger is None:
        logger = setup_logging()
    return logger