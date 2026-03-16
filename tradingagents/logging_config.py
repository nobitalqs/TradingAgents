"""Structured logging setup."""

import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    """Configure logging. 0=WARNING, 1=INFO, 2=DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    if root.handlers:
        return  # avoid duplicate handlers

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(handler)
    root.setLevel(level)

    # quiet noisy libraries
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
