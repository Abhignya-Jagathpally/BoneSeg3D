"""
boneseg3d/utils/logging_utils.py
==================================
Structured logging with consistent formatting across all modules and subagents.
Writes to both console (INFO+) and logs/boneseg3d.log (DEBUG+).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "boneseg3d.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_FILE_HANDLER   = None
_INITIALISED    = False


def _ensure_file_handler() -> logging.FileHandler:
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _FILE_HANDLER = logging.FileHandler(LOG_FILE, encoding="utf-8")
        _FILE_HANDLER.setLevel(logging.DEBUG)
        _FILE_HANDLER.setFormatter(_FORMATTER)
    return _FILE_HANDLER


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with console + file handlers."""
    logger = logging.getLogger(f"boneseg3d.{name}")
    if logger.handlers:
        return logger   # already configured

    logger.setLevel(logging.DEBUG)

    # Console handler (INFO and above)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_FORMATTER)
    logger.addHandler(console)

    # File handler (DEBUG and above)
    logger.addHandler(_ensure_file_handler())

    logger.propagate = False
    return logger