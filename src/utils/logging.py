from __future__ import annotations

import sys

from loguru import logger as _logger


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru: stderr for human-readable output, file for debug trace."""
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:^7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — {message}"
        ),
    )
    _logger.add(
        "data/paperpilot.log",
        level="DEBUG",
        rotation="10 MB",
        encoding="utf-8",
    )


logger = _logger
