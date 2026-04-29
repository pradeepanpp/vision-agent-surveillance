# src/utils/logger.py
"""
Shared logger for the entire project.
Uses loguru for clean, coloured, timestamped output.
"""

import sys
from loguru import logger

# Remove default handler
logger.remove()

# Console handler — coloured, readable
logger.add(
    sys.stdout,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan> - {message}"
    ),
    level="INFO",
    colorize=True,
)

# File handler — full detail, rotates at 10MB
logger.add(
    "logs/surveillance.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
)

__all__ = ["logger"]
