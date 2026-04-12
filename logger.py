import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

import config

LOG_DIR = os.path.join(config.PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "trading_agent.log")

def setup_logger(name="ai_trading_agent"):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # File Formatter
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Formatter (Keep it clean for chat.py UI)
    console_formatter = logging.Formatter("%(message)s")

    # File Handler — daily rotation at midnight, keep 30 days
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"  # trading_agent.log.2026-04-09
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
