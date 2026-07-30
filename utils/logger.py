"""
utils/logger.py
Centralised logging setup – writes to console + rotating file.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import LOGS_DIR

def get_logger(name: str = "surveillance") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler (5 MB × 3 backups)
    fh = RotatingFileHandler(
        os.path.join(LOGS_DIR, "surveillance.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
