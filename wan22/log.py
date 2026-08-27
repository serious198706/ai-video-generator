from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from wan22 import config

_configured = False
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(*, force: bool = False) -> None:
    """当前文件 `wan22.log`，午夜滚成 `wan22.log.YYYY-MM-DD`。"""
    global _configured
    if _configured and not force:
        return

    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "wan22.log"

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    handlers: list[logging.Handler] = []

    file_handler = TimedRotatingFileHandler(
        logfile,
        when="midnight",
        interval=1,
        backupCount=config.LOG_BACKUP_DAYS,
        encoding="utf-8",
        utc=False,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    if config.LOG_CONSOLE:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        handlers.append(console)

    for logger_name in ("wan22", "uvicorn", "uvicorn.error", "uvicorn.access"):
        existing = logging.getLogger(logger_name)
        for handler in list(existing.handlers):
            existing.removeHandler(handler)
            handler.close()

    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    root_wan22 = logging.getLogger("wan22")
    for handler in handlers:
        root_wan22.addHandler(handler)
    root_wan22.setLevel(level)
    root_wan22.propagate = False

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    _configured = True
    root_wan22.info(
        "logging to %s level=%s rotate=midnight keep=%sd",
        logfile,
        logging.getLevelName(level),
        config.LOG_BACKUP_DAYS,
    )


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if name.startswith("wan22."):
        return logging.getLogger(name)
    return logging.getLogger(f"wan22.{name}")
