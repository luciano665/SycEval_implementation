import logging
import os
from typing import Optional, Union

_CONFIGURED = False


def _resolve_level(level: Optional[Union[int, str]]) -> int:
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_str = level.strip().upper()
        if level_str.isdigit():
            return int(level_str)
        return logging._nameToLevel.get(level_str, logging.INFO)
    return logging.INFO


def _configure_root(level: int) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(level)
            if handler.formatter is None:
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )

    _CONFIGURED = True


def get_logger(name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    resolved_level = _resolve_level(level)
    _configure_root(resolved_level)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(resolved_level)
    return logger
