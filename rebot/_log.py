"""Internal lazy logging for rebot.

Zero module-level logging imports. Only imports the standard library `logging`
module if REBOT_DEBUG is set in the environment or if a custom logger is provided.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class _LazyLogger:
    def __call__(
        self,
        msg: str,
        *args: Any,
        level: str = "info",
        logger: Optional[Any] = None,
    ) -> None:
        if logger is not None:
            fn = getattr(logger, level, None) or getattr(logger, "info", None)
            if callable(fn):
                fn(msg, *args)
            return
        if not os.environ.get("REBOT_DEBUG"):
            return
        import logging

        log = logging.getLogger("rebot")
        fn = getattr(log, level, None) or log.info
        fn(msg, *args)

    def debug(self, msg: str, *args: Any, logger: Optional[Any] = None) -> None:
        self(msg, *args, level="debug", logger=logger)

    def info(self, msg: str, *args: Any, logger: Optional[Any] = None) -> None:
        self(msg, *args, level="info", logger=logger)

    def warning(self, msg: str, *args: Any, logger: Optional[Any] = None) -> None:
        self(msg, *args, level="warning", logger=logger)

    def error(self, msg: str, *args: Any, logger: Optional[Any] = None) -> None:
        self(msg, *args, level="error", logger=logger)


_log = _LazyLogger()
