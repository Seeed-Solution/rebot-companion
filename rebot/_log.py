"""Internal lazy logging for rebot.

Zero module-level logging imports. Warnings and errors are written to ``sys.stderr`` as
``rebot: <level>: <message>``; info and debug are dropped. With ``REBOT_DEBUG`` set, every level
goes to the standard library ``logging`` logger ``rebot`` (imported only then), and a custom
``logger`` passed to a call receives that call.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

_STDERR_LEVELS = ("warning", "error", "critical")


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
            if level in _STDERR_LEVELS:
                try:
                    text = msg % args if args else msg
                except (TypeError, ValueError):
                    text = f"{msg} {args!r}"
                try:
                    sys.stderr.write(f"rebot: {level}: {text}\n")
                except (OSError, ValueError, AttributeError):  # stderr closed or missing
                    pass
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
