"""In-process HTTP transport for ``rebot._http`` (the shape of httpx.MockTransport, standard library only)."""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from urllib.parse import parse_qsl, urlsplit

from rebot._http import HttpResponse


class _Url:
    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        self.path = parts.path
        self.params = dict(parse_qsl(parts.query))
        self.raw = url


class Request:
    def __init__(self, method: str, url: str, body: bytes | None, headers: dict) -> None:
        self.method = method
        self.url = _Url(url)
        self.content = body or b""
        self.headers = {k.lower(): v for k, v in headers.items()}


def Response(status: int, json=None) -> HttpResponse:  # noqa: N802 (mirrors httpx.Response)
    return HttpResponse(status, b"" if json is None else _json.dumps(json).encode())


class MockTransport:
    def __init__(self, handler: Callable[[Request], HttpResponse]) -> None:
        self.handler = handler

    def __call__(self, method: str, url: str, body: bytes | None, headers: dict, timeout: float) -> HttpResponse:
        return self.handler(Request(method, url, body, headers))
