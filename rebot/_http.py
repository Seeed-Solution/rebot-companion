"""Small JSON-over-HTTP helper on ``urllib`` (standard library only).

Every client takes an optional ``transport``: a callable
``(method, url, body: bytes | None, headers: dict, timeout: float) -> HttpResponse``.
Tests pass an in-process fake; the default sends real requests, bypassing proxy
environment variables (armd is on the local network).
"""

from __future__ import annotations

import json as _json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

from .errors import ArmConnectionError


class HttpResponse:
    def __init__(self, status: int, body: bytes = b"", headers: Optional[Dict[str, str]] = None) -> None:
        self.status = int(status)
        self.body = body
        self.headers = headers or {}

    @property
    def status_code(self) -> int:
        return self.status

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        return _json.loads(self.body.decode("utf-8")) if self.body else None

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def raise_for_status(self) -> None:
        if not self.ok:
            raise HttpStatusError(self)


class HttpError(ArmConnectionError):
    """The request could not be sent or timed out."""


class HttpTimeout(HttpError):
    code = "TIMEOUT"


class HttpStatusError(HttpError):
    """The server answered with a non-2xx status (``response`` holds it)."""

    code = "HTTP_ERROR"

    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        detail = response.text[:200].strip()
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("message") or body.get("error") or body.get("code") or detail)
        except ValueError:
            pass
        super().__init__(f"HTTP {response.status}" + (f": {detail}" if detail else ""))


Transport = Callable[[str, str, Optional[bytes], Dict[str, str], float], HttpResponse]

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def urllib_transport(
    method: str, url: str, body: Optional[bytes], headers: Dict[str, str], timeout: float
) -> HttpResponse:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with _opener.open(req, timeout=timeout) as res:
            return HttpResponse(res.status, res.read(), dict(res.headers.items()))
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read() or b"", dict(exc.headers.items()) if exc.headers else {})
    except socket.timeout as exc:
        raise HttpTimeout(f"{method} {url} timed out after {timeout:g}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise HttpTimeout(f"{method} {url} timed out after {timeout:g}s") from exc
        raise HttpError(f"cannot reach {url}: {exc.reason}") from exc
    except OSError as exc:
        raise HttpError(f"cannot reach {url}: {exc}") from exc


def request(
    method: str,
    url: str,
    json: Any = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    transport: Optional[Transport] = None,
) -> HttpResponse:
    if params:
        url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"
    headers = {"Accept": "application/json"}
    body = None
    if json is not None:
        body = _json.dumps(json).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return (transport or urllib_transport)(method, url, body, headers, timeout)
