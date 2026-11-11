"""Small JSON-over-HTTP helper using a minimal HTTP/1.1 client over standard library socket.

Zero third-party packages, zero urllib.request / http.client / email imports.
Every client takes an optional ``transport``: a callable
``(method, url, body: bytes | None, headers: dict, timeout: float) -> HttpResponse``.
Tests pass an in-process fake; the default sends real requests over raw sockets.
``timeout`` bounds the whole request (connect, send and the full response), not each read.
"""

from __future__ import annotations

import json as _json
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode, urlsplit

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


def host_url(scheme: str, host: str, default_port: int, port: Optional[int] = None) -> str:
    """``scheme://host:port`` from ``host``, ``host:port``, an IPv6 literal (``fe80::1``,
    ``[fe80::1]`` or ``[fe80::1]:9000``) or a full URL (passed through without a trailing ``/``)."""
    host = host.strip()
    if "://" in host:
        return host.rstrip("/")
    if host.startswith("["):  # bracketed IPv6, maybe with a port
        if port is None and "]:" in host:
            return f"{scheme}://{host}"
        return f"{scheme}://{host.split(']', 1)[0]}]:{port or default_port}"
    if host.count(":") == 1:  # "host:port"
        if port is None:
            return f"{scheme}://{host}"
        host = host.split(":", 1)[0]
    elif ":" in host:  # bare IPv6 literal
        host = f"[{host}]"
    return f"{scheme}://{host}:{port or default_port}"


Transport = Callable[[str, str, Optional[bytes], Dict[str, str], float], HttpResponse]


class _DeadlineSocket:
    """``recv``/``sendall`` on ``sock`` bounded by one monotonic ``deadline`` for the whole request."""

    def __init__(self, sock: socket.socket, deadline: float) -> None:
        self.sock = sock
        self.deadline = deadline

    def _arm(self) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("request deadline passed")
        self.sock.settimeout(remaining)

    def recv(self, n: int) -> bytes:
        self._arm()
        return self.sock.recv(n)

    def sendall(self, data: bytes) -> None:
        self._arm()
        self.sock.sendall(data)


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise socket.timeout("request deadline passed")
    return left


# A lookup has no timeout of its own, so it runs in a thread that may outlive its request. At most
# this many lookups run at once; a request that cannot get a slot in time times out without a thread.
_RESOLVERS = threading.BoundedSemaphore(2)


def _resolve(host: str, port: int, deadline: float) -> list:
    """getaddrinfo bounded by ``deadline``; IP literals resolve inline without a thread."""
    try:
        return socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST)
    except socket.gaierror:
        pass
    if not _RESOLVERS.acquire(timeout=_remaining(deadline)):
        raise socket.timeout(f"resolving {host}: no free resolver within the request deadline")
    out: dict = {}

    def run() -> None:
        try:
            out["infos"] = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        except OSError as exc:
            out["error"] = exc
        finally:
            _RESOLVERS.release()

    t = threading.Thread(target=run, name="rebot-resolve", daemon=True)
    try:
        t.start()
    except BaseException:
        _RESOLVERS.release()
        raise
    t.join(_remaining(deadline))
    if t.is_alive():
        raise socket.timeout(f"resolving {host} took longer than the request deadline")
    if "error" in out:
        raise out["error"]
    return out["infos"]


def _connect(host: str, port: int, deadline: float) -> socket.socket:
    """Try each address with the time left before ``deadline``; stop once it has passed."""
    err: Optional[OSError] = None
    for family, kind, proto, _, addr in _resolve(host, port, deadline):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(_remaining(deadline))
            sock.connect(addr)
            return sock
        except OSError as exc:
            sock.close()
            if isinstance(exc, socket.timeout) and deadline <= time.monotonic():
                raise
            err = exc
    raise err or OSError(f"no address for {host}")


def _read_chunked(sock: _DeadlineSocket, initial: bytes) -> bytes:
    buf = bytearray(initial)
    body = bytearray()

    def read_until(delim: bytes) -> bytes:
        while delim not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise HttpError("connection closed before chunk delimiter")
            buf.extend(chunk)
        idx = buf.index(delim)
        line = bytes(buf[:idx])
        del buf[: idx + len(delim)]
        return line

    def read_exact(n: int) -> bytes:
        while len(buf) < n:
            chunk = sock.recv(max(4096, n - len(buf)))
            if not chunk:
                raise HttpError("connection closed while reading chunk data")
            buf.extend(chunk)
        data = bytes(buf[:n])
        del buf[:n]
        return data

    while True:
        line = read_until(b"\r\n").strip()
        if not line:
            continue
        size_str = line.split(b";", 1)[0].strip()
        try:
            chunk_size = int(size_str, 16)
        except ValueError as exc:
            raise HttpError(f"invalid chunk size: {line!r}") from exc
        if chunk_size == 0:
            while True:
                trailer = read_until(b"\r\n").strip()
                if not trailer:
                    break
            break
        body.extend(read_exact(chunk_size))
        crlf = read_exact(2)
        if crlf != b"\r\n":
            raise HttpError(f"expected CRLF after chunk data, got {crlf!r}")

    return bytes(body)


def socket_transport(
    method: str, url: str, body: Optional[bytes], headers: Dict[str, str], timeout: float
) -> HttpResponse:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme == "https":
        raise HttpError(f"rebot supports http:// only; armd and the reCamera services have no TLS: {url}")
    if scheme != "http":
        raise HttpError(f"unsupported URL scheme: {scheme}")

    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    host_hdr = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    req_headers = dict(headers)
    req_headers.setdefault("Host", host_hdr)
    req_headers.setdefault("Connection", "close")
    if body is not None and "Content-Length" not in req_headers:
        req_headers["Content-Length"] = str(len(body))

    req_lines = [f"{method.upper()} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in req_headers.items()]
    payload = "\r\n".join(req_lines).encode("latin-1") + b"\r\n\r\n"
    if body:
        payload += body

    deadline = time.monotonic() + timeout
    raw: Optional[socket.socket] = None
    try:
        raw = _connect(host, port, deadline)
        sock = _DeadlineSocket(raw, deadline)
        sock.sendall(payload)

        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise HttpError("connection closed before response headers were received")
            buf.extend(chunk)
            if len(buf) > 65536:
                raise HttpError("response headers too large")

        header_bytes, rest = buf.split(b"\r\n\r\n", 1)
        status_line, *header_lines = header_bytes.decode("latin-1").split("\r\n")
        status_parts = status_line.split(" ", 2)
        if len(status_parts) < 2:
            raise HttpError(f"malformed HTTP status line: {status_line}")
        try:
            status = int(status_parts[1])
        except ValueError as exc:
            raise HttpError(f"malformed HTTP status code: {status_line}") from exc

        res_headers: Dict[str, str] = {}
        for line in header_lines:
            hk, sep, hv = line.partition(":")
            if sep:
                res_headers[hk.strip()] = hv.strip()

        lower_headers = {k.lower(): v.lower() for k, v in res_headers.items()}
        transfer_encoding = lower_headers.get("transfer-encoding", "")

        if status in (204, 304) or method.upper() == "HEAD":
            resp_body = b""
        elif "chunked" in transfer_encoding:
            resp_body = _read_chunked(sock, rest)
        elif "content-length" in lower_headers:
            try:
                cl = int(lower_headers["content-length"])
            except ValueError as exc:
                raise HttpError(f"invalid Content-Length: {lower_headers['content-length']}") from exc
            body_buf = bytearray(rest)
            while len(body_buf) < cl:
                chunk = sock.recv(min(4096, cl - len(body_buf)))
                if not chunk:
                    raise HttpError("connection closed before Content-Length bytes were received")
                body_buf.extend(chunk)
            resp_body = bytes(body_buf[:cl])
        else:
            body_buf = bytearray(rest)
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                body_buf.extend(chunk)
            resp_body = bytes(body_buf)

        return HttpResponse(status, resp_body, res_headers)

    except (socket.timeout, TimeoutError) as exc:
        raise HttpTimeout(f"{method} {url} timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise HttpError(f"cannot reach {url}: {exc}") from exc
    finally:
        if raw is not None:
            try:
                raw.close()
            except OSError:
                pass


default_transport = socket_transport


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
    return (transport or default_transport)(method, url, body, headers, timeout)
