"""Minimal RFC 6455 WebSocket client on the standard library (no third-party packages).

Covers what armd and the camera stream use: the opening handshake (with an ``Origin`` header,
which armd's same-origin check compares with ``Host``), masked client frames, text and binary
messages, fragmentation, ping -> pong, close, and receive timeouts. No extensions
(no permessage-deflate), no subprotocols.

One thread may call :meth:`WebSocket.recv` while others call :meth:`WebSocket.send`.
"""

from __future__ import annotations

import base64
import hashlib
import os
import select
import socket
import ssl
import struct
import threading
import time
from typing import Optional, Union
from urllib.parse import urlsplit

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(OSError):
    """Handshake failed or the peer broke the protocol."""


class WebSocketClosed(WebSocketError):
    """The connection is closed (close frame received, or the socket ended)."""

    def __init__(self, code: Optional[int] = None, reason: str = "") -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"closed (code {code}{', ' + reason if reason else ''})")


class WebSocketTimeout(TimeoutError):
    """No complete message arrived within the receive timeout."""


class WebSocket:
    def __init__(self, sock: socket.socket, buffered: bytes = b"", max_size: Optional[int] = None) -> None:
        self.sock = sock
        self.max_size = max_size
        self._buf = bytearray(buffered)
        self._send_lock = threading.Lock()
        self._closed: Optional[WebSocketClosed] = None
        self._close_sent = False
        self._fragments: list = []
        self._frag_op = 0

    # ---- sending -------------------------------------------------------------

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 1 << 16:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        mask = os.urandom(4)
        masked = _mask(payload, mask)
        with self._send_lock:
            try:
                self.sock.sendall(header + mask + masked)
            except OSError as exc:
                raise WebSocketClosed(None, f"send failed: {exc}") from exc

    def send(self, data: Union[str, bytes]) -> None:
        """Send one text (``str``) or binary (``bytes``) message."""
        if self._closed is not None:
            raise self._closed
        if isinstance(data, str):
            self._send_frame(OP_TEXT, data.encode("utf-8"))
        else:
            self._send_frame(OP_BINARY, bytes(data))

    def ping(self, payload: bytes = b"") -> None:
        self._send_frame(OP_PING, payload)

    # ---- receiving -----------------------------------------------------------

    def _fill(self, deadline: Optional[float]) -> None:
        """Read more bytes into the buffer; raise WebSocketTimeout at ``deadline``."""
        pending = isinstance(self.sock, ssl.SSLSocket) and self.sock.pending() > 0
        if not pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                ready, _, _ = select.select([self.sock], [], [], remaining)
            except (OSError, ValueError) as exc:  # socket closed by another thread
                raise WebSocketClosed(None, "socket closed") from exc
            if not ready:
                raise WebSocketTimeout("no message within the timeout")
        try:
            chunk = self.sock.recv(65536)
        except (OSError, ValueError) as exc:
            raise WebSocketClosed(None, f"recv failed: {exc}") from exc
        if not chunk:
            raise WebSocketClosed(None, "connection ended")
        self._buf += chunk

    def _parse(self) -> Optional[tuple]:
        """One complete frame ``(fin, opcode, payload)`` from the buffer, or None if incomplete."""
        buf = self._buf
        if len(buf) < 2:
            return None
        b0, b1 = buf[0], buf[1]
        n = b1 & 0x7F
        pos = 2
        if n == 126:
            if len(buf) < 4:
                return None
            n = struct.unpack_from("!H", buf, 2)[0]
            pos = 4
        elif n == 127:
            if len(buf) < 10:
                return None
            n = struct.unpack_from("!Q", buf, 2)[0]
            pos = 10
        if self.max_size is not None and n > self.max_size:
            raise WebSocketError(f"frame of {n} bytes exceeds max_size {self.max_size}")
        mask = None
        if b1 & 0x80:  # servers must not mask, but accept it
            if len(buf) < pos + 4:
                return None
            mask = bytes(buf[pos : pos + 4])
            pos += 4
        if len(buf) < pos + n:
            return None
        payload = bytes(buf[pos : pos + n])
        del buf[: pos + n]
        if mask is not None:
            payload = _mask(payload, mask)
        return bool(b0 & 0x80), b0 & 0x0F, payload

    def recv(self, timeout: Optional[float] = None) -> Union[str, bytes]:
        """Next message: ``str`` for text, ``bytes`` for binary.

        Raises :class:`WebSocketTimeout` after ``timeout`` seconds (None waits forever; a partly
        received frame stays buffered for the next call) and :class:`WebSocketClosed` when the
        connection closes. Pings are answered here.
        """
        if self._closed is not None:
            raise self._closed
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            frame = self._parse()
            if frame is None:
                try:
                    self._fill(deadline)
                except WebSocketClosed as exc:
                    self._closed = exc
                    raise
                continue
            fin, op, payload = frame
            if op == OP_PING:
                try:
                    self._send_frame(OP_PONG, payload)
                except WebSocketClosed:
                    pass
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else None
                reason = payload[2:].decode("utf-8", "replace")
                if not self._close_sent:
                    try:
                        self._close_sent = True
                        self._send_frame(OP_CLOSE, payload[:2])
                    except WebSocketClosed:
                        pass
                self._closed = WebSocketClosed(code, reason)
                self._shutdown()
                raise self._closed
            if op in (OP_TEXT, OP_BINARY):
                if self._fragments:
                    raise WebSocketError("new message before the previous fragmented one ended")
                if fin:
                    return payload.decode("utf-8") if op == OP_TEXT else payload
                self._frag_op, self._fragments = op, [payload]
                continue
            if op == OP_CONT:
                if not self._fragments:
                    raise WebSocketError("continuation frame without a message")
                self._fragments.append(payload)
                if self.max_size is not None and sum(len(p) for p in self._fragments) > self.max_size:
                    raise WebSocketError(f"message exceeds max_size {self.max_size}")
                if fin:
                    data, self._fragments = b"".join(self._fragments), []
                    return data.decode("utf-8") if self._frag_op == OP_TEXT else data
                continue
            raise WebSocketError(f"unknown opcode {op}")

    # ---- closing -------------------------------------------------------------

    def close(self, code: int = 1000) -> None:
        """Send a close frame (best effort) and close the socket; unblocks a thread in :meth:`recv`."""
        if not self._close_sent and self._closed is None:
            self._close_sent = True
            try:
                self._send_frame(OP_CLOSE, struct.pack("!H", code))
            except WebSocketClosed:
                pass
        if self._closed is None:
            self._closed = WebSocketClosed(code, "closed by client")
        self._shutdown()

    def _shutdown(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self) -> WebSocket:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _mask(data: bytes, mask: bytes) -> bytes:
    if not data:
        return b""
    n = len(data)
    key = int.from_bytes((mask * (n // 4 + 1))[:n], "big")
    return (int.from_bytes(data, "big") ^ key).to_bytes(n, "big")


def connect(
    url: str,
    timeout: float = 5.0,
    origin: Union[str, bool] = True,
    headers: Optional[dict] = None,
    max_size: Optional[int] = None,
    send_timeout: float = 10.0,
) -> WebSocket:
    """Open ``ws://`` or ``wss://`` ``url``.

    ``origin=True`` sends ``Origin: http://<host>:<port>`` matching the ``Host`` header (armd's
    same-origin check accepts that); a string sends that value; ``False`` sends none.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("ws", "wss"):
        raise ValueError(f"not a ws:// or wss:// URL: {url}")
    secure = parts.scheme == "wss"
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if secure else 80)
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    host_hdr = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        raise WebSocketError(f"cannot connect to {host}:{port}: {exc}") from exc
    try:
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host_hdr}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if origin is True:
            lines.append(f"Origin: {'https' if secure else 'http'}://{host_hdr}")
        elif origin:
            lines.append(f"Origin: {origin}")
        for k, v in (headers or {}).items():
            lines.append(f"{k}: {v}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(4096)
            if not chunk:
                raise WebSocketError("connection closed during the handshake")
            resp += chunk
            if len(resp) > 65536:
                raise WebSocketError("handshake response too large")
        head, rest = resp.split(b"\r\n\r\n", 1)
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        status = status_line.split(" ", 2)
        if len(status) < 2 or status[1] != "101":
            raise WebSocketError(f"handshake refused: {status_line}")
        hdrs = {}
        for line in header_lines:
            k, _, v = line.partition(":")
            hdrs[k.strip().lower()] = v.strip()
        expect = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if hdrs.get("sec-websocket-accept") != expect:
            raise WebSocketError("handshake failed: bad Sec-WebSocket-Accept")
        sock.settimeout(send_timeout)  # reads use select(); this bounds blocking sends
        return WebSocket(sock, rest, max_size)
    except BaseException:
        sock.close()
        raise
