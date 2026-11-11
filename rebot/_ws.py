"""Minimal RFC 6455 WebSocket client on the standard library (no third-party packages).

Covers what armd and the camera stream use: the opening handshake (with an ``Origin`` header,
which armd's same-origin check compares with ``Host``), masked client frames, text and binary
messages, fragmentation, ping -> pong, close, and receive timeouts. No extensions
(no permessage-deflate), no subprotocols. Frames RFC 6455 forbids (reserved bits, masked server
frames, fragmented or oversized control frames, bad close payloads, invalid UTF-8) close the
connection with 1002 (1007 for UTF-8, 1009 for max_size) and raise :class:`WebSocketProtocolError`.

One thread may call :meth:`WebSocket.recv` while others call :meth:`WebSocket.send`.
``ws://`` only: armd and the camera stream serve no TLS, so ``wss://`` raises :class:`ValueError`.
"""

from __future__ import annotations

import binascii
import hashlib
import os
import select
import socket
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


class WebSocketProtocolError(WebSocketError):
    """The server sent a frame RFC 6455 forbids; the connection was closed with ``close_code``."""

    def __init__(self, message: str, close_code: int = 1002) -> None:
        self.close_code = close_code
        super().__init__(message)


def _valid_close_code(code: int) -> bool:
    return code in (1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011) or 3000 <= code <= 4999


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

    def _send_frame(self, opcode: int, payload: bytes, lock_timeout: Optional[float] = None) -> None:
        """Send one frame. ``lock_timeout`` bounds the wait for the send lock;
        :class:`WebSocketClosed` when it passes."""
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 1 << 16:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        mask = os.urandom(4)
        masked = _mask(payload, mask)
        if not self._send_lock.acquire(timeout=-1 if lock_timeout is None else lock_timeout):
            raise WebSocketClosed(None, "send lock not available")
        try:
            self.sock.sendall(header + mask + masked)
        except OSError as exc:
            raise WebSocketClosed(None, f"send failed: {exc}") from exc
        finally:
            self._send_lock.release()

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
        op, n = b0 & 0x0F, b1 & 0x7F
        if b0 & 0x70:
            raise WebSocketProtocolError("reserved bits set without a negotiated extension")
        if b1 & 0x80:
            raise WebSocketProtocolError("server frames must not be masked")
        if op not in (OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG):
            raise WebSocketProtocolError(f"unknown opcode {op}")
        if op >= 0x8 and (not b0 & 0x80 or n > 125):
            raise WebSocketProtocolError("control frames must not be fragmented or exceed 125 bytes")
        pos = 2
        if n == 126:
            if len(buf) < 4:
                return None
            n = struct.unpack_from("!H", buf, 2)[0]
            if n < 126:
                raise WebSocketProtocolError("payload length not minimally encoded")
            pos = 4
        elif n == 127:
            if len(buf) < 10:
                return None
            n = struct.unpack_from("!Q", buf, 2)[0]
            if n >> 63:
                raise WebSocketProtocolError("64-bit payload length with the most significant bit set")
            if n < 1 << 16:
                raise WebSocketProtocolError("payload length not minimally encoded")
            pos = 10
        if self.max_size is not None and n > self.max_size:
            raise WebSocketProtocolError(f"frame of {n} bytes exceeds max_size {self.max_size}", 1009)
        if len(buf) < pos + n:
            return None
        payload = bytes(buf[pos : pos + n])
        del buf[: pos + n]
        return bool(b0 & 0x80), op, payload

    def _fail(self, exc: WebSocketProtocolError) -> None:
        """Close with ``exc.close_code`` (best effort) and remember the failure for later calls."""
        if not self._close_sent:
            self._close_sent = True
            try:
                self._send_frame(OP_CLOSE, struct.pack("!H", exc.close_code))
            except WebSocketClosed:
                pass
        self._closed = WebSocketClosed(exc.close_code, str(exc))
        self._shutdown()

    def recv(self, timeout: Optional[float] = None) -> Union[str, bytes]:
        """Next message: ``str`` for text, ``bytes`` for binary.

        Raises :class:`WebSocketTimeout` after ``timeout`` seconds (None waits forever; a partly
        received frame stays buffered for the next call) and :class:`WebSocketClosed` when the
        connection closes. Pings are answered here.
        """
        if self._closed is not None:
            raise self._closed
        try:
            return self._recv(timeout)
        except WebSocketProtocolError as exc:
            self._fail(exc)
            raise

    def _recv(self, timeout: Optional[float]) -> Union[str, bytes]:
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
                if len(payload) == 1:
                    raise WebSocketProtocolError("close frame with a 1-byte payload")
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else None
                if code is not None and not _valid_close_code(code):
                    raise WebSocketProtocolError(f"invalid close code {code}")
                reason = _utf8(payload[2:])
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
                    raise WebSocketProtocolError("new message before the previous fragmented one ended")
                if fin:
                    return _utf8(payload) if op == OP_TEXT else payload
                self._frag_op, self._fragments = op, [payload]
                continue
            if op == OP_CONT:
                if not self._fragments:
                    raise WebSocketProtocolError("continuation frame without a message")
                self._fragments.append(payload)
                if self.max_size is not None and sum(len(p) for p in self._fragments) > self.max_size:
                    raise WebSocketProtocolError(f"message exceeds max_size {self.max_size}", 1009)
                if fin:
                    data, self._fragments = b"".join(self._fragments), []
                    return _utf8(data) if self._frag_op == OP_TEXT else data
                continue

    # ---- closing -------------------------------------------------------------

    def close(self, code: int = 1000) -> None:
        """Send a close frame (best effort) and close the socket; unblocks a thread in :meth:`recv`."""
        if not self._close_sent and self._closed is None:
            self._close_sent = True
            try:  # bounded: a stuck sender or reader never delays the shutdown below for long
                self._send_frame(OP_CLOSE, struct.pack("!H", code), lock_timeout=0.5)
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


def _utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WebSocketProtocolError("text is not valid UTF-8", 1007) from exc


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
    """Open ``ws://`` ``url`` (``wss://`` raises :class:`ValueError`: armd has no TLS).

    ``origin=True`` sends ``Origin: http://<host>:<port>`` matching the ``Host`` header (armd's
    same-origin check accepts that); a string sends that value; ``False`` sends none.
    """
    parts = urlsplit(url)
    if parts.scheme == "wss":
        raise ValueError(f"rebot supports ws:// only; armd has no TLS: {url}")
    if parts.scheme != "ws":
        raise ValueError(f"not a ws:// URL: {url}")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    host_hdr = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        raise WebSocketError(f"cannot connect to {host}:{port}: {exc}") from exc
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = binascii.b2a_base64(os.urandom(16), newline=False).decode("ascii")
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host_hdr}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if origin is True:
            lines.append(f"Origin: http://{host_hdr}")
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
        digest = hashlib.sha1((key + _GUID).encode("ascii")).digest()
        expect = binascii.b2a_base64(digest, newline=False).decode("ascii")
        if hdrs.get("sec-websocket-accept") != expect:
            raise WebSocketError("handshake failed: bad Sec-WebSocket-Accept")
        sock.settimeout(send_timeout)  # reads use select(); this bounds blocking sends
        return WebSocket(sock, rest, max_size)
    except BaseException:
        sock.close()
        raise
