"""rebot._ws against the ``websockets`` reference server and a raw-socket server."""

from __future__ import annotations

import base64
import hashlib
import socket
import struct
import threading
import time

import pytest
from websockets.sync.server import serve

from rebot import _ws


@pytest.fixture
def server():
    seen: dict = {}

    def handler(ws):
        seen["origin"] = ws.request.headers.get("Origin")
        seen["host"] = ws.request.headers.get("Host")
        seen["path"] = ws.request.path
        for msg in ws:
            if msg == "fragments":
                ws.send(iter(["ab", "cd", "ef"]))
            elif msg == "ping-then-text":
                ws.ping(b"x")
                ws.send("after-ping")
            elif msg == "close":
                ws.close(4001, "bye")
                return
            elif msg == "slow":
                time.sleep(0.3)
                ws.send("late")
            else:
                ws.send(msg)

    srv = serve(handler, "127.0.0.1", 0, max_size=None)
    port = srv.socket.getsockname()[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"ws://127.0.0.1:{port}", seen
    srv.shutdown()


def test_text_binary_and_sizes(server):
    url, seen = server
    with _ws.connect(url + "/api/v1/ws?obs_hz=10") as ws:
        ws.send("hello")
        assert ws.recv(timeout=2) == "hello"
        for n in (0, 125, 126, 65535, 65536, 300_000):
            data = bytes(range(256)) * (n // 256) + bytes(n % 256)
            ws.send(data)
            assert ws.recv(timeout=5) == data
        ws.send("héllo ✓")
        assert ws.recv(timeout=2) == "héllo ✓"
    port = url.rsplit(":", 1)[1]
    assert seen["origin"] == f"http://127.0.0.1:{port}"
    assert seen["host"] == f"127.0.0.1:{port}"
    assert seen["path"] == "/api/v1/ws?obs_hz=10"


def test_fragments_ping_and_close(server):
    url, _ = server
    ws = _ws.connect(url)
    ws.send("fragments")
    assert ws.recv(timeout=2) == "abcdef"
    ws.send("ping-then-text")
    assert ws.recv(timeout=2) == "after-ping"
    ws.send("close")
    with pytest.raises(_ws.WebSocketClosed) as info:
        ws.recv(timeout=2)
    assert info.value.code == 4001 and info.value.reason == "bye"
    with pytest.raises(_ws.WebSocketClosed):
        ws.send("x")


def test_recv_timeout_then_message(server):
    url, _ = server
    with _ws.connect(url) as ws:
        ws.send("slow")
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.05)
        assert ws.recv(timeout=2) == "late"


def test_close_from_other_thread_unblocks_recv(server):
    url, _ = server
    ws = _ws.connect(url)
    errors: list = []

    def reader():
        try:
            ws.recv()
        except _ws.WebSocketClosed as exc:
            errors.append(exc)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.1)
    ws.close()
    t.join(2)
    assert not t.is_alive() and errors


def test_origin_options(server):
    url, seen = server
    with _ws.connect(url, origin=False) as ws:
        ws.send("a")
        ws.recv(timeout=2)
    assert seen["origin"] is None
    with _ws.connect(url, origin="http://example") as ws:
        ws.send("a")
        ws.recv(timeout=2)
    assert seen["origin"] == "http://example"


def test_handshake_refused():
    with pytest.raises(_ws.WebSocketError):
        _ws.connect("ws://127.0.0.1:9/", timeout=1)
    with pytest.raises(ValueError):
        _ws.connect("http://127.0.0.1/")


def _raw_server(script):
    """Accept one client, answer the handshake, then run ``script(conn)``."""
    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)

    def run():
        conn, _ = lsock.accept()
        req = b""
        while b"\r\n\r\n" not in req:
            req += conn.recv(4096)
        lines = req.split(b"\r\n")
        key = [ln.split(b":", 1)[1].strip() for ln in lines if ln.lower().startswith(b"sec-websocket-key")][0]
        acc = base64.b64encode(hashlib.sha1(key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest())
        conn.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + acc + b"\r\n\r\n"
        )
        script(conn)
        conn.close()
        lsock.close()

    threading.Thread(target=run, daemon=True).start()
    return f"ws://127.0.0.1:{lsock.getsockname()[1]}/"


def test_partial_frame_survives_timeout():
    payload = b"x" * 1000
    frame = struct.pack("!BBH", 0x82, 126, len(payload)) + payload
    gate = threading.Event()

    def script(conn):
        conn.sendall(frame[:300])
        gate.wait(2)
        conn.sendall(frame[300:])
        time.sleep(0.2)

    ws = _ws.connect(_raw_server(script))
    with pytest.raises(TimeoutError):
        ws.recv(timeout=0.1)
    gate.set()
    assert ws.recv(timeout=2) == payload
    ws.close()


def test_client_frames_are_masked_and_eof_is_closed():
    got: list = []

    def script(conn):
        hdr = conn.recv(2)
        got.append(hdr)
        conn.recv(4 + 5)

    ws = _ws.connect(_raw_server(script))
    ws.send("hello")
    with pytest.raises(_ws.WebSocketClosed):
        ws.recv(timeout=2)
    assert got[0][0] == 0x81 and got[0][1] == 0x80 | 5


def test_mask_roundtrip():
    data = bytes(range(37))
    mask = b"\x01\x02\x03\x04"
    assert _ws._mask(_ws._mask(data, mask), mask) == data
    assert _ws._mask(b"\x00\x00\x00\x00\x00", mask) == b"\x01\x02\x03\x04\x01"
