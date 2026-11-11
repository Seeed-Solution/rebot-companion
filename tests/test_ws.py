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


@pytest.mark.parametrize(
    "frame, close_code",
    [
        (b"\xc1\x02hi", 1002),  # RSV1 set, no extension negotiated
        (b"\x81\x82\x00\x00\x00\x00hi", 1002),  # masked server frame
        (b"\x09\x00", 1002),  # fragmented ping
        (b"\x89\x7e\x00\x7e" + b"x" * 126, 1002),  # ping payload > 125
        (b"\x88\x01\x03", 1002),  # close payload of 1 byte
        (b"\x88\x02\x03\xed", 1002),  # close code 1005 must not be sent
        (b"\x88\x04\x03\xe8\xff\xfe", 1007),  # close reason not UTF-8
        (b"\x81\x02\xff\xfe", 1007),  # text not UTF-8
        (b"\x83\x00", 1002),  # reserved opcode
        (b"\x82\x7e\x00\x05hello", 1002),  # 16-bit length form for 5 bytes
        (b"\x82\x7f" + (5).to_bytes(8, "big") + b"hello", 1002),  # 64-bit length form for 5 bytes
        (b"\x82\x7f" + (65535).to_bytes(8, "big"), 1002),  # 64-bit form below 65536
        (b"\x82\x7f" + (1 << 63).to_bytes(8, "big"), 1002),  # most significant bit set
    ],
)
def test_protocol_violations_close_with_an_error_code(frame, close_code):
    got: list = []
    done = threading.Event()

    def script(conn):
        conn.sendall(frame)
        conn.settimeout(2)
        data = b""
        try:
            while len(data) < 8:
                chunk = conn.recv(64)
                if not chunk:
                    break
                data += chunk
        except OSError:
            pass
        got.append(data)
        done.set()

    ws = _ws.connect(_raw_server(script))
    with pytest.raises(_ws.WebSocketError) as info:
        ws.recv(timeout=2)
    assert not isinstance(info.value, _ws.WebSocketClosed)
    assert done.wait(3)
    data = got[0]
    assert data[0] == 0x88 and data[1] & 0x7F == 2  # masked close frame with a status code
    code = struct.unpack("!H", _ws._mask(data[6:8], data[2:6]))[0]
    assert code == close_code
    with pytest.raises(_ws.WebSocketClosed):  # later calls see the closed connection
        ws.recv(timeout=0.1)


def test_tls_socket_is_never_read_and_written_at_once():
    """An ssl.SSLSocket (it has pending()) must not see recv and sendall from two threads together."""
    a, b = socket.socketpair()
    overlap = []
    state = {"in_recv": False}

    class FakeTls:
        def pending(self):
            return 2

        def recv(self, n):
            state["in_recv"] = True
            time.sleep(0.2)
            state["in_recv"] = False
            return b"\x81\x00"  # empty text message

        def sendall(self, data):
            if state["in_recv"]:
                overlap.append(len(data))

        def fileno(self):
            return a.fileno()

        def shutdown(self, how):
            pass

        def close(self):
            a.close()
            b.close()

    ws = _ws.WebSocket(FakeTls())
    t = threading.Thread(target=ws.recv, kwargs={"timeout": 2})
    t.start()
    time.sleep(0.05)
    ws.send("heartbeat")
    t.join(2)
    ws.close()
    assert overlap == []
