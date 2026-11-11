import json
import os
import pathlib
import socket
import subprocess
import sys
import threading

import pytest

from rebot import _http


def test_import_rebot_footprint():
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    code = f"import sys\nsys.path.insert(0, {repo_root!r})\nimport rebot\nprint(repr(list(sys.modules.keys())))\n"
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded_modules = eval(res.stdout.strip())
    forbidden = [
        "urllib.request",
        "http.client",
        "email",
        "ssl",
        "logging",
        "uuid",
        "platform",
        "base64",
        "numpy",
        "httpx",
        "websockets",
    ]
    violations = []
    for mod in forbidden:
        matched = [m for m in loaded_modules if m == mod or m.startswith(f"{mod}.")]
        if matched:
            violations.extend(matched)

    assert not violations, f"Forbidden modules loaded upon 'import rebot': {violations}"


def test_lazy_logging():
    # Without REBOT_DEBUG, logging should not be imported
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    code = (
        f"import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        f"from rebot._log import _log\n"
        f"_log('silent message')\n"
        f"_log.warning('another silent message')\n"
        f"print('logging' in sys.modules)\n"
    )
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "False"

    # With REBOT_DEBUG=1, logging should be imported
    env = os.environ.copy()
    env["REBOT_DEBUG"] = "1"
    code_debug = (
        f"import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        f"from rebot._log import _log\n"
        f"_log('debug message')\n"
        f"print('logging' in sys.modules)\n"
    )
    res_debug = subprocess.run(
        [sys.executable, "-I", "-c", code_debug],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert res_debug.stdout.strip() == "True"


def test_http_socket_transport_content_length():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def server_thread():
        conn, _ = srv.accept()
        with conn:
            req = b""
            while b"\r\n\r\n" not in req:
                req += conn.recv(1024)
            body = json.dumps({"status": "ok", "value": 42}).encode("utf-8")
            resp = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )
            conn.sendall(resp)

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    res = _http.request("GET", f"http://127.0.0.1:{port}/test")
    assert res.status == 200
    assert res.ok
    assert res.json() == {"status": "ok", "value": 42}
    t.join(timeout=2.0)
    srv.close()


def test_http_socket_transport_chunked():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def server_thread():
        conn, _ = srv.accept()
        with conn:
            req = b""
            while b"\r\n\r\n" not in req:
                req += conn.recv(1024)
            resp_header = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: close\r\n\r\n"
            )
            conn.sendall(resp_header)
            chunk1 = b'{"msg": '
            chunk2 = b'"chunked"}'
            conn.sendall(f"{len(chunk1):x}\r\n".encode() + chunk1 + b"\r\n")
            conn.sendall(f"{len(chunk2):x}\r\n".encode() + chunk2 + b"\r\n")
            conn.sendall(b"0\r\n\r\n")

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    res = _http.request("POST", f"http://127.0.0.1:{port}/chunked", json={"ping": True})
    assert res.status == 200
    assert res.ok
    assert res.json() == {"msg": "chunked"}
    t.join(timeout=2.0)
    srv.close()


def test_http_socket_transport_delete_and_timeout():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def server_thread():
        conn, _ = srv.accept()
        with conn:
            req = b""
            while b"\r\n\r\n" not in req:
                req += conn.recv(1024)
            # Sleep longer than client timeout to trigger timeout
            import time

            time.sleep(0.3)
            conn.sendall(b"HTTP/1.1 204 No Content\r\n\r\n")

    t = threading.Thread(target=server_thread, daemon=True)
    t.start()

    with pytest.raises(_http.HttpTimeout):
        _http.request("DELETE", f"http://127.0.0.1:{port}/timeout", timeout=0.1)

    t.join(timeout=2.0)
    srv.close()
