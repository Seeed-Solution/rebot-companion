"""Single-port Same-Origin Playground and API Reverse Proxy.

Serves static assets (HTML/JS/CSS/WASM/ONNX) from `--static-dir` with SPA fallback,
MIME type mappings, and optional COOP/COEP isolation headers.
Proxies `/api/v1/*` HTTP calls and WebSocket connections (`/api/v1/ws`) to armd,
rewriting `Host`, `Origin`, and `Referer` so armd's same-origin guard accepts requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import urllib.parse
from pathlib import Path
from typing import Any

# Static file extension to MIME type
MIME_TYPES: dict[str, str] = {
    ".wasm": "application/wasm",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".json": "application/json",
    ".onnx": "application/octet-stream",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".map": "application/json",
}


def _send_response(
    writer: asyncio.StreamWriter,
    status: int,
    phrase: str,
    body: bytes,
    headers: list[tuple[str, str]] | None = None,
) -> None:
    lines = [f"HTTP/1.1 {status} {phrase}\r\n".encode("latin1")]
    if headers:
        for k, v in headers:
            lines.append(f"{k}: {v}\r\n".encode("latin1"))
    else:
        lines.append(f"Content-Length: {len(body)}\r\n".encode("latin1"))
        lines.append(b"Connection: close\r\n")
    lines.append(b"\r\n")
    lines.append(body)
    writer.write(b"".join(lines))


class PlaygroundProxyServer:
    """Async single-port static asset and armd HTTP/WS proxy server."""

    def __init__(
        self,
        static_dir: str | Path,
        armd_url: str = "http://127.0.0.1:8765",
        host: str = "127.0.0.1",
        port: int = 4173,
        enable_coop: bool = False,
    ) -> None:
        self.static_dir = Path(static_dir).resolve()
        self.armd_url = armd_url.rstrip("/")
        self._configured_host = host
        self._configured_port = int(port)
        self.enable_coop = bool(enable_coop)
        self._server: asyncio.Server | None = None
        self._port: int | None = None

        u = urllib.parse.urlsplit(self.armd_url)
        self._u_host = u.hostname or "127.0.0.1"
        self._u_port = u.port or (80 if u.scheme == "http" else 443)
        self._u_netloc = u.netloc or f"{self._u_host}:{self._u_port}"
        self._target_origin = f"{u.scheme or 'http'}://{self._u_netloc}"

    @property
    def host(self) -> str:
        return self._configured_host

    @property
    def port(self) -> int:
        if self._port is not None:
            return self._port
        if self._server and self._server.sockets:
            return int(self._server.sockets[0].getsockname()[1])
        return self._configured_port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._handle_client,
            self._configured_host,
            self._configured_port,
        )
        if self._server.sockets:
            self._port = int(self._server.sockets[0].getsockname()[1])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> PlaygroundProxyServer:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                return

            req_line = line.decode("latin1").strip()
            parts = req_line.split(" ", 2)
            if len(parts) < 2:
                writer.close()
                return

            method = parts[0].upper()
            raw_path = parts[1]
            parsed_url = urllib.parse.urlsplit(raw_path)
            pathname = parsed_url.path

            headers: dict[str, str] = {}
            header_list: list[tuple[str, str]] = []
            while True:
                hline = await reader.readline()
                if hline in (b"\r\n", b"\n", b""):
                    break
                h_str = hline.decode("latin1").rstrip("\r\n")
                k, _, v = h_str.partition(":")
                key = k.strip()
                val = v.strip()
                headers[key.lower()] = val
                header_list.append((key, val))

            # The upstream gets armd's own origin below, so armd's same-origin check no longer
            # sees the browser's: apply it here (Origin must match this proxy's Host).
            origin = headers.get("origin")
            if origin is not None and origin != f"http://{headers.get('host', '')}":
                _send_response(writer, 403, "Forbidden", b"Cross-origin request refused")
                writer.close()
                return

            # 1. WebSocket upgrade check
            is_ws = headers.get("upgrade", "").lower() == "websocket"
            if is_ws:
                await self._proxy_websocket(reader, writer, line, header_list)
                return

            # 2. HTTP API proxy check (/api/v1/*)
            if pathname.startswith("/api/v1/"):
                await self._proxy_http(reader, writer, line, method, headers, header_list)
                return

            # 3. Static asset serving
            await self._serve_static(writer, method, pathname)

        except Exception:
            with contextlib.suppress(Exception):
                writer.close()

    async def _proxy_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        initial_line: bytes,
        header_list: list[tuple[str, str]],
    ) -> None:
        try:
            u_reader, u_writer = await asyncio.open_connection(self._u_host, self._u_port)
        except Exception as exc:
            _send_response(writer, 502, "Bad Gateway", f"Cannot connect to armd: {exc}".encode())
            writer.close()
            return

        u_writer.write(initial_line)
        for k, v in header_list:
            k_low = k.lower()
            if k_low == "host":
                u_writer.write(f"Host: {self._u_netloc}\r\n".encode("latin1"))
            elif k_low in ("origin", "referer"):
                continue
            else:
                u_writer.write(f"{k}: {v}\r\n".encode("latin1"))
        u_writer.write(f"Origin: {self._target_origin}\r\n".encode("latin1"))
        u_writer.write(b"\r\n")
        await u_writer.drain()

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await src.read(65536)
                    if not chunk:
                        break
                    dst.write(chunk)
                    await dst.drain()
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    dst.close()

        t1 = asyncio.create_task(pipe(reader, u_writer))
        t2 = asyncio.create_task(pipe(u_reader, writer))
        await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

    async def _proxy_http(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        initial_line: bytes,
        method: str,
        headers: dict[str, str],
        header_list: list[tuple[str, str]],
    ) -> None:
        # Read body if present
        cl_str = headers.get("content-length")
        content_length = int(cl_str) if cl_str and cl_str.isdigit() else 0
        body = await reader.readexactly(content_length) if content_length > 0 else b""

        try:
            u_reader, u_writer = await asyncio.open_connection(self._u_host, self._u_port)
        except Exception as exc:
            _send_response(writer, 502, "Bad Gateway", f"Cannot connect to armd: {exc}".encode())
            writer.close()
            return

        u_writer.write(initial_line)
        has_conn = False
        for k, v in header_list:
            k_low = k.lower()
            if k_low == "host":
                u_writer.write(f"Host: {self._u_netloc}\r\n".encode("latin1"))
            elif k_low in ("origin", "referer"):
                continue
            elif k_low == "connection":
                u_writer.write(b"Connection: close\r\n")
                has_conn = True
            else:
                u_writer.write(f"{k}: {v}\r\n".encode("latin1"))
        if not has_conn:
            u_writer.write(b"Connection: close\r\n")
        u_writer.write(f"Origin: {self._target_origin}\r\n".encode("latin1"))
        u_writer.write(f"Referer: {self._target_origin}/\r\n".encode("latin1"))
        u_writer.write(b"\r\n")
        if body:
            u_writer.write(body)
        await u_writer.drain()

        # Read upstream response and pipe to client
        try:
            while True:
                resp_chunk = await u_reader.read(65536)
                if not resp_chunk:
                    break
                writer.write(resp_chunk)
                await writer.drain()
        finally:
            with contextlib.suppress(Exception):
                u_writer.close()
            with contextlib.suppress(Exception):
                writer.close()

    async def _serve_static(self, writer: asyncio.StreamWriter, method: str, pathname: str) -> None:
        rel = pathname.lstrip("/")
        target_path = (self.static_dir / rel).resolve()

        # Safe path containment
        try:
            target_path.relative_to(self.static_dir)
        except ValueError:
            _send_response(writer, 403, "Forbidden", b"Access denied")
            writer.close()
            return

        file_to_serve: Path | None = None
        if target_path.is_file():
            file_to_serve = target_path
        elif target_path.is_dir() and (target_path / "index.html").is_file():
            file_to_serve = target_path / "index.html"
        else:
            filename = pathname.split("/")[-1]
            has_ext = "." in filename and not filename.startswith(".")
            if not has_ext:
                # SPA fallback only for extensionless routes
                index_candidate = self.static_dir / "index.html"
                if index_candidate.is_file():
                    file_to_serve = index_candidate

        if file_to_serve is None or not file_to_serve.is_file():
            _send_response(writer, 404, "Not Found", b"File not found")
            writer.close()
            return

        data = file_to_serve.read_bytes()
        ext = file_to_serve.suffix.lower()
        mime = MIME_TYPES.get(ext, "application/octet-stream")

        resp_headers = [
            ("Content-Type", mime),
            ("Content-Length", str(len(data))),
            ("Connection", "close"),
            ("Access-Control-Allow-Origin", "*"),
        ]
        if self.enable_coop:
            resp_headers.append(("Cross-Origin-Opener-Policy", "same-origin"))
            resp_headers.append(("Cross-Origin-Embedder-Policy", "require-corp"))
            resp_headers.append(("Cross-Origin-Resource-Policy", "same-origin"))

        body = b"" if method == "HEAD" else data
        _send_response(writer, 200, "OK", body, resp_headers)
        writer.close()
