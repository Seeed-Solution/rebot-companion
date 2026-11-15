"""HTTP client for hot-swapping scenes/tools and driving virtual I/O on rebot-sim."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .spec import resolve_builtin_path


class SimClientError(RuntimeError):
    """Raised when rebot-sim HTTP control returns an error or cannot be reached."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _default_base_url() -> str:
    host = os.environ.get("REBOT_SIM_HTTP_HOST", os.environ.get("REBOT_HOST", "127.0.0.1"))
    port = os.environ.get("REBOT_SIM_HTTP_PORT", "8090")
    return f"http://{host}:{port}"


def resolve_pack_argument(kind: str, ref: str | Path | None) -> str | None:
    """Resolve local path or built-in pack id to an absolute path for rebot-sim."""
    if ref is None:
        return None
    raw = str(ref).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.exists():
        return str(p.resolve())
    builtin = resolve_builtin_path(kind, raw)
    if builtin is not None:
        return str(builtin.resolve())
    return raw


class SimSceneClient:
    """Thin client for `rebot-sim` HTTP routes (`/sim/scene`, `/sim/io`, `/sim/state`, `/sim/reset`)."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 10.0):
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.timeout_s = float(timeout_s)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(err_body).get("error", err_body)
            except json.JSONDecodeError:
                msg = err_body or exc.reason
            raise SimClientError(f"{method} {path} failed ({exc.code}): {msg}", status_code=exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise SimClientError(f"Cannot reach rebot-sim at {url}: {exc}") from exc

    def get_scene(self) -> dict[str, Any]:
        return self._request("GET", "/sim/scene")

    def load_scene(
        self,
        scene: str | Path | None = None,
        *,
        tool: str | Path | None = None,
        seed: int | None = None,
        objects: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if scene is not None:
            payload["scene"] = resolve_pack_argument("scene", scene)
        if tool is not None:
            payload["tool"] = resolve_pack_argument("tool", tool)
        if seed is not None:
            payload["seed"] = int(seed)
        if objects is not None:
            payload["objects"] = objects
        if extra_payload:
            payload.update(extra_payload)
        return self._request("POST", "/sim/scene", payload)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return self._request("POST", "/sim/reset", {"seed": seed})

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/sim/state")

    def get_io(self) -> dict[str, Any]:
        return self._request("GET", "/sim/io")

    def set_io(self, outputs: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if outputs:
            merged.update(outputs)
        merged.update(kwargs)
        return self._request("POST", "/sim/io", merged)


DEFAULT_ARMD_HTTP_URL = "http://127.0.0.1:8765"


def _collect_pack_files(pack_dir: str | Path) -> dict[str, str]:
    """Encode all files under `pack_dir` for JSON upload to /api/v1/sandbox/*."""
    import base64

    root = Path(pack_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pack directory does not exist: {root}")
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            try:
                p.resolve().relative_to(root)
            except ValueError:
                raise ValueError(f"Symlink {p} resolves outside pack directory {root}") from None
        if not p.is_file():
            continue
        try:
            p.resolve().relative_to(root)
        except ValueError:
            raise ValueError(f"Symlink {p} resolves outside pack directory {root}") from None
        rel = p.relative_to(root).as_posix()
        if any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts):
            continue
        raw = p.read_bytes()
        if p.suffix.lower() in {".json", ".xml", ".txt", ".md"}:
            try:
                out[rel] = raw.decode("utf-8")
                continue
            except UnicodeDecodeError:
                pass
        out[rel] = "base64:" + base64.b64encode(raw).decode("ascii")
    return out


class ArmdSandboxClient:
    """HTTP client for `armd` sandbox management (`/api/v1/sandbox/*`)."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 15.0) -> None:
        if base_url:
            raw = base_url
        elif os.environ.get("REBOT_ARMD_HTTP"):
            raw = os.environ["REBOT_ARMD_HTTP"]
        elif os.environ.get("REBOT_HOST"):
            raw = f"http://{os.environ['REBOT_HOST']}:8765"
        else:
            raw = DEFAULT_ARMD_HTTP_URL
        if not str(raw).startswith("http"):
            raw = f"http://{raw}"
        self.base_url = str(raw).rstrip("/")
        self.timeout = float(timeout)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(detail).get("error", detail)
            except Exception:
                msg = detail
            raise SimClientError(f"{method} {path} returned HTTP {exc.code}: {msg}") from exc
        except OSError as exc:
            raise SimClientError(f"Cannot reach armd at {self.base_url}{path}: {exc}") from exc

    def catalog(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/sandbox/catalog")

    def upload_scene(self, scene_dir: str | Path, *, pack_id: str | None = None) -> dict[str, Any]:
        from .spec import validate_scene_dir

        root = Path(scene_dir).expanduser().resolve()
        validate_scene_dir(root)
        payload: dict[str, Any] = {"files": _collect_pack_files(root)}
        if pack_id:
            payload["id"] = pack_id
        return self._request("POST", "/api/v1/sandbox/scenes", payload)

    def upload_tool(self, tool_dir: str | Path, *, pack_id: str | None = None) -> dict[str, Any]:
        from .spec import validate_tool_dir

        root = Path(tool_dir).expanduser().resolve()
        validate_tool_dir(root)
        payload: dict[str, Any] = {"files": _collect_pack_files(root)}
        if pack_id:
            payload["id"] = pack_id
        return self._request("POST", "/api/v1/sandbox/tools", payload)

    def upload_policy(self, policy_dir: str | Path, *, pack_id: str | None = None) -> dict[str, Any]:
        from ..onnx_policy import validate_policy_dir

        root = Path(policy_dir).expanduser().resolve()
        manifest = validate_policy_dir(root)
        payload: dict[str, Any] = {"id": pack_id or manifest.name, "files": _collect_pack_files(root)}
        return self._request("POST", "/api/v1/sandbox/policies", payload)

    def activate(
        self,
        mode: str,
        *,
        scene: str | None = None,
        tool: str | None = None,
        policy: str | None = None,
        seed: int | None = None,
        auto_start_policy: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode, "auto_start_policy": bool(auto_start_policy)}
        if scene is not None:
            payload["scene"] = scene
        if tool is not None:
            payload["tool"] = tool
        if policy is not None:
            payload["policy"] = policy
        if seed is not None:
            payload["seed"] = seed
        return self._request("POST", "/api/v1/sandbox/activate", payload)

    def bundle(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/sandbox/bundle")

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/sandbox/state")

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {} if seed is None else {"seed": seed}
        return self._request("POST", "/api/v1/sandbox/reset", payload)

    def set_io(self, updates: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if updates:
            merged.update(updates)
        merged.update(kwargs)
        return self._request("POST", "/api/v1/sandbox/io", {"updates": merged})
