"""On-demand scene asset caching following solution-assets naming & verification rules."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_OSS_CDN_BASE = "https://sensecraft-statics.seeed.cc/solution-app"
DEFAULT_WEB_CDN_BASE = "https://files.seeedstudio.com/Solution/landpage_asset"

logger = logging.getLogger(__name__)


class AssetError(RuntimeError):
    """Base error for scene asset operations."""


class AssetChecksumError(AssetError):
    """Raised when a downloaded or cached file fails SHA-256 verification."""


class AssetDownloadError(AssetError):
    """Raised when downloading a required asset fails."""


def default_cache_dir() -> Path:
    """Return local cache directory for on-demand scene assets."""
    env = os.environ.get("REBOT_SCENE_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "rebot" / "scenes"


def idempotent_asset_name(file_path_or_name: str | Path, data: bytes | None = None) -> str:
    """Return `<basename>-<md5-first-8>.<ext>` filename per solution-assets convention."""
    p = Path(file_path_or_name)
    raw = data if data is not None else p.read_bytes()
    digest8 = hashlib.md5(raw).hexdigest()[:8]
    stem = p.stem
    suffix = p.suffix
    return f"{stem}-{digest8}{suffix}"


def compute_sha256(path: str | Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def prepare_asset_metadata(
    file_path: str | Path,
    remote_prefix: str = "rebot/scenes",
    cdn_base: str = DEFAULT_OSS_CDN_BASE,
) -> dict[str, Any]:
    """Compute MD5-8 CDN name, SHA-256 checksum, size, and target URLs for publishing."""
    p = Path(file_path)
    raw = p.read_bytes()
    md5_8 = hashlib.md5(raw).hexdigest()[:8]
    sha256 = hashlib.sha256(raw).hexdigest()
    hashed_name = f"{p.stem}-{md5_8}{p.suffix}"
    clean_prefix = remote_prefix.strip("/")
    rel = f"{clean_prefix}/{hashed_name}" if clean_prefix else hashed_name
    return {
        "source_name": p.name,
        "idempotent_filename": hashed_name,
        "md5_8": md5_8,
        "sha256": sha256,
        "size_bytes": len(raw),
        "oss_url": f"{cdn_base.rstrip('/')}/{rel}",
        "web_cdn_url": f"{DEFAULT_WEB_CDN_BASE.rstrip('/')}/{rel}",
    }


def _safe_relpath(base_dir: Path, rel: str) -> Path:
    candidate = (base_dir / rel).resolve()
    base_resolved = base_dir.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise ValueError(f"Asset target escapes scene cache directory: {rel!r}")
    return candidate


def _get_attr(asset: Any, key: str, default: Any = None) -> Any:
    if isinstance(asset, dict):
        return asset.get(key, default)
    return getattr(asset, key, default)


class SceneAssetManager:
    """Lazy downloader and verifier for Tier-1 optional scene assets."""

    def __init__(self, cache_dir: str | Path | None = None, timeout_s: float = 30.0):
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
        self.timeout_s = float(timeout_s)

    def scene_dir(self, scene_id: str) -> Path:
        return _safe_relpath(self.cache_dir, scene_id)

    def asset_path(self, scene_id: str, filename: str, target: str | None = None) -> Path:
        rel = target or filename
        if not rel:
            raise ValueError("Asset must define filename or target")
        return _safe_relpath(self.scene_dir(scene_id), rel)

    def verify_file(self, path: Path, expected_sha256: str = "") -> bool:
        """Verify file exists and matches optional SHA-256 hex digest."""
        if not path.is_file():
            return False
        if not expected_sha256:
            return True
        return compute_sha256(path).lower() == expected_sha256.strip().lower()

    def resolve_cached(self, scene_id: str, asset: Any) -> Path | None:
        filename = _get_attr(asset, "filename", "")
        target = _get_attr(asset, "target", "") or None
        sha256 = _get_attr(asset, "sha256", "") or ""
        path = self.asset_path(scene_id, filename, target)
        if not path.is_file():
            return None
        if sha256 and not self.verify_file(path, sha256):
            return None
        return path

    def ensure_asset(
        self,
        scene_id: str,
        asset: Any,
        *,
        allow_download: bool = True,
        progress_cb: Callable[[int, int | None], None] | None = None,
    ) -> Path | None:
        """Return verified cached path, downloading from configured URLs if needed."""
        filename = _get_attr(asset, "filename", "") or _get_attr(asset, "id", "")
        target = _get_attr(asset, "target", "") or None
        sha256 = (_get_attr(asset, "sha256", "") or "").strip().lower()
        required = bool(_get_attr(asset, "required", False))
        urls = [u for u in [_get_attr(asset, "url", ""), *(_get_attr(asset, "fallback_urls", []) or [])] if u]

        dest = self.asset_path(scene_id, filename, target)
        if self.verify_file(dest, sha256):
            return dest

        if not allow_download:
            if required:
                raise AssetDownloadError(f"Missing required cached asset {filename!r} at {dest}")
            return None

        if not urls:
            if required:
                raise AssetDownloadError(f"No download URL configured for required asset {filename!r}")
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(f"{dest.name}.part")
        last_err: Exception | None = None

        for url in urls:
            scheme = urllib.parse.urlparse(url).scheme.lower()
            if scheme not in {"http", "https", "file"}:
                last_err = AssetDownloadError(f"Unsupported URL scheme {scheme!r} for {url!r}")
                continue
            try:
                digest = hashlib.sha256()
                total = 0
                with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                    cl_header = resp.headers.get("Content-Length") if resp.headers else None
                    total_bytes = int(cl_header) if cl_header and cl_header.isdigit() else None
                    with part.open("wb") as out_f:
                        while chunk := resp.read(65536):
                            out_f.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
                            if progress_cb is not None:
                                progress_cb(total, total_bytes)
                got_sha = digest.hexdigest().lower()
                if sha256 and got_sha != sha256:
                    part.unlink(missing_ok=True)
                    raise AssetChecksumError(
                        f"SHA-256 mismatch for {filename!r} from {url}: expected {sha256}, got {got_sha}"
                    )
                os.replace(part, dest)
                return dest
            except AssetChecksumError:
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                part.unlink(missing_ok=True)
                last_err = exc
                logger.warning("Asset download attempt failed for %s (%s): %s", filename, url, exc)

        if required:
            raise AssetDownloadError(f"Failed to fetch required asset {filename!r}: {last_err}") from last_err
        return None

    def ensure_scene_assets(
        self,
        manifest: Any,
        *,
        allow_download: bool = True,
        include_optional: bool = True,
    ) -> dict[str, Path]:
        """Ensure assets declared in a SceneManifest or ToolManifest."""
        scene_id = _get_attr(manifest, "id", "")
        assets = _get_attr(manifest, "assets", []) or []
        resolved: dict[str, Path] = {}
        for asset in assets:
            required = bool(_get_attr(asset, "required", False))
            if not include_optional and not required:
                continue
            asset_id = _get_attr(asset, "id", "") or _get_attr(asset, "filename", "")
            path = self.ensure_asset(scene_id, asset, allow_download=allow_download)
            if path is not None:
                resolved[asset_id] = path
        return resolved

    def status(self, manifest: Any) -> list[dict[str, Any]]:
        """Report local cache status for every asset in a manifest."""
        scene_id = _get_attr(manifest, "id", "")
        assets = _get_attr(manifest, "assets", []) or []
        rows: list[dict[str, Any]] = []
        for asset in assets:
            filename = _get_attr(asset, "filename", "")
            target = _get_attr(asset, "target", "") or None
            sha256 = (_get_attr(asset, "sha256", "") or "").strip().lower()
            path = self.asset_path(scene_id, filename, target)
            exists = path.is_file()
            verified = self.verify_file(path, sha256) if exists else False
            rows.append(
                {
                    "id": _get_attr(asset, "id", filename),
                    "filename": filename,
                    "tier": int(_get_attr(asset, "tier", 1)),
                    "required": bool(_get_attr(asset, "required", False)),
                    "cached": verified,
                    "present": exists,
                    "checksum_ok": verified if sha256 else exists,
                    "path": str(path),
                    "url": _get_attr(asset, "url", ""),
                }
            )
        return rows
