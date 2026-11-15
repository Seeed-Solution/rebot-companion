"""On-demand simulation scenes, custom end-effectors, and asset caching for rebot."""

from .assets import (
    DEFAULT_OSS_CDN_BASE,
    DEFAULT_WEB_CDN_BASE,
    AssetChecksumError,
    AssetDownloadError,
    AssetError,
    SceneAssetManager,
    compute_sha256,
    default_cache_dir,
    idempotent_asset_name,
    prepare_asset_metadata,
)
from .bundle_export import (
    compose_mjcf_via_wasm,
    export_static_bundle,
)
from .client import ArmdSandboxClient, SimClientError, SimSceneClient
from .playground_proxy import (
    PlaygroundProxyServer,
)
from .spec import (
    AssetRef,
    SceneManifest,
    ToolManifest,
    builtin_root,
    list_builtin_packs,
    load_registry,
    resolve_builtin_path,
    scaffold_pack,
    validate_scene_dir,
    validate_tool_dir,
)

__all__ = [
    "ArmdSandboxClient",
    "AssetChecksumError",
    "AssetDownloadError",
    "AssetError",
    "AssetRef",
    "DEFAULT_OSS_CDN_BASE",
    "DEFAULT_WEB_CDN_BASE",
    "PlaygroundProxyServer",
    "SceneAssetManager",
    "SceneManifest",
    "SimClientError",
    "SimSceneClient",
    "ToolManifest",
    "builtin_root",
    "compose_mjcf_via_wasm",
    "compute_sha256",
    "default_cache_dir",
    "export_static_bundle",
    "idempotent_asset_name",
    "list_builtin_packs",
    "load_registry",
    "prepare_asset_metadata",
    "resolve_builtin_path",
    "scaffold_pack",
    "validate_scene_dir",
    "validate_tool_dir",
]
