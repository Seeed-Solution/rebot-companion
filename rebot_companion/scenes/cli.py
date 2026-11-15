"""CLI entry point `rebot-scene` for discovering, validating, scaffolding, and loading scenes/tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .assets import DEFAULT_OSS_CDN_BASE, SceneAssetManager, prepare_asset_metadata
from .client import SimClientError, SimSceneClient
from .spec import (
    SceneManifest,
    ToolManifest,
    list_builtin_packs,
    resolve_builtin_path,
    scaffold_pack,
    validate_scene_dir,
    validate_tool_dir,
)


def _coerce_io_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "on", "yes"}:
        return True
    if low in {"false", "off", "no"}:
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _load_pack(target: str, kind: str = "auto") -> tuple[str, SceneManifest | ToolManifest | dict[str, Any]]:
    p = Path(target).expanduser()
    if p.exists():
        if kind == "scene" or (kind == "auto" and (p / "scene.json").is_file() or p.name == "scene.json"):
            return "scene", validate_scene_dir(p)
        if kind == "tool" or (kind == "auto" and (p / "tool.json").is_file() or p.name == "tool.json"):
            return "tool", validate_tool_dir(p)
        raise FileNotFoundError(f"Path does not contain scene.json or tool.json: {p}")

    if kind in {"auto", "scene"}:
        builtin_scene = resolve_builtin_path("scene", target)
        if builtin_scene is not None:
            return "scene", validate_scene_dir(builtin_scene)
    if kind in {"auto", "tool"}:
        builtin_tool = resolve_builtin_path("tool", target)
        if builtin_tool is not None:
            return "tool", validate_tool_dir(builtin_tool)

    packs = list_builtin_packs()
    if kind in {"auto", "scene"}:
        for item in packs["scenes"]:
            if item.get("id") == target:
                return "scene", item
    if kind in {"auto", "tool"}:
        for item in packs["tools"]:
            if item.get("id") == target:
                return "tool", item
    raise FileNotFoundError(f"Unknown scene or tool pack: {target!r}")


def _cmd_list(args: argparse.Namespace) -> int:
    packs = list_builtin_packs()
    if args.json:
        print(json.dumps(packs, indent=2, ensure_ascii=False))
        return 0
    print("Scenes:")
    for s in packs["scenes"]:
        print(f"  - {s['id']:<20} {s.get('name', '')} :: {s.get('description', '')}")
    print("\nTools / End-effectors:")
    for t in packs["tools"]:
        kind = t.get("kind", "custom")
        print(f"  - {t['id']:<20} [{kind}] {t.get('name', '')} :: {t.get('description', '')}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    kind, pack = _load_pack(args.target, kind=args.kind)
    data = pack.to_dict() if hasattr(pack, "to_dict") else dict(pack)
    mgr = SceneAssetManager(cache_dir=args.cache_dir)
    if isinstance(pack, (SceneManifest, ToolManifest)):
        data["asset_status"] = mgr.status(pack)
        if pack.source_dir is not None:
            data["source_dir"] = str(pack.source_dir)
    if args.json:
        print(json.dumps({"type": kind, **data}, indent=2, ensure_ascii=False))
        return 0
    print(f"[{kind}] {data.get('id')} — {data.get('name', '')}")
    if data.get("description"):
        print(f"  {data['description']}")
    if "source_dir" in data:
        print(f"  dir: {data['source_dir']}")
    if kind == "scene":
        print(f"  robots: {', '.join(data.get('robots', []))} | default_tool: {data.get('default_tool', 'default')}")
        stages = [st.get("id", "?") for st in data.get("task_stages", [])]
        if stages:
            print(f"  task_stages: {' -> '.join(stages)}")
        if data.get("poses"):
            print(f"  suggested_poses: {', '.join(data['poses'].keys())}")
    else:
        print(f"  kind: {data.get('kind')} | robots: {', '.join(data.get('robots', []))}")
        if data.get("driven_joint"):
            print(f"  driven_joint: {data['driven_joint']} (scale={data.get('motor_to_joint_scale')})")
    if data.get("asset_status"):
        print("  assets:")
        for st in data["asset_status"]:
            flag = "cached" if st["cached"] else "tier0-only/not-cached"
            print(f"    - {st['id']} ({st['filename']}) [{flag}]")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    if args.kind == "policy":
        from pathlib import Path

        from ..onnx_policy import scaffold_onnx_policy

        raw_dir = Path(args.target_dir).expanduser().resolve()
        pid = args.id or raw_dir.name or "custom_policy"
        dest = scaffold_onnx_policy(raw_dir, policy_id=pid, overwrite=args.force)
        print(f"Created custom policy pack in {dest}")
        print(f"Upload to armd sandbox with: rebot-scene upload {dest}")
        return 0
    dest = scaffold_pack(args.target_dir, kind=args.kind, pack_id=args.id, overwrite=args.force)
    print(f"Created custom {args.kind} pack in {dest}")
    flag = f"--{args.kind}"
    print(f"Load into rebot-sim with: rebot-scene load {flag} {dest}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from pathlib import Path

    p = Path(args.path).expanduser().resolve()
    if args.kind == "policy" or (
        args.kind == "auto" and ((p.is_dir() and (p / "policy.json").is_file()) or p.name == "policy.json")
    ):
        from ..onnx_policy import validate_policy_dir

        manifest = validate_policy_dir(p.parent if p.is_file() else p)
        print(f"OK [policy] {manifest.name}")
        return 0
    kind, pack = _load_pack(args.path, kind=args.kind)
    print(f"OK [{kind}] {pack.id if hasattr(pack, 'id') else pack['id']}")
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .client import ArmdSandboxClient

    p = Path(args.path).expanduser().resolve()
    kind = args.kind
    if kind == "auto":
        if (p / "policy.json").is_file():
            kind = "policy"
        elif (p / "tool.json").is_file():
            kind = "tool"
        elif (p / "scene.json").is_file():
            kind = "scene"
        else:
            raise ValueError(f"Could not detect scene.json, tool.json, or policy.json in {p}")
    client = ArmdSandboxClient(base_url=args.armd_url)
    if kind == "policy":
        res = client.upload_policy(p, pack_id=args.id)
    elif kind == "tool":
        res = client.upload_tool(p, pack_id=args.id)
    else:
        res = client.upload_scene(p, pack_id=args.id)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_activate(args: argparse.Namespace) -> int:
    from .client import ArmdSandboxClient

    client = ArmdSandboxClient(base_url=args.armd_url)
    res = client.activate(
        args.mode,
        scene=args.scene,
        tool=args.tool,
        policy=args.policy,
        seed=args.seed,
        auto_start_policy=not args.no_auto_start,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .client import ArmdSandboxClient

    client = ArmdSandboxClient(base_url=args.armd_url)
    res = client.catalog() if args.catalog else client.state()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    client = SimSceneClient(base_url=args.http_url)
    res = client.load_scene(scene=args.scene, tool=args.tool, seed=args.seed)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_io(args: argparse.Namespace) -> int:
    client = SimSceneClient(base_url=args.http_url)
    if args.set:
        updates: dict[str, Any] = {}
        for item in args.set:
            if "=" not in item:
                raise ValueError(f"Expected KEY=VALUE for --set, got {item!r}")
            k, v = item.split("=", 1)
            updates[k.strip()] = _coerce_io_value(v)
        res = client.set_io(updates)
    else:
        res = client.get_io()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    _, pack = _load_pack(args.target, kind=args.kind)
    if not isinstance(pack, (SceneManifest, ToolManifest)):
        print("No downloadable assets for virtual preset.")
        return 0
    mgr = SceneAssetManager(cache_dir=args.cache_dir)
    if not pack.assets:
        print(f"{pack.id}: uses Tier-0 native MuJoCo primitives (0 remote assets declared).")
        return 0
    resolved = mgr.ensure_scene_assets(pack, allow_download=True, include_optional=not args.required_only)
    print(json.dumps({"id": pack.id, "cached": {k: str(v) for k, v in resolved.items()}}, indent=2))
    return 0


def _cmd_asset_hash(args: argparse.Namespace) -> int:
    meta = prepare_asset_metadata(args.file, remote_prefix=args.prefix, cdn_base=args.cdn)
    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0
    print(f"filename: {meta['idempotent_filename']}")
    print(f"sha256:   {meta['sha256']}")
    print(f"size:     {meta['size_bytes']} bytes")
    print(f"oss_url:  {meta['oss_url']}")
    print(f"web_url:  {meta['web_cdn_url']}")
    return 0


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    from .bundle_export import export_static_bundle

    base_xml_text = None
    if args.base_xml:
        base_xml_text = Path(args.base_xml).read_text(encoding="utf-8")

    out = export_static_bundle(
        out_dir=args.out,
        scene_id=args.scene,
        tool_id=args.tool,
        policy_dir=args.policy,
        robot=args.robot,
        wasm_path=args.wasm,
        base_xml=base_xml_text,
        overwrite=True,
    )
    print(f"Exported static bundle to: {out}")
    return 0


def _cmd_apply_profile(args: argparse.Namespace) -> int:
    from ..deploy_profile import apply_deploy_profile

    result = apply_deploy_profile(
        args.profile,
        host=args.host,
        port=args.port,
        apply_changes=args.yes,
        backup_path=args.backup,
        out_script=args.out_script,
        clear_keepout=args.clear_keepout,
        apply_collision_guard=args.apply_collision_guard,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_playground_serve(args: argparse.Namespace) -> int:
    import asyncio

    from .playground_proxy import PlaygroundProxyServer

    server = PlaygroundProxyServer(
        static_dir=args.static_dir,
        armd_url=args.armd_url,
        host=args.host,
        port=args.port,
        enable_coop=args.coop,
    )

    async def run() -> None:
        await server.start()
        print(f"reBot Cyclops serving at {server.url} (armd: {args.armd_url})")
        stop = asyncio.Event()
        try:
            await stop.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await server.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rebot-scene", description="Manage rebot-sim scenes, tools, and assets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List built-in scenes and tools")
    p_list.add_argument("--json", action="store_true", help="Output JSON")
    p_list.set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="Show manifest details for a built-in or local pack")
    p_info.add_argument("target", help="Pack ID or directory path")
    p_info.add_argument("--kind", choices=("auto", "scene", "tool"), default="auto")
    p_info.add_argument("--cache-dir", default=None)
    p_info.add_argument("--json", action="store_true", help="Output JSON")
    p_info.set_defaults(func=_cmd_info)

    p_init = sub.add_parser("init", help="Scaffold a starter custom scene, tool, or ONNX policy directory")
    p_init.add_argument("target_dir", help="Destination directory")
    p_init.add_argument("--kind", choices=("scene", "tool", "policy"), default="scene")
    p_init.add_argument("--id", default=None, help="Pack identifier")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing starter files")
    p_init.set_defaults(func=_cmd_init)

    p_val = sub.add_parser("validate", help="Validate a custom scene, tool, or ONNX policy directory")
    p_val.add_argument("path", help="Path to scene, tool, or policy directory")
    p_val.add_argument("--kind", choices=("auto", "scene", "tool", "policy"), default="auto")
    p_val.set_defaults(func=_cmd_validate)

    p_up = sub.add_parser("upload", help="Upload a validated scene, tool, or ONNX policy pack to armd sandbox")
    p_up.add_argument("path", help="Path to local scene, tool, or policy directory")
    p_up.add_argument("--kind", choices=("auto", "scene", "tool", "policy"), default="auto")
    p_up.add_argument("--id", default=None, help="Override uploaded pack identifier")
    p_up.add_argument("--armd-url", default=None, help="armd HTTP URL (default http://127.0.0.1:8080)")
    p_up.set_defaults(func=_cmd_upload)

    p_act = sub.add_parser("activate", help="Switch active sandbox mode/scene/tool/policy on armd")
    p_act.add_argument("--mode", choices=("none", "wasm", "sim"), required=True, help="Target sandbox runtime mode")
    p_act.add_argument("--scene", default=None, help="Scene ID (built-in or uploaded)")
    p_act.add_argument("--tool", default=None, help="Tool ID (built-in or uploaded)")
    p_act.add_argument("--policy", default=None, help="Uploaded ONNX policy ID (or omit)")
    p_act.add_argument("--seed", type=int, default=None, help="Optional layout randomization seed")
    p_act.add_argument("--no-auto-start", action="store_true", help="Load policy without auto-starting inference")
    p_act.add_argument("--armd-url", default=None, help="armd HTTP URL (default http://127.0.0.1:8080)")
    p_act.set_defaults(func=_cmd_activate)

    p_stat = sub.add_parser("status", help="Inspect armd sandbox state or catalog")
    p_stat.add_argument("--catalog", action="store_true", help="Print full sandbox catalog instead of live state")
    p_stat.add_argument("--armd-url", default=None, help="armd HTTP URL (default http://127.0.0.1:8080)")
    p_stat.set_defaults(func=_cmd_status)

    p_load = sub.add_parser("load", help="Hot-swap scene and/or tool on running rebot-sim")
    p_load.add_argument("--scene", default=None, help="Scene ID or local directory")
    p_load.add_argument("--tool", default=None, help="Tool ID, preset ('default'/'suction'), or local directory")
    p_load.add_argument("--seed", type=int, default=None, help="Optional layout randomization seed")
    p_load.add_argument("--http-url", default=None, help="rebot-sim HTTP URL (default http://127.0.0.1:8090)")
    p_load.set_defaults(func=_cmd_load)

    p_io = sub.add_parser("io", help="Read or write simulator virtual I/O channels")
    p_io.add_argument("--set", nargs="+", default=None, help="KEY=VALUE pairs, e.g. vacuum=true")
    p_io.add_argument("--http-url", default=None, help="rebot-sim HTTP URL (default http://127.0.0.1:8090)")
    p_io.set_defaults(func=_cmd_io)

    p_pull = sub.add_parser("pull", help="Download/verify cached Tier-1 assets for a scene or tool")
    p_pull.add_argument("target", help="Pack ID or directory path")
    p_pull.add_argument("--kind", choices=("auto", "scene", "tool"), default="auto")
    p_pull.add_argument("--cache-dir", default=None)
    p_pull.add_argument("--required-only", action="store_true")
    p_pull.set_defaults(func=_cmd_pull)

    p_hash = sub.add_parser("asset-hash", help="Compute MD5-8 CDN name and SHA-256 checksum for an asset file")
    p_hash.add_argument("file", help="Path to mesh/texture/asset file")
    p_hash.add_argument("--prefix", default="rebot/scenes", help="OSS key prefix under solution-app")
    p_hash.add_argument("--cdn", default=DEFAULT_OSS_CDN_BASE, help="CDN base URL")
    p_hash.add_argument("--json", action="store_true", help="Output JSON")
    p_hash.set_defaults(func=_cmd_asset_hash)

    p_exp = sub.add_parser("export-bundle", help="Export static self-contained reBot Cyclops bundle")
    p_exp.add_argument("--out", required=True, help="Destination directory for bundle.json and model.onnx")
    p_exp.add_argument("--scene", default=None, help="Scene ID or path (default None)")
    p_exp.add_argument("--tool", default=None, help="Tool ID or path (default scene's tool or default)")
    p_exp.add_argument("--policy", default=None, help="Path to local policy directory")
    p_exp.add_argument("--robot", choices=("b601dm", "so-arm101"), default="b601dm", help="Target robot")
    p_exp.add_argument("--wasm", default=None, help="Path to armd.wasm (default built artifact)")
    p_exp.add_argument("--base-xml", default=None, help="Path to custom base robot XML")
    p_exp.set_defaults(func=_cmd_export_bundle)

    p_prof = sub.add_parser("apply-profile", help="Validate and apply deploy profile poses and keepout to armd")
    p_prof.add_argument("profile", help="Path to profile JSON")
    p_prof.add_argument("--host", default="127.0.0.1", help="armd host")
    p_prof.add_argument("--port", type=int, default=8765, help="armd port")
    p_prof.add_argument("--yes", action="store_true", help="Mutate device state (otherwise dry-run)")
    p_prof.add_argument("--backup", default=None, help="Path to backup existing device configuration")
    p_prof.add_argument("--out-script", default=None, help="Path to output generated Python 3.8 main.py")
    p_prof.add_argument(
        "--clear-keepout",
        action="store_true",
        help="Clear existing device keepout boxes when profile specifies none",
    )
    p_prof.add_argument(
        "--apply-collision-guard",
        action="store_true",
        help="Apply collision guard configuration from profile to armd",
    )
    p_prof.set_defaults(func=_cmd_apply_profile)

    p_serve = sub.add_parser("playground-serve", help="Run static asset and same-origin armd reverse proxy for reBot Cyclops")
    p_serve.add_argument("--static-dir", required=True, help="Static build directory")
    p_serve.add_argument("--armd-url", default="http://127.0.0.1:8765", help="Upstream armd URL")
    p_serve.add_argument("--host", default="127.0.0.1", help="Proxy listening host")
    p_serve.add_argument("--port", type=int, default=4173, help="Proxy listening port")
    p_serve.add_argument("--coop", action="store_true", help="Enable COOP/COEP isolation headers")
    p_serve.set_defaults(func=_cmd_playground_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, SimClientError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
