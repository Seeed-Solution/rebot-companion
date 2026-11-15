"""Scene and custom end-effector manifest schema, discovery, validation, and scaffolding."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
VALID_TOOL_KINDS = {"gripper", "suction", "screwdriver", "custom"}


def builtin_root() -> Path:
    """Return directory containing built-in scene and tool packs."""
    return Path(__file__).resolve().parent / "builtin"


def _resolve_within(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    root_resolved = root.resolve()
    if p != root_resolved and root_resolved not in p.parents:
        raise ValueError(f"Path {rel!r} escapes pack directory {root}")
    return p


@dataclass
class AssetRef:
    """Remote/cached asset descriptor aligned with solution-assets."""

    id: str
    filename: str
    url: str = ""
    fallback_urls: list[str] = field(default_factory=list)
    sha256: str = ""
    size_bytes: int | None = None
    tier: int = 1
    required: bool = False
    target: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AssetRef:
        filename = str(raw.get("filename") or raw.get("target") or "").strip()
        asset_id = str(raw.get("id") or filename).strip()
        if not asset_id or not filename:
            raise ValueError("Asset entry requires non-empty 'id' and 'filename'")
        return cls(
            id=asset_id,
            filename=filename,
            url=str(raw.get("url") or "").strip(),
            fallback_urls=[str(u).strip() for u in raw.get("fallback_urls", []) if str(u).strip()],
            sha256=str(raw.get("sha256") or "").strip().lower(),
            size_bytes=int(raw["size_bytes"]) if raw.get("size_bytes") is not None else None,
            tier=int(raw.get("tier", 1)),
            required=bool(raw.get("required", False)),
            target=str(raw.get("target") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "url": self.url,
            "fallback_urls": list(self.fallback_urls),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "tier": self.tier,
            "required": self.required,
            "target": self.target,
        }


@dataclass
class SceneManifest:
    """Declarative specification for a simulation scene pack."""

    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    world_xml: str = "world.xml"
    robots: list[str] = field(default_factory=lambda: ["b601dm"])
    default_tool: str = "default"
    recommended_tools: list[str] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    zones: dict[str, Any] = field(default_factory=dict)
    io: list[dict[str, Any]] = field(default_factory=list)
    task_stages: list[dict[str, Any]] = field(default_factory=list)
    poses: dict[str, Any] = field(default_factory=dict)
    keepout_boxes: list[dict[str, Any]] = field(default_factory=list)
    training: dict[str, Any] = field(default_factory=dict)
    camera_extrinsics: dict[str, Any] = field(default_factory=dict)
    assets: list[AssetRef] = field(default_factory=list)
    source_dir: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_dir: Path | None = None) -> SceneManifest:
        pack_id = str(raw.get("id") or "").strip()
        if not VALID_ID_RE.match(pack_id):
            raise ValueError(f"Invalid scene id {pack_id!r}; expected match for {VALID_ID_RE.pattern}")
        name = str(raw.get("name") or pack_id).strip()
        assets = [AssetRef.from_dict(item) for item in raw.get("assets", [])]
        return cls(
            id=pack_id,
            name=name,
            version=str(raw.get("version") or "1.0.0"),
            description=str(raw.get("description") or ""),
            world_xml=str(raw.get("world_xml") or "world.xml"),
            robots=[str(r) for r in raw.get("robots", ["b601dm"])],
            default_tool=str(raw.get("default_tool") or "default"),
            recommended_tools=[str(t) for t in raw.get("recommended_tools", [])],
            objects=[dict(o) for o in raw.get("objects", [])],
            zones=dict(raw.get("zones", {})),
            io=[dict(ch) for ch in raw.get("io", [])],
            task_stages=[dict(st) for st in raw.get("task_stages", [])],
            poses=dict(raw.get("poses", {})),
            keepout_boxes=[dict(b) for b in raw.get("keepout_boxes", [])],
            training=dict(raw.get("training", {})),
            camera_extrinsics=dict(raw.get("camera_extrinsics", {})),
            assets=assets,
            source_dir=source_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "world_xml": self.world_xml,
            "robots": list(self.robots),
            "default_tool": self.default_tool,
            "recommended_tools": list(self.recommended_tools),
            "objects": list(self.objects),
            "zones": dict(self.zones),
            "io": list(self.io),
            "task_stages": list(self.task_stages),
            "poses": dict(self.poses),
            "keepout_boxes": list(self.keepout_boxes),
            "training": dict(self.training),
            "assets": [a.to_dict() for a in self.assets],
        }
        if self.camera_extrinsics:
            data["camera_extrinsics"] = dict(self.camera_extrinsics)
        return data


@dataclass
class ToolManifest:
    """Declarative specification for a custom gripper or end-effector pack."""

    id: str
    name: str
    kind: str = "gripper"
    version: str = "1.0.0"
    description: str = ""
    tool_xml: str = "tool.xml"
    robots: list[str] = field(default_factory=lambda: ["b601dm"])
    attach_body: str = "end_link"
    replace_bodies: list[str] = field(default_factory=lambda: ["finger_left", "finger_right"])
    replace_geoms: list[str] = field(default_factory=lambda: ["palm", "motor_housing"])
    replace_equalities: list[str] = field(default_factory=lambda: ["jaw_coupling"])
    replace_actuators: list[str] = field(default_factory=lambda: ["m7"])
    driven_joint: str | None = None
    coupled_joint: str | None = None
    actuator: str | None = None
    motor_to_joint_scale: float = 1.0
    finger_geoms: list[str] = field(default_factory=list)
    tcp_site: str = "tcp"
    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    io: list[dict[str, Any]] = field(default_factory=list)
    assets: list[AssetRef] = field(default_factory=list)
    source_dir: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_dir: Path | None = None) -> ToolManifest:
        pack_id = str(raw.get("id") or "").strip()
        if not VALID_ID_RE.match(pack_id):
            raise ValueError(f"Invalid tool id {pack_id!r}; expected match for {VALID_ID_RE.pattern}")
        kind = str(raw.get("kind") or "gripper").strip().lower()
        if kind not in VALID_TOOL_KINDS:
            raise ValueError(f"Invalid tool kind {kind!r}; expected one of {sorted(VALID_TOOL_KINDS)}")
        tcp_offset = [float(v) for v in raw.get("tcp_offset", [0.0, 0.0, 0.0])]
        if len(tcp_offset) != 3:
            raise ValueError("tcp_offset must have exactly 3 numeric elements")
        assets = [AssetRef.from_dict(item) for item in raw.get("assets", [])]
        return cls(
            id=pack_id,
            name=str(raw.get("name") or pack_id).strip(),
            kind=kind,
            version=str(raw.get("version") or "1.0.0"),
            description=str(raw.get("description") or ""),
            tool_xml=str(raw.get("tool_xml") or "tool.xml"),
            robots=[str(r) for r in raw.get("robots", ["b601dm"])],
            attach_body=str(raw.get("attach_body") or "end_link"),
            replace_bodies=[str(b) for b in raw.get("replace_bodies", ["finger_left", "finger_right"])],
            replace_geoms=[str(g) for g in raw.get("replace_geoms", ["palm", "motor_housing"])],
            replace_equalities=[str(e) for e in raw.get("replace_equalities", ["jaw_coupling"])],
            replace_actuators=[str(a) for a in raw.get("replace_actuators", ["m7"])],
            driven_joint=raw.get("driven_joint"),
            coupled_joint=raw.get("coupled_joint"),
            actuator=raw.get("actuator"),
            motor_to_joint_scale=float(raw.get("motor_to_joint_scale", 1.0)),
            finger_geoms=[str(g) for g in raw.get("finger_geoms", [])],
            tcp_site=str(raw.get("tcp_site") or "tcp"),
            tcp_offset=tcp_offset,
            io=[dict(ch) for ch in raw.get("io", [])],
            assets=assets,
            source_dir=source_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "description": self.description,
            "tool_xml": self.tool_xml,
            "robots": list(self.robots),
            "attach_body": self.attach_body,
            "replace_bodies": list(self.replace_bodies),
            "replace_geoms": list(self.replace_geoms),
            "replace_equalities": list(self.replace_equalities),
            "replace_actuators": list(self.replace_actuators),
            "driven_joint": self.driven_joint,
            "coupled_joint": self.coupled_joint,
            "actuator": self.actuator,
            "motor_to_joint_scale": self.motor_to_joint_scale,
            "finger_geoms": list(self.finger_geoms),
            "tcp_site": self.tcp_site,
            "tcp_offset": list(self.tcp_offset),
            "io": list(self.io),
            "assets": [a.to_dict() for a in self.assets],
        }


def validate_scene_dir(path: str | Path) -> SceneManifest:
    """Validate and load a scene directory (or direct `scene.json` file)."""
    p = Path(path).expanduser().resolve()
    scene_json = p if p.is_file() and p.name == "scene.json" else p / "scene.json"
    if not scene_json.is_file():
        raise FileNotFoundError(f"Missing scene.json in {p}")
    scene_dir = scene_json.parent
    raw = json.loads(scene_json.read_text(encoding="utf-8"))
    manifest = SceneManifest.from_dict(raw, source_dir=scene_dir)
    world_path = _resolve_within(scene_dir, manifest.world_xml)
    if not world_path.is_file():
        raise FileNotFoundError(f"Scene world XML not found: {world_path}")
    ET.parse(world_path)
    return manifest


def validate_tool_dir(path: str | Path) -> ToolManifest:
    """Validate and load a custom tool/gripper directory (or direct `tool.json` file)."""
    p = Path(path).expanduser().resolve()
    tool_json = p if p.is_file() and p.name == "tool.json" else p / "tool.json"
    if not tool_json.is_file():
        raise FileNotFoundError(f"Missing tool.json in {p}")
    tool_dir = tool_json.parent
    raw = json.loads(tool_json.read_text(encoding="utf-8"))
    manifest = ToolManifest.from_dict(raw, source_dir=tool_dir)
    if manifest.tool_xml:
        tool_path = _resolve_within(tool_dir, manifest.tool_xml)
        if not tool_path.is_file():
            raise FileNotFoundError(f"Tool XML not found: {tool_path}")
        ET.parse(tool_path)
    return manifest


def load_registry() -> dict[str, Any]:
    """Load the built-in pack registry metadata."""
    reg_path = builtin_root() / "registry.json"
    if not reg_path.is_file():
        return {"scenes": [], "tools": []}
    return json.loads(reg_path.read_text(encoding="utf-8"))


def resolve_builtin_path(kind: str, pack_id: str) -> Path | None:
    """Return built-in pack directory if one exists for `kind` ('scene'|'tool') and `pack_id`."""
    if not VALID_ID_RE.match(pack_id):
        return None
    sub = "scenes" if kind in {"scene", "scenes"} else "tools"
    candidate = builtin_root() / sub / pack_id
    return candidate if candidate.is_dir() else None


def list_builtin_packs() -> dict[str, list[dict[str, Any]]]:
    """Return summarized built-in scenes and tools from disk + registry presets."""
    reg = load_registry()
    scenes: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    scenes_dir = builtin_root() / "scenes"
    if scenes_dir.is_dir():
        for d in sorted(scenes_dir.iterdir()):
            if (d / "scene.json").is_file():
                m = validate_scene_dir(d)
                seen_scenes.add(m.id)
                scenes.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "version": m.version,
                        "description": m.description,
                        "robots": m.robots,
                        "default_tool": m.default_tool,
                        "path": str(d),
                        "builtin": True,
                    }
                )
    for item in reg.get("scenes", []):
        if item.get("id") not in seen_scenes:
            scenes.append({**item, "builtin": True})

    tools: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    tools_dir = builtin_root() / "tools"
    if tools_dir.is_dir():
        for d in sorted(tools_dir.iterdir()):
            if (d / "tool.json").is_file():
                t = validate_tool_dir(d)
                seen_tools.add(t.id)
                tools.append(
                    {
                        "id": t.id,
                        "name": t.name,
                        "kind": t.kind,
                        "version": t.version,
                        "description": t.description,
                        "robots": t.robots,
                        "path": str(d),
                        "builtin": True,
                    }
                )
    for item in reg.get("tools", []):
        if item.get("id") not in seen_tools:
            tools.append({**item, "builtin": True})

    return {"scenes": scenes, "tools": tools}


_STARTER_WORLD_XML = """<mujoco model="{pack_id}_world">
  <worldbody>
    <!-- Visual zone marker on table (z = 0.001) -->
    <geom name="target_mat" type="box" pos="0.28 -0.10 0.001" size="0.06 0.06 0.001"
          rgba="0.18 0.48 0.70 1" contype="0" conaffinity="0"/>

    <!-- Movable object detected by wrist segmentation & reported in /sim/state -->
    <body name="part_a" pos="0.25 0.08 0.02">
      <freejoint name="part_a_joint"/>
      <inertial pos="0 0 0" mass="0.04" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="part_a" type="box" size="0.018 0.018 0.020" rgba="0.88 0.36 0.22 1"
            friction="1.2 0.01 0.001" condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""

_STARTER_TOOL_XML = """<mujoco model="{pack_id}_tool">
  <worldbody>
    <geom name="custom_palm" type="box" pos="-0.085 0 0" size="0.014 0.058 0.022"
          rgba="0.24 0.26 0.30 1"/>
    <body name="custom_jaw_left" pos="0 0 0">
      <joint name="custom_jaw_left" type="slide" axis="0 1 0" range="-0.001 0.048" damping="5"/>
      <inertial pos="-0.03 0.01 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="custom_pad_left" type="box" pos="-0.016 0.007 0" size="0.016 0.007 0.018"
            rgba="0.18 0.50 0.65 1" friction="1.6 0.02 0.002" condim="4"/>
    </body>
    <body name="custom_jaw_right" pos="0 0 0">
      <joint name="custom_jaw_right" type="slide" axis="0 -1 0" range="-0.001 0.048" damping="5"/>
      <inertial pos="-0.03 -0.01 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="custom_pad_right" type="box" pos="-0.016 -0.007 0" size="0.016 0.007 0.018"
            rgba="0.18 0.50 0.65 1" friction="1.6 0.02 0.002" condim="4"/>
    </body>
  </worldbody>
  <equality>
    <joint name="custom_jaw_coupling" joint1="custom_jaw_left" joint2="custom_jaw_right"
           polycoef="0 1 0 0 0" solimp="0.99 0.999 0.0001" solref="0.002 1"/>
  </equality>
  <actuator>
    <position name="m7_custom" joint="custom_jaw_left" kp="100" dampratio="1" forcerange="-20 20"/>
  </actuator>
</mujoco>
"""


def scaffold_pack(
    target_dir: str | Path,
    *,
    kind: str = "scene",
    pack_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a starter custom scene or custom tool directory."""
    dest = Path(target_dir).expanduser().resolve()
    resolved_id = (pack_id or dest.name or f"custom_{kind}").strip()
    if not VALID_ID_RE.match(resolved_id):
        raise ValueError(f"Invalid pack id {resolved_id!r}")
    if kind not in {"scene", "tool"}:
        raise ValueError(f"kind must be 'scene' or 'tool', got {kind!r}")

    dest.mkdir(parents=True, exist_ok=True)
    if kind == "scene":
        manifest_path = dest / "scene.json"
        xml_path = dest / "world.xml"
        if not overwrite and (manifest_path.exists() or xml_path.exists()):
            raise FileExistsError(f"Scene files already exist in {dest}; pass overwrite=True/--force")
        manifest = {
            "id": resolved_id,
            "name": resolved_id.replace("_", " ").title(),
            "version": "1.0.0",
            "description": "Custom local MuJoCo scene for rebot-sim.",
            "world_xml": "world.xml",
            "robots": ["b601dm"],
            "default_tool": "default",
            "objects": [
                {
                    "name": "part_a",
                    "body": "part_a",
                    "geom": "part_a",
                    "class_name": "part_a",
                    "default_pos": [0.25, 0.08, 0.02],
                    "jitter_xy": 0.02,
                }
            ],
            "zones": {"target": {"center": [0.28, -0.10, 0.02], "half_extents": [0.06, 0.06, 0.04]}},
            "task_stages": [
                {
                    "id": "part_grasped",
                    "type": "object_grasped",
                    "object": "part_a",
                    "min_z": 0.04,
                    "latch": True,
                },
                {
                    "id": "part_placed",
                    "type": "object_in_zone",
                    "object": "part_a",
                    "zone": "target",
                    "requires": ["part_grasped"],
                    "latch": True,
                },
            ],
            "assets": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        xml_path.write_text(_STARTER_WORLD_XML.format(pack_id=resolved_id), encoding="utf-8")
        return validate_scene_dir(dest).source_dir or dest

    manifest_path = dest / "tool.json"
    xml_path = dest / "tool.xml"
    if not overwrite and (manifest_path.exists() or xml_path.exists()):
        raise FileExistsError(f"Tool files already exist in {dest}; pass overwrite=True/--force")
    manifest = {
        "id": resolved_id,
        "name": resolved_id.replace("_", " ").title(),
        "kind": "gripper",
        "version": "1.0.0",
        "description": "Custom local gripper attachment for B601-DM end_link.",
        "tool_xml": "tool.xml",
        "robots": ["b601dm"],
        "attach_body": "end_link",
        "replace_bodies": ["finger_left", "finger_right"],
        "replace_geoms": ["palm", "motor_housing"],
        "replace_equalities": ["jaw_coupling"],
        "replace_actuators": ["m7"],
        "driven_joint": "custom_jaw_left",
        "coupled_joint": "custom_jaw_right",
        "actuator": "m7_custom",
        "motor_to_joint_scale": -0.007456140350877194,
        "finger_geoms": ["custom_pad_left", "custom_pad_right"],
        "tcp_site": "tcp",
        "tcp_offset": [0.0, 0.0, 0.0],
        "io": [],
        "assets": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    xml_path.write_text(_STARTER_TOOL_XML.format(pack_id=resolved_id), encoding="utf-8")
    return validate_tool_dir(dest).source_dir or dest
