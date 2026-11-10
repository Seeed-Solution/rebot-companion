"""examples/templates/*.py against the fake armd (capabilities version 6)."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from rebot import Arm

TEMPLATES = Path(__file__).resolve().parents[1] / "examples" / "templates"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TEMPLATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def det(label: str, score: float = 0.9) -> dict:
    return {"format": "xywh_center", "class_name": label, "confidence": score, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}


def open_arm(armd) -> Arm:
    armd.devplat = True
    return Arm(armd.url, transport=armd.transport)


def teach(arm, *names):
    for n in names:
        arm.save_pose(n)


def kinds(armd) -> list:
    return [p.rsplit("/", 1)[-1] for p, _ in armd.action_bodies]


def test_demo_wave(armd):
    mod = load("demo_wave")
    with open_arm(armd) as arm:
        teach(arm, "wave_1", "wave_2", "wave_3")
        assert mod.run(arm, dict(mod.CONFIG, rounds=1)) == 1
    k = kinds(armd)
    assert k.count("move_to_pose") == 3 and k.count("gripper") == 6 and armd.state == "Idle"
    assert [b["name"] for p, b in armd.action_bodies if p.endswith("move_to_pose")] == ["wave_1", "wave_2", "wave_3"]


def test_demo_wave_missing_pose_does_not_move(armd):
    mod = load("demo_wave")
    with open_arm(armd) as arm:
        teach(arm, "wave_1")
        assert mod.run(arm, mod.CONFIG) == 0
    assert armd.action_bodies == []


def test_sort_by_label_sorts_into_bins(armd):
    mod = load("sort_by_label")
    armd.detections = [det("bottle", 0.8), det("cup", 0.9), det("person", 0.99)]
    with open_arm(armd) as arm:
        teach(arm, "scan", "bin_a", "bin_b")
        ok, bad = mod.run(arm, dict(mod.CONFIG, max_cycles=2))
    assert (ok, bad) == (2, 0)
    places = [b["pose"] for p, b in armd.action_bodies if p.endswith("/place")]
    assert places == ["bin_a", "bin_a"]  # best score first: cup -> bin_a; "person" has no bin
    assert all(b["speed_scale"] == 0.3 for p, b in armd.action_bodies if p.endswith("/place"))


def test_sort_by_label_stops_when_table_is_empty(armd):
    mod = load("sort_by_label")
    with open_arm(armd) as arm:
        teach(arm, "scan", "bin_a", "bin_b")
        assert mod.run(arm, dict(mod.CONFIG, max_cycles=10, max_empty_scans=2)) == (0, 0)
    assert kinds(armd).count("move_to_pose") == 2


def test_sort_by_label_counts_missed_grasps_and_continues(armd):
    mod = load("sort_by_label")
    armd.detections = [det("cup")]
    armd.gripper_contact = None  # gripper closes on nothing: NOT_HOLDING
    with open_arm(armd) as arm:
        teach(arm, "scan", "bin_a", "bin_b")
        ok, bad = mod.run(arm, dict(mod.CONFIG, max_cycles=3))
    assert (ok, bad) == (0, 3)
    assert "place" not in kinds(armd)


def test_sort_by_label_stops_on_unrecoverable_fault(armd):
    mod = load("sort_by_label")
    armd.detections = [det("cup")]
    with open_arm(armd) as arm:
        teach(arm, "scan", "bin_a", "bin_b")
        armd.trigger_fault("FollowingError")  # not recoverable
        time.sleep(0.3)  # let the observation show the fault
        ok, bad = mod.run(arm, dict(mod.CONFIG, max_cycles=5))
        assert (ok, bad) == (0, 1)
        assert armd.state == "Fault"  # never cleared automatically


def test_sort_by_label_recovers_a_recoverable_fault(armd):
    mod = load("sort_by_label")
    armd.detections = [det("cup")]
    with open_arm(armd) as arm:
        teach(arm, "scan", "bin_a", "bin_b")
        armd.trigger_fault("WatchdogTimeout")
        time.sleep(0.3)  # let the observation show the fault
        ok, bad = mod.run(arm, dict(mod.CONFIG, max_cycles=2))
    assert bad == 1 and ok == 1


def test_sort_by_label_missing_pose_does_not_move(armd):
    mod = load("sort_by_label")
    with open_arm(armd) as arm:
        teach(arm, "scan")
        assert mod.run(arm, mod.CONFIG) == (0, 0)
    assert armd.action_bodies == []


@pytest.mark.parametrize("name", ["demo_wave", "sort_by_label"])
def test_templates_are_short_and_stdlib_only(name):
    src = (TEMPLATES / f"{name}.py").read_text()
    assert len(src.splitlines()) < 120
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert all(ln.split()[1].split(".")[0] in {"rebot", "sys"} for ln in imports)
