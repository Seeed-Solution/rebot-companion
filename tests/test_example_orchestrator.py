"""examples/03_agent_orchestrator.py: placeholder calibration must not drive motion."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from rebot_companion import HandEyeCalibration, MarkerDetection, Transform

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "03_agent_orchestrator.py"


def _load():
    spec = importlib.util.spec_from_file_location("agent_orchestrator", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fail(*_a, **_k):
    raise AssertionError("must not reach the camera or the arm")


def test_pick_refused_without_calibration(monkeypatch):
    mod = _load()
    tools = mod.RobotCompanionAgentTools(ik_solver=lambda _pose: [0.0] * 7)
    monkeypatch.setattr(tools.camera, "trigger", _fail)
    monkeypatch.setattr(tools.arm, "get_observation", _fail)
    monkeypatch.setattr(mod.DeterministicPickPlacePipeline, "run_cycle", _fail)
    out = json.loads(tools.execute_tool("plan_and_execute_pick", {"marker_id": 0}))
    assert out["success"] is False and out["stage"] == "refused"
    assert tools.calibrated is False


def test_dry_run_without_calibration_has_no_ik(monkeypatch):
    mod = _load()
    tools = mod.RobotCompanionAgentTools(ik_solver=lambda _pose: [0.0] * 7, dry_run=True)
    assert tools.ik_solver is None
    seen = {}

    def run_cycle(self):
        seen["ik"] = self.ik_solver
        return _Res()

    class _Res:
        success, stage, error_message, target_pose_base, warnings = False, "dry_run", "", None, []

    monkeypatch.setattr(mod.DeterministicPickPlacePipeline, "run_cycle", run_cycle)
    out = json.loads(tools.execute_tool("plan_and_execute_pick", {"marker_id": 0}))
    assert out["stage"] == "dry_run" and out["motion_executed"] is False and seen["ik"] is None


def test_inspect_scene_uses_loaded_calibration(tmp_path: Path, monkeypatch):
    mod = _load()
    X = Transform.from_translation_euler([0.1, 0.2, 0.3], 0.0, 0.0, 0.0)
    f = tmp_path / "calibration.json"
    f.write_text(
        json.dumps(
            {"kind": "hand_eye_calibration", "setup": "eye_to_hand", "result": {"T_cam_to_base": X.matrix.tolist()}}
        )
    )
    tools = mod.RobotCompanionAgentTools(hand_eye=HandEyeCalibration.load_result(f))
    marker = MarkerDetection(
        id=3,
        center=[0, 0],
        corners=[],
        pose=Transform.from_translation_euler([0.0, 0.0, 0.5], 0, 0, 0),
        reprojection_error=0.0,
    )
    monkeypatch.setattr(tools.camera, "trigger", lambda wait_ms: [marker])
    out = json.loads(tools.execute_tool("inspect_scene", {}))
    assert out["calibrated"] is True and "warning" not in out
    np.testing.assert_allclose(out["detected_markers"][0]["position_base_xyz"], [0.1, 0.2, 0.8], atol=1e-4)
