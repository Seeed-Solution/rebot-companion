"""Stage 2: Hardened Deterministic Pick & Place Pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rebot.client import ArmCommandRejected, ArmdClient, ArmdError, ArmSession, SafeExitOutcome

from . import kinematics
from .calibration import HandEyeResult
from .camera_client import CameraTriggerTimeout, MarkerDetection, ReCameraClient
from .ik import GRIPPER_CLOSED, GRIPPER_INDEX, GRIPPER_OPEN
from .transforms import Transform


@dataclass
class PipelineConfig:
    target_marker_id: int = 0
    approach_height_offset: float = 0.08  # +80mm in Z
    place_position_base: tuple[float, float, float] = (0.20, 0.20, 0.05)
    max_allowed_distance: float = 0.80  # safety bound: max reach 800mm
    trigger_wait_ms: int = 2500
    # Requested segment durations (s); armd stretches them to respect joint limits.
    approach_duration: float = 1.5
    grasp_duration: float = 1.0
    lift_duration: float = 1.0
    place_duration: float = 1.5
    gripper_duration: float = 1.0
    # Gripper joint (index 6 of armd's 7-joint vector) targets; open is negative.
    gripper_open: float = GRIPPER_OPEN
    gripper_closed: float = GRIPPER_CLOSED
    enable_torque: bool = True  # arm torque if the arm is Idle; otherwise fail the cycle
    # After the last segment: "unload" (park + torque off), "home", or "hold".
    # With a pipeline-owned session, "home"/"hold" leave the arm torqued without a
    # lease holder, and armd latches Fault(WatchdogTimeout) after lease_timeout_ms.
    finish_action: str = "unload"
    # On a motion failure, run ArmSession.safe_exit() (unload) on a caller-provided session too.
    safe_exit_on_failure: bool = True
    # eye_in_hand: max joint change (rad) between the reads before and after the trigger.
    max_joint_drift_during_detection: float = 0.01
    # Motor sign/offset for FK (eye_in_hand), same as kinematics.to_model.
    kinematics_sign: tuple[float, ...] = kinematics.DEFAULT_SIGN
    kinematics_offset: tuple[float, ...] = kinematics.DEFAULT_OFFSET


@dataclass
class PipelineResult:
    """Cycle outcome. ``stage`` is ``completed`` only when the arm ran the whole cycle.

    ``dry_run``: no IK solver configured; the target was computed and checked, the arm did not move.
    """

    success: bool
    stage: str
    target_pose_base: Transform | None = None
    elapsed_ms: float = 0.0
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)
    safe_exit: SafeExitOutcome | None = None


class DeterministicPickPlacePipeline:
    """
    Stage 2 execution engine: Runs a deterministic, closed-loop pick & place
    cycle directly on the local machine without cloud/remote LLM dependency.

    ``cam_to_base`` is either a fixed ``T_cam_to_base`` (eye_to_hand) or a
    :class:`HandEyeResult`; for an eye_in_hand result the camera pose is
    ``T_base_gripper(q) @ T_cam_to_gripper`` with ``q`` read at detection time.

    ``ik_solver`` maps a base-frame pose to joint targets (for the B601-DM use
    :class:`rebot_companion.ik.B601IkAdapter`, which returns armd's 7-joint
    vector). Without one the cycle ends with ``stage="dry_run"``, ``success=False``.

    Motion runs inside an armd control lease. Pass ``session`` (an open
    :class:`ArmSession`) to keep one lease across cycles; otherwise each cycle
    opens and releases its own session.
    """

    def __init__(
        self,
        camera: ReCameraClient,
        arm: ArmdClient,
        cam_to_base: Transform | HandEyeResult,
        config: PipelineConfig | None = None,
        ik_solver: Callable[[Transform], list[float]] | None = None,
        session: ArmSession | None = None,
    ) -> None:
        if config is not None and config.finish_action not in ("unload", "home", "hold"):
            raise ValueError("finish_action must be 'unload', 'home' or 'hold'")
        self.camera = camera
        self.arm = arm
        self.hand_eye = (
            cam_to_base if isinstance(cam_to_base, HandEyeResult) else HandEyeResult("eye_to_hand", cam_to_base)
        )
        self.config = config or PipelineConfig()
        self.ik_solver = ik_solver
        self.session = session

    @property
    def cam_to_base(self) -> Transform:
        """Fixed ``T_cam_to_base`` (eye_to_hand only)."""
        return self.hand_eye.cam_to_base()

    def run_cycle(self) -> PipelineResult:
        """Execute one complete deterministic pick & place cycle."""
        t0 = time.perf_counter()
        cfg = self.config
        eye_in_hand = self.hand_eye.needs_joints

        # Step 1: Health check (and arm pose for eye_in_hand)
        try:
            arm_status = self.arm.get_observation()
            if arm_status.e_stopped:
                return PipelineResult(success=False, stage="health_check", error_message="Arm is currently E-STOPPED")
            if arm_status.faulted:
                return PipelineResult(
                    success=False,
                    stage="health_check",
                    error_message=f"Arm is in FAULT ({arm_status.fault_reason}); clear_fault() required",
                )
            if eye_in_hand and arm_status.safety_state == "Moving":
                return PipelineResult(
                    success=False, stage="health_check", error_message="eye_in_hand detection needs a still arm"
                )
        except Exception as e:
            return PipelineResult(success=False, stage="health_check", error_message=f"Arm communication failed: {e}")

        # Step 2: Atomic single-shot vision trigger
        try:
            detections = self.camera.trigger(wait_ms=cfg.trigger_wait_ms)
        except CameraTriggerTimeout as e:
            return PipelineResult(success=False, stage="vision_trigger", error_message=str(e))
        except Exception as e:
            return PipelineResult(success=False, stage="vision_trigger", error_message=f"reCamera trigger failed: {e}")

        target_marker: MarkerDetection | None = None
        for m in detections:
            if m.id == cfg.target_marker_id:
                target_marker = m
                break

        if not target_marker:
            return PipelineResult(
                success=False,
                stage="marker_detection",
                error_message=f"Target marker #{cfg.target_marker_id} not detected",
            )

        # Step 3: Coordinate frame transformation (Camera Frame -> Arm Base Frame)
        # T_marker_in_base = T_cam_to_base @ T_marker_in_cam
        try:
            if eye_in_hand:
                after = self.arm.get_observation()
                drift = max(
                    (abs(a - b) for a, b in zip(after.positions, arm_status.positions)),
                    default=float("inf"),
                )
                if len(after.positions) != len(arm_status.positions) or drift > cfg.max_joint_drift_during_detection:
                    return PipelineResult(
                        success=False,
                        stage="frame_transform",
                        error_message=f"Arm moved during eye_in_hand detection (max joint change {drift:.4f} rad)",
                    )
            cam_to_base = self.hand_eye.cam_to_base(arm_status.positions, cfg.kinematics_sign, cfg.kinematics_offset)
        except Exception as e:
            return PipelineResult(success=False, stage="frame_transform", error_message=f"Camera pose unavailable: {e}")
        target_in_base = cam_to_base @ target_marker.pose
        t_pos = target_in_base.translation

        # Step 4: Safety boundary validation
        dist_from_base = float(np.linalg.norm(t_pos[:2]))
        if dist_from_base > cfg.max_allowed_distance or t_pos[2] < -0.05:
            return PipelineResult(
                success=False,
                stage="safety_check",
                target_pose_base=target_in_base,
                error_message=f"Target position {t_pos.tolist()} exceeds safety workspace boundary",
            )

        # Step 5: Planned waypoints (Approach -> Grasp -> Lift -> Place)
        approach_pos = np.array([t_pos[0], t_pos[1], t_pos[2] + cfg.approach_height_offset])
        grasp_pos = np.array([t_pos[0], t_pos[1], t_pos[2]])
        lift_pos = approach_pos
        place_pos = np.array(cfg.place_position_base)

        if self.ik_solver is None:
            return PipelineResult(
                success=False,
                stage="dry_run",
                target_pose_base=target_in_base,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error_message="No IK solver configured: target computed and checked, arm not moved",
            )

        warnings: list[str] = []
        # Solve every waypoint before moving so an unreachable pose aborts with the arm still.
        try:
            set_seed = getattr(self.ik_solver, "set_seed", None)
            if callable(set_seed) and arm_status.positions:
                set_seed(arm_status.positions)
            rot = target_in_base.rotation
            q_approach = list(self.ik_solver(Transform.from_rot_trans(rot, approach_pos)))
            q_grasp = list(self.ik_solver(Transform.from_rot_trans(rot, grasp_pos)))
            q_lift = list(self.ik_solver(Transform.from_rot_trans(rot, lift_pos)))
            q_place = list(self.ik_solver(Transform.from_rot_trans(rot, place_pos)))
        except Exception as e:
            return PipelineResult(
                success=False, stage="ik_solve", target_pose_base=target_in_base, error_message=f"IK failed: {e}"
            )

        has_gripper = all(len(q) > GRIPPER_INDEX for q in (q_approach, q_grasp, q_lift, q_place))
        if not has_gripper:
            warnings.append(f"IK targets have no gripper joint (index {GRIPPER_INDEX}); gripper not commanded")

        def grip(q: list[float], value: float) -> list[float]:
            q = list(q)
            if has_gripper:
                q[GRIPPER_INDEX] = value
            return q

        opened, closed = cfg.gripper_open, cfg.gripper_closed

        def execute(s: ArmSession) -> None:
            obs = s.latest_observation() or s.wait_for_observation()
            if not obs.is_operational:
                if not cfg.enable_torque:
                    raise ArmdError(f"Arm is {obs.safety_state}, torque disabled by config")
                s.torque(True)
            if has_gripper:
                if len(obs.positions) != len(q_approach):
                    raise ArmdError(f"armd reports {len(obs.positions)} joints, IK returned {len(q_approach)}")
                s.move_to(grip(obs.positions, opened), duration=cfg.gripper_duration)  # 0. open gripper in place
            s.move_to(grip(q_approach, opened), duration=cfg.approach_duration)  # 1. Approach
            s.move_to(grip(q_grasp, opened), duration=cfg.grasp_duration)  # 2. Grasp pose
            if has_gripper:
                s.move_to(grip(q_grasp, closed), duration=cfg.gripper_duration)  # 3. Close gripper
            s.move_to(grip(q_lift, closed), duration=cfg.lift_duration)  # 4. Lift
            s.move_to(grip(q_place, closed), duration=cfg.place_duration)  # 5. Place
            if has_gripper:
                s.move_to(grip(q_place, opened), duration=cfg.gripper_duration)  # 6. Release
            if cfg.finish_action == "home":
                s.home()
            elif cfg.finish_action == "unload":
                try:
                    s.unload()
                except ArmCommandRejected as e:
                    if e.code != "UNSUPPORTED":
                        raise
                    warnings.append(f"unload unsupported by armd, arm left holding: {e}")

        outcome: SafeExitOutcome | None = None
        try:
            if self.session is not None:
                try:
                    execute(self.session)
                except BaseException:
                    if cfg.safe_exit_on_failure:
                        outcome = self.session.safe_exit()
                    raise
            else:
                session = self.arm.session()
                try:
                    with session as s:  # default safe_exit="error" unloads on failure
                        execute(s)
                finally:
                    outcome = getattr(session, "safe_exit_outcome", None)
        except Exception as e:
            message = f"Motion execution failed: {e}"
            if isinstance(outcome, SafeExitOutcome):
                message += f"; safe exit: {outcome.action}" + (f" ({outcome.message})" if outcome.message else "")
            return PipelineResult(
                success=False,
                stage="motion_execution",
                target_pose_base=target_in_base,
                error_message=message,
                warnings=warnings,
                safe_exit=outcome if isinstance(outcome, SafeExitOutcome) else None,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            success=True,
            stage="completed",
            target_pose_base=target_in_base,
            elapsed_ms=elapsed,
            warnings=warnings,
        )
