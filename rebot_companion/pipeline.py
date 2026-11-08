"""Stage 2: Hardened Deterministic Pick & Place Pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .arm_client import ArmCommandRejected, ArmdClient, ArmdError, ArmSession
from .camera_client import MarkerDetection, ReCameraClient
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
    enable_torque: bool = True  # arm torque if the arm is Idle; otherwise fail the cycle
    # After the last segment: "unload" (park + torque off), "home", or "hold".
    # With a pipeline-owned session, "home"/"hold" leave the arm torqued without a
    # lease holder, and armd latches Fault(WatchdogTimeout) after lease_timeout_ms.
    finish_action: str = "unload"


@dataclass
class PipelineResult:
    success: bool
    stage: str
    target_pose_base: Transform | None = None
    elapsed_ms: float = 0.0
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)


class DeterministicPickPlacePipeline:
    """
    Stage 2 execution engine: Runs a deterministic, closed-loop pick & place
    cycle directly on the local machine without cloud/remote LLM dependency.

    Motion runs inside an armd control lease. Pass ``session`` (an open
    :class:`ArmSession`) to keep one lease across cycles; otherwise each cycle
    opens and releases its own session.
    """

    def __init__(
        self,
        camera: ReCameraClient,
        arm: ArmdClient,
        cam_to_base: Transform,
        config: PipelineConfig | None = None,
        ik_solver: Callable[[Transform], list[float]] | None = None,
        session: ArmSession | None = None,
    ) -> None:
        if config is not None and config.finish_action not in ("unload", "home", "hold"):
            raise ValueError("finish_action must be 'unload', 'home' or 'hold'")
        self.camera = camera
        self.arm = arm
        self.cam_to_base = cam_to_base
        self.config = config or PipelineConfig()
        self.ik_solver = ik_solver
        self.session = session

    @contextmanager
    def _control(self) -> Iterator[ArmSession]:
        if self.session is not None:
            yield self.session
            return
        with self.arm.session() as s:
            yield s

    def run_cycle(self) -> PipelineResult:
        """Execute one complete deterministic pick & place cycle."""
        t0 = time.perf_counter()

        # Step 1: Health check & Move arm to observation pose
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
        except Exception as e:
            return PipelineResult(success=False, stage="health_check", error_message=f"Arm communication failed: {e}")

        # Step 2: Atomic single-shot vision trigger
        try:
            detections = self.camera.trigger(wait_ms=self.config.trigger_wait_ms)
        except Exception as e:
            return PipelineResult(success=False, stage="vision_trigger", error_message=f"reCamera trigger failed: {e}")

        target_marker: MarkerDetection | None = None
        for m in detections:
            if m.id == self.config.target_marker_id:
                target_marker = m
                break

        if not target_marker:
            return PipelineResult(
                success=False,
                stage="marker_detection",
                error_message=f"Target marker #{self.config.target_marker_id} not detected",
            )

        # Step 3: Coordinate frame transformation (Camera Frame -> Arm Base Frame)
        # T_marker_in_base = T_cam_to_base @ T_marker_in_cam
        target_in_base = self.cam_to_base @ target_marker.pose
        t_pos = target_in_base.translation

        # Step 4: Safety boundary validation
        dist_from_base = float(np.linalg.norm(t_pos[:2]))
        if dist_from_base > self.config.max_allowed_distance or t_pos[2] < -0.05:
            return PipelineResult(
                success=False,
                stage="safety_check",
                target_pose_base=target_in_base,
                error_message=f"Target position {t_pos.tolist()} exceeds safety workspace boundary",
            )

        # Step 5: Planned waypoints (Approach -> Grasp -> Lift -> Place)
        approach_pos = np.array([t_pos[0], t_pos[1], t_pos[2] + self.config.approach_height_offset])
        grasp_pos = np.array([t_pos[0], t_pos[1], t_pos[2]])
        lift_pos = approach_pos
        place_pos = np.array(self.config.place_position_base)

        warnings: list[str] = []
        if self.ik_solver is not None:
            # Solve every waypoint before moving so an unreachable pose aborts with the arm still.
            try:
                rot = target_in_base.rotation
                q_approach = self.ik_solver(Transform.from_rot_trans(rot, approach_pos))
                q_grasp = self.ik_solver(Transform.from_rot_trans(rot, grasp_pos))
                q_lift = self.ik_solver(Transform.from_rot_trans(rot, lift_pos))
                q_place = self.ik_solver(Transform.from_rot_trans(rot, place_pos))
            except Exception as e:
                return PipelineResult(
                    success=False, stage="ik_solve", target_pose_base=target_in_base, error_message=f"IK failed: {e}"
                )
            cfg = self.config
            try:
                with self._control() as s:
                    obs = s.latest_observation() or s.wait_for_observation()
                    if not obs.is_operational:
                        if not cfg.enable_torque:
                            raise ArmdError(f"Arm is {obs.safety_state}, torque disabled by config")
                        s.torque(True)
                    s.move_to(q_approach, duration=cfg.approach_duration)  # 1. Approach
                    s.move_to(q_grasp, duration=cfg.grasp_duration)  # 2. Grasp
                    s.move_to(q_lift, duration=cfg.lift_duration)  # 3. Lift
                    s.move_to(q_place, duration=cfg.place_duration)  # 4. Place
                    if cfg.finish_action == "home":
                        s.home()
                    elif cfg.finish_action == "unload":
                        try:
                            s.unload()
                        except ArmCommandRejected as e:
                            if e.code != "UNSUPPORTED":
                                raise
                            warnings.append(f"unload unsupported by armd, arm left holding: {e}")
            except Exception as e:
                return PipelineResult(
                    success=False,
                    stage="motion_execution",
                    target_pose_base=target_in_base,
                    error_message=f"Motion execution failed: {e}",
                )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            success=True,
            stage="completed",
            target_pose_base=target_in_base,
            elapsed_ms=elapsed,
            warnings=warnings,
        )
