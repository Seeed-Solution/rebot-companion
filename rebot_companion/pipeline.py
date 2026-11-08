"""Stage 2: Hardened Deterministic Pick & Place Pipeline."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from .camera_client import ReCameraClient, MarkerDetection
from .arm_client import ArmdClient
from .transforms import Transform


@dataclass
class PipelineConfig:
    target_marker_id: int = 0
    approach_height_offset: float = 0.08  # +80mm in Z
    place_position_base: tuple[float, float, float] = (0.20, 0.20, 0.05)
    max_allowed_distance: float = 0.80     # safety bound: max reach 800mm
    trigger_wait_ms: int = 2500


@dataclass
class PipelineResult:
    success: bool
    stage: str
    target_pose_base: Optional[Transform] = None
    elapsed_ms: float = 0.0
    error_message: str = ""


class DeterministicPickPlacePipeline:
    """
    Stage 2 execution engine: Runs a deterministic, closed-loop pick & place
    cycle directly on the local machine without cloud/remote LLM dependency.
    """

    def __init__(
        self,
        camera: ReCameraClient,
        arm: ArmdClient,
        cam_to_base: Transform,
        config: Optional[PipelineConfig] = None,
        ik_solver: Optional[Callable[[Transform], list[float]]] = None,
    ) -> None:
        self.camera = camera
        self.arm = arm
        self.cam_to_base = cam_to_base
        self.config = config or PipelineConfig()
        self.ik_solver = ik_solver

    def run_cycle(self) -> PipelineResult:
        """Execute one complete deterministic pick & place cycle."""
        t0 = time.perf_counter()

        # Step 1: Health check & Move arm to observation pose
        try:
            arm_status = self.arm.get_observation()
            if arm_status.e_stopped:
                return PipelineResult(
                    success=False, stage="health_check",
                    error_message="Arm is currently E-STOPPED"
                )
        except Exception as e:
            return PipelineResult(
                success=False, stage="health_check",
                error_message=f"Arm communication failed: {e}"
            )

        # Step 2: Atomic single-shot vision trigger
        try:
            detections = self.camera.trigger(wait_ms=self.config.trigger_wait_ms)
        except Exception as e:
            return PipelineResult(
                success=False, stage="vision_trigger",
                error_message=f"reCamera trigger failed: {e}"
            )

        target_marker: Optional[MarkerDetection] = None
        for m in detections:
            if m.id == self.config.target_marker_id:
                target_marker = m
                break

        if not target_marker:
            return PipelineResult(
                success=False, stage="marker_detection",
                error_message=f"Target marker #{self.config.target_marker_id} not detected"
            )

        # Step 3: Coordinate frame transformation (Camera Frame -> Arm Base Frame)
        # T_marker_in_base = T_cam_to_base @ T_marker_in_cam
        target_in_base = self.cam_to_base @ target_marker.pose
        t_pos = target_in_base.translation

        # Step 4: Safety boundary validation
        dist_from_base = float(np.linalg.norm(t_pos[:2]))
        if dist_from_base > self.config.max_allowed_distance or t_pos[2] < -0.05:
            return PipelineResult(
                success=False, stage="safety_check",
                target_pose_base=target_in_base,
                error_message=f"Target position {t_pos.tolist()} exceeds safety workspace boundary"
            )

        # Step 5: Planned waypoints (Approach -> Grasp -> Lift -> Place)
        approach_pos = np.array([t_pos[0], t_pos[1], t_pos[2] + self.config.approach_height_offset])
        grasp_pos = np.array([t_pos[0], t_pos[1], t_pos[2]])
        lift_pos = approach_pos
        place_pos = np.array(self.config.place_position_base)

        # If an inverse kinematics solver is provided, calculate joint positions and command the arm
        if self.ik_solver is not None:
            try:
                # 1. Approach
                q_approach = self.ik_solver(Transform.from_rot_trans(target_in_base.rotation, approach_pos))
                self.arm.servo_joints(q_approach, duration_sec=1.5)
                time.sleep(1.6)

                # 2. Grasp
                q_grasp = self.ik_solver(Transform.from_rot_trans(target_in_base.rotation, grasp_pos))
                self.arm.servo_joints(q_grasp, duration_sec=1.0)
                time.sleep(1.2)

                # 3. Lift
                self.arm.servo_joints(q_approach, duration_sec=1.0)
                time.sleep(1.2)

                # 4. Place
                q_place = self.ik_solver(Transform.from_rot_trans(target_in_base.rotation, place_pos))
                self.arm.servo_joints(q_place, duration_sec=1.5)
                time.sleep(1.6)
            except Exception as e:
                return PipelineResult(
                    success=False, stage="motion_execution",
                    target_pose_base=target_in_base,
                    error_message=f"IK motion execution failed: {e}"
                )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            success=True,
            stage="completed",
            target_pose_base=target_in_base,
            elapsed_ms=elapsed,
        )
