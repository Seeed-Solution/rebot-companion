"""rebot-armd REST API Client."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import httpx


@dataclass
class ArmStatus:
    """Current telemetry from the robot arm supervisor."""
    connected: bool
    e_stopped: bool
    joint_positions: list[float]
    joint_velocities: list[float]
    joint_temperatures: list[float]
    timestamp_ms: int

    @classmethod
    def from_dict(cls, d: dict) -> ArmStatus:
        obs = d.get("observation", d)
        q = obs.get("positions", obs.get("q", []))
        dq = obs.get("velocities", obs.get("dq", []))
        temp = obs.get("temperatures", obs.get("temp", []))
        return cls(
            connected=bool(d.get("connected", True)),
            e_stopped=bool(d.get("e_stopped", False)),
            joint_positions=[float(x) for x in q],
            joint_velocities=[float(x) for x in dq],
            joint_temperatures=[float(x) for x in temp],
            timestamp_ms=int(d.get("timestamp_ms", 0)),
        )


class ArmdClient:
    """Client for controlling the arm daemon (rebot-armd)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout_sec: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def get_observation(self) -> ArmStatus:
        """Fetch current joint observation."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.get(f"{self.base_url}/api/v1/observation")
            res.raise_for_status()
            return ArmStatus.from_dict(res.json())

    def servo_joints(self, target_positions: list[float], duration_sec: float = 1.0) -> dict:
        """Send smooth servo step to joints."""
        payload = {
            "positions": target_positions,
            "duration": duration_sec,
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.post(f"{self.base_url}/api/v1/motion/servo", json=payload)
            res.raise_for_status()
            return res.json()

    def home(self) -> dict:
        """Move arm to predefined Home position."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.post(f"{self.base_url}/api/v1/motion/home", json={})
            res.raise_for_status()
            return res.json()

    def e_stop(self) -> dict:
        """Trigger emergency stop immediately."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.post(f"{self.base_url}/api/v1/safety/e_stop", json={})
            res.raise_for_status()
            return res.json()

    def clear_fault(self) -> dict:
        """Clear latch fault / e-stop condition."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.post(f"{self.base_url}/api/v1/safety/clear_fault", json={})
            res.raise_for_status()
            return res.json()
