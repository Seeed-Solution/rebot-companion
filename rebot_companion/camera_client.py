"""reCamera Atomic API Client (HTTP & WebSocket)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import httpx

from .transforms import Transform


@dataclass
class MarkerDetection:
    """Represents a detected fiducial marker with 6-DoF pose."""
    id: int
    center: list[float]
    corners: list[list[float]]
    pose: Transform
    reprojection_error: float

    @classmethod
    def from_dict(cls, d: dict) -> MarkerDetection:
        p = d.get("pose", {})
        t = p.get("t", [0.0, 0.0, 0.0])
        r_deg = p.get("r_deg", [0.0, 0.0, 0.0])
        tf = Transform.from_translation_euler(t, r_deg[0], r_deg[1], r_deg[2])
        return cls(
            id=int(d.get("id", -1)),
            center=list(d.get("center", [0.0, 0.0])),
            corners=list(d.get("corners", [])),
            pose=tf,
            reprojection_error=float(p.get("error", 0.0)),
        )


class ReCameraClient:
    """Client for invoking reCamera atomic vision capabilities."""

    def __init__(
        self,
        base_url: str = "http://192.168.42.1:8090",
        snapshot_url: Optional[str] = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # If snapshot url not specified, infer port 80 /snapshot.jpg
        if snapshot_url is None:
            host = self.base_url.split("//")[-1].split(":")[0]
            self.snapshot_url = f"http://{host}:80/snapshot.jpg"
        else:
            self.snapshot_url = snapshot_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def get_inference_state(self) -> dict:
        """Query current inference state and FPS."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.get(f"{self.base_url}/api/inference")
            res.raise_for_status()
            return res.json()

    def set_inference(self, enabled: bool) -> dict:
        """Toggle inference on or off (controls camera idle throttling)."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.post(f"{self.base_url}/api/inference", json={"enabled": enabled})
            res.raise_for_status()
            return res.json()

    def trigger(self, wait_ms: int = 2500) -> list[MarkerDetection]:
        """Atomic one-shot trigger: activates detector, grabs one fresh frame, returns to standby."""
        with httpx.Client(timeout=float(wait_ms) / 1000.0 + 2.0) as client:
            res = client.post(f"{self.base_url}/api/trigger?wait_ms={wait_ms}")
            res.raise_for_status()
            data = res.json()
            if not data.get("success", False):
                return []
            markers = data.get("markers", [])
            return [MarkerDetection.from_dict(m) for m in markers]

    def get_latest_result(self, after_frame: int = 0, wait_ms: int = 1000) -> list[MarkerDetection]:
        """Long poll the latest detection result."""
        with httpx.Client(timeout=float(wait_ms) / 1000.0 + 2.0) as client:
            res = client.get(f"{self.base_url}/api/results/latest?after={after_frame}&wait_ms={wait_ms}")
            res.raise_for_status()
            data = res.json()
            markers = data.get("markers", [])
            return [MarkerDetection.from_dict(m) for m in markers]

    def get_snapshot(self) -> bytes:
        """Fetch raw JPEG snapshot for VLM / multimodal reasoning."""
        with httpx.Client(timeout=self.timeout_sec) as client:
            res = client.get(self.snapshot_url)
            res.raise_for_status()
            return res.content
