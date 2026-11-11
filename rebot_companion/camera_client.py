"""reCamera Atomic API Client (HTTP & WebSocket)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rebot import _http

from .transforms import Transform

log = logging.getLogger(__name__)


class CameraError(Exception):
    """reCamera API error other than a trigger timeout."""


def _check(res: _http.HttpResponse, what: str) -> None:
    if res.status_code == 415:
        raise CameraError(
            f"{what} rejected with 415 Unsupported Media Type: the "
            "aruco-detector (>= 0.1.3) requires 'Content-Type: application/json' on POST"
        )
    res.raise_for_status()


class CameraTriggerTimeout(Exception):
    """The trigger produced no result in time (HTTP timeout or ``success: false``).

    Distinct from "no marker": a completed trigger without markers returns an empty list.
    """

    def __init__(self, message: str, raw: dict | None = None) -> None:
        super().__init__(message)
        self.raw = raw or {}


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
        """Parse one marker. Raises ``ValueError`` when the marker has no ``pose`` (no ``t`` vector)."""
        p = d.get("pose")
        if not isinstance(p, dict) or p.get("t") is None:
            raise ValueError(f"marker {d.get('id')} has no pose")
        t = p["t"]
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
        snapshot_url: str | None = None,
        timeout_sec: float = 5.0,
        transport: _http.Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        # No default: a JPEG snapshot endpoint has not been verified on the device firmware.
        self.snapshot_url = snapshot_url.rstrip("/") if snapshot_url else None
        self.timeout_sec = timeout_sec

    def _req(self, method: str, url: str, timeout: float, **kw) -> _http.HttpResponse:
        return _http.request(method, url, timeout=timeout, transport=self._transport, **kw)

    def get_inference_state(self) -> dict:
        """Query current inference state and FPS."""
        res = self._req("GET", f"{self.base_url}/api/inference", self.timeout_sec)
        res.raise_for_status()
        return res.json()

    def set_inference(self, enabled: bool) -> dict:
        """Toggle inference on or off (controls camera idle throttling)."""
        res = self._req("POST", f"{self.base_url}/api/inference", self.timeout_sec, json={"enabled": enabled})
        _check(res, "POST /api/inference")
        return res.json()

    def trigger(self, wait_ms: int = 2500) -> list[MarkerDetection]:
        """Atomic one-shot trigger: activates detector, grabs one fresh frame, returns to standby.

        Returns the markers that carry a pose (an empty list means the frame had no usable
        marker). Raises :class:`CameraTriggerTimeout` when no result arrived in time.
        """
        try:
            # aruco-detector >= 0.1.3 answers 415 to a POST without a JSON content type.
            url = f"{self.base_url}/api/trigger?wait_ms={wait_ms}"
            res = self._req("POST", url, float(wait_ms) / 1000.0 + 2.0, json={})
            _check(res, "POST /api/trigger")
            data = res.json()
        except _http.HttpTimeout as exc:
            raise CameraTriggerTimeout(f"reCamera trigger timed out: {exc}") from exc
        if not data.get("success", False):
            detail = data.get("error") or data.get("message") or "success=false"
            raise CameraTriggerTimeout(f"reCamera trigger returned no result within {wait_ms} ms ({detail})", data)
        return _parse_markers(data.get("markers", []))

    def get_latest_result(self, after_frame: int = 0, wait_ms: int = 1000) -> list[MarkerDetection]:
        """Long poll the latest detection result."""
        url = f"{self.base_url}/api/results/latest?after={after_frame}&wait_ms={wait_ms}"
        res = self._req("GET", url, float(wait_ms) / 1000.0 + 2.0)
        res.raise_for_status()
        return _parse_markers(res.json().get("markers", []))

    def get_snapshot(self) -> bytes:
        """Fetch raw JPEG snapshot for VLM / multimodal reasoning.

        Requires ``snapshot_url`` to be passed to the constructor; there is no verified default endpoint.
        """
        if not self.snapshot_url:
            raise RuntimeError(
                "snapshot_url is not configured: pass ReCameraClient(snapshot_url=...) with an endpoint that "
                "serves a JPEG (no snapshot endpoint is verified on the reCamera firmware; RTSP "
                "rtsp://<host>:8554/live0 is available for frame grabs)"
            )
        res = self._req("GET", self.snapshot_url, self.timeout_sec)
        res.raise_for_status()
        return res.body


def _parse_markers(markers: list) -> list[MarkerDetection]:
    """Parse markers, skipping any without a pose (they must not become a target at the camera origin)."""
    out: list[MarkerDetection] = []
    for m in markers:
        try:
            out.append(MarkerDetection.from_dict(m))
        except ValueError as exc:
            log.warning("skipping marker: %s", exc)
    return out
