"""Detections through armd's detector proxy (``/api/v1/detection/*``, rebot-armd ``docs/detection_api.md``).

Standard library only. :func:`parse_detections` reads the same payload formats as the cockpit's
``parseDetections``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from . import _http
from .errors import DetectorUnavailable, RebotError

log = logging.getLogger(__name__)

PIXEL_UNIT_THRESHOLD = 1.05  # cx|cy|w|h above this means pixels (cockpit parseDetections)
PIPELINES = ("yolo", "aruco")


@dataclass
class Detection:
    label: str
    score: float
    box_norm: tuple[float, float, float, float]  # x0, y0, x1, y1 in [0, 1], clamped to the frame
    track_id: Any = None
    center_norm: tuple[float, float] = (0.0, 0.0)  # detector centre, clamped to [0, 1]
    corners_norm: list[tuple[float, float]] | None = None  # marker corners (not clamped)
    frame_id: int | None = None  # detector frame the box comes from (set by Arm.detect)
    ts_ms: float | None = None

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "score": round(self.score, 4),
            "box_norm": [round(v, 4) for v in self.box_norm],
            "center_norm": [round(v, 4) for v in self.center_norm],
            "track_id": self.track_id,
        }
        if self.corners_norm is not None:
            d["corners_norm"] = [[round(u, 4), round(v, 4)] for u, v in self.corners_norm]
        return d


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _flatten_xy(v: Any) -> list[Any]:
    if isinstance(v, dict):
        return [v.get("x"), v.get("y")]
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in _flatten_xy(e)]
    return [v]


def _corners(item: dict) -> list[tuple[float, float]] | None:
    """Four marker corners from ``corners`` / ``corner`` / ``polygon`` (also under ``pose`` / ``detail``)."""
    sources = [item.get(k) for k in ("pose", "pose3d", "pose_3d") if isinstance(item.get(k), dict)]
    det = item.get("detail") if isinstance(item.get("detail"), dict) else {}
    if isinstance(det.get("pose"), dict):
        sources.append(det["pose"])
    for src in [*sources[:1], item, det]:
        for k in ("corners", "corner", "polygon"):
            raw = src.get(k)
            if not isinstance(raw, list):
                continue
            flat = _flatten_xy(raw)
            nums = [_num(v) for v in flat[:8]]
            if len(nums) == 8 and all(n is not None for n in nums):
                return [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]  # type: ignore[misc]
    return None


def _items(raw: Any) -> tuple[list, float | None, float | None]:
    if isinstance(raw, list):
        if len(raw) >= 5 and isinstance(raw[0], (int, float)):
            return [raw], None, None
        return list(raw), None, None
    if not isinstance(raw, dict):
        return [], None, None
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    items: list = []
    if raw.get("format"):
        items = [raw]
    elif isinstance(raw.get("boxes"), list):
        items = raw["boxes"]
    elif isinstance(raw.get("detections"), list):
        items = raw["detections"]
    elif isinstance(data.get("boxes"), list):
        items = data["boxes"]
    elif isinstance(data.get("detections"), list):
        items = data["detections"]
    elif isinstance(raw.get("data"), list):
        items = raw["data"]
    elif isinstance(raw.get("results"), list):
        items = raw["results"]
    items = list(items)
    for k in ("markers", "aruco"):
        m = raw.get(k) if isinstance(raw.get(k), list) else data.get(k) if isinstance(data.get(k), list) else None
        if m:
            items += [{"label": "aruco", **x} if isinstance(x, dict) else x for x in m]
    ref_w = raw.get("width") or raw.get("w") or data.get("width") or data.get("w")
    ref_h = raw.get("height") or raw.get("h") or data.get("height") or data.get("h")
    for src in (raw, data):
        res = src.get("resolution")
        if not ref_w and isinstance(res, list) and len(res) >= 2:
            ref_w, ref_h = res[0], res[1]
    return items, _num(ref_w), _num(ref_h)


def _raw_box(item: Any, corners: list[tuple[float, float]] | None):
    """``(cx, cy, w, h, score, label, track_id)`` in the payload's units, or None to drop."""
    if isinstance(item, (list, tuple)) and len(item) >= 5:
        # SSCMA array [cx, cy, w, h, score, target(, track)], centre per the cockpit (index.html:2471);
        # needs verification against real yolo-detector output.
        vals = [_num(v) for v in item[:4]]
        if any(v is None for v in vals):
            return None
        cx, cy, w, h = vals  # type: ignore[misc]
        if w <= 0 or h <= 0:
            return None
        label = str(item[5]) if len(item) > 5 else "obj"
        return cx, cy, w, h, _num(item[4]), label, item[6] if len(item) > 6 else None
    if not isinstance(item, dict):
        return None
    fmt = item.get("format") or ("corners" if corners else None)
    if fmt == "corners":
        if not corners:
            return None
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w <= 0 or h <= 0:
            return None
        cx, cy = sum(xs) / 4, sum(ys) / 4
    elif fmt == "xywh_center":
        cx = _num(item.get("cx", item.get("x")))
        cy = _num(item.get("cy", item.get("y")))
        w = _num(item.get("w", item.get("width")))
        h = _num(item.get("h", item.get("height")))
        if None in (cx, cy, w, h) or w <= 0 or h <= 0:
            return None
    elif fmt == "xyxy":
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else item.get("box")
        bbox = bbox if isinstance(bbox, list) and len(bbox) >= 4 else None
        if "x1" in item and "x2" in item:
            x0, x1 = _num(item["x1"]), _num(item["x2"])
        elif "x0" in item and "x1" in item:
            x0, x1 = _num(item["x0"]), _num(item["x1"])
        elif bbox:
            x0, x1 = _num(bbox[0]), _num(bbox[2])
        else:
            return None
        if "y1" in item and "y2" in item:
            y0, y1 = _num(item["y1"]), _num(item["y2"])
        elif "y0" in item and "y1" in item:
            y0, y1 = _num(item["y0"]), _num(item["y1"])
        elif bbox:
            y0, y1 = _num(bbox[1]), _num(bbox[3])
        else:
            return None
        if None in (x0, y0, x1, y1) or x1 <= x0 or y1 <= y0:
            return None
        w, h, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    elif fmt == "xywh_top_left":
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x0, y0, w, h = (_num(v) for v in bbox[:4])
        else:
            x0 = _num(item.get("x", item.get("x0")))
            y0 = _num(item.get("y", item.get("y0")))
            w = _num(item.get("w", item.get("width")))
            h = _num(item.get("h", item.get("height")))
        if None in (x0, y0, w, h) or w <= 0 or h <= 0:
            return None
        cx, cy = x0 + w / 2, y0 + h / 2
    else:
        return None
    score = item.get("score", item.get("confidence"))
    score = _num(score) if score is not None else (1.0 if corners else 0.0)  # marker decoders send no score
    label = next(
        (item[k] for k in ("label", "name", "target", "class_name") if item.get(k) is not None),
        "aruco" if corners else "obj",
    )
    track = item.get("track_id", item.get("id"))
    return cx, cy, w, h, score, str(label), track


def parse_detections(raw: Any, ref_res: tuple[int, int] | None = None) -> list[Detection]:
    """Detector payload -> normalised boxes (port of the cockpit's ``parseDetections``).

    Accepts the envelopes ``result`` / ``results`` / ``boxes`` / ``detections`` / ``data.*`` plus
    ``markers`` / ``aruco`` lists; items as SSCMA arrays ``[cx, cy, w, h, score, target(, track)]``
    or dicts with ``format`` ``xywh_center`` / ``xyxy`` / ``xywh_top_left`` / ``corners``. Values
    with ``cx|cy|w|h > 1.05`` are pixels and are divided by the payload ``width/height/resolution``,
    else by ``ref_res`` (usually the snapshot size); pixel items without a resolution are dropped.
    Scores are clamped to [0, 1]; boxes are clamped to the frame.
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        inner = raw.get("result") or (raw.get("results") if isinstance(raw.get("results"), dict) else None)
        if isinstance(inner, dict) and not raw.get("boxes") and not raw.get("detections"):
            ref = ref_res or ((raw["width"], raw["height"]) if raw.get("width") and raw.get("height") else None)
            return parse_detections(inner, ref)
    items, ref_w, ref_h = _items(raw)
    if ref_res and len(ref_res) >= 2:
        ref_w = ref_w or _num(ref_res[0])
        ref_h = ref_h or _num(ref_res[1])
    has_ref = bool(ref_w and ref_h and ref_w > 0 and ref_h > 0)

    out: list[Detection] = []
    for item in items:
        corners = _corners(item) if isinstance(item, dict) else None
        box = _raw_box(item, corners)
        if box is None:
            continue
        cx, cy, w, h, score, label, track = box
        if score is None:
            continue
        score = max(0.0, min(1.0, score))
        pixel = max(cx, cy, w, h) > PIXEL_UNIT_THRESHOLD
        sx, sy = (ref_w, ref_h) if pixel else (1.0, 1.0)
        if pixel and not has_ref:
            continue
        clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
        x0, y0 = clamp((cx - w / 2) / sx), clamp((cy - h / 2) / sy)
        x1, y1 = clamp((cx + w / 2) / sx), clamp((cy + h / 2) / sy)
        if x1 - x0 <= 0.001 or y1 - y0 <= 0.001:
            continue
        out.append(
            Detection(
                label=label,
                score=score,
                box_norm=(x0, y0, x1, y1),
                track_id=track,
                center_norm=(clamp(cx / sx), clamp(cy / sy)),
                corners_norm=[(u / sx, v / sy) for u, v in corners] if corners else None,
            )
        )
    return out


class DetectionClient:
    """armd's detector proxy (``/api/v1/detection/*``, rebot-armd ``docs/detection_api.md``)."""

    def __init__(
        self,
        armd_url: str = "http://192.168.42.1:8765",
        timeout: float = 3.0,
        transport: _http.Transport | None = None,
    ) -> None:
        self.base_url = armd_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self.trigger_supported: bool | None = None  # None until the first trigger call
        # perception API 2 (rebot-vision >= 1.3): triggers run the pipelines they name
        self.pipelines_supported: bool | None = None  # None until probed

    def _call(self, method: str, path: str, wait_ms: int = 0, **kw: Any) -> _http.HttpResponse:
        # armd waits wait_ms (capped at 5000) + 2 s for the upstream.
        timeout = self.timeout + min(max(wait_ms, 0), 5000) / 1000.0 + 2.0
        try:
            res = _http.request(method, f"{self.base_url}{path}", timeout=timeout, transport=self._transport, **kw)
        except _http.HttpError as exc:
            raise DetectorUnavailable(f"armd detection proxy unreachable: {exc}") from exc
        if res.status_code == 502:
            raise DetectorUnavailable("detector unreachable (502 from armd)")
        return res

    def latest(self, after: int | None = None, wait_ms: int = 0) -> dict:
        params: dict = {"wait_ms": int(wait_ms)}
        if after is not None:
            params["after"] = int(after)
        res = self._call("GET", "/api/v1/detection/latest", wait_ms, params=params)
        res.raise_for_status()
        return res.json()

    def inference(self) -> dict:
        res = self._call("GET", "/api/v1/detection/inference")
        res.raise_for_status()
        return res.json()

    def set_inference(self, enabled: bool) -> dict:
        res = self._call("POST", "/api/v1/detection/inference", json={"enabled": bool(enabled)})
        res.raise_for_status()
        return res.json()

    def supports_pipelines(self) -> bool:
        """Whether the provider runs the pipelines a trigger names (perception API 2).

        Read once from ``GET /api/v1/vision/mode`` (``per_request_pipelines``); a provider
        without the route (404) or any other error counts as no.
        """
        if self.pipelines_supported is None:
            try:
                res = self._call("GET", "/api/v1/vision/mode")
                ok = res.status_code == 200 and res.json().get("per_request_pipelines") is True
            except (DetectorUnavailable, ValueError):
                return False  # not cached: the provider may come up later
            self.pipelines_supported = ok
        return self.pipelines_supported

    def detect_once(self, wait_ms: int = 2500, pipelines: Sequence[str] | None = None) -> dict:
        """One fresh inference result, normalised to ``{frame_id, ts_ms, raw, timeout, pipelines}``.

        Uses ``POST /api/v1/detection/trigger``. ``pipelines`` (``"yolo"`` / ``"aruco"``) is sent
        when the provider supports per-request pipelines (perception API 2); ``None`` then means
        both. Older providers run their current mode and ``pipelines`` is ignored. ``pipelines``
        in the result lists what ran on the frame when the provider reports it, else ``None``.
        Detector builds without the trigger answer 404; then inference is enabled, ``latest`` is
        long-polled for a newer frame, and inference is disabled again only if this call
        enabled it.
        """
        wait_ms = max(0, min(int(wait_ms), 5000))
        if pipelines is not None:
            pipelines = list(dict.fromkeys(pipelines))
            if not pipelines or any(p not in PIPELINES for p in pipelines):
                raise ValueError(f"pipelines must be a non-empty subset of {list(PIPELINES)}")
        if self.trigger_supported is not False:
            kw: dict = {"params": {"wait_ms": wait_ms}}
            if self.supports_pipelines():
                kw["json"] = {"pipelines": pipelines or list(PIPELINES)}
            res = self._call("POST", "/api/v1/detection/trigger", wait_ms, **kw)
            if res.status_code != 404:
                self.trigger_supported = True
                res.raise_for_status()
                body = res.json()
                return {
                    "frame_id": body.get("frame_id"),
                    "ts_ms": body.get("ts_ms"),
                    "timeout": bool(body.get("timeout")) or body.get("success") is False,
                    "pipelines": _ran(body),
                    "raw": body,
                }
            self.trigger_supported = False
        before = self.latest(wait_ms=0)
        enabled_here = False
        if not self.inference().get("enabled"):
            self.set_inference(True)
            enabled_here = True
        try:
            body = self.latest(after=before.get("frame_id") or 0, wait_ms=wait_ms)
        finally:
            if enabled_here:
                try:
                    self.set_inference(False)
                except RebotError as exc:
                    log.warning("could not disable inference again: %s", exc)
        return {
            "frame_id": body.get("frame_id"),
            "ts_ms": body.get("ts_ms"),
            "timeout": bool(body.get("timeout")),
            "pipelines": _ran(body),
            "raw": body,
        }


PIPELINES = ("yolo", "aruco")


def _ran(body: dict) -> list[str] | None:
    """Pipelines that ran on a result frame (rebot-vision ``pipelines.*.ran``), else ``None``."""
    p = body.get("pipelines")
    if not isinstance(p, dict):
        return None
    return [k for k in PIPELINES if isinstance(p.get(k), dict) and p[k].get("ran") is True]


def _ran(body: dict) -> list[str] | None:
    """Pipelines that ran on a result frame (rebot-vision ``pipelines.*.ran``), else ``None``."""
    p = body.get("pipelines")
    if not isinstance(p, dict):
        return None
    return [k for k in PIPELINES if isinstance(p.get(k), dict) and p[k].get("ran") is True]


__all__ = ["Detection", "DetectionClient", "DetectorUnavailable", "parse_detections", "PIPELINES"]
