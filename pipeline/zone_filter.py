"""Drops detections outside the configured detection zone(s) for the
camera's current view, "wide" or "zoomed" (configs/detection_zones.yaml). Zones are
normalized polygons (0..1 image coordinates) so they stay valid
regardless of stream resolution.

An empty (or missing) zone list for a view means "no filtering" —
the deliberate default, since most decks won't need zones on day one
and an empty zone list must never mean "reject everything".
"""

from __future__ import annotations

from inference.base import Detection


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    # Standard ray-casting test.
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class ZoneFilter:
    def __init__(self, zones_config: dict):
        # zones_config: {"wide": [[[x,y],...]], "zoomed": [...]}
        self.zones = zones_config.get("zones", zones_config)

    def filter(
        self, detections: list[Detection], preset_name: str, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        polygons = self.zones.get(preset_name) or []
        if not polygons:
            return detections

        h, w = frame_shape
        kept = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            if any(_point_in_polygon(cx, cy, poly) for poly in polygons):
                kept.append(det)
        return kept
