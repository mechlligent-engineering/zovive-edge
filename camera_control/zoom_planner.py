"""Pure math: animal bounding box -> how far to optically zoom in.
No camera I/O here (that's zoom_controller.py), so it's cheaply
unit-testable.

The camera cannot pan or tilt, and optical zoom magnifies about the
frame centre. So an animal off to one side moves further toward the
edge as we zoom, and zooming too far pushes it out of frame. The plan
therefore takes the SMALLER of:
  - the magnification that would make the animal fill
    `target_fill_ratio` of the frame, and
  - the largest magnification that keeps the whole box inside the frame
    (minus `edge_margin_ratio`), and
  - the lens's `max_optical_magnification`.
If that is less than `min_useful_magnification`, zooming isn't worth
the lost wide-view coverage and the plan is None.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoomPlanConfig:
    # Animal counts as "far" when its larger box side fills less than
    # this fraction of the matching frame side.
    far_fill_ratio: float = 0.25
    # Zoom until the animal fills about this fraction of the frame.
    target_fill_ratio: float = 0.5
    # Tele / wide focal length of the lens, e.g. 2.8-12 mm ≈ 4.3x.
    max_optical_magnification: float = 4.0
    min_useful_magnification: float = 1.5
    # Keep the box at least this far (fraction of half-frame) from the edge.
    edge_margin_ratio: float = 0.1
    # ONVIF zoom-space values for fully wide and fully tele.
    wide_zoom_level: float = 0.0
    max_zoom_level: float = 1.0

    @classmethod
    def from_config(cls, zoom_cfg: dict) -> ZoomPlanConfig:
        d = cls()
        return cls(
            far_fill_ratio=float(zoom_cfg.get("far_fill_ratio", d.far_fill_ratio)),
            target_fill_ratio=float(zoom_cfg.get("target_fill_ratio", d.target_fill_ratio)),
            max_optical_magnification=float(
                zoom_cfg.get("max_optical_magnification", d.max_optical_magnification)
            ),
            min_useful_magnification=float(zoom_cfg.get("min_useful_magnification", d.min_useful_magnification)),
            edge_margin_ratio=float(zoom_cfg.get("edge_margin_ratio", d.edge_margin_ratio)),
            wide_zoom_level=float(zoom_cfg.get("wide_zoom_level", d.wide_zoom_level)),
            max_zoom_level=float(zoom_cfg.get("max_zoom_level", d.max_zoom_level)),
        )


def box_fill_ratio(box: tuple[float, float, float, float], frame_shape: tuple[int, int]) -> float:
    """Fraction of the frame the box fills along its larger relative side."""
    h, w = frame_shape
    x1, y1, x2, y2 = box
    return max(max(0.0, x2 - x1) / w, max(0.0, y2 - y1) / h)


def max_in_frame_magnification(
    box: tuple[float, float, float, float], frame_shape: tuple[int, int], edge_margin_ratio: float
) -> float:
    """Largest centre-zoom magnification that keeps every box edge inside
    the frame, with `edge_margin_ratio` of half-frame to spare."""
    h, w = frame_shape
    x1, y1, x2, y2 = box
    # Box edges in normalized half-frame units: -1 = left/top edge, +1 = right/bottom.
    extent = max(
        abs((x1 - w / 2.0) / (w / 2.0)),
        abs((x2 - w / 2.0) / (w / 2.0)),
        abs((y1 - h / 2.0) / (h / 2.0)),
        abs((y2 - h / 2.0) / (h / 2.0)),
    )
    if extent <= 0.0:
        return float("inf")
    return (1.0 - edge_margin_ratio) / extent


def magnification_to_zoom_level(magnification: float, config: ZoomPlanConfig) -> float:
    """Linear approximation of the lens's zoom space. Real motorized
    lenses are not perfectly linear; tune `max_optical_magnification`
    on site if zoomed frames come out tighter or looser than expected."""
    span = config.max_optical_magnification - 1.0
    if span <= 0.0:
        return config.wide_zoom_level
    frac = (min(magnification, config.max_optical_magnification) - 1.0) / span
    return config.wide_zoom_level + (config.max_zoom_level - config.wide_zoom_level) * max(0.0, frac)


def plan_zoom(
    box: tuple[float, float, float, float], frame_shape: tuple[int, int], config: ZoomPlanConfig
) -> float | None:
    """Returns the magnification to zoom to, or None if the animal is
    close enough already or can't be usefully magnified without leaving
    the frame."""
    fill = box_fill_ratio(box, frame_shape)
    if fill <= 0.0 or fill >= config.far_fill_ratio:
        return None

    magnification = min(
        config.target_fill_ratio / fill,
        max_in_frame_magnification(box, frame_shape, config.edge_margin_ratio),
        config.max_optical_magnification,
    )
    if magnification < config.min_useful_magnification:
        return None
    return magnification
