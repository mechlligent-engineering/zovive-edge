"""Animal-aware optical zoom + autofocus for the fixed-direction CP Plus
motorized-lens camera. The camera never pans or tilts; this only
changes the zoom and asks the lens to refocus.

    WIDE       : normal wide view, frames go through the full pipeline.
    ZOOMING    : zoom_to() issued, lens travelling. Frames are skipped for
                 `zoom_settle_sec` (blurred, scene scaling under the tracker).
    FOCUSING   : trigger_autofocus() issued. Frames are skipped for
                 `focus_settle_sec` while the lens hunts for focus.
    ZOOMED     : zoomed + focused. Frames go through the pipeline again, so
                 the animal is re-detected and re-classified at higher
                 resolution. The first event dispatched here completes the
                 session. If nothing confirms within `zoomed_timeout_sec`,
                 tick() hands the session back so the caller can alert with
                 the wide-view evidence instead (an animal is never dropped
                 just because the zoomed re-detection failed).
    RETURNING  : zoom_to(wide) issued, then autofocus again, then WIDE.

After a session, the animal (and any other animal already alerted when
the zoom started) is remembered for `suppress_realert_sec`, so the new
track id it gets in the wide view doesn't cause a second zoom or a
duplicate alert.

Takes its camera and clock as constructor arguments so it can be
unit-tested with FakeLensCamera and a manually advanced fake clock.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from camera_control.camera_api import LensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_planner import ZoomPlanConfig, magnification_to_zoom_level, plan_zoom

log = logging.getLogger(__name__)

WIDE_VIEW = "wide"
ZOOMED_VIEW = "zoomed"


class ZoomState(enum.StrEnum):
    WIDE = "WIDE"
    ZOOMING = "ZOOMING"
    FOCUSING = "FOCUSING"
    ZOOMED = "ZOOMED"
    RETURNING = "RETURNING"


@dataclass
class ZoomSession:
    track_id: str
    class_name: str
    wide_box: tuple[float, float, float, float]
    magnification: float
    started_ts: float
    # Best wide-view material, kept for the fallback alert if the zoomed
    # re-detection fails (the wide track itself is gone by then).
    wide_crops: list = field(default_factory=list)
    wide_frame: np.ndarray | None = None
    event_id: str | None = None
    species: str | None = None
    confidence: float = 0.0


@dataclass
class KnownAnimal:
    box: tuple[float, float, float, float]
    class_name: str
    species: str | None
    confidence: float
    event_id: str | None
    expires_ts: float


def _center_inside(
    box: tuple[float, float, float, float], region: tuple[float, float, float, float], grow: float
) -> bool:
    x1, y1, x2, y2 = region
    gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return (x1 - gx) <= cx <= (x2 + gx) and (y1 - gy) <= cy <= (y2 + gy)


class AnimalZoomController:
    def __init__(
        self,
        camera: LensCamera,
        plan_config: ZoomPlanConfig | None = None,
        mode_manager: ModeManager | None = None,
        enabled: bool = True,
        zoom_settle_sec: float = 2.0,
        focus_settle_sec: float = 1.5,
        zoomed_timeout_sec: float = 10.0,
        suppress_realert_sec: float = 60.0,
        clock: Callable[[], float] = time.time,
    ):
        self.camera = camera
        self.plan_config = plan_config or ZoomPlanConfig()
        self.mode_manager = mode_manager
        self.enabled = enabled
        self.zoom_settle_sec = zoom_settle_sec
        self.focus_settle_sec = focus_settle_sec
        self.zoomed_timeout_sec = zoomed_timeout_sec
        self.suppress_realert_sec = suppress_realert_sec
        self.clock = clock

        self.state = ZoomState.WIDE
        self.session: ZoomSession | None = None
        self._deadline: float | None = None
        self._focus_then: ZoomState = ZoomState.ZOOMED
        self._known: list[KnownAnimal] = []

    @property
    def view_name(self) -> str:
        return ZOOMED_VIEW if self.state == ZoomState.ZOOMED else WIDE_VIEW

    def frames_usable(self) -> bool:
        """False while the lens is moving or focusing."""
        return self.state in (ZoomState.WIDE, ZoomState.ZOOMED)

    def plan(self, box: tuple[float, float, float, float], frame_shape: tuple[int, int]) -> float | None:
        """Magnification to engage this animal with, or None to classify it
        in the wide view as-is."""
        if not self.enabled or self.state != ZoomState.WIDE:
            return None
        if self.mode_manager is not None and not self.mode_manager.is_auto():
            return None
        return plan_zoom(box, frame_shape, self.plan_config)

    def engage(
        self,
        track_id: str,
        class_name: str,
        box: tuple[float, float, float, float],
        magnification: float,
        wide_crops: list | None = None,
        wide_frame: np.ndarray | None = None,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else self.clock()
        level = magnification_to_zoom_level(magnification, self.plan_config)
        self.session = ZoomSession(
            track_id=track_id,
            class_name=class_name,
            wide_box=box,
            magnification=magnification,
            started_ts=now,
            wide_crops=list(wide_crops or []),
            wide_frame=wide_frame,
        )
        self.camera.zoom_to(level)
        self.state = ZoomState.ZOOMING
        self._deadline = now + self.zoom_settle_sec
        log.info(
            "zooming in on far animal",
            extra={
                "track_id": track_id,
                "magnification": round(magnification, 2),
                "zoom_level": round(level, 3),
            },
        )

    def tick(self, now: float | None = None) -> ZoomSession | None:
        """Call once per pipeline cycle. Returns a session only when the
        zoomed re-detection timed out; the caller should then alert using
        `session.wide_crops` / `session.wide_frame` and call `complete()`."""
        now = now if now is not None else self.clock()

        if self.state in (ZoomState.ZOOMING, ZoomState.RETURNING) and now >= self._deadline:
            self._focus_then = ZoomState.ZOOMED if self.state == ZoomState.ZOOMING else ZoomState.WIDE
            self.camera.trigger_autofocus()
            self.state = ZoomState.FOCUSING
            self._deadline = now + self.focus_settle_sec

        elif self.state == ZoomState.FOCUSING and now >= self._deadline:
            self.state = self._focus_then
            if self.state == ZoomState.ZOOMED:
                self._deadline = now + self.zoomed_timeout_sec
                log.info("zoomed and focused; re-detecting", extra={"track_id": self.session.track_id})
            else:
                self._deadline = None
                log.info("back to wide view")

        elif self.state == ZoomState.ZOOMED and now >= self._deadline:
            log.info(
                "no animal confirmed in zoomed view; falling back to wide evidence",
                extra={"track_id": self.session.track_id},
            )
            return self.session

        return None

    def complete(
        self, event_id: str | None, species: str | None, confidence: float, now: float | None = None
    ) -> None:
        """Record the session's result and zoom back out to wide."""
        now = now if now is not None else self.clock()
        if self.session is not None:
            self.session.event_id = event_id
            self.session.species = species
            self.session.confidence = confidence
            self.remember(self.session.wide_box, self.session.class_name, species, confidence, event_id, now)
        self.camera.zoom_to(self.plan_config.wide_zoom_level)
        self.state = ZoomState.RETURNING
        self._deadline = now + self.zoom_settle_sec

    def abort(self, now: float | None = None) -> None:
        """Drop any zoom session without alerting and head back to wide.

        Used when the pipeline thread crashed and is being restarted: the
        session's evidence may be what caused the crash, and the animal is
        normally re-detected once the lens is wide again. Goes through the
        normal RETURNING -> FOCUSING -> WIDE path so frames stay skipped
        until the lens has settled and refocused.
        """
        now = now if now is not None else self.clock()
        dropped = self.session
        self.session = None
        returning_already = self.state == ZoomState.RETURNING or (
            self.state == ZoomState.FOCUSING and self._focus_then == ZoomState.WIDE
        )
        if self.state == ZoomState.WIDE or returning_already:
            return
        self.camera.zoom_to(self.plan_config.wide_zoom_level)
        self.state = ZoomState.RETURNING
        self._deadline = now + self.zoom_settle_sec
        log.warning(
            "zoom session aborted; returning to wide view",
            extra={"track_id": dropped.track_id if dropped else None},
        )

    def remember(
        self,
        box: tuple[float, float, float, float],
        class_name: str,
        species: str | None,
        confidence: float,
        event_id: str | None,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else self.clock()
        self._known.append(
            KnownAnimal(box, class_name, species, confidence, event_id, now + self.suppress_realert_sec)
        )

    def match_known(
        self, class_name: str, box: tuple[float, float, float, float], now: float | None = None
    ) -> KnownAnimal | None:
        """A recently handled wide-view animal this new track is probably
        the same individual as (same class, centre within its old box)."""
        now = now if now is not None else self.clock()
        self._known = [k for k in self._known if k.expires_ts > now]
        for k in self._known:
            if k.class_name == class_name and _center_inside(box, k.box, grow=0.5):
                return k
        return None

    @classmethod
    def from_config(
        cls, camera: LensCamera, camera_modes_cfg: dict, mode_manager: ModeManager | None = None
    ) -> AnimalZoomController:
        z = camera_modes_cfg.get("mode", {}).get("zoom", {})
        return cls(
            camera,
            plan_config=ZoomPlanConfig.from_config(z),
            mode_manager=mode_manager,
            enabled=bool(z.get("enabled", True)),
            zoom_settle_sec=float(z.get("zoom_settle_sec", 2.0)),
            focus_settle_sec=float(z.get("focus_settle_sec", 1.5)),
            zoomed_timeout_sec=float(z.get("zoomed_timeout_sec", 10.0)),
            suppress_realert_sec=float(z.get("suppress_realert_sec", 60.0)),
        )
