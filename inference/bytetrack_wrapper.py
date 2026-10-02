"""Multi-object tracker that assigns a stable track_id to detections
across frames, so one animal produces one event instead of a new
"detection" every frame.

Named bytetrack_wrapper.py in the skeleton; what's implemented here is
a dependency-free greedy IoU tracker (match each detection to the
track with highest IoU above threshold, in descending IoU order),
NOT the full ByteTrack algorithm (which also uses low-confidence
detections in a second association pass and needs the `motmetrics`/
`lap` stack). This is intentional for the first pass: it needs no
extra pip packages, and IoU-only tracking is enough for a slow-moving
or stationary camera with a handful of animals per frame. If track
identity proves unstable in the field (fast-moving animals, frequent
occlusion), swap this module for real ByteTrack — pipeline code only
depends on the Track dataclass shape below, not on how it was produced.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from inference.base import Detection


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: str
    box: tuple[float, float, float, float]
    class_id: int
    class_name: str
    score: float
    hits: int = 1
    age_since_seen: int = 0
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    # last N (is_animal_present) votes for stage1_gate's 3-of-5 rule
    presence_history: list[bool] = field(default_factory=list)
    confirmed: bool = False


class IouTracker:
    def __init__(
        self,
        iou_match_threshold: float = 0.3,
        max_age_frames: int = 15,
        min_hits_to_confirm: int = 2,
        presence_window: int = 5,
    ):
        self.iou_match_threshold = iou_match_threshold
        self.max_age_frames = max_age_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.presence_window = presence_window
        self._tracks: dict[str, Track] = {}
        self._id_counter = itertools.count(1)

    def _new_id(self) -> str:
        return f"t-{next(self._id_counter)}"

    def update(self, detections: list[Detection]) -> list[Track]:
        now = time.time()

        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[str] = set()

        # Greedy matching: consider all (track, detection) pairs above
        # threshold, sorted by IoU descending, claim the best first.
        candidates = []
        for tid, track in self._tracks.items():
            for di in unmatched_dets:
                score = iou(track.box, detections[di].box)
                if score >= self.iou_match_threshold:
                    candidates.append((score, tid, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        used_dets: set[int] = set()
        for _score, tid, di in candidates:
            if tid in matched_track_ids or di in used_dets:
                continue
            det = detections[di]
            track = self._tracks[tid]
            track.box = det.box
            track.class_id = det.class_id
            track.class_name = det.class_name
            track.score = det.score
            track.hits += 1
            track.age_since_seen = 0
            track.last_seen_ts = now
            track.presence_history.append(True)
            track.presence_history = track.presence_history[-self.presence_window :]
            if track.hits >= self.min_hits_to_confirm:
                track.confirmed = True
            matched_track_ids.add(tid)
            used_dets.add(di)

        # Unmatched detections become new tracks.
        newly_created_ids: set[str] = set()
        for di in unmatched_dets:
            if di in used_dets:
                continue
            det = detections[di]
            tid = self._new_id()
            self._tracks[tid] = Track(
                track_id=tid,
                box=det.box,
                class_id=det.class_id,
                class_name=det.class_name,
                score=det.score,
                presence_history=[True],
            )
            newly_created_ids.add(tid)

        # Unmatched existing tracks age; drop ones past max_age_frames.
        # Tracks created THIS cycle (newly_created_ids) are excluded here —
        # without this, a brand-new track would be aged (an extra `False`
        # appended to presence_history, age_since_seen incremented) in the
        # very same update() call that created it.
        dead_ids = []
        for tid, track in self._tracks.items():
            if tid in matched_track_ids or tid in newly_created_ids:
                continue
            track.age_since_seen += 1
            track.presence_history.append(False)
            track.presence_history = track.presence_history[-self.presence_window :]
            if track.age_since_seen > self.max_age_frames:
                dead_ids.append(tid)
        for tid in dead_ids:
            del self._tracks[tid]

        return list(self._tracks.values())

    def reset(self) -> None:
        """Drop every track, e.g. when the camera zooms and every box in
        the old view is meaningless in the new one. Ids keep counting up
        so a new track never reuses an old id."""
        self._tracks.clear()

    def get_track(self, track_id: str) -> Track | None:
        return self._tracks.get(track_id)

    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    @classmethod
    def from_config(cls, cfg: dict, presence_window: int = 5) -> IouTracker:
        t = cfg.get("tracker", {})
        return cls(
            iou_match_threshold=float(t.get("iou_match_threshold", 0.3)),
            max_age_frames=int(t.get("max_age_frames", 15)),
            min_hits_to_confirm=int(t.get("min_hits_to_confirm", 2)),
            presence_window=presence_window,
        )
