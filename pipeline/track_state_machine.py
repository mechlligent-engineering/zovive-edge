"""Per-track lifecycle, separate from the tracker's own bookkeeping
(inference/bytetrack_wrapper.py just does frame-to-frame association)
and separate from the camera's lens state machine
(camera_control/zoom_controller.py just zooms and focuses).

States:
    NEW        -> just created by the tracker, not yet gate-passed
    CONFIRMING -> being tracked, waiting on stage1_gate's presence vote
    CONFIRMED  -> stage1_gate passed; ready for stage2 classification
    CLASSIFIED -> species verified (or unknown_animal); ready to alert
    ALERTED    -> alert dispatched; track kept alive so it doesn't
                  immediately re-trigger, but no more work happens on it
    LOST       -> tracker dropped it (max_age_frames exceeded)

Pure state transitions, no I/O — edge_main.py's pipeline thread calls
`advance()` once per frame per track and reacts to the returned state.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from inference.bytetrack_wrapper import Track
from pipeline.stage1_gate import Stage1Gate


class TrackState(enum.StrEnum):
    NEW = "NEW"
    CONFIRMING = "CONFIRMING"
    CONFIRMED = "CONFIRMED"
    CLASSIFIED = "CLASSIFIED"
    ALERTED = "ALERTED"
    LOST = "LOST"


@dataclass
class TrackRecord:
    track_id: str
    state: TrackState = TrackState.NEW
    event_id: str | None = None
    species: str | None = None
    confidence: float = 0.0
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)


class TrackStateMachine:
    """Owns one TrackRecord per track_id, advancing it each frame based
    on the current Track (from the tracker) and the Stage1Gate's vote."""

    def __init__(self, gate: Stage1Gate):
        self.gate = gate
        self._records: dict[str, TrackRecord] = {}

    def _get_or_create(self, track_id: str) -> TrackRecord:
        rec = self._records.get(track_id)
        if rec is None:
            rec = TrackRecord(track_id=track_id)
            self._records[track_id] = rec
        return rec

    def advance(self, track: Track, now: float | None = None) -> TrackRecord:
        now = now if now is not None else time.time()
        rec = self._get_or_create(track.track_id)

        if rec.state == TrackState.NEW:
            rec.state = TrackState.CONFIRMING

        if rec.state == TrackState.CONFIRMING:
            if self.gate.evaluate(track, now):
                rec.state = TrackState.CONFIRMED

        rec.updated_ts = now
        return rec

    def mark_classified(self, track_id: str, species: str, confidence: float) -> TrackRecord:
        rec = self._get_or_create(track_id)
        rec.species = species
        rec.confidence = confidence
        rec.state = TrackState.CLASSIFIED
        rec.updated_ts = time.time()
        return rec

    def mark_alerted(self, track_id: str, event_id: str) -> TrackRecord:
        rec = self._get_or_create(track_id)
        rec.event_id = event_id
        rec.state = TrackState.ALERTED
        rec.updated_ts = time.time()
        self.gate.mark_alerted(track_id)
        return rec

    def mark_lost(self, track_id: str) -> None:
        if track_id in self._records:
            self._records[track_id].state = TrackState.LOST
            self._records[track_id].updated_ts = time.time()

    def sweep_lost(self, active_track_ids: set[str]) -> list[str]:
        """Call once per frame with the tracker's currently-active ids.
        Marks anything missing as LOST and returns the ids removed, so
        the caller can evict them from BestSnapshotStore etc."""
        gone = [tid for tid in self._records if tid not in active_track_ids]
        for tid in gone:
            self.mark_lost(tid)
            del self._records[tid]
        return gone

    def get(self, track_id: str) -> TrackRecord | None:
        return self._records.get(track_id)

    def in_state(self, state: TrackState) -> list[TrackRecord]:
        return [r for r in self._records.values() if r.state == state]
