"""Turns a CLASSIFIED track into a durable event: saves the snapshot
FIRST (storage/evidence_store.py), then writes the event to the SQLite
outbox with the snapshot's final path, then hands a lightweight payload
to the in-memory alert_queue for transfer_main.py to pick up promptly.

Snapshot before DB row: an outbox row never claims evidence that does
not exist. If the snapshot cannot be saved, dispatch() raises with no
row written and nothing queued (edge_main's supervisor restarts the
pipeline thread). A crash between the save and the insert leaves at
most a snapshot no row refers to, never a row without its snapshot.

DB row before in-memory queue push: this is what makes the alert_queue
drop-oldest policy (queues/alert_queue.py) safe: a drop just means the
sender learns about it a little later, by polling
db.outbox.get_pending(), rather than losing it.

dispatch() runs the four steps in one call: prepare() (policy, event_id,
priority), save_snapshot(), insert(), announce(). edge_main runs the
three disk steps on a separate thread instead, through
pipeline/dispatch_worker.py, in the same order.
"""

from __future__ import annotations

import logging

import numpy as np

import paths
from db.outbox import EventRecord, insert_event
from queues.alert_queue import AlertPayload, AlertQueue
from storage.evidence_store import EvidenceStore
from utils.config_loader import load

log = logging.getLogger(__name__)


class AlertDispatcher:
    def __init__(
        self,
        alert_queue: AlertQueue,
        evidence_store: EvidenceStore | None = None,
        node_config: dict | None = None,
        alert_policy: dict | None = None,
    ):
        self.alert_queue = alert_queue
        self.evidence_store = evidence_store or EvidenceStore()
        self.node_cfg = (node_config or load(paths.NODE_CONFIG)).get("node", {})
        self.policy = (alert_policy or load(paths.ALERT_POLICY_CONFIG)).get("alert_policy", {})

    def _priority_for(self, species: str) -> str:
        high = set(self.policy.get("high_priority_species", []))
        return "high" if species in high else "normal"

    def dispatch(
        self,
        track_id: str,
        species: str,
        confidence: float,
        snapshot_image: np.ndarray,
        preset_name: str = "",
        model_version: str = "",
    ) -> str | None:
        """Returns the event_id, or None if this alert was suppressed by policy."""
        record = self.prepare(track_id, species, confidence, preset_name, model_version)
        if record is None:
            return None
        self.save_snapshot(record, snapshot_image)
        self.insert(record)
        self.announce(record)
        return record.event_id

    def prepare(
        self, track_id: str, species: str, confidence: float, preset_name: str = "", model_version: str = ""
    ) -> EventRecord | None:
        """The event to record, or None if this alert is suppressed by policy.
        No I/O. event_id is created here so the same id is used for the DB
        row, the snapshot filename, the queued payload and the event clip."""
        if species == "unknown_animal" and not self.policy.get("send_unknown_animal_alerts", True):
            log.info("unknown_animal alert suppressed by policy", extra={"track_id": track_id})
            return None
        return EventRecord(
            track_id=track_id,
            camera_id=self.node_cfg.get("camera_id", "unknown"),
            camera_name=self.node_cfg.get("camera_name", ""),
            forest_name=self.node_cfg.get("forest_name", ""),
            gps_lat=(self.node_cfg.get("gps", {}) or {}).get("lat"),
            gps_lon=(self.node_cfg.get("gps", {}) or {}).get("lon"),
            animal_name=species,
            confidence=confidence,
            snapshot_path="",  # set by save_snapshot()
            model_version=model_version,
            preset_name=preset_name,
            priority=self._priority_for(species),
        )

    def save_snapshot(self, record: EventRecord, snapshot_image: np.ndarray) -> str:
        """Step 1: save the snapshot and set record.snapshot_path. Raises
        (storage/evidence_store.py SnapshotWriteError) with no file left if
        it cannot be saved; never overwrites an existing snapshot."""
        snapshot_path = self.evidence_store.save_snapshot(
            snapshot_image,
            camera_id=record.camera_id,
            track_id=record.track_id,
            event_id=record.event_id,
            ts=record.created_ts,
        )
        record.snapshot_path = str(snapshot_path)
        return record.snapshot_path

    def insert(self, record: EventRecord) -> None:
        """Step 2: the outbox row, which points at the saved snapshot."""
        insert_event(record)

    def announce(self, record: EventRecord) -> None:
        """Step 3: hand the event to transfer_main.py via the alert_queue."""
        self.alert_queue.put(
            AlertPayload(
                event_id=record.event_id,
                track_id=record.track_id,
                camera_id=record.camera_id,
                species=record.animal_name,
                confidence=record.confidence,
                snapshot_path=record.snapshot_path,
                timestamp=record.created_ts,
                priority=record.priority,
            )
        )
        log.info(
            "event dispatched",
            extra={
                "event_id": record.event_id,
                "track_id": record.track_id,
                "species": record.animal_name,
                "confidence": round(record.confidence, 3),
                "priority": record.priority,
            },
        )
