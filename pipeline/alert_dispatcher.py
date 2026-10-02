"""Turns a CLASSIFIED track into a durable event: writes it to the
SQLite outbox FIRST (so it's never lost even if everything after this
crashes), saves the snapshot, then hands a lightweight payload to the
in-memory alert_queue for transfer_main.py to pick up promptly.

This ordering (DB row before in-memory queue push) is what makes the
alert_queue drop-oldest policy (queues/alert_queue.py) safe: a drop
just means the sender learns about it a little later, by polling
db.outbox.get_pending(), rather than losing it.
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
        if species == "unknown_animal" and not self.policy.get("send_unknown_animal_alerts", True):
            log.info("unknown_animal alert suppressed by policy", extra={"track_id": track_id})
            return None

        # event_id is created here so the same id is used for the DB row,
        # the snapshot filename, and the queued payload.
        record = EventRecord(
            track_id=track_id,
            camera_id=self.node_cfg.get("camera_id", "unknown"),
            camera_name=self.node_cfg.get("camera_name", ""),
            forest_name=self.node_cfg.get("forest_name", ""),
            gps_lat=(self.node_cfg.get("gps", {}) or {}).get("lat"),
            gps_lon=(self.node_cfg.get("gps", {}) or {}).get("lon"),
            animal_name=species,
            confidence=confidence,
            snapshot_path="",  # filled in after save below
            model_version=model_version,
            preset_name=preset_name,
            priority=self._priority_for(species),
        )

        snapshot_path = self.evidence_store.save_snapshot(
            snapshot_image,
            camera_id=record.camera_id,
            track_id=track_id,
            event_id=record.event_id,
            ts=record.created_ts,
        )
        record.snapshot_path = str(snapshot_path)

        insert_event(record)

        self.alert_queue.put(
            AlertPayload(
                event_id=record.event_id,
                track_id=track_id,
                camera_id=record.camera_id,
                species=species,
                confidence=confidence,
                snapshot_path=str(snapshot_path),
                timestamp=record.created_ts,
                priority=record.priority,
            )
        )

        log.info(
            "event dispatched",
            extra={
                "event_id": record.event_id,
                "track_id": track_id,
                "species": species,
                "confidence": round(confidence, 3),
                "priority": record.priority,
            },
        )
        return record.event_id
