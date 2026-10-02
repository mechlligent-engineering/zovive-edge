"""End-to-end integration test of edge_main's pipeline loop: feeds a
sequence of detection batches for the same animal through the real
tracker, stage1 gate, track state machine, stage2 classifier,
dispatcher and SQLite outbox — exactly the code edge_main.py wires
together, just without real capture/inference threads or a real
camera/model.
"""

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import paths
from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import AnimalZoomController
from db.init_db import get_connection, init_db
from db.outbox import count_by_status, get_pending
from edge_main import _pipeline_loop
from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker
from pipeline.alert_dispatcher import AlertDispatcher
from pipeline.best_snapshot import BestSnapshotStore
from pipeline.motion_gate import MotionGate
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackStateMachine
from pipeline.zone_filter import ZoneFilter
from queues.alert_queue import AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from storage.evidence_store import EvidenceStore
from tests.helpers import AlwaysClassifier, blank_frame


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_pipeline_test_")
        self._orig_db_path = paths.DB_PATH
        self._orig_snapshot_dir = paths.SNAPSHOT_DIR
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        paths.SNAPSHOT_DIR = Path(self.tmpdir) / "snapshots"
        paths.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        init_db()

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path
        paths.SNAPSHOT_DIR = self._orig_snapshot_dir

    def _build_zoom(self):
        return AnimalZoomController(FakeLensCamera(), mode_manager=ModeManager())

    def test_consistent_tiger_detections_produce_one_event(self):
        detection_queue = DetectionQueue(maxsize=20)
        tracker = IouTracker(
            iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5
        )
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5, track_alert_cooldown_sec=300.0))
        track_sm = TrackStateMachine(gate)
        zone_filter = ZoneFilter({"zones": {}})
        motion_gate = MotionGate()
        snapshot_store = BestSnapshotStore()
        evidence_store = EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR)
        alert_queue = AlertQueue()
        node_cfg = {
            "node": {"camera_id": "cam01", "camera_name": "Test Cam", "forest_name": "Test Forest", "gps": {}}
        }
        dispatcher = AlertDispatcher(alert_queue, evidence_store=evidence_store, node_config=node_cfg)
        classifier = AlwaysClassifier("tiger", 0.92)
        zoom = self._build_zoom()

        frame = blank_frame(320, 240)
        box = (100, 80, 180, 160)
        det = Detection(box=box, score=0.9, class_id=5, class_name="tiger")

        # 5 consecutive frames with the same tiger box -> tracker keeps one
        # track, presence vote reaches 5/5 (>= 3-of-5), should confirm and
        # produce exactly one dispatched event.
        for i in range(5):
            detection_queue.put(
                DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=[det], camera_id="cam01")
            )

        stop_event = threading.Event()
        _pipeline_loop(
            detection_queue,
            classifier,
            zoom,
            tracker,
            track_sm,
            zone_filter,
            motion_gate,
            snapshot_store,
            dispatcher,
            inf_cfg={"classifier": {"max_crops_per_track": 5, "min_confidence": 0.5}},
            model_version="test-v1",
            stop_event=stop_event,
            max_cycles=5,
        )

        pending = get_pending(limit=10)
        self.assertEqual(len(pending), 1, f"expected exactly one event, got {pending}")
        self.assertEqual(pending[0]["animal_name"], "tiger")
        self.assertTrue(Path(pending[0]["snapshot_path"]).exists())
        self.assertEqual(count_by_status().get("pending"), 1)

    def test_no_detections_produce_no_events(self):
        detection_queue = DetectionQueue(maxsize=20)
        tracker = IouTracker(presence_window=5)
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))
        track_sm = TrackStateMachine(gate)
        zone_filter = ZoneFilter({"zones": {}})
        motion_gate = MotionGate()
        snapshot_store = BestSnapshotStore()
        evidence_store = EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR)
        dispatcher = AlertDispatcher(
            AlertQueue(), evidence_store=evidence_store, node_config={"node": {"camera_id": "cam01"}}
        )
        classifier = AlwaysClassifier("tiger", 0.92)
        zoom = self._build_zoom()
        frame = blank_frame(320, 240)

        for i in range(5):
            detection_queue.put(
                DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=[], camera_id="cam01")
            )

        _pipeline_loop(
            detection_queue,
            classifier,
            zoom,
            tracker,
            track_sm,
            zone_filter,
            motion_gate,
            snapshot_store,
            dispatcher,
            inf_cfg={"classifier": {"max_crops_per_track": 5, "min_confidence": 0.5}},
            model_version="test-v1",
            stop_event=threading.Event(),
            max_cycles=5,
        )

        self.assertEqual(get_pending(limit=10), [])


if __name__ == "__main__":
    unittest.main()
