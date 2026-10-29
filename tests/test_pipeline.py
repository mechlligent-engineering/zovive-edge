"""End-to-end integration test of edge_main's pipeline loop: feeds a
sequence of detection batches for the same animal through the real
tracker, stage1 gate, track state machine, stage2 species vote (over the
detector's own per-frame classes; there is no classifier model),
dispatcher and SQLite outbox — exactly the code edge_main.py wires
together, just without real capture/inference threads or a real
camera/model.
"""

import errno
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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
from storage.evidence_store import EvidenceStore, SnapshotWriteError
from tests.helpers import blank_frame

STAGE2_CFG = {"stage2": {"min_confidence": 0.5, "voting_strategy": "average"}}


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

    def _run_frames(self, per_frame_detections: list[list[Detection]], send_unknown: bool = True) -> None:
        """One tracker/gate/dispatcher stack, one batch per entry, real SQLite
        outbox. The alert policy is the test's own, not configs/alert_policy.yaml."""
        detection_queue = DetectionQueue(maxsize=20)
        tracker = IouTracker(
            iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5
        )
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5, track_alert_cooldown_sec=300.0))
        node_cfg = {
            "node": {"camera_id": "cam01", "camera_name": "Test Cam", "forest_name": "Test Forest", "gps": {}}
        }
        evidence_store = EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR)
        dispatcher = AlertDispatcher(
            AlertQueue(),
            evidence_store=evidence_store,
            node_config=node_cfg,
            alert_policy={"alert_policy": {"send_unknown_animal_alerts": send_unknown}},
        )
        frame = blank_frame(320, 240)
        for i, detections in enumerate(per_frame_detections):
            detection_queue.put(
                DetectionBatch(
                    seq=i, timestamp=float(i), frame=frame, detections=detections, camera_id="cam01"
                )
            )
        _pipeline_loop(
            detection_queue,
            self._build_zoom(),
            tracker,
            TrackStateMachine(gate),
            ZoneFilter({"zones": {}}),
            MotionGate(),
            BestSnapshotStore(),
            dispatcher,
            inf_cfg=STAGE2_CFG,
            model_version="test-v1",
            stop_event=threading.Event(),
            max_cycles=len(per_frame_detections),
        )

    def test_consistent_tiger_detections_produce_one_event(self):
        # 5 consecutive frames with the same tiger box -> tracker keeps one
        # track, presence vote reaches 3-of-5, confirms and produces exactly
        # one event whose species and confidence are the detector's own.
        det = Detection(box=(100, 80, 180, 160), score=0.9, class_id=5, class_name="tiger")
        self._run_frames([[det]] * 5)

        pending = get_pending(limit=10)
        self.assertEqual(len(pending), 1, f"expected exactly one event, got {pending}")
        self.assertEqual(pending[0]["animal_name"], "tiger")
        self.assertAlmostEqual(pending[0]["confidence"], 0.9, places=6)
        self.assertTrue(Path(pending[0]["snapshot_path"]).exists())
        self.assertEqual(count_by_status().get("pending"), 1)

    def test_species_is_voted_across_frames_not_taken_from_the_last_one(self):
        # The track confirms on frame 3, whose detection says leopard; the
        # two earlier frames said tiger. "average": tiger 1.8/3 = 0.6 wins.
        box = (100, 80, 180, 160)
        tiger = Detection(box=box, score=0.9, class_id=5, class_name="tiger")
        leopard = Detection(box=box, score=0.9, class_id=2, class_name="leopard")
        self._run_frames([[tiger], [tiger], [leopard]])

        pending = get_pending(limit=10)
        self.assertEqual([e["animal_name"] for e in pending], ["tiger"])
        self.assertAlmostEqual(pending[0]["confidence"], 0.6, places=6)

    def test_ambiguous_track_is_reported_as_unknown_animal(self):
        # Votes split tiger/leopard/elephant: the winner's averaged score is
        # 0.3, below stage2.min_confidence 0.5 -> unknown_animal, not a guess.
        box = (100, 80, 180, 160)
        frames = [
            [Detection(box=box, score=0.9, class_id=5, class_name="tiger")],
            [Detection(box=box, score=0.9, class_id=2, class_name="leopard")],
            [Detection(box=box, score=0.9, class_id=3, class_name="elephant")],
        ]
        self._run_frames(frames)

        pending = get_pending(limit=10)
        self.assertEqual([e["animal_name"] for e in pending], ["unknown_animal"])

    def test_ambiguous_track_produces_no_event_when_unknown_alerts_are_off(self):
        # The deployed policy (alert_policy.yaml send_unknown_animal_alerts:
        # false): the same ambiguous track is dropped, not sent as unknown.
        box = (100, 80, 180, 160)
        frames = [
            [Detection(box=box, score=0.9, class_id=5, class_name="tiger")],
            [Detection(box=box, score=0.9, class_id=2, class_name="leopard")],
            [Detection(box=box, score=0.9, class_id=3, class_name="elephant")],
        ]
        self._run_frames(frames, send_unknown=False)
        self.assertEqual(get_pending(limit=10), [])

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
        zoom = self._build_zoom()
        frame = blank_frame(320, 240)

        for i in range(5):
            detection_queue.put(
                DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=[], camera_id="cam01")
            )

        _pipeline_loop(
            detection_queue,
            zoom,
            tracker,
            track_sm,
            zone_filter,
            motion_gate,
            snapshot_store,
            dispatcher,
            inf_cfg=STAGE2_CFG,
            model_version="test-v1",
            stop_event=threading.Event(),
            max_cycles=5,
        )

        self.assertEqual(get_pending(limit=10), [])

    def test_failed_snapshot_save_records_no_event_and_queues_nothing(self):
        # The real dispatcher and store: the snapshot is saved before the
        # outbox row, so a failed save leaves no row claiming evidence exists.
        real_link = os.link

        def another_writer_first(src, dst):
            Path(dst).write_bytes(b"other writer")
            return real_link(src, dst)

        failures = {
            "another writer took the final name": mock.patch("os.link", side_effect=another_writer_first),
            "disk I/O error": mock.patch("os.fsync", side_effect=OSError(errno.EIO, "Input/output error")),
        }
        for n, (name, failure) in enumerate(failures.items()):
            with self.subTest(name):
                alert_queue = AlertQueue()
                dispatcher = AlertDispatcher(
                    alert_queue,
                    evidence_store=EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR),
                    node_config={"node": {"camera_id": "cam01"}},
                    alert_policy={},
                )
                with failure, self.assertRaises(SnapshotWriteError) as cm:
                    dispatcher.dispatch(f"track-{n}", "tiger", 0.92, blank_frame())
                self.assertIsInstance(cm.exception, OSError, "what the pipeline supervisor restarts on")
                self.assertEqual(get_connection().execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
                self.assertEqual(alert_queue.qsize(), 0)

        # Only the other writer's file is left, unchanged; no temp files.
        self.assertEqual([p.read_bytes() for p in paths.SNAPSHOT_DIR.iterdir()], [b"other writer"])


if __name__ == "__main__":
    unittest.main()
