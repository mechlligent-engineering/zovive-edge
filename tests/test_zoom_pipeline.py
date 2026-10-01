"""Far animal -> zoom in -> autofocus -> re-detect/classify -> one alert.

Runs edge_main's real pipeline loop with FakeLensCamera and zero settle
times, so each lens phase takes exactly one pipeline cycle:

    cycle 0-2  wide view, far tiger confirms -> zoom in
    cycle 3    ZOOMING -> FOCUSING (frame skipped)
    cycle 4-6  ZOOMED, tiger re-detected large -> alert, zoom out
    cycle 7    RETURNING -> FOCUSING (frame skipped)
    cycle 8-10 wide view again, same tiger, new track id -> NOT re-alerted
"""

import tempfile
import threading
import unittest
from pathlib import Path

import paths
from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import AnimalZoomController, ZoomState
from db.init_db import init_db
from db.outbox import get_pending
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

FAR_BOX = (300, 170, 340, 190)  # small, centred in a 640x360 frame
ZOOMED_BOX = (160, 60, 480, 300)


def _det(box):
    return Detection(box=box, score=0.9, class_id=5, class_name="tiger")


class TestZoomPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_zoom_test_")
        self._orig_db_path = paths.DB_PATH
        self._orig_snapshot_dir = paths.SNAPSHOT_DIR
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        paths.SNAPSHOT_DIR = Path(self.tmpdir) / "snapshots"
        paths.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        init_db()

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path
        paths.SNAPSHOT_DIR = self._orig_snapshot_dir

    def _run(self, per_cycle_detections, zoomed_timeout_sec=30.0):
        self.camera = FakeLensCamera()
        self.zoom = AnimalZoomController(
            self.camera, mode_manager=ModeManager(), zoom_settle_sec=0.0, focus_settle_sec=0.0,
            zoomed_timeout_sec=zoomed_timeout_sec,
        )
        queue = DetectionQueue(maxsize=50)
        frame = blank_frame(640, 360)
        for i, dets in enumerate(per_cycle_detections):
            queue.put(DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=dets, camera_id="cam01"))

        dispatcher = AlertDispatcher(
            AlertQueue(),
            evidence_store=EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR),
            node_config={"node": {"camera_id": "cam01"}},
            alert_policy={},
        )
        _pipeline_loop(
            queue,
            AlwaysClassifier("tiger", 0.92),
            self.zoom,
            IouTracker(iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5),
            TrackStateMachine(Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))),
            ZoneFilter({"zones": {}}),
            MotionGate(),
            BestSnapshotStore(),
            dispatcher,
            inf_cfg={"classifier": {"max_crops_per_track": 5, "min_confidence": 0.5}},
            model_version="test-v1",
            stop_event=threading.Event(),
            max_cycles=len(per_cycle_detections),
        )
        return get_pending(limit=10)

    def test_far_animal_is_zoomed_refocused_and_alerted_once(self):
        far, zoomed = [_det(FAR_BOX)], [_det(ZOOMED_BOX)]
        pending = self._run([far] * 3 + [far] + [zoomed] * 3 + [zoomed] + [far] * 3)

        self.assertEqual(len(pending), 1, pending)
        self.assertEqual(pending[0]["animal_name"], "tiger")
        self.assertEqual(pending[0]["preset_name"], "zoomed")
        self.assertTrue(Path(pending[0]["snapshot_path"]).exists())

        calls = [name for name, _ in self.camera.call_log]
        self.assertEqual(calls, ["zoom_to", "trigger_autofocus", "zoom_to", "trigger_autofocus"])
        self.assertGreater(self.camera.call_log[0][1]["level"], 0.0)
        self.assertEqual(self.camera.get_zoom(), 0.0)
        self.assertEqual(self.zoom.state, ZoomState.WIDE)

    def test_failed_zoomed_redetection_alerts_with_wide_evidence(self):
        far = [_det(FAR_BOX)]
        # Zoomed view sees nothing; timeout 0 -> fallback on the next cycle.
        pending = self._run([far] * 3 + [far] + [[]] * 3, zoomed_timeout_sec=0.0)

        self.assertEqual(len(pending), 1, pending)
        self.assertEqual(pending[0]["preset_name"], "wide")
        self.assertTrue(Path(pending[0]["snapshot_path"]).exists())

    def test_close_animal_is_alerted_without_zoom(self):
        close = [_det((100, 60, 400, 300))]
        pending = self._run([close] * 3)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["preset_name"], "wide")
        self.assertEqual(self.camera.call_log, [])


if __name__ == "__main__":
    unittest.main()
