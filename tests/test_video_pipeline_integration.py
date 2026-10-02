"""End-to-end test of the full enhancement chain, on top of the same
real pipeline wiring test_pipeline.py exercises: a consistent tiger
detection sequence confirms a track, dispatches an image alert (batch
1's existing path, unchanged), *and* now also feeds pipeline/ram_buffer.py
+ pipeline/clip_extractor.py, which writes an event MP4 and sets
db.outbox's video_path once it's ready. Finally transfer_main.py's
video-send loop picks that clip up and uploads it — proving the video
half of "Video Upload to Base Station" actually reaches a (fake) base
station starting from real detections, not just from hand-built
outbox rows like tests/test_video_upload.py.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import paths
from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import AnimalZoomController
from db.init_db import init_db
from db.outbox import get_pending, get_pending_videos, set_video_path
from edge_main import _pipeline_loop
from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker
from network.transfer_state import TransferState
from pipeline.alert_dispatcher import AlertDispatcher
from pipeline.best_snapshot import BestSnapshotStore
from pipeline.clip_extractor import ClipExtractor
from pipeline.motion_gate import MotionGate
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackStateMachine
from pipeline.zone_filter import ZoneFilter
from queues.alert_queue import AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from storage.evidence_store import EvidenceStore
from tests.helpers import AlwaysClassifier, blank_frame
from transfer_main import _send_pending_videos


class TestVideoPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_video_pipeline_test_")
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

    def test_confirmed_track_produces_image_alert_and_uploaded_video(self):
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

        clip_ready = threading.Event()

        def on_clip_ready(event_id: str, video_path: str) -> None:
            set_video_path(event_id, video_path)
            clip_ready.set()

        clip_extractor = ClipExtractor(
            config={
                "clip": {
                    "enabled": True,
                    "pre_event_seconds": 1.0,
                    "post_event_seconds": 0.1,  # short so the test doesn't wait long
                    "fps": 5.0,
                    "max_width": None,
                    "codec": "mp4v",
                    "max_concurrent_recordings": 4,
                }
            },
            clip_dir=Path(self.tmpdir) / "clips",
            on_clip_ready=on_clip_ready,
        )

        frame = blank_frame(320, 240)
        box = (100, 80, 180, 160)
        det = Detection(box=box, score=0.9, class_id=5, class_name="tiger")

        now = time.time()
        for i in range(5):
            detection_queue.put(
                DetectionBatch(
                    seq=i, timestamp=now + i * 0.05, frame=frame, detections=[det], camera_id="cam01"
                )
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
            inf_cfg={
                "classifier": {"max_crops_per_track": 5, "min_confidence": 0.5, "voting_strategy": "average"}
            },
            model_version="test-v1",
            stop_event=stop_event,
            max_cycles=5,
            clip_extractor=clip_extractor,
        )

        # 1. Image alert path (batch 1, unaffected by any of this).
        pending = get_pending(limit=10)
        self.assertEqual(len(pending), 1)
        event_id = pending[0]["event_id"]
        self.assertEqual(pending[0]["animal_name"], "tiger")

        # 2. A recording should have started for that event.
        self.assertGreaterEqual(clip_extractor.active_count(), 0)  # may have already finalized

        # Feed a few more frames past the (very short) post_event_seconds
        # deadline so clip_extractor finalizes and fires on_clip_ready.
        for i in range(5, 10):
            clip_extractor.offer_frame(frame, timestamp=now + i * 0.05 + 0.2)

        self.assertTrue(clip_ready.wait(timeout=5.0), "clip was never finalized")

        # 3. video_path should now be set and pending upload.
        pending_videos = get_pending_videos()
        self.assertEqual(len(pending_videos), 1)
        self.assertEqual(pending_videos[0]["event_id"], event_id)
        video_path = Path(pending_videos[0]["video_path"])
        self.assertTrue(video_path.exists())
        self.assertGreater(video_path.stat().st_size, 0)

        # 4. transfer_main's video loop should upload it to a fake client.
        uploaded = []

        class _FakeClient:
            def send_video(self, event_id, video_bytes, timeout_sec=None):
                uploaded.append((event_id, len(video_bytes)))

        _send_pending_videos(_FakeClient(), TransferState(), batch_size=10, timeout_sec=30.0)

        self.assertEqual(len(uploaded), 1)
        self.assertEqual(uploaded[0][0], event_id)
        self.assertEqual(get_pending_videos(), [], "video should be marked sent after upload")


if __name__ == "__main__":
    unittest.main()
