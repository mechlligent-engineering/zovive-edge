"""pipeline/recent_alerts.py and its use in edge_main's pipeline loop: a
detection that stays in view alerts once, however often the tracker splits
it into new tracks (docs/REALERT_SUPPRESSION_SPEC.md)."""

import itertools
import tempfile
import threading
import unittest
from pathlib import Path

import paths
from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import AnimalZoomController
from db.init_db import init_db
from db.outbox import get_pending
from edge_main import _pipeline_loop
from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker
from pipeline.alert_dispatcher import AlertDispatcher
from pipeline.best_snapshot import BestSnapshotStore
from pipeline.motion_gate import MotionGate
from pipeline.recent_alerts import RecentAlerts, center_inside
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackStateMachine
from pipeline.zone_filter import ZoneFilter
from queues.alert_queue import AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from storage.evidence_store import EvidenceStore
from tests.helpers import blank_frame

BOX = (100, 80, 180, 160)
NEARBY = (110, 85, 190, 165)  # same thing, box jittered
ELSEWHERE = (250, 10, 300, 60)


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestRecentAlerts(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.alerts = RecentAlerts(window_sec=300.0, clock=self.clock)
        self.alerts.remember(BOX, "elephant", "elephant", 0.7, "evt-1")

    def test_same_class_at_the_same_place_matches(self):
        match = self.alerts.match("elephant", NEARBY)
        self.assertIsNotNone(match)
        self.assertEqual(match.event_id, "evt-1")

    def test_other_class_or_other_place_does_not(self):
        self.assertIsNone(self.alerts.match("tiger", BOX))
        self.assertIsNone(self.alerts.match("elephant", ELSEWHERE))

    def test_forgets_after_the_window_without_sightings(self):
        self.clock.now += 300.0
        self.assertIsNone(self.alerts.match("elephant", BOX))
        self.assertEqual(len(self.alerts), 0)

    def test_sightings_keep_the_memory_alive(self):
        for _ in range(10):  # 50 minutes, seen every 5
            self.clock.now += 299.0
            self.assertTrue(self.alerts.seen("elephant", NEARBY))
        self.assertIsNotNone(self.alerts.match("elephant", BOX))

    def test_sightings_elsewhere_or_of_another_class_do_not(self):
        self.clock.now += 200.0
        self.assertFalse(self.alerts.seen("elephant", ELSEWHERE))
        self.assertFalse(self.alerts.seen("tiger", BOX))
        self.clock.now += 100.0
        self.assertIsNone(self.alerts.match("elephant", BOX))

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            RecentAlerts(window_sec=0)

    def test_center_inside_grows_the_region(self):
        self.assertTrue(center_inside((180, 80, 220, 160), BOX, grow=0.5))  # centre x 200, region to 220
        self.assertFalse(center_inside((180, 80, 220, 160), BOX, grow=0.0))

    def test_zoom_controller_shares_an_injected_store(self):
        zoom = AnimalZoomController(FakeLensCamera(), recent_alerts=self.alerts)
        self.assertIs(zoom.recent_alerts, self.alerts)
        self.assertEqual(zoom.match_known("elephant", NEARBY).event_id, "evt-1")
        zoom.remember(ELSEWHERE, "tiger", "tiger", 0.8, "evt-2")
        self.assertIsNotNone(self.alerts.match("tiger", ELSEWHERE))


class TestStaticDetectionInThePipeline(unittest.TestCase):
    """The overnight-soak case: one object, never moving, whose detections
    drop out for longer than max_age_frames now and then, so the tracker
    gives it a new track id each time."""

    FRAGMENTS = 6

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="zovive_recent_alerts_"))
        self._orig = (paths.DB_PATH, paths.SNAPSHOT_DIR)
        paths.DB_PATH = tmp / "test.sqlite3"
        paths.SNAPSHOT_DIR = tmp / "snapshots"
        paths.SNAPSHOT_DIR.mkdir()
        init_db()

    def tearDown(self):
        paths.DB_PATH, paths.SNAPSHOT_DIR = self._orig

    def _run(self, frames: list[list[Detection]], recent_alerts: RecentAlerts) -> list[dict]:
        queue = DetectionQueue(maxsize=len(frames))
        frame = blank_frame(320, 240)
        for i, dets in enumerate(frames):
            queue.put(
                DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=dets, camera_id="cam01")
            )
        dispatcher = AlertDispatcher(
            AlertQueue(),
            evidence_store=EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR),
            node_config={"node": {"camera_id": "cam01"}},
            alert_policy={"alert_policy": {"send_unknown_animal_alerts": False}},
        )
        _pipeline_loop(
            queue,
            AnimalZoomController(FakeLensCamera(), mode_manager=ModeManager()),
            IouTracker(iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5),
            TrackStateMachine(
                Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5, track_alert_cooldown_sec=300.0))
            ),
            ZoneFilter({"zones": {}}),
            MotionGate(),
            BestSnapshotStore(),
            dispatcher,
            inf_cfg={"stage2": {"min_confidence": 0.5, "voting_strategy": "average"}},
            model_version="test-v1",
            stop_event=threading.Event(),
            max_cycles=len(frames),
            recent_alerts=recent_alerts,
        )
        return get_pending(limit=50)

    def _fragmented(self, *detections: Detection) -> list[list[Detection]]:
        # 10 frames seen, then 20 unseen (> max_age_frames 15): a new track each time.
        return ([list(detections)] * 10 + [[]] * 20) * self.FRAGMENTS

    def test_one_alert_however_often_the_track_breaks(self):
        # A large box at the frame edge, like the soak's false positive: no zoom plan.
        fabric = Detection(box=(0, 20, 120, 230), score=0.7, class_id=3, class_name="elephant")
        events = self._run(self._fragmented(fabric), RecentAlerts())
        self.assertEqual([e["animal_name"] for e in events], ["elephant"])

    def test_without_the_memory_every_fragment_alerts(self):
        # What the soak saw: a memory that forgets at once is the old behaviour.
        # Its clock moves a second per read, so nothing outlives the read that stored it.
        fabric = Detection(box=(0, 20, 120, 230), score=0.7, class_id=3, class_name="elephant")
        ticks = itertools.count()
        forgetful = RecentAlerts(window_sec=1e-9, clock=lambda: float(next(ticks)))
        events = self._run(self._fragmented(fabric), forgetful)
        self.assertEqual(len(events), self.FRAGMENTS)

    def test_a_different_species_at_the_same_place_still_alerts(self):
        elephant = Detection(box=(0, 20, 120, 230), score=0.7, class_id=3, class_name="elephant")
        tiger = Detection(box=(0, 20, 120, 230), score=0.9, class_id=5, class_name="tiger")
        frames = [[elephant]] * 10 + [[]] * 20 + [[tiger]] * 10
        events = self._run(frames, RecentAlerts())
        self.assertEqual(sorted(e["animal_name"] for e in events), ["elephant", "tiger"])


if __name__ == "__main__":
    unittest.main()
