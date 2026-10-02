import time
import unittest

from queues.alert_queue import AlertPayload, AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from queues.frame_queue import LatestFrameSlot
from queues.queue_monitor import BoundedDropOldestQueue, snapshot


class TestBoundedDropOldestQueue(unittest.TestCase):
    def test_drops_oldest_when_full(self):
        q = BoundedDropOldestQueue("test_q1", maxsize=2)
        self.assertFalse(q.put("a"))
        self.assertFalse(q.put("b"))
        dropped = q.put("c")  # should drop "a"
        self.assertTrue(dropped)
        self.assertEqual(q.get(timeout=0), "b")
        self.assertEqual(q.get(timeout=0), "c")
        self.assertIsNone(q.get(timeout=0.05))

    def test_get_blocks_until_put(self):
        q = BoundedDropOldestQueue("test_q2", maxsize=2)
        self.assertIsNone(q.get(timeout=0.1))
        q.put(42)
        self.assertEqual(q.get(timeout=0.1), 42)

    def test_stats_recorded_in_registry(self):
        q = BoundedDropOldestQueue("test_q3", maxsize=1)
        q.put(1)
        q.put(2)  # drop
        stats = snapshot()["test_q3"]
        self.assertEqual(stats["put_count"], 2)
        self.assertEqual(stats["drop_count"], 1)


class TestLatestFrameSlot(unittest.TestCase):
    def test_get_latest_returns_newest(self):
        slot = LatestFrameSlot("test_slot1")
        slot.put("frame1")
        slot.put("frame2")
        env = slot.get_latest(timeout=0.1)
        self.assertEqual(env.frame, "frame2")
        self.assertEqual(env.seq, 2)

    def test_wait_for_next_blocks_for_new_frame(self):
        slot = LatestFrameSlot("test_slot2")
        slot.put("frame1")
        first = slot.get_latest()
        self.assertIsNone(slot.wait_for_next(first.seq, timeout=0.1))
        slot.put("frame2")
        env = slot.wait_for_next(first.seq, timeout=0.5)
        self.assertIsNotNone(env)
        self.assertEqual(env.frame, "frame2")


class TestDetectionAndAlertQueue(unittest.TestCase):
    def test_detection_queue_defaults_from_config(self):
        q = DetectionQueue()
        self.assertGreaterEqual(q.maxsize, 1)
        batch = DetectionBatch(seq=1, timestamp=time.time(), frame=None, detections=[])
        q.put(batch)
        self.assertEqual(q.get(timeout=0.1).seq, 1)

    def test_alert_queue_accepts_payload(self):
        q = AlertQueue(maxsize=1)
        payload = AlertPayload(
            event_id="e1",
            track_id="t1",
            camera_id="cam01",
            species="tiger",
            confidence=0.9,
            snapshot_path="snapshots/x.jpg",
            timestamp=time.time(),
        )
        q.put(payload)
        got = q.get(timeout=0.1)
        self.assertEqual(got.species, "tiger")


if __name__ == "__main__":
    unittest.main()
