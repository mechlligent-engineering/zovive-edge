"""StreamHealth.snapshot() must not deadlock: it holds the lock while it
calls fps() and is_stalled(), which take the same lock. edge_main writes
the snapshot into the edge status on the first supervisor loop, so a
hang there freezes the service until systemd's watchdog kills it."""

import threading
import unittest
from unittest.mock import patch

from capture import stream_health
from capture.stream_health import StreamHealth

# Generous: every call here takes microseconds, so only a deadlock can
# hit this. The worker threads are daemons so a hung one can't block exit.
_TIMEOUT_SEC = 2.0

_SNAPSHOT_FIELDS = {
    "camera_id",
    "frames_received",
    "reconnect_count",
    "stalled_count",
    "last_frame_ts",
    "last_reconnect_ts",
    "fps",
    "is_stalled",
    "uptime_sec",
}


class _Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _call_with_timeout(fn):
    """Return (finished, result) for fn() run on a daemon thread."""
    out = {}

    def target():
        out["result"] = fn()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(_TIMEOUT_SEC)
    return not t.is_alive(), out.get("result")


class _HealthTestCase(unittest.TestCase):
    def snapshot(self, health: StreamHealth) -> dict:
        # Never call snapshot() on the test thread: a deadlock there would
        # hang the whole suite instead of failing this test.
        finished, snap = _call_with_timeout(health.snapshot)
        self.assertTrue(finished, "snapshot() deadlocked")
        return snap


class TestSnapshotDoesNotDeadlock(_HealthTestCase):
    def test_snapshot_returns_before_any_frame(self):
        snap = self.snapshot(StreamHealth("cam01"))
        self.assertEqual(set(snap), _SNAPSHOT_FIELDS)

    def test_snapshot_returns_after_frames(self):
        health = StreamHealth("cam01")
        for _ in range(5):
            health.record_frame()
        snap = self.snapshot(health)
        self.assertEqual(set(snap), _SNAPSHOT_FIELDS)
        self.assertEqual(snap["frames_received"], 5)

    def test_lock_released_after_snapshot(self):
        # A nested acquire that isn't fully released would block the
        # capture thread's next record_frame() forever.
        health = StreamHealth("cam01")
        self.snapshot(health)
        finished, _ = _call_with_timeout(health.record_frame)
        self.assertTrue(finished, "lock still held after snapshot()")
        self.assertEqual(self.snapshot(health)["frames_received"], 1)


class TestSnapshotValues(_HealthTestCase):
    def setUp(self):
        self.clock = _Clock()
        patcher = patch.object(stream_health.time, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.health = StreamHealth("cam01", stall_timeout_sec=5.0)
        # _Counters' default_factory bound the real time.time at import,
        # so start the fake clock at the stream's real start time.
        self.start = self.clock.t = self.health._c.stream_started_ts

    def _frames(self, n: int, interval: float) -> None:
        for _ in range(n):
            self.clock.t += interval
            self.health.record_frame()

    def test_matches_individual_accessors(self):
        self._frames(4, 0.1)  # 10 fps
        last_ts = self.clock.t
        self.health.record_reconnect()
        self.health.record_stall()
        self.clock.t = self.start + 1.0

        snap = self.snapshot(self.health)

        self.assertEqual(
            snap,
            {
                "camera_id": "cam01",
                "frames_received": 4,
                "reconnect_count": 1,
                "stalled_count": 1,
                "last_frame_ts": last_ts,
                "last_reconnect_ts": last_ts,
                "fps": 10.0,
                "is_stalled": False,
                "uptime_sec": 1.0,
            },
        )
        self.assertEqual(snap["fps"], round(self.health.fps(), 2))
        self.assertEqual(snap["is_stalled"], self.health.is_stalled())

    def test_reports_stall_after_timeout(self):
        self._frames(3, 0.1)
        self.clock.t += 5.5
        self.assertTrue(self.snapshot(self.health)["is_stalled"])
        self.assertTrue(self.health.is_stalled())

    def test_fresh_stream_not_stalled_until_timeout(self):
        self.assertFalse(self.snapshot(self.health)["is_stalled"])
        self.clock.t += 5.5
        snap = self.snapshot(self.health)
        self.assertTrue(snap["is_stalled"])
        self.assertEqual(snap["fps"], 0.0)
        self.assertIsNone(snap["last_frame_ts"])


class TestConcurrentAccess(_HealthTestCase):
    def test_snapshot_while_capture_thread_records(self):
        health = StreamHealth("cam01")
        frames = 2000
        done = threading.Event()
        seen = []

        def capture():
            for _ in range(frames):
                health.record_frame()
            done.set()

        def heartbeat():
            # At least one snapshot even if capture finishes first.
            while True:
                seen.append(health.snapshot()["frames_received"])
                if done.is_set():
                    return

        threads = [
            threading.Thread(target=capture, daemon=True),
            threading.Thread(target=heartbeat, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(_TIMEOUT_SEC)
            self.assertFalse(t.is_alive(), "capture/heartbeat threads deadlocked")

        self.assertEqual(self.snapshot(health)["frames_received"], frames)
        self.assertTrue(seen)
        self.assertEqual(seen, sorted(seen), "frames_received went backwards")


if __name__ == "__main__":
    unittest.main()
