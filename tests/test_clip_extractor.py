"""Tests for pipeline/clip_extractor.py: pre-roll capture from
RamBuffer, post-roll collection, async MP4 finalization, and the
max_concurrent_recordings safety cap.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from pipeline.clip_extractor import ClipExtractor
from tests.helpers import blank_frame

BASE_CONFIG = {
    "clip": {
        "enabled": True,
        "pre_event_seconds": 1.0,
        "post_event_seconds": 0.2,
        "fps": 5.0,
        "max_width": None,
        "codec": "mp4v",
        "max_concurrent_recordings": 2,
    }
}


class TestClipExtractor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_clip_test_")
        self.ready_events: dict[str, str] = {}
        self._ready_signal = threading.Event()

        def on_clip_ready(event_id: str, path: str) -> None:
            self.ready_events[event_id] = path
            self._ready_signal.set()

        self.on_clip_ready = on_clip_ready

    def _wait_for_ready(self, timeout: float = 5.0) -> None:
        self.assertTrue(self._ready_signal.wait(timeout=timeout), "clip was never finalized")

    def _extractor(self, config: dict | None = None) -> ClipExtractor:
        return ClipExtractor(
            config=config or BASE_CONFIG, clip_dir=Path(self.tmpdir), on_clip_ready=self.on_clip_ready
        )

    def test_disabled_start_is_noop(self):
        cfg = {**BASE_CONFIG, "clip": {**BASE_CONFIG["clip"], "enabled": False}}
        extractor = self._extractor(cfg)
        started = extractor.start(event_id="evt1", track_id="t1", camera_id="cam01")
        self.assertFalse(started)
        self.assertEqual(extractor.active_count(), 0)

    def test_start_captures_pre_roll_and_writes_clip_on_deadline(self):
        extractor = self._extractor()
        now = time.time()

        # Feed pre-roll frames before the event fires.
        for i in range(5):
            extractor.offer_frame(blank_frame(), timestamp=now + i * 0.1)

        started = extractor.start(event_id="evt1", track_id="t1", camera_id="cam01")
        self.assertTrue(started)
        self.assertEqual(extractor.active_count(), 1)

        # Feed post-roll frames past the (short, 0.2s) post_event_seconds
        # deadline so offer_frame finalizes it.
        t = now + 0.5
        for i in range(4):
            extractor.offer_frame(blank_frame(), timestamp=t + i * 0.1)

        self._wait_for_ready()
        self.assertIn("evt1", self.ready_events)
        out_path = Path(self.ready_events["evt1"])
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)
        self.assertEqual(extractor.active_count(), 0)

    def test_duplicate_start_for_same_event_is_ignored(self):
        extractor = self._extractor()
        extractor.start(event_id="evt1", track_id="t1", camera_id="cam01")
        second = extractor.start(event_id="evt1", track_id="t1", camera_id="cam01")
        self.assertFalse(second)
        self.assertEqual(extractor.active_count(), 1)

    def test_max_concurrent_recordings_caps_active_set(self):
        extractor = self._extractor()
        self.assertTrue(extractor.start(event_id="evt1", track_id="t1"))
        self.assertTrue(extractor.start(event_id="evt2", track_id="t2"))
        # Third exceeds max_concurrent_recordings=2 in BASE_CONFIG.
        self.assertFalse(extractor.start(event_id="evt3", track_id="t3"))
        self.assertEqual(extractor.active_count(), 2)

    def test_flush_all_finalizes_early(self):
        extractor = self._extractor()
        extractor.offer_frame(blank_frame(), timestamp=time.time())
        extractor.start(event_id="evt1", track_id="t1", camera_id="cam01")
        self.assertEqual(extractor.active_count(), 1)

        extractor.flush_all()

        self._wait_for_ready()
        self.assertEqual(extractor.active_count(), 0)
        self.assertIn("evt1", self.ready_events)


if __name__ == "__main__":
    unittest.main()
