"""Tests for pipeline/ram_buffer.py."""

from __future__ import annotations

import unittest

from pipeline.ram_buffer import RamBuffer
from tests.helpers import blank_frame


class TestRamBuffer(unittest.TestCase):
    def test_for_duration_sizes_from_fps(self):
        buf = RamBuffer.for_duration(window_seconds=8.0, fps=8.0)
        self.assertEqual(buf.max_frames, 64)

    def test_for_duration_rounds_and_floors_at_one(self):
        buf = RamBuffer.for_duration(window_seconds=0.01, fps=1.0)
        self.assertEqual(buf.max_frames, 1)

    def test_offer_and_len(self):
        buf = RamBuffer(max_frames=5)
        for i in range(3):
            buf.offer(blank_frame(), timestamp=float(i))
        self.assertEqual(len(buf), 3)

    def test_drops_oldest_when_full(self):
        buf = RamBuffer(max_frames=3)
        for i in range(5):
            buf.offer(blank_frame(), timestamp=float(i))
        frames = buf.frames()
        self.assertEqual(len(frames), 3)
        # Should hold the 3 most recent timestamps: 2, 3, 4.
        self.assertEqual([f.timestamp for f in frames], [2.0, 3.0, 4.0])

    def test_frames_since_filters_by_timestamp(self):
        buf = RamBuffer(max_frames=10)
        for i in range(10):
            buf.offer(blank_frame(), timestamp=float(i))
        recent = buf.frames_since(5.0)
        self.assertEqual([f.timestamp for f in recent], [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_frames_returns_independent_snapshot(self):
        buf = RamBuffer(max_frames=5)
        buf.offer(blank_frame(), timestamp=1.0)
        snapshot = buf.frames()
        buf.offer(blank_frame(), timestamp=2.0)
        # The earlier snapshot should not see the frame offered after it was taken.
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(buf.frames()), 2)

    def test_rejects_non_positive_max_frames(self):
        with self.assertRaises(ValueError):
            RamBuffer(max_frames=0)


if __name__ == "__main__":
    unittest.main()
