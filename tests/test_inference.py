import unittest

import numpy as np

from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker, iou
from inference.preprocess import letterbox, unletterbox_box
from inference.yolo_postprocess import decode_yolo_detections, softmax
from tests.helpers import blank_frame


class TestPreprocess(unittest.TestCase):
    def test_letterbox_preserves_aspect_and_size(self):
        img = blank_frame(width=640, height=360)
        padded, ratio, pad = letterbox(img, new_size=320)
        self.assertEqual(padded.shape[:2], (320, 320))
        self.assertAlmostEqual(ratio, 320 / 640, places=4)

    def test_unletterbox_roundtrip(self):
        img = blank_frame(width=640, height=480)
        _, ratio, pad = letterbox(img, new_size=320)
        orig_box = (100.0, 50.0, 300.0, 200.0)
        # forward-map orig -> letterboxed space
        x1, y1, x2, y2 = orig_box
        pad_x, pad_y = pad
        lb_box = (x1 * ratio + pad_x, y1 * ratio + pad_y, x2 * ratio + pad_x, y2 * ratio + pad_y)
        recovered = unletterbox_box(lb_box, ratio, pad)
        for a, b in zip(orig_box, recovered, strict=True):
            self.assertAlmostEqual(a, b, places=3)


class TestYoloPostprocess(unittest.TestCase):
    def test_decode_filters_low_confidence(self):
        class_names = ["deer", "tiger"]
        # 2 anchors: one confident tiger, one low-confidence noise anchor
        arr = np.array(
            [
                # cx,  cy,  w,   h,  deer, tiger
                [50, 50, 20, 20, 0.05, 0.90],
                [10, 10, 5, 5, 0.10, 0.10],
            ],
            dtype=np.float32,
        ).T  # shape (6, 2) = (4+num_classes, N)

        dets = decode_yolo_detections(
            arr, class_names, ratio=1.0, pad=(0, 0), orig_shape=(100, 100), conf_threshold=0.5, iou_threshold=0.5
        )
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_name, "tiger")
        self.assertAlmostEqual(dets[0].score, 0.90, places=3)

    def test_decode_empty_when_all_below_threshold(self):
        class_names = ["deer", "tiger"]
        arr = np.array([[50, 50, 20, 20, 0.1, 0.2]], dtype=np.float32).T
        dets = decode_yolo_detections(arr, class_names, 1.0, (0, 0), (100, 100), conf_threshold=0.5)
        self.assertEqual(dets, [])

    def test_softmax_sums_to_one(self):
        scores = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=5)


class TestIouTracker(unittest.TestCase):
    def test_overlapping_detection_keeps_same_track_id(self):
        tracker = IouTracker(iou_match_threshold=0.3, max_age_frames=5, min_hits_to_confirm=1)
        det1 = Detection(box=(10, 10, 50, 50), score=0.9, class_id=5, class_name="tiger")
        tracks1 = tracker.update([det1])
        self.assertEqual(len(tracks1), 1)
        tid = tracks1[0].track_id

        det2 = Detection(box=(12, 11, 52, 51), score=0.9, class_id=5, class_name="tiger")  # slight shift
        tracks2 = tracker.update([det2])
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, tid)
        self.assertEqual(tracks2[0].hits, 2)

    def test_non_overlapping_detection_creates_new_track(self):
        tracker = IouTracker(iou_match_threshold=0.3, max_age_frames=5, min_hits_to_confirm=1)
        tracker.update([Detection(box=(10, 10, 50, 50), score=0.9, class_id=5, class_name="tiger")])
        tracks = tracker.update([Detection(box=(200, 200, 240, 240), score=0.9, class_id=5, class_name="tiger")])
        # two distinct tracks should now exist (first aged, second new)
        self.assertEqual(len(tracker.active_tracks()), 2)
        self.assertNotEqual(tracks[0].track_id if tracks else None, None)

    def test_track_dropped_after_max_age(self):
        tracker = IouTracker(iou_match_threshold=0.3, max_age_frames=2, min_hits_to_confirm=1)
        tracker.update([Detection(box=(10, 10, 50, 50), score=0.9, class_id=5, class_name="tiger")])
        for _ in range(3):
            tracker.update([])  # no detections; track should age out
        self.assertEqual(tracker.active_tracks(), [])

    def test_presence_history_records_misses(self):
        tracker = IouTracker(iou_match_threshold=0.3, max_age_frames=5, min_hits_to_confirm=1, presence_window=3)
        tracker.update([Detection(box=(10, 10, 50, 50), score=0.9, class_id=5, class_name="tiger")])
        tracks = tracker.update([])  # miss
        self.assertEqual(tracks[0].presence_history, [True, False])

    def test_iou_function(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


if __name__ == "__main__":
    unittest.main()
