"""tools/hef_parity.py's comparison logic (the HEF itself only runs on the Pi)."""

from __future__ import annotations

import unittest

from inference.base import Detection
from tools.hef_parity import compare, summarize


def _det(box, name="tiger", score=0.9) -> Detection:
    return Detection(box=box, score=score, class_id=0, class_name=name)


class CompareTests(unittest.TestCase):
    def test_pairs_by_overlap_and_counts_species_agreement(self):
        onnx = [_det((0, 0, 100, 100), "tiger", 0.9), _det((200, 200, 300, 300), "elephant", 0.8)]
        hef = [_det((210, 205, 300, 300), "gaur", 0.7), _det((2, 0, 100, 100), "tiger", 0.85)]
        result = compare(onnx, hef)
        self.assertEqual(
            (result["onnx"], result["hef"], result["matched"], result["same_species"]), (2, 2, 2, 1)
        )
        self.assertEqual(sorted(result["score_diffs"]), [0.05, 0.1])

    def test_a_box_far_off_does_not_pair(self):
        result = compare([_det((0, 0, 100, 100))], [_det((60, 60, 160, 160))])  # IoU ~0.09
        self.assertEqual(result["matched"], 0)

    def test_summary_over_images(self):
        per_image = {
            "a.jpg": compare([_det((0, 0, 100, 100))], [_det((0, 0, 100, 100))]),
            "b.jpg": compare([_det((0, 0, 100, 100), "deer")], []),
        }
        summary = summarize(per_image)
        self.assertEqual((summary["images"], summary["onnx"], summary["matched"]), (2, 2, 1))
        self.assertEqual(summary["matched_of_onnx"], 0.5)
        self.assertEqual(summary["same_species_of_matched"], 1.0)
        self.assertEqual(summary["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
