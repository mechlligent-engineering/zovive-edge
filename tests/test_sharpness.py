import unittest

import cv2
import numpy as np

from pipeline.sharpness_filter import is_blurry, sharpness_score


def _checkerboard(size=128, square=8):
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square * 2):
        for x in range(0, size, square * 2):
            img[y : y + square, x : x + square] = 255
            img[y + square : y + square * 2, x + square : x + square * 2] = 255
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestSharpnessFilter(unittest.TestCase):
    def test_blurred_image_scores_lower_than_sharp(self):
        sharp = _checkerboard()
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

        sharp_score = sharpness_score(sharp)
        blur_score = sharpness_score(blurred)

        self.assertGreater(sharp_score, blur_score)

    def test_is_blurry_threshold(self):
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        self.assertTrue(is_blurry(flat, threshold=1.0))

    def test_empty_image_is_zero(self):
        self.assertEqual(sharpness_score(np.zeros((0, 0, 3), dtype=np.uint8)), 0.0)


if __name__ == "__main__":
    unittest.main()
