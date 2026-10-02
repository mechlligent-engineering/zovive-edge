"""Shared test doubles and synthetic-data generators. Plain functions
and classes (not pytest fixtures) so they work identically whether a
test file uses stdlib `unittest` (the default in this repo — see
requirements.txt) or `pytest`, which collects unittest.TestCase
classes natively.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import numpy as np

from inference.base import ClassificationResult, Classifier, Detection, Detector
from utils import config_loader

# Fake camera credentials for tests that load configs/rtsp_config.yaml or
# camera.yaml, which require ZOVIVE_CAMERA_USERNAME/PASSWORD. Not real secrets.
TEST_CAMERA_ENV = {"ZOVIVE_CAMERA_USERNAME": "test-user", "ZOVIVE_CAMERA_PASSWORD": "test-pass"}  # nosec B105


@contextmanager
def camera_credentials(env: dict[str, str] | None = None) -> Iterator[None]:
    """Put fake camera credentials in os.environ. Clears the config_loader.load()
    cache on entry and exit so interpolated values never leak into other tests."""
    config_loader.clear_cache()
    try:
        with mock.patch.dict(os.environ, TEST_CAMERA_ENV if env is None else env):
            yield
    finally:
        config_loader.clear_cache()


def blank_frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 60, dtype=np.uint8)  # dark "forest" gray


def frame_with_box(width: int = 320, height: int = 240, box=(100, 80, 180, 160), value=200) -> np.ndarray:
    frame = blank_frame(width, height)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = value
    return frame


class ScriptedDetector(Detector):
    """Returns a pre-programmed sequence of detection lists, one per
    call to infer() (repeats the last entry once exhausted). Lets a
    test drive exact detector behavior across frames without a real
    model — e.g. "present for 3 of 5 frames" for stage1_gate tests.
    """

    def __init__(self, script: list[list[Detection]], class_names: list[str] | None = None):
        self.class_names = class_names or ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]
        self.script = script
        self._i = 0
        self.calls = 0

    def infer(self, frame) -> list[Detection]:
        self.calls += 1
        if not self.script:
            return []
        result = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return result


class AlwaysClassifier(Classifier):
    """Always classifies as `species` with `confidence`, regardless of
    the crop — used where the test cares about pipeline wiring, not
    classifier accuracy."""

    def __init__(self, species: str, confidence: float, class_names: list[str] | None = None):
        self.class_names = class_names or ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]
        self.species = species
        self.confidence = confidence

    def infer(self, crop) -> ClassificationResult:
        scores = [0.0] * len(self.class_names)
        idx = self.class_names.index(self.species) if self.species in self.class_names else 0
        scores[idx] = self.confidence
        remainder = (1.0 - self.confidence) / max(1, len(self.class_names) - 1)
        for i in range(len(scores)):
            if i != idx:
                scores[i] = remainder
        return ClassificationResult(
            class_id=idx, class_name=self.species, confidence=self.confidence, all_scores=scores
        )


class ScriptedClassifier(Classifier):
    """Returns a pre-programmed sequence of ClassificationResults, one
    per call to infer() (repeats the last entry once exhausted) —
    the classifier counterpart to ScriptedDetector above. Lets a test
    drive stage2_verifier's voting strategies (average vs majority)
    with an exact, known sequence of per-crop results instead of a
    single fixed answer like AlwaysClassifier gives.
    """

    def __init__(self, script: list[ClassificationResult], class_names: list[str] | None = None):
        self.class_names = class_names or ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]
        self.script = script
        self._i = 0

    def infer(self, crop) -> ClassificationResult:
        if not self.script:
            return ClassificationResult(class_id=-1, class_name="invalid_crop", confidence=0.0)
        result = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return result


def scored_result(class_names: list[str], species: str, confidence: float) -> ClassificationResult:
    """Builds one ClassificationResult with a full softmax-like
    `all_scores` vector, for feeding ScriptedClassifier — the winning
    class gets `confidence`, the rest split the remainder evenly."""
    idx = class_names.index(species)
    scores = [(1.0 - confidence) / max(1, len(class_names) - 1)] * len(class_names)
    scores[idx] = confidence
    return ClassificationResult(class_id=idx, class_name=species, confidence=confidence, all_scores=scores)


def make_detection(box=(100, 80, 180, 160), score=0.9, class_id=5, class_name="tiger") -> Detection:
    return Detection(box=box, score=score, class_id=class_id, class_name=class_name)


def write_synthetic_video(path, num_frames: int = 20, width: int = 320, height: int = 240) -> None:
    """Writes a small MJPEG video with a bright square sweeping left to
    right, for FileFrameSource-based integration tests."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (width, height))
    try:
        for i in range(num_frames):
            x = 20 + int((width - 100) * (i / max(1, num_frames - 1)))
            frame = frame_with_box(width, height, box=(x, 80, x + 60, 160))
            writer.write(frame)
    finally:
        writer.release()
