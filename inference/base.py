"""Backend-agnostic interfaces for detection and classification.

This is the seam that lets the exact same pipeline code
(pipeline/stage1_gate.py, stage2_verifier.py, edge_main.py) run on a
laptop against ONNX Runtime and on the Pi against the Hailo-8, without
an `if backend == "hailo"` anywhere outside this module and the two
backend implementations (onnx_inference.py, hailo_inference.py).

Not present in the original skeleton listing — added because
onnx_inference.py and hailo_inference.py both need a shared contract
to implement, and pipeline code needs something to type-hint against
that doesn't import either backend directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in ORIGINAL frame pixels
    score: float
    class_id: int
    class_name: str


@dataclass
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float
    # per-class softmax scores, for averaging across multiple crops in stage2_verifier.
    all_scores: list[float] = field(default_factory=list)


class Detector(ABC):
    """Stage-1 "is there an animal, and roughly where" model."""

    class_names: list[str]

    @abstractmethod
    def infer(self, frame: Any) -> list[Detection]:
        """Run detection on one BGR frame (numpy HxWx3 uint8). Returns
        detections already filtered by confidence + NMS, in original
        frame coordinates."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: run a dummy inference to pay JIT/allocation cost up front."""
        return None

    def close(self) -> None:
        return None


class Classifier(ABC):
    """Stage-2 species classifier, run only on cropped, triggered regions."""

    class_names: list[str]

    @abstractmethod
    def infer(self, crop: Any) -> ClassificationResult:
        """Classify one cropped BGR image (numpy HxWx3 uint8)."""
        raise NotImplementedError

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class InferenceBackendError(RuntimeError):
    """Raised when a backend can't be constructed (missing runtime, bad model path, ...)."""
