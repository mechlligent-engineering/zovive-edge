"""Per-class confidence thresholds (docs/PIPELINE_HARDENING_SPEC.md, change 1).

The detector decodes every class at one threshold,
`detector.confidence_threshold` in configs/inference_config.yaml. Some
species need more confidence than that before the pipeline acts on them:
`detector.class_confidence_thresholds` raises the bar per class, and a
class not listed keeps the decode threshold.

Applied once, in edge_main's inference thread right after
Detector.infer() and before the DetectionQueue, so the Hailo, ONNX and
SAHI paths are filtered identically. pipeline/stage1_gate.py uses the
same thresholds for a track's mean confidence. utils/config_loader.py
rejects a threshold for a class the model doesn't have, or one below the
decode threshold (it could never take effect).

Pure logic, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from inference.base import Detection

# What the detectors fall back to when the config has no confidence_threshold.
DEFAULT_CONFIDENCE_THRESHOLD = 0.45


@dataclass(frozen=True)
class ClassThresholds:
    default: float = DEFAULT_CONFIDENCE_THRESHOLD
    per_class: Mapping[str, float] = field(default_factory=dict)

    def for_class(self, class_name: str) -> float:
        return self.per_class.get(class_name, self.default)

    def keep(self, detections: Sequence[Detection]) -> list[Detection]:
        """The detections that reach their own class's threshold (inclusive)."""
        return [d for d in detections if d.score >= self.for_class(d.class_name)]

    @classmethod
    def from_config(cls, detector_cfg: Mapping) -> ClassThresholds:
        """From the `detector` section of configs/inference_config.yaml."""
        default = float(detector_cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        raw = detector_cfg.get("class_confidence_thresholds") or {}
        return cls(default=default, per_class={str(name): float(value) for name, value in raw.items()})
