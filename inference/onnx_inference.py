"""ONNX Runtime backend — runs on the laptop, in CI, and on the Pi CPU
(used there for the classifier, which only runs on-trigger so CPU cost
is acceptable; the always-on detector uses hailo_inference.py instead).

This is what lets the entire pipeline be developed and tested without
a Hailo device: configs/node_config.yaml's `runtime.backend: onnx`
selects this module in edge_main.py.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from inference.base import ClassificationResult, Classifier, Detection, Detector, InferenceBackendError
from inference.preprocess import letterbox, to_chw_float32
from inference.yolo_postprocess import decode_yolo_detections, softmax

log = logging.getLogger(__name__)


def _make_session(onnx_path: str) -> ort.InferenceSession:
    try:
        # CPUExecutionProvider works everywhere including the Pi;
        # if onnxruntime-gpu/CUDA is present elsewhere it's picked up first.
        providers = [p for p in ort.get_available_providers()]
        return ort.InferenceSession(onnx_path, providers=providers)
    except Exception as exc:  # noqa: BLE001 - surface as our own error type
        raise InferenceBackendError(f"failed to load ONNX model {onnx_path}: {exc}") from exc


class OnnxDetector(Detector):
    def __init__(
        self,
        onnx_path: str,
        class_names: list[str],
        input_size: int = 640,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
    ):
        self.class_names = class_names
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = _make_session(onnx_path)
        self._input_name = self.session.get_inputs()[0].name

    def infer(self, frame) -> list[Detection]:
        padded, ratio, pad = letterbox(frame, self.input_size)
        tensor = to_chw_float32(padded)
        outputs = self.session.run(None, {self._input_name: tensor})
        return decode_yolo_detections(
            outputs[0],
            self.class_names,
            ratio,
            pad,
            orig_shape=frame.shape[:2],
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

    def warmup(self) -> None:
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.infer(dummy)


class OnnxClassifier(Classifier):
    def __init__(self, onnx_path: str, class_names: list[str], input_size: int = 224):
        self.class_names = class_names
        self.input_size = input_size
        self.session = _make_session(onnx_path)
        self._input_name = self.session.get_inputs()[0].name

    def infer(self, crop) -> ClassificationResult:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) == 0:
            return ClassificationResult(class_id=-1, class_name="invalid_crop", confidence=0.0)

        padded, _ratio, _pad = letterbox(crop, self.input_size)
        tensor = to_chw_float32(padded)
        outputs = self.session.run(None, {self._input_name: tensor})
        raw = np.asarray(outputs[0]).reshape(-1)

        # YOLO11-cls export is usually already softmaxed; normalize defensively
        # in case a raw-logit export is swapped in later.
        scores = raw if np.isclose(raw.sum(), 1.0, atol=1e-2) else softmax(raw)

        class_id = int(np.argmax(scores))
        return ClassificationResult(
            class_id=class_id,
            class_name=self.class_names[class_id] if class_id < len(self.class_names) else str(class_id),
            confidence=float(scores[class_id]),
            all_scores=scores.tolist(),
        )

    def warmup(self) -> None:
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.infer(dummy)
