"""ONNX Runtime backend — runs on the laptop, in CI, and on the Pi CPU
(there only as the fallback when the Hailo backend in hailo_inference.py
can't start).

This is what lets the entire pipeline be developed and tested without
a Hailo device: configs/node_config.yaml's `runtime.backend: onnx`
selects this module in edge_main.py.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from inference.base import Detection, Detector, InferenceBackendError
from inference.class_names import checked_class_names, embedded_class_names
from inference.preprocess import letterbox, to_chw_float32
from inference.yolo_postprocess import decode_yolo_detections

log = logging.getLogger(__name__)


def _make_session(onnx_path: str) -> ort.InferenceSession:
    try:
        # CPUExecutionProvider works everywhere including the Pi;
        # if onnxruntime-gpu/CUDA is present elsewhere it's picked up first.
        providers = [p for p in ort.get_available_providers()]
        return ort.InferenceSession(onnx_path, providers=providers)
    except Exception as exc:  # noqa: BLE001 - surface as our own error type
        raise InferenceBackendError(f"failed to load ONNX model {onnx_path}: {exc}") from exc


def _embedded_names(session: ort.InferenceSession) -> list[str] | None:
    return embedded_class_names(session.get_modelmeta().custom_metadata_map)


class OnnxDetector(Detector):
    def __init__(
        self,
        onnx_path: str,
        class_names: list[str],
        input_size: int = 640,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = _make_session(onnx_path)
        self.class_names = checked_class_names(
            class_names, onnx_path, model_names=_embedded_names(self.session)
        )
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
