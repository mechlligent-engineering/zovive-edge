"""Hailo-8 backend — only runs on the Raspberry Pi with HailoRT and a
compiled .hef installed (see provisioning/install_hailort.sh and
zovive-ml's export pipeline: .pt -> ONNX -> Hailo Dataflow Compiler -> .hef).

`hailo_platform` cannot be installed in this dev/CI environment (no
device, and the wheel isn't on PyPI), so the import is deferred into
__init__ rather than module load time — the module stays importable
everywhere (for tests that only check config wiring), and only
*constructing* HailoDetector/HailoClassifier on a non-Pi machine raises
a clear InferenceBackendError instead of an ImportError stack trace.

IMPORTANT: this follows the HailoRT Python API shape as of HailoRT
4.x (HEF / VDevice / InferVStreams). Verify method names against
`python3 -c "import hailo_platform; help(hailo_platform)"` on the
actual Pi once HailoRT is installed — Hailo has changed this API
across major versions before, and this code has not been run against
real hardware.
"""

from __future__ import annotations

import logging

import numpy as np

from inference.base import ClassificationResult, Classifier, Detection, Detector, InferenceBackendError
from inference.npu_scheduler import NpuScheduler, default_scheduler
from inference.preprocess import crop_box, letterbox  # noqa: F401 (crop_box re-exported for callers)
from inference.yolo_postprocess import decode_yolo_detections, softmax

log = logging.getLogger(__name__)


def _import_hailo():
    try:
        import hailo_platform as hpf  # type: ignore

        return hpf
    except ImportError as exc:
        raise InferenceBackendError(
            "hailo_platform is not installed. This backend only runs on the "
            "Raspberry Pi after provisioning/install_hailort.sh. Use "
            "configs/node_config.yaml runtime.backend: onnx for development."
        ) from exc


class _HailoRunner:
    """Shared HEF-loading + single-input/single-output inference helper
    used by both HailoDetector and HailoClassifier."""

    def __init__(self, hef_path: str, scheduler: NpuScheduler):
        hpf = _import_hailo()
        self._hpf = hpf
        self.scheduler = scheduler

        self.hef = hpf.HEF(hef_path)
        self.target = hpf.VDevice()
        configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        shape = self.input_vstream_info.shape  # (H, W, C)
        self.input_size = int(shape[0])

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """`input_tensor`: HxWxC float32, NHWC (Hailo's native layout)."""
        hpf = self._hpf

        def _infer():
            with hpf.InferVStreams(
                self.network_group, self.input_vstreams_params, self.output_vstreams_params
            ) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    input_data = {self.input_vstream_info.name: input_tensor[np.newaxis, ...]}
                    results = infer_pipeline.infer(input_data)
                    return results[self.output_vstream_info.name]

        return self.scheduler.run(_infer)

    def close(self) -> None:
        try:
            self.target.release()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            log.exception("error releasing Hailo VDevice")


def _to_nhwc_float(image_letterboxed) -> np.ndarray:
    # letterbox() returns BGR; Hailo models compiled from Ultralytics
    # exports generally expect RGB, matching the ONNX path.
    import cv2

    rgb = cv2.cvtColor(image_letterboxed, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


class HailoDetector(Detector):
    def __init__(
        self,
        hef_path: str,
        class_names: list[str],
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        scheduler: NpuScheduler | None = None,
    ):
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._runner = _HailoRunner(hef_path, scheduler or default_scheduler)
        self.input_size = self._runner.input_size

    def infer(self, frame) -> list[Detection]:
        padded, ratio, pad = letterbox(frame, self.input_size)
        tensor = _to_nhwc_float(padded)
        raw_output = self._runner.run(tensor)
        return decode_yolo_detections(
            raw_output,
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

    def close(self) -> None:
        self._runner.close()


class HailoClassifier(Classifier):
    def __init__(
        self,
        hef_path: str,
        class_names: list[str],
        scheduler: NpuScheduler | None = None,
    ):
        self.class_names = class_names
        self._runner = _HailoRunner(hef_path, scheduler or default_scheduler)
        self.input_size = self._runner.input_size

    def infer(self, crop) -> ClassificationResult:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) == 0:
            return ClassificationResult(class_id=-1, class_name="invalid_crop", confidence=0.0)

        padded, _ratio, _pad = letterbox(crop, self.input_size)
        tensor = _to_nhwc_float(padded)
        raw = np.asarray(self._runner.run(tensor)).reshape(-1)
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

    def close(self) -> None:
        self._runner.close()
