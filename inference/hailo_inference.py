"""Hailo-8 backend — only runs on the Raspberry Pi with HailoRT and a
compiled .hef installed (see provisioning/install_hailort.sh and
zovive-ml's export pipeline: .pt -> ONNX -> Hailo Dataflow Compiler -> .hef).

`hailo_platform` cannot be installed in this dev/CI environment (no
device, and the wheel isn't on PyPI), so the import is deferred into
__init__ rather than module load time — the module stays importable
everywhere (for tests that only check config wiring), and only
*constructing* HailoDetector can fail: FileNotFoundError
for a missing .hef, and a clear InferenceBackendError for everything on
the HailoRT side (not installed, no/busy device, a bad .hef, or outputs
that are not the expected split head). edge_main falls back to the ONNX
CPU backend on either.

The HEF contract (zovive-ml hailo_config/yolov11.alls):
- input: the 640x640 letterboxed frame (grey 114) as uint8 RGB, NHWC;
  the HEF normalises (/ 255) on the device.
- outputs: the detection head cut before its DFL decode — per stride
  (8/16/32) a box conv (64 channels) and a class conv (7 channels), NHWC,
  read as float32 and decoded on the CPU by
  inference/yolo_postprocess.decode_split_heads, the same selection the
  ONNX backend uses. No on-chip NMS.

IMPORTANT: this follows the HailoRT Python API shape as of HailoRT
4.x (HEF / VDevice / InferVStreams) and the calls of the Pi test scripts
that ran the HEF. tools/hef_parity.py compares its detections with the
ONNX model's on the Pi; numbers only come from running it there.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np

from inference.base import Detection, Detector, InferenceBackendError
from inference.class_names import checked_class_names
from inference.npu_scheduler import NpuScheduler, default_scheduler
from inference.preprocess import crop_box, letterbox  # noqa: F401 (crop_box re-exported for callers)
from inference.yolo_postprocess import decode_split_heads, split_head_layout

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
    """HEF-loading + single-input/multi-output inference helper behind
    HailoDetector.

    The InferVStreams pipeline is opened and the network group activated
    once, on the first run(), and kept for the runner's lifetime; building
    them for every frame cost about 19 new threads per inference on the Pi
    (docs/PERSISTENT_VSTREAMS_SPEC.md). Holding the group active assumes
    this is the only network group on the device, which it is: the node
    runs one model. A failed inference tears the pipeline down and the next
    run() opens it again. Every touch of the pipeline holds the NPU lock."""

    # How long close() waits for an inference in progress before leaving
    # the pipeline and device to process exit (a hung native call).
    CLOSE_LOCK_TIMEOUT_SEC = 5.0

    def __init__(self, hef_path: str, scheduler: NpuScheduler):
        # Checked first, so a model that was never deployed is reported as
        # exactly that, whether or not HailoRT is installed.
        if not Path(hef_path).is_file():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
        hpf = _import_hailo()
        self._hpf = hpf
        self.scheduler = scheduler
        self.target = None
        self._stack: contextlib.ExitStack | None = None
        self._pipeline = None
        self._closed = False

        try:
            self.hef = hpf.HEF(hef_path)
            self.target = hpf.VDevice()
            configure_params = hpf.ConfigureParams.create_from_hef(
                self.hef, interface=hpf.HailoStreamInterface.PCIe
            )
            self.network_group = self.target.configure(self.hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_infos = list(self.hef.get_output_vstream_infos())

            # uint8 in (the HEF normalises on the device), float32 out.
            self.input_vstreams_params = hpf.InputVStreamParams.make(
                self.network_group, format_type=hpf.FormatType.UINT8
            )
            self.output_vstreams_params = hpf.OutputVStreamParams.make(
                self.network_group, format_type=hpf.FormatType.FLOAT32
            )

            shape = self.input_vstream_info.shape  # (H, W, C)
            self.input_size = int(shape[0])
        except Exception as exc:  # noqa: BLE001 - HailoRT's exception classes only exist on the Pi
            # Bad .hef, no/busy device or a HailoRT API mismatch: release a
            # device that was already opened, and report a backend error so
            # the caller's CPU fallback runs.
            self.close()
            raise InferenceBackendError(
                f"HailoRT could not load {hef_path}: {type(exc).__name__}: {exc}"
            ) from exc

    def _open(self) -> None:
        """Pipeline first, then activation: the nesting HailoRT's own
        examples use. Caller holds the NPU lock."""
        stack = contextlib.ExitStack()
        try:
            pipeline = stack.enter_context(
                self._hpf.InferVStreams(
                    self.network_group, self.input_vstreams_params, self.output_vstreams_params
                )
            )
            stack.enter_context(self.network_group.activate(self.network_group_params))
        except BaseException:
            stack.close()
            raise
        self._stack, self._pipeline = stack, pipeline

    def _teardown(self) -> None:
        """Deactivate, then release the vstreams. Caller holds the NPU lock."""
        stack, self._stack, self._pipeline = self._stack, None, None
        if stack is None:
            return
        try:
            stack.close()
        except Exception:  # noqa: BLE001 - HailoRT's exception classes only exist on the Pi
            log.exception("error closing the Hailo inference pipeline")

    def run(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        """`input_tensor`: HxWxC uint8, NHWC (Hailo's native layout).
        Returns every output by name, batch dimension removed."""

        def _infer():
            if self._closed:
                raise InferenceBackendError("the Hailo runner is closed")
            if self._pipeline is None:
                self._open()
            input_data = {self.input_vstream_info.name: input_tensor[np.newaxis, ...]}
            try:
                results = self._pipeline.infer(input_data)
            except Exception:
                self._teardown()  # reopened on the next call
                raise
            return {name: np.asarray(value)[0] for name, value in results.items()}

        return self.scheduler.run(_infer)

    def close(self) -> None:
        """Idempotent. Waits for an inference in progress, up to
        CLOSE_LOCK_TIMEOUT_SEC; past that a native call is hung, and the
        pipeline and device are left to process exit."""
        with self.scheduler.exclusive(timeout=self.CLOSE_LOCK_TIMEOUT_SEC) as locked:
            if not locked:
                log.warning("Hailo device busy at close; leaving it to process exit")
                return
            self._closed = True
            self._teardown()
            target, self.target = self.target, None
            if target is None:
                return
            try:
                target.release()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                log.exception("error releasing Hailo VDevice")


def _to_nhwc_uint8_rgb(image_letterboxed) -> np.ndarray:
    # letterbox() returns BGR; the model was trained on RGB, matching the
    # ONNX path. Pixel values stay 0-255: the HEF divides by 255 itself.
    import cv2

    return np.ascontiguousarray(cv2.cvtColor(image_letterboxed, cv2.COLOR_BGR2RGB))


class HailoDetector(Detector):
    def __init__(
        self,
        hef_path: str,
        class_names: list[str],
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        scheduler: NpuScheduler | None = None,
    ):
        self.class_names = checked_class_names(class_names, hef_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._runner = _HailoRunner(hef_path, scheduler or default_scheduler)
        self.input_size = self._runner.input_size
        try:
            self._heads = split_head_layout(
                {info.name: tuple(info.shape) for info in self._runner.output_vstream_infos},
                len(self.class_names),
                self.input_size,
            )
        except ValueError as exc:
            # Not the split head this decoder reads (e.g. a different cut
            # or on-chip NMS): release the device and let the CPU fallback run.
            self._runner.close()
            raise InferenceBackendError(f"{hef_path}: {exc}") from exc

    def infer(self, frame) -> list[Detection]:
        padded, ratio, pad = letterbox(frame, self.input_size)
        outputs = self._runner.run(_to_nhwc_uint8_rgb(padded))
        return decode_split_heads(
            outputs,
            self._heads,
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
