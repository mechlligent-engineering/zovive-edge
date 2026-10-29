"""The Hailo split-head decoder (inference/yolo_postprocess.split_head_layout /
decode_split_heads) and HailoDetector with HailoRT faked.

The HEF ends at the YOLO11 head's six convs (ZOVIVE_FINAL_150E compile log):
per stride 8/16/32 a box conv with 64 DFL channels and a class conv with 7
channels, NHWC. Synthetic outputs encode known boxes (one-hot DFL bins,
Ultralytics' layout channel = side * 16 + bin) and class logits (channel i =
class i, no objectness channel), and must decode exactly — and identically
to the ONNX path's decode_yolo_detections for the same rows. The match with
the real float model is checked by zovive-ml
scripts/eval_audit/verify_split_head_decoder.py; with the quantized HEF, on
the Pi, by tools/hef_parity.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from inference.base import InferenceBackendError
from inference.hailo_inference import HailoDetector
from inference.yolo_postprocess import decode_split_heads, decode_yolo_detections, split_head_layout

NAMES = ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]
GRID = {8: 80, 16: 40, 32: 20}
# Output names of the compiled HEF; the decoder must not depend on them.
LAYERS = {8: ("conv71", "conv74"), 16: ("conv87", "conv90"), 32: ("conv102", "conv105")}


def _name(stride: int, kind: str) -> str:
    return f"zovive_detector/{LAYERS[stride][0 if kind == 'box' else 1]}"


def _heads(objects=(), *, batch: bool = False) -> dict[str, np.ndarray]:
    """objects: (stride, grid_y, grid_x, class_id, logit, (left, top, right, bottom) in grid cells)."""
    out = {}
    for stride, n in GRID.items():
        out[_name(stride, "box")] = np.zeros((n, n, 64), np.float32)
        out[_name(stride, "cls")] = np.full((n, n, 7), -20.0, np.float32)
    for stride, gy, gx, class_id, logit, ltrb in objects:
        out[_name(stride, "cls")][gy, gx, class_id] = logit
        for side, cells in enumerate(ltrb):
            out[_name(stride, "box")][gy, gx, side * 16 + cells] = 30.0  # one-hot DFL bin
    return {k: v[None] for k, v in out.items()} if batch else out


def _expected_box(stride, gy, gx, ltrb):
    ax, ay = (gx + 0.5) * stride, (gy + 0.5) * stride
    left, top, right, bottom = (d * stride for d in ltrb)
    return (ax - left, ay - top, ax + right, ay + bottom)


def _layout(outputs):
    return split_head_layout({k: v.shape for k, v in outputs.items()}, len(NAMES), 640)


def _decode(outputs, *, ratio=1.0, pad=(0, 0), orig=(640, 640)):
    return decode_split_heads(outputs, _layout(outputs), NAMES, ratio, pad, orig)


TIGER = (8, 40, 30, 5, 5.0, (2, 3, 4, 5))


class LayoutTests(unittest.TestCase):
    def test_outputs_are_matched_by_shape_not_name_or_order(self):
        outputs = _heads()
        shuffled = {f"out{i}": outputs[k].shape for i, k in enumerate(reversed(list(outputs)))}
        layout = split_head_layout(shuffled, 7, 640)
        self.assertEqual([h.stride for h in layout], [8, 16, 32])
        for head in layout:
            self.assertEqual(shuffled[head.box][-1], 64)
            self.assertEqual(shuffled[head.cls][-1], 7)

    def test_a_leading_batch_dimension_is_accepted(self):
        layout = _layout(_heads(batch=True))
        self.assertEqual([h.stride for h in layout], [8, 16, 32])

    def test_outputs_this_decoder_cannot_read_are_rejected(self):
        good = {k: v.shape for k, v in _heads().items()}
        cases = {
            "single exported head": {"output0": (1, 11, 8400)},
            "8 class channels (objectness + 7)": {
                **good,
                _name(8, "cls"): (80, 80, 8),
            },
            "a stride missing its class output": {k: v for k, v in good.items() if k != _name(16, "cls")},
            "only two strides": {k: v for k, v in good.items() if "conv10" not in k},
            "grid that does not fit the input": {
                **good,
                _name(32, "box"): (21, 21, 64),
                _name(32, "cls"): (21, 21, 7),
            },
        }
        for name, shapes in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                split_head_layout(shapes, 7, 640)


class DecodeTests(unittest.TestCase):
    def test_one_encoded_animal_decodes_exactly(self):
        [det] = _decode(_heads([TIGER]))
        self.assertEqual((det.class_id, det.class_name), (5, "tiger"))
        self.assertAlmostEqual(det.score, 1 / (1 + np.exp(-5.0)), places=6)
        np.testing.assert_allclose(det.box, _expected_box(8, 40, 30, (2, 3, 4, 5)), atol=1e-3)

    def test_class_channel_i_is_species_i(self):
        for class_id, species in enumerate(NAMES):
            with self.subTest(species=species):
                [det] = _decode(_heads([(16, 20, 20, class_id, 4.0, (3, 3, 3, 3))]))
                self.assertEqual(det.class_name, species)

    def test_every_stride_and_a_batch_dimension(self):
        objects = [
            (8, 10, 10, 0, 4.0, (1, 1, 1, 1)),
            (16, 20, 30, 3, 4.0, (4, 2, 4, 2)),
            (32, 15, 5, 6, 4.0, (2, 2, 3, 3)),
        ]
        dets = sorted(_decode(_heads(objects, batch=True)), key=lambda d: d.class_id)
        self.assertEqual([d.class_name for d in dets], ["deer", "elephant", "wild_boar"])
        for det, (stride, gy, gx, _c, _l, ltrb) in zip(dets, objects, strict=True):
            np.testing.assert_allclose(det.box, _expected_box(stride, gy, gx, ltrb), atol=1e-3)

    def test_below_the_threshold_nothing_is_decoded(self):
        self.assertEqual(_decode(_heads([(8, 40, 30, 5, -1.0, (2, 2, 2, 2))])), [])

    def test_same_detections_as_the_onnx_path_for_the_same_rows(self):
        x1, y1, x2, y2 = _expected_box(8, 40, 30, (2, 3, 4, 5))
        raw = np.zeros((1, 11, 8400), np.float32)
        raw[0, :4, 123] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        raw[0, 4 + 5, 123] = 1 / (1 + np.exp(-5.0))
        onnx = decode_yolo_detections(raw, NAMES, 1.0, (0, 0), (640, 640))
        hailo = _decode(_heads([TIGER]))
        self.assertEqual(
            [(d.class_name, round(d.score, 6)) for d in hailo],
            [(d.class_name, round(d.score, 6)) for d in onnx],
        )
        np.testing.assert_allclose(hailo[0].box, onnx[0].box, atol=1e-3)

    def test_overlapping_boxes_are_de_duplicated_once(self):
        # The same animal seen at stride 8 and stride 16 (IoU ~0.8): one survives, the stronger.
        dets = _decode(_heads([(8, 40, 40, 5, 6.0, (4, 4, 4, 4)), (16, 20, 20, 2, 3.0, (2, 2, 2, 2))]))
        self.assertEqual(
            [(d.class_name, round(d.score, 4)) for d in dets], [("tiger", round(1 / (1 + np.exp(-6.0)), 4))]
        )

    def test_boxes_centred_in_the_letterbox_padding_are_dropped_by_both_paths(self):
        # A 640x360 frame letterboxed into 640x640: content rows 140..500.
        pad_obj = (8, 5, 40, 1, 5.0, (2, 2, 2, 2))  # centre y = 44: padding
        real_obj = (8, 40, 40, 5, 5.0, (2, 2, 2, 2))  # centre y = 324: content
        dets = _decode(_heads([pad_obj, real_obj]), pad=(0, 140), orig=(360, 640))
        self.assertEqual([d.class_name for d in dets], ["tiger"])
        self.assertAlmostEqual(dets[0].box[1], 324 - 16 - 140, places=3)

        raw = np.zeros((1, 11, 8400), np.float32)
        raw[0, :4, 0], raw[0, 4 + 1, 0] = (324, 44, 32, 32), 0.99
        self.assertEqual(decode_yolo_detections(raw, NAMES, 1.0, (0, 140), (360, 640)), [])


def _fake_hailort(outputs: dict[str, np.ndarray]):
    """HailoRT stand-in serving `outputs` (with a batch dimension, as HailoRT
    returns them) and recording what it was fed."""
    hpf = MagicMock()
    hpf.FormatType.UINT8, hpf.FormatType.FLOAT32 = "UINT8", "FLOAT32"
    hef = hpf.HEF.return_value
    hef.get_input_vstream_infos.return_value = [
        SimpleNamespace(name="zovive_detector/input_layer1", shape=(640, 640, 3))
    ]
    hef.get_output_vstream_infos.return_value = [
        SimpleNamespace(name=k, shape=v.shape) for k, v in outputs.items()
    ]
    device = hpf.VDevice.return_value
    device.configure.return_value = [MagicMock()]
    fed: list[np.ndarray] = []

    def infer(data):
        fed.extend(data.values())
        return {k: v[None] for k, v in outputs.items()}

    hpf.InferVStreams.return_value.__enter__.return_value.infer.side_effect = infer
    return hpf, device, fed


class HailoDetectorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.hef_path = str(Path(tmp.name) / "detector.hef")
        Path(self.hef_path).write_bytes(b"")

    def test_reads_all_six_outputs_and_feeds_uint8_rgb(self):
        hpf, _device, fed = _fake_hailort(_heads([(8, 40, 40, 5, 5.0, (2, 3, 4, 5))]))
        with patch("inference.hailo_inference._import_hailo", return_value=hpf):
            detector = HailoDetector(self.hef_path, NAMES)
            frame = np.zeros((360, 640, 3), np.uint8)
            frame[:, :] = (10, 20, 30)  # BGR
            dets = detector.infer(frame)

        hpf.InputVStreamParams.make.assert_called_once()
        self.assertEqual(hpf.InputVStreamParams.make.call_args.kwargs["format_type"], "UINT8")
        self.assertEqual(hpf.OutputVStreamParams.make.call_args.kwargs["format_type"], "FLOAT32")
        [tensor] = fed
        self.assertEqual((tensor.dtype, tensor.shape), (np.uint8, (1, 640, 640, 3)))
        self.assertEqual(tuple(tensor[0, 320, 320]), (30, 20, 10), "RGB, 0-255, not divided by 255")
        self.assertEqual(tuple(tensor[0, 10, 320]), (114, 114, 114), "letterbox padding")
        [det] = dets
        self.assertEqual(det.class_name, "tiger")
        x1, y1, x2, y2 = _expected_box(8, 40, 40, (2, 3, 4, 5))
        np.testing.assert_allclose(det.box, (x1, y1 - 140, x2, y2 - 140), atol=1e-3)

    def test_a_hef_that_is_not_the_split_head_falls_back_and_releases_the_device(self):
        hpf, device, _fed = _fake_hailort({"output0": np.zeros((1, 11, 8400), np.float32)})
        with patch("inference.hailo_inference._import_hailo", return_value=hpf):
            with self.assertRaises(InferenceBackendError) as cm:
                HailoDetector(self.hef_path, NAMES)
        self.assertIn("8400 channels", str(cm.exception))
        device.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
