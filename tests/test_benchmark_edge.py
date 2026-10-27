"""tools/benchmark_edge.py: Phase 0 benchmark. The Hailo SDK is faked; the
real numbers only come from running the tool on the Pi."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2

from inference.sahi_slicer import SahiDetector
from queues.frame_queue import LatestFrameSlot
from tests.helpers import AlwaysClassifier, ScriptedDetector, blank_frame
from tools import benchmark_edge
from tools.benchmark_edge import (
    bench_models,
    bench_sahi,
    config_for_stream,
    inspect_hefs,
    is_nms_format,
    latency_stats,
    raw_yolo_compatible,
    soak,
)
from utils.thermal_monitor import read_soc_temp_c


def _info(name, shape, order="FormatOrder.NHWC"):
    return NS(name=name, shape=shape, format=NS(order=order, type="FormatType.FLOAT32"))


def _fake_hpf(outputs_by_path, configure_fails_for=None, broken_paths=()):
    configured = []

    class FakeHef:
        def __init__(self, path):
            if path in broken_paths:
                raise RuntimeError("HEF file is corrupt")
            self.path = path

        def get_network_group_names(self):
            return [Path(self.path).stem]

        def get_input_vstream_infos(self):
            return [_info("input", (640, 640, 3))]

        def get_output_vstream_infos(self):
            return outputs_by_path[self.path]

    class FakeVDevice:
        @staticmethod
        def create_params():
            return NS(scheduling_algorithm=None)

        def __init__(self, params):
            self.params = params

        def configure(self, hef, params):
            if configure_fails_for and configure_fails_for in hef.path:
                raise RuntimeError("not enough device resources")
            configured.append(hef.path)

        def release(self):
            pass

    hpf = NS(
        __version__="4.20.0",
        HEF=FakeHef,
        VDevice=FakeVDevice,
        HailoSchedulingAlgorithm=NS(ROUND_ROBIN="ROUND_ROBIN"),
        ConfigureParams=NS(create_from_hef=lambda hef, interface=None: {}),
        HailoStreamInterface=NS(PCIe="PCIe"),
    )
    return hpf, configured


RAW_DETECTOR = {"det.hef": [_info("yolo11m/conv", (1, 11, 8400))], "cls.hef": [_info("cls/softmax", (1, 7))]}
NUM_CLASSES = {"detector": 7, "classifier": 7}
HEF_PATHS = {"detector": "det.hef", "classifier": "cls.hef"}


class TestHelpers(unittest.TestCase):
    def test_latency_stats(self):
        stats = latency_stats([0.001, 0.002, 0.003, 0.004, 0.005])
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean_ms"], 3.0)
        self.assertEqual(stats["p50_ms"], 3.0)
        self.assertEqual(stats["p95_ms"], 5.0)
        self.assertEqual(stats["max_ms"], 5.0)
        self.assertEqual(latency_stats([]), {"count": 0})

    def test_raw_yolo_compatible(self):
        self.assertTrue(raw_yolo_compatible([(1, 11, 8400)], 7))
        self.assertTrue(raw_yolo_compatible([(8400, 11)], 7))
        self.assertFalse(raw_yolo_compatible([(1, 11, 8400), (1, 64, 80, 80)], 7))  # several outputs
        self.assertFalse(raw_yolo_compatible([(80, 80, 64)], 7))
        self.assertFalse(raw_yolo_compatible([(1, 12, 8400)], 7))  # wrong class count

    def test_is_nms_format(self):
        self.assertTrue(is_nms_format("FormatOrder.HAILO_NMS_BY_CLASS"))
        self.assertFalse(is_nms_format("FormatOrder.NHWC"))

    def test_config_for_stream_copies(self):
        cfg = {"camera": {"sub_stream": {"url": "sub"}, "main_stream": {"url": "main"}}}
        out = config_for_stream(cfg, "main_stream")
        self.assertEqual(out["camera"]["sub_stream"]["url"], "main")
        self.assertEqual(cfg["camera"]["sub_stream"]["url"], "sub")  # original untouched

    def test_read_soc_temp(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "temp"
            p.write_text("52345\n")
            self.assertEqual(read_soc_temp_c(p), 52.3)
            p.write_text("garbage")
            self.assertIsNone(read_soc_temp_c(p))
            self.assertIsNone(read_soc_temp_c(Path(d) / "missing"))


class TestInspectHefs(unittest.TestCase):
    def test_raw_head_detector_is_decodable_and_models_share_a_vdevice(self):
        hpf, configured = _fake_hpf(RAW_DETECTOR)
        report = inspect_hefs(HEF_PATHS, NUM_CLASSES, hpf=hpf)
        det = report["models"]["detector"]
        self.assertEqual(report["hailort_version"], "4.20.0")
        self.assertFalse(det["on_chip_nms"])
        self.assertTrue(det["decode_yolo_detections_compatible"])
        self.assertEqual(report["shared_vdevice"], {"ok": True, "models": ["classifier", "detector"]})
        self.assertEqual(configured, ["det.hef", "cls.hef"])

    def test_on_chip_nms_is_flagged_incompatible(self):
        outputs = dict(
            RAW_DETECTOR, **{"det.hef": [_info("nms", (), order="FormatOrder.HAILO_NMS_BY_CLASS")]}
        )
        hpf, _ = _fake_hpf(outputs)
        det = inspect_hefs(HEF_PATHS, NUM_CLASSES, hpf=hpf)["models"]["detector"]
        self.assertTrue(det["on_chip_nms"])
        self.assertFalse(det["decode_yolo_detections_compatible"])

    def test_per_stride_outputs_are_incompatible(self):
        heads = [_info(f"conv{i}", (80 // 2**i, 80 // 2**i, 71)) for i in range(3)]
        hpf, _ = _fake_hpf(dict(RAW_DETECTOR, **{"det.hef": heads}))
        det = inspect_hefs(HEF_PATHS, NUM_CLASSES, hpf=hpf)["models"]["detector"]
        self.assertFalse(det["decode_yolo_detections_compatible"])

    def test_shared_vdevice_failure_is_reported(self):
        hpf, _ = _fake_hpf(RAW_DETECTOR, configure_fails_for="cls")
        shared = inspect_hefs(HEF_PATHS, NUM_CLASSES, hpf=hpf)["shared_vdevice"]
        self.assertFalse(shared["ok"])
        self.assertEqual(shared["stage"], "configure:classifier")

    def test_broken_hef_reported_without_hiding_the_other(self):
        hpf, _ = _fake_hpf(RAW_DETECTOR, broken_paths=("cls.hef",))
        report = inspect_hefs(HEF_PATHS, NUM_CLASSES, hpf=hpf)
        self.assertIn("corrupt", report["models"]["classifier"]["error"])
        self.assertIn("outputs", report["models"]["detector"])
        self.assertNotIn("shared_vdevice", report)  # needs two loadable HEFs

    def test_hailo_not_installed(self):
        with patch.object(benchmark_edge, "_import_hailo_platform", side_effect=ImportError("no module")):
            report = inspect_hefs(HEF_PATHS, NUM_CLASSES)
        self.assertFalse(report["available"])


class TestModelBenchmarks(unittest.TestCase):
    def test_bench_models_reports_backend_types_and_sizes(self):
        result = bench_models(ScriptedDetector([[]]), AlwaysClassifier("tiger", 0.9), iterations=3)
        self.assertEqual(result["detector_type"], "ScriptedDetector")
        self.assertEqual(result["classifier_type"], "AlwaysClassifier")
        self.assertEqual(set(result["detector"]), {"720p", "4mp"})
        self.assertEqual(result["detector"]["4mp"]["count"], 3)
        self.assertEqual(result["classifier"]["count"], 3)

    def test_bench_models_unwraps_sahi(self):
        result = bench_models(
            SahiDetector(ScriptedDetector([[]])), AlwaysClassifier("tiger", 0.9), iterations=1
        )
        self.assertEqual(result["detector_type"], "ScriptedDetector")

    def test_sahi_full_frame_vs_roi_call_counts(self):
        result = bench_sahi(ScriptedDetector([[]]), iterations=2)
        self.assertEqual(result["full_frame_tiles"], 15)  # 5 x 3 tiles of 640 at 2688x1520
        self.assertEqual(result["full_frame"]["detector_calls_per_run"], 16)  # + full-frame pass
        self.assertEqual(result["roi"]["region"], "640x640")
        self.assertEqual(result["roi"]["detector_calls_per_run"], 1)


class TestSoak(unittest.TestCase):
    def test_soak_runs_real_capture_and_inference_loop(self):
        from capture.file_source import FileFrameSource

        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                cv2.imwrite(str(Path(d) / f"{i}.jpg"), blank_frame(320, 240))
            slot = LatestFrameSlot(name="test_bench_soak_slot")
            capture = FileFrameSource(slot, d, camera_id="bench", target_fps=50)
            result = soak(
                capture,
                slot,
                ScriptedDetector([[]]),
                seconds=0.5,
                sample_every_sec=0.1,
                temp_reader=lambda: 55.0,
                throttle_reader=lambda: {"raw": "throttled=0x0"},
            )
        self.assertGreater(result["frames_captured"], 0)
        self.assertGreater(result["frames_inferred"], 0)
        self.assertLessEqual(result["frames_inferred"], result["frames_captured"])
        self.assertEqual(result["max_soc_temp_c"], 55.0)
        self.assertEqual(result["detector_type"], "ScriptedDetector")
        self.assertGreater(result["stream_fps"], 0)
        self.assertFalse(result["stream_stalled"])


class TestMain(unittest.TestCase):
    def test_hef_section_without_hailo_runs_and_prints_json(self):
        out = io.StringIO()
        with (
            patch.object(benchmark_edge, "_import_hailo_platform", side_effect=ImportError("no module")),
            contextlib.redirect_stdout(out),
        ):
            code = benchmark_edge.main(["--sections", "hef"])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertFalse(report["hef"]["available"])
        self.assertIn("platform", report)

    def test_rejects_unknown_section_and_bad_soak_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmark_edge.main(["--sections", "warp"])
            with self.assertRaises(SystemExit):
                benchmark_edge.main(["--sections", "soak", "--soak-source", "usb"])


if __name__ == "__main__":
    unittest.main()
