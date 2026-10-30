"""Phase 0 performance measurements on the Raspberry Pi 5 + Hailo-8
(docs/TE41ZL6C_MODES_PLAN.md). Replaces the plan's estimates with numbers.

Sections (--sections, comma-separated):

  hef     Decision D13. For the detector .hef in configs/inference_config.yaml
          (the node's only model): its input/output vstreams, whether it was
          compiled with on-chip NMS, and whether HailoDetector can read its
          outputs (the head split before the DFL decode:
          inference/yolo_postprocess.split_head_layout).
  models  Latency of the detector at 720p and 4MP input, letterbox alone,
          and SAHI on a 4MP frame (every tile vs. one ROI around a distant
          animal). Built with edge_main.build_detector, so it measures exactly
          what edge_main would run, and the report shows when that silently
          fell back from Hailo to the CPU.
  decode  Decode fps and CPU of the configured sub and main streams
          (tools.camera_probe.measure_stream).
  soak    The real capture thread (RtspReader, or FileFrameSource for
          --soak-source file:...) feeding edge_main._inference_loop for
          --soak-seconds: achieved inference fps, frames skipped, inference
          latency, CPU, SoC temperature over time and throttle flags.

Defaults run hef,models,decode. Run on the Pi from /opt/zovive with the .env
loaded, and stop zovive-detect.service first (the Hailo-8 normally accepts one
process at a time, and a running pipeline would skew every number):

    sudo systemctl stop zovive-detect
    set -a; . /opt/zovive/.env; set +a
    .venv/bin/python -m tools.benchmark_edge --backend hailo --out /tmp/bench.json
    .venv/bin/python -m tools.benchmark_edge --backend hailo --sections soak \\
        --soak-source sub --soak-seconds 600 --out /tmp/soak.json

--image PATH uses a real photo (resized per frame size) instead of noise, so
NMS/decode cost reflects a frame that actually contains animals.
The JSON report goes to stdout (and --out); log messages go to stderr.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import paths
from capture.file_source import FileFrameSource
from capture.rtsp_reader import RtspReader, build_rtsp_url
from edge_main import _inference_loop, build_detector
from health_main import _pi_throttle_status
from inference.base import Detection, Detector
from inference.preprocess import letterbox
from inference.sahi_slicer import SahiConfig, SahiDetector, compute_tiles
from inference.yolo_postprocess import split_head_layout
from pipeline.roi_cluster import cluster_box
from queues.detection_queue import DetectionQueue
from queues.frame_queue import LatestFrameSlot
from tools.camera_probe import measure_stream
from utils.config_loader import ConfigError, load
from utils.thermal_monitor import read_soc_temp_c

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CONFIG = 78

DEFAULT_SECTIONS = ("hef", "models", "decode")
ALL_SECTIONS = ("hef", "models", "decode", "soak")
# The two frame sizes the approved design runs YOLO on (detection stream,
# evidence snapshots).
FRAME_SIZES = {"720p": (1280, 720), "4mp": (2688, 1520)}
# One distant animal near the frame centre at 4MP: the case conditional SAHI
# (Phase 4) is meant for.
SAMPLE_FAR_ANIMAL_BOX = (1300.0, 700.0, 1380.0, 760.0)


# -- small pure helpers -----------------------------------------------------


def latency_stats(samples_sec: list[float]) -> dict:
    if not samples_sec:
        return {"count": 0}
    s = sorted(samples_sec)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

    return {
        "count": len(s),
        "mean_ms": round(1000.0 * sum(s) / len(s), 2),
        "p50_ms": round(1000.0 * pct(50), 2),
        "p95_ms": round(1000.0 * pct(95), 2),
        "max_ms": round(1000.0 * s[-1], 2),
    }


def time_calls(
    fn: Callable[[], Any], iterations: int, clock: Callable[[], float] = time.perf_counter
) -> list:
    samples = []
    for _ in range(iterations):
        t0 = clock()
        fn()
        samples.append(clock() - t0)
    return samples


def is_nms_format(format_order: Any) -> bool:
    """HailoRT names on-chip-NMS outputs HAILO_NMS, HAILO_NMS_BY_CLASS, ..."""
    return "NMS" in str(format_order).upper()


def split_heads_compatible(outputs: list[dict], num_classes: int, input_size: int) -> tuple[bool, str | None]:
    """Whether HailoDetector can read these outputs: the detection head cut
    before its DFL decode, three box + three class outputs
    (inference/yolo_postprocess.split_head_layout, the same check HailoDetector
    runs at load). Returns (ok, reason it is not)."""
    try:
        split_head_layout({o["name"]: o["shape"] for o in outputs}, num_classes, input_size)
    except (TypeError, ValueError) as exc:  # TypeError: an NMS output has no plain shape
        return False, str(exc)
    return True, None


def make_frame(width: int, height: int, image: np.ndarray | None = None) -> np.ndarray:
    if image is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class TimedDetector(Detector):
    """Wraps a Detector, recording each infer() call's duration."""

    def __init__(self, inner: Detector, clock: Callable[[], float] = time.perf_counter):
        self.inner = inner
        self.class_names = inner.class_names
        self.clock = clock
        self.samples: list[float] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            return len(self.samples)

    def infer(self, frame: Any) -> list[Detection]:
        t0 = self.clock()
        try:
            return self.inner.infer(frame)
        finally:
            with self._lock:
                self.samples.append(self.clock() - t0)


def _unwrap(detector: Detector) -> Detector:
    """build_detector wraps the detector in SahiDetector when SAHI is
    enabled in config; the benchmark times SAHI separately."""
    return detector.inner if isinstance(detector, SahiDetector) else detector


# -- hef (D13) ---------------------------------------------------------------


def _import_hailo_platform():
    import hailo_platform  # type: ignore

    return hailo_platform


def _hailort_version(hpf: Any) -> str | None:
    version = getattr(hpf, "__version__", None)
    if version:
        return str(version)
    for dist in ("hailort", "hailo_platform"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _vstream_info(info: Any) -> dict:
    fmt = getattr(info, "format", None)
    try:
        shape = [int(x) for x in getattr(info, "shape", ())]
    except (TypeError, ValueError):
        shape = None  # NMS outputs have no plain tensor shape
    order = getattr(fmt, "order", None)
    nms_shape = None
    if is_nms_format(order):
        # HailoRT 4.23 raises its own exception (not AttributeError) when
        # nms_shape is read on a raw-tensor output; its classes only exist on the Pi
        try:
            nms_shape = getattr(info, "nms_shape", None)
        except Exception:
            nms_shape = None
    return {
        "name": str(getattr(info, "name", "")),
        "shape": shape,
        "format_order": str(order),
        "format_type": str(getattr(fmt, "type", None)),
        "nms_shape": str(nms_shape) if nms_shape is not None else None,
    }


def inspect_hefs(hef_paths: dict[str, str], num_classes: dict[str, int], hpf: Any = None) -> dict:
    if hpf is None:
        try:
            hpf = _import_hailo_platform()
        except ImportError as exc:
            return {"available": False, "error": f"hailo_platform not importable: {exc}"}
    report: dict[str, Any] = {"available": True, "hailort_version": _hailort_version(hpf), "models": {}}
    for name, path in hef_paths.items():
        entry: dict[str, Any] = {"path": str(path)}
        try:
            hef = hpf.HEF(str(path))
            entry["network_groups"] = [str(n) for n in hef.get_network_group_names()]
            entry["inputs"] = [_vstream_info(i) for i in hef.get_input_vstream_infos()]
            entry["outputs"] = [_vstream_info(o) for o in hef.get_output_vstream_infos()]
            entry["on_chip_nms"] = any(is_nms_format(o["format_order"]) for o in entry["outputs"])
            if name == "detector":
                input_shape = entry["inputs"][0]["shape"] if entry["inputs"] else None
                ok, reason = split_heads_compatible(
                    entry["outputs"], num_classes.get(name, 0), int(input_shape[0]) if input_shape else 640
                )
                entry["hailo_detector_compatible"] = ok and not entry["on_chip_nms"]
                if reason:
                    entry["incompatible_because"] = reason
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["models"][name] = entry
    return report


# -- models ------------------------------------------------------------------


def bench_models(
    detector: Detector,
    iterations: int,
    image: np.ndarray | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    detector = _unwrap(detector)
    input_size = int(getattr(detector, "input_size", 640))
    out: dict[str, Any] = {
        "detector_type": type(detector).__name__,
        "detector_input_size": input_size,
        "detector": {},
        "letterbox": {},
    }
    for name, (w, h) in FRAME_SIZES.items():
        frame = make_frame(w, h, image)
        detector.infer(frame)  # first call at a new size can allocate
        out["detector"][name] = latency_stats(
            time_calls(lambda f=frame: detector.infer(f), iterations, clock)
        )
        out["letterbox"][name] = latency_stats(
            time_calls(lambda f=frame: letterbox(f, input_size), iterations, clock)
        )
    return out


def bench_sahi(
    detector: Detector,
    iterations: int,
    image: np.ndarray | None = None,
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """SAHI at 4MP: every tile + full-frame pass, vs. one ROI around a
    distant animal (pipeline/roi_cluster.cluster_box)."""
    w, h = FRAME_SIZES["4mp"]
    frame = make_frame(w, h, image)
    config = SahiConfig(tile_size=tile_size, overlap_ratio=overlap_ratio, run_full_frame_pass=True)

    def run(region: np.ndarray) -> dict:
        timed = TimedDetector(_unwrap(detector), clock)
        sahi = SahiDetector(timed, config)
        samples = time_calls(lambda: sahi.infer(region), iterations, clock)
        return {
            "region": f"{region.shape[1]}x{region.shape[0]}",
            "detector_calls_per_run": round(timed.calls / max(iterations, 1), 2),
            "latency": latency_stats(samples),
        }

    x1, y1, x2, y2 = cluster_box([SAMPLE_FAR_ANIMAL_BOX], frame.shape, min_size=tile_size)
    return {
        "tile_size": tile_size,
        "overlap_ratio": overlap_ratio,
        "full_frame_tiles": len(compute_tiles(w, h, tile_size, overlap_ratio)),
        "full_frame": run(frame),
        "roi": run(frame[y1:y2, x1:x2]),
    }


# -- soak --------------------------------------------------------------------


def config_for_stream(rtsp_cfg: dict, stream_key: str) -> dict:
    """A copy of rtsp_config whose sub_stream is `stream_key`: RtspReader
    always reads sub_stream, so this is how the soak test reads main."""
    cfg = copy.deepcopy(rtsp_cfg)
    cfg["camera"]["sub_stream"] = copy.deepcopy(rtsp_cfg["camera"][stream_key])
    return cfg


def soak(
    capture: Any,
    frame_slot: LatestFrameSlot,
    detector: Detector,
    seconds: float,
    camera_id: str = "bench",
    sample_every_sec: float = 5.0,
    temp_reader: Callable[[], float | None] = read_soc_temp_c,
    throttle_reader: Callable[[], dict] = _pi_throttle_status,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run `capture` (RtspReader / FileFrameSource writing into `frame_slot`)
    and the real edge_main._inference_loop together for `seconds`."""
    timed = TimedDetector(_unwrap(detector))
    detection_queue = DetectionQueue(maxsize=4)
    stop = threading.Event()
    worker = threading.Thread(
        target=_inference_loop,
        args=(timed, frame_slot, detection_queue, camera_id, stop),
        name="bench-inference",
        daemon=True,
    )
    throttle_start = throttle_reader()
    temps: list[dict] = []
    capture.start()
    t0, cpu0 = clock(), time.process_time()
    worker.start()
    try:
        while clock() - t0 < seconds:
            temps.append({"t_sec": round(clock() - t0, 1), "soc_temp_c": temp_reader()})
            time.sleep(max(0.0, min(sample_every_sec, seconds - (clock() - t0))))
    finally:
        stop.set()
        capture.stop()
        worker.join(timeout=5.0)
    elapsed = max(clock() - t0, 1e-6)
    cpu_cores = (time.process_time() - cpu0) / elapsed
    captured, inferred = frame_slot.last_seq, timed.calls
    readings = [t["soc_temp_c"] for t in temps if t["soc_temp_c"] is not None]
    stream = capture.health.snapshot()
    return {
        "seconds": round(elapsed, 1),
        "detector_type": type(timed.inner).__name__,
        "frames_captured": captured,
        "capture_fps": round(captured / elapsed, 2),
        "frames_inferred": inferred,
        "inference_fps": round(inferred / elapsed, 2),
        "frames_skipped": max(0, captured - inferred),
        "inference_latency": latency_stats(timed.samples),
        "process_cpu_cores": round(cpu_cores, 3),
        "max_soc_temp_c": max(readings) if readings else None,
        "temps": temps,
        "throttle_start": throttle_start,
        "throttle_end": throttle_reader(),
        "stream_fps": stream["fps"],
        "stream_stalled": stream["is_stalled"],
    }


# -- report ------------------------------------------------------------------


def platform_info() -> dict:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    try:
        info["onnxruntime"] = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        info["onnxruntime"] = None
    model_path = Path("/proc/device-tree/model")
    try:
        info["board"] = model_path.read_text(errors="replace").strip("\x00\n ")
    except OSError:
        info["board"] = None
    return info


def _section(report: dict, name: str, fn: Callable[[], Any]) -> None:
    """One failing section (missing model, no camera) never hides the others."""
    try:
        report[name] = fn()
    except Exception as exc:
        log.exception("benchmark section failed", extra={"section": name})
        report[name] = {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZOVIVE Phase 0 edge benchmark")
    parser.add_argument(
        "--sections", default=",".join(DEFAULT_SECTIONS), help=f"any of {','.join(ALL_SECTIONS)}"
    )
    parser.add_argument("--backend", choices=("hailo", "onnx"), default=None, help="override runtime.backend")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--image", type=Path, default=None, help="real photo to use instead of noise")
    parser.add_argument("--decode-seconds", type=float, default=10.0)
    parser.add_argument("--soak-seconds", type=float, default=120.0)
    parser.add_argument("--soak-source", default="sub", help="sub | main | file:PATH")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in sections if s not in ALL_SECTIONS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}")
    if args.soak_source not in ("sub", "main") and not args.soak_source.startswith("file:"):
        parser.error("--soak-source must be sub, main or file:PATH")

    image = None
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            parser.error(f"could not read image {args.image}")

    node_cfg = copy.deepcopy(load(paths.NODE_CONFIG))
    if args.backend:
        node_cfg.setdefault("runtime", {})["backend"] = args.backend
    inf_cfg = load(paths.INFERENCE_CONFIG)
    needs_camera = "decode" in sections or ("soak" in sections and not args.soak_source.startswith("file:"))
    rtsp_cfg: dict = {}
    if needs_camera:
        try:
            rtsp_cfg = load(paths.RTSP_CONFIG)
        except ConfigError as exc:
            print("invalid configuration:", file=sys.stderr)
            for issue in exc.issues:
                print(f"  {issue}", file=sys.stderr)
            return EXIT_CONFIG

    report: dict[str, Any] = {
        "benchmark_version": 1,
        "ts": time.time(),
        "platform": platform_info(),
        "backend_requested": node_cfg.get("runtime", {}).get("backend"),
        "model_version": inf_cfg.get("model_version"),
    }

    if "hef" in sections:
        det = inf_cfg["detector"]
        _section(
            report,
            "hef",
            lambda: inspect_hefs(
                {"detector": str(paths.model_file(det["hef_path"]))},
                {"detector": len(det["classes"])},
            ),
        )

    detector = None
    if "models" in sections or "soak" in sections:
        # Built after the hef section released its VDevice. Uses edge_main's
        # own builder, including its Hailo -> ONNX fallback.
        try:
            detector = build_detector(node_cfg, inf_cfg)
            detector.warmup()
        except Exception as exc:
            log.exception("could not build the detector")
            report["models"] = {"error": f"{type(exc).__name__}: {exc}"}

    if "models" in sections and detector is not None:
        _section(report, "models", lambda: bench_models(detector, args.iterations, image))
        _section(report, "sahi", lambda: bench_sahi(detector, max(3, args.iterations // 5), image))

    if "decode" in sections:
        cam = rtsp_cfg["camera"]

        def decode_all() -> dict:
            return {
                key: measure_stream(
                    build_rtsp_url(cam[key]["url"], cam["host"], cam["username"], cam["password"]),
                    args.decode_seconds,
                )
                for key in ("sub_stream", "main_stream")
            }

        _section(report, "decode", decode_all)

    if "soak" in sections and detector is not None:

        def run_soak() -> dict:
            slot = LatestFrameSlot(name="bench_frame_slot")
            if args.soak_source.startswith("file:"):
                capture = FileFrameSource(slot, args.soak_source[len("file:") :], camera_id="bench")
            else:
                key = {"sub": "sub_stream", "main": "main_stream"}[args.soak_source]
                capture = RtspReader(slot, camera_id="bench", config=config_for_stream(rtsp_cfg, key))
            result = soak(capture, slot, detector, args.soak_seconds)
            result["source"] = args.soak_source
            return result

        _section(report, "soak", run_soak)

    if detector is not None:
        try:
            detector.close()
        except Exception:  # noqa: BLE001 - shutting down anyway
            log.debug("detector close failed", exc_info=True)

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
