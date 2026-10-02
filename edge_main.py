"""Entry point for the detection process (systemd: zovive-detect.service).

Three threads, deliberately kept separate so a slow stage never stalls
an earlier one:

    capture thread   (RtspReader or FileFrameSource, started internally)
        writes into `frame_slot` (queues/frame_queue.py — latest-frame
        only, never a backlog)
            v
    inference thread (this file: `_inference_loop`)
        Detector.infer() on the newest frame, pushes a DetectionBatch
        into `detection_queue` (bounded, drop-oldest)
            v
    pipeline thread  (this file: `_pipeline_loop`)
        tracker -> per-track state machine -> zone filter -> best-crop
        collection -> stage2 classification -> alert dispatch. A confirmed
        animal that is far away is first zoomed in on (optical zoom +
        autofocus only; the camera never pans or tilts), re-detected and
        re-classified in the zoomed view, and alerted with that snapshot

transfer_main.py and health_main.py are separate processes/systemd
units on purpose (see their own docstrings): nothing in this file
talks to the network directly except through the DB outbox.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

import paths
from camera_control.camera_api import FakeLensCamera, OnvifLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import WIDE_VIEW, ZOOMED_VIEW, AnimalZoomController, ZoomState
from capture.file_source import FileFrameSource
from capture.rtsp_reader import RtspReader
from db.init_db import init_db
from db.migrations import migrate
from db.outbox import set_video_path
from inference.base import Classifier, Detector
from inference.bytetrack_wrapper import IouTracker
from inference.onnx_inference import OnnxClassifier, OnnxDetector
from inference.preprocess import crop_box
from inference.sahi_slicer import SahiConfig, SahiDetector, is_sahi_enabled
from logger_setup import configure_root, get_logger
from pipeline.alert_dispatcher import AlertDispatcher
from pipeline.best_snapshot import BestSnapshotStore
from pipeline.clip_extractor import ClipExtractor
from pipeline.motion_gate import MotionGate
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.stage2_verifier import VerifiedResult, verify_track
from pipeline.track_state_machine import TrackState, TrackStateMachine
from pipeline.zone_filter import ZoneFilter
from queues.alert_queue import AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from queues.frame_queue import LatestFrameSlot
from queues.queue_monitor import snapshot as queues_snapshot
from utils.config_loader import ConfigError, load
from watchdog import sd_notify
from watchdog.edge_status import EdgeStatus, write_status_file
from watchdog.restart_history import record_exit, record_start
from watchdog.supervisor import RestartPolicy, SupervisedThread, Supervisor

log = get_logger(__name__)

_shutdown_event = threading.Event()

EXIT_OK = 0
# A worker hung or ran out of restarts; systemd (Restart=always) restarts us.
EXIT_SUPERVISOR = 70
# Configuration invalid, e.g. ZOVIVE_CAMERA_USERNAME/PASSWORD missing (sysexits EX_CONFIG).
EXIT_CONFIG = 78
SUPERVISOR_POLL_SEC = 0.5


def _handle_signal(signum, frame):  # noqa: ARG001
    log.info("shutdown signal received", extra={"signal": signum})
    _shutdown_event.set()


def _build_onnx_backend(det_cfg: dict, cls_cfg: dict) -> tuple[Detector, Classifier]:
    detector: Detector = OnnxDetector(
        det_cfg["onnx_path"],
        det_cfg["classes"],
        input_size=int(det_cfg.get("input_size", 640)),
        conf_threshold=float(det_cfg.get("confidence_threshold", 0.45)),
        iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
    )
    classifier: Classifier = OnnxClassifier(
        cls_cfg["onnx_path"], cls_cfg["classes"], input_size=int(cls_cfg.get("input_size", 224))
    )
    return detector, classifier


def build_detector_and_classifier(node_cfg: dict, inf_cfg: dict) -> tuple[Detector, Classifier]:
    backend = node_cfg.get("runtime", {}).get("backend", "onnx")
    det_cfg = inf_cfg["detector"]
    cls_cfg = inf_cfg["classifier"]

    if backend == "hailo":
        # Imported lazily: hailo_platform isn't installed off-Pi, and we
        # don't want importing this module to fail on a laptop just
        # because the ONNX branch was the one actually taken.
        from inference.base import InferenceBackendError
        from inference.hailo_inference import HailoClassifier, HailoDetector

        try:
            detector: Detector = HailoDetector(
                det_cfg["hef_path"],
                det_cfg["classes"],
                conf_threshold=float(det_cfg.get("confidence_threshold", 0.45)),
                iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
            )
            classifier: Classifier = HailoClassifier(cls_cfg["hef_path"], cls_cfg["classes"])
        except InferenceBackendError:
            # CPU fallback: HailoRT missing, no device present, or a bad
            # .hef — rather than crash-looping the whole process (systemd
            # would just restart it into the same failure), fall back to
            # the ONNX backend on the Pi's CPU so detection keeps running
            # in a degraded-but-alive state until the Hailo path is fixed.
            log.exception(
                "hailo backend failed to initialize; falling back to onnx (CPU) backend",
                extra={"hef_detector": det_cfg.get("hef_path"), "hef_classifier": cls_cfg.get("hef_path")},
            )
            detector, classifier = _build_onnx_backend(det_cfg, cls_cfg)
    else:
        detector, classifier = _build_onnx_backend(det_cfg, cls_cfg)

    if is_sahi_enabled(inf_cfg):
        sahi_cfg = SahiConfig.from_config(inf_cfg)
        log.info(
            "SAHI tiled inference enabled",
            extra={"tile_size": sahi_cfg.tile_size, "overlap_ratio": sahi_cfg.overlap_ratio},
        )
        detector = SahiDetector(detector, sahi_cfg)

    return detector, classifier


def build_camera(node_cfg: dict, rtsp_cfg: dict):
    environment = node_cfg.get("runtime", {}).get("environment", "dev")
    if environment == "pi":
        return OnvifLensCamera.from_config(rtsp_cfg)
    log.warning("using FakeLensCamera (runtime.environment != 'pi'); no real zoom/focus will happen")
    return FakeLensCamera()


def _inference_loop(
    detector: Detector,
    frame_slot: LatestFrameSlot,
    detection_queue: DetectionQueue,
    camera_id: str,
    stop_event: threading.Event,
    max_cycles: int | None = None,
    status: EdgeStatus | None = None,
) -> None:
    last_seq = 0
    cycles = 0
    while not stop_event.is_set():
        if status is not None:
            status.inference_loop_tick()
        env = frame_slot.wait_for_next(last_seq, timeout=1.0)
        if env is None:
            continue
        last_seq = env.seq
        if status is not None:
            status.frame_received()
        try:
            detections = detector.infer(env.frame)
        except Exception:
            log.exception("detector.infer failed; skipping frame")
            if status is not None:
                status.inference_failed()
            continue
        if status is not None:
            status.inference_ok(len(detections))
        detection_queue.put(
            DetectionBatch(
                seq=env.seq, timestamp=env.timestamp, frame=env.frame, detections=detections, camera_id=camera_id
            )
        )
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
    log.info("inference loop stopped")


def _classify_and_dispatch(
    classifier: Classifier,
    cls_cfg: dict,
    crops: list,
    snapshot_image,
    track_id: str,
    dispatcher: AlertDispatcher,
    view_name: str,
    model_version: str,
) -> tuple[VerifiedResult, str | None]:
    result = verify_track(
        classifier,
        crops,
        min_confidence=float(cls_cfg.get("min_confidence", 0.55)),
        strategy=cls_cfg.get("voting_strategy", "average"),
    )
    event_id = dispatcher.dispatch(
        track_id=track_id,
        species=result.species,
        confidence=result.confidence,
        snapshot_image=snapshot_image,
        preset_name=view_name,
        model_version=model_version,
    )
    return result, event_id


def _reset_tracking(tracker: IouTracker, track_sm: TrackStateMachine, snapshot_store: BestSnapshotStore) -> None:
    """The lens is moving: every box from the previous view is meaningless."""
    tracker.reset()
    track_sm.sweep_lost(set())
    for tid in list(snapshot_store.active_track_ids()):
        snapshot_store.discard(tid)


def _pipeline_loop(
    detection_queue: DetectionQueue,
    classifier: Classifier,
    zoom: AnimalZoomController,
    tracker: IouTracker,
    track_sm: TrackStateMachine,
    zone_filter: ZoneFilter,
    motion_gate: MotionGate,
    snapshot_store: BestSnapshotStore,
    dispatcher: AlertDispatcher,
    inf_cfg: dict,
    model_version: str,
    stop_event: threading.Event,
    max_cycles: int | None = None,
    clip_extractor: ClipExtractor | None = None,
    status: EdgeStatus | None = None,
) -> None:
    cls_cfg = inf_cfg["classifier"]
    max_crops = int(cls_cfg.get("max_crops_per_track", 5))
    cycles = 0
    while not stop_event.is_set():
        if status is not None:
            # Every iteration, idle or not: a stale tick means this thread is stuck or dead.
            status.pipeline_tick(zoom_state=zoom.state.value)
        batch = detection_queue.get(timeout=1.0)
        if batch is None:
            continue

        motion_gate.score(batch.frame)  # advisory metric only; see motion_gate.py
        if clip_extractor is not None:
            # Fed every cycle regardless of whether anything is
            # recording — cheap (ring-buffer append), and it's what lets
            # a clip started below include the seconds *before* the
            # track was confirmed (pipeline/ram_buffer.py).
            clip_extractor.offer_frame(batch.frame, batch.timestamp)

        expired = zoom.tick()
        if expired is not None:
            # Zoomed re-detection found nothing: alert with the wide-view
            # evidence captured when the zoom started.
            snapshot = expired.wide_frame if expired.wide_frame is not None else batch.frame
            result, event_id = _classify_and_dispatch(
                classifier, cls_cfg, expired.wide_crops, snapshot, expired.track_id,
                dispatcher, WIDE_VIEW, model_version,
            )
            zoom.complete(event_id, result.species, result.confidence)
            if event_id and clip_extractor is not None:
                clip_extractor.start(event_id=event_id, track_id=expired.track_id, camera_id=batch.camera_id)

        if not zoom.frames_usable():
            # Lens zooming or focusing: frames are blurred and the scene is
            # rescaling, so skip them and start tracking fresh afterwards.
            _reset_tracking(tracker, track_sm, snapshot_store)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            continue

        view = zoom.view_name
        frame_shape = batch.frame.shape[:2]
        detections = zone_filter.filter(batch.detections, view, frame_shape)

        tracks = tracker.update(detections)
        if status is not None:
            status.pipeline_tick(active_tracks=len(tracks))
        active_ids = {t.track_id for t in tracks}
        track_sm.sweep_lost(active_ids)
        for tid in list(snapshot_store.active_track_ids()):
            if tid not in active_ids:
                snapshot_store.discard(tid)

        engaged_this_cycle = False
        for track in tracks:
            rec = track_sm.advance(track)

            crop = crop_box(batch.frame, track.box)
            snapshot_store.offer(track.track_id, crop, batch.frame)

            if rec.state != TrackState.CONFIRMED:
                continue

            if view == WIDE_VIEW:
                known = zoom.match_known(track.class_name, track.box)
                if known is not None:
                    # Same animal already handled before/while zooming, back
                    # under a new track id: don't re-zoom or re-alert it.
                    track_sm.mark_classified(track.track_id, known.species or track.class_name, known.confidence)
                    if known.event_id:
                        track_sm.mark_alerted(track.track_id, known.event_id)
                    continue

                magnification = zoom.plan(track.box, frame_shape)
                if magnification is not None:
                    best = snapshot_store.best(track.track_id)
                    zoom.engage(
                        track.track_id,
                        track.class_name,
                        track.box,
                        magnification,
                        wide_crops=snapshot_store.top(track.track_id, max_crops),
                        wide_frame=best.full_frame if best else batch.frame,
                    )
                    engaged_this_cycle = True
                    continue  # classified after zoom + focus

            crops = snapshot_store.top(track.track_id, max_crops)
            best = snapshot_store.best(track.track_id)
            snapshot_image = best.full_frame if best else batch.frame
            result, event_id = _classify_and_dispatch(
                classifier, cls_cfg, crops, snapshot_image, track.track_id, dispatcher, view, model_version
            )
            track_sm.mark_classified(track.track_id, result.species, result.confidence)
            if event_id:
                track_sm.mark_alerted(track.track_id, event_id)
                if clip_extractor is not None:
                    clip_extractor.start(event_id=event_id, track_id=track.track_id, camera_id=batch.camera_id)

            if view == ZOOMED_VIEW and zoom.state == ZoomState.ZOOMED:
                # Final zoomed snapshot stored: zoom back out.
                zoom.complete(event_id, result.species, result.confidence)

        if engaged_this_cycle:
            # These animals get new track ids after the zoom; remember them
            # so they aren't alerted a second time.
            for track in tracks:
                rec = track_sm.get(track.track_id)
                if rec is not None and rec.state == TrackState.ALERTED:
                    zoom.remember(track.box, track.class_name, rec.species, rec.confidence, rec.event_id)

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
    log.info("pipeline loop stopped")


def run(source_override: str | None = None, max_cycles: int | None = None) -> int:
    """Runs until shutdown. Returns the process exit code: EXIT_OK, or
    EXIT_SUPERVISOR when a worker hung or used up its restarts."""
    configure_root("detect")
    history = record_start(paths.RESTART_HISTORY_PATH, time.time())
    try:
        return _run(history, source_override, max_cycles)
    except BaseException as exc:
        # Startup failures (missing model, bad config) and anything else that
        # escapes: record why, then let it propagate (systemd restarts us).
        if history.get("running"):
            reason = f"fatal: {type(exc).__name__}: {exc}"[:300]
            record_exit(paths.RESTART_HISTORY_PATH, history, reason, 1, time.time())
        raise


def _run(history: dict, source_override: str | None, max_cycles: int | None) -> int:
    conn = init_db()
    migrate(conn)

    node_cfg = load(paths.NODE_CONFIG)
    rtsp_cfg = load(paths.RTSP_CONFIG)
    inf_cfg = load(paths.INFERENCE_CONFIG)
    tracking_cfg = load(paths.TRACKING_CONFIG)
    zones_cfg = load(paths.DETECTION_ZONES_CONFIG)

    camera_id = node_cfg.get("node", {}).get("camera_id", "cam01")
    model_version = inf_cfg.get("model_version", "unknown")

    detector, classifier = build_detector_and_classifier(node_cfg, inf_cfg)
    detector.warmup()
    classifier.warmup()

    frame_slot = LatestFrameSlot()

    if source_override:
        kind, _, value = source_override.partition(":")
        if kind != "file":
            raise ValueError("--source must be 'file:<path>' or omitted for the real RTSP camera")
        capture = FileFrameSource(frame_slot, value, camera_id=camera_id)
    else:
        capture = RtspReader(frame_slot, camera_id=camera_id, config=rtsp_cfg)

    lens_camera = build_camera(node_cfg, rtsp_cfg)
    camera_modes_cfg = load(paths.CAMERA_MODES_CONFIG)
    mode_manager = ModeManager.from_config(camera_modes_cfg)
    zoom = AnimalZoomController.from_config(lens_camera, camera_modes_cfg, mode_manager=mode_manager)
    # Start from a known lens position: fully wide and focused.
    lens_camera.zoom_to(zoom.plan_config.wide_zoom_level)
    lens_camera.trigger_autofocus()

    tracker = IouTracker.from_config(
        tracking_cfg, presence_window=int(inf_cfg["detector"].get("window_size", 5))
    )
    stage1 = Stage1Gate(Stage1Config.from_config(inf_cfg))
    track_sm = TrackStateMachine(stage1)
    zone_filter = ZoneFilter(zones_cfg)
    motion_gate = MotionGate()
    snapshot_store = BestSnapshotStore()

    detection_queue = DetectionQueue()
    alert_queue = AlertQueue()
    dispatcher = AlertDispatcher(alert_queue, node_config=node_cfg)

    # on_clip_ready=set_video_path is the only link between clip_extractor
    # and the DB: it never touches db.outbox directly otherwise, matching
    # how alert_dispatcher.py owns the actual insert_event() call.
    clip_extractor = ClipExtractor(on_clip_ready=set_video_path)

    watchdog_cfg = load(paths.WATCHDOG_CONFIG, required=False).get("watchdog", {})
    status_interval = max(1.0, float(watchdog_cfg.get("edge_status_write_interval_sec", 5)))
    edge_status = EdgeStatus(camera_id)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def reset_pipeline() -> None:
        # Runs on the main thread after the pipeline thread died, so nothing
        # else is touching this state. A zoom in progress is dropped without
        # an alert and the lens goes back to wide.
        _reset_tracking(tracker, track_sm, snapshot_store)
        zoom.abort()

    supervisor = Supervisor(RestartPolicy.from_config(watchdog_cfg))
    supervisor.add(SupervisedThread("capture", capture.run, liveness=lambda: capture.loop_ts))
    supervisor.add(
        SupervisedThread(
            "inference",
            lambda: _inference_loop(
                detector, frame_slot, detection_queue, camera_id, _shutdown_event, max_cycles, edge_status
            ),
            liveness=edge_status.inference_loop_ts,
        )
    )
    supervisor.add(
        SupervisedThread(
            "pipeline",
            lambda: _pipeline_loop(
                detection_queue, classifier, zoom, tracker, track_sm, zone_filter, motion_gate,
                snapshot_store, dispatcher, inf_cfg, model_version, _shutdown_event, max_cycles,
                clip_extractor, edge_status,
            ),
            liveness=edge_status.pipeline_loop_ts,
            on_restart=reset_pipeline,
        )
    )

    supervisor.start_all()
    log.info("capture started", extra={"camera_id": camera_id, "backend": node_cfg["runtime"]["backend"]})
    sd_notify.ready()

    reason, exit_code = "shutdown requested", EXIT_OK
    try:
        reason, exit_code = _supervise(
            supervisor, edge_status, capture, history, status_interval, _shutdown_event
        )
    finally:
        _shutdown_event.set()
        sd_notify.stopping()
        capture.stop()
        supervisor.join_all(timeout_each=5.0)
        if exit_code == EXIT_OK:
            # Finish in-progress event clips with whatever post-roll they
            # have. Skipped when escalating: the process is about to be torn
            # down and a half-written MP4 is worse than none.
            clip_extractor.flush_all()
            detector.close()
            classifier.close()
        # On escalation a hung worker may hold the NPU or a camera socket;
        # closing those could hang too, so they are left to process exit.
        if zoom.state != ZoomState.WIDE:
            lens_camera.zoom_to(zoom.plan_config.wide_zoom_level)
        _write_edge_status(edge_status, capture, supervisor, history)
        record_exit(paths.RESTART_HISTORY_PATH, history, reason, exit_code, time.time())
        log.info("edge_main stopped", extra={"reason": reason, "exit_code": exit_code})
    return exit_code


def _supervise(
    supervisor: Supervisor,
    edge_status: EdgeStatus,
    capture,
    history: dict,
    status_interval: float,
    stop_event: threading.Event,
) -> tuple[str, int]:
    """Main-thread loop: restart crashed workers, escalate hung or exhausted
    ones, publish status, and keep the systemd watchdog fed. Returns
    (reason, exit code)."""
    next_status_write = 0.0
    while not stop_event.is_set():
        escalation = supervisor.check()
        if escalation:
            log.critical("supervisor escalating; exiting for systemd restart", extra={"reason": escalation})
            return escalation, EXIT_SUPERVISOR
        if supervisor.stopped("pipeline"):
            return "pipeline finished (--max-cycles)", EXIT_OK
        # Fed only while check() runs and finds nothing to escalate: if this
        # thread hangs, systemd's WatchdogSec kills and restarts the process.
        sd_notify.watchdog_ping()
        if time.monotonic() >= next_status_write:
            _write_edge_status(edge_status, capture, supervisor, history)
            next_status_write = time.monotonic() + status_interval
        stop_event.wait(SUPERVISOR_POLL_SEC)
    return "shutdown requested", EXIT_OK


def _write_edge_status(
    edge_status: EdgeStatus, capture, supervisor: Supervisor | None = None, history: dict | None = None
) -> None:
    """Publish this process's health for health_main.py. A failed write
    (disk full, permissions) is logged, never fatal: health_main reports
    the resulting stale file as edge_process_alive = false."""
    try:
        snapshot = edge_status.snapshot(
            stream=capture.health.snapshot(),
            queues=queues_snapshot(),
            supervisor=supervisor.snapshot() if supervisor else None,
            process=history,
        )
        write_status_file(paths.EDGE_STATUS_PATH, snapshot)
    except (OSError, TypeError, ValueError):
        log.exception("failed to write edge status file", extra={"path": str(paths.EDGE_STATUS_PATH)})


def main() -> None:
    parser = argparse.ArgumentParser(description="ZOVIVE edge detection process")
    parser.add_argument(
        "--source",
        default=None,
        help="override the camera source, e.g. 'file:/path/to/video.mp4' for dev/testing without a camera",
    )
    parser.add_argument("--max-cycles", type=int, default=None, help="stop after N pipeline cycles (testing)")
    args = parser.parse_args()
    try:
        exit_code = run(source_override=args.source, max_cycles=args.max_cycles)
    except ConfigError as exc:
        # Missing camera credentials or a bad config file: one clear line
        # naming the file, key and variable (never its value), no traceback.
        log.critical("refusing to start: invalid configuration", extra={"problems": [str(i) for i in exc.issues]})
        logging.shutdown()
        sys.exit(EXIT_CONFIG)
    if exit_code != EXIT_OK:
        # os._exit, not sys.exit: a hung worker thread stuck in native code
        # (Hailo, FFmpeg) can block normal interpreter shutdown.
        logging.shutdown()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
