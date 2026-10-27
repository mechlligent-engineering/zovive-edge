"""Reads the CP Plus camera's RTSP sub-stream in its own thread and
writes each frame into a `LatestFrameSlot` (see queues/frame_queue.py).

Deliberately does NOT queue frames: only the newest one matters for
detection, and a queue here is exactly how RTSP lag turns into growing
latency. On disconnect it retries with `ReconnectPolicy` backoff and
updates `StreamHealth` so health_main.py can report it.

Runs the sub-stream by default (low resolution -> cheap CPU decode,
since the Pi 5 has no H.264 hardware decoder). `grab_main_stream_frame`
is used on-trigger only, for a single high-res snapshot.

Reads at whatever rate the camera encoder sends and never sleeps between
frames. H.264 frames cannot be skipped without decoding them, and the
FFmpeg backend ignores CAP_PROP_BUFFERSIZE, so a reader that throttles
below the encoder rate falls further behind real time every second. The
frame rate is therefore set on the camera (configs/rtsp_config.yaml
documents the expected value; tools/camera_probe.py checks it).
"""

from __future__ import annotations

import logging
import threading
import time
from urllib.parse import quote

import cv2

import paths
from capture.reconnect import ReconnectPolicy
from capture.stream_health import StreamHealth
from queues.frame_queue import LatestFrameSlot
from utils.config_loader import load

log = logging.getLogger(__name__)

# Longer than reconnect.stall_timeout_sec (5 s) in rtsp_config.yaml, and they
# do different jobs: StreamHealth.is_stalled() reports a silent camera to the
# heartbeat while this read is still waiting; at this timeout the read fails
# and the reader reconnects.
_FFMPEG_TIMEOUT_MSEC = 10_000


def build_rtsp_url(template: str, host: str, user: str, password: str) -> str:
    # Percent-encode the credentials (RFC 3986 userinfo) so '@', ':', '%' and
    # the like reach the camera intact: FFmpeg decodes them before it
    # authenticates. Letters, digits and - . _ ~ are left as they are.
    user, password = quote(user, safe=""), quote(password, safe="")
    return template.format(host=host, user=user, pass_=password, **{"pass": password})


def open_rtsp_capture(url: str) -> cv2.VideoCapture | None:
    """Open an RTSP (or file) URL with bounded open/read timeouts, so an
    unreachable camera returns within ~10 s instead of blocking on the OS
    TCP timeout (~2 min), which would otherwise look like a hung thread
    to the supervisor. Returns None if it could not be opened."""
    params = []
    for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, prop):
            params += [getattr(cv2, prop), _FFMPEG_TIMEOUT_MSEC]
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    # Keep OpenCV's internal buffer at 1 frame where the backend honours it.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


class RtspReader:
    def __init__(
        self,
        frame_slot: LatestFrameSlot,
        camera_id: str = "cam01",
        config: dict | None = None,
    ):
        self.frame_slot = frame_slot
        self.camera_id = camera_id
        self.cfg = config or load(paths.RTSP_CONFIG)
        cam = self.cfg.get("camera", {})
        self.sub_url = build_rtsp_url(cam["sub_stream"]["url"], cam["host"], cam["username"], cam["password"])
        self.main_url = build_rtsp_url(
            cam["main_stream"]["url"], cam["host"], cam["username"], cam["password"]
        )
        stall_timeout = float(self.cfg.get("reconnect", {}).get("stall_timeout_sec", 5.0))
        self.health = StreamHealth(camera_id, stall_timeout_sec=stall_timeout)
        self._reconnect_policy = ReconnectPolicy.from_config(self.cfg)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Wall-clock time of the latest loop iteration, connected or not;
        # watchdog/supervisor.py uses it to tell "stuck" from "camera down".
        self.loop_ts: float | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name=f"rtsp-reader-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)

    def _open_capture(self) -> cv2.VideoCapture | None:
        return open_rtsp_capture(self.sub_url)

    def run(self) -> None:
        """The capture loop. start() runs it on a private thread;
        edge_main.py runs it under watchdog/supervisor.py instead."""
        # The open capture lives in a one-item list so it is released here
        # even when _loop raises, instead of leaking an FFmpeg handle per crash.
        cap_holder: list = [None]
        try:
            self._loop(cap_holder)
        finally:
            if cap_holder[0] is not None:
                cap_holder[0].release()

    def _loop(self, cap_holder: list) -> None:
        # Set by a failed read: the next successful open is a reconnect even
        # when its first attempt succeeds (no backoff attempt was counted).
        lost_stream = False
        while not self._stop_event.is_set():
            self.loop_ts = time.time()
            cap = cap_holder[0]
            if cap is None:
                cap = cap_holder[0] = self._open_capture()
                if cap is None:
                    delay = self._reconnect_policy.sleep_next()
                    log.warning(
                        "rtsp connect failed, backing off",
                        extra={"camera_id": self.camera_id, "backoff_sec": round(delay, 2)},
                    )
                    continue
                else:
                    if lost_stream or self._reconnect_policy.attempt_count > 0:
                        self.health.record_reconnect()
                    lost_stream = False
                    self._reconnect_policy.reset()
                    log.info("rtsp connected", extra={"camera_id": self.camera_id})

            # No throttle here (see module docstring): read at the encoder rate.
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("rtsp read failed, reconnecting", extra={"camera_id": self.camera_id})
                cap.release()
                cap_holder[0] = None
                self.health.record_stall()
                lost_stream = True
                continue

            self.health.record_frame()
            self.frame_slot.put(frame, camera_id=self.camera_id)

    def grab_main_stream_frame(self, timeout_sec: float = 3.0):
        """One-shot high-res grab from the main stream, used on trigger
        for the alert snapshot. Opens and closes its own capture so it
        never interferes with the continuous sub-stream reader.
        """
        cap = open_rtsp_capture(self.main_url)
        if cap is None:
            log.warning("main-stream grab failed to open", extra={"camera_id": self.camera_id})
            return None
        try:
            deadline = time.monotonic() + timeout_sec
            frame = None
            while time.monotonic() < deadline:
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                    break
            return frame
        finally:
            cap.release()
