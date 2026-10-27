"""Phase 0 check of a real CP-UNC-TE41ZL6C-VMDS-Q against what the design
assumes (docs/TE41ZL6C_MODES_PLAN.md, Phase 0): stream count, resolution,
fps and codec per stream, ONVIF snapshot support and resolution, zoom
range, zoom/focus status reporting, and which motion event topics the
firmware publishes.

Read-only by default: ONVIF queries, RTSP reads and snapshot fetches. It
never changes encoder, imaging or network settings. The lens moves only
with --lens-test (zoom fully in and back out, then autofocus), which
measures lens travel and settle times.

RTSP is measured on the two configured streams only (sub_stream, main_stream:
the URLs the edge service reads). Every ONVIF profile is still discovered and
reported with its stream URI, but those URIs are never opened (see
stream_identity()).

Run on the Pi from /opt/zovive with the node's .env loaded, and stop
zovive-detect.service first so the RTSP numbers aren't shared with it:

    sudo systemctl stop zovive-detect
    set -a; . /opt/zovive/.env; set +a
    .venv/bin/python -m tools.camera_probe --out /tmp/camera_probe.json
    .venv/bin/python -m tools.camera_probe --lens-test --out /tmp/camera_probe_lens.json

Everything comes from configs/rtsp_config.yaml, the same file the edge
service reads: the camera address from ZOVIVE_CAMERA_HOST (or its default
there), RTSP credentials from ZOVIVE_CAMERA_USERNAME/PASSWORD and ONVIF
ones from ZOVIVE_ONVIF_USERNAME/PASSWORD. Every URL in the report has its
password removed.
The JSON report goes to stdout (and --out); log messages go to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import cv2
import numpy as np

import paths
from camera_control.camera_api import OnvifLensCamera
from capture.rtsp_reader import build_rtsp_url, open_rtsp_capture
from utils.config_loader import ConfigError, load, redact_url

EXIT_OK = 0
EXIT_CONFIG = 78  # same as edge_main.py: invalid configuration / missing credentials

CONFIGURED_STREAMS = ("sub_stream", "main_stream")
# A measured fps within this fraction (or 1 fps) of the configured value is a match.
FPS_TOLERANCE = 0.15
# Rest between a settled zoom move and the next zoom command. The CP-UNC-TE41ZL6C
# accepts an AbsoluteMove sent sooner without a fault but never moves the lens.
LENS_COMMAND_COOLDOWN_SEC = 15


def _fourcc(value: float) -> str | None:
    code = int(value or 0)
    if code <= 0:
        return None
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    return text or None


def measure_stream(
    url: str,
    seconds: float,
    opener: Callable[[str], Any] = open_rtsp_capture,
    clock: Callable[[], float] = time.monotonic,
    cpu_clock: Callable[[], float] = time.process_time,
    first_frame_timeout_sec: float = 10.0,
) -> dict:
    """Open `url`, time the first frame, then read for `seconds` and report
    resolution, codec, measured fps and the decode CPU it cost. CPU is the
    whole process's CPU time (FFmpeg decodes on its own threads) divided by
    wall time, so 1.0 = one fully busy core."""
    result: dict[str, Any] = {"url": redact_url(url), "opened": False}
    t0 = clock()
    cap = opener(url)
    if cap is None:
        return result
    try:
        result["opened"] = True
        result["open_sec"] = round(clock() - t0, 3)

        frame = None
        deadline = clock() + first_frame_timeout_sec
        while frame is None and clock() < deadline:
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            result["first_frame_sec"] = None
            return result
        result["first_frame_sec"] = round(clock() - t0, 3)
        result["height"], result["width"] = (int(v) for v in frame.shape[:2])
        result["codec"] = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        result["reported_fps"] = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)

        frames = failures = 0
        start, cpu_start = clock(), cpu_clock()
        while clock() - start < seconds:
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frames += 1
                continue
            failures += 1
            if failures >= 3:
                break
        elapsed = max(clock() - start, 1e-6)
        result.update(
            seconds=round(elapsed, 2),
            frames=frames,
            read_failures=failures,
            measured_fps=round(frames / elapsed, 2),
            decode_cpu_cores=round((cpu_clock() - cpu_start) / elapsed, 3),
        )
        return result
    finally:
        cap.release()


# Query parameters the CP Plus firmware appends to every ONVIF GetStreamUri
# answer (".../video/live?channel=1&subtype=1&unicast=true&proto=Onvif"). They
# set up the RTSP session, not which stream is sent. With proto=Onvif the
# camera answers 401 to the credentials the configured URLs open with
# (verified on the bench), so the probe never opens ONVIF URIs.
_ONVIF_SESSION_PARAMS = frozenset({("unicast", "true"), ("proto", "onvif")})

StreamIdentity = tuple[str, tuple[tuple[str, str], ...]]


def stream_identity(uri: str) -> StreamIdentity | None:
    """What selects the stream in `uri`: its path and query parameters
    (case-insensitive, in any order), minus the ONVIF session parameters
    above. Every other parameter still counts, including unicast or proto
    with another value. Host, port and credentials are ignored. None if
    `uri` can't be parsed."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return None
    params = ((k.lower(), v.lower()) for k, v in parse_qsl(parts.query, keep_blank_values=True))
    return parts.path.lower(), tuple(sorted(p for p in params if p not in _ONVIF_SESSION_PARAMS))


def collect_onvif(camera: Any, configured: dict[str, StreamIdentity | None]) -> dict:
    """ONVIF inventory for the report. Every profile is listed with its stream
    URI as the camera sent it (credentials masked) and `configured_stream`:
    the key in `configured` ({stream key: stream_identity of its URL}) whose
    stream the profile carries, or None for one the Pi doesn't use."""
    profiles = camera.describe_media_profiles()
    for profile in profiles:
        token = profile.get("token")
        uri = camera.get_stream_uri(token)
        snapshot_uri = camera.get_snapshot_uri(token)
        identity = stream_identity(uri) if uri else None
        profile["stream_uri"] = redact_url(uri) if uri else None
        profile["configured_stream"] = next(
            (key for key, ident in configured.items() if identity is not None and ident == identity), None
        )
        profile["snapshot_uri"] = redact_url(snapshot_uri) if snapshot_uri else None
    inventory = {
        "device": camera.get_device_info(),
        "profiles": profiles,
        "zoom_limits": camera.get_zoom_limits(),
        "zoom_position": camera.get_zoom(),
        "move_status": camera.get_move_status(),
        "focus_status": camera.get_focus_status(),
        "imaging": camera.get_imaging_options(),
        "event_topics": camera.get_event_topics(),
    }
    return inventory


def largest_profile_token(profiles: list[dict]) -> str | None:
    """The highest-resolution profile: the 4MP evidence stream on this camera."""
    sized = [p for p in profiles if p.get("width") and p.get("height")]
    if not sized:
        return profiles[0].get("token") if profiles else None
    return max(sized, key=lambda p: int(p["width"]) * int(p["height"])).get("token")


def check_snapshots(
    camera: Any, profile_token: str | None, count: int, clock: Callable[[], float] = time.monotonic
) -> dict:
    attempts = []
    for _ in range(count):
        t0 = clock()
        data = camera.get_snapshot_jpeg(profile_token)
        entry: dict[str, Any] = {"ok": data is not None, "sec": round(clock() - t0, 3)}
        if data is not None:
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            entry["bytes"] = len(data)
            entry["width"] = int(image.shape[1]) if image is not None else None
            entry["height"] = int(image.shape[0]) if image is not None else None
        attempts.append(entry)
    return {
        "profile_token": profile_token,
        "ok_count": sum(1 for a in attempts if a["ok"]),
        "attempts": attempts,
    }


def _wait_lens_settled(
    camera: Any,
    start: float,
    target: float,
    timeout_sec: float,
    poll_sec: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    stable_polls: int = 5,
) -> dict:
    """Wait for a zoom move to finish, judged from the zoom position alone.
    The lens has moved once its position is 0.02 or more from `start`, and
    has settled once it has moved and then held still (< 0.002 per poll) for
    `stable_polls` polls. A lens already at `target` (within 0.02) has
    nothing to move, so holding still there counts too; holding still
    anywhere else is a motor that hasn't started yet, not "settled".
    MoveStatus is reported but never decides anything: the CP Plus answers
    IDLE for the whole move. `saw_moving` tells whether it ever said MOVING."""
    t0 = clock()
    last_position = None
    stable = 0
    moved = False
    saw_moving = False
    status = None
    while clock() - t0 < timeout_sec:
        status = camera.get_move_status()
        position = camera.get_zoom()
        if status is not None and str(status).upper() == "MOVING":
            saw_moving = True
        moved = moved or abs(position - start) >= 0.02
        at_target = abs(position - target) <= 0.02
        same = last_position is not None and abs(position - last_position) < 0.002
        stable = stable + 1 if same else 0
        last_position = position
        if (moved or at_target) and stable >= stable_polls:
            return {
                "settled": True,
                "settle_sec": round(clock() - t0, 2),
                "final_position": position,
                "move_status_reported": status is not None,
                "saw_moving": saw_moving,
            }
        sleep(poll_sec)
    return {
        "settled": False,
        "settle_sec": None,
        "final_position": last_position,
        "move_status_reported": status is not None,
        "saw_moving": saw_moving,
    }


def _wait_focus_settled(
    camera: Any,
    timeout_sec: float,
    poll_sec: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict:
    t0 = clock()
    saw_moving = False
    while clock() - t0 < timeout_sec:
        status = camera.get_focus_status()
        if status is None:
            return {"focus_status_reported": False}
        move = str(status.get("move_status") or "").upper()
        if move == "MOVING":
            saw_moving = True
        elif move == "IDLE" and (saw_moving or clock() - t0 > 1.0):
            return {
                "focus_status_reported": True,
                "settled": True,
                "settle_sec": round(clock() - t0, 2),
                "saw_moving": saw_moving,
                "position": status.get("position"),
            }
        sleep(poll_sec)
    return {"focus_status_reported": True, "settled": False, "saw_moving": saw_moving}


def lens_travel_test(
    camera: Any,
    zoom_limits: tuple[float, float] | None,
    timeout_sec: float = 20.0,
    poll_sec: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Wide -> full tele -> wide, then autofocus. Zoom only, never pan/tilt.
    Leaves the lens fully wide (the patrol position)."""
    wide, tele = zoom_limits if zoom_limits else (0.0, 1.0)
    result: dict[str, Any] = {"wide": wide, "tele": tele}
    start = camera.get_zoom()
    camera.zoom_to(wide)
    result["prepare_wide"] = _wait_lens_settled(camera, start, wide, timeout_sec, poll_sec, clock, sleep)
    for label, target in (("wide_to_tele", tele), ("tele_to_wide", wide)):
        sleep(LENS_COMMAND_COOLDOWN_SEC)
        start = camera.get_zoom()
        ok = camera.zoom_to(target)
        settled = _wait_lens_settled(camera, start, target, timeout_sec, poll_sec, clock, sleep)
        result[label] = {"command_ok": ok, **settled}
    ok = camera.trigger_autofocus()
    result["autofocus"] = {
        "command_ok": ok,
        **_wait_focus_settled(camera, timeout_sec, poll_sec, clock, sleep),
    }
    return result


def compare_expected(rtsp_cfg: dict, streams: dict[str, dict]) -> list[str]:
    """Differences between configs/rtsp_config.yaml and what the camera sends."""
    cam = rtsp_cfg.get("camera", {})
    issues = []
    for key in CONFIGURED_STREAMS:
        expected = cam.get(key, {}) or {}
        got = streams.get(key) or {}
        if not got.get("opened"):
            issues.append(f"{key}: could not open the stream")
            continue
        if got.get("first_frame_sec") is None:
            issues.append(f"{key}: opened but no frame arrived")
            continue
        exp_w, exp_h = expected.get("width"), expected.get("height")
        if exp_w and exp_h and (got.get("width"), got.get("height")) != (int(exp_w), int(exp_h)):
            issues.append(
                f"{key}: resolution {got.get('width')}x{got.get('height')}, expected {exp_w}x{exp_h}"
            )
        exp_fps, measured = expected.get("fps"), got.get("measured_fps")
        if exp_fps and measured is not None:
            if abs(measured - float(exp_fps)) > max(1.0, FPS_TOLERANCE * float(exp_fps)):
                issues.append(f"{key}: measured {measured} fps, expected {exp_fps}")
    return issues


def phase0_checklist(report: dict) -> dict:
    """The Phase 0 questions, answered from the raw report."""
    onvif = report.get("onvif") or {}
    streams = report.get("streams") or {}
    snapshots = report.get("snapshots") or {}
    good_snaps = [a for a in snapshots.get("attempts", []) if a.get("ok")]
    topics = onvif.get("event_topics") or []

    def ok(key: str) -> bool:
        return bool(streams.get(key, {}).get("first_frame_sec") is not None)

    return {
        "sub_stream_ok": ok("sub_stream"),
        "main_stream_ok": ok("main_stream"),
        "main_stream_codec": streams.get("main_stream", {}).get("codec"),
        "onvif_ok": report.get("onvif") is not None,
        "onvif_profile_count": len(onvif.get("profiles") or []),
        # Discovered over ONVIF only, never opened: a third stream for the
        # base station's live view (D1), not one the Pi processes.
        "extra_onvif_profiles": sorted(
            str(p.get("token"))
            for p in onvif.get("profiles") or []
            if p.get("stream_uri") and p.get("configured_stream") is None
        ),
        "snapshot_ok": bool(good_snaps),
        "snapshot_resolution": (
            f"{good_snaps[0].get('width')}x{good_snaps[0].get('height')}" if good_snaps else None
        ),
        "snapshot_latency_sec": max((a["sec"] for a in good_snaps), default=None),
        "zoom_limits": onvif.get("zoom_limits"),
        "move_status_reported": onvif.get("move_status") is not None,
        "focus_status_reported": onvif.get("focus_status") is not None,
        "motion_event_topics": [t for t in topics if "motion" in t.lower()],
        "mismatches": report.get("mismatches", []),
    }


def run_probe(
    rtsp_cfg: dict,
    rtsp_seconds: float,
    snapshot_count: int,
    lens_test: bool,
    skip_onvif: bool = False,
    camera_factory: Callable[[dict], Any] = OnvifLensCamera.from_config,
    stream_measurer: Callable[[str, float], dict] = measure_stream,
) -> dict:
    cam_cfg = rtsp_cfg["camera"]
    user, password, host = cam_cfg["username"], cam_cfg["password"], cam_cfg["host"]
    report: dict[str, Any] = {"probe_version": 1, "ts": time.time(), "host": host}
    urls = {key: build_rtsp_url(cam_cfg[key]["url"], host, user, password) for key in CONFIGURED_STREAMS}

    camera = None
    if not skip_onvif:
        try:
            camera = camera_factory(rtsp_cfg)
            report["onvif"] = collect_onvif(camera, {key: stream_identity(url) for key, url in urls.items()})
        except Exception as exc:  # report it; the RTSP checks still run
            camera = None
            report["onvif"] = None
            report["onvif_error"] = f"{type(exc).__name__}: {exc}"

    # Each configured stream once; ONVIF URIs only repeat these or point at
    # streams the Pi never processes.
    streams = {key: stream_measurer(url, rtsp_seconds) for key, url in urls.items()}
    report["streams"] = streams

    if camera is not None and snapshot_count > 0:
        token = largest_profile_token(report["onvif"]["profiles"])
        report["snapshots"] = check_snapshots(camera, token, snapshot_count)
    if camera is not None and lens_test:
        report["lens_test"] = lens_travel_test(camera, report["onvif"].get("zoom_limits"))

    report["mismatches"] = compare_expected(rtsp_cfg, streams)
    report["checklist"] = phase0_checklist(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZOVIVE Phase 0 camera verification (read-only by default)")
    parser.add_argument("--rtsp-seconds", type=float, default=10.0, help="read each stream this long")
    parser.add_argument("--snapshots", type=int, default=3, help="ONVIF snapshots to fetch (0 = skip)")
    parser.add_argument(
        "--lens-test", action="store_true", help="MOVES THE LENS: zoom in/out and autofocus, timing each"
    )
    parser.add_argument("--skip-onvif", action="store_true", help="RTSP checks only")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    try:
        rtsp_cfg = load(paths.RTSP_CONFIG)
    except ConfigError as exc:
        print("invalid configuration:", file=sys.stderr)
        for issue in exc.issues:
            print(f"  {issue}", file=sys.stderr)
        return EXIT_CONFIG

    report = run_probe(rtsp_cfg, args.rtsp_seconds, args.snapshots, args.lens_test, args.skip_onvif)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
