# ZOVIVE Edge

Raspberry Pi 5 + Hailo-8 wildlife monitoring node: animal-aware optical zoom + autofocus, on-device
detection + species classification, evidence storage, and store-and-forward
alerting to a base station.

## Status

This is the **first coded batch** on top of the original skeleton,
plus a second enhancement pass (SAHI, event video recording + upload,
species-voting strategies, Hailo CPU fallback — see below). It covers
the full detect -> track -> classify -> alert (+ video clip) path end
to end, runnable and tested on a laptop/CI with no camera or Hailo
device required. See "What's still not here" below for what's deferred.

## Quickstart (no camera, no Hailo — runs anywhere)

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v      # 131 tests, no camera/model needed

# Run the full pipeline against a local video file instead of a real camera.
# (You'll still need models/detector.onnx and models/classifier.onnx —
#  export those from zovive-ml first; see docs/ARCHITECTURE.md.)
# Camera credentials are always required (edge_main.py exits with code 78
# if either is unset or empty); any non-empty value works for --source file:.
export ZOVIVE_CAMERA_USERNAME=dev ZOVIVE_CAMERA_PASSWORD=dev
python3 edge_main.py --source file:/path/to/some_video.mp4
```

Credentials and API keys are never stored in `configs/*.yaml`; those files
reference them as `${VAR}`. On a node they go in `/opt/zovive/.env` (copy
`.env.example`), which systemd loads and git ignores.

Set `configs/node_config.yaml`'s `runtime.environment: pi` and
`runtime.backend: hailo` for the real Raspberry Pi + Hailo-8 deployment;
everything else (zoom/focus, tracking, gating, classification, alerting)
is identical code on both.

## Layout

| Folder | What's in it |
|---|---|
| `capture/` | RTSP reader + reconnect/health tracking, and a file-based source for dev/tests |
| `camera_control/` | ONVIF lens control (optical zoom + autofocus only, never pan/tilt; + a fake for tests), animal-aware zoom planner and state machine |
| `inference/` | Detector/Classifier interfaces, ONNX backend (laptop/Pi-CPU), Hailo backend (Pi), IoU tracker |
| `pipeline/` | Presence-vote gating, per-track state machine, best-snapshot selection, species verification (2 voting strategies), alert dispatch, event video recording (ram_buffer + clip_extractor) |
| `queues/` | Bounded drop-oldest queues + latest-frame slot, all metered |
| `db/` | SQLite outbox schema + read/write helpers (image + video status, and image/video deletion timestamps, tracked separately) |
| `storage/` | Snapshot evidence store + post-ACK local-file cleanup |
| `network/` | Base-station HTTP client (image + video upload) + retry backoff |
| `tests/` | 139 tests, stdlib `unittest`, zero required installs |

`edge_main.py`, `transfer_main.py`, `health_main.py` are the three
processes (matching the three systemd units) — see each file's
docstring for why they're separate.

## Enhancements on top of batch 1

Added in a second pass (see `docs/ENHANCEMENT_REPORT.md` for the full
before/after, file-by-file):

- **`inference/sahi_slicer.py`**: SAHI tiled inference — wraps the
  existing `Detector` (ONNX or Hailo, doesn't care which), off by
  default (`configs/inference_config.yaml` `detector.sahi.enabled`).
- **`pipeline/ram_buffer.py` + `pipeline/clip_extractor.py`**: event
  video recorder — a rolling pre-event buffer plus a live post-event
  window, written to MP4 next to the JPEG snapshot
  (`configs/clip_config.yaml`).
- **`network/base_station_client.py`'s `send_video`** + `db/outbox.py`'s
  `video_path`/`video_status` columns + `transfer_main.py`'s video-send
  loop: uploads the clip on its own retry timeline, independent of the
  image alert (`configs/transfer_config.yaml` `transfer.video_upload_*`).
- **`pipeline/stage2_verifier.py`**: a second voting strategy,
  `"majority"`, alongside the original `"average"` softmax blend
  (`configs/inference_config.yaml` `classifier.voting_strategy`).
- **`edge_main.py`'s `build_detector_and_classifier`**: falls back from
  Hailo to the ONNX (CPU) backend if HailoRT/the device isn't available
  at startup, instead of crash-looping.
- **`storage/local_file_cleanup.py`** + `db/outbox.py`'s
  `snapshot_deleted_ts`/`video_deleted_ts` columns + `transfer_main.py`'s
  send loops: matches the intended storage model — Pi disk is temporary
  (images, videos, SQLite metadata), the base station is permanent. On a
  successful send (an ACK), the local file is deleted; on failure it's
  kept and retried, unchanged. Controlled by `transfer_config.yaml`'s
  `transfer.delete_local_files_after_ack` (default `true`), which is
  forced off automatically whenever `base_station.url` is empty, so a
  dev/test run with no real base station never deletes the only copy of
  an evidence file.

## What's still not here (deferred)

Listed here instead of stubbed out so nothing in this repo *looks*
done without being done:

- **network/**: priority_dispatcher, ack_listener, sync_daemon, boot_registration
- **storage/**: retention_manager, purge_service, disk_guard, orphan_scanner, cache_manager
- **pipeline/**: clip_quality_gate (skip recording a clip below some quality bar), roi_cluster
- **ota/**: model_updater, checksum_verify, rollback, version_checker
- **watchdog/** + systemd `WatchdogSec`/`sd_notify` wiring
- **provisioning/**: setup_pi.sh, install_hailort.sh, static IP, SSH hardening
- **tools/**: benchmark_edge, db_inspect, debug_viewer, replay_clip, zone_editor

## Testing without pytest

If your environment can't reach PyPI (this happens in some sandboxed
dev environments), the whole suite still runs:

```bash
python3 -m unittest discover -s tests -v
```

`pytest` (see `requirements-dev.txt`) works too and collects the same
`unittest.TestCase` classes — CI runs both, so neither path silently rots.
