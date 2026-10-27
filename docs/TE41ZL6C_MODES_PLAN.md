# CP-UNC-TE41ZL6C-VMDS-Q: patrol/trigger modes, events and evidence — approved plan

Status (2026-10-03): design decisions D1–D13 approved; **Phase 0 (verification) tools
implemented**, waiting for results from a real camera and Pi. Phases 1–7 are not started.
Each phase is presented for approval before any code is written.

---

## 1. Decision record

| # | Decision | Approved choice |
|---|---|---|
| D1 | Streams / live view | Fixed camera encoder profiles, **never reconfigured at runtime**: main = 4MP evidence, sub = 720p detection. "Patrol 5 fps / Trigger best fps" = how many frames the Pi *processes*. Live view: base station pulls RTSP **directly from the camera** (sub-stream, or a third stream if the camera has one) |
| D2 | Event model | **Encounter = one event = one video.** Within an encounter, **one alert per species**. New `encounters` table; alert rows (`events`) get `encounter_id`; the video belongs to the encounter |
| D3 | Track IDs | Alert `track_id` = the track that triggered it; the encounter keeps the list of all its track IDs |
| D4 | Pre-roll | 10 s pre-roll from the RAM segment ring |
| D5 | Recording process | Separate `zovive-record.service` (ffmpeg stream copy into the `/dev/shm/zovive/segments` ring, always on). edge_main publishes "encounter active"; the recorder moves segments to `/data` while it is |
| D6 | Long videos | 2 s segments (crash-safe), joined by stream copy into **one** `event_<id>.mp4` per stream at encounter end. Only the 720p file is uploaded, through the chunked/resumable uploader |
| D7 | Voting | **Detector → crop → classifier → confidence-weighted voting** (new strategy next to `average` / `majority` in `pipeline/stage2_verifier.py`) |
| D8 | Low-confidence label | Below 0.60: internal `unknown_animal`, display **"Animal Detected"** |
| D9 | Unknown → species upgrade | **Local pending-update queue** in the outbox until the base station has an update API |
| D10 | 4MP archive | Kept locally under `configs/retention_policy.yaml` (30 days / 120 GB, oldest first). Requires `storage/disk_guard.py` + `retention_manager.py` before 4MP recording goes live |
| D11 | Max event duration (6 h) | Close the encounter, finalize its video, re-validate; if the animal is still present, open a **new** encounter (alert flagged as continuation) |
| D12 | 120 s presence confirmation | Confirmation = a detection at the normal detection threshold; before presence lapses, one conditional SAHI re-check |
| D13 | Hailo model fit / HEF format | **Needs Phase 0 verification** (`tools/benchmark_edge.py --sections hef`) |

### Behaviour these decisions add up to

| Rule | Value |
|---|---|
| Patrol triggers | MOG2 **or** camera ONVIF motion event **or** YOLO safety sweep |
| Safety sweep | YOLO once every **10 s** in PATROL (sleeping / stationary animals) |
| Unconfirmed trigger | Back to PATROL after **10 s** with no confirmed animal |
| Presence | Tracker alive **and** a detector confirmation at least every **120 s**. Motion is ignored once an animal is confirmed |
| Exit | PATROL only after **30 s** with no animal |
| Snapshots | 5–10 at encounter start (ONVIF snapshot, 4MP JPEG) → YOLO → SAHI only for low-confidence / small / distant → classifier → voting |
| SAHI | Never per frame. Only on the ROI (`pipeline/roi_cluster.cluster_box`) when a detection is low-confidence, small or distant |
| Video | Stream copy, starts at confirmation (with pre-roll), stops 30 s after the animal disappears |
| Upload | 720p only, queued, deleted locally only after base-station ACK, retried forever |
| Max encounter | 6 h, then re-validation (D11) |

### Not created (each would duplicate existing code)

`camera_control/profile_switcher.py`, `trigger_modes.py`, `patrol_controller.py`,
`preset_context.py`, `pipeline/target_handoff.py`, `watchdog/hardware_watchdog.py`
(systemd `RuntimeWatchdogSec` drives the Pi's hardware watchdog), a third SAHI module.
The camera is a fixed bullet: zoom and focus only, never pan, tilt or presets.

> `docs/CAMERA_MODES.md`, `docs/PATROL_AND_TRIGGER.md` and parts of `docs/DEVELOPER_GUIDE.md`
> / `docs/ARCHITECTURE.md` still describe a PTZ head and modules that don't exist.
> They are rewritten in Phase 7; until then this file is authoritative.

---

## 2. Runtime layout (approved)

```
CAMERA (fixed encoder profiles)
 ├─ main  4MP  ──► zovive-record.service (ffmpeg copy → RAM ring → /data while an encounter is active)
 ├─ sub   720p ──► edge_main RtspReader → motion / YOLO / tracking
 │            └─► zovive-record.service (ffmpeg copy → 720p upload video)
 ├─ live (sub or third stream) ──► base station / dashboard, directly from the camera
 ├─ ONVIF events   ──► edge_main OnvifMotionEventListener (new thread, own connection)
 └─ ONVIF snapshot ──► edge_main snapshot burst (4MP JPEG, no video decode on the Pi)

edge_main.py      capture → inference (PATROL: MOG2 + sweep / TRIGGER: YOLO) → pipeline
                  (mode timers, tracker, encounter, snapshots, SAHI-when-needed, voting, alerts)
transfer_main.py  alerts + chunked 720p video upload, delete after ACK, pending-update queue
health_main.py    heartbeat + camera / live-stream / recorder liveness
```

---

## 3. Phases

| Phase | Content | Done when |
|---|---|---|
| **0 Verify** *(tools done; results pending)* | Camera: stream count, resolution/fps/codec per stream, ONVIF snapshot, zoom range, zoom/focus status, motion event topics. Pi: YOLO11m HEF format + throughput, both models on one Hailo-8, decode cost, thermal soak. Reader throttle removed | Measured numbers replace every estimate below; D13 answered |
| 1 Modes | `camera_control/mode_manager.py` `ModeManager` (PATROL/TRIGGER, 10 s / 30 s / 120 s / 6 h timers); `edge_main._inference_loop` / `_pipeline_loop` | Transition tests with a fake clock; file-source end-to-end run |
| 2 Triggers | `capture/motion_detector.py` (MOG2), `pipeline/motion_gate.py` `MotionGate.evaluate` (+10 s sweep), new `camera_control/motion_event_listener.py` | `tests/test_trigger_arbitration.py`; a still video triggers through the sweep |
| 3 Presence | `inference/bytetrack_wrapper.py` `IouTracker` (time-based age, low-confidence keep-alive); 120 s confirmation | Stump-lock and occlusion tests (no new track ID) |
| 4 Encounter + alert | `TrackStateMachine` encounters; `AlertDispatcher`; DB v4 (`encounters`, `events.encounter_id`, pending updates); snapshot burst; conditional SAHI; confidence-weighted voting at 0.60 | 3 tiger tracks → 1 alert; tiger + deer → 2 alerts, 1 video |
| 5 Recording | `zovive-record.service` + `capture/segment_recorder.py`; `ClipExtractor` segments mode; metadata in MP4 tags + snapshot JSON sidecars | 1 × 720p + 1 × 4MP per encounter; killing edge_main mid-event loses no segments |
| 6 Storage + upload | `storage/disk_guard.py`, `retention_manager.py`; `network/chunked_uploader.py`; streaming upload in `transfer_main._send_pending_videos` | `/data` never fills; a 2 GB upload resumes after a link drop |
| 7 Watchdog + docs | `WatchdogSec` + sd_notify in `transfer_main`; recorder + live-stream liveness in `health_main`; rewrite the PTZ-era docs | Kill each worker/process → recovers unattended |

---

## 4. Phase 0 — what was implemented

| File | Change |
|---|---|
| `capture/rtsp_reader.py` | `RtspReader._loop` no longer sleeps between frames (a throttled reader falls behind real time: H.264 frames can't be skipped). Opening moved into public `open_rtsp_capture()` (bounded timeouts), reused by the tools and `grab_main_stream_frame`; `_build_url` → public `build_rtsp_url` |
| `configs/rtsp_config.yaml` | Expected TE41ZL6C encoder settings: sub 1280×720 @ 10 fps, main 2688×1520 @ 25 fps. `target_fps` removed: frame rate is set on the camera |
| `configs/clip_config.yaml` | `fps: 10` to match the sub-stream (clips would otherwise play at the wrong speed) |
| `camera_control/camera_api.py` | Read-only `OnvifLensCamera` queries: `get_device_info`, `describe_media_profiles`, `get_stream_uri`, `get_snapshot_uri`, `get_snapshot_jpeg`, `get_move_status`, `get_focus_status`, `get_zoom_limits`, `get_imaging_options`, `get_event_topics`; module `fetch_snapshot` (HTTP **digest only**: basic auth would put the camera password in clear text on the bridge) |
| `tools/camera_probe.py` (new) | Camera verification, read-only unless `--lens-test` |
| `tools/benchmark_edge.py` | Was a stub: `hef`, `models`, `decode`, `soak` sections |
| `utils/thermal_monitor.py` | Was a stub: `read_soc_temp_c()` |

### How to run Phase 0 (on the node)

```bash
sudo systemctl stop zovive-detect          # frees the Hailo-8 and the camera streams
cd /opt/zovive && set -a && . ./.env && set +a

.venv/bin/python -m tools.camera_probe --out /tmp/camera_probe.json
.venv/bin/python -m tools.camera_probe --lens-test --snapshots 0 --out /tmp/camera_lens.json   # MOVES THE LENS
.venv/bin/python -m tools.benchmark_edge --backend hailo --out /tmp/bench.json
.venv/bin/python -m tools.benchmark_edge --backend hailo --sections soak --soak-source sub --soak-seconds 600 --out /tmp/soak_sub.json

sudo systemctl start zovive-detect
```

Add `--image /path/to/animal.jpg` to the benchmark for realistic NMS/decode cost.

### What each result decides

| Result (JSON key) | Decides |
|---|---|
| `camera_probe` `checklist.sub_stream_ok`, `mismatches` | Whether the camera's sub-stream must be set to 720p @ 10 fps in its web UI |
| `checklist.extra_onvif_profiles` (`onvif.profiles[].configured_stream`) | Live view on a separate third stream (preferred) vs. sharing the sub-stream (D1). Discovered over ONVIF only: the probe opens just `sub_stream` and `main_stream`, because the camera's ONVIF URIs carry `proto=Onvif` and answer 401 to the configured credentials |
| `checklist.main_stream_codec` | H.264 vs H.265 for the 4MP archive (H.265 ≈ 40 % less storage) |
| `checklist.snapshot_ok`, `snapshot_resolution`, `snapshot_latency_sec` | Whether the Phase 4 snapshot burst can use ONVIF snapshots at 4MP, and how many fit in the first seconds |
| `checklist.move_status_reported`, `focus_status_reported`, `lens_test.*.settle_sec` | Phase 3 zoom/focus settle logic (status-driven vs. timers) |
| `checklist.motion_event_topics` | Phase 2 ONVIF topic subscription |
| `bench.hef.models.detector.on_chip_nms`, `decode_yolo_detections_compatible` | **D13**: whether `inference/yolo_postprocess.py` / `_HailoRunner` must change for the YOLO11m HEF |
| `bench.hef.shared_vdevice.ok` | **D13**: whether detector + classifier fit on one Hailo-8 (fix for `_HailoRunner`'s one-VDevice-per-model) |
| `bench.models.detector.720p`, `bench.sahi.roi` | Achievable TRIGGER fps; SAHI cost per snapshot |
| `soak.inference_fps`, `process_cpu_cores`, `max_soc_temp_c`, `throttle_end` | Sustained load and thermal margin of the approved design |

---

## 5. Defects found during review — resolution status

| Defect | Impact | Status |
|---|---|---|
| `capture/stream_health.py` `StreamHealth.snapshot()` takes `self._lock`, then calls `fps()` / `is_stalled()`, which take the same non-reentrant lock → **deadlock** | `edge_main._write_edge_status` calls it on the supervisor thread on the first loop: the main thread hangs (holding the lock, so the capture thread blocks on `record_frame` too), systemd's watchdog kills the service after 30 s, and it restarts every ~60 s | **Fixed**: `StreamHealth` now uses `threading.RLock()`; regression tests in `tests/test_stream_health.py` verify `snapshot()` no longer deadlocks |
| `inference/hailo_inference.py` `_HailoRunner`: one `VDevice` per model; `InferVStreams` + `activate()` on every call; reads output `[0]` only | Second model may fail to open (silent CPU fallback); per-call overhead at 10 fps | Fix after Phase 0 results (D13) |
| `pipeline/clip_extractor.py` re-encodes RAM-held frames | Unusable for long / 4MP recording | Replaced in Phase 5 |
| `transfer_main._send_pending_videos` reads the whole file into RAM, single POST | Fails for multi-GB encounter videos | Phase 6 |
| `storage/disk_guard.py`, `retention_manager.py` are stubs | `/data` can fill | Phase 6, required before 4MP archiving (D10) |
| `zovive-transfer.service` has no `WatchdogSec` | A hung uploader is never restarted | Phase 7 |
| Two config systems (`camera.yaml`/`inference.yaml`/`node.yaml` vs `rtsp_config.yaml`/`inference_config.yaml`/`node_config.yaml`), unused `sahi_config.yaml`, duplicate tile code in `inference/sahi_batched.py` | Drift | New keys go only into the files edge_main loads; cleanup to be scheduled |
