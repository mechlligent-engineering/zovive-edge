# ZOVIVE Edge

The software for a ZOVIVE edge node: a Raspberry Pi 5 with a Hailo-8L NPU, watching one fixed CP Plus motorised-lens camera at a forest site. It decides on the device whether an animal is one of seven target species, keeps the evidence (snapshot + short video), and forwards alerts to a base station over the local network. The node has no internet connection.

This README is the starting point for a new developer. It covers how the node works end to end, how to develop and test on a laptop, and how code reaches the Pi. The `docs/` folder goes deeper on each part, and is linked from each section.

**State on 31 Oct 2026:** the field Pi runs branch `feature/pipeline-hardening` at `689b72d`, with the R1 HEF.

| Measure (on the Pi, 30 Oct 2026) | Value |
|---|---|
| Live inference rate | **11.16 FPS** (the chip's own ceiling for this HEF: 11.18 FPS, 84.0 ms) |
| Per-frame latency p50 / p95 | 88.9 / 89.6 ms |
| Detector process CPU | 0.23 cores |
| Detection parity, chip vs ONNX | 95.2% of ONNX boxes matched, 100% same species (102 images) |

Known open issues are listed under [Current state and known issues](#current-state-and-known-issues).

---

## Contents

1. [What the node does](#what-the-node-does)
2. [Hardware and platform](#hardware-and-platform)
3. [Processes and systemd units](#processes-and-systemd-units)
4. [The detection pipeline (`zovive-detect`)](#the-detection-pipeline-zovive-detect)
5. [Inference: model, HEF and the Hailo runner](#inference-model-hef-and-the-hailo-runner)
6. [Alert decision and suppression](#alert-decision-and-suppression)
7. [Data on the node](#data-on-the-node)
8. [Configuration](#configuration)
9. [Developer workflow](#developer-workflow)
10. [Tools](#tools)
11. [Telemetry](#telemetry)
12. [Deploying to the Pi](#deploying-to-the-pi)
13. [Current state and known issues](#current-state-and-known-issues)
14. [Further reading](#further-reading)

---

## What the node does

```
camera (RTSP) ──> zovive-detect ──> SQLite outbox + snapshot + MP4 ──> zovive-transfer ──> base station
                      │                                                       (HTTP, retried, ACKed)
                      └── edge_status.json ──> zovive-health ──> heartbeats (local table + base station)
```

1. **Capture:** frames arrive from the camera's sub stream (~25 FPS).
2. **Inference:** each frame the detector can take (~11 FPS) goes through the YOLO11m species detector on the Hailo-8L.
3. **Decision:** detections are tracked across frames. A track becomes an alert only after a confirmation gate, a species vote and the re-alert checks.
4. **Recording:** an alert writes a JPEG snapshot and an outbox row, and starts an MP4 clip with 8 s before and 8 s after.
5. **Transfer:** a separate process sends alerts and clips to the base station and retries until they are acknowledged. Local media is deleted only after the base station confirms receipt.

The seven species are deer, gaur, leopard, elephant, sloth_bear, tiger and wild_boar. A high-priority alert is raised for tiger, leopard, elephant and sloth_bear.

---

## Hardware and platform

| Part | What's used |
|---|---|
| Computer | Raspberry Pi 5 |
| NPU | Hailo-8L (M.2 HAT), HailoRT **4.23.0**, kernel driver `hailo_pci` 4.23.0 |
| OS | Raspberry Pi OS, kernel `6.18.39+rpt-rpi-2712`, **Python 3.13** |
| Camera | CP Plus CP-UNC-TE41ZL6C: fixed direction, motorised optical zoom and focus, RTSP and ONVIF |
| Network | static IPs on a local network, no DHCP, no internet ([docs/NETWORK_MAP.md](docs/NETWORK_MAP.md)). The test Pi is 192.168.1.111 |
| Storage | `/var/lib/zovive` for runtime data ([docs/STORAGE_LAYOUT.md](docs/STORAGE_LAYOUT.md)) |
| Clock | **no RTC battery fitted yet**, and no NTP (see known issues) |

Code is developed on Windows laptops and deployed to the Pi. `.gitattributes` forces LF line endings everywhere, so shell scripts and systemd units work unchanged on the Pi.

---

## Processes and systemd units

Three processes, so a stuck one can't stall the others. The units are in `systemd/`, and provisioning scripts are in `provisioning/`.

| Unit | Entry point | What it does | Restart |
|---|---|---|---|
| `zovive-detect.service` | `edge_main.py` | the detection pipeline (next section). `Type=notify`, `WatchdogSec=30` | `always`, after 30 s |
| `zovive-health.service` | `health_main.py` | heartbeat every 15 s: detector liveness (from `edge_status.json`), queue depths, stream health, disk, Pi throttle flags | `always`, after 3 s |
| `zovive-transfer.service` | `transfer_main.py` | sends due events and clips from the outbox, with retry and dead-lettering; deletes local media after an ACK | `always`, after 5 s |
| `zovive-retention.timer` | | daily at 03:30: ACKed media older than 14 days | |
| `zovive-storage-pressure.timer` | | every 5 min: cleanup above 82% media filesystem use | |
| `zovive_cleanup.timer` | | hourly cleanup | |

All three services read secrets from `/opt/zovive/.env` (`EnvironmentFile=`). On the Pi, `/opt/zovive` is a symlink to the git checkout.

---

## The detection pipeline (`zovive-detect`)

One process, four supervised worker threads. The main thread supervises them.

```
capture thread      RtspReader (or FileFrameSource for --source file:)
   │  LatestFrameSlot: one slot, newest frame wins, never a backlog
   ▼
inference thread    Detector.infer() -> per-class confidence filter
   │  DetectionQueue: bounded (4), drop-oldest
   ▼
pipeline thread     zone filter -> IoU tracker -> per-place alert memory -> stage-1 gate
   │                -> stage-2 species vote -> zoom decision -> dispatch decision
   │                (also feeds the clip ring buffer every cycle)
   │  dispatch queue: bounded (256); a full queue is logged as an error
   ▼
dispatch thread     snapshot file -> outbox row -> alert_queue (64, drop-oldest) for zovive-transfer
                    each step retried from the one that failed; clip paths follow the same queue
```

| Queue | Size | When full |
|---|---|---|
| `frame_slot` (`queues/frame_queue.py`) | 1 | the newer frame replaces the older one |
| `detection_queue` (`queues/detection_queue.py`) | 4 (`queue_config.yaml`) | drops the oldest batch and counts it in `drop_count` |
| `dispatch_queue` (`pipeline/dispatch_worker.py`) | 256 | refuses, logs an error, and the track isn't alerted |
| `alert_queue` (`queues/alert_queue.py`) | 64 (`queue_config.yaml`) | drops the oldest |

Every queue reports its depth and drops to `edge_status.json` (see [Telemetry](#telemetry)).

**Self-recovery** comes in three layers; the details are in [docs/WATCHDOG.md](docs/WATCHDOG.md).

1. **Worker restart.** Each worker is a `SupervisedThread` (`watchdog/supervisor.py`) with a liveness timestamp.
   - A crashed worker restarts with backoff (1 s, doubling, up to 60 s).
   - A pipeline restart resets tracking and aborts any zoom session.
   - A worker that crashes more than 5 times in 600 s, or whose liveness stops for 60 s, escalates.
2. **Process exit.** On escalation the process exits with code **70**, and systemd restarts it 30 s later. An invalid configuration exits with **78** and a one-line reason naming the key, never the value.
3. **systemd watchdog.** The main thread pings `WATCHDOG=1` only while its checks pass. If the main thread itself hangs, systemd kills and restarts the process after 30 s.

The main thread also writes `edge_status.json` every 5 s. `zovive-health` reports the detector as stale if the file is older than 30 s.

---

## Inference: model, HEF and the Hailo runner

**The model.** YOLO11m, trained for 150 epochs on the 7-class dataset (`ZOVIVE_FINAL_150E`). It's the node's only model: species come straight from the detector's classes, and no classifier runs. Training, export and HEF compilation live in the sibling `zovive-ml` project.

**The deployed HEF: R1** (`det150e_h8l_r1`):

- **sha256:** `b999f7842b6bcc0326a57de06c66958f38d521310ac5e03a79bc4c4a52bfc1f1`.
- **Build:** compiled for the Hailo-8L at optimisation level 1, from an optimized HAR with 256 calibration images. The build record is `zovive-ml/hailo_builds/det150e_h8l_r1/BUILD.md`.
- **Replaces** an earlier faulty build (`2d460f58…`) that skipped the /255 input normalisation and mislabelled species.
- **Deployed as** `/opt/zovive/models/detector.hef`, next to the ONNX fallback `detector.onnx` (sha256 `404de609…`, recorded in `models/manifest.json`). Neither file is in git.

**The HEF contract** (`inference/hailo_inference.py`):

- **Input:** the frame letterboxed to 640×640 (grey 114 padding), RGB, **uint8**, NHWC. The HEF divides by 255 on the chip, so the host must not.
- **Outputs: six tensors, the detection head cut before its DFL decode.** For each stride (8, 16, 32) there's a box conv with 64 channels and a class conv with 7 channels (`conv71/74`, `conv87/90`, `conv102/105`), read as float32.
- **No NMS on the chip.** `inference/yolo_postprocess.decode_split_heads` decodes the DFL boxes on the CPU, then applies:
  - the decode threshold (0.50);
  - a filter for boxes that fall in the letterbox padding;
  - one class-agnostic NMS (IoU 0.45), so one object can't come out as two species.
- **Layout check.** `split_head_layout` checks the output names and shapes strictly when the detector loads. A HEF with on-chip NMS or merged heads is rejected, and `edge_main` falls back to the ONNX model on the CPU rather than crash-looping. `tools/benchmark_edge.py --sections hef` reports this compatibility without loading the detector.

**The persistent runner** (`_HailoRunner`, since `689b72d`). The `InferVStreams` pipeline is opened and the network group activated **once**, on the first inference, and kept open.

- A failed inference tears the pipeline down, and the next call opens it again.
- `close()` deactivates and releases the device under the NPU lock. It gives up after 5 s if a native call is hung.
- Every device call goes through `inference/npu_scheduler.NpuScheduler`, one lock per device.
- This assumes one network group on the device, which is the design: one model.
- Results:
  - output is bit-identical to the old per-frame setup;
  - latency fell from 98.5 to 88.9 ms, and FPS rose from 10.05 to 11.16;
  - new threads fell from 191 to 78 a second. The 78 left are most likely HailoRT starting one thread per vstream inside each `infer()` call.
  - The design and measurements are in [docs/PERSISTENT_VSTREAMS_SPEC.md](docs/PERSISTENT_VSTREAMS_SPEC.md).

**R1 validation on the Pi (gates C and D, 30 Oct 2026):**

| Gate | Result |
|---|---|
| C parity, chip vs ONNX (102 images) | matched 0.952, same species 1.000, IoU 0.968, score diff 0.046: **pass** |
| C AP50 (679 images) | mAP50 0.936 ONNX vs 0.925 chip. Sloth bear drops 4.3 points (limit 3.0), **accepted as a documented exception** |
| D live soak | 10.08 FPS before the persistent runner, p95 100 ms, 45-50 °C, no throttling: **pass** |

SAHI (tiled inference, `inference/sahi_slicer.py`) exists but is **off**: one full 4 MP frame takes about 1.5 s on the Pi.

---

## Alert decision and suppression

The order is the pipeline thread's own (`edge_main._pipeline_loop`):

1. **Per-class confidence** (`inference/class_thresholds.py`). Detections below their species' threshold are dropped before tracking: elephant and gaur 0.60, leopard and wild boar 0.55, tiger, sloth bear and deer 0.50.
2. **Zone filter** (`pipeline/zone_filter.py`, `configs/detection_zones.yaml`).
3. **IoU tracker** (`inference/bytetrack_wrapper.IouTracker`, `configs/tracking_config.yaml`). It matches at IoU 0.3. A track survives **40 pipeline cycles** (about 5 s) without a match before it's dropped.
4. **Per-place alert memory** (`pipeline/recent_alerts.py`, since `52bb464`).
   - **What's remembered:** every dispatched alert, by place and class. That covers the wide view and the wide box of a zoom session.
   - **Matching:** a new track of the same class whose centre falls inside a remembered box is treated as the same animal. The box is grown by half its width and height on each side. A match is marked alerted, with no new event.
   - **Expiry:** any live track at that place keeps the entry alive. The place can alert again only after it has been out of view for `realert_window_sec` (300 s).
   - **Why:** before this, the tracker gave a static object a new track ID after every detection gap. Each ID got past the per-track cooldown, and one indoor false positive produced 59 elephant alerts in a night. Design: [docs/REALERT_SUPPRESSION_SPEC.md](docs/REALERT_SUPPRESSION_SPEC.md).
5. **Stage-1 gate** (`pipeline/stage1_gate.py`). A track is confirmed only when all of these hold:
   - present in 5 of the last 7 frames;
   - seen for at least 1.0 s;
   - its leading species averages at least that species' threshold.
   
   There's also a per-track cooldown of 300 s.
6. **Stage-2 species vote** (`pipeline/stage2_verifier.py`). The species is voted from the track's per-frame classes ("average" strategy). Below 0.50 the result is `unknown_animal`, which is **not alerted** (`alert_policy.yaml send_unknown_animal_alerts: false`).
7. **Zoom** (`camera_control/zoom_controller.py`). A small, far animal triggers an optical zoom and a re-detection, with tracking paused while the lens moves. If the re-detection fails, the alert falls back to the wide-view evidence.
8. **Dispatch.** The pipeline decides the event (policy, ID, priority); the dispatch thread writes it. A clip starts with the alert.

---

## Data on the node

Everything is under `ZOVIVE_HOME` (`/var/lib/zovive` on the Pi; `./var` on a laptop by default). See `paths.py`:

| Path | Contents |
|---|---|
| `db/zovive.sqlite3` | the outbox (`events`: one row per alert, image and video status, retry state, ACK times) and `heartbeats`. WAL mode. Schema: `db/schema.sql` |
| `snapshots/` | alert JPEGs, named `<local time>_<camera>_<track>_<event id>.jpg` |
| `clips/` | alert MP4s (8 s before + 8 s after, 10 FPS) |
| `edge_status.json` | the detector's live status, rewritten every 5 s |
| `restart_history.json` | process starts and exit reasons |
| `logs/`, `cache/`, `quarantine/` | logs, cache, files that failed validation |

Outbox rows are never deleted; they're the permanent local record. Media files are deleted only after the base station acknowledges that file ([docs/PURGE_AND_CACHE.md](docs/PURGE_AND_CACHE.md)). With no base station URL configured, deletion after ACK is forced off.

---

## Configuration

All settings live in `configs/*.yaml`. Each file is validated when it loads (`utils/config_loader.py`), and an invalid value stops `edge_main` with exit 78.

| File | Controls |
|---|---|
| `node_config.yaml` | node identity, `runtime.backend` (`hailo`/`onnx`), `runtime.environment` (`pi` drives the real lens over ONVIF; `dev` uses a fake lens) |
| `inference_config.yaml` | model paths, decode and per-class thresholds, the stage-1 gate, `realert_window_sec`, the stage-2 vote, SAHI |
| `tracking_config.yaml` | tracker IoU, `max_age_frames`, `min_hits_to_confirm` |
| `alert_policy.yaml` | high-priority species, `send_unknown_animal_alerts`, minimum snapshot sharpness |
| `camera_modes.yaml` | zoom planning and timing |
| `rtsp_config.yaml`, `camera.yaml` | camera streams and ONVIF; credentials come from the environment |
| `clip_config.yaml` | event clips |
| `queue_config.yaml` | queue sizes |
| `watchdog_config.yaml` | heartbeat interval, status-file timing, worker restart policy |
| `transfer_config.yaml`, `retention_policy.yaml` | sending, retries, deletion after ACK, retention |
| `detection_zones.yaml` | detection zones per view |

**Secrets never go in `configs/`.** Files refer to them as `${VAR}`, and the values live in `/opt/zovive/.env` (from `.env.example`, git-ignored). The variables are:

- `ZOVIVE_CAMERA_USERNAME`, `ZOVIVE_CAMERA_PASSWORD`;
- `ZOVIVE_ONVIF_USERNAME`, `ZOVIVE_ONVIF_PASSWORD`;
- `ZOVIVE_BASE_STATION_URL`, `ZOVIVE_BASE_STATION_API_KEY`;
- `ZOVIVE_HOME`, `ZOVIVE_NODE_ID`;
- optional log and config overrides.

Never print `.env`. On the Pi, don't share `systemctl status` or the full journal either: both can show the camera URL. Kernel-only (`journalctl -k`) and systemd-only (`journalctl _PID=1`) output is safe.

---

## Developer workflow

**Setup (laptop, Python 3.11+):**

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt   # Linux/macOS: .venv/bin/pip
```

`requirements-pi.txt` lists the Pi's extras. `hailo_platform` only exists on the Pi, so on a laptop the detector falls back to ONNX automatically.

**Checks.** These are the same ones CI runs (`.github/workflows/ci.yml`). All must pass before a commit.

```bash
python -m pytest tests/ -q                  # about 750 tests; no camera, NPU or model needed
python -m unittest discover -s tests        # the same suite with the standard library only
python -m ruff check .
python -m ruff format --check .
python -m bandit --ini .bandit -r .
```

Tests are stdlib `unittest` classes. HailoRT, the camera and the base station are faked; real numbers only come from the Pi.

**Run the whole pipeline on a laptop** against a video file. This uses the ONNX model in `models/detector.onnx`, a fake lens and a scratch data folder:

```bash
export ZOVIVE_HOME=/tmp/zovive-dev
export ZOVIVE_CAMERA_USERNAME=dev ZOVIVE_CAMERA_PASSWORD=dev ZOVIVE_ONVIF_USERNAME=dev ZOVIVE_ONVIF_PASSWORD=dev
python edge_main.py --source file:/path/to/clip.mp4 --source-fps 25
```

Any non-empty value works for the camera variables with `--source file:`, but they must be set. Stop the run with Ctrl+C, which gives a clean shutdown.

**Ways of working on this project:**

- **Design first.** Any feature or behaviour change gets a short design and the owner's approval before code. The specs in `docs/*_SPEC.md` are the record (decisions numbered, e.g. D1-D5, A1-A4, P1-P4).
- **Branches.**
  - `feature/pipeline-hardening` is what the field Pi runs. It's pushed to GitHub but not yet merged into `develop`.
  - `develop` and `main`: PR #2 (develop into main) is still open.
- **Commits.** Small, one change each, conventional prefixes (`fix(pipeline):`, `perf(inference):`, `docs:`). No co-author trailers.
- **Hardware claims.** No performance number goes into docs or commits unless it was measured on the real Pi.

---

## Tools

| Tool | Use |
|---|---|
| `tools/benchmark_edge.py` | on-Pi measurements, in sections: `hef` (outputs and whether HailoDetector can read them), `models` (latency at 720p and 4 MP, letterbox, SAHI), `decode` (stream decode rate), `soak` (live capture + inference: FPS, latency, CPU, temperature, throttling). Stop `zovive-detect` first |
| `tools/hef_parity.py` | the HEF vs the ONNX model on the same images, on the Pi |
| `tools/camera_probe.py` | checks the real camera's streams against what the design assumes |
| `tools/mode_simulator.py` | replays wide-view footage through the zoom/mode state machine offline |
| `tools/profile_threads.py` | cProfile on every thread. **Doesn't work on Python 3.12+**, which allows only one profiler per process, so it doesn't run on the Pi |
| `edge_main.py --source file:<clip> --source-fps N` | replays a clip through the real pipeline |
| `tools/db_inspect.py`, `debug_viewer.py`, `motion_tuner.py`, `replay_clip.py`, `zone_editor.py` | **not implemented yet** (placeholders) |

Gate and soak scripts used on the Pi live in `zovive-ml/hailo_builds/det150e_h8l_r1/` (sibling project):

| Script | Purpose |
|---|---|
| `pi/soak_sampler.py` | read-only, one CSV row a minute (43 columns) |
| `pi/dump_detections.py` | per-image detections and raw heads |
| `pi/bench_compare.py` | compares benchmark runs |
| `score_pi_gates.py`, `ap50_eval.py` | scores the gates |

---

## Telemetry

**`/var/lib/zovive/edge_status.json`** is rewritten every 5 s:

- liveness timestamps for the inference and pipeline loops;
- the inference error count and active tracks;
- the zoom state;
- `stream` (FPS, stalled, reconnects, frames);
- `queues` (depth, maxsize, `drop_count` for each);
- `supervisor` (state and restarts for each worker);
- `dispatch` (dispatched, rejected, failed, retries, queued, video paths);
- `process` (start and exit history).

The loop timestamps are only as fresh as the file, so judge liveness as `loop age − file age`.

**Heartbeats** are written every 15 s by `zovive-health` to the local `heartbeats` table and the base station.

**Soak CSV** (`soak_sampler.py`) columns include:

- the service state, PID and restarts;
- process memory, CPU and threads;
- SoC temperature, throttle flags, free memory and disk;
- loop ages, stream FPS, queue drops;
- dispatch counters, sampler errors.

The agreed pass criteria for a 7-hour soak are in the R1 build's `pi/RUNBOOK_R1_PI.md` ("Production soak").

---

## Deploying to the Pi

The Pi has no internet, so code reaches it as a **git bundle**, never by editing files on the Pi. Detailed, tested runbooks live in `zovive-ml/hailo_builds/det150e_h8l_r1/pi/`:

| Runbook | For |
|---|---|
| `RUNBOOK_DEPLOY_689b72d.md` | a code deploy with restart, checks and rollback (the template for future deploys) |
| `RUNBOOK_BENCH_689b72d.md` | benchmarking two commits side by side in a staging clone, without touching production |
| `RUNBOOK_R1_PI.md` | gates C and D, and the production soak criteria |

**The deploy flow:**

1. **Laptop:** all checks pass, then commit and push. Create the bundle from the commit the Pi has:

   ```bash
   git bundle create zovive-edge_<branch>_<newhash>.bundle <pi-hash>..<branch>
   ```

   Write `SHA256SUMS` beside it.
2. **Copy** the bundle with `scp` to `/home/zovive1/r1_gate/transfer_<newhash>/`.
3. **Pi, read-only checks:**
   - the checksums match;
   - production `HEAD` is the expected commit, with no tracked changes;
   - no file the update adds would land on an untracked file in `/opt/zovive`.
4. **Pi, deploy:**
   - Fetch the bundle into `refs/remotes/bundle/<branch>`, then `git merge --ff-only <newhash>`.
   - Before restarting, load the changed configs and import the changed modules.
   - Then `sudo systemctl restart zovive-detect`.
   - Check: `active`, the Hailo device held, all four workers running, the same PID after 2 minutes.
5. **Soak** with `soak_sampler.py`, 30 minutes or 7 hours, scored on the laptop.

**Rollback** to the previous good commit:

```bash
git -C /opt/zovive switch --detach <good-hash> && sudo systemctl restart zovive-detect
```

The good commits for the current deploy are `52bb464` (the re-alert fix without the persistent runner) and `b5b8bc8` (before both). Rollback deletes nothing.

**Safety rules on the Pi:**

- Production stays unchanged until the owner approves a deploy.
- Arm a `systemd-run` safety timer whenever the service is stopped for a test.
- Use `export GIT_OPTIONAL_LOCKS=0` for read-only git commands.
- Never print `.env`.
- Delete evidence only after the owner acknowledges each item.

---

## Current state and known issues

| Item | Status |
|---|---|
| Production code | `689b72d`, deployed 30 Oct 2026 (21:48Z by the Pi's clock). Content verified against git; 30-minute soak passed |
| Detection-queue drops while a track is active | **open**. The pipeline thread can fall behind inference while tracking (2.2 drops/s overnight on `b5b8bc8`; 4,857 drops in the first hour after the latest boot). Not yet measured under a live track on `689b72d` |
| Lens control | `runtime.environment` is `dev` in this repository and on the Pi. Production therefore runs the zoom logic against a **fake lens**: the real lens doesn't move, while tracking still pauses during simulated zooms. The CP Plus lens also has known command limits (`MoveStatus` always IDLE, ~7 s travel, commands within 15 s dropped) |
| Clock | no RTC battery or NTP. The Pi's clock goes back at every power loss, and the camera's on-screen time differs from the Pi's by 5 h 30 min (a time-zone setting on one device). An RTC cell or time sync from the base station/GPS is needed before field deployment |
| Power | repeated site power cuts. The service recovers by itself on boot; a UPS is recommended |
| Hailo driver warning | `hailo_pci` 4.23.0 on kernel 6.18 logs 30 `find_vma` warnings each time a process opens the device. They appear with old and new code alike, and are worth reporting to Hailo |
| Sloth bear accuracy | 4.3-point AP50 drop on the chip, accepted. Needs sloth bear images from more than one source in the next model round |
| `models/manifest.json` | `sha256_hef` is still `null`, although R1 is deployed |
| Profiling on the Pi | `tools/profile_threads.py` needs rework for Python 3.12+ (a sampling profiler) |

---

## Further reading

| Doc | Covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | threads, processes, backend swap, SAHI, clips, retention |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | the long-form guide: versions, the detection path, folder by folder |
| [docs/WATCHDOG.md](docs/WATCHDOG.md) | self-recovery layers in detail |
| [docs/PIPELINE_HARDENING_SPEC.md](docs/PIPELINE_HARDENING_SPEC.md) | per-class thresholds, the stage-1 gate, async dispatch (D1-D5) |
| [docs/REALERT_SUPPRESSION_SPEC.md](docs/REALERT_SUPPRESSION_SPEC.md) | the per-place alert memory (A1-A4) |
| [docs/PERSISTENT_VSTREAMS_SPEC.md](docs/PERSISTENT_VSTREAMS_SPEC.md) | the persistent Hailo runner and its Pi benchmark (P1-P4) |
| [docs/STORAGE_LAYOUT.md](docs/STORAGE_LAYOUT.md), [docs/PURGE_AND_CACHE.md](docs/PURGE_AND_CACHE.md) | disks, deletion on confirmed receipt |
| [docs/CAMERA_MODES.md](docs/CAMERA_MODES.md), [docs/PATROL_AND_TRIGGER.md](docs/PATROL_AND_TRIGGER.md), [docs/TE41ZL6C_MODES_PLAN.md](docs/TE41ZL6C_MODES_PLAN.md) | camera modes and zoom |
| [docs/FIELD_INSTALL.md](docs/FIELD_INSTALL.md), [docs/NETWORK_MAP.md](docs/NETWORK_MAP.md), [docs/RUNBOOK.md](docs/RUNBOOK.md) | installing and running a node |
| [docs/TESTING.md](docs/TESTING.md) | test approach |
| [docs/WORKLOG.md](docs/WORKLOG.md) | the history behind the decisions |
