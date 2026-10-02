# ZOVIVE Developer Guide

How the node actually works, what runs in parallel, what to write and in what order.
Read `docs/ARCHITECTURE.md`, `docs/CAMERA_MODES.md` and `docs/STORAGE_LAYOUT.md`
alongside this.

---

# Part 0 — Versions

## Pin these, and record them

The one rule that will save you a week: **the Hailo Dataflow Compiler version used to
build a `.hef` must match the HailoRT version on the node.** A mismatch does not fail
cleanly - it either refuses to load or produces wrong tensors. Record the pair in
`zovive-edge/models/manifest.json` for every model you ship.

| Component | Version | Notes |
|---|---|---|
| Raspberry Pi OS | Bookworm or Trixie, 64-bit | 64-bit is not optional for HailoRT |
| Kernel | >= 6.6.31 | below this the Hailo PCIe driver will not load |
| Python (node) | 3.11 (Bookworm) / 3.13 (Trixie) | use the system Python; do not build your own |
| HailoRT + driver | whatever `hailo-all` installs on your Pi today | **read it off the node, then pin it** |
| Hailo Dataflow Compiler | the version matching HailoRT above | training machine only |
| ultralytics | 8.4.x | pin exactly, e.g. `ultralytics==8.4.120` |
| Model | **YOLO11** | see below |
| torch / torchvision | whatever the pinned ultralytics requires | training machine only |
| opencv-python | 4.10–4.13 | node: prefer `opencv-python-headless` |
| numpy | 2.x | must match the opencv wheel you install |
| sahi | 0.11.x | training-side tuning only; the node uses our own tiling |
| ffmpeg | 6.x or 7.x, system package | remux only, never encode |
| pytest | 8.x | dev machine |

Get the node's actual versions before pinning anything:

```bash
apt list --installed 2>/dev/null | grep -i hailo
hailortcli fw-control identify
uname -r && python3 -VV
```

## Why YOLO11 and not YOLO26

YOLO26 shipped in January 2026 with NMS-free end-to-end inference, which sounds ideal
for edge. But NMS-free changes exactly the part that is hardest to compile for Hailo -
the `nms_postprocess` block in `hailo_config/yolov11.alls`. YOLO11 has a proven Hailo
Model Zoo compilation path and known-good `.alls` references.

Build on YOLO11 now. Revisit YOLO26 once Hailo publishes a reference `.alls` for it -
at that point it is a retrain and recompile, not an architecture change, because
nothing in `zovive-edge/` depends on which YOLO generation produced the `.hef`.

---

# Part 1 — What runs in parallel

## Three processes, not one

```
┌─ zovive-detect.service ────────────────────────────────────────────────┐
│  T1 rtsp_reader (sub)      T2 rtsp_reader (main)   T3 segment_recorder │
│  T4 motion+Stage1 loop     T5 Stage2 worker        T6 mode_manager     │
└──────────────────────┬─────────────────────────────────────────────────┘
                       │ writes evidence + rows
                       ▼
                  SQLite (/var/lib/zovive)  +  blobs (/data/evidence)
                       │
┌─ zovive-transfer.service ──────────────┐  ┌─ zovive-health.service ────┐
│  T1 priority_dispatcher (owns socket)  │  │  T1 hardware watchdog      │
│  T2 chunked_uploader                   │  │  T2 health_reporter        │
│  T3 ack_listener                       │  │  T3 disk_guard             │
│  T4 retention_manager / purge          │  │  T4 thermal_monitor        │
└────────────────────────────────────────┘  └────────────────────────────┘
```

Separate processes because a crash in the uploader must not stop detection, and a
crash in detection must not strand evidence that was already recorded. They share
state only through SQLite - never in-process globals, never a shared queue object.

**Real-life analogy.** A forest checkpost. One person watches the road (detect), one
runs the radio and the logbook (transfer), one checks the generator and the water
(health). If the radio operator falls asleep, the watcher still watches, and the log
is still there when he wakes.

## Threads inside zovive-detect

| Thread | Module | Runs | Blocks on |
|---|---|---|---|
| T1 | `capture/rtsp_reader.py` (sub) | always | socket read |
| T2 | `capture/rtsp_reader.py` (main) | always | socket read |
| T3 | `capture/segment_recorder.py` | always except during PTZ moves | ffmpeg subprocess |
| T4 | `pipeline/motion_gate.py` → `stage1_gate.py` | ~10 fps patrol, ~25 fps trigger | frame_queue |
| T5 | `pipeline/stage2_verifier.py` | only on a confirmed animal | detection_queue |
| T6 | `camera_control/mode_manager.py` | state ticks + PTZ commands | ONVIF round trip |

T4 and T5 both want the NPU. `inference/npu_scheduler.py` is the only thing that
touches the device: T4's requests are high priority, T5's are preemptible. Two threads
opening HailoRT contexts independently is the fastest way to a node that stalls under
load.

## One event, on the clock

```
t+0.00  T4  motion blob passes the gate
t+0.02  T4  npu_scheduler → generic model → animal 0.71
t+0.30  T4  track 17 reaches min_age → NEW → event opens
t+0.31  T6  mode_manager: PATROL → ACQUIRING   (T3 suspends the ring)
t+0.31  T5  pulls 5 frames from ram_buffer, starts sharpness scoring   ← parallel
t+2.40  T6  head settled, zoomed, ACQUIRING → TRIGGER
t+2.45  T5  roi_cluster → 4 tiles → npu_scheduler → species model → elephant 0.88
t+2.60      alert_dispatcher → alert_queue → (transfer process picks it up)
t+2.9   transfer: tier-1 alert + snapshot on the wire
t+12.0  T3  post-roll done → clip_extractor muxes the MP4      ← parallel with tracking
t+13.0  transfer: chunked upload starts, yielding between chunks
t+31.0  transfer: FINAL_ACK → purge_service deletes blobs + cache
```

The thing to notice: Stage 2 sharpness scoring starts *while* the camera is still
moving, because the burst frames were already in RAM before the move began. That is
what the RAM buffer buys you.

---

# Part 2 — How the model detects

## Two models, on purpose

| | Stage 1 gate | Stage 2 species |
|---|---|---|
| classes | 1 (`animal`) | 5 (tiger, elephant, boar, deer, bear) |
| input | 640×640 letterboxed from 1080p sub-stream | 640×640 tiles from the cluster ROI |
| runs | continuously during patrol | only on a confirmed animal |
| optimise for | **recall** | **precision** |
| cost of an error | a missed animal is unrecoverable | a wrong species is embarrassing, not fatal |

Why not one 5-class model at 30 fps? Because the expensive question (*which* species)
only needs answering a few times an hour, while the cheap question (*is anything
there*) needs answering constantly. Splitting them is what lets the node idle at low
power for 99% of its life.

Train the gate to over-trigger. A false positive costs one Stage 2 pass — about 8 ms
of NPU. A false negative costs an elephant.

## The detection path, precisely

```python
# 1. letterbox - preserve aspect ratio, remember scale and padding
img, scale, pad = letterbox(frame, (640, 640))

# 2. NPU. Normalisation happens inside the .hef (see yolov11.alls), so do NOT
#    divide by 255 here as well - doing it twice is a silent accuracy killer.
raw = npu_scheduler.submit(model="generic", tensor=img, priority=HIGH)

# 3. decode + NMS -> boxes in 640x640 letterboxed space
boxes, scores, classes = yolo_postprocess.decode(raw, conf=cfg["conf_threshold"])

# 4. undo the letterbox -> boxes in real frame pixels
boxes = unletterbox_boxes(boxes, scale, pad, frame.shape)
```

Then, only if something was found:

```python
cluster = cluster_box(boxes, frame.shape, pad_ratio=0.10, min_size=640)
tiles = compute_tiles(cluster_w, cluster_h, 640, 640, 0.2, 0.2)
crops = [region[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]
raw = npu_scheduler.submit_batch("species", crops, priority=LOW)
boxes, scores, classes = merge_detections(per_tile_results, iou_thr=0.5)
```

`merge_detections` is not optional. Overlapping tiles deliberately see the same animal
twice; without NMS across the merged set you report two elephants.

## Where SAHI actually helps

SAHI is for **small targets**. A deer 200 m away in a 4K frame is maybe 40×30 px; after
downscaling to 640×640 it is 7×5 px and invisible. Cropping the ROI and running it at
native resolution turns that into a 40×30 px target in a 640 tile - detectable.

So the rule: SAHI on the cluster ROI at native resolution, never on the whole downscaled
frame. If `cluster_area_ratio` approaches 1.0, the animals are spread across the whole
scene, the cluster is saving nothing, and you should split them into groups.

## Day and night are two different problems

IR frames are monochrome, lower contrast, and lit unevenly by the illuminator. A single
confidence threshold underperforms at night, which is when most wildlife moves.

- separate thresholds: `inference_config.yaml` has `day:` and `ir:` blocks
- IR frames in training, in the `*_ir` splits
- IR frames in the Hailo calibration set (`calibration_images/ir/`)
- report metrics separately — `evaluate_ir_vs_rgb.py`. A blended mAP hides night failure
  until the field trial exposes it.

---

# Part 3 — Folder by folder

Legend: **[done]** working code you can run today · **[you]** you write this ·
**[stub]** contract written, needs hardware

## zovive-edge/capture — the eyes

Real-life: the person who keeps the binoculars steady. Does not decide anything, just
guarantees a clean, current view.

| File | Role | |
|---|---|---|
| `rtsp_reader.py` | dual-stream reader, bounded queue, drop-oldest | **[you]** |
| `segment_recorder.py` | 2 s segments into tmpfs ring, `-c copy` | **[you]** |
| `motion_detector.py` | MOG2 on downscaled sub-stream | **[you]** |
| `stream_health.py` | fps, gaps, corruption | **[you]** |
| `reconnect.py` | backoff helper | **[you]** |

Non-obvious: **drop-oldest, not block.** If the pipeline stalls for two seconds, you
want the newest frame, not a two-second-old one. Blocking here means the queue grows
until the Pi OOMs.

## zovive-edge/inference — the NPU

Real-life: one microscope, several researchers, a booking sheet. `npu_scheduler` is the
booking sheet.

| File | Role | |
|---|---|---|
| `preprocess.py` | letterbox / unletterbox / IR handling | **[done]** |
| `sahi_batched.py` | tile geometry, remap, NMS merge | **[done]** |
| `hailo_inference.py` | HailoRT: load `.hef`, run, return tensors | **[stub]** |
| `npu_scheduler.py` | single device owner, priority queue | **[you]** |
| `yolo_postprocess.py` | decode + NMS | **[you]** |
| `bytetrack_wrapper.py` | association over Stage 1 output | **[you]** |

## zovive-edge/pipeline — the decisions

Real-life: the ranger who watches the feed and decides what is worth radioing in. This
is the folder that makes ZOVIVE ZOVIVE; everything else is plumbing.

| File | Role | |
|---|---|---|
| `sharpness_filter.py` | crop Laplacian variance over the burst | **[done]** |
| `roi_cluster.py` | one enclosing box for N animals | **[done]** |
| `zone_filter.py` | polygons, exclusions, feet-not-head anchor | **[done]** |
| `track_state_machine.py` | NEW / ALERTED / EXPIRED | **[done]** |
| `target_selector.py` | who gets the PTZ lock | **[done]** |
| `motion_gate.py` | blob size, persistence, cooldown, sweep | **[you]** |
| `stage1_gate.py` | animal-or-not orchestration | **[you]** |
| `stage2_verifier.py` | burst → sharpness → cluster → SAHI → species | **[you]** |
| `clip_extractor.py` | concat segments, remux | **[you]** |
| `clip_quality_gate.py` | reject blurred / truncated clips | **[you]** |
| `best_snapshot.py` | write the chosen JPEG | **[you]** |
| `target_handoff.py` | keep event_id across the zoom | **[you]** |
| `ram_buffer.py` | 5-frame deque | **[you]** |
| `alert_dispatcher.py` | build payload, hand to transfer | **[you]** |

The five **[done]** files come with 27 passing tests. Run them now:

```bash
cd zovive-edge && python -m pytest tests/ -q
```

## zovive-edge/camera_control — the neck and the lens

Real-life: the tripod head plus the person turning it. Only one person may touch it.

| File | Role | |
|---|---|---|
| `ptz_mapping.py` | box → pan/tilt/zoom, FOV interpolation, deadband | **[done]** |
| `mode_manager.py` | PATROL/ACQUIRING/TRIGGER/SEARCHING/RETURNING | **[you]** |
| `ptz_tracker.py` | closed-loop correction, rate limit, lead | **[you]** |
| `patrol_controller.py` | preset tour, `is_settled()` | **[you]** |
| `preset_context.py` | per-preset background/zones/track namespace | **[you]** |
| `profile_switcher.py` | encoder + IR profile per mode | **[you]** |
| `camera_api.py` | ONVIF / CGI client | **[stub]** |
| `motion_event_listener.py` | ONVIF pull-point subscription | **[stub]** |
| `ir_mode_manager.py` | day/night detection and thresholds | **[you]** |

## zovive-edge/storage — the evidence locker

Real-life: the store room with a register at the door. Nothing enters or leaves without
a line in the register.

| File | Role | |
|---|---|---|
| `evidence_store.py` | blob + DB row atomically, or neither | **[you]** |
| `purge_service.py` | delete on FINAL_ACK, journalled, idempotent | **[you]** |
| `cache_manager.py` | `/var/cache/zovive`, `/run/zovive/staging` | **[you]** |
| `retention_manager.py` | applies `retention_policy.yaml` | **[you]** |
| `disk_guard.py` | mount sentinel, free space, eviction | **[you]** |
| `orphan_scanner.py` | reconcile disk vs DB after a crash | **[you]** |

Non-obvious: write the blob to a temp name, fsync, insert the DB row, then rename. A
blob with no row is an orphan; a row with no blob is a broken upload that retries
forever. Both are avoidable with ordering.

## zovive-edge/network — the radio

| File | Role | |
|---|---|---|
| `priority_dispatcher.py` | owns the socket, tier 1 preempts tier 2 | **[you]** |
| `chunked_uploader.py` | BEGIN / CHUNK / END, per-chunk SHA-256 | **[you]** |
| `ack_listener.py` | FINAL_ACK is the only purge trigger | **[you]** |
| `transfer_state.py` | artifact state machine in SQLite | **[you]** |
| `transfer_protocol.md` | the contract with the base station | **[done]** |
| `boot_registration.py`, `heartbeat.py`, `sync_daemon.py`, `alert_schema.py` | | **[you]** |

## The rest, briefly

- **queues/** — bounded queues. Real-life: a counter with room for eight forms; the
  ninth pushes the oldest off. Unbounded means OOM.
- **watchdog/** — the person who checks everyone else is awake, and kicks `/dev/watchdog`
  so the kernel reboots the Pi if he himself stops.
- **ota/** — model updates with checksum and rollback. Never ship without rollback to a
  node you cannot drive to.
- **utils/** — thermal (`vcgencmd get_throttled`), disk, network diagnostics.
- **provisioning/** — turns a blank Pi into a node. `partition_nvme.sh` is destructive:
  read `docs/STORAGE_LAYOUT.md` first.
- **tools/** — your daily drivers: `debug_viewer.py` for aiming, `zone_editor.py` for
  polygons, `motion_tuner.py` and `mode_simulator.py` for tuning against footage,
  `ptz_calibrate.py` once per camera model.
- **db/** — `schema.sql` on `/var/lib/zovive`, deliberately not on `/data`.

## zovive-ml

Real-life: the classroom. Nothing here runs in the forest.

Order: `validate_dataset.py` → `deduplicate_frames.py` → `split_dataset.py` →
`compute_class_weights.py` → `train_*.py` → `evaluate_test_set.py` +
`evaluate_ir_vs_rgb.py` → `export_onnx.py` → `calibrate_hailo.py` →
`compile_hailo.sh` → `model_versioning.py`.

Two rules worth repeating: dedupe **before** splitting (burst frames leak across splits
and inflate your mAP), and never tune against the test split.

---

# Part 4 — Where you actually code

## Week 1 — laptop only, no hardware

1. `python -m pytest tests/ -q` — 27 tests pass against the **[done]** modules. Read
   them; they are the specification.
2. `pipeline/ram_buffer.py` — a bounded deque. 30 lines.
3. `pipeline/motion_gate.py` — pure logic over a frame sequence, testable with a video
   file. Write `tests/test_motion_gate.py` alongside.
4. `pipeline/stage1_gate.py` with a **fake** inference backend that returns canned
   detections. This proves the whole decision chain — gate → detect → track → zone →
   alert — before any NPU exists.

## Week 2 — the Pi, no camera

5. `provisioning/` end to end. Partition, HailoRT, static IP, systemd, `db/init_db.py`.
6. `inference/hailo_inference.py` against a stock model zoo `.hef`, then
   `npu_scheduler.py`, then `yolo_postprocess.py`. Verify with `tools/benchmark_edge.py`.

## Week 3 — the camera

7. `capture/rtsp_reader.py` dual-stream, then `segment_recorder.py`, then
   `clip_extractor.py`. Prove you can produce a correct 10 s clip with pre-roll and no
   transcode. Everything downstream depends on this one thing working.
8. `camera_control/camera_api.py`, then `tools/ptz_calibrate.py`. Do the calibration
   properly here; every trigger-mode bug later traces back to it.

## Week 4 — the modes

9. `mode_manager.py`, `ptz_tracker.py`, `patrol_controller.py`, `target_handoff.py`.
   Tune against footage with `tools/mode_simulator.py` before touching the real head.

## Week 5 — evidence and delivery

10. `storage/evidence_store.py` → `network/chunked_uploader.py` +`ack_listener.py`
    against a stub base station → `purge_service.py` → `disk_guard.py`.
    Test resume by killing the process mid-transfer.

## Week 6 — the field

11. `watchdog/`, `ota/`, `utils/thermal_monitor.py`, the runbook.

Rule of thumb: if a module needs hardware to test, its hardware-touching part is in the
wrong file. Push it down into `capture/` or `hailo_inference.py` and keep the decision
logic pure.

---

# Part 5 — Where to put your own ideas

Every one of these is a defined seam. Adding an idea should mean writing one new module
and changing one config key — if it means editing five files, the seam is in the wrong
place and it is worth telling me what you are trying to do.

| Your idea | Where it goes | Config key |
|---|---|---|
| Different motion algorithm (optical flow, frame-diff, thermal) | new module in `capture/`, selected by `motion_config.yaml → source` | `source` |
| Extra filter before alerting (herd size, direction of travel, dwell time) | new module in `pipeline/`, called from `stage2_verifier.py` after species | add to `alert_policy.yaml` |
| Different lock priority (nearest to gate, fastest moving, calf present) | `pipeline/target_selector.py` — swap `_rank()` | `alert_policy.yaml` |
| Species-specific behaviour (elephants get longer clips, tigers get max zoom) | `alert_policy.yaml → per_species`, read in `stage2_verifier` and `mode_manager` | `per_species` |
| Deterrent output (siren, strobe, SMS) | new `actuators/` package, triggered from `alert_dispatcher.py` | new `actuator_config.yaml` |
| Second sensor (thermal camera, PIR, fence vibration) | new reader in `capture/`, fused in `motion_gate.py` as another trigger source | `motion_config.yaml` |
| Re-identification (is this the same elephant as last week?) | new `pipeline/reid.py` after species; embedding stored on the `events` row | new `reid_config.yaml` |
| Counting / crossing direction | `pipeline/zone_filter.py` already gives zone entry; add a line-crossing module beside it | `detection_zones.yaml` |
| On-device active learning (save hard negatives for retraining) | `storage/evidence_store.py`, a second artifact kind `training_sample` | `retention_policy.yaml` |
| Different model architecture | nothing in `zovive-edge/` changes — swap the `.hef` and `manifest.json` | `inference_config.yaml` |

Two seams deliberately left open for you:

- **`motion_gate.py` trigger fusion.** It takes trigger sources and returns one decision.
  Adding a PIR sensor or a fence-vibration input is a new source, not a rewrite.
- **`stage2_verifier.py` post-species hook.** Everything you want to compute about a
  confirmed animal — direction, count, re-ID, behaviour — hangs here, after the species
  is known and before the alert is built.

Tell me which of your ideas you want to build first and I will write that module
properly, with tests, in the same style as the five that are already done.
