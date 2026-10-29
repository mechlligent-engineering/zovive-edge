# Pipeline hardening spec

Status: approved 29 Oct 2026 with decisions D1-D5 as recommended (section 8).
Baseline: `develop` at `e28cc9d`. Phase B (changes 1 and 2) is implemented on
`feature/pipeline-hardening`; change 3 is Phase C. Not yet reconciled with the
nine locally modified tracked files on the Pi (section 6).

## 1. Goals and non-goals

**Goals:**

- Fewer false and duplicate alerts.
- No disk writes on the capture, inference or pipeline threads.
- Keep detecting real animals, including still and distant ones.

**Non-goals:**

- Model retraining: rhinos and empty-scene false positives need data.
- Moving the outbox (it stays in `zovive.sqlite3`).
- Replacing `IouTracker` with ByteTrack.
- Changing the MP4 encoder.
- Changing the zoom or camera logic.
- Deploying to the Pi.

## 2. Threads after the change

```
capture  --LatestFrameSlot-->  inference  --DetectionQueue-->  pipeline  --DispatchQueue-->  dispatch (new)
(newest frame only)             (decode, then per-class          (tracker, gate, vote,          (snapshot JPEGs, outbox row,
                                 thresholds: change 1)            zoom, clip start: change 2)    alert_queue: change 3)
                                                                                                  clip writer threads (exist)
```

Every thread is supervised (`watchdog/supervisor.py`). The new dispatch worker
gets its own liveness timestamp.

## 3. Change 1: per-class confidence thresholds

**Config** (`configs/inference_config.yaml`, under `detector:`):

```yaml
detector:
  confidence_threshold: 0.50          # decode threshold, and the default for any class not listed
  class_confidence_thresholds:        # new; D1 values
    elephant: 0.60
    gaur: 0.60
    leopard: 0.55
    wild_boar: 0.55
    tiger: 0.50
    sloth_bear: 0.50
    deer: 0.50
```

The first proposal (elephant 0.70, tiger 0.65, leopard 0.65, gaur, wild boar
and sloth bear 0.60, deer 0.55) was replaced by the data-backed set in the
table below (decision D1).

The spec's top-level `global_min_confidence` maps onto the existing `detector.confidence_threshold` key. That keeps one key and one place, and `edge_main.py` (both detector builders) and `tools/hef_parity.py` already read it.

**Behaviour:**

- Detectors decode at `confidence_threshold`.
- In the inference thread, right after `detector.infer()` and before `DetectionQueue`, each detection is dropped if its score is below its class's threshold.
- The filter is a pure function in a new module, `inference/class_thresholds.py`. It covers the Hailo, ONNX and SAHI paths alike.

**Validation** (`utils/config_loader.py`):

- Every key must be one of `detector.classes`.
- Every value must be in (0, 1] and at least `confidence_threshold`.
- A bad config fails at startup with `EXIT_CONFIG`.

**Measured effect** (laptop, 679 test images, 730 boxes; the R1 quantized emulator approximates the deployed HEF). The report is `zovive-ml/hailo_builds/det150e_h8l_r1/reports/ap50/threshold_sim_r1.json`. These are still images: no tracking, and decoded before the `acf1192` decoder restore.

| Class | 0.45 now: precision / recall | Proposed value: precision / recall | Data-backed alternative: precision / recall |
|---|---|---|---|
| elephant | 0.894 / 0.808 | 0.70: 0.935 / **0.737** | **0.60: 0.937 / 0.763** |
| tiger | 0.976 / 0.790 | 0.65: 0.973 / **0.686** | **0.50: 0.976 / 0.790** |
| leopard | 0.959 / 0.940 | 0.65: 0.977 / 0.860 | **0.55: 0.978 / 0.880** |
| sloth_bear | 0.956 / 0.782 | 0.60: 0.953 / 0.745 | **0.50: 0.956 / 0.782** |
| gaur | 1.000 / 0.886 | 0.60: 1.000 / 0.886 | 0.60: 1.000 / 0.886 |
| wild_boar | 0.978 / 0.884 | 0.60: 0.983 / 0.864 | **0.55: 0.983 / 0.864** |
| deer | 0.980 / 0.942 | 0.55: 0.980 / 0.923 | **0.50: 0.980 / 0.942** |
| all classes | 624 TP, 26 FP | 591 TP, 17 FP | |

**What the numbers show:**

- **Tiger 0.65** loses 10 points of recall and gains no precision.
- **Elephant 0.70** reaches the same precision as 0.60 but loses 2.6 more points of recall.
- **Sloth bear 0.60** gains nothing.
- With `min_hits_in_window: 5` on top, a tiger whose per-frame confidence sits between 0.50 and 0.65 (distant or night IR) might never confirm at all.
- Neither set of values stops a **stable** false positive. The empty desk frame scores elephant 0.88. A rhino detected confidently as an elephant also passes. Those need exclusion zones (`zone_filter`) and new training data.

**Tests:**

- The filter: per class, default fallback, empty input.
- Config validation.
- The inference loop applies it to every backend.

## 4. Change 2: stricter confirmation gate

**Config** (under `detector:`, read by `Stage1Config`):

```yaml
  min_hits_in_window: 5           # was 3
  window_size: 7                  # was 5. "5 of the last 7 frames" tolerates two missed frames (leaves, blur); 5 = strictly consecutive
  min_track_duration_sec: 1.0     # new
  min_box_area_px: 0              # new; 0 = off until measured on real camera frames
  track_alert_cooldown_sec: 300   # unchanged (D3)
```

**A track is confirmed only when all of these hold:**

1. **Hits:** `presence_history` has at least `min_hits_in_window` hits in the last `window_size` frames. The existing `presence_vote_passed`.
2. **Duration:** `last_seen_ts - first_seen_ts >= min_track_duration_sec`. Both fields already exist on `Track`.
3. **Confidence:** the mean confidence of the track's `class_votes` for its leading class (up to `max_frames_per_track` = 10 frames) is at least that class's threshold from change 1.
4. **Size:** the box area in original-frame pixels is at least `min_box_area_px`.
5. **Cooldown:** the track is not in its alert cooldown (`in_cooldown`, existing).

**Where:**

- `pipeline/stage1_gate.py`: new fields in `Stage1Config`, and new checks in `evaluate()`.
- `track_state_machine.py` keeps its API. The stage-2 species vote and `unknown_animal` behaviour are unchanged.

**Frame rate:** the reported rate is about 10-12 FPS, so 5 hits take about 0.5 s and the 1.0 s duration is the binding condition.

**Tests:** in `tests/test_stage1_gate.py`, one case per condition, plus the cooldown and a missed-frame tolerance case. `test_pipeline` and `test_video_pipeline_integration` are updated.

## 5. Change 3: alert disk writes off the pipeline thread

**Today:**

- `AlertDispatcher.dispatch()` saves the snapshot JPEG and inserts the outbox row **on the pipeline thread**.
- MP4 clips are already written by background threads: 8 s pre-roll from the RAM ring buffer plus 8 s post-roll, at most 4 at once.

**New `pipeline/dispatch_worker.py`:**

- **Submitting:** a `DispatchWorker` thread with a bounded `queue.Queue(maxsize=64)`. The pipeline thread builds an `AlertJob` and calls `submit()` (`put_nowait`).
  - The job carries the event id, track id, species, confidence, priority, view, model version, camera id and the snapshot image.
  - The event id is a `uuid4`, generated at submit time so the clip can start immediately.
- **The worker:** saves the snapshot(s), then inserts the outbox row, then pushes to `alert_queue`. That's the same order as today, so an outbox row never points to a missing snapshot.
- **Clips:** `ClipExtractor.start()` stays on the pipeline thread; it only copies frame references from the RAM buffer. The clip's `set_video_path` callback is routed through the dispatch queue, so it always runs after its event row exists.
- **Queue full:** the pipeline thread never blocks. The job goes to a bounded overflow list (256) on the pipeline side and is retried every cycle. Beyond that, an ERROR log and an edge-status counter record it. An alert is lost only if the disk stalls for minutes.
- **Write failures:**
  - A job is retried up to 3 times with backoff, then logged as ERROR and counted. This replaces today's "raise, so the supervisor restarts the pipeline thread".
  - The worker is a `SupervisedThread`, so a restart keeps queued jobs.
- **Shutdown:** the worker drains for up to 10 s before `clip_extractor.flush_all()`.

**Snapshots:** one per event, the best frame from `BestSnapshotStore`, as today (decision D4). A second snapshot would need a DB migration and a transfer change, and is out of scope.

**Tests:**

- Ordering (snapshot, then row, then queue).
- Queue full and overflow.
- Retries.
- The `set_video_path` ordering.
- The shutdown drain.
- A pipeline thread that never blocks on a slow fake disk.

## 6. Reconciliation with the Pi (before any deployment)

- The Pi runs `84652ff` plus nine locally modified tracked files, which haven't been reviewed yet. If any of them touch thresholds, the gate, snapshots or clips, this spec is adjusted to keep working fixes.
- Deployment follows `RUNBOOK_R1_PI.md`, only after:
  - a verified backup;
  - the reconciliation above;
  - a PR merged with CI green.
- Thresholds and `min_box_area_px` get reviewed against real CP Plus frames and the Pi's event counts. These are read-only database queries per hour, per track and per event.

## 7. Delivery

- **Branch:** `feature/pipeline-hardening` from `develop`.
- **Commits:** one per change, in order: 1, then 2, then 3. Each comes with its tests and passes the CI checks locally: ruff, pytest, unittest and bandit.
- **Pi:** no deployment and no config change on the Pi as part of this work.

## 8. Decisions (approved 29 Oct 2026)

| ID | Decision |
|---|---|
| D1 | Data-backed thresholds: elephant 0.60, gaur 0.60, leopard 0.55, wild boar 0.55, tiger 0.50, sloth bear 0.50, deer 0.50; decode threshold 0.50 |
| D2 | 5 hits in the last 7 frames (`min_hits_in_window: 5`, `window_size: 7`) |
| D3 | Cooldown stays at 300 s per track |
| D4 | One snapshot per event |
| D5 | `min_box_area_px: 0` (off) until measured on real frames |
