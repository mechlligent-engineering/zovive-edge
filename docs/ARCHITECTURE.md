# Architecture

## Threads (edge_main.py, one process)

```
capture thread (RtspReader | FileFrameSource)
    -> queues/frame_queue.LatestFrameSlot  (latest frame only, never a backlog)
        -> inference thread (Detector.infer per frame — optionally
           inference/sahi_slicer.SahiDetector wrapping the real
           ONNX/Hailo detector; see "SAHI" below)
            -> queues/detection_queue.DetectionQueue  (bounded, drop-oldest)
                -> pipeline thread:
                     inference/bytetrack_wrapper.IouTracker.update()
                     -> pipeline/zone_filter (drop out-of-zone detections)
                     -> every cycle: pipeline/clip_extractor.offer_frame()
                        (feeds the rolling pre-event buffer regardless of
                        whether anything is recording — see "Event video")
                     -> pipeline/track_state_machine.advance() per track
                        (NEW -> CONFIRMING -> CONFIRMED, gated by
                         pipeline/stage1_gate's 3-of-5 presence vote
                         + per-track cooldown)
                     -> on CONFIRMED: pipeline/stage2_verifier votes the
                        species from the track's per-frame detector
                        (class, score) results via stage2.voting_strategy
                        ("average" or "majority"); no second model runs.
                        pipeline/best_snapshot picks the alert snapshot
                     -> pipeline/alert_dispatcher: db/outbox row first
                        (durable), then storage/evidence_store snapshot,
                        then queues/alert_queue for the sender
                     -> on a dispatched alert: pipeline/clip_extractor.start()
                        begins an event video clip (pre-roll already
                        buffered + live post-roll), written async and
                        attached to the same event_id via
                        db/outbox.set_video_path() when ready
                     -> camera_control/patrol_controller.tick() each
                        cycle: MOVING -> SETTLING -> SCANNING -> (ENGAGED
                        on a confirmed target) -> COOLDOWN -> MOVING
```

## Processes

Three separate processes/systemd units, so a stuck one can't stall the
others:

- `edge_main.py` — the pipeline above.
- `transfer_main.py` — claims due events *and* due video clips from
  `db/outbox` (two independent queues — see "Event video" below),
  sends each to the base station (`network/base_station_client.py`).
  Retry state is stored on the outbox rows (`db/outbox.claim_due`), so it
  survives restarts; `network/retry_policy.py` has the rules: an HTTP 4xx
  rejection costs a retry (30 s, 1 m, 2 m, 5 m, 15 m, then dead-lettered),
  an unreachable or busy base station pauses the queue and is probed
  about once a minute at no cost.
- `health_main.py` — periodic heartbeat: queue depths, stream health,
  disk usage, Pi throttle status.

## Backend swap (ONNX <-> Hailo)

`inference/base.py` defines `Detector` as the only contract the
pipeline depends on (single-stage: the 7-species detector is the
node's only model). `configs/node_config.yaml`'s `runtime.backend`
picks `inference/onnx_inference.py` (laptop, CI, or the Pi's CPU) or
`inference/hailo_inference.py` (Pi + Hailo-8) in
`edge_main.build_detector()` — nothing else in the codebase branches
on backend. If `backend: hailo` is set but `HailoDetector` fails to
construct (a missing `.hef`, HailoRT missing, no device, a bad
`.hef`, or outputs that are not the split head below), `build_detector()`
catches `FileNotFoundError` / `InferenceBackendError` and falls back to
the ONNX backend on the Pi's CPU with a logged warning, rather than
crash-looping the whole process under systemd. `runtime.backend` is
`hailo` in the repo, so a laptop takes that fallback too.

The two backends get the same model in two forms. `detector.onnx` outputs
the decoded head, (1, 11, 8400). `detector.hef` is cut before the head's
DFL decode (zovive-ml `hailo_config/yolov11.alls`): it takes the 640x640
letterboxed frame as uint8 RGB and divides by 255 on the device, and has
six outputs — per stride 8/16/32 a box conv (64 channels) and a class
conv (7 channels). `inference/yolo_postprocess.py` turns those into the
same rows the ONNX head gives (`decode_split_heads`; outputs matched by
shape, not layer name) and both go through the same confidence filter,
NMS and letterbox reversal. zovive-ml's
`scripts/eval_audit/verify_split_head_decoder.py` checks the decoder
against the float model; `python -m tools.hef_parity --images ...`
compares the compiled HEF with `detector.onnx` on the Pi.

## SAHI (tiled inference)

`inference/sahi_slicer.SahiDetector` wraps whichever `Detector` backend
was built above — it implements the same `Detector` interface, so
nothing downstream knows the difference. When
`configs/inference_config.yaml`'s `detector.sahi.enabled` is true, it
slices a frame larger than `tile_size` into overlapping tiles, runs the
wrapped detector on each (plus one full-frame pass by default), and
merges the results with per-class NMS. Off by default and a no-op on a
frame that already fits in one tile, so it never changes behavior on
the existing 640x360 sub-stream path unless `tile_size` is set smaller
than the capture resolution to deliberately force tiling for small/far
animals.

## Event video (clip recording + upload)

`pipeline/ram_buffer.RamBuffer` holds the last `pre_event_seconds` of
frames at all times (fed every pipeline cycle, cheap). The moment an
alert is dispatched, `pipeline/clip_extractor.ClipExtractor.start()`
snapshots that buffer as the clip's pre-roll and keeps collecting
`post_event_seconds` more via the same per-cycle feed; once the window
closes, the MP4 is written on a short-lived background thread (never
the pipeline thread) and `db/outbox.set_video_path()` marks it ready.
`transfer_main.py` uploads it via `BaseStationClient.send_video()` on
its own retry timeline (`video_status`, independent of the image
alert's `status`) — a lost or slow clip never risks the durability
guarantee the image alert already has once it's in `db/outbox`.

## ACK-based local retention

The Pi holds temporary copies of the evidence; the base station holds the
permanent ones. A local snapshot or clip may be deleted only once the base
station has ACKed that file itself:

- `transfer_main.py` records `snapshot_acked_ts` / `video_acked_ts` (db/outbox)
  in the same UPDATE that marks the artifact sent, and only when a real base
  station answered 2xx, without a redirect, to a request that carried the
  complete file. A metadata-only send, the `NullBaseStationClient` (empty
  `base_station.url`) and every row sent before schema v4 leave it NULL.
- `storage/retention_manager.py` deletes ACKed media recorded and ACKed at
  least 14 days ago (daily, `zovive-retention.timer`), and, when the
  filesystem holding the media reaches 82 %, ACKed media oldest first at any
  age until it is back to 79 % (`zovive-storage-pressure.timer`). Thresholds
  live in `configs/retention_policy.yaml`.
- Unacknowledged evidence is never deleted automatically. It is counted
  (overdue after 14 days) in `retention_status.json` and the heartbeat's
  `retention` section.

`transfer_config.yaml`'s `delete_local_files_after_ack` (now `false`) still
deletes a file right after its own ACK when turned on, bypassing the 14-day
local window; `transfer_main.run()` forces it off without a base station URL.

## Why gating exists

The field report's false positives (Tiger->Leopard/Elephant,
empty-forest->Leopard, 1-of-45 Wild Boar recall) are a dataset/model
problem the retraining plan (see the project report) addresses
directly. What the software adds on top:

1. A track needs 3 of the last 5 frames as "animal present"
   (`pipeline/stage1_gate.py`) before anything happens — a single-frame
   false positive is just noise, not an alert.
2. The species is voted over up to 10 of the track's recent detector
   results (`pipeline/stage2_verifier.py`), not taken from one frame.
3. Below `stage2.min_confidence`, the event is reported as
   `unknown_animal` with the snapshot rather than a guessed species.

None of this fixes a genuinely bad model — it only stops a good-but-imperfect
model's occasional bad frame from becoming a false alert.

See `../README.md` for what's implemented vs. deferred.
