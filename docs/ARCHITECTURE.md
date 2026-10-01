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
                     -> on CONFIRMED: pipeline/stage2_verifier classifies
                        the best few crops (pipeline/best_snapshot) and
                        combines them via classifier.voting_strategy
                        ("average" softmax-blend or "majority" vote)
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
- `transfer_main.py` — polls `db/outbox` for pending events *and*
  pending video clips (independently — see "Event video" below),
  sends each to the base station (`network/base_station_client.py`),
  retries failed ones with backoff (`network/transfer_state.py`, one
  instance per kind so a slow video upload never delays image alerts).
- `health_main.py` — periodic heartbeat: queue depths, stream health,
  disk usage, Pi throttle status.

## Backend swap (ONNX <-> Hailo)

`inference/base.py` defines `Detector`/`Classifier` as the only
contract the pipeline depends on. `configs/node_config.yaml`'s
`runtime.backend` picks `inference/onnx_inference.py` (laptop, CI, or
the Pi's CPU) or `inference/hailo_inference.py` (Pi + Hailo-8) in
`edge_main.build_detector_and_classifier()` — nothing else in the
codebase branches on backend. If `backend: hailo` is set but
`HailoDetector`/`HailoClassifier` fail to construct (HailoRT missing,
no device, a bad `.hef`), `build_detector_and_classifier()` catches
`InferenceBackendError` and falls back to the ONNX backend on the
Pi's CPU with a logged warning, rather than crash-looping the whole
process under systemd.

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

## ACK-triggered local cleanup

The intended storage model: the Pi holds temporary copies (images,
videos, SQLite metadata) while the base station holds the permanent
ones (images, and optionally videos, plus the event database). Once a
send gets an ACK, the Pi's copy of that artifact is no longer needed;
if it doesn't, the Pi's copy is the only copy and must be kept.

`transfer_main.py`'s `_send_pending_images`/`_send_pending_videos` call
`storage/local_file_cleanup.delete_local_file()` right after
`mark_sent()`/`mark_video_sent()` — i.e. only on the success path,
never speculatively — and record the deletion in `db/outbox`'s
`snapshot_deleted_ts`/`video_deleted_ts` columns. A failed send takes
the existing `except BaseStationError` path unchanged: nothing is
deleted, the file stays, and the existing per-event backoff
(`network/transfer_state.py`) retries it. This is gated by
`transfer_config.yaml`'s `transfer.delete_local_files_after_ack`
(default `true`), which `transfer_main.run()` forces off whenever
`base_station.url` is empty — `NullBaseStationClient` "succeeds"
unconditionally so the rest of the pipeline can be exercised without a
real base station, and without this guard that would delete every
evidence file immediately with nothing backing it up anywhere.

## Why gating exists

The field report's false positives (Tiger->Leopard/Elephant,
empty-forest->Leopard, 1-of-45 Wild Boar recall) are a dataset/model
problem the retraining plan (see the project report) addresses
directly. What the software adds on top:

1. A track needs 3 of the last 5 frames as "animal present"
   (`pipeline/stage1_gate.py`) before anything happens — a single-frame
   false positive is just noise, not an alert.
2. Classification averages softmax scores over up to 5 of the
   sharpest crops (`pipeline/stage2_verifier.py`), not one frame.
3. Below `classifier.min_confidence`, the event is reported as
   `unknown_animal` with the snapshot rather than a guessed species.

None of this fixes a genuinely bad model — it only stops a good-but-imperfect
model's occasional bad frame from becoming a false alert.

See `../README.md` for what's implemented vs. deferred.
