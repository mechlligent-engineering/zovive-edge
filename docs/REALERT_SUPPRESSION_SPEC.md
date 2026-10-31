# Re-alert suppression for detections that stay in view: design

Status: **approved 31 Oct 2026 (A1-A4) and implemented.**

- S1 is `pipeline/recent_alerts.py`, used by `edge_main._pipeline_loop` and `AnimalZoomController`, with `detector.realert_window_sec` in `inference_config.yaml`.
- S2 is `tracker.max_age_frames: 40`.
- Tests are in `tests/test_recent_alerts.py`.
- Written from the overnight soak on `b5b8bc8`.

## 1. Evidence

Sources: the R1 build folder's `reports/pi/soak_events_0827Z.csv` and the soak snapshots.

- **59 alerts**, all `elephant`, all from the wide view, confidence 0.62-0.76.
- **59 different track IDs**; the tracker issued `t-1` to `t-230` during the night, while `active_tracks` stayed at 1 or more the whole time.
- **Bursts:** 33 alerts in the first 17 minutes, often 5-60 s apart. Some came while two tracks were alive at once.
- **Snapshots:** an indoor bench scene. A large dark fabric close to the lens sits at the left edge of every frame. One frame is almost blank (overexposed) and still scored elephant 0.73. **Every alert was a false positive on something that never left the view.**

## 2. Root cause (code)

1. **The cooldown is per track ID.** `Stage1Gate.in_cooldown` keys `_last_alert_ts` by `track_id` (`pipeline/stage1_gate.py`), so a new track over the same object starts with no cooldown.
2. **The tracker breaks one detection into many tracks:**
   - `IouTracker` drops a track after `max_age_frames: 15` pipeline cycles without a match (`configs/tracking_config.yaml`). That's about 2 s at the ~8 cycles/s the pipeline managed with a track active.
   - Detections below the class threshold (elephant 0.60) are removed before tracking, and these scores hovered just above it.
   - About a fifth of batches never reached the tracker (`detq_drops`).
   - Any gap longer than 15 cycles ends the track, and the next detection starts a new one, which can confirm and alert again.
3. **The same-animal memory only works on the zoom path:**
   - `ZoomController.remember` and `match_known` (60 s) exist exactly to stop re-alerts under new track IDs. But only zoom sessions add entries (`complete()`, and `engaged_this_cycle` in `edge_main._pipeline_loop`).
   - A box that is already large, or at the frame edge, gets `plan_zoom() -> None` and is alerted from the wide view without being remembered. That covers all 59 alerts here.

## 3. Proposed changes

**S1, the main fix: remember every alert, and keep the memory fresh while the object stays in view.**

- After any dispatch that returns an `event_id`, wide or zoomed, record (box, class, species, confidence, `event_id`).
- **Refresh:** every pipeline cycle, any live track (in any state) whose centre lies inside a remembered box of the same class pushes that entry's expiry to `now + window`. This uses today's `_center_inside` test (box grown by 50%).
- **Window:** 300 s by default (the same as `track_alert_cooldown_sec`), configurable. Effect: one alert per stay in view. The same class at the same place alerts again only after it has been out of view for 5 minutes.
- **Where:**
  - Move the memory out of `ZoomController` into a small pipeline module (`pipeline/recent_alerts.py`). Both the wide and zoom paths use it, and it works with zoom disabled.
  - `ZoomController` keeps its re-zoom check by asking the same store.
- **Expected effect on this soak:** at most a handful of alerts instead of 59. A new alert would come only where the box moved off the remembered spot (the camera view changed during the night) or the object was out of view for 5 minutes. This is to be confirmed with a replay test, not assumed.
- **Accepted risk:** a second animal of the *same* species at the *same* spot within 5 minutes isn't alerted separately. Different species alert normally.

**S2, a config change: keep tracks through short gaps.** Change `max_age_frames` from 15 to 40, about 5 s at 8 cycles/s.

- This gives fewer track fragments and fewer re-confirmations.
- Cost: a track stays about 5 s after an animal leaves. That's no extra alert, since a track alerts once.
- Later, measuring the age in seconds rather than cycles would stop it shrinking when the pipeline is slow.

**S3, deferred: two-stage association.** Existing tracks could also match detections below the class threshold, for example down to 0.30, while new tracks still need the full threshold.

- That needs sub-threshold detections passed downstream, a larger change.
- Revisit after S1 and S2 have run in the field.

**S4, ML backlog: hard negatives.** Add these snapshots, plus empty forest frames, as negatives in the next training round. Indoor scenes were never in the training data.

**Not proposed: raising the elephant threshold.**

- This false positive scored up to 0.76, so the threshold would have to go above that.
- Elephant recall at 0.45 on the chip is already the lowest of the classes (0.80).

## 4. Tests

- **Unit:**
  - the memory records, refreshes and expires correctly;
  - a wide-view alert is remembered;
  - a new track over a remembered box is marked alerted with no new event;
  - a different class at the same spot still alerts;
  - after 300 s out of view, the same spot alerts again.
- **Pipeline:** a scripted detector holds a static box for 20 minutes, with its score dropping out for 20-cycle gaps every minute.
  - Expect exactly **one** event.
  - Today's code produces one per track fragment; the test documents the before and after.
- **Optional replay on the Pi**, in the same session as the InferVStreams benchmark: one soak clip through `edge_main --source file:`, with a scratch `ZOVIVE_HOME` and zoom off.

## 5. Decisions needed

- **A1:** S1 as described: a 300 s window refreshed while seen, in a new `pipeline/recent_alerts.py`.
- **A2:** S2, `max_age_frames` 40.
- **A3:** defer S3, and add S4 to the ML backlog.
- **A4:** ship S1+S2 as their own change, before the InferVStreams work. This is recommended: it fixes the alert flood, and it keeps the benchmark a one-variable comparison.
