# Patrol and trigger modes

## What exists

| Function | Module | Default |
|---|---|---|
| Trigger source registry + arbitration | `camera_control/trigger_modes.py` | active |
| Camera ONVIF motion events | `camera_control/motion_event_listener.py` | active |
| Software motion (MOG2) | `capture/motion_detector.py` | active |
| Periodic safety sweep | `pipeline/motion_gate.py` (`sweep_interval_s`) | active |
| PTZ preset patrol | `camera_control/patrol_controller.py` | **disabled** |
| Per-preset state | `camera_control/preset_context.py` | pass-through on fixed cameras |

## Trigger priority

    1  remote / manual trigger      (operator asks for a look)
    2  camera ONVIF motion event
    3  software motion (MOG2)
    4  patrol preset arrival        (PTZ only)
    5  periodic safety sweep

A higher source suppresses lower ones inside `cooldown_s`, so two detectors seeing the
same animal produce one event. The safety sweep is exempt in both directions: it never
suppresses and is never suppressed, because its whole job is catching the animal that
stopped moving.

Every event records `gate_source`. After a month in the field that column tells you
whether camera-side motion is pulling its weight or just adding false triggers.

## Do you actually need patrol?

Only if the node has a pan/tilt head. The blueprint specifies a fixed 4K bullet, and a
fixed bullet cannot patrol - `zoom_controller.py` gives you varifocal zoom, not
coverage. For a fixed camera the sweep already provides what patrol is for: periodic
re-examination of the scene independent of motion.

Patrol earns its place when one node must cover several fence segments that do not fit
in one field of view. The cost is coverage gaps - while the head is looking west, an
animal crossing east is simply not seen. Three presets at 20 s dwell means each segment
is unwatched for 40 seconds out of every 60. Decide whether that is acceptable before
choosing PTZ over three fixed cameras; three fixed nodes usually beat one patrolling
node for a fence line.

## The three things patrol breaks

**1. Motion detection during movement.** While the head moves, every pixel changes.
Background subtraction reports the entire frame as motion, and a naive gate will fire on
every single move. Fix: `is_settled()` gates everything, plus `settle_ms` dead time and
`warmup_frames` before the per-preset background model is trusted again.

**2. One background model cannot serve several presets.** MOG2 learns a scene. Point it
somewhere else and its model is wrong until it relearns, which takes long enough that
you miss the first animal at every preset. Fix: `per_preset_models: true` - a separate
model instance per preset, kept warm across visits.

**3. Track identity does not survive a move.** ByteTrack ids, cooldowns and zone
polygons are all scoped to a view. Carrying them across a preset change produces
phantom tracks and suppressed alerts. Fix: `preset_context.py` swaps the whole context
on every move; ids restart per preset.

## Lock-on

When Stage 1 confirms an animal, patrol pauses. The verifier needs a stable scene for
the 5-frame burst, the sharpness filter, the ROI cluster and the 10 s clip. Moving the
head mid-event produces a blurred snapshot and an unusable clip.

`max_lock_s` exists because without it a herd that settles in front of preset 2 stops
the tour indefinitely and the other segments go dark for hours. When the cap trips, log
it - repeated cap trips mean the site needs another node, not a longer cap.

## Evidence during a move

`suspend_ring_during_move: true` stops writing segments to the tmpfs ring while the head
is in motion. A clip whose pre-roll is a smear of a moving camera is worse than a
shorter clip, and it still costs full bandwidth to upload.
