# Patrol mode and trigger mode

## The intent

**PATROL** — camera on the preset tour, wide view, continuous stream, Stage 1 gate
running at a low rate on the 1080p sub-stream. Cheap, always on.

**TRIGGER** — an animal is confirmed, the head drives onto it, zooms until it fills
~40% of frame, switches the encoder to full 4K, and tracks it closed-loop until it
leaves. Then back to patrol.

Implemented by `camera_control/mode_manager.py`. It is the only component permitted to
issue PTZ commands.

## The five states

```
PATROL ──animal confirmed──► ACQUIRING ──locked──► TRIGGER
  ▲                              │                    │
  │                         acquire fail         target lost
  │                              ▼                    ▼
  └──────── RETURNING ◄──── SEARCHING ◄───────────────┘
```

`ACQUIRING` and `RETURNING` exist because the camera is *moving* during them, and a
moving camera produces detections, motion and video that are all worthless. Both
suspend detection and suspend the pre-roll ring.

## Five things that will bite you

### 1. Calibration is the whole game
Trigger mode is a coordinate transform: a box at (0.62, 0.41) in the wide frame becomes
an absolute pan/tilt and a zoom factor. That needs the true FOV at each zoom step, the
slew rate, the command latency and the backlash — measured, not from the datasheet.
Run `tools/ptz_calibrate.py` on site. A mapping three degrees off gives you an empty
zoomed frame and no way to recover except returning to patrol.

Zoom in **steps** (2x → check → 4x → check), never one blind jump to maximum.

### 2. The tracker does not survive the zoom
After the move, ByteTrack sees a completely different scene. New id, invalid background
model, and to the alert logic it looks like a brand new animal — so you get two alerts
for one elephant. `pipeline/target_handoff.py` keeps `event_id` stable across the move
and validates re-acquisition (same class, plausible size for the zoom factor, near
centre). If re-acquisition fails, the event still closes with the wide-view snapshot.
A wide-only alert is a valid alert.

### 3. Ping-pong is the classic failure
Zoom in → target drifts out of the narrow frame → lost → return to patrol → see it
again → zoom in → lose it. The camera spends the whole encounter swinging and records
nothing usable.

Three guards, all in `camera_modes.yaml`: `deadband` (don't correct for small drift),
`min_patrol_dwell_s` (must be > 0 — never re-trigger instantly on return), and
`retrigger_cooldown_s` per target. Tune them on footage with `tools/mode_simulator.py`
before you tune them in the field.

### 4. While zoomed, the rest of the fence is blind
At 8x you see perhaps 5% of the patrol coverage. A second animal crossing elsewhere is
simply not seen. That is the real cost of one PTZ node versus several fixed ones, and
it is why `max_trigger_s` (default 180 s) is a hard cap, not a suggestion. When the cap
trips repeatedly at one site, the answer is another node, not a longer cap.

`pipeline/target_selector.py` decides who gets the lock when several animals are
visible — species priority, then zone, then size, then confidence. It does not
re-evaluate every frame; a selector that keeps changing its mind swings between two
animals and records neither.

### 5. Chasing wears the motor and drains the node
`max_moves_per_min` caps corrections. If a target needs more than the cap, it is moving
too fast to hold at that zoom — zoom out one step rather than chase. Use the ByteTrack
velocity to lead the target; a correction aimed where the animal *was* always lags by
the command latency plus the settle time.

## Where the energy saving actually comes from

Be realistic about this. The Pi 5 and the Hailo-8 dominate the node's power draw, and
the PTZ motors only draw while moving — mode switching saves little there.

The real savings, all handled by `profile_switcher.py`:

| | PATROL | TRIGGER |
|---|---|---|
| detection stream | sub 1080p | main 4K |
| detection rate | ~10 fps Stage 1 only | ~25 fps + SAHI + species model |
| decode load | low | high |
| IR illuminator | low / auto | full |

Running the species model and SAHI only during trigger events, and decoding 1080p
instead of 4K the other 99% of the time, is where the compute, heat and power go down.
The camera profile switch also cuts IR illuminator draw at night, which is not nothing
on a solar node.

Switch profiles during `ACQUIRING`, while the head is moving anyway — a profile change
briefly interrupts the stream and must never land mid-recording.

## Evidence in trigger mode

You end up with two views of the same event, and both are worth keeping:

- **wide pre-roll** — the approach, direction of travel, context (from the tmpfs ring)
- **zoomed** — the clear species-ID frames

PTZ zoom is optical, so the frame geometry and codec do not change between them. The
segments concatenate with a stream copy, no re-encode. Keep the move itself in the clip
rather than cutting mid-GOP; a two-second swish is cheaper than a transcode.

Best snapshot comes from the zoomed view when acquisition succeeded, from the wide view
when it did not. `best_snapshot.py` picks by sharpness across both sets.
