# Deletion on confirmed receipt

## Trigger

`FINAL_ACK {event_id, verified: true, sha256}` from the base station, meaning it
re-hashed the fully assembled file and it matched. Nothing else deletes evidence:

- a per-chunk `CHUNK_ACK` is **not** permission to delete;
- `TRANSFER_END` sent by us is not permission to delete;
- a timer, a reboot, or "the queue looks long" is not permission to delete.

## What gets deleted

`purge_service.purge_event(event_id)` removes, in this order:

1. **Blobs** - `/data/evidence/<event_id>/clip.mp4`, `snapshot.jpg`, `meta.json`, then the directory.
2. **Cache derivatives** - `/var/cache/zovive/thumbs/<event_id>*`, the quality-scaled JPEG that was actually transmitted, any generated preview.
3. **Staging** - `/run/zovive/staging/<event_id>/` (tmpfs; partial mux output, temp files).
4. **Segment pins** - any tmpfs ring segments still pinned for this event are released back to the ring.
5. **Chunk rows** - `DELETE FROM chunks WHERE artifact_id IN (...)`. These exist only to support resume; once verified they are dead weight.
6. **Queue entries** - offline_queue and alert_queue rows for the event.

## What is kept

The `events` row. A few hundred bytes: timestamp, species, confidence, zone, IR mode,
count. Keep it indefinitely. It is your field history, your false-alarm analysis, and
what you show a forest department that asks how many elephants crossed in March. It is
the blobs that are expensive, not the metadata.

`artifacts` rows are kept too, with `state = PURGED`, `path` cleared and `purged_utc`
set - so you can prove an artifact existed and was delivered.

## Crash safety

Purge is journalled and idempotent. `purge_journal` gets a row before the first
destructive step and is closed after the last. On startup, `crash_recovery` finds any
open journal row and re-runs `purge_event` for it - re-running a completed purge is a
no-op by design (every step is "delete if exists").

`orphan_scanner` runs hourly and reconciles both directions:
- a file on disk with no `artifacts` row -> orphan, delete it and log;
- an `artifacts` row in a live state whose file is missing -> mark DISCARDED, log, and
  do not keep retrying an upload of a file that no longer exists.

## What never happens silently

If `disk_guard` has to evict an artifact that is not yet ACKed, it raises a health
alert to the base station and increments a counter that appears in the heartbeat.
Evidence loss is always visible. A system that quietly deletes un-delivered evidence is
worse than one that fills up, because you find out during the review, not during the
outage.
