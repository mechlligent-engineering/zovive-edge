# Transfer protocol (edge <-> base station)

Both sides implement this. Change it in both repositories or not at all.

```
-> TRANSFER_BEGIN {event_id, artifact, filename, size, sha256, chunk_size, chunk_count}
<- TRANSFER_READY {event_id, resume_from_chunk}
-> CHUNK          {event_id, index, sha256, data}
<- CHUNK_ACK      {event_id, index}   |   CHUNK_NAK {event_id, index, reason}
-> TRANSFER_END   {event_id}
<- FINAL_ACK      {event_id, verified: true, sha256}
<- FINAL_NAK      {event_id, reason}
```

Rules:
- Purge only on FINAL_ACK with verified true.
- `resume_from_chunk` is authoritative and comes from the base station. Local
  bookkeeping can be stale after an unclean shutdown.
- Every message carries `event_id`. The base station must handle a duplicate
  `event_id` idempotently: a node that reboots after TRANSFER_END but before
  FINAL_ACK will retry the whole artifact.
- Tier 1 (alert JSON + snapshot) is a single message on the same connection and
  preempts chunk sending.
