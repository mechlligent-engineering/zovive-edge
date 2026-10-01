# Storage layout

## Why partition at all

One partition means one failure mode shared by everything. A week of unsent clips,
a log loop, or a debug dump fills the disk, and then: the OS cannot write, systemd
cannot restart services, the database cannot record an ACK - so nothing can ever be
purged. The node is bricked until someone drives to it.

Separate partitions turn that into a contained fault: the evidence partition fills,
the disk guard notices, eviction runs, alerts still flow, and the node keeps working.

## Layout (256 GB NVMe)

| Part | Mount | Size | Holds | If it fills |
|---|---|---|---|---|
| p1 | `/boot/firmware` | 512 MB | boot | n/a |
| p2 | `/` | 32 GB | OS, `/opt/zovive` code, venv, `.hef` models | should never grow; nothing writes here at runtime |
| p3 | `/var/log/zovive` | 8 GB | logs only | logrotate caps it; worst case logging degrades |
| p4 | `/var/lib/zovive` | 8 GB | SQLite state | bounded by row count; alarms long before full |
| p5 | `/data` | ~190 GB | evidence blobs only | disk guard evicts; detection continues |

Mount options: `noatime` everywhere (NVMe wear), `nodev,nosuid,noexec` on the data,
log and state partitions. `nofail` on `/data` so a failed evidence disk does not block
boot - the node comes up degraded and says so.

Run `tune2fs -m 1 /dev/nvme0n1p5` on the data partition. The ext4 default reserves 5%
for root, which is ~9 GB wasted on a partition root never writes to.

## The database does not live on /data

This is the important one. If `/data` is full and the database is on `/data`, the node
cannot write the row that says "this was ACKed", so `purge_service` never runs, so
`/data` stays full. Permanent deadlock. The database lives on its own small partition
that cannot be filled by evidence.

## Volatile storage

| Path | Backing | Contents | Lifetime |
|---|---|---|---|
| `/dev/shm/zovive/segments` | tmpfs (RAM) | pre-roll segment ring, 2 s each | ~30 s rolling |
| `/run/zovive/staging` | tmpfs (RAM) | partial mux output, in-progress JPEG | until the event completes |
| `/var/cache/zovive` | disk (p2) | thumbnails, compressed transmit copies | until purge, safe to delete any time |

The pre-roll ring is in RAM by design. Writing the stream to SSD 24/7 to keep 30
seconds of history would destroy the NVMe write budget for no benefit. Only confirmed
events reach the disk.

## Mount sentinel

`/data/.zovive_mounted` is created on the data filesystem itself. On startup and before
every write, `disk_guard` checks for it. If `/data` failed to mount, the sentinel is
absent - the directory underneath is empty - and without this check the node would
happily write evidence into the root partition and fill it in a day. Missing sentinel
means: refuse to write evidence, raise a health alert, keep detecting and alerting.
