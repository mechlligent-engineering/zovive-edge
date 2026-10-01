# Runbook

**Node dark.** Check heartbeat gap, then link (RSSI at the AP), then power. A node that
is up but not reporting will still be recording to `/data` - do not reflash it, pull the
evidence first.

**Backlog growing.** `tools/db_inspect.py --unacked`. If the base station is up, check
`max_backlog_bandwidth_mbps` and the link throughput. Backlog with a healthy link means
the base station is NAKing - check the protocol version on both sides.

**Disk filling.** `storage/disk_guard.py --report`. If unacked count is high the link has
been down; if acked artifacts are not being purged, `purge_service` is failing - check
`purge.log` and the journal table.

**Thermal throttling.** `utils/thermal_monitor.py --report`. Sustained throttling in an
enclosure means the heatsink or airflow is wrong; the software cannot fix it.

**Replacing a Pi.** New unit, same static IP, boots and registers its new MAC. The base
station operator re-binds the node. Restore `/data` from the old NVMe if it survived.
