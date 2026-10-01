# Cloning a node

Build node-01 fully, verify it in the lab, then image it. Per-node differences are
exactly three files - keep it that way:

1. `.env` (NODE_ID, camera credentials)
2. `configs/node_config.yaml` (node_id, static_ip, site)
3. `configs/detection_zones.yaml` (site-specific polygons)

Everything else is identical across nodes. If you find yourself hand-editing a
fourth file per node, that value belongs in one of these three.

After cloning: regenerate SSH host keys, clear `/data/evidence`, clear the database,
and re-run boot registration so the base station sees the new MAC.
