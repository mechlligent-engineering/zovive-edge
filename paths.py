"""Central path and directory-layout definitions for the ZOVIVE edge node.

Every other module imports paths from here instead of building them
ad-hoc, so the on-disk layout only has to be decided once. Directories
are created on import (idempotent) so callers never need `mkdir -p`
sprinkled around.

The root can be overridden with the ZOVIVE_HOME environment variable,
which is how tests and the CI runner keep the real Pi data directory
(/var/lib/zovive by default) out of the way of a laptop checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Roots ------------------------------------------------------------

# Where the code lives (this file's directory).
PROJECT_ROOT = Path(__file__).resolve().parent

# Where runtime *data* (db, snapshots, logs, models) lives. On the Pi
# this is a separate mount (often the USB SSD); on a dev laptop it
# defaults to <project>/var so nothing touches system paths.
DATA_ROOT = Path(os.environ.get("ZOVIVE_HOME", PROJECT_ROOT / "var")).resolve()

# --- Config -------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs"

NODE_CONFIG = CONFIG_DIR / "node_config.yaml"
RTSP_CONFIG = CONFIG_DIR / "rtsp_config.yaml"
INFERENCE_CONFIG = CONFIG_DIR / "inference_config.yaml"
CAMERA_MODES_CONFIG = CONFIG_DIR / "camera_modes.yaml"
TRACKING_CONFIG = CONFIG_DIR / "tracking_config.yaml"
QUEUE_CONFIG = CONFIG_DIR / "queue_config.yaml"
LOGGING_CONFIG = CONFIG_DIR / "logging_config.yaml"
DETECTION_ZONES_CONFIG = CONFIG_DIR / "detection_zones.yaml"
ALERT_POLICY_CONFIG = CONFIG_DIR / "alert_policy.yaml"
RETENTION_POLICY_CONFIG = CONFIG_DIR / "retention_policy.yaml"
TRANSFER_CONFIG = CONFIG_DIR / "transfer_config.yaml"
WATCHDOG_CONFIG = CONFIG_DIR / "watchdog_config.yaml"
CLIP_CONFIG = CONFIG_DIR / "clip_config.yaml"


# --- Models ---------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_BACKUP_DIR = MODELS_DIR / "backup"
MODEL_MANIFEST = MODELS_DIR / "manifest.json"

# --- Runtime data (under DATA_ROOT so it's easy to point at a big disk) --

DB_DIR = DATA_ROOT / "db"
DB_PATH = DB_DIR / "zovive.sqlite3"

SNAPSHOT_DIR = DATA_ROOT / "snapshots"
CLIP_DIR = DATA_ROOT / "clips"
# Written by edge_main.py every few seconds, read by health_main.py
# (watchdog/edge_status.py).
EDGE_STATUS_PATH = DATA_ROOT / "edge_status.json"
# edge_main.py process starts and why the previous run ended
# (watchdog/restart_history.py); survives systemd restarts.
RESTART_HISTORY_PATH = DATA_ROOT / "restart_history.json"
LOG_DIR = DATA_ROOT / "logs"
CACHE_DIR = DATA_ROOT / "cache"
QUARANTINE_DIR = DATA_ROOT / "quarantine"  # files that failed validation

_RUNTIME_DIRS = (
    DATA_ROOT,
    DB_DIR,
    SNAPSHOT_DIR,
    CLIP_DIR,
    LOG_DIR,
    CACHE_DIR,
    QUARANTINE_DIR,
    MODELS_DIR,
    MODELS_BACKUP_DIR,
)


def ensure_runtime_dirs() -> None:
    """Create every runtime directory if missing. Safe to call repeatedly."""
    for d in _RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()
