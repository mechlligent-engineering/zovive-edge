#!/usr/bin/env bash
# DESTRUCTIVE. Partitions the 256 GB NVMe for a ZOVIVE node.
# Read docs/STORAGE_LAYOUT.md first. This script refuses to run without --yes-i-am-sure.
set -euo pipefail

DISK="${DISK:-/dev/nvme0n1}"

# Target layout (256 GB):
#   p1  /boot/firmware    512M  vfat
#   p2  /                  32G  ext4   OS + /opt/zovive code + venv + models
#   p3  /var/log/zovive     8G  ext4   logs only
#   p4  /var/lib/zovive     8G  ext4   SQLite state - must stay writable when /data is full
#   p5  /data             rest  ext4   evidence blobs only
#
# Rationale: a runaway evidence directory or a log flood must never fill the
# partition that holds the OS, the code, or the database.

echo "TODO: implement with sgdisk/parted. Refuse unless DISK is empty or --yes-i-am-sure is passed."
