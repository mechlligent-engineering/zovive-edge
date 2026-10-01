#!/usr/bin/env bash
# One-shot node build. Run once on a fresh Raspberry Pi OS (64-bit) install.
set -euo pipefail

# 1. base packages
# 2. create the zovive user and /opt/zovive
# 3. ./install_hailort.sh
# 4. ./partition_nvme.sh        <- destructive, read it first
# 5. apply config.txt.snippet   (PCIe Gen 3, watchdog)
# 6. install static_ip.nmconnection
# 7. install systemd units, tmpfiles, logrotate, chrony
# 8. python -m db.init_db
# 9. systemctl enable --now zovive-detect zovive-transfer zovive-health

echo "TODO: implement. Read docs/STORAGE_LAYOUT.md before running partition_nvme.sh."
