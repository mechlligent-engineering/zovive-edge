"""vcgencmd get_throttled + SoC/NVMe temperature; reports under-voltage.

Only the SoC temperature reader exists so far (used by
tools/benchmark_edge.py). The throttle flags are read by
health_main._pi_throttle_status; the TRIGGER-mode thermal guard is a
later phase (docs/TE41ZL6C_MODES_PLAN.md).
"""

from __future__ import annotations

from pathlib import Path

# Raspberry Pi OS exposes the SoC sensor here, in millidegrees Celsius.
SOC_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def read_soc_temp_c(path: Path = SOC_TEMP_PATH) -> float | None:
    """SoC temperature in °C, or None where the sensor doesn't exist (off-Pi)."""
    try:
        return round(int(path.read_text().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return None
