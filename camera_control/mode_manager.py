"""Top-level operating mode: AUTO (zoom_controller may zoom and focus on
its own) vs MANUAL (an operator has taken control of the lens and
zoom_controller must not start new zoom sessions).

Thread-safe because edge_main.py's own signal handling and any future
operator tooling both need to flip this safely.
"""

from __future__ import annotations

import threading


class OperatingMode:
    AUTO = "auto"
    MANUAL = "manual"


class ModeManager:
    def __init__(self, default_mode: str = OperatingMode.AUTO):
        self._lock = threading.Lock()
        self._mode = default_mode

    def set_manual(self) -> None:
        with self._lock:
            self._mode = OperatingMode.MANUAL

    def set_auto(self) -> None:
        with self._lock:
            self._mode = OperatingMode.AUTO

    def is_auto(self) -> bool:
        with self._lock:
            return self._mode == OperatingMode.AUTO

    @classmethod
    def from_config(cls, cfg: dict) -> ModeManager:
        m = cfg.get("mode", {})
        return cls(default_mode=m.get("default_mode", OperatingMode.AUTO))
