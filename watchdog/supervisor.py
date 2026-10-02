"""In-process supervisor for edge_main.py's worker threads.

Python cannot kill or replace a thread that is stuck, only one that has
already exited. So the supervisor recovers the two cases differently:

    crashed (target raised) : run the worker's reset hook, wait the backoff
                              (1, 2, 4 ... up to 60 s), start a new thread.
                              More than `max_restarts` within `window_sec`
                              -> escalate.
    hung (alive, but its liveness timestamp is older than
          `hang_threshold_sec`) : escalate immediately.

"Escalate" means edge_main.py records the reason and exits the whole
process non-zero, so systemd (Restart=always) starts it fresh. A hang of
the main thread itself is caught by the systemd watchdog instead
(watchdog/sd_notify.py, WatchdogSec in zovive-detect.service): main only
pings while `check()` is running and finding nothing to escalate.

`check()` is meant to be called from one thread (edge_main's main loop);
workers only touch their own SupervisedThread via the wrapper.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

RUNNING = "running"
BACKOFF = "backoff"
STOPPED = "stopped"  # target returned normally (shutdown, --max-cycles, end of a file source)
NOT_STARTED = "not_started"

_MAX_ERROR_CHARS = 300


@dataclass
class RestartPolicy:
    max_restarts: int = 5
    window_sec: float = 600.0
    initial_backoff_sec: float = 1.0
    max_backoff_sec: float = 60.0
    hang_threshold_sec: float = 60.0

    @classmethod
    def from_config(cls, watchdog_cfg: dict) -> RestartPolicy:
        s = watchdog_cfg.get("supervisor", {}) or {}
        d = cls()
        return cls(
            max_restarts=max(0, int(s.get("max_restarts", d.max_restarts))),
            window_sec=max(1.0, float(s.get("restart_window_sec", d.window_sec))),
            initial_backoff_sec=max(0.0, float(s.get("initial_backoff_sec", d.initial_backoff_sec))),
            max_backoff_sec=max(0.0, float(s.get("max_backoff_sec", d.max_backoff_sec))),
            hang_threshold_sec=max(5.0, float(s.get("hang_threshold_sec", d.hang_threshold_sec))),
        )

    def backoff(self, recent_restarts: int) -> float:
        return min(self.initial_backoff_sec * (2 ** recent_restarts), self.max_backoff_sec)


class SupervisedThread:
    """One worker: a zero-arg `target` run on a daemon thread, plus how to
    tell whether it is alive (`liveness`, a wall-clock timestamp the worker
    refreshes every loop iteration) and how to make restarting it safe
    (`on_restart`, called before each new thread starts)."""

    def __init__(
        self,
        name: str,
        target: Callable[[], None],
        liveness: Callable[[], float | None] | None = None,
        on_restart: Callable[[], None] | None = None,
        restartable: bool = True,
    ):
        self.name = name
        self.target = target
        self.liveness = liveness
        self.on_restart = on_restart
        self.restartable = restartable

        self.state = NOT_STARTED
        self.restart_count = 0
        self.last_error: str | None = None
        self.last_failure_wall_ts: float | None = None
        self.last_restart_wall_ts: float | None = None
        self.started_wall_ts: float | None = None
        self._thread: threading.Thread | None = None
        self._crashed = False
        self._recent_restarts: deque[float] = deque()
        self._restart_at: float | None = None

    def start(self, wall_now: float) -> None:
        self._crashed = False
        self.started_wall_ts = wall_now
        self.state = RUNNING
        self._thread = threading.Thread(target=self._run, name=f"supervised-{self.name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.target()
        except Exception as exc:
            self._crashed = True
            self.last_error = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
            log.exception("worker thread crashed", extra={"worker": self.name})

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "restarts": self.restart_count,
            "last_error": self.last_error,
            "last_failure_ts": self.last_failure_wall_ts,
            "last_restart_ts": self.last_restart_wall_ts,
        }


class Supervisor:
    def __init__(
        self,
        policy: RestartPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.policy = policy or RestartPolicy()
        self.clock = clock
        self.wall_clock = wall_clock
        self.workers: dict[str, SupervisedThread] = {}

    def add(self, worker: SupervisedThread) -> SupervisedThread:
        self.workers[worker.name] = worker
        return worker

    def start_all(self) -> None:
        for w in self.workers.values():
            w.start(self.wall_clock())

    def check(self) -> str | None:
        """Inspect every worker once. Restarts crashed ones when due and
        allowed. Returns an escalation reason, or None if all is handled."""
        now, wall = self.clock(), self.wall_clock()
        for w in self.workers.values():
            reason = self._check_one(w, now, wall)
            if reason:
                return reason
        return None

    def _check_one(self, w: SupervisedThread, now: float, wall: float) -> str | None:
        if w.state == RUNNING and w.is_alive():
            if w.liveness is None:
                return None
            # The later of its last tick and its (re)start: a tick left over
            # from before a crash must not count against the new thread.
            last = max(w.liveness() or 0.0, w.started_wall_ts or 0.0)
            if wall - last > self.policy.hang_threshold_sec:
                return f"{w.name} hung: no progress for {wall - last:.0f} s"
            return None

        if w.state == RUNNING:  # thread has exited
            if not w._crashed:
                w.state = STOPPED
                return None
            w.last_failure_wall_ts = wall
            if not w.restartable:
                return f"{w.name} crashed and is not restartable: {w.last_error}"
            while w._recent_restarts and now - w._recent_restarts[0] > self.policy.window_sec:
                w._recent_restarts.popleft()
            if len(w._recent_restarts) >= self.policy.max_restarts:
                return (
                    f"{w.name} restart budget exhausted ({self.policy.max_restarts} in "
                    f"{self.policy.window_sec:.0f} s); last error: {w.last_error}"
                )
            delay = self.policy.backoff(len(w._recent_restarts))
            w._restart_at = now + delay
            w.state = BACKOFF
            log.warning(
                "worker crashed; restarting after backoff",
                extra={"worker": w.name, "backoff_sec": delay, "error": w.last_error},
            )

        if w.state == BACKOFF and now >= w._restart_at:
            if w.on_restart is not None:
                try:
                    w.on_restart()
                except Exception as exc:
                    return f"{w.name} reset before restart failed: {type(exc).__name__}: {exc}"
            w._recent_restarts.append(now)
            w.restart_count += 1
            w.last_restart_wall_ts = wall
            w.start(wall)
            log.warning("worker restarted", extra={"worker": w.name, "restarts": w.restart_count})
        return None

    def stopped(self, name: str) -> bool:
        w = self.workers.get(name)
        return w is not None and w.state == STOPPED

    def join_all(self, timeout_each: float = 5.0) -> None:
        for w in self.workers.values():
            w.join(timeout_each)

    def snapshot(self) -> dict:
        return {name: w.snapshot() for name, w in self.workers.items()}
