# Detection self-recovery and watchdog

`edge_main.py` recovers from failures in three layers, each catching what the one
before it cannot.

| Layer | Catches | Action | Where |
| --- | --- | --- | --- |
| In-process supervisor | A worker thread that **raised** | Reset its state, restart the thread after a backoff | `watchdog/supervisor.py` |
| In-process supervisor | A worker thread that is **hung** (alive, no progress for `hang_threshold_sec`) or out of restarts | Exit code 70; systemd restarts the process | `watchdog/supervisor.py`, `edge_main._supervise` |
| systemd watchdog | The **main/supervisor thread** itself hung | systemd sends SIGABRT and restarts the service | `WatchdogSec=30` in `systemd/zovive-detect.service` |

A hung thread cannot be killed or replaced from inside Python, and a replacement would
fight the stuck one for the same NPU lock or camera socket. That is why hangs go
straight to a process restart, while crashes get an in-process restart first.

## Supervised workers

| Worker | Liveness signal | Reset before restart |
| --- | --- | --- |
| `capture` (`RtspReader.run` / `FileFrameSource.run`) | `loop_ts`, set every loop iteration including reconnect attempts | none; the capture handle is released on the way out |
| `inference` (`_inference_loop`) | `EdgeStatus.inference_loop_ts` | none |
| `pipeline` (`_pipeline_loop`) | `EdgeStatus.pipeline_loop_ts` | clear tracker, track records and snapshots; `zoom.abort()` drops any zoom session without an alert and returns the lens to wide |

Every loop ticks on each iteration, idle or busy, so an old timestamp means stuck, not
"no animals". The batch that caused a pipeline crash has already been taken off the
queue, so it is not replayed after the restart.

Timeouts that keep normal waits well under the 60 s hang threshold:

- RTSP open/read: 10 s (`CAP_PROP_OPEN_TIMEOUT_MSEC` / `CAP_PROP_READ_TIMEOUT_MSEC`); the
  reconnect backoff is at most 36 s.
- ONVIF zoom/focus calls: 5 s (`camera_api.ONVIF_TIMEOUT_SEC`).
- Slowest normal pipeline step: about 13.5 s (zoom settle + focus + zoomed timeout), all
  non-blocking.

## Restart policy (`configs/watchdog_config.yaml` → `watchdog.supervisor`)

| Key | Default | Meaning |
| --- | --- | --- |
| `initial_backoff_sec` | 1 | First restart delay; doubles each restart |
| `max_backoff_sec` | 60 | Cap on the delay |
| `max_restarts` | 5 | Restarts allowed per worker within the window |
| `restart_window_sec` | 600 | Sliding window; older restarts stop counting |
| `hang_threshold_sec` | 60 | No progress for this long = hung (minimum 5) |

The 6th crash of one worker within 10 minutes exits the process. With systemd's
`RestartSec=30` and `StartLimitIntervalSec=0`, a persistent fault (disk full, broken
model) retries every 30 s or so forever. It never gives up and never reboots the Pi.

## Exit reasons

Written to `var/restart_history.json` and reported in the heartbeat as
`edge.restarts.last_exit_reason` / `last_exit_code`:

| Reason starts with | Exit code | Meaning |
| --- | --- | --- |
| `shutdown requested` | 0 | SIGTERM/SIGINT (deploy, `systemctl stop`) |
| `<worker> hung: no progress for N s` | 70 | Supervisor escalation |
| `<worker> restart budget exhausted …; last error: …` | 70 | Supervisor escalation |
| `<worker> reset before restart failed` | 70 | The reset hook itself raised |
| `fatal: …` | 1 | Startup failure (missing model, bad config) or an unexpected exception |
| `unclean exit (killed, systemd watchdog, …)` | none | The previous run never reached its exit handler |

## Heartbeat fields (additive; nothing existing changed)

`edge.restarts`:

```json
{
  "process_starts": 4,
  "last_exit_reason": "pipeline restart budget exhausted (5 in 600 s); last error: OSError: ...",
  "last_exit_code": 70,
  "last_exit_ts": 1790000000.0,
  "thread_restarts": 2,
  "threads": {
    "pipeline": {"state": "running", "restarts": 2, "last_error": "OSError: ...",
                 "last_failure_ts": 1790000100.0, "last_restart_ts": 1790000101.0},
    "inference": {"state": "running", "restarts": 0, "...": "..."},
    "capture": {"state": "running", "restarts": 0, "...": "..."}
  }
}
```

These are reported even when `edge_process_alive` is false, since a crash loop is when
they matter most.

## systemd unit (`zovive-detect.service`)

- `Type=notify`, `NotifyAccess=main`: `READY=1` is sent once the worker threads are
  running (`watchdog/sd_notify.py`, standard library only, fixed messages, no shell).
- `WatchdogSec=30`: the supervisor loop pings about every 0.5 s, and only while
  `check()` finds nothing to escalate.
- `TimeoutStartSec=180`: covers model load and warmup, including the ONNX CPU fallback.
- `Restart=always`, `RestartSec=30`, `StartLimitIntervalSec=0`: never give up, slowly.

`zovive-transfer` and `zovive-health` are unchanged (`Type=simple`, restart on crash).
They are single-threaded loops with network timeouts; the same notify helper can be
added to them later.

## Deploying on the Pi

1. Check `systemctl --version`. Pi OS Bookworm ships systemd 252, which supports
   everything used here.
2. Copy the new unit, then `sudo systemctl daemon-reload && sudo systemctl restart zovive-detect`.
3. `systemctl show zovive-detect -p Type -p WatchdogUSec -p NRestarts`: expect
   `notify`, `30s`, and a restart count to watch over time.
4. `systemctl status zovive-detect` shows `active (running)` only after `READY=1`. If it
   sits in `activating` and fails after 180 s, the process never reached the supervisor
   loop; the journal says why.
5. Test once on a bench Pi: `sudo kill -STOP $(pidof -s python)` on the detect process.
   systemd should log a watchdog timeout and restart it within about 90 s: 30 s with no
   ping, then SIGABRT (a stopped process can't act on it, so SIGKILL follows after the
   30 s abort timeout), then 30 s `RestartSec`. A process that is hung but not stopped
   dies on the SIGABRT straight away.

Not done here: the hardware watchdog. `config.txt.snippet` enables the device
(`dtparam=watchdog=on`), but nothing arms it until `RuntimeWatchdogSec=15` is set in
`/etc/systemd/system.conf`. That recovers kernel hangs by rebooting, so try it on a bench
Pi first.
