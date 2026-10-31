# Persistent Hailo inference pipeline: design and benchmark plan

Status: **approved 31 Oct 2026 (P1-P4); option A implemented, waiting for the Pi benchmark.**

- The runner is `inference/hailo_inference._HailoRunner`, with `NpuScheduler.exclusive()` for close. Tests are in `tests/test_hailo_runner.py`.
- The replay tooling is `tools/profile_threads.py` and `edge_main --source-fps`.
- Written after the R1 gates and the overnight soak on `b5b8bc8`.

## 1. Problem

`_HailoRunner.run` (`inference/hailo_inference.py`) does the following for **every frame**, inside the NPU lock:

1. builds a new `InferVStreams` pipeline (one input and six output vstreams);
2. activates the network group;
3. runs one inference;
4. deactivates the network group and tears the pipeline down.

Evidence from the Pi (R1 HEF, HailoRT 4.23):

- The detector's thread count swings between 27 and 48 across soak samples. The process uses about 190 new thread IDs per second, about 19 per inference at about 10 inferences per second. Source: daytime soaks, the 0355Z, 0431Z and 0649Z runs.
- One inference cycle takes about 98 ms (gate D p50: 98.18 ms live, 97 ms on photos), and inference runs at 10.08 FPS. The loop is bound by latency, so any per-frame setup time comes straight out of FPS.
- **Not known:** how much of the 98 ms is setup rather than device compute, letterbox and decode. No speed-up is claimed until it is measured (section 5).

## 2. Goal and non-goals

Goals:

- Open the pipeline and activate the network group **once**, keep them open for the detector's lifetime, and release them cleanly on `close()`.
- Produce outputs identical to today's: same HEF, same input bytes, so the raw heads are bit-identical.
- Keep the NPU lock: every device call, open and close still goes through `NpuScheduler.run`.
- Recover from a HailoRT error without restarting the service.

Non-goals:

- Batching, async pipelining and the newer `InferModel` API (option B in section 3). Each is a bigger change, worth looking at only if option A isn't enough.
- Any change to the decoder, thresholds, gate or HEF.

## 3. Design

**Option A (proposed): a persistent pipeline inside `_HailoRunner`.**

- `_open()` enters `hpf.InferVStreams(...)` and then `network_group.activate(...)`. This keeps today's nesting order, with both held in a `contextlib.ExitStack`. It runs on the first `run()` call (warm-up triggers it), under the NPU lock.
- `run()` sends the input through the open pipeline (`self._pipeline.infer(...)`), under the NPU lock as today.
- **On any exception from the device call:**
  - close the stack (deactivate, then release the vstreams), log once, and re-raise;
  - the next `run()` reopens it;
  - the inference loop already logs, counts `inference_failed` and skips the frame;
  - if reopening fails repeatedly, the existing error path is unchanged: the frame is skipped and the status shows inference errors.
- `close()` closes the stack, then releases the VDevice. It is idempotent.
  - It takes the NPU lock (`NpuScheduler.exclusive`, which records no latency sample), so it can't race an inference.
  - If an inference holds the lock for more than 5 s, a native call is hung, and the pipeline and device are left to process exit.
- **Constraint:** this holds the network group active, so only one network group can use the device at a time. That's true today: the node runs one model (`npu_scheduler.py`).
  - A second model would need HailoRT's own model scheduler instead of manual activation.
  - The spec records the constraint; it does not build for that case.
- Callers don't change: `HailoDetector`, `tools/hef_parity.py` and `tools/benchmark_edge.py` all go through `_HailoRunner`.

**Option B (not proposed now):** HailoRT's `InferModel` API (`create_infer_model` / `configure` / `run_async`).

- It allows several frames in flight, which helps throughput, not single-frame latency.
- It means rewriting `_HailoRunner` and the test fakes.
- Revisit it only if option A's measured gain is small and more FPS is still needed.

## 4. Pipeline-stage capacity: check first

The overnight soak dropped **42,797 detection batches** (`detq_drops`), a steady 2.2 per second while a track was active. The pipeline thread was already slower than about 10 batches per second coming from inference.

Faster inference alone would mostly mean more drops, not more frames reaching the tracker. So the benchmark also measures the pipeline stage (section 5, step 4). If the pipeline is the limit, that gets fixed before, or together with, option A. Candidates, to be measured rather than assumed:

- the motion score on the full frame;
- the clip ring-buffer append;
- the best-snapshot copy;
- zoom-controller calls.

## 5. Benchmark plan

These steps run on the Pi. Steps 2 and 3 **stop `zovive-detect`**, so they need the owner's explicit go-ahead and a time window (about 30 minutes). The candidate runs from a staging checkout under `/home/zovive1/r1_gate`, never from `/opt/zovive`. The deployed tree, HEF, manifest and configs stay unchanged.

| Step | What | Measures |
|---|---|---|
| 1 | Laptop: unit tests with a fake HailoRT (section 6), full CI | correctness of open, reuse, close and recovery |
| 2 | Pi, service stopped, **current code**: `tools.benchmark_edge --sections models` and the gate D 5-minute live soak | baseline: per-stage latency (letterbox, device call from `NpuScheduler`, decode), FPS, CPU, threads, thread IDs used per second |
| 2b | Pi, service stopped: the HailoRT CLI's benchmark on the R1 HEF | the device-only floor (no Python, no setup per frame) |
| 3 | Pi, service stopped, **candidate**: same commands as step 2 | the change in each figure |
| 3b | Pi: `dump_detections` on the 102-image parity set with both runners | raw heads identical (max abs diff 0), and identical detections |
| 4 | Pi: replay one of the overnight alert clips through the pipeline, current vs candidate. Details below the table. | `detq_drops` per second, pipeline time per batch with a track active, and which function takes the time |
| 5 | Pi: service restarted on the deployed code; check it is active and holds the Hailo device | production restored |

**How step 4 runs** (changed from the first draft, 31 Oct 2026):

- It uses `edge_main.py --source file:<clip> --source-fps 25` from the staging checkout, under `tools.profile_threads`.
  - Plain `python -m cProfile` profiles only the main thread, and the pipeline runs in a worker thread.
  - `FileFrameSource` plays at 8 FPS by default, below the camera's 25.
- **Isolation:**
  - `ZOVIVE_HOME` points at a scratch folder, so nothing is written under `/var/lib/zovive`.
  - The checked-in `node_config.yaml` has `runtime.environment: "dev"`, so `edge_main` uses `FakeLensCamera` and never touches the real lens.
- **Stopping:** `timeout -s INT` stops the run, which `edge_main` handles as a normal shutdown, so every thread writes its profile.
  - `--max-cycles` can't be used here: inference and the pipeline count cycles separately, and dropped batches leave the pipeline waiting.

**Adopt option A only if:**

- outputs are identical (step 3b);
- live inference FPS improves by at least 10% or p50 latency drops by at least 10 ms;
- thread IDs used per second fall to near zero;
- a 30-minute candidate soak shows flat RSS and no inference errors.

Otherwise keep today's simpler code. Deploying the candidate is a separate decision with a separate approval.

## 6. Tests (laptop)

A fake `hpf` whose `InferVStreams` and `activate` are context managers that count enters and exits:

- N `run()` calls enter each context once;
- `close()` exits them in reverse order, then releases the VDevice; a second `close()` is a no-op;
- an exception from `infer` exits both contexts, and the next `run()` reopens and succeeds;
- `run()` after `close()` raises a clear error and doesn't reopen;
- every pipeline step (open, infer, teardown, close) runs with the NPU lock held. Only inferences count in the latency stats.
- `close()` gives up on a device held past its timeout, without releasing it.

Existing tests (`tests/test_hailo_decoder.py`, `tests/test_benchmark_edge.py`) must pass unchanged.

## 7. Decisions needed

- **P1:** option A as described, with option B deferred. Yes or no.
- **P2:** measure the pipeline-stage bottleneck (section 4, step 4) in the same Pi session.
- **P3:** the adoption thresholds in section 5.
- **P4:** a Pi window for steps 2 to 5 (service stopped for about 30 minutes).
