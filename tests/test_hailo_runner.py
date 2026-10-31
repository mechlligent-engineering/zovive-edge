"""inference/hailo_inference._HailoRunner keeps one InferVStreams pipeline open
and the network group active for its lifetime (docs/PERSISTENT_VSTREAMS_SPEC.md).
HailoRT is faked; real numbers come only from the Pi."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from inference.base import InferenceBackendError
from inference.hailo_inference import _HailoRunner
from inference.npu_scheduler import NpuScheduler

OUT = np.arange(4, dtype=np.float32).reshape(2, 2, 1)


class _Context:
    """A HailoRT context manager that logs its enter/exit and whether the
    NPU lock was held at the time."""

    def __init__(self, name: str, fake: "_FakeHailo", value=None, fail_enter: bool = False):
        self.name, self.fake, self.value, self.fail_enter = name, fake, value, fail_enter

    def __enter__(self):
        self.fake.record("enter", self.name)
        if self.fail_enter:
            raise RuntimeError(f"{self.name} failed")
        return self.value if self.value is not None else self

    def __exit__(self, *exc):
        self.fake.record("exit", self.name)
        return False


class _FakeHailo:
    def __init__(self, scheduler: NpuScheduler):
        self.scheduler = scheduler
        self.log: list[tuple[str, str]] = []
        self.lock_held: list[bool] = []
        self.infer_error: Exception | None = None
        self.activate_fails = False

        self.hpf = MagicMock()
        hef = self.hpf.HEF.return_value
        hef.get_input_vstream_infos.return_value = [SimpleNamespace(name="det/input", shape=(640, 640, 3))]
        hef.get_output_vstream_infos.return_value = [SimpleNamespace(name="det/out", shape=OUT.shape)]
        self.device = self.hpf.VDevice.return_value
        self.network_group = MagicMock()
        self.device.configure.return_value = [self.network_group]

        self.pipeline = MagicMock()
        self.pipeline.infer.side_effect = self._infer
        self.hpf.InferVStreams.side_effect = lambda *a, **k: _Context("vstreams", self, self.pipeline)
        self.network_group.activate.side_effect = lambda params: _Context(
            "activate", self, fail_enter=self.activate_fails
        )

    def record(self, what: str, name: str) -> None:
        self.log.append((what, name))
        self.lock_held.append(self.scheduler._lock.locked())

    def _infer(self, data):
        self.lock_held.append(self.scheduler._lock.locked())
        if self.infer_error is not None:
            raise self.infer_error
        return {"det/out": OUT[None]}


class HailoRunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hef_path = Path(tmp.name) / "detector.hef"
        hef_path.write_bytes(b"")
        self.scheduler = NpuScheduler()
        self.fake = _FakeHailo(self.scheduler)
        with patch("inference.hailo_inference._import_hailo", return_value=self.fake.hpf):
            self.runner = _HailoRunner(str(hef_path), self.scheduler)
        self.frame = np.zeros((640, 640, 3), np.uint8)

    def test_one_pipeline_and_activation_serve_every_inference(self):
        for _ in range(5):
            out = self.runner.run(self.frame)
        np.testing.assert_array_equal(out["det/out"], OUT)  # batch dimension removed
        self.assertEqual(self.fake.hpf.InferVStreams.call_count, 1)
        self.assertEqual(self.fake.network_group.activate.call_count, 1)
        self.assertEqual(self.fake.log, [("enter", "vstreams"), ("enter", "activate")])
        self.assertEqual(self.fake.pipeline.infer.call_count, 5)

    def test_close_deactivates_releases_vstreams_then_the_device_once(self):
        self.runner.run(self.frame)
        self.runner.close()
        self.runner.close()
        self.assertEqual(self.fake.log[-2:], [("exit", "activate"), ("exit", "vstreams")])
        self.fake.device.release.assert_called_once()

    def test_a_failed_inference_tears_down_and_the_next_run_reopens(self):
        self.runner.run(self.frame)
        self.fake.infer_error = RuntimeError("HAILO_TIMEOUT")
        with self.assertRaisesRegex(RuntimeError, "HAILO_TIMEOUT"):
            self.runner.run(self.frame)
        self.assertEqual(self.fake.log[-2:], [("exit", "activate"), ("exit", "vstreams")])
        self.fake.infer_error = None
        np.testing.assert_array_equal(self.runner.run(self.frame)["det/out"], OUT)
        self.assertEqual(self.fake.hpf.InferVStreams.call_count, 2)

    def test_a_failed_activation_releases_the_vstreams_and_is_retried(self):
        self.fake.activate_fails = True
        with self.assertRaisesRegex(RuntimeError, "activate failed"):
            self.runner.run(self.frame)
        self.assertEqual(self.fake.log[-1], ("exit", "vstreams"))
        self.fake.activate_fails = False
        self.runner.run(self.frame)
        self.assertEqual(self.fake.hpf.InferVStreams.call_count, 2)

    def test_run_after_close_raises_without_reopening(self):
        self.runner.close()
        with self.assertRaises(InferenceBackendError):
            self.runner.run(self.frame)
        self.fake.hpf.InferVStreams.assert_not_called()

    def test_every_pipeline_step_holds_the_npu_lock(self):
        self.runner.run(self.frame)
        self.fake.infer_error = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.runner.run(self.frame)
        self.fake.infer_error = None
        self.runner.run(self.frame)
        self.runner.close()
        self.assertTrue(self.fake.lock_held)
        self.assertTrue(all(self.fake.lock_held), self.fake.lock_held)

    def test_close_leaves_a_busy_device_to_process_exit(self):
        self.runner.run(self.frame)
        self.runner.CLOSE_LOCK_TIMEOUT_SEC = 0.05
        self.scheduler._lock.acquire()  # an inference that never returns
        try:
            with self.assertLogs("inference.hailo_inference", level="WARNING"):
                self.runner.close()
        finally:
            self.scheduler._lock.release()
        self.fake.device.release.assert_not_called()
        self.assertNotIn(("exit", "vstreams"), self.fake.log)

    def test_only_inferences_count_in_the_latency_stats(self):
        self.runner.run(self.frame)
        self.runner.run(self.frame)
        self.runner.close()
        self.assertEqual(self.scheduler.snapshot()["calls"], 2)


if __name__ == "__main__":
    unittest.main()
