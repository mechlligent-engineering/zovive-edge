"""tools/profile_threads.py (per-thread cProfile) and edge_main's --source-fps,
the two pieces of the pipeline-stage replay in docs/PERSISTENT_VSTREAMS_SPEC.md."""

import contextlib
import io
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

import edge_main
from tools import profile_threads

SCRIPT = textwrap.dedent(
    """
    import sys, threading

    def busy_worker_function():
        return sum(i * i for i in range(20000))

    t = threading.Thread(target=busy_worker_function, name="supervised-pipeline")
    t.start()
    t.join()
    sys.exit(int(sys.argv[1]))
    """
)


class TestProfileThreads(unittest.TestCase):
    def _run(self, *script_args: str) -> tuple[int, Path]:
        d = Path(tempfile.mkdtemp())
        script = d / "prog.py"
        script.write_text(SCRIPT)
        out = d / "prof"
        with contextlib.redirect_stderr(io.StringIO()):
            code = profile_threads.main(["--out", str(out), "--", str(script), *script_args])
        return code, out

    def test_worker_thread_gets_its_own_profile(self):
        code, out = self._run("0")
        self.assertEqual(code, 0)
        names = sorted(p.name.split(".")[0] for p in out.glob("*.prof"))
        self.assertEqual(names, ["MainThread", "supervised-pipeline"])
        summary = (out / "summary.txt").read_text()
        self.assertIn("busy_worker_function", summary)

    def test_exit_code_passes_through_and_run_is_restored(self):
        original = threading.Thread.run
        code, _ = self._run("3")
        self.assertEqual(code, 3)
        self.assertIs(threading.Thread.run, original)

    def test_refuses_an_existing_out_dir(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileExistsError):
            profile_threads.main(["--out", d, "--", "prog.py"])


class TestSourceFps(unittest.TestCase):
    def _main(self, *argv: str):
        with mock.patch.object(edge_main, "run", return_value=edge_main.EXIT_OK) as run:
            with mock.patch("sys.argv", ["edge_main.py", *argv]):
                edge_main.main()
        return run

    def test_passed_through_with_a_file_source(self):
        run = self._main("--source", "file:clip.mp4", "--source-fps", "25")
        run.assert_called_once_with(source_override="file:clip.mp4", max_cycles=None, source_fps=25.0)

    def test_default_leaves_file_source_rate_alone(self):
        run = self._main("--source", "file:clip.mp4")
        self.assertIsNone(run.call_args.kwargs["source_fps"])

    def test_rejected_without_a_source(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._main("--source-fps", "25")


if __name__ == "__main__":
    unittest.main()
