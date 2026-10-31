"""Run a Python program with cProfile on every thread, not just the main one.

`python -m cProfile` only profiles the thread that runs the script, and
edge_main does its work in supervised worker threads (capture, inference,
pipeline, dispatch). This wraps threading.Thread.run so each thread gets its
own profiler, writes one .prof file per thread when the thread finishes, and
a summary.txt with each thread's top functions by cumulative time.

Used for the pipeline-stage measurement in docs/PERSISTENT_VSTREAMS_SPEC.md:

    timeout -s INT -k 60 330 .venv/bin/python -m tools.profile_threads \\
        --out /tmp/prof -- edge_main.py --source file:clip.mp4 --source-fps 25

Stop the program the way it normally stops (edge_main: SIGINT or SIGTERM).
A daemon thread still running when the interpreter exits writes nothing.
Read the .prof files with `python -m pstats`. Standard library only.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import re
import runpy
import sys
import threading
from collections.abc import Callable
from pathlib import Path

DEFAULT_TOP = 25


def _prof_path(out_dir: Path, thread_name: str, ident: int | None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", thread_name) or "thread"
    return out_dir / f"{safe}.{ident}.prof"


def install(out_dir: Path) -> Callable[[], None]:
    """Profile every thread started from now on; returns the undo function."""
    original_run = threading.Thread.run

    def profiled_run(self: threading.Thread) -> None:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            original_run(self)
        finally:
            profiler.disable()
            profiler.dump_stats(_prof_path(out_dir, self.name, self.native_id))

    threading.Thread.run = profiled_run  # type: ignore[method-assign]

    def uninstall() -> None:
        threading.Thread.run = original_run  # type: ignore[method-assign]

    return uninstall


def summarize(out_dir: Path, top: int = DEFAULT_TOP) -> str:
    parts = []
    for path in sorted(out_dir.glob("*.prof")):
        buf = io.StringIO()
        stats = pstats.Stats(str(path), stream=buf)
        buf.write(f"=== {path.name}: {stats.total_tt:.1f} s profiled\n")
        stats.sort_stats("cumulative").print_stats(top)
        parts.append(buf.getvalue())
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cProfile every thread of a Python program")
    parser.add_argument("--out", type=Path, required=True, help="new directory for the .prof files")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="functions per thread in summary.txt")
    parser.add_argument("script", help="the program to run, then its own arguments (put -- before it)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=False)  # never mixes two runs' profiles
    uninstall = install(args.out)
    main_profiler = cProfile.Profile()
    saved_argv = sys.argv
    sys.argv = [args.script, *args.script_args]
    exit_code = 0
    try:
        main_profiler.enable()
        try:
            runpy.run_path(args.script, run_name="__main__")
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        finally:
            main_profiler.disable()
    finally:
        sys.argv = saved_argv
        uninstall()
        main_profiler.dump_stats(_prof_path(args.out, "MainThread", threading.main_thread().native_id))
        summary = summarize(args.out, args.top)
        (args.out / "summary.txt").write_text(summary)
    print(f"profiles and summary.txt in {args.out}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
