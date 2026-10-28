"""Durable upload retries: network/retry_policy.py, the outbox queue in
db/outbox.py (claim_due / record_send_failure / defer_due / requeue_dead)
and transfer_main.py's send loops, against a real WAL SQLite file. The clock
is injected and jitter is 0 or seeded, so nothing sleeps.

Rules under test: only an HTTP 4xx rejection (not 429) costs a retry
(30 s, 1 m, 2 m, 5 m, 15 m, then dead); an unreachable or busy base station
pauses the queue (~60 s probes) at no cost; retry state survives a restart;
concurrent senders never claim the same row.
"""

from __future__ import annotations

import random
import tempfile
import threading
import unittest
from pathlib import Path

import paths
import transfer_main
from db.init_db import get_connection, init_db
from db.migrations import migrate
from db.outbox import (
    EventRecord,
    claim_due,
    get_event,
    insert_event,
    mark_sent,
    media_backlog,
    record_send_failure,
    requeue_dead,
    set_video_path,
)
from network.base_station_client import BaseStationError, BaseStationRejected, BaseStationUnreachable
from network.retry_policy import RetryPolicy
from utils.config_loader import load

EXACT = RetryPolicy(jitter=0.0)  # schedule 30/60/120/300/900, max 5 retries, 60 s pauses


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)  # nosec B311 - retry jitter in tests, not security


def _rejected(code: int = 422) -> BaseStationRejected:
    return BaseStationRejected(f"HTTP {code}", status_code=code)


class _ScriptedClient:
    """Plays back one outcome per send (None = success, else the exception to
    raise); the last outcome repeats. Records (kind, event_id) per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def _next(self):
        return self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]

    def send_event(self, event, snapshot_bytes=None):
        self.calls.append(("image", event["event_id"]))
        outcome = self._next()
        if outcome is not None:
            raise outcome

    def send_video(self, event_id, video_bytes, timeout_sec=None):
        self.calls.append(("video", event_id))
        outcome = self._next()
        if outcome is not None:
            raise outcome


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class RetryPolicyTests(unittest.TestCase):
    def test_each_rejection_waits_its_step_within_jitter(self):
        policy, rng = RetryPolicy(), _rng(7)
        for rejections, step in enumerate((30, 60, 120, 300, 900), start=1):
            for _ in range(50):
                wait = policy.retry_at(0.0, rejections, rng)
                self.assertGreaterEqual(wait, step * 0.8)
                self.assertLessEqual(wait, step * 1.2)

    def test_sixth_rejection_dead_letters(self):
        self.assertIsNotNone(EXACT.retry_at(0.0, 5, _rng()))
        self.assertIsNone(EXACT.retry_at(0.0, 6, _rng()))

    def test_pause_and_retry_after(self):
        policy, rng = RetryPolicy(), _rng(1)
        self.assertTrue(48 <= policy.pause_until(0.0, rng) <= 72)
        self.assertTrue(300 <= policy.pause_until(0.0, rng, retry_after=300) <= 360, "never shortened")
        self.assertTrue(900 <= policy.pause_until(0.0, rng, retry_after=10_000) <= 1080, "capped")

    def test_invalid_settings_are_rejected(self):
        for bad in (
            {"schedule_sec": ()},
            {"schedule_sec": (60.0, 30.0)},
            {"schedule_sec": (0.0, 30.0)},
            {"max_retries": -1},
            {"jitter": 1.0},
            {"pause_sec": 0.0},
            {"pause_sec": 1000.0, "max_pause_sec": 900.0},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RetryPolicy(**bad)

    def test_shipped_config_is_the_agreed_policy(self):
        policy = RetryPolicy.from_config(load(paths.TRANSFER_CONFIG)["transfer"])
        self.assertEqual(policy, RetryPolicy(schedule_sec=(30.0, 60.0, 120.0, 300.0, 900.0), max_retries=5))


class _QueueTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmp.name) / "test.sqlite3"
        migrate(init_db())

    def tearDown(self):
        get_connection().close()
        paths.DB_PATH = self._orig_db_path
        self.tmp.cleanup()

    def add(self, name: str, created_ts: float = 900.0, *, snapshot: bool = True, video: bool = False) -> str:
        snap = Path(self.tmp.name) / f"{name}.jpg"
        if snapshot:
            snap.write_bytes(b"jpeg")
        event_id = insert_event(
            EventRecord(
                track_id=name,
                camera_id="cam01",
                animal_name="tiger",
                confidence=0.9,
                snapshot_path=str(snap),
                created_ts=created_ts,
            )
        )
        if video:
            clip = Path(self.tmp.name) / f"{name}.mp4"
            clip.write_bytes(b"mp4")
            set_video_path(event_id, str(clip))
        return event_id

    def claim(self, queue: str = "image", now: float = 1000.0, limit: int = 10, lease: float = 50.0):
        return claim_due(
            queue, now=now, limit=limit, lease_sec=lease, max_retries=5, max_wait_sec=EXACT.max_wait_sec
        )

    def ids(self, rows) -> list[str]:
        return [r["event_id"] for r in rows]


class ClaimTests(_QueueTestCase):
    def test_oldest_pending_first_and_a_claim_hides_rows_until_its_lease_ends(self):
        a, b, c = self.add("a", 1.0), self.add("b", 2.0), self.add("c", 3.0)
        _, first = self.claim(limit=2)
        self.assertEqual(self.ids(first), [a, b])
        _, second = self.claim(limit=10)
        self.assertEqual(self.ids(second), [c], "claimed rows are hidden from other senders")
        _, after_lease = self.claim(now=1000.0 + 50.0, limit=10)
        self.assertEqual(sorted(self.ids(after_lease)), sorted([a, b, c]), "a dead sender's rows come back")

    def test_failed_row_waits_for_its_retry_time(self):
        a = self.add("a")
        claim, _ = self.claim()
        record_send_failure("image", a, claim, now=1000.0, error="HTTP 422", retry_at=1030.0, charged=True)
        self.assertEqual(self.claim(now=1029.9)[1], [])
        self.assertEqual(self.ids(self.claim(now=1030.0)[1]), [a])

    def test_dead_and_over_budget_rows_are_never_claimed(self):
        dead, over = self.add("dead"), self.add("over")
        conn = get_connection()
        with conn:
            conn.execute("UPDATE events SET status = 'dead' WHERE event_id = ?", (dead,))
            conn.execute("UPDATE events SET status = 'failed', attempt_count = 6 WHERE event_id = ?", (over,))
        self.assertEqual(self.claim()[1], [])
        self.assertEqual(get_event(over)["status"], "dead", "over-budget failed rows become dead")

    def test_a_retry_time_far_ahead_is_due_after_a_clock_jump_back(self):
        a = self.add("a")
        conn = get_connection()
        with conn:
            conn.execute(
                "UPDATE events SET status = 'failed', next_retry_ts = ? WHERE event_id = ?",
                (1000.0 + EXACT.max_wait_sec + 1, a),
            )
        self.assertEqual(self.ids(self.claim()[1]), [a])

    def test_video_queue_holds_only_rows_with_a_clip(self):
        self.add("no_clip")
        with_clip = self.add("clip", video=True)
        self.assertEqual(self.ids(self.claim("video")[1]), [with_clip])

    def test_a_stale_claim_cannot_record_a_result(self):
        a = self.add("a")
        old_claim, _ = self.claim(lease=50.0)
        new_claim, rows = self.claim(now=1051.0)  # old lease ran out; another sender took it
        self.assertEqual(self.ids(rows), [a])
        self.assertFalse(mark_sent(a, claim=old_claim))
        self.assertFalse(
            record_send_failure("image", a, old_claim, now=1052.0, error="x", retry_at=2000.0, charged=True)
        )
        self.assertTrue(mark_sent(a, claim=new_claim))
        self.assertEqual(get_event(a)["status"], "sent")


class ConcurrentClaimTests(_QueueTestCase):
    def test_four_senders_claim_every_row_exactly_once(self):
        expected = {self.add(f"e{i}", float(i), snapshot=False) for i in range(200)}
        claimed: list[str] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def sender():
            try:
                while True:
                    _, rows = claim_due(
                        "image", now=1000.0, limit=7, lease_sec=3600.0, max_retries=5, max_wait_sec=9e9
                    )
                    if not rows:
                        return
                    with lock:
                        claimed.extend(self.ids(rows))
            except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
                errors.append(exc)
            finally:
                get_connection().close()

        threads = [threading.Thread(target=sender) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(claimed), 200, "no row claimed twice")
        self.assertEqual(set(claimed), expected)


class SendLoopTests(_QueueTestCase):
    def send_images(self, client, clock, **kwargs) -> bool:
        return transfer_main._send_pending_images(
            client, EXACT, 10, clock=clock, rng=_rng(0), lease_sec=50.0, **kwargs
        )

    def send_videos(self, client, clock) -> bool:
        return transfer_main._send_pending_videos(client, EXACT, 10, 30.0, clock=clock, rng=_rng(0))

    def test_rejections_follow_the_schedule_then_dead_letter(self):
        a = self.add("a")
        client, clock = _ScriptedClient([_rejected()]), _Clock(1000.0)
        self.send_images(client, clock)  # first send
        for due in (1030.0, 1090.0, 1210.0, 1510.0, 2410.0):  # the 5 retries
            clock.now = due - 0.5
            self.send_images(client, clock)
            calls_before = len(client.calls)
            clock.now = due
            self.send_images(client, clock)
            self.assertEqual(len(client.calls), calls_before + 1, f"exactly one try at {due}")
        row = get_event(a)
        self.assertEqual((row["status"], row["attempt_count"]), ("dead", 6))
        clock.now = 100_000.0
        self.send_images(client, clock)
        self.assertEqual(len(client.calls), 6, "dead rows are never retried automatically")
        self.assertTrue(Path(row["snapshot_path"]).exists(), "dead-lettered evidence is kept")
        self.assertEqual(media_backlog(0.0)["dead_snapshots"], 1)

    def test_success_after_two_rejections_then_ack_and_delete(self):
        a = self.add("a")
        client, clock = _ScriptedClient([_rejected(), _rejected(), None]), _Clock(1000.0)
        for now in (1000.0, 1030.0, 1090.0):
            clock.now = now
            self.send_images(client, clock, delete_after_ack=True, record_ack=True)
        row = get_event(a)
        self.assertEqual((row["status"], row["attempt_count"]), ("sent", 2))
        self.assertIsNotNone(row["snapshot_acked_ts"])
        self.assertFalse(Path(row["snapshot_path"]).exists(), "deleted only after the ACK")
        self.assertIsNone(row["next_retry_ts"])
        self.assertIsNone(row["send_claim"])

    def test_two_hour_outage_costs_nothing_and_probes_about_once_a_minute(self):
        events = [self.add(name, ts) for name, ts in (("a", 1.0), ("b", 2.0), ("c", 3.0))]
        clip_event = self.add("v", 4.0, video=True)
        down = BaseStationUnreachable("connection refused")
        client, clock = _ScriptedClient([down]), _Clock(0.0)
        while clock.now < 7200.0:  # poll every 5 s for two hours
            self.send_images(client, clock)
            self.send_videos(client, clock)
            clock.now += 5.0
        self.assertLessEqual(len(client.calls), 7200 / 60 + 2, "about one probe per minute, not per row")
        for event_id in events + [clip_event]:
            row = get_event(event_id)
            self.assertEqual((row["attempt_count"], row["video_attempt_count"]), (0, 0), "outage is free")
            self.assertNotEqual(row["status"], "dead")

        client.outcomes = [None]  # base station back
        for _ in range(30):
            self.send_images(client, clock)
            self.send_videos(client, clock)
            clock.now += 5.0
        self.assertEqual({get_event(e)["status"] for e in events}, {"sent"})
        self.assertEqual(get_event(clip_event)["video_status"], "sent")

    def test_unreachable_also_pauses_the_video_queue(self):
        a = self.add("a", 1.0)
        self.add("v", 2.0, video=True)
        client, clock = _ScriptedClient([BaseStationUnreachable("dns")]), _Clock(1000.0)
        self.assertTrue(self.send_images(client, clock))
        self.send_videos(client, clock)
        self.assertEqual(client.calls, [("image", a)], "the video queue waits for the probe too")

    def test_busy_base_station_rotates_past_an_item_it_always_fails(self):
        bad, good1, good2 = self.add("bad", 1.0), self.add("g1", 2.0), self.add("g2", 3.0)
        clip_event = self.add("v", 4.0, video=True)
        busy = BaseStationError("HTTP 500", status_code=500)

        class _BadItemClient(_ScriptedClient):
            def send_event(self, event, snapshot_bytes=None):
                self.calls.append(("image", event["event_id"]))
                if event["event_id"] == bad:
                    raise busy

        client, clock = _BadItemClient([None]), _Clock(1000.0)
        self.assertTrue(self.send_images(client, clock), "a 500 pauses the image queue")
        self.assertFalse(self.send_videos(client, clock), "but not the video queue")
        self.assertEqual(get_event(clip_event)["video_status"], "sent")
        clock.now = 1060.0
        self.send_images(client, clock)
        self.assertEqual({get_event(e)["status"] for e in (good1, good2)}, {"sent"}, "others are not blocked")
        row = get_event(bad)
        self.assertEqual((row["status"], row["attempt_count"]), ("failed", 0), "a 500 costs nothing")

    def test_retry_after_is_honoured(self):
        a = self.add("a")
        limited = BaseStationError("HTTP 429", status_code=429, retry_after=300.0)
        client, clock = _ScriptedClient([limited, None]), _Clock(1000.0)
        self.send_images(client, clock)
        clock.now = 1300.0  # the failing item is 1 s behind the pause
        self.send_images(client, clock)
        self.assertEqual(len(client.calls), 1)
        clock.now = 1301.0
        self.send_images(client, clock)
        self.assertEqual(get_event(a)["status"], "sent")

    def test_retry_state_survives_a_restart(self):
        a = self.add("a")
        client, clock = _ScriptedClient([_rejected(), None]), _Clock(1000.0)
        self.send_images(client, clock)

        def restarted():
            try:
                clock.now = 1029.0
                self.send_images(client, clock)
                clock.now = 1030.0
                self.send_images(client, clock)
            finally:
                get_connection().close()

        thread = threading.Thread(target=restarted)  # fresh connection, as after a restart
        thread.start()
        thread.join()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(get_event(a)["status"], "sent")

    def test_missing_clip_is_dead_lettered_without_a_send(self):
        clip_event = self.add("v", video=True)
        Path(get_event(clip_event)["video_path"]).unlink()
        client = _ScriptedClient([None])
        self.send_videos(client, _Clock(1000.0))
        self.assertEqual(client.calls, [])
        self.assertEqual(get_event(clip_event)["video_status"], "dead")

    def test_requeue_dead_gives_a_fresh_budget(self):
        a = self.add("a")
        client, clock = _ScriptedClient([_rejected()]), _Clock(1000.0)
        for due in (1000.0, 1030.0, 1090.0, 1210.0, 1510.0, 2410.0):
            clock.now = due
            self.send_images(client, clock)
        self.assertEqual(get_event(a)["status"], "dead")
        self.assertEqual(requeue_dead("image"), 1)
        row = get_event(a)
        self.assertEqual((row["status"], row["attempt_count"], row["next_retry_ts"]), ("failed", 0, None))
        client.outcomes = [None]
        self.send_images(client, clock)
        self.assertEqual(get_event(a)["status"], "sent")


if __name__ == "__main__":
    unittest.main()
