import tempfile
import unittest
from pathlib import Path

import paths
from inference.bytetrack_wrapper import Track
from inference.class_thresholds import ClassThresholds
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from utils import config_loader
from utils.config_loader import ConfigError, IssueKind, load


def _track(track_id="t1", presence=(True, True, True), **fields):
    return Track(
        track_id=track_id,
        box=fields.pop("box", (0, 0, 10, 10)),
        class_id=5,
        class_name="tiger",
        score=0.9,
        presence_history=list(presence),
        **fields,
    )


THRESHOLDS = ClassThresholds(default=0.50, per_class={"elephant": 0.60})


class TestStage1Gate(unittest.TestCase):
    def test_passes_with_enough_hits_in_window(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))
        track = _track(presence=(True, False, True, False, True))  # 3 of 5
        self.assertTrue(gate.evaluate(track, now=0.0))

    def test_fails_with_too_few_hits(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))
        track = _track(presence=(True, False, False, False, True))  # 2 of 5
        self.assertFalse(gate.evaluate(track, now=0.0))

    def test_only_considers_last_window_size_entries(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=3))
        # 10 hits far in the past, but the recent window only has 1 hit
        track = _track(presence=[True] * 10 + [False, False, True])
        self.assertFalse(gate.evaluate(track, now=0.0))

    def test_cooldown_suppresses_repeat_alert(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=1, window_size=1, track_alert_cooldown_sec=100.0))
        track = _track(presence=(True,))
        self.assertTrue(gate.evaluate(track, now=1000.0))
        gate.mark_alerted(track.track_id, now=1000.0)
        # Still within cooldown window
        self.assertTrue(gate.in_cooldown(track, now=1050.0))
        self.assertFalse(gate.evaluate(track, now=1050.0))
        # Cooldown expired
        self.assertFalse(gate.in_cooldown(track, now=1200.0))
        self.assertTrue(gate.evaluate(track, now=1200.0))

    def test_evaluate_all_filters_list(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=2, window_size=2))
        passing = _track("t-pass", presence=(True, True))
        failing = _track("t-fail", presence=(False, True))
        result = gate.evaluate_all([passing, failing], now=0.0)
        self.assertEqual([t.track_id for t in result], ["t-pass"])


class TestStricterGate(unittest.TestCase):
    """The checks added by docs/PIPELINE_HARDENING_SPEC.md change 2."""

    def test_defaults_are_the_original_presence_and_cooldown_gate(self):
        # Seen once, a split second ago, flip-flopping species, tiny box:
        # still promoted, exactly as before the new checks existed.
        gate = Stage1Gate(Stage1Config(min_hits_in_window=1, window_size=1))
        track = _track(
            presence=(True,),
            box=(0, 0, 1, 1),
            first_seen_ts=10.0,
            last_seen_ts=10.0,
            class_votes=[("tiger", 0.46), ("leopard", 0.46)],
        )
        self.assertTrue(gate.evaluate(track, now=10.0))

    def test_five_of_seven_tolerates_two_missed_frames(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=5, window_size=7))
        self.assertTrue(
            gate.presence_vote_passed(_track(presence=(True, True, False, True, False, True, True)))
        )
        self.assertFalse(
            gate.presence_vote_passed(_track(presence=(True, False, True, False, True, False, True)))
        )

    def test_window_equal_to_hits_means_strictly_consecutive(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=5, window_size=5))
        self.assertTrue(gate.presence_vote_passed(_track(presence=(True,) * 5)))
        self.assertFalse(gate.presence_vote_passed(_track(presence=(True, True, False, True, True))))

    def test_track_must_last_the_minimum_duration(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=1, window_size=1, min_track_duration_sec=1.0))
        self.assertFalse(
            gate.evaluate(_track(presence=(True,), first_seen_ts=100.0, last_seen_ts=100.9), now=100.9)
        )
        self.assertTrue(
            gate.evaluate(_track(presence=(True,), first_seen_ts=100.0, last_seen_ts=101.0), now=101.0)
        )

    def test_box_must_cover_the_minimum_area_when_set(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=1, window_size=1, min_box_area_px=400))
        self.assertFalse(gate.box_area_passed(_track(box=(0, 0, 19, 20))))  # 380 px
        self.assertTrue(gate.box_area_passed(_track(box=(0, 0, 20, 20))))  # 400 px
        off = Stage1Gate(Stage1Config(min_box_area_px=0))
        self.assertTrue(off.box_area_passed(_track(box=(5, 5, 5, 5))))

    def test_mean_confidence_must_reach_the_leading_species_threshold(self):
        gate = Stage1Gate(Stage1Config(class_thresholds=THRESHOLDS))
        self.assertTrue(gate.mean_confidence_passed(_track(class_votes=[("elephant", 0.62)] * 5)))
        self.assertFalse(gate.mean_confidence_passed(_track(class_votes=[("elephant", 0.58)] * 5)))
        # Unlisted species use the default threshold.
        self.assertTrue(gate.mean_confidence_passed(_track(class_votes=[("deer", 0.52)] * 5)))
        self.assertFalse(gate.mean_confidence_passed(_track(class_votes=[])))

    def test_a_track_that_keeps_changing_species_does_not_confirm(self):
        gate = Stage1Gate(Stage1Config(class_thresholds=THRESHOLDS))
        # Every frame is confident, but elephant leads with only half the
        # frames: 1.4 / 4 = 0.35, below elephant's 0.60.
        votes = [("elephant", 0.7), ("wild_boar", 0.7), ("elephant", 0.7), ("wild_boar", 0.69)]
        self.assertFalse(gate.mean_confidence_passed(_track(class_votes=votes)))

    def test_every_check_has_to_pass(self):
        cfg = Stage1Config(
            min_hits_in_window=5,
            window_size=7,
            min_track_duration_sec=1.0,
            min_box_area_px=100,
            class_thresholds=THRESHOLDS,
            track_alert_cooldown_sec=300.0,
        )
        good = {
            "presence": (True,) * 7,
            "box": (0, 0, 20, 20),
            "first_seen_ts": 0.0,
            "last_seen_ts": 2.0,
            "class_votes": [("elephant", 0.8)] * 7,
        }
        self.assertTrue(Stage1Gate(cfg).evaluate(_track(**good), now=2.0))
        one_bad = {
            "presence": {"presence": (True, False) * 3 + (True,)},
            "duration": {"last_seen_ts": 0.5},
            "box": {"box": (0, 0, 5, 5)},
            "confidence": {"class_votes": [("elephant", 0.55)] * 7},
        }
        for name, change in one_bad.items():
            with self.subTest(failing=name):
                self.assertFalse(Stage1Gate(cfg).evaluate(_track(**{**good, **change}), now=2.0))
        gate = Stage1Gate(cfg)
        gate.mark_alerted("t1", now=2.0)
        self.assertFalse(gate.evaluate(_track(**good), now=3.0))  # cooldown

    def test_from_config_reads_every_setting(self):
        cfg = Stage1Config.from_config(
            {
                "detector": {
                    "confidence_threshold": 0.5,
                    "class_confidence_thresholds": {"elephant": 0.6},
                    "min_hits_in_window": 5,
                    "window_size": 7,
                    "min_track_duration_sec": 1.0,
                    "min_box_area_px": 64,
                    "track_alert_cooldown_sec": 300,
                }
            }
        )
        self.assertEqual(
            (cfg.min_hits_in_window, cfg.window_size, cfg.min_track_duration_sec, cfg.min_box_area_px),
            (5, 7, 1.0, 64.0),
        )
        self.assertEqual(cfg.track_alert_cooldown_sec, 300.0)
        self.assertEqual(cfg.class_thresholds.for_class("elephant"), 0.6)
        self.assertEqual(cfg.class_thresholds.for_class("deer"), 0.5)

    def test_from_config_without_new_keys_keeps_old_behaviour(self):
        cfg = Stage1Config.from_config({"detector": {}})
        self.assertEqual((cfg.min_hits_in_window, cfg.window_size), (3, 5))
        self.assertEqual((cfg.min_track_duration_sec, cfg.min_box_area_px), (0.0, 0.0))


_DETECTOR = """\
detector:
  classes: [deer, gaur, leopard, elephant, sloth_bear, tiger, wild_boar]
  confidence_threshold: 0.50
"""


class TestGateConfigChecks(unittest.TestCase):
    def setUp(self):
        config_loader.clear_cache()
        self.addCleanup(config_loader.clear_cache)
        self.dir = Path(tempfile.mkdtemp(prefix="zovive_gatecfg_"))

    def _issues(self, extra: str) -> list:
        path = self.dir / "inference_config.yaml"
        path.write_text(_DETECTOR + extra, encoding="utf-8")
        try:
            load(path)
        except ConfigError as exc:
            return list(exc.issues)
        finally:
            config_loader.clear_cache()
        return []

    def test_valid_gate_settings_load(self):
        extra = (
            "  min_hits_in_window: 5\n  window_size: 7\n  min_track_duration_sec: 1.0\n"
            "  min_box_area_px: 0\n  track_alert_cooldown_sec: 300\n"
        )
        self.assertEqual(self._issues(extra), [])

    def test_more_hits_than_the_window_is_rejected(self):
        issues = self._issues("  min_hits_in_window: 8\n  window_size: 7\n")
        self.assertEqual([i.key for i in issues], ["detector.min_hits_in_window"])
        self.assertIn("no track could ever be confirmed", issues[0].message)

    def test_hits_above_the_default_window_is_rejected(self):
        self.assertEqual(
            [i.key for i in self._issues("  min_hits_in_window: 6\n")], ["detector.min_hits_in_window"]
        )

    def test_non_integer_or_zero_counts_are_rejected(self):
        issues = self._issues("  min_hits_in_window: 5.5\n  window_size: 0\n")
        self.assertEqual(
            sorted((i.key, i.kind) for i in issues),
            [
                ("detector.min_hits_in_window", IssueKind.INVALID_TYPE),
                ("detector.window_size", IssueKind.INVALID_VALUE),
            ],
        )

    def test_negative_or_non_numeric_durations_are_rejected(self):
        issues = self._issues(
            "  min_track_duration_sec: -1\n  min_box_area_px: big\n  track_alert_cooldown_sec: true\n"
        )
        self.assertEqual(
            sorted(i.key for i in issues),
            [
                "detector.min_box_area_px",
                "detector.min_track_duration_sec",
                "detector.track_alert_cooldown_sec",
            ],
        )


class TestDeployedGateConfig(unittest.TestCase):
    def setUp(self):
        config_loader.clear_cache()
        self.addCleanup(config_loader.clear_cache)

    def test_repository_config_sets_the_approved_gate(self):
        cfg = Stage1Config.from_config(load(paths.INFERENCE_CONFIG))
        self.assertEqual((cfg.min_hits_in_window, cfg.window_size), (5, 7))
        self.assertEqual(cfg.min_track_duration_sec, 1.0)
        self.assertEqual(cfg.min_box_area_px, 0.0)
        self.assertEqual(cfg.track_alert_cooldown_sec, 300.0)
        self.assertEqual(cfg.class_thresholds.for_class("elephant"), 0.60)


if __name__ == "__main__":
    unittest.main()
