import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import json
import subprocess
from tools.monitor_training import inspect_health
from tools.monitor_training import main


class MonitorTests(unittest.TestCase):
    def health(self, **kwargs):
        return {"timestamp": 1000, "actors_alive": 8, "actors_expected": 8,
                "transitions": 100, "updates": 20, "alpha": .3,
                "q_loss": 50, "policy_loss": -1, **kwargs}

    def test_zero_distance_and_large_finite_loss_are_not_failures(self):
        problems, *_ = inspect_health(self.health(q_loss=1000000), 1000, (99, 19), 900)
        self.assertEqual(problems, [])

    def test_stale_dead_actor_nan_and_stall_detected(self):
        problems, *_ = inspect_health(self.health(actors_alive=7, q_loss=float('nan')), 1400, (100, 20), 1000)
        self.assertEqual(set(problems), {"stale_heartbeat", "missing_actor", "nonfinite_loss", "no_progress"})

    def test_updates_progress_without_new_episode(self):
        problems, _, timestamp = inspect_health(self.health(), 1000, (100, 19), 500)
        self.assertEqual(problems, [])
        self.assertEqual(timestamp, 1000)

    def test_command_timeouts_respect_deadline_and_write_final_report(self):
        class Clock:
            value = 0.0
            def monotonic(self): return self.value
            def time(self): return 1000 + self.value
            def sleep(self, seconds): self.value += seconds
        clock = Clock()
        def stuck_command(cmd, **kwargs):
            timeout = kwargs["timeout"]
            self.assertLessEqual(timeout, 250 - clock.value)
            clock.value += timeout
            raise subprocess.TimeoutExpired(cmd, timeout)
        with tempfile.TemporaryDirectory() as temp:
            argv = ["monitor", "--session", "fake.service", temp + "/missing",
                    "--duration", "250", "--interval", "60", "--output", temp]
            with patch("sys.argv", argv), patch("tools.monitor_training.time", clock), \
                 patch("tools.monitor_training.subprocess.run", side_effect=stuck_command):
                main()
            report = json.loads((Path(temp)/"finished.json").read_text())
            self.assertEqual(clock.value, 250)
            self.assertEqual(report["states"]["fake.service"]["restarts"], 1)
            checks = (Path(temp)/"checks.jsonl").read_text()
            self.assertIn("status_query_TimeoutExpired", checks)
            self.assertIn("restart_TimeoutExpired", checks)
            self.assertFalse(report["training_stop_requested"])
