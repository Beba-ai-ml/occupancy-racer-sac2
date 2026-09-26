from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from src.train_ssac import _save_checkpoint_atomic


class CheckpointGuardTests(unittest.TestCase):
    def test_bad_parameters_or_optimizer_preserve_last_good_checkpoint(self):
        for corrupt in ("parameter", "optimizer"):
            models = [torch.nn.Linear(1, 1) for _ in range(5)]
            opt = SimpleNamespace(state={})
            agent = SimpleNamespace(policy=models[0], critic1=models[1], critic2=models[2],
                critic1_target=models[3], critic2_target=models[4], log_alpha=torch.tensor(-1.),
                policy_optimizer=opt, critic_optimizer=opt, alpha_optimizer=opt,
                # Simulate stale finite losses from the preceding learn() call.
                last_q_loss=1.0, last_policy_loss=1.0,
                save_checkpoint=lambda *a, **k: self.fail("corrupt checkpoint was serialized"))
            if corrupt == "parameter":
                with torch.no_grad(): models[0].weight.fill_(float('nan'))
            else:
                opt.state[0] = {"exp_avg": torch.tensor(float('inf'))}
            with tempfile.TemporaryDirectory() as temp:
                p = Path(temp)/"weights.pth"
                p.write_bytes(b"LAST GOOD CHECKPOINT")
                with self.assertRaises(FloatingPointError):
                    _save_checkpoint_atomic(agent, str(p), {})
                self.assertEqual(p.read_bytes(), b"LAST GOOD CHECKPOINT")
                self.assertFalse(Path(str(p)+".bak").exists())
