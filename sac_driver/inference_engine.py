"""Inference wrapper for SAC policy."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .policy_loader import GaussianPolicy, load_policy

_LOGGER = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        policy_path: str | Path,
        device: str | torch.device | None = "cpu",
        *,
        weights_only: bool = False,
        action_scale: Optional[np.ndarray] = None,
        action_bias: Optional[np.ndarray] = None,
    ) -> None:
        self.policy: GaussianPolicy = load_policy(
            policy_path,
            device=device,
            weights_only=weights_only,
        )
        self.device = next(self.policy.parameters()).device
        if action_scale is not None:
            self.policy.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        if action_bias is not None:
            self.policy.action_bias = torch.as_tensor(action_bias, dtype=torch.float32, device=self.device)

        self.last_inference_ms: Optional[float] = None

    def get_action(self, state: np.ndarray) -> Tuple[float, float]:
        state_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state_np).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            action_t = self.policy.get_mean_action(state_t)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.last_inference_ms = elapsed_ms
        _LOGGER.debug("Policy inference: %.3f ms", elapsed_ms)

        action = action_t.detach().cpu().numpy().reshape(-1)
        if action.size < 2:
            raise ValueError(f"Expected action_dim=2, got {action.size}")
        return float(action[0]), float(action[1])


__all__ = ["InferenceEngine"]
