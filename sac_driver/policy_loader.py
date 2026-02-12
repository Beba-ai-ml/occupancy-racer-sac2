"""Standalone policy loader for SAC GaussianPolicy (offline)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

_LOGGER = logging.getLogger(__name__)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        action_scale: np.ndarray,
        action_bias: np.ndarray,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        if not hidden_sizes:
            hidden_sizes = [128, 128]
        layers: list[nn.Module] = []
        last_dim = int(state_dim)
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(size)))
            layers.append(nn.ReLU())
            last_dim = int(size)
        self.backbone = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(last_dim, int(action_dim))
        self.log_std_layer = nn.Linear(last_dim, int(action_dim))
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))
        self._eps = 1e-6

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        safe_scale = torch.clamp(self.action_scale, min=self._eps)
        log_prob = log_prob - torch.log(safe_scale * (1.0 - y_t.pow(2)) + self._eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_mean_action(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return self.get_mean_action(state)


def _infer_arch_from_state_dict(
    state_dict: dict,
) -> Tuple[int, int, list[int], np.ndarray, np.ndarray]:
    if "backbone.0.weight" not in state_dict or "mean_layer.weight" not in state_dict:
        raise ValueError("State dict missing required backbone/mean_layer weights.")

    state_dim = int(state_dict["backbone.0.weight"].shape[1])
    action_dim = int(state_dict["mean_layer.weight"].shape[0])

    hidden_layers: list[tuple[int, int]] = []
    for key, value in state_dict.items():
        if key.startswith("backbone.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            hidden_layers.append((idx, int(value.shape[0])))
    hidden_layers.sort(key=lambda item: item[0])
    hidden_sizes = [size for _, size in hidden_layers]

    action_scale = state_dict.get("action_scale")
    action_bias = state_dict.get("action_bias")
    if action_scale is None or action_bias is None:
        _LOGGER.warning("action_scale/action_bias missing in state dict; using defaults.")
        action_scale = np.ones(action_dim, dtype=np.float32)
        action_bias = np.zeros(action_dim, dtype=np.float32)
    else:
        action_scale = action_scale.detach().cpu().numpy()
        action_bias = action_bias.detach().cpu().numpy()

    return state_dim, action_dim, hidden_sizes, action_scale, action_bias


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_policy(
    pth_path: str | Path,
    device: str | torch.device | None = "cpu",
    *,
    weights_only: bool = False,
) -> GaussianPolicy:
    """Load a policy from a .pth checkpoint and return an eval() model.

    NOTE: If the checkpoint contains non-weight objects, weights_only=False is required.
    Only use weights_only=False on trusted files.
    """
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    device_obj = _resolve_device(device)
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=weights_only)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = checkpoint["policy"]
    else:
        policy_state = checkpoint

    state_dim, action_dim, hidden_sizes, action_scale, action_bias = _infer_arch_from_state_dict(
        policy_state
    )
    model = GaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        action_scale=action_scale,
        action_bias=action_bias,
    )
    model.load_state_dict(policy_state)
    model.to(device_obj)
    model.eval()

    _LOGGER.info(
        "Loaded policy from %s (state_dim=%d, action_dim=%d, hidden=%s, device=%s)",
        pth_path,
        state_dim,
        action_dim,
        hidden_sizes,
        device_obj.type,
    )
    return model


def export_policy_weights(
    pth_path: str | Path,
    output_path: str | Path,
    *,
    weights_only: bool = False,
) -> Path:
    """Extract and save only policy weights to a smaller file."""
    pth_path = Path(pth_path)
    output_path = Path(output_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=weights_only)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy_state = checkpoint["policy"]
    else:
        policy_state = checkpoint

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_state, output_path)
    _LOGGER.info("Saved policy weights to %s", output_path)
    return output_path


__all__ = [
    "GaussianPolicy",
    "load_policy",
    "export_policy_weights",
]
