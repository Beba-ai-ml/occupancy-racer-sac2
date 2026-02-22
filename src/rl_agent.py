from __future__ import annotations

import random as _random
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _build_mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.ReLU())
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


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
        last_dim = state_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        self.backbone = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(last_dim, action_dim)
        self.log_std_layer = nn.Linear(last_dim, action_dim)
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

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        if not hidden_sizes:
            hidden_sizes = [128, 128]
        self.net = _build_mlp(state_dim + action_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def to_list(self) -> list[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]:
        return [
            (self.states[i].copy(), self.actions[i].copy(), float(self.rewards[i]),
             self.next_states[i].copy(), float(self.dones[i]))
            for i in range(self.size)
        ]

    def load_list(self, items: list[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]) -> None:
        self.ptr = 0
        self.size = 0
        for state, action, reward, next_state, done in items:
            if self.size >= self.capacity:
                break
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = float(done)
            self.ptr = (self.ptr + 1) % self.capacity
            self.size += 1

    def __len__(self) -> int:
        return self.size


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: np.ndarray,
        action_bias: np.ndarray,
        policy_lr: float,
        q_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        batch_size: int,
        memory_size: int,
        target_entropy: float | None,
        init_alpha: float,
        start_steps: int,
        learn_after: int,
        update_every: int,
        updates_per_step: int,
        hidden_sizes: Iterable[int],
        grad_clip: float = 0.0,
        alpha_min: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.start_steps = max(0, int(start_steps))
        self.learn_after = max(0, int(learn_after))
        self.update_every = max(1, int(update_every))
        self.updates_per_step = max(1, int(updates_per_step))
        self.grad_clip = float(grad_clip)
        self.alpha_min = float(alpha_min)
        self.total_steps = 0
        self.total_updates = 0
        self.last_loss: float | None = None
        self.last_q_loss: float | None = None
        self.last_policy_loss: float | None = None
        self.last_alpha_loss: float | None = None
        self.last_entropy: float | None = None

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        action_scale = np.asarray(action_scale, dtype=np.float32)
        action_bias = np.asarray(action_bias, dtype=np.float32)
        self.action_low = (action_bias - action_scale).astype(np.float32)
        self.action_high = (action_bias + action_scale).astype(np.float32)

        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            action_scale=action_scale,
            action_bias=action_bias,
        ).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        _compile_mode = "default"
        if hasattr(torch, 'compile'):
            try:
                self.critic1 = torch.compile(self.critic1, mode=_compile_mode)
                self.critic2 = torch.compile(self.critic2, mode=_compile_mode)
                print(f"torch.compile applied to critics (mode={_compile_mode})")
            except Exception as e:
                print(f"torch.compile failed for critics: {e}")
            try:
                self.policy = torch.compile(self.policy, mode=_compile_mode)
                print(f"torch.compile applied to policy (mode={_compile_mode})")
            except Exception as e:
                print(f"torch.compile failed for policy: {e}")

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(policy_lr))
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(q_lr),
        )

        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        init_alpha = max(float(init_alpha), 1e-6)
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(alpha_lr))

        self.memory = ReplayBuffer(memory_size, state_dim, action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _random_action(self) -> np.ndarray:
        return np.random.uniform(self.action_low, self.action_high).astype(np.float32)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.total_steps < self.start_steps:
            return self._random_action()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic(state_t)
            else:
                action, _, _ = self.policy.sample(state_t)
        return action.cpu().numpy()[0]

    def select_action_batch(self, states: list[np.ndarray], deterministic: bool = False) -> list[np.ndarray]:
        if not states:
            return []
        if not deterministic and self.total_steps < self.start_steps:
            return [self._random_action() for _ in states]
        states_t = torch.tensor(np.array(states, dtype=np.float32), device=self.device)
        with torch.no_grad():
            if deterministic:
                actions = self.policy.deterministic(states_t)
            else:
                actions, _, _ = self.policy.sample(states_t)
        actions_np = actions.cpu().numpy()
        return [actions_np[idx] for idx in range(actions_np.shape[0])]

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.total_steps += 1
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size or self.total_steps < self.learn_after:
            self.last_loss = None
            return
        if self.update_every > 0 and self.total_steps % self.update_every != 0:
            return
        for _ in range(self.updates_per_step):
            self.last_loss = self.learn()

    def learn(self) -> float:
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        if self.device.type == "cuda":
            states_t = torch.from_numpy(states).pin_memory().to(self.device, non_blocking=True)
            actions_t = torch.from_numpy(actions).pin_memory().to(self.device, non_blocking=True)
            rewards_t = torch.from_numpy(rewards).pin_memory().to(self.device, non_blocking=True).unsqueeze(1)
            next_states_t = torch.from_numpy(next_states).pin_memory().to(self.device, non_blocking=True)
            dones_t = torch.from_numpy(dones).pin_memory().to(self.device, non_blocking=True).unsqueeze(1)
        else:
            states_t = torch.from_numpy(states)
            actions_t = torch.from_numpy(actions)
            rewards_t = torch.from_numpy(rewards).unsqueeze(1)
            next_states_t = torch.from_numpy(next_states)
            dones_t = torch.from_numpy(dones).unsqueeze(1)

        alpha = self.alpha.detach()
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.policy.sample(next_states_t)
            q1_next = self.critic1_target(next_states_t, next_actions)
            q2_next = self.critic2_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            q_target = rewards_t + (1.0 - dones_t) * self.gamma * q_next

        q1 = self.critic1(states_t, actions_t)
        q2 = self.critic2(states_t, actions_t)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self.grad_clip,
            )
        self.critic_optimizer.step()

        new_actions, log_prob, _ = self.policy.sample(states_t)
        q1_pi = self.critic1(states_t, new_actions)
        q2_pi = self.critic2(states_t, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (alpha * log_prob - min_q_pi).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        alpha_loss = None
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.alpha_min > 0 and self.log_alpha is not None:
                min_log_alpha = float(np.log(max(self.alpha_min, 1e-10)))
                with torch.no_grad():
                    self.log_alpha.data.clamp_(min=min_log_alpha)

        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        self.total_updates += 1
        self.last_q_loss = float(q_loss.item())
        self.last_policy_loss = float(policy_loss.item())
        self.last_alpha_loss = float(alpha_loss.item()) if alpha_loss is not None else None
        self.last_entropy = float(-log_prob.mean().item())
        return self.last_q_loss

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.tau
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, path: str, meta: dict | None = None) -> None:
        def _clean_sd(sd: dict) -> dict:
            return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

        payload = {
            "format": "sac_checkpoint_v1",
            "policy": _clean_sd(self.policy.state_dict()),
            "critic1": _clean_sd(self.critic1.state_dict()),
            "critic2": _clean_sd(self.critic2.state_dict()),
            "critic1_target": _clean_sd(self.critic1_target.state_dict()),
            "critic2_target": _clean_sd(self.critic2_target.state_dict()),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": float(self.log_alpha.item()) if self.log_alpha is not None else None,
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            # "memory": self.memory.to_list(),  # WYŁĄCZONE - warmup i tak zbiera nowe doświadczenia
            "rng": {
                "python": _random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "meta": meta or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, reset_optimizers: bool = False) -> dict:
        # Load with CPU mapping first to handle RNG states correctly
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, dict) or "policy" not in payload:
            raise ValueError("Unsupported checkpoint format")

        # Move model state dicts to target device
        def _to_device(state_dict: dict) -> dict:
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}

        def _adapt_keys(model: nn.Module, state_dict: dict) -> dict:
            """Handle torch.compile _orig_mod. prefix mismatch."""
            model_keys = set(model.state_dict().keys())
            sd_keys = set(state_dict.keys())
            if model_keys == sd_keys:
                return state_dict
            prefixed = {"_orig_mod." + k: v for k, v in state_dict.items()}
            if set(prefixed.keys()) == model_keys:
                return prefixed
            stripped = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            if set(stripped.keys()) == model_keys:
                return stripped
            return state_dict

        self.policy.load_state_dict(_to_device(_adapt_keys(self.policy, payload["policy"])))
        self.critic1.load_state_dict(_to_device(_adapt_keys(self.critic1, payload["critic1"])))
        self.critic2.load_state_dict(_to_device(_adapt_keys(self.critic2, payload["critic2"])))
        self.critic1_target.load_state_dict(_to_device(_adapt_keys(self.critic1_target, payload.get("critic1_target", payload["critic1"]))))
        self.critic2_target.load_state_dict(_to_device(_adapt_keys(self.critic2_target, payload.get("critic2_target", payload["critic2"]))))

        def _move_optimizer_state(optimizer: optim.Optimizer) -> None:
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)

        # Load optimizer states and move tensors to the correct device
        if reset_optimizers:
            print("Optimizers reset: keeping model weights, fresh Adam states")
        else:
            if "policy_optimizer" in payload:
                self.policy_optimizer.load_state_dict(payload["policy_optimizer"])
                _move_optimizer_state(self.policy_optimizer)
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
                _move_optimizer_state(self.critic_optimizer)

        if self.log_alpha is not None and payload.get("log_alpha") is not None:
            self.log_alpha.data.fill_(float(payload["log_alpha"]))
        if not reset_optimizers:
            if self.alpha_optimizer is not None and payload.get("alpha_optimizer") is not None:
                try:
                    self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
                    _move_optimizer_state(self.alpha_optimizer)
                except Exception as e:
                    print(f"Warning: Failed to load alpha_optimizer state: {e}")

        self.total_steps = int(payload.get("total_steps", self.total_steps))
        self.total_updates = int(payload.get("total_updates", self.total_updates))

        mem = payload.get("memory")
        if mem is not None:
            self.memory.load_list(mem)

        # Restore RNG states - torch.set_rng_state requires CPU ByteTensor
        rng = payload.get("rng") or {}
        rng_restored = False
        try:
            if rng.get("python") is not None:
                _random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                # Ensure CPU ByteTensor for torch RNG state
                torch_rng = rng["torch"]
                if isinstance(torch_rng, torch.Tensor):
                    torch_rng = torch_rng.cpu()
                torch.set_rng_state(torch_rng)
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                cuda_states = rng["torch_cuda"]
                # Ensure CPU tensors for CUDA RNG states
                if cuda_states:
                    cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
            rng_restored = True
        except Exception as e:
            print(f"Warning: Failed to restore RNG state: {e}")

        if not rng_restored:
            print("RNG state not restored - randomness may differ from previous session")

        return payload.get("meta", {}) or {}
