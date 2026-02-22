from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
from typing import List

import numpy as np

from .config import load_yaml, resolve_path
from .racer_env import LIDAR_ANGLES_DEG
from .rl_agent import SACAgent
from .sim_config import build_sim_config, register_sim_args
from .vec_env import VecRacerEnv


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _parse_hidden_sizes(raw: object, default: list[int]) -> list[int]:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            return default
        return [int(part) for part in parts]
    return default


def _parse_target_entropy(raw: object, action_dim: int) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("", "auto", "none"):
            return None
        return float(value)
    return float(raw)


def _resolve_accel_range(accel_bins: object) -> tuple[float, float]:
    if accel_bins is None:
        values = [-2.0, 2.0]
    else:
        values = [float(v) for v in accel_bins]
    if not values:
        values = [-2.0, 2.0]
    accel_min = float(min(values))
    accel_max = float(max(values))
    if accel_min > accel_max:
        accel_min, accel_max = accel_max, accel_min
    return accel_min, accel_max


def _resolve_checkpoint_path(load_from: str) -> str | None:
    if os.path.isdir(load_from):
        try:
            entries = [name for name in os.listdir(load_from) if name.endswith(".pth")]
        except FileNotFoundError:
            return None
        if not entries:
            return None
        session_entries = [name for name in entries if name.startswith("session_")]
        if session_entries:
            entries = session_entries
        entries.sort(
            key=lambda name: os.path.getmtime(os.path.join(load_from, name)),
            reverse=True,
        )
        return os.path.join(load_from, entries[0])
    return load_from


def train() -> None:
    parser = argparse.ArgumentParser(description="Multi-env training for Occupancy Racer")
    parser.add_argument("--config", default="config/game.yaml", help="Path to game config")
    parser.add_argument("--physics", default="config/physics.yaml", help="Path to physics config")
    parser.add_argument("--map", default=None, help="Override map path")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of envs")
    parser.add_argument("--render-every", type=int, default=None, help="Render every N episodes (global)")
    parser.add_argument(
        "--render-envs",
        default=None,
        help="Comma-separated env ids to render (e.g. 0,3) or 'all'/'none'",
    )
    parser.add_argument("--fps", type=int, default=None, help="Simulated fps")
    parser.add_argument("--max-episodes", type=int, default=None, help="Stop after N episodes (0=unlimited)")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint every N episodes")
    parser.add_argument("--session-id", default=None, help="Session id for checkpoint/log names")
    parser.add_argument("--save-dir", default=None, help="Directory for checkpoints/logs")
    parser.add_argument("--load-from", default=None, help="Load checkpoint from path")
    parser.add_argument("--device", default=None, help="Device for training (cpu/cuda/auto)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh even if checkpoints exist")
    parser.add_argument("--stack-frames", type=int, default=None, help="Number of frames to stack")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--tau", type=float, default=None, help="Target smoothing coefficient")
    parser.add_argument("--policy-lr", type=float, default=None, help="Actor learning rate")
    parser.add_argument("--q-lr", type=float, default=None, help="Critic learning rate")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Entropy learning rate")
    parser.add_argument("--init-alpha", type=float, default=None, help="Initial entropy coefficient")
    parser.add_argument("--target-entropy", default=None, help="Target entropy (float or 'auto')")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--memory-size", type=int, default=None, help="Replay buffer size")
    parser.add_argument("--start-steps", type=int, default=None, help="Random action steps")
    parser.add_argument("--learn-after", type=int, default=None, help="Start learning after N steps")
    parser.add_argument("--update-every", type=int, default=None, help="Update frequency in steps")
    parser.add_argument("--updates-per-step", type=int, default=None, help="Gradient updates per update step")
    parser.add_argument("--hidden-sizes", default=None, help="Comma-separated hidden layer sizes")
    register_sim_args(parser)
    args = parser.parse_args()

    game_cfg = load_yaml(resolve_path(args.config))
    physics_cfg = load_yaml(resolve_path(args.physics))

    map_cfg = game_cfg.get("map", {})
    map_path = args.map or map_cfg.get("path") or map_cfg.get("yaml_path") or "assets/maps/K_02.pgm"

    rl_cfg = game_cfg.get("rl", {})
    train_cfg = game_cfg.get("train", {})
    sim_cfg = build_sim_config(game_cfg, args)
    num_envs = args.num_envs if args.num_envs is not None else int(train_cfg.get("num_envs", 4))
    render_every = args.render_every if args.render_every is not None else int(train_cfg.get("render_every", 0))
    fps = args.fps if args.fps is not None else int(train_cfg.get("fps", 60))
    max_episodes = args.max_episodes if args.max_episodes is not None else int(train_cfg.get("max_episodes", 0))
    render_envs_raw = args.render_envs if args.render_envs is not None else train_cfg.get("render_envs")

    def _parse_render_envs(raw: object) -> List[int] | None:
        if render_every <= 0:
            return None
        if raw is None:
            return [0] if num_envs > 1 else None
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "all":
                return None
            if value in ("none", "off"):
                return []
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if not parts:
                return [0] if num_envs > 1 else None
            ids: List[int] = []
            for part in parts:
                try:
                    idx = int(part)
                except ValueError:
                    parser.error(f"Invalid --render-envs entry: {part}")
                if idx < 0 or idx >= num_envs:
                    parser.error(f"--render-envs id {idx} out of range 0..{num_envs - 1}")
                ids.append(idx)
            return sorted(set(ids))
        if isinstance(raw, list):
            ids = []
            for part in raw:
                try:
                    idx = int(part)
                except (TypeError, ValueError):
                    parser.error(f"Invalid --render-envs entry: {part}")
                if idx < 0 or idx >= num_envs:
                    parser.error(f"--render-envs id {idx} out of range 0..{num_envs - 1}")
                ids.append(idx)
            return sorted(set(ids))
        return None

    render_env_ids = _parse_render_envs(render_envs_raw)

    steer_bins = list(range(0, 21))
    accel_bins = [float(v) for v in rl_cfg.get("accel_bins", [-2.0, -1.0, 0.0, 1.0, 2.0])]
    accel_min, accel_max = _resolve_accel_range(accel_bins)
    action_dim = 2
    action_scale = [1.0, (accel_max - accel_min) * 0.5]
    action_bias = [0.0, (accel_max + accel_min) * 0.5]
    stack_frames = int(args.stack_frames if args.stack_frames is not None else rl_cfg.get("stack_frames", 1))

    device = args.device or rl_cfg.get("device") or "auto"
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes, rl_cfg.get("hidden_sizes", [128, 128]))
    target_entropy = _parse_target_entropy(args.target_entropy or rl_cfg.get("target_entropy"), action_dim)

    state_dim = (len(LIDAR_ANGLES_DEG) + 5) * max(1, stack_frames)  # +5: collision, speed, servo, linear_accel, angular_vel
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=np.array(action_scale, dtype=np.float32),
        action_bias=np.array(action_bias, dtype=np.float32),
        policy_lr=float(args.policy_lr if args.policy_lr is not None else rl_cfg.get("policy_lr", 3e-4)),
        q_lr=float(args.q_lr if args.q_lr is not None else rl_cfg.get("q_lr", 3e-4)),
        alpha_lr=float(args.alpha_lr if args.alpha_lr is not None else rl_cfg.get("alpha_lr", 3e-4)),
        gamma=float(args.gamma if args.gamma is not None else rl_cfg.get("gamma", 0.99)),
        tau=float(args.tau if args.tau is not None else rl_cfg.get("tau", rl_cfg.get("soft_tau", 0.005))),
        batch_size=int(args.batch_size if args.batch_size is not None else rl_cfg.get("batch_size", 256)),
        memory_size=int(args.memory_size if args.memory_size is not None else rl_cfg.get("memory_size", 100000)),
        target_entropy=target_entropy,
        init_alpha=float(args.init_alpha if args.init_alpha is not None else rl_cfg.get("init_alpha", 0.2)),
        start_steps=int(args.start_steps if args.start_steps is not None else rl_cfg.get("start_steps", 0)),
        learn_after=int(args.learn_after if args.learn_after is not None else rl_cfg.get("learn_after", 1000)),
        update_every=int(args.update_every if args.update_every is not None else rl_cfg.get("update_every", 1)),
        updates_per_step=int(args.updates_per_step if args.updates_per_step is not None else rl_cfg.get("updates_per_step", 1)),
        hidden_sizes=hidden_sizes,
        device=device,
        grad_clip=float(args.grad_clip if hasattr(args, 'grad_clip') and args.grad_clip is not None else rl_cfg.get("grad_clip", 0.0)),
        alpha_min=float(args.alpha_min if hasattr(args, 'alpha_min') and args.alpha_min is not None else rl_cfg.get("alpha_min", 0.0)),
    )
    print(f"Using device: {agent.device}")

    save_dir = str(args.save_dir or rl_cfg.get("save_dir", "runs"))
    save_every = int(args.save_every if args.save_every is not None else rl_cfg.get("save_every", 0))
    session_id = str(args.session_id or rl_cfg.get("session_id", "default"))
    load_from = args.load_from or rl_cfg.get("load_from")
    os.makedirs(save_dir, exist_ok=True)
    session_dir = os.path.join(save_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    session_ckpt = os.path.join(session_dir, f"session_{session_id}.pth")
    session_meta = os.path.join(session_dir, f"session_{session_id}.json")
    session_csv = os.path.join(session_dir, f"session_{session_id}.csv")
    legacy_session_ckpt = os.path.join(save_dir, f"session_{session_id}.pth")

    episodes_completed = 0
    if not args.no_resume:
        if load_from:
            resolved_load = _resolve_checkpoint_path(load_from)
            if resolved_load is None:
                print(f"Failed to locate checkpoint in {load_from}; starting fresh.")
            else:
                print(f"Resuming from checkpoint: {resolved_load}")
                try:
                    meta = agent.load_checkpoint(resolved_load)
                    episodes_completed = int(meta.get("episodes_trained", 0))
                except Exception as exc:
                    print(f"Failed to load checkpoint ({exc}); starting fresh.")
        elif os.path.exists(session_ckpt):
            print(f"Resuming from checkpoint: {session_ckpt}")
            try:
                meta = agent.load_checkpoint(session_ckpt)
                episodes_completed = int(meta.get("episodes_trained", 0))
            except Exception as exc:
                print(f"Failed to load checkpoint ({exc}); starting fresh.")
        elif os.path.exists(legacy_session_ckpt):
            print(f"Resuming from checkpoint: {legacy_session_ckpt}")
            try:
                meta = agent.load_checkpoint(legacy_session_ckpt)
                episodes_completed = int(meta.get("episodes_trained", 0))
            except Exception as exc:
                print(f"Failed to load checkpoint ({exc}); starting fresh.")
        else:
            print("Starting new session (no checkpoint found).")
    else:
        print("Starting new session (resume disabled).")

    env = VecRacerEnv(
        num_envs=num_envs,
        map_path=map_path,
        physics_cfg=physics_cfg,
        fps=fps,
        render_every=render_every,
        render_env_ids=render_env_ids,
        global_render=render_every > 0,
        stack_frames=stack_frames,
        steer_bins=steer_bins,
        accel_bins=accel_bins,
        sim_cfg=sim_cfg,
    )

    states = env.reset()
    render_env_set: set[int] = set()
    render_pending = 0
    render_active_env: int | None = None
    if render_every > 0:
        if render_env_ids is None:
            render_env_set = set(range(num_envs))
        else:
            render_env_set = set(render_env_ids)
    if render_every > 0 and render_env_set:
        for env_id in render_env_set:
            env.set_render(env_id, False)
    episode_rewards = [0.0 for _ in range(num_envs)]
    episode_penalties = [0.0 for _ in range(num_envs)]
    episode_distances = [0.0 for _ in range(num_envs)]
    episode_times = [0.0 for _ in range(num_envs)]
    distance_history: deque[float] = deque(maxlen=100)
    total_episodes = episodes_completed
    episodes_done = 0
    step_time = 1.0 / fps if fps > 0 else 0.05

    csv_fieldnames = [
        "total_episode",
        "episode_in_run",
        "env_id",
        "distance_m",
        "time_s",
        "reward_sum",
        "penalty_sum",
        "mean_20",
        "mean_100",
        "alpha",
        "q_loss",
        "policy_loss",
        "alpha_loss",
        "entropy",
        "death_reason",
    ]
    csv_header_written = os.path.exists(session_csv) and os.path.getsize(session_csv) > 0

    def log_episode(
        env_id: int,
        distance: float,
        time_s: float,
        reward_sum: float,
        penalty_sum: float,
        mean_20: float,
        mean_100: float,
        death_reason: str,
    ) -> None:
        nonlocal csv_header_written
        alpha_value = float(agent.alpha.item())
        row = {
            "total_episode": total_episodes,
            "episode_in_run": episodes_done,
            "env_id": env_id,
            "distance_m": f"{distance:.4f}",
            "time_s": f"{time_s:.2f}",
            "reward_sum": f"{reward_sum:.6f}",
            "penalty_sum": f"{penalty_sum:.6f}",
            "mean_20": f"{mean_20:.4f}",
            "mean_100": f"{mean_100:.4f}",
            "alpha": f"{alpha_value:.6f}",
            "q_loss": f"{agent.last_q_loss:.6f}" if agent.last_q_loss is not None else "",
            "policy_loss": f"{agent.last_policy_loss:.6f}" if agent.last_policy_loss is not None else "",
            "alpha_loss": f"{agent.last_alpha_loss:.6f}" if agent.last_alpha_loss is not None else "",
            "entropy": f"{agent.last_entropy:.6f}" if agent.last_entropy is not None else "",
            "death_reason": death_reason or "collision",
        }
        write_header = not csv_header_written
        with open(session_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            if write_header:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(row)
            f.flush()
        q_loss_str = f"{agent.last_q_loss:.4f}" if agent.last_q_loss is not None else ""
        policy_loss_str = f"{agent.last_policy_loss:.4f}" if agent.last_policy_loss is not None else ""
        print(
            f"[Ep {total_episodes:05d}] env={env_id} time={time_s:.2f}s distance={distance:.2f}m "
            f"mean(20)={mean_20:.2f}m mean(100)={mean_100:.2f}m "
            f"alpha={alpha_value:.3f} q_loss={q_loss_str} policy_loss={policy_loss_str} "
            f"death={death_reason or 'collision'} reward_sum={reward_sum:.2f} penalty_sum={penalty_sum:.2f}"
        )

    try:
        while True:
            if max_episodes > 0 and total_episodes >= max_episodes:
                break
            actions = agent.select_action_batch(states)
            next_states, rewards, dones, distances, deaths = env.step(actions)
            for idx in range(num_envs):
                agent.step(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
                episode_rewards[idx] += rewards[idx]
                if rewards[idx] < 0:
                    episode_penalties[idx] += -rewards[idx]
                episode_distances[idx] += distances[idx]
                episode_times[idx] += step_time

                if dones[idx]:
                    total_episodes += 1
                    episodes_done += 1
                    if render_every > 0 and render_env_set and total_episodes % render_every == 0:
                        render_pending += 1
                    distance_history.append(episode_distances[idx])
                    mean_20 = _mean(list(distance_history)[-20:])
                    mean_100 = _mean(list(distance_history)[-100:])
                    log_episode(
                        idx,
                        episode_distances[idx],
                        episode_times[idx],
                        episode_rewards[idx],
                        episode_penalties[idx],
                        mean_20,
                        mean_100,
                        deaths[idx] if deaths else "",
                    )
                    if save_every > 0 and total_episodes % save_every == 0:
                        meta = {
                            "session_id": session_id,
                            "episodes_trained": total_episodes,
                            "alpha": float(agent.alpha.item()),
                            "num_envs": num_envs,
                            "distance_history": list(distance_history),
                        }
                        agent.save_checkpoint(session_ckpt, meta=meta)
                        with open(session_meta, "w", encoding="utf-8") as handle:
                            json.dump(meta, handle)

                    if render_active_env == idx:
                        env.set_render(idx, False)
                        render_active_env = None

                    states[idx] = env.reset_at(idx)
                    episode_rewards[idx] = 0.0
                    episode_penalties[idx] = 0.0
                    episode_distances[idx] = 0.0
                    episode_times[idx] = 0.0
                    if render_pending > 0 and render_active_env is None and idx in render_env_set:
                        env.set_render(idx, True)
                        render_active_env = idx
                        render_pending -= 1
                else:
                    states[idx] = next_states[idx]
    finally:
        env.close()


if __name__ == "__main__":
    train()
