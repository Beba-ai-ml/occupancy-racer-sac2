from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
import queue as queue_mod
import multiprocessing as mp
from pathlib import Path
import re
try:
    import resource
except ImportError:
    resource = None  # not available on Windows
import shutil
import sys
from typing import List

import numpy as np
import torch
torch.set_float32_matmul_precision('high')

from .config import load_yaml, resolve_path
from .map_loader import load_map
from .params import build_map_params, build_vehicle_params
from .racer_env import LIDAR_ANGLES_DEG, RacerEnv
from .rl_agent import SACAgent, GaussianPolicy
from .sim_config import build_sim_config, register_sim_args


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _cpu_state_dict(model: torch.nn.Module) -> dict:
    return {k.removeprefix("_orig_mod."): v.detach().cpu() for k, v in model.state_dict().items()}


def _bump_nofile_limit() -> None:
    if resource is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return
    if hard == resource.RLIM_INFINITY:
        target = soft
    else:
        target = hard
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, target))
            print(f"Raised file descriptor limit: {soft} -> {target}")
        except Exception:
            pass


def _drain_queue(q: mp.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue_mod.Empty:
            break


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


def _parse_map_pool(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if str(value).strip()]
    if isinstance(raw, str):
        parts = [part.strip() for part in re.split(r"[,\s]+", raw) if part.strip()]
        return parts
    return []


def _parse_target_entropy(raw: object, action_dim: int) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("", "auto", "none"):
            return None
        return float(value)
    return float(raw)


def _load_checkpoint(agent: SACAgent, path: str, reset_optimizers: bool = False) -> tuple[dict | None, Exception | None]:
    try:
        meta = agent.load_checkpoint(path, reset_optimizers=reset_optimizers)
        return meta, None
    except Exception as exc:
        return None, exc


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


def _save_checkpoint_atomic(agent: SACAgent, path: str, meta: dict) -> None:
    tmp_path = f"{path}.tmp"
    backup_path = f"{path}.bak"
    agent.save_checkpoint(tmp_path, meta=meta)
    if os.path.exists(path):
        os.replace(path, backup_path)
    os.replace(tmp_path, path)


def _unique_checkpoint_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{base}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _save_checkpoint_backup(path: str, backup_path: str) -> None:
    backup_path = _unique_checkpoint_path(backup_path)
    try:
        os.link(path, backup_path)
    except OSError:
        shutil.copy2(path, backup_path)


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


def _actor_worker(
    actor_id: int,
    map_path: str,
    map_pool: list[str],
    map_switch_start_episode: int,
    map_switch_every: int,
    physics_cfg: dict,
    fps: int,
    steer_bins: List[int],
    accel_bins: List[float],
    stack_frames: int,
    render: bool,
    render_every: int,
    weights_queue: mp.Queue,
    transition_queue: mp.Queue,
    stats_queue: mp.Queue,
    stop_event: mp.Event,
    action_dim: int,
    hidden_sizes: list[int],
    action_scale: list[float],
    action_bias: list[float],
    device: str,
    episode_offset: int,
    sim_cfg: dict,
) -> None:
    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    vehicle_params = build_vehicle_params(physics_cfg)
    map_params = build_map_params(physics_cfg)

    def _build_env(map_path_local: str, episode_offset_local: int) -> RacerEnv:
        map_data = load_map(resolve_path(map_path_local))
        return RacerEnv(
            map_data,
            vehicle_params,
            map_params,
            steer_bins=steer_bins,
            accel_bins=accel_bins,
            fps=fps,
            render=False,
            stack_frames=stack_frames,
            episode_offset=episode_offset_local,
            sim_cfg=sim_cfg,
        )

    current_map_path = map_path
    map_block_remaining = 0
    if map_pool and map_switch_every > 0 and map_switch_start_episode <= 0:
        current_map_path = map_pool[int(np.random.randint(0, len(map_pool)))]
        map_block_remaining = map_switch_every

    env = _build_env(current_map_path, episode_offset)

    obs = env.reset()
    policy = GaussianPolicy(
        state_dim=obs.shape[0],
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        action_scale=np.array(action_scale, dtype=np.float32),
        action_bias=np.array(action_bias, dtype=np.float32),
    ).to(torch.device(device))
    policy.eval()
    device_t = torch.device(device)

    try:
        weights = weights_queue.get(timeout=60.0)
        policy.load_state_dict(weights)
        # Diagnostic: compute weight checksum and check action_scale/bias
        weight_sum = sum(v.sum().item() for v in policy.state_dict().values() if hasattr(v, 'sum'))
        a_scale = policy.action_scale.cpu().numpy()
        a_bias = policy.action_bias.cpu().numpy()
        print(f"Actor {actor_id}: Loaded weights, checksum={weight_sum:.4f}, action_scale={a_scale}, action_bias={a_bias}", flush=True)
    except queue_mod.Empty:
        print(f"Actor {actor_id}: WARNING - No weights received, using random initialization!", flush=True)

    episode_idx = 1
    episode_reward = 0.0
    episode_penalty = 0.0
    episode_distance = 0.0
    episode_time = 0.0
    episode_dt_sum = 0.0
    episode_dt_max = 0.0
    episode_steps = 0
    fixed_dt = 1.0 / fps if fps > 0 else 0.05

    def _update_render() -> None:
        if not render or render_every <= 0:
            return
        if episode_idx % render_every == 0:
            env.enable_render()
        else:
            env.disable_render()

    _update_render()

    def _maybe_switch_map(next_episode_idx: int) -> None:
        nonlocal env, current_map_path, map_block_remaining
        if not map_pool or map_switch_every <= 0:
            return
        absolute_episode = episode_offset + next_episode_idx - 1
        if absolute_episode < map_switch_start_episode:
            return
        if map_block_remaining <= 0:
            new_map_path = map_pool[int(np.random.randint(0, len(map_pool)))]
            if new_map_path != current_map_path:
                env.close()
                current_map_path = new_map_path
                env = _build_env(current_map_path, absolute_episode)
            map_block_remaining = map_switch_every
        map_block_remaining -= 1

    while not stop_event.is_set():
        try:
            while True:
                weights = weights_queue.get_nowait()
                policy.load_state_dict(weights)
        except queue_mod.Empty:
            pass

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device_t).unsqueeze(0)
            action_t, _, _ = policy.sample(obs_t)
            action = action_t.cpu().numpy()[0]

        next_obs, reward, done, distance_delta, death_reason = env.step(action)
        step_dt = env.last_dt
        episode_dt_sum += step_dt
        episode_dt_max = max(episode_dt_max, step_dt)
        episode_steps += 1
        transition_queue.put((obs, action, reward, next_obs, done))

        episode_reward += reward
        if reward < 0:
            episode_penalty += -reward
        episode_distance += distance_delta
        episode_time += fixed_dt

        if done:
            episode_idx += 1
            mean_dt = episode_dt_sum / max(1, episode_steps)
            fps_mean = 1.0 / mean_dt if mean_dt > 1e-9 else 0.0
            stats_queue.put(
                (
                    actor_id,
                    episode_distance,
                    episode_time,
                    episode_reward,
                    episode_penalty,
                    death_reason or "collision",
                    mean_dt,
                    episode_dt_max,
                    fps_mean,
                )
            )
            _maybe_switch_map(episode_idx)
            obs = env.reset()
            episode_reward = 0.0
            episode_penalty = 0.0
            episode_distance = 0.0
            episode_time = 0.0
            episode_dt_sum = 0.0
            episode_dt_max = 0.0
            episode_steps = 0
            _update_render()
        else:
            obs = next_obs

    env.close()


def _extract_config_file(raw_args: list[str]) -> tuple[str | None, list[str]]:
    config_file = None
    cleaned: list[str] = []
    skip_next = False
    for idx, arg in enumerate(raw_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--config-file":
            if idx + 1 >= len(raw_args):
                raise SystemExit("Missing value after --config-file")
            config_file = raw_args[idx + 1]
            skip_next = True
            continue
        if arg.startswith("--config-file="):
            config_file = arg.split("=", 1)[1]
            continue
        if arg == "config_file:":
            if idx + 1 >= len(raw_args):
                raise SystemExit("Missing value after config_file:")
            config_file = raw_args[idx + 1]
            skip_next = True
            continue
        if arg.startswith("config_file:"):
            config_file = arg.split(":", 1)[1]
            if config_file == "":
                if idx + 1 >= len(raw_args):
                    raise SystemExit("Missing value after config_file:")
                config_file = raw_args[idx + 1]
                skip_next = True
            continue
        if arg.startswith("config_file="):
            config_file = arg.split("=", 1)[1]
            continue
        cleaned.append(arg)
    return config_file, cleaned


def _resolve_config_file(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    candidates: list[Path] = []
    if path.suffix:
        candidates.append(path)
        candidates.append(Path("config") / path)
    else:
        candidates.extend(
            [
                Path(raw),
                Path(f"{raw}.yaml"),
                Path(f"{raw}.yml"),
                Path("config") / f"{raw}.yaml",
                Path("config") / f"{raw}.yml",
            ]
        )
    for cand in candidates:
        resolved = resolve_path(str(cand))
        if resolved.exists():
            return resolved
    return None


def _resolve_map_override(raw: str | None) -> str | None:
    if raw is None:
        return None
    resolved = resolve_path(raw)
    if resolved.exists():
        return str(resolved)
    base = Path(raw).stem
    candidates = [
        Path("assets/maps") / raw,
        Path("assets/maps") / f"{base}.pgm",
        Path("assets/maps") / f"{base}.yaml",
        Path("assets/maps") / f"{base}.yml",
    ]
    for cand in candidates:
        cand_path = resolve_path(str(cand))
        if cand_path.exists():
            return str(cand_path)
    raise SystemExit(f"Map not found: {raw}")


def _has_session_id_arg(raw_args: list[str]) -> bool:
    for arg in raw_args:
        if arg == "--session-id":
            return True
        if arg.startswith("--session-id="):
            return True
    return False


def _parse_config_index(config_path: Path | None, raw_name: str | None) -> str | None:
    name = None
    if config_path is not None:
        name = config_path.stem
    elif raw_name:
        name = Path(raw_name).stem
    if not name:
        return None
    match = re.search(r"config_sac_(\d+)", name)
    if not match:
        return None
    return match.group(1)


def _next_session_id(save_dir: str, config_idx: str) -> str:
    file_pattern = re.compile(rf"^session_Sesja_Sac_{re.escape(config_idx)}_(\d+)\\.(?:csv|pth|json)$")
    dir_pattern = re.compile(rf"^session_Sesja_Sac_{re.escape(config_idx)}_(\d+)$")
    max_seen = 0
    try:
        entries = os.listdir(save_dir)
    except FileNotFoundError:
        entries = []
    for name in entries:
        match = file_pattern.match(name) or dir_pattern.match(name)
        if not match:
            continue
        try:
            max_seen = max(max_seen, int(match.group(1)))
        except ValueError:
            continue
    return f"Sesja_Sac_{config_idx}_{max_seen + 1}"


def _validate_config_keys(parser: argparse.ArgumentParser, cfg: dict) -> None:
    allowed = {action.dest for action in parser._actions}
    unknown = sorted(key for key in cfg.keys() if key not in allowed)
    if unknown:
        raise SystemExit(f"Unknown config keys: {', '.join(unknown)}")


def train_ssac() -> None:
    parser = argparse.ArgumentParser(description="SSAC training for Occupancy Racer")
    parser.add_argument("--config-file", default=None, help="Path or name of a training config file")
    parser.add_argument("--config", default="config/game.yaml", help="Path to game config")
    parser.add_argument("--physics", default="config/physics.yaml", help="Path to physics config")
    parser.add_argument("--map", default=None, help="Override map path")
    parser.add_argument("--map-pool", default=None, help="Maps to rotate (comma-separated or YAML list)")
    parser.add_argument(
        "--map-switch-start-episode",
        type=int,
        default=None,
        help="Start rotating maps after N episodes",
    )
    parser.add_argument(
        "--map-switch-every",
        type=int,
        default=None,
        help="Episodes to keep a map before re-sampling",
    )
    parser.add_argument("--num-actors", type=int, default=None, help="Number of actor processes")
    parser.add_argument("--num-envs", type=int, default=None, help="Alias for --num-actors")
    parser.add_argument("--fps", type=int, default=None, help="Simulated fps")
    parser.add_argument("--max-episodes", type=int, default=None, help="Stop after N episodes (0=unlimited)")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint every N episodes")
    parser.add_argument("--session-id", default=None, help="Session id for checkpoint/log names")
    parser.add_argument("--save-dir", default=None, help="Directory for checkpoints/logs")
    parser.add_argument("--load-from", default=None, help="Load checkpoint from path")
    parser.add_argument("--device", default=None, help="Device for training (cpu/cuda/auto)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh even if checkpoints exist")
    parser.add_argument("--reset-optimizers", action="store_true", help="Keep model weights but reset Adam optimizer states (for fine-tuning with different hyperparams)")
    parser.add_argument("--stack-frames", type=int, default=None, help="Number of frames to stack")
    parser.add_argument("--sync-every", type=int, default=None, help="Sync actor weights every N transitions")
    parser.add_argument("--queue-size", type=int, default=None, help="Max transitions in queue")
    parser.add_argument("--render-actor", type=int, default=None, help="Actor id to render (optional)")
    parser.add_argument("--render-every", type=int, default=None, help="Render every N episodes for render actor")
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
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient norm clipping (0=disabled)")
    parser.add_argument("--alpha-min", type=float, default=None, help="Minimum alpha floor")
    parser.add_argument("--utd-ratio", type=float, default=None,
                        help="Update-to-data ratio (default from config or 1.0)")
    register_sim_args(parser)

    raw_args = sys.argv[1:]
    session_id_from_cli = _has_session_id_arg(raw_args)
    config_file_arg, cleaned_args = _extract_config_file(raw_args)
    config_file = config_file_arg
    config_path = None
    if config_file:
        config_path = _resolve_config_file(config_file)
        if config_path is None:
            raise SystemExit(f"Config file not found: {config_file}")
        config_data = load_yaml(config_path)
        if not isinstance(config_data, dict):
            raise SystemExit("Config file must be a YAML mapping of keys to values")
        _validate_config_keys(parser, config_data)
        parser.set_defaults(**config_data)

    args = parser.parse_args(cleaned_args)

    game_cfg = load_yaml(resolve_path(args.config))
    physics_cfg = load_yaml(resolve_path(args.physics))

    map_cfg = game_cfg.get("map", {})
    map_override = _resolve_map_override(args.map) if args.map is not None else None
    map_path = map_override or map_cfg.get("path") or map_cfg.get("yaml_path") or "assets/maps/K_02.pgm"
    map_pool_names = _parse_map_pool(args.map_pool)
    map_pool_paths: list[str] = []
    if map_pool_names:
        for name in map_pool_names:
            map_pool_paths.append(_resolve_map_override(name))
    map_switch_start = max(0, int(args.map_switch_start_episode or 0))
    map_switch_every = max(0, int(args.map_switch_every or 0))

    rl_cfg = game_cfg.get("rl", {})
    train_cfg = game_cfg.get("train", {})
    async_cfg = game_cfg.get("async", {})
    sim_cfg = build_sim_config(game_cfg, args)

    num_actors = args.num_actors if args.num_actors is not None else args.num_envs
    if num_actors is None:
        num_actors = int(async_cfg.get("num_actors", train_cfg.get("num_envs", 4)))
    _bump_nofile_limit()
    fps = args.fps if args.fps is not None else int(train_cfg.get("fps", 60))
    max_episodes = args.max_episodes if args.max_episodes is not None else int(train_cfg.get("max_episodes", 0))
    save_every = int(args.save_every if args.save_every is not None else rl_cfg.get("save_every", 0))
    stack_frames = int(args.stack_frames if args.stack_frames is not None else rl_cfg.get("stack_frames", 1))
    sync_every = int(args.sync_every if args.sync_every is not None else async_cfg.get("sync_every", 1000))
    queue_size = int(args.queue_size if args.queue_size is not None else async_cfg.get("queue_size", 5000))
    render_actor = args.render_actor if args.render_actor is not None else async_cfg.get("render_actor")
    render_every = int(args.render_every if args.render_every is not None else async_cfg.get("render_every", 0))

    steer_bins = list(range(0, 21))
    accel_bins = [float(v) for v in rl_cfg.get("accel_bins", [-2.0, -1.0, 0.0, 1.0, 2.0])]
    accel_min, accel_max = _resolve_accel_range(accel_bins)
    action_dim = 2
    action_scale = [1.0, (accel_max - accel_min) * 0.5]
    action_bias = [0.0, (accel_max + accel_min) * 0.5]

    device = args.device or rl_cfg.get("device") or "auto"
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes, rl_cfg.get("hidden_sizes", [128, 128]))
    target_entropy = _parse_target_entropy(args.target_entropy or rl_cfg.get("target_entropy"), action_dim)

    state_dim = (len(LIDAR_ANGLES_DEG) + 3) * max(1, stack_frames)
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
        grad_clip=float(args.grad_clip if args.grad_clip is not None else rl_cfg.get("grad_clip", 0.0)),
        alpha_min=float(args.alpha_min if args.alpha_min is not None else rl_cfg.get("alpha_min", 0.0)),
        device=device,
    )
    utd_ratio = float(args.utd_ratio if args.utd_ratio is not None else rl_cfg.get("utd_ratio", 1.0))
    print(f"Using device: {agent.device}")

    save_dir = str(args.save_dir or rl_cfg.get("save_dir", "runs"))
    load_from = args.load_from or rl_cfg.get("load_from")
    os.makedirs(save_dir, exist_ok=True)
    session_id = str(args.session_id or rl_cfg.get("session_id", "default"))
    if not session_id_from_cli:
        config_idx = _parse_config_index(config_path, config_file)
        if config_idx is not None:
            session_id = _next_session_id(save_dir, config_idx)
            args.session_id = session_id
    session_dir = os.path.join(save_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    session_ckpt = os.path.join(session_dir, f"session_{session_id}.pth")
    session_meta = os.path.join(session_dir, f"session_{session_id}.json")
    session_csv = os.path.join(session_dir, f"session_{session_id}.csv")
    legacy_session_ckpt = os.path.join(save_dir, f"session_{session_id}.pth")

    base_fieldnames = [
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
    extended_fieldnames = base_fieldnames + ["row_type", "config_json"]
    csv_has_data = os.path.exists(session_csv) and os.path.getsize(session_csv) > 0
    csv_fieldnames = extended_fieldnames
    csv_supports_config = True
    if csv_has_data:
        with open(session_csv, "r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header:
            csv_fieldnames = header
            csv_supports_config = "row_type" in header and "config_json" in header
        else:
            csv_fieldnames = extended_fieldnames
            csv_supports_config = True

    episodes_completed = 0
    loaded_distance_history: list[float] | None = None
    loaded_meta: dict | None = None
    if not args.no_resume:
        if load_from:
            resolved_load = _resolve_checkpoint_path(load_from)
            if resolved_load is None:
                print(f"Failed to locate checkpoint in {load_from}; starting fresh.")
            else:
                meta, error = _load_checkpoint(agent, resolved_load, reset_optimizers=args.reset_optimizers)
                if meta is not None:
                    print(f"Resuming from checkpoint: {resolved_load}")
                    loaded_meta = meta
                    episodes_completed = int(meta.get("episodes_trained_total", meta.get("episodes_trained", 0)))
                    loaded_distance_history = meta.get("distance_history")
                else:
                    backup_path = f"{resolved_load}.bak"
                    if os.path.exists(backup_path):
                        meta, backup_error = _load_checkpoint(agent, backup_path, reset_optimizers=args.reset_optimizers)
                        if meta is not None:
                            print(f"Resuming from backup checkpoint: {backup_path}")
                            loaded_meta = meta
                            episodes_completed = int(meta.get("episodes_trained_total", meta.get("episodes_trained", 0)))
                            loaded_distance_history = meta.get("distance_history")
                        else:
                            print(
                                f"Failed to load checkpoint ({error}); failed to load backup ({backup_error}); starting fresh."
                            )
                    else:
                        print(f"Failed to load checkpoint ({error}); starting fresh.")
        elif os.path.exists(session_ckpt):
            meta, error = _load_checkpoint(agent, session_ckpt, reset_optimizers=args.reset_optimizers)
            if meta is not None:
                print(f"Resuming from checkpoint: {session_ckpt}")
                loaded_meta = meta
                episodes_completed = int(meta.get("episodes_trained_total", meta.get("episodes_trained", 0)))
                loaded_distance_history = meta.get("distance_history")
            else:
                backup_path = f"{session_ckpt}.bak"
                if os.path.exists(backup_path):
                    meta, backup_error = _load_checkpoint(agent, backup_path, reset_optimizers=args.reset_optimizers)
                    if meta is not None:
                        print(f"Resuming from backup checkpoint: {backup_path}")
                        loaded_meta = meta
                        episodes_completed = int(meta.get("episodes_trained_total", meta.get("episodes_trained", 0)))
                        loaded_distance_history = meta.get("distance_history")
                    else:
                        print(
                            f"Failed to load checkpoint ({error}); failed to load backup ({backup_error}); starting fresh."
                        )
                else:
                    print(f"Failed to load checkpoint ({error}); starting fresh.")
        elif os.path.exists(legacy_session_ckpt):
            meta, error = _load_checkpoint(agent, legacy_session_ckpt, reset_optimizers=args.reset_optimizers)
            if meta is not None:
                print(f"Resuming from checkpoint: {legacy_session_ckpt}")
                loaded_meta = meta
                episodes_completed = int(meta.get("episodes_trained_total", meta.get("episodes_trained", 0)))
                loaded_distance_history = meta.get("distance_history")
            else:
                backup_path = f"{legacy_session_ckpt}.bak"
                if os.path.exists(backup_path):
                    meta, backup_error = _load_checkpoint(agent, backup_path, reset_optimizers=args.reset_optimizers)
                    if meta is not None:
                        print(f"Resuming from backup checkpoint: {backup_path}")
                        loaded_meta = meta
                        episodes_completed = int(meta.get("episodes_trained_total", meta.get("episodes_trained", 0)))
                        loaded_distance_history = meta.get("distance_history")
                    else:
                        print(
                            f"Failed to load checkpoint ({error}); failed to load backup ({backup_error}); starting fresh."
                        )
                else:
                    print(f"Failed to load checkpoint ({error}); starting fresh.")
        else:
            print("Starting new session (no checkpoint found).")
    else:
        print("Starting new session (resume disabled).")

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    transition_queue: mp.Queue = ctx.Queue(maxsize=max(1, queue_size))
    stats_queue: mp.Queue = ctx.Queue()
    weights_queues: list[mp.Queue] = [ctx.Queue(maxsize=1) for _ in range(num_actors)]
    processes: list[mp.Process] = []

    for actor_id in range(num_actors):
        render_flag = render_actor is not None and int(render_actor) == actor_id and render_every > 0
        proc = ctx.Process(
            target=_actor_worker,
            args=(
                actor_id,
                map_path,
                map_pool_paths,
                map_switch_start,
                map_switch_every,
                physics_cfg,
                fps,
                steer_bins,
                accel_bins,
                stack_frames,
                render_flag,
                render_every,
                weights_queues[actor_id],
                transition_queue,
                stats_queue,
                stop_event,
                action_dim,
                hidden_sizes,
                action_scale,
                action_bias,
                "cpu",
                episodes_completed,
                sim_cfg,
            ),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    init_weights = _cpu_state_dict(agent.policy)
    # Diagnostic: compute weight checksum before sending to actors
    weight_sum = sum(v.sum().item() for v in init_weights.values() if hasattr(v, 'sum'))
    a_scale = init_weights.get('action_scale', 'N/A')
    a_bias = init_weights.get('action_bias', 'N/A')
    print(f"Main process: Sending weights to actors, checksum={weight_sum:.4f}, action_scale={a_scale}, action_bias={a_bias}", flush=True)
    for q in weights_queues:
        _drain_queue(q)
        q.put(init_weights)

    distance_history: deque[float] = deque(maxlen=100)
    if loaded_distance_history:
        distance_history.extend(loaded_distance_history[-100:])
        print(f"Restored distance_history with {len(distance_history)} entries, mean100={_mean(list(distance_history)):.2f}m")
    elif episodes_completed > 0 and os.path.exists(session_csv):
        # Fallback: load from CSV if checkpoint doesn't have distance_history
        try:
            with open(session_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                distances = []
                for row in reader:
                    if row.get("row_type") == "episode" or (row.get("distance_m") and row.get("row_type", "episode") == "episode"):
                        try:
                            dist = float(row.get("distance_m", 0))
                            distances.append(dist)
                        except (ValueError, TypeError):
                            pass
                if distances:
                    distance_history.extend(distances[-100:])
                    print(f"Restored distance_history from CSV with {len(distance_history)} entries, mean100={_mean(list(distance_history)):.2f}m")
        except Exception as e:
            print(f"Could not restore distance_history from CSV: {e}")

    last_logged_episode: int | None = None
    if loaded_meta is not None and loaded_meta.get("episodes_trained_logged") is not None:
        try:
            last_logged_episode = int(loaded_meta.get("episodes_trained_logged", 0))
        except (TypeError, ValueError):
            last_logged_episode = None
    if last_logged_episode is None and os.path.exists(session_csv):
        try:
            with open(session_csv, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("row_type") not in (None, "", "episode"):
                        continue
                    raw_episode = row.get("total_episode")
                    if not raw_episode:
                        continue
                    try:
                        last_logged_episode = int(float(raw_episode))
                    except (TypeError, ValueError):
                        continue
        except Exception as e:
            print(f"Could not restore last logged episode from CSV: {e}")

    # Warmup after resume: scale with actor count to keep per-actor warmup comparable
    base_warmup_steps = 37500
    base_actors = 4
    warmup_scale = max(1.0, float(num_actors) / float(base_actors))
    resume_warmup_steps = int(base_warmup_steps * warmup_scale) if episodes_completed > 0 else 0
    if resume_warmup_steps > 0:
        print(f"Resume warmup: delaying learning and weight sync for {resume_warmup_steps} steps")
    print(f"  UTD ratio: {utd_ratio} (drain 512 × {utd_ratio} = ~{int(512 * utd_ratio)} updates/cycle)")

    total_episodes = episodes_completed
    episodes_done = 0
    logged_total_episodes = last_logged_episode if last_logged_episode is not None else episodes_completed
    logged_episodes_done = 0
    transitions = 0

    def _ensure_csv_header() -> None:
        if os.path.exists(session_csv) and os.path.getsize(session_csv) > 0:
            return
        with open(session_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            writer.writeheader()

    def _build_csv_row(values: dict) -> dict:
        row = {name: "" for name in csv_fieldnames}
        for key, value in values.items():
            if key in row:
                row[key] = value
        return row

    def _write_config_snapshot() -> None:
        if not csv_supports_config:
            return
        _ensure_csv_header()
        resolved = {
            "config": args.config,
            "physics": args.physics,
            "map_path": map_path,
            "map_pool": map_pool_paths,
            "map_switch_start_episode": map_switch_start,
            "map_switch_every": map_switch_every,
            "num_actors": num_actors,
            "fps": fps,
            "max_episodes": max_episodes,
            "save_every": save_every,
            "stack_frames": stack_frames,
            "sync_every": sync_every,
            "queue_size": queue_size,
            "render_actor": render_actor,
            "render_every": render_every,
            "hidden_sizes": hidden_sizes,
            "device": str(device),
            "target_entropy": target_entropy,
            "utd_ratio": utd_ratio,
        }
        config_snapshot = {
            "config_file": str(config_path) if config_path is not None else None,
            "args": vars(args),
            "resolved": resolved,
            "sim_cfg": sim_cfg,
        }
        row = _build_csv_row(
            {
                "row_type": "config",
                "config_json": json.dumps(config_snapshot, sort_keys=True),
            }
        )
        with open(session_csv, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            writer.writerow(row)

    def log_episode(
        episode_id: int,
        episode_in_run: int,
        env_id: int,
        distance: float,
        time_s: float,
        reward_sum: float,
        penalty_sum: float,
        mean_20: float,
        mean_100: float,
        death_reason: str,
        mean_dt: float,
        max_dt: float,
        fps_mean: float,
    ) -> None:
        alpha_value = float(agent.alpha.item())
        _ensure_csv_header()
        with open(session_csv, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            row = _build_csv_row(
                {
                    "total_episode": episode_id,
                    "episode_in_run": episode_in_run,
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
                    "row_type": "episode",
                    "config_json": "",
                }
            )
            writer.writerow(row)
            csv_file.flush()
        q_loss_str = f"{agent.last_q_loss:.4f}" if agent.last_q_loss is not None else ""
        policy_loss_str = f"{agent.last_policy_loss:.4f}" if agent.last_policy_loss is not None else ""
        print(
            f"[Ep {episode_id:05d}] env={env_id} time={time_s:.2f}s distance={distance:.2f}m "
            f"mean(20)={mean_20:.2f}m mean(100)={mean_100:.2f}m "
            f"alpha={alpha_value:.3f} q_loss={q_loss_str} policy_loss={policy_loss_str} "
            f"death={death_reason or 'collision'} reward_sum={reward_sum:.2f} penalty_sum={penalty_sum:.2f} "
            f"dt_mean={mean_dt:.4f}s fps={fps_mean:.1f} dt_max={max_dt:.4f}s"
        )

    prev_sync_bucket = 0  # track sync_every crossings across batch drains

    try:
        if not csv_has_data:
            _write_config_snapshot()
        while True:
            if max_episodes > 0 and total_episodes >= max_episodes:
                break

            # --- Batch drain: pull up to 64 transitions at once ---
            transitions_received = 0
            # Block on first transition to avoid busy-looping
            try:
                state, action, reward, next_state, done = transition_queue.get(timeout=0.1)
            except queue_mod.Empty:
                state = None
            if state is not None:
                agent.memory.add(state, np.asarray(action, dtype=np.float32), float(reward), next_state, bool(done))
                agent.total_steps += 1
                transitions += 1
                transitions_received += 1
                if transitions <= resume_warmup_steps:
                    if transitions == 1:
                        print(f"Warmup: collecting experiences without learning for {resume_warmup_steps} steps...")
                    if transitions % 10000 == 0:
                        print(f"Warmup progress: {transitions}/{resume_warmup_steps} steps")
                # Drain remaining without blocking
                drain_limit = min(512, max(0, transition_queue.qsize()))
                for _ in range(drain_limit):
                    try:
                        state, action, reward, next_state, done = transition_queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    agent.memory.add(state, np.asarray(action, dtype=np.float32), float(reward), next_state, bool(done))
                    agent.total_steps += 1
                    transitions += 1
                    transitions_received += 1
                    if transitions <= resume_warmup_steps:
                        if transitions % 10000 == 0:
                            print(f"Warmup progress: {transitions}/{resume_warmup_steps} steps")

            # --- Learning: batch updates proportional to transitions received ---
            if transitions_received > 0 and transitions > resume_warmup_steps:
                if transitions - transitions_received < resume_warmup_steps and resume_warmup_steps > 0:
                    print(f"Warmup complete! Starting learning...")
                can_learn = (len(agent.memory) >= agent.batch_size
                             and agent.total_steps >= agent.learn_after)
                if can_learn:
                    num_updates = max(1, int(transitions_received * utd_ratio * agent.updates_per_step))
                    for _ in range(num_updates):
                        agent.last_loss = agent.learn()

            # --- Weight sync: check if we crossed a sync_every boundary ---
            if sync_every > 0 and transitions > resume_warmup_steps:
                cur_bucket = transitions // sync_every
                if cur_bucket > prev_sync_bucket:
                    prev_sync_bucket = cur_bucket
                    latest = _cpu_state_dict(agent.policy)
                    for q in weights_queues:
                        _drain_queue(q)
                        q.put(latest)

            while True:
                try:
                    (
                        env_id,
                        distance,
                        time_s,
                        reward_sum,
                        penalty_sum,
                        death_reason,
                        mean_dt,
                        max_dt,
                        fps_mean,
                    ) = stats_queue.get_nowait()
                except queue_mod.Empty:
                    break
                total_episodes += 1
                episodes_done += 1
                distance_history.append(distance)
                mean_20 = _mean(list(distance_history)[-20:])
                mean_100 = _mean(list(distance_history)[-100:])
                if resume_warmup_steps <= 0 or transitions >= resume_warmup_steps:
                    logged_total_episodes += 1
                    logged_episodes_done += 1
                    log_episode(
                        logged_total_episodes,
                        logged_episodes_done,
                        env_id,
                        distance,
                        time_s,
                        reward_sum,
                        penalty_sum,
                        mean_20,
                        mean_100,
                        death_reason,
                        mean_dt,
                        max_dt,
                        fps_mean,
                    )
                if save_every > 0 and total_episodes % save_every == 0:
                    meta = {
                        "session_id": session_id,
                        "episodes_trained": total_episodes,
                        "episodes_trained_total": total_episodes,
                        "episodes_trained_logged": logged_total_episodes,
                        "alpha": float(agent.alpha.item()),
                        "num_envs": num_actors,
                        "distance_history": list(distance_history),
                    }
                    _save_checkpoint_atomic(agent, session_ckpt, meta)
                    with open(session_meta, "w", encoding="utf-8") as handle:
                        json.dump(meta, handle)
                    backup_name = f"session_{session_id}_Backup_{total_episodes}.pth"
                    backup_path = os.path.join(session_dir, backup_name)
                    _save_checkpoint_backup(session_ckpt, backup_path)
    finally:
        stop_event.set()
        for proc in processes:
            proc.join(timeout=1.0)


if __name__ == "__main__":
    train_ssac()
