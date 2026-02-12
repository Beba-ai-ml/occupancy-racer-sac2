from __future__ import annotations

import multiprocessing as mp
import os
from typing import List, Tuple

from .config import resolve_path
from .map_loader import load_map
from .params import build_map_params, build_vehicle_params
from .racer_env import RacerEnv

DEFAULT_RENDER_COLS = 2


def _set_window_position(env_id: int, window_width: int, window_height: int) -> None:
    cols_raw = os.environ.get("RACER_RENDER_COLS")
    cols = DEFAULT_RENDER_COLS
    if cols_raw:
        try:
            cols = int(cols_raw)
        except ValueError:
            cols = DEFAULT_RENDER_COLS
    if cols < 1:
        cols = DEFAULT_RENDER_COLS

    x = (env_id % cols) * window_width
    y = (env_id // cols) * window_height
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    map_path: str,
    physics_cfg: dict,
    fps: int,
    render_every: int,
    env_id: int,
    steer_bins: List[int],
    accel_bins: List[float],
    stack_frames: int,
    sim_cfg: dict | None,
) -> None:
    parent_remote.close()
    render_allowed = render_every != 0
    manual_render = render_every < 0
    auto_render = render_every > 0
    if not render_allowed:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    map_data = load_map(resolve_path(map_path))
    vehicle_params = build_vehicle_params(physics_cfg)
    map_params = build_map_params(physics_cfg)

    if render_allowed:
        window_width = map_data.image.shape[1]
        window_height = map_data.image.shape[0]
        _set_window_position(env_id, window_width, window_height)

    env = RacerEnv(
        map_data,
        vehicle_params,
        map_params,
        steer_bins=steer_bins,
        accel_bins=accel_bins,
        fps=fps,
        render=auto_render,
        stack_frames=stack_frames,
        sim_cfg=sim_cfg,
    )
    episode_idx = 0
    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == "reset":
                episode_idx += 1
                if auto_render:
                    env.render_enabled = episode_idx % render_every == 0
                remote.send(env.reset())
            elif cmd == "step":
                state, reward, done, distance_delta, death_reason = env.step(data)
                remote.send((state, reward, done, distance_delta, death_reason))
            elif cmd == "render":
                if render_allowed:
                    if bool(data):
                        env.enable_render()
                    else:
                        env.disable_render()
                remote.send(None)
            elif cmd == "close":
                remote.send(None)
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    finally:
        env.close()
        remote.close()


class VecRacerEnv:
    def __init__(
        self,
        num_envs: int,
        map_path: str,
        physics_cfg: dict,
        fps: int,
        render_every: int = 0,
        render_env_ids: List[int] | None = None,
        global_render: bool = False,
        stack_frames: int = 1,
        steer_bins: List[int] | None = None,
        accel_bins: List[float] | None = None,
        sim_cfg: dict | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.remotes = list(self.remotes)
        self.work_remotes = list(self.work_remotes)
        self.ps: List[mp.Process] = []

        steer_bins = steer_bins or list(range(0, 21))
        accel_bins = accel_bins or [-2.0, -1.0, 0.0, 1.0, 2.0]

        render_env_set = set(render_env_ids) if render_env_ids is not None else None
        for env_id, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            render_every_env = render_every
            if render_env_set is not None and env_id not in render_env_set:
                render_every_env = 0
            elif global_render and render_every > 0:
                render_every_env = -1
            proc = ctx.Process(
                target=_worker,
                args=(
                    work_remote,
                    remote,
                    map_path,
                    physics_cfg,
                    fps,
                    render_every_env,
                    env_id,
                    steer_bins,
                    accel_bins,
                    stack_frames,
                    sim_cfg,
                ),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.ps.append(proc)

    def reset(self) -> List[Tuple]:
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def reset_at(self, index: int) -> Tuple:
        self.remotes[index].send(("reset", None))
        return self.remotes[index].recv()

    def step(self, actions: List[object]) -> tuple[List[Tuple], List[float], List[bool], List[float], List[str]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, dones, distances, deaths = zip(*results)
        return list(next_states), list(rewards), list(dones), list(distances), list(deaths)

    def set_render(self, index: int, enabled: bool) -> None:
        self.remotes[index].send(("render", bool(enabled)))
        self.remotes[index].recv()

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for remote in self.remotes:
            try:
                remote.recv()
            except Exception:
                pass
        for proc in self.ps:
            proc.join(timeout=1.0)
