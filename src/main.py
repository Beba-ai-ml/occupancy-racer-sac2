from __future__ import annotations

import argparse
from .config import load_yaml, resolve_path
from .game import Game
from .map_loader import load_map
from .params import build_map_params, build_vehicle_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Occupancy grid racing prototype")
    parser.add_argument("--config", default="config/game.yaml", help="Path to game config")
    parser.add_argument("--physics", default="config/physics.yaml", help="Path to physics config")
    parser.add_argument("--map", default=None, help="Override map YAML path")
    parser.add_argument("--control-mode", choices=["human", "rl"], help="Override control mode")
    args = parser.parse_args()

    game_cfg = load_yaml(resolve_path(args.config))
    physics_cfg = load_yaml(resolve_path(args.physics))
    if args.control_mode:
        game_cfg.setdefault("control", {})["mode"] = args.control_mode

    map_cfg = game_cfg.get("map", {})
    map_path = args.map or map_cfg.get("path") or map_cfg.get("yaml_path") or "assets/maps/K_02.pgm"
    map_data = load_map(resolve_path(map_path))

    display_cfg = game_cfg.get("display", {})
    window_cfg = game_cfg.get("window", {})

    vehicle_params = build_vehicle_params(physics_cfg)
    map_params = build_map_params(physics_cfg)
    control_cfg = game_cfg.get("control", {})
    rl_cfg = game_cfg.get("rl", {})

    game = Game(map_data, vehicle_params, map_params, display_cfg, window_cfg, control_cfg, rl_cfg)
    game.run()


if __name__ == "__main__":
    main()
