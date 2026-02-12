from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any


def register_sim_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Sim randomization")
    group.add_argument("--dr-enable", dest="dr_enable", action="store_true", default=None, help="Enable sim randomization")
    group.add_argument("--dr-disable", dest="dr_enable", action="store_false", default=None, help="Disable sim randomization")

    group.add_argument("--dr-physics", dest="dr_physics", action="store_true", default=None, help="Enable physics randomization")
    group.add_argument("--dr-no-physics", dest="dr_physics", action="store_false", default=None, help="Disable physics randomization")
    group.add_argument("--dr-accel-scale-range", nargs=2, type=float, default=None, help="Acceleration scale range")
    group.add_argument("--dr-brake-scale-range", nargs=2, type=float, default=None, help="Brake decel scale range")
    group.add_argument("--dr-reverse-scale-range", nargs=2, type=float, default=None, help="Reverse accel scale range")
    group.add_argument("--dr-max-speed-scale-range", nargs=2, type=float, default=None, help="Max speed scale range")
    group.add_argument(
        "--dr-max-reverse-speed-scale-range",
        nargs=2,
        type=float,
        default=None,
        help="Max reverse speed scale range",
    )
    group.add_argument("--dr-friction-scale-range", nargs=2, type=float, default=None, help="Friction scale range")
    group.add_argument("--dr-drag-scale-range", nargs=2, type=float, default=None, help="Drag scale range")
    group.add_argument("--dr-wheelbase-scale-range", nargs=2, type=float, default=None, help="Wheelbase scale range")
    group.add_argument("--dr-max-steer-scale-range", nargs=2, type=float, default=None, help="Max steer scale range")

    group.add_argument("--dr-surface", dest="dr_surface", action="store_true", default=None, help="Enable surface randomization")
    group.add_argument("--dr-no-surface", dest="dr_surface", action="store_false", default=None, help="Disable surface randomization")
    group.add_argument(
        "--dr-surface-friction-scale-range",
        nargs=2,
        type=float,
        default=None,
        help="Surface friction scale range",
    )
    group.add_argument("--dr-surface-drag-range", nargs=2, type=float, default=None, help="Surface drag range")

    group.add_argument("--dr-obs-noise", dest="dr_obs_noise", action="store_true", default=None, help="Enable obs noise")
    group.add_argument("--dr-no-obs-noise", dest="dr_obs_noise", action="store_false", default=None, help="Disable obs noise")
    group.add_argument("--dr-lidar-noise-std", type=float, default=None, help="Lidar noise std (normalized)")
    group.add_argument("--dr-lidar-drop-prob", type=float, default=None, help="Lidar dropout prob")
    group.add_argument("--dr-lidar-spike-prob", type=float, default=None, help="Lidar spike prob")
    group.add_argument("--dr-speed-noise-std", type=float, default=None, help="Speed noise std (normalized)")
    group.add_argument("--dr-servo-noise-std", type=float, default=None, help="Servo noise std (normalized)")

    group.add_argument("--dr-control", dest="dr_control", action="store_true", default=None, help="Enable control lag")
    group.add_argument("--dr-no-control", dest="dr_control", action="store_false", default=None, help="Disable control lag")
    group.add_argument(
        "--dr-delay-steps",
        nargs="+",
        type=int,
        default=None,
        help="Control delay steps (one value or min max)",
    )
    group.add_argument("--dr-steer-rate-limit", type=float, default=None, help="Steer rate limit per step")
    group.add_argument("--dr-accel-rate-limit", type=float, default=None, help="Accel rate limit per step")

    group.add_argument("--dr-dt-jitter", dest="dr_dt_jitter", action="store_true", default=None, help="Enable dt jitter")
    group.add_argument("--dr-no-dt-jitter", dest="dr_dt_jitter", action="store_false", default=None, help="Disable dt jitter")
    group.add_argument("--dr-dt-scale-range", nargs=2, type=float, default=None, help="Dt scale range")

    group.add_argument("--dr-action-noise", dest="dr_action_noise", action="store_true", default=None, help="Enable action noise")
    group.add_argument("--dr-no-action-noise", dest="dr_action_noise", action="store_false", default=None, help="Disable action noise")
    group.add_argument(
        "--dr-action-steer-scale-range", nargs=2, type=float, default=None, help="Action steer scale range"
    )
    group.add_argument(
        "--dr-action-steer-bias-range", nargs=2, type=float, default=None, help="Action steer bias range"
    )
    group.add_argument(
        "--dr-action-accel-scale-range", nargs=2, type=float, default=None, help="Action accel scale range"
    )
    group.add_argument(
        "--dr-action-accel-bias-range", nargs=2, type=float, default=None, help="Action accel bias range"
    )

    group.add_argument("--dr-perturb", dest="dr_perturb", action="store_true", default=None, help="Enable perturbations")
    group.add_argument("--dr-no-perturb", dest="dr_perturb", action="store_false", default=None, help="Disable perturbations")
    group.add_argument("--dr-perturb-prob", type=float, default=None, help="Perturb prob per step")
    group.add_argument("--dr-perturb-yaw-rate-sigma-deg", type=float, default=None, help="Yaw rate sigma (deg/s)")
    group.add_argument("--dr-perturb-speed-sigma", type=float, default=None, help="Speed sigma (m/s)")

    group.add_argument("--dr-obs-delay", dest="dr_obs_delay", action="store_true", default=None, help="Enable obs delay")
    group.add_argument("--dr-no-obs-delay", dest="dr_obs_delay", action="store_false", default=None, help="Disable obs delay")
    group.add_argument("--dr-obs-delay-p1", type=float, default=None, help="Obs delay prob (t-1)")
    group.add_argument("--dr-obs-delay-p2", type=float, default=None, help="Obs delay prob (t-2)")

    group.add_argument("--action-repeat", type=int, default=None, help="Repeat each action N steps")

    group.add_argument("--obst-enable", dest="obst_enable", action="store_true", default=None, help="Enable obstacles")
    group.add_argument("--obst-disable", dest="obst_enable", action="store_false", default=None, help="Disable obstacles")
    group.add_argument("--obst-episode-prob", type=float, default=None, help="Obstacle episode probability")
    group.add_argument("--obst-start-episode", type=int, default=None, help="Enable obstacles after N episodes")
    group.add_argument("--obst-max-static", type=int, default=None, help="Max static obstacles per episode")
    group.add_argument("--obst-max-total", type=int, default=None, help="Max total obstacles active")
    group.add_argument("--obst-static-size-range", nargs=2, type=float, default=None, help="Static obstacle size range (m)")
    group.add_argument("--obst-static-radius-range", nargs=2, type=float, default=None, help="Static obstacle radius range (m)")
    group.add_argument("--obst-static-persist-prob", type=float, default=None, help="Static obstacle persist prob")
    group.add_argument("--obst-static-min-distance", type=float, default=None, help="Min distance from vehicle spawn (m)")
    group.add_argument("--obst-static-spawn-attempts", type=int, default=None, help="Static spawn attempts")
    group.add_argument("--obst-min-gap", type=float, default=None, help="Min passable gap on either side (m)")
    group.add_argument("--obst-min-wall-clearance", type=float, default=None, help="Min wall clearance (m)")
    group.add_argument("--obst-min-separation", type=float, default=None, help="Min obstacle separation (m)")
    group.add_argument("--obst-pass-buffer", type=float, default=None, help="Pass buffer distance (m)")
    group.add_argument(
        "--obst-pass-activate-distance", type=float, default=None, help="Distance to activate pass tracking (m)"
    )
    group.add_argument(
        "--obst-allow-wall-overlap",
        dest="obst_allow_wall_overlap",
        action="store_true",
        default=None,
        help="Allow obstacles to overlap walls",
    )
    group.add_argument(
        "--obst-no-wall-overlap",
        dest="obst_allow_wall_overlap",
        action="store_false",
        default=None,
        help="Disallow obstacle overlap with walls",
    )
    group.add_argument("--obst-dynamic", dest="obst_dynamic", action="store_true", default=None, help="Enable dynamic obstacle")
    group.add_argument("--obst-no-dynamic", dest="obst_dynamic", action="store_false", default=None, help="Disable dynamic obstacle")
    group.add_argument("--obst-dynamic-spawn-rate", type=float, default=None, help="Dynamic spawn rate (per second)")
    group.add_argument("--obst-dynamic-distance-range", nargs=2, type=float, default=None, help="Dynamic spawn distance range (m)")
    group.add_argument("--obst-dynamic-size-range", nargs=2, type=float, default=None, help="Dynamic obstacle size range (m)")
    group.add_argument("--obst-dynamic-radius-range", nargs=2, type=float, default=None, help="Dynamic obstacle radius range (m)")
    group.add_argument("--obst-dynamic-lateral-range", nargs=2, type=float, default=None, help="Dynamic lateral offset range (m)")
    group.add_argument("--obst-dynamic-ttl-range", nargs=2, type=float, default=None, help="Dynamic obstacle TTL range (s)")
    group.add_argument("--obst-dynamic-max", type=int, default=None, help="Max dynamic obstacles active")
    group.add_argument("--obst-dynamic-spawn-attempts", type=int, default=None, help="Dynamic spawn attempts")

    reward_group = parser.add_argument_group("Reward shaping")
    reward_group.add_argument("--reward-scale", type=float, default=None, help="Reward scale factor")
    reward_group.add_argument("--reward-clip", type=float, default=None, help="Reward clip value (<=0 disables)")
    reward_group.add_argument("--reward-no-clip", action="store_true", default=None, help="Disable reward clipping")
    reward_group.add_argument(
        "--reward-collision-penalty", type=float, default=None, help="Reward value on collision"
    )
    reward_group.add_argument(
        "--reward-front-penalty",
        type=float,
        default=None,
        help="Penalty for low front clearance at speed",
    )
    reward_group.add_argument("--reward-side-penalty", type=float, default=None, help="Penalty for low side clearance")
    reward_group.add_argument("--reward-min-clear-penalty", type=float, default=None, help="Penalty for minimal clearance")
    reward_group.add_argument(
        "--reward-balance-penalty", type=float, default=None, help="Penalty for left/right imbalance"
    )
    reward_group.add_argument("--reward-reverse-penalty", type=float, default=None, help="Penalty for reverse speed")
    reward_group.add_argument("--reward-alignment-bonus", type=float, default=None, help="Alignment bonus scale")
    reward_group.add_argument(
        "--reward-forward-speed-weight", type=float, default=None, help="Forward speed weight in reward"
    )
    reward_group.add_argument(
        "--reward-front-speed-weight",
        type=float,
        default=None,
        help="Front clearance speed weight in reward",
    )
    reward_group.add_argument(
        "--reward-front-cone-deg",
        type=float,
        default=None,
        help="Front cone half-angle (deg) for clearance",
    )

    track_group = parser.add_argument_group("Track guidance")
    track_group.add_argument(
        "--track-center",
        choices=["map", "free"],
        default=None,
        help="Center used for track direction (map center or free-space centroid)",
    )
    track_group.add_argument(
        "--track-direction",
        choices=["clockwise", "counterclockwise"],
        default=None,
        help="Preferred travel direction for alignment reward",
    )
    track_group.add_argument(
        "--spawn-face-forward",
        dest="spawn_face_forward",
        action="store_true",
        default=None,
        help="Flip spawn heading if backward has more clearance",
    )
    track_group.add_argument(
        "--spawn-no-face-forward",
        dest="spawn_face_forward",
        action="store_false",
        default=None,
        help="Disable spawn heading flip",
    )

    episode_group = parser.add_argument_group("Episode limits")
    episode_group.add_argument(
        "--episode-time-limit-s",
        type=float,
        default=None,
        help="End episode after N seconds (0 disables)",
    )


def build_sim_config(game_cfg: dict, args: argparse.Namespace | None) -> dict:
    sim_cfg = deepcopy(game_cfg.get("sim_randomization") or {})
    if args is None:
        return sim_cfg

    def _set_nested(cfg: dict, path: tuple[str, ...], value: Any) -> None:
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    def _range_from(values: list[float]) -> list[float]:
        if len(values) < 2:
            return [float(values[0]), float(values[0])]
        lo, hi = float(values[0]), float(values[1])
        if lo > hi:
            lo, hi = hi, lo
        return [lo, hi]

    if args.dr_enable is not None:
        sim_cfg["enabled"] = args.dr_enable

    if args.dr_physics is not None:
        _set_nested(sim_cfg, ("physics", "enabled"), args.dr_physics)
    if args.dr_accel_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "accel_scale"), _range_from(args.dr_accel_scale_range))
    if args.dr_brake_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "brake_scale"), _range_from(args.dr_brake_scale_range))
    if args.dr_reverse_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "reverse_scale"), _range_from(args.dr_reverse_scale_range))
    if args.dr_max_speed_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "max_speed_scale"), _range_from(args.dr_max_speed_scale_range))
    if args.dr_max_reverse_speed_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "max_reverse_speed_scale"), _range_from(args.dr_max_reverse_speed_scale_range))
    if args.dr_friction_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "friction_scale"), _range_from(args.dr_friction_scale_range))
    if args.dr_drag_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "drag_scale"), _range_from(args.dr_drag_scale_range))
    if args.dr_wheelbase_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "wheelbase_scale"), _range_from(args.dr_wheelbase_scale_range))
    if args.dr_max_steer_scale_range is not None:
        _set_nested(sim_cfg, ("physics", "max_steer_scale"), _range_from(args.dr_max_steer_scale_range))

    if args.dr_surface is not None:
        _set_nested(sim_cfg, ("surface", "enabled"), args.dr_surface)
    if args.dr_surface_friction_scale_range is not None:
        _set_nested(sim_cfg, ("surface", "friction_scale"), _range_from(args.dr_surface_friction_scale_range))
    if args.dr_surface_drag_range is not None:
        _set_nested(sim_cfg, ("surface", "drag_range"), _range_from(args.dr_surface_drag_range))

    if args.dr_obs_noise is not None:
        _set_nested(sim_cfg, ("observation_noise", "enabled"), args.dr_obs_noise)
    if args.dr_lidar_noise_std is not None:
        _set_nested(sim_cfg, ("observation_noise", "lidar_noise_std"), float(args.dr_lidar_noise_std))
    if args.dr_lidar_drop_prob is not None:
        _set_nested(sim_cfg, ("observation_noise", "lidar_dropout_prob"), float(args.dr_lidar_drop_prob))
    if args.dr_lidar_spike_prob is not None:
        _set_nested(sim_cfg, ("observation_noise", "lidar_spike_prob"), float(args.dr_lidar_spike_prob))
    if args.dr_speed_noise_std is not None:
        _set_nested(sim_cfg, ("observation_noise", "speed_noise_std"), float(args.dr_speed_noise_std))
    if args.dr_servo_noise_std is not None:
        _set_nested(sim_cfg, ("observation_noise", "servo_noise_std"), float(args.dr_servo_noise_std))

    if args.dr_control is not None:
        _set_nested(sim_cfg, ("control", "enabled"), args.dr_control)
    if args.dr_delay_steps is not None:
        steps = [int(v) for v in args.dr_delay_steps if v is not None]
        if steps:
            if len(steps) == 1:
                steps = [steps[0], steps[0]]
            if steps[0] > steps[1]:
                steps = [steps[1], steps[0]]
            _set_nested(sim_cfg, ("control", "delay_steps"), steps)
    if args.dr_steer_rate_limit is not None:
        _set_nested(sim_cfg, ("control", "steer_rate_limit"), float(args.dr_steer_rate_limit))
    if args.dr_accel_rate_limit is not None:
        _set_nested(sim_cfg, ("control", "accel_rate_limit"), float(args.dr_accel_rate_limit))

    if args.dr_dt_jitter is not None:
        _set_nested(sim_cfg, ("dt_jitter", "enabled"), args.dr_dt_jitter)
    if args.dr_dt_scale_range is not None:
        _set_nested(sim_cfg, ("dt_jitter", "dt_scale_range"), _range_from(args.dr_dt_scale_range))

    if args.dr_action_noise is not None:
        _set_nested(sim_cfg, ("action_noise", "enabled"), args.dr_action_noise)
    if args.dr_action_steer_scale_range is not None:
        _set_nested(sim_cfg, ("action_noise", "steer_scale_range"), _range_from(args.dr_action_steer_scale_range))
    if args.dr_action_steer_bias_range is not None:
        _set_nested(sim_cfg, ("action_noise", "steer_bias_range"), _range_from(args.dr_action_steer_bias_range))
    if args.dr_action_accel_scale_range is not None:
        _set_nested(sim_cfg, ("action_noise", "accel_scale_range"), _range_from(args.dr_action_accel_scale_range))
    if args.dr_action_accel_bias_range is not None:
        _set_nested(sim_cfg, ("action_noise", "accel_bias_range"), _range_from(args.dr_action_accel_bias_range))

    if args.dr_perturb is not None:
        _set_nested(sim_cfg, ("perturb", "enabled"), args.dr_perturb)
    if args.dr_perturb_prob is not None:
        _set_nested(sim_cfg, ("perturb", "prob"), float(args.dr_perturb_prob))
    if args.dr_perturb_yaw_rate_sigma_deg is not None:
        _set_nested(sim_cfg, ("perturb", "yaw_rate_sigma_deg"), float(args.dr_perturb_yaw_rate_sigma_deg))
    if args.dr_perturb_speed_sigma is not None:
        _set_nested(sim_cfg, ("perturb", "speed_sigma"), float(args.dr_perturb_speed_sigma))

    if args.dr_obs_delay is not None:
        _set_nested(sim_cfg, ("observation_delay", "enabled"), args.dr_obs_delay)
    if args.dr_obs_delay_p1 is not None:
        _set_nested(sim_cfg, ("observation_delay", "p1"), float(args.dr_obs_delay_p1))
    if args.dr_obs_delay_p2 is not None:
        _set_nested(sim_cfg, ("observation_delay", "p2"), float(args.dr_obs_delay_p2))

    if args.action_repeat is not None:
        steps = max(1, int(args.action_repeat))
        _set_nested(sim_cfg, ("action_repeat", "steps"), steps)

    if args.obst_enable is not None:
        _set_nested(sim_cfg, ("obstacles", "enabled"), args.obst_enable)
    if args.obst_episode_prob is not None:
        _set_nested(sim_cfg, ("obstacles", "episode_prob"), float(args.obst_episode_prob))
    if args.obst_start_episode is not None:
        _set_nested(sim_cfg, ("obstacles", "start_episode"), int(args.obst_start_episode))
    if args.obst_max_static is not None:
        _set_nested(sim_cfg, ("obstacles", "max_static"), int(args.obst_max_static))
    if args.obst_max_total is not None:
        _set_nested(sim_cfg, ("obstacles", "max_total"), int(args.obst_max_total))
    if args.obst_static_size_range is not None:
        _set_nested(sim_cfg, ("obstacles", "static_size_range"), _range_from(args.obst_static_size_range))
    if args.obst_static_radius_range is not None:
        _set_nested(sim_cfg, ("obstacles", "static_radius_range"), _range_from(args.obst_static_radius_range))
    if args.obst_static_persist_prob is not None:
        _set_nested(sim_cfg, ("obstacles", "static_persist_prob"), float(args.obst_static_persist_prob))
    if args.obst_static_min_distance is not None:
        _set_nested(sim_cfg, ("obstacles", "static_min_distance"), float(args.obst_static_min_distance))
    if args.obst_static_spawn_attempts is not None:
        _set_nested(sim_cfg, ("obstacles", "static_spawn_attempts"), int(args.obst_static_spawn_attempts))
    if args.obst_min_gap is not None:
        _set_nested(sim_cfg, ("obstacles", "min_gap"), float(args.obst_min_gap))
    if args.obst_min_wall_clearance is not None:
        _set_nested(sim_cfg, ("obstacles", "min_wall_clearance"), float(args.obst_min_wall_clearance))
    if args.obst_min_separation is not None:
        _set_nested(sim_cfg, ("obstacles", "min_separation"), float(args.obst_min_separation))
    if args.obst_pass_buffer is not None:
        _set_nested(sim_cfg, ("obstacles", "pass_buffer"), float(args.obst_pass_buffer))
    if args.obst_pass_activate_distance is not None:
        _set_nested(sim_cfg, ("obstacles", "pass_activate_distance"), float(args.obst_pass_activate_distance))
    if args.obst_allow_wall_overlap is not None:
        _set_nested(sim_cfg, ("obstacles", "allow_wall_overlap"), bool(args.obst_allow_wall_overlap))
    if args.obst_dynamic is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "enabled"), args.obst_dynamic)
    if args.obst_dynamic_spawn_rate is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "spawn_rate"), float(args.obst_dynamic_spawn_rate))
    if args.obst_dynamic_distance_range is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "distance_range"), _range_from(args.obst_dynamic_distance_range))
    if args.obst_dynamic_size_range is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "size_range"), _range_from(args.obst_dynamic_size_range))
    if args.obst_dynamic_radius_range is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "radius_range"), _range_from(args.obst_dynamic_radius_range))
    if args.obst_dynamic_lateral_range is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "lateral_range"), _range_from(args.obst_dynamic_lateral_range))
    if args.obst_dynamic_ttl_range is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "ttl_range"), _range_from(args.obst_dynamic_ttl_range))
    if args.obst_dynamic_max is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "max_active"), int(args.obst_dynamic_max))
    if args.obst_dynamic_spawn_attempts is not None:
        _set_nested(sim_cfg, ("obstacles", "dynamic", "spawn_attempts"), int(args.obst_dynamic_spawn_attempts))

    if args.reward_scale is not None:
        _set_nested(sim_cfg, ("reward", "scale"), float(args.reward_scale))
    if args.reward_no_clip:
        _set_nested(sim_cfg, ("reward", "clip"), None)
    elif args.reward_clip is not None:
        _set_nested(sim_cfg, ("reward", "clip"), None if args.reward_clip <= 0 else float(args.reward_clip))
    if args.reward_collision_penalty is not None:
        _set_nested(sim_cfg, ("reward", "collision_penalty"), float(args.reward_collision_penalty))
    if args.reward_front_penalty is not None:
        _set_nested(sim_cfg, ("reward", "front_penalty"), float(args.reward_front_penalty))
    if args.reward_side_penalty is not None:
        _set_nested(sim_cfg, ("reward", "side_penalty"), float(args.reward_side_penalty))
    if args.reward_min_clear_penalty is not None:
        _set_nested(sim_cfg, ("reward", "min_clear_penalty"), float(args.reward_min_clear_penalty))
    if args.reward_balance_penalty is not None:
        _set_nested(sim_cfg, ("reward", "balance_penalty"), float(args.reward_balance_penalty))
    if args.reward_reverse_penalty is not None:
        _set_nested(sim_cfg, ("reward", "reverse_penalty"), float(args.reward_reverse_penalty))
    if args.reward_alignment_bonus is not None:
        _set_nested(sim_cfg, ("reward", "alignment_bonus"), float(args.reward_alignment_bonus))
    if args.reward_forward_speed_weight is not None:
        _set_nested(sim_cfg, ("reward", "forward_speed_weight"), float(args.reward_forward_speed_weight))
    if args.reward_front_speed_weight is not None:
        _set_nested(sim_cfg, ("reward", "front_speed_weight"), float(args.reward_front_speed_weight))
    if args.reward_front_cone_deg is not None:
        _set_nested(sim_cfg, ("reward", "front_cone_deg"), float(args.reward_front_cone_deg))

    if args.track_center is not None:
        _set_nested(sim_cfg, ("track", "center"), str(args.track_center))
    if args.track_direction is not None:
        _set_nested(sim_cfg, ("track", "direction"), str(args.track_direction))
    if args.spawn_face_forward is not None:
        _set_nested(sim_cfg, ("track", "spawn_face_forward"), bool(args.spawn_face_forward))
    if args.episode_time_limit_s is not None:
        _set_nested(sim_cfg, ("episode", "time_limit_s"), float(args.episode_time_limit_s))

    return sim_cfg
