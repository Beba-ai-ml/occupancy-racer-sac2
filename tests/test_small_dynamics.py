"""Behavior tests through the same configuration and environment as async training."""

import argparse
from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from src.config import load_yaml
from src.map_loader import MapData
from src.params import build_vehicle_params, build_map_params
from src.racer_env import RacerEnv, build_lidar_angles
from src.sim_config import build_sim_config, register_sim_args
from src.steering import SteeringActuator, SteeringProfile
from src.vehicle import Vehicle, MapParams

ROOT = Path(__file__).resolve().parents[1]
PHYSICS = load_yaml(ROOT / "config/physics_small.yaml")


def profile():
    return build_vehicle_params(PHYSICS).steering_profile


def make_env(full_profile=False, render=False):
    # Fixed spawn in a large free area; walls cannot confound the response tests.
    image = np.full((200, 200), 255, dtype=np.uint8)
    occupied = np.zeros_like(image, dtype=bool)
    spawn = np.zeros_like(occupied)
    spawn[100, 100] = True
    lookat = np.zeros_like(occupied)
    lookat[100, 150] = True
    data = MapData(image, .1, (0, 0, 0), 0, .65, .25, ~occupied, occupied,
                   spawn_mask=spawn, lookat_mask=lookat)
    params = build_vehicle_params(PHYSICS)
    if full_profile:
        parser = argparse.ArgumentParser()
        register_sim_args(parser)
        args = parser.parse_args([])
        vars(args).update(load_yaml(ROOT / "config/config_sac_small.yaml"))
        cfg = build_sim_config(load_yaml(ROOT / args.config), args)
        cfg["lidar"] = {"front_step_deg": args.lidar_front_step_deg, "rear_step_deg": args.lidar_rear_step_deg}
    else:
        params = replace(params, steering_profile=replace(profile(), timing_scale_range=(1, 1)))
        cfg = {"enabled": False, "lidar": {"front_step_deg": .5, "rear_step_deg": 2}}
    env = RacerEnv(data, params, build_map_params(PHYSICS),
                   steer_bins=[-1, 0, 1], accel_bins=[0, 1, 2],
                   stack_frames=4, fps=60, sim_cfg=cfg, render=render)
    return env


class SteeringTests(unittest.TestCase):
    def test_neutral_to_full_150_plus_150_ms(self):
        actuator = SteeringActuator(profile())
        self.assertEqual(actuator.step(1, .150), 0)
        self.assertAlmostEqual(actuator.step(1, .075), .5)
        self.assertAlmostEqual(actuator.step(1, .075), 1)

    def test_full_reversal_150_plus_300_ms(self):
        for side in (-1, 1):
            actuator = SteeringActuator(profile())
            actuator.step(side, .3)
            self.assertAlmostEqual(actuator.step(-side, .15), side)
            self.assertAlmostEqual(actuator.step(-side, .15), 0)
            self.assertAlmostEqual(actuator.step(-side, .15), -side)

    def test_variable_dt_and_interrupted_commands(self):
        # Timed waveform independent of update frequency. Opposite command is
        # issued before the first reaches the servo, so both must remain queued.
        def run(parts):
            actuator = SteeringActuator(profile())
            for command, duration in [(1, .1), (-1, .1), (0, .3)]:
                for _ in range(parts):
                    actuator.step(command, duration / parts)
            return actuator.actual, actuator.target
        self.assertEqual(run(1), (0, 0))
        for parts in (2, 7, 30):
            np.testing.assert_allclose(run(parts), run(1), atol=1e-12)
        actuator = SteeringActuator(profile())
        self.assertEqual(actuator.step(1, .1), 0)
        self.assertAlmostEqual(actuator.step(-1, .1), 1/3)
        self.assertAlmostEqual(actuator.step(-1, .05), 2/3)
        self.assertAlmostEqual(actuator.step(-1, .05), 1/3)

    def test_reset_drops_inflight_commands(self):
        actuator = SteeringActuator(profile())
        actuator.step(1, .1)
        actuator.reset()
        self.assertEqual(actuator.step(0, 1), 0)

    def test_timing_variation_is_per_episode(self):
        actuator = SteeringActuator(profile())
        for scale in (.9, 1, 1.1):
            actuator.reset(scale, scale)
            self.assertEqual(actuator.step(1, .15 * scale), 0)
            self.assertAlmostEqual(actuator.step(1, .15 * scale), 1)

    def test_invalid_profile_rejected(self):
        for field, value in [("delay_s", -1), ("full_travel_s", 0), ("slow_diameter_m", math.nan),
                             ("reference_speed_mps", 0), ("timing_scale_range", (1.1, .9)),
                             ("mid_speed_mps", 0), ("mid_diameter_m", 4),
                             ("fast_diameter_range_m", (1, 2))]:
            with self.assertRaises(ValueError):
                replace(profile(), **{field: value})


class VehicleTests(unittest.TestCase):
    def test_full_circles_measured_from_positions(self):
        p = replace(build_vehicle_params(PHYSICS), friction=0, drag=0)
        # The two user-specified speeds must produce the requested circles.
        for speed, expected in [(1.5, 2.5), (2, 3)]:
            for side in (-1, 1):
                v = Vehicle(p, (0, 0), 10, render_enabled=False)
                v.speed = speed
                for _ in range(300):
                    v.update(1/120, False, False, side, MapParams(0, 0), accel_cmd=0)
                positions = []
                for _ in range(math.ceil(math.pi * expected / speed * 120)):
                    v.update(1/120, False, False, side, MapParams(0, 0), accel_cmd=0)
                    positions.append(tuple(v.position))
                points = np.asarray(positions)
                diameter_x, diameter_y = np.ptp(points, axis=0)
                self.assertAlmostEqual(diameter_x, expected, delta=.01)
                self.assertAlmostEqual(diameter_y, expected, delta=.01)

    def test_physical_cap_keeps_acceleration_request(self):
        for cap in (1, 1.5, 2):
            v = Vehicle(build_vehicle_params(PHYSICS), (0, 0), 10, render_enabled=False)
            v.speed_limit_mps = cap
            for _ in range(300):
                v.update(1/60, False, False, 0, MapParams(0, 0), accel_cmd=2)
                self.assertLessEqual(v.speed, cap)
            self.assertEqual(v.speed, cap)
            self.assertAlmostEqual(v.accel_actual, 2)

    def test_legacy_physics_remains_immediate(self):
        p = build_vehicle_params(load_yaml(ROOT / "config/physics.yaml"))
        v = Vehicle(p, (0, 0), 10, render_enabled=False)
        v.speed = 2
        v.update(1/60, False, False, 1, MapParams(0, 0), accel_cmd=0)
        self.assertIsNone(v.steering)
        self.assertGreater(v.servo_actual, 0)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(20260926)

    def tearDown(self):
        pygame.quit()

    def test_new_config_observation_angles_and_true_speed(self):
        env = make_env(full_profile=True)
        obs = env.reset()
        self.assertEqual(obs.shape, (1820,))
        self.assertEqual(env.lidar_angles_deg, build_lidar_angles(.5, 2))
        self.assertEqual(env.vehicle_params.speed_observation_scale_mps, 2.5)
        for cap in (1, 1.5, 2):
            env.vehicle.speed_limit_mps = cap
            env.vehicle.speed = cap
            obs, *_ = env.step([0, 2])
            self.assertLessEqual(env.vehicle.speed, cap)
            # Last frame: 450 rays, collision, measured speed, servo, IMU x2.
            self.assertAlmostEqual(float(obs[-4]), env.vehicle.speed / 2.5, places=6)

    def test_free_navigation_rewards_both_directions_equally(self):
        env = make_env(full_profile=True)
        env.reset()
        self.assertTrue(env.free_navigation)
        env.vehicle.speed = 1.0
        readings = [(angle, env.lidar_max_range_m, pygame.Vector2()) for angle in env.lidar_angles_deg]
        rewards = []
        for angle in (0, math.pi):
            env.vehicle.angle = angle
            env.vehicle.position = pygame.Vector2(10, 12)
            direction = pygame.Vector2(math.cos(angle), math.sin(angle))
            env.prev_position = env.vehicle.position - direction * .1
            rewards.append(env._compute_reward(readings, False))
        self.assertAlmostEqual(rewards[0], rewards[1])
        # A stationary car receives no forward-progress reward.
        env.prev_position = env.vehicle.position.copy()
        stopped_progress = env._compute_reward(readings, False)
        self.assertGreater(rewards[0], stopped_progress)

    def test_episode_randomization_reset_and_speed_cap(self):
        env = make_env(full_profile=True)
        caps, delays, fast_diameters = set(), set(), set()
        for _ in range(60):
            env.reset()
            caps.add(env.vehicle.speed_limit_mps)
            delays.add(env.vehicle.steering.delay_s)
            steering = env.vehicle_params.steering_profile
            fast_diameters.add(round(steering.fast_diameter_m, 3))
            self.assertGreaterEqual(steering.fast_diameter_m, 2)
            self.assertLessEqual(steering.fast_diameter_m, 3)
            expected_mid = 1.5 + (steering.fast_diameter_m - 1.5) * (2/3)
            self.assertAlmostEqual(2 * steering.turning_radius(1.5), expected_mid)
            self.assertGreaterEqual(env.vehicle.steering.delay_s, .135)
            self.assertLessEqual(env.vehicle.steering.delay_s, .165)
            self.assertEqual(env.vehicle.steering.actual, 0)
            self.assertEqual(len(env.vehicle.steering._pending), 0)
            env.step([1, 2])
        self.assertEqual(caps, {1, 1.5, 2})
        self.assertGreater(len(delays), 1)
        self.assertGreater(len(fast_diameters), 1)

    def test_legacy_zero_speed_delay_keeps_historical_jitter(self):
        env = make_env()
        env.reset()
        env.vehicle_params = replace(env.vehicle_params, steering_profile=None)
        env.sensor_delay_enabled = True
        env.speed_delay_range = (0, 0)
        env._ep_lidar_delay = env._ep_speed_delay = env._ep_imu_delay = 0
        env._speed_obs_history.append(.1)
        env.vehicle.speed = 1.0
        readings = [(angle, 20.0, pygame.Vector2()) for angle in env.lidar_angles_deg]
        # Preserve the previous legacy sample and all three RNG draws. The
        # opt-in profile alone promises an undelayed, true speed observation.
        with patch("src.racer_env.np.random.randint", side_effect=[0, 1, 0]) as rng:
            obs = env._build_observation(readings, False)
        self.assertAlmostEqual(float(obs[451]), .1)
        self.assertEqual(rng.call_count, 3)

    def test_real_env_latency_and_reset(self):
        env = make_env()
        env.reset()
        for _ in range(9):
            env.step([1, 0])
            self.assertAlmostEqual(env.vehicle.servo_actual, 0)
        for _ in range(9):
            env.step([1, 0])
        self.assertAlmostEqual(env.vehicle.servo_actual, env.vehicle_params.max_steer_angle)
        env.step([-1, 0])
        env.reset()
        for _ in range(30):
            env.step([0, 0])
        self.assertEqual(env.vehicle.servo_actual, 0)

    def test_slow_render_does_not_change_simulation_clock(self):
        env = make_env(render=True)
        class SlowClock:
            def tick(self, fps):
                return 250  # Render stalls for 250 ms each frame.
        env.clock = SlowClock()
        env.reset()
        for _ in range(9):
            env.step([1, 0])
        self.assertAlmostEqual(env.episode_time_s, .15)
        self.assertAlmostEqual(env.vehicle.servo_actual, 0)
        for _ in range(9):
            env.step([1, 0])
        self.assertAlmostEqual(env.vehicle.servo_actual, env.vehicle_params.max_steer_angle)


if __name__ == "__main__":
    unittest.main()
