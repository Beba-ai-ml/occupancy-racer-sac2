# Navigation - Soft_Actor_Critic_2

Table of contents for the SAC-based autonomous racing project.

---

## Root

- `run.py` - Entry point; calls `src.main.main()`
- `test_offline.py` - Offline end-to-end test for the SAC inference pipeline (no ROS2)
  - `class MockLaserScan` - Fake ROS2 LaserScan message for testing
  - `def build_scan()` - Creates a simulated laser scan from scenario distances
  - `def build_obs()` - Builds a full observation vector from scan + vehicle state
  - `def run_scenario()` - Runs a named scenario through the inference pipeline
  - `def main()` - Orchestrates all test scenarios and prints results
- `requirements.txt` - Python dependencies (pygame, pyyaml, numpy, torch)
- `README.md` - Project documentation
- `runtime.md` - Runtime configuration guide for deploying on f1tenth robot

---

## config/

Training and physics configuration files.

- `game.yaml` - Master game config: map path, display, RL hyperparameters, training params, async params, full domain randomization settings
- `physics.yaml` - Vehicle physics parameters (acceleration, braking, speed, friction, drag, steering, dimensions) and map surface parameters
- `config_sac_8.yaml` - SSAC training config variant: 8 actors (baseline)
- `config_sac_10.yaml` - SSAC training config variant: 64 actors, large buffer (1.6M), grad_clip + alpha_min
- `config_sac_11.yaml` - SSAC training config variant: 128 actors, Dom_01 only, grad_clip + alpha_min
- `config_sac_12.yaml` - SSAC training config variant: 64 actors, 10 maps, utd_ratio=0.25, batch_size=256
- `config_sac_13.yaml` - SSAC training config variant: latest iteration with tuned hyperparameters

---

## src/

Core source code: environment, agent, training loops, physics.

- `__init__.py` - Package marker (empty)

- `main.py` - Entry point for interactive mode (human or RL control)
  - `def main()` - Parses CLI args (--config, --physics, --map, --control-mode), loads configs, creates Game

- `config.py` - YAML config loading utilities and path resolution
  - `ROOT_DIR` - Project root directory constant
  - `def load_yaml(path)` - Load and parse a YAML file
  - `def resolve_path(path_str)` - Resolve a path relative to ROOT_DIR
  - `def resolve_from(base, path_str)` - Resolve a path relative to a base directory

- `game.py` - Interactive pygame-based game loop with RL or human control (807 lines)
  - `class Game` - Main game class handling rendering, physics, RL training loop
    - `def run()` - Main game loop (events, physics step, rendering, RL updates)
    - `def _compute_lidar()` - Cast LIDAR rays on the occupancy grid
    - `def _vehicle_collision()` - Check vehicle-wall collision via occupancy mask
    - `def _build_observation()` - Assemble observation vector (lidar + speed + steer + collision)
    - `def _compute_reward()` - Compute step reward (alignment, speed, clearance, collision)
    - `def _reset_episode()` - Reset vehicle to a random spawn position
    - `def _stuck_penalty()` - Penalize the agent for being stuck
    - `def _backward_penalty()` - Penalize the agent for driving backwards
    - `def _clockwise_alignment()` - Compute alignment score relative to track direction
    - `def _random_spawn()` - Pick a random spawn point from the spawn mask or free mask
  - `def _parse_hidden_sizes()` - Parse hidden layer sizes from config string
  - `def _parse_target_entropy()` - Parse target entropy from config string
  - `def _resolve_accel_range()` - Resolve acceleration range from config
  - `def _resolve_checkpoint_path()` - Resolve checkpoint file path
  - `def create_map_surface()` - Create pygame surface from map data with zone overlays

- `map_loader.py` - PGM occupancy grid map loading with optional zone overlays (153 lines)
  - `class MapData` - Frozen dataclass holding map image, metadata, and zone masks
  - `DEFAULT_MAP_META` - Default map metadata values
  - `def read_pgm(path)` - Read P5 (binary) or P2 (ASCII) PGM files
  - `def _map_data_from_meta(image, meta)` - Build MapData from image and metadata dict
  - `def _load_zones_png(map_pgm_path, expected_shape)` - Load zone overlay (R=kill, G=spawn, B=lookat)
  - `def load_map(map_path)` - Load map from YAML or PGM path with automatic zone detection

- `params.py` - Builder functions for vehicle and map parameter dataclasses (37 lines)
  - `def build_vehicle_params(physics_cfg)` - Build VehicleParams from physics config dict
  - `def build_map_params(physics_cfg)` - Build MapParams from physics config dict

- `racer_env.py` - Core racing environment for RL training (1379 lines)
  - `class _Obstacle` - Dataclass for static/dynamic obstacles (position, size, velocity)
  - `class RacerEnv` - Full RL environment with step/reset interface
    - `def reset()` - Reset environment, respawn vehicle, clear obstacles
    - `def step(action)` - Execute one environment step (physics, collision, reward, obs)
    - `def _compute_lidar()` - Cast 27 LIDAR rays over 210-degree arc
    - `def _vehicle_collision()` - Check vehicle-wall collision
    - `def _build_observation()` - Build observation vector with frame stacking
    - `def _compute_reward()` - Compute shaped reward (alignment, speed, clearance, collision)
    - `def _apply_domain_randomization()` - Randomize physics and observation parameters
    - `def _prepare_action()` - Apply action noise, delays, and repeat
    - `def _spawn_obstacles()` - Spawn static obstacles at episode start
    - `def _spawn_dynamic_obstacle()` - Spawn a moving obstacle during episode
    - `def _random_spawn()` - Random spawn from spawn_mask or free_mask
    - `def _clockwise_alignment()` - Track-direction alignment score
    - `def render()` - Pygame-based rendering
  - `def create_map_surface()` - Create pygame surface with zone visualization
  - `def _build_lidar_angles()` - Generate 27 LIDAR ray angles (dense center, sparse edges)
  - LIDAR constants: `LIDAR_CENTER_DEG=90`, `LIDAR_TOTAL_ARC_DEG=210`, `LIDAR_DENSE_HALF_DEG=45`, `LIDAR_DENSE_STEP_DEG=5`, `LIDAR_SPARSE_STEP_DEG=15`, `LIDAR_FRONT_CONE_DEG=20`

- `rl_agent.py` - SAC algorithm implementation (479 lines)
  - `class GaussianPolicy` - Actor network: backbone MLP + mean/log_std heads, tanh squashing
    - `def forward(state)` - Compute mean and log_std
    - `def sample(state)` - Sample action with reparameterization trick
    - `def deterministic(state)` - Return deterministic (mean) action
  - `class QNetwork` - Critic MLP (state+action -> Q-value)
  - `class ReplayBuffer` - Numpy-based circular experience replay buffer
    - `def push(state, action, reward, next_state, done)` - Store transition
    - `def sample(batch_size)` - Sample random batch as tensors
  - `class SACAgent` - Full SAC with twin critics, target networks, auto entropy tuning
    - `def select_action(state)` - Sample action for single state
    - `def select_action_batch(states)` - Sample actions for batch of states
    - `def step(state, action, reward, next_state, done)` - Store + learn
    - `def learn()` - SAC update: critic loss, actor loss, alpha loss, soft target update
    - `def _soft_update()` - Polyak averaging for target networks
    - `def save_checkpoint(path)` - Save full training state (v1 format)
    - `def load_checkpoint(path)` - Restore from checkpoint with RNG states
  - `def _build_mlp()` - Helper to construct Q-network MLP layers
  - Constants: `LOG_STD_MIN=-20.0`, `LOG_STD_MAX=2.0`

- `sim_config.py` - Domain randomization CLI arguments and config builder (412 lines)
  - `def register_sim_args(parser)` - Register 80+ CLI arguments for all DR categories
  - `def build_sim_config(game_cfg, args)` - Merge YAML config with CLI overrides
  - DR categories: physics, surface, observation_noise, control, dt_jitter, action_noise, perturb, observation_delay, action_repeat, obstacles (static + dynamic), reward shaping, track guidance, episode limits

- `train.py` - Synchronous multi-env training using VecRacerEnv (417 lines)
  - `def train()` - Main synchronous training loop: parse args, create SACAgent + VecRacerEnv, collect transitions, learn, log, checkpoint

- `train_ssac.py` - Asynchronous multi-process training (1081 lines) -- primary training entry point
  - `def train_ssac()` - Main orchestrator: launch actor workers, central learner loop, checkpoint management
  - `def _actor_worker()` - Actor subprocess: collect transitions with local policy copy, send to learner
  - `def _cpu_state_dict()` - Move state dict to CPU for inter-process transfer
  - `def _bump_nofile_limit()` - Increase OS file descriptor limit for multiprocessing
  - `def _drain_queue()` - Batch-drain a multiprocessing queue (up to 64 items)
  - `def _parse_map_pool()` - Parse map pool string into list of map paths
  - `def _save_checkpoint_atomic()` - Atomic checkpoint save with tmp+rename
  - `def _save_checkpoint_backup()` - Create timestamped backup of checkpoint
  - `def _extract_config_file()` - Extract config file path from CLI args
  - `def _resolve_config_file()` - Resolve config file with fallback logic
  - `def _resolve_map_override()` - Resolve map override from CLI or config
  - `def _next_session_id()` - Generate next session ID from config file name
  - `def _validate_config_keys()` - Validate config YAML keys against known schema

- `vec_env.py` - Vectorized environment using multiprocessing (191 lines)
  - `class VecRacerEnv` - Manages N worker processes each running a RacerEnv
    - `def reset()` - Reset all environments
    - `def reset_at(index)` - Reset a single environment by index
    - `def step(actions)` - Step all environments in parallel
    - `def set_render(index, enabled)` - Enable/disable rendering for one env
    - `def close()` - Shut down all worker processes
  - `def _worker()` - Subprocess loop handling reset/step/render/close commands

- `vehicle.py` - Vehicle physics model with Ackermann steering (249 lines)
  - `class VehicleParams` - Dataclass for vehicle physical parameters
  - `class MapParams` - Dataclass for map surface parameters (friction, drag)
  - `class Vehicle` - Vehicle with position, heading, speed, steering
    - `def update(dt, accel_input, steer_input, map_params)` - Physics step (friction, drag, Ackermann yaw rate)
    - `def draw(surface, camera_x, camera_y, zoom)` - Render vehicle sprite on pygame surface
    - `def _create_texture()` - Generate detailed car sprite with body, windshield, wheels

---

## sac_driver/

Sim-to-real bridge: inference and ROS2 integration utilities.

- `__init__.py` - Package docstring: "SAC driver utilities (offline staging)."

- `control_mapper.py` - Maps policy actions to vehicle commands (111 lines)
  - `class ControlMapper` - Configurable action-to-command mapper
    - `def map_to_ackermann()` - Convert (steer, accel) to Ackermann dict (steering_angle, speed, acceleration)
    - `def map_to_twist()` - Convert (steer, accel) to Twist dict (linear_x, angular_z)

- `inference_engine.py` - Inference wrapper for trained SAC policy (58 lines)
  - `class InferenceEngine` - Load checkpoint, run deterministic inference
    - `def get_action(state)` - Return (steer, accel) from state vector

- `lidar_converter.py` - ROS2 LaserScan to training-aligned LIDAR converter (83 lines)
  - `class LidarConverter` - Converts real LIDAR scans to match training format
    - `def convert(scan_msg)` - Interpolate/nearest-neighbor, normalize to [0,1]

- `policy_loader.py` - Standalone policy loader for inference without full agent (192 lines)
  - `class GaussianPolicy` - Standalone copy of the policy network
    - `def get_mean_action(state)` - Deterministic action from policy mean
  - `def load_policy(pth_path, device)` - Auto-infer architecture from state dict, load weights
  - `def export_policy_weights(pth_path, output_path)` - Export policy weights to separate file
  - `def _infer_arch_from_state_dict()` - Infer hidden sizes and input/output dims from weight shapes

- `state_builder.py` - Builds stacked state vectors matching training layout (81 lines)
  - `class StateBuilder` - Maintains frame stack for temporal information
    - `def reset(first_obs)` - Initialize stack with first observation
    - `def update(lidar, speed, servo, collision)` - Push new frame, return stacked state

---

## tools/

Standalone GUI utilities for map preparation.

- `map_zone_painter.py` - Tkinter GUI for painting spawn/kill/lookat zones on PGM maps (575 lines)
  - `class ZonePainter` - Canvas-based zone editor with zoom, pan, undo, layer toggling
    - Saves `{map_name}_zones.png` (RGB channels: R=kill, G=spawn, B=lookat)

- `pgm_outline_ui.py` - Tkinter GUI for batch-adding outlines to PGM map files (203 lines)
  - `class App` - Batch outline processor with prefix/range/folder configuration
  - `def add_outline(image)` - Add dark outline around occupied regions
  - `def write_pgm(path, image)` - Write P5 PGM file
  - `def _dilate(mask, radius)` - Morphological dilation for outline generation

---

Last mapped: 2026-02-12
