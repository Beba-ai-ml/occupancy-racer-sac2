# Occupancy Racer SAC v2 -- Technical Documentation

## Project Overview

Occupancy Racer SAC v2 is a reinforcement learning system that trains an autonomous
racing agent to navigate 2D occupancy-grid maps using a Soft Actor-Critic (SAC)
algorithm. The agent receives LiDAR scans (up to 450 rays at 360 degrees), speed,
servo position, and IMU readings, then outputs continuous steering and acceleration
commands. The system targets sim-to-real transfer onto a 1/10-scale RC car equipped
with a 2D LiDAR, VESC motor controller, and onboard compute.

Key design choices: asynchronous multi-actor training with a centralized learner
(up to 64 actor processes); extensive domain randomization (14 sim-to-real features
covering physics, sensors, actuators, and environment); action repeat at 8 Hz to
match real LiDAR scan rate; zone-based spawn/lookat system for multi-map curriculum;
and a standalone `sac_driver/` package for ROS2-free deployment inference.

## Architecture

```
+------------------------------------------------------------------+
|                        train_ssac.py                              |
|                                                                   |
|  +------------------+    transitions     +--------------------+   |
|  | Actor Process 0  |  ------------->    |                    |   |
|  | Actor Process 1  |  ------------->    |   Learner Thread   |   |
|  | ...              |  ------------->    |   (SACAgent.learn) |   |
|  | Actor Process N  |  ------------->    |                    |   |
|  +------------------+    <-----------    +--------------------+   |
|         ^                 policy weights         |                |
|         |                                        |                |
|  +------+--------+                     +--------+--------+       |
|  | RacerEnv       |                     | ReplayBuffer    |       |
|  | + Vehicle       |                     | (stratified)    |       |
|  | + MapData       |                     +-----------------+       |
|  | + DR + Obstacles|                                              |
|  +-----------------+                                              |
+------------------------------------------------------------------+

+--------------------------+
| sac_driver/ (inference)  |
|  PolicyLoader            |
|  InferenceEngine         |
|  LidarConverter          |      ROS2 / real hardware
|  StateBuilder            |  <-- LaserScan, Odometry
|  ControlMapper           |  --> AckermannDrive / Twist
+--------------------------+
```

**Data flow:** Each actor process runs a `RacerEnv` instance and a local copy of
`GaussianPolicy`. It collects `(state, action, reward, next_state, done)` transitions
and pushes them to a shared queue. The learner drains the queue into a `ReplayBuffer`,
runs SAC gradient updates, and periodically broadcasts updated policy weights back to
all actors via per-actor weight queues.

## Module Reference

### src/rl_agent.py

#### class GaussianPolicy(nn.Module)

Stochastic policy network with squashed Gaussian output.

```python
def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Iterable[int],
             action_scale: np.ndarray, action_bias: np.ndarray) -> None
```

Architecture: `backbone` (Linear+ReLU layers) followed by parallel `mean_layer` and
`log_std_layer` heads. Log-std is clamped to [-20, 2]. Stores `action_scale` and
`action_bias` as registered buffers.

- `forward(state) -> (mean, log_std)` -- raw network outputs.
- `sample(state) -> (action, log_prob, mean_action)` -- reparameterized sample with
  tanh squashing and corrected log-probability.
- `deterministic(state) -> action` -- tanh(mean) scaled and biased.

#### class QNetwork(nn.Module)

Twin Q-function critic with LayerNorm.

```python
def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None
```

Architecture: `[state, action]` concatenated input through Linear+LayerNorm+ReLU
blocks, final Linear to scalar Q-value.

- `forward(state, action) -> Q-value`

#### class ReplayBuffer

Pre-allocated NumPy circular buffer with optional stratified sampling by map ID.

```python
def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None
```

- `add(state, action, reward, next_state, done, map_id=-1)` -- insert transition.
- `sample(batch_size, stratified=False) -> (states, actions, rewards, next_states, dones)`
- `sample_into(batch_size, np_s, np_a, np_r, np_ns, np_d, stratified=False)` -- zero-copy
  sample into pre-allocated pinned-memory arrays (CUDA fast path).
- `_stratified_indices(batch_size)` -- equal samples per map ID with cached index lookup,
  rebuilds cache every 1000 insertions.

#### class SACAgent

Full SAC algorithm with automatic entropy tuning, twin critics, and torch.compile.

```python
def __init__(self, state_dim, action_dim, action_scale, action_bias,
             policy_lr, q_lr, alpha_lr, gamma, tau, batch_size, memory_size,
             target_entropy, init_alpha, start_steps, learn_after, update_every,
             updates_per_step, hidden_sizes, grad_clip=0.0, alpha_min=0.0,
             alpha_max=0.0, device=None) -> None
```

Key methods:

- `select_action(state, deterministic=False) -> np.ndarray` -- single action; returns
  random during `start_steps`.
- `select_action_batch(states, deterministic=False) -> list[np.ndarray]` -- vectorized.
- `step(state, action, reward, next_state, done, map_id=-1)` -- store transition and
  optionally trigger learning based on `update_every` / `updates_per_step`.
- `learn(stratified=False) -> float` -- one gradient step: critic loss (MSE on
  clipped double-Q target), policy loss (entropy-regularized), alpha loss (dual
  gradient descent), soft target update. Returns Q-loss. Uses pinned-memory transfer
  on CUDA.
- `save_checkpoint(path, meta=None)` -- saves format `sac_checkpoint_v1` with all
  network weights, optimizer states, RNG states, step counters, and metadata.
- `load_checkpoint(path, reset_optimizers=False) -> dict` -- restores full state;
  handles `torch.compile` prefix mismatches (`_orig_mod.`); optionally resets
  optimizer states for fine-tuning.

### src/racer_env.py

#### class RacerEnv

2D occupancy-grid racing environment with LiDAR simulation, physics, domain
randomization, obstacles, and zone-based spawning.

```python
def __init__(self, map_data: MapData, vehicle_params: VehicleParams,
             map_params: MapParams, steer_bins, accel_bins, fps=60, render=False,
             stack_frames=1, episode_offset=0, sim_cfg=None) -> None
```

**Observation space:** flat float32 vector of size `(num_lidar_rays + 5) * stack_frames`.
Per frame: `[lidar_0..lidar_N, collision_flag, speed_norm, servo_norm, linear_accel_norm, angular_vel_norm]`.
LiDAR values normalized to [0, 1] by `max_range` (20 m). Speed normalized by max_speed
in km/h. IMU values normalized by configurable max ranges.

**Action space:** continuous 2D vector `[steering, acceleration]`. Steering in [-1, 1],
acceleration in [accel_min, accel_max] (typically [-2, 2] m/s^2).

Key methods:

- `reset() -> np.ndarray` -- randomize physics/surface/noise params (DR), pick spawn
  position from zone-aware positions, reset vehicle, return initial stacked observation.
- `step(action) -> (obs, reward, done, distance_delta, death_reason)` -- apply action
  pipeline (repeat, noise, delay, rate-limit), step physics, compute LiDAR, check
  collision, compute reward, apply perturbations/wind, update obstacles.
- `_compute_reward(readings, collision) -> float` -- see reward components below.
- `_cast_ray(origin, angle) -> (distance_m, hit_point)` -- DDA grid traversal raycaster.
- `build_lidar_angles(front_step_deg, rear_step_deg) -> list[float]` -- module-level
  function; 360-degree layout with configurable front/rear density.
- `_build_legacy_lidar_angles() -> list[float]` -- classic 27-ray, 210-degree arc.

**Reward function components** (in `_compute_reward`):

| Component | Weight/Value | Description |
|-----------|-------------|-------------|
| Collision | -20.0 | Terminal penalty on wall/obstacle hit |
| Forward speed | 0.6 | `forward_speed_norm` (positive speed / max_speed) |
| Front+speed | 0.2 | `forward_speed_norm * front_clearance` |
| Distance progress (S9) | 0.1 | Dot product of displacement onto track heading |
| Alignment bonus | 0.02 | Clockwise/raceline alignment (when S9 disabled) |
| Front penalty | 0.06 | `speed * (1 - front_clearance)` |
| Side penalty | 0.02 | `1 - avg(left, right)` |
| Min clearance | 0.04 | `1 - min(all_distances)` |
| Balance penalty | 0.015 | `abs(left - right)` |
| Reverse penalty | 0.03 | Negative speed normalized |
| Stuck penalty | exponential | Grows with time spent below 0.1 m/s |
| Soft collision (S8) | -5.0 | Light contact penalty (up to N contacts before death) |

Final reward is scaled by `reward_scale` (0.8) and clipped to [-20, 20].

### src/train_ssac.py

Entry point: `train_ssac()` function, invoked via `python -m src.train_ssac`.

**Flow:**
1. Parse CLI args + config YAML (config file overrides game.yaml defaults).
2. Build `SACAgent` with computed `state_dim = (num_lidar_rays + 5) * stack_frames`.
3. Attempt checkpoint resume (session dir, legacy path, or explicit `--load-from`).
4. Spawn N actor processes via `multiprocessing.spawn` context.
5. Broadcast initial policy weights to all actors.
6. Enter learner loop.

**Actor process** (`_actor_worker`):
- Runs its own `RacerEnv` with a local `GaussianPolicy` (CPU inference).
- On each step: check for weight updates (non-blocking), query policy when
  `wants_new_action` (action-repeat window expired), step environment, push
  one transition per decision window with accumulated reward.
- Handles map rotation: after `map_switch_start_episode`, randomly samples from
  `map_pool` every `map_switch_every` episodes.
- Reports episode stats (distance, reward, death reason, FPS) via stats queue.

**Learner loop** (main process):
- Drains transition queue into `ReplayBuffer` with UTD ratio control.
- Calls `agent.learn(stratified=stratified_sampling)` with appropriate frequency.
- Every `sync_every` transitions: broadcasts new policy weights to all actors.
- Every `save_every` episodes: atomic checkpoint save with backup rotation.
- Logs per-episode stats to CSV via threaded `LogBuffer`.
- Resume warmup: delays learning for ~37500 * (num_actors/4) steps after checkpoint
  load to refill the replay buffer with on-policy data.

**Config validation:** `_validate_config_keys` warns on unknown YAML keys not
matching any argparse destination.

### src/vehicle.py

#### class Vehicle

2D bicycle-model vehicle with sim-to-real fidelity features.

```python
def __init__(self, params: VehicleParams, position, pixels_per_meter,
             angle=0.0, render_enabled=True, servo_tau=0.050, motor_tau=0.100,
             slip_speed_threshold=4.0, min_grip_factor=0.4,
             grip_reduction_rate=0.3, yaw_inertia_tau=0.100) -> None
```

Physics model in `update(dt, throttle, brake, steer, map_params, accel_cmd)`:

- **S1: Servo lag** -- first-order low-pass filter on steering angle (`servo_tau`).
- **S1: Motor lag** -- first-order low-pass filter on acceleration (`motor_tau`).
- **Speed-dependent steering** -- max steer angle reduces linearly with speed ratio.
- **S5: Quadratic drag** -- deceleration uses `friction + drag * v^2`.
- **Ackermann steering** -- yaw rate = `(speed / wheelbase) * tan(steer_actual)`.
- **S4: Tire slip** -- grip factor reduces above `slip_speed_threshold`.
- **S11: Yaw inertia** -- low-pass filter on yaw rate (`yaw_inertia_tau`).

`VehicleParams` dataclass: acceleration, brake_deceleration, reverse_acceleration,
max_speed, max_reverse_speed, friction, drag, max_steer_angle, wheelbase, length, width.

`MapParams` dataclass: surface_friction, surface_drag.

### src/map_loader.py

#### MapData dataclass

```python
@dataclass(frozen=True)
class MapData:
    image: np.ndarray           # uint8 grayscale
    resolution: float           # meters per pixel
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    free_mask: np.ndarray       # bool
    occupied_mask: np.ndarray   # bool
    spawn_mask: np.ndarray | None
    kill_mask: np.ndarray | None
    lookat_mask: np.ndarray | None
    raceline_mask: np.ndarray | None
    spawn_zones: np.ndarray | None   # uint8: 0=none, 1-3=zone id
    lookat_zones: np.ndarray | None  # uint8: 0=none, 1-3=zone id
```

Key functions:

- `load_map(map_path: Path) -> MapData` -- loads PGM image + YAML metadata + optional
  `_zones.png` companion file. Supports both `.yaml` and bare `.pgm` paths.
- `read_pgm(path) -> np.ndarray` -- P5/P2 PGM parser with maxval normalization.
- `decode_zone_channel(channel) -> np.ndarray` -- PNG channel to zone IDs (1-3)
  using threshold midpoints: >=213 -> zone 1, >=128 -> zone 2, >=43 -> zone 3.
- `encode_zone_channel(zones) -> np.ndarray` -- zone IDs to PNG values:
  zone 1 -> 255, zone 2 -> 170, zone 3 -> 85.

Zone PNG format (RGBA): R=kill mask, G=spawn zones, B=lookat zones, A=raceline mask.

### src/sim_config.py

- `register_sim_args(parser)` -- adds ~80 argparse arguments for domain randomization,
  obstacles, reward shaping, track guidance, and episode limits.
- `build_sim_config(game_cfg, args) -> dict` -- merges `game_cfg["sim_randomization"]`
  with CLI overrides into a nested dict consumed by `RacerEnv`.

### sac_driver/

Standalone inference package for deploying trained policies. No dependency on
training code or pygame.

#### InferenceEngine

```python
class InferenceEngine:
    def __init__(self, policy_path, device="cpu", *, weights_only=False,
                 action_scale=None, action_bias=None) -> None
    def get_action(self, state: np.ndarray) -> Tuple[float, float]
```

Loads checkpoint, runs deterministic inference, returns `(steering, acceleration)`.
Tracks `last_inference_ms` for latency monitoring.

#### PolicyLoader

- `load_policy(pth_path, device, weights_only) -> GaussianPolicy` -- auto-infers
  architecture (state_dim, action_dim, hidden_sizes, action_scale, action_bias) from
  state dict keys. Handles full checkpoints and bare policy weights.
- `export_policy_weights(pth_path, output_path)` -- extracts policy-only weights
  to a smaller file.

#### LidarConverter

```python
@dataclass
class LidarConverter:
    target_angles_deg: Sequence[float]
    max_range_m: float
    angle_offset_deg: float = 0.0
    angle_direction: float = 1.0
    use_interpolation: bool = True
```

`convert(scan_msg) -> np.ndarray` -- re-samples a ROS2 `LaserScan` message onto the
training LiDAR angle layout via linear interpolation. Output normalized to [0, 1].
Accepts both ROS2 message objects and dicts.

#### StateBuilder

```python
class StateBuilder:
    def __init__(self, stack_frames, lidar_dim, max_speed_mps, use_imu=False) -> None
```

- `single_obs_dim` -> `lidar_dim + 5`
- `state_dim` -> `single_obs_dim * stack_frames`
- `reset(first_obs) -> np.ndarray` -- initialize frame stack.
- `update(lidar_normalized, speed_mps, servo_normalized, collision_flag,
         linear_accel, angular_vel) -> np.ndarray` -- build observation and
  return concatenated stacked state.

#### ControlMapper

```python
@dataclass
class ControlMapper:
    max_steering_angle_deg: float
    max_speed_mps: float
    max_accel_mps2: float
    speed_limit_mps: Optional[float] = None
    steer_rate_limit_deg_s: Optional[float] = None
    accel_rate_limit_mps2: Optional[float] = None
    safe_mode: bool = False
    ...
```

- `map_to_ackermann(steering_raw, accel_raw, current_speed, dt) -> dict` -- returns
  `{steering_angle, speed, acceleration}` with rate limiting and safety scaling.
- `map_to_twist(steering_raw, accel_raw, current_speed, dt) -> dict` -- returns
  `{linear_x, angular_z}` for differential-drive or Twist-based interfaces.

### tools/map_zone_painter.py

Tkinter GUI tool for painting zone overlays on map PGM files.

#### class ZonePainter

Paint modes: Kill (red, R channel), Spawn (green, G channel with zone ID 1-3),
LookAt (yellow/blue, B channel with zone ID 1-3), Raceline (purple, A channel),
Eraser (clears all layers).

Features: scroll zoom, middle-click pan, Ctrl+LMB pan, adjustable brush size (1-60),
per-layer visibility toggles, undo stack (200 levels), per-zone color coding.

Saves `{map_name}_zones.png` as RGBA PNG companion to the PGM file.

## Configuration Reference

Parameters from `config/config_sac_20.yaml`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| config | str | config/game.yaml | Path to game config YAML |
| physics | str | config/physics.yaml | Path to physics config YAML |
| map | str | R_01 | Initial map name or path |
| map_pool | list | [] | Maps to rotate through |
| map_switch_start_episode | int | 1200 | Start map rotation after N episodes |
| map_switch_every | int | 15 | Episodes per map before re-sampling |
| track_center | str | free | Center for track direction (map/free) |
| track_direction | str | clockwise | Preferred travel direction |
| spawn_face_forward | bool | true | Flip spawn heading if backward is clearer |
| num_actors | int | 32 | Number of parallel actor processes |
| fps | int | 60 | Simulated frames per second |
| sync_every | int | 800 | Weight sync every N transitions |
| queue_size | int | 32000 | Max transitions in shared queue |
| save_every | int | 250 | Checkpoint every N episodes |
| session_id | str | SAC_P_13 | Session identifier |
| save_dir | str | runs | Checkpoint/log directory |
| load_from | str | null | Checkpoint to resume from |
| no_resume | bool | false | Disable auto-resume |
| max_episodes | int | 0 | Stop after N episodes (0=unlimited) |
| episode_time_limit_s | float | 90 | Episode timeout in seconds |
| device | str | cuda | Training device |
| stack_frames | int | 4 | Observation frame stacking |
| gamma | float | 0.99 | Discount factor |
| tau | float | 0.005 | Target network smoothing |
| policy_lr | float | 0.0003 | Actor learning rate |
| q_lr | float | 0.0003 | Critic learning rate |
| alpha_lr | float | 0.0001 | Entropy coefficient learning rate |
| init_alpha | float | 0.3 | Initial entropy coefficient |
| target_entropy | float | -1.5 | Target entropy (or "auto" for -action_dim) |
| batch_size | int | 256 | Mini-batch size |
| memory_size | int | 1000000 | Replay buffer capacity |
| start_steps | int | 0 | Random actions before policy |
| learn_after | int | 5000 | Minimum steps before learning |
| update_every | int | 1 | Learning frequency in steps |
| updates_per_step | int | 1 | Gradient updates per learning step |
| hidden_sizes | list | [512, 512, 256] | Network hidden layer sizes |
| grad_clip | float | 0.5 | Gradient norm clipping |
| alpha_min | float | 0.05 | Minimum alpha floor |
| alpha_max | float | 0.3 | Maximum alpha ceiling |
| utd_ratio | float | 2.0 | Update-to-data ratio |
| stratified_sampling | bool | false | Stratified replay by map ID |
| reward_scale | float | 0.8 | Reward scaling factor |
| reward_clip | float | 20 | Reward clipping magnitude |
| reward_collision_penalty | float | -20 | Collision terminal reward |
| reward_alignment_bonus | float | 0.02 | Track alignment bonus |
| reward_forward_speed_weight | float | 0.6 | Forward speed weight |
| reward_front_speed_weight | float | 0.2 | Front clearance + speed weight |
| reward_front_penalty | float | 0.06 | Low front clearance penalty |
| reward_side_penalty | float | 0.02 | Low side clearance penalty |
| reward_min_clear_penalty | float | 0.04 | Minimum clearance penalty |
| reward_balance_penalty | float | 0.015 | Left/right imbalance penalty |
| reward_reverse_penalty | float | 0.03 | Reverse speed penalty |
| reward_front_cone_deg | float | 20 | Front cone half-angle (degrees) |
| lidar_front_step_deg | float | 0.5 | Front hemisphere angular step |
| lidar_rear_step_deg | float | 2.0 | Rear hemisphere angular step |
| action_repeat | int | 8 | Steps per policy decision (~8 Hz) |

**Domain randomization parameters** (prefixed `dr_`): physics randomization (accel,
brake, reverse, max_speed, friction, drag, wheelbase, max_steer -- all with scale
ranges), surface randomization (friction scale, drag range), observation noise
(LiDAR std/dropout/spike, speed/servo noise), control (delay steps, rate limits),
dt jitter, action noise (steer/accel scale + bias), perturbations (yaw rate + speed
sigma). All parameters with defaults are listed in `config/config_sac_20.yaml`.

**Obstacle parameters** (prefixed `obst_`): enable/disable obstacles, episode
probability (0.3), start after episode 5000, max 1 static + 2 dynamic (3 total),
spawn rate 0.25/s. See `config_sac_20.yaml` for full obstacle parameter list.

## Domain Randomization Reference

Sim-to-real (S) features implemented in `racer_env.py` and `vehicle.py`:

| ID | Feature | Location | Description |
|----|---------|----------|-------------|
| S1 | Actuator lag | vehicle.py | First-order servo + motor lag filters |
| S4 | Tire slip | vehicle.py | Grip reduction above speed threshold |
| S5 | Quadratic drag | vehicle.py | Drag proportional to v^2 |
| S6 | LiDAR beam divergence | racer_env.py | Distance-dependent noise + ego-motion blur |
| S7 | IMU + encoder noise | racer_env.py | Linear accel + yaw rate in obs; VESC noise |
| S8 | Soft collision | racer_env.py | Allow N light contacts before termination |
| S9 | Distance progress | racer_env.py | Forward progress reward along track heading |
| S10 | Async sensor delay | racer_env.py | Per-sensor stale frames (LiDAR, speed, IMU) |
| S11 | Yaw inertia | vehicle.py | Low-pass filter on yaw rate |
| S12 | Wind perturbations | racer_env.py | OU-process lateral wind gusts |
| S13 | Continuous DR | racer_env.py | Battery sag + friction cycle within episode |
| S14 | Thermal drift | racer_env.py | Ornstein-Uhlenbeck sensor bias drift |

Per-episode DR: physics params scaled by uniform ranges, surface friction/drag,
observation noise (LiDAR Gaussian/dropout/spike, speed, servo), action noise
(scale + bias on steer/accel), control delay (0-1 steps + rate limiting),
dt jitter ([0.85, 1.15]), random yaw/speed perturbations.

## Checkpoint Format

Saved by `SACAgent.save_checkpoint()` as a PyTorch `.pth` file via `torch.save()`.

```python
{
    "format": "sac_checkpoint_v1",
    "policy": OrderedDict,           # GaussianPolicy state_dict (prefix-stripped)
    "critic1": OrderedDict,          # QNetwork 1 state_dict
    "critic2": OrderedDict,          # QNetwork 2 state_dict
    "critic1_target": OrderedDict,   # Target Q1 state_dict
    "critic2_target": OrderedDict,   # Target Q2 state_dict
    "policy_optimizer": dict,        # Adam state for policy
    "critic_optimizer": dict,        # Adam state for both critics
    "log_alpha": float,              # Current log(alpha) value
    "alpha_optimizer": dict,         # Adam state for alpha
    "total_steps": int,              # Cumulative env steps
    "total_updates": int,            # Cumulative gradient updates
    "rng": {                         # RNG states for reproducibility
        "python": tuple,
        "numpy": dict,
        "torch": ByteTensor,
        "torch_cuda": list[ByteTensor] | None,
    },
    "meta": {                        # Training metadata
        "episodes_trained_total": int,
        "distance_history": list[float],  # last 100 episode distances
        "session_id": str,
        ...
    },
}
```

Resume logic in `train_ssac()`:
1. Check `--load-from` path (explicit).
2. Check `runs/session_{id}/session_{id}.pth` (session directory).
3. Check `runs/session_{id}.pth` (legacy flat path).
4. For each: try primary, then `.bak` backup on failure.
5. After load: warmup period (~37500 * num_actors/4 steps) before learning resumes.

`torch.compile` `_orig_mod.` prefix is handled transparently during load via
`_adapt_keys()`.
