# Occupancy Racer - Soft Actor-Critic v2

2D racing simulator that trains an autonomous vehicle using the Soft Actor-Critic (SAC) reinforcement learning algorithm on occupancy grid maps. Designed for sim-to-real transfer to an f1tenth robot platform.

![Training curve - mean reward over 100 episodes](photos/run22_mean100.png)

---

## Overview

The project trains a neural network policy to race around tracks represented as PGM occupancy grids. The agent perceives the world through 27 simulated LIDAR rays and outputs steering and acceleration commands. Training uses the SAC algorithm with twin critics, automatic entropy tuning, and extensive domain randomization to enable transfer to physical hardware.

---

## Architecture

```
run.py / src/main.py          CLI entry points
        |
  src/game.py                 Interactive mode (pygame, human or RL control)
        |
  src/train.py                Synchronous multi-env training (VecRacerEnv)
  src/train_ssac.py           Asynchronous multi-process training (primary)
        |
  src/rl_agent.py             SAC agent (GaussianPolicy, QNetwork, ReplayBuffer)
  src/racer_env.py            RL environment (LIDAR, physics, reward, obstacles)
  src/vehicle.py              Ackermann vehicle physics model
  src/vec_env.py              Vectorized environment (multiprocessing)
  src/sim_config.py           Domain randomization configuration
  src/map_loader.py           PGM map + zone overlay loading
  src/params.py               Parameter builders
  src/config.py               YAML config utilities
        |
  sac_driver/                 Sim-to-real bridge (inference, ROS2 integration)
    inference_engine.py        Load policy, run deterministic inference
    policy_loader.py           Standalone policy loader (auto-infer architecture)
    lidar_converter.py         ROS2 LaserScan -> training-format LIDAR
    state_builder.py           Frame-stacked state vector builder
    control_mapper.py          Policy actions -> Ackermann/Twist commands
```

### Data Flow (Training)

1. **Environment** (`racer_env.py`): Vehicle spawns on a free pixel of the occupancy map. LIDAR rays are cast on the grid. Domain randomization is applied.
2. **Observation**: 27 normalized LIDAR ranges + speed + steering angle + collision flag = 30 values per frame. With 4-frame stacking = 120-dimensional state vector.
3. **Agent** (`rl_agent.py`): GaussianPolicy outputs 2D action (steering, acceleration) via tanh squashing. Twin Q-networks evaluate state-action pairs.
4. **Reward**: Shaped reward combining alignment with track direction, forward speed, wall clearance, and collision penalty (-20).
5. **Learning**: SAC update with critic loss (MSE on Bellman target), actor loss (max Q - alpha * log_prob), alpha loss (entropy constraint), and soft target update (Polyak averaging).

### Data Flow (Inference / Sim-to-Real)

1. **ROS2 LaserScan** -> `LidarConverter.convert()` -> normalized 27-ray vector
2. Normalized LIDAR + vehicle speed + servo position + collision flag -> `StateBuilder.update()` -> 120-dim stacked state
3. Stacked state -> `InferenceEngine.get_action()` -> (steering, acceleration)
4. (steering, acceleration) -> `ControlMapper.map_to_ackermann()` -> Ackermann drive command

---

## Directory Breakdown

| Directory | Purpose |
|-----------|---------|
| `src/` | Core source: environment, agent, training loops, physics, config |
| `sac_driver/` | Sim-to-real bridge: inference engine, LIDAR conversion, state building, control mapping |
| `config/` | YAML configurations for game, physics, and training variants (config_sac_8, config_sac_10 through config_sac_13) |
| `tools/` | Standalone GUI utilities: zone painter (spawn/kill/lookat), PGM outline processor |
| `runs/` | Training outputs (checkpoints, CSV logs) -- gitignored |
| `assets/maps/` | PGM occupancy grid maps and zone overlay PNGs |

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Beba-ai-ml/occupancy-racer-sac2.git
cd occupancy-racer-sac2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The GUI tools (`tools/map_zone_painter.py`, `tools/pgm_outline_ui.py`) require `tkinter`. On Debian/Ubuntu, install it with:

```bash
sudo apt install python3-tk
```

## Quick Start

```bash
source .venv/bin/activate
python run.py                    # Interactive mode (pygame)
```

### Training

```bash
# Synchronous training (simpler, single process + VecRacerEnv)
python -m src.train --config config/game.yaml --physics config/physics.yaml

# Asynchronous training (primary, multi-process actors + central learner)
python -m src.train_ssac --config-file config/config_sac_10.yaml
```

### Controls (Interactive Mode)

- **W**: accelerate
- **S**: brake / reverse
- **A**: steer left
- **D**: steer right
- **Mouse wheel**: zoom in/out
- **ESC**: quit

---

## Dependencies

From `requirements.txt`:

- `pygame >= 2.5` -- rendering and interactive mode
- `pyyaml >= 6.0` -- configuration file parsing
- `numpy >= 1.24` -- array operations, LIDAR simulation, replay buffer
- `torch >= 2.0` -- neural networks, SAC algorithm, `torch.compile`

Optional (for tools and zone loading):
- `Pillow` -- zone PNG loading in map_loader, zone painter tool
- `tkinter` -- GUI tools (map_zone_painter, pgm_outline_ui); system package `python3-tk` on Debian/Ubuntu

---

## Entry Points

| Command | Description |
|---------|-------------|
| `python run.py` | Interactive mode (calls `src.main.main()`) |
| `python -m src.main` | Same as above |
| `python -m src.train` | Synchronous multi-env training |
| `python -m src.train_ssac` | Asynchronous multi-process training (primary) |
| `python test_offline.py` | Offline inference pipeline test (no ROS2) |
| `python tools/map_zone_painter.py` | GUI zone painter for maps |
| `python tools/pgm_outline_ui.py` | GUI outline processor for PGM maps |

---

## Configuration

### config/game.yaml (Master Config)

The main configuration file controlling all aspects of training and gameplay:

- **map**: path to the occupancy grid map (YAML or PGM)
- **display**: window width, height, FPS, zoom
- **rl**: hidden_sizes, learning_rate, gamma, tau, alpha, batch_size, buffer_size, learn_every, learn_after, target_entropy, stack_frames
- **training**: total_steps, checkpoint_every, render_every
- **async**: num_actors, sync_every, transition_batch
- **sim_randomization**: Full domain randomization settings (see below)

### config/physics.yaml (Vehicle Physics)

- acceleration: 2.0 m/s^2, brake_deceleration: 3.5 m/s^2
- max_speed: 8.0 m/s, max_reverse_speed: 0.5 m/s
- friction: 0.6, drag: 0.2
- max_steer_angle: 20 degrees, wheelbase: 0.27 m (computed as length * 0.6)
- Vehicle size: 0.45 x 0.30 m

### config/config_sac_N.yaml (Training Variants)

Numbered configs for the SSAC trainer with varying scale:
- **config_sac_8**: 8 actors, medium buffer (baseline)
- **config_sac_10**: 64 actors, 1.6M buffer, learn_after=40000, grad_clip + alpha_min (full scale)
- **config_sac_11**: 128 actors, Dom_01 only, grad_clip + alpha_min
- **config_sac_12**: 64 actors, 10 maps, utd_ratio=0.25, batch_size=256
- **config_sac_13**: Latest iteration with tuned hyperparameters

### Domain Randomization Categories

Configured in `sim_randomization` section of game.yaml or config_sac_N.yaml:

| Category | What it randomizes |
|----------|--------------------|
| `physics` | acceleration, braking, max_speed, friction, drag, steering angle, wheelbase |
| `surface` | surface_friction, surface_drag |
| `observation_noise` | Gaussian noise on LIDAR, speed, steering observations |
| `control` | Steering and throttle delays (in steps) |
| `dt_jitter` | Simulation timestep variation |
| `action_noise` | Gaussian noise on actions |
| `perturb` | Random velocity/angular perturbations during episodes |
| `observation_delay` | Delayed observation delivery (in steps) |
| `action_repeat` | Repeat actions for multiple steps |
| `obstacles` | Static obstacles at episode start, dynamic obstacles during episode |
| `reward` | Reward shaping weights (alignment, speed, clearance, collision) |
| `track` | Track direction (clockwise/counter), center mode, direction-change probability |
| `episode` | Max steps, max distance |

---

## LIDAR System

- **27 rays** over a **210-degree arc** centered at 90 degrees (vehicle front)
- Dense region: 5-degree steps within +/-45 degrees of center (19 rays)
- Sparse region: 15-degree steps outside the dense zone (8 rays)
- Max range: 20 meters (configurable)
- Front cone: 20 degrees for clearance penalty computation
- Output: normalized distances in [0, 1] (distance / max_range)

---

## Reward Function

The reward at each step combines:

1. **Alignment bonus**: Dot product of velocity direction with track-following direction (clockwise or counterclockwise around the track center)
2. **Forward speed reward**: Proportional to forward speed (encouraging fast driving)
3. **Clearance penalty**: Negative reward when front LIDAR rays detect walls too close
4. **Collision penalty**: -20 on wall contact (episode terminates)
5. **Reverse penalty**: Penalty for driving backwards
6. **Balance penalty**: Discourages extreme steering at speed

---

## Checkpoint System

- Format version: `sac_checkpoint_v1`
- Saved atomically (write to tmp, then rename) with timestamped backups
- Contents: policy weights, critic weights, target network weights, all optimizer states, log_alpha, RNG states (Python, NumPy, PyTorch), training metadata (step count, episode count, best reward)
- CSV logging alongside checkpoints with per-episode statistics
- Resume training from checkpoint with `--resume` flag

---

## Key Files

| File | Why it matters |
|------|----------------|
| `src/racer_env.py` | The environment -- defines the entire RL problem (observations, actions, rewards, dynamics) |
| `src/rl_agent.py` | The SAC algorithm -- policy, critics, replay buffer, learning updates |
| `src/train_ssac.py` | Primary training script -- async multi-process architecture for fast data collection |
| `src/vehicle.py` | Vehicle physics -- Ackermann steering model that must match the real robot |
| `src/sim_config.py` | Domain randomization -- critical for sim-to-real transfer |
| `config/game.yaml` | Master config -- all hyperparameters and DR settings in one place |
| `sac_driver/` | The entire sim-to-real bridge -- everything needed to run the trained policy on hardware |

---

## Map System

Maps are PGM occupancy grids (P5 binary or P2 ASCII format) with optional YAML metadata files:

- **White pixels** (value >= 250): free space (driveable)
- **Dark pixels**: walls/obstacles
- **Zone overlay**: `{map_name}_zones.png` with RGB channels:
  - Red channel: kill zones (episode terminates if vehicle enters)
  - Green channel: spawn zones (vehicle spawns here)
  - Blue channel: lookat zones (vehicle faces toward these on spawn)

The `tools/map_zone_painter.py` GUI allows painting these zones interactively.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
