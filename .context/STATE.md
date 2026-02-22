# SAC v2 - Current State

## What Works
- Async training with 64 actors on GPU — stable, no crashes
- All 9 performance optimizations (P1-P9) implemented and validated
- All 14 sim-to-real features (S1-S14) implemented, configured, and enabled
- S1-S5, S11 active in vehicle.py (hardcoded defaults, always on)
- S6-S14 configured in game.yaml sim_randomization (added 2026-02-16)
- sac_driver updated for 32-dim obs + IMU (backward compatible)
- Session 33 (pre-sim2real): best run, 11K episodes, agent completes full laps
- Session 39 (with action_repeat=8 + delay_steps=[4,6]): 8K episodes, learned but 20x slower due to double-counted latency
- **Session 53**: current best — **24K episodes, mean(100)=250m**, config_sac_13, all fixes applied
- game.py interactive mode now synchronized with training obs format (+5 dim)
- Replay buffer stores correct actions during action_repeat windows
- **Phase 2 optimizations implemented** (P10-P13): batch insert, pinned memory pool, shared memory weights, batched actor transitions
- **Learning quality fix** (diag_12): Direct queue access with backpressure replaces TransitionDrainer. Proportional learning (UTD ≈ 1.0) matches pre-Phase-2 agent.step() behavior.

## Work in Progress
- **diag_12 validation pending** — need to run and confirm UTD ≈ 1.0 breaks the 50m plateau
- **A/B testing on hold** — configs 20-25 created but learning quality must be restored first
- If diag_12 too slow with batch_size=1024: switch to config_sac_13 (batch=256) for 4.3x throughput

## Recent Changes (2026-02-21)

### Phase 2: Performance Optimizations (4 tasks, all implemented)

**P10: Batch insert to replay buffer (Task 2.1)**
- Added `ReplayBuffer.add_batch()` — numpy slice assignment with circular wraparound
- File: `rl_agent.py:120-161`

**P11: Pinned memory pool (Task 2.2)**
- Pre-allocated 5 pinned CPU tensors + numpy views in `SACAgent.__init__()`
- Added `ReplayBuffer.sample_into()` — writes directly into pinned arrays via `np.take(out=)`
- Files: `rl_agent.py:176-191, 315-326, 382-389`

**P12: Shared weights via POSIX shared memory (Task 2.3)**
- `SharedWeightSync` + `SharedWeightReader` replace 64 mp.Queue pipes with POSIX shared memory
- ~350x faster weight sync
- Files: `train_ssac.py:184-262`

**P13: Batched actor transitions (Task 2.4)**
- Actors accumulate 32 transitions locally (`_ACTOR_BATCH=32`), flush as numpy batch to queue
- ~32x fewer IPC calls
- File: `train_ssac.py` actor loop

### Learning Quality Fix (7 iterations, diag through diag_12)

**Root cause:** Phase 2's TransitionDrainer removed queue backpressure. Actors flooded data at 3840/sec, learner only did 72/sec → UTD dropped from 1.0 to 0.02 → agent plateaued at 50m (vs 250m pre-Phase-2).

**Failed approaches (diag through diag_11):**
1. Learning debt system (5 iterations): accumulated gradient "debt" processed in chunks. Treated symptom (bursty output) not disease (no backpressure).
2. Simple loop (diag_10): 1 learn per iteration. UTD still ~0.02 — drainer buffered thousands of transitions.
3. Proportional learn + drainer cap (diag_11): capped drainer at 1024, learned proportionally. Pipeline clogged — 1024 learns × 13ms = 13.3s blocking, queue=96K too big for backpressure.

**Current fix (diag_12, pending validation):**
- Removed TransitionDrainer from main loop — pull directly from `queue.get_nowait()`
- Capped queue_size at 512 (line 816): `queue_size = min(queue_size, 512)`. 512 × 32 = 16K transitions ≈ 4s buffer.
- Proportional learning: N learn() calls per batch of N transitions → UTD ≈ 1.0
- Natural backpressure restored: actors block on `put()` when queue full
- File: `train_ssac.py:1184-1323` (main loop), `train_ssac.py:814-816` (queue cap)

### Scaling & Bottleneck Analysis
- **350 actors would crash** on current hardware (46GB RAM, Ryzen 7 5700X 16T). RAM limit ~160 actors, CPU sweet spot ~128.
- **Post-buffer-fill bottleneck identified**: CPU 30%, GPU 30% because single-threaded main process alternates drain/learn. Actors block on queue.put().
- **batch_size=256 identified as config_13's biggest weakness** — GPU severely underutilized. Config_sac_10 had batch_size=1024 but worse other params.
- **Config 13 confirmed as best config** over 10 and 12: utd_ratio=2.0, sync_every=1600, alpha_lr=0.0001, learn_after=5000 are all superior choices.
- **RAM purchase analysis**: buying more RAM helps for bigger replay buffer (memory_size), NOT for more actors. Sweet spot stays at 64-160 actors regardless of RAM.

### New A/B Test Configs (config_sac_20-25)
- Created 6 configs branching from config_13, each testing one hyperparameter change
- All designed to fit within 46GB RAM (memory_size capped at 3M instead of 6M)

## Earlier Changes (2026-02-16)

### Critical Bug Fixes (from 4-tester audit)

**C1: Action repeat replay buffer corruption (FIXED)**
- Problem: With action_repeat=8, policy was queried every sim frame but env used stale repeated action. 7/8 transitions stored wrong action in replay buffer, corrupting Q-function learning.
- Fix: Added `wants_new_action` property to RacerEnv. Actor loop in train_ssac.py now only queries policy when repeat window expired. Same action variable reused during repeat frames.
- Files: `racer_env.py:853-856`, `train_ssac.py:341-358`
- Bonus: ~8x fewer GPU inference calls in actors.

**B1: Vehicle actuator state not reset between episodes (FIXED)**
- Problem: `servo_actual`, `accel_actual`, `yaw_rate` carried over from previous episode, corrupting first few steps.
- Fix: Added zeroing of all three in `reset()`.
- File: `racer_env.py:1427-1430`

**C2: game.py obs_dim +3 → +5 (FIXED)**
- Problem: After S7 added IMU channels, game.py still used +3 (30-dim). Trained checkpoints couldn't load.
- Fix: Updated `_build_observation()` to include linear_accel and angular_vel. Updated `lidar_state_dim` to +5.
- File: `game.py:112, 539-566`

**B2: game.py _obs_stack.clear() before defined (FIXED)**
- Problem: Triple-duplicated init code called `self._obs_stack.clear()` before deque was created (line 165 vs 184). AttributeError on instantiation.
- Fix: Removed all duplicated assignments, proper ordering.
- File: `game.py:156-185`

**B3: game.py _reset_episode missing resets (FIXED)**
- Problem: `backward_death`, `backward_penalized`, `backward_distance_m`, `stuck_time` not reset. After first backward death, every subsequent episode terminated immediately.
- Fix: Added all missing resets + vehicle actuator zeroing + `_prev_speed/_prev_angle` for IMU.
- File: `game.py:583-606`

**B4: game.py reward_clip=1.0, np.round (FIXED)**
- Problem: game.py had reward_clip=1.0 (vs 20.0 everywhere else) and `np.round(obs, 3)` (training doesn't round).
- Fix: reward_clip → 20.0, removed np.round().
- File: `game.py:178, 555`

**B5: train.py missing grad_clip and alpha_min (FIXED)**
- Problem: SACAgent created without these params, defaulting to 0.0 (disabled).
- Fix: Now reads from config and passes through.
- File: `train.py:257-258`

**B6: StateBuilder extra_dims=3 when use_imu=False (FIXED)**
- Problem: Deployment with use_imu=False produced 30-dim obs, model expects 32.
- Fix: `_extra_dims` always 5 (IMU channels always present in training obs).
- File: `sac_driver/state_builder.py:23`

### Warning-Level Fixes

**W1: Sensor delays per-episode instead of per-frame random (FIXED)**
- Problem: S10 sensor delays re-randomized via `np.random.randint()` every frame — unrealistic jitter.
- Fix: Base delay sampled once per episode in `reset()`, with ±1 frame jitter per step.
- File: `racer_env.py:1443-1449, 1345-1347`

**W2: Wind OU process (FIXED)**
- Problem: Wind used i.i.d. Gaussian per frame — no temporal correlation (vibration, not wind).
- Fix: Replaced with Ornstein-Uhlenbeck process modulated by sinusoidal envelope.
- File: `racer_env.py:940-952`

**W3: lidar_delay_range default (7,8) → (3,4) (FIXED)**
- Code default now matches config value. Prevents surprise if key deleted from YAML.
- File: `racer_env.py:342`

**W5: _bump_nofile_limit no-op when hard==INFINITY (FIXED)**
- Target now set to 65536 when hard limit is infinity.
- File: `train_ssac.py:104-105`

**W6: auto_entropy dead code removed (FIXED)**
- Both branches set auto_entropy=True, dead fixed_alpha path. Cleaned up.
- File: `rl_agent.py:242-259`

**W7: save_checkpoint strips _orig_mod. prefix (FIXED)**
- torch.compile prefixed keys now cleaned before saving. Consistent with _cpu_state_dict.
- File: `rl_agent.py:399-401`

**W8: stats_queue bounded (FIXED)**
- Added maxsize=num_actors*100 to prevent unbounded memory growth.
- File: `train_ssac.py:809`

**W11: train.py saves distance_history in checkpoint (FIXED)**
- Mean distance metrics now survive resume.
- File: `train.py:435`

### Earlier Config Fixes (same session)
- action_repeat: kept at 8 (correct 8Hz)
- dr_delay_steps: changed from [4,6] to [0,1] (was double-counting vehicle.py lag)
- dr_obs_delay: set to false (S10 sensor_delay replaces it, prevents stacking)
- Added S6-S14 sections to game.yaml sim_randomization (were missing — features never enabled)
- S10 lidar_delay_frames: code default aligned from [7,8] to [3,4]
- S12 slope: disabled (slope_enabled=false, indoor flat floor)
- obs_dim fix: +3 → +5 in train_ssac.py:647, train.py:237

### Performance Optimizations (P1-P9) — from previous session
- P1: Background CSV writer (threading + queue)
- P2: Batched GPU transfers (pin all → transfer all with non_blocking)
- P3: Pre-allocated replay buffer index (fixed np.random.randint bug)
- P4: Parallel VecEnv recv via multiprocessing.connection.wait()
- P5: Vectorized action selection
- P6: Vectorized observation building, removed np.round()
- P7: LiDAR ray-obstacle with AABB cache + sphere pre-filter + early exit
- P8: Cached critic_params list
- P9: torch.compile mode changed from 'reduce-overhead' to 'default'

## Decisions

0. **batch_size=256 gives 4.3x faster wall-clock progress** than batch=1024 at same UTD≈1.0. Session 53 (250m) used batch=256. batch=1024 underutilizes GPU but the real constraint is learn() throughput for UTD, not GPU %.
1. **8Hz control rate (action_repeat=8)** — matches real LiDAR. Non-negotiable.
2. **delay_steps=[0,1] not [4,6]** — delay_steps only models inference pipeline (~15ms). Servo/motor lag is already in vehicle.py first-order filters.
3. **IMU always in obs (32-dim)** — even when imu sim_cfg not enabled, IMU channels exist. obs_dim is always 32.
4. **S9: distance_progress replaces alignment** — code has if/else branch.
5. **Train from scratch required** — obs_dim changed, action_repeat behavior fixed, env dynamics changed.
6. **Policy only queried when wants_new_action=True** — during action_repeat window, same action reused. Saves GPU compute and ensures replay buffer integrity.
7. **Sensor delays per-episode with ±1 jitter** — real hardware delay is constant (wiring/buffering), not random per-read.
8. **Wind as OU process** — temporally correlated, matches real sustained gusts vs high-frequency vibration.

## Known Issues

### Fixed (this session)
- ~~observation_delay + sensor_delay stacking~~ → obs_delay disabled
- ~~S12 slope ±2°~~ → slope disabled (indoor flat floor)
- ~~Action repeat replay buffer corruption~~ → C1 fix
- ~~Vehicle actuator state leaking between episodes~~ → B1 fix
- ~~game.py completely out of sync~~ → C2, B2, B3, B4 fixes
- ~~Bursty episode output (hundreds dumped at once)~~ → proportional learning + stats drain every 64 learns
- ~~weights_queues 64x pickle overhead~~ → POSIX shared memory (P12)
- ~~TransitionDrainer removing backpressure → UTD=0.02~~ → direct queue access + queue cap at 512
- ~~Learning debt system bugs~~ → debt system removed entirely, replaced with proportional learning

### Open concerns
1. **game.py reward function still diverges from racer_env.py** — hardcoded weights, different structure (no front_cone_deg, no distance_progress). Low priority since game.py is for interactive play, not training.
2. **Duplicated utility functions** — `_parse_hidden_sizes`, `_parse_target_entropy`, `_resolve_accel_range` etc. copy-pasted across train.py, train_ssac.py, game.py. Should be in src/utils.py.
3. **GaussianPolicy duplicated** in rl_agent.py and sac_driver/policy_loader.py with different method names.
4. **S11 yaw_inertia_tau=0.100** — estimated, not measured on real vehicle.
5. **reward_alignment_bonus: 0.02 still in config_sac_*.yaml** — harmless (ignored when S9 enabled) but confusing.
6. **train.py (sync) still has action_repeat issue** — VecRacerEnv makes it harder to fix. train_ssac.py (primary) is fixed.
7. **Missing CLI mappings for S6-S14** — can only configure via game.yaml, not CLI overrides.
8. **Battery sag horizon hardcoded to 60s** — should be configurable.
9. **imu_enabled flag in config is misleading** — parsed but never checked, IMU channels always present.
10. **Lidar rays pass through kill boundaries** — collision_mask used for collision but occupied_mask used for raycasting.

## Next Steps

1. **Run diag_12** — verify UTD ≈ 1.0 and distance breaks past 50m plateau
2. **If too slow with batch=1024** — switch to config_sac_13 (batch=256, 4.3x faster)
3. **Long training run** (20K+ episodes) once learning quality confirmed
4. **Clean up dead code** — remove TransitionDrainer class, old comments
5. **A/B test configs 20-25** — only after baseline is restored
6. **Phase 3 optimizations** — CUDA streams, shared map data, pygame removal
7. **Validate on real vehicle** — use hardware_profile.md checklist

## Training Session History

| Session | Episodes | Config | batch | Mean(100) | Notes |
|---------|----------|--------|-------|-----------|-------|
| 33 | 11,475 | sac_13 | 256 | 800m+ | Pre-sim2real best. Completes laps. |
| 36 | ~480 | sac_11 | — | — | Hang at ~400ep (np.random.randint bug) |
| 39 | 8,131 | sac_13 | 256 | — | delay_steps double-counted, 20x slower |
| 43 | ~12K+ | sac_13 | 256 | — | Intermediate run |
| **53** | **24,000** | **sac_13** | **256** | **250m** | **Pre-Phase-2 baseline. Best ever.** |
| diag | ? | sac_13 | 256 | ~50m | Phase 2 + debt v1. Main thread blocked. |
| diag_2 | 20,000 | sac_13 | 256 | ~25m | Debt tuning. Worse. |
| diag_5 | 26,500 | sac_13 | 256 | ~56m | MIN_DEBT/MAX_BURST. UTD=0.016. |
| diag_10 | 29,750 | sac_20 | 1024 | ~56m | Simple loop. UTD ~0.02. |
| diag_11 | 21,750 | sac_20 | 1024 | ~34m | Proportional + drainer cap. Clogged. |
| **diag_12** | **pending** | **sac_20** | **1024** | **?** | **Direct queue + cap. UTD target ≈ 1.0.** |

### Config Comparison (10 vs 12 vs 13)
| Param | 10 | 12 | **13 (best)** |
|-------|----|----|---------------|
| batch_size | 1024 | 256 | 256 |
| utd_ratio | 1.0 | 0.25 | **2.0** |
| sync_every | 6400 | 12800 | **1600** |
| learn_after | 20K | 40K | **5K** |
| alpha_lr | 0.0003 | 0.0003 | **0.0001** |
| map_pool | 40 maps | 10 maps | 10 maps |

Last updated: 2026-02-21
