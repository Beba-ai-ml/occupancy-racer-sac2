# SAC v2 - Current State

## What Works
- Async training with 64 actors on GPU — stable, no crashes
- All 14 sim-to-real features (S1-S14) implemented, configured, and enabled
- S1-S5, S11 active in vehicle.py (hardcoded defaults, always on)
- S6-S14 configured in game.yaml sim_randomization (added 2026-02-16)
- sac_driver updated for 32-dim obs + IMU (backward compatible)
- **Session 53**: best run — **24K episodes, mean(100)=250m**, config_sac_13, all fixes applied
- game.py interactive mode synchronized with training obs format (+5 dim)
- Replay buffer stores correct actions during action_repeat windows (C1 fix)
- **Phase 1/2 fully reverted** (2026-02-22): all performance optimizations removed, clean simple architecture restored

## Work in Progress
- **Fresh training run needed** with updated config_sac_20 (batch=256, learn_after=5000, memory_size=8M)
- Need to verify learning curves match Session 53 baseline quality

## Recent Changes (2026-02-22)

### Phase 1/2 Complete Revert
All performance optimizations (P1-P13) surgically removed from 4 source files while preserving all bug fixes (C1, B1-B6, W1-W8) and sim-to-real features (S1-S14).

**Removed from train_ssac.py:**
- LogBuffer class (P1 background CSV writer) → direct file writes
- SharedWeightSync + SharedWeightReader (P12 POSIX shared memory) → per-actor mp.Queue(maxsize=1)
- _ACTOR_BATCH=32 (P13 batched actor transitions) → single transition puts
- forkserver → spawn context
- queue_size cap at 512 → config-driven value
- DIAG prints and _last_diag_time

**Removed from rl_agent.py:**
- add_batch() (P10 batch insert) → single memory.add()
- sample_into() + pinned memory pool (P11) → per-call pin_memory()
- Pre-allocated index buffer (P3)
- Cached _critic_params (P8)
- _foreach soft update (P2) → for-loop

**Removed from vec_env.py:**
- Parallel recv via multiprocessing.connection.wait() (P4) → sequential list comprehension

**Removed from train.py:**
- LogBuffer class (P1) → direct CSV writes with csv_header_written flag

### Config Tuning (config_sac_20.yaml)
- `learn_after`: 40000 → **5000** (Session 54 proved faster ramp-up)
- `memory_size`: 1600000 → **8000000** (~2.5 hours retention at 64 actors)
- `batch_size`: 1024 → **256** (proven in Session 53 at 250m)

### Why Phase 1/2 Was Reverted
Phase 2 optimizations (especially SharedWeightSync, TransitionDrainer, batched transitions) broke the queue backpressure mechanism that maintains UTD ≈ 1.0. Learning quality degraded catastrophically: agent plateaued at ~50m instead of climbing to 250m+. After 12 diagnostic iterations (diag through diag_25), the root cause was confirmed as broken UTD ratio. Surgical revert was chosen over further patching.

## Earlier Changes (2026-02-16)

### Critical Bug Fixes
- **C1: Action repeat replay buffer corruption** — Actor loop now only queries policy when `wants_new_action=True`
- **B1: Vehicle actuator state not reset** — servo_actual, accel_actual, yaw_rate zeroed on reset()
- **C2: game.py obs_dim +3 → +5** — IMU channels added to match training
- **B2-B4: game.py fixes** — deque init ordering, missing resets, reward_clip/np.round
- **B5: train.py missing grad_clip/alpha_min** — now passed to SACAgent
- **B6: StateBuilder extra_dims=3 → 5** — deployment obs matches training

### Warning-Level Fixes
- W1: Sensor delays per-episode (not per-frame)
- W2: Wind OU process (not i.i.d.)
- W3: lidar_delay_range default aligned
- W5: _bump_nofile_limit for INFINITY
- W6: auto_entropy dead code removed
- W7: save_checkpoint strips _orig_mod. prefix
- W8: stats_queue bounded
- W11: train.py saves distance_history

### Config Fixes
- action_repeat=8 (correct 8Hz)
- dr_delay_steps: [4,6] → [0,1] (no double-counting)
- dr_obs_delay: false (S10 replaces it)
- S6-S14 sections added to game.yaml
- S12 slope disabled (indoor flat floor)

## Decisions

0. **Phase 1/2 reverted** — performance optimizations broke learning quality (UTD). Clean simple code preferred over fast but broken code. Can re-implement carefully later if needed.
1. **batch_size=256** gives faster wall-clock progress than 1024 at same UTD. Session 53 (250m) used 256.
2. **memory_size=8M** provides ~2.5 hours retention at 64 actors. Longer retention = more diverse samples.
3. **learn_after=5000** — Session 54 (learn_after=5000) climbed faster than diag_25 (learn_after=40000).
4. **8Hz control rate (action_repeat=8)** — matches real LiDAR. Non-negotiable.
5. **delay_steps=[0,1] not [4,6]** — delay_steps only models inference pipeline. Actuator lag is in vehicle.py.
6. **IMU always in obs (32-dim)** — even when imu sim_cfg not enabled, IMU channels exist.
7. **Policy only queried when wants_new_action=True** — saves GPU compute, ensures replay buffer integrity.
8. **Sensor delays per-episode with ±1 jitter** — matches real hardware.
9. **Wind as OU process** — temporally correlated, matches real sustained gusts.

## Known Issues

### Open concerns
1. **game.py reward function diverges from racer_env.py** — hardcoded weights, different structure. Low priority (interactive play only).
2. **Duplicated utility functions** — `_parse_hidden_sizes`, `_parse_target_entropy`, etc. copy-pasted across train.py, train_ssac.py, game.py. Should be in src/utils.py.
3. **GaussianPolicy duplicated** in rl_agent.py and sac_driver/policy_loader.py with different method names.
4. **S11 yaw_inertia_tau=0.100** — estimated, not measured on real vehicle.
5. **reward_alignment_bonus: 0.02 still in configs** — harmless (ignored when S9 enabled) but confusing.
6. **train.py (sync) still has action_repeat issue** — VecRacerEnv makes it harder to fix. train_ssac.py (primary) is fixed.
7. **Missing CLI mappings for S6-S14** — can only configure via game.yaml, not CLI overrides.
8. **config_sac_13.yaml has learn_after=40000** — but Session 53 actually ran with 5000. Config file doesn't match historical run.

## Next Steps

1. **Run fresh training** with config_sac_20 (batch=256, learn_after=5000, memory_size=8M)
2. **Verify learning curves** match Session 53 baseline quality (should reach 100m+ by ~8K episodes)
3. **Long training run** (20K+ episodes) to see if 8M replay buffer helps
4. **A/B test** config variations once baseline confirmed
5. **Validate on real vehicle** — use hardware_profile.md checklist
6. **Consider re-implementing Phase 2 carefully** — one optimization at a time, verifying UTD stays ≈ 1.0 after each

## Training Session History

| Session | Episodes | Config | batch | learn_after | Mean(100) | Notes |
|---------|----------|--------|-------|-------------|-----------|-------|
| 33 | 11,475 | sac_13 | 256 | 5K | 800m+ | Pre-sim2real best. Completes laps. |
| 39 | 8,131 | sac_13 | 256 | 5K | — | delay_steps double-counted, 20x slower |
| **53** | **24,000** | **sac_13** | **256** | **5K** | **250m** | **Pre-Phase-2 baseline. Best post-sim2real.** |
| 54 | 14,700 | sac_20 | 1024 | 5K | 112m | config_sac_20, cut short |
| diag | ? | sac_13 | 256 | 5K | ~50m | Phase 2 + debt v1. Broken UTD. |
| diag_2 | 20,000 | sac_13 | 256 | 5K | ~25m | Debt tuning. Worse. |
| diag_5 | 26,500 | sac_13 | 256 | 5K | ~56m | MIN_DEBT/MAX_BURST. UTD=0.016. |
| diag_10 | 29,750 | sac_20 | 1024 | 5K | ~56m | Simple loop. UTD ~0.02. |
| diag_11 | 21,750 | sac_20 | 1024 | 5K | ~34m | Proportional + drainer cap. Clogged. |
| diag_25 | 8,500 | sac_20 | 1024 | 40K | ~49m | Direct queue, cut short at 8.5K eps |

### Config Comparison (10 vs 12 vs 13 vs 20)
| Param | 10 | 12 | **13** | **20 (current)** |
|-------|----|----|--------|------------------|
| batch_size | 1024 | 256 | 256 | **256** |
| utd_ratio | 1.0 | 0.25 | **2.0** | **2.0** |
| sync_every | 6400 | 12800 | **1600** | **1600** |
| learn_after | 20K | 40K | **5K** | **5K** |
| alpha_lr | 0.0003 | 0.0003 | **0.0001** | **0.0001** |
| memory_size | — | — | 1.6M | **8M** |
| map_pool | 40 maps | 10 maps | 10 maps | 10 maps |

Last updated: 2026-02-22
