# SAC Training Optimization — Complete History

> **Goal:** Speed up training WITHOUT changing learning dynamics.
> **Hardware:** AMD Ryzen 7 5700X (8C/16T), 46GB RAM, CUDA GPU, 64 actor processes.
> **Baseline (pre-optimization):** Session 53 — config_sac_13, batch_size=256, 24K episodes, mean(100)=250m.

---

## PHASE 1 — Quick Wins (COMPLETE, VERIFIED)

All 4 tasks implemented and validated. No regressions.

### P1: Threaded Queue Drainer (Task 1.1) — IMPLEMENTED ➜ LATER REMOVED
- **What:** `TransitionDrainer` class — background daemon thread continuously drains `mp.Queue` into `ReplayBuffer`.
- **File:** `train_ssac.py:95-148`
- **Result:** Unblocked GPU from queue drain pauses.
- **Status:** Class still exists in code but **no longer used** in the main loop (see Phase 2 Fixes below). Direct queue access replaced it.

### P2: Vectorized Soft Update (Task 1.2) — IMPLEMENTED, ACTIVE
- **What:** Replaced Python for-loop over parameters with `torch._foreach_mul_` / `torch._foreach_add_`.
- **File:** `rl_agent.py` `_soft_update()` method
- **Result:** Eliminates ~32K kernel launches per update cycle. +2-3% GPU utilization.

### P3: forkserver (Task 1.3) — IMPLEMENTED, ACTIVE
- **What:** Changed `mp.get_context("spawn")` → `mp.get_context("forkserver")`.
- **File:** `train_ssac.py` actor spawning
- **Result:** ~1.5GB RAM savings. 3-5x faster actor startup.

### P4: gc.collect on Map Switch (Task 1.4) — IMPLEMENTED, ACTIVE
- **What:** Added `del env; gc.collect()` before rebuilding env on map switch.
- **File:** `train_ssac.py` actor map rotation logic
- **Result:** Prevents ~2GB transient memory spikes.

---

## PHASE 2 — Medium Effort (IMPLEMENTED, THEN PARTIALLY REVERTED)

Phase 2 introduced 4 performance optimizations (P10-P13) plus a main loop rewrite. The optimizations themselves are sound, but the **main loop changes broke learning quality** — UTD dropped from ~1.0 to ~0.02, causing all post-Phase-2 sessions to plateau at mean(100)≈50m vs 250m pre-Phase-2. This required 7 iterations of debugging (diag through diag_12).

### P10: Batch Insert to Replay Buffer (Task 2.1) — IMPLEMENTED, ACTIVE
- **What:** `ReplayBuffer.add_batch()` — numpy slice assignment with circular wraparound.
- **File:** `rl_agent.py:120-161`
- **How it works:**
  - Accepts 5 numpy arrays (states, actions, rewards, next_states, dones)
  - If batch fits without wraparound: single slice assignment `self.states[ptr:end] = states`
  - If wraparound needed: two slice assignments (end-of-buffer + start-of-buffer)
  - If batch >= capacity: trims to last `capacity` elements
- **Result:** 5 slice assignments instead of N×5 index assignments per drain cycle. ~80% faster buffer insertion.

### P11: Pinned Memory Pool (Task 2.2) — IMPLEMENTED, ACTIVE
- **What:** Pre-allocated 5 pinned CPU tensors + numpy views in `SACAgent.__init__()`. `ReplayBuffer.sample_into()` writes directly into pinned memory via `np.take(out=)`.
- **Files:**
  - `rl_agent.py:315-326` — pinned tensor allocation + numpy views
  - `rl_agent.py:176-191` — `sample_into()` method using `np.take(out=)`
  - `rl_agent.py:382-389` — `learn()` uses pinned pool path when CUDA available
- **How it works:**
  - 5 pinned tensors: `_pin_states`, `_pin_actions`, `_pin_rewards`, `_pin_next_states`, `_pin_dones`
  - 5 numpy views: `_np_states`, etc. (zero-copy views into pinned memory)
  - `sample_into()` calls `np.take(array, idx, axis=0, out=numpy_view)` — writes directly into pinned memory
  - `learn()` then does `async non_blocking .to(device)` for H2D transfer
- **Result:** Eliminates `cudaHostRegister` syscall per learn() call (~50-200µs each). Zero per-call allocation.

### P12: Shared Weights via POSIX Shared Memory (Task 2.3) — IMPLEMENTED, ACTIVE
- **What:** `SharedWeightSync` + `SharedWeightReader` replace 64 separate `mp.Queue` pipes with a single POSIX shared memory block in `/dev/shm`.
- **File:** `train_ssac.py:184-262`
- **How it works:**
  - **Main process (`SharedWeightSync`):** Flattens policy `state_dict` into a contiguous float32 buffer + int64 version counter at first 8 bytes. `update()` copies new flat weights + increments version.
  - **Actor side (`SharedWeightReader`):** Checks version counter. If changed, copies the flat buffer, slices into tensors by stored offsets/shapes, reconstructs `state_dict`, calls `policy.load_state_dict()`.
  - One `memcpy` (~0.2ms) replaces 64 pipe writes of pickled 517KB state_dicts (~70ms).
- **Result:** ~350x faster weight sync. Eliminated ~33MB/sync pickle overhead.

### P13: Batched Actor Transitions (Task 2.4) — IMPLEMENTED, ACTIVE
- **What:** Actors accumulate 32 transitions locally, then send one stacked numpy batch to queue.
- **File:** `train_ssac.py` actor loop, `_ACTOR_BATCH = 32` (line 181)
- **How it works:**
  - Actor appends `(obs, action, reward, next_obs, done)` to local list
  - When list reaches 32 items OR episode ends: stack into 5 numpy arrays, `queue.put(batch)`
  - Queue items are now `(np.ndarray[N,128], np.ndarray[N,2], np.ndarray[N], np.ndarray[N,128], np.ndarray[N])`
- **Result:** ~32x fewer IPC calls (from ~480/sec to ~15/sec). Less pipe contention, less pickle overhead.

---

## THE LEARNING QUALITY CRISIS (diag through diag_12)

### Root Cause Analysis

**Pre-Phase-2 (Session 53, the 250m baseline):**
- `agent.step()` called `learn()` per transition (UTD ≈ 1.0)
- Queue acted as natural backpressure: when queue filled, actors blocked on `put()`, throttling data flow to match learner speed
- Each transition was sampled ~256 times on average (256 updates per transition in buffer of 1.6M)
- Result: agent learned nuanced driving skills

**Post-Phase-2 (all diag sessions, plateau at 50m):**
- `TransitionDrainer` removed queue backpressure — background thread continuously emptied queue into an unbounded buffer
- Actors flooded data at ~3840 transitions/sec, but learner only did ~72 learn/sec (with batch=1024)
- UTD dropped to ~0.02 (98% of transitions never influenced gradient updates)
- Each transition sampled only ~19 times on average
- Result: enough learning for basic driving (50m) but not for nuanced skills (250m+)

### Iteration History

#### Iteration 1-5: Learning Debt System (ALL FAILED)
The initial post-Phase-2 main loop introduced a "learning debt" system — a float that accumulated gradient updates owed, processed in chunks. This went through 5 iterations:

1. **diag (initial):** Learning debt with unlimited accumulation. Main thread blocked for seconds doing thousands of gradient steps. Stats accumulated for 20-40s then dumped hundreds of episodes at once with frozen loss values.
2. **diag_1:** Added `_LEARN_CHUNK=64` max per iteration. Spread work across iterations. Didn't fix the fundamental UTD problem — debt accumulated faster than it was processed.
3. **diag_2:** Tuned chunk sizes and thresholds. UTD still ~0.02. mean(100) stuck at ~25m.
4. **diag_3:** More debt tuning. No improvement.
5. **diag_4:** Added `_MIN_DEBT=128` and `_MAX_BURST=512` constants. Still didn't wire them into the learn loop properly.
6. **diag_5 (completing diag_4):** Wired `_MIN_DEBT`/`_MAX_BURST` into the loop. Result: 448K steps, 16,436 updates, UTD=0.016, GPU 43%. Still plateaued.

**Conclusion:** The debt system was treating a symptom. The real problem was that TransitionDrainer removed all backpressure, letting actors flood data at 50x the learner's capacity.

#### Iteration 6: Simple Loop (PARTIAL SUCCESS — diag_10)
- **Ripped out entire debt system.** Single `agent.learn()` per main loop iteration.
- Stats drain every 64 learns.
- Sleep only when no data AND can't learn.
- **Result:** GPU 93% utilization, 72 learn/sec. But still only 1 learn per iteration → UTD still ~0.02 because drainer buffered thousands of transitions and loop processed them one-at-a-time.
- **User observed:** CPU 100%, GPU 2%. Explained: CPU saturated by 64 actors on 16 threads, GPU 2% because model [256,256,128] too tiny. These are expected.

#### Iteration 7: Proportional Learning + Drainer Cap (FAILED — diag_11)
- Changed drainer `max_buffered` from 65536 to 1024.
- Main loop learns N times per N transitions received (proportional to data).
- **Result:** Pipeline completely clogged. Drainer caps at 1024 transitions, main loop learns 1024 times (13.3s with batch=1024). During learning, queue (96K capacity) fills but drainer is paused. Only 1024 transitions per 13.3s cycle reach the buffer. total_steps=11,248 for 13K episodes.
- **Root cause:** Two problems: (1) batch=1024 makes each learn() take ~13ms, so 1024 learns = 13.3s blocking the loop, (2) queue_size=96K is too big — actors never feel backpressure.

#### Iteration 8: Direct Queue + Queue Cap (CURRENT — diag_12 pending)
- **Removed TransitionDrainer from main loop entirely.** Pull directly from `transition_queue.get_nowait()`.
- **Capped queue_size at 512** regardless of config value (line 816: `queue_size = min(queue_size, 512)`). 512 batches × 32 transitions = 16K transitions ≈ 4 seconds of actor output.
- **Proportional learning:** For each batch of N transitions received, do N learn() calls (UTD ≈ 1.0).
- **Natural backpressure restored:** When queue fills (512 items), actors block on `put()`, throttling data flow to match learner speed. This is exactly how pre-Phase-2 worked.
- **File:** `train_ssac.py:1184-1317` (main loop), `train_ssac.py:814-816` (queue cap)

**Current main loop structure (lines 1249-1323):**
```
while True:
    1. Pull ONE batch from queue (get_nowait — non-blocking)
    2. If got batch: add_batch to replay buffer, track transitions
    3. If can_learn AND got_any: learn N times (N = transitions in batch)
       - Every 64 learns: drain stats queue
    4. If not learning: drain stats queue
    5. Weight sync: check if crossed sync_every boundary
    6. Periodic DIAG print every 30s (shows UTD, updates, steps)
    7. Sleep 1ms only when truly idle (no data AND can't learn)
```

**Expected behavior with config_sac_20 (batch_size=1024):**
- Each learn() takes ~13ms with batch=1024
- Batch of 32 transitions → 32 learn() calls → ~416ms per queue item
- Actors produce ~120 queue items/sec → learner processes ~2.4/sec → queue fills in ~4s
- Once queue full: actors block, throughput drops to ~77 transitions/sec
- UTD ≈ 1.0 (32 learns per 32 transitions)
- Actors ~84% blocked waiting on queue.put()

**Expected behavior with config_sac_13 (batch_size=256):**
- Each learn() takes ~2-3ms with batch=256
- Batch of 32 transitions → 32 learn() calls → ~64-96ms per queue item
- Much higher throughput: ~333 transitions/sec
- UTD ≈ 1.0
- Actors ~30-50% blocked
- **4.3x faster wall-clock progress than config_sac_20**

---

## CURRENT STATE OF CODE

### What's Active (contributing to training)
| ID | Optimization | File | Status |
|----|-------------|------|--------|
| P2 | `_foreach` soft update | `rl_agent.py` | Active |
| P3 | forkserver | `train_ssac.py` | Active |
| P4 | gc.collect on map switch | `train_ssac.py` | Active |
| P10 | `add_batch()` replay buffer | `rl_agent.py:120-161` | Active |
| P11 | Pinned memory pool + `sample_into()` | `rl_agent.py:176-191, 315-326` | Active |
| P12 | Shared memory weights | `train_ssac.py:184-262` | Active |
| P13 | Batched actor transitions | `train_ssac.py` actor loop | Active |

### What's Dead Code (exists but unused)
| Code | File | Why Unused |
|------|------|-----------|
| `TransitionDrainer` class | `train_ssac.py:95-148` | Main loop bypasses it — pulls directly from queue |
| Learning debt variables | (removed) | Ripped out in iteration 6 |
| `agent.step()` method | `rl_agent.py:364-380` | Only used by `train.py` (sync). Async loop calls `learn()` directly |

### Key Constants
| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `_ACTOR_BATCH` | 32 | `train_ssac.py:181` | Transitions accumulated per actor before queue.put() |
| `queue_size` cap | 512 | `train_ssac.py:816` | Max queue items for backpressure |
| `_STATS_DRAIN_EVERY` | 64 | `train_ssac.py:1185` | Drain stats queue every N learn() calls |
| `resume_warmup_steps` | 50,000 | `train_ssac.py:~1072` | Flat 50K steps warmup on resume |

---

## PHASE 3 — Advanced (NOT STARTED)

These are independent of the learning quality fix. Implement only after diag_12 confirms UTD≈1.0 works.

### Task 3.1: `torch.compile` on `learn()` with `reduce-overhead`
- **Risk:** HIGH. CUDA graphs require constant tensor shapes. May conflict with pinned memory pool.
- **Expected:** +10-20% GPU utilization.
- **Note:** Current per-network `torch.compile(mode='default')` works. `reduce-overhead` was tried before and caused CUDA graph deadlocks — needs careful testing.

### Task 3.2: CUDA Streams Prefetching (Double-Buffered H2D)
- **What:** Pre-fetch next batch on a separate CUDA stream while current batch computes on default stream.
- **Expected:** +5-10% GPU utilization. CPU sampling overlapped with GPU compute.

### Task 3.3: Shared Memory Map Data
- **What:** Pre-load all maps in main process, store in `multiprocessing.shared_memory`. Actors read zero-copy.
- **Expected:** ~900MB RAM savings (64 copies → 1 per map).

### Task 3.4: Eliminate Pygame in Headless Actors
- **What:** Replace `pygame.Vector2` with lightweight alternative. Skip `pygame.init()` when `render=False`.
- **Expected:** ~960MB RAM savings. Highest effort task.

---

## TRAINING SESSION HISTORY

| Session | Episodes | Config | batch_size | Mean(100) | Notes |
|---------|----------|--------|------------|-----------|-------|
| **53** | **24,000** | **sac_13** | **256** | **250m** | **Pre-Phase-2 baseline. Best ever.** |
| diag | ? | sac_13 | 256 | ~50m | Phase 2 + debt system v1. Blocked main thread. |
| diag_1 | ? | sac_13 | 256 | ~50m | Debt chunking. UTD still ~0.02. |
| diag_2 | 20,000 | sac_13 | 256 | ~25m | Tuned debt. Worse. |
| diag_3 | ? | sac_13 | 256 | ~50m | More debt tuning. No improvement. |
| diag_4 | ? | sac_13 | 256 | ~50m | MIN_DEBT/MAX_BURST not wired. |
| diag_5 | 26,500 | sac_13 | 256 | ~56m | MIN_DEBT/MAX_BURST wired. UTD=0.016. |
| diag_10 | 29,750 | sac_20 | 1024 | ~56m | Simple loop (1 learn/iter). UTD ~0.02. |
| diag_11 | 21,750 | sac_20 | 1024 | ~34m | Proportional + drainer cap. Pipeline clogged. |
| **diag_12** | **pending** | **sac_20** | **1024** | **?** | **Direct queue + queue cap. UTD should ≈ 1.0.** |

### Config Reference
| Param | sac_13 (best) | sac_20 (current) |
|-------|---------------|-----------------|
| batch_size | 256 | 1024 |
| utd_ratio | 2.0 | 2.0 |
| sync_every | 1600 | 1600 |
| learn_after | 5000 | 5000 |
| alpha_lr | 0.0001 | 0.0001 |
| memory_size | 1,600,000 | 1,600,000 |
| queue_size (config) | 96,000 | 96,000 |
| queue_size (actual) | 512 (capped) | 512 (capped) |

---

## KEY LESSONS LEARNED

1. **Never remove queue backpressure.** The original `mp.Queue` blocking on `put()` was the primary mechanism ensuring UTD≈1.0. TransitionDrainer removed this by continuously emptying the queue, letting actors produce data 50x faster than the learner could process it.

2. **UTD matters more than throughput.** 72 learn/sec with UTD=1.0 produces vastly better agents than 5000 learn/sec with UTD=0.02. The agent needs enough gradient updates *per transition* to learn fine-grained driving skills.

3. **batch_size affects UTD throughput.** batch=256 gives ~500 learn/sec (each 2-3ms). batch=1024 gives ~77 learn/sec (each ~13ms). Both achieve UTD≈1.0, but batch=256 processes data 4.3x faster. Session 53 used batch=256.

4. **Debugging RL main loops is treacherous.** 7 iterations were needed to find the correct fix. The debt system was a red herring — it treated a symptom (bursty output) while the real disease (no backpressure → low UTD) went undiagnosed.

5. **CPU 100% / GPU 2% is expected.** The model [256,256,128] is too small to saturate a modern GPU. Training is CPU-bound due to 64 actor processes + physics sim. Increasing batch_size or network size would increase GPU utilization, but at the cost of fewer gradient updates.

---

## NEXT STEPS (PRIORITY ORDER)

1. **Run diag_12** and verify:
   - DIAG output shows `utd=` climbing toward 1.0
   - `total_updates ≈ total_steps` in checkpoint
   - Distance progression breaks past 50m plateau
2. **If diag_12 too slow** (77 trans/sec with batch=1024): switch to config_sac_13 (batch=256) for 4.3x throughput
3. **Long training run** (20K+ episodes) with winning config to match/exceed Session 53's 250m
4. **Phase 3 optimizations** only after learning quality is confirmed restored
5. **Clean up dead code:** Remove TransitionDrainer class, remove dead debt system references in comments

Last updated: 2026-02-21
