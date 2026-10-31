# M2 v2 plan, v3: train the defender against the eval's own full-body opponent

Written 2026-10-29. Supersedes `M2V2_PLAN.md` (v2). Every blocker B1-B7 and minor m1-m19 of `CRITIQUE_M2V2.md` is fixed here;
the change map is in section 12.
- Repo `r1-sparring-gym-spar`, branch feat/sparring-m2v2 @ b81fa16: read only for this revision. No tracked file was changed,
  nothing was committed, no MUJOCO_LOG.TXT was left.
- CPU only, 4 worker processes. New simulation: **R4-pre** only (67000-67767 and 68000-68767, train clips, real EO bout), logged in
  `SEED_LEDGER.md`. No GPU, no val seed (21000-21999), no test seed (22000-22511), no kfab_202/205 content, no kfab_204 simulation.

## Plain-language summary (for a phone)

1. Our robot learned to block two floating fists, but the judge throws a whole robot at it, so in training we now use exactly the judge's robot.
2. A new laptop test (4,600 judge fights) shows the old robot mainly fell when its motors lagged or after the opponent went down, not by tripping it.
3. So training adds the judge's motor lag and a price on stepping into a fallen opponent, and the laptop copy must reproduce those falls before any money.
4. Every pass/fail check now uses enough fights (4,000 to 12,000 per check) that a correct copy passes about 99% of the time and a wrong one fails.
5. GPU cost for the full best plan (3 training pairs, longer pilot) is about $38, up to about $66 if the GPU is slow; the expected spend is about $17. A top-up of about $65 is needed.
6. The chance of passing every M2 rule is about 1 in 12. That is lower than v2's 1 in 10, but v2's gates would almost never have passed.

---

## 0. Decision (unchanged winner, updated rows)

**Winner: design 2, "learned-tracker"** (`design_learned-tracker.md`, D2), with grafts from D3 "cheap-staged" (replica
gates, driver interface, case against a kinematic opponent) and D1 "physics-tracked" (LOCO pilot, keypoint-use sign check,
hygiene asserts). The core is unchanged:
- the eval opponent itself (shipped tracker `model_13749.pt` in torch, never the all-clip ONNX) on a free-floating second R1, no
  root assist;
- shell contact on hulls for all 32 links of both robots;
- keypoints from the opponent's real links;
- the eval's latency, per-episode DR, fall rule and sim options.

| question | **v3 choice** | why (what changed from v2) |
|---|---|---|
| Defender self-collision | **off**, as at eval | unchanged |
| Leg learning | **zero-init residual on G's legs + L2 anchor + stage-A fallback** | unchanged |
| Opponent diversity | **T0 50% / T1 50%**, D3 admission rule if T1 fails | T1 gates are now powered (T1-c N 6144) |
| Knockdown and post-knockdown | **-3 he per knockdown attributed by defender work (m1)** + down-neutral charge on the eval's non-sticky both-up state (m4) + **new post-knockdown terms** (B1 fix 5) + S4 | R4-pre: on train clips A3 falls after 58% of knockdowns (G 7%), but knocks down *less* than G |
| Motor and obs delay | **the eval's per-episode act_delay flag and obs lag k** | R4-pre: 71 of A3's 80 train-clip falls are in act_delay bouts; this is now the first lever |
| Pre-pod CPU gate | **powered equivalence gates vs EO** (R-EQ), ladder as diagnostics vs step 0 | B2: v2's ladder false-failed with probability ~1 |
| Blind twin | **H(blind3)/M3s**, identical pipeline, per seed | unchanged |
| Pushes | **none in T0 rows; <= 0.3 m/s at half rate in other rows; hit terms masked 0.5 s after a push** | m5 |
| Seeds and budget | **3 seed pairs default, LOCO pilot 2000 iterations, throughput fixes not cuts, written check-ins at $45 / $80** | B6 |
| Test size | **512 bouts (22000-22511), decided now, blind** | m17 |

Rejected outright (D3 section 1): a kinematic or mocap opponent (cannot be knocked down, pushes with infinite force, saves < 10%
of a launch-bound run).

---

## 0.R R4-pre (done, CPU, $0): the failure signature on the train clips

Full notes: `r4pre\NOTES.md`; numbers: `r4pre\R4PRE_RESULT.json`.

**What ran.** The real eval bout (P1's `p1_eval.run()` = `sim/dev_eval.run()`, aim `standoff+armwarp_vz`, hard-tier distances,
20 s, DR on, delays as drawn, no pushes, keypoints truth) for G, H(A3s)/M3s and the full run-3 A3 on kfab_200/201/203.
- 768 paired bouts per controller at 67000-67767 (eval distance mix) and 768 more at 68000-68767 (0.40/0.45 m only).
- 4 processes; the 4-process split is bit identical to P1's in-process path (checked on 8 bouts). 0.58-0.64 s wall per bout.

| main block, 768 paired bouts | G | H(A3s)/M3s | full A3 |
|---|---|---|---|
| falls / survival | 15 / 0.980 | 17 / 0.978 | **80 / 0.896** |
| knockdowns / falls after a knockdown / P(fall given kd) | 56 / 4 / 0.07 | 60 / 5 / 0.08 | 43 / 25 / **0.58** |
| falls, act_delay 1 (n 368) / act_delay 0 (n 400) | 10 / 5 | 14 / 3 | **71 / 9** |
| head clean per up-minute; R vs G | 4.20 | 3.53; **0.841 [0.772, 0.917]** | 2.71; 0.646 |

- A3 - G fall rate **+0.085 [0.063, 0.108]**; A3 / G knockdowns **0.77 [0.55, 1.04]**; P(fall given kd) A3 - G **+0.51 [0.34, 0.67]**.
- H vs G: survival +0.003 [-0.010, +0.016], knockdowns 1.07 [0.80, 1.46]. The start is G-like in every fall statistic.
- Close range (0.40/0.45 m, 1152 bouts) gives the same fall gap and a *weaker* knockdown ratio (0.56), so R4 stays on the eval mix.

**Frozen R4 contrasts** (equivalence-style, B1 fix 2; gated in M6 R-EQ and P1-R4g, margins and N in section 3.1):

| id | quantity | rule |
|---|---|---|
| gap_A3_G | (A3 - G fall rate), replica minus EO | 90% CI inside [-0.03, +0.03] |
| gap_sign | the replica's A3 - G fall rate | 90% one-sided lower bound > 0 |
| pfk_contrast | (P(fall given kd) A3 - G), replica minus EO | 90% CI inside [-0.20, +0.20] |
| R5_latency | the replica's A3 fall rate, act_delay 1 minus 0 | 90% one-sided lower bound > 0 |

Report only: the A3/G knockdown ratio, the H - G survival CI, G's latency interaction. Deleted: "A3 knockdowns >= 1.5x G",
"A3 survival <= G - 0.03" as a point rule, "H within 0.015 of G".

---

## 1. Measured causes and the lever for each

| cause (measured) | evidence | lever in this plan | CPU check before any $ |
|---|---|---|---|
| **Untrained motor delay** | R4-pre: A3 falls 71/368 with act_delay vs 9/400 without; G 10/368 vs 5/400. P1: G 12/231 vs 1/217 | per-episode `act_delay` flag in the action term and one obs lag k for the whole 111-d actor obs, drawn in a reset event (B3) | M4 G0-6/G0-7; M6 R-EQ R5_latency |
| **Cannot survive a knockdown** (train clips) | R4-pre: P(fall given kd) A3 0.58 vs G 0.07; falls after a knockdown 25 vs 4 | `fb_down_approach` + `fb_down_contact` (new), down-neutral charge, residual legs | M5 G0-12; M6 R-EQ pfk_contrast |
| Lower body tangles with a real body (val clip, frozen evidence) | SYNTHESIS 1A on kfab_204: knockdowns 13.3% vs 5.1%; leg contact 0 N*s/min in training vs 363 at eval | full-body opponent with leg hulls that can fall; attributed knockdown penalty; tangle penalty | M6 R-EQ kd_G; P1-R4g |
| Keypoint channel fitted to a fist attacker | A3s truth-zeros +2.85/min; P2b: conversion removes 0.6% | `KeypointSource("truth")` on real links (KP:18-21), bit-checked | M4 G0-5 |
| Fall rule 70 deg vs eval z < 0.45 / 60 deg | about 1 pp | `fell_eval` | M4 G0-8 |
| Delta warp adds reach; eval vz cannot | glove-head at apex 0.36 m eval vs 0.21 m train | the eval's StandOff + ArmWarp_vz; jitter never adds reach | M3 G0-4 |
| Through-the-arm follow-through | 96% of A3's clean head hits come after an arm touch | a 29 kg body behind the arm; HIT_R2 pay cancelled by any same-side limb landing (m2) | M5 G0-11 |

Kept from run 3: A3s's arms (R 0.84 vs G on train clips against this opponent, R4-pre), G's legs (survival 0.980 on train
clips), the P1-passing start H(A3s)/M3s.

---

## 2. The chosen design (compact spec; D2 with the grafts and v3 fixes)

### 2.1 Scene
- **Entities:** `robot` (defender, unchanged indices and ONNX metadata), then `opponent` = a second `get_r1_robot_cfg()`
  (actuators kp 100/40/100/40/20, kd 2/2/2/2/1, caps 60/50/60/60/33, armature 0.01; equal to the eval's, C1). Never
  randomised, never delayed.
- **Collision bits** (MOD:65-99): defender capsules 0/1, defender shells 0/4, **floor 1/3**, opponent capsules 2/2, opponent
  shells 4/0. Shells condim 1, solref (0.01, 1), solimp as MOD:130-133. Feet condim 3, priority 1, mu 0.6.
  - **m9:** mjlab's terrain compiles as 1/**1** (C1). `spec_fn` sets 3, and M9 asserts it **on mjlab's own compiled model**.
- **Hulls (m8):** farthest-point hulls; **256 points (or the exact hull if smaller) for head_pitch, head_yaw, waist_yaw, both
  elbows and both wrists**, 64 points elsewhere. Gate: max Hausdorff distance to the full eval hull <= 3 mm on those links and
  <= 1 cm elsewhere, all 32 links of both robots. Never `maxhullvert`. P0 prices {this mix, all 256, full}.
- **Sizes:** `njmax` 600, `nconmax` 128; overflow and NaN asserted at P0/P1.
- **Sim options pinned to the eval (B4), asserted at P0:** timestep 0.005, implicitfast, pyramidal cone, impratio 1,
  **`ccd_iterations` 35** (mjlab default 50, eval 35). Solver iterations/ls: chosen in M6 from the fallback list (10/20 -> 25/30 ->
  50/50 -> eval 100/50). The pod records `mujoco` / `mujoco_warp` versions vs local 3.17.0 and reruns M9's model diff with the
  pod's `mujoco`.

### 2.2 Opponent driver (`gpu/fb_opponent.py`, a CommandTerm with a pluggable `driver`)
- **TrackerDriver(T0):** `model_13749.pt` actor (132 -> 512-256-128 -> 24, ELU, normaliser), deterministic. Obs order
  OPP:307-309. Target = default + 0.25 x cap / kp x a.
- Clock `round((t_ep + t0 + dt) x 50) + aim_offset`, looping; facing tau 0.5 s; steering lag 0.5 s, cap 0.3 m/s, axis fixed at
  step 0 (OPP:286-306).
- **Timing:** `_update_command` runs after the single `sim.forward()` (mjlab `manager_based_rl_env.py` step body), which is the
  eval's order (defender target written, `mj_kinematics`, tracker acts, 4 substeps).
- **No mid-episode resample (B3):** `resampling_time_range = (1e9, 1e9)`, so `CommandTerm.compute()` never calls
  `_resample` inside an episode; placement happens only from `command_manager.reset`. Asserted at P1 (command_counter == 1 per
  episode).
- **Placement** in `_resample_command` (called from `command_manager.reset`, which runs after the reset events): read the
  defender root from `sim.data.qpos` (written by the reset event, not derived), draw (clip, t0 from the clean-start mask,
  distance), write the opponent root and joints with finite-difference qvel (OPP:77-82); zero the tracker's last action, yaw and
  spot state. No derived quantity is read here.
- **First-ever reset:** mjlab's `env.reset()` does not run `command_manager.compute`, so the task's env wrapper calls
  `command_manager.compute(dt=0.0)` once after the initial reset (time_left unchanged, no resample, targets computed on the
  forwarded state). P1 asserts the opponent targets are non-zero and equal a fresh tracker call at step 0.
- **Downed opponent:** never reset; its tracker keeps acting. Two flags:
  - `opp_down_now` = pelvis up-z < 0.5 or z < 0.25 (EO:86), **non-sticky**, recomputed every step (eval's `both-up` logic,
    `eval_opponent.py:192-202`);
  - `opp_down_ever` (sticky) for metrics and the knockdown event edge only.
- **Aim:** torch port of `standoff+armwarp_vz` (AIM:116-184): batched 5-joint FK and Jacobian, 12 damped least-squares
  iterations, x replaced by the reference palm's forward distance, clip +-0.8 rad / 95% of range, smoothstep, gates tilt > 15 deg
  and a strike-limb touch within 0.3 s. `torch.compile`d. **m12:** `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`, `torch.set_float32_matmul_precision("highest")`, asserted at P0.
- **Population per episode (D2 3.7):**

  | share | controller | motion | aim | distance | pushes (m5) |
  |---|---|---|---|---|---|
  | 40% | T0 | train clips | nominal eval aim | eval 0.40-0.55 | **none** |
  | 10% | T0 | train clips | shadow | eval | **none** |
  | 35% | T1 | L1 | aim + jitter | U(0.35, 0.65) | <= 0.3 m/s, half rate |
  | 15% | T1 | L1 | aim + jitter + opponent DR (mass, kp x U(0.9, 1.1), foot mu U(0.5, 1.25)) | U(0.35, 0.65) | <= 0.3 m/s, half rate |

  - Iterations 1-100 of every run: 100% T0 nominal, no pushes (the in-distribution WATCH baseline, m6).
  - T1 fails a gate -> 100% T0; mirrors/rate warps then only if T0 performs them (D3 7.2).
- **Library L1** (train clips only): L/R mirror; per-strike time warp U(0.7, 1.3); approach re-angling (hooks +-40 deg,
  straights +-20 deg, uppercuts +30..+70 deg) with **no added reach**; 2-3 strike combos. Priors fixed in this file.
  Disclosure: re-angling widens coverage toward rising hooks, a class SYNTHESIS identified on kfab_204.
- **T1 = `R1-Track-Spar`:** T0 fine-tuned on L1 + originals; gates T1-a..e (D2 9.3) with the powered T1-c of section 3.1.
  T1 never sees val2/test2 (m18).

### 2.3 Defender observations, latency, DR, termination
- **`EvalActorObs`:** one class term, 111 dims (the warm start's 83 in r1gym order, then the 28-d keypoint block), a ring of
  the last 3 clean rows, uniform noise added after the delay (SEN:9-13). mjlab's `delay_max_lag` = 0. The critic is undelayed.
- **Reset ordering (B3), verified against mjlab 1.2.0 `_reset_idx`:** curriculum, `sim.reset`, `scene.reset`, **reset events**,
  observation reset, action reset, reward, metrics, curriculum, **command reset (placement)**, event and termination reset;
  then `write_data_to_sim`, **one `sim.forward()`**, `command_manager.compute`, observation compute. Therefore:
  1. **`fb_draw_latency`, a reset-mode event listed first**, draws k ~ U{0, 1, 2} and act_delay ~ Bernoulli(0.5) into
     env-level buffers (`env.fb_k`, `env.fb_act_delay`). Events run before every manager reset, so no reset reads a stale draw.
  2. `EvalActorObs.reset(env_ids)` only sets `needs_prefill[env_ids] = True` and zeroes `KeypointHistory`'s previous points.
     It computes nothing (positions are stale and the opponent is not yet placed).
  3. `EvalActorObs.compute()` (after the forward and the command compute) builds the clean row; for `needs_prefill` envs it
     fills the ring with k + 1 copies of that row (SEN:9-10 semantics: the eval pre-fills with the true reset obs after
     placement and `mj_forward`), sets keypoint velocities to 0 as `KeypointSource.reset` does, clears the flag, then returns
     `ring[k]` + noise.
  4. `HybridDelayedAction.reset(env_ids)`: a_prev = 0; reads `env.fb_act_delay` (already drawn).
  - Tests: M4 G0-6b runs this exact call order in the CPU replica (section 3); P1 asserts on the pod that at the first
    observation of every episode all ring rows equal the fresh row and the keypoint block equals one computed from the placed
    opponent; G1m (if ever approved) repeats it in mjlab on Warp's CPU device.
- **`HybridDelayedAction` (m11):** the action term only does `applied = where(act_delay, a_prev, a)`, a_prev <- a, and exposes
  `applied` to the `actions` obs slot (ENV:96, :125). **The composition G(M3s(obs83)) + residual + arm net lives in the actor**,
  which consumes the same delayed, noisy 111-d obs as at eval. A unit test asserts the action term never reads env observations.
- **Keypoints:** opponent `{l,r}_wrist_roll + (0.08, 0, 0)`, `{l,r}_elbow_link` origin, `torso_link + (0.01, 0, 0.357)`, in the
  defender's `torso_link` frame; FOV mask, heading yaw, velocity, `keypoint_block` (ST:305-344). No noise. Blind twin: 28 zeros.
- **DR per reset, defender only (ENV:187-199), written as the eval writes it (m7):** `fb_dr.py` writes `body_mass`,
  `body_ipos`, `dof_armature`, friction, gains and caps **directly into the model fields with no `set_const` recompute** (mjlab's
  `dr.body_mass` / `dr.body_com_offset` trigger `set_const`; the eval never calls `mj_setConst`). Ranges: mass x U(0.85, 1.15);
  torso + U(-0.5, 2.0) kg; pelvis and torso COM +-2.5 cm; foot mu U(0.5, 1.25); kp and kd x U(0.8, 1.2); caps x U(0.8, 1.05);
  armature x U(0.5, 1.5). No encoder bias, no startup events.
- **Pushes (m5):** none in T0 rows; in T1 rows velocity pushes capped at 0.3 m/s at half rate, no 100 N shoves. Knockdown,
  tangle and post-knockdown terms are masked for 0.5 s after a push. All CPU gates: pushes off.
- **Termination:** `fell_eval` (pelvis z < 0.45 or tilt > 60 deg); `ring_out` 1 m (training only); nan; 20 s time-out,
  bootstrapped. An opponent fall never terminates.
- **Outer loops:** face the opponent's pelvis bearing (ENV:133-154), hold the spot (deadband 0.25/0.08, gain 1.5).

### 2.4 Hits and rewards
- **`FullBodyHitAccount(striker, target)`** (symmetric): eight strike-limb contact sensors x 5 zones (HIT:29-36), a `tangle`
  sensor (defender leg and pelvis shells vs the opponent subtree), closing speed from a per-substep velocity ring read after
  `mj_step`, HitMeter rules (HIT:51-60), intent windows. Opponent -> defender drives rewards; defender -> opponent is logged.
- **Rewards** (x 0.02 s; "he" = one clean head hit = 0.8):

  | term | weight | stage | v3 change |
  |---|---|---|---|
  | spar_head_clean | -40 | A, B | |
  | spar_head_impulse | -20 | A, B | |
  | spar_torso_impulse | -5 | A, B | |
  | spar_block (HIT_R2) | +40 | A, B | **m2:** run 2's clause verbatim: pay only an arm-first intercept by apex + 0.05 s **and only if nothing landed in the window after the touch** (`gpu/sparring_task.py:116-120`, `_finish` :588-596); additionally cancelled when **any same-side strike limb (hand or forearm)** lands on head or torso in the window |
  | spar_clearance | +0.5 | **A only** | **m3:** dropped in B (in B it pays the lean-back) |
  | spar_ring | +2.5 | A, B | |
  | spar_alive | +4 | A, B | |
  | base Unitree/RS terms | unchanged | A, B | |
  | **fb_knockdown** | -120 (-3 he) | A, B | **m1:** event on the rising edge of `opp_down_ever`, charged only if **defender work** on the opponent in the previous 1 s, W = sum over contacts of the defender's **leg, pelvis and torso shells** (not arm shells) of max(0, F_on_opp . v_def_link) dt, is >= W0, **and** no opponent strike limb landed on the defender in the last 0.3 s. W0 is calibrated in M5 G0-12 on EO-recorded knockdowns (the 10th percentile of W over G's knockdowns with contact) and frozen in the prereg. Masked 0.5 s after a push. |
  | **fb_down_neutral** | about -(4-6/min) x 0.8 / 60 per s | A, B | **m4:** accrues while `opp_down_now` (non-sticky) only; the head-clean EMA is kept **per population row** (T0 nominal, T0 shadow, T1, T1+DR) |
  | **fb_down_approach** (new, B1 fix 5) | -2.0 per (m/s x s) | A, B | while `opp_down_now`: max(0, v_pelvis . u), u = unit xy vector from the defender's pelvis to the downed opponent's pelvis |
  | **fb_down_contact** (new, B1 fix 5) | -0.1 per N*s | A, B | while `opp_down_now`: impulse of the defender's leg and pelvis shells on the opponent |
  | fb_leg_tangle | -0.05 per N*s | A, B | masked 0.5 s after a push |
  | fb_lean_away | -0.01 x max(0, lean - 5 deg)^2 | B | |
  | res_l2 | -0.5 annealed to -0.1 over B 0-2000 | B | |

  The two new terms target the route R4-pre found (stepping into or onto a downed body); their per-second magnitude is checked
  in M10 (within 2x of this table) and both are monitored in E6.

### 2.5 Policy and stages
- **Start (sighted):** H(A3s)/M3s: legs G `model_7496.pt` frozen with the M3s rewiring `g[:, 73:83] = obs[:, 25:35] / scale[14:24]`;
  arms A3s `model_8995.pt`. **Start (blind twin):** H(blind3)/M3s with 28 zeros, identical pipeline.
- **Critic:** run 3's 138 inputs + privileged (opponent root state, `opp_down_now` and its age, intent features, act_delay, k, DR
  scalars, tangle impulse, defender work W), zero-initialised.
- **Stage A0** 100 iterations critic only; **A** 1500 arms only (`ArmOnlyGaussian` dims 14-23), lr 2e-4, entropy 0.003; **B** 4000
  with the residual r = MLP(111 -> 256 -> 128 -> 14), last layer zero, Gaussian over 24 dims, leg log-std init log 0.15, clamp
  <= 0.35.
- **Seeds (B6): three pairs, 42, 43, 44** (sighted + blind each). Candidates per seed: A-final, B-2000, B-4000.
- **Export:** one ONNX 111 -> 24 (G + M3s + arm net + residual), donor metadata; `observation_names` ends `,opp_keypoints` or
  `,opp_keypoints_blind`. Contract T9 (M7).

---

## 3. Build order: every milestone ends in a CPU check on this machine

All code in **new files only** on feat/sparring-m2v2 (`gpu/fb_*.py`, `scripts/fb_*.py`, `tests/test_fb_*.py`, and the files
named below). Runs 1-3 stay bit for bit. The main session commits.

**Local limits:** scratchpad venv python with `PYTHONIOENCODING=utf-8` and `-I`; **at most 4 worker processes**; delete any
MUJOCO_LOG.TXT in the repo; seeds only from 6.4; every block appended to `SEED_LEDGER.md` when first used.

**Measured CPU speed (m14):** the real EO bout runs at **0.58-0.64 s wall per bout on 4 workers** (2.3-2.6 s per
bout-worker; R4-pre). The CPU replica with torch modules at batch 1 is unmeasured; budget **2x EO (1.25 s per bout on 4 workers)**,
pessimistic 3x. M6's wall times below use these.

| # | milestone | new files | done when (CPU check, blocking; N) | CPU wall (4 workers) | effort |
|---|---|---|---|---|---|
| **M0** | Hygiene scaffold, seed asserts, ledger | `tests/test_fb_hygiene.py`, `scripts/fb_seeds.py` | (1) Grep test: `kfab_202/204/205/206`, `r1_kungfu.csv`, the tracker `policy.onnx` never appear in a fullbody task input. (2) **B7 seed guard** `fb_seeds.check(seeds, role)`: every role refuses 21000-21999 (hard_r2 val) and **22000-22511** (hard_r2 test) except role `test` (and only via the test-once guard); refuses 23000-23255 except `screen_val`; refuses 24000-25999 except `track2`; and every role asserts its **clip set** (dev roles: train clips only; `screen_val`: kfab_204 only; `test`: kfab_202/205 only). (3) Disjointness of every 6.4 block from each other and from `SEED_LEDGER.md` used blocks (asserted in the test). | < 1 min | 0.25 d |
| **M1** | Train-only exporters | `gpu/export_fb_motion.py`, `gpu/export_fb_hulls.py` | (1) `assert clips == {kfab_200, 201, 203}` + sha256 manifest. (2) T-ref: every train frame equals `Tracker.frame(c, f)` within 1e-4, indices asserted inside train ranges. (3) Clean-start masks equal `SC.clean_starts` exactly. (4) Strike table = strikes.json train rows (37 hand + 1 kick). (5) **m8:** per-link max Hausdorff to the full eval hull <= 3 mm (head_pitch, head_yaw, waist_yaw, elbows, wrists) and <= 1 cm (all other links), 64 links. | about 10 min | 0.5 d |
| **M2** | Torch tracker, facing, steering, placement | `gpu/fb_opponent.py` (driver), `scripts/fb_record.py` (records CPU eval states, train clips, 60000-60015) | G0-1: torch MLP vs ONNX on 10k recorded obs rows <= 1e-5 **in float32 as deployed** and <= 1e-9 in float64 (m12). G0-2: the 132-d obs <= 1e-6 (float64). G0-3: facing yaw, spot, axis over 16 recorded bouts <= 1e-6. G0-3b: tracker alone in the CPU world, 24 starts x 3 train clips: fall times within 0.1 s of r1gym's `Tracker`, or both stay up. | about 40 min | 1 d |
| **M3** | Torch aim (StandOff + ArmWarp_vz) | `gpu/fb_opponent.py` (AimOp, ArmKinematics) | G0-4 **in float32 and float64** (m12): FK and Jacobian vs `mj_jacSite` <= 1e-6 (f64) / 1e-4 (f32); `AimOp` vs `Aim.reference` on recorded bouts: intents and gates identical, StandOff dist <= 1e-6, warped q <= 1e-4 rad, closest palm-to-target per intent <= 1 mm. Jitter test: palm forward distance at apex unchanged <= 1 mm (no reach added). | about 20 min | 1 d |
| **M4** | Keypoints, `EvalActorObs`, `HybridDelayedAction`, latency event, fall rule, DR writer | `gpu/fb_obs.py`, `gpu/fb_actions.py`, `gpu/fb_dr.py`, `gpu/fb_events.py` | G0-5: keypoint block vs `KeypointSource("truth", delay=k)` on 2000 recorded states, k in {0,1,2}, reset pre-fill included: positions, masks, yaw bit-equal; velocities <= 1e-6; glove-elbow 0.194862 m. G0-6: obs lag constant per episode and shared by all 111 dims; output = `DelayedNoisyObs` with noise 0. **G0-6b (B3): the mjlab reset call order replayed in the replica** (event draw -> obs reset -> action reset -> command reset/placement -> forward -> command compute -> obs compute) on 64 resets including resets right after a fall and right after a knockdown: every ring row at the first obs equals the fresh row; keypoints equal those of the placed opponent; k and act_delay equal the event's draw; zero rows from the previous episode. G0-7: applied(t) = a(t-1) when delayed, first delayed step = 0, `actions` slot = applied; **the action term reads no env obs** (m11). G0-8: fall rule and `opp_down_now` vs `R1.fallen` / EO:86 on 1e4 states. G0-9 (m7): DR ranges = ENV:187-199; opponent fields untouched; **`dof_invweight0`, `body_subtreemass` and constraint regularisation stay nominal** after a DR write (no `set_const`), exactly as in the eval. G0-10: hybrid composition and M3s equal P1's `p1_eval.py` hybrid on recorded obs (exact). | about 15 min | 1 d |
| **M5** | FullBodyHitAccount, attribution, reward terms | `gpu/fb_hits.py`, `gpu/fb_rewards.py` | G0-11 (after M6 R1): on 64 replica bouts (60200-60263) in eval mode, **same-state** comparison: landed counts per (limb, zone) exact vs HitMeter; clean-vs-not agreement >= 0.95; median impulse within 10%. G0-12 (scripted events + recorded EO bouts): an attributed knockdown gives exactly one -2.4 event; a tracker-alone fall gives 0; **a knockdown right after an opponent strike landed (< 0.3 s) gives 0 (m1)**; a blocked strike followed by a stumble with only arm-shell contact gives 0; W0 calibrated on the R4-pre G knockdowns re-recorded with contact logs (60300-60363) and frozen; the down-neutral charge accrues only while `opp_down_now` and stops when the opponent is up again (m4); `fb_down_approach` / `fb_down_contact` are zero while both are up; HIT_R2 pays 0 when a same-side forearm lands after the glove touch (m2); tangle impulse signed and summed per substep; all hit terms masked 0.5 s after a push (m5). | about 30 min | 1 d |
| **M6** | **CPU replica and the replica gates** | `scripts/fb_replica.py` (r1gym two-R1 world driven by the M2-M5 modules, N = 1 per process; switches: hulls, solver, reset distribution, scorer, backend numpy/torch64/torch32), `scripts/fb_eq.py` (gate statistics, section 3.1) | **R1 eval identity (B7):** r1gym's own numpy Tracker/Aim/KeypointSource, full hulls, Newton 100/50, EO's episodes, G and H(A3s)/M3s on 16 bouts (60100-60115): **bit identical** to `EO.run` (fell_s and every HitMeter count). Torch float64 modules: per-step qpos difference <= 1e-9 **until the first defender-opponent shell contact**, then same-state checks only (module outputs on the replica's own states, <= 1e-9). **R0-EO reference:** EO cells for G, H(A3s)/M3s and full A3 on **70000-82287 (12288 bouts each)**. **R-EQ (the gate):** the final replica configuration vs R0-EO on the same episodes, group-sequential looks at **4096 / 8192 / 12288** bouts per controller, all gated tests of 3.1 (survival, head clean, knockdowns, R(H vs G) and the four frozen R4 contrasts). **R2 diagnostics (report, B2):** each ladder step alone vs step 0 (EO = R4-pre), 768 bouts per controller on 67000-67767: (a) torch float32 modules, (b) hulls of 2.1, (c) solver 10/20, (d) training reset distribution, (e) FullBodyHitAccount scoring (same-state, not distributional). **T1-c T0 arm:** replica G vs T0 on 70000-76143 (6144), precomputed here. **T-opp:** opponent tracker-alone self-fall rate with no defender contact (E6 baseline). **R6 (report):** defender self-collision on vs off; eval solver vs chosen. | R1 0.2 h; R0-EO 3 x 12288 x 0.62 s = **6.4 h**; R-EQ mean 3 x 6600 x 1.25 s = **6.9 h** (max 12.8 h); R2 3 x 5 x 768 x 1.25 s = 4.0 h; T1-c T0 arm <= 0.7 h. **About 18 h expected, 24 h max** | 2 d |
| **M7** | Policy pieces and export | `gpu/hybrid_actor.py`, `scripts/export_hybrid_fb.py` | (1) Stage A forward = H(A3s)/M3s bit for bit; leg dims have no gradient. (2) Stage B at init = stage A output bit for bit. (3) Anchor-loss gradient reaches only the residual. (4) T9: ONNX = torch on 1e4 rows (<= 1e-6); 16 dev bouts (60400-60415) through an **unchanged** `EO.run()` bit identical to the replica's composition. | about 15 min | 1 d |
| **M8** | Library L1 and T1 config (CPU part) | `gpu/export_fb_library.py`, `gpu/r1_track_spar_cfg.py` | (1) Manifest: every row's source in TRAIN_CLIPS; sha256. (2) Rejection stats per family; re-angled apex forward distance = original (no reach). (3) T0-alone performability per family, 24 starts each (60500-60599), reported; decides which families need T1. (4) Mirrored R1 joint ranges symmetric (asserted). | about 1 h | 0.75 d |
| **M9** | mjlab task, pod script, WATCH6 | `gpu/fullbody_task.py`, `gpu/pod_fb.sh`, `watch --rule 6`, `tests/test_fb_watch.py` | **m10:** compile mjlab 1.2.0's **real** `Scene` (both entities, prefixes, CollisionCfg, actuators, `spec_fn`) on CPU with `warp`, `mujoco_warp`, `mediapy`, `scipy` stubbed in `sys.modules` (C1 pattern, no install). Diff that MjModel **by geom, body and actuator name** against the replica's model: contype/conaffinity/condim/solref/solimp, priority, friction, hull vertex sets, actuator gains/caps/armature, body order, and `opt` (timestep, integrator, cone, impratio, **ccd_iterations 35**, solver iterations). **Floor conaffinity 3 asserted (m9).** (2) Every task input file is in the M1/M8 manifests. (3) WATCH6 replays: synthetic logs that should stop do; run-3 and blind logs do not trip load checks. (4) Pod script lint + dry run on a fake GPU flag; `taskset` pinning plan (training cores vs background-eval cores) present (m13). | < 30 min | 1.5 d |
| **M10** | Tiny CPU training smoke | `scripts/fb_cpu_smoke.py` (8-16 replica envs; minimal PPO with rsl_rl 5.0.1 hyperparameters and the same distribution classes) | **Plumbing smoke (blocking)**, 8 envs x 24 steps, 20 A0 + 30 A + 30 B iterations, seeds 69000-69099: no NaN; legs bit-frozen in A0/A; residual exactly 0 at B start; value loss falls >= 30% over A0; every reward term fires with the right sign and per-second magnitude within 2x of 2.4 (including `fb_down_approach` / `fb_down_contact` on scripted knockdowns); act_delay and k constant per episode (zero switches); knockdown, tangle, lean, W metrics populated. **Learning canary: report only (m19)**, 16 envs, 400 A iterations, seeds 69100-69999; a "fail" triggers no work and no gate. | about 1 h | 0.5 d |
| **M11** | Prereg frozen and hashed | `results/sparring/prereg_m2_fb.json`, `scripts/m2_verdict.py --run fb` roles | (1) `m2_verdict --dry-run` on synthetic rows gives the expected go/no-go for every screen condition. (2) The test-once guard refuses a second hard_r2 test and refuses without `go = true` against the current hash; **test = 512 bouts, 22000-22511**. (3) The hash covers this file, D2, the L1 priors, WATCH6 thresholds, the R-EQ/R4g/T1-c/LOCO statistics and margins, W0, the seed ledger and the Track 2 decision (approved or not; section 6.2). | < 5 min | 0.5 d |

**Totals:** about **11 agent-days** of code (v2: 9.5; + B3 event/lazy pre-fill, m1 attribution, the new terms, `fb_eq.py`),
about 3.5k new lines; CPU wall **about 20 h expected, 27 h max** on 4 workers (M6 dominates).

**Order and critical path:** M0 -> M1 -> M2 -> M3 -> M4 -> M5 (G0-12) -> M6 R1 -> M5 G0-11 -> M6 R0-EO (can start right after
M0, it needs only EO) -> M6 R-EQ, R2, T1-c T0 arm -> M7 -> M10 -> M11. M8 and M9 run beside M6's CPU wall.
**Start R0-EO first:** it is 6.4 h of pure EO and needs none of the new modules.

**Go/no-go for the top-up:** only when M0-M11 all pass under 3.2's fail handling.

### 3.1 Gate statistics and required N (B2; `critique_checks\gate_power_v3.py` -> `gate_power_v3_all.json`)

**Model.** No assumed rates: the simulation resamples the R4-pre EO rows (768 paired bouts per controller). A replica bout on
the same episode copies the EO bout's outcome with probability q, otherwise it is a fresh bout from the same (clip, distance,
act_delay) stratum. q = 0.1 gives per-bout EO-vs-replica correlations of 0.15 (falls), 0.14 (knockdowns), 0.44 (head clean),
at or below the measured perturbed-replicate correlations (P1(b) M3s vs plain: 0.24-0.31, 0.15-0.20, 0.38-0.45); q = 0 is also
reported. Every gate is TOST on a 90% CI from the paired delta method, which is also the formula `fb_eq.py` uses.

**Margins** (critique B2; all gates two-sided unless marked):

| gate | statistic | margin |
|---|---|---|
| surv_G, surv_H | survival, replica - EO | +-0.015 |
| hc_G, hc_H | head clean per up-minute, replica / EO | x[0.90, 1.10] |
| kd_G | knockdowns, replica / EO | x[0.80, 1.25] |
| R_HvsG | log R(H vs G), replica - EO | +-0.0625 (= +-0.05 at R 0.8) |
| gap_A3_G, gap_sign, pfk_contrast, R5_latency | section 0.R | as frozen there |
| kd_A3 | knockdowns A3, replica / EO | report only (A3's knockdowns are not its signature) |

**Pre-declared wrong replicas** (each must fail >= 80%): W1 survival -0.03 (G and H); W2 head clean x1.2; W2b x0.83; W3
knockdowns x1.6; W3b x0.6; W4 R(H vs G) +0.12; W5 A3 falls like G (no gap); W6 no P(fall given kd) contrast.

**Results (2000 trials each):**

| gate set | N | correct passes | wrong fails |
|---|---|---|---|
| R-EQ, fixed N | 4096 / 8192 / 12288 | 0.46 / 0.92 / 0.99 (bottlenecks kd_G and R_HvsG) | - |
| **R-EQ, group-sequential looks 4096 / 8192 / 12288** (pass at the first look where every gated test passes) | **mean about 6,500**, max 12,288 | **0.995** (q 0: 0.98) | **W1-W6 all 1.00** |
| **P1-R4g (GPU worlds vs R0-EO, unpaired)** | **12,288 GPU worlds per controller vs 12,288 EO** | **0.998** | (same effects; GPU N is cheap) |
| **T1-c** (one-sided: lower 90% bound of kd(T1)/kd(T0) > 0.8, and the point in [0.8, 1.25]) | **6,144** per arm | **0.97** at a true ratio 1.0 | true 0.8: fails 0.95; true 0.6: fails 1.00 |
| **P3 LOCO** (section 4, P3 bars) | **1,152 per clip set** | **0.98** for a good pilot (held-out R 0.80, seen 0.72) | no transfer (held-out 0.98): fails 0.999; no learning (0.95/0.95): fails 0.999; start-equal: fails 1.00 |

v2's gates for comparison (gate_power.json): R2 any band one step false-failed 0.82, five steps ~1.0; R3 0.31; T1-c 0.24 false
fail and passed a dangerous 0.6x T1 32% of the time.

### 3.2 Fail handling (pre-registered, B2)
1. A gate fails -> diagnose with the R2 per-step diagnostics (which step moves which statistic) and fix code, or move to the
   next configuration in the **pre-registered fallback list** (hulls: 2.1 mix -> all 256 -> full; solver: 10/20 -> 25/30 -> 50/50
   -> 100/50). Thresholds and margins never change.
2. Rerun **once** on the fresh reserve block **100000-112287** (EO and replica both).
3. A second fail:
   - **noise-sized** (the 90% CI overlaps the margin): a written report and proceed only with your OK (this is a real decision,
     not a routine question);
   - **conclusive** (the CI lies entirely outside the margin): stop. No pod; find the cause.
4. The same rule applies to P1-R4g and P1-Hit on the pod (reserve: a second set of GPU worlds, same EO reference).

---

## 4. GPU stages, costs, throughput rules (B6; RTX 4090 or equal, >= 16 vCPU)

**Pod (m13):** one RTX 4090 with **>= 16 vCPU** (if none is free: an L40S). Training processes are pinned to their own cores and
background CPU evals (E7, E8, T1-c, LOCO evals) to disjoint cores with `taskset`; the run is launch-bound, so host cores matter.
Price model: $0.89/h central, **$0.99/h pessimistic** (larger hosts).

**Speed model** (unmeasured; P0 measures): defender task central 3.5 s/iteration at 4096 envs, pessimistic 5.5; tracker task 1.2 /
1.6 s. **Throughput rules instead of cuts:** if P0's best setting is > 6 s/iteration, rent a **second pod** and run seed pairs in
parallel (same dollars, half the wall). If > 9 s/iteration, also time an H100/L40S for 20 iterations and use whichever costs less
per iteration. Never drop a seed pair, shorten a run or skip a gate for speed. If P0 shows >= 80% per-process throughput with two
processes per GPU, run two processes per GPU (cost drops; the central line below assumes it does not).

| stage | what | iterations / work | central h / $ | pessimistic h / $ | gate to continue (fixed now) |
|---|---|---|---|---|---|
| **P0** | Setup, uploads (G, A3s, blind3, A3; `model_13749.pt`; manifests; hulls; md5); assert sim options (B4) and tf32 off (m12); versions recorded and M9 model diff rerun with the pod's `mujoco`. Timing smoke: hulls {2.1 mix, all 256, full} x envs {4096, 8192} x solver {M6's choice, 25/30}, a two-process concurrency test; 20 iterations each | about 13 x 20 | 0.8 / 0.71 | 1.1 / 1.09 | No overflow, NaN or `maxmatch` warnings. Pick the most env-steps/s. Throughput rules above. |
| builds (m13a) | 16-18 env builds (A0 -> A -> B restarts x 6 runs, 2 pilots, T1 runs, P0 configs) with Warp graph capture | 2-5 min each | 0.8 / 0.71 | 1.3 / 1.29 | - |
| **P1** | Parity. (1) T0 alone: fall times vs CPU within 0.1 s, 24 starts x 3 clips. (2) **Restricted (B4):** tracker-alone trajectory parity over the contact-free first 0.3 s only (32 worlds, pelvis within 2 cm). (3) **P1-R4g:** G, H(A3s)/M3s and full A3 in the task (T0 nominal rows, eval distances, aimed, pushes off, eval fall rule, delays on) at **12,288 worlds x 1000 steps per controller**, judged by the 3.1 equivalence tests against R0-EO. (4) **P1-Hit:** dump Warp's per-substep contact list and `efc_force` for 32 worlds x 1000 steps, run the CPU HitMeter on exactly those contacts; GPU FullBodyHitAccount must match (clean calls >= 0.95; counts per (limb, zone) exact on >= 90% of worlds). Same-state, so chaos does not matter. (5) **B3 asserts:** at every first observation of an episode all ring rows equal the fresh row and the keypoint block equals one from the placed opponent; command_counter == 1 per episode; act_delay and k shares 0.50 / {1/3}. (6) A 3-iteration rsl_rl smoke of the real runner. | about 25 min GPU + 15 min pod CPU | 0.7 / 0.62 | 1.2 / 1.19 | Every item passes, under 3.2's fail handling. |
| fix cycle | one P0/P1 fix and re-smoke (m13d) | - | 0.5 / 0.45 | 1.0 / 0.99 | - |
| **P2** | T1 fine-tune (4000 it) and T1-LOCO (L1 without kfab_203, 3000 it); T1-e GPU-CPU parity. **T1-c on the pod CPU** (G vs T1 in the replica on **70000-76143**, against the T0 arm precomputed in M6) while the GPU trains T1-LOCO: 6144 bouts x 2.5-5 s / 12 workers = 25-45 min | 7000 at 1.2 s | 2.5 / 2.23 | 3.6 / 3.56 | T1-a..e pass (T1-c as 3.1), else 100% T0. |
| **P3** | **LOCO pilot (B5, B6):** sighted + blind, kfab_203 and its L1 descendants removed from every training tensor, **2000 iterations each** (A0 100, A 800, B 1100). Eval on the pod CPU: held-out kfab_203 at 88000-89151 and seen kfab_200/201 at 89500-90651 (1152 each), pilots only; **the start's and G's EO numbers on these episodes are precomputed locally** before the pod. | 2 x 2000 + 0.3 h eval wait | 4.2 / 3.74 | 6.6 / 6.53 | Bars below. |
| **P4-P6** | **Main pairs, seeds 42, 43, 44:** sighted + blind; A0 100 + A 1500 + B 4000 each. Concurrent on 2 pods if the throughput rule says so. | 6 x 5600 = 33,600 | 32.7 / 29.10 | 51.3 / 50.79 | WATCH6. The blind twin is never stopped. |
| **P7** | Pack, md5, download, terminate | - | 0.3 / 0.27 | 0.3 / 0.30 | - |
| | **Total, best plan, Track 1** | | **42.5 h / about $38** | **66.4 h / about $66** | |
| | with two processes per GPU (if P0 shows it) | | **about $27** | about $45 | |
| | **Track 2 (awaiting approval)** | broad tracker T2 for val2/test2 (eval apparatus only, m18) + train-pool T1' | **+$6-13** | +$15 | |
| | optional same-start twin ablation (D3's twin, seed 42, report only) | 5600 | +$4.8 | +$7.6 | |

**P3 LOCO bars (B5), all relative to the start H(A3s)/M3s on the same episodes, fixed now:**
- held-out kfab_203: R(pilot vs start) point <= 0.90 **and** paired 90% CI upper < 1.0;
- seen kfab_200/201: R(pilot vs start) point <= 0.85 **and** 90% CI upper < 0.95;
- survival (pooled 2304): pilot - start point >= -0.015 **and** 90% lower bound >= -0.04;
- generalisation ratio g = log R_heldout / log R_seen >= 0.4 (point).
- **Proceed** iff all four hold. **Track 2 first** (if approved; otherwise stop and report) iff seen passes but held-out or g
  fails. **Stop and diagnose** iff seen fails or survival fails.
- Reported: R(pilot vs blind pilot) per clip set; g's bootstrap CI.
- **Disclosure:** kfab_203's 10 hand strikes were in run 3's GPU fist library, so the A3s/blind3 arms in the start (and T0)
  trained on it as fist paths. The relative bars measure the pilot's *change* from that start; the contamination cannot be
  removed, only measured against.

**Budget (B6, m13):**
- Spent so far about $5.57; RunPod balance $4.73.
- **Recommended top-up before P0: $65** (covers the pessimistic Track 1 total). Track 2 adds about $15.
- **Check-ins, not stops:** at $45 and at $80 of cumulative M2 v2 pod spend, a written status (spend, s/iteration, gates passed,
  projected remaining). The pod keeps running; the only automatic stops are the gates and WATCH6.

**WATCH6 early stops** (iterations 50, 100, 200, then every 100; 20-iteration windows). A stopped sighted run is exported
`_stopped`; a stopped stage B falls back to the stage-A final.

| rule | stop the sighted run if |
|---|---|
| E0 load | wrong start iteration; arm std != donor's +- 0.01; `leg_dev` > 1e-5 in A0/A; residual != 0 at B start; act_delay share outside 0.50 +- 0.05; k share outside 1/3 +- 0.05; any within-episode lag switch; T0 share outside 0.50 +- 0.03 (100% before iteration 100); command_counter != 1 per episode; clip-manifest hash mismatch |
| E2 hazard (m6) | eval-rule falls + ring_out per live s, **no-push envs only**, > 3 x the run's own iterations 1-100 mean (T0 nominal, no pushes), from iteration 200 |
| E3 std | mean arm std > 0.80; leg std > 0.35 (B) |
| E4 progress | from A-500 / B-1000: time-weighted head clean per live minute > 0.97 x its own iteration 101-120 mean |
| E5 knockdowns (m6) | knockdowns per live minute, no-push envs, > max(1.5 x the run's own iterations 1-100 mean, that mean + 0.15/min) (G's train-clip EO rate is about 0.22/min) |
| E6 opponent health and stance | opponent no-contact self-fall rate > 2 x the T-opp baseline; in B: lean_away mean > 6 deg, tangle N*s/min > 1.3 x the A final, **`fb_down_approach` or `fb_down_contact` per downed second > 1.5 x the A final** (B1 fix 5) |
| E7 eval probe (m15, m16) | pod CPU, background, `EO.run`, **one fixed paired set of 384 train-clip bouts (63000-63383) reused at every probe and every seed**; at A-1500, B-1000, B-2500, B-4000. Stop if survival < 0.94, or R vs G > 0.95 at A-1500 / > 0.85 from B-2500. **kfab_204 is not used during training.** |
| E8 keypoint-use sign | report at B-1000; stop at B-2500 if the policy with zeroed keypoints beats itself with true keypoints by > 10% on E7's bouts |

---

## 5. Selection (dev only, after the pod, local CPU)
- **Block 64000-65535** (1536 bouts: train clips x 4 distances x 128), aimed, through an unchanged `EO.run()` with each exported
  ONNX. All candidates (A-final, B-2000, B-4000) x 3 seeds, plus G. CPU: 10 x 1536 x 0.62 s = **2.6 h**.
- **Eligible:** survival >= 0.965 and knockdown share <= 1.25 x G's on the same bouts.
- **Choice:** lowest head clean R vs G; B over A only if its R is lower by >= 0.05 and its survival >= A's - 0.005; across seeds the
  lowest R; ties to the lower seed. The twin is the blind run of the same seed, stage and iteration. No other checkpoint picking.

---

## 6. Fresh preregistration (`prereg_m2_fb.json`, hashed before P0)

### 6.1 Rules (unchanged, `prereg_m2_run3.json:58`)
1. head R vs warm <= 0.60 AND CI upper < 0.80; 2. head vs blind CI upper < 0.90; 3. torso R <= 1.5; 4. survival >= 0.95;
5. ring drop <= 0.05. All five, on the hard_r2 test.

### 6.2 Clean validation

1. **Track 1, recipe-level clean val (proceeds now): the LOCO pilot (P3)** with the relative bars of section 4.
2. **Candidate screen (conjunctive; a pass is necessary, not evidence):**

   | id | condition |
   |---|---|
   | S1 | val hard_r4, kfab_204 x 4 x 64 = 256 bouts, 23000-23255, aimed: survival >= 0.95, ring drop <= 0.05, R_warm <= 0.80 |
   | S2 | the same with `l_hand@13` windows excluded: R_warm <= 0.80 |
   | S3 | train clips, 56000-56191: R_warm <= 0.80 AND survival >= 0.95 |
   | S4 | knockdown share <= max(1.25 x G's, G's + 0.03), AND R on both-up minutes <= 0.80 (the tackle exploit) |
   | S5 | R(candidate vs twin) <= 1.0 (point) on S1's bouts |

   kfab_204 is touched only here, once, after the pod (m15: no E7 use during training).

3. **Track 2: AWAITING USER APPROVAL (downloads and license confirmation).** Track 1 does not wait for it. If you approve it
   **before M11's prereg hash**, it slots in as follows; if not, the prereg is hashed without it and Track 2 can only come later
   under a new prereg.
   - T2-0: confirm licenses (KungFuAthlete MIT vs Apache-2.0; CMU and ACCAD if used). T2-1: download (needs your approval).
   - T2-2: partition **by source video**, from titles only, committed seed and sha256: 60% train pool / 20% val2 / 20% test2;
     duan-6 routine (IDs 199-206) excluded; 206 quarantined.
   - T2-3: sealed processing writes only per-clip eligibility (>= 4 s kept, >= 2 head-band strikes, tracker stay-up >= 90%);
     val2 needs >= 30 head-band hand strikes; seeds val2 24000-24999, test2 25000-25999 (M0 guard role `track2`).
   - T2-4: **the defender trains only against trackers that never saw val2/test2 (m18)**: T1' = T0 fine-tuned on L1 plus the
     train pool (P2, same T1 gates). **T2** (broad tracker that learns val2/test2 references) is **eval apparatus only**.
   - Prereg changes: S1/S2 replaced by S1' on val2 (same bars); kfab_204 joins the training library (decided in the prereg) and
     stays report-only; test2 is an **additional** confirmation (the kfab test still uses T0).
   - Cost: +$6-13 GPU (T2) and about 2-3 agent-days; P(M2) about +0.02.

### 6.3 Test (m17: decided now)
- hard_r2 test, kfab_202/205, **512 bouts: reps 64, seeds 22000-22511**. Decided blind (no test data exists).
- Why: rule 4's false-fail at a true 0.96 drops from 25% (128) to 13% (512); rule 2's power at a true sighted/blind ratio of 0.75
  rises from 0.50 (128) to 0.97 (512) (gate_power.json). CPU cost about 16 min for G + candidate + twin.
- Run **once**, only on go, under the test-once guard: G + candidate + paired twin in one invocation.

### 6.4 Seed ledger (one ledger; m16; all blocks disjoint; used blocks also in `SEED_LEDGER.md`)

| block | use | status |
|---|---|---|
| 50000-59255 (various), 52000-52255, 53000-53191, 54000-54255, 55000-55691 | earlier analyses, P1, P2, P2b | used / retired |
| 57000-58071 | RUN4_PLAN_v3 E7 reservations | retired, never used |
| 66000-66047, 66100-66291 | CRITIQUE_M2V2 C3 | used |
| **67000-67767, 68000-68767** | **R4-pre** (EO, G/H/A3); 67000-67767 is also R2 diagnostics' step 0 | **used** |
| 60000-60015 | M2 recording | reserved |
| 60100-60115 | M6 R1 identity | reserved |
| 60200-60263 | M5 G0-11 (same-state hit scoring) | reserved |
| 60300-60363 | M5 G0-12 W0 calibration (EO with contact logs) | reserved |
| 60400-60415 | M7 T9 | reserved |
| 60500-60599 | M8 L1 performability | reserved |
| 63000-63383 | E7 fixed paired probe set (all seeds, all probes) | reserved |
| 64000-65535 | selection | reserved |
| 69000-69099 / 69100-69999 | M10 plumbing smoke / canary | reserved |
| **70000-82287** | **M6 R0-EO reference + R-EQ replica (looks at 4096/8192/12288); P1-R4g's EO side; 70000-76143 is T1-c's episode set** | reserved |
| 88000-89151 / 89500-90651 | P3 LOCO held-out kfab_203 / seen kfab_200,201 | reserved |
| 100000-112287 | 3.2 rerun reserve (R-EQ, once) | reserved |
| 56000-56191 | S3 | reserved |
| 23000-23255 | S1/S2 val (kfab_204) | reserved |
| 24000-24999 / 25000-25999 | Track 2 val2 / test2 (only if approved) | reserved |
| **22000-22511** | **test, once** | untouched |

### 6.5 Disclosures
- Everything in RUN4_PLAN_v3 section 6, P1/P2b results and the V3 restatement; R4-pre (train clips only).
- T0 (and T2) weights trained on all clips: eval apparatus. kfab_203's warm-start contamination (section 4, P3).
- The L1 priors, including re-angling toward a class seen on kfab_204.
- The knockdown, attribution (W0), down-neutral, down-approach/contact and tangle terms; pushes only in T1 rows.
- Test-clip tracker-only summary rows and scene titles were seen in recon and informed nothing.
- The LOCO outcome; the test clips are scenes of the train routine (a within-routine claim, not a claim about boxers).

---

## 7. Expected outcome per rule (judgement, given a trained, unstopped candidate)

| rule | P(pass on test) | reasoning |
|---|---|---|
| 4 survival >= 0.95 | 0.82 | The start is 0.978-0.980 on train clips and 0.969 on kfab_204 (P1); training now contains the delay that causes most falls and a priced post-knockdown route. 512 bouts cut the luck: false fail 13% at a true 0.96 (25% at 128). |
| 5 ring drop <= 0.05 | 0.95 | Run 3 was 0.016 with the same ring terms. |
| 3 torso R <= 1.5 | 0.92 | Run 3 halved torso landed; torso impulse is still priced. |
| 1 head R <= 0.60, CI upper < 0.80 | 0.40 | For: A3's guard gives R 0.65 on train clips vs this opponent (R4-pre) even with the wrong training; the evade lever becomes learnable in B. Against: the start is 0.84 (R4-pre) / 0.806 (P1), and the point must reach about 0.6. |
| 2 vs blind, CI upper < 0.90 | 0.42 | For: real keypoints; the real arm telegraphs strikes; 512 bouts give 0.97 power at a true 0.75. Against: the twin gets the same body training; on the start, sight is worth only +0.18/min. |

Rules 1 and 2 correlate (both need vision to pay): jointly about 0.27. With the rest (about 0.72 jointly, partly correlated),
**P(all five | trained, screened) is about 0.45** after conditioning on the screen (survival and R already shown on val).

---

## 8. Probability of passing M2 (from today)

| step | P | reasoning |
|---|---|---|
| CPU gates M0-M11 pass (code fixes and the pre-registered fallback configurations allowed; thresholds not) | 0.80 | A correct replica now passes R-EQ 0.995 of the time (v2: about 0.1). The remaining risk is a real replica difference that even full hulls and the eval solver do not remove (0.85 x 0.95). |
| P0/P1 pass on the pod, including P1-R4g and P1-Hit, after at most one fix cycle | 0.75 | Warp float32 + fewer solver iterations on tangles is the new, now-tested risk; P1-Hit is same-state and should pass. |
| P3 LOCO says proceed (or Track 2 approved and proceeds) | 0.65 | Power 0.98 for a good pilot; the risk is that the recipe does not transfer to an unseen scene. |
| A candidate trained and selected (3 seeds, A fallback) | 0.95 | |
| **P(candidate)** | **about 0.37** | 0.80 x 0.75 x 0.65 x 0.95 |
| Screen S1-S5, given a candidate | 0.50 | LOCO pre-filtered, best of three seeds; kfab_204 optimistic. |
| All five on test, given the screen | 0.45 | Section 7 (512 bouts). |
| **P(M2)** | **about 0.08 (range 0.04-0.14)** | 0.37 x 0.50 x 0.45. With Track 2: about 0.10. v2 claimed 0.10 but its own gates would have stopped a correct replica with probability about 0.9, i.e. about 0.01 as written. |

**Expected GPU spend: about $17** (0.80 reach the pod; of those 0.25 stop at P0/P1 for about $2.5, the rest reach P3 for about
$8 and 0.65 of those continue to the main runs for about $30 more). Full path: about $38 central, $66 pessimistic. Even a failed
M2 leaves a validated two-R1 GPU world and an EO reference set for M3/M4.

---

## 9. Setup for M3 and M4 (unchanged)
- `OpponentCommand.driver`: `TrackerDriver` (M2) -> `ResidualStrikerDriver` (M3) -> `PolicyDriver` (M4).
- `FullBodyHitAccount(striker, target)` and `kp(source, observer)` are symmetric; defender -> opponent is logged now.
- M3 striker: T1 (never T2, m18) + a goal residual on arms, waist and steering; observes 132-d tracker obs + the defender's
  keypoints; reward clean head hits minus own falls and tracking error; trained against a frozen pool (G, H(A3s), the M2
  candidate and its twin).
- M4 cross-play: alternating best response with PFSP; T0 + eval aim stays as the human-motion anchor; CPU judge
  `EO --opponent policy:<onnx>`; Track 2's val2/test2 (if approved) are the league's held-out references.
- The R0-EO set (12,288 bouts x 3 controllers) is the reference every later GPU world is checked against.

## 10. Risks (ranked) and mitigations
1. **Warp physics or scoring differs from the CPU eval in tangles** -> P1-R4g (12,288 worlds) and P1-Hit (same-state); 3.2's
   fallback list (hulls, solver iterations); B4's pinned sim options.
2. **The CPU replica cannot reach equivalence** (hulls or solver) -> the fallback list up to full hulls and the eval solver; P0
   prices the result; never a threshold change.
3. **Warp hull-vs-hull speed or njmax overflow** -> throughput rules (second pod, faster GPU), not cuts; 32-point hulls are no
   longer a fallback for the head and arm links (m8).
4. **Stage B relearns falling** (lean-back, stepping into a downed body) -> residual + anchor, clearance dropped in B (m3), new
   post-knockdown terms, E5/E6, stage-A fallback.
5. **Tackle exploit** -> down-neutral charge on the non-sticky state (m4), attributed knockdown penalty (m1), S4.
6. **T1 more robust than T0** -> T1-c powered (6,144 bouts, one-sided), T0's 50% share.
7. **Rule 2 unreachable** -> 512-bout test, E8 and P3 report it early, Track 2.
8. **mjlab internals** (reset order, command resample, sensor slots) -> B3 design, G0-6b, P1 asserts.
9. **CPU wall** (about 20-27 h for M6) -> R0-EO starts first, in parallel with M1-M5 coding.
10. **Track 2 depends on you** (downloads, licenses).

## 11. Decisions needed from you
1. **Approve the build** (about 11 agent-days of new files on feat/sparring-m2v2, then about 20-27 h of CPU gates). No money is
   spent until M11 passes.
2. **Top up RunPod by about $65** after M11 passes. Check-ins at $45 and $80; no automatic budget stop.
3. **Track 2 (awaiting approval):** may KungFuAthlete clips be downloaded once licenses are confirmed? Needed before M11's hash
   to count for this prereg. About +$6-13 GPU and +2-3 days; P(M2) about +0.02.

Decided in this plan without asking: test size 512 (m17), 3 seed pairs, LOCO at 2000 iterations, no optional installs (G1m and the
real-rsl_rl CPU smoke move to P1 on the pod, as v2 allowed).

---

## 12. Change map: critique item -> fix -> where

| item | fix adopted | where | deviation from the critique |
|---|---|---|---|
| B1 | R4-pre run (2,304 + 2,304 real EO bouts); R4 redefined as replica-vs-EO equivalence on four frozen contrasts; "H within 0.015" deleted; post-knockdown terms; section 1 row 1 and E5 rechecked | 0.R, 1, 2.4, 3.1, 4 (E5, E6), `r4pre\NOTES.md` | Close-distance R4 not adopted: R4-pre shows close range weakens the knockdown contrast and leaves the fall gap unchanged |
| B2 | ladder vs step 0 (diagnostics); TOST with the critique's margins; N by simulation from real EO rows; group-sequential looks; pre-registered fail handling; probabilities recomputed | 3 (M6), 3.1, 3.2, 8 | The critique's "1,500-2,000 bouts per cell" is too few for kd and R(H vs G) at its own margins: 12,288 fixed or sequential mean about 6,500 is needed |
| B3 | latency draws in a first reset event; lazy pre-fill in `compute()`; no mid-episode resample; initial `compute(dt=0)`; G0-6b and P1 asserts | 2.2, 2.3, M4, P1 | Also found: `resampling_time_range` must be infinite and mjlab's `reset()` skips the command compute (two traps the critique did not list) |
| B4 | P1-R4g (12,288 worlds), P1-Hit (same-state), P1(2) restricted to tracker-alone first 0.3 s, sim options pinned and versions recorded | 2.1, P0, P1 | none |
| B5 | relative bars, N 1,152 per clip set, generalisation ratio, start precomputed locally, contamination disclosed | 4 (P3) | N 1,152 instead of >= 576: 576 gives only 0.85 power for a good pilot |
| B6 | 3 seed pairs, pilot 2000 iterations, check-ins at $45/$80, >= 16 vCPU pod, throughput rules | 2.5, 4 | none |
| B7 | role-based seed guard with explicit test block 22000-22511 and clip-set asserts; float64 rule until first shell contact, then same-state | M0, M6 R1 | none |
| m1 | attribution by defender work on non-arm shells; excluded 0.3 s after an opponent strike lands | 2.4, M5 | W0 calibrated on EO knockdowns instead of a fixed 2 N*s |
| m2 | HIT_R2 clause verbatim + any same-side limb cancels | 2.4, M5 | none |
| m3 | spar_clearance dropped in B | 2.4 | none |
| m4 | down-neutral on the non-sticky state; EMA per population row | 2.2, 2.4 | none |
| m5 | no pushes in T0 rows; <= 0.3 m/s elsewhere; terms masked 0.5 s; E2/E5 on no-push envs | 2.2, 2.3, 4 | none |
| m6 | WATCH baselines from the run's own iterations 1-100 | 4 | none |
| m7 | direct field writes, no set_const; G0-9 checks invweight0 nominal | 2.3, M4 | none |
| m8 | 256-point hulls for head/waist/elbows/wrists; Hausdorff gate | 2.1, M1, P0 | none |
| m9 | floor 3 asserted on mjlab's compiled model | 2.1, M9 | none |
| m10 | mjlab's real Scene with stubs, diff by name | M9 | none |
| m11 | composition in the actor; action term reads no obs | 2.3, M4 | none |
| m12 | float32 G0-1/G0-4; tf32 off asserted | M2, M3, 2.2, P0 | none |
| m13 | builds, P3 eval wait, host-core contention, fix cycle priced; >= 16 vCPU; taskset; top-up $65 | 4 | central $38 (critique: $45 with 3 pairs) |
| m14 | real CPU timing (0.62 s per EO bout on 4 workers) used for every CPU wall | 3 | none |
| m15 | kfab_204 removed from E7; train-clip fixed set | 4 (E7), 6.2 | none |
| m16 | one ledger; E7 one fixed 384-bout paired set; critique and R4-pre blocks recorded | 6.4, `SEED_LEDGER.md` | none |
| m17 | test 512 (22000-22511) decided now | 6.3 | none |
| m18 | defender never trains against T2; T1' on the train pool | 6.2 (Track 2) | none |
| m19 | canary report-only | M10 | none |

## Files
- This plan: `C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody\M2V2_PLAN_v3.md`
- R4-pre: `...\m1m2\fullbody\r4pre\NOTES.md`, `R4PRE_RESULT.json`, `r4pre_eval.py`, `analyze_r4pre.py`, `runs\*.pkl`
- Gate power: `...\m1m2\fullbody\critique_checks\gate_power_v3.py`, `gate_power_v3_all.json`
- Seed ledger: `...\m1m2\fullbody\SEED_LEDGER.md`
- Previous: `M2V2_PLAN.md` (v2), `CRITIQUE_M2V2.md`, designs, recon

## Amendment A1 (2026-10-29, main session, after M3 build + independent verify; decided before any re-run)

Two frozen G0-4 gates were shown impossible as specified: the eval's own operator fails them (M3_NOTES.md, M3_VERIFY.md
probes 1 and 2). They are replaced as follows. Every other G0-4 gate is unchanged.

1. **Float32 warped q.** "<= 1e-4 rad vs the eval" is replaced by all three of these:
   - (i) same-input check: AimOp-f32 vs the float64 port on the **same float32-rounded states**, <= 1e-4 rad on every row;
   - (ii) <= 1e-4 rad vs the eval on >= 99.9% of rows;
   - (iii) the closest palm-to-target per intent stays <= 1 mm vs the eval (unchanged).

   Reason: the eval's 12-step bounded IK amplifies float32 input rounding about 3e4x. Any float32 simulator, the GPU included,
   sees this.
2. **Jitter reach.** "Palm forward distance at apex unchanged within 1 mm" is replaced by all four of these:
   - (i) at every warped step, the IK x target equals the reference palm's own and the nominal one (bit-exact);
   - (ii) the mean of (jitter - nominal) forward distance at the aimed apexes has its 90% CI inside +-5 mm;
   - (iii) the maximum forward excess over the clip with jitter is <= nominal's + 5 mm;
   - (iv) T0 rows are bit-equal to no-jitter rows (mixed-mask probe).

   Reason: the eval's vz IK never reaches its target, so any yz change moves x. The quantity that matters, systematic reach
   added, is what (ii) and (iii) bound.
3. **Recorded finding.** At the reference level, the eval's own vz warp can put the palm up to 12.2 cm beyond the clip's
   forward distance, despite aim.py's docstring. The training opponent is the same operator, so this matches the eval and is
   not a train/eval gap. It goes into the M2 v2 write-up.

## Amendment A2 (2026-10-29, main session, after M3A verify; decided before any re-run)

M3A_VERIFY showed that A1's jitter gate passes at only 9 of 20 jitter seeds. Its (iii) tail grows with draws, up to +14 mm of
forward excess over nominal. Training draws millions of intents, so a single-seed pass (option A) is not acceptable.
Restating (iii) as a quantile (option B) would still let the tail through. Decision: make "no added reach" true by
construction on T1 rows, and gate on the distribution.

1. **Reach guard on jittered rows only.** AimOp computes the nominal warp (no jitter) alongside the jittered one on T1 rows.
   - At each warped step, take the palm's forward distance in the opponent torso frame. Use the same FK the vz target uses.
   - If the jittered x exceeds the nominal x, replace the jittered delta by nominal + s x (jittered - nominal), where s is in
     [0, 1].
   - s is the largest value with excess <= 0. Find it by bisection, 8 steps, per limb.
   - T0 rows are untouched and stay eval-exact; A1 (iv) still applies.
   - JITTER ranges are unchanged. The guard, not narrower ranges, carries the reach constraint.
2. **Gates, replacing A1 jitter (ii) and (iii).** Run them over jitter seeds 0-19, on the same recorded states with no new
   simulation.
   - (ii') The pooled mean of (jitter - nominal) forward distance at the aimed apexes has a **bout-cluster** bootstrap 90% CI
     inside +-5 mm.
   - (iii') On **every** warped step of every seed, jittered palm x - nominal palm x <= 1e-6 m (float64) or <= 1e-5 m
     (float32).
   - (v) Report the share of intents where the guard is active and the mean s. If the guard binds on more than 50% of
     jittered intents, report it; the diversity of training strikes would then be mostly vertical and lateral.
   - A1 (i) and (iv) are unchanged.
3. **Recorded finding (extends A1.3).** Over every warped step, not just the apexes, the eval's own warp puts the palm up to
   27.9 cm beyond the clip's forward distance. This goes into the write-up.

## Amendment A3 (2026-10-29, main session, after M3B verify; decided before any re-run)

M3B_VERIFY proved that A2 (ii') cannot pass under A2.1. The one-sided guard sets every apex dx to min(r, 0), so the mean is
(mean r - mean|r|)/2 = -6.58 mm, predicted and measured exactly. The guard also sets s to about 0 on 98% of intents, so it
deletes most of the yz diversity it was meant to keep. A2's gates stay; the mechanism changes.

1. **Target-space reach correction (F'), on T1 rows only.** This applies per warped step and limb, before the A2.1 guard.
   - Keep the jittered target's yz. Shift only its forward (torso-frame x) coordinate so that the palm x after the same IK
     equals the nominal palm x.
   - Use secant iterations on that one scalar, with at most 4 extra IK calls.
   - Then apply the A2.1 bisection guard unchanged, as a safety net for any positive residual. (iii') therefore stays true by
     construction.
   - T0 rows are untouched. JITTER ranges are unchanged. Report the extra IK cost per step.
   - The joint-space Newton projection (verifier option F) is rejected: it moved the arm up to 1.29 rad in one step.
2. **Gates, over jitter seeds 0-19, on the same recorded states.**
   - A1 (i), A1 (iv), A2 (ii') (two-sided +-5 mm, bout-cluster CI) and A2 (iii') are unchanged.
   - (v) Report the share of intents where the guard is active and the mean s. Also report the share of steps where the
     secant correction converged to within 1e-6 m.
   - (vi), new: the mean yz apex spread after the correction and guard is >= 50% of the raw jittered spread. This proves the
     jitter still adds diversity.
   - (vii), new, smoothness: take the per-step change of the commanded arm q (max over arm joints) on T1 warped limb-steps.
     Its 99.9th percentile must be <= 1.25x the same statistic on T0 rows of the same seeds and recording, and its max must
     be <= the T0 max + 0.2 rad.
3. **Fallback, fixed now.** If F' fails (ii'), (vi) or (vii), adopt D instead and re-gate.
   - D sets the arm-target yz/z jitter to zero, keeps the commit_s and depth jitter, and keeps the A2.1 guard.
   - Its gates are A1 (i), A1 (iv), A2 (ii') and A2 (iii'), with (v) reported.
   - If D also fails, stop and report. No threshold is changed, and no JITTER range is tuned on seeds 0-19.

## Amendment A4 (2026-10-30, main session, after M5 verify; decided before any W0 data is seen)

M3 (A3, F' adopted), M4 and M5 are built and verified, and committed as 82c1d29 and e9fdd66. M5_VERIFY left one item open.
The frozen W0 clause names "R4-pre G knockdowns re-recorded (60300-60363)", but 60300-60363 are fresh episodes. They
yield 4 G knockdowns, 2 with contact, so W0 = 0.299 J is the 10th percentile of two values.

1. **W0 uses option (a): the actual R4-pre G knockdown bouts.**
   - **New fb_seeds role `w0_r4pre`.**
     - It allows only controller G, and only the R4-pre episodes in 67000-67767 and 68000-68767 where G's bout had a knockdown.
       There are about 144 such episodes.
     - The list is derived from the R4-pre run outputs before any re-recording, and is stored with its sha256.
   - **Re-recording.** Re-record those bouts with contact logs. The hooked bout must equal the plain bout, as in M5.
   - **W0 = the 10th percentile of W** over those knockdowns that had contact, using M5's contact definition.
   - At least 30 contact knockdowns are required. If there are fewer, stop and report. W0 is not tuned.
   - The 60300-60363 value becomes report only. W0 is then frozen and enters the prereg hash.
2. **Early knockdowns.** For knockdowns before t = 0.5 s, report how many occur among G's R4-pre bouts and among the eval's own
   bouts, and whether the eval reset produces them. This is report only; no rule changes.
3. **Closing speed for hit scoring on GPU: contact-point velocity.**
   - M6 checks whether the contact sensor's pos field is the true contact point under netforce. It passes if pos lies within
     1 cm of MuJoCo's contact position on >= 99% of contacts.
   - If it passes, use the velocity at that point. Otherwise use the palm and foot site velocities.
   - G0-11 runs on the replica with this deployed input. The COM input is report only.
4. **Order.** R0-EO starts first, as in section 3. It uses 4 workers for about 6.4 h; parallel lanes use at most 2 workers each.

## Amendment A5 (2026-10-30, main session, after the R0E, M5W and T2 verifies; decided before any replica or gate data)

1. **W0 and knockdown attribution.** A4.1 was applied exactly as frozen.
   - P10 over the 98 contact knockdowns is 0.0 J, because 32 of them have W = 0. W0 = 0.0 stays the calibrated threshold, and
     the charge keeps M5's W > 0 clause.
   - Two mechanism fixes, not tuned on outcomes:
     - (a) **Numeric floor.** W must exceed W_eps. W_eps = 10 x the max |W_f32 - W_f64| over the 144 w0_r4pre bouts, computed by
       the torch account in float32 and float64 on the same recorded rows. It is re-measured on Warp at P1-Hit, and if the GPU
       floor is larger, the larger value is used. Spurious float32 power can therefore not cost -2.4.
     - (b) **Start window.** Knockdowns with an edge in the first 0.5 s of an episode are never charged. M5W and the R0E verify
       show that about 35% of G's charged knockdowns happen there: the same episodes are early for G, H and A3, and the opponent
       is already falling (68666 falls at 0.12 s with no contact).
   - These knockdowns still count in every eval metric (S4, R-EQ kd gates, E5). Report the charged counts under the final rule.
2. **Track 2 leaves this preregistration.**
   - T2_VERIFY proves that the frozen "val2 needs >= 30 head-band hand strikes" cannot be met by any partition of the 80 clips.
     The cap is 29, even with a perfect tracker.
   - Frozen T0 stays up only 0-0.21 on the unseen forms, so a train pool would need new tracker training for little head-band
     content.
   - Decision: option (a) without the train pool.
     - S1/S2 stay on kfab_204. The LOCO pilot (P3) is the clean recipe-level val. T1' and T2 are not trained.
     - The split (results/sparring/track2_split.json) and the sealed scripts stay committed and sealed for a later prereg.
     - Seeds 24000-25999 stay unused.
     - P3's "Track 2 first" branch becomes "stop and report".
   - Disclose this, and the routine overlap across the split.
3. **R-EQ looks and the T1-c set, by whole balanced shards.** R0-EO episodes are clip-major (192 per shard, 16 per stratum).
   - The R-EQ looks are at the first **22 / 43 / 64 shards: 4224 / 8256 / 12288 bouts.** Each look is at least the planned N,
     so the power is at least 3.1's.
   - **The T1-c episode set is shards 0-31: 6144 bouts, balanced.** It replaces "70000-76143".
   - The R2 diagnostics stay on 67000-67767, which is balanced.

## Amendment A6 (2026-10-30, main session, after the M8 verify and the M6a build; decided before any R-EQ data)

1. **M8 combos (V1).** Implement the stated speed rule. The allowance is max(vmax, the host window's own fastest, the source
   segment's own fastest) + 0.05 rad/s, and the join is never counted. For each splice, the fade length is the shortest one in
   the frozen prior range [0.15, 0.35] s (frame grid) that passes the rule. If none passes, the splice is rejected. This stays
   inside the frozen prior; no option (B) speeds above the retarget's cap. Report how many splices survive.
2. **Mirrored rows (M8 issue 1): option B.** The opponent on mirrored L1 rows, and on mirrored strikes, is driven by the
   mirror-conjugated tracker (obs and action mirrored). That applies to T0, and to T1 where T1 exists. T1 then has to learn only
   the non-mirror families. T0-alone performability is re-reported with the conjugated tracker on mirrored rows, and T1-b's 90%
   bar is judged per family on that basis.
3. **LOCO hygiene (V2).** vmax is computed per clip set. T1-LOCO's library uses kfab_200/201 only.
4. **Disclosure (M8 issue 2).** The accepted warps skew slow (mean speed 0.884), as a structural consequence of export_warps'
   rule.
5. **M6 R1 at 1e-9 (M6a issue 1).** The eval tracker is a float32 ONNX, and the task may not read it, so a float64 torch chain
   cannot match the eval at 1e-9.
   - The 1e-9 gates apply to the **diagnostic configuration**: the torch modules with the eval's tracker session and reference
     arrays plugged in. Measured there: 2e-15.
   - The deployed configuration is gated through the existing M2 G0-1/G0-2 (tracker) and M3 (aim, f32 and f64) gates, and
     distributionally through R-EQ, which is the real gate.
   - Aim rdq is gated at <= 1e-9 x 50 (the IK's measured amplification) and reported.
6. **A4.3 reading.** Literal: over all contacts, 88.3% < 99%, so the deployed closing-speed input is the **palm/foot site
   velocity**. G0-11 passes with it (0.9675 / 0.9744).

## Amendment A7 (2026-10-30, main session, after the M6a verify; decided before any R-EQ data)

M6a measured the replica at R-EQ's configuration (torch32 modules + account, N = 1) at about 40 s per 20-s bout per worker.
At that speed R-EQ takes about 55 h on 4 workers (100 h at most), against the plan's 6.9 h. The modules are op-overhead
bound at batch 1.

1. **Batched replica.**
   - fb_replica runs K bouts in lockstep per process. Each bout has its own MjData. The torch modules (tracker, AimOp,
     EvalActorObs, HybridDelayedAction, account) run once per step on the batch of K, as they will on the GPU.
   - Choose K by measurement, target <= 2x EO per bout. Pursue profiling fixes first: torch threads, small-op overhead, and
     offline scoring of logged sensors, since the account is row-independent.
2. **Batch invariance is reported, not required to be bitwise.**
   - Rows at batch K are compared with N = 1 on 64 bouts per controller (60100-60115 and the M6 R1 block).
   - Report every bitwise difference and its source. Batched float32 matmuls may differ from batch 1, and chaos then amplifies
     that.
   - The **batched configuration is the replica configuration that R-EQ judges.** R-EQ is distributional, and the GPU will run
     batched too. R1 eval identity (numpy, N = 1) and A6.5's diagnostic gates stay as they are.
3. **A5.1 implementation.**
   - W_eps (10 x max |W_f32 - W_f64| over the 144 w0_r4pre bouts) and the 0.5 s start-window no-charge rule go into
     gpu/fb_rewards.py.
   - G0-12 is re-run with scripted cases for both, and the charged counts are reported under the final rule.
4. **If the batched replica is still > 3x EO,** rent CPU (a RunPod CPU pod, >= 32 vCPU) for R-EQ rather than cutting N or
   looks. Report the expected cost first. Thresholds, N and looks never change.

## Amendment A8 (2026-10-30, main session; decided before any R-EQ data)

1. **The heavy M6 stages run on a cloud CPU pod.** These are R-EQ, R2, the T1-c T0 arm and T-opp.
   - The pod is RunPod cpu3g: 32 vCPU, AMD EPYC 9655, 128 GB.
   - Package versions are identical to the laptop venv: mujoco 3.17.0, numpy 2.5.4, torch 2.14.1+cpu, onnxruntime 1.31.0.
   - Paths are rewritten mechanically by pod/port.py.
2. **The platforms are not bitwise identical.** The pod's EO rows for shard 1 differ from the laptop's in nearly every row (G 191 of 192): bout outcomes diverge chaotically, from floating-point differences between CPUs.
   - So **R-EQ compares replica and EO on the same platform.** R0-EO is re-run on the pod into fullbody/r0eo_pod; it is 3 x 12288 bouts, about 0.8 s per bout-core. R-EQ, R2 and T1-c use that reference.
   - The laptop R0-EO stays as the laptop-platform record. It is the reference for anything run on the laptop, such as R1 identity and same-state checks.
   - Pairing by episode is preserved within the pod platform, so 3.1's power analysis holds.
3. **Platform check (report).** The pod's G, H and A3 EO summary rates are compared with the laptop's R0-EO: survival, head clean per up-minute and knockdowns, each with a 90% CI, unpaired. A difference outside 3.1's margins is reported as a platform effect before any R-EQ look. It is not a gate.
4. **Measured speed.** One pod core runs EO at about 3x the laptop's per-core speed: 149-166 s against 438-509 s per 192-bout shard.

## Amendment A9 (2026-10-30, main session, after the M8b and M7 verifies)

1. **ONNX precision (M7 issue 1): option (i).**
   - The exported hybrid graph computes in float64 on the exact float32 weights and returns float32. It matches the torch
     reference bit for bit.
   - No float32 graph can meet T9's 1e-6, because the run-3 donors' own exports differ from their checkpoints by 1.7-2.9e-6.
   - The start H(A3s)/M3s used in LOCO, selection and the test is the exported start ONNX through the same path. The A3s/G
     donors stay as shipped where used as references.
   - Disclose the 2.9e-6 forward gap to float32 training.
2. **Blind flag (M7_VERIFY hazard 3).**
   - The runner factory stores `blind` in every checkpoint. export_hybrid_fb's build() asserts that the stored flag equals
     --blind.
   - The selection and test scripts read the flag, never an argument.
   - This is implemented in M9/M10.
3. **L1 (M8b issues).**
   - T1 trains on the reangle family only (warp 0.917, combo 0.917, mirror 0.958 pass the 90% bar). Mirrored rows are driven by
     the conjugated tracker.
   - Combo rows with no surviving splice stay as rows, with the duplication disclosed.
   - The 4 reangle falls are T0's own fall at kfab_200 t0 2.5 s and stay in. No post-hoc start exclusion.
4. **Design 4.2's p 0.2 mirrored-strike swap** is implemented in M9's AimOp wiring as designed. fb_mirror is wired into
   fb_replica and the task per the M8b contract: the mask is set from the npz `conj` at reset, and the npz reference is fed.

## Amendment A10 (2026-10-30, main session, after R-EQ failed; decided before the rerun)

1. **Result.** R-EQ on the pod (12288 x 3, all three looks) failed **conclusively** on hc_G, hc_H and R_HvsG. Survival matched
   (est 0.000). Every replica row in REQ_CONF has head_clean = head_landed = 0, both in the account and in the HitMeter that
   rides along.
   - R2 isolates the cause: with only `hulls-mix`, head hits are 0 for G, H and A3. Every other single step (torch32, solver
     10x20, train reset, account/site, self-collision) keeps the test's head-hit rates: G hc about 1000 per 768 bouts.
2. **Root cause: a naming bug, not physics.**
   - r1gym/hits.py `r1_zone` identifies the head by **mesh name** (HEAD_MESHES = head_pitch_link, head_yaw_link).
   - fb_replica.build_model and gpu/fullbody_task.py swap in hull meshes named `fbhull_<mesh>`, so the head shells lose their
     zone. fb_hits.zone_of and the task's zone lists key on the same names.
   - The M6a/M6b checks used full hulls (R1 identity) or did not compare zone maps, so they missed it.
3. **Fix.**
   - Hull meshes take the **original mesh names**, and the replaced originals are renamed (e.g. `fborig_<mesh>`). This applies
     in both fb_replica and the mjlab task.
   - New blocking test: for every hulls option (full / mix / all256) and both entities, the HitMeter zone_of / limb_of / keys
     maps, the account's zone_idx and the block-shell set equal those of the full-hull model, geom for geom by body.
   - The M9 model diff adds the same zone-map identity against the replica.
4. **Rerun (3.2 step 2): once, on the fresh reserve block 100000-112287.**
   - EO and replica both run on the pod platform, with the same configuration REQ_CONF and the same looks (A5.3 by shards).
   - R2 is re-run for the hulls-mix step only.
   - Thresholds are unchanged. A second fail is handled by 3.2 step 3.

## Amendment A11 (2026-10-30, main session, after the M9 verify)

1. **mujoco on the GPU pod.** Keep the stack run 3 trained with: unitree_rl_mjlab's pin, mujoco / mujoco-warp 3.5.0. Do not
   install 3.17.0 over it.
   - P0 re-runs the M9 model diff, including zone maps, under the pod's mujoco against the replica model. The replica model is
     exported on the laptop as a pickled spec dump, or rebuilt on the pod.
   - Any mismatch beyond the version's known option defaults stops P0.
   - P1-R4g remains the distributional gate for GPU physics against EO.
2. **"Iterations 1-100".** This is read per training process: A0 entirely, plus the first 100 iterations of A and of B, are
   T0-only. Each stage's own iterations 1-100 serve as the WATCH6 E2/E5 baselines. That is the task's implementation, and is
   fixed now.
3. **Remaining before the top-up for GPU:**
   - R-EQ rerun on the reserve block (A10.4, cloud CPU, needs a small top-up);
   - M10 (CPU training smoke);
   - M11 (prereg hash).
   The P1 / T1-c / LOCO evaluation scripts named by pod_fb.sh (fb_p1.py, fb_t1c.py, fb_loco_eval.py) are written in M10/M11
   and dry-run there.
