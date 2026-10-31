# M4 independent verification (M2V2_PLAN_v3 section 3, M4: G0-5, G0-6, G0-6b, G0-7, G0-8, G0-9, G0-10), 2026-10-30

**Verdict: M4 PASS (ok = true).**
- Every frozen check passes when I re-run it myself from a fresh recording.
- No threshold was changed. Amendments A1-A3 concern M3 only and do not touch M4's gates.
- Reading the code line by line against the eval found no unfixed semantic mismatch. I made no code changes.

**Rules kept:**
- Only new files are present, and the tracked `git diff` is empty. Branch feat/sparring-m2v2.
- Nothing was committed, stashed or branch-switched.
- CPU only, at most 4 workers. $0 spent.
- One ledger row was appended ("M4 verify").

## 1. Frozen check re-run (fresh recording, not the builder's file)
Command (cwd = repo, PYTHONIOENCODING=utf-8):
`python -I scripts/fb_m4_check.py --out build\verify_m4\m4_check_verify.json --m4rec build\verify_m4\rec\fb_m4rec_60000-60015.npz --rec build\rec\fb_rec_60000-60015.npz --p1 ...\gap\run4pre\P1\p1_eval.py --workers 4`

**Result:** exit 0 in 96.8 s, with 39.9 s of that spent recording. Log: `build\verify_m4\m4_check_verify.log`.

**The recording:**
- All 32 hooked bouts equal their plain twins (asserted).
- G is bit-equal to the M2 recording (`same_as_m2_recording` true).
- My new npz is **bit-identical to the builder's** `build\rec\fb_m4rec_60000-60015.npz` on all 353 arrays.

| gate | frozen rule | re-run result |
|---|---|---|
| G0-5 | positions, masks and yaw bit-equal; velocities <= 1e-6; glove-elbow 0.194862 m; 2000 states, k in {0,1,2}, pre-fill included | **PASS.**<ul><li>2000 states in 8 segments, giving 6000 rows.</li><li>Mismatches 0/0/0; velocity max diff 0.0.</li><li>On the eval's delivered block over 15,380 H steps: 0/0/0 and 0.0.</li><li>Glove-elbow distance 0.19486223237 (min) to 0.19486223237 (max).</li></ul> |
| G0-6 | one lag per episode, shared by all 111 dims; output = DelayedNoisyObs with noise 0 | **PASS** on 30,808 steps.<ul><li>proprio83 vs `_raw_obs`: 0.</li><li>Eval noise replayed vs the delivered 83: 0; vs the delivered keypoints: 0.</li><li>Noise 0 vs DelayedNoisyObs(k, 0): 0; vs clean(t-k): 0.</li><li>ks {0,1,2}; k_switches 0.</li></ul> |
| G0-6b | B3 order in the replica, 64 resets incl. after a fall and after a knockdown: ring rows = fresh row, keypoints = the placed opponent's, k/act_delay = the draw, no previous-episode rows | **PASS, 64/64 on all 9 items.**<ul><li>Resets: 2 after a fall, 3 at a knockdown, 3 at knockdown+10, 56 random.</li><li>Negative controls all fail: eager (ring 0/64), late_draw (draw 27/64), no_prefill (ring 0/64), no_forward (keypoints 0/64).</li></ul> |
| G0-7 | applied = a(t-1) when delayed, first delayed step = 0, actions slot = applied, reads no obs | **PASS.**<ul><li>applied/targets/slot mismatches: 0/0/0.</li><li>First delayed step is 0 in 18/18 delayed bouts; act_switches 0.</li><li>Stub and AST: no obs read.</li></ul> |
| G0-8 | fall rule and opp_down_now vs R1.fallen / EO:86 on 1e4 states | **PASS** on 40,776 states (30,776 recorded + 10,000 random).<ul><li>All mismatches 0, including EO's both-up intervals and first-fall times.</li></ul> |
| G0-9 | ranges = ENV:187-199; opponent untouched; invweight0, subtreemass and constraint regularisation nominal (no set_const) | **PASS.**<ul><li>Fields and act_delay vs `_dr_sample_into_model`: 0 mismatches over 216 seeds.</li><li>Ranges inside, reaching both ends.</li><li>Only the 7 FIELDS change, with 0 off-defender rows.</li><li>The 5 constants are nominal (ours and the eval's).</li><li>efc_R/D/KBIP/diagA equal the eval DR model's on a 69-row contact state.</li><li>Negative control: mj_setConst changes all of them.</li></ul> |
| G0-10 | composition and M3s = P1's hybrid on recorded obs (exact) | **PASS.** 15,364 rows; compose and m3s_input mismatches 0/0. |

## 2. Independent probe (`build\verify_m4\probe_v1.py`, recorded states only, no simulation)
**(a) An independent G0-6 loop.**
- It runs one EvalActorObs per bout, separate from `fb_m4_check.pipeline`, with the eval's noise replayed.
- Results: 0 mismatches against the delivered 83 and 0 against the delivered keypoint block, on 30,808 rows.

**(b) Command and phase coverage of the recording.**
- The outer-loop command is non-zero on **every** step, including all 32 reset obs; the eval's `_face` runs at reset.
- The phase is on for 30,036 of 30,808 steps.
- So proprio83's phase and command path is fully exercised by G0-6, reset rows included.
- G0-6b sets the reset command to 0 on both sides. That is a simplification of the replica (the outer loops are not M4), not a module gap. M6 must feed the real reset command.

**(c) The float32 (GPU) path on all 30,808 recorded states, against float64.**
- Keypoint positions differ by at most 5.5e-7 m and yaw by 7.9e-7 rad, with 0 mask flips.
- EvalActorObs and HybridDelayedAction run in float32 end to end.
- Feeding float32 poses with the default kp_dtype float64 also works: the history buffer is replaced by float32 on the first block.

## 3. Line-by-line review against the eval (no mismatch found)
- **Delay (SEN:9-13).** `ring[k]`, with the newest row at index 0, equals `deque(maxlen=k+1)[0]`.
  - The pre-fill writes k+1 (in fact all 3) copies of the fresh row, and noise is added after the delay.
  - Output = float32 row (cast to float64) + float64 noise, the eval's arithmetic.
  - Noise half-widths equal `R1HitEnv.noise` (mjlab contract), with 0 on the keypoints.
- **Keypoints (KP:18-21, 49-77, 96-120).**
  - Bodies, offsets, the T frame, heading, wrap, the camera FOV test and pack (zero-plus-mask, yaw present if any keypoint is) all match the eval.
  - The velocity uses the previous control step's T-frame points, with 0 on the first block after a reset, as `KeypointSource.reset` (`prev = None`) does.
  - The velocity is computed undelayed and then delayed with the row.
  - `rowmat` and `fma` reproduce numpy's FMA matmul. It is verified against Fraction and numpy in the suite; the probe is machine-specific, as M4_NOTES says.
- **proprio83 (ENV:50-58).**
  - Gravity is −R[2,:] from the qpos quaternion, equal to `R.T @ (0,0,-1)`.
  - Angular velocity is qvel[3:6] (body frame).
  - The phase is `fmod(t·dt, 0.6)` (Python's `%` for t >= 0) in the same operation order.
  - The actions slot is the applied action.
- **Action (ENV:94-98, :125).**
  - Clip, then cast to float32; applied = where(delay, a_prev, a); target = home + scale·applied.
  - The EMA with alpha 0 is exact; alpha == 0 is asserted on every recorded bout.
  - a_prev is zeroed at reset, and act_delay is read only at reset.
- **Latency (EO:103, ENV:199).** k ~ U{0,1,2} and act_delay ~ Bernoulli(0.5), drawn in a reset event.
- **Reset order, checked against the mjlab 1.2.0 source** (`run2b\src\mjlab_x\mjlab\envs\manager_based_rl_env.py`):
  - In `_reset_idx`: reset events, then observation reset, then action reset, …, then command reset.
  - In `step`: `write_data_to_sim`, a single `sim.forward()`, `command_manager.compute`, `sim.sense`, then observation compute.
  - In `reset()`: `_reset_idx`, forward, then obs compute, with no command compute.
  - This matches `B3_ORDER` and the lazy pre-fill design.
- **Fall rules (MOD:174-176, EO:86, :192-202).**
  - `up_z` follows mju_quat2Mat's [2,2] term order.
  - The constants are the cfg's and EO's own floats.
  - `now` is non-sticky; `edge` is the rising edge of `ever`.
- **DR (ENV:187-199).** It matches the eval in four respects:
  - the draw order and operations: mass, then torso add, then ipos, then one foot mu, then kp/kd/cap, then armature;
  - nominal is restored first;
  - only the defender is written;
  - there is no set_const.

## 4. Notes for M6/M9 (integration hazards, not M4 defects)
1. **The stateful obs term must advance exactly once per control step.**
   - mjlab's `ObservationManager.compute(update_history=False)` returns its cache. However, every real compute calls the term, and so does `_prepare_terms`' shape probe at construction.
   - The startup probe is harmless, because `env.reset()` re-flags every env.
   - A critic term must read `EvalActorObs.clean` from the same step: put the actor group first, or compute both from one shared step.
   - In blind mode `clean` has the keypoints zeroed. If the blind twin's critic should keep privileged keypoints, M9 must take them from elsewhere.
2. **`DRIndex`'s default `is_defender` tests for the `opp_` prefix (the eval model).**
   - mjlab prefixes entities as `robot/` and `opponent/`.
   - M9 must pass `is_defender` (and `prefix`) explicitly, or the opponent would be randomised.
   - G0-9's "opponent untouched" is proven on the eval model only. M9's model diff should re-assert it on mjlab's compiled model.
3. **Where `OppDown.update` runs.**
   - It must run in the termination/reward phase, on the one-substep-stale quantities, to reproduce EO's both-up test.
   - If it is called after the forward (for example in the command compute), it reads fresh kinematics, one substep different from EO.
4. **`fb_draw_latency` must be the first reset event,** and no `min_step_count_between_reset` may skip it.
5. **Inherited from M4_NOTES (confirmed).**
   - base_ang_vel comes from the free joint's qvel; gravity comes from the root quaternion.
   - `delay_max_lag` is 0 and there is no DelayedActuatorCfg.
   - Float32 single rounding gives a difference of at most 1 ulp (M3s inputs 3.4e-7).
   - The torch .pt actors against the ONNX actors is M7's T9.

## 5. Tests and housekeeping
- Full suite with the frozen command: **300 passed, 2 skipped** in 615 s (the 2 skips are the env-gated tests) (`build\verify_m4\suite_verify.log`).
- With FB_P1_EVAL and FB_SEED_LEDGER set, the env-gated tests: `tests/test_fb_actions.py tests/test_fb_m4.py tests/test_fb_hygiene.py` **62 passed**, 0 skipped (`build\verify_m4\gated_verify.log`).
- `fb_seeds --audit --scan`, after my ledger row: ok, 19 roles disjoint, ledger consistent, 17 task inputs clean.
- Seed hygiene:
  - The check and the tests simulate only 60000-60015 in role `record`, train clips only.
  - `test_fb_m4` passes 21000/22000 only to assert that `check()` refuses them before any simulation.
  - No hard_r2 test seed, no kfab_202/205/204 and no val seed is touched.
- MUJOCO_LOG.TXT: the suite wrote one (untracked; the known empty-prefix warning); deleted. There is no `Users*` folder in the repo.
