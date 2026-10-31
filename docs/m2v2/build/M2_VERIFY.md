# M2 independent verification (M2V2_PLAN_v3 section 3, M2), 2026-10-29

**Verdict: PASS (ok = true).** Every frozen check passes on my own re-run, using unchanged thresholds. I read the driver line by line
against `r1gym/opponent.py` (Tracker.act, Tracker.frame, FightReplay.place/qpos_at), `r1gym/env.py` (step/reset order), `r1gym/aim.py`
(clock, reference, dist) and `scripts/eval_opponent.py` / `scripts/eval_tracker.py`. I found no semantic mismatch in the M2 scope.

I found one latent hazard and fixed it in a new file: `MjTracker` now refuses to serve more than one env (details below). There is a
new test for it, and all checks re-ran after the fix. Nothing tracked changed (`git diff` is empty) and nothing was committed. CPU only,
at most 4 workers. MUJOCO_LOG.TXT never appeared in the repo.

## Re-runs (everything frozen, thresholds unchanged)
Outputs are in `...\fullbody\build\verify_m2\`.

| item | result |
|---|---|
| `fb_record.py`, seeds 60000-60015, 4 workers (21.5 s) | **Bit-identical to the build file.** All 35 arrays and every meta key match (`cmp_rec.py`). 16 bouts, 15,412 steps, defender fall 60014 at 8.24 s, opponent knockdowns 60004/60005. Hooked rows equal unhooked rows (the script's own assert). |
| G0-1 f32, <= 1e-5 | PASS 7.63e-6 batched, 6.68e-6 at batch 1. The weights and the folded divisor are bit-equal. onnxruntime reproduces the recorded eval actions exactly. |
| G0-1 f64, <= 1e-9 | PASS 1.07e-14 |
| G0-2 (a), <= 1e-6 | PASS 5.6e-16 |
| G0-2 (b), <= 1e-6 | PASS 2.69e-7. This is M1's anchor quaternion difference of 1.19e-7, which M1 recomputed by `mj_kinematics` without reading the ONNX. Clock and frame are equal at every step. The joint reference is bit-equal. |
| G0-3, <= 1e-6 | PASS: yaw 2.8e-17, spot 1.1e-16, axis 0 |
| G0-3b, fall within 0.1 s or both up | PASS in both f32 and f64. 72 runs each: 67 both up, 5 both fell, max dt 0.00 s. Re-run after the fix gives the same result. Mean wall per 10 s run: eval 0.59-0.64 s, driver 1.2-1.3 s. |
| placement vs FightReplay.place | recorded bouts: qpos 0, qvel 4.4e-16. G0-3b starts: 8.9e-16 / 1.8e-13. |
| `fb_m2_check.py` exit code | 0 both times (before and after the fix) |

## Extra independent check: `TrackerDriver.step()` end to end, batched (`verify_m2\step_e2e.py`)
`fb_m2_check` tests the driver's pieces (`face`, `steer`, `observation`) one at a time. This check calls `step()` itself, as M9/M6 will:
- **Batch:** N = 16, one env per bout, all bouts stepped together. A finished bout leaves the batch and the live envs keep their state.
- **Partial resets:** two groups (odd ids, then even ids).
- **Inputs:** the recorded target and opponent state, the aim's warp passed as `ref`, the stand-off passed as `dist`, and the recorded
  last action.

| dtype | obs vs eval obs64 | action vs eval action | ctrl vs eval d.ctrl | yaw | spot | axis | clock |
|---|---|---|---|---|---|---|---|
| float64 | 2.69e-7 | 7.7e-6 | 2.9e-6 | 5.6e-17 | 1.1e-16 | 0 | equal |
| float32 | 3.8e-6 | 7.6e-6 | 2.2e-6 | 2.4e-6 | 1.2e-6 | 2.5e-9 | equal |

The batched placement with partial resets gives qpos 0 and qvel 4.4e-16. Batching, per-env masks (`yaw_set`, `axis_set`) and the target
formula `default + scale x a` all match the eval.

## Line-by-line review against the eval (M2 scope)
- **Clock.** `round(t x 50) + offset` in float64, round half to even. This is the same as `int(round(t * fps * speed))` and
  `Aim.clock`. `t` is passed in as the eval passes it: `d.time + t0 + dt`.
- **Frame.** `row0 + fi mod T` is equivalent to `Tracker.frame(clip, fi)` (bit-equal at every recorded step). For negative fi, Python
  `%` and `torch.remainder` agree.
- **Reference.** Joints are bit-equal. The anchor differs by at most 1.2e-7, a float32 ulp (M1).
- **Facing.**
  - `want = atan2(to) - pi`.
  - First step: `yaw = want`.
  - Then `yaw += wrap(want - yaw) x min(1, dt / 0.5)`. `wrap` uses atan2(sin, cos) where the eval uses `np.angle(exp(1j x))`, a
    difference of 3e-17.
  - `R = quat2mat(conj(torso_xquat) x (rz x aori))`, with the 6 entries `R[:, :2]` in row-major order.
- **Steering.**
  - Spot starts at (d, 0).
  - The axis is fixed at the first step from (spot - target) / (norm + 1e-9).
  - Speed cap is `min(1, 0.3 / (|v| + 1e-9))`.
  - Steering runs after the aim's `dist` is set in the same step, as in Tracker.act.
  - `pos = xmat^T (world - torso_xpos)`.
- **Obs order.** `rq, rdq, pos, R6, qvel[3:6]` (free-joint angular velocity, body frame), then `q - default, dq, last`. The eval casts
  float64 to float32 at the end, while the driver's float32 path computes in float32 throughout. That is the deployed precision.
- **Actor.**
  - The eval computes `(x - mean) / (std + 0.01)` and the driver uses the float32-folded divisor.
  - Activations are ELU with alpha 1.
  - `time_step` does not reach `actions`: onnxruntime with the recorded `time_step` matches torch, which has no such input, to 7.6e-6.
- **Targets.** `default + 0.25 cap / kp x a`. The constants equal the ONNX metadata, `HOME` and the mjlab caps (test).
- **Placement.**
  - `qpos_at(t0)` and `qpos_at(t0 + 0.005)`: fmod, truncation, hemisphere match, renormalisation.
  - Velocities by `mj_differentiatePos`: free joint, then `subQuat`, then `quat2Vel`, wrapped at pi.
  - Distance is added before interpolation, as in the eval's baked `_canon`. The 8.9e-16 at fractional t0 comes from the `_seamless`
    blend being applied with the distance baked in. Docstring nit: "bit-identical" holds only to about 1e-15 at fractional t0.
  - Bout heading/origin: the rotation is applied to positions and quaternions. The angular qvel stays in the body frame and does not
    change (test).
- **Recorder** (`fb_record`).
  - The `Tracker` instance is the shared `Tracker.get` cache entry. `reference` and `act` are instance overrides and removed in
    `finally`.
  - The numpy proxy captures the only (132,) concatenate in `Tracker.act`. `aim.py` has its own `np`.
  - `standoff+armwarp_vz` has no `select` op, so `offset` stays 0 and `reference` is called once per step (no `_blend`).
  - Episodes are `eval_opponent.episodes` with hits off, per seed (`rng = default_rng(seed)`, env and hit seeds from SeedSequence).
- **Hygiene.**
  - Train clips only (asserted in `Motion`, `fb_record.assign` / `episodes`, `fb_seeds.check('record')`).
  - `aim.json` holds no clip rows (checked).
  - The driver opens no ONNX (test). `fb_m2_check` reads the ONNX only as the reference.
  - `fb_seeds --audit --scan`: ok, 19 roles disjoint, ledger consistent, 11 task inputs clean.

## Fix applied (new file only)
- **`MjTracker` served any MjData** (`gpu/fb_opponent.py`).
  - The eval's Tracker keeps per-env state in `opp_state`. MjTracker keeps yaw, spot, axis and last in one `TrackerDriver(n=1)`
    attached to the FightReplay.
  - `eval_opponent.run` shares one FightReplay between every episode of the same (clip, distance) in a call (`reps[key]`). Two such
    bouts in one call would silently mix their facing, steering and last-action state.
  - The current checks never hit this: eval_tracker builds a replay per run. M6's replica could, if it batches bouts through
    `EO.run`.
  - Fix: `place()` records the MjData it placed into, and `act()` asserts it is acting on that same MjData.
  - New test `test_mjtracker_serves_one_env`. `test_fb_opponent` now has 15 tests, all passing.
  - G0-3b re-ran with the guard and gives identical results (`m2_check_after_fix.json`).

## Notes for later milestones (not M2 defects)
- **M3 aim.**
  - `step(ref=...)` overrides only (rq, rdq). That is exact for `standoff+armwarp_vz`.
  - An operator with `select` would also blend apos/aori (`Aim._blend`). `step` would then need those too.
- **M9 clock.**
  - The eval's `t` is the accumulated `d.time + t0 + dt`. mjlab must pass the same quantity.
  - Training t0 lies on the 0.02 s clean-start grid, so `t x 50` sits near an integer, and an ulp drift cannot flip the rounding.
  - Arbitrary t0 values, as in eval_tracker's starts, can land on .5.
- **M9 inputs.**
  - `ang_vel` must be the opponent free joint's qvel[3:6] (body frame).
  - The target must be the defender pelvis xy after the forward.
  - In float32 with large env origins, pass origin-relative positions, or keep spot/yaw in float64 if training uses widely spaced
    origins.
- **G0-1 f32 margin is 1.3x on CPU** (builder's note, confirmed: 7.63e-6). P0 must repeat G0-1 on the pod's GPU with `precision_ok()`
  asserted.

## Seeds
60000-60015 (role `record`) was re-simulated once for this verification, and test_fb_opponent re-ran 60001 for 1 s. I appended a row
to SEED_LEDGER.md and the audit stays consistent. G0-3b uses eval_tracker's env seed 0 with no draws. No val (21000-21999) or test
(22000-22511) seeds and no test clips were touched. kfab_204 was not simulated.

## Tests
- `test_fb_opponent` + `test_fb_hygiene`: 67 passed before the fix. `test_fb_opponent`: 15 passed after the fix.
- Full suite with the prescribed command, after the fix: **251 passed, 1 skipped** (the FB_SEED_LEDGER-gated test), in 196 s.
  `verify_m2\suite.log` has the output.
- The full suite wrote a MUJOCO_LOG.TXT to the repo root. It held only "spec with an empty prefix" warnings from existing tests, it is
  untracked, and I deleted it.

Correction to the header: MUJOCO_LOG.TXT never appeared during the M2 scripts. The full suite did create one, as described above.
