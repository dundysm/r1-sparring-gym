# M3 independent verification (M2V2_PLAN_v3 M3, G0-4), 2026-10-29

**Verdict: ok = false.** Two frozen G0-4 gates fail: float32 warped q <= 1e-4 rad, and jitter forward distance <= 1 mm. I
reproduced both failures exactly, and I confirmed independently that both are **impossible as specified**: the eval's own
operator behaves the same way. I found **no semantic mismatch** between `AimOp`/`ArmKinematics` and r1gym/aim.py, and I fixed
nothing. Both gates need a plan decision; the builder's options in M3_NOTES.md stand. All other frozen checks pass.

## What I re-ran
- **The full check, with a fresh re-recording** (`scripts/fb_m3_check.py`, unchanged; new aimrec in `build\verify_m3\rec\`):
  - `same_as_m2_recording: true`.
  - `m3_check_v.json` equals the builder's `m3_check.json` in every field except wall time. Exit 1, with the same two gates
    failing.
  - The new recording is bit-identical to the builder's `fb_aimrec_60000-60015.npz`: every array, plus the intents and rows
    metadata.
- **Full suite** (prescribed command, FB_SEED_LEDGER set): **264 passed**, exit 0 (`build\verify_m3\suite_v.log`). This includes
  the 12 tests in test_fb_aim.py.
- **Housekeeping:**
  - `fb_seeds.py --audit --scan`: ok, ledger consistent, 12 task inputs clean.
  - `git diff` on tracked files is empty. Only untracked new files exist, and no MUJOCO_LOG.TXT is left.
  - CPU only, at most 4 workers, $0.
- **Seeds:** no new block. 60000-60015 were re-run in role `record`, and I appended a row to SEED_LEDGER.md. No val or test seed,
  no test clip and no kfab_204 was used.

## Gate results (identical to the build)
| gate | f64 | f32 (f32 states/outputs, f64 aim arithmetic) |
|---|---|---|
| params = AIM_P | PASS | |
| FK / Jacobian vs Aim._palm_b, Rt^T mj_jacSite | PASS 8.9e-16 / 9.4e-16 | PASS 1.3e-7 / 1.4e-7 |
| intents and gates identical (482: 226 aimed, 172 tilt, 84 touch) | PASS, times bit-equal | PASS |
| StandOff dist <= 1e-6 | PASS 0.0 | PASS 1.3e-8 |
| warped q <= 1e-4 rad | PASS 1.26e-10 | **FAIL** 1.94e-3, 4 of 15,412 rows |
| closest palm-to-target <= 1 mm | PASS 7.8e-16 | PASS 8.0e-8 |
| ring 8 = 64 slots, 0 overflow | PASS | |
| jitter: forward distance at apex <= 1 mm | **FAIL** 0.143 m | **FAIL** 0.143 m |

## Independent probes (`build\verify_m3\probe_v1.py`, output `probe_v1.out`)
These probes do not use AimOp; they use only the eval's own code and the recorded states.

1. **Float32 floor, rebuilt from eval quantities only.**
   - On the 4 missed rows I rebuilt the eval's vz IK target from the recorded opponent torso pose, the eval intent's target, and
     `Aim._palm_b(rq0)` x.
   - With this target, `Aim.ik` reproduces the **eval's recorded warp to <= 1.1e-16** on all 5 limb-rows.
   - Rounding only those inputs to float32 (a 2e-8 to 7e-8 m change in the target) moves the eval's own warp by 8.6e-5,
     2.6e-3, 5.7e-4 and 5.7e-4 rad.
   - So the eval IK, not the port, amplifies float32 rounding beyond 1e-4. This confirms the builder's mechanism: bounds active,
     12 iterations, not converged.
2. **Jitter premise, eval only.**
   - I ran MuJoCo FK on the eval's recorded warped `rq` against the clip's `rq0`, at each aimed apex.
   - Result: 218 apexes, **212 move the palm's forward distance by more than 1 mm**. The median is 4.07 cm and the range is
     -28.6 to +12.2 cm, the same as the builder's.
   - The eval's own full-target IK residual at the apex has a median of 20 cm.
   - So "vz holds x" is false for the eval. Under either reading of the gate (jitter vs nominal, or warp vs clip), it cannot
     hold while the yz target or the commit time is jittered.
3. **Library keys:** every strike onset lies in [0, T) with onset <= apex <= end. So keys cannot collide across loop
   occurrences, and replacing `Aim.made` with "last key per strike" (plus dropping repeated onsets) is exact.
4. **Mixed jitter mask** (alternate envs jittered, seed 1):
   - Nominal rows are bit-equal to the run with no jitter (rq and dist), and 1.26e-10 from the eval.
   - Jittered rows differ.
   - So the per-env mask leaks nothing into T0 rows.

## Line-by-line read against r1gym/aim.py, opponent.py and eval_opponent.py
I found no mismatch. Items checked:
- **Schedule:**
  - `tc = fi/rate`, and `occ = floor(tc*rate/T)` in the eval's operation order.
  - Occurrences k and k+1 are visited k-major, then in library order, which preserves the eval's intent order and its
    first-made tie-break.
  - Window `tc <= key/rate <= tc + 0.8`; `sim(f) = (now + (kT+f)/rate) - tc`.
  - The gate is decided once, at creation. Tilt comes from the opponent torso_link R[2,2]; touch updates touch_t before the gate.
  - Target updates apply to all intents, gated ones included, until committed or t_end + 1 s. Commit is set at
    `now >= t_apex - commit_s`.
- **StandOff:** `min(home, clip(reach - depth, 0.35, home))` over aimed intents with t_on - 0.8 <= now <= t_end. Shadow rows
  return home, which equals the eval's `state.get("dist", replay.distance)`.
- **ArmWarp:**
  - Intent selection: active window `[t_on - 0.8, t_end + 0.25]`, then w > 0, then the highest weight per limb, the first made on
    a tie.
  - Target: the target is taken in the opponent's torso frame (R^T), and x is replaced by the FK x of the **unwarped** rq.
  - IK: `lo/hi` as in Aim.ik, then the same 12 damped steps with the prior term.
  - Output: `delta = w (qa - rq)`, with `rdq += (delta - prev)/dt` and prev reset per bout.
- **Ordering:** aim.reference runs before steering inside Tracker.act (opponent.py 288-303), so a dist fed to
  `TrackerDriver.step(dist=)` in the same step matches.
- **Log:**
  - Covers every intent, gated ones included, over the window `[t_on, t_apex + 0.15]`.
  - Closest palm: computed from FK of the actual joints plus the torso pose, which equals `site_xpos` to 1.2e-15.
  - Arm RMSE: aggregated as sqrt(mean of per-step mean-squares), as `as_dict` does.
  - IK residual: computed once at the apex, from the warped rq.
- **Float precision:** times, clock and gates are float64 in every mode. I/O and compute are split as the docstring says.
- **Seed and clip hygiene:**
  - Motion asserts train clips only.
  - `work()` calls `check(seeds, "record")` before it simulates anything; the test covers 21000, 22000 and 23000.
  - Nothing reads tracker results files apart from aim.json's operator field.

## Notes for the plan decision and later milestones
These are not gate items.
- **Stand-off depth jitter can bring the opponent up to 3 cm closer on a given intent.**
  - Depth U(0, 0.06) vs nominal 0.03: measured dist change -2.9 to +3.0 cm, mean +0.8 mm.
  - This follows design 4.2 and has no systematic bias. Strictly, though, it is per-intent reach through the stand-off, which
    the plan's "jitter never adds reach" (section 1 table) does not cover.
  - It should be part of the jitter-gate decision. Builder option (b), jitter only the depth, does not avoid this; a symmetric
    depth range around 0.03 would bound it.
- **The f32 gate needs a re-statement,** with options (a), (b) or (c) from the notes. On the evidence, (b) is the meaningful
  same-input test: f32 AimOp vs the float64 port on the same float32-rounded states.
- **P0 should time AimOp with float64 compute on the A40,** which has low FP64 throughput. The work is tiny (batched 5x5 solves,
  2 limbs, 12 iterations), but it is unmeasured.
- **Still open, as the builder listed:** the MjTracker+AimOp adapter (M6), the exact `_touching` contact set (M5/M9), the mirror
  jitter knob (M8), and `log_ik` (a host sync, off in training).

## Files
- Re-run: `build\verify_m3\m3_check_v.json`, `m3_check_v.log`, `rec\fb_aimrec_v.npz`, `suite_v.log`.
- Probes: `build\verify_m3\probe_v1.py`, `probe_v1.out`.
- Ledger: one row appended (M3 verify, 60000-60015 re-run in role `record`).
