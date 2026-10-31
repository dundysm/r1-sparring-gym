# M3C verify: M3 under Amendment A3 (M2V2_PLAN_v3, G0-4), 2026-10-30

**Verdict: ok = true.** I agree with the builder.
- Every frozen G0-4 check, as amended by A1, A2 and A3, passes in float64 and float32.
- F' is adopted. Fallback D was correctly not triggered.
- I found no semantic mismatch against the eval. I changed no code, loosened no threshold and tuned no JITTER range.
- Two readings of the amended text are noted below for the main session (A1 (i) under F', and what "spread" means in (vi)). Neither
  changes the verdict.

## What I re-ran (CPU only, at most 4 workers, $0)
| run | result |
|---|---|
| `fb_m3_check.py --aimrec build\rec\fb_aimrec_60000-60015.npz --rec build\rec\fb_rec_60000-60015.npz --workers 4` -> `verify_m3c\v_m3c_check.json` / `.log` | **exit 0** in 2110 s. A recursive diff against the builder's `m3c_check.json` (wall times excluded) shows **0 differences**. `adopted` = "F'", and all 7 ok fields are true |
| `verify_m3c\probe_indep.py`: recorded states, f64. Seeds 0 and 13 use mujoco palm sites. A step-by-step pass on all 16 bouts at seed 0 uses the eval's own `Aim.ik` | see below |
| `verify_m3c\probe_variant.py`: recorded states, f64, jitter seeds **100-103** (outside the gated 0-19) | see below |
| Full suite (`pytest ... --basetemp="...\pytest-tmp" tests`, FB_SEED_LEDGER set) -> `verify_m3c\suite_v.log` | **271 passed**, 473 s |
| `fb_seeds.py --audit SEED_LEDGER.md --scan`, run before and after my ledger row | ok: 19 roles disjoint; ledger consistent; 12 task inputs clean |

**Hygiene:**
- The aim recording loaded is the same file M3A and M3B used (md5 e91038d7..., dated Oct 29 20:34). No episode was simulated.
- Seeds 0-19 and 100-103 are AimOp generator seeds, not episode seeds.
- The suite re-ran 60001 in role `record`, and I appended one ledger row ("M3 A3 verify").
- The suite's MUJOCO_LOG.TXT (the empty-prefix warning) is deleted. There is no `Users*` folder in the repo.
- The tracked diff is empty. Only the workstream's untracked new files exist; nothing was committed or stashed, and the branch is
  feat/sparring-m2v2.
- The three edited files have LF line endings.

## Gates (my run, identical to the builder's)
| gate | f64 | f32 |
|---|---|---|
| params, FK, AimOp f64, AimOp f32 (A1 float32), ring | PASS | PASS |
| A1 (i): vz IK x target = reference palm x = nominal, bit-exact | PASS 20/20 | PASS 20/20 |
| **A2 (ii'):** pooled apex dx, bout-cluster bootstrap 90% CI inside +-5 mm | **PASS**: mean -1.339 mm, CI [-2.084, -0.762] mm (4,360 apexes, 16 bouts) | **PASS**: -1.334 mm, [-2.080, -0.756] mm |
| **A2 (iii'):** jittered x - nominal x at every warped step (132,540 steps) | **PASS**: max 0.0 | **PASS**: max 2.7e-8 m (51 steps > 0 from output rounding, 0 over 1e-5) |
| A1 (iv): mixed mask, F' on the T1 bouts | PASS: T0 rq, rdq and dist bit-equal (7,412 rows); T1 rows differ | PASS |
| **(vi):** pooled mean yz apex spread >= 0.5x raw | **PASS**: 12.98 / 21.88 mm = 0.593 (per seed 0.456-0.708) | PASS 0.593 |
| **(vii):** T1 vs T0 arm q change per step | **PASS**: q99.9 1.8500 vs 1.8476 rad (<= 1.25x); max 1.85 vs 1.85 (<= +0.2) | PASS |
| (v), reported | The guard binds on 91.5% of jittered intents (> 50%, reported) but on 17.8% of T1 warped limb-steps. Mean s is 0.030, and s = 0 on 95.9% of guarded steps. F' converged within 1e-6 m on 67.6% of limb-steps (89,577 / 132,540). IK calls per step: 6 jittered, 1 raw | same |

## Code read against A3 and the eval (`r1gym/aim.py` `_arm`, `_palm_b`, `ik`)
**Diff vs `build\m3c\*.pre_A3.py`.** These are the only changes:
- `fb_opponent.py`: `_guard` becomes `_reach`, with F' added ahead of the unchanged guard. It adds the `reach` modes, the D draw
  switch, the `ik_calls` and `g_conv` counters, and debug fields.
- `fb_m3_check.py`: the A3.2 gates, the D branch and `adopted`.
- `test_fb_aim.py`: the A3 tests and pins.

The T0 path, the scheduler, StandOff and `_log` are untouched.

**Checked line by line:**
- **Nominal dn.** This is A2.1's code, unchanged. The step-by-step probe gives dn = the T0 run's delta to within 2.2e-16 on 6,627
  limb-steps.
- **F' target.** It uses `torch.cat([c, p_t[..., 1:]])`, where p_t's yz is the jittered target in the opponent's current torso
  frame (eval `_arm`: `Rt.T @ (target_w - xt)`). Only x moves, and the yz is bit-equal (probe).
- **Same IK.** `D(c) = w (ik(rq5, target) - rq5)` uses the same weight, slot and reference arm as the eval's `_arm`. Rebuilt with
  the **eval's own `Aim.ik`** on the shifted target, the corrected delta matches to within 1.8e-10 rad. That is the known port-vs-eval
  ik floor, and it is not a mismatch.
- **Palm x.** It uses the ArmKinematics FK in the torso frame, the same frame as the vz target. Against mujoco sites (`_palm_b`'s
  way), x_cor matches to within 9.5e-12 m.
- **Secant safeguards.**
  - NaN or inf from a zero denominator becomes a capped step of 0.
  - The bracket test `(cs - cl)(cs - ch) < 0` is False on NaN, which forces bisection or a capped step.
  - The kept iterate is gated by `on = jit & has`, so T0 rows and non-warped limbs keep d5 bit for bit, and garbage in the masked
    lanes cannot leak through `torch.where`.
  - For a limb with no active intent, wmax = -1 is clamped to 1e-9 in the first step only. That lane is masked.
- **Guard.** `g = on & (x_cor > xn)`. The bisection starts from lo = 0, and x(dn) = xn exactly, so s = 0 is always feasible. Its
  output is `dn + lo (d - dn)`. Measured independently with mujoco sites on the final output rq, the maximum excess is 5.6e-17 m
  (FK rounding between the two engines).
- **IK cost.** 1 (main) + 1 (nominal) + 4 (F') = 6 calls per step, so F' adds 4 extra IK calls, as A3.1 allows. The guard keeps
  its 2 + 8 FK calls, and F' adds 4 more.
- **rdq.** It uses the final delta, `(delta - prev) / dt`, as the eval's rate term does.
- **D.** It skips only the target offset. The generator draws the same `u` per step, so commit_s and depth equal A3's
  (test_fallback_d_draws).
- **Precision.** F' and the guard run in the compute dtype (f64) and are cast to the io dtype at the end. The f32 excess of 2.7e-8 m
  is that cast.
- **The check's gates.**
  - (ii') resamples the 16 bouts with every seed's apexes attached (`np.tile(A["bout"][r], 20)`). The rule is `-5 mm < lo` and
    `hi < 5 mm`.
  - (iii') compares the output with the separate T0 run, not with the op's internal xn.
  - (vi) runs the raw jitter on the same seeds. It draws the same numbers, because `u` is drawn per step independently of the
    reach mode.
  - (vii) takes the consecutive-row diff within a bout, masked by the later row's `has`. T0 is the nominal run on the same
    recording, with 6,623 limb-steps against 20 x 6,623 for T1.
  - The D trigger is (ii'), (vi) or (vii) failing in either dtype, as in A3.3.
- **Independent recomputation (mujoco sites, not ArmKinematics):**
  - Seed 0: apex mean -1.598 mm and yz ratio 0.708.
  - Seed 13: apex mean -1.420 mm and yz ratio 0.574.
  - Both equal the check's per_seed values. Every-step excess is at most 5.6e-17 m, and `has` is identical to the nominal run's.

**Cosmetic, not fixed:** two new docstring lines in `fb_opponent.py` are longer than their neighbours (the AimOp jitter paragraph
and the `_reach` docstring).

## Readings and facts for the main session (none blocks ok)
1. **A1 (i) under F'.** A3.1 moves the IK target's x by design, while A3.2 keeps A1 (i) "unchanged". The builder checks A1 (i) on
   the vz target F' starts from, which is bit-exact. That is the only reading under which both clauses hold. The moved target's yz
   is bit-equal to the jittered one (test and probe).
2. **(vi) "spread".** It is measured as the mean |yz_jit - yz_nom| of the palm at the aimed apexes, against the same quantity for
   the raw jitter on the same seeds. It is a displacement magnitude, not a positional variance. I accept it as the natural measure
   of the diversity the jitter adds.
3. **(vii) is weak as frozen (I confirm).** T0's own q99.9 and max (1.848 and 1.85 rad) are single-step jumps at clip loops and
   slot switches, so the gate cannot see moderate added roughness. The q50 and q99 match too (0.1255 vs 0.1250; 1.4946 vs 1.4939).
4. **Held-out jitter seeds 100-103 (probe_variant, f64): F' as built passes every A3.2 gate there too.**
   - (ii') mean -1.05 mm, CI [-1.81, -0.44] mm.
   - (vi) 0.561, per seed 0.511-0.603.
   - (vii) passes.
   - The guard acts on 18.2% of limb-steps, and F' converges on 67.1%.
   So the result is not specific to seeds 0-19.
5. **The iterate-choice trade-off (probe_variant, the same seeds 100-103).** I tried a "feasible first" variant: keep the closest
   iterate with palm x <= nominal x when one exists, otherwise the closest.
   - It keeps far more yz: (vi) 0.873, per seed 0.819-0.910.
   - The guard acts on only 5.7% of limb-steps.
   - But it **fails (ii')**: mean -4.85 mm, CI [-6.90, -2.99] mm, because the feasible iterates sit short of nominal.
   The built smallest-|g| rule is therefore the one that passes (ii'). The low per-seed (vi) values (0.456, 0.498) and the 91.5%
   intent binding come from the guard snapping s to 0 when an iterate lands slightly ahead. This is information only; I changed
   nothing.
6. **Apex shortfall.** 78.8% of apexes are below nominal, the minimum is -143.9 mm, and none is above (f64). These are steps where
   neither the 4 secant steps nor the joint bounds allow reaching nominal x. The F' residual ranges from -0.279 to +0.422 m, and the
   guard handles the positive tail.
7. **GPU cost.** A mixed batch runs F' on every env whenever any env is T1, to avoid a device sync. That is 6 IK calls against 1
   for T0, measured at 5.6x the wall time per step on CPU at 16 envs. P0 timing should include this.

## Files
- `build\verify_m3c\v_m3c_check.json`, `v_m3c_check.log`: my re-run of the check.
- `build\verify_m3c\probe_indep.py`, `.json`, `.log`: mujoco-site and eval-`Aim.ik` checks.
- `build\verify_m3c\probe_variant.py`, `.json`, `.log`: held-out seeds and the iterate-choice variant.
- `build\verify_m3c\jdiff.py`: the JSON diff.
- `build\verify_m3c\suite_v.log`: the full suite.
- `SEED_LEDGER.md`: one row appended ("M3 A3 verify").
