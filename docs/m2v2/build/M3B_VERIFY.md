# M3B verify: M3 under Amendment A2 (M2V2_PLAN_v3, G0-4), 2026-10-29

**Verdict: ok = false.** I agree with the builder. Every frozen G0-4 check, as amended by A1 and A2, passes except **A2 (ii')**,
which fails in both dtypes. I proved that (ii') cannot pass as specified: with A2.1's guard and the frozen JITTER ranges, its
value follows from an identity. The implementation follows A2.1 literally, and I found no unfixed semantic mismatch. I changed
no code, and I loosened no threshold. A plan decision is needed (options below).

## What I re-ran (CPU only, at most 4 workers, $0)
| run | result |
|---|---|
| `fb_m3_check.py --aimrec build\rec\fb_aimrec_60000-60015.npz --workers 4` -> `verify_m3b\m3b_check_v.json` / `.log` | exit 1 in 926 s. Every field equals `m3b_check.json` (recursive diff, wall times excluded). ok: params, FK, AimOp_f64, AimOp_f32 and ring true; jitter_f64 and jitter_f32 false |
| Full suite (`pytest ... --basetemp="...\pytest-tmp" tests`, FB_SEED_LEDGER set) -> `verify_m3b\suite_v.log` | **269 passed**, 331 s |
| `fb_seeds.py --audit SEED_LEDGER.md --scan` | ok: 19 roles disjoint; ledger consistent; 12 task inputs clean. Run before and after my ledger row |
| `verify_m3b\probe_identity.py` (recorded states; guard on and guard off, f64, seeds 0-19; 40 replays on 4 workers) | see below |
| `verify_m3b\probe_proj.py` (recorded states, seed 0, f64; a diagnostic for one option only) | see options |

**Hygiene:**
- The suite's MUJOCO_LOG.TXT (an empty-prefix warning) is deleted.
- There is no `Users*` folder in the repo.
- The tracked diff is empty. Only the builder's untracked new files exist, nothing was committed, and the branch is feat/sparring-m2v2.
- There was no new simulation. Seeds 0-19 are AimOp generator seeds.
- The suite re-ran 60001 in role `record`, and I appended one ledger row for this.

## Gates (identical in my run and the builder's)
| gate | f64 | f32 |
|---|---|---|
| params, FK, AimOp f64, AimOp f32 (A1 float32), ring | PASS | PASS |
| A1 (i): IK x target = reference palm x = nominal, bit-exact, every seed | PASS, 20/20 | PASS, 20/20 |
| **A2 (ii'):** pooled apex (jitter - nominal) dx, bout-cluster bootstrap 90% CI inside +-5 mm | **FAIL**: mean -6.577 mm, CI [-8.284, -5.062] mm | **FAIL**, same to 1e-8 m |
| A2 (iii'): jittered x - nominal x at every warped step (132,540 steps) | PASS, max 0.0 | PASS, max 7.0e-8 m (output rounding) |
| A1 (iv): mixed mask, T0 rows bit-equal (7,412 rows) | PASS | PASS |
| A2 (v), reported | binds on 98.3% of jittered intents (> 50%, reported); mean s 0.0029; s = 0 on 99.4% of guarded steps | same |

## Code read against A2.1 and the eval (r1gym/aim.py `reference`, `_arm`, `_palm_b`, `ik`)
**Diff vs `build\m3b\*.pre_A2.py`.** The only changes are:
- the guard;
- the nominal-intent state (`target_wn`, `committed_n`);
- the counters, debug fields and `guard=` flag;
- in the check script, the A2 gates and the spawn pool;
- in the test file, the new tests and pins.

The T0 path is untouched.

**Checked line by line:**
- **Nominal intent.** It uses the unjittered `target` and the nominal `p["commit_s"]`. The update window (`now <= t_end + 1`) and the
  order (update the target, then commit) are the same as for the jittered intent and the eval's `_schedule`.
- **Slot and weight.** The same slot `j` and weight `wmax` serve both. That is correct, because jitter changes neither the timing
  nor the gating; only target, commit_s and depth are drawn.
- **Same IK.** `p_n` keeps the vz x (`p_t[..., :1]`, the reference palm's own), so dn is the T0 run's arithmetic. The test proves it
  bit for bit, and A1 (iv) holds.
- **Palm x.** It is `kin.fk(rq5 + d)[..., 0]`: the torso frame, with the same ArmKinematics that gives the vz target (eval
  `_palm_b`, 6.7e-16 m).
- **Bisection.** It starts at lo = 0, which is always feasible since x(dn) = xn. It runs 8 steps, and the output is
  `dn + lo (d5 - dn)`, applied only where `jit & has & xj > xn`. This is A2.1 as written.
- **rdq.** It uses the guarded delta (`(delta - prev) / dt`), as the eval's rate term does.
- **Precision.** The guard works in the compute dtype (f64) and casts to the io dtype at the end. The f32 excess of 7e-8 m comes
  from that cast and is within 1e-5.
- **(ii') in the check.** The 20 seeds' apex rows are aligned with `np.tile(A["bout"][r], 20)`. The percentile cluster bootstrap
  resamples the 16 bouts. The rule is `-5 mm < lo and hi < 5 mm`. Apexes are the first control step with now >= t_apex for
  non-gated intents, the same definition A1 and M3A used.
- **(iii') in the check.** It compares the jittered output with the separate T0 run's output, not with the guard's internal xn,
  so it is an independent check.

**Cosmetic, not fixed:**
- The `cluster_ci` docstring still says "Reported only", but it is now gated.
- The `test_jitter_a2...` docstring says "1 bout", while rec2 is the same bout in two envs (`bouts == 2`).

Neither affects any result.

**Noted, not a mismatch (frozen design):** the depth jitter (0-6 cm vs the nominal 3 cm) moves the stand-off by up to +-3 cm,
symmetric about the nominal. A2 defines reach as palm x in the torso frame, so the guard rightly ignores it.

## Proof that (ii') is impossible as specified (`verify_m3b\probe_identity.json`, 4,360 apexes)
1. **The guard gives min(raw, 0) at each apex.**
   - Palm x is close to linear along the segment dn -> d5. So where xj > xn, x(dn + s dd) - xn is about s (xj - xn) > 0 for every
     s > 0. The largest feasible s is 0, or one grid step above it.
   - The raw jittered delta at a step does not depend on earlier guard outputs: the target is per step, the IK starts from rq5,
     and the states are recorded. The generator draws the same numbers whether the guard is on or off.
   - So the guarded apex dx is min(dx_off, 0). Measured:
     - it is exact on 99.59% of apexes;
     - the largest deviation is 0.18 mm;
     - the guarded apex max is 0.0;
     - the guard-off dx is above 0 on 48.0% of apexes.
2. **Identity:** mean(min(r, 0)) = (mean r - mean |r|) / 2.
   - Guard off: mean r = -1.804 mm and mean |r| = 11.350 mm.
   - That predicts -6.5774 mm, and guard on measures -6.5774 mm.
3. **The eval's own operator as the nominal.**
   - Using the eval's recorded warped `A["rq"]` (r1gym Aim.reference) as the nominal instead of AimOp's, the result is the same:
     mean -6.5774 mm and CI [-8.284, -5.062] mm.
   - Port vs eval apex x is 8.6e-14 m.
   - So the failure belongs to the guard and JITTER, not to the port.
4. **What a pass would need.** Even with a CI of zero width, it needs mean |r| < 10 mm + mean r = 8.20 mm.
   - mean |r| comes from the JITTER yz/z and commit ranges, which A2.1 freezes. The seed set is frozen at 0-19, and the recording
     is fixed.
   - The result is therefore deterministic, and no allowed choice passes.
   - For comparison, guard off would have passed (ii') on this pooled set: cluster CI [-3.61, -0.11] mm. But that fails (iii')
     by up to 45 cm.
5. **Consequences.**
   - The guard makes no reach added true by snapping to nominal, not by trading x for yz. Training palms reach 6.6 mm short of the
     eval's at the apex on average, a train/eval gap.
   - On guarded steps no yz diversity is left: the yz spread at the apex is 12.1 mm, against 23.0 mm without the guard.

## Options for the main session (not decided)
**The builder's options:**
- **A. One-sided (ii').** An upper bound of +5 mm passes by construction. It needs disclosure of the -6.6 mm mean, and it keeps
  the short-reach gap.
- **B. Two-sided guard on the same segment.** It collapses to s = 0, which means no arm jitter.
- **C. Narrower JITTER yz/z.** Use ranges with mean |r| at or below about 5 mm. This needs an amendment to "ranges unchanged".
- **D. Drop the arm-target jitter.** Keep the commit_s and depth jitter.
- **E. Widen the band.** This loosens a frozen threshold and is listed only for completeness.

**My additional option, F: an x-projection guard (two-sided, keeps yz). It needs an amendment to A2.1's mechanism.**

How it works:
- Instead of interpolating toward dn, move the jittered arm q along the palm-x gradient (Newton, minimum-norm in joint space,
  clipped to the IK's bounds) until x equals the nominal warp's x.
- Then apply A2.1's one-sided bisection as a fallback where the bounds stop the projection, so that (iii') stays exact.

Probe on seed 0 (`verify_m3b\probe_proj.json`, 8 Newton steps, no fallback):
- **Apex dx:** the mean is **+0.09 mm** (the max is 4.1 mm).
- **yz spread at the apex:** **20.6 mm** of the raw 23.0 mm is kept, against 12.1 mm under A2.1.
- **x residual:** above 1e-6 m on 322 of 6,627 warped limb-steps (bound-limited), with a max of 1.5 cm. The fallback would cover
  these, and those steps would then be at or below nominal.
- **Joint change:** up to 1.29 rad within the IK's bounds. It would need a look at rdq.

This is a diagnostic sketch only. It is not built, and it is not gated.

## Files
- `build\verify_m3b\m3b_check_v.json`, `m3b_check_v.log`: the check re-run.
- `build\verify_m3b\suite_v.log`: the full suite.
- `build\verify_m3b\probe_identity.py` with `.json` and `.out`.
- `build\verify_m3b\probe_proj.py` with `.json` and `.out`.
- `SEED_LEDGER.md`: one row appended, "M3 A2 verify".
