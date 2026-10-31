# M3 + Amendment A3 (M2V2_PLAN_v3, G0-4), 2026-10-30

**Status: ok = true. F' is adopted; fallback D was not needed and was not run.**
- `fb_m3_check.py` exits 0. Every `ok` field is true: params, FK, AimOp_f64, AimOp_f32, ring, jitter_f64, jitter_f32.
  `adopted` = "F'".
- Every A3.2 gate passes in both dtypes: A1 (i), A2 (ii') two-sided, A2 (iii'), A1 (iv), (vi) and (vii). (v) is reported.
- No threshold, JITTER range or seed set was changed. T0 rows are untouched: A1 (iv) is bit-equal on 7,412 rows.
- **Extra IK cost:** 6 IK calls per control step on T1 rows. That is +4 over A2.1 (2) and +5 over T0 (1). On this CPU at
  16 envs, the jittered op takes 134 ms per step against 24 ms for the 1-IK op (5.6x).
- **Unchanged groups:** params, FK, AimOp f64, AimOp f32 (the A1 float32 gate), info f32 arithmetic, the eval Aim.ik floor,
  ring and the f32 part of superseded_by_A1 are **identical, field for field,** to `m3b_check.json` (wall times excluded).
  The jitter part of superseded_by_A1 moved, because the jitter changed: v0 max |dx| is 92.8 mm, against 143.4 mm under A2.
- **Full suite:** **271 passed**, 471 s (`build\m3c_suite.log`). That is 269 before, plus 2 new tests.
- **Housekeeping:**
  - New and untracked files only, and the tracked `git diff` is empty. Nothing was committed or stashed, and the branch is
    feat/sparring-m2v2.
  - I deleted the suite's MUJOCO_LOG.TXT (the empty-prefix warning), and no `Users*` folder exists.
  - CPU only, at most 4 workers, $0.

## Files (untracked files of this workstream; pre-A3 copies in `build\m3c\*.pre_A3.py`)
| file | change |
|---|---|
| `gpu/fb_opponent.py` | **New constants:** `SECANT_IK = 4`, `SECANT_TOL = 1e-6`, `SECANT_STEP = 0.10` and `REACH = ("A3", "A2", "D", "raw")`. `AimOp(reach=...)` replaces `guard=` and defaults to "A3".<br>**`_reach`** replaces `_guard`. On T1 rows it first computes the nominal warp dn exactly as A2.1 did (the T0 run's arithmetic). Then, in mode "A3", it runs F':<ul><li>keep the jittered IK target's yz, and run secant iterations on its torso-frame x c;</li><li>the root is g(c) = palm x after the same weighted IK - nominal palm x + SECANT_TOL/2;</li><li>the first step is unit slope over the weight, `c0 - g0 / w`;</li><li>steps are capped at 0.10 m until a sign change brackets the root; after that, a secant step that leaves the bracket is replaced by bisection;</li><li>the iterate kept is the one with the smallest abs(g), with c0 itself as a candidate.</li></ul>The **A2.1 guard is then unchanged**: where palm x > nominal x, d -> dn + s (d - dn), with 8 bisection steps. This keeps (iii') true by construction.<br>**Mode "D" (A3.3):** draws no arm-target offset. The same generator draws are made, and commit_s and depth stay jittered. It uses the guard alone.<br>**Modes "A2" and "raw":** the guard alone, and the pre-A2 jitter; both for diagnostics.<br>**Counters and debug:** `ik_calls` counts every IK run. The new per-slot counter `secant_conv_steps` is in `intents()`. `debug` adds d_cor, x_cor, shift and conv.<br>**Unchanged:** T0 rows and JITTER. The module and AimOp docstrings are updated. |
| `scripts/fb_m3_check.py` | **TOL:** adds `yz_keep_frac` 0.5, `smooth_q999_ratio` 1.25 and `smooth_max_add_rad` 0.2.<br>**`replay(reach=)`:** returns the IK calls and the wall per step, plus conv, xres and shift per limb.<br>**`jitter_test(reach=)`:** runs 20 jittered and 20 raw seeds in a spawn pool of 4 workers, on the recorded states. It gates on:<ul><li>A1 (i);</li><li>(ii'), a bout-cluster bootstrap whose 90% CI must lie inside +-5 mm (two-sided);</li><li>(iii');</li><li>A1 (iv), the mixed mask, in the same reach mode;</li><li>for "A3" only: (vi), the pooled mean yz apex spread >= 0.5x the raw jitter's on the same seeds, and (vii), the per-step change of the commanded arm q (max over the limb's 5 joints) on T1 warped limb-steps, where q99.9 must be <= 1.25x and max <= +0.2 rad of the same statistic on the T0 rows (the nominal run).</li></ul>(v) is reported, with the secant convergence share (1e-6 m), the residual and shift quantiles, and the IK calls and wall per step.<br>**`main`:** if F' fails (ii'), (vi) or (vii) in either dtype, it runs D, gates D on A1 (i), (ii'), (iii') and A1 (iv) with (v) reported, and sets `adopted`.<br>**Cosmetic (M3B_VERIFY):** the `cluster_ci` docstring now says it is gated. |
| `tests/test_fb_aim.py` | 17 to 19 tests.<br>**Pins:** the A3 TOL values, `SECANT_IK <= 4` and `SECANT_TOL`. JITTER is still pinned as unchanged.<br>**`test_reach_guard_bisection`:** now runs A2.1 alone (`reach="A2"`), and its negative control uses `reach="raw"`.<br>**New `test_fprime_reach_correction`**, step by step on 60001:<ul><li>the corrected delta is the weighted IK on the jittered target with only x moved (yz bit-equal);</li><li>x_cor and conv are as reported;</li><li>output = guard(dc), and no step lies ahead of nominal;</li><li>exactly 2 + SECANT_IK IK calls per step, and 1 for T0;</li><li>the intent counters match;</li><li>F' keeps more yz at the apex than A2 and lands nearer the nominal x.</li></ul>**New `test_fallback_d_draws`:** D's target is the nominal one, its commit and depth draws equal A3's, and no step lies ahead.<br>**`test_jitter_a2...` becomes `test_jitter_a3_on_a_recorded_bout[f64, f32]`:** it checks the (vi) and (vii) rules, the gated list, the D trigger, the convergence and the IK calls per step.<br>**Cosmetic (M3B_VERIFY):** the docstring now says "one bout in two envs" instead of "1 bout". |

Scratch files are in `build\m3c\`:
- `patch_*.py` and `jitter_fn.py`;
- `probe_fprime.py` and `probe_curve.py`, which use the recorded states and seed 0 only;
- `fb_opponent.slope_variant.py`, a rejected variant.

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, F = ...\m1m2\fullbody\build)
```
$PY -I scripts/fb_m3_check.py --out $F\m3c_check.json --aimrec $F\rec\fb_aimrec_60000-60015.npz --rec $F\rec\fb_rec_60000-60015.npz --workers 4   # exit 0, 2094 s (m3c_check.log)
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp="...\scratchpad\pytest-tmp" tests/test_fb_aim.py   # 19 passed, 392 s
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp="...\scratchpad\pytest-tmp" tests                   # 271 passed, 471 s (m3c_suite.log)
$PY -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan   # ok: 19 roles disjoint; ledger consistent; 12 task inputs clean
```

**Seeds:**
- The check loaded the existing recording (60000-60015, role `record`) and ran no new simulation.
- Seeds 0-19 are AimOp generator seeds.
- The tests re-ran 60001 (4 s and 1 s) in role `record`.
- I appended one ledger row ("M3 A3").

## Results (`build\m3c_check.json`; 16 bouts, 15,412 steps, 218 aimed apexes per seed, jitter seeds 0-19)
| gate | float64 | float32 (f32 states and outputs, f64 aim arithmetic) |
|---|---|---|
| params, FK, AimOp f64/f32, ring | PASS, identical to m3b | PASS, identical |
| A1 (i): IK x target = reference palm x = nominal, bit-exact | **PASS** 20/20 seeds | **PASS** 20/20 |
| **A2 (ii'):** pooled apex dx, bout-cluster bootstrap 90% CI inside +-5 mm | **PASS**: mean **-1.34 mm**, CI **[-2.08, -0.76] mm** (4,360 apexes, 16 bouts, 4000 resamples). Per-seed means -2.50 to -0.60 mm | **PASS**: -1.33 mm, [-2.08, -0.76] mm |
| **A2 (iii'):** jittered x - nominal x at every warped step | **PASS**: max **0.0 m** (132,540 steps, 0 > 0) | **PASS**: max 2.7e-8 m (51 steps > 0 from output rounding, 0 > 1e-5) |
| A1 (iv): mixed mask, F' on the T1 bouts | **PASS**: T0 rq, rdq and dist bit-equal (7,412 rows); T1 rows differ | **PASS** |
| **(vi):** mean yz apex spread >= 50% of raw | **PASS**: 12.98 mm vs raw 21.88 mm = **0.593** (per-seed range 0.456-0.708) | **PASS**: 0.593 |
| **(vii):** arm q change per step on T1 warped limb-steps vs T0 | **PASS**: q99.9 **1.8500** vs T0 1.8476 rad (ratio 1.001 <= 1.25); max 1.85 vs 1.85 (<= +0.2) | **PASS**: same to 1e-7 |
| **(v)**, reported | **Guard:** binds on **91.5% of jittered intents** (4,045 / 4,420; > 50%, reported), but on only **17.8% of T1 warped limb-steps**. Mean s 0.030 over guarded steps, and s = 0 on 95.9% of them.<br>**Secant:** converged within 1e-6 m on **67.6%** of limb-steps. Residual quantiles (0 / 0.1% / 50% / 99.9% / 100%): -0.279 / -0.096 / -5e-7 / +0.085 / +0.422 m. Target shift abs: median 9.4 mm, q99 0.299 m, max 0.40 m (4 x the 0.10 m cap) | same: 91.5%, 17.8%, 67.6% |
| **IK cost** | **6 IK calls per step** (raw/T0: 1; A2.1: 2): **+4 over A2.1**. Each IK is 12 x (FK + Jacobian + 5x5 solve) on [N, 2]. F' also adds 4 FK; the guard keeps its 2 + 8 FK. CPU at 16 envs: **133.8 ms/step** vs 23.8 ms (5.6x) | 139.6 vs 25.3 ms |

**Reported alongside (f64):**
- **Apex dx shape:**
  - 78.8% of apexes are below 0 and 0% above.
  - The mean abs value is 1.34 mm, and the minimum is **-143.9 mm**: a few apexes where neither the secant nor the bounds
    allow reaching nominal x.
  - A2.1 gave a mean of -6.58 mm, s about 0 on 99.4% of guarded steps, and a yz spread of 12.1 mm out of 23.0 mm at seed 0.
- **A1's replaced gates (`superseded_by_A2`):** (ii) t CI passes on 20/20 seeds; (iii) apex excess passes on 20/20.
- **Raw jitter at seed 0** reproduces M3A: mean -1.77 mm, every-step excess up to 45.0 cm, yz 23.0 mm.
- **rdq vs nominal at seed 0:** q99.9 69.7, max 81.3 (A2.1: 69.2 and 82.5; raw: 76.0 and 129.6).
- **The eval's own warp** goes up to 27.86 cm beyond the clip over every warped step (A2 finding 3, unchanged).

## Notes for the main session (facts; nothing decided here)
1. **How A1 (i) is read.** A1 (i) is checked on the vz IK target that F' starts from. That target's x is the reference palm's
   own and the nominal one, bit-exact. A3.1 then moves x by design and keeps A1 (i) unchanged, so this is the only consistent
   reading. The moved target's yz is asserted bit-equal to the jittered one in `test_fprime_reach_correction`.
2. **My mechanism choices inside "secant iterations, <= 4 extra IK calls".** Each was fixed before the full check:
   - The bracket and the step cap. A plain secant (probe, seed 0) converged on 56% and threw targets up to 200 m. f(c) is
     piecewise and plateaus at the IK's joint bounds (`probe_curve.py`).
   - Aiming 0.5 um below nominal, so that a converged iterate never triggers the guard. Under A2.1's near-linear segment the
     guard snaps s to 0 and deletes the yz.
   - Keeping the smallest abs(g).
   - **Seed-0 probes, not tuned:** step caps of 0.05, 0.2 and 0.4 m all give 67-68% convergence. A linearised first-step
     slope gives 68.1% against 67.7% and was not adopted. 16 iterations (diagnostic only) would reach 88%, so the 4-call cap
     costs about 20 points of convergence. The default 0.10 m cap was kept. No JITTER range was touched.
3. **(vii) is weak as frozen.**
   - Its q99.9 and max are set by single-step jumps of 1.6 to 1.85 rad that the T0 operator itself makes. The IK's +-0.8 rad
     clamp flips across a slot or limb switch, and the clip reference loops. The warp-delta change alone is 1.6 rad max in T0
     and T1 alike.
   - T1 matches T0 at the median (0.1255 vs 0.1250 rad) and at q99 (1.4946 vs 1.4939 rad), so F' adds no roughness that the
     statistic can see. It would also not see a moderate added roughness below those T0 jumps.
4. **(v) binds above 50% of intents, which A2.2 says to report.** That share counts an intent once if any of its steps was
   guarded, so it overstates the effect. Per step the guard acts on 17.8%, and on the other 82.2% the jittered yz is kept with
   the reach equalised. Per seed, 2 of 20 seeds fall below 0.5x (0.456 and 0.498). The frozen gate (vi) is the pooled mean,
   0.593.
5. **GPU cost.** The jittered T1 rows run about 6x the IK work of T0. If a mixed batch computes F' on every env, as this
   implementation does to avoid a device sync, then every env pays it whenever any env is T1.
