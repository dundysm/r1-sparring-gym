# M3 + Amendment A2 (M2V2_PLAN_v3, G0-4), 2026-10-29

**Status: ok = false. A2 (ii') fails in both dtypes; every other gate passes.** I did not change a threshold, the guard or
JITTER. (ii') is impossible as specified: under A2.1's guard with A2's unchanged JITTER ranges, its value is fixed by an
identity (shown below). A plan decision is needed.

- **Code:** the reach guard is in AimOp exactly as A2.1 states, and the A2 gates are in `fb_m3_check.py` and `test_fb_aim.py`.
- **Exit code:** `fb_m3_check.py` exits 1. `ok` is: params, FK, AimOp_f64, AimOp_f32 and ring true; jitter_f64 and jitter_f32
  false, both only because of (ii').
- **Unchanged groups:** FK, AimOp f64, AimOp f32 (A1 float32 gate), info f32 arithmetic, the eval Aim.ik floor, ring,
  params_equal and superseded_by_A1 are **identical, field for field,** to `m3a_check.json`.
- **Full suite:** **269 passed**, 334 s (`build\m3b_suite.log`). That is 268 before, plus 1 new test.
- **Housekeeping:**
  - New and untracked files only. `git diff` on tracked files is empty, nothing was committed, and the branch is
    feat/sparring-m2v2.
  - I deleted the suite's MUJOCO_LOG.TXT, and no `Users*` folder exists.
  - CPU only, at most 4 workers, $0.
- **Seeds:**
  - The check loads the existing recording (60000-60015, role `record`) with no new simulation.
  - Seeds 0-19 are AimOp's jitter generator seeds, not episode seeds.
  - Only 60001 was re-run (4 s and 1 s), by the tests.
  - I appended a ledger row, and `fb_seeds.py --audit --scan` is ok.

## Files (all untracked new files of this workstream; pre-A2 copies in `build\m3b\*.pre_A2.py`)
| file | change |
|---|---|
| `gpu/fb_opponent.py` | **New:** `GUARD_STEPS = 8`, and `AimOp._guard`.<br>**Nominal intent:** on T1 rows, AimOp keeps the nominal intent's world target (`target_wn`, `committed_n`), with the unjittered target and the nominal commit_s, using the same expression as a T0 run.<br>**Per step and limb:** it computes `dn`, the nominal warp of the same slot, with the same IK. It then takes the palm x of the jittered and nominal warps in the opponent torso frame, using `ArmKinematics.fk`, the FK the vz target uses.<br>**Guard (`jit & has & xj > xn`):** the delta becomes `dn + s (d5 - dn)`. s is the lower end after 8 bisection steps on [0, 1], always feasible, with excess <= 0.<br>**Untouched:** T0 rows, and the JITTER ranges.<br>**Per-slot counters** in `intents()`: `guard_warped_steps`, `guard_steps`, `guard_s_mean`.<br>**`debug`:** adds d_jit, d_nom, x_jit, x_nom, guard and s.<br>**`guard=False`:** the pre-A2 jitter, for diagnostics only. |
| `scripts/fb_m3_check.py` | **TOL:** drops `jitter_excess_m` (now `V1`, report only) and adds `guard_x_f64` 1e-6, `guard_x_f32` 1e-5 and `guard_bind_report_frac` 0.5. Also adds `JITTER_SEEDS = range(20)` and `BOOT = 4000`.<br>**`replay(guard=)`:** its debug output carries the guard flag and s.<br>**`jitter_runs`:** runs the seeds in a spawn pool of at most 4 workers, on the recorded states.<br>**`jitter_test`**, rewritten. It gates on:<ul><li>A1 (i) on every seed;</li><li>A2 (ii'): the pooled apexes over the 20 seeds, with a bout-cluster percentile bootstrap (16 clusters, 4000 resamples, seed 0) whose 90% CI must lie inside +-5 mm;</li><li>A2 (iii') on every warped step of every seed;</li><li>A1 (iv): the mixed mask, unchanged, at seed 0.</li></ul>It reports A2 (v); A1 (ii)/(iii) per seed (`superseded_by_A2`); the seed-0 info as before; and in f64 the guard-off run at seed 0 (`info_guard_off_seed0`).<br>**Other G0-4 gates:** unchanged. |
| `tests/test_fb_aim.py` | 16 to 17 tests.<br>**`test_a1_float32_q_rule`:** also pins the A2 TOL values, `JITTER_SEEDS`, `GUARD_STEPS`, and that JITTER is unchanged.<br>**New `test_reach_guard_bisection`** (60001 bout, f64, step by step):<ul><li>output arm = `rq5 + dn + s (d5 - dn)` bit-exact where guarded, and `rq5 + d5` elsewhere;</li><li>guard = has and xj > xn;</li><li>s lies on the 1/256 grid, s < 1, x(s) <= xn, and x(s + 1/256) > xn;</li><li>the alongside nominal equals the T0 run's arm bit for bit;</li><li>the intent counters match;</li><li>guard off: excess > 1e-6 (negative control); guard on: excess <= 0.</li></ul>**`test_jitter_a1_on_a_recorded_bout` becomes `test_jitter_a2_on_a_recorded_bout[f64, f32]`**, over seeds 0-1:<ul><li>asserts A1 (i), A1 (iv) and the (iv) negative control;</li><li>gates (iii') too, since it holds by construction;</li><li>checks the (ii') rule and the (v) fields.</li></ul> |

Scratch files are in `build\m3b\`: `patch_op*.py`, `patch_check.py`, `jitter_fn.py`, `fix_test.py`, and `probe_guard.py` /
`probe_guard2.py`, which use the recorded states only.

## Commands (cwd = repo, PYTHONIOENCODING=utf-8)
```
PY=...\scratchpad\venv\Scripts\python.exe ; F=...\m1m2\fullbody\build
$PY -I scripts/fb_m3_check.py --out $F\m3b_check.json --aimrec $F\rec\fb_aimrec_60000-60015.npz --rec $F\rec\fb_rec_60000-60015.npz --workers 4   # exit 1, 923 s (m3b_check.log)
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp="...\scratchpad\pytest-tmp" tests/test_fb_aim.py -k "guard or a2 or q_rule"   # 4 passed, 118 s
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp="...\scratchpad\pytest-tmp" tests   # 269 passed, 334 s (m3b_suite.log)
$PY -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan   # ok: 19 roles disjoint; ledger consistent; 12 task inputs clean
```

## Results (`build\m3b_check.json`; 16 bouts, 15,412 steps, 218 aimed apexes per seed, jitter seeds 0-19)
| gate | float64 | float32 (f32 states and outputs, f64 aim arithmetic) |
|---|---|---|
| params, FK, AimOp f64/f32 (A1 float32 gate), ring | PASS, identical to m3a_check.json | PASS, identical |
| A1 (i): x target = reference palm x = nominal, bit-exact | **PASS** on 20 of 20 seeds (6,627 warped (row, limb) per seed) | **PASS** 20/20 |
| **A2 (ii'):** pooled mean (jitter - nominal) fwd at apex, bout-cluster bootstrap 90% CI inside +-5 mm | **FAIL**: mean **-6.58 mm**, CI **[-8.28, -5.06] mm** (4,360 apexes, 16 bouts). Per-seed means -8.62 to -4.70 mm. Naive t CI [-7.14, -6.02] | **FAIL**, same to 1e-8 m |
| **A2 (iii'):** jittered x - nominal x at every warped step of every seed | **PASS**: max **0.0 m** (132,540 steps, 0 > 0) | **PASS**: max 7.0e-8 m (<= 1e-5; 1 step > 0, output rounding) |
| A1 (iv): mixed mask (8 T0 / 8 T1 bouts, seed 0) | **PASS**: T0 rq, rdq and dist bit-equal (7,412 rows); T1 differ | **PASS**, same |
| **A2 (v)**, reported | **binds on 98.3% of jittered intents (4,346 / 4,420) > 50%: REPORTED.** Mean s = 0.0029 over guarded steps, and s = 0 on 99.4% of them. 44.6% of the T1 warped limb-steps are guarded | same: 98.3%, mean s 0.0029 |

**Reported alongside (f64):**
- **A1's replaced gates, per seed (`superseded_by_A2`):**
  - A1 (ii), the t CI: 0 of 20 seeds pass, all on the low side.
  - A1 (iii), apex excess: 20 of 20 pass (margin 0).
- **Guard off at seed 0 reproduces M3A exactly:**
  - mean -1.77 mm, t CI [-4.35, +0.81] mm, cluster CI [-5.10, +1.39] mm, and +1.57 mm apex margin;
  - every-step excess over nominal up to **+45.0 cm**;
  - yz spread at apex 23.0 mm.
- **With the guard:**
  - the yz spread at apex falls to **12.1 mm**;
  - the rdq difference from nominal is not larger than without it: q99.9 69.2 vs 76.0, and max 82.5 vs 129.6.
- **A2 finding 3 is confirmed by the check:** the eval's own warp reaches 27.86 cm beyond the clip over every warped step
  (`eval_every_warped_step_x_over_clip_max_m`).

## Why (ii') cannot pass as specified (for the main session; not decided)
1. **The guard caps every apex at nominal.** At each warped step it leaves dx = x_jit - x_nom unchanged when dx <= 0. Otherwise
   it sets dx = x(s) - x_nom <= 0. Palm x is close to linear in s, so s = 0 on 99.4% of guarded steps, and there dx = 0.
   - At every apex, then, dx_guarded is min(dx_raw, 0) (measured: `info_share_above_0` = 0.0).
   - The raw delta at a step does not depend on earlier steps' guard (target_w and the IK are per step), so dx_raw is the
     guard-off jitter.
2. **Identity:** mean(min(r, 0)) = (mean r - mean |r|) / 2.
   - Over the same 20 seeds, the guard-off apex dx (M3A verify `probe_seeds_*.json`) has mean r = -1.80 mm and mean |r| = 11.35 mm.
   - This predicts -6.5774 mm, and the check measures **-6.5774 mm**. Seed 0: predicted -6.7647, measured -6.7648.
   - The AimOp and FK behind these numbers equal the eval's Aim.reference (q 1.3e-10 rad) and `_palm_b` (6.7e-16 m), from this
     same run.
3. **What a pass would need:**
   - Even with zero CI width, (ii') needs mean |r| < 10 mm + mean r = **8.2 mm**. With the observed cluster half-width
     (about 1.6 mm), it needs about 5 mm.
   - mean |r| is set by the JITTER yz and commit ranges, which A2.1 freezes ("ranges unchanged").
   - So, with the guard and ranges as written, no seed set or bootstrap choice makes (ii') pass. The seed set is frozen
     anyway, at 0-19.
4. **What it means:**
   - The training opponent's palms reach on average **6.6 mm less** far forward at apex than the eval's (nominal), while
     never reaching further.
   - For a defender, shorter training strikes are a train/eval gap, not a safe margin: it could learn a stand-off that the
     eval's strikes reach.
   - The guard binds on 98% of intents with s about 0. So on guarded steps the arm is fully nominal: no vertical or lateral
     diversity is left there, contrary to A2's expectation. The yz spread at apex halves, from 23 to 12 mm.

**Options:**
- **A.** Make (ii') one-sided, as the upper bound +5 mm. It passes, since the guard enforces <= 0. Disclose the -6.6 mm mean
  and accept less reach in training.
- **B.** Make the guard two-sided: also cap the deficit, with s chosen so that x_jit is close to x_nom from both sides.
  Because x is near linear in s, this degenerates to s = 0 everywhere, which removes the arm-target jitter entirely. Only
  commit_s and stand-off depth jitter would remain.
- **C.** Narrow the JITTER yz (and z asymmetry) ranges until mean |r| is about 5 mm or less, then re-gate. This needs an
  amendment to A2.1's "ranges unchanged". It is cheap to scan on the recorded states (about 45 s per seed and setting per
  worker).
- **D.** Drop the arm-target yz jitter and keep the commit_s and depth jitter. Given (v), the guard already removes most of it.
- **E.** Keep the guard and gate (ii') on |mean| with a wider band. This loosens a frozen threshold, so it is listed only for
  completeness.
