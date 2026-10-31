# M3 + Amendment A1 (M2V2_PLAN_v3, G0-4), 2026-10-29

**Status: all G0-4 gates pass** with A1's float32 gate (i)-(iii) and jitter gate (i)-(iv). `scripts/fb_m3_check.py` exits 0 and
`ok` holds for all 7 groups (params, FK, AimOp_f64, AimOp_f32, ring, jitter_f64, jitter_f32). The full suite passes:
**268 passed**, 250 s.

- **Scope of the change:** only the two gates A1 replaces were changed; every other G0-4 gate and tolerance is unchanged.
  The f64 run's results are identical, field for field, to `m3_check.json`, as are FK, ring and params.
- **Housekeeping:**
  - The repo has new, untracked files only. `git diff` on tracked files is empty, and nothing was committed.
  - The suite's MUJOCO_LOG.TXT was deleted, and no stray `Users...` folder exists.
  - CPU only, $0.
- **Seeds:** no new block was used.
  - The check loads the existing M3 recording (60000-60015, role `record`) and does not re-simulate it.
  - Only 60001 was re-run, for 4 s and 1 s, by the tests and a probe.
  - A row was appended to SEED_LEDGER.md, and `fb_seeds.py --audit --scan` is ok.

## Files
| file | change |
|---|---|
| `scripts/fb_m3_check.py` | **TOL:** adds `q_rows_frac` 0.999, `jitter_ci` 0.90, `jitter_ci_m` 5e-3 and `jitter_excess_m` 5e-3. The replaced gates are kept as `V0`, for reporting only.<br>**`replay(round_to=)`:** the float64 port on float32-rounded states.<br>**`q_gate_a1`, and `compare(port=)`:** float32 gate (i) and (ii). (iii), the closest palm, stays in `compare`. The f64 rule is unchanged.<br>**Jitter helpers:** `t_quantile` (Student-t by Simpson + bisection, since there is no scipy), `mean_ci`, `cluster_ci` (info only), `mixed_mask` (iv).<br>**`jitter_test`:** rewritten for (i)-(iv).<br>**`res["superseded_by_A1"]`:** the two replaced gates as measured; they are not in `ok`. |
| `tests/test_fb_aim.py` | 12 to 16 tests.<br>**Fixtures:** `bouts`, `rec`, and `rec2` (the 60001 bout twice, for the mixed mask).<br>**Changed:** the f32 AimOp test now uses A1 (port).<br>**New:** `test_a1_float32_q_rule` (the rule's edges, and TOL pinned at its frozen values); `test_t_quantile_and_mean_ci` (t at df 1/4/10/30/217/inf); `test_jitter_a1_on_a_recorded_bout[f64, f32]`, which asserts (i), (iv) and the (iv) negative control, plus the (ii)/(iii) rule arithmetic. |

Backups of the pre-A1 versions are in `build\fb_m3_check.pre_A1.py` and `build\test_fb_aim.pre_A1.py`. Scratch files:
`build\m3a_jitter_fn.py`, `m3a_splice.py`, `m3a_main.txt`, `probe_m3a_test.py`.

## Commands (cwd = repo, PYTHONIOENCODING=utf-8)
```
PY=...\scratchpad\venv\Scripts\python.exe ; F=...\m1m2\fullbody\build
$PY -I scripts/fb_m3_check.py --out $F\m3a_check.json --aimrec $F\rec\fb_aimrec_60000-60015.npz --rec $F\rec\fb_rec_60000-60015.npz --workers 4   # exit 0, 280 s (log m3a_check.log)
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp="...\scratchpad\pytest-tmp" tests/test_fb_aim.py   # 16 passed, 86 s
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp="...\scratchpad\pytest-tmp" tests                   # 268 passed, 251 s (m3a_suite.log)
$PY -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan    # ok: 19 roles disjoint; ledger consistent; 12 task inputs clean
```

## Results (`build\m3a_check.json`, 16 bouts, 15,412 steps, 482 intents, 218 aimed apexes)
| gate | float64 | float32 (f32 states/outputs, f64 aim arithmetic) |
|---|---|---|
| params, FK/Jac, intents, StandOff dist, ring (unchanged) | PASS (identical to m3_check.json) | PASS: dist 1.29e-8, intents 482/482 |
| warped q, every row vs the eval (unchanged, f64 only) | **PASS** 1.26e-10 | superseded |
| A1 f32 (i): same-input vs the float64 port, every row <= 1e-4 | | **PASS** 1.19e-7 (output rounding; rdq 3.8e-6, dist 1.3e-8) |
| A1 f32 (ii): vs the eval <= 1e-4 on >= 99.9% of rows | | **PASS** 99.974% (4 of 15,412 over, max 1.94e-3) |
| A1 f32 (iii): closest palm-to-target <= 1 mm | PASS 7.8e-16 m | **PASS** 8.0e-8 m |
| A1 jitter (i): x target at every warped step = reference palm, = nominal (bit-exact) | **PASS**: 6,627 warped steps, same in both runs, reference arm bit-equal | **PASS**, same |
| A1 jitter (ii): mean (jitter - nominal) fwd at apex, 90% t CI inside +-5 mm | **PASS**: -1.77 mm, SE 1.56 mm, df 217, CI [-4.35, +0.81] mm | **PASS**, same to 1e-8 |
| A1 jitter (iii): max forward excess at apex, jitter <= nominal + 5 mm | **PASS**: 123.72 vs 122.16 mm (+1.57 mm) | **PASS** (+1.57 mm) |
| A1 jitter (iv): mixed mask (8 T0 / 8 T1 bouts) | **PASS**: T0 rq, rdq and dist bit-equal (7,412 rows); T1 differ; T0 intents equal | **PASS**, same |

**Info, the pure float32 arithmetic run (not deployed):**
- (i) fails at 0.032 rad and (ii) fails at 99.63%.
- This is why `compute` stays float64.

**Superseded gates, as measured** (`superseded_by_A1`):
- f32 q on every row vs the eval: 1.94e-3 on 4 rows. The eval's own Aim.ik explains every row.
- Jitter forward distance at apex: 0.143 m, with 162 of 218 apexes over 1 mm. The eval's own warp misses x by more than 1 mm at
  212 of 218 apexes, and the IK reaches its target at 0 apexes.

## For the main session (reported, not decided)
1. **Bout-cluster bootstrap of jitter (ii) sits just outside the band.**
   - A1 (ii) is gated on the plain t interval over the 218 apexes, which is how A1 is written. That interval is [-4.35, +0.81] mm.
   - The apexes cluster in 16 bouts. A (bout) cluster bootstrap 90% CI is **[-5.10, +1.39] mm** (4000 resamples, info only).
   - Its lower end is 0.1 mm outside -5 mm, on the "less reach" side; the upper end is well inside.
   - If the plan means a cluster CI, (ii) would narrowly fail on the reach-reducing side. A1 limits added reach, and this
     would not show any.
2. **Forward excess over every warped step, not just the apex** (info).
   - The eval's own warp puts the palm up to **27.9 cm** beyond the clip's forward distance. A1's finding 3 quotes 12.2 cm, which
     is at the apexes only.
   - Jitter's maximum over all steps is lower, 19.5 cm (-8.3 cm vs nominal).
   - The 27.9 cm figure belongs in the M2 v2 write-up next to finding 3.
3. **Gates (ii) and (iii) are statistics over the 16 bouts.**
   - On the single 4 s test bout (4 apexes) both fail: CI +-22 mm, and excess +19 mm.
   - So the test file checks their arithmetic, while the gate itself runs in `fb_m3_check.py`.
