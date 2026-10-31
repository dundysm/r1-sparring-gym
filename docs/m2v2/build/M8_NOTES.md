# M8 build notes: library L1 and the T1 config, CPU part (M2V2_PLAN_v3 section 3, M8), 2026-10-30

Frozen M8 row: `gpu/export_fb_library.py`, `gpu/r1_track_spar_cfg.py`; (1) manifest, every row's source in TRAIN_CLIPS, sha256;
(2) rejection stats per family, re-angled apex forward distance = original (no reach); (3) T0-alone performability per family,
24 starts each (60500-60599), reported, decides which families need T1; (4) mirrored R1 joint ranges symmetric (asserted).
Spec used: plan 2.2 (L1, T1), design_learned-tracker 9.1-9.3 (priors, rejection rules, T1 task). Track 2 is not used (A5.2).

**Result: all four checks pass as specified. Performability (3) puts every family below T1-b's 90% bar, so as frozen all four
families need T1. The cause is mirrored motion: T0 is not mirror-equivariant. On the original-base rows T0 keeps up 11/12, 10/12
and 11/12; on mirrored-base rows 1/24, 1/12, 0/12, 0/12. A diagnostic proves the mirror itself is correct (below). That opens an
option the main session may want to decide (section 5).**

## 1. Files (new; nothing tracked was modified, no commit)

| file | what |
|---|---|
| `gpu/export_fb_library.py` | L1 generation (mirror, warp, reangle, combo), the T1 motion CSV + index, the reference a tracker trained on it is shown (csv_to_npz float32 grid at Tracker.frame, export_fb_motion's method on the whole T1 motion), strike tables (the eval's `strikes.detect` on every row), clean-start masks (`scenario.clean_starts`, the eval's call, through a row replay), manifest |
| `gpu/r1_track_spar_cfg.py` | `R1-Track-Spar` / `R1-Track-Spar-LOCO`: Steer2's env_cfg + the Aim task's hand impulses; `SparMotionCommand` = Steer2's command with mjlab 1.2.0's adaptive sampling renormalised per group and mixed 0.5 originals / 0.5 L1; `build_motion(loco)`; CLI `motion [--loco]`, `convert`, training. Imports mjlab lazily (pod only) |
| `scripts/fb_m8_check.py` | the M8 checks (1)-(4) + extras (clean-mask identity, shown-reference comparison, mirror-conjugated T0 diagnostic) |
| `tests/test_fb_library.py` | 13 tests, about 25 s |
| `gpu/motions/fb/fb_l1.npz` (7.1 MB, sha256 413f4802...), `fb_l1_track.csv` (8.8 MB, 5c21a07b...), `fb_l1_track_index.json` (789c6420...), `fb_l1_manifest.json` | outputs (deterministic zip timestamps; regeneration is bit-equal, tested) |

Untouched: the other lanes' uncommitted diffs (fb_opponent, fb_rewards, fb_seeds, fb_m5_check, tests) and untracked M6/T2 files.
The manifest's git field reads `e9fdd66...+dirty` because of those diffs.

## 2. Library L1 as built (281 s = 4.7 min; design 9.1 expected 4-5 min)

Source: `fb_motion.npz`'s q only (M1, train clips). Every row is a whole loopable clip; generator `default_rng([0, family, clip,
mirror, draw])`; one draw per strike, no redraw (redraws would reshape the frozen prior); a rejected strike stays as in the base.

| family | rows | s | strikes seen | tried | accepted | rejected (reasons, multi-reason counted per reason) | skipped |
|---|---|---|---|---|---|---|---|
| mirror | 3 | 25.0 | - | - | - | - | - |
| warp | 12 | 102.9 | 148 | 97 | 53 | 44 (speed 44) | seam 24, overlap 27 |
| reangle | 12 | 99.8 | 148 | 112 | 48 | 64 (speed 54, path 22, reach 21) | type 36 (downward: no prior) |
| combo | 6 | 53.3 | 74 | 13 | 8 | 5 (speed 5) | max 26, guard 19, seam 15, gap 1 |

- Warp: drawn speeds mean 1.006 (symmetric prior); **accepted mean 0.884 (9 faster, 44 slower), every rejection a speed-up**
  (mean 1.154). The frozen export_warps speed rule (|dq| <= max(p99, the window's own fastest) + 0.05 rad/s) passes a sped-up strike
  only where p99 is above 1.3x its own speed. The accepted set is therefore slow-biased although the prior is not. Disclosure item.
- Re-angle: accepted delta mean -0.13 deg (27 up, 21 down), mean |delta| 13.7 deg vs 19.9 deg rejected. Most rejections are the speed
  rule: rotating the last 0.10 s of a strike moves its fastest joint frame, and the window's own fastest is binding.
- Combo: 8 splices in 6 rows (1, 1, 1, 2, 2, 1). Pairing 200 <- 201 <- 203 <- 200 keeps 2 combo rows in the LOCO motion.
- Strikes detected on the 33 rows: 420 (hook 191, downward 110, straight 62, uppercut 57).
- Clean masks: [33, 31, 300], every row has >= 2805 clean (distance, start) cells of 9300.
- T1 motion: 3 originals + 33 rows, each twice, cross-fade 0.4 s: 29,888 frames. LOCO (`build_motion(loco=True)`): 20,302 frames,
  originals 200/201 and 20 L1 rows (mirror 2, warp 8, reangle 8, combo 2).
- Sampling: originals hold 49 of 598 one-second bins (8%); the group mix gives them 0.5 of the sampling mass whatever the failures
  (tested). Reading of "half the sampling weight on originals, to protect them": a 50% share.

## 3. Check results (`scripts/fb_m8_check.py`, `build\m8\m8_check.json`, `m8_check.log`; 6.4 min, 2 workers)

| check | result |
|---|---|
| (1) manifest | **PASS**: 33 rows; row sources and every frame's src_clip in {kfab_200, 201, 203}; sha256 of the 3 outputs, of the 13 sources and of every row's q match; T1 index = npz rows; `fb_seeds.scan` finds no banned name in the outputs or the new code |
| (2) rejection stats | reported (section 2) |
| (2) no reach | **PASS**: 48 accepted re-angles, apex frame bit-equal to the base in 48/48 (so the apex palm forward distance is the original's exactly); max palm forward excess over the re-angled window **-0.31 mm** (<= 0, enforced as a rejection rule: 21 rejected for reach). Re-detection with the eval's rule: 47/48 strikes found again, all 47 with the **same apex forward distance (diff 0.0)**; 5 hooks re-classify as downward; 1 not re-detected (l1_reangle_kfab_201_m1, r_hand apex 541: its onset speed no longer crosses 1.5 m/s). Report: warp resampling puts the palm up to 4.7 mm beyond the base's own sampled frames (interpolation between frames) |
| (3) T0 alone, ids 60500-60595 | reported: mirror **1/24** (0.042), warp **12/24** (0.50), reangle **10/24** (0.42), combo **11/24** (0.46). All below 0.9: **as frozen, all four families need T1** |
| (3) by base | original-base rows: warp 11/12, reangle 10/12, combo 11/12; mirrored-base rows: mirror 1/24, warp 1/12, reangle 0/12, combo 0/12. Of the 14 reangle falls only 3 come within 1 s after a modified frame; both original-base reangle falls are T0's own fall on kfab_200 at t0 2.5 s (0.52 s, also in the originals) |
| (3) runner identity | **PASS**: originals 72 starts through the M8 runner = the eval's own Tracker (fb_m2_check._alone 'eval'): 67/67 up, 0 mismatches |
| (3) diagnostic | T0 conjugated by the mirror (`MirroredMLP`: obs mirrored, T0, action mirrored) on the mirrored-base rows at the same starts: mirror **23/24**, warp 11/12, reangle 10/12, combo 10/12. On mirror rows it reproduces T0-on-originals start for start (kfab_200 start 2.5 s falls at 0.52 s in both). So the mirrored rows, their references and the physics are correct; T0 is not mirror-equivariant |
| (4) mirror | **PASS**: max |mirrored range - range| **9.0e-5 rad** (shoulder roll 2.4784 vs 2.47849, 0.22689 vs 0.2268; knees 3e-6, ankle roll 1e-6) <= RANGE_TOL 1e-4 (asserted; the MJCF's 4-decimal print precision; mirrored q is clipped to the ranges, max change 9e-5 rad); FK of mirror(q) = reflected FK of q on every frame of the 3 clips: bodies **0.0 m**, sites **0.0 m**; mirror is an involution (0.0); the eval's strike detector on each mirrored clip = the mirrored strikes (8/20/10, limbs swapped, frames, types, bands equal, apex within rounding); detect on the originals = strikes.json train rows |
| extra: clean masks | **PASS**: the row replay through `scenario.clean_starts` on the originals = fb_motion's masks, 27,900 cells, 0 mismatches |
| extra: shown reference | report: the originals' reference in the T1 motion vs fb_motion's (T0's own): joint pos 7.3e-5 rad, joint vel 3.1e-3 rad/s, anchor pos 1.5e-4 m, anchor quat 1.0e-3 (up to sign). Two float32 csv_to_npz time grids of different motion lengths; the same scale as T2's C4, which C5 showed does not change behaviour. P2 should compare against the pod's converted npz |

Tests: `tests/test_fb_library.py` 13 passed (25 s). Full suite 379 passed, 2 skipped (section 6).

## 4. Commands (python = scratchpad venv, `-I`, PYTHONIOENCODING=utf-8, cwd = repo)

```
$PY -I gpu/export_fb_library.py --workers 2                      # 22 min (clean masks dominate; 15 s with --no-clean)
$PY -I scripts/fb_m8_check.py --workers 2 --out <build>\m8\m8_check.json     # 6.4 min
$PY -I scripts/fb_seeds.py --audit <SEED_LEDGER.md> --scan       # ok: 20 roles disjoint, ledger consistent, 36 task inputs clean
$PY -m pytest -q -p no:cacheprovider --basetemp="<scratchpad>\pytest-tmp" tests
```
Ledger: one row appended (60500-60595, role `l1_perf`, seed-free bookkeeping ids as T2 used them; re-runs noted).
Before the final run: one check run crashed before any tracker run (replay factory keyword), one full run preceded the diagnostic.
No MUJOCO_LOG.TXT in the repo; no stray pytest folder.

## 5. Open items and options (not decided here)

1. **Mirrored rows and T1 (decision for the main session).** As frozen, (3) says all families need T1 and T1 must learn the
   mirrored half of L1 from T0, which falls on it 23-24 times in 24. Option A (frozen path): train T1 on all of L1; T1-b then
   needs >= 90% per family after cuts. Option B: drive mirrored rows with the mirror-conjugated tracker (T0 for mirror rows now,
   T1 for warp/reangle/combo), which costs nothing and performs mirrored rows as T0 performs the originals. It would also make
   the conjugated T1 cover both sides. B changes the plan's controller table, so it is not adopted here. Either way, T1-c
   (knockdown parity) must hold for whatever drives the T1 rows.
2. **Slow bias of warps** (section 2): a consequence of the frozen speed rule; disclose, or decide a rule change before P2.
3. **D2 4.2 "p 0.2 swap to the other hand's mirrored strike"** (M3_NOTES open item): L1's mirrors now exist; AimOp still has no
   such knob. M9 decides.
4. `SparMotionCommand` copies mjlab 1.2.0's `_adaptive_sampling` (local source `scratchpad\mjlab-src`, commands.py 246-288); it
   cannot run here (no mjlab). P2's first iterations must smoke it (sampling metrics; originals' share 0.5).
5. The fullbody task's T1 rows (M9) can read `fb_l1.npz`: q, reference, strike tables (aim intents), clean masks at 0.35-0.65 m.

## 6. Full suite

`$PY -m pytest -q -p no:cacheprovider --basetemp=<scratchpad>\pytest-tmp tests`: **379 passed, 2 skipped** in 1283 s (the 2 skips are
the env-gated tests), log `build\m8\m8_suite.log`. The suite run left a MUJOCO_LOG.TXT ("Attaching a spec with an empty prefix"
warnings, 09:22); it was deleted. No stray pytest folder in the repo.
