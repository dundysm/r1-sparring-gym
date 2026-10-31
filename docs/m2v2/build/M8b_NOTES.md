# M8b build notes: Amendment A6.1-A6.4 applied to M8 (L1, LOCO, mirror conjugation), 2026-10-30

Builds on M8 (build\M8_NOTES.md, M8_VERIFY.md). Every frozen M8 check (1)-(4) still passes. A new check (5) proves the mirror
conjugation exact on the recorded eval rows; it passes at 2e-15 (f64) and 3.8e-6 (f32 action). L1 and LOCO were regenerated with
clean masks. The full suite passed (section 6). Nothing was committed. No GPU was used, and every process ran with at most 2 workers.

**Headline.**
- **A6.1 (combos):** 4 of 13 tried splices survive (fades 8, 9, 10, 17 frames). The M8 code accepted 8 of 13 and the M8 verifier
  predicted 1 at the drawn fades.
- **A6.2 (performability with the conjugated tracker on mirrored rows):** mirror 23/24 (0.958), warp 22/24 (0.917), reangle 20/24
  (0.833), combo 22/24 (0.917). On that basis only **reangle** is below T1-b's 90% bar.
- **The 4 reangle falls are all T0's own fall** on kfab_200 at t0 2.5 s, which falls at 0.52 s on the unmodified original as well.
  No reangle fall comes within 1 s after a modified frame.
- **Plain T0 on mirrored-base rows** (M8's reading, now a diagnostic): 0/24, 0/12, 1/12 and 1/12 stay up.

## 1. Files

| file | change |
|---|---|
| `gpu/fb_mirror.py` (new) | Reusable conjugation, pure torch + numpy on any device. `mirror_obs` (132-d, Tracker.act's order) and `mirror_act` are signed permutations: exact involutions in every float type. `mirror_qpos` / `mirror_qvel` (31 / 30), `mirror_ref`, `mirror_joints`. `MirrorPolicy(policy, mask)` wraps any callable obs -> actions (TrackerMLP, or the replica's ONNX actor) with a per-env bool mask (`.set(ids, value)` at resets), using one policy call per step. `conjugate(drv, mask=None)` wraps `drv.mlp` of a `TrackerDriver` and never double-wraps. The driver's own obs, facing, steering and `last` stay in the real world. Usage in the replica or the mjlab task: `pol = FM.conjugate(drv, mask)`, then at reset `pol.set(ids, conj[row])` with `conj` from `fb_l1.npz`, and feed the row's npz reference as usual |
| `gpu/export_fb_library.py` | **A6.1:** `speed_over(new, [spans], vmax)` measures each span on its own frames, so a join is never a speed. `splice()`. `combo_row`: per splice, the fade (both ends) is the shortest in `FADES` = 8..17 frames (the frame grid of [0.15, 0.35] s) that fits (seam, guard) and passes max(vmax, host window's own fastest, source segment's own fastest) + 0.05; if none passes, the splice is rejected. The two U(0.15, 0.35) draws are still made but not used, so the generator stream (source strike, mirror) is M8's. **A6.3:** `bases(Q, clips)` computes vmax per clip set; `library(Q, clips)`; `--loco` builds the library over kfab_200/201 with the same generator seeds, vmax of that set, and no combo row whose source is kfab_203. It writes `fb_l1_loco*` and `fb_l1_track_loco*`. **A6.2:** `t1_rows` puts the T1 motion together: originals, original-base rows, and mirror(row) for the mirrored-base warp, re-angle and combo rows. The mirror family is left out. Index rows carry `conj`. `fed()`: the npz reference of a mirrored-base row is `mirror_ref` of T1's shown reference of its un-mirrored entry (the host original for the mirror family), so the conjugated driver shows T1 exactly its training reference. npz `conj` [R]. **A6.4:** manifest `disclosures` (warp skew with the numbers, conjugation, combo survival, vmax set), `vmax`, `combo_fades` |
| `gpu/r1_track_spar_cfg.py` | `build_motion(loco)` reads `fb_l1.npz` or `fb_l1_loco.npz` and lays the motion out with `L.track_motion` (conjugated rows un-mirrored, no mirror family). The index json equals the export's. Docstring updated |
| `scripts/fb_m8_check.py` | (1) on L1 and LOCO, plus: conj flags, T1 CSV = rebuild, vmax = p99 of the manifest's clip set, `check_combos` (A6.1 recomputed independently: the accepted fade passes, every shorter fitting fade fails, rejected splices fail at every fitting fade, the row q equals the splices applied to the base). (2) on L1 and LOCO. (3) now drives mirrored-base rows with `FM.conjugate` (`src 'l1conj'`), reports plain T0 on them as a diagnostic, and compares the mirror rows start for start with T0 on the originals. **(5) new:** conjugation exact on recorded rows (`--rec`, required). Shown-reference extra: every row's fed reference, rebuilt from the CSV, equals the npz bit for bit (L1 and LOCO). `MirroredMLP` was removed (replaced by fb_mirror) |
| `tests/test_fb_mirror.py` (new) | Joint map = the library's. Involutions. qpos mirror = `L.mirror`. dtype kept. Obs and action conjugation on 48 synthetic two-robot states (MuJoCo FK of the mirrored world) in f64 and f32. Per-env mask and re-wrap. Recorded rows (skips unless `FB_REC` is set). 4 passed + 1 env-gated skip; with `FB_REC` set, 5 passed |
| `tests/test_fb_library.py` | Updated for A6: LOCO manifest and reach, the combo rule never counts the join, `FADES`, regeneration of L1 and LOCO bit-equal without opening a held-out file, T1 motion rows = row or mirror(row) and LOCO's own vmax, conjugated runner on the kfab_201 mirror row (3 s, stays up) |
| outputs, `gpu/motions/fb/` | `fb_l1.npz` cb0bc0aa..., `fb_l1_track.csv` ec090404..., `fb_l1_track_index.json` 94ba46de..., `fb_l1_manifest.json`; `fb_l1_loco.npz` 9f939255..., `fb_l1_track_loco.csv` 38459be3..., `fb_l1_track_loco_index.json` 6c99e876..., `fb_l1_loco_manifest.json` (EM.sha256, LF-normalised as in M8) |

The pre-A6 versions of the four M8 files and the M8 manifest are kept in `build\m8b\pre\`. The edit scripts are `build\m8b\edit*.py`.
I did not touch other lanes' diffs (fb_opponent, fb_rewards, fb_m5/m6_check, fb_replica, tests). The fb_mirror wrapper is not wired
into fb_replica.py or the M9 task: those belong to other lanes, and the contract above is what they need.

## 2. Library as regenerated

L1: 33 rows, 279.5 s, 21.2 min with clean masks. LOCO: 20 rows, 187.4 s, 14.3 min.

| family | L1 rows | accept / tried | LOCO rows | accept / tried |
|---|---|---|---|---|
| mirror | 3 | - | 2 | - |
| warp | 12 | 53 / 97 (as M8) | 8 | 39 / 74 |
| reangle | 12 | 48 / 112 (as M8) | 8 | 43 / 88 |
| combo | 6 | **4 / 13** | 2 | **1 / 4** |

**Combos (A6.1).** The fade is in frames; the "failed" list gives the shorter fades that fit but failed the rule.

| row | splice at host frame | result |
|---|---|---|
| kfab_200_o0 | 41 | fade 9 accepted (8 failed) |
| kfab_201_o0 | 471 | fade 8 accepted |
| kfab_201_m0 | 306 | fade 17 accepted (8-16 failed) |
| kfab_203_o0 | 245 | fade 10 accepted (8, 9 failed) |

- The other 9 tried splices fail at every fade that fits.
- Rows kfab_200_m0 and kfab_203_m0 now have no splice, so they equal their mirrored base, the same motion as the mirror row. Reported, not acted on.
- LOCO keeps only kfab_200_o0's splice at fade 9.

**T1 motion.**
- L1: the 3 originals plus 30 rows (15 conjugated, stored un-mirrored), 27,306 frames.
- LOCO: originals 200/201 plus 18 rows (9 conjugated), 18,356 frames.

**Disclosures (A6.4).** These are in both manifests:
- **Warp speed skew.** Accepted warps average speed 0.884: 9 faster and 44 slower of 53. All 44 rejected warps are speed-ups (mean 1.154), while the drawn mean is 1.006. LOCO: 0.888 (6 faster, 33 slower); 35 of 35 rejected are speed-ups. This is a structural consequence of export_warps' rule.
- **Mirror conjugation.**
- **Combo survival.**
- **The vmax clip set.**

## 3. Check results

`scripts/fb_m8_check.py --workers 2 --rec build\rec\fb_rec_60000-60015.npz --out build\m8b\m8b_check.json` ran in 453 s and exited 0.
The log is `build\m8b\m8b_check.log`. Every ok key is true.

| check | result |
|---|---|
| (1) manifest L1 | **PASS**. 33 rows; row and frame sources in {200, 201, 203}. Output, source and row sha256 match. Index = the 30 non-mirror rows plus originals, with conj = mirror. The T1 CSV equals the rebuild. vmax equals the p99 of the clip set. Combos 4/13, recomputation bad = []. No banned name |
| (1) manifest LOCO | **PASS**. 20 rows; sources only {kfab_200, kfab_201}. Index 2 + 18 (9 conj). CSV equals the rebuild. vmax is LOCO's own (differs from L1's, tested). Combos 1/4 |
| (2) reach L1 / LOCO | **PASS**. Apex frames bit-equal 48/48 and 43/43; max window excess -0.31 mm in both. Re-detected 47/48 and 42/43, all with the same apex forward distance (0.0). Warp resampling excess (report): 4.7 mm and 2.5 mm |
| (3) performability, ids 60500-60595, as frozen + A6.2 | mirror 23/24, warp 22/24, reangle **20/24 (0.833)**, combo 22/24. families_needing_T1 = **[reangle]**. By base, T0 on the original base vs the conjugated T0 on the mirrored base: warp 11/12 vs 11/12, reangle 10/12 vs 10/12, combo 11/12 vs 11/12; mirror 23/24 conjugated. Plain-T0 diagnostic on the mirrored base: 0/24, 0/12, 1/12, 1/12 |
| (3) falls | reangle: all 4 are kfab_200 rows at t0 2.5 s, falling at 0.52 s (o0, o1, m0, m1), which is T0's own fall on the original at that start. mirror: l1_mirror_kfab_200_m0 at 2.5 s, 0.52 s (the same). warp: kfab_200_o0 (t0 2.55, 2.62 s) and kfab_200_m1 (2.61, 0.40). combo: kfab_200_m0 (2.5, 0.52) and kfab_203_o0 (t0 1.535, falls at 0.18 s; not within 1 s after a modified frame). falls_after_modified: mirror 1 (every frame counts as modified), warp 1, reangle 0, combo 0 |
| (3) mirror vs originals | 24/24 starts: the conjugated T0 on the mirror rows has the same outcome as T0 on the original at the same start (fall time within 0.1 s) |
| (3) runner identity | **PASS**. 72 original starts: runner = the eval's Tracker, 67/67, 0 mismatches |
| (4) mirror | **PASS**, unchanged: ranges 9.0e-5 <= 1e-4; FK 0.0; involution 0.0; strike detection equal |
| **(5) conjugation on recorded rows** | **PASS**. 15,412 control steps of the 16 recorded eval bouts. MuJoCo FK of the mirrored states = the reflected FK: **0.0**. Target = defender pelvis, also in the mirrored world: 0.0. The world mirror's gyro and joints: 0.0. **f64:** obs vs M_o(obs) **2.0e-15**, action vs M_a(action) **7.1e-15**, joint targets 1.8e-15; bit-exact rows 14,125 (obs) and 14,147 (action); yaw 1.8e-15, spot and axis 0.0; tolerance 1e-9. **f32:** obs 8.6e-7, action **3.8e-6**, targets 1.1e-6; bit-exact 14,018 / 14,046; tolerance 1e-5. Sanity: unmirrored obs vs the recorded obs64 is 5.6e-16 (f64) |
| shown / fed reference L1 / LOCO | **PASS**. Fed reference rebuilt from the CSV = npz on 33/33 and 20/20 rows, bit for bit. Report: originals vs fb_motion's are 7.3e-5 rad joint pos and 2.6e-3 rad/s vel (LOCO 3.1e-5 and 1.5e-3), the scale M8 reported |
| clean identity | **PASS**. 27,900 cells, 0 mismatches |

The CONJ_TOL values (1e-9 and 1e-5) are not frozen. I defined them from A6.5's diagnostic 1e-9 and G0-1's float32 action tolerance.
The rows that are not bit-exact come from the facing update: for an unset yaw, want = atan2(...) - pi sits 2 pi apart between the two
worlds, and that rounding is not sign-symmetric.

## 4. Commands

The Python is the scratchpad venv, run with `-I`, PYTHONIOENCODING=utf-8, cwd = repo.

```
$PY -I gpu/export_fb_library.py --workers 2            # 21.2 min -> L1
$PY -I gpu/export_fb_library.py --workers 2 --loco     # 14.3 min -> LOCO
$PY -I scripts/fb_m8_check.py --workers 2 --rec <build>\rec\fb_rec_60000-60015.npz --out <build>\m8b\m8b_check.json   # 453 s, exit 0
$PY -I scripts/fb_seeds.py --audit <SEED_LEDGER.md> --scan      # ok: 20 roles disjoint, ledger consistent, 41 task inputs clean
$PY -m pytest -q -p no:cacheprovider --basetemp="<scratchpad>\pytest-tmp" tests
```

- **Ledger.** One row appended: 60500-60595 re-run in role `l1_perf`, seed-free. Recording 60000-60015 was only read, not simulated.
- **Pre-checks.** A quick no-clean export ran into `build\m8b\quick` (not the repo), plus `build\m8b\probe_lib.py` and `conj.py`.
- **Repo hygiene.** No MUJOCO_LOG.TXT in the repo and no stray pytest folder.

## 5. Open items (not decided here)

1. **Reangle is below T1-b's bar (0.833).** All 4 falls are T0's own kfab_200 t0 2.5 s fall, which also happens on the original. T1-b's bar is "after cuts", and how cuts treat a start where T0 falls on the original too is for the main session. As frozen, (3) says reangle needs T1; mirror, warp and combo do not. T1 is still trained on the T1 motion (A6.2: only the non-mirror families).
2. **Two combo rows (kfab_200_m0, kfab_203_m0) have no splice under A6.1**, so they duplicate the mirrored base in L1 and the originals in the T1 motion. LOCO's combo family has only 1 splice. Keep, drop, or disclose: the main session decides.
3. **Wiring.** fb_mirror is not yet wired into fb_replica.py (M6 lane) or the M9 task. The contract is in section 1.
4. **The "p 0.2 swap" knob is still open.** It is design 4.2's swap to the other hand's mirrored strike. AimOp still has no such knob. If M9 adds it, the per-env mask can follow the swapped strike: the obs is always built in the real world, so switching the mask mid-bout is consistent.

## 6. Full suite

`$PY -m pytest -q -p no:cacheprovider --basetemp=<scratchpad>\pytest-tmp tests`: **390 passed, 3 skipped** in 3692 s. The machine was loaded by R0-EO and the M6 lanes. The 3 skips are env-gated tests, including test_fb_mirror's recorded-rows test, which passes with FB_REC set. The log is `build\m8b\m8b_suite.log`. The suite left a MUJOCO_LOG.TXT (12:04, the "Attaching a spec with an empty prefix" warning, as in M8). I deleted it. No stray pytest folder.
