# M0 + M1 build notes (M2V2_PLAN_v3 section 3), 2026-10-29

Status: **M0 PASS, M1 PASS**, every check as frozen; no threshold changed. Repo `r1-sparring-gym-spar` on feat/sparring-m2v2 @ b81fa16:
only new, untracked files (`git diff` empty), nothing committed, no MUJOCO_LOG.TXT left. CPU only, single process. **No seed block
used** (no episode was simulated; `clean_starts` uses the eval's own constant env seed 0), so nothing was appended to SEED_LEDGER.md.

## Files (all new)
| file | what |
|---|---|
| `scripts/fb_seeds.py` | held-out guard: `check(seeds, role, clips, guard)`, 19 roles = the 6.4 ledger blocks, `TestOnce`, `ledger_problems()`, `audit(SEED_LEDGER.md)`, banned-name `scan()` over `task_inputs()`; CLI `--audit --scan` |
| `tests/test_fb_hygiene.py` | M0 (1)-(3), 53 tests |
| `gpu/export_fb_motion.py` | train-only reference motion, clean-start masks, strike table -> `gpu/motions/fb/fb_motion.npz` + `fb_motion_manifest.json` |
| `gpu/export_fb_hulls.py` | m8 hulls + Hausdorff gate -> `gpu/motions/fb/fb_hulls.npz` + `fb_hulls_manifest.json` |
| `scripts/fb_m1_check.py` | M1 (1)-(5) against the eval's own code (the only place the tracker ONNX is read) |
| `tests/test_fb_export.py` | M1 checks in pytest (clean starts at 6 of 31 distances), Hausdorff cross-check, regeneration, no held-out file opened; 10 tests |
| `gpu/motions/fb/*` | fb_motion.npz (627 kB, sha256 b66f0f7e7d5aba7f...), fb_hulls.npz (205 kB, sha256 bbe073fc5c5b312e...), two manifests (deterministic zip timestamps, so the sha256 is reproducible) |

## Commands
```
PY=...\scratchpad\venv\Scripts\python.exe   (PYTHONIOENCODING=utf-8, cwd = repo)
$PY -I gpu/export_fb_motion.py                 # 110 s (clean starts dominate)
$PY -I gpu/export_fb_hulls.py                  # 8 s
$PY -I scripts/fb_m1_check.py --out ...\build\m1_check.json      # 111 s, full M1 check, exit 0
$PY -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan      # "ok: 19 roles disjoint; ... consistent; ... task inputs clean"
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp=...\pytest-tmp-m1 tests/test_fb_export.py tests/test_fb_hygiene.py   # 63 passed, 40 s
$PY -m pytest -q -p no:cacheprovider --basetemp=...\pytest-tmp tests      # full suite: 235 passed, 1 skipped (FB_SEED_LEDGER unset), 257 s
```
Full numbers: `build\m1_check.json`.

## M0 results
1. **Grep test** PASS: `scan(task_inputs())` = [] over 8 files (gpu/export_fb_*.py, scripts/fb_*.py, gpu/motions/fb/*; globs also cover
   the later gpu/fb_*.py, fullbody_task.py, r1_track_spar_cfg.py, hybrid_actor.py, pod_fb.sh, export_hybrid_fb.py). Banned: `kfab_20[2456]`,
   `r1_kungfu*.csv|npz` (broader than the plan's `r1_kungfu.csv`: the aim CSV and the npz also hold every clip), and
   `opponent_tracker*/policy.onnx` (not every `policy.onnx`: the defender's own mjlab export has that name). The scanner reads bytes as
   ASCII, UTF-16 and UTF-32 and npz string arrays/keys; a planted-name test catches all 8 tokens in .py, UTF-16 .json and npz.
2. **B7 seed guard** PASS: all 19 roles refuse 21000-21999 and (except `test`) 22000-22511; 23000-23255 only `screen_val`; 24000-25999
   only `track2`; seeds must lie in the role's own 6.4 block; clip sets: dev roles a non-empty subset of the train clips (LOCO roles
   their own subset), `screen_val` the val clip, `test` the test clips, `track2` none of kfab_199-206. `test` needs a granted `TestOnce`
   (fb prereg exists, selection go = true against the prereg's LF-normalised sha256[:16], no `raw_test_hard_r2*.json` under
   results/eval_opponent). Today it refuses (no `prereg_m2_fb.json`); M11 wires it to `m2_verdict --run fb`.
3. **Disjointness** PASS: role blocks pairwise disjoint and disjoint from the pre-v3 used/retired blocks (50000-50255, 51000-51999,
   52000-52255, 53000-53191, 54000-54255, 55000-55255, 55500-55691, 58100-58143, 59000-59255, 66000-66047, 66100-66291, 67000-67767,
   68000-68767; retired 57000-58071), with the one reuse 6.4 declares (r2_diag = R4-pre 67000-67767). SEED_LEDGER.md audit: all 4 rows
   consistent. T1-c's 70000-76143 sits inside role `eq` (70000-82287) by design, not as a separate block.
   - Ledger interpretation (6.4 lists "50000-59255 (various)"): encoded as a legacy zone in which only S3 56000-56191 may be reserved;
     51000-51xxx taken as the whole 51000-51999; 58100-58143 (P1/P2b verifier spot checks) added from the run4pre notes.
   - The workstream-ledger test needs `FB_SEED_LEDGER` (the ledger lives outside the repo); without it that one test skips.

## M1 results
| check | rule | result |
|---|---|---|
| (1) clip set + sha256 | `== {kfab_200, 201, 203}` | PASS; outputs' sha256 match; 0 stale sources (19 + 5 source files hashed: clip npz, retarget caches, cut/strike json, r1.xml, config, motion_index.json, r1gym modules, exporters) |
| (2) T-ref vs `Tracker.frame(c, f)` | <= 1e-4, indices inside train rows | PASS. joint_pos **0** and joint_vel **0** (bit-identical, 100% of 1248 frames x 24); anchor pos **6.0e-8** m, anchor quat **1.2e-7**. Rows used: kfab_200 21-270, kfab_201 501-1222, kfab_203 2151-2426 (each inside its own [start, start + 2T)). q at the 4 eval distances = q0 + (d, 0): 1.1e-16 |
| (3) clean masks vs `SC.clean_starts` | exact | PASS: 93 (clip, distance) cells x 300 starts = 27,900, **0 mismatches** (fresh call in a separate process). Grid 0.35-0.65 m by 0.01 (design 3.2); clean share 0.441 / 0.311 / 0.456 |
| (4) strike table | strikes.json train rows | PASS: 38 rows = 37 hand + 1 kick, every field equal |
| (5) Hausdorff, 64 meshes | <= 3 mm fine (7 links x 2 robots), <= 1 cm rest | PASS. Worst fine **waist_yaw 2.971 mm** (head_yaw 2.519, elbows 2.29/2.09, wrists 1.05, head_pitch 0.29); worst coarse **pelvis 9.366 mm** (knees 8.24/8.30, left shoulder pitch 6.02). Sizes: 256 on the 7 fine links, 64 elsewhere, imu_in_pelvis exact (42 vertices) |

### Findings that matter downstream
- **The tracker's reference is not FightReplay's clip.** A straightforward export (6-decimal CSV, float64 central differences) misses
  the ONNX initializers by up to 7.5e-5 rad in joint_pos and **1.5e-3 rad/s in joint_vel** (fails (2) and would fail M2's G0-2 at 1e-6).
  Cause: csv_to_npz on CUDA interpolates on a float32 time grid of the whole 3234-row motion, so frames late in the motion get
  interpolation weights 1 float32 ulp off (up to 2.4e-4 at rows > 2048). Reproduced exactly: `t = f32(i) * f32(0.02)`,
  `p = f32(t * f32(1/duration)) * f32(n - 1)`, float32 lerp, mjlab slerp, gradient `(x[i+1] - x[i-1]) * f32(50) / f32(2)`
  (identified by search over 15 arithmetic variants; only this one is exact on all frames). This needs each train frame's row in the
  tracker motion, so the exporter reads `results/opponent_tracker/motion_index.json` (train rows, the total length 3234 and the 20-frame
  cross-fade). The total length depends on the held-out clips' lengths (metadata, not motion); nothing of it is stored in the task
  inputs, and no held-out motion is read (tested by intercepting `open`).
- kfab_201's first rows are cross-faded with kfab_200 in the tracker motion; they are rebuilt from kfab_200 (train). Without that, one
  kfab_201 frame missed by 1.1e-4 rad/s (would fail (2)). kfab_203's predecessor is a held-out clip: its cross-faded rows are taken
  unblended; measured effect 0 (all kfab_203 frames exact).
- Hausdorff here = between the convex solids; the reduced hull lies inside the full one, so it is the largest distance from a full-hull
  vertex to the reduced hull's triangles. Cross-checked against the support-function identity on 20k directions (3 meshes).
- **waist_yaw passes with 0.029 mm of margin** (2.971 vs 3 mm). Any change to MuJoCo's hull or the FPS seed point could flip it; the
  plan's fallback (all 256 / full) would then apply at P0, not a threshold change.
- Clean starts call the eval's own `clean_starts` with the eval config (`opponent_mode: policy`), which loads the tracker ONNX as a side
  effect of building FightReplay; the ONNX is not used for the masks and is not a task input.

## Disclosures
- Read only train rows of `motion_index.json`; the file (also read by `Tracker` at every eval) lists the held-out clips' start/frame
  counts, which tests/test_strikes.py already pins. No tracker results file, no val/test seed, no kfab_204 simulation.
- MUJOCO_LOG.TXT (MuJoCo's "attaching a spec with an empty prefix" warning from `build_spec`) appeared twice in the repo during the
  runs and was deleted each time. A mistyped `--basetemp` once created a temp folder in the repo; removed.
- Probe scripts used to find the float32 pipeline: `build\probe_*.py` (scratchpad only).
