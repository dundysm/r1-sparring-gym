# M2 build notes (M2V2_PLAN_v3 section 3, M2), 2026-10-29

Status: **M2 PASS** (G0-1, G0-2, G0-3, G0-3b). Every check ran as frozen and no threshold was changed. Repo `r1-sparring-gym-spar` is on
feat/sparring-m2v2 @ b81fa16. Only new, untracked files were added (`git diff` is empty) and nothing was committed. Any MUJOCO_LOG.TXT
was deleted. CPU only, with at most 4 worker processes. Seed block **60000-60015** (role `record`) was used and appended to SEED_LEDGER.md.

## Files (all new)
| file | what |
|---|---|
| `gpu/fb_opponent.py` | Driver part of the opponent (pure torch and numpy, float64 or float32). `pin_precision()` / `precision_ok()` (m12: tf32 off, matmul precision "highest"). MuJoCo quaternion ops and `differentiate_pos` (mj_differentiatePos). `TrackerMLP` loads `model_13749.pt`; the divisor is std + 0.01 summed in float32, which is the constant the ONNX folded. `Motion` reads fb_motion.npz and asserts train clips only. `TrackerDriver` is batched: `reset` (placement, with an optional bout origin and heading), `clock`, `reference`, `face`, `steer`, `observation`, `step`. `MjTracker` is a drop-in for r1gym's Tracker inside R1HitEnv, for the M6 replica. It never reads the tracker ONNX (tested). |
| `scripts/fb_record.py` | Records CPU eval states from `eval_opponent.run` itself: G, aim standoff+armwarp_vz from aim.json, hard distances, 20 s, DR, delays as drawn, no pushes. It uses train clips at seeds 60000-60015 (k = s - 60000: clip k mod 3, distance (k // 3) mod 4) and episodes from `eval_opponent.episodes` with the M1 clean masks. Hooks are read-only: Tracker.reference / act and a numpy proxy that captures the float64 obs row before its float32 cast. Each bout also runs without hooks and the rows must be identical (asserted). It calls `fb_seeds.check(..., 'record')` and asserts G's home pose is in REF_POSES. |
| `scripts/fb_m2_check.py` | G0-1, G0-2 (a) and (b), G0-3, and G0-3b (eval_tracker.run unchanged, with the driver swapped in for the Tracker), plus placement vs FightReplay.place. Like fb_m1_check, it reads the tracker ONNX only as the reference, through the eval config. |
| `tests/test_fb_opponent.py` | 14 tests, about 9 s: the contract constants (HOME, kp, mjlab caps, scale = Tracker's from the ONNX metadata, taus), precision flags, quaternion ops and differentiatePos vs mujoco, placement vs `FightReplay.place` (3 clips x 6 t0, bit level) and with a bout heading/origin, the actor vs the ONNX (f32 and f64) and the obs/facing/steering on a 1 s recorded bout (seed 60001), the recorder refusing other blocks and assigning seeds independently of the worker split, short tracker-alone parity (2 starts, 3 s), and no ONNX opened by the driver. |

Recording (scratch, not in the repo; 50 MB): `...\fullbody\build\rec\fb_rec_60000-60015.npz`. Check output: `...\fullbody\build\m2_check.json`, `m2_check.log`.

## Commands
```
PY=...\scratchpad\venv\Scripts\python.exe   (PYTHONIOENCODING=utf-8, cwd = repo)
$PY -I scripts/fb_record.py --out ...\build\rec\fb_rec_60000-60015.npz --seeds 60000-60015 --workers 4     # 26 s
$PY -I scripts/fb_m2_check.py --rec ...\build\rec\fb_rec_60000-60015.npz --out ...\build\m2_check.json --workers 4   # 132 s, exit 0
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp=...\pytest-tmp tests/test_fb_opponent.py tests/test_fb_hygiene.py  # 67 passed, 9 s
$PY -m pytest -q -p no:cacheprovider --basetemp=...\pytest-tmp tests     # full suite: 250 passed, 1 skipped (the FB_SEED_LEDGER test), 189 s
$PY -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan              # ok: 19 roles disjoint; ledger consistent; 11 task inputs clean
```

## Recording
16 bouts with 15,412 opponent control steps. One defender fall (60014 at 8.24 s) and 2 opponent knockdowns (60004, 60005), so downed-opponent
states are in the file. With and without hooks, all 16 bout rows are identical.

## Results (thresholds as frozen)
| check | rule | result |
|---|---|---|
| G0-1 f32 | torch MLP vs onnxruntime, 15,412 recorded obs rows, <= 1e-5 | **PASS 7.63e-6** (batched); 6.68e-6 at batch 1 (every 30th row). The weights and the folded divisor equal the checkpoint bit for bit. onnxruntime here reproduces the recorded eval actions exactly. |
| G0-1 f64 | torch float64 vs the ONNX graph in float64, <= 1e-9 | **PASS 1.07e-14** |
| G0-2 (a) | obs builder on the recorded internals vs the eval's float64 obs row, <= 1e-6 | **PASS 5.6e-16** |
| G0-2 (b) | driver path (placement, clock, fb_motion reference + the aim's recorded warp and distance, recurrent facing/steering, torso matrix from the quaternion), <= 1e-6 | **PASS 2.69e-7**. This is the M1 anchor difference (quaternion 1.19e-7 vs the ONNX initializers). The clock and frame equal the eval's at every step, and the joint reference is bit-equal. |
| G0-3 | facing yaw, spot, axis over 16 bouts, recurrent, <= 1e-6 | **PASS** yaw 2.8e-17, spot 1.1e-16, axis 0 |
| G0-3b | tracker alone, 24 starts x 3 train clips, fall within 0.1 s or both up | **PASS**, float32 and float64. 67 both up and 5 both fell, with max dt 0.00 s (eval falls: kfab_200 starts 12/19 at 0.52/0.30 s, kfab_203 starts 4/7/11 at 0.70/0.10/0.18 s). Mean wall per 10 s run: eval 0.56 s, driver 1.13 s. |
| placement | vs FightReplay.place | recorded bouts qpos 0 / qvel 4.4e-16; G0-3b starts (fractional t0) qpos 8.9e-16 / qvel 1.8e-13 |
| info, float32 obs path | not gated (m12 gates float32 for G0-1 and G0-4 only) | obs 6.6e-6, yaw 2.4e-6, spot 1.2e-6 against the float64 eval: the expected float32 rounding |

## Findings and fixes during the build
- **The seed-to-episode assignment depended on the worker split** (first recording). k came from the position inside a worker's slice, so the
  clip and distance per seed changed with `--workers`. Fixed: k = seed - 60000. A test now asserts the result is independent of the split. The
  block was re-recorded and all checks re-ran on the new file. The first file was deleted.
- **G0-3b worker contamination.** A pool worker kept eval_tracker's patched FightReplay after a driver job. Fixed: it is restored in `finally`
  and asserted unpatched before each eval job. The run made before the fix was discarded before it reported anything.
- The float64 reference for G0-1 cannot be onnxruntime, which is float32. It is the graph's own Sub/Div/Gemm/Elu evaluated in float64 on its
  float32 initializers. That needs the folded divisor in the torch module (std + 0.01 in float32, not float64), otherwise the gap is about 1e-9.
- The clock is computed in float64 in both modes (round half to even, like Python's round). In float32, t x 50 can land on .5 for the
  eval_tracker starts (for example kfab_200 s = 6).

## Open notes
- **G0-1 float32 margin is about 1.3x** (7.6e-6 vs 1e-5; |a| up to 17). The difference comes from GEMM summation order (MLAS vs MKL). A GPU
  cuBLAS order could come close to 1e-5. P0 should repeat G0-1 on the pod's device before training.
- G0-2 (b)'s 2.7e-7 floor comes from M1's anchor export (6e-8 m position, 1.2e-7 quaternion). It is within the gate by 3.7x.
- The driver's `reset` supports a bout origin and heading for training resets; the eval case (0, 0) is bit-identical. M9 must pass mjlab's
  env origin there.
- `MjTracker` asserts no aim in the state. M3 adds AimOp and feeds `step(ref=..., dist=...)`, which G0-2 (b) already exercises with the
  recorded warp.
- The recording also holds full qpos/qvel, the defender's pelvis/torso and the base reference per step, for M3 G0-4 and M4 G0-5.
