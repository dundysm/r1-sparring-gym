# M3 build notes (M2V2_PLAN_v3 section 3, M3: torch aim, G0-4), 2026-10-29

**Status:** M3 is built. Five of the seven G0-4 gate groups pass as frozen: the parameters, FK and Jacobian (float64 and float32),
AimOp in float64, and the slot ring. Two frozen gates fail, and each one is **impossible as specified**: the eval's own operator does
the same thing, shown with r1gym's own code. No threshold was changed, and both gates stay failed in `m3_check.json` `ok` (exit 1).
1. **AimOp float32, warped q <= 1e-4 rad.** Fails on 4 of 15,412 rows (max 1.94e-3 rad). The eval's own `Aim.ik` moves by the same
   amount when given float32-rounded states.
2. **Jitter, palm forward distance at apex <= 1 mm.** The eval's own nominal vz warp already moves the palm's forward distance by
   more than 1 mm at 212 of 218 aimed apexes. The premise "vz holds x" is false.

Both need a plan decision (options below). Repo `r1-sparring-gym-spar` is on feat/sparring-m2v2 @ b81fa16. Only new, untracked
files were added; `git diff` on tracked files is empty and nothing was committed. CPU only, at most 4 workers, $0. Any MUJOCO_LOG.TXT
was deleted. Seed block 60000-60015 (role `record`) was re-run for the aim recording and logged in SEED_LEDGER.md. No new block was
used. No val or test seed, no test clip, and no kfab_204 was simulated.

## Files
| file | what |
|---|---|
| `gpu/fb_opponent.py` (M2 file, extended) | `Motion.strikes` (fb_motion's strike table). **`ArmKinematics`**: torso-frame FK and Jacobian of both arms (5 joints to the palm site), read from any compiled model and prefix (sub-model, the eval's `opp_`, mjlab later) in mj_kinematics' order. **`AimOp`**: r1gym/aim.py `standoff+armwarp_vz` for N envs (see below). Constants `ARM_IX`, `LIMBS`, `PALMS`, `AIM_P` (= the eval's merged config), `JITTER` (design_learned-tracker 4.2, T1 rows). |
| `scripts/fb_m3_check.py` | Records the eval's Aim on the 16 record bouts with a read-only hook on `Aim.reference`. Then G0-4: FK/Jacobian, AimOp float64 and float32 vs the eval, the eval-`Aim.ik` floor on the float32 misses, the ring, the jitter test, and the impossible-as-specified classification. |
| `tests/test_fb_aim.py` | 12 tests, about 40 s: the contract (ARM/PALM/params/aim.json), FK and Jacobian vs mj_jacSite in f64/f32, the chain from the `opp_` model = sub-model, the library = Aim.lib with the repeated-onset rule, the weight equal to Aim.weight (exact), AimOp = Aim.reference on a recorded 4 s bout (60001) in f64 and f32, ring 8 = 64 and overflow detected at 1 slot, reset, shadow rows, jitter bounds and the IK x target, the recorder refusing other blocks. |

Scratch outputs (not in the repo): `...\fullbody\build\m3_check.json`, `m3_check.log`, `rec\fb_aimrec_60000-60015.npz` (aim recording, 2 runs bit-identical; `rec\fb_aimrec_first.npz`), probes `probe_fk.py`, `probe_m3a.py`, `probe_m3b.py`, `m3_append.py`, `m3_patch_check.py`.

### AimOp design (all of it checked against Aim.reference)
- **Intents** use K ring slots per env (default 8; at most 4 live per clip in the strike table), plus a dump slot so the scatter needs
  no host sync. Times, clock, gates and weights are float64 in every mode.
- **Aim.made** becomes "last key made per strike": time only moves forward, and T >= 5 s > lead 0.8 s.
- **Eval quirk reproduced:** `made` is one key set per Aim, so in **kfab_203 the r_hand strike at onset 0 never gets an intent**
  (the l_hand one at the same onset takes the key). AimOp drops repeated onsets statically. The test checks this against `Aim.lib`.
- **Gates:** tilt is computed in float64 from the opponent torso R[2,2]. Touch is a per-env flag input: the same contact set as
  `Aim._touching`, supplied by M5's sensors in mjlab.
- **StandOff, ArmWarp vz, logs:** these follow the eval. Logs are closest palm, arm RMSE and, with `log_ik`, the apex IK residual. The
  best intent per limb is the max weight, ties going to the first made.
- **Precision:** `dtype` is the I/O dtype (sim state in, reference out). `compute` (default **float64**) is the aim's own arithmetic.
  This is a deliberate design choice, see G0-4 f32 below. The cost is negligible: a launch-bound 5x5 IK, 12 iterations, 2 limbs.
- **Modes:** shadow rows (`aimed=False`) make and log intents but leave rq, rdq and dist bit-unchanged. Jitter is a per-env flag.
  `reset()` makes no host sync; `overflow` counts overwritten live intents.

## Commands
```
PY=...\scratchpad\venv\Scripts\python.exe   (PYTHONIOENCODING=utf-8, cwd = repo)
$PY -I scripts/fb_m3_check.py --out ...\build\m3_check.json --aimrec ...\build\rec\fb_aimrec_60000-60015.npz --rec ...\build\rec\fb_rec_60000-60015.npz --workers 4
    # 231 s wall (recording 22 s on 4 workers); exit 1 = the two impossible-as-specified gates
$PY -m pytest -q -p no:cacheprovider --basetemp=...\pytest-tmp tests/test_fb_aim.py           # 12 passed, 40 s
FB_SEED_LEDGER=...\SEED_LEDGER.md $PY -m pytest -q -p no:cacheprovider --basetemp=...\pytest-tmp tests   # full suite (below)
$PY -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan    # ok: 19 roles disjoint; ledger consistent; 12 task inputs clean
```

## Recording
- **Bouts:** the same 16 bouts as M2 (G, standoff+armwarp_vz, DR, no pushes, 20 s), re-run through `eval_opponent.run`. Every bout
  is also run without the hook, and the rows are identical (asserted).
- **Same states as M2:** opponent qpos per step is bit-identical to M2's recording (`same_as_m2_recording: true`). A second recording
  is bit-identical to the first (arrays and intents).
- **Contents:** 15,412 Aim.reference calls and 482 intents (226 aimed, 172 tilt-gated, 84 touch-gated). 5,466 steps carry a
  warp, up to the 0.8 rad clip, and 3,024 steps have a stand-off below the bout distance. There are 1 defender fall and 2 opponent
  knockdowns, so tilt gates are covered.

## Results (thresholds as frozen)
| check | rule | float64 | float32 |
|---|---|---|---|
| params | recorded Aim.p (merged config) = AIM_P | PASS | |
| FK | palm vs Aim._palm_b; 1248 reference frames, 20k uniform, 15,412 eval-warped | **PASS** 8.9e-16 | **PASS** 1.3e-7 |
| Jacobian | vs Rt^T mj_jacSite, same sets | **PASS** 9.4e-16 | **PASS** 1.4e-7 |
| FK on real states | world palm (torso pose + FK of recorded joints) vs d.site_xpos, 15,412 | 1.2e-15 | 2.5e-7 |
| intents and gates | identical (limb, onset, t_on/t_apex/t_end exactly, gate, commit, reach) | **PASS**, 482/482, time diff 0 | **PASS**, 482/482, time diff 0 |
| StandOff dist | <= 1e-6 | **PASS** 0.0 | **PASS** 1.3e-8 |
| warped q | <= 1e-4 rad | **PASS** 1.26e-10 | **FAIL** 1.94e-3 on 4/15,412 rows (q99 5.6e-7, q99.9 2.2e-5): impossible as specified |
| closest palm-to-target per intent | <= 1 mm | **PASS** 7.8e-16 m | **PASS** 8.0e-8 m |
| ring | 8 slots = 64 slots bit for bit, no overflow | **PASS**, 0 overflow | |
| jitter | palm forward distance at apex, jitter vs nominal <= 1 mm | **FAIL** 0.143 m: impossible as specified | **FAIL** 0.143 m |
| info | rdq 6.3e-9; apex IK residual 3.5e-11; target_w 2.2e-16; arm RMSE 1.3e-12 | | rdq 0.097; ik_res 2.0e-3 |

### Why G0-4 float32 "warped q <= 1e-4 rad" is impossible as specified
- **As deployed:** float32 states in, float64 aim arithmetic, float32 out. It misses on 4 rows, max 1.94e-3 rad.
- **Pure float32 arithmetic** (reported, not deployed): 57 rows, max 0.030 rad. This is why `compute` defaults to float64.
- **The eval's own `Aim.ik`, given the IK target from the exact states and from the float32-rounded states, moves just as much on
  every missed row:**
  - row 10316: the target differs by 6.0e-8 m. The eval IK moves 1.964e-3 rad and the port moves 1.964e-3 rad. Eval vs port in
    float64: 7e-11.
  - row 6128: the target differs by 1.4e-8 m. The eval IK moves 4.9e-3 rad (the warp weight is below 1, so q moves 1.16e-3).
  - row 11639: the target differs by 3.6e-8 m. The eval IK moves 4.7e-3 rad.
  - row 1519: the target differs by 3.6e-8 m. The eval IK moves 1.1e-4 rad.
- **Mechanism:** the 12-step damped least-squares IK runs with bounds active (elbow and shoulder joints at the +-0.8 rad or range
  clip in every case; `at_bound` in the json). It does not converge in 12 steps, and it amplifies 1e-8 m input differences about
  3e4 times.
- **Conclusion:** any implementation fed a float32 sim's states misses 1e-4 rad on these rows. The float64 port equals the eval to
  1.3e-10 on all 15,412 steps.
- **Options for the main session:**
  - (a) Accept float32 G0-4 as measured: 0.026% of steps above 1e-4, everything else passing.
  - (b) Re-state the float32 gate as a same-input comparison. AimOp-f32 vs the float64 port on the same float32-rounded states
    differs only by output rounding.
  - (c) Gate the 99.9th percentile.

  None is applied here.

### Why the jitter test "palm forward distance at apex unchanged <= 1 mm" is impossible as specified
The test assumes that vz (x of the IK target = the reference palm's own forward distance) holds the warped palm's forward distance.
The eval's own operator does not.
- **The eval itself:** at the 218 aimed apexes of the recording, the eval's warped reference (recorded `rq`, not the port) puts the
  palm off the clip's own forward distance by more than 1 mm at **212/218 apexes**. The median is 4.1 cm, q90 11.6 cm, and the range
  is **-28.6 cm to +12.2 cm**.
- **Why:** the vz IK **never reaches its target** at an apex (0/218 within 1 mm; residual median 14.4 cm, max 1.06 m). With +-0.8 rad
  and 95%-range clips and the pull toward the clip's arm, the least-squares compromise trades x error against yz error.
- **Consequence:** any change of the yz target moves x. Jitter vs nominal exceeds 1 mm at 162/218 apexes (median 5.1 mm, q90 2.9 cm,
  max 14.3 cm). The commit-time jitter also moves the committed target.
- **What does hold, checked:**
  - The IK's x target under jitter is the reference palm's own, bit for bit, at every warped step, and equal to the nominal one (the
    x component of the target jitter is discarded).
  - The signed mean of (jitter - nominal) forward distance at apex is **-1.8 mm +- 1.6 mm (SE)**: no systematic reach added.
  - The largest forward excess over the clip is 12.37 cm with jitter vs 12.22 cm nominal (+1.6 mm).
  - Stand-off depth jitter changes dist by -2.9 to +3.0 cm (mean +0.8 mm).
- **Finding for the record:** aim.py's docstring says "the arm can't add reach". At the reference level, the eval's own vz warp puts
  the palm **up to 12.2 cm further forward than the clip** at some apexes (and up to 28.6 cm back).
- **Options for the main session:**
  - (a) Replace the gate with what holds: the IK x target is the reference palm's own (bit-exact), plus no systematic reach. For
    example, the mean forward change at apex has its 90% CI inside +-5 mm, and the forward-excess maximum is within the nominal's +
    a margin.
  - (b) Jitter only the stand-off depth (the arm then sees bit-identical targets, so x is unchanged trivially). This drops the
    target and commit-time jitter.
  - (c) A reach-safe IK for jittered rows (hard x constraint). This departs from the eval operator on T1 rows.

  Not decided here.

## Full suite
With the prescribed command and FB_SEED_LEDGER set: **264 passed, 0 skipped, 242 s** (`build\m3_suite.log`). The run includes the
12 new tests, and M0-M2's tests still pass on the extended fb_opponent.py. The full suite wrote a MUJOCO_LOG.TXT to the repo root,
which was deleted. `git diff` on tracked files is empty.

## Open notes for later milestones
- **M9:**
  - `now` must reach AimOp in float64 (d.time + dt, as the clock). `touching` must be exactly Aim._touching's set: opponent
    strike-limb geoms vs defender head/torso geoms, from M5's sensors.
  - Pass the opponent's torso_link pose from mjlab and `q_act` only for logging.
  - `torch.compile` (plan 2.2) is not applied on CPU; P0 should time AimOp with compute float64 on the pod's GPU (batched 5x5 solves)
    and assert `precision_ok()`.
- **M6:** there is no MjTracker+AimOp adapter yet (MjTracker still asserts no aim). M6's torch backend needs one; AimOp.step takes
  exactly what Aim.reference reads.
- **M8:** the D2 4.2 jitter knob "p 0.2 swap to the other hand's mirrored strike" needs L1's mirrors and is not implemented. JITTER
  holds the target, commit and depth knobs only. Their fate depends on the decision above.
- `log_ik` uses `nonzero` (a host sync); it is meant for validation and metrics, so leave it off in training.
