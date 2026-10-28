# Head camera (phase 3, sim)

The R1's head camera in simulation, and its first closed-loop use. **M1: the R1 turns toward what its own head camera sees,
with no learning.** The preregistered test passed all five criteria. It shows the camera chain works end to end (mount,
clocks, latency, drops, ego compensation, wiring). Because the detector is a colour key, it does **not** show that the R1
can perceive a real opponent (see **Limitations**).

![camera turn](../media/camera_turn.gif)

One bout from `python scripts/demo.py --headcam --offaxis 40`, guard policy, train clip `kfab_201`. The opponent starts
40° off the R1's heading. **White ray:** the opponent's true bearing. **Green ray:** the camera estimate the R1 acts on.
**Grey ray:** its heading. Top right is the head-camera frame in use, with the detector's mask tinted. In this bout the facing
error over the last 6 s has a median of 7.1°, and the R1 first faces within 15° at 3.08 s
(`results/perception/camera_turn.json`). That is one illustrative bout, not evidence. Full video:
[`media/camera_turn.mp4`](../media/camera_turn.mp4).

## The camera (M0)
`configs/r1_camera.yaml` is the single source for every camera number. Each number carries its source (RBTX datasheet,
mybotshop, r1.xml) or is tagged ASSUMPTION / INFERENCE. Unitree publishes no intrinsics, mount pose or head-pitch range.
- **Network image:** 160×120 at 30 Hz, a rectified virtual pinhole (fovy 100°, 115.6° horizontal) inside the 146°×124°
  fisheye.
- **Mount:** the lens is on `torso_link` at the visor front, at head pitch **-50°**. That pitch is the best-scoring one in
  `results/perception/coverage.json`, which measures the opponent's unoccluded keypoints on train + val clips.
- **Timing:** random capture phase, 2–20 ms exposure, 20–60 ms latency per episode with ±3 ms jitter, and 1–2% dropped
  frames. All of these are ASSUMPTIONs.
- **Render twin:** `r1gym/perception/render_scene.py` is a separate MjModel with LOD meshes. Physics is untouched
  (`tests/test_perception_render.py`).
- **Results:** `mount.json`, `coverage.json`, `lod_report.json` and `render_bench.json` in `results/perception/`.

## M1: what it is
Nothing is trained. The stepping policy (guard) is unchanged, and so is the existing turn-to-face command (`env._face`: a
yaw-rate command toward the opponent's bearing). The only change is where `_face` gets the bearing.

```
substep hook (percept.py)  -- exposure falls due -->  snapshot: qpos (renderer only) + sensed encoders / IMU gravity + gyro yaw
PerceptHub (hub.py)        -- render the snapshot on the twin, main thread -->  ColorKeyDetector (detect_colorkey.py)
LatencyQueue (camera.py)   -- delivered at t_capture + latency, some frames dropped -->
PerceptStream.bearing()    -- latest estimate, carried to "now" by the integrated gyro yaw (ego compensation) -->  env._face
```

- **Detector:** an HSV mask for the opponent's orange (hue 4–22°, saturation ≥ 0.4, value ≥ 0.08, at least 12 pixels).
  - **Bearing:** the median azimuth of the mask pixels' rays. The rays use the nominal intrinsics and the mount pose from
    *sensed* joint angles and gravity.
  - **Range:** from the nearest floor-plane hit of the mask pixels. It is used only for the bearing's parallax
    correction.
- **Lost track:** with nothing seen for `percept_lost_s` = 1 s, the R1 stands still.
- **Firewall:** no estimate reads MjData (`test_camera_estimates_never_read_mjdata`). There is one stated exception. The
  gyro yaw integration projects onto the true vertical, which assumes an ideal roll/pitch estimate (module docstring of
  `r1gym/perception/percept.py`).
- **Settings,** all on the same episodes with the same perception seed per episode (paired):

  | setting | `_face` reads |
  |---|---|
  | blind | nothing: turn command off |
  | truth | the true bearing, as before M1 |
  | truth_lag | the exact bearing, but on the camera's clock with its latency and drops: the timing cost alone |
  | truth_lag_nocomp | the same without ego compensation (an ablation) |
  | camera | the head camera: render twin → colour key |
  | shuffled | another bout's camera estimates at the same times since its start (a seeded derangement): a control |

## Setup
`scripts/face_ab.py --offaxis` (its module docstring and `M1_DOC` give the full protocol).
- **Main ladder:** guard policy (`results/policy_mjlab_guard/policy.onnx`; it turns both ways). The bouts cover:
  - test clips `kfab_202` and `kfab_205`;
  - distances 0.55 and 0.65 m;
  - starting bearings ±30° and ±50°;
  - 12 start draws per clip and distance, so 192 bouts per setting in 24 (clip, start draw) clusters;
  - 12 s each, with the demo's three scripted pushes.

  The R1 starts turned (`init_yaw`). The steered opponent keeps to its starting world axis, so it stays off-axis unless the
  R1 turns.
- **Arms-down row:** the arms-down base, which can only turn right. Offsets -30° and -50° only, 96 bouts per setting, and
  no truth_lag_nocomp.
- **Empty ring:** camera setting. The R1 starts facing away from the opponent, which is parked 5 m behind it, beyond the
  room's wall. 24 bouts.
- **Tuning, on val only** (`kfab_204`, 6 start draws, 48 bouts): only `percept_filter_s` ∈ {0, 0.1} × `face_deadband` ∈
  {none, 15/5°, 20/6°}. The rule was fixed before the val run: the lowest camera objective wins, but within 0.5° of it the
  widest deadband wins, then the shortest filter.
  - **Choice:** no filter and deadband [0.35, 0.1] rad, i.e. turn above 20°, stop below 6° (objective 11.94° vs 12.05° with
    no deadband; `results/perception/m1_offaxis_val.json`).
  - **Val outcome:** (a) failed on val because its CI was too wide: -0.83° [-3.77, 2.05]. The other four passed.
  - **Test size:** from that CI's half-width (2.9° with 6 clusters), the test was set before any test bout to 12 start draws
    per clip and distance instead of the design's 6. The expected half-width was 1.45°. `preregistration_m1.json` → `power`
    records the arithmetic.
- **Preregistration:** `results/perception/preregistration_m1.json` was written after the val run and before the test run
  (`written_utc` 2026-10-28T01:52:11Z). It fixes:
  - the criteria and the statistics;
  - the val choice and both policies;
  - the test design;
  - the val file's hash;
  - the hashes of the code, configs and policies a bout runs on.

## Preregistered criteria and the test result
**Per-bout statistics:**
- **Facing error:** |true bearing of the opponent's pelvis from the R1's pelvis heading| at every 20 ms tick, as its
  median over the last 6 s.
- **Time to face:** the first tick within 15°; a bout that never gets there counts as 12 s.

**Comparisons:** each is the mean over bouts of the paired per-bout difference. Its 95% CI resamples whole (clip, start
draw) clusters (`r1gym/stats.py` `cluster_paired_diff`, 4000 resamples, seed 0). No bout fell in any setting, so every
pair has n = 192.

Source: `results/perception/m1_offaxis_test.json` → `criteria`.

| | rule | test result | |
|---|---|---|---|
| (a) | facing error, camera − truth_lag: 95% CI within [-3, +3]° | -1.02° [-2.32, 0.54] | pass |
| (b) | blind − camera and blind − truth_lag: both CIs above 0 | 37.16° [34.11, 40.62] and 36.15° [32.78, 39.68] | pass |
| (c) | time to face, camera − truth_lag: CI upper bound ≤ 0.5 s | +0.20 s [0.09, 0.32] | pass |
| (d) | blind − shuffled (facing error): CI lower bound ≤ 0 | -16.61° [-22.71, -10.76] | pass |
| (e) | empty ring (camera): turn command on ≤ 2% of control ticks | 0.0% of ticks (24 bouts) | pass |

**Recomputed in review.** An independent script recomputed every pass/fail from the file's `raw` rows (no `r1gym`
imports). With the same bootstrap it reproduces every stored diff and CI exactly. With a different bootstrap (20,000
resamples, another seed) every verdict is unchanged; (a) is then [-2.30, 0.58]. `tests/test_perception_percept.py`
(`test_m1_results_carry_run_meta_and_recompute`) does the same recompute with the code's own functions.

**Per setting** (`summary`; facing error = per-bout median over the last 6 s):

| setting | facing error, mean / median of bouts | time to face (mean) | faced within 15° | ticks turning | drift (mean) | ring-out |
|---|---|---|---|---|---|---|
| blind | 46.3° / 43.7° | 8.78 s | 38% | 0% | 0.17 m | 0% |
| truth | 10.4° / 9.3° | 2.12 s | 100% | 46% | 0.21 m | 1.6% |
| truth_lag | 10.2° / 9.7° | 2.08 s | 100% | 45% | 0.23 m | 1.0% |
| truth_lag_nocomp | 10.6° / 9.8° | 2.07 s | 100% | 47% | 0.24 m | 1.0% |
| camera | 9.1° / 7.5° | 2.28 s | 99.5% (one bout never) | 46% | 0.23 m | 1.0% |
| shuffled | 62.9° / 68.9° | 7.09 s | 51% | 45% | 0.20 m | 0.5% |

**The percept itself** (camera, the median over bouts of each bout's value; `summary.camera.percept_bout_median`):
- **Bearing error** against the truth at capture: median 3.8°, 90th percentile 10.5°. At use (after latency and
  compensation): 3.9° and 10.9°.
- **Range error:** median 0.095 m (15%), bias +0.04 m.
- **Latency:** median 39 ms, max 42 ms.
- **Drops and track:** 1.4% of frames dropped; presence 1.0 and no_track 0 (bout medians).
- **Calibration:** the stated 1σ covers 86% of bearing errors. The σ is a heuristic, not fitted.

**The ladder's other steps** (`paired`): these are reported, not preregistered.
- **Timing cost:** truth_lag − truth is -0.20° [-1.16, 0.87].
- **Detector error:** camera − truth is -1.22° [-2.52, 0.58]. Camera's point estimate is lower than truth's, but the CI
  includes 0.
- **Ego compensation:** truth_lag_nocomp − truth_lag is +0.42° [-0.43, 1.13], so not measurable here.
- **Time to face:** camera − truth is +0.16 s [0.00, 0.34]. Camera is slower than truth_lag by the (c) margin above.

**Arms-down row** (`arms_down`; right offsets only; reported, not preregistered):
- **Facing error:** blind 29.8°, truth 17.3°, truth_lag 19.6°, camera 18.2°, shuffled 37.1° (means of bouts).
- **camera − truth_lag:** -1.46° [-4.16, 1.24].
- **camera − blind:** -11.59° [-17.69, -6.31].
- **shuffled − blind:** +7.33° [1.16, 13.70].
- **truth_lag − truth:** +2.30° [0.01, 4.37]. This is the only ladder step whose CI excludes 0, here at the edge.

**Hits on the R1** (`summary.*.landed_per_bout`): head landed 0.10 per bout blind vs 0.036 with camera. There is no CI and
no claim, as the design asked: these were not tested.

**Cost:**
- **Test run:** 3199 s wall in total, of which the camera setting took 1254 s.
- **Rendering:** 12.3 ms per render and 2.7 ms per detection over 110,729 frames.
- **Machine:** one Windows laptop, CPU physics, Intel Arc 140V iGPU (`cost`, `gl_renderer`).

## The controls: what each shows and doesn't
- **truth_lag** has the camera's clock, latency and drops but exact values. So camera − truth_lag (criteria a and c) isolates
  what the detector's errors cost the closed loop.
- **shuffled** has realistic estimate streams from another bout. It turns as often as camera does (45% vs 46% of ticks) but
  toward the wrong place: its bearing error has a median of 63°. It ends up worse than blind, so (d) is an easy bar.
  - **What it shows:** camera's gain comes from this bout's own images. It is not from the turning itself or from a bearing
    prior.
  - **What it can't show:** anything finer than that.
- **The empty ring** shows no false detections on the R1's own body or the empty room. The opponent is never in view there
  (`criteria.e`: presence 0, no_track 1). So it says nothing about how the percept behaves after losing an opponent it had
  seen.
- **blind** is the pre-M1 shipped behaviour (both `gym.json` files have `face_opponent: false`).

## Integrity of the test run
Checked in review, without re-running the test split.
- **Order:** the run records show a clean chain: val run → preregistration → test run → demo.
  - The preregistration lists `m1_offaxis_val.json` in its `git_dirty` and pins its sha (`4b61ad235c0dc11d`, which matches
    the file today).
  - The test result's `git_dirty` lists the preregistration but not the test file itself. It pins the preregistration's sha
    (`bf0a4888d6fb9897`, which matches the file today).
  - The demo's `git_dirty` lists the test result.
  - Shas are sha256 of the LF-normalised file, first 16 hex characters (`face_ab.py` `sha`).
- **Not committed.** The design asked for the preregistration to be committed before the test run, and it was not. It was
  untracked when the test ran (the test result's `git_dirty` says so), and it still is. So the order rests on the run's own
  records (timestamps and the hash chain above), not on git history. `face_ab.py` now refuses a test run unless the
  preregistration is committed and unmodified (`m1_tracked`). That check was added after this run.
- **Dry run on the test clips.** Before the preregistration was written, the `--prereg` and `--split test` code paths were
  dry-run once on the test clips (3 s, 1 start, against a scratch preregistration, output discarded). Its printed
  criteria and summary were seen. Nothing was chosen from it: the val choice and the test size come from the val file.
  This is disclosed in the result (`pre_test_smoke_run`), and `--split smoke` (a train clip) now exists so it can't recur.
- **Code changed since the preregistration.** Of the files it hashed, two have changed since. Both changes came after
  the test run:
  - `r1gym/perception/percept.py`: docstrings only. Removing the two added passages (the firewall exception in the module
    docstring and its mention in `substep`) reproduces the preregistered hash `8b2309d0bdd84627` exactly.
  - `scripts/face_ab.py`: review follow-ups (`--annotate`, `--split smoke`, the commit check, a wider hash set, docstrings).
    The stored criteria recompute exactly from the raw rows with the current code, and with the independent script.
- **Annotations.** `--annotate` later added `criteria.e.presence`, `no_track` and `scope`, `pre_test_smoke_run` and the new
  `about` to the test result. Compared with a copy of the file taken before `--annotate` ran (not in the repo), no other
  field changed.
- **Gaps in the hash set.** The preregistration's hand-picked list missed `r1gym/policy.py`, `perturb.py`, `sensors.py`,
  `contacts.py` and the package `__init__` files. They are unchanged since commit `cdddb94`, the preregistration's
  recorded `git`, and none appears in its or the test's `git_dirty`, so the test ran them as committed. The guard now
  hashes every `r1gym/**/*.py` and `configs/*.yaml` (`test_m1_guard_hashes_the_whole_package`).
- **Path separators.** The preregistration keys the two `gym.json` files with Windows path separators. Their hashes still
  match today. The guard now normalises separators, so a re-check doesn't report them as unhashed.
- **Run as preregistered.** Every test episode matches the preregistered design: clips, distances, bearings, 12 starts,
  perception seeds from 30000, 192 + 96 + 24 bouts. The test result's choice equals the preregistration's.

## Limitations
- **Simulation only.** The camera renders the same simulator it acts in: a nominal 6×6 m grey room, fixed lights, no
  appearance randomisation, and no sensor noise, motion blur or compression in the image. The renderer and the detector
  use the same nominal intrinsics and mount, so there is no calibration error. Every timing number is an ASSUMPTION.
  Nothing has run on an R1.
- **The colour key relies on the opponent's colours.** It finds the opponent because it is the only orange thing in the
  scene; the R1 is silver, the floor blue-grey and the walls grey. A real sparring partner, or a second R1 (silver, like
  ours), would defeat it, and so would orange in the room. So M1 checks the plumbing, not perception
  (`detect_colorkey.py` says so).
- **No learning.** Nothing is trained on images yet, and the policy doesn't see the camera: only the yaw-rate command does.
- **The scenario is easy for perception.** All the starting bearings (±30°, ±50°) are inside the 115.6° view, and the
  opponent stays near its starting axis. So the R1 never has to search for it, track it around the room, or recover a
  lost track. M1 covers one turn and holding the facing through strikes and pushes. It does not cover tracking a moving
  opponent.
- **The facing metric mostly measures the controller.**
  - The deadband turns only above 20° and stops below 6°, and the guard turns slowly (`results/policy_mjlab_guard/gym.json`
    `_why`: +0.21 / -0.29 rad/s measured for a ±0.8 rad/s command). Most of the camera's bearing error (median 3.8°) sits inside that band.
  - So (a)'s ±3° says the closed loop does as well on the camera as on exact values. It does not say the estimate is
    accurate to 3°.
  - For the same reason this setup can't resolve the costs of latency or ego compensation: truth vs truth_lag and the
    nocomp ablation are not distinguishable (guard).
- **The arms-down policy can only turn one way.** It turns right only (`results/policy_mjlab/gym.json`: +0.8 rad/s →
  -0.04 measured), so its row uses right offsets only and isn't part of the criteria.
- **Head pitch.** -50° is the best pitch for the sim's keypoint coverage (`coverage.json`; the selection rule was revised
  once, on train + val, and `coverage.json` keeps both rules). The real R1's head-pitch range is unpublished.
- **Ideal attitude.** The yaw integration assumes an ideal roll/pitch estimate (the firewall's one exception, above).
- **Clusters.** A start draw's 4 bearings share a clip time, but each distance picks its own clean start for the same draw.
  On test those differ by up to 0.54 s (the `episodes` field). The clustering is by (clip, start draw), as preregistered.
- **Determinism.** Rendering is deterministic on one machine (`render_bench.json`). Other GPUs or drivers may give other
  pixels.
- **No change to the shipped policies.** Both `gym.json` files keep `face_opponent: false` and `perception` defaults to
  truth. The headline eval is unchanged.

## Reproduce
```bash
python scripts/face_ab.py --offaxis --split smoke --starts 1 --seconds 3   # code check on a train clip (never evidence)
python scripts/face_ab.py --offaxis --split val                            # tuning + val ladder -> m1_offaxis_val.json (~36 min)
python scripts/face_ab.py --offaxis --prereg                               # preregistration_m1.json (once)
python scripts/face_ab.py --offaxis --split test                           # once, refuses unless the preregistration is committed (~53 min)
python scripts/demo.py --headcam --offaxis 40                              # media/camera_turn.{mp4,gif}, camera_turn.json
```
Wall times are from the stored `cost.wall_s` on the machine above.

## Next: M2
**M2** builds the image dataset that a learned estimator needs.
- **Labels:** keypoints, presence, bearing, range and the opponent's relative heading, with the camera pose from sensed
  proprioception.
- **Visual domain randomisation on the render twin:** robot colours (including a silver, look-alike opponent), materials,
  rooms and lights, camera extrinsics and intrinsics, and image effects (exposure, noise, blur, JPEG).

**M3** replaces the colour key with a CNN estimator trained on that data. Its open-loop accuracy is preregistered against
constant baselines.

**M4** puts the estimator back in this loop on the same ladder (truth → truth_lag → camera, with shuffled and blind), and
adds a moving (orbiting) opponent that the R1 has to track.

Until M3, the result above should be read as "the camera chain works", not "the R1 can see".
