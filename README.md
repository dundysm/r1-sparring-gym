# r1-sparring-gym

MuJoCo simulation and RL for turning a **Unitree R1** into a sparring partner that can take hits from a real opponent and stay up.

[![showcase](media/showcase_loop.webp)](media/showcase.mp4)

One held-out hard-tier bout (test episode 16, picked by a fixed rule in [`showcase/pick_bouts.py`](showcase/pick_bouts.py)) run once with each controller on identical physics and hits: the rigid PD stance (left) falls at 5.5 s, the footwork policy (right) stays up. Rendered in Blender Cycles from the recorded simulation, not animated by hand; every number on screen is read from the results files. **Full 34 s video: [`media/showcase.mp4`](media/showcase.mp4)** ([GIF fallback](media/showcase_loop.gif)). Pipeline: [`showcase/README.md`](showcase/README.md).

![demo](media/demo.gif)

The raw eval view. **Silver:** our R1 running the two shipped policies, stepping balance controllers trained on GPU from Unitree's own R1 task and fine-tuned for footwork: arms down (left) and holding a boxer's guard (right). When the bout pushes them off their spot they walk back to it, and the guard turns to face the opponent. **Orange:** the opponent, a second R1 **on its own motors**. A motion-tracking policy performs cleaned KungFuAthleteBot fight clips (here `kfab_201`, with its idle stretches and posing cut), and the sim steers it to stay in front of our R1 and face it. No invisible forces hold it up. **Yellow spheres** are real contacts between the two robots (size ∝ force); **orange arrows** are scripted pushes from the side and behind. Full video: [`media/demo.mp4`](media/demo.mp4). Reproduce: `python scripts/demo.py`.

**Status (all simulation, held-out test clips):**
- **Hard tier** (20 s bouts; the opponent 0.40–0.55 m away and stepping in; six 8–14 N·s strikes and pushes from any direction; randomised robot physics): the guard survives **99.2%** of bouts (95% CI 95.7%–99.9%) and the arms-down policy 90.6%. PD and the residual policies survive 0%.
- **Holding the ring:** 81.2% (guard) and 72.7% (arms down) of hard bouts end with the R1 still up *and* never pushed more than 1 m from where it started. The policies shipped before this PR managed 28.9% and 30.5%, with no outer loop. This PR's hold-your-spot loop alone lifts those same policies to 36.7% and 64.8%; the rest comes with the footwork fine-tune (for arms down together with a looser loop setting, and for the guard with turning to face; below).
- **Standard tier** (10 s, two side/back pushes): 100% (guard) and 100% (arms down) on test; 100% and 100% on validation.
- **Single hits:** fall rate 0.021 (guard) and 0.030 (arms down) over 2–30 N·s, vs 0.494–0.499 for the residual policies (v1–v3). Both stay up at least half the time up to **35 N·s**.
- **The opponent** stays upright in **46/48** 10 s runs on its own, and in bouts on train + val clips (`selection.json`) it stays a median 4° off straight in front of our R1.
- **Contact:** the robots collide on their visible shells. In 48 bouts on train + val clips (`scripts/clip_check.py`) the shells' deepest overlap is a median 0.9 cm (guard) and 0.7 cm (arms down); with Unitree's collision capsules it would be 5.4 and 6.1 cm ("clipping").

## Headline eval
`scripts/eval_opponent.py`. Two tiers, the same opponents, and every controller sees exactly the same episodes.
- **Hard (`--tier hard`):**
  - 20 s bouts with the opponent 0.40–0.55 m away.
  - Six scripted strikes and pushes per bout, 8–14 N·s each, from any direction at any hit location (head and knees included); they can overlap.
  - The R1's mass, COM, foot friction, motor strength, armature and command latency are randomised per bout, with the residual (v3) trainer's ranges (`r1gym/env.py`; listed under "v3 attempt" below). The stepping policies trained in mjlab with only foot friction, encoder bias and torso COM randomised, so mass, motor strength, armature and latency are outside their training.
  - The settings were fixed before looking at results and checked on validation only, once the stepping policies had reached the ceiling of the standard tier.
- **Standard (`--tier standard`, the default):** 10 s, the opponent 0.50–0.70 m away, two scripted side/back pushes (4–6 N·s).
- **Episodes:** each samples a clip, a distance, a start time, the obs delay (0–2 steps) and the hits.
  - Start times where the opponent's limbs would begin inside the R1 are skipped: 7%–85% of them, by clip and distance.
  - Before that fix, up to a quarter of standard bouts began with a foot or fist already inside the R1.
- **Size:** test is 128 hard and 160 standard bouts (kfab_202, kfab_205); validation is 64 hard and 80 standard bouts (kfab_204).
- **Pairing:** differences are paired, with bootstrap 95% CIs. `survival_minus` and `survival_in_ring_minus` in `results/eval_opponent/summary_*.json` have every pair.
- **Ring:** "up and within 1 m" counts a bout only if the R1 is still standing at the end and its pelvis never got more than 1 m from where it started.
- **Selection:** candidates and settings are chosen on validation only, and test is reported after the choice.

There are two opponents:
- **Tracked (the default):** the R1 performs the clip on its own motors through a tracking policy and is steered to stay in front, as in the demo.
- **Puppet (stress test, `--opponent puppet`):** the same trimmed clips forced through a bounded root assist (up to 250 N horizontal, 600 N vertical), without steering. It is stronger than any robot could be.

| | **hard: test survival** [95% CI] | hard: val | hard: test, up and within 1 m [95% CI] | hard: val, up and within 1 m | standard: test [95% CI] | standard: val | puppet: test | puppet: val |
|---|---|---|---|---|---|---|---|---|
| PD stand | **0.000** [0.00, 0.03] | 0.000 | 0.000 [0.00, 0.03] | 0.000 | 0.375 [0.30, 0.45] | 0.338 | 0.075 | 0.000 |
| RL v1 (residual) | **0.000** [0.00, 0.03] | 0.000 | 0.000 [0.00, 0.03] | 0.000 | 0.656 [0.58, 0.72] | 0.637 | 0.169 | 0.000 |
| RL v2 (residual) | **0.000** [0.00, 0.03] | 0.000 | 0.000 [0.00, 0.03] | 0.000 | 0.650 [0.57, 0.72] | 0.588 | 0.200 | 0.000 |
| RL v3 cand. (residual) | **0.000** [0.00, 0.03] | 0.000 | 0.000 [0.00, 0.03] | 0.000 | 0.619 [0.54, 0.69] | 0.588 | 0.175 | 0.000 |
| Stepping base, stage 1 | **0.422** [0.34, 0.51] | 0.406 | 0.273 [0.20, 0.36] | 0.250 | 0.912 [0.86, 0.95] | 0.875 | 0.869 | 0.575 |
| Stepping base, arms down (stage 2) + hold | **0.898** [0.83, 0.94] | 0.938 | 0.648 [0.56, 0.73] | 0.547 | 0.994 [0.96, 1.00] | 0.988 | 0.994 | 0.963 |
| Stepping base + guard (stage 2) + hold | **0.945** [0.89, 0.97] | 0.922 | 0.367 [0.29, 0.45] | 0.375 | 0.994 [0.96, 1.00] | 1.000 | 0.988 | 0.963 |
| **Footwork, arms down (shipped)** | **0.906** [0.84, 0.95] | 0.969 | 0.727 [0.64, 0.80] | 0.797 | 1.000 [0.98, 1.00] | 1.000 | 0.975 | 0.950 |
| **Footwork + guard (shipped)** | **0.992** [0.96, 1.00] | 0.969 | 0.812 [0.74, 0.87] | 0.766 | 1.000 [0.98, 1.00] | 1.000 | 1.000 | 1.000 |

The stage-2 and footwork policies run with their own outer-loop settings (`gym.json` next to each ONNX; below): each holds its spot, and the footwork guard also turns to face the opponent. So the stage-2 rows differ from PR #13's, which had neither. Stage 1 has no outer loops.

Hits without the opponent (`scripts/evaluate.py`): 1120 single hits per controller at 2–30 N·s plus 448 at 35–60 N·s, and 96 three-hit sequences per impulse. With no opponent the outer loops do nothing.

| | single-hit fall rate, 2–30 N·s [95% CI] | fall rate, 35–60 N·s | ≥50% survival up to | 3 hits @ 12 N·s, randomised physics |
|---|---|---|---|---|
| PD stand | 0.606 [0.58, 0.63] | 1.000 | 6 N·s | 0.04 |
| RL v1 (residual) | 0.495 [0.47, 0.52] | 0.996 | 8 N·s | 0.05 |
| RL v2 (residual) | 0.499 [0.47, 0.53] | 0.987 | 8 N·s | 0.05 |
| RL v3 cand. (residual) | 0.494 [0.47, 0.52] | 0.991 | 8 N·s | 0.04 |
| Stepping base, stage 1 | 0.212 [0.19, 0.24] | 0.842 | 20 N·s | 0.56 |
| Stepping base, arms down (stage 2) + hold | 0.057 [0.04, 0.07] | 0.750 | 35 N·s | 0.95 |
| Stepping base + guard (stage 2) + hold | 0.044 [0.03, 0.06] | 0.768 | 30 N·s | 0.96 |
| **Footwork, arms down (shipped)** | 0.030 [0.02, 0.04] | 0.701 | 35 N·s | 0.98 |
| **Footwork + guard (shipped)** | 0.021 [0.01, 0.03] | 0.674 | 35 N·s | 0.93 |

**Read it plainly:**
- **Only stepping survives the hard tier.** PD and the residual policies (v1–v3) fall in every one of the 128 bouts. They sit on a stiff stand that can't step (`docs/MODEL_AUDIT.md`).
- **Footwork vs the policies shipped in PR #13.** The episodes are identical to PR #13's (PD and the residual policies reproduce exactly), so the comparison is paired (`scripts/paired_vs_ref.py` → `vs_3fcbb3b.json`). Hard test, then hard validation (where the settings were chosen, so test is the clean estimate):

  | | survival | up and within 1 m |
  |---|---|---|
  | guard: PR #13 → footwork | 0.922 → 0.992, +0.070 [0.02, 0.12] | 0.289 → 0.812, +0.523 [0.44, 0.61] |
  | arms down: PR #13 → footwork | 0.891 → 0.906, +0.016 [-0.05, 0.08] | 0.305 → 0.727, +0.422 [0.33, 0.51] |
  | guard, validation | +0.078 [0.02, 0.14] | +0.516 [0.39, 0.64] |
  | arms down, validation | +0.078 [0.00, 0.16] | +0.484 [0.33, 0.62] |

  - **What did it:** the hold-your-spot loop and the fine-tune together. How the credit splits depends on the data split and on which is added first.
    - *Arms down:* on hard test the loop alone takes PR #13's base from 0.305 to 0.648 up and within 1 m (+0.344 [0.26, 0.43]), and the footwork policy adds +0.078 [-0.02, 0.16] (survival +0.008 [-0.06, 0.08]): mostly the loop. That second step is not the fine-tune alone: it also swaps the base's tight loop (12 / 5 cm) for the standard one (25 / 8 cm), each chosen on validation, and on validation the swap alone moves the footwork policy 0.719 → 0.797. On hard validation with the tight loop on both, the split is the same in either order: the loop 0.312 → 0.547 (or 0.484 → 0.719 after the fine-tune) and the fine-tune 0.312 → 0.484 (or 0.547 → 0.719); the standard loop then takes the footwork policy to 0.797 (shipped vs shipped, +0.250 [0.12, 0.38]).
    - *Guard:* the loop alone takes PR #13's guard only from 0.289 to 0.367 (+0.078 [0.00, 0.16]); it still drifts 0.9 m. The footwork guard, which also turns to face, adds +0.445 [0.35, 0.54] (survival +0.047 [0.02, 0.09]). Test can't separate the fine-tune from the turn. On validation the fine-tune needs the loop (without it, 0.266 vs 0.250); with it, the fine-tune takes the guard from 0.922 / 0.375 to 0.938 / 0.719 and turning to 0.969 / 0.766 (survival / up and within 1 m). So the ring gain is mostly the fine-tune's. Of the 3 extra surviving bouts, 1 comes with the fine-tune and 2 with the turn, but those 2 are the margin the turn was chosen on, so they don't show that turning adds survival.
  - **Hits:** the single-hit fall rate halves for both (0.057 → 0.030 arms down, 0.044 → 0.021 guard; the CIs don't overlap), and the guard now holds to 35 N·s (was 30). Three hits at 12 N·s under randomised physics: arms down 0.95 → 0.98, guard 0.96 → 0.93.
  - **Worse:** the arms-down footwork policy falls more against the puppet (test 1.000 → 0.975, -0.025 [-0.05, -0.01]; validation 1.000 → 0.950) and ends fewer of them up and within 1 m (test 0.994 → 0.956, -0.037 [-0.07, -0.01]; validation 0.825 → 0.812, -0.013 [-0.10, 0.07]). That mixes the loop and the fine-tune: the stage-2 base with its loop already falls in 1 of 160 test and 3 of 80 validation puppet bouts, and footwork minus that base is -0.019 [-0.05, 0.01] (test) and -0.013 [-0.06, 0.04] (validation), so neither step is significant on its own. The loop settings were chosen on the tracked opponent only. The footwork guard doesn't fall more (1.000 → 1.000 on test, 0.950 → 1.000 on validation).
- **Arms down vs guard.** Footwork guard minus footwork arms-down, paired:
  - test: hard +0.086 [0.04, 0.13], up and within 1 m +0.086 [-0.01, 0.18]; standard +0.000 [0.00, 0.00]
  - validation: hard +0.000 [-0.05, 0.05], up and within 1 m -0.031 [-0.16, 0.09]; standard +0.000 [0.00, 0.00]

  They tie on validation, where both were chosen, and the guard survives more of the hard test bouts. Both ship: the guard looks like sparring. (With Unitree's capsules carrying the contact, PR #12, the guard looked worse; PR #13 traced that to the contact model: `clip_check.json`, `crosscheck_val_hard`.)
- **Pushed around, less.** In the hard tier surviving bouts end 0.31 m (guard) and 0.26 m (arms down) from where they started, and 18.1% and 19.8% of them leave a 1 m circle at some point. PR #13's policies ended 1.3–1.4 m away, and two thirds of their surviving bouts left the circle.
  - Without the scripted shoves (`summary_test_hard_nohits.json`) they end 0.24 and 0.19 m away, and 95.3% and 91.4% of bouts end up and within 1 m (PR #13: 54.7% and 53.9%; the stage-2 bases with the loop: 65.6% and 91.4%, so for arms down this gain is all the loop).
  - In the standard tier they drift 0.16 and 0.11 m.
- **The opponent falls too.** It falls in 0%–12.5% of tracked bouts against the stepping policies (`opponent_fell`, per controller, counted while the R1 is up). On its own it falls in 2 of 48 10 s runs.
- **What changed since the last version of this table (PR #13):**
  - Two new shipped policies, the footwork fine-tunes (stage 3, below).
  - The hold-your-spot loop for the stage-2 and footwork policies, and the turn command for the footwork guard, set per policy on validation (below). The stage-2 rows now include their hold loop, paired against PR #13 (`vs_3fcbb3b.json`): hard test survival 0.891 → 0.898 (arms down, +0.008 [-0.06, 0.07]) and 0.922 → 0.945 (guard, +0.023 [-0.02, 0.08]); up and within 1 m 0.305 → 0.648 and 0.289 → 0.367. Elsewhere they lose a few bouts, none significantly: standard test 1.000 → 0.994 (arms down); puppet test 1.000 → 0.994 (arms down) and 1.000 → 0.988 (guard); puppet validation 1.000 → 0.963 (arms down; the guard gains, 0.950 → 0.963). Against the puppet, which isn't steered, the loop doesn't keep them in the ring: up and within 1 m goes 0.994 → 0.988 (arms down) and 0.975 → 0.950 (guard, -0.025 [-0.06, 0.01]) on test, and 0.825 → 0.725 (-0.100 [-0.20, 0.00]) and 0.825 → 0.775 (-0.050 [-0.14, 0.04]) on validation, none significantly.
  - The episodes, opponent and contact are unchanged.

## Holding the spot, and turning to face
Two outer loops drive a stepping policy through its velocity command (`r1gym/env.py`), standing in for what the R1 will do with perception. Both use true positions, as the opponent's steering does.
- **Hold your spot** (`_hold`): once the pelvis is more than 25 cm from where the bout started, command (vx, vy) back toward it (gain 1.5, in the heading frame, at most 0.8 m/s forward, 0.5 back and 0.5 sideways) until it is within 8 cm. "Tight" is 12 cm / 5 cm.
- **Turn to face** (`_face`): command the yaw rate toward the opponent's bearing (gain 2, at most 0.8 rad/s). The deadband version turns only past 20° off and stops within 6°.

Each policy's settings were chosen on hard-tier validation (64 bouts): **the most bouts up and within 1 m among the settings within one bout of the best survival.** Cells are survival / up and within 1 m; bold is the setting shipped (`gym.json`, with the numbers in its `_why_*` fields).

| policy | neither | hold | hold, tight | hold + turn | hold + turn, deadband |
|---|---|---|---|---|---|
| stage 2, arms down | 0.891 / 0.312 | 0.859 / 0.500 | **0.938 / 0.547** | – | – |
| stage 2, guard | 0.891 / 0.250 | **0.922 / 0.375** | 0.875 / 0.281 | – | – |
| **footwork, arms down (shipped)** | 0.938 / 0.484 | **0.969 / 0.797** | 0.922 / 0.719 | 0.938 / 0.703 | 0.938 / 0.766 |
| **footwork + guard (shipped)** | 0.953 / 0.266 | 0.938 / 0.719 | 0.922 / 0.734 | **0.969 / 0.766** | 0.938 / 0.781 |

`scripts/face_ab.py --hold` → `results/hold_ab.json` (turn command off), and `scripts/face_ab.py` → `results/face_opponent_ab.json` (hold on for the footwork policies). The stage-2 bases' turn command was measured in PR #13 with shell contact (the stage-2 rows of `face_opponent_ab.json`), without the hold loop, and left off. The same files have 48 bouts of 12 s on train + val clips with the demo's pushes:
- **Hold** cuts the footwork policies' drift in those bouts from 0.46 to 0.17 m (guard) and 0.21 to 0.13 m (arms down), and walking back costs the guard some facing (median 22° → 29° off).
- **Turning** brings the footwork guard back to 18°. The footwork arms-down policy already faces the opponent within a median 11° without it (the steered opponent keeps itself in front), and with it survives less on validation.
- **Turning is weak, and the fine-tune made the right turn weaker.** Measured rad/s for a yaw-rate command held from standing, here (`face_opponent_ab.json`) and in mjlab, where they trained (`gpu/turn_check_mjlab.py --reps 8` → `results/turn_check_mjlab.json`; 8 envs per command there, each with its own draw of mjlab's training randomisation of foot friction, torso COM and encoder bias, and no sensor noise; 3 seeds here at nominal physics, with sensor noise and a 20 ms obs delay):

  | | +0.8 here | −0.8 here | +0.8 mjlab | −0.8 mjlab |
  |---|---|---|---|---|
  | stage 2, arms down | -0.04 | -0.32 | -0.04 | -0.26 |
  | **footwork, arms down** | 0.09 | -0.12 | 0.08 | -0.15 |
  | stage 2, guard | 0.21 | -0.29 | 0.20 | -0.30 |
  | **footwork + guard** | 0.18 | -0.08 | 0.19 | -0.11 |

  The stage-2 arms-down base drifted right for any turn command (it turned the wrong way for a left turn); the footwork one lost that drift but turns slowly both ways. The stage-2 guard turned both ways; the footwork guard turns left about as fast in mjlab (0.19 vs 0.20) but right at about a third of its speed (-0.11 vs -0.30), and here it also yaws at +0.09 rad/s with no command (a 3-seed mean; mjlab's 8 envs read 0.00, so treat it as rough). In mjlab no policy gets past 0.30 rad/s for a 0.8 command, and for these four the two simulators' means agree within 0.06 rad/s at ±0.8 (randomised physics there, nominal here; one command's spread across mjlab's 8 envs is up to 0.08), so this is the policies, not the simulator. An earlier single-env mjlab check read up to 0.11 rad/s differently between runs, each run drawing new physics; it is what the older claims of a faster left turn in mjlab and a 'simulator gap' for the guard came from, and `turn_check_mjlab.json` keeps it.
- **A turning pass didn't fix it (stage 4, not shipped).** Fine-tunes with turn-in-place commands for 25% of the moving envs, tighter yaw tracking and looser hip and waist yaw posture (`gpu/README.md`) turn at 0.19 / -0.03 rad/s for +0.8 / −0.8 here (arms down, now drifting left; mjlab +0.22 / +0.01) and 0.16 / -0.21 (guard; mjlab +0.14 / -0.13). The guard's right turn got faster and its left turn slower, here and in mjlab. The rule written before training (`results/turn_prereg.json`) needed both directions at 0.25 rad/s or more; the weaker direction is 0.03 and 0.16 (`scripts/turn_select.py` → `results/turn_selection.json`), so the footwork policies stay. The validation bouts weren't run, and neither was the test the rule said would be reported for both candidates: no turn setting was chosen to test with.

## What the shipped policies are, and aren't
- **Training:** Unitree's official R1 velocity task in mjlab (MuJoCo Warp), in three stages on one RTX 4090 (`gpu/README.md`):
  1. Unitree's task with more standing (3000 iterations).
  2. Pushes every 1.5–3.5 s up to 0.8 m/s and 0–40 ms obs delay (1500 iterations; about 440M env steps in all). The guard is this base fine-tuned for 1500 iterations with a **guard pose** as its default arm pose.
  3. **Footwork** (1500 iterations each, about 36 min): Unitree's command curriculum removed and the command ranges set explicitly (vx −0.6…0.8 m/s, vy ±0.5 m/s, yaw rate ±1.0 rad/s, 30% standing; as in Unitree's task, the yaw-rate command comes from a random heading target); velocity and yaw tracking weighted 1.5×; torso shoves (each force component up to ±100 N for 0.05–0.15 s, every 1–3 s) on top of stage 2's pushes.
- **Contract:** Unitree's own (kp 100/40, per-joint action scale 0.15–0.41, 83-d obs, zero joint friction as mjlab sets it; `configs/r1_mjlab.yaml`, `docs/DEPLOY.md`). They are evaluated here in CPU MuJoCo, a different simulator. `scripts/crosssim_mjlab.py` checks that they stand and track commands here.
- **Footwork, measured** (`crosssim.json` next to each ONNX: commanded → measured, mean over the last 6 of 8 s, 4 seeds, no falls):

  | | 0.5 m/s forward | 0.3 m/s back | 0.3 m/s sideways | 0.5 rad/s turn |
  |---|---|---|---|---|
  | stage 2, arms down | 0.32 | 0.14 | 0.19 | -0.11 |
  | **footwork, arms down** | 0.35 | 0.12 | 0.08 | 0.06 |
  | stage 2, guard | 0.21 | 0.14 | 0.21 | 0.12 |
  | **footwork + guard** | 0.34 | 0.23 | 0.27 | 0.13 |

  The footwork guard tracks forward, back and sideways better here, but it yaws at 0.13 rad/s while walking forward (stage 2: -0.03), and in mjlab it tracks 0.3 m/s sideways at 0.26 (stage 2: 0.22; `turn_check_mjlab.json`). The arms-down one is about the same forward and back, worse sideways, and now turns the right way for a left turn, slowly. The turn column is a left turn only; right turns got worse (above). Holding a spot takes short steps, which both manage; quick footwork and turning need another pass.
- **They don't see, block, counter or clinch.** They stay standing and hold their ground. Phase 3 (perception, strikes, blocks) builds on them.
- **No hardware.** They use Unitree's deploy gains, not the old kp 300 stand. The R1 EDU still needs system ID (phase 4). Unitree's own R1 `deploy.yaml` lists a flat 0.25 action scale where its exported policies use 0.15–0.41 per joint (`docs/DEPLOY.md`).

## The opponent
`r1gym/opponent.py`, `gpu/r1_track_task.py`, `results/opponent_tracker/`.
- **Motion:** KungFuAthleteBot G1 clips, retargeted to the R1 by keypoint IK anchored to the G1's own joint angles. The retarget is cleaned:
  - unit quaternions;
  - short mocap glitches interpolated;
  - aerial flips cut;
  - heading-safe joins.

  Moves the R1 can't perform are cut. So are idle stretches and posing with both arms overhead (`scripts/style_cuts.py`), so what is left is mostly strikes. The clip is placed in place, facing the R1, with seamless loops, and grounded so the supporting foot is on the floor.
- **Performer:** a BeyondMimic-style tracking policy (unitree_rl_mjlab's tracking task on the R1) on Unitree's motors with no assist.
  - It also reads where its reference is, so the sim steers it: the reference follows a point in front of our R1 at ≤ 0.3 m/s. The opponent stays in front and steps in when the R1 is pushed back, instead of circling to its side.
  - In 48 bouts on train + val clips (`selection.json`), it holds a median 4° off straight in front (worst bout's 90th percentile 28°) and a median 6 cm from its distance. Before steering (stage 3) that was 11° (worst 85°) and 7 cm. In the old demo, where the R1 was also shoved, the opponent ended up at its side within 2 s.
  - It stays up in 46/48 10 s runs on its own (`scripts/eval_tracker.py`).
- **Contact between the robots** (`robot_contact: shell`, `r1gym/model.py`; measured by `scripts/clip_check.py` → `results/clip_check.json`).
  - Unitree's R1 model collides through capsules and spheres that are thinner than its visible shell (pelvis mesh ~16.5 cm across vs a 10 cm sphere; elbow ~9 vs 6 cm), and 9 of its 25 links have none. So the two robots' shells passed through each other before anything touched.
  - Now each link's visual mesh (its convex hull) collides with the other robot's, with stiffer contact, and the capsules no longer collide across robots. Ground contact and the opponent's self-collision keep the capsules the policies trained with.
  - Robot-to-robot contact is frictionless (condim 1, as mjlab uses for every collision shape but the feet). With capsule contact only the feet's contacts had friction.

  | policy | contact | deepest overlap per bout (median / 95th pct / worst) | steps with > 1 cm overlap |
  |---|---|---|---|
  | footwork, arms down | capsule | 6.1 / 7.8 / 8.2 cm | 9.9% |
  | footwork, arms down | shell | 0.7 / 1.4 / 2.0 cm | 0.1% |
  | footwork + guard | capsule | 5.4 / 7.8 / 8.3 cm | 25.1% |
  | footwork + guard | shell | 0.9 / 2.2 / 2.6 cm | 0.1% |

  48 bouts of 12 s per row on train + val clips, with the demo's pushes and each policy's outer loops. The remaining overlap is the contact giving way for a few steps in hard impacts.
- **Parity:** run alone from the same 48 starts, it stays up 45/48 in mjlab, where it trained, and 46/48 here; the falls they share happen within 0.1 s of each other (`parity_mjlab.json`). The observation matches mjlab's training loop: the reference one frame ahead, fresh kinematics, and the anchor position in the torso frame.

## Head camera (sim, phase 3)
![camera turn](media/camera_turn.gif)

The R1 turning toward what its own head camera sees (M1, no learning).
- **The chain:** a render twin of the sim draws the head camera's frame (160×120 at 30 Hz, pitched -50°). A colour-key detector finds the orange opponent. The estimate arrives after 20–60 ms of latency, with some frames dropped, and is carried to "now" by the gyro. It then feeds the same turn-to-face command as before.
- **Test:** on the test clips (guard policy, 192 paired bouts per setting, starts ±30° / ±50° off-axis), all five preregistered criteria passed (`results/perception/m1_offaxis_test.json`):
  - The camera faces as well as exact bearings on the camera's own clock: -1.0° [-2.3, 0.5].
  - It beats no turning by 37.2° [34.1, 40.6].
  - It is 0.20 s [0.09, 0.32] slower to face.
  - A shuffled-estimates control does no better than no turning.
  - With an empty ring it never commands a turn.
- **What it doesn't show:** the colour key only works because the opponent is the only orange thing in the scene. So this checks the camera plumbing (mount, clocks, latency, compensation), not perception of a real opponent. A learned estimator comes next.

Setup, controls, the integrity checks and every limitation are in [docs/HEAD_CAMERA.md](docs/HEAD_CAMERA.md).
Reproduce: `python scripts/demo.py --headcam --offaxis 40`.

## Aimed opponent (M1: the shipped tracker, aimed)
The opponent above performs fight forms in front of the R1. M1 makes those strikes go for the R1's head without retraining
the tracker: `r1gym/aim.py` changes the reference the tracking policy is shown, per strike, and the hit metric scores what
lands. The definitions, the primary metric (head clean strikes per minute), the splits and the rule that picks the operator
were fixed before anything aimed ran (`docs/AIMED_OPPONENT.md`, its amendment 1 included). Everything below is generated from
the stored JSON by `python scripts/aim_tables.py --write`; the aim reads the R1's true pose (privileged, like the steering).
- **Operators:** StandOff steers in so a strike's apex reaches 3 cm into its target; ArmWarp re-solves the striking arm every
  control step so the palm meets the target from where the opponent's torso is now (or only in height and sideways); the
  strike selector splices the clip to its head-level strikes. A strike's target follows the R1 until 0.15 s before its apex,
  then stays put; no strike is aimed while the opponent leans more than 15 degrees or right after it touched the R1.
- **Off by default:** `opponent_aim: none` is the shipped opponent bit for bit (`tests/test_aim.py`).

<!-- aim:begin -->
**Strike library** (`r1gym/strikes.py`, `scripts/strike_library.py`): 55 strikes (51 with a hand) in the motion the tracker performs; 5 train, 1 val and 1 test strikes are level with the head. Against 106 hand-labelled candidates (9 unsure; `strike_labels.json` says how they were labelled): precision 0.78 [0.58, 0.90], recall 0.79 [0.59, 0.91]. Sheet: [`media/strikes_sheet.png`](media/strikes_sheet.png).

**What the shipped tracker tolerates** (`scripts/tracker_tolerance.py`, opponent alone, 43 train + val hand strikes): its palm comes within 5 cm of the reference apex in 70% of strikes unwarped, 67% / 54% when the apex is moved 10 / 20 cm up, 47% for 20 cm in any of three directions; no warp up to 25 cm made it fall. A 20 cm lunge at 1 m/s adds 3.5 cm of reach (median). Slowing strikes to 0.5x: 16/48 falls (unchanged: 2/48). Splicing to head-level strikes: 15/48 falls, so the selector fails its preregistered tolerance and is out. Its lag: 20 ms (median, train).

**Paired aim check** (`scripts/aim_check.py`, train + val, 72 bouts of 10 s per operator and target at 0.40-0.65 m; against PR #13's arms-down base; the unaimed opponent lands 5.3 head clean strikes/min):

| operator | Δ head clean strikes/min [95% CI] | Δ head-landed share per strike window | opponent falls, pooled (unaimed 0.5%) | sign under the other hit rules |
|---|---|---|---|---|
| StandOff | +1.0 [0.0, 2.1] | +0.013 [0.002, 0.025] | 0.0% | landed +1.4 A_peak20 +3.4 B_strike +0.9 impulse_only +1.4 |
| ArmWarp | +2.1 [0.3, 3.8] | +0.006 [-0.010, 0.019] | 0.5% | landed +1.3 A_peak20 +0.7 B_strike +2.0 impulse_only +1.3 |
| ArmWarp, height and sideways only | +2.6 [0.7, 4.5] | +0.033 [0.011, 0.056] | 0.0% | landed +3.5 A_peak20 +4.8 B_strike +2.5 impulse_only +3.5 |
| StandOff + ArmWarp (height, sideways) **(chosen)** | +3.2 [1.1, 5.2] | +0.030 [0.014, 0.049] | 0.9% | landed +6.0 A_peak20 +7.9 B_strike +3.1 impulse_only +6.0 |
| StrikeSelector | not run: failed its splice tolerance | | | |

Chosen by the preregistered rule (`results/opponent_tracker/aim.json`, before any aimed test run): **StandOff + ArmWarp (height, sideways)**, the largest paired difference whose CI is above 0. The next best, ArmWarp, height and sideways only, is 0.6/min behind (the tie margin is 0.5). Opponent falls: 0.9% of bouts against 0.5% unaimed (the rule allows +2 pp). Its sign is the same under every hit rule. StandOff alone misses the bar: its CI starts at 0.0.

The chosen operator on every target (same episodes; peak forces at the shell solref named):

| target | head clean strikes/min, unaimed → aimed | Δ [95% CI] | Δ on train clips (9 start times) / on the val clip (3) | Δ head-landed/min | Δ head impulse, N·s/min | strike windows ending on the head, unaimed → aimed | windows not aimed (lean / touch gate) | windows whose target is >= 8 cm beyond the arm's reach | head-hit peak force, median / p95 |
|---|---|---|---|---|---|---|---|---|---|
| mannequin (welded R1: the aim alone) | 6.5 → 8.2 | +1.8 [-1.0, 4.4] | +3.7 [1.2, 6.0] / -4.0 [-9.0, -1.0] | +4.2 [1.9, 6.5] | +25.8 [15.3, 38.4] | 5.1% → 8.9% | 30% / 23% | 72% | 220 / 1027 N at 0.01 |
| arms-down base (PR #13) | 5.3 → 8.5 | +3.2 [1.1, 5.2] | +3.4 [0.9, 5.6] / +2.3 [-1.0, 7.0] | +6.0 [3.3, 8.8] | +23.7 [11.6, 35.3] | 3.3% → 6.3% | 27% / 21% | 75% | 215 / 808 N at 0.01 |
| guard (PR #13) | 2.4 → 3.3 | +0.9 [-0.2, 2.0] | +0.3 [-0.8, 1.6] / +2.7 [2.0, 4.0] | +3.2 [1.2, 5.4] | +10.5 [3.2, 18.6] | 1.7% → 3.0% | 30% / 10% | 85% | 205 / 903 N at 0.01 |
| arms-down (PR #13), shell solref 0.02 | 4.9 → 9.7 | +4.8 [2.7, 7.3] | +3.8 [2.1, 5.4] / +7.7 [0.0, 15.0] | +6.2 [3.4, 9.0] | +24.6 [13.3, 36.2] | 3.7% → 5.9% | 28% / 22% | 76% | 113 / 419 N at 0.02 |

![unaimed vs aimed](media/aim_frames.png)

*Left unaimed, right aimed (standoff+armwarp_vz): the same kfab_204 bout at 0.6 m, at the apex of each aimed strike that ended on the head (`scripts/aim_frames.py`; the bout where the aim added the most head windows, so the best case, not a typical one).*

Reach is the limit, as the design predicted: on the arms-down base, strikes whose target the arm can reach from where the opponent stands (IK residual < 3 cm, 125 windows) end on the head 18% of the time, those 3-8 cm short 18%, and the 796 that are 8 cm or more short 3%. The actual pelvis spacing stays near 0.56 m (median) at nominal 0.40-0.65 m. Whole-body aiming (M2, GPU) is what can change that.

**Hard tier with the aimed opponent** (`--aim chosen`: standoff+armwarp_vz, chosen before the footwork policies existed and not re-chosen; `results/eval_opponent/aim/`: the same episodes as the unaimed tier, the scheduler in shadow there, paired; nominal distance 0.40-0.55 m):

| split | controller | survival, unaimed → aimed | Δ survival [95% CI] | head clean strikes/min, unaimed → aimed | Δ [95% CI] | Δ opponent falls [95% CI] | actual spacing (median, unaimed) |
|---|---|---|---|---|---|---|---|
| val | Stepping base, stage 1 | 0.406 → 0.469 | +0.062 [-0.016, 0.149] | 10.4 → 9.0 | -1.4 [-3.4, 0.6] | -0.016 [-0.065, 0.034] | 0.547 m |
| val | Stepping base, arms down (PR #13) | 0.938 → 0.922 | -0.016 [-0.081, 0.052] | 7.3 → 8.8 | +1.4 [-0.1, 2.9] | +0.062 [-0.017, 0.149] | 0.536 m |
| val | Stepping base + guard (PR #13) | 0.922 → 0.859 | -0.062 [-0.138, 0.000] | 3.2 → 5.5 | +2.3 [1.0, 3.7] | +0.109 [0.000, 0.215] | 0.588 m |
| val | **Footwork, arms down (shipped)** | 0.969 → 0.891 | -0.078 [-0.161, -0.015] | 7.5 → 8.7 | +1.2 [-0.4, 2.8] | +0.016 [-0.101, 0.117] | 0.536 m |
| val | **Footwork + guard (shipped)** | 0.969 → 0.953 | -0.016 [-0.065, 0.034] | 4.6 → 5.9 | +1.4 [0.2, 2.5] | +0.094 [0.000, 0.197] | 0.54 m |
| val, no scripted hits | Stepping base, stage 1 | 0.453 → 0.422 | -0.031 [-0.106, 0.045] | 14.5 → 9.3 | -5.2 [-7.8, -2.6] | +0.000 [0.000, 0.000] | 0.537 m |
| val, no scripted hits | Stepping base, arms down (PR #13) | 0.969 → 0.969 | +0.000 [0.000, 0.000] | 9.8 → 11.6 | +1.8 [0.0, 3.5] | +0.047 [0.000, 0.104] | 0.51 m |
| val, no scripted hits | Stepping base + guard (PR #13) | 0.969 → 0.969 | +0.000 [-0.048, 0.045] | 5.0 → 7.6 | +2.6 [1.1, 4.2] | +0.016 [-0.051, 0.085] | 0.546 m |
| val, no scripted hits | **Footwork, arms down (shipped)** | 0.938 → 0.922 | -0.016 [-0.067, 0.034] | 9.8 → 11.4 | +1.5 [-0.3, 3.4] | +0.031 [0.000, 0.078] | 0.516 m |
| val, no scripted hits | **Footwork + guard (shipped)** | 0.953 → 0.969 | +0.016 [0.000, 0.051] | 5.0 → 7.7 | +2.7 [1.5, 3.7] | +0.047 [0.000, 0.102] | 0.525 m |
| test | Stepping base, stage 1 | 0.422 → 0.406 | -0.016 [-0.072, 0.047] | 3.7 → 8.9 | +5.2 [3.9, 6.7] | +0.000 [0.000, 0.000] | 0.568 m |
| test | Stepping base, arms down (PR #13) | 0.898 → 0.875 | -0.023 [-0.084, 0.033] | 5.4 → 7.9 | +2.5 [1.4, 3.5] | +0.141 [0.083, 0.205] | 0.56 m |
| test | Stepping base + guard (PR #13) | 0.945 → 0.883 | -0.062 [-0.116, -0.015] | 1.1 → 2.3 | +1.1 [0.5, 1.8] | +0.133 [0.077, 0.194] | 0.618 m |
| test | **Footwork, arms down (shipped)** | 0.906 → 0.922 | +0.016 [-0.038, 0.071] | 5.4 → 8.7 | +3.3 [2.2, 4.4] | +0.133 [0.061, 0.211] | 0.55 m |
| test | **Footwork + guard (shipped)** | 0.992 → 0.969 | -0.023 [-0.060, 0.008] | 2.1 → 4.2 | +2.2 [1.2, 3.2] | +0.133 [0.074, 0.197] | 0.582 m |
| test, no scripted hits | Stepping base, stage 1 | 0.500 → 0.414 | -0.086 [-0.156, -0.024] | 4.0 → 8.5 | +4.5 [2.9, 5.9] | +0.023 [0.000, 0.054] | 0.57 m |
| test, no scripted hits | Stepping base, arms down (PR #13) | 0.992 → 0.992 | +0.000 [-0.024, 0.023] | 5.7 → 10.4 | +4.7 [3.6, 5.9] | +0.094 [0.047, 0.146] | 0.54 m |
| test, no scripted hits | Stepping base + guard (PR #13) | 0.992 → 0.992 | +0.000 [-0.023, 0.024] | 0.8 → 2.5 | +1.7 [1.1, 2.3] | +0.109 [0.060, 0.161] | 0.564 m |
| test, no scripted hits | **Footwork, arms down (shipped)** | 0.984 → 0.977 | -0.008 [-0.024, 0.000] | 5.5 → 10.3 | +4.9 [3.7, 6.1] | +0.070 [0.031, 0.116] | 0.536 m |
| test, no scripted hits | **Footwork + guard (shipped)** | 1.000 → 0.977 | -0.023 [-0.063, 0.000] | 2.2 → 4.8 | +2.6 [1.7, 3.6] | +0.086 [0.029, 0.150] | 0.554 m |

On test (run once, after val): head clean strikes/min +3.3 [2.2, 4.4] against the footwork arms-down policy and +2.2 [1.2, 3.2] against the footwork guard; survival +0.016 [-0.038, 0.071] and -0.023 [-0.060, 0.008]. PD, the residual policies and PR #13's bases are in the JSON and in `docs/AIMED_OPPONENT.md`, as is the first run against PR #13's bases (`results/eval_opponent/aim_pre14/`). **The aimed opponent falls more in the hard tier:** in 12.5% → 25.8% (footwork arms down), 11.7% → 25.0% (footwork guard) of test bouts (6.2% → 13.3%, 7.8% → 16.4% without the scripted hits). The preregistered fall rule is judged on the aim-check grid (10 s, nominal physics, 0.40-0.65 m), where it stayed within +2 pp; in the hard tier (20 s, randomised R1, 0.40-0.55 m) it doesn't, with or without the scripted hits, so M1's operator is not fit to be the default opponent.

<!-- aim:end -->

## Plan
1. **Sim.** R1 model on the official Unitree MJCF, a fight-motion opponent, procedural hits, sensor delay and noise. Done.
2. **Hit-robust balance against a physical opponent.** The residual PPO on a PD stand (v1–v3) can't step. It was replaced by the **stepping base**, arms down and with a guard, fine-tuned for footwork, with an outer loop that walks it back to its spot (`gpu/`, `results/policy_mjlab_footwork/`, `results/policy_mjlab_guard_footwork/`). The opponent is a tracked R1 on its own motors, steered to stay in front (`results/opponent_tracker/`). Done in sim.
3. **Perception + block / counter / clinch** from the head camera. Roadmap: [ROADMAP.md](ROADMAP.md). Head camera in sim (spec, mount, render twin) and M1, turning on a colour-key detector, are done ([docs/HEAD_CAMERA.md](docs/HEAD_CAMERA.md)).
4. **Safety + sim-to-real** on the R1 EDU (system ID, e-stop, padded mode). Roadmap: [ROADMAP.md](ROADMAP.md).

## What's in the repo
| path | what |
|---|---|
| `r1gym/model.py` | R1 from `unitree_rl_mjlab` r1.xml (24 actuated DoF, 28.93 kg), PD servos, torque caps; the opponent R1 with its own capped motors |
| `r1gym/strikes.py`, `r1gym/aim.py`, `r1gym/hits.py` | the strike library (`assets/motions/strikes.json`, audited against `strike_labels.json`); the aimed opponent's scheduler and operators (M1); the landed-hit metric |
| `results/opponent_aim/`, `results/opponent_tracker/{tolerance,aim}.json`, `results/eval_opponent/aim/`, `results/eval_opponent/aim_pre14/` | the paired aim check on train + val, the tracker's tolerance, the M1 operator with its `_why`; the aimed hard tier against the shipped footwork policies, and the archived as-run one against PR #13's bases |
| `r1gym/opponent.py` | the opponent: clip retarget, cleaning, cuts and grounding (`FightReplay`); the tracking-policy performer with facing and steering (`Tracker`); puppet and kinematic modes |
| `results/opponent_tracker/` | the opponent's tracking policy (`policy.onnx` with the motion bundled, `motion_index.json`, `model_13749.pt`, `params/`, `train_log.json`, `eval_tracker.json`, `parity_mjlab.json`, `selection.json`) |
| `results/policy_mjlab_footwork/` | **shipped** footwork policy, arms down (`policy.onnx`, `model_5997.pt`, `params/`, `crosssim.json`, `gym.json`) |
| `results/policy_mjlab_guard_footwork/` | **shipped** footwork policy with a guard (`policy.onnx`, `params/`, `crosssim.json`, `gym.json`; its checkpoint was lost, `gpu/README.md`) |
| `results/policy_mjlab/`, `results/policy_mjlab_guard/` | the stage-2 bases they were fine-tuned from (`model_4498.pt`, `model_5997.pt`, `train_log.json`, ...); `results/policy_mjlab_stage1/` for comparison |
| `results/hold_ab.json`, `results/face_opponent_ab.json`, `results/turn_check_mjlab.json` | the evidence behind each policy's `gym.json` (hold your spot, turn to face) and how well they turn |
| `results/policy_mjlab_turn/`, `results/policy_mjlab_guard_turn/` | stage-4 turning candidates, **not shipped** (`policy.onnx`, `model_*.pt`, `params/`, `crosssim.json`; `results/turn_prereg.json`, `results/turn_selection.json`) |
| `r1gym/scenario.py` | starting a bout (start times without overlap) and the demo scenario shared by `demo.py`, `demo_check.py` and `eval_opponent.py` |
| `configs/r1_mjlab.yaml`, `r1gym/policy.py` (`MjlabPolicy`) | runs unitree_rl_mjlab policies here under their own contract (checked against the ONNX metadata); same opponent |
| `gpu/` | the mjlab tasks (stepping base, guard, footwork, opponent tracker), the motion export, the mjlab turn check, and how to reproduce them on RunPod |
| `r1gym/perturb.py` | strikes / pushes / grabs by impulse, body and direction; peak force capped at 400 N |
| `r1gym/env.py` | env (78-dim obs, 24-dim residual action), 0–40 ms obs delay, noise, optional opponent in training, thread/process batching; the hold-your-spot and turn-to-face outer loops for the stepping policies |
| `r1gym/policy.py` | PD baseline, torch policy, ONNX policy |
| `scripts/` | `eval_opponent.py` (headline, both tiers), `evaluate.py` (hit sweeps), `eval_tracker.py`, `style_cuts.py`, `clip_check.py`, `face_ab.py`, `paired_vs_ref.py`, `turn_select.py`, `select_opponent.py`, `crosssim_mjlab.py`, `verify_r1.py`, `train.py`, `demo_check.py`, `demo.py`, `export_onnx.py`; aimed opponent: `strike_library.py`, `strike_sheet.py`, `tracker_tolerance.py`, `aim_check.py`, `aim_frames.py`, `aim_tables.py` |
| `configs/r1_assumptions.yaml` | every parameter not taken from Unitree (armature, foot friction, PD gains) |
| `results/policy/` | v2 policy (`policy.pt`, `policy.onnx`, `train_log.json`, `latency.json`); `results/policy_v1/` = v1 |
| `results/eval/`, `results/eval_v1/` | hit eval for v2 / v1 (`summary.json`, `survival.png`) |
| `results/policy_v3_candidate/`, `results/eval_v3/` | v3 attempt (DR + recovery reward + multi-hit curriculum): **not adopted**, see below |
| `results/demo_check.json` | demo fall times + 8 variants, with opponent contact counts and peak forces |
| `r1gym/perception/`, `configs/r1_camera.yaml` | head camera (sim): spec, mount and clocks (`camera.py`), render twin (`render_scene.py`), percept, sources and ego compensation (`percept.py`), main-thread renderer (`hub.py`), colour-key detector (`detect_colorkey.py`); results in `results/perception/`, write-up in `docs/HEAD_CAMERA.md` |
| `docs/MODEL_AUDIT.md`, `docs/GPU.md`, `docs/DEPLOY.md` | model audit; GPU-batched (mjlab / MuJoCo Warp) path; deploy contract (`r1gym/deploy.py`, `deploy_contract.json` next to each ONNX) |

## Quickstart
```bash
pip install -e .[dev]
python scripts/verify_r1.py                  # R1 spec checks -> results/r1_verification.json
python scripts/train.py --envs 256 --horizon 32 --workers 10 --minutes 30 --imp0 6 --opp_prob 0.5 \
    --init results/policy_v1/policy.pt --out results/policy_new   # v2 recipe, stage 1
P="v1=results/policy_v1/policy.pt v2=results/policy/policy.pt v3=results/policy_v3_candidate/policy.pt step_s1=results/policy_mjlab_stage1/policy.onnx step=results/policy_mjlab/policy.onnx guard=results/policy_mjlab_guard/policy.onnx step_fw=results/policy_mjlab_footwork/policy.onnx guard_fw=results/policy_mjlab_guard_footwork/policy.onnx"
python scripts/eval_opponent.py --tier hard --split test --policy $P   # headline -> results/eval_opponent/ (also --split val; --tier standard; --opponent puppet; --no_pushes)
python scripts/style_cuts.py                                     # idle / posing cuts for the opponent's clips -> r1gym/assets/motions/style_cuts.json
python scripts/select_opponent.py --opponent final=results/opponent_tracker/policy.onnx --out selection_check.json   # opponent in bouts; the stored selection.json also has stage 3 and iteration 11250
S2="arms_down=results/policy_mjlab/policy.onnx guard=results/policy_mjlab_guard/policy.onnx"; FW="footwork_arms_down=results/policy_mjlab_footwork/policy.onnx footwork_guard=results/policy_mjlab_guard_footwork/policy.onnx"
python scripts/face_ab.py --hold --policy $S2 $FW && python scripts/face_ab.py --hold --val --policy $S2 $FW   # hold-your-spot A/B -> results/hold_ab.json
python scripts/face_ab.py && python scripts/face_ab.py --val      # turn-toward-the-opponent A/B, footwork policies (hold on) -> results/face_opponent_ab.json
python scripts/face_ab.py --no_hold --policy $S2 && python scripts/face_ab.py --no_hold --val --policy $S2   # its stage-2 rows (no hold loop; rows merge per policy)
python scripts/paired_vs_ref.py --ref 3fcbb3b --pair "RL step_fw=RL step" "RL guard_fw=RL guard" "RL step=RL step" "RL guard=RL guard" --out results/eval_opponent/vs_3fcbb3b.json   # vs PR #13, same episodes
python scripts/clip_check.py                                     # how far the robots' shells overlap, capsule vs shell contact -> results/clip_check.json
git show 97a22e2:results/eval_opponent/raw_val_hard.json > old_val_hard.json && python scripts/clip_check.py --crosscheck old_val_hard.json results/eval_opponent/raw_val_hard.json   # its crosscheck_val_hard rows (stage-2 bases, no hold loop)
python scripts/strike_library.py && python scripts/strike_sheet.py   # strike library + audit -> r1gym/assets/motions/strikes.json, media/strikes_sheet.png
python scripts/tracker_tolerance.py && python scripts/aim_check.py   # tolerance, paired aim check (train + val) and the M1 operator -> results/opponent_tracker/aim.json
python scripts/eval_opponent.py --tier hard --split val --policy $P --out results/eval_opponent/aim && python scripts/eval_opponent.py --tier hard --split val --policy $P --out results/eval_opponent/aim --aim chosen   # aimed hard tier, paired with the unaimed (shadow) run; then once on test; also --no_pushes
python scripts/aim_tables.py --write                             # README aim block + docs/AIMED_OPPONENT.md results, from the stored JSON
python scripts/eval_tracker.py                                   # how well the opponent performs its clips on its own
python scripts/crosssim_mjlab.py results/policy_mjlab_footwork/policy.onnx --out results/policy_mjlab_footwork/crosssim.json   # mjlab-trained policy stands and tracks commands in this sim
python scripts/evaluate.py --out results/eval_hits --policy $P   # single hits to 60 N·s, 3-hit survival, CIs
python scripts/demo_check.py --policy v1=results/policy_v1/policy.pt v2=results/policy/policy.pt v3=results/policy_v3_candidate/policy.pt step=results/policy_mjlab/policy.onnx guard=results/policy_mjlab_guard/policy.onnx step_fw=results/policy_mjlab_footwork/policy.onnx guard_fw=results/policy_mjlab_guard_footwork/policy.onnx --out results/demo_check.json   # demo + 8 variants
python scripts/demo.py   # the README clip, both shipped policies side by side (headless Linux: MUJOCO_GL=egl); --left none for one panel
python scripts/export_onnx.py --ckpt results/policy/policy.pt    # -> results/policy/policy.onnx + latency
pytest -q
```

## History
Earlier stages, kept for the record. The numbers below were measured against earlier opponents (a kinematic replay, then the assisted puppet), so they are not comparable with the headline table.

### Results (v2 vs v1 vs PD)
Eval: 896 single-hit episodes per controller (strike/push × 7 locations × 8 impulses × 4 directions × 2 reps), 0–40 ms obs delay, sensor noise.

| | PD stand | RL v1 | **RL v2** |
|---|---|---|---|
| Overall fall rate | 0.52 | 0.413 | **0.411** |
| Fall rate @ 4 / 6 / 8 / 10 N·s | 0.15 / 0.39 / 0.61 / 0.71 | 0.00 / 0.16 / 0.37 / 0.63 | 0.00 / 0.19 / 0.39 / 0.57 |
| Largest impulse with ≥50% survival | 6 N·s | 8 N·s | 8 N·s |
| Fall rate chest / head / pelvis | 0.72 / 0.76 / 0.57 | 0.59 / 0.67 / 0.42 | 0.56 / 0.66 / 0.43 |
| Old demo (kinematic opponent, mostly out of reach) fall time | 9.1 s | 6.1 s | 11.0 s |
| Old demo variants: mean time to fall (8 runs) | 9.16 s | 7.34 s | 9.74 s |

On single hits, v2 is about the same as v1 (0.411 vs 0.413 is within noise). It is slightly better at 10 N·s and slightly worse at 6–8 N·s. In the old demo it looked like the gain came with an opponent in contact range; against the physical opponent that gain is gone (see the top of this README). v1 never saw one in training and usually fell about 1.5 s after the 4.6 s head strike plus the opponent's knee contact. v2 trains with an opponent in 50–70% of episodes.

**Demo history.** The first demo's opponent was a kinematic (infinite-mass) replay placed by its clip's mean position. The clips drift up to 2.4 m, so it started 1.46 m away, touched the R1 for only 2 control steps in 12 s, and teleported when the clip looped. Every fall in that demo came from invisible scripted forces. It also had soft "padded" contact to stop the infinite-mass body producing unbounded forces (4–12 kN once in range). The retargeted clips also had IK flips (shoulder yaw jumping limit to limit, up to 3.7 rad in one frame); the retarget now anchors to the G1's own joint angles and caps joint speed (0.25 rad/frame, keypoint error ~5 cm).

![survival](results/eval/survival.png)

**Puppet opponent force.** The puppet's joints are PD servos with the R1's own torque caps; its pelvis gets a force-limited assist toward the clip (horizontal ≤ 250 N, about the friction limit of a 29 kg robot; vertical ≤ 600 N; torque ≤ 150 N·m) so it can follow kung-fu motion without falling. In the demo its contacts peak at roughly 100–300 N with about 5 N·s per punch; kicks and lunges are larger. Scripted hit peak force is capped at 400 N, with the impulse preserved.

### v3 attempt: harder hits, recovery reward, domain randomisation (not adopted)
Feedback on v1 was that it fell over too easily. v3 tried to fix that, starting from v2:
- **Harder, denser hits:** 3–8 hits per 10 s episode, 0.5–1.8 s apart (run B: 1.0–2.5 s). The curriculum grows the impulse whenever the episode fall rate is under 0.35, up to a cap of 40 N·s.
- **Recovery reward** (`--reward v3`): no penalty for leaving the stance, a weak leg-velocity penalty, and a strong upright/height term. The residual limit was already ±1 rad (clip ±4 × 0.25), so it did not need loosening.
- **Domain randomisation** (`--dr 1`), sampled per episode:
  - link mass ×0.85–1.15, plus −0.5…+2 kg on the torso
  - pelvis/torso COM ±2.5 cm
  - foot friction 0.5–1.25
  - per-joint kp/kd ×0.8–1.2 and torque cap ×0.8–1.05
  - armature ×0.5–1.5
  - 0–20 ms command latency (on top of the 0–40 ms observation delay)
- **Opponent:** present in 50% of episodes.

Two runs on one RTX 4090 pod:
- **Run B:** 80 min, 13M steps.
- **Run A:** 33 min, then it crashed (OOM from per-env model copies, since fixed: DR now swaps parameters into a shared model). It was resumed as A2 for 55 min, 10.5M steps.

**The curriculum stalled at a 7–9 N·s max impulse in both runs.** Under DR with dense hits, the episode fall rate stayed at 0.4–0.6, so the impulse never grew toward sparring levels.

New eval (`scripts/evaluate.py`): 1120 single hits per controller up to 30 N·s, plus 96 three-hit sequences per impulse (random kind, location and direction, 0.5–1 s apart), on the nominal model and under DR.

| | PD | RL v1 | **RL v2 (current)** | RL v3 cand. (run B) | RL v3 run A2 |
|---|---|---|---|---|---|
| Single-hit fall rate (1120, 2–30 N·s) | 0.606 | 0.495 | **0.498** | 0.495 | 0.505 |
| Fall rate @ 6 / 8 / 10 / 15 / 20 / 30 N·s | .40/.63/.71/.76/.82/.96 | .16/.35/.60/.71/.75/.90 | .19/.38/.57/.71/.73/.88 | .17/.42/.57/.70/.74/.88 | .21/.43/.61/.71/.71/.88 |
| 3-hit survival @ 4 / 6 / 8 / 10 N·s (nominal) | .48/.20/.10/.04 | .95/.57/.29/.08 | .91/.49/.21/.04 | .99/.53/.23/.09 | .94/.47/.16/.05 |
| 3-hit survival @ 4 / 6 / 8 / 10 N·s (DR) | .45/.25/.09/.04 | .89/.46/.28/.08 | .79/.40/.12/.05 | .93/.47/.16/.06 | .83/.44/.14/.03 |
| Old demo fall time (kinematic opponent) | 9.1 s | 6.1 s | 11.0 s | 10.8 s | 9.2 s |
| Old demo variants: mean survival (12 s = survived) | 9.16 s | 7.34 s | 9.74 s | 11.05 s | 10.34 s |
| Old demo variants survived the full 12 s | 0/8 | 0/8 | 0/8 | 0/8 | 4/8 |

**Verdict at the time: v3 was not a clear robustness improvement, so v2 stayed the shipped policy.** It has since been replaced by the stepping base.
- **Where the v3 candidate is better:**
  - It is more consistent in the demo variants. All 8 fall at 10.8–11.7 s, versus two early 6.2 s falls for v2.
  - It does a little better on 3-hit sequences, mostly under DR.
- **Where it is not better:**
  - Single-hit fall rates are unchanged, and slightly worse at 8 N·s.
  - Survival above ~10 N·s is still poor for every controller, and v1 is as good as anything on multiple hits.
- **Bottom line:** none of these controllers survives sparring-level hits. Every one falls to 15–30 N·s hits more than 70% of the time.
- **What would likely help next:**
  - a curriculum that does not gate on the fall rate under full DR (e.g. ramp DR in after the impulse grows)
  - a stepping-capable base controller (Unitree's walking policy) instead of a residual on a stiff stand
  - much more compute

![v3 eval](results/eval_v3/survival.png)

### Compute and cost (residual policies)
- **v1:** PPO, 60 min, 65.8M env steps, one RunPod RTX 4090 pod (training is CPU MuJoCo on the pod's ~10-vCPU quota; the GPU is barely used).
- **v2:** warm-started from v1 on one RunPod RTX 4090 pod ($0.89/h), with runs in parallel:
  - Stage 1: 27 min, 4.2M steps, opponent in 50% of episodes.
  - Stage 2: 25 min, 3.9M steps, opponent in 70%, starting impulse 7 N·s.
  - The pod was up about 1.9 h in total, about $1.7, including 3 discarded fine-tunes without the opponent and a few crashed launches (the pod has a 31 GB memory limit).
- **v3 attempt:** one RTX 4090 pod for ~2.3 h (~$2), two runs in parallel (8 workers each).
- **ONNX:** 227 KB, p50 8.4 µs / p99 17 µs at batch 1 on one x86 thread; max error vs torch 2.4e-7.

## Limitations
- The shipped policies are proprioceptive only. Neither upstream R1 MJCF has a head camera. The sim head camera (`docs/HEAD_CAMERA.md`) so far only feeds the turn command, through a colour-key detector that relies on the opponent being orange. The opponent's steering and the R1's hold-your-spot loop use true positions.
- Nothing has run on a real R1. Armature, friction and latency are assumptions until system ID (`docs/MODEL_AUDIT.md`, ROADMAP phase 4).
- The robots' shells can still overlap by up to 2–3 cm for a few steps in hard impacts (contact can't be stiffer at this timestep).
- The opponent performs fight *forms* (KungFuAthleteBot, trimmed to the strikes), not sparring aimed at the R1. It is kept in front and faces the R1, but it doesn't adapt its strikes to it. M1's aim (off by default; "Aimed opponent" above) bends the shipped tracker's strikes toward the head; this line stays until the whole-body aimed tracker (M2) passes on test.
- Turning is weak: a +0.8 rad/s command turns the footwork policies at 0.09–0.18 rad/s here (0.08–0.19 in mjlab, where they trained); right turns are slower still for the guard (0.08 rad/s here, 0.11 in mjlab) but not for the arms-down one (0.12 / 0.15), which also tracks sideways commands poorly. A turning pass (stage 4) didn't fix it.
- Holding the spot is an outer loop, not learned: in the hard tier about one surviving bout in five still leaves the 1 m circle at some point.
- The arms-down footwork policy, with its hold loop, falls more against the puppet stress test than PR #13's base without one (test -0.025 [-0.05, -0.01]; up and within 1 m -0.037 [-0.07, -0.01]); the loop and the fine-tune each account for part of it, neither significantly.

## Credits
See [CREDITS.md](CREDITS.md): Unitree R1 MJCF (unitree_rl_mjlab, Apache-2.0), KungFuAthleteBot motions, MuJoCo, mjlab.
