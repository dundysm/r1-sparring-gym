# r1-sparring-gym

MuJoCo simulation and RL for turning a **Unitree R1** into a sparring partner that can take hits from a real opponent and stay up.

![demo](media/demo.gif)

**Silver:** our R1 running the shipped policy, a stepping balance controller trained on GPU from Unitree's own R1 task, holding a boxer's guard. **Orange:** the opponent, a second R1 **on its own motors**. It is driven by a motion-tracking policy that performs cleaned KungFuAthleteBot fight clips (here the held-out `kfab_202`) and turns to face our R1. No invisible forces hold it up. **Yellow spheres** are real contacts between the two robots (size ∝ force); **orange arrows** are scripted pushes from the side and behind. Full video: [`media/demo.mp4`](media/demo.mp4). Reproduce: `python scripts/demo.py`.

**Status (all simulation):**
- Against the opponent on held-out clips, the shipped policy survives 10 s in **100%** of test episodes (95% CI 98%–100%) and 97.5% on validation.
- Against the harder assisted "puppet" opponent it survives 99.4% (test).
- Its single-hit fall rate is **0.044**, vs 0.499 for the best residual policy (v2). It stays up at least half the time at every tested impulse up to 30 N·s.
- The opponent itself performs its clips upright in **47/48** 10 s runs.

## Headline eval
`scripts/eval_opponent.py`.
- **Episodes:** each lasts 10 s and samples a clip, a distance of 0.50–0.70 m, the clip start time and the obs delay (0–2 steps). Each adds two scripted side/back pushes (4–6 N·s).
- **Size:** 160 test episodes (kfab_202, kfab_205) and 80 validation episodes (kfab_204).
- **Pairing:** every controller sees exactly the same episodes, so differences vs PD are paired (bootstrap 95% CI; see `results/eval_opponent/`).
- **Selection:** candidates are chosen on validation only, and test is reported after the choice.

There are two opponents:
- **Tracked (default):** the R1 performs the clip on its own motors through a tracking policy, as in the demo.
- **Puppet (stress test, `--opponent puppet`):** the same clips forced through a bounded root assist (up to 250 N horizontal, 600 N vertical). It is stronger than any robot could be, and it levitated in the old demo.

| | **tracked: test survival** [95% CI] | tracked: val | puppet: test survival [95% CI] | puppet: val | single-hit fall rate [95% CI] | ≥50% survival up to | 3 hits @ 12 N·s, randomised physics |
|---|---|---|---|---|---|---|---|
| PD stand | **0.206** [0.15, 0.28] | 0.125 | 0.056 [0.03, 0.10] | 0.000 | 0.606 [0.58, 0.63] | 6 N·s | 0.04 |
| RL v1 (residual) | **0.531** [0.45, 0.61] | 0.250 | 0.225 [0.17, 0.30] | 0.000 | 0.495 [0.47, 0.52] | 8 N·s | 0.05 |
| RL v2 (residual) | **0.438** [0.36, 0.52] | 0.188 | 0.225 [0.17, 0.30] | 0.000 | 0.499 [0.47, 0.53] | 8 N·s | 0.05 |
| RL v3 cand. (residual) | **0.450** [0.38, 0.53] | 0.225 | 0.200 [0.14, 0.27] | 0.000 | 0.494 [0.47, 0.52] | 8 N·s | 0.04 |
| Stepping base, stage 1 | **0.819** [0.75, 0.87] | 0.738 | 0.713 [0.64, 0.78] | 0.600 | 0.212 [0.19, 0.24] | 20 N·s | 0.56 |
| Stepping base, arms down | **0.994** [0.96, 1.00] | 0.975 | 0.988 [0.96, 1.00] | 0.963 | 0.057 [0.04, 0.07] | 30 N·s | 0.95 |
| **Stepping base + guard (shipped)** | **1.000** [0.98, 1.00] | 0.975 | 0.994 [0.96, 1.00] | 0.950 | 0.044 [0.03, 0.06] | 30 N·s | 0.96 |

**Read it plainly:**
- **Residual policies (v1–v3).** They beat PD on single hits by about 11 points and survive the tracked opponent about half the time on test (about a fifth on validation), but survive the puppet only about 1 in 5 on test and never on validation. They can't step: the stiff stand they sit on falls from 0.04 rad of exploration noise (`docs/MODEL_AUDIT.md`).
- **The stepping base.** It survives both opponents in ≥95% of episodes. The guard version ties the arms-down base on validation (tracked 0.975 vs 0.975; puppet 0.950 vs 0.963, within CI), so it ships: same robustness, and it looks like sparring.
- **Not just retreat.** Surviving episodes drift 0.28 m on average, only 0.12 m of it backward, and the opponent keeps facing the R1.
- **The opponent falls too.** It falls in 5%–21% of tracked episodes, from contact with the R1. That is reported per controller (`opponent_fell`), not hidden.
- **The eval is near its ceiling** for the shipped policy on both opponents. A harder tier (longer bouts, opponents that chase, heavier pushes) is next.

## What the shipped policy is, and isn't
- **Training:** Unitree's official R1 velocity task in mjlab (MuJoCo Warp), plus more standing, sparring-range footwork, pushes every 1.5–3.5 s up to 0.8 m/s, 0–40 ms obs delay, and a **guard pose** as its default arm pose. About 470M env steps on one RTX 4090 (`gpu/README.md`).
- **Contract:** Unitree's own (kp 100/40, per-joint action scale 0.15–0.41, 83-d obs, zero joint friction as mjlab sets it; `configs/r1_mjlab.yaml`, `docs/DEPLOY.md`). It is evaluated here in CPU MuJoCo, a different simulator. `scripts/crosssim_mjlab.py` checks that it stands and tracks commands here.
- **Footwork is weak.** Robustness and the guard cost velocity tracking: commanded 0.5 m/s forward gives 0.21 m/s, 0.3 m/s sideways gives 0.21, and turning 0.5 rad/s gives 0.12. That is fine for sparring in place, but footwork needs another pass.
- **It doesn't see, block, counter or clinch.** It keeps its guard up and stays standing. Phase 3 (perception, strikes, blocks) builds on it.
- **No hardware.** It uses Unitree's deploy gains, not the old kp 300 stand. The R1 EDU still needs system ID (phase 4). Unitree's own R1 `deploy.yaml` lists a flat 0.25 action scale where its exported policies use 0.15–0.41 per joint (`docs/DEPLOY.md`).

## The opponent
`r1gym/opponent.py`, `gpu/r1_track_task.py`, `results/opponent_tracker/`.
- **Motion:** KungFuAthleteBot G1 clips, retargeted to the R1 by keypoint IK anchored to the G1's own joint angles. The retarget is cleaned:
  - unit quaternions;
  - short mocap glitches interpolated;
  - aerial flips cut;
  - heading-safe joins.

  Then it is placed in place, facing the R1, with seamless loops, and shifted so the supporting foot is on the floor.
- **Performer:** a BeyondMimic-style tracking policy (unitree_rl_mjlab's tracking task on the R1) on Unitree's motors with no assist. Moves it couldn't perform even after training (`r1gym/assets/motions/tracker_cuts.json`, located inside mjlab) are cut, and `kfab_199` is dropped. It stays up in 47/48 10 s runs on its own (`scripts/eval_tracker.py`).
- **Parity:** run from the same starts, the tracker falls at the same moments in mjlab (where it trained) and in this repo. The observation matches mjlab's training loop: the reference one frame ahead and fresh kinematics, checked by a review against mjlab's source.

## Plan
1. **Sim.** R1 model on the official Unitree MJCF, a fight-motion opponent, procedural hits, sensor delay and noise. Done.
2. **Hit-robust balance against a physical opponent.** The residual PPO on a PD stand (v1–v3) can't step. It was replaced by the **stepping base with a guard** (`gpu/`, `results/policy_mjlab_guard/`), and the opponent is a tracked R1 on its own motors (`results/opponent_tracker/`). Done in sim.
3. **Perception + block / counter / clinch** from the head camera. Roadmap: [ROADMAP.md](ROADMAP.md).
4. **Safety + sim-to-real** on the R1 EDU (system ID, e-stop, padded mode). Roadmap: [ROADMAP.md](ROADMAP.md).

## What's in the repo
| path | what |
|---|---|
| `r1gym/model.py` | R1 from `unitree_rl_mjlab` r1.xml (24 actuated DoF, 28.93 kg), PD servos, torque caps; the opponent R1 with its own capped motors |
| `r1gym/opponent.py` | the opponent: clip retarget, cleaning, cuts and grounding (`FightReplay`); the tracking-policy performer (`Tracker`); puppet and kinematic modes |
| `results/opponent_tracker/` | the opponent's tracking policy (`policy.onnx` with the motion bundled, `motion_index.json`, `model_9748.pt`, `params/`, `eval_tracker.json`) |
| `results/policy_mjlab_guard/` | **shipped** stepping base with guard (`policy.onnx`, `model_5997.pt`, `params/`, `crosssim.json`) |
| `r1gym/scenario.py` | the demo scenario shared by `demo.py` and `demo_check.py` |
| `configs/r1_mjlab.yaml`, `r1gym/policy.py` (`MjlabPolicy`) | runs unitree_rl_mjlab policies here under their own contract (checked against the ONNX metadata); same opponent |
| `gpu/` | the mjlab tasks (stepping base, guard, opponent tracker), the motion export, and how to reproduce them on RunPod |
| `results/policy_mjlab/` | stepping base, arms down (`policy.onnx`, `model_4498.pt`, `params/`, `train_log.json`, `crosssim.json`); `results/policy_mjlab_stage1/` for comparison |
| `r1gym/perturb.py` | strikes / pushes / grabs by impulse, body and direction; peak force capped at 400 N |
| `r1gym/env.py` | env (78-dim obs, 24-dim residual action), 0–40 ms obs delay, noise, optional opponent in training, thread/process batching |
| `r1gym/policy.py` | PD baseline, torch policy, ONNX policy |
| `scripts/` | `verify_r1.py`, `train.py`, `evaluate.py`, `demo_check.py`, `demo.py`, `export_onnx.py` |
| `configs/r1_assumptions.yaml` | every parameter not taken from Unitree (armature, foot friction, PD gains) |
| `results/policy/` | v2 policy (`policy.pt`, `policy.onnx`, `train_log.json`, `latency.json`); `results/policy_v1/` = v1 |
| `results/eval/`, `results/eval_v1/` | hit eval for v2 / v1 (`summary.json`, `survival.png`) |
| `results/policy_v3_candidate/`, `results/eval_v3/` | v3 attempt (DR + recovery reward + multi-hit curriculum): **not adopted**, see below |
| `results/demo_check.json` | demo fall times + 8 variants, with opponent contact counts and peak forces |
| `docs/MODEL_AUDIT.md`, `docs/GPU.md`, `docs/DEPLOY.md` | model audit; GPU-batched (mjlab / MuJoCo Warp) path; deploy contract (`r1gym/deploy.py`, `deploy_contract.json` next to each ONNX) |

## Quickstart
```bash
pip install -e .[dev]
python scripts/verify_r1.py                  # R1 spec checks -> results/r1_verification.json
python scripts/train.py --envs 256 --horizon 32 --workers 10 --minutes 30 --imp0 6 --opp_prob 0.5 \
    --init results/policy_v1/policy.pt --out results/policy_new   # v2 recipe, stage 1
python scripts/eval_opponent.py --split test --policy v2=results/policy/policy.pt guard=results/policy_mjlab_guard/policy.onnx   # headline (--split val, --opponent puppet)
python scripts/eval_tracker.py                                   # how well the opponent performs its clips on its own
python scripts/crosssim_mjlab.py results/policy_mjlab_guard/policy.onnx   # mjlab-trained policy stands and tracks commands in this sim
python scripts/evaluate.py --out results/eval_hits               # PD + v1 + v2 + v3: single hits to 30 N·s, 3-hit survival, CIs
python scripts/demo_check.py --policy v2=results/policy/policy.pt v3=results/policy_v3_candidate/policy.pt   # demo fall times + 8 variants
python scripts/demo.py   # the README clip (headless Linux: MUJOCO_GL=egl); add --left "v2=results/policy/policy.pt" for side by side
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
- Proprioceptive only. Neither upstream R1 MJCF has a head camera, so nothing here sees the opponent (that is phase 3).
- Nothing has run on a real R1. Armature, friction and latency are assumptions until system ID (`docs/MODEL_AUDIT.md`, ROADMAP phase 4).
- The opponent performs fight *forms* (KungFuAthleteBot), not sparring motion aimed at the R1. It faces the R1 but doesn't chase or adapt to it.
- The shipped policy trades footwork (velocity tracking) for robustness and the guard.
- The headline eval is near its ceiling for the shipped policy; it needs a harder tier.

## Credits
See [CREDITS.md](CREDITS.md): Unitree R1 MJCF (unitree_rl_mjlab, Apache-2.0), KungFuAthleteBot motions, MuJoCo, mjlab.
