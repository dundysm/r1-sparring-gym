# r1-sparring-gym

MuJoCo simulation and RL for turning a **Unitree R1** into a near-real-time sparring / grappling partner (built for Zoon Robotics' R1 EDU).

![demo](media/demo.gif)

Left: stiff PD stand (baseline). Right: the RL balance policy. Both face the same R1 opponent (KungFuAthleteBot clip `kfab_202`, held out of training) and the same scripted hits. In this demo PD falls at 9.1 s and RL at 11.0 s. The demo is not hand-picked; see "Demo robustness" below for 8 re-sampled variants. Full video: [`media/demo.mp4`](media/demo.mp4).

## Plan
1. **Sim.** R1 model on the official Unitree MJCF, a fight-motion opponent, procedural hits, sensor delay and noise. Done.
2. **Hit-robust balance policy.** Residual PPO on top of a PD stand, exported to ONNX. Done (v2 in this README).
3. **Perception + block / counter / clinch** from the head camera. Roadmap: [ROADMAP.md](ROADMAP.md).
4. **Safety + sim-to-real** on the R1 EDU (system ID, e-stop, padded mode). Roadmap: [ROADMAP.md](ROADMAP.md).

## What's in the repo
| path | what |
|---|---|
| `r1gym/model.py` | R1 from `unitree_rl_mjlab` r1.xml (24 actuated DoF, 28.93 kg), PD servos, torque caps, opponent attachment with compliant (padded) contact |
| `r1gym/opponent.py` | kinematic sparring partner: a second R1 replaying KungFuAthleteBot G1 clips retargeted to R1 by keypoint IK |
| `r1gym/perturb.py` | strikes / pushes / grabs by impulse, body and direction; peak force capped at 400 N |
| `r1gym/env.py` | env (78-dim obs, 24-dim residual action), 0–40 ms obs delay, noise, optional opponent in training, thread/process batching |
| `r1gym/policy.py` | PD baseline, torch policy, ONNX policy |
| `scripts/` | `verify_r1.py`, `train.py`, `evaluate.py`, `demo_check.py`, `demo.py`, `export_onnx.py` |
| `configs/r1_assumptions.yaml` | every parameter not taken from Unitree (armature, foot friction, PD gains) |
| `results/policy/` | v2 policy (`policy.pt`, `policy.onnx`, `train_log.json`, `latency.json`); `results/policy_v1/` = v1 |
| `results/eval/`, `results/eval_v1/` | hit eval for v2 / v1 (`summary.json`, `survival.png`) |
| `results/demo_check.json` | demo fall times + 8 variants |
| `docs/MODEL_AUDIT.md`, `docs/GPU.md` | model audit; GPU-batched (mjlab / MuJoCo Warp) path |

## Quickstart
```bash
pip install -e .[dev]
python scripts/verify_r1.py                  # R1 spec checks -> results/r1_verification.json
python scripts/train.py --envs 256 --horizon 32 --workers 10 --minutes 30 --imp0 6 --opp_prob 0.5 \
    --init results/policy_v1/policy.pt --out results/policy_new   # v2 recipe, stage 1
python scripts/evaluate.py --policy results/policy/policy.pt     # -> results/eval/
python scripts/demo_check.py --variants 8                        # demo fall times, no rendering
MUJOCO_GL=egl python scripts/demo.py                             # -> media/demo.mp4, demo.gif, still_*.png
python scripts/export_onnx.py --ckpt results/policy/policy.pt    # -> results/policy/policy.onnx + latency
pytest -q
```

## Results (v2 vs v1 vs PD)
Eval: 896 single-hit episodes per controller (strike/push × 7 locations × 8 impulses × 4 directions × 2 reps), 0–40 ms obs delay, sensor noise.

| | PD stand | RL v1 | **RL v2** |
|---|---|---|---|
| Overall fall rate | 0.52 | 0.413 | **0.411** |
| Fall rate @ 4 / 6 / 8 / 10 N·s | 0.15 / 0.39 / 0.61 / 0.71 | 0.00 / 0.16 / 0.37 / 0.63 | 0.00 / 0.19 / 0.39 / 0.57 |
| Largest impulse with ≥50% survival | 6 N·s | 8 N·s | 8 N·s |
| Fall rate chest / head / pelvis | 0.72 / 0.76 / 0.57 | 0.59 / 0.67 / 0.42 | 0.56 / 0.66 / 0.43 |
| Demo (`kfab_202`) fall time | 9.1 s | 6.1 s | **11.0 s** |
| Demo variants: mean time to fall (8 runs, 12 s clip) | 9.16 s | 7.34 s | **9.74 s** |
| Demo variants where RL outlasts PD | – | 2 / 8 | **6 / 8** |

On single hits, v2 is about the same as v1 (0.411 vs 0.413 is within noise). It is slightly better at 10 N·s and slightly worse at 6–8 N·s. The real gain is with an opponent in contact range. v1 never saw one in training and usually fell about 1.5 s after the 4.6 s head strike plus the opponent's knee contact. v2 trains with an opponent in 50–70% of episodes.

**Demo robustness.** Each of the 8 variants re-samples hit durations, the obs delay (0–2 steps) and the opponent distance (0.80–0.95 m). RL v2 fall times are 6.28, 11.0, 10.8, 10.84, 6.18, 10.8, 11.04 and 10.96 s; PD falls at 9.0–9.36 s in all 8. Neither controller survives the 8 N·s push at 9.5 s every time. The candidate was chosen on eval fall rate plus mean variant survival across 5 trained candidates (3 fine-tunes without the opponent, 2 with it; numbers in the PR). It was not chosen on the main demo alone.

![survival](results/eval/survival.png)

**Opponent force.** The opponent is kinematic (infinite mass), so its contacts are made compliant like a padded surface (`OPP_SOLREF`/`OPP_SOLIMP` in `r1gym/model.py`). Peak contact force in the demo is 315 N (it was 630 N before). Scripted hit peak force is capped at 400 N, with the impulse preserved.

## Compute and cost
- **v1:** PPO, 60 min, 65.8M env steps, one RunPod RTX 4090 pod (training is CPU MuJoCo on the pod's ~10-vCPU quota; the GPU is barely used).
- **v2:** warm-started from v1 on one RunPod RTX 4090 pod ($0.89/h), with runs in parallel:
  - Stage 1: 27 min, 4.2M steps, opponent in 50% of episodes.
  - Stage 2: 25 min, 3.9M steps, opponent in 70%, starting impulse 7 N·s.
  - The pod was up about 1.9 h in total, about $1.7, including 3 discarded fine-tunes without the opponent and a few crashed launches (the pod has a 31 GB memory limit).
- **ONNX:** 227 KB, p50 8.4 µs / p99 17 µs at batch 1 on one x86 thread; max error vs torch 2.4e-7.

## Limitations
- Proprioceptive only. Neither upstream R1 MJCF has a head camera, so nothing here sees the opponent (that is phase 3).
- Armature, foot friction and PD gains are assumptions, not identified on hardware (`docs/MODEL_AUDIT.md`). Nothing has been run on a real R1.
- The single-hit eval has no grabs and no opponent contact. The demo variants cover those, but only 8 runs on one held-out clip.
- The policy only absorbs hits standing in place. It doesn't step, block or counter yet.
- Small training budget (about 74M steps total) on CPU MuJoCo. The GPU path in `docs/GPU.md` isn't benchmarked here.
- CI isn't set up. Tests run locally (`pytest -q`).

## Credits
See [CREDITS.md](CREDITS.md): Unitree R1 MJCF (unitree_rl_mjlab, Apache-2.0), KungFuAthleteBot motions, MuJoCo, mjlab.
