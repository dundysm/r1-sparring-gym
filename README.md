# r1-sparring-gym

**A MuJoCo sparring gym for the Unitree R1, plus a balance policy that withstands hits better than a stiff PD stand.** It uses the official Unitree R1 MJCF (24 actuated DoF, 28.93 kg). Every R1 spec is checked by `scripts/verify_r1.py`. The sparring partner is a second R1 replaying KungFuAthleteBot fight clips retargeted from the G1. Procedural strikes, pushes and grabs, sensor delay and noise, and per-body contact forces are all built in. The policy is ~8 µs per inference on CPU, exported to ONNX for Jetson. This is the base for phase 3: strike / block / clinch from the head camera on a real R1 EDU.

![demo](media/demo.gif)

`media/demo.mp4` shows a PD stand (left) and the RL policy (right) under the same opponent clip and hit schedule. In this particular 12 s demo both R1s eventually fall: PD at 7.0 s, RL at 6.3 s, after a 4 N·s grab. **Judge the policy on the eval below, not on the demo.**

## Headline numbers (all measured in this repo; see `results/`)
| | PD stand baseline | RL policy |
|---|---|---|
| Overall fall rate, 896 hits per controller (8 impulses × 7 locations × 4 directions × strike/push × 2 reps) | **0.52** | **0.41** |
| Largest impulse with ≥50% survival | 6 N·s | **8 N·s** |
| Fall rate @ 4 / 6 / 8 / 10 N·s | 0.15 / 0.39 / 0.61 / 0.71 | **0.00 / 0.16 / 0.37 / 0.63** |
| Fall rate, chest / head / pelvis hits | 0.72 / 0.76 / 0.57 | 0.59 / 0.67 / 0.42 |

- **Recovery time:** median time until tilt <5° and angular velocity <0.3 rad/s, among survivors. The RL policy takes 1.0–1.6 s for 2–15 N·s. PD shows 0.0 s because its stiff survivors barely tilt past 5°; PD doesn't move, it either holds or topples.
- **Training:** PPO, 60 min on one RTX 4090 pod (30 CPU MuJoCo workers), 65.8M env steps at ~18k env-steps/s.
  - Curriculum on impulse: 2 → ~9 N·s.
  - Observation delay of 0–40 ms, sensor noise.
- **ONNX:** 227 KB. Batch-1 latency is p50 8.2 µs / p99 12.9 µs on 1 x86 thread; batch 4096 takes 5.3 ms. Max error vs torch is 1.8e-7.
- **Throughput (CPU MuJoCo):** 4.2k env-steps/s on an 8-vCPU box, 24.9k on 32 vCPU. A GPU-batched path (mjlab / MuJoCo Warp) is documented in `docs/GPU.md`; it isn't benchmarked here.

![survival](results/eval/survival.png)

## Quickstart
```bash
pip install -e .[dev]
python scripts/verify_r1.py            # R1 spec checks -> results/r1_verification.json
python -m r1gym.check_model            # joints, limits, torque caps, stand test
python scripts/train.py --envs 512 --horizon 32 --workers 30 --minutes 60 --out results/policy   # (--device cuda)
python scripts/evaluate.py --policy results/policy/policy.pt   # survival curve vs PD
python scripts/export_onnx.py --ckpt results/policy/policy.pt  # ONNX + latency
MUJOCO_GL=egl python scripts/demo.py   # media/demo.mp4, stills, gif
pytest -q
```

## What's in it
- **`r1gym/model.py`:** R1 from `unitree_rl_mjlab` r1.xml, with PD position servos and torque caps = min(unitree_mujoco ctrlrange, mjlab effort limit). Every assumption lives in `configs/r1_assumptions.yaml`; the audit is in `docs/MODEL_AUDIT.md`.
- **`r1gym/opponent.py`:** a kinematic sparring partner.
  - The default (`opponent_model: r1`) is a second R1. KungFuAthleteBot G1 clips are retargeted with keypoint IK on knees, ankles, torso, elbows and wrists.
  - `g1` replays the raw G1.
- **`r1gym/perturb.py`:** strikes, pushes and grabs, specified by impulse, body and direction.
- **`r1gym/env.py`:** observation delay and noise, an action filter, per-body contact logging, and thread- and process-batched envs.
- **`scripts/`:** train (PPO), evaluate, export_onnx, demo, verify_r1.

## Honest gaps
- **Unidentified parameters:** armature 0.03, foot μ 1.0 and the stand PD gains are assumptions; they aren't identified on hardware. See `docs/MODEL_AUDIT.md`.
- **No head camera:** neither upstream R1 MJCF has one, so phases 1–2 are proprioceptive only.
- **Infinite-mass opponent:** it is kinematic, so its kicks can apply kN-level contact forces. The eval uses procedural impulses only.
- **Single seed:** one policy and one training run, no seed variance yet.

Roadmap: [ROADMAP.md](ROADMAP.md) · Credits: [CREDITS.md](CREDITS.md)
