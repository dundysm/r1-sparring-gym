> **Done:** the stepping base was trained this way. See [`gpu/README.md`](../gpu/README.md) for the task, the version pins and the commands that worked (2026-10-26). The plan below is kept for reference.

# Batched GPU path (thousands of parallel copies)

The CPU gym here (`r1gym/`) is the reference: model, hit model, sensor delay/noise, eval harness. It runs about 4.2k env-steps/s on
8 vCPUs (measured). To train at scale, you run the same physics on GPU through **MuJoCo Warp**. Three ready stacks, all
built on the same `r1.xml` we use:

| stack | what it gives you | R1 support |
|---|---|---|
| [mjlab](https://github.com/mujocolab/mjlab) (Apache-2.0) | Isaac-Lab-style manager API on MuJoCo Warp, rsl_rl PPO, 4096+ envs per GPU | via unitree_rl_mjlab |
| [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (Apache-2.0) | Unitree's official tasks, incl. `Unitree-R1-Flat` velocity and sim2real deploy with ONNX export | yes (official) |
| wbc-mjlab | whole-body-control / tracking tasks on mjlab | G1-first; port R1 via the same asset cfg |

## Full-scale command (Unitree's official R1 task)
```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab && cd unitree_rl_mjlab && uv sync   # CUDA 12 GPU
python scripts/train.py Unitree-R1-Flat --env.scene.num-envs=4096      # official velocity task (has push events)
python scripts/play.py  Unitree-R1-Flat --checkpoint_file=logs/rsl_rl/<run>/model_<n>.pt
```
## Porting this gym's hit task to it (Phase 2b, about 1 day)
1. Copy `configs/r1_assumptions.yaml` values into the R1 actuator cfg (`stand_pd` gains, caps = min, armature, foot μ).
2. Add an event that samples `r1gym.perturb.random_schedule` per env and writes `xfrc_applied` (mjlab exposes it on the
   Warp data). Use the same impulse curriculum as `scripts/train.py` (grow ×1.15 when episode fall rate < 25%).
3. Add an observation-delay term (0–2 steps; mjlab#1230 pattern) and the same noise levels.
4. Rewards and terminations: copy them from `r1gym/env.py`. Command = zero velocity (stand and absorb). Then a second stage with
   velocity commands so it can step out of hits; that's where the big survival gains are.
5. Evaluate with *this* repo's `scripts/evaluate.py` (CPU MuJoCo, same MJCF). That gives you a cross-simulator check for free.

Rough cost: on a single RTX 4090 / L40S, mjlab velocity tasks reach 100M+ steps/hour, so a 1–2 h run costs about $1–3 on RunPod.
That's an estimate, not something we measured here.
