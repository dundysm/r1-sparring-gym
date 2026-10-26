# Stepping base on GPU (mjlab)

`r1_spar_task.py` registers two tasks on top of Unitree's official R1 velocity task (`unitree_rl_mjlab` @1425b15):
- `R1-Spar-Stand`: 40% standing envs, footwork commands of vx −0.6…0.8, vy ±0.5, wz ±0.8. Unitree's pushes are unchanged (±0.5 m/s every 5–6 s).
- `R1-Spar-Stand-Push`: same as above, plus:
  - pushes every 1.5–3.5 s, up to ±0.8 m/s linear and ±0.6 rad/s roll/pitch
  - 0–2 steps (0–40 ms) of observation delay on every actor term

The shipped policy (`results/policy_mjlab/`) is stage 1 for 3000 iterations, then stage 2 resumed for 1500 more
(4096 envs, 24 steps per iteration, about 440M env steps in total).

## Run (RunPod, 1× RTX 4090, ~1.7 h, ~$1.5)
```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab && cd unitree_rl_mjlab && git checkout 1425b15
uv venv -p 3.11 /root/venv && . /root/venv/bin/activate && uv pip install -e .
# pins needed on 2026-10-26: mjlab 1.2.0 breaks with newer mujoco (mjENBL_MULTICCD) and warp (wp.context),
# and doesn't declare scipy
uv pip install "mujoco==3.5.0" "warp-lang>=1.12,<1.13" scipy tensorboard
python /path/to/r1_spar_task.py R1-Spar-Stand --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard
python /path/to/r1_spar_task.py R1-Spar-Stand-Push --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.logger tensorboard \
  --agent.resume True --agent.load-run <stage-1 run dir name>
```
Stage 1 runs at about 1.1 s per iteration and stage 2 at about 1.3 s per iteration (about 90k env steps/s). Don't set `MUJOCO_GL=egl` on the RunPod
PyTorch image: it has no EGL, and training doesn't render. The run writes `logs/rsl_rl/r1_spar/<run>/policy.onnx` with
the contract in its metadata.

## Bring it back
Copy `policy.onnx` and evaluate it here: `python scripts/eval_opponent.py --split val --policy step=<path>/policy.onnx`.
Choose candidates on `--split val`, report on `--split test`. `scripts/crosssim_mjlab.py <onnx>` checks that it stands
and tracks velocity commands in this repo's CPU MuJoCo, which is the check that the obs/action contract (`configs/r1_mjlab.yaml`)
is right.
