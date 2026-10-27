# GPU training (mjlab)

Two policies are trained here: the **stepping base** our R1 runs (`r1_spar_task.py`) and the **opponent's motion tracker** (`r1_track_task.py`). Both use Unitree's own R1 contract in mjlab and are evaluated in this repo's CPU MuJoCo.

## Stepping base

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

### Guard pose (shipped)
`R1-Spar-Guard` is stage 2 with the default arm pose replaced by a boxer's guard: shoulder pitch -0.85, roll ±0.15, yaw ±0.2, elbow -0.7. Fists sit at cheek height and the elbows hang down by the ribs. The pose was found by rendering FK candidates on the R1 (its elbow flexes forward for *negative* angles). It is inside mjlab's soft joint limits (90% of range; the elbow's is -0.82). The default pose is the action offset and what the posture reward holds, so the policy keeps its guard up while it steps. The run fine-tuned 1500 iterations from the shipped stepping base (`--agent.load-run <dir> --agent.load-checkpoint model_4498.pt`), about 28 min. On validation it ties the arms-down base (0.975 vs 0.975 against the tracked opponent; 0.950 vs 0.963 against the puppet, within CI), and it was selected because it keeps the guard.

## Opponent tracker (`r1_track_task.py`)
This is unitree_rl_mjlab's BeyondMimic-style tracking task on the R1: R1 robot and action scale, R1 body names, and no anchor position or linear velocity in the actor (IMU + encoders only). The motion is `export_motion.py`'s concatenation of the cleaned clips (each twice, cross-faded), exactly what `FightReplay` replays.

```bash
python export_motion.py                                         # in the repo -> gpu/motions/r1_kungfu.csv + index
python r1_track_task.py convert r1_kungfu.csv motions/r1_kungfu.npz      # on the GPU box (unitree_rl_mjlab checkout)
python r1_track_task.py R1-Track-KungFu-Relaxed2 --env.scene.num-envs 4096 --agent.max-iterations N   --agent.logger tensorboard --motion-file motions/r1_kungfu.npz [--agent.resume True --agent.load-run <dir> --agent.load-checkpoint model_<k>.pt]
```

What it took (2026-10-26, about 3.3 h on one RTX 4090, ~$3):
1. **Clean the source clips first** (`r1gym/opponent.py`):
   - The KungFuAthleteBot root quaternions aren't unit length (0.91–1.12).
   - There are short mocap glitches; these are interpolated.
   - kfab_204/205 contain aerial flips; these are cut.
   - Before cleaning, the converter's velocity check was off by 6.8; after, it is exact.
2. **Relax the end-effector termination.** With 0.25 m on hands and feet, 91% of episodes ended on a hand: the R1's elbows and wrists are kp 20, so fast arm swings lag. Hands get 0.5 m (`-Relaxed`), then feet get 0.4 m and hands 0.6 m (`-Relaxed2`), since high kicks were the next limit. Balance stays guarded by the anchor height and orientation terms.
3. **Cut the moves the R1 can't do.** After 5750 iterations the opponent stayed up in only 38% of runs, and the falls clustered on specific moves. Running the tracker inside mjlab from 24 starts per clip located them, and the sim parity was exact (mjlab 21/56 runs vs this repo 21/56). `r1gym/assets/motions/tracker_cuts.json` lists the cuts; kfab_199 failed everywhere and is dropped.
4. **Ground the motion.** The retarget only scaled root height, so the supporting foot sat a few cm off the floor. `ground()` shifts root height on stance frames.
5. **Resume on the trimmed, grounded motion** for 4000 iterations (9748 in total). Episodes went from ~1 s to ~8.4 s of 10. In this repo the opponent stays up in **47/48** 10 s runs (`scripts/eval_tracker.py`, `results/opponent_tracker/eval_tracker.json`).

In this repo (`r1gym/opponent.py` `Tracker`), these follow the training contract:
- The opponent's motors always use Unitree's gains, the mjlab caps, armature 0.01 and zero joint friction, with feet μ 0.6 and self-collision on.
- The reference shown is the frame to reach by the end of the control step, as mjlab's `MotionCommand` does after its increment.
- Kinematics are refreshed before the observation.
- The whole reference is turned about z, smoothed over 0.5 s, so the opponent keeps facing the R1. The tracker only sees the reference relative to its torso, so this equals turning the world, which training covers.

A test checks nothing pushes on its pelvis.
