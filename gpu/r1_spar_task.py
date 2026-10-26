"""Stepping-capable R1 base policy on GPU (mjlab / MuJoCo Warp), built on Unitree's official R1 velocity task.

Why: the residual-on-stiff-stand policies can only explore ~0.015 rad (docs/MODEL_AUDIT.md) and stall at 8-9 N*s.
Unitree's own R1 task uses its deploy gains (kp 100/40), per-joint action scales of 0.15-0.41 rad, init std 1.0 and
velocity pushes, so the policy learns to step.

R1-Spar-Stand changes to Unitree-R1-Flat (unitree_rl_mjlab @1425b15, mjlab 1.2.0):
  * more standing envs (sparring is mostly in place) and sparring-range footwork commands
  * stage 2: pushes every 1.5-3.5 s instead of 5-6 s, |dv| up to 0.8 m/s (~23 N*s on 29 kg) instead of 0.5 m/s,
    and 0-40 ms observation delay
Run from a unitree_rl_mjlab checkout:  python <this file> R1-Spar-Stand --env.scene.num-envs=4096 [...train.py flags]"""
import sys
from pathlib import Path

def _cfg(stage):
    from src.tasks.velocity.config.r1.env_cfgs import unitree_r1_flat_env_cfg
    def make(play=False):
        cfg = unitree_r1_flat_env_cfg(play=play)
        tw = cfg.commands["twist"]
        tw.rel_standing_envs = 0.4
        tw.ranges.lin_vel_x = (-0.6, 0.8); tw.ranges.lin_vel_y = (-0.5, 0.5); tw.ranges.ang_vel_z = (-0.8, 0.8)
        if stage == 2 and not play:
            push = cfg.events["push_robot"]; push.interval_range_s = (1.5, 3.5)
            push.params["velocity_range"] = {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "z": (-0.3, 0.3),
                                             "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.8, 0.8)}
            # observation latency 0-2 control steps (0-40 ms), as the residual policies trained and as every eval
            # in this repo applies; Unitree's task trains with none
            for term in cfg.observations["actor"].terms.values(): term.delay_min_lag, term.delay_max_lag = 0, 2
        return cfg
    return make

def register():
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.rl import VelocityOnPolicyRunner
    from src.tasks.velocity.config.r1.rl_cfg import unitree_r1_ppo_runner_cfg
    for stage, tid in [(1, "R1-Spar-Stand"), (2, "R1-Spar-Stand-Push")]:
        make = _cfg(stage); rl = unitree_r1_ppo_runner_cfg(); rl.experiment_name = "r1_spar"
        register_mjlab_task(task_id=tid, env_cfg=make(), play_env_cfg=make(play=True), rl_cfg=rl, runner_cls=VelocityOnPolicyRunner)

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts"))
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
