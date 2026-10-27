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

# Guard: the stepping base's default arm pose (its action offset and the pose the posture reward holds) becomes a boxer's
# guard: fists at cheek height, elbows down by the ribs. Found by FK/IK on the R1 (gpu/README.md), inside mjlab's soft
# joint limits (90% of range). Fine-tuned from the shipped stepping base.
GUARD_ARMS = {"left_shoulder_pitch_joint": -0.85, "left_shoulder_roll_joint": 0.15, "left_shoulder_yaw_joint": 0.2, "left_elbow_joint": -0.7,
              "right_shoulder_pitch_joint": -0.85, "right_shoulder_roll_joint": -0.15, "right_shoulder_yaw_joint": -0.2, "right_elbow_joint": -0.7}

def _guard(make):
    def g(play=False):
        cfg = make(play); robot = cfg.scene.entities["robot"]; jp = dict(robot.init_state.joint_pos)
        jp = {k: v for k, v in jp.items() if not any(a in k for a in ("shoulder", "elbow"))}
        jp.update(GUARD_ARMS); jp.update({".*_wrist_roll_joint": 0.0})
        import dataclasses; robot.init_state = dataclasses.replace(robot.init_state, joint_pos=jp)   # HOME_KEYFRAME is shared: copy
        return cfg
    return g

# Stage 3, footwork (fine-tuned from the shipped bases): the outer loops in this repo (hold your spot, turn to face;
# r1gym/env.py) drive the policy through its velocity command, and the shipped bases track it poorly (0.5 m/s -> 0.2-0.3,
# 0.3 sideways -> 0.2) and the arms-down one doesn't turn left in CPU MuJoCo (results/face_opponent_ab.json).
#  * Unitree's command_vel curriculum is removed, so the ranges below are the ones trained (it overrode the task's
#    ranges before; gpu/README.md); 30% standing; velocity and yaw tracking weighted 1.5x.
#  * Upper-body shoves like the opponent's: apply_body_impulse on the torso, 10 cm above its COM, each force component
#    uniform in +-100 N for 0.05-0.15 s (typically ~5-15 N*s) every 1-3 s, on top of stage 2's velocity pushes.
def _footwork(make):
    def f(play=False):
        cfg = make(play); cfg.curriculum.pop("command_vel", None)
        tw = cfg.commands["twist"]; tw.rel_standing_envs = 0.3
        tw.ranges.lin_vel_x = (-0.6, 0.8); tw.ranges.lin_vel_y = (-0.5, 0.5); tw.ranges.ang_vel_z = (-1.0, 1.0)
        cfg.rewards["track_linear_velocity"].weight = 1.5; cfg.rewards["track_angular_velocity"].weight = 1.5
        if not play:
            from mjlab.envs import mdp as envs_mdp
            from mjlab.managers.event_manager import EventTermCfg
            from mjlab.managers.scene_entity_config import SceneEntityCfg
            cfg.events["shove"] = EventTermCfg(func=envs_mdp.apply_body_impulse, mode="step", params={
                "force_range": (-100.0, 100.0), "torque_range": (0.0, 0.0), "duration_s": (0.05, 0.15), "cooldown_s": (1.0, 3.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "body_point_offset": (0.0, 0.0, 0.1)})
        return cfg
    return f

def register():
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.rl import VelocityOnPolicyRunner
    from src.tasks.velocity.config.r1.rl_cfg import unitree_r1_ppo_runner_cfg
    for stage, tid in [(1, "R1-Spar-Stand"), (2, "R1-Spar-Stand-Push")]:
        make = _cfg(stage); rl = unitree_r1_ppo_runner_cfg(); rl.experiment_name = "r1_spar"
        register_mjlab_task(task_id=tid, env_cfg=make(), play_env_cfg=make(play=True), rl_cfg=rl, runner_cls=VelocityOnPolicyRunner)
    make = _guard(_cfg(2)); rl = unitree_r1_ppo_runner_cfg(); rl.experiment_name = "r1_spar"
    register_mjlab_task(task_id="R1-Spar-Guard", env_cfg=make(), play_env_cfg=make(play=True), rl_cfg=rl, runner_cls=VelocityOnPolicyRunner)
    for tid, make in [("R1-Spar-Footwork", _footwork(_cfg(2))), ("R1-Spar-Footwork-Guard", _footwork(_guard(_cfg(2))))]:
        rl = unitree_r1_ppo_runner_cfg(); rl.experiment_name = "r1_spar"
        register_mjlab_task(task_id=tid, env_cfg=make(), play_env_cfg=make(play=True), rl_cfg=rl, runner_cls=VelocityOnPolicyRunner)

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts"))
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
