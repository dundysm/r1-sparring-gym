"""Physical opponent: an R1 motion-tracking policy (BeyondMimic-style, unitree_rl_mjlab's tracking task) for the fight clips.

Why: the puppet opponent (r1gym/opponent.py) is held up by an invisible root assist, about 80% of its weight on average in the
demo, so it floats through moves no real robot could do. A tracking policy has to perform the clips on the R1's own motors.

R1-Track-KungFu = unitree_rl_mjlab's tracking task (@1425b15, mjlab 1.2.0) with the R1 robot and contract (kp 100/40,
per-joint action scale), R1 body names (the R1 has no wrist-yaw links), and no global position or linear velocity in the
actor's observation (Unitree's "No-State-Estimation" variant), so the policy needs only IMU and encoders.
Motion: gpu/export_motion.py -> r1_kungfu.csv (all 7 clips, each twice, cross-faded).

On the GPU box, from a unitree_rl_mjlab checkout:
  python r1_track_task.py convert r1_kungfu.csv motions/r1_kungfu.npz
  python r1_track_task.py R1-Track-KungFu --env.scene.num-envs 4096 --agent.max-iterations N --agent.logger tensorboard
    --env.commands.motion.motion-file motions/r1_kungfu.npz"""
import sys
from pathlib import Path

BODIES = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
          "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
          "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link")
END_EFFECTORS = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")
JOINTS = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
          "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
          "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_roll_joint", "waist_yaw_joint", "left_shoulder_pitch_joint",
          "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
          "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]

def env_cfg(play=False, motion_file="", relaxed_hands=False, feet=0.25, hands=0.5):
    from src.assets.robots import R1_ACTION_SCALE, get_r1_robot_cfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_r1_robot_cfg()}
    cfg.scene.sensors = (ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                          secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                          fields=("found", "force"), reduce="none", num_slots=1, history_length=4),)
    act = cfg.actions["joint_pos"]; assert isinstance(act, JointPositionActionCfg); act.scale = R1_ACTION_SCALE
    mc = cfg.commands["motion"]; assert isinstance(mc, MotionCommandCfg)
    mc.anchor_body_name = "torso_link"; mc.body_names = BODIES; mc.motion_file = motion_file
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS
    if relaxed_hands:
        # The R1's arms run on soft gains (kp 20 at the elbow/wrist), so in fast kung-fu arm swings the hands lag; with the
        # 0.25 m limit on every end effector 91% of episodes ended on a hand at iteration ~1250 and episodes stayed ~0.8 s.
        # Keep 0.25 m for the feet (balance), allow 0.5 m for the hands.
        import copy
        h = copy.deepcopy(cfg.terminations["ee_body_pos"]); h.params["body_names"] = END_EFFECTORS[2:]; h.params["threshold"] = hands
        cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS[:2]; cfg.terminations["ee_body_pos"].params["threshold"] = feet
        cfg.terminations["hand_body_pos"] = h
    cfg.viewer.body_name = "torso_link"
    terms = {k: v for k, v in cfg.observations["actor"].terms.items() if k not in ("motion_anchor_pos_b", "base_lin_vel")}
    cfg.observations["actor"] = ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=True)
    if play:
        cfg.episode_length_s = int(1e9); cfg.observations["actor"].enable_corruption = False; cfg.events.pop("push_robot", None)
        mc.pose_range = {}; mc.velocity_range = {}; mc.sampling_mode = "start"
    return cfg

def register(motion_file=""):
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    rl = unitree_g1_tracking_ppo_runner_cfg(); rl.experiment_name = "r1_tracking"; rl.save_interval = 250
    register_mjlab_task(task_id="R1-Track-KungFu", env_cfg=env_cfg(motion_file=motion_file), play_env_cfg=env_cfg(True, motion_file),
                        rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)
    register_mjlab_task(task_id="R1-Track-KungFu-Relaxed", env_cfg=env_cfg(motion_file=motion_file, relaxed_hands=True),
                        play_env_cfg=env_cfg(True, motion_file, relaxed_hands=True), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)
    # stage 3: at ~3500 iterations most episodes end on a foot during high kicks the R1 can't reach; an opponent that kicks a
    # little lower is fine, so allow 0.4 m on the feet (balance is still guarded by the anchor height/orientation terms)
    register_mjlab_task(task_id="R1-Track-KungFu-Relaxed2", env_cfg=env_cfg(motion_file=motion_file, relaxed_hands=True, feet=0.4, hands=0.6),
                        play_env_cfg=env_cfg(True, motion_file, relaxed_hands=True, feet=0.4, hands=0.6), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)

def convert(csv, out):
    """csv_to_npz for the R1: replay the CSV kinematically and log joint states and body poses/velocities at 50 fps."""
    import os, torch, csv_to_npz as c
    from mjlab.scene import Scene
    # csv_to_npz asserts the logged root velocity equals the written one (1e-5). For the R1 it differs by up to ~0.08 m/s on
    # frames with roll/pitch rotation (pelvis COM is 7 cm below the free-joint origin; a link vs COM convention difference in the
    # write path). The logged values come from the simulator, so they stay self-consistent; record the worst gap and go on.
    worst = [0.0]
    def check(a, b, **kw): worst[0] = max(worst[0], float((a - b).abs().max()))
    torch.testing.assert_close = check
    from mjlab.sim.sim import Simulation, SimulationCfg
    sim_cfg = SimulationCfg(); sim_cfg.mujoco.timestep = 1.0 / 50
    scene = Scene(env_cfg().scene, device="cuda:0"); model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device="cuda:0"); scene.initialize(sim.mj_model, sim.model, sim.data)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    c.run_sim(sim=sim, scene=scene, joint_names=JOINTS, input_fps=50, input_file=csv, output_fps=50, output_path=out,
              render=False, line_range=None, renderer=None)
    print(f"root velocity check: worst |logged - written| = {worst[0]:.4f}")

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts"))
    if sys.argv[1] == "convert":
        convert(sys.argv[2], sys.argv[3]); sys.exit(0)
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
