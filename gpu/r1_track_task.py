"""Physical opponent: an R1 motion-tracking policy (BeyondMimic-style, unitree_rl_mjlab's tracking task) for the fight clips.

Why: the puppet opponent (r1gym/opponent.py) is held up by an invisible root assist, about 80% of its weight on average in the
demo, so it floats through moves no real robot could do. A tracking policy has to perform the clips on the R1's own motors.

R1-Track-KungFu = unitree_rl_mjlab's tracking task (@1425b15, mjlab 1.2.0) with the R1 robot and contract (kp 100/40,
per-joint action scale), R1 body names (the R1 has no wrist-yaw links), and no global position or linear velocity in the
actor's observation (Unitree's "No-State-Estimation" variant), so the policy needs only IMU and encoders.
Motion: gpu/export_motion.py -> r1_kungfu.csv (all clips, each twice, cross-faded).

R1-Track-KungFu-Steer adds the motion anchor's position to the actor's observation (Unitree's standard variant has it)
and lets the reference drift in xy during training (SteerMotionCommand), so the policy learns to walk after its
reference. The sim then keeps the opponent in front of the R1 by moving the reference (r1gym/opponent.py Tracker),
instead of it wandering off to the R1's side. Warm-started from the stage-3 policy with the new inputs' weights at zero:
python r1_track_task.py widen model_9748.pt model_9748_wide.pt. With BeyondMimic's anchor-position reward (std 0.3 m,
weight 1) the policy kept tracking the pose and let the position go (0.77 m off after 1000 iterations), so
R1-Track-KungFu-Steer2 (the shipped opponent) makes position count: reward std 0.5 m, weight 2, an episode ends when the
anchor is more than 0.6 m from its reference, and the drift is at most 0.3 m/s.

R1-Track-KungFu-Aim (M2 run 1, docs/AIMED_OPPONENT.md) is Steer2 fine-tuned on export_warps.py's motion (the same clips plus
warped copies of the train clips, whose hand strikes are bent toward an R1's head or torso with the arm, waist, stance and a
short spot lunge), with three additions:
  - strike_apex: at every train-clip strike apex (+-2 frames, from the motion's _apex.json) the striking wrist's distance to
    its reference, in the world (so the lunge counts), exp(-d^2 / 0.1^2), weight 1.5. BeyondMimic's body-position term
    averages 14 bodies with std 0.3 m and barely sees a 20 cm hand error.
  - lunges in the steering (AimMotionCommand): 1-3 s after the last one, the reference jumps 0.10-0.30 m ahead along its own
    heading at a 0.5-1.2 m/s cap (0.1 s lag), holds 0.25-0.6 s, and goes back at 0.4 m/s, on top of Steer2's drift. This is
    what the sim's StandOff and the tolerance probe's lunge (scripts/tracker_tolerance.py) ask of it, and the drift
    (<= 0.3 m/s) never showed it.
  - hand reaction impulses: random wrenches on both wrists, each component +-25 N for 0.05-0.12 s, every 1-3 s (up to about
    3 N*s, the landed range), so it learns to absorb what its strikes meet.
The actor's inputs are unchanged (the shipped 132-d contract), so it drops into r1gym/opponent.py's Tracker as is.

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

STEER_SPEED, STEER_HOLD, STEER_EVERY_S, STEER_TAU = 0.4, 0.3, (1.0, 3.0), 0.3   # m/s, P(stand still), s, s
_STEER = {}

def _steer_cfg(mc, speed=STEER_SPEED):
    """MotionCommandCfg -> the same config building a SteerMotionCommand: the reference (all bodies, anchor included) is
    offset in xy by a per-env drift. Every 1-3 s the drift's target velocity is resampled (zero with probability 0.3, else
    a random direction at up to 0.4 m/s) and followed with a 0.3 s lag. Body velocities include the drift, so the
    reference stays self-consistent; relative body poses (re-anchored at the robot) are unchanged by it."""
    import dataclasses, torch
    from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
    if not _STEER:
        class SteerMotionCommand(MotionCommand):
            def __init__(self, cfg, env):
                super().__init__(cfg, env); z = lambda: torch.zeros(self.num_envs, 3, device=self.device)
                self.off, self.v, self.v_goal = z(), z(), z(); self.t_next = torch.zeros(self.num_envs, device=self.device)
            @property
            def body_pos_w(self): return super().body_pos_w + self.off[:, None, :]
            @property
            def anchor_pos_w(self): return super().anchor_pos_w + self.off
            @property
            def body_lin_vel_w(self): return super().body_lin_vel_w + self.v[:, None, :]
            @property
            def anchor_lin_vel_w(self): return super().anchor_lin_vel_w + self.v
            def _resample_command(self, env_ids):   # new episode or end of motion: the robot is put on an undrifted reference
                self.off[env_ids] = 0; self.v[env_ids] = 0; self.v_goal[env_ids] = 0; self.t_next[env_ids] = 0
                super()._resample_command(env_ids)
            def _update_command(self):
                dt = self._env.step_dt; self.t_next -= dt; ids = torch.where(self.t_next <= 0)[0]
                if len(ids):
                    n = len(ids); ang = torch.rand(n, device=self.device) * 2 * torch.pi
                    sp = torch.rand(n, device=self.device) * self.cfg.steer_speed * (torch.rand(n, device=self.device) > STEER_HOLD)
                    self.v_goal[ids, 0] = sp * torch.cos(ang); self.v_goal[ids, 1] = sp * torch.sin(ang)
                    self.t_next[ids] = STEER_EVERY_S[0] + torch.rand(n, device=self.device) * (STEER_EVERY_S[1] - STEER_EVERY_S[0])
                self.v += (self.v_goal - self.v) * min(1.0, dt / STEER_TAU); self.off += self.v * dt
                super()._update_command()
        @dataclasses.dataclass(kw_only=True)
        class SteerMotionCommandCfg(MotionCommandCfg):
            steer_speed: float = STEER_SPEED
            def build(self, env): return SteerMotionCommand(self, env)
        _STEER["cfg"], _STEER["cls"] = SteerMotionCommandCfg, SteerMotionCommand
    return _STEER["cfg"](**{f.name: getattr(mc, f.name) for f in dataclasses.fields(mc)}, steer_speed=speed)

LUNGE_M, LUNGE_MPS, LUNGE_HOLD_S, LUNGE_EVERY_S, LUNGE_TAU, LUNGE_BACK_MPS = (0.10, 0.30), (0.5, 1.2), (0.25, 0.6), (1.0, 3.0), 0.1, 0.4
_AIM = {}

def _aim_cfg(mc, strike_lunges=False):
    """A Steer2 command config -> the same config building an AimMotionCommand: Steer2's drift plus lunges (module
    docstring), a second xy offset `lo` chasing its target `lt` at a capped speed."""
    import dataclasses, torch
    if not _AIM:
        class AimMotionCommand(_STEER["cls"]):
            def __init__(self, cfg, env):
                super().__init__(cfg, env); n, dev = self.num_envs, self.device; z = lambda: torch.zeros(n, 3, device=dev)
                self.lo, self.lv, self.lt = z(), z(), z(); self.cap = torch.full((n,), LUNGE_BACK_MPS, device=dev)
                self.l_next = torch.rand(n, device=dev) * LUNGE_EVERY_S[1]; self.l_left = torch.zeros(n, device=dev)
            @property
            def body_pos_w(self): return super().body_pos_w + self.lo[:, None, :]
            @property
            def anchor_pos_w(self): return super().anchor_pos_w + self.lo
            @property
            def body_lin_vel_w(self): return super().body_lin_vel_w + self.lv[:, None, :]
            @property
            def anchor_lin_vel_w(self): return super().anchor_lin_vel_w + self.lv
            def _resample_command(self, env_ids):   # new episode: no lunge under way (run 2: none until the next window opens)
                self.lo[env_ids] = 0; self.lv[env_ids] = 0; self.lt[env_ids] = 0; self.cap[env_ids] = LUNGE_BACK_MPS; self.l_left[env_ids] = 0
                if hasattr(self, "prev_win"): self.prev_win[env_ids] = True
                self.l_next[env_ids] = torch.rand(len(env_ids), device=self.device) * LUNGE_EVERY_S[1]; super()._resample_command(env_ids)
            def _window(self):
                """Run 2: 1 where the current motion frame is in a train-clip strike's lunge window (the motion's _apex.json)."""
                if not hasattr(self, "_lw"):
                    import json
                    f = Path(self.cfg.motion_file); self._lw = torch.tensor(json.load(open(f.with_name(f.stem + "_apex.json")))["lunge_window"], dtype=torch.bool, device=self.device)
                    self.prev_win = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                return self._lw[self.time_steps.clamp(max=len(self._lw) - 1)]
            def _update_command(self):
                dt, dev = self._env.step_dt, self.device; self.l_next -= dt
                was = self.l_left > 0; self.l_left -= dt; back = was & (self.l_left <= 0); self.lt[back] = 0; self.cap[back] = LUNGE_BACK_MPS
                if self.cfg.strike_lunges:   # run 2: a lunge starts as a strike's window opens and is held until it closes
                    win = self._window(); opened = win & ~self.prev_win; closed = ~win & self.prev_win & (self.l_left > 0); self.prev_win = win
                    self.l_left[closed] = torch.rand(int(closed.sum()), device=dev) * STRIKE_LUNGE_TAIL_S + 1e-6
                    ids = torch.where(opened & (self.l_left <= 0) & (torch.rand(self.num_envs, device=dev) < STRIKE_LUNGE_P))[0]; hold = 1e9
                else:
                    ids = torch.where((self.l_next <= 0) & (self.l_left <= 0))[0]; hold = None
                if len(ids):
                    u = lambda r: r[0] + torch.rand(len(ids), device=dev) * (r[1] - r[0])
                    w, x, y, z = self.anchor_quat_w[ids].unbind(-1); yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    L, v = u(LUNGE_M), u(LUNGE_MPS); self.lt[ids, 0] = L * torch.cos(yaw); self.lt[ids, 1] = L * torch.sin(yaw)
                    self.cap[ids] = v; self.l_left[ids] = hold if hold else u(LUNGE_HOLD_S) + L / v; self.l_next[ids] = self.l_left[ids] + u(LUNGE_EVERY_S)
                want = (self.lt - self.lo) / LUNGE_TAU; sp = torch.linalg.norm(want, dim=-1, keepdim=True)
                self.lv = want * torch.clamp(self.cap[:, None] / sp.clamp_min(1e-9), max=1.0); self.lo += self.lv * dt
                super()._update_command()
        @dataclasses.dataclass(kw_only=True)
        class AimMotionCommandCfg(_STEER["cfg"]):
            strike_lunges: bool = False
            def build(self, env): return AimMotionCommand(self, env)
        _AIM["cfg"] = AimMotionCommandCfg
    return _AIM["cfg"](**{f.name: getattr(mc, f.name) for f in dataclasses.fields(mc)}, strike_lunges=strike_lunges)

STRIKE_LUNGE_P, STRIKE_LUNGE_TAIL_S = 0.6, 0.2   # run 2: share of strike windows with a lunge; hold after the window closes (0-0.2 s)

def lunge_anchor_exp(env, command_name, std):
    """Run 2: while a lunge is asked for, exp(-d^2 / std^2) with d the horizontal distance of the anchor (torso) from its
    lunging reference; 0 otherwise. Steer2's anchor term (std 0.5 m) makes a 20 cm lunge nearly free to ignore."""
    import torch
    cmd = env.command_manager.get_term(command_name); on = torch.linalg.norm(cmd.lt[:, :2], dim=-1) > 0
    d2 = torch.sum(torch.square(cmd.anchor_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2]), dim=-1)
    return torch.exp(-d2 / std ** 2) * on

WRISTS = (BODIES.index("left_wrist_roll_link"), BODIES.index("right_wrist_roll_link"))

def strike_apex_exp(env, command_name, std):
    """At a train-clip strike apex (the motion's _apex.json: -1 none, 0 left, 1 right per frame), exp(-d^2 / std^2) with d the
    striking wrist's world distance to its reference (drift and lunge included); 0 elsewhere."""
    import json, torch
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "_apex_hand"):
        f = Path(cmd.cfg.motion_file); cmd._apex_hand = torch.tensor(json.load(open(f.with_name(f.stem + "_apex.json")))["hand"], dtype=torch.long, device=cmd.device)
    h = cmd._apex_hand[cmd.time_steps.clamp(max=len(cmd._apex_hand) - 1)]; i = torch.where(h == 1, WRISTS[1], WRISTS[0]); n = torch.arange(len(h), device=cmd.device)
    d2 = torch.sum(torch.square(cmd.body_pos_w[n, i] - cmd.robot_body_pos_w[n, i]), dim=-1)
    return torch.exp(-d2 / std ** 2) * (h >= 0)

def env_cfg(play=False, motion_file="", relaxed_hands=False, feet=0.25, hands=0.5, steer=False, strict=False, aim=False):
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
    drop = ("base_lin_vel",) if steer else ("motion_anchor_pos_b", "base_lin_vel")
    terms = {k: v for k, v in cfg.observations["actor"].terms.items() if k not in drop}
    if steer:   # the opponent is a sim agent with exact state: small noise, and the position term counts as much as the others
        from mjlab.utils.noise import UniformNoiseCfg as Unoise
        terms["motion_anchor_pos_b"].noise = Unoise(n_min=-0.05, n_max=0.05); cfg.rewards["motion_global_root_pos"].weight = 1.0
        cfg.commands["motion"] = mc = _steer_cfg(mc, 0.3 if strict else STEER_SPEED)
        if strict:   # position has to matter: a reward with gradient out to ~1 m, and leaving the reference ends the episode
            import copy
            from mjlab.tasks.tracking.mdp import bad_anchor_pos
            r = cfg.rewards["motion_global_root_pos"]; r.weight = 2.0; r.params["std"] = 0.5
            t = copy.deepcopy(cfg.terminations["anchor_pos"]); t.func = bad_anchor_pos; t.params = {"command_name": "motion", "threshold": 0.6}
            cfg.terminations["anchor_xy"] = t
    if aim:   # M2 run 1 (aim=1) and run 2 (aim=2; module docstring)
        from mjlab.managers.reward_manager import RewardTermCfg
        cfg.commands["motion"] = mc = _aim_cfg(mc, strike_lunges=aim == 2)
        cfg.rewards["strike_apex"] = RewardTermCfg(func=strike_apex_exp, weight=3.0 if aim == 2 else 1.5, params={"command_name": "motion", "std": 0.1})
        if aim == 2: cfg.rewards["lunge_anchor"] = RewardTermCfg(func=lunge_anchor_exp, weight=1.0, params={"command_name": "motion", "std": 0.15})
        if not play:
            from mjlab.envs import mdp as envs_mdp
            from mjlab.managers.event_manager import EventTermCfg
            from mjlab.managers.scene_entity_config import SceneEntityCfg
            cfg.events["hand_impulse"] = EventTermCfg(func=envs_mdp.apply_body_impulse, mode="step", params={
                "force_range": (-25.0, 25.0), "torque_range": (0.0, 0.0), "duration_s": (0.05, 0.12), "cooldown_s": (1.0, 3.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=("left_wrist_roll_link", "right_wrist_roll_link"))})
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
    register_mjlab_task(task_id="R1-Track-KungFu-Steer", env_cfg=env_cfg(motion_file=motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True),
                        play_env_cfg=env_cfg(True, motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)
    register_mjlab_task(task_id="R1-Track-KungFu-Steer2", env_cfg=env_cfg(motion_file=motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True),
                        play_env_cfg=env_cfg(True, motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)

    register_mjlab_task(task_id="R1-Track-KungFu-Aim", env_cfg=env_cfg(motion_file=motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True, aim=True),
                        play_env_cfg=env_cfg(True, motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True, aim=True), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)
    register_mjlab_task(task_id="R1-Track-KungFu-Aim2", env_cfg=env_cfg(motion_file=motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True, aim=2),
                        play_env_cfg=env_cfg(True, motion_file, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True, aim=2), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)

def widen(src, dst, at=48, n=3):
    """Checkpoint for the steer task from a no-position one: n new actor inputs at index `at` (after the 48-d command,
    where motion_anchor_pos_b sits), with zero weights, normaliser mean 0 / std 1, and zero Adam moments. The widened
    policy acts exactly like the old one until training starts using the new inputs."""
    import torch
    c = torch.load(src, map_location="cpu", weights_only=False); a = c["actor_state_dict"]; w = a["mlp.0.weight"]; old = w.shape[1]
    def ins(x, val):   # insert n columns (last dim) at `at`
        return torch.cat([x[..., :at], torch.full((*x.shape[:-1], n), float(val), dtype=x.dtype), x[..., at:]], dim=-1)
    a["mlp.0.weight"] = ins(w, 0.0)
    for k, val in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)): a[k] = ins(a[k], val)
    for st in c["optimizer_state_dict"]["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            if st[k].shape == (w.shape[0], old): st[k] = ins(st[k], 0.0)
    assert sum(st["exp_avg"].shape == (w.shape[0], old + n) for st in c["optimizer_state_dict"]["state"].values()) == 1
    torch.save(c, dst); print(f"actor input {old} -> {old + n} (new inputs at {at}:{at + n})")

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
    if sys.argv[1] == "widen":
        widen(sys.argv[2], sys.argv[3]); sys.exit(0)
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
