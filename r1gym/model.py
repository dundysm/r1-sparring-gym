"""Build the R1 sparring world (R1 + optional kinematic opponent: R1 by default, G1 via opponent_model) and the PD/torque-cap layer."""
from __future__ import annotations
import re, numpy as np, mujoco, yaml
from . import ASSETS, CONFIGS

# 24 actuated joints in Unitree's deploy order (unitree_rl_mjlab deploy.yaml joint_ids_map)
CTRL_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]
HOME = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0.35, -0.18, 0, 0.87, 0])
# unitree_rl_mjlab r1_constants.py effort_limit, by regex
MJLAB_CAPS = [(r"_hip_|_knee", 60.0), (r"_ankle_", 50.0), (r"waist_", 60.0), (r"shoulder_(pitch|roll)", 60.0), (r"shoulder_yaw|elbow|wrist_roll", 33.0)]
HIT_BODIES = ["pelvis", "torso_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_knee_link", "right_knee_link"]

def load_cfg(path=None):
    return yaml.safe_load(open(path or CONFIGS / "r1_assumptions.yaml"))

def _group(j):
    if "ankle" in j: return "ankle"
    if re.search(r"hip|knee", j): return "leg"
    if "waist" in j: return "waist"
    if re.search(r"shoulder_(pitch|roll)", j): return "shoulder"
    return "arm"

def build_spec(opponent: bool = False, cfg=None, gains: str = "stand_pd") -> mujoco.MjSpec:
    """R1 from unitree_rl_mjlab (the MJCF Unitree trains its own R1 policies on) + PD servos + floor [+ G1 opponent]."""
    cfg = cfg or load_cfg()
    s = mujoco.MjSpec.from_file(str(ASSETS / "r1" / "r1.xml"))
    s.meshdir = str(ASSETS / "r1" / "assets")
    # upstream r1.xml (unitree_rl_mjlab 1425b15) excludes contacts with *_wrist_pitch_link, a body that does not exist
    # in that file -> plain MuJoCo refuses to compile. Drop dangling excludes (see docs/MODEL_AUDIT.md).
    names = {b.name for b in s.bodies}
    for e in list(s.excludes):
        if e.bodyname1 not in names or e.bodyname2 not in names: s.delete(e)
    s.option.timestep = cfg["model"]["timestep"]
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # PD damping implicit: explicit Euler is unstable for kd>~4 on light links
    s.worldbody.add_light(pos=[1, 0, 3.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    s.worldbody.add_light(pos=[-2, -2, 3], dir=[1, 1, -1.2], castshadow=False)
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.22, 0.26, 0.3, 1])
    s.visual.global_.offwidth = 1280; s.visual.global_.offheight = 720
    # collision setup = unitree_rl_mjlab FULL_COLLISION_WITHOUT_SELF (feet condim 3, mu 0.6)
    for g in s.geoms:
        if g.name.endswith("_collision"):
            g.contype, g.conaffinity = 0, 1
            if "foot" in g.name: g.condim, g.priority, g.friction = 3, 1, [cfg.get("foot_mu", 1.0), 0.005, 0.0001]
            else: g.condim = 1
    caps = torque_caps(cfg)
    for i, j in enumerate(CTRL_JOINTS):
        grp = cfg[gains][_group(j)]; s.joint(j).armature = cfg.get("armature", 0.01)
        s.add_actuator(name=j.replace("_joint", ""), target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                       gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[grp["kp"]] + [0] * 9,
                       biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -grp["kp"], -grp["kd"]] + [0] * 7,
                       forcelimited=1, forcerange=[-caps[i], caps[i]])
    if opponent:
        kind = opponent if isinstance(opponent, str) else cfg.get("opponent_model", "r1")
        g = opponent_spec(kind)
        for a in list(g.actuators): g.delete(a)          # opponent is kinematic (fight-motion replay), no motors
        for se in list(g.sensors): g.delete(se)
        for ge in g.geoms:                               # opponent hits the R1 but not itself
            ge.contype, ge.conaffinity = (1, 0) if ge.contype or ge.conaffinity else (0, 0)
            ge.rgba = [0.85, 0.35, 0.25, 1]
        f = s.worldbody.add_frame(); f.attach_body(g.body("pelvis"), "opp_", "")
    return s

# Torque caps: unitree_mujoco ctrlrange (R1_C++.xml, commit 1eb6642) vs unitree_rl_mjlab effort_limit. See docs/MODEL_AUDIT.md
MUJOCO_CAPS = [(r"_hip_", 88.0), (r"_knee", 139.0), (r"_ankle_", 50.0), (r"waist_roll", 50.0), (r"waist_yaw", 88.0), (r"shoulder|elbow|wrist", 25.0)]

def torque_caps(cfg):
    mj = np.array([next(c for r, c in MUJOCO_CAPS if re.search(r, j)) for j in CTRL_JOINTS])
    lab = np.array([next(c for r, c in MJLAB_CAPS if re.search(r, j)) for j in CTRL_JOINTS])
    return {"min": np.minimum(mj, lab), "mujoco": mj, "mjlab": lab}[cfg["torque_cap_policy"]]

class R1Robot:
    """Index maps + PD torque law with explicit caps for the R1 inside a compiled model."""
    def __init__(self, m: mujoco.MjModel, cfg=None):
        cfg = cfg or load_cfg(); self.cfg = cfg
        jid = lambda n: m.joint(n).id
        self.qadr = np.array([m.jnt_qposadr[jid(j)] for j in CTRL_JOINTS]); self.vadr = np.array([m.jnt_dofadr[jid(j)] for j in CTRL_JOINTS])
        self.act = np.array([m.actuator(j.replace("_joint", "")).id for j in CTRL_JOINTS])
        self.caps = torque_caps(cfg)
        self.pel = m.body("pelvis").id; self.torso = m.body("torso_link").id
        self.free_q = m.jnt_qposadr[jid("floating_base_joint")]; self.free_v = m.jnt_dofadr[jid("floating_base_joint")]
        self.hit_bodies = {b: m.body(b).id for b in HIT_BODIES}
        self.r1_bodies = set(range(1, m.nbody)) - {b for b in range(m.nbody) if m.body(b).name.startswith("opp_")}

    def reset(self, m, d, height=0.74):
        mujoco.mj_resetData(m, d)
        d.qpos[self.free_q:self.free_q + 3] = [0, 0, height]; d.qpos[self.free_q + 3:self.free_q + 7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = HOME; mujoco.mj_forward(m, d)

    def pd(self, d, target):
        """Set joint position targets (rad) for the 24 actuated joints."""
        d.ctrl[self.act] = target

    def torques(self, d):
        return d.actuator_force[self.act].copy()

    def fallen(self, d):
        z = d.xpos[self.pel][2]; up = d.xmat[self.pel].reshape(3, 3)[2, 2]
        return z < self.cfg["fall"]["min_pelvis_z"] or up < np.cos(np.deg2rad(self.cfg["fall"]["max_tilt_deg"]))


def opponent_spec(kind: str = "r1") -> mujoco.MjSpec:
    """Bare opponent body tree: 'r1' (default, the same official R1 MJCF) or 'g1' (original KungFuAthleteBot embodiment)."""
    if kind == "g1":
        g = mujoco.MjSpec.from_file(str(ASSETS / "g1" / "g1_29dof.xml")); g.meshdir = str(ASSETS / "g1" / "meshes"); return g
    g = mujoco.MjSpec.from_file(str(ASSETS / "r1" / "r1.xml")); g.meshdir = str(ASSETS / "r1" / "assets")
    names = {b.name for b in g.bodies}
    for e in list(g.excludes):
        if e.bodyname1 not in names or e.bodyname2 not in names: g.delete(e)
    return g
