"""Sparring M1-M2 on GPU (docs/SPARRING_M1M2.md): an attacker in the stepping-base scene, and a defender that sees its fists.

M1, R1-Spar-Attack = R1-Spar-Footwork-Guard (r1_spar_task.py) + a two-fist attacker + the defender's 16 collision proxies + two
contact sensors + the `attack` command + a `nan` termination; actor, critic and rewards unchanged. Each fist is a 5-joint chain
(slide x/y/z, hinge yaw/pitch) on force-capped position servos ending in a forearm+fist capsule named opp_*_wrist_roll_link:
a weld has no force limit and gave 18-60 N*s per head hit in the architect's probe, where the CPU aimed opponent lands 2.37.
The fists replay the 37 train-split hand strikes (export_attacks.py -> motions/attacks_train.npz) around the defender, each
aimed at its head (p 0.8) or torso, the aim latched at the strike's onset so a slip after onset makes it miss (AttackSchedule).
They touch only the proxies (bit 4), which touch nothing else, so the robot's own contacts and rewards are unchanged.
M2, R1-Spar-Block-{A,B} add to it: the 28-d opponent-keypoint block on the actor (83 -> 111; 1.3 of the doc: five keypoints in
the defender's torso frame, glove velocities, relative yaw, head-camera FOV masks; training noise and dropout inside the term),
the clean block plus the strike phase on the critic (98 -> 136), half the envs on the CPU's face/hold outer loops, looser arm
and waist-roll posture, and the sparring rewards (A, B: x2). -Blind / -Blind-B: the same with the actor's block zeroed.
Hits are counted on GPU by HitAccount, which mirrors r1gym/hits.py's rules on the proxies' per-substep contact forces.

Pure parts (no mjlab import; CPU tests and probes use them): the MjSpec builders attacker_spec / add_shell_proxies /
proxy_bits / add_contact_sensors_cpu, and torch AttackSchedule, HitAccount, keypoint_block, in_view_t, joint_targets. The
mjlab glue lives in _classes() and the cfg builders, importing mjlab only inside functions, like r1_spar_task.py.

Subcommands (python sparring_task.py ...):
  widen <src.pt> <dst.pt>          warm start for R1-Spar-Block-*: zero-weight columns for the new inputs (3.1)
  calibrate [--bouts 64 --secs 20 --recheck 1]  CPU replica in r1gym (R1HitEnv + the frozen warm start ONNX), HitMeter vs HitAccount (2.5)
  calcheck [m1_calibration.json]   the preregistered 'CPU calibration passes first', for ATTACK as set here (pod, before m1check)
  m1check --checkpoint model_7496.pt [--num-envs 4096 --steps 1000 --out m1_gate.json --calibration m1_calibration.json]   gate 1 (2.6)
  gate --m1 m1_gate.json --smoke-log smoke.log [--proxy mesh --num-envs 4096]               gate 2 from the smoke log
  <task id> [train.py flags]       training (unitree_rl_mjlab's train.main), from a unitree_rl_mjlab checkout
Environment: SPAR_SHELL_PROXY=capsule (2.6's speed fallback), SPAR_GRAVCOMP=0 (feed-forward instead of body gravcomp),
SPAR_CALIB_DEVIATION="<reason>" (train although the calibration did not choose ATTACK: recorded as a deviation)."""
import os, sys, json, math, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LIB_PATH, SHELLS_PATH = HERE / "motions" / "attacks_train.npz", HERE / "motions" / "attacks_shells.npz"

# ---- 1.1-1.3 the shared contract (identical in r1gym/keypoints.py; change docs/SPARRING_M1M2.md first)
KP_NAMES = ("l_glove", "r_glove", "l_elbow", "r_elbow", "head")
KP_DIM, N_KP, HEAD = 28, 5, 4
POS, VEL, YAW, MASK = slice(0, 15), slice(15, 21), slice(21, 23), slice(23, 28)
CAM_C = (0.1310, 0.0, 0.2473)                                                       # lens in T [m]
CAM_R = ((0.0, -0.7660, 0.6428), (-1.0, 0.0, 0.0), (0.0, -0.6428, -0.7660))          # columns x_c, y_c, z_c (OpenCV) in T
CAM_F, CAM_CX, CAM_CY, CAM_ZMIN = 50.346, 80.0, 60.0, 0.05
CTRL_DT, SUB_DT, DECIM = 0.02, 0.005, 4
ELBOW_TO_FIST, TIP = 0.195, 0.075               # elbow keypoint = fist body origin - 0.195 x; fist tip = origin + 0.075 x
DEF_HEAD_T = (0.01, 0.0, 0.357)                 # the defender's head point in T (clearance reward)
CORRUPT = dict(pos=0.02, bias=0.02, vel=0.25, yaw=0.1, drop=0.10)   # 3.2 training corruption of the actor's block

# ---- 2.1 collision proxies, zone order (= export_attacks.py SHELLS); the sensors' primaries
ZONES = ("head", "torso", "pelvis", "block", "upper_arm")
SHELLS = (("head", ("head_pitch_link", "head_yaw_link")), ("torso", ("waist_yaw_link",)),
          ("pelvis", ("pelvis_link", "imu_in_pelvis_link", "waist_roll_link")),
          ("block", ("left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link")),
          ("upper_arm", tuple(f"{s}_shoulder_{j}_link" for s in ("left", "right") for j in ("pitch", "roll", "yaw"))))
LINKS = tuple(l for _, ls in SHELLS for l in ls)
ZONE_OF_LINK = {l: z for z, ls in SHELLS for l in ls}
SHELL_BIT, SHELL_SOLREF, SHELL_SOLIMP = 4, (0.01, 1.0), (0.95, 0.99, 0.001, 0.5, 2.0)   # r1gym/model.py SHELL_*
SIDES = (("l", "left"), ("r", "right"))
FISTS = tuple(f"opp_{nm}_wrist_roll_link" for _, nm in SIDES)
ATT_JOINTS = tuple(f"att_{s}_{j}" for s, _ in SIDES for j in ("x", "y", "z", "yaw", "pitch"))   # joint_targets' column order

# ---- 2.1 / 2.5 attacker. Calibrated on CPU (python sparring_task.py calibrate: the first passing row of CALIB_GRID wins; set
# mass / fmax / delta here to it before the pod). 2026-10-28, final schedule, 64 bouts x 20 s: the first four rows fail on
# impulse only (HitMeter median 4.42 / 4.04 / 3.59 / 3.65 N*s > 3.56; the 1.0 kg row also 3.95 on seeds 64-127); (1.5, 80, -2 cm)
# passes: median 3.52 N*s, head clean 11.0 / min, survival 0.88 (results/sparring/m1_calibration.json, section 8 item 28).
ATTACK = dict(mass=1.5, fmax=80.0, delta=-0.02, kp=1500.0, rot_kp=60.0, rot_kv=3.0, rot_fmax=30.0, rot_armature=0.01, radius=0.045,
              link_mass=0.01, link_inertia=1e-5, capsule=(-0.195, 0.03))
def slide_kv(mass, kp=ATTACK["kp"]): return 2.0 * math.sqrt(kp * (mass + 2 * ATTACK["link_mass"]))   # critical on the z slide's load (95.5 at 1.5 kg)
CALIB_GRID = ((1.5, 150.0, 0.005), (1.5, 150.0, -0.02), (1.5, 80.0, 0.005), (1.0, 150.0, 0.005), (1.5, 80.0, -0.02))
CALIB_PASS = dict(impulse_Ns=(1.58, 3.56), head_clean_pm=7.7, speed_ratio=0.85, agree_rate=0.15, agree_impulse=0.10)

# ---- 2.4 schedule (times in s, frames at the library's 50 fps). D, alpha_deg and the per-episode head_dz (the virtual head's
# height), beta_deg (bearing offset) and base_* (the whole opponent's frame offset: forward, left, up; 0 here) are set so the
# keypoints' in-view rates come within 0.15 of the CPU eval's (doc 8: the spec's D 0.35-0.55 and alpha 30 deg saw the head 0.87 of
# the time against 0.41 on CPU). Raising the whole base instead (base_dz 0-0.12) also matched, but lifted the calibrated median
# impulse per head hit from 3.50 to 4.07 N*s; the head is virtual, so head_dz leaves the fists' physics alone.
SCHED = dict(D=(0.35, 0.65), lam=(0.5, 2.0), first=(0.5, 1.5), alpha_deg=50.0, alpha_every=(2.0, 5.0), alpha_rate=0.5, spot_tau=0.5,
             spot_vmax=0.3, base_dx=(0.0, 0.0), base_dy=(0.0, 0.0), base_dz=(0.0, 0.0), head_dz=(0.0, 0.12), beta_deg=(-30.0, 0.0),
             sigma=(0.7, 1.3), p_head=0.8, min_gap=0.25, aim_head=(0.087, 0.0, 0.36), aim_torso=(0.070, 0.0, 0.15), delta=ATTACK["delta"],
             max_delta=0.5, idle_blend_s=0.2, lead=7.5, hold=2.5, tail=5.0, win_post=0.15)
# ---- 2.5 hit accounting (r1gym/hits.py DEFAULTS)
HIT = dict(dt=SUB_DT, gap=0.1, min_impulse=0.5, min_peak=5.0, v_close=1.0, max_dur=0.15, ring=4096)
T_TOL = 1e-4   # duration / gap slack: the float32 episode clock is off by ~1e-6 s at 20 s, a substep is 5e-3 s (HitMeter: float64)
W_TOL = 5e-5   # window-edge slack: HitMeter compares times rounded to 4 decimals (substep times are exact at 4 dp, window edges are not)
EP_KEYS = ("head_clean", "torso_clean", "head_landed", "torso_landed", "head_imp_landed", "windows", "blocked", "block_first", "head_landed_vneg")
# ---- 3.2 rewards per step (mjlab multiplies by step_dt) and 3.1 widening
WEIGHTS = {"A": dict(spar_head_impulse=-20.0, spar_head_clean=-40.0, spar_torso_impulse=-5.0, spar_block=20.0, spar_clearance=0.5)}
WEIGHTS["B"] = {k: 2.0 * v for k, v in WEIGHTS["A"].items()}
POSTURE_ARMS = (r".*shoulder_pitch.*", r".*shoulder_roll.*", r".*shoulder_yaw.*", r".*elbow.*", r".*wrist.*")
STD_STANDING = {**{k: 0.05 for k in (r".*hip_pitch.*", r".*hip_roll.*", r".*hip_yaw.*", r".*knee.*", r".*ankle_pitch.*", r".*ankle_roll.*",
                                     r".*waist_yaw.*")}, r".*waist_roll.*": 0.15, **{k: 0.35 for k in POSTURE_ARMS}}
ACTOR_DIM, CRITIC_DIM, PHASE_DIM = 83, 98, 10

# =============================================================================================== pure MjSpec builders
def load_shells(path=SHELLS_PATH):
    z = np.load(path); assert tuple(z["links"].tolist()) == LINKS, "attacks_shells.npz link order differs from SHELLS"
    return {k: z[k] for k in z.files}

def load_library(path=LIB_PATH):
    z = np.load(path); return {k: z[k] for k in z.files}

def attacker_spec(mass=ATTACK["mass"], fmax=ATTACK["fmax"], actuators=False, gravcomp=True, spec=None):
    """The attacker as its own MjSpec: under one root body att_root (no joint; mjlab wraps a fixed-base entity in a mocap body
    left at the world origin, so slide values are world coordinates), per side s: att_s_x (slide x) -> att_s_y (slide y) ->
    att_s_z (slide z) -> att_s_yaw (hinge z) -> body opp_{left,right}_wrist_roll_link (hinge y, joint att_s_pitch) carrying the
    forearm+fist capsule (origin = the glove keypoint, tip 0.075 m beyond). actuators=True adds the position servos mjlab's
    BuiltinPositionActuatorCfg would (the CPU replica); for mjlab they come from attacker_entity_cfg's articulation. spec: add
    the attacker into that spec's worldbody instead of a new one (no attach, so names stay unprefixed: HitMeter needs opp_)."""
    import mujoco
    A = ATTACK; s = spec if spec is not None else mujoco.MjSpec(); root = s.worldbody.add_body(name="att_root")   # spec: build into it (CPU replica)
    chain = (("x", mujoco.mjtJoint.mjJNT_SLIDE, (1, 0, 0)), ("y", mujoco.mjtJoint.mjJNT_SLIDE, (0, 1, 0)), ("z", mujoco.mjtJoint.mjJNT_SLIDE, (0, 0, 1)),
             ("yaw", mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1)), ("pitch", mujoco.mjtJoint.mjJNT_HINGE, (0, 1, 0)))
    for side, nm in SIDES:
        b = root
        for jn, jt, ax in chain:
            last = jn == "pitch"
            b = b.add_body(name=f"opp_{nm}_wrist_roll_link" if last else f"att_{side}_{jn}", gravcomp=1.0 if gravcomp else 0.0)
            b.add_joint(name=f"att_{side}_{jn}", type=jt, axis=list(ax), damping=0.0, frictionloss=0.0,
                        armature=A["rot_armature"] if (actuators and jt == mujoco.mjtJoint.mjJNT_HINGE) else 0.0)
            if not last:
                b.mass = A["link_mass"]; b.inertia = [A["link_inertia"]] * 3; b.explicitinertial = True
            else:
                g = b.add_geom(name=f"att_{side}_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[A["capsule"][0], 0, 0, A["capsule"][1], 0, 0],
                               size=[A["radius"], 0, 0], mass=mass, contype=SHELL_BIT, conaffinity=0, condim=1, rgba=[0.2, 0.8, 0.2, 1])
                g.solref = list(SHELL_SOLREF); g.solimp = list(SHELL_SOLIMP)
    if actuators:
        kv = slide_kv(mass)
        for side, _ in SIDES:
            for jn in ("x", "y", "z", "yaw", "pitch"):
                k, c, f = (A["kp"], kv, fmax) if jn in "xyz" else (A["rot_kp"], A["rot_kv"], A["rot_fmax"])
                s.add_actuator(name=f"att_{side}_{jn}", target=f"att_{side}_{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                               gainprm=[k] + [0] * 9, biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -k, -c] + [0] * 7, forcelimited=1, forcerange=[-f, f])
    return s

def add_shell_proxies(spec, prefix="", shells=None, proxy="mesh", rename_full=False):
    """The defender's 16 collision proxies (2.1) on its bodies: one geom <link>_shell per link (density 0, group 3, collision
    bits 0 here: proxy_bits sets them), a 64-vertex mesh from attacks_shells.npz (MuJoCo collides its hull) or, proxy="capsule",
    the PCA capsule. Points are in the body frame, so the geom sits at the body origin. rename_full (CPU replica only): the
    visual mesh <link> becomes <link>_full and the proxy mesh takes the name <link>, so r1gym's HitMeter, which zones the head
    by mesh name, labels the proxies as it labels the eval's shells."""
    import mujoco
    sh = shells or load_shells()
    for i, link in enumerate(LINKS):
        body = spec.body(prefix + str(sh["bodies"][i]))
        if proxy == "mesh":
            mname = prefix + link + ("" if rename_full else "_shell")
            if rename_full:
                old = spec.mesh(prefix + link); old.name = prefix + link + "_full"
                for g in spec.geoms:
                    if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == prefix + link: g.meshname = prefix + link + "_full"
            spec.add_mesh(name=mname, uservert=sh["points"][i].astype(float).ravel().tolist())
            body.add_geom(name=prefix + link + "_shell", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, density=0.0, group=3, contype=0, conaffinity=0)
        elif proxy == "capsule":
            c = sh["capsule"][i].astype(float)
            body.add_geom(name=prefix + link + "_shell", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=c[:6].tolist(), size=[float(c[6]), 0, 0],
                          density=0.0, group=3, contype=0, conaffinity=0)
        else: raise ValueError(f"shell proxy {proxy!r}: mesh or capsule")
    return spec

def proxy_bits(spec, prefix="robot/"):
    """Collision bits of the proxies (SceneCfg.spec_fn in mjlab: the robot's CollisionCfg zeroes any extra geom's bits): touched
    by the fists only (contype 0, conaffinity 4), condim 1, r1gym's shell contact."""
    names = {prefix + l + "_shell" for l in LINKS}; n = 0
    for g in spec.geoms:
        if g.name in names:
            g.contype, g.conaffinity, g.condim = 0, SHELL_BIT, 1; g.solref = list(SHELL_SOLREF); g.solimp = list(SHELL_SOLIMP); n += 1
    assert n == len(LINKS), f"proxy_bits: {n} of {len(LINKS)} proxies found under prefix {prefix!r}"
    return spec

def add_contact_sensors_cpu(spec, robot_prefix="", att_prefix=""):
    """The two hit sensors as mjlab's ContactSensorCfg(fields=("found", "force"), reduce="netforce", num_slots=1) writes them:
    one mjSENS_CONTACT per (primary proxy, field), named hit_<s>_<link>_shell_<field>, secondary = that fist's body."""
    import mujoco
    for s, nm in SIDES:
        for l in LINKS:
            for field, bit in (("found", 0), ("force", 1)):
                spec.add_sensor(name=f"hit_{s}_{l}_shell_{field}", type=mujoco.mjtSensor.mjSENS_CONTACT, objtype=mujoco.mjtObj.mjOBJ_GEOM,
                                objname=robot_prefix + l + "_shell", reftype=mujoco.mjtObj.mjOBJ_BODY, refname=att_prefix + f"opp_{nm}_wrist_roll_link",
                                intprm=[1 << bit, 3, 1])
    return spec

# =============================================================================================== torch: keypoints (1.1-1.3)
_TC = {}
def _t(x, like):
    """x as a tensor of like's dtype / device; a tuple (a constant) is built once per (value, dtype, device) and reused, so the
    per-step terms make no small host-to-device copies. Callers never write into it."""
    import torch
    if not isinstance(x, tuple): return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    k = (x, like.dtype, like.device)
    if k not in _TC: _TC[k] = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return _TC[k]

def wrap_t(a):
    import torch; return torch.remainder(a + math.pi, 2 * math.pi) - math.pi

def smoothstep_t(x):
    x = x.clamp(0.0, 1.0); return x * x * (3.0 - 2.0 * x)

def quat_to_mat(q):
    """(..., 4) wxyz -> (..., 3, 3) rotation (columns = the body's axes in world)."""
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)

def heading_t(R):
    import torch; return torch.atan2(R[..., 1, 0], R[..., 0, 0])

def to_torso_t(P_w, x_T, R_T):
    """World points (N, K, 3) -> T, given the torso origin (N, 3) and rotation (N, 3, 3)."""
    return (P_w - x_T[:, None]) @ R_T

def in_view_t(P_T):
    """1.1: True where a point in T is inside the nominal head camera's 160 x 120 pinhole (FOV only)."""
    q = (P_T - _t(CAM_C, P_T)) @ _t(CAM_R, P_T); z = q[..., 2]; zs = z.clamp(min=1e-9)
    return (z > CAM_ZMIN) & ((q[..., 0] / zs).abs() <= CAM_CX / CAM_F) & ((q[..., 1] / zs).abs() <= CAM_CY / CAM_F)

def pack_t(P, V, rel_yaw, mask):
    """The 28-d block from positions (N, 5, 3), glove velocities (N, 2, 3), relative yaw (N) and masks (N, 5): zero-plus-mask
    (absent entries exactly +0.0; the yaw pair is present when any keypoint is, 8 of the doc). Same as r1gym/keypoints.py pack."""
    import torch
    on = mask > 0.5 if mask.dtype != torch.bool else mask; z = torch.zeros((), dtype=P.dtype, device=P.device)
    P = torch.where(on[..., :, None], P, z); V = torch.where(on[..., :2, None], V, z)
    sc = torch.where(on.any(-1, keepdim=True), torch.stack([torch.sin(rel_yaw), torch.cos(rel_yaw)], -1), z)
    return torch.cat([P.reshape(*P.shape[:-2], 15), V.reshape(*V.shape[:-2], 6), sc, on.to(P.dtype)], -1).float()

def keypoint_block(P_T, P_T_prev, rel_yaw, mask):
    """1.3 without corruption: true positions in T (N, 5, 3), the previous control step's (None at an episode's first step:
    zero velocity), wrap(yaw_opp - yaw_def_torso) (N) and masks (N, 5) -> (N, 28). Equals r1gym/keypoints.py assemble."""
    import torch
    V = torch.zeros_like(P_T[..., :2, :]) if P_T_prev is None else (P_T[..., :2, :] - P_T_prev[..., :2, :]) / CTRL_DT
    return pack_t(P_T, V, rel_yaw, mask)

class KeypointHistory:
    """The glove velocities' bookkeeping (1.3), as r1gym/keypoints.py KeypointSource keeps it: per env the previous control
    step's T-frame block (N, 5, 3), so a velocity is the difference of two T-frame blocks, each in its own step's T (never
    world positions re-expressed in the current T, which would add the defender's own rotation). step(P_T) returns
    (P_T, previous block); an env flagged by reset(ids) gets its current block as the previous one (velocity 0), as the CPU's
    P_T_prev = None at reset. Pure torch: a CPU test drives it with a recorded P_T sequence and reset flags."""
    def __init__(self, n, device="cpu"):
        import torch
        self.prev = torch.zeros(n, N_KP, 3, device=device); self.first = torch.ones(n, dtype=torch.bool, device=device)
    def reset(self, ids): self.first[ids] = True
    def step(self, P_T):
        import torch
        prev = torch.where(self.first[:, None, None], P_T, self.prev); self.prev = P_T.clone(); self.first[:] = False
        return P_T, prev

def joint_targets(glove, elbow, cur=None):
    """Glove and elbow world targets (N, 2, 3) -> attacker joint targets (N, 10) in ATT_JOINTS order: slides = glove; the fist
    axis u = glove - elbow gives yaw = atan2(u_y, u_x) (unwrapped to `cur`, the current joint values, when given) and
    pitch = -atan2(u_z, |u_xy|)."""
    import torch
    u = glove - elbow; yaw = torch.atan2(u[..., 1], u[..., 0]); pitch = -torch.atan2(u[..., 2], torch.linalg.norm(u[..., :2], dim=-1))
    if cur is not None: cy = cur[:, [3, 8]]; yaw = cy + wrap_t(yaw - cy)
    return torch.cat([glove, yaw[..., None], pitch[..., None]], -1).reshape(glove.shape[0], 10)

# =============================================================================================== torch: the schedule (2.4)
class AttackSchedule:
    """Batched strike schedule (2.4). Per episode a stand-off D, a strike rate lam, a first strike time, a bearing offset beta,
    a frame offset koff (SCHED base_*) and the virtual head's height offset hdz; a live frame: the bout axis alpha (re-aimed
    within +-alpha_deg of the defender's yaw + beta every 2-5 s, followed at <= 0.5 rad/s), the spot S chasing pelvis + D (cos,
    sin)(alpha) with a 0.5 s lag capped at 0.3 m/s, facing f toward the pelvis; keypoints k sit at World(k + koff). A strike (one
    of its library segments, speed sigma, head p 0.8 else torso) replays the segment in its frame frozen at the start, guard-
    blended so it starts and ends at that hand's guard, bent by w(tau) * Delta onto a point 5 mm inside the aimed zone; Delta
    follows the defender until the strike's onset, then is world-fixed. Ended strikes blend back to the live guard in 0.2 s.
    step() returns the glove and elbow world targets (N, 2, 3), the virtual opponent head (N, 3) and yaw (N). Episode times t
    are per env (s since reset); random draws use `gen` (a torch.Generator) when given. No mjlab import."""
    def __init__(self, lib, n, device="cpu", dt=CTRL_DT, gen=None, p=None):
        import torch
        self.p = {**SCHED, **(p or {})}; self.n, self.dev, self.dt, self.gen = n, device, dt, gen
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.kp, self.yk, self.G, self.fps = f(lib["kp"]), f(lib["yaw"]), f(lib["guard"]), float(lib["fps"])
        self.seg = torch.as_tensor(np.asarray(lib["seg"]), dtype=torch.long, device=device); st, nn, hand = self.seg[:, 0], self.seg[:, 1], self.seg[:, 5]
        self.Gyaw = torch.stack([torch.quantile(torch.cat([self.yk[st[hand == h]], self.yk[(st + nn - 1)[hand == h]]]), 0.5) for h in (0, 1)])
        ids = [torch.nonzero(hand == h).flatten() for h in (0, 1)]; M = max(len(i) for i in ids)
        self.hseg = torch.stack([torch.cat([i, i[:1].repeat(M - len(i))]) for i in ids]); self.hcnt = torch.tensor([len(i) for i in ids], device=device)
        self.Ghead = self.G[:, HEAD].mean(0); self.ar = torch.arange(2, device=device); self.rows = torch.arange(n, device=device)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.D, self.lam, self.t_next, self.alpha, self.alpha_tgt, self.t_alpha = z(n), z(n), z(n), z(n), z(n), z(n)
        self.S, self.f = z(n, 2), z(n, 2); self.f[:, 0] = -1.0
        self.phase, self.seg_id, self.zone, self.strike_id = z(n, 2, dt=torch.long), z(n, 2, dt=torch.long), z(n, 2, dt=torch.long), z(n, 2, dt=torch.long)
        self.t0, self.sig, self.t_ret, self.tau, self.u = z(n, 2), torch.ones(n, 2, device=device), z(n, 2), z(n, 2), z(n, 2)
        self.win_on, self.win_off, self.t_apex = z(n, 2), z(n, 2), z(n, 2)
        self.S0, self.f0, self.delta, self.latched = z(n, 2, 2), z(n, 2, 2), z(n, 2, 3), z(n, 2, dt=torch.bool)
        self.glove, self.elbow, self.head, self.yaw = z(n, 2, 3), z(n, 2, 3), z(n, 3), z(n)
        self.n_strikes = z(n); self.koff, self.beta, self.hdz = z(n, 3), z(n), z(n)

    def _u(self, *shape, lo=0.0, hi=1.0):
        import torch; return lo + (hi - lo) * torch.rand(*shape, device=self.dev, generator=self.gen)

    @staticmethod
    def world(S, f, k):
        """Frame (spot S (..., 2), facing f (..., 2)) -> world: S + k_x f + k_y l + k_z z, l = z x f; the floor is z = 0."""
        import torch
        x = S[..., 0] + k[..., 0] * f[..., 0] - k[..., 1] * f[..., 1]; y = S[..., 1] + k[..., 0] * f[..., 1] + k[..., 1] * f[..., 0]
        return torch.stack([x, y, k[..., 2] + torch.zeros_like(x)], -1)

    def _hz(self, ids=None):
        import torch; h = self.hdz if ids is None else self.hdz[ids]; return torch.stack([torch.zeros_like(h), torch.zeros_like(h), h], -1)

    def _W(self, S, f, k, ids=None):
        """world() with this episode's frame offset koff (forward, left, up in the opponent's frame) added to the keypoints."""
        o = self.koff if ids is None else self.koff[ids]; o = o.reshape(o.shape[0], *([1] * (max(k.dim(), S.dim()) - 2)), 3)
        return self.world(S, f, k + o)

    def _kprime(self, st, nn, tau):
        """Guard-blended keypoints K'(tau) (..., 5, 3), yaw'(tau) and the blend u, for segments (start row, length) at frame tau."""
        import torch
        last = (nn - 1).clamp(min=1); tau = torch.minimum(tau.clamp(min=0.0), last.float())
        i0 = torch.minimum(tau.floor().long(), nn - 1); i1 = torch.minimum(i0 + 1, nn - 1); w = tau - i0.float()
        K = self.kp[st + i0] * (1 - w)[..., None, None] + self.kp[st + i1] * w[..., None, None]
        y = self.yk[st + i0] * (1 - w) + self.yk[st + i1] * w
        u = smoothstep_t(tau / last.float()); G = self.G   # (2, 5, 3): broadcasts over envs, per hand
        Kp = K + (1 - u)[..., None, None] * (G - self.kp[st]) + u[..., None, None] * (G - self.kp[st + nn - 1])
        yp = y + (1 - u) * (self.Gyaw - self.yk[st]) + u * (self.Gyaw - self.yk[st + nn - 1])
        return Kp, yp, u

    def aim_weight(self, tau, on, ap, en):
        """w(tau): smoothstep up from onset - lead to the apex, 1 until apex + hold, smoothstep down to end + tail (frames)."""
        import torch
        p = self.p; a = on - p["lead"]; b = ap + p["hold"]; c = en + p["tail"]
        up = smoothstep_t((tau - a) / (ap - a).clamp(min=1e-6)); dn = smoothstep_t((c - tau) / (c - b).clamp(min=1e-6))
        return torch.where(tau < ap, up, torch.where(tau <= b, torch.ones_like(tau), dn))

    def reset(self, ids, pel_xy, yaw_def):
        """New episodes for envs `ids` (defender pelvis xy (k, 2) and yaw (k), read from qpos): draws D, lam, the first strike,
        koff, beta, hdz, the bout axis; both hands idle at guard. Returns (glove, elbow, head, yaw) for ids."""
        import torch
        p, k = self.p, len(ids)
        self.D[ids] = self._u(k, lo=p["D"][0], hi=p["D"][1]); self.lam[ids] = self._u(k, lo=p["lam"][0], hi=p["lam"][1])
        self.t_next[ids] = self._u(k, lo=p["first"][0], hi=p["first"][1])
        self.koff[ids] = torch.stack([self._u(k, lo=p[x][0], hi=p[x][1]) for x in ("base_dx", "base_dy", "base_dz")], -1)
        self.beta[ids] = math.radians(1.0) * self._u(k, lo=p["beta_deg"][0], hi=p["beta_deg"][1]); self.hdz[ids] = self._u(k, lo=p["head_dz"][0], hi=p["head_dz"][1])
        a = wrap_t(yaw_def + self.beta[ids] + math.radians(p["alpha_deg"]) * self._u(k, lo=-1.0)); self.alpha[ids] = a; self.alpha_tgt[ids] = a
        self.t_alpha[ids] = self._u(k, lo=p["alpha_every"][0], hi=p["alpha_every"][1])
        S = pel_xy + self.D[ids, None] * torch.stack([torch.cos(a), torch.sin(a)], -1); self.S[ids] = S
        self.f[ids] = torch.nn.functional.normalize(pel_xy - S, dim=-1)
        for x in (self.phase, self.strike_id, self.seg_id, self.zone): x[ids] = 0
        for x in (self.t0, self.t_ret, self.tau, self.u, self.win_on, self.win_off, self.t_apex, self.delta, self.S0, self.f0, self.n_strikes): x[ids] = 0
        self.sig[ids] = 1.0; self.latched[ids] = False
        Gg, Ge = self.G[self.ar, self.ar], self.G[self.ar, self.ar + 2]
        Sx, fx = self.S[ids, None], self.f[ids, None]
        self.glove[ids] = self._W(Sx, fx, Gg, ids); self.elbow[ids] = self._W(Sx, fx, Ge, ids)
        self.head[ids] = self._W(self.S[ids], self.f[ids], self.Ghead.expand(k, 3), ids) + self._hz(ids); self.yaw[ids] = torch.atan2(self.f[ids, 1], self.f[ids, 0])
        return self.glove[ids], self.elbow[ids], self.head[ids], self.yaw[ids]

    def step(self, t, pel_xy, yaw_def, x_T, R_T):
        """One control step at episode times t (N): live frame, strike ends / starts, then the targets. x_T, R_T: the defender's
        torso_link origin (N, 3) and rotation (N, 3, 3), for the aim points. Returns (glove, elbow, head, yaw)."""
        import torch
        p, dt, n = self.p, self.dt, self.n
        # live frame
        due = t >= self.t_alpha
        if bool(due.any()):
            i = torch.nonzero(due).flatten(); self.alpha_tgt[i] = wrap_t(yaw_def[i] + self.beta[i] + math.radians(p["alpha_deg"]) * self._u(len(i), lo=-1.0))
            self.t_alpha[i] = t[i] + self._u(len(i), lo=p["alpha_every"][0], hi=p["alpha_every"][1])
        self.alpha = wrap_t(self.alpha + wrap_t(self.alpha_tgt - self.alpha).clamp(-p["alpha_rate"] * dt, p["alpha_rate"] * dt))
        tgt = pel_xy + self.D[:, None] * torch.stack([torch.cos(self.alpha), torch.sin(self.alpha)], -1)
        v = (tgt - self.S) / p["spot_tau"]; sp = torch.linalg.norm(v, dim=-1, keepdim=True)
        self.S = self.S + v * torch.clamp(p["spot_vmax"] / sp.clamp(min=1e-9), max=1.0) * dt
        self.f = torch.nn.functional.normalize(pel_xy - self.S, dim=-1)
        # strike ends, returns
        sg = self.seg[self.seg_id]; st, nn, on, ap, en = sg[..., 0], sg[..., 1], sg[..., 2].float(), sg[..., 3].float(), sg[..., 4].float()
        tau = (t[:, None] - self.t0) * self.fps * self.sig
        ended = (self.phase == 1) & (tau >= (nn - 1).float()); self.phase[ended] = 2; self.t_ret = torch.where(ended, t[:, None].expand(n, 2), self.t_ret)
        self.phase[(self.phase == 2) & (t[:, None] - self.t_ret >= p["idle_blend_s"])] = 0
        # strike starts
        idle = self.phase == 0; go = (t >= self.t_next) & idle.any(1)
        if bool(go.any()):
            i = torch.nonzero(go).flatten(); k = len(i)
            h = torch.where(idle[i].all(1), (self._u(k) >= 0.5).long(), idle[i, 1].long())
            j = torch.minimum((self._u(k) * self.hcnt[h]).long(), self.hcnt[h] - 1); sid = self.hseg[h, j]; sig = self._u(k, lo=p["sigma"][0], hi=p["sigma"][1])
            self.phase[i, h] = 1; self.seg_id[i, h] = sid; self.t0[i, h] = t[i]; self.sig[i, h] = sig; self.zone[i, h] = (self._u(k) >= p["p_head"]).long()
            self.S0[i, h] = self.S[i]; self.f0[i, h] = self.f[i]; self.latched[i, h] = False; self.delta[i, h] = 0.0; self.strike_id[i, h] += 1; self.n_strikes[i] += 1
            r = self.fps * sig; self.win_on[i, h] = t[i] + self.seg[sid, 2].float() / r; self.t_apex[i, h] = t[i] + self.seg[sid, 3].float() / r
            self.win_off[i, h] = self.t_apex[i, h] + p["win_post"]
            self.t_next[i] = t[i] + torch.clamp(-torch.log1p(-self._u(k)) / self.lam[i], min=p["min_gap"])
            sg = self.seg[self.seg_id]; st, nn, on, ap, en = sg[..., 0], sg[..., 1], sg[..., 2].float(), sg[..., 3].float(), sg[..., 4].float()
            tau = (t[:, None] - self.t0) * self.fps * self.sig
        act = self.phase == 1; self.tau = torch.where(act, tau, torch.zeros_like(tau))
        # striking targets: K'(tau) + w Delta, Delta aimed until the onset, then latched
        Kp, yp, u = self._kprime(st, nn, tau); Kap, _, _ = self._kprime(st, nn, ap); self.u = torch.where(act, u, torch.zeros_like(u))
        w = self.aim_weight(tau, on, ap, en)
        pT = torch.where((self.zone == 0)[..., None], _t(tuple(p["aim_head"]), tau), _t(tuple(p["aim_torso"]), tau))
        surf = x_T[:, None] + torch.einsum("nij,nhj->nhi", R_T, pT)
        uh = torch.nn.functional.normalize(surf[..., :2] - self.S0, dim=-1); uh = torch.cat([uh, torch.zeros_like(uh[..., :1])], -1)
        nd = surf - (TIP - p["delta"]) * uh - self._W(self.S0, self.f0, Kap[:, self.ar, self.ar])
        nd = nd * torch.clamp(p["max_delta"] / torch.linalg.norm(nd, dim=-1, keepdim=True).clamp(min=1e-9), max=1.0)
        upd = act & ~self.latched; self.delta = torch.where(upd[..., None], nd, self.delta); self.latched = self.latched | (act & (tau >= on))
        g_s = self._W(self.S0, self.f0, Kp[:, self.ar, self.ar]) + w[..., None] * self.delta
        e_s = self._W(self.S0, self.f0, Kp[:, self.ar, self.ar + 2]) + w[..., None] * self.delta
        # guard: live (idle), frozen -> live (returning)
        Gg, Ge = self.G[self.ar, self.ar], self.G[self.ar, self.ar + 2]
        g_l, e_l = self._W(self.S[:, None], self.f[:, None], Gg), self._W(self.S[:, None], self.f[:, None], Ge)
        b = smoothstep_t((t[:, None] - self.t_ret) / p["idle_blend_s"])[..., None]
        g_r = self._W(self.S0, self.f0, Gg) * (1 - b) + g_l * b; e_r = self._W(self.S0, self.f0, Ge) * (1 - b) + e_l * b
        ph = self.phase[..., None]
        self.glove = torch.where(ph == 1, g_s, torch.where(ph == 2, g_r, g_l)); self.elbow = torch.where(ph == 1, e_s, torch.where(ph == 2, e_r, e_l))
        # virtual head and yaw: the most recently started strike's, else the live guard head and facing
        hs = torch.argmax(torch.where(act, self.t0, torch.full_like(self.t0, -1e9)), 1); anyact = act.any(1); r_ = self.rows
        head_s = self._W(self.S0[r_, hs], self.f0[r_, hs], Kp[r_, hs, HEAD]); yaw_s = torch.atan2(self.f0[r_, hs, 1], self.f0[r_, hs, 0]) + yp[r_, hs]
        self.head = torch.where(anyact[:, None], head_s, self._W(self.S, self.f, self.Ghead.expand(n, 3))) + self._hz()
        self.yaw = wrap_t(torch.where(anyact, yaw_s, torch.atan2(self.f[:, 1], self.f[:, 0])))
        return self.glove, self.elbow, self.head, self.yaw

    def phase_obs(self, t=None):
        """strike_phase (critic, 10): per hand l then r: active, segment progress u (the guard blend's smoothstep), time to the
        apex in s clipped to +-0.5 (from the last step's clock, or at times t), zone head, zone torso; zeros when not striking."""
        import torch
        act = (self.phase == 1).float()
        tta = (self.t_apex - (self.t0 + self.tau / (self.fps * self.sig)) if t is None else self.t_apex - t[:, None]).clamp(-0.5, 0.5)
        return torch.stack([act, self.u * act, tta * act, (self.zone == 0).float() * act, (self.zone == 1).float() * act], -1).reshape(self.n, 10)

# =============================================================================================== torch: hit accounting (2.5)
class HitAccount:
    """r1gym/hits.py's rules on GPU contact data, per (fist, zone): per-substep zone forces (sum of |F| over the zone's proxies)
    merged into contact episodes over gaps <= 0.1 s; impulse = sum F * 5 ms; landed = head/torso, >= 0.5 N*s, peak >= 5 N;
    clean = landed, closing speed >= 1 m/s, duration (first to last substep, both included) <= 0.15 s. The closing speed is
    the one the episode's first substep gets (closing_speeds). An episode is scored when it closes (more than 0.1 s after its
    last contact). Strike windows [onset, apex + 0.15 s], as HitMeter.windows: only episodes of the striking fist that OPEN
    inside the window count (a contact already going on when it opens never does; pelvis contacts never do); the window is
    blocked iff an arm episode (block or upper_arm zone, merged as HitMeter's one 'arm' zone) opens before the first landed
    head / torso episode that opened in it, or opens and none lands. Landing is monotone (impulse and peak only grow), so a
    window is decided once it is past and no unlanded head / torso episode that opened in it before its arm episode is still
    open; a new strike of that fist, a reset or flush decides it on what is known. Per step: head and torso N*s (both
    fists), clean head episodes closed, blocked windows whose arm episode was on the block zone (spar_block). Per episode
    (EP_KEYS) and in total; a ring buffer of head-landed impulses. Everything stays on device (no host sync)."""
    ARM = (3, 4)   # zone order head, torso, pelvis, block, upper_arm: HitMeter's 'arm' = block + upper_arm

    def __init__(self, n, device="cpu", zone_of_prim=None, p=None):
        import torch
        self.p = {**HIT, **(p or {})}; self.n, self.dev = n, device; Z = len(ZONES)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.open, self.ewin = z(n, 2, Z, dt=torch.bool), z(n, 2, Z, dt=torch.bool); self.t0, self.t1, self.imp, self.peak, self.vc, self.vcs = (z(n, 2, Z) for _ in range(6))
        self.w_open = z(n, 2, dt=torch.bool); self.w_on, self.w_off = z(n, 2), z(n, 2); self.inf = torch.tensor(float("inf"), device=device)
        self.w_arm, self.w_hit, self.w_armz = torch.full((n, 2), float("inf"), device=device), torch.full((n, 2), float("inf"), device=device), z(n, 2, dt=torch.long)
        self.seen = z(n, 2, dt=torch.long); self.zi = torch.arange(Z, device=device)
        self.ep = {k: z(n) for k in EP_KEYS}; self.step = {k: z(n) for k in ("head_imp", "torso_imp", "head_clean", "block")}
        self.tot = {k: z(()) for k in EP_KEYS}; R = int(self.p["ring"]); self.ring = z(R + 1); self.ring_i = z((), dt=torch.long); self.ring_n = z((), dt=torch.long)
        self.Zm = None
        if zone_of_prim is not None:
            self.Zm = torch.nn.functional.one_hot(torch.as_tensor(zone_of_prim, device=device), Z).float()

    def zone_forces(self, F_hist):
        """Sensor force_history (B, P, H, 3), index 0 = latest substep -> (zone forces (B, 5, H): sum of |F| over the zone's
        proxies, zone force vectors (B, 5, H, 3)), oldest substep first."""
        import torch
        F = F_hist.flip(-2); return torch.einsum("bph,pz->bzh", torch.linalg.norm(F, dim=-1), self.Zm), torch.einsum("bphc,pz->bzhc", F, self.Zm)

    def reset(self, ids):
        self.open[ids] = False; self.ewin[ids] = False; self.w_open[ids] = False; self.seen[ids] = 0
        for d in (self.ep, self.step):
            for v in d.values(): v[ids] = 0.0

    def reset_totals(self):
        for v in self.tot.values(): v.zero_()
        self.ring.zero_(); self.ring_i.zero_(); self.ring_n.zero_()

    def _landed(self):
        p = self.p; return self.open & (self.zi <= 1) & (self.imp >= p["min_impulse"]) & (self.peak >= p["min_peak"])

    def _close(self, C):
        import torch
        p = self.p
        land = C & (self.zi <= 1) & (self.imp >= p["min_impulse"]) & (self.peak >= p["min_peak"])
        clean = land & (self.vc >= p["v_close"]) & (self.t1 - self.t0 + p["dt"] <= p["max_dur"] + T_TOL)
        hl, tl, hc, tc = land[..., 0].float().sum(1), land[..., 1].float().sum(1), clean[..., 0].float().sum(1), clean[..., 1].float().sum(1)
        hi = (self.imp[..., 0] * land[..., 0]).sum(1); hn = (land[..., 0] & (self.vcs[..., 0] < 0)).float().sum(1)
        for k, v in (("head_landed", hl), ("torso_landed", tl), ("head_clean", hc), ("torso_clean", tc), ("head_imp_landed", hi), ("head_landed_vneg", hn)):
            self.ep[k] += v; self.tot[k] += v.sum()
        self.step["head_clean"] += hc
        m = land[..., 0].flatten(); R = self.ring.shape[0] - 1   # ring: landed head impulses, scattered without a host sync
        pos = torch.where(m, (self.ring_i + torch.cumsum(m.long(), 0) - 1) % R, torch.full_like(self.ring_i, R).expand(m.shape))
        self.ring.scatter_(0, pos, self.imp[..., 0].flatten()); c = m.long().sum(); self.ring_i = (self.ring_i + c) % R; self.ring_n = torch.clamp(self.ring_n + c, max=R)
        self.open &= ~C

    def _pending(self):
        """(N, 2): the window may still change: an open head / torso episode that opened in it, has not landed yet and opened
        before its arm episode or on the same substep (it could still land and turn 'blocked' into a hit: HitMeter calls a
        window blocked only when the arm episode opened strictly first)."""
        import torch
        w = self.open & self.ewin & ~self._landed() & (self.t0 <= self.w_arm[..., None])
        return w[..., :2].any(-1) & torch.isfinite(self.w_arm)

    def _finish(self, E):
        """Decide windows E (N, 2): blocked iff the arm episode opened before the first landed head / torso one (HitMeter)."""
        bl = E & (self.w_arm < self.w_hit); bf = (bl & (self.w_armz == self.ARM[0])).float().sum(1)
        for k, v in (("windows", E.float().sum(1)), ("blocked", bl.float().sum(1)), ("block_first", bf)):
            self.ep[k] += v; self.tot[k] += v.sum()
        self.step["block"] += bf; self.w_open &= ~E; self.ewin &= ~E[..., None]

    def process(self, Fz, t_end, strike_id, w_on, w_off, vc, vcs=None):
        """One control step: Fz (N, 2, 5, S) zone forces per fist, oldest substep first, the last ending at episode time t_end
        (N); the schedule's strike_id / window bounds (N, 2); vc (N, 2, 5, S): the closing speed an episode opening at that
        substep gets (closing_speeds), vcs the same signed (diagnostic: head_landed_vneg; default vc)."""
        import torch
        p = self.p; S = Fz.shape[-1]; vcs = vc if vcs is None else vcs; a3, a4 = self.ARM
        for v in self.step.values(): v.zero_()
        new = strike_id != self.seen
        self._finish(self.w_open & new)   # that fist's next strike started: decide its last window on what is known
        self.w_open |= new; self.w_on = torch.where(new, w_on, self.w_on); self.w_off = torch.where(new, w_off, self.w_off)
        self.w_arm = torch.where(new, self.inf, self.w_arm); self.w_hit = torch.where(new, self.inf, self.w_hit); self.ewin &= ~new[..., None]
        self.seen = strike_id.clone()
        for j in range(S):
            tj = (t_end - (S - 1 - j) * p["dt"])[:, None]; F = Fz[..., j]
            self._close(self.open & (tj[..., None] - self.t1 > p["gap"] + T_TOL))
            touch = F > 0; nw = touch & ~self.open
            inw = self.w_open & (tj >= self.w_on - W_TOL) & (tj <= self.w_off + W_TOL)
            arm = (nw[..., a3] | nw[..., a4]) & ~(self.open[..., a3] | self.open[..., a4]) & inw & ~torch.isfinite(self.w_arm)
            self.w_armz = torch.where(arm, torch.where(nw[..., a3], a3, a4), self.w_armz); self.w_arm = torch.where(arm, tj.expand_as(self.w_arm), self.w_arm)
            self.ewin = torch.where(nw, inw[..., None].expand_as(nw), self.ewin)
            self.t0 = torch.where(nw, tj[..., None].expand_as(F), self.t0); self.imp = torch.where(nw, torch.zeros_like(F), self.imp)
            self.peak = torch.where(nw, torch.zeros_like(F), self.peak); self.vc = torch.where(nw, vc[..., j], self.vc); self.vcs = torch.where(nw, vcs[..., j], self.vcs); self.open |= nw
            self.t1 = torch.where(touch, tj[..., None].expand_as(F), self.t1); self.imp = self.imp + F * p["dt"]; self.peak = torch.maximum(self.peak, F)
            self.step["head_imp"] += F[:, :, 0].sum(1) * p["dt"]; self.step["torso_imp"] += F[:, :, 1].sum(1) * p["dt"]
            self.w_hit = torch.minimum(self.w_hit, torch.where(self._landed() & self.ewin, self.t0, self.inf).amin(-1))
            self._finish(self.w_open & (tj > self.w_off + W_TOL) & ~self._pending())

    def close_envs(self, ids):
        """Score the open episodes and decide the open windows of envs `ids` (an episode ending: the reset would drop them)."""
        import torch
        m = torch.zeros(self.n, dtype=torch.bool, device=self.dev); m[ids] = True
        self._close(self.open & m[:, None, None]); self._finish(self.w_open & m[:, None])

    def flush(self, t_end):
        """Close every open episode and window (end of a CPU bout)."""
        import torch
        z = torch.zeros(self.n, 2, len(ZONES), 1, device=self.dev); self.process(z, t_end + 10.0, self.seen, self.w_on, self.w_off, z)

    def median_head_impulse(self):
        import torch
        k = int(self.ring_n); return float(torch.median(self.ring[:k])) if k else float("nan")


def fist_substep_vel(x0, v0, q_prev, q_next, mass=ATTACK["mass"], fmax=ATTACK["fmax"], S=DECIM):
    """Glove velocities (N, 2, 3, S) at the start of each physics substep of a control step, predicted from the slides' state
    at the step's start (x0, v0 (N, 2, 3): slide qpos / qvel, i.e. the glove's world position / velocity) under the
    AttackerEntity's target law (lerp(q_prev, q_next) + damping feed-forward), the force cap and MuJoCo's implicit damping,
    contact-free. Why: the servo brakes the fist hard (up to F_max / m = 100 m/s^2) as it nears its target 5 mm inside the
    shell, so the velocity at the control step before contact overstates the closing speed at contact (CPU: +0.5 m/s mean,
    +33% head clean strikes vs HitMeter); HitMeter takes it at the contact's first substep."""
    import torch
    kp, kv = ATTACK["kp"], slide_kv(mass); L = ATTACK["link_mass"]; m = _t((mass + 4 * L, mass + 3 * L, mass + 2 * L), x0)   # x, y, z slide loads
    dq = q_next - q_prev; x, v, out = x0, v0, []
    for j in range(S):
        out.append(v); f = kp * (q_prev + min(1.0, (j + 1) / S) * dq + (kv / kp) * dq / CTRL_DT - x) - kv * v
        v = torch.where(f.abs() > fmax, v + SUB_DT * f.clamp(-fmax, fmax) / m, v + SUB_DT * f / (m + SUB_DT * kv)); x = x + SUB_DT * v
    return torch.stack(out, -1)

FORCE_SIGN = -1.0   # the hit sensors' net force points against the fist's push (measured on CPU, MuJoCo 3.17: 66 landed episodes, corr with HitMeter 0.94, clean calls agree 97%)

def closing_speeds(v_rel, Fv, signed=False):
    """Closing speeds (N, 2, 5, S): the fist's velocity relative to the defender's torso_link at each substep's start (v_rel
    (N, 2, 3, S), fist_substep_vel - v_torso) along that substep's contact normal, the direction of the zone's net contact
    force (Fv (N, 2, 5, S, 3); condim 1 contacts are frictionless, so the force is along the normal). HitMeter takes the
    relative velocity of the contact point along the normal at the episode's first substep. Sign-free by default: at a
    contact's first substep fist and target are closing, so |v . n| is the closing speed whichever way the sensor's force
    points (MuJoCo Warp's sign is unchecked); signed=True applies FORCE_SIGN (m1check logs how often that is < 0 on head hits)."""
    import torch
    c = (v_rel.transpose(-1, -2)[:, :, None] * torch.nn.functional.normalize(Fv, dim=-1)).sum(-1)
    return FORCE_SIGN * c if signed else c.abs()

def zone_index(primary_names):
    """Zone per sensor primary (geom names <link>_shell, in the sensor's own order: mjlab resolves them in geom order)."""
    return [ZONES.index(ZONE_OF_LINK[n.split("/")[-1][:-len("_shell")]]) for n in primary_names]

# =============================================================================================== mjlab glue: obs, rewards
def _cmd(env, name="attack"): return env.command_manager.get_term(name)

class opp_keypoints:
    """Actor term (3.2): the 28-d block from the attack command's true keypoints, with training corruption when corrupt=True:
    per-episode bias U(+-0.02) m per keypoint axis plus U(+-0.02) m per step on positions, U(+-0.25) m/s on glove velocities,
    U(+-0.1) rad on the relative yaw (before sin/cos), each keypoint dropped with p 0.10; then the mask (in view and not
    dropped) zeroes it. Noise is applied here (the term's own `noise` stays None) so the zeros stay exact."""
    def __init__(self, cfg, env):
        import torch
        self.n, self.dev = env.num_envs, env.device; self.bias = torch.zeros(self.n, N_KP, 3, device=self.dev)
    def reset(self, env_ids=None):
        import torch
        ids = slice(None) if env_ids is None else env_ids; k = self.n if env_ids is None or isinstance(env_ids, slice) else len(env_ids)
        self.bias[ids] = CORRUPT["bias"] * (2 * torch.rand(k, N_KP, 3, device=self.dev) - 1)
    def __call__(self, env, command_name="attack", corrupt=True):
        import torch
        P, Pp, yaw, vis = _cmd(env, command_name).kp_true(); V = (P[:, :2] - Pp[:, :2]) / CTRL_DT
        if corrupt:
            c, r = CORRUPT, lambda *s: 2 * torch.rand(*s, device=P.device) - 1
            P = P + self.bias + c["pos"] * r(*P.shape); V = V + c["vel"] * r(*V.shape); yaw = yaw + c["yaw"] * r(*yaw.shape)
            vis = vis & (torch.rand(vis.shape, device=P.device) >= c["drop"])
        return pack_t(P, V, yaw, vis)

def opp_keypoints_truth(env, command_name="attack"):
    """Critic term: the clean block (no noise, dropout or delay; mask = in view)."""
    P, Pp, yaw, vis = _cmd(env, command_name).kp_true(); return keypoint_block(P, Pp, yaw, vis)

def opp_keypoints_blind(env, command_name="attack"):
    """The blind twin's actor term: 28 zeros (CPU feeds zeros to a policy whose metadata ends in opp_keypoints_blind)."""
    import torch; return torch.zeros(env.num_envs, KP_DIM, device=env.device)

def strike_phase(env, command_name="attack"):
    return _cmd(env, command_name).sched.phase_obs()

def spar_head_impulse(env, command_name="attack"):
    c = _cmd(env, command_name); c.hit_update(); return c.hits.step["head_imp"]
def spar_head_clean(env, command_name="attack"):
    c = _cmd(env, command_name); c.hit_update(); return c.hits.step["head_clean"]
def spar_torso_impulse(env, command_name="attack"):
    c = _cmd(env, command_name); c.hit_update(); return c.hits.step["torso_imp"]
def spar_block(env, command_name="attack"):
    c = _cmd(env, command_name); c.hit_update(); return c.hits.step["block"]
def spar_hits(env, command_name="attack"):
    """Metrics term (every spar task): runs the hit update after the rewards and before the reset block, so an ending
    episode's last step is counted; logs clean head episodes closed per step."""
    c = _cmd(env, command_name); c.hit_update(); return c.hits.step["head_clean"]
def spar_clearance(env, command_name="attack"):
    """Per head-aimed strike in [onset - 0.15 s, apex + 0.15 s]: clip((d - 0.10) / 0.15, 0, 1), d = glove tip to the defender's
    head point, summed over hands."""
    return _cmd(env, command_name).clearance()

# =============================================================================================== mjlab glue: classes
_C = {}
def _classes():
    """mjlab classes, built once on first use (mjlab is imported only here): AttackerEntity(Cfg), AttackCommand(Cfg),
    OuterLoopTwistCommand(Cfg)."""
    if _C: return _C
    from dataclasses import dataclass
    import torch
    from mjlab.entity import Entity, EntityCfg
    from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
    from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

    class AttackerEntity(Entity):
        """Per physics substep k of a control step, the servo targets are lerp(q_prev, q_next, min(1, (k + 1) / decimation)),
        plus the damping feed-forward (kv / kp) * (q_next - q_prev) / step_dt on the slides (and, without body gravcomp, the
        z offset m g / kp). The attack command sets q_prev <- q_next, q_next <- new, k <- 0 once per env step."""
        def initialize(self, mj_model, model, data, device):
            super().initialize(mj_model, model, data, device); n = data.nworld; J = self.num_joints
            self.q_prev, self.q_next, self.k = torch.zeros(n, J, device=device), torch.zeros(n, J, device=device), 0
            self.slides = torch.tensor(self.find_joints(r"att_._[xyz]")[0], device=device); self.zs = torch.tensor(self.find_joints(r"att_._z")[0], device=device)
        def set_targets(self, q): self.q_prev, self.q_next, self.k = self.q_next, q, 0
        def reset_targets(self, env_ids, q): self.q_prev[env_ids] = q; self.q_next[env_ids] = q
        def update(self, dt): super().update(dt); self.k += 1
        def write_data_to_sim(self):
            c = self.cfg; a = min(1.0, (self.k + 1) / c.decimation); dq = self.q_next - self.q_prev; tgt = self.q_prev + a * dq
            tgt[:, self.slides] += (c.kv / c.kp) * dq[:, self.slides] / c.step_dt
            if c.grav_offset: tgt[:, self.zs] += c.grav_offset
            self._data.joint_pos_target[:] = tgt; super().write_data_to_sim()

    @dataclass
    class AttackerEntityCfg(EntityCfg):
        kp: float = ATTACK["kp"]; kv: float = slide_kv(ATTACK["mass"]); decimation: int = DECIM; step_dt: float = CTRL_DT; grav_offset: float = 0.0
        def build(self): return AttackerEntity(self)

    class AttackCommand(CommandTerm):
        """The attack (2.4-2.6): AttackSchedule -> attacker joint targets each step; the virtual head / yaw and the true
        keypoints for the observations; HitAccount fed by the hit_l / hit_r sensors (update idempotent per env step, done by
        the spar_hits metrics term, before the reset block: mjlab 1.2.0 step = physics, terminations, rewards, metrics, reset,
        forward, commands, events, observations); an ending episode's open contacts and windows are scored at its reset;
        per-episode metrics logged at reset. The attacker's root must stay at the world origin (slide values are world
        coordinates): asserted here and, on the device, by check_root()."""
        METRICS = ("head_clean_pm", "torso_clean_pm", "strikes_pm", "blocked_share", "head_impulse_per_landed")
        def __init__(self, cfg, env):
            super().__init__(cfg, env); n, dev = self.num_envs, self.device
            self.robot, self.att = env.scene[cfg.robot_name], env.scene[cfg.attacker_name]
            ist = self.att.cfg.init_state; assert tuple(ist.pos) == (0.0, 0.0, 0.0) and tuple(ist.rot) == (1.0, 0.0, 0.0, 0.0), \
                f"attacker root must sit at the world origin (slides are world coordinates): init_state {ist.pos} {ist.rot}"
            ev = getattr(getattr(env, "cfg", None), "events", None) or {}
            bad = [k for k, e in ev.items() if e is not None and (getattr(e.func, "__name__", "") == "reset_scene_to_default"
                   or any(getattr(v, "name", None) == cfg.attacker_name for v in (getattr(e, "params", None) or {}).values()))]
            assert not bad, f"events {bad} would move the attacker's root off the world origin (slides are world coordinates)"
            self.sched = AttackSchedule(load_library(cfg.library), n, dev, env.step_dt, p={"delta": cfg.delta})
            self.torso = self.robot.find_bodies("torso_link")[0][0]; self.fists = self.att.find_bodies(list(FISTS), preserve_order=True)[0]
            self.jid = torch.tensor(self.att.find_joints(list(ATT_JOINTS), preserve_order=True)[0], device=dev)
            prims = [list(dict.fromkeys(s.primary_name for s in env.scene[nm]._slots)) for nm in cfg.sensor_names]
            assert prims[0] == prims[1] and len(prims[0]) == len(LINKS), prims
            self.hits = HitAccount(n, dev, zone_index(prims[0])); self._hit_step = -1; z3 = torch.zeros(n, 2, 3, device=dev)
            self._hs = (z3, z3, torch.zeros(n, 3, device=dev))
            self._hs_ok = torch.zeros(n, device=dev)
            self.head_w, self.opp_yaw = torch.zeros(n, 3, device=dev), torch.zeros(n, device=dev)
            self._ver, self._kp_ver, self.kp_hist = 0, -1, KeypointHistory(n, dev)
            self._kp = None; self.metrics = {k: torch.zeros(n, device=dev) for k in self.METRICS}
        @property
        def command(self): return self.sched.phase_obs()
        def _t_ep(self): return self._env.episode_length_buf.float() * self._env.step_dt
        def _torso(self):
            d = self.robot.data; return d.body_link_pos_w[:, self.torso], quat_to_mat(d.body_link_quat_w[:, self.torso])
        def check_root(self):
            """Host-side check (m1check, after a forward): every env's attacker root at the world origin."""
            p = self.att.data.root_link_pos_w; assert float(p.abs().max()) < 1e-6, f"attacker root off the world origin: max |p| {float(p.abs().max()):.3g} m"
        def hit_update(self):
            if self._hit_step == self._env.common_step_counter: return
            self._hit_step = self._env.common_step_counter
            Fz, Fv = (torch.stack(x, 1) for x in zip(*[self.hits.zone_forces(self._env.scene[nm].data.force_history) for nm in self.cfg.sensor_names]))
            n = self.num_envs; sl = lambda q: q[:, self.jid].reshape(n, 2, 5)[..., :3]
            x0, v0, vT0 = self._hs   # this step's start, saved by _update_command after the forward (slides: qpos / qvel, exact)
            vr = fist_substep_vel(x0, v0, sl(self.att.q_prev), sl(self.att.q_next)) - vT0[:, None, :, None]; ok = self._hs_ok[:, None, None, None]
            s = self.sched; self.hits.process(Fz, self._t_ep(), s.strike_id, s.win_on, s.win_off, closing_speeds(vr, Fv) * ok, closing_speeds(vr, Fv, signed=True) * ok)
        def clearance(self):
            s = self.sched; t = self._t_ep()[:, None]; ad = self.att.data; x_T, R_T = self._torso()
            tip = ad.body_link_pos_w[:, self.fists] + TIP * quat_to_mat(ad.body_link_quat_w[:, self.fists])[..., :, 0]
            hp = x_T + torch.einsum("nij,j->ni", R_T, _t(DEF_HEAD_T, x_T)); d = torch.linalg.norm(tip - hp[:, None], dim=-1)
            win = (s.phase == 1) & (s.zone == 0) & (t >= s.win_on - 0.15) & (t <= s.t_apex + 0.15)
            return (((d - 0.10) / 0.15).clamp(0.0, 1.0) * win).sum(1)
        def _refresh_metrics(self, ids=None):
            ids = slice(None) if ids is None else ids; h, mins = self.hits.ep, (self._t_ep()[ids].clamp(min=self._env.step_dt) / 60.0)
            self.metrics["head_clean_pm"][ids] = h["head_clean"][ids] / mins; self.metrics["torso_clean_pm"][ids] = h["torso_clean"][ids] / mins
            self.metrics["strikes_pm"][ids] = self.sched.n_strikes[ids] / mins
            self.metrics["blocked_share"][ids] = h["blocked"][ids] / h["windows"][ids].clamp(min=1.0)
            self.metrics["head_impulse_per_landed"][ids] = h["head_imp_landed"][ids] / h["head_landed"][ids].clamp(min=1.0)
        def _update_metrics(self): self.hit_update(); self._refresh_metrics()
        def reset(self, env_ids):
            if isinstance(env_ids, torch.Tensor) and len(env_ids): self.hits.close_envs(env_ids); self._refresh_metrics(env_ids)
            return super().reset(env_ids)
        def _resample_command(self, env_ids):   # new episode (resampling_time_range is effectively infinite): derived state is stale, read qpos
            q = self._env.sim.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr]; w, x, y, z = q[:, 3:7].unbind(-1)
            g, e, hd, yw = self.sched.reset(env_ids, q[:, :2], torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            full = torch.zeros(len(env_ids), self.att.num_joints, device=self.device); full[:, self.jid] = joint_targets(g, e)
            self.att.write_joint_state_to_sim(full, torch.zeros_like(full), env_ids=env_ids); self.att.reset_targets(env_ids, full)
            self.hits.reset(env_ids); self._hs_ok[env_ids] = 0.0; self.head_w[env_ids] = hd; self.opp_yaw[env_ids] = yw; self.kp_hist.reset(env_ids); self._ver += 1
        def _update_command(self):
            """After the forward: the head / yaw the gloves were driven to over the last physics step (the schedule's state
            before this advance: gloves and head share a timestamp), then the next targets; and the next step's start state
            for the closing speeds (fresh for every env, reset ones included; an interval push applied after this is missed)."""
            rd, ad = self.robot.data, self.att.data; x_T, R_T = self._torso(); s = self.sched
            self.head_w, self.opp_yaw = s.head.clone(), s.yaw.clone()
            g, e, _, _ = s.step(self._t_ep(), rd.root_link_pos_w[:, :2], rd.heading_w, x_T, R_T)
            q = self.att.q_next.clone(); q[:, self.jid] = joint_targets(g, e, ad.joint_pos[:, self.jid]); self.att.set_targets(q)
            n = self.num_envs; sl = lambda v: v[:, self.jid].reshape(n, 2, 5)[..., :3]
            self._hs = (sl(ad.joint_pos).clone(), sl(ad.joint_vel).clone(), rd.body_link_lin_vel_w[:, self.torso].clone()); self._hs_ok[:] = 1.0; self._ver += 1
        def kp_true(self):
            """(P_T (N, 5, 3), the previous step's T-frame block (KeypointHistory; = P_T at an episode's first step), rel_yaw (N),
            in view (N, 5)): the 1.2 GPU keypoints in the defender's T, recomputed once per command update (shared by the actor
            and critic terms)."""
            if self._kp_ver != self._ver:
                ad = self.att.data; x_T, R_T = self._torso(); gl = ad.body_link_pos_w[:, self.fists]; Rf = quat_to_mat(ad.body_link_quat_w[:, self.fists])
                P, prev = self.kp_hist.step(to_torso_t(torch.cat([gl, gl - ELBOW_TO_FIST * Rf[..., :, 0], self.head_w[:, None]], 1), x_T, R_T))
                self._kp = (P, prev, wrap_t(self.opp_yaw - heading_t(R_T)), in_view_t(P)); self._kp_ver = self._ver
            return self._kp

    @dataclass(kw_only=True)
    class AttackCommandCfg(CommandTermCfg):
        library: str = str(LIB_PATH); robot_name: str = "robot"; attacker_name: str = "attacker"; sensor_names: tuple = ("hit_l", "hit_r")
        delta: float = ATTACK["delta"]
        def build(self, env): return AttackCommand(self, env)

    class OuterLoopTwistCommand(UniformVelocityCommand):
        """Per episode a share rel_outer_envs of the envs take r1gym/env.py's outer loops every step (face: wz = clip(2.0 x the
        bearing error of the live spot S, +-0.8); hold: back to the episode's start pelvis xy with hysteresis 0.25 / 0.08 m,
        vx = clip(1.5 e_fwd, -0.5, 0.8), vy = clip(1.5 e_left, +-0.5), else 0); the rest keep the footwork command."""
        def __init__(self, cfg, env):
            super().__init__(cfg, env); n = self.num_envs
            self.outer = torch.zeros(n, dtype=torch.bool, device=self.device); self.spot = torch.zeros(n, 2, device=self.device)
            self.returning = torch.zeros_like(self.outer)
        def reset(self, env_ids):
            out = super().reset(env_ids)
            if isinstance(env_ids, torch.Tensor) and len(env_ids):
                self.outer[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_outer_envs; self.returning[env_ids] = False
                self.spot[env_ids] = self._env.sim.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[:2]]
            return out
        def _update_command(self):
            super()._update_command(); d = self.robot.data; pel, yaw = d.root_link_pos_w[:, :2], d.heading_w
            S = self._env.command_manager.get_term(self.cfg.attack_name).sched.S; v = S - pel
            wz = (2.0 * wrap_t(torch.atan2(v[:, 1], v[:, 0]) - yaw)).clamp(-0.8, 0.8)
            e = self.spot - pel; dist = torch.linalg.norm(e, dim=-1); self.returning = torch.where(self.returning, dist > 0.08, dist > 0.25)
            c, s = torch.cos(yaw), torch.sin(yaw); r = self.returning.float()
            vx = (1.5 * (c * e[:, 0] + s * e[:, 1])).clamp(-0.5, 0.8) * r; vy = (1.5 * (-s * e[:, 0] + c * e[:, 1])).clamp(-0.5, 0.5) * r
            self.vel_command_b = torch.where(self.outer[:, None], torch.stack([vx, vy, wz], -1), self.vel_command_b)

    @dataclass(kw_only=True)
    class OuterLoopTwistCommandCfg(UniformVelocityCommandCfg):
        rel_outer_envs: float = 0.5; attack_name: str = "attack"
        def build(self, env): return OuterLoopTwistCommand(self, env)

    _C.update(AttackerEntity=AttackerEntity, AttackerEntityCfg=AttackerEntityCfg, AttackCommand=AttackCommand, AttackCommandCfg=AttackCommandCfg,
              OuterLoopTwistCommand=OuterLoopTwistCommand, OuterLoopTwistCommandCfg=OuterLoopTwistCommandCfg)
    return _C

def attacker_entity_cfg():
    """EntityCfg of the attacker: attacker_spec + the two BuiltinPositionActuatorCfg groups (slides 1500 / 95.5 / F_max, armature 0;
    hinges 60 / 3 / 30, armature 0.01). SPAR_GRAVCOMP=0: no body gravcomp, the slides' z targets get + m g / kp instead."""
    import functools
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    A, gc = ATTACK, os.environ.get("SPAR_GRAVCOMP", "1") != "0"
    art = EntityArticulationInfoCfg(actuators=(
        BuiltinPositionActuatorCfg(target_names_expr=(r"att_._[xyz]",), stiffness=A["kp"], damping=slide_kv(A["mass"]), effort_limit=A["fmax"], armature=0.0),
        BuiltinPositionActuatorCfg(target_names_expr=(r"att_._(yaw|pitch)",), stiffness=A["rot_kp"], damping=A["rot_kv"], effort_limit=A["rot_fmax"], armature=A["rot_armature"])))
    return _classes()["AttackerEntityCfg"](init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        spec_fn=functools.partial(attacker_spec, A["mass"], A["fmax"], False, gc), articulation=art,
        grav_offset=0.0 if gc else (A["mass"] + 2 * A["link_mass"]) * 9.81 / A["kp"])

# =============================================================================================== task configs
def _robot_spec(base, proxy):
    s = base(); add_shell_proxies(s, proxy=proxy); return s

def _scene_spec(prev, spec):
    if prev is not None: prev(spec)
    proxy_bits(spec, prefix="robot/")

def _attack(make, outer=None, check=False):
    """R1-Spar-Attack (2.6) on top of `make`: attacker, proxies (SPAR_SHELL_PROXY), sensors, attack command, nan termination,
    the spar_hits metrics term (the hit update before resets).
    outer: also replace the twist command by OuterLoopTwistCommand with that share of outer-loop envs (twist stays first).
    check: m1check's variant (push_robot and shove removed)."""
    import functools
    def f(play=False):
        from mjlab.envs.mdp.terminations import nan_detection
        from mjlab.managers.metrics_manager import MetricsTermCfg
        from mjlab.managers.termination_manager import TerminationTermCfg
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        C = _classes(); cfg = make(play); robot = cfg.scene.entities["robot"]
        robot.spec_fn = functools.partial(_robot_spec, robot.spec_fn, os.environ.get("SPAR_SHELL_PROXY", "mesh"))
        cfg.scene.entities["attacker"] = attacker_entity_cfg()
        cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(ContactSensorCfg(
            name=f"hit_{s}", primary=ContactMatch(mode="geom", entity="robot", pattern=tuple(l + "_shell" for l in LINKS)),
            secondary=ContactMatch(mode="body", entity="attacker", pattern=f"opp_{nm}_wrist_roll_link"),
            fields=("found", "force"), reduce="netforce", num_slots=1, history_length=DECIM) for s, nm in SIDES)
        cfg.scene.spec_fn = functools.partial(_scene_spec, cfg.scene.spec_fn)
        if outer is not None:
            import dataclasses
            tw = cfg.commands["twist"]; cfg.commands["twist"] = C["OuterLoopTwistCommandCfg"](**{x.name: getattr(tw, x.name) for x in dataclasses.fields(tw)}, rel_outer_envs=outer)
        cfg.commands["attack"] = C["AttackCommandCfg"](resampling_time_range=(1e9, 1e9), library=str(LIB_PATH))
        cfg.terminations["nan"] = TerminationTermCfg(func=nan_detection)
        cfg.metrics = {**(getattr(cfg, "metrics", None) or {}), "spar_hits": MetricsTermCfg(func=spar_hits, params={"command_name": "attack"})}
        if check:
            for k in ("push_robot", "shove"): cfg.events.pop(k, None)
        return cfg
    return f

def _block(make, weights="A", blind=False):
    """R1-Spar-Block-{A,B} / -Blind(-B) (3.2) on top of `make` (the footwork guard)."""
    def f(play=False):
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        cfg = _attack(make, outer=0.5)(play); lag = 0 if play else 2
        if blind: term = ObservationTermCfg(func=opp_keypoints_blind, params={"command_name": "attack"}, delay_min_lag=0, delay_max_lag=lag)
        else: term = ObservationTermCfg(func=opp_keypoints, params={"command_name": "attack", "corrupt": not play}, delay_min_lag=0, delay_max_lag=lag)
        cfg.observations["actor"].terms["opp_keypoints_blind" if blind else "opp_keypoints"] = term
        cr = cfg.observations["critic"].terms
        cr["opp_keypoints_truth"] = ObservationTermCfg(func=opp_keypoints_truth, params={"command_name": "attack"})
        cr["strike_phase"] = ObservationTermCfg(func=strike_phase, params={"command_name": "attack"})
        pp = cfg.rewards["pose"].params; pp["std_standing"] = dict(STD_STANDING)
        for k in ("std_walking", "std_running"):
            d = dict(pp[k]); d.update({a: 0.35 for a in POSTURE_ARMS}); d[r".*waist_roll.*"] = 0.15; pp[k] = d
        for name, w in WEIGHTS[weights].items():
            cfg.rewards[name] = RewardTermCfg(func=globals()[name], weight=w, params={"command_name": "attack"})
        return cfg
    return f

def _base():
    import r1_spar_task as RS
    return RS._footwork(RS._guard(RS._cfg(2)))

TASKS = {"R1-Spar-Attack": lambda: _attack(_base()), "R1-Spar-Block-A": lambda: _block(_base(), "A"), "R1-Spar-Block-B": lambda: _block(_base(), "B"),
         "R1-Spar-Block-Blind": lambda: _block(_base(), "A", blind=True), "R1-Spar-Block-Blind-B": lambda: _block(_base(), "B", blind=True)}

def register():
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.velocity.rl import VelocityOnPolicyRunner
    from src.tasks.velocity.config.r1.rl_cfg import unitree_r1_ppo_runner_cfg
    for tid, mk in TASKS.items():
        make = mk(); rl = unitree_r1_ppo_runner_cfg(); rl.experiment_name = "r1_spar"
        register_mjlab_task(task_id=tid, env_cfg=make(), play_env_cfg=make(play=True), rl_cfg=rl, runner_cls=VelocityOnPolicyRunner)

# =============================================================================================== widen (3.1)
def widen_norm(critic=False):
    """Normaliser (mean, std) for the new inputs: positions 0 / 0.5, glove velocities 0 / 2.0, yaw pair (0, -1) / 0.5, masks
    0.5 / 0.5; the critic also strike phase 0 / 1."""
    mean = [0.0] * 15 + [0.0] * 6 + [0.0, -1.0] + [0.5] * 5 + ([0.0] * PHASE_DIM if critic else [])
    std = [0.5] * 15 + [2.0] * 6 + [0.5, 0.5] + [0.5] * 5 + ([1.0] * PHASE_DIM if critic else [])
    return mean, std

def widen(src, dst):
    """Checkpoint for R1-Spar-Block-* from the warm start (rsl_rl 5.0.1 layout): actor 28 columns at 83, critic 38 at 98, first
    layer weights zero, Adam exp_avg / exp_avg_sq zero for those columns (matched by shape, asserted unique), normaliser
    mean/var/std of widen_norm (count 7.4e8: they stay put). The widened policy acts exactly as the warm start."""
    import torch
    c = torch.load(src, map_location="cpu", weights_only=False)
    def ins(x, at, vals): v = torch.as_tensor(vals, dtype=x.dtype).expand(*x.shape[:-1], len(vals)); return torch.cat([x[..., :at], v, x[..., at:]], -1)
    rep = {}
    for key, at, critic in (("actor_state_dict", ACTOR_DIM, False), ("critic_state_dict", CRITIC_DIM, True)):
        sd = c[key]; w = sd["mlp.0.weight"]; assert w.shape[1] == at, (key, tuple(w.shape)); mean, std = widen_norm(critic); n = len(mean)
        sd["mlp.0.weight"] = ins(w, at, [0.0] * n); sd["obs_normalizer._mean"] = ins(sd["obs_normalizer._mean"], at, mean)
        sd["obs_normalizer._std"] = ins(sd["obs_normalizer._std"], at, std); sd["obs_normalizer._var"] = ins(sd["obs_normalizer._var"], at, [s * s for s in std])
        hit = [st for st in c["optimizer_state_dict"]["state"].values() if tuple(st["exp_avg"].shape) == (w.shape[0], at)]
        assert len(hit) == 1, f"{key}: {len(hit)} Adam states of shape {(w.shape[0], at)}"
        for k in ("exp_avg", "exp_avg_sq"): hit[0][k] = ins(hit[0][k], at, [0.0] * n)
        rep[key] = f"{at} -> {at + n} (new inputs at {at}:{at + n}), count {float(sd['obs_normalizer.count']):.3g}"
    Path(dst).parent.mkdir(parents=True, exist_ok=True); torch.save(c, dst); print("widen:", json.dumps(rep), "iter", c.get("iter"))

def load_actor(path, device="cpu"):
    """The deterministic actor mean of an rsl_rl 5.0.1 checkpoint: (x - mean) / (std + 0.01) -> 512-256-128 ELU MLP (eps 0.01
    matches the exported ONNX to 1e-5 on model_7496.pt)."""
    import torch
    a = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]; mean, std = a["obs_normalizer._mean"], a["obs_normalizer._std"]
    L = [(a[f"mlp.{i}.weight"], a[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]
    def act(x):
        z = (x - mean) / (std + 1e-2)
        for i, (W, b) in enumerate(L): z = z @ W.T + b; z = torch.nn.functional.elu(z) if i < 3 else z
        return z
    return act

# =============================================================================================== M1 gate (2.6), on the pod
CALIB_FILE = HERE.parent / "results" / "sparring" / "m1_calibration.json"

def calibration_check(path=CALIB_FILE):
    """The preregistration's m1 gate item 'cpu_calibration: results/sparring/m1_calibration.json passes first', for the attacker
    this file trains with: the calibration chose a row and it is ATTACK's (constants_match). Else pass only with a reason in
    SPAR_CALIB_DEVIATION, which is then recorded as a deviation (m1_gate.json, the pod's progress file). Pure json, no torch."""
    att = (ATTACK["mass"], ATTACK["fmax"], ATTACK["delta"]); res = dict(file=str(path), attack=list(att))
    try: b = Path(path).read_bytes(); j = json.loads(b)
    except (OSError, ValueError) as ex: j, res["error"] = None, repr(ex)
    if j is not None:
        row = [r for r in j.get("rows", []) if r.get("geometry") == "proxy" and (r["mass"], r["fmax"], r["delta"]) == att]; rc = j.get("attack_recheck") or {}
        med = lambda r: ((r.get("hitmeter") or {}).get("impulse_per_head_hit_Ns") or {}).get("median")
        res.update(sha256_16=hashlib.sha256(b).hexdigest()[:16], chosen=j.get("chosen"), constants_match=j.get("constants_match"),
                   attack_row=dict(pass_=row[0].get("pass"), median_Ns=med(row[0]), checks=row[0].get("checks")) if row else None,
                   attack_recheck=dict(seeds=rc.get("seeds"), pass_=rc.get("pass"), median_Ns=med(rc), checks=rc.get("checks")) if rc else None)
    ch = res.get("chosen"); ok = bool(ch) and (ch["mass"], ch["fmax"], ch["delta"]) == att and res.get("constants_match") is True
    dev = os.environ.get("SPAR_CALIB_DEVIATION", "").strip(); res["match"] = ok; res["deviation"] = None if ok or not dev else dev
    res["pass"] = bool(ok or (dev and j is not None)); return res

def m1check(checkpoint, num_envs=4096, steps=1000, out="m1_gate.json", device="cuda:0", calibration=CALIB_FILE):
    """Gate 1: R1-Spar-Attack with the frozen warm start (deterministic mean, normaliser applied), push_robot and shove removed
    (the CPU 7.72 is 'no scripted hits'), corruption and DR on, every env on the CPU outer loops (as the CPU eval). Pooled
    head clean strikes per minute (count / env-minutes) >= 7.7 and median impulse per head-landed hit in [1.58, 3.56] N*s;
    and calibration_check on the calibration file (recorded; part of the pass)."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    cfg = _attack(_base(), outer=1.0, check=True)(play=False); cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device); actor = load_actor(checkpoint, device)
    obs, _ = env.reset(); cmd = env.command_manager.get_term("attack"); cmd.check_root(); cmd.hits.reset_totals(); falls = 0; strikes0 = 0.0; t0 = time.time()
    with torch.no_grad():
        for _ in range(steps):
            obs, _, term, _, _ = env.step(actor(obs["actor"])); falls += int(term.sum())
    cmd.hits.flush(cmd._t_ep()); T = {k: float(v) for k, v in cmd.hits.tot.items()}; mins = steps * num_envs * env.step_dt / 60.0
    med = cmd.hits.median_head_impulse(); rate = T["head_clean"] / mins
    res = dict(task="R1-Spar-Attack (m1check: outer loops on every env, no push_robot / shove)", checkpoint=str(checkpoint), num_envs=num_envs, steps=steps,
               env_minutes=round(mins, 1), shell_proxy=os.environ.get("SPAR_SHELL_PROXY", "mesh"), gravcomp=os.environ.get("SPAR_GRAVCOMP", "1") != "0",
               attack=dict(ATTACK), head_clean_pm=round(rate, 3), torso_clean_pm=round(T["torso_clean"] / mins, 3), head_landed_pm=round(T["head_landed"] / mins, 3),
               median_impulse_per_head_landed_Ns=round(med, 3), mean_impulse_per_head_landed_Ns=round(T["head_imp_landed"] / max(T["head_landed"], 1.0), 3),
               impulse_ring_n=int(cmd.hits.ring_n), blocked_share=round(T["blocked"] / max(T["windows"], 1.0), 3), windows=int(T["windows"]),
               head_landed_vneg_share=round(T["head_landed_vneg"] / max(T["head_landed"], 1.0), 4),   # signed closing speed < 0 (FORCE_SIGN check; clean uses |v.n|)
               falls_per_min=round(falls / mins, 4), wall_s=round(time.time() - t0, 1),
               pass_head_clean=rate >= CALIB_PASS["head_clean_pm"], pass_impulse=CALIB_PASS["impulse_Ns"][0] <= med <= CALIB_PASS["impulse_Ns"][1])
    res["calibration"] = cal = calibration_check(calibration); res["pass_calibration"] = cal["pass"]
    res["pass"] = bool(res["pass_head_clean"] and res["pass_impulse"] and cal["pass"]); j = {"m1": res}
    if cal["deviation"]: j["deviations"] = [f"calibration: ATTACK {cal['attack']} is not the calibration's chosen row {cal.get('chosen')}: {cal['deviation']}"]
    json.dump(j, open(out, "w"), indent=1); print("M1CHECK", json.dumps(res)); return res

def gate(m1_json, smoke_log, proxy=None, num_envs=None, max_iter_s=2.0, first=20, prereg_envs=4096):
    """Gate 2 from the 200-iteration smoke log: mean 'Iteration time' over iterations first..end and no nan termination
    (Episode_Termination/nan > 0 anywhere). Merged into the m1 gate file with gate 1; exit code 0 iff both pass. Also logged:
    iteration first's time and the mean 'Collection time' (env stepping: a launch-bound miss shows there, not in learning);
    a pass at num_envs other than the preregistered 4096 (2.6's last speed fallback) is listed under 'deviations'."""
    import re
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(smoke_log, errors="replace").read())
    its = [float(x) for x in re.findall(r"Iteration time:\s*([0-9.]+)\s*s", txt)]; col = [float(x) for x in re.findall(r"Collection time:\s*([0-9.]+)\s*s", txt)]; nans = [float(x) for x in re.findall(r"Episode_Termination/nan:\s*([0-9.eE+-]+)", txt)]
    sel = its[first - 1:]; mean_it = float(np.mean(sel)) if sel else float("nan")
    j = json.load(open(m1_json)) if Path(m1_json).exists() else {}
    j.setdefault("smoke", []).append(dict(log=str(smoke_log), iterations=len(its), mean_iter_s=round(mean_it, 3), iter_s_at_first=its[first - 1] if len(its) >= first else None,
                                          mean_collection_s=round(float(np.mean(col[first - 1:])), 3) if col[first - 1:] else None, nan_terminations=float(sum(nans)), nan_logged=len(nans),
                                          shell_proxy=proxy or os.environ.get("SPAR_SHELL_PROXY", "mesh"), num_envs=num_envs,
                                          pass_speed=bool(sel) and mean_it <= max_iter_s, pass_nan=bool(nans) and sum(nans) == 0))
    s = j["smoke"][-1]; s["pass"] = bool(s["pass_speed"] and s["pass_nan"]); j["pass"] = bool(j.get("m1", {}).get("pass", False) and s["pass"])
    if s["pass"] and num_envs and num_envs != prereg_envs:
        s["deviation"] = d = f"speed gate passed at {num_envs} envs; prereg_m2.json m1_gate_before_training.speed says {prereg_envs}"
        if d not in j.setdefault("deviations", []): j["deviations"].append(d)
    json.dump(j, open(m1_json, "w"), indent=1); print("GATE", json.dumps(s), "overall", j["pass"]); return j

# =============================================================================================== CPU calibration (2.5)
_CAL = {}
def calibration_model(mass, fmax, geometry="proxy", cfg=None):
    """r1gym's R1 (mjlab contract, guard home) + the 16 proxies (renamed for HitMeter) + the attacker chain + the hit sensors,
    built with the same pure-MjSpec functions as the GPU task. geometry="shell": the eval's full hulls instead of proxies
    (every R1 mesh collides, as robot_contact: shell), no hit sensors (HitMeter only)."""
    import mujoco
    from r1gym.model import build_spec
    s = build_spec(cfg=cfg)
    if geometry == "proxy": add_shell_proxies(s, rename_full=True); proxy_bits(s, prefix="")
    else:
        for g in s.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH: g.contype, g.conaffinity, g.condim = 0, SHELL_BIT, 1; g.solref = list(SHELL_SOLREF); g.solimp = list(SHELL_SOLIMP)
    attacker_spec(mass, fmax, actuators=True, spec=s)
    if geometry == "proxy": add_contact_sensors_cpu(s)
    return s.compile()

class _CpuAttacker:
    """The AttackerEntity's per-substep target law on a CPU MjData (substep hook: runs after each mj_step, sets the next one)."""
    def __init__(self, m, mass):
        self.act = np.array([m.actuator(j).id for j in ATT_JOINTS]); self.qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in ATT_JOINTS])
        self.va = np.array([m.jnt_dofadr[m.joint(j).id] for j in ATT_JOINTS]); self.sl = np.array([j[-1] in "xyz" for j in ATT_JOINTS])
        self.ff = slide_kv(mass) / ATTACK["kp"]; self.q_prev = self.q_next = np.zeros(10); self.k = 0; self.d = None
    def place(self, d, q): self.d = d; d.qpos[self.qa] = q; d.qvel[self.va] = 0.0; self.q_prev = self.q_next = q.copy(); self.k = 0; self._write()
    def set(self, q): self.q_prev, self.q_next, self.k = self.q_next, q.copy(), 0; self._write()
    def _write(self):
        dq = self.q_next - self.q_prev; tgt = self.q_prev + min(1.0, (self.k + 1) / DECIM) * dq; tgt[self.sl] += self.ff * dq[self.sl] / CTRL_DT
        self.d.ctrl[self.act] = tgt
    def reset(self): pass
    def substep(self, m, d): self.k += 1; self._write()

class _CpuSensors:
    """Substep hook: the hit_<s>_<link>_shell_force sensors after each mj_step -> (2, 16, 3) per substep."""
    def __init__(self, m):
        self.adr = np.array([[m.sensor(f"hit_{s}_{l}_shell_force").adr[0] for l in LINKS] for s, _ in SIDES]); self.buf = []
    def reset(self): self.buf = []
    def substep(self, m, d): self.buf.append(d.sensordata[self.adr[..., None] + np.arange(3)].copy())
    def take(self):
        F = np.array(self.buf); self.buf = []; return F   # (S, 2, 16, 3)

def _cal_bout(args):
    """One CPU bout: R1HitEnv (warm-start ONNX, zero command: no FightReplay), AttackSchedule (torch CPU, seeded) driving the
    attacker, HitMeter (+ HitAccount on the proxies' sensors). Returns per-bout counts and samples."""
    import mujoco, torch
    seed, mass, fmax, delta, geometry, secs, onnx = args
    from r1gym.env import R1HitEnv
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.hits import HitMeter
    torch.set_num_threads(1); key = (mass, fmax, geometry)
    if "ctrl" not in _CAL: _CAL["ctrl"] = load_controller(f"w={onnx}"); _CAL["cfg"] = controller_cfg(_CAL["ctrl"]); _CAL["lib"] = load_library()
    ctrl, cfg = _CAL["ctrl"], _CAL["cfg"]
    if key not in _CAL: _CAL[key] = calibration_model(mass, fmax, geometry, cfg)
    m = _CAL[key]; e = R1HitEnv(seed=seed, cfg=cfg, model=m, episode_s=secs, max_impulse=0, train=False)
    hm = HitMeter(m, cfg.get("hit_metric")); drv = _CpuAttacker(m, mass); sens = _CpuSensors(m) if geometry == "proxy" else None
    e.substep_hooks += [hm] + ([sens] if sens else []) + [drv]
    rng = np.random.default_rng(seed); o = e.reset(hits=[], delay=int(rng.integers(0, 3))); d = e.d
    sch = AttackSchedule(_CAL["lib"], 1, "cpu", CTRL_DT, gen=torch.Generator().manual_seed(seed), p={"delta": delta})
    ha = HitAccount(1, "cpu", zone_index([l + "_shell" for l in LINKS])) if sens else None
    fb = [m.body(b).id for b in FISTS]
    def state():
        q = d.qpos[e.r.free_q:e.r.free_q + 7]; w, x, y, z = q[3:]
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)[None]
        return T(q[:2]), T([np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])[0], T(d.xpos[e.r.torso]), T(d.xmat[e.r.torso].reshape(3, 3))
    pel, yaw, xT, RT = state(); g, el, _, _ = sch.reset(torch.tensor([0]), pel, yaw); drv.place(d, joint_targets(g, el)[0].numpy()); mujoco.mj_forward(m, d)
    wins, vref, vact, prev, fell, seen = [], [], [], g[0].numpy().copy(), None, np.zeros(2, int)
    T3 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)[None]; sl = lambda q: T3(q.reshape(2, 5)[:, :3])
    for k in range(e.T):
        t = k * CTRL_DT; pel, yaw, xT, RT = state()
        g, el, _, _ = sch.step(torch.tensor([t]), pel, yaw, xT, RT)
        for h in np.flatnonzero(sch.strike_id[0].numpy() != seen): wins.append((round(float(sch.win_on[0, h]), 4), round(float(sch.win_off[0, h]), 4), ("l_hand", "r_hand")[h]))
        seen = sch.strike_id[0].numpy().copy(); gn = g[0].numpy(); vref += list(np.linalg.norm(gn - prev, axis=1) / CTRL_DT); prev = gn.copy()
        vact += list(np.linalg.norm(d.qvel[drv.va].reshape(2, 5)[:, :3], axis=1))
        drv.set(joint_targets(g, el, torch.as_tensor(d.qpos[drv.qa], dtype=torch.float32)[None])[0].numpy())
        if ha is not None:   # the step's start, for HitAccount's per-substep closing speed (as AttackCommand.hit_update saves it)
            v6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, e.r.torso, v6, 0)
            hs = (sl(d.qpos[drv.qa]), sl(d.qvel[drv.va]), T3(v6[3:]))
        o, _, _, info = e.step(ctrl.act(o[None])[0]); mujoco.mj_forward(m, d)
        if ha is not None:
            Fz, Fv = ha.zone_forces(torch.as_tensor(sens.take(), dtype=torch.float32).permute(1, 2, 0, 3).flip(-2))   # (2, 16, S, 3), newest first as mjlab
            vr = fist_substep_vel(hs[0], hs[1], sl(drv.q_prev), sl(drv.q_next), mass, fmax) - hs[2][:, None, :, None]
            ha.process(Fz[None], torch.tensor([t + CTRL_DT]), sch.strike_id, sch.win_on, sch.win_off, closing_speeds(vr, Fv[None]), closing_speeds(vr, Fv[None], signed=True))
        if info["fallen"]: fell = (k + 1) * CTRL_DT; break
    up = fell or secs; row = dict(seed=seed, fell=fell, secs=up, hm=hm.summary(up, wins=wins), strikes=float(sch.n_strikes[0]),
                                  vref_all=np.array(vref, np.float32), vact_all=np.array(vact, np.float32))
    if ha is not None:
        ha.flush(torch.tensor([up])); row["ha"] = {k: float(v[0]) for k, v in ha.ep.items()}; row["ha_imp"] = ha.ring[:int(ha.ring_n)].tolist()
    return row

def calibrate(bouts=64, secs=20.0, workers=8, out=None, onnx=None, grid=None, shell_check=True, recheck=True):
    """2.5: the CALIB_GRID rows in order (or `grid`, indices), 64 bouts x 20 s each (seeds 0..63); the first that passes wins:
    HitMeter median impulse per head hit in [1.58, 3.56] N*s, head clean / min >= 7.7, glove speed p98 >= 0.85 x the reference
    p98, HitAccount within +-15% of HitMeter on head clean / min and +-10% on the median impulse. The winner (else the first row)
    is rerun on the full hulls (geometry shell: reported, not gated). recheck: ATTACK's row again on the next seed set (seeds
    bouts..2 bouts - 1; reported as attack_recheck, it never changes the choice): the median moves between seed sets by about
    as much as the margins to 3.56. Writes `out` (results/sparring/m1_calibration.json)."""
    import multiprocessing as mp
    from r1gym.hits import landed_block
    root = HERE.parent; onnx = onnx or str(root / "results" / "policy_mjlab_guard_footwork_r" / "policy.onnx")
    out = out or str(root / "results" / "sparring" / "m1_calibration.json"); P = CALIB_PASS
    ctx = mp.get_context("spawn"); pool = [ctx.Pool(workers)]
    def run(mass, fmax, delta, geometry, seed0=0):
        t0 = time.time(); args = [(s, mass, fmax, delta, geometry, secs, onnx) for s in range(seed0, seed0 + bouts)]
        for attempt in range(3):   # Windows spawn pools occasionally lose a worker (WinError 6) and hang: time out, rebuild, retry
            try: rows = pool[0].map_async(_cal_bout, args, chunksize=1).get(timeout=60 + 3 * secs * bouts / workers); break
            except Exception as ex:
                print("calibration pool retry:", repr(ex), flush=True); pool[0].terminate(); pool[0] = ctx.Pool(workers)
        else: raise RuntimeError("calibration pool failed 3 times")
        blk = landed_block([r["hm"] for r in rows], [(i, 0) for i in range(len(rows))]); mins = sum(r["secs"] for r in rows) / 60
        vr, va = np.concatenate([r["vref_all"] for r in rows]), np.concatenate([r["vact_all"] for r in rows])
        res = dict(mass=mass, fmax=fmax, delta=delta, geometry=geometry, bouts=bouts, secs=secs, seeds=[seed0, seed0 + bouts - 1], wall_s=round(time.time() - t0, 1),
                   survival=round(float(np.mean([r["fell"] is None for r in rows])), 3), minutes_up=round(mins, 2),
                   hitmeter=dict(head_clean_pm=blk["head_clean_pm"]["rate"], head_clean_ci95=blk["head_clean_pm"]["ci95"], torso_clean_pm=blk["torso_clean_pm"]["rate"],
                                 head_landed_pm=blk["head_landed_pm"]["rate"], impulse_per_head_hit_Ns=blk["impulse_per_head_hit_Ns"], windows=blk.get("windows")),
                   strikes_pm=round(sum(r["strikes"] for r in rows) / mins, 2),
                   glove_speed_p98=dict(reference=round(float(np.percentile(vr, 98)), 3), actual=round(float(np.percentile(va, 98)), 3)))
        res["speed_ratio"] = round(res["glove_speed_p98"]["actual"] / max(res["glove_speed_p98"]["reference"], 1e-9), 3)
        if geometry == "proxy":
            H = {k: sum(r["ha"][k] for r in rows) for k in EP_KEYS}; imp = [x for r in rows for x in r["ha_imp"]]
            res["hitaccount"] = dict(head_clean_pm=round(H["head_clean"] / mins, 3), torso_clean_pm=round(H["torso_clean"] / mins, 3), head_landed_pm=round(H["head_landed"] / mins, 3),
                                     median_impulse_per_head_landed_Ns=round(float(np.median(imp)), 3) if imp else None, n_head_landed=len(imp),
                                     blocked_share=round(H["blocked"] / max(H["windows"], 1), 3), head_landed_vneg_share=round(H["head_landed_vneg"] / max(H["head_landed"], 1), 4))
            hm_r, hm_i, ha_r, ha_i = res["hitmeter"]["head_clean_pm"], res["hitmeter"]["impulse_per_head_hit_Ns"].get("median"), res["hitaccount"]["head_clean_pm"], res["hitaccount"]["median_impulse_per_head_landed_Ns"]
            res["checks"] = dict(impulse=hm_i is not None and P["impulse_Ns"][0] <= hm_i <= P["impulse_Ns"][1], head_clean=hm_r >= P["head_clean_pm"],
                                 speed=res["speed_ratio"] >= P["speed_ratio"], agree_rate=hm_r > 0 and abs(ha_r - hm_r) <= P["agree_rate"] * hm_r,
                                 agree_impulse=bool(hm_i and ha_i) and abs(ha_i - hm_i) <= P["agree_impulse"] * hm_i)
            res["pass"] = all(res["checks"].values())
        print("CALIB", json.dumps(res), flush=True); return res
    rows, chosen = [], None
    for i in (grid if grid is not None else range(len(CALIB_GRID))):
        r = run(*CALIB_GRID[i], "proxy"); rows.append(r)
        if r["pass"]: chosen = r; break
    ref = chosen or rows[0]; shell = run(ref["mass"], ref["fmax"], ref["delta"], "shell") if shell_check else None
    again = run(ATTACK["mass"], ATTACK["fmax"], ATTACK["delta"], "proxy", seed0=bouts) if recheck else None; pool[0].close(); pool[0].join()
    res = dict(rule=dict(grid=CALIB_GRID, pass_=P, reference="policy_mjlab_guard_footwork_r/summary_val_hard.json: 2.37 N*s per head hit; x1.5 window"),
               onnx=onnx, rows=rows, chosen=None if chosen is None else dict(mass=chosen["mass"], fmax=chosen["fmax"], delta=chosen["delta"]), shell_geometry=shell, attack_recheck=again,
               constants_match=chosen is not None and (chosen["mass"], chosen["fmax"], chosen["delta"]) == (ATTACK["mass"], ATTACK["fmax"], ATTACK["delta"]))
    Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    print("CHOSEN", res["chosen"], "constants match ATTACK:" if chosen else "no row passed", res["constants_match"] if chosen else "")
    return res

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts")); sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent))
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "widen": widen(sys.argv[2], sys.argv[3]); sys.exit(0)
    if cmd == "calcheck":
        r = calibration_check(sys.argv[2] if len(sys.argv) > 2 else CALIB_FILE); print("CALCHECK", json.dumps(r)); sys.exit(0 if r["pass"] else 1)
    if cmd in ("calibrate", "m1check", "gate"):
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("cmd")
        if cmd == "calibrate":
            ap.add_argument("--bouts", type=int, default=64); ap.add_argument("--secs", type=float, default=20.0); ap.add_argument("--workers", type=int, default=8)
            ap.add_argument("--out"); ap.add_argument("--onnx"); ap.add_argument("--grid", type=int, nargs="*"); ap.add_argument("--shell_check", type=int, default=1)
            ap.add_argument("--recheck", type=int, default=1)
            a = ap.parse_args(); calibrate(a.bouts, a.secs, a.workers, a.out, a.onnx, a.grid, bool(a.shell_check), bool(a.recheck)); sys.exit(0)
        if cmd == "gate":
            ap.add_argument("--m1", required=True); ap.add_argument("--smoke-log", required=True); ap.add_argument("--proxy"); ap.add_argument("--num-envs", type=int)
            a = ap.parse_args(); sys.exit(0 if gate(a.m1, a.smoke_log, a.proxy, a.num_envs)["pass"] else 1)
        ap.add_argument("--checkpoint", required=True); ap.add_argument("--num-envs", type=int, default=4096); ap.add_argument("--steps", type=int, default=1000)
        ap.add_argument("--out", default="m1_gate.json"); ap.add_argument("--calibration", default=str(CALIB_FILE)); a = ap.parse_args()
        import mjlab.tasks, src.tasks  # noqa: F401
        sys.exit(0 if m1check(a.checkpoint, a.num_envs, a.steps, a.out, calibration=a.calibration)["pass"] else 1)
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
