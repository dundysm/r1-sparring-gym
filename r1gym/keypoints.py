"""Opponent keypoints for the sparring defender (docs/SPARRING_M1M2.md section 1, the shared contract; section 4, the CPU eval).

The defender `R1-Spar-Block-*` sees a 28-d block appended after its 83 proprioceptive inputs: five opponent keypoints in its
own torso frame (T), the two gloves' velocities, the opponent's relative yaw and five presence masks. The GPU task builds
the same block from its attacker (gpu/sparring_task.py keypoint_block); this module is the CPU side, read from the real
tracker opponent (bodies prefixed opp_). Every constant here is pinned by section 1 and must change there first.

  KeypointSource(m, "truth", delay)   the eval's source: exact state on its own MjData twin, mask = in the head camera's view,
                                      no noise or dropout (corrupt=True: the GPU training corruption of 3.2, for studies);
                                      delayed by the episode's 0-2 control steps exactly as DelayedNoisyObs delays the 83
  KeypointSource(m, "camera", delay)  M3: gloves from the head camera's gloves detector; the interface is fixed, the body raises."""
from __future__ import annotations
import numpy as np, mujoco
from .sensors import DelayedNoisyObs

# ---- 1.2 keypoints: order, names, CPU bodies and offsets (body frame, m). Left/right are the opponent's own sides.
KP_NAMES = ("l_glove", "r_glove", "l_elbow", "r_elbow", "head")
KP_BODIES = {"l_glove": ("opp_left_wrist_roll_link", (0.08, 0.0, 0.0)), "r_glove": ("opp_right_wrist_roll_link", (0.08, 0.0, 0.0)),
             "l_elbow": ("opp_left_elbow_link", (0.0, 0.0, 0.0)), "r_elbow": ("opp_right_elbow_link", (0.0, 0.0, 0.0)),
             "head": ("opp_torso_link", (0.01, 0.0, 0.357))}   # = r1gym/perception/render_scene.py KEYPOINTS l_fist, l_elbow, head (rigid head)
OPP_YAW_BODY, DEF_BODY = "opp_torso_link", "torso_link"   # opponent yaw: heading of its torso x axis; T = the defender's torso_link
ELBOW_TO_FIST = 0.195   # GPU: elbow keypoint = attacker body origin - 0.195 x its body x axis (the R1's elbow-origin-to-fist distance)
N_KP, GLOVES, HEAD = 5, (0, 1), 4

# ---- 1.3 the block (float32)
KP_DIM = 28
POS, VEL, YAW, MASK = slice(0, 15), slice(15, 21), slice(21, 23), slice(23, 28)
CTRL_DT = 0.02   # velocities: one control step apart

# ---- 1.4 ONNX metadata
OBS_NAMES = "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"   # the 83 of every shipped mjlab policy
OBS_NAMES_SIGHTED, OBS_NAMES_BLIND = OBS_NAMES + ",opp_keypoints", OBS_NAMES + ",opp_keypoints_blind"
PROPRIO_DIM = 83

# ---- 1.1 the head camera, as numbers (configs/r1_camera.yaml: HeadCameraSpec().mount_torso, K(); tests check they agree)
CAM_C = np.array([0.1310, 0.0, 0.2473])                       # lens in T [m]
CAM_R = np.array([[0.0, -0.7660, 0.6428], [-1.0, 0.0, 0.0], [0.0, -0.6428, -0.7660]])   # columns x_c, y_c, z_c (OpenCV) in T
CAM_F, CAM_CX, CAM_CY, CAM_ZMIN = 50.346, 80.0, 60.0, 0.05     # fx = fy [px], principal point, near limit [m]

# ---- 3.2 GPU training corruption (corrupt=True here: off for the preregistered eval)
CORRUPT = dict(pos=0.02, bias=0.02, vel=0.25, yaw=0.1, drop=0.10)

def wrap(a): return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

def heading(R):
    """Yaw [rad] of a body's x axis on the ground, from its rotation matrix (3, 3)."""
    R = np.asarray(R); return np.arctan2(R[..., 1, 0], R[..., 0, 0])

def to_torso(P_w, x_T, R_T):
    """World points (..., 3) -> the defender's torso frame T, given the torso origin x_T and rotation R_T (columns = T axes)."""
    return (np.asarray(P_w, float) - x_T) @ np.asarray(R_T, float)

def in_view(P_T):
    """1.1: True where a point in T is inside the nominal head camera's 160 x 120 pinhole (FOV only, no self-occlusion)."""
    q = (np.asarray(P_T, float) - CAM_C) @ CAM_R; z = q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (z > CAM_ZMIN) & (np.abs(q[..., 0] / z) <= CAM_CX / CAM_F) & (np.abs(q[..., 1] / z) <= CAM_CY / CAM_F)

def glove_vel(P_T, P_T_prev):
    """Glove velocities (..., 2, 3) in T from true positions one control step apart; zeros when P_T_prev is None (first step)."""
    P_T = np.asarray(P_T, float)
    return np.zeros(P_T.shape[:-2] + (2, 3)) if P_T_prev is None else (P_T[..., :2, :] - np.asarray(P_T_prev, float)[..., :2, :]) / CTRL_DT

def pack(P, V, rel_yaw, mask):
    """The 28-d block from (possibly corrupted) positions (..., 5, 3), glove velocities (..., 2, 3), relative yaw (...) and
    masks (..., 5): zero-plus-mask (a keypoint with mask 0 has its position, and for gloves its velocity, zeroed; the yaw
    pair is present when any of the five keypoints is: docs/SPARRING_M1M2.md 8, the head alone is in view 36% of the time)."""
    on = np.asarray(mask) > 0.5; mask = on.astype(float)   # np.where, not a product: absent entries are +0.0, never -0.0
    P = np.where(on[..., :, None], np.asarray(P, float), 0.0); V = np.where(on[..., :2, None], np.asarray(V, float), 0.0)
    y = np.asarray(rel_yaw, float); sc = np.where(on.any(-1, keepdims=True), np.stack([np.sin(y), np.cos(y)], -1), 0.0)
    lead = P.shape[:-2]
    return np.concatenate([P.reshape(lead + (15,)), V.reshape(lead + (6,)), sc, mask], -1).astype(np.float32)

def assemble(P_T, P_T_prev, rel_yaw, mask):
    """1.3 without corruption: true positions in T (..., 5, 3), the previous step's (or None), wrap(yaw_opp - yaw_def_torso)
    and masks -> the (..., 28) float32 block. The reference the GPU's torch keypoint_block must match."""
    return pack(P_T, glove_vel(P_T, P_T_prev), rel_yaw, mask)

class KeypointSource:
    """Per env: reset(d) right after env.reset, observe(d) right after each env.step; both return the delayed 28-d block.

    Reads the env's MjData (qpos, mocap) only: kinematics run on its own MjData twin, because after mj_step the env's xpos
    are one substep stale while its qpos are current. delay: the episode's obs delay in control steps (ep["delay"]), applied
    by a zero-noise DelayedNoisyObs on a private rng, so the block lags exactly as the 83 proprioceptive dims do.
    fov=False: every keypoint present (mask 1) wherever it is. corrupt=True (needs rng): 3.2's training corruption, in the
    contract's order (true positions -> velocities -> noise and dropout -> mask -> delay); off for every preregistered eval."""
    MODES = ("truth", "camera")
    def __init__(self, m, mode="truth", delay=0, fov=True, corrupt=False, rng=None):
        if mode not in self.MODES: raise ValueError(f"keypoint source {mode!r}: one of {self.MODES}")
        if corrupt and rng is None: raise ValueError("corrupt=True needs an rng")
        self.m, self.mode, self.fov, self.corrupt, self.rng = m, mode, bool(fov), bool(corrupt), rng
        self.dk = mujoco.MjData(m); self.kb = np.array([m.body(KP_BODIES[k][0]).id for k in KP_NAMES])
        self.koff = np.array([KP_BODIES[k][1] for k in KP_NAMES], float); self.tb, self.ob = m.body(DEF_BODY).id, m.body(OPP_YAW_BODY).id
        self.line = DelayedNoisyObs(delay, np.zeros(KP_DIM), np.random.default_rng(0), uniform=True); self.prev = None; self.bias = np.zeros((N_KP, 3))

    def true_state(self, d):
        """(P_T (5, 3) true positions in T, rel_yaw [rad], in_view (5,)) at d's current qpos, computed on the twin."""
        if self.mode == "camera":   # M3: PerceptStream gloves (xyz in the pelvis heading frame -> T via mount_fk); elbows, head, yaw absent
            raise NotImplementedError("--keypoints camera: the gloves-detector source comes with M3 (docs/SPARRING_M1M2.md section 4)")
        k = self.dk; k.qpos[:] = d.qpos
        if self.m.nmocap: k.mocap_pos[:] = d.mocap_pos; k.mocap_quat[:] = d.mocap_quat
        mujoco.mj_kinematics(self.m, k)
        X, R = k.xpos[self.kb], k.xmat[self.kb].reshape(-1, 3, 3); P_w = X + np.einsum("kij,kj->ki", R, self.koff)
        RT = k.xmat[self.tb].reshape(3, 3); P_T = to_torso(P_w, k.xpos[self.tb], RT)
        yaw = float(wrap(heading(k.xmat[self.ob].reshape(3, 3)) - heading(RT)))
        return P_T, yaw, in_view(P_T) if self.fov else np.ones(N_KP, bool)

    def block(self, d):
        """The undelayed block at d's current state; advances the velocity memory (call once per control step)."""
        P_T, yaw, vis = self.true_state(d); V = glove_vel(P_T, self.prev); self.prev = P_T; P = P_T
        if self.corrupt:
            c, g = CORRUPT, self.rng
            P = P_T + self.bias + g.uniform(-c["pos"], c["pos"], (N_KP, 3)); V = V + g.uniform(-c["vel"], c["vel"], (2, 3))
            yaw = yaw + g.uniform(-c["yaw"], c["yaw"]); vis = vis & (g.random(N_KP) >= c["drop"])
        return pack(P, V, yaw, vis)

    def reset(self, d):
        self.prev = None
        if self.corrupt: self.bias = self.rng.uniform(-CORRUPT["bias"], CORRUPT["bias"], (N_KP, 3))   # per episode, per keypoint axis
        b = self.block(d); self.line.reset(b); return self.line(b).astype(np.float32)

    def observe(self, d): return self.line(self.block(d)).astype(np.float32)
