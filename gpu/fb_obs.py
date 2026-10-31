"""Full-body M2 v2 defender observation (M2V2_PLAN_v3 2.3, build order M4): the eval's 111-d actor input in batched torch. Pure
torch + numpy; the mjlab observation term (M9) and the CPU replica (M6) both drive it.

keypoints     r1gym/keypoints.py KeypointSource('truth') on the opponent's real links (KP:18-21): world points from the bodies'
              poses, the defender's torso frame T, the head camera's FOV mask, the opponent's relative yaw (ST:305-344). Glove
              velocities from the previous control step's T-frame points (KeypointHistory; 0 at an episode's first obs). Poses are
              (xpos, xquat) as mjlab hands them (body_link_pos_w / _quat_w); quat2mat is mju_quat2Mat bit for bit. float64 runs
              numpy's arithmetic on this machine: its small matmuls (to_torso, in_view) are FMA chains, emulated exactly by fma(),
              its einsum (the body offsets) is a plain left-to-right sum, so the float32 block equals the eval's bit for bit (M4
              G0-5). float32 (GPU): plain matmuls.
proprio83     R1HitEnv._raw_obs under the mjlab contract (ENV:50-58): base ang vel, projected gravity, command, phase, q - home, dq
              and the applied action (the `actions` slot = HybridDelayedAction.applied, ENV:96, :125), cast to float32.
EvalActorObs  one class term for the actor, 111 dims (the 83 in r1gym order, then the 28-d block): a ring of the last 3 clean rows,
              ONE obs lag k per env and episode shared by all 111 dims (DelayedNoisyObs, SEN:9-13), uniform noise after the delay
              (the 83's half-widths, 0 on the keypoints). reset(ids) only flags a lazy pre-fill and drops the keypoint history
              (B3: at observation-manager reset positions are stale and the opponent is not placed yet). compute() runs after the
              forward and the command compute: a flagged env's ring takes k + 1 copies of its fresh row (the eval pre-fills with
              the true obs after placement and mj_forward), then the output is ring[k] + noise. k is read from the latency event's
              buffer (fb_events.Latency), drawn before every manager reset. mjlab's own obs delay stays off (delay_max_lag 0): its
              DelayBuffer sets a reset env's lag to 0 at observation reset (the lag-reset trap). The critic reads `clean`."""
import numpy as np, torch
from fb_opponent import quat2mat

# ---- the shared contract, as r1gym/keypoints.py (asserted equal in tests/test_fb_obs.py)
KP_NAMES = ("l_glove", "r_glove", "l_elbow", "r_elbow", "head")
KP_BODIES = ("left_wrist_roll_link", "right_wrist_roll_link", "left_elbow_link", "right_elbow_link", "torso_link")   # opponent bodies
KP_OFF = ((0.08, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.01, 0.0, 0.357))
DEF_BODY, OPP_YAW_KP = "torso_link", 4          # T = the defender's torso_link; opponent yaw = heading of its torso (keypoint 4's body)
N_KP, KP_DIM, PROPRIO_DIM, OBS_DIM, CTRL_DT = 5, 28, 83, 111, 0.02
POS, VEL, YAW, MASK = slice(0, 15), slice(15, 21), slice(21, 23), slice(23, 28)   # inside the block
CAM_C = (0.1310, 0.0, 0.2473); CAM_R = ((0.0, -0.7660, 0.6428), (-1.0, 0.0, 0.0), (0.0, -0.6428, -0.7660))
CAM_F, CAM_CX, CAM_CY, CAM_ZMIN = 50.346, 80.0, 60.0, 0.05
# the 83 (mjlab contract, r1gym/env.py _raw_obs): slices into a row
ANG, GRAV, CMD, PHASE, JPOS, JVEL, ACT = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 11), slice(11, 35), slice(35, 59), slice(59, 83)
PHASE_S = 0.6
RING = 3                                         # k in {0, 1, 2}

def noise_halfwidths(cfg):
    """The 111 uniform half-widths: R1HitEnv's mjlab-contract noise (command, phase, actions 0), then 28 zeros (keypoints: none)."""
    n = cfg["sensors"]["noise"]
    return np.concatenate([np.full(3, n["ang_vel"]), np.full(3, n["gravity"]), np.zeros(5), np.full(24, n["joint_pos"]), np.full(24, n["joint_vel"]), np.zeros(24), np.zeros(KP_DIM)])

def body_names(opp_prefix="opp_", def_prefix=""):
    """The 5 keypoint bodies (opponent) then T (defender), under the eval model's prefixes (mjlab: entity-relative, no prefix)."""
    return tuple(opp_prefix + b for b in KP_BODIES) + (def_prefix + DEF_BODY,)

# ---- exact arithmetic
def fma(a, b, c):
    """round(a x b + c) in float64 without hardware FMA: Dekker's exact product, TwoSum, one final rounding."""
    p = a * b; s1 = a * 134217729.0; ah = s1 - (s1 - a); al = a - ah; s2 = b * 134217729.0; bh = s2 - (s2 - b); bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl; s = p + c; bb = s - p; t = (p - (s - bb)) + (c - bb)
    return s + (t + e)

def rowmat(d, R):
    """d [..., 3] @ R [..., 3, 3] (R broadcast to d's rows). float64: numpy's BLAS order here, fma(d2, R2, fma(d1, R1, d0 R0))."""
    if d.dtype != torch.float64: return torch.matmul(d.unsqueeze(-2), R).squeeze(-2)
    x = d[..., 0:1] * R[..., 0, :]; x = fma(d[..., 1:2], R[..., 1, :], x); return fma(d[..., 2:3], R[..., 2, :], x)

def np_wrap(a):
    """numpy's (a + pi) % 2pi - pi (npy_divmod: fmod, then + 2pi where the sign differs; +0.0 for an exact multiple)."""
    b = 2 * np.pi; m = torch.fmod(a + np.pi, b); m = torch.where(m != 0, torch.where(m < 0, m + b, m), torch.zeros_like(m)); return m - np.pi

def heading(R): return torch.atan2(R[..., 1, 0], R[..., 0, 0])

# ---- keypoints
def kp_state(xpos, xquat):
    """Poses of the 6 bodies in body_names() order, xpos [N, 6, 3], xquat [N, 6, 4] -> (P_T [N, 5, 3] true keypoints in T,
    rel_yaw [N] = wrap(yaw_opp - yaw_T), in_view [N, 5]) = KeypointSource.true_state."""
    R = quat2mat(xquat); off = torch.as_tensor(KP_OFF, dtype=xpos.dtype, device=xpos.device); Rk = R[:, :N_KP]
    P_w = xpos[:, :N_KP] + ((Rk[..., 0] * off[:, 0:1] + Rk[..., 1] * off[:, 1:2]) + Rk[..., 2] * off[:, 2:3])   # numpy einsum's order
    RT = R[:, N_KP]; P_T = rowmat(P_w - xpos[:, N_KP, None], RT[:, None])
    yaw = np_wrap(heading(R[:, OPP_YAW_KP]) - heading(RT)); return P_T, yaw, in_view(P_T)

def in_view(P_T):
    """1.1: inside the nominal head camera's 160 x 120 pinhole (FOV only, no self-occlusion)."""
    t = lambda x: torch.as_tensor(x, dtype=P_T.dtype, device=P_T.device)
    q = rowmat(P_T - t(CAM_C), t(CAM_R)); z = q[..., 2]
    return (z > CAM_ZMIN) & ((q[..., 0] / z).abs() <= CAM_CX / CAM_F) & ((q[..., 1] / z).abs() <= CAM_CY / CAM_F)

def pack(P, V, rel_yaw, mask):
    """r1gym/keypoints.py pack: zero-plus-mask (absent entries +0.0), the yaw pair present when any keypoint is; float32."""
    on = mask if mask.dtype == torch.bool else mask > 0.5; z = torch.zeros((), dtype=P.dtype, device=P.device)
    P = torch.where(on[..., :, None], P, z); V = torch.where(on[..., :2, None], V, z)
    sc = torch.where(on.any(-1, keepdim=True), torch.stack([torch.sin(rel_yaw), torch.cos(rel_yaw)], -1), z)
    return torch.cat([P.reshape(*P.shape[:-2], 15), V.reshape(*V.shape[:-2], 6), sc, on.to(P.dtype)], -1).float()

class KeypointHistory:
    """Per env the previous control step's T-frame points (velocities are differences of two T-frame blocks, each in its own
    step's T, as KeypointSource). reset(ids): the next block has zero glove velocity (KeypointSource.reset: prev = None)."""
    def __init__(self, n, dtype=torch.float64, device="cpu"):
        self.prev = torch.zeros(n, N_KP, 3, dtype=dtype, device=device); self.first = torch.ones(n, dtype=torch.bool, device=device)
    def reset(self, ids): self.first[ids] = True; self.prev[ids] = 0
    def block(self, P_T, rel_yaw, mask):
        """The undelayed 28-d block; advances the memory (once per control step)."""
        prev = torch.where(self.first[:, None, None], P_T, self.prev); V = (P_T[:, :2] - prev[:, :2]) / CTRL_DT
        self.prev = P_T.clone(); self.first[:] = False; return pack(P_T, V, rel_yaw, mask)

# ---- the 83
def proprio83(root_quat, ang_vel, q, dq, home, command, t_step, last_a, dt=CTRL_DT):
    """R1HitEnv._raw_obs (mjlab contract) for N envs: root_quat [N, 4] (qpos), ang_vel [N, 3] (qvel, body frame), q, dq [N, 24],
    home [24], command [N, 3], t_step [N] control steps since reset, last_a [N, 24] the applied action -> [N, 83] float32.
    Gravity R^T (0, 0, -1) = -R[2, :] exactly (numpy's matvec with two zero terms)."""
    R = quat2mat(root_quat); grav = -R[:, 2, :]; f = torch.float64
    ph = 2 * np.pi * torch.fmod(torch.as_tensor(t_step, device=q.device).to(f) * dt, PHASE_S) / PHASE_S   # Python's float % (t >= 0)
    on = (torch.linalg.vector_norm(command.to(f), dim=-1) >= 0.1)[:, None]
    phase = torch.where(on, torch.stack([torch.sin(ph), torch.cos(ph)], -1), torch.zeros((), dtype=f, device=q.device))
    return torch.cat([ang_vel.to(f), grav.to(f), command.to(f), phase, q.to(f) - torch.as_tensor(home, dtype=f, device=q.device), dq.to(f), last_a.float().to(f)], -1).float()

class EvalActorObs:
    """The delayed, noisy 111-d actor obs for N envs (module docstring). latency: fb_events.Latency (k [N] drawn per episode).
    noise: 111 half-widths (noise_halfwidths). Rows are float32 (the eval's 83 and block are); out_dtype float32 (GPU) or float64
    (the eval adds float64 noise to the float32 83)."""
    def __init__(self, n, latency, noise, device="cpu", kp_dtype=torch.float64, out_dtype=torch.float32, generator=None):
        self.n, self.lat, self.device, self.out_dtype, self.gen = n, latency, device, out_dtype, generator
        self.noise = torch.as_tensor(noise, dtype=out_dtype, device=device); assert self.noise.shape == (OBS_DIM,)
        self.ring = torch.zeros(n, RING, OBS_DIM, dtype=torch.float32, device=device); self.clean = torch.zeros(n, OBS_DIM, dtype=torch.float32, device=device)
        self.prefill = torch.ones(n, dtype=torch.bool, device=device); self.hist = KeypointHistory(n, kp_dtype, device)
        self.k_ep = torch.zeros(n, dtype=torch.long, device=device); self.k_switches = torch.zeros((), dtype=torch.long, device=device)
        self.ar = torch.arange(n, device=device)

    def reset(self, ids):
        """Observation-manager reset: flag the lazy pre-fill, drop the keypoint history. Reads no state (stale here)."""
        self.prefill[ids] = True; self.hist.reset(ids)

    def row(self, proprio, P_T, rel_yaw, mask):
        """The clean 111-d rows (undelayed, no noise); advances the keypoint history."""
        return torch.cat([proprio.float(), self.hist.block(P_T, rel_yaw, mask)], -1)

    def compute(self, proprio, P_T, rel_yaw, mask, noise=None, blind=False):
        """After forward and command compute: push the fresh rows (pre-fill flagged envs), return ring[k] + noise [N, 111].
        noise: an explicit [N, 111] sample (checks replay the eval's); default uniform(-1, 1) x half-widths. blind: 28 zeros."""
        r = self.row(proprio, P_T, rel_yaw, mask)
        if blind: r[:, PROPRIO_DIM:] = 0
        k, f = self.lat.k, self.prefill
        self.k_switches += ((self.k_ep != k) & ~f).sum(); self.k_ep = torch.where(f, k, self.k_ep)   # one lag per episode (M10: zero switches)
        self.ring = torch.where(f[:, None, None], r[:, None].expand(-1, RING, -1), torch.cat([r[:, None], self.ring[:, :-1]], 1))
        self.prefill[:] = False; self.clean = r; o = self.ring[self.ar, self.k_ep].to(self.out_dtype)
        if noise is None: noise = (2 * torch.rand(self.n, OBS_DIM, dtype=self.out_dtype, device=self.device, generator=self.gen) - 1) * self.noise
        return o + noise
