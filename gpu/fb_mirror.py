"""L/R mirror conjugation of the full-body opponent's tracker (M2V2_PLAN_v3 A6.2): on a mirrored L1 row, and on a mirrored strike,
the opponent is driven by a(obs) = M_a(pi(M_o(obs))), pi the tracker (T0, or T1 where it exists), so it performs a mirrored motion
as pi performs the unmirrored one. The mirror is the world reflection y -> -y (export_fb_library.mirror on qpos): left_* / right_*
joints swap, the roll and yaw joints (waist included) change sign. Pure torch + numpy on any device: the replica
(scripts/fb_replica.py), the mjlab task and the CPU tools wrap a TrackerDriver with conjugate(drv, mask), mask [n] per env (True:
that env's row is mirrored); the driver's own obs, facing, steering and `last` stay in the real world.
  M_o  the 132-d obs in Tracker.act's order: ref q, ref dq: joint map; the anchor position in the torso frame (x, -y, z); the two
       columns of R_torso^T R_ref, interleaved (c0 (x, -y, z), c1 (-x, y, -z)); the gyro (-x, y, -z); q - default, dq, last action:
       joint map (DEFAULT, KP and the caps are mirror-symmetric, so the joint targets mirror with the action).
  M_a  the joint map on the 24 actions.
Both are signed permutations: involutions, exact in every float type. World state (checks, references): qpos [.., 31] of a
floating-base R1: (x, -y, z), quaternion (w, -x, y, -z), joint map; qvel [.., 30]: linear (x, -y, z) (world frame), angular
(-x, y, -z) (local frame, a pseudovector), joint map; a reference (joint pos, joint vel, anchor pos, anchor quat) likewise."""
import numpy as np, torch

JOINTS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
          "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
          "waist_roll_joint", "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
          "left_wrist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
          "right_wrist_roll_joint")   # r1gym.model.CTRL_JOINTS (asserted in tests/test_fb_mirror.py)

def joint_map(names=JOINTS):
    """(perm, sign): joint i of the mirror = sign[i] x joint perm[i] of the source."""
    sw = lambda s: s.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_"); n = list(names)
    return np.array([n.index(sw(j)) for j in n]), np.array([-1.0 if ("_roll" in j or "_yaw" in j) else 1.0 for j in n])

JP, JS = joint_map(); NJ = len(JOINTS)
def _cat(*blocks):
    """[(perm or width, sign)] -> (perm, sign) over the concatenated vector."""
    p, s, o = [], [], 0
    for b, sg in blocks:
        b = np.arange(b) if np.isscalar(b) else np.asarray(b); p.append(o + b); s.append(np.broadcast_to(np.asarray(sg, float), b.shape)); o += len(b)
    return np.concatenate(p), np.concatenate(s)
V3, W3, Q4 = (1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, 1.0, -1.0)
OBS = _cat((JP, JS), (JP, JS), (3, V3), (6, (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)), (3, W3), (JP, JS), (JP, JS), (JP, JS))   # c0 at 0, 2, 4; c1 at 1, 3, 5
ACT, QPOS, QVEL = (JP, JS), _cat((3, V3), (4, Q4), (JP, JS)), _cat((3, V3), (3, W3), (JP, JS))
APOS, AQUAT = _cat((3, V3)), _cat((4, Q4))
assert len(OBS[0]) == 132 and len(QPOS[0]) == 31 and len(QVEL[0]) == 30

_T = {}
def _apply(x, ps):
    """x[..., perm] x sign, numpy or torch (any device / dtype); ps: one of the module's maps (cached by identity)."""
    p, s = ps
    if not torch.is_tensor(x): x = np.asarray(x); return x[..., p] * s.astype(x.dtype)
    k = (id(p), x.device, x.dtype)
    if k not in _T: _T[k] = (torch.as_tensor(p, device=x.device), torch.as_tensor(s, device=x.device, dtype=x.dtype))
    tp, ts = _T[k]; return x.index_select(-1, tp) * ts

def mirror_obs(o): return _apply(o, OBS)
def mirror_act(a): return _apply(a, ACT)
def mirror_joints(q): return _apply(q, ACT)
def mirror_qpos(q): return _apply(q, QPOS)
def mirror_qvel(v): return _apply(v, QVEL)
def mirror_ref(jp, jv, ap, aq):
    """A shown reference (joint pos, joint vel [.., 24], anchor pos [.., 3], anchor quat wxyz [.., 4]) of the mirrored motion."""
    return _apply(jp, ACT), _apply(jv, ACT), _apply(ap, APOS), _apply(aq, AQUAT)

class MirrorPolicy:
    """policy conjugated by the mirror on the envs where mask is True (mask None: every env). policy: any callable obs [n, 132] ->
    actions [n, 24] (TrackerMLP, the replica's ONNX actor). One policy call per step; rows are independent."""
    def __init__(self, policy, mask=None):
        self.policy, self.mask = policy, mask; self.dtype = getattr(policy, "dtype", None)
    def set(self, ids, value):
        """mask[ids] = value (envs reset onto a mirrored / unmirrored row)."""
        self.mask[ids] = torch.as_tensor(value, dtype=torch.bool, device=self.mask.device)
    @torch.no_grad()
    def __call__(self, obs):
        m = self.mask
        if m is None or bool(m.all()): return mirror_act(self.policy(mirror_obs(obs)))
        if not bool(m.any()): return self.policy(obs)
        a = self.policy(torch.where(m[:, None], mirror_obs(obs), obs)); return torch.where(m[:, None], mirror_act(a), a)

def conjugate(drv, mask=None):
    """Wrap drv.mlp (a TrackerDriver's policy, unwrapped first if already wrapped) by the mirror. mask: [n] bool, None = every env
    mirrored. Returns the MirrorPolicy (.set(ids, value) at resets)."""
    if mask is not None: mask = torch.as_tensor(mask, dtype=torch.bool, device=drv.device).reshape(drv.n).clone()
    pol = drv.mlp.policy if isinstance(drv.mlp, MirrorPolicy) else drv.mlp; drv.mlp = MirrorPolicy(pol, mask); return drv.mlp
