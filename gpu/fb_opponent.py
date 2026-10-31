"""Full-body M2 v2 opponent driver (M2V2_PLAN_v3 2.2, build order M2): the eval's tracker opponent in batched torch, float64 or
float32 (m12: tf32 off). Pure torch + numpy; the mjlab CommandTerm (M9) and the CPU replica (M6) both drive it.

TrackerMLP     the shipped tracker's actor (results/opponent_tracker/model_13749.pt, R1-Track-KungFu-Steer2): (x - mean) / div ->
               132-512-256-128-24 ELU, deterministic mean. div = std + 0.01 summed in float32, the constant the tracker's export
               folded into its graph; float64 runs that graph's arithmetic on the float32 weights upcast.
Motion         gpu/motions/fb/fb_motion.npz (export_fb_motion.py, train clips only): the in-place clip q for placement and the
               reference the tracker is shown at clip frame f (bit-equal to the tracker's own at Tracker.frame(clip, f), M1).
TrackerDriver  r1gym/opponent.py Tracker.act and FightReplay.place for N envs, step for step:
  reset()  placement: q(t0) and q(t0 + 5 ms) of the in-place clip at the bout distance (qpos_at), qvel by mj_differentiatePos,
           optionally turned by a bout heading about a bout origin (the eval: origin 0, heading 0, bit-identical); last action 0,
           facing yaw and steering axis unset, spot = origin + R (d, 0).
  step()   clock fi = round(t x 50) + offset with t = t_ep + t0 + dt (float64 always); reference at fi mod T (or the aim's rq/rdq);
           facing: want = bearing to the target - pi, yaw += wrap(want - yaw) x min(1, dt / 0.5) (yaw = want at the first step);
           steering: axis fixed at the first step, spot chases target + dist x axis with a 0.5 s lag capped at 0.3 m/s; the 132-d
           obs in Tracker.act's order; a = MLP(obs); joint targets default + 0.25 cap / kp x a; last = a.
MjTracker      drop-in for the Tracker inside an r1gym R1HitEnv (one env): place(m, d, t) and act(m, d, replay, state, t, dt).
ArmKinematics  (M3) torso-frame FK and Jacobian of both arms (5 joints -> palm site) from a compiled model's offsets and axes.
AimOp          (M3) r1gym/aim.py's standoff+armwarp_vz (aimed) or shadow for N envs: intents, gates, StandOff, ArmWarp vz, logs;
               training-only jitter (JITTER) per intent, reach-corrected (plan A3.1 F', then the A2.1 guard). Feeds
               TrackerDriver.step(ref=(rq, rdq), dist=dist)."""
import json
from pathlib import Path
import numpy as np, torch

GPU = Path(__file__).resolve().parent; ROOT = GPU.parent
MOTION = GPU / "motions" / "fb" / "fb_motion.npz"
TRACKER_PT = ROOT / "results" / "opponent_tracker" / "model_13749.pt"
TRAIN = ("kfab_200", "kfab_201", "kfab_203")
FPS, CTRL_DT, TIMESTEP = 50.0, 0.02, 0.005
FACE_TAU, STEER_TAU, STEER_MAX = 0.5, 0.5, 0.3      # r1gym/opponent.py Tracker
OBS_DIM, ACT_DIM, NQ, NV = 132, 24, 31, 30
# the tracker's training contract (its export metadata; r1gym HOME, mjlab torque caps, opponent kp): asserted in tests/test_fb_opponent.py
DEFAULT = (-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0.35, -0.18, 0, 0.87, 0)
KP = (100, 100, 100, 100, 40, 40, 100, 100, 100, 100, 40, 40, 100, 100, 40, 40, 20, 20, 20, 40, 40, 20, 20, 20)
CAP = (60, 60, 60, 60, 50, 50, 60, 60, 60, 60, 50, 50, 60, 60, 60, 60, 33, 33, 33, 60, 60, 33, 33, 33)
SCALE = tuple(0.25 * c / k for c, k in zip(np.array(CAP, float), np.array(KP, float)))
MJ_MINVAL = 1e-15

def pin_precision():
    """m12: no TF32 anywhere, full float32 matmuls (asserted at P0 by precision_ok)."""
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")

def precision_ok():
    return torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False and torch.get_float32_matmul_precision() == "highest"

# ---- MuJoCo's quaternion arithmetic (wxyz, last dim 4), term order as engine_util_spatial.c
def qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

def qconj(q): return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])

def quat2mat(q):
    """mju_quat2Mat: [..., 3, 3]."""
    w, x, y, z = q.unbind(-1); ww, xx, yy, zz = w * w, x * x, y * y, z * z; wx, wy, wz, xy, xz, yz = w * x, w * y, w * z, x * y, x * z, y * z
    return torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy), 2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
                        2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], -1).reshape(*q.shape[:-1], 3, 3)

def _norm4(q):
    """mju_normalize4: identity below MINVAL, unchanged within MINVAL of unit length."""
    n = torch.linalg.vector_norm(q, dim=-1, keepdim=True); one = torch.zeros_like(q); one[..., 0] = 1.0
    return torch.where(n < MJ_MINVAL, one, torch.where((n - 1).abs() > MJ_MINVAL, q / n, q))

def differentiate_pos(q0, q1, h):
    """mj_differentiatePos for a free joint followed by hinges: [N, 31] x2 -> [N, 30]."""
    a, b = _norm4(q0[:, 3:7]), _norm4(q1[:, 3:7]); dq = qmul(qconj(a), b)      # mju_subQuat(b, a)
    ax = dq[:, 1:]; s = torch.linalg.vector_norm(ax, dim=-1, keepdim=True)
    ax = torch.where(s < MJ_MINVAL, ax.new_tensor([1.0, 0.0, 0.0]).expand_as(ax), ax / s)
    sp = 2 * torch.atan2(s, dq[:, :1]); sp = torch.where(sp > np.pi, sp - 2 * np.pi, sp)
    return torch.cat([(q1[:, :3] - q0[:, :3]) / h, ax * sp * (1 / h), (q1[:, 7:] - q0[:, 7:]) / h], 1)

def wrap(x): return torch.atan2(torch.sin(x), torch.cos(x))     # np.angle(np.exp(1j x))

class TrackerMLP(torch.nn.Module):
    def __init__(self, path=TRACKER_PT, dtype=torch.float32, device="cpu"):
        super().__init__(); a = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
        assert a["mlp.0.weight"].shape == (512, OBS_DIM) and a["mlp.6.weight"].shape == (ACT_DIM, 128) and "mlp.8.weight" not in a, "not the 132-d tracker actor"
        div = a["obs_normalizer._std"].float() + 0.01   # float32 sum, as exported
        self.register_buffer("mean", a["obs_normalizer._mean"].float().reshape(-1).to(device, dtype)); self.register_buffer("div", div.reshape(-1).to(device, dtype))
        self.W = [a[f"mlp.{i}.weight"].to(device, dtype) for i in (0, 2, 4, 6)]; self.b = [a[f"mlp.{i}.bias"].to(device, dtype) for i in (0, 2, 4, 6)]
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, x):
        z = (x.to(self.dtype) - self.mean) / self.div
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = torch.nn.functional.linear(z, W, b); z = torch.nn.functional.elu(z) if i < 3 else z
        return z

class Motion:
    """fb_motion.npz on `device`: q float64 (placement), the shown reference float32 (upcast exactly when used in float64)."""
    def __init__(self, path=MOTION, device="cpu"):
        with np.load(path, allow_pickle=False) as z: A = {k: z[k] for k in z.files}
        self.clips = tuple(A["clips"].tolist()); assert self.clips == TRAIN, f"fb_motion holds {self.clips}: train clips only"
        assert float(A["fps"]) == FPS
        t = lambda k, dt=None: torch.as_tensor(A[k], device=device, dtype=dt)
        self.frames, self.row0 = t("frames", torch.long), t("row0", torch.long); self.q = t("q", torch.float64)
        self.ref_q, self.ref_dq, self.ref_apos, self.ref_aquat = (t(k) for k in ("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"))
        self.clean, self.distances = (t("clean"), t("distances", torch.float64)) if "clean" in A else (None, None)
        self.strikes = {k[7:]: A[k] for k in A if k.startswith("strike_")}   # numpy: the strikes.json train rows (M1 (4))
        self.device = device

    def row(self, clip, fi): return self.row0[clip] + torch.remainder(fi, self.frames[clip])

    def qpos_at(self, clip, t, distance):
        """FightReplay.qpos_at at a bout distance (float64): linear between frames, quaternion hemisphere matched, renormalised."""
        T = self.frames[clip]; f = torch.fmod(t * FPS, T.double()); i = f.long(); a = (f - i)[:, None]; i = i % T
        q0 = self.q[self.row0[clip] + i].clone(); q1 = self.q[self.row0[clip] + (i + 1) % T].clone()
        q0[:, 0] += distance; q1[:, 0] += distance
        q1[:, 3:7] = torch.where(((q0[:, 3:7] * q1[:, 3:7]).sum(-1, keepdim=True) < 0), -q1[:, 3:7], q1[:, 3:7])
        q = (1 - a) * q0 + a * q1; q[:, 3:7] = q[:, 3:7] / torch.linalg.vector_norm(q[:, 3:7], dim=-1, keepdim=True); return q

class TrackerDriver:
    def __init__(self, n, dtype=torch.float32, device="cpu", motion=None, policy=TRACKER_PT, dt=CTRL_DT):
        pin_precision(); self.n, self.dtype, self.device, self.dt = n, dtype, device, dt
        self.M = motion or Motion(device=device); self.mlp = TrackerMLP(policy, dtype, device)
        f = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
        self.default, self.scale = torch.tensor(DEFAULT, dtype=dtype, device=device), torch.tensor(SCALE, dtype=dtype, device=device)
        self.clip, self.offset = torch.zeros(n, dtype=torch.long, device=device), torch.zeros(n, dtype=torch.long, device=device)
        self.t0, self.dist = torch.zeros(n, dtype=torch.float64, device=device), f(n)
        self.yaw, self.spot, self.axis, self.last = f(n), f(n, 2), f(n, 2), f(n, ACT_DIM)
        self.yaw_set, self.axis_set = torch.zeros(n, dtype=torch.bool, device=device), torch.zeros(n, dtype=torch.bool, device=device)

    def _ids(self, ids): return torch.arange(self.n, device=self.device) if ids is None else torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def reset(self, ids, clip, t0, distance, origin=None, heading=None):
        """Placement for envs `ids`: (qpos [k, 31], qvel [k, 30]) float64 in the world frame; driver state zeroed."""
        ids = self._ids(ids); k = len(ids); D = lambda x: torch.as_tensor(x, dtype=torch.float64, device=self.device).expand(k).clone()
        clip = torch.as_tensor(clip, dtype=torch.long, device=self.device).expand(k).clone(); t0, d = D(t0), D(distance)
        q0, q1 = self.M.qpos_at(clip, t0, d), self.M.qpos_at(clip, t0 + TIMESTEP, d)
        spot = torch.stack([d, torch.zeros_like(d)], 1)
        if heading is not None or origin is not None:   # a bout frame turned by heading about origin (training resets)
            h = D(0.0 if heading is None else heading); o = torch.zeros(k, 2, dtype=torch.float64, device=self.device) if origin is None else torch.as_tensor(origin, dtype=torch.float64, device=self.device).reshape(k, 2)
            c, s = torch.cos(h), torch.sin(h); rz = torch.stack([torch.cos(h / 2), 0 * h, 0 * h, torch.sin(h / 2)], 1)
            for q in (q0, q1):
                x, y = q[:, 0].clone(), q[:, 1].clone(); q[:, 0], q[:, 1] = o[:, 0] + c * x - s * y, o[:, 1] + s * x + c * y; q[:, 3:7] = qmul(rz, q[:, 3:7])
            spot = torch.stack([o[:, 0] + c * d, o[:, 1] + s * d], 1)
        self.clip[ids], self.t0[ids], self.dist[ids], self.offset[ids] = clip, t0, d.to(self.dtype), 0
        self.last[ids] = 0; self.yaw[ids] = 0; self.axis[ids] = 0; self.yaw_set[ids] = False; self.axis_set[ids] = False; self.spot[ids] = spot.to(self.dtype)
        return q0, differentiate_pos(q0, q1, TIMESTEP)

    def clock(self, t):
        """fi = round(t x fps) + aim offset, t the absolute clip time passed to Tracker.act (t_ep + t0 + dt); float64, half to even."""
        return torch.round(torch.as_tensor(t, dtype=torch.float64, device=self.device) * FPS).long() + self.offset

    def reference(self, fi):
        r = self.M.row(self.clip, fi); c = lambda x: x[r].to(self.dtype)
        return c(self.M.ref_q), c(self.M.ref_dq), c(self.M.ref_apos), c(self.M.ref_aquat)

    def face(self, target, pel_xy):
        to = target - pel_xy; want = torch.atan2(to[:, 1], to[:, 0]) - np.pi
        y0 = torch.where(self.yaw_set, self.yaw, want); self.yaw = y0 + wrap(want - y0) * min(1.0, self.dt / FACE_TAU); self.yaw_set[:] = True
        return self.yaw

    def steer(self, target, dist):
        spot = self.spot; u = spot - target
        self.axis = torch.where(self.axis_set[:, None], self.axis, u / (torch.linalg.vector_norm(u, dim=1, keepdim=True) + 1e-9)); self.axis_set[:] = True
        v = (target + dist[:, None] * self.axis - spot) / STEER_TAU; sp = torch.linalg.vector_norm(v, dim=1, keepdim=True)
        self.spot = spot + v * torch.clamp(STEER_MAX / (sp + 1e-9), max=1.0) * self.dt; return self.spot

    def observation(self, rq, rdq, apos, aori, torso_pos, torso_quat, ang_vel, q, dq, torso_mat=None):
        """The 132-d obs after face() and steer() of this step (Tracker.act, r1gym/opponent.py lines 292-309)."""
        yaw = self.yaw; rz = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
        R = quat2mat(qmul(qconj(torso_quat), qmul(rz, aori)))
        c, s = torch.cos(yaw), torch.sin(yaw); a0, a1 = apos[:, 0], apos[:, 1]
        world = torch.stack([self.spot[:, 0] + c * a0 - s * a1, self.spot[:, 1] + s * a0 + c * a1, apos[:, 2]], 1)
        Rt = quat2mat(torso_quat) if torso_mat is None else torso_mat
        pos = torch.einsum("nji,nj->ni", Rt, world - torso_pos)
        return torch.cat([rq, rdq, pos, R[:, :, :2].reshape(-1, 6), ang_vel, q - self.default, dq, self.last], 1)

    def step(self, t, target, pel_xy, torso_pos, torso_quat, ang_vel, q, dq, torso_mat=None, ref=None, dist=None):
        """One control step for all envs on the forwarded state. ref: the aim's (rq, rdq) (default: the clip's reference at the
        clock); dist: the aim's stand-off distance (default: the bout distance). Returns joint targets and a dict of internals."""
        c = lambda x: torch.as_tensor(x, device=self.device).to(self.dtype)
        target, pel_xy, torso_pos, torso_quat, ang_vel, q, dq = map(c, (target, pel_xy, torso_pos, torso_quat, ang_vel, q, dq))
        fi = self.clock(t); rq, rdq, apos, aori = self.reference(fi)
        if ref is not None: rq, rdq = c(ref[0]), c(ref[1])
        self.face(target, pel_xy); self.steer(target, self.dist if dist is None else c(dist))
        obs = self.observation(rq, rdq, apos, aori, torso_pos, torso_quat, ang_vel, q, dq, None if torso_mat is None else c(torso_mat))
        a = self.mlp(obs); self.last = a
        return self.default + self.scale * a, dict(fi=fi, obs=obs, action=a, rq=rq, rdq=rdq)

class MjTracker:
    """Drop-in for r1gym's Tracker inside R1HitEnv (one env per instance): placement and acting by a TrackerDriver(n=1).
    install(replay) sets replay.tracker and replay.place; the env's opp_state carries only 'target' (and 'last', unused)."""
    def __init__(self, replay, dtype=torch.float64, driver=None):
        self.replay = replay; self.k = TRAIN.index(replay.clip); self.drv = driver or TrackerDriver(1, dtype)
        assert int(self.drv.M.frames[self.k]) == replay.T and replay.speed == 1.0, (replay.clip, replay.T)

    def install(self):
        self.replay.tracker = self; self.replay.place = self.place; return self

    def place(self, m, d, t):
        r = self.replay; qp, qv = self.drv.reset(None, self.k, float(t), float(r.distance)); self.d = d   # the env it now serves
        d.qpos[r.qadr:r.qadr + r.nq] = qp[0].cpu().numpy(); d.qvel[r.vadr:r.vadr + r.nq - 1] = qv[0].cpu().numpy()

    def act(self, m, d, replay, state, t, dt=CTRL_DT):
        # the driver's facing/steering state is per instance, unlike Tracker's (per env in opp_state): a FightReplay that
        # eval_opponent.run shares between envs of one (clip, distance) would mix their states, so one MjData only
        assert d is getattr(self, "d", None), "MjTracker serves the one env it last placed (eval_opponent.run shares replays: one bout per run)"
        assert state.get("aim") is None and state.get("target") is not None, "M2 driver: facing target required, no aim (M3)"
        assert abs(dt - self.drv.dt) < 1e-12
        r = replay; T = lambda x: torch.as_tensor(np.asarray(x, np.float64)[None])
        tgt, _ = self.drv.step(t, T(state["target"]), T(d.xpos[r.pel][:2]), T(d.xpos[r.torso]), T(d.xquat[r.torso]), T(d.qvel[r.vadr + 3:r.vadr + 6]),
                               T(d.qpos[r.qadr + 7:r.qadr + r.nq]), T(d.qvel[r.vadr + 6:r.vadr + r.nq - 1]), torso_mat=T(d.xmat[r.torso].reshape(3, 3)))
        state["last"] = self.drv.last[0].cpu().numpy(); d.ctrl[r.act] = tgt[0].double().cpu().numpy()

# ---- M3: the eval's aim (r1gym/aim.py, operator standoff+armwarp_vz = results/opponent_tracker/aim.json), batched
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
ARM_IX = ((14, 15, 16, 17, 18), (19, 20, 21, 22, 23))   # r1gym.aim.ARM: l_hand, r_hand in the 24 controlled joints
LIMBS, PALMS = ("l_hand", "r_hand"), ("left_palm", "right_palm")
# r1gym.aim.merged(configs/r1_assumptions.yaml aim) for zone 'head' (asserted in tests/test_fb_aim.py and vs the recorded Aim.p)
AIM_P = dict(target=(0.057, 0.0, 0.36), lead_s=0.8, commit_s=0.15, window_tail_s=0.15, upright_gate_deg=15.0, gate_after_touch_s=0.3,
             arm=dict(lead_s=0.15, hold_s=0.05, tail_s=0.1, min_end_s=0.1, max_dq=0.8, range_frac=0.95, iters=12, prior=1e-3, damp=1e-3),
             standoff=dict(min_m=0.35, depth_m=0.03))
# training-only jitter, T1 rows (design_learned-tracker 4.2): per intent, the target in the defender's torso frame, the commit
# time and the stand-off depth. The arm keeps 'vz' (x = the reference palm's own forward distance); the reach correction (plan
# A3.1 F') moves the IK target's x so the warped palm's forward distance equals the nominal warp's, and the guard (A2.1) keeps it at
# or below the nominal's at every step.
JITTER = dict(target=((-0.03, 0.03), (-0.04, 0.04), (-0.05, 0.03)), commit_s=(0.10, 0.20), depth_m=(0.0, 0.06))
GUARD_STEPS = 8          # bisection steps on s per limb (plan A2.1)
SECANT_IK = 4            # F' secant iterations = extra IK calls per step (plan A3.1: at most 4)
SECANT_TOL = 1e-6        # F' converged: |palm x - nominal palm x| <= this (m; A3.2 (v), reported); the root aimed at is TOL / 2 below
SECANT_STEP = 0.10       # F' step cap on the target's x (m) until a sign change brackets the root (then secant inside, else bisection)
# reach modes on T1 rows: 'A3' F' then the guard (training); 'A2' the guard alone; 'D' A3.3's fallback (no arm-target jitter, the
# guard); 'raw' the pre-A2 jitter (diagnostics only)
REACH = ("A3", "A2", "D", "raw")
RATE, NO_KEY = FPS, -(1 << 62)

def _r1_model():
    """The opponent-only R1 model the eval's Aim runs its IK on (r1gym.opponent._sub('r1'))."""
    import sys
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from r1gym.opponent import _sub
    return _sub("r1")

def _rot(ax, q):
    """Rotation by q about the unit axis ax (Rodrigues): ax [..., 3], q [...] -> [..., 3, 3]."""
    c, s = torch.cos(q), torch.sin(q); C = 1 - c; x, y, z = ax.unbind(-1)
    return torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                        z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1).reshape(*q.shape, 3, 3)

def _mv(R, v): return (R @ v[..., None])[..., 0]

def smoothstep(x): x = x.clamp(0.0, 1.0); return x * x * (3 - 2 * x)

class ArmKinematics:
    """Both arms' 5 joints -> palm site in the torso_link frame, and its Jacobian, from a compiled model's body offsets, joint axes
    and anchors (mj_kinematics' order: body pos and quat, then the hinge about its anchor). Equals Aim._palm_b and
    Rt^T mj_jacSite on the eval's opponent sub-model (G0-4). Joints: shoulder pitch, roll, yaw, elbow, wrist roll."""
    def __init__(self, C, dtype=torch.float64, device="cpu"):
        t = lambda k: torch.as_tensor(C[k], dtype=dtype, device=device)
        self.pos, self.rot, self.axis, self.jpos, self.q0, self.site = (t(k) for k in ("pos", "rot", "axis", "jpos", "q0", "site"))
        self.range = torch.as_tensor(C["range"], dtype=torch.float64, device=device); self.C, self.dtype, self.device = C, dtype, device

    @staticmethod
    def chain(m, prefix=""):
        """Constants of the two chains torso_link -> {left,right}_palm in model m (joint and body names under `prefix`)."""
        import mujoco
        tor = m.body(prefix + "torso_link").id; C = {k: [] for k in ("pos", "rot", "axis", "jpos", "q0", "range", "site")}
        for side, palm in zip(("left", "right"), PALMS):
            s = m.site(prefix + palm).id; b, bodies = int(m.site_bodyid[s]), []
            while b != tor: assert b > 0, f"{palm} not under torso_link"; bodies.append(b); b = int(m.body_parentid[b])
            assert len(bodies) == 5 and np.allclose(m.site_quat[s], (1, 0, 0, 0)), (palm, len(bodies))
            for k in C: C[k].append([])
            for b, jn in zip(bodies[::-1], ARM_JOINTS):
                j = m.joint(f"{prefix}{side}_{jn}_joint").id; R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_quat[b])
                assert m.jnt_bodyid[j] == b and m.body_jntnum[b] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                for k, v in zip(("pos", "rot", "axis", "jpos", "q0", "range"), (m.body_pos[b], R.reshape(3, 3), m.jnt_axis[j], m.jnt_pos[j], m.qpos0[m.jnt_qposadr[j]], m.jnt_range[j])):
                    C[k][-1].append(np.array(v, np.float64))
            C["site"][-1] = np.array(m.site_pos[s], np.float64)
        return {k: np.array(v) for k, v in C.items()}

    @classmethod
    def from_model(cls, m=None, prefix="", dtype=torch.float64, device="cpu"):
        return cls(cls.chain(_r1_model() if m is None else m, prefix), dtype, device)

    def fk(self, q, jac=False):
        """q [N, 2, 5] (l_hand, r_hand) -> palm [N, 2, 3] in the torso frame; with jac also J [N, 2, 3, 5] = d palm / d q."""
        q = q.to(self.dtype); N = q.shape[0]; R = torch.eye(3, dtype=self.dtype, device=self.device).expand(N, 2, 3, 3); p = q.new_zeros(N, 2, 3); A, X = [], []
        for i in range(5):
            p = p + _mv(R, self.pos[:, i]); R = R @ self.rot[:, i]; an = p + _mv(R, self.jpos[:, i]); X.append(_mv(R, self.axis[:, i]))
            R = R @ _rot(self.axis[:, i].expand(N, 2, 3), q[..., i] - self.q0[:, i]); p = an - _mv(R, self.jpos[:, i]); A.append(an)
        palm = p + _mv(R, self.site)
        if not jac: return palm
        return palm, torch.stack([torch.linalg.cross(X[i], palm - A[i], dim=-1) for i in range(5)], -1)

class AimOp:
    """r1gym/aim.py Aim for N envs, operator standoff+armwarp_vz (aimed rows) or shadow (intents made and logged, reference and
    distance untouched), step for step as Aim.reference. Times (clock, intents, gates, weights) are float64 in both dtypes.
      intents  per env K ring slots (+1 dump slot for the scatter): the clip's hand strikes (fb_motion's strike table in the
               library's order) whose onset, at loop occurrence k or k + 1 of the clip clock, lies within lead_s ahead. Aim.made
               (one key set per Aim) becomes the last key made per strike (time only moves forward and T >= 5 s > lead_s), and a
               strike whose onset repeats an earlier one of the same clip never gets an intent (kfab_203 onset 0: l_hand is
               made, r_hand is skipped, as the eval's key set does). gated 1 'tilt' (opponent torso tilt > 15 deg), 2 'touch'
               (a strike limb touched the defender's head or torso within 0.3 s; `touching` is that contact flag this step).
               The world target follows the defender's torso until t_apex - commit_s, then stays.
      standoff dist = min(home, clip(reach_head - depth, 0.35, home)) over live aimed intents.
      arm      per limb the highest-weight aimed intent (the first made on a tie): target in the opponent's current torso frame,
               x replaced by the reference palm's own (vz); 12 damped least-squares steps from the reference arm, clipped to
               +-0.8 rad and 95% of range; delta = w (q_ik - q_ref) added to rq and its rate to rdq.
      log      (with q_act) the closest palm to target over [t_on, t_apex + 0.15], arm tracking error, and (log_ik) the
               full-target IK residual at the apex, as Aim._log.
    jitter (per env, T1 rows): JITTER drawn per intent from the op's generator. The nominal intent (target and commit time
               unjittered, same slot) is warped alongside; palm x is the forward distance in the opponent torso frame (ArmKinematics,
               as the vz target). reach 'A3' (plan A3.1, F'): the jittered IK target keeps its yz and its x is moved by secant
               iterations (SECANT_IK extra IK calls; the first step unit slope over the weight, steps capped at SECANT_STEP until a
               sign change brackets the root, then inside the bracket or bisected) until the warped palm x equals the nominal's (the
               root aimed at SECANT_TOL / 2 below it, so a converged iterate needs no guard); the iterate kept is the closest. Then the guard (A2.1; alone in
               'A2' and 'D'): where the palm still lies further forward than the nominal's, the limb's delta becomes nominal + s
               (corrected - nominal), s in [0, 1) the largest feasible of an 8-step bisection (excess <= 0). 'D' (A3.3) draws no
               arm-target offset (commit time and depth still jittered). T0 rows are not touched. Per slot: warped T1 steps, guarded
               steps, the sum of s and F'-converged steps (intents()). Slot overflow (a live intent overwritten) is counted; ik_calls
               counts the IK runs.
    skip_idle (CPU replica, M6): no IK on a step where no limb has a live aimed intent (one host sync; outputs unchanged).
    dtype is the inputs' and outputs' (the sim state, the reference shown); compute (default float64) is the aim's own arithmetic:
    the 5x5 damped IK is ill-conditioned near the clip bounds, and float32 arithmetic moves its result by up to 0.03 rad where a
    float64 IK on the same float32 inputs moves it by 2e-3 (scripts/fb_m3_check.py). debug keeps the last step's IK (and the reach
    step's nominal, jittered and corrected deltas, palm x, the F' shift and convergence, the guard flag and s) in self.dbg."""
    def __init__(self, n, dtype=torch.float32, device="cpu", motion=None, kin=None, p=AIM_P, slots=8, seed=0, log_ik=False, compute=torch.float64, debug=False, reach="A3", skip_idle=False):
        assert reach in REACH, reach; self.skip_idle = skip_idle
        pin_precision(); self.io, dtype = dtype, compute or dtype; self.n, self.dtype, self.device, self.K, self.p, self.log_ik, self.debug, self.reach_mode = n, dtype, device, slots, p, log_ik, debug, reach
        M = motion or Motion(device=device); S = M.strikes; kin = kin or ArmKinematics.from_model(dtype=dtype, device=device)
        self.kin = kin if kin.dtype == dtype else ArmKinematics(kin.C, dtype, device)
        hand = np.char.endswith(S["limb"].astype(str), "hand"); rows = []
        for c in range(len(M.clips)):
            r = np.flatnonzero((S["clip"] == c) & hand); rows.append([int(i) for j, i in enumerate(r) if S["onset"][i] not in S["onset"][r[:j]]])
        self.lib_rows = rows; L = max(map(len, rows)); P = lambda k, f=0: [[S[k][i] for i in r] + [f] * (L - len(r)) for r in rows]
        lt = lambda x, dt=torch.long: torch.as_tensor(np.array(x), dtype=dt, device=device)
        self.lib_on, self.lib_ap, self.lib_en = lt(P("onset")), lt(P("apex")), lt(P("end"))
        self.lib_limb = lt([[LIMBS.index(str(S["limb"][i])) for i in r] + [0] * (L - len(r)) for r in rows])
        self.lib_reach = lt(P("reach_head_m", 0.0), torch.float64); self.lib_ok = lt([[True] * len(r) + [False] * (L - len(r)) for r in rows], torch.bool)
        self.frames = M.frames.to(device); a = p["arm"]; rg = self.kin.range; mid, half = rg.mean(-1), (rg[..., 1] - rg[..., 0]) / 2
        self.lo, self.hi = (mid - a["range_frac"] * half).to(dtype), (mid + a["range_frac"] * half).to(dtype)   # [2, 5]
        self.ix = torch.tensor(ARM_IX, device=device); self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)
        self.target = torch.tensor(p["target"], dtype=dtype, device=device)
        z = lambda *s, dt=torch.float64, v=0: torch.full(s, v, dtype=dt, device=device); K1 = slots + 1
        self.clip, self.home = z(n, dt=torch.long), z(n, dt=dtype); self.aimed, self.jit = z(n, dt=torch.bool), z(n, dt=torch.bool)
        self.touch_t, self.last_key, self.prev = z(n, v=-9.0), z(n, L, dt=torch.long, v=NO_KEY), z(n, ACT_DIM, dt=dtype)
        self.nseq, self.overflow = z(n, dt=torch.long), z(n, dt=torch.long)
        self.valid, self.committed, self.ik_done = z(n, K1, dt=torch.bool), z(n, K1, dt=torch.bool), z(n, K1, dt=torch.bool)
        self.seq, self.lib, self.limb, self.key, self.gated, self.nerr = (z(n, K1, dt=torch.long) for _ in range(6))
        self.t_on, self.t_apex, self.t_end, self.commit_s = (z(n, K1) for _ in range(4))
        self.tgt_r1, self.target_w = z(n, K1, 3, dt=dtype), z(n, K1, 3, dt=dtype); self.reach, self.depth, self.err2 = z(n, K1, dt=dtype), z(n, K1, dt=dtype), z(n, K1, dt=dtype)
        self.closest, self.ik_res = z(n, K1, dt=dtype, v=9.9), z(n, K1, dt=dtype, v=float("nan")); self.any_jit = False
        self.target_wn, self.committed_n = z(n, K1, 3, dt=dtype), z(n, K1, dt=torch.bool)   # the nominal intent (reach guard)
        self.g_warp, self.g_steps, self.g_s, self.g_conv = z(n, K1, dt=torch.long), z(n, K1, dt=torch.long), z(n, K1, dt=dtype), z(n, K1, dt=torch.long); self.ik_calls = 0

    def _ids(self, ids): return torch.arange(self.n, device=self.device) if ids is None else torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def reset(self, ids, clip, distance, aimed=True, jitter=False):
        """A new Aim for envs `ids` (the eval makes one per bout): clip index, bout distance (StandOff's home), aimed (else
        shadow) and jitter flags."""
        ids = self._ids(ids); k = len(ids); E = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device).expand(k)
        self.clip[ids], self.home[ids], self.aimed[ids], self.jit[ids] = E(clip, torch.long), E(distance, self.dtype), E(aimed, torch.bool), E(jitter, torch.bool)
        self.any_jit = self.any_jit or not isinstance(jitter, bool) or jitter   # a tensor flag: draw every step (no device sync), masked per env
        self.touch_t[ids] = -9.0; self.last_key[ids] = NO_KEY; self.prev[ids] = 0; self.nseq[ids] = 0; self.overflow[ids] = 0
        for x in (self.valid, self.committed, self.ik_done, self.committed_n): x[ids] = False
        for x in (self.seq, self.lib, self.limb, self.key, self.gated, self.nerr, self.t_on, self.t_apex, self.t_end, self.commit_s, self.tgt_r1, self.target_w, self.reach, self.depth, self.err2,
                  self.target_wn, self.g_warp, self.g_steps, self.g_s, self.g_conv): x[ids] = 0
        self.closest[ids] = 9.9; self.ik_res[ids] = float("nan")

    def weight(self, now):
        """Aim.weight for every slot: smoothstep in from t_on - 0.15 s to the apex, 1 for hold_s, out by max(t_end, apex +
        min_end_s) + tail_s; 0 outside. Float64 [N, K + 1]."""
        a = self.p["arm"]; now = now[:, None]; t0, tp = self.t_on - a["lead_s"], self.t_apex; th = tp + a["hold_s"]; te = torch.maximum(self.t_end, tp + a["min_end_s"]) + a["tail_s"]
        w = torch.where(now < tp, smoothstep((now - t0) / (tp - t0).clamp(min=1e-6)), torch.where(now <= th, torch.ones_like(tp), smoothstep((te - now) / (te - th).clamp(min=1e-6))))
        return torch.where((now <= t0) | (now >= te), torch.zeros_like(w), w)

    def ik(self, rq5, p_t):
        """Aim.ik for both limbs: rq5 [N, 2, 5] reference arm, p_t [N, 2, 3] palm target in the torso frame -> arm angles [N, 2, 5]."""
        a = self.p["arm"]; q = rq5.clone(); self.ik_calls += 1; lo = torch.minimum(torch.maximum(self.lo, q - a["max_dq"]), q); hi = torch.maximum(torch.minimum(self.hi, q + a["max_dq"]), q)
        I = torch.eye(5, dtype=self.dtype, device=self.device) * (a["damp"] + a["prior"])
        for _ in range(a["iters"]):
            pb, A = self.kin.fk(q, jac=True); At = A.transpose(-1, -2)
            q = torch.minimum(torch.maximum(q + torch.linalg.solve(At @ A + I, (_mv(At, p_t - pb) + a["prior"] * (rq5 - q))[..., None])[..., 0], lo), hi)
        return q

    @torch.no_grad()
    def step(self, fi, now, rq, rdq, opp_x, opp_R, def_x, def_R, touching, q_act=None, dt=CTRL_DT):
        """One Aim.reference for all envs. fi [N] the clip clock (aim offset 0), now [N] float64 sim time at the end of the control
        step (d.time + dt), rq/rdq [N, 24] the clip's reference at fi, opp_x/opp_R and def_x/def_R torso_link positions [N, 3] and
        rotations [N, 3, 3] of the opponent and the defender, touching [N] bool, q_act [N, 24] the opponent's joints (logging).
        Returns (rq, rdq) shown to the tracker and the steering distance [N]."""
        p, a, K, n, dv = self.p, self.p["arm"], self.K, self.n, self.device; c = lambda x: torch.as_tensor(x, device=dv).to(self.dtype)
        now = torch.as_tensor(now, dtype=torch.float64, device=dv).expand(n); fi = torch.as_tensor(fi, dtype=torch.long, device=dv).expand(n)
        rq, rdq, opp_x, opp_R, def_x, def_R = map(c, (rq, rdq, opp_x, opp_R, def_x, def_R)); touching = torch.as_tensor(touching, dtype=torch.bool, device=dv).expand(n)
        # scheduler (Aim._schedule): touch time, tilt gate, new intents at occurrences k and k + 1, targets until committed
        self.touch_t = torch.where(touching, now, self.touch_t)
        tilt = torch.rad2deg(torch.arccos(opp_R[:, 2, 2].double().clamp(-1.0, 1.0))); T = self.frames[self.clip]; tc = fi.double() / RATE
        occ = torch.floor(tc * RATE / T.double()).long(); kk = torch.stack([occ, occ + 1], 1)[:, :, None]; on = self.lib_on[self.clip][:, None, :]
        key = kk * T[:, None, None] + on; ton = key.double() / RATE; L = on.shape[-1]
        ok = self.lib_ok[self.clip][:, None, :] & (key > self.last_key[:, None, :]) & (tc[:, None, None] <= ton) & (ton <= tc[:, None, None] + p["lead_s"])
        self.last_key = torch.maximum(self.last_key, torch.where(ok, key, NO_KEY).amax(1)); ok, key = ok.reshape(n, 2 * L), key.reshape(n, 2 * L)
        rank = ok.long().cumsum(1) - 1; slot = torch.where(ok, (self.nseq[:, None] + rank) % K, K)
        live = self.valid & (now[:, None] <= self.t_end + a["tail_s"] + p["window_tail_s"]); self.overflow += (ok & live.gather(1, slot)).sum(1)
        gate = torch.where(tilt > p["upright_gate_deg"], 1, torch.where(now - self.touch_t < p["gate_after_touch_s"], 2, 0))
        s_ix = torch.arange(L, device=dv).repeat(2)[None].expand(n, -1); kT = (kk[:, :, 0] * T[:, None]).repeat_interleave(L, 1)
        g = lambda x: x[self.clip].gather(1, s_ix); sim = lambda f: (now[:, None] + (kT + g(f)).double() / RATE) - tc[:, None]
        W = lambda x, v: x.scatter_(1, slot.reshape(n, -1, *[1] * (x.dim() - 2)).expand(n, -1, *x.shape[2:]), v)
        tgt = self.target.expand(n, 2 * L, 3); cs = torch.full((n, 2 * L), p["commit_s"], dtype=torch.float64, device=dv)
        dep = torch.full((n, 2 * L), p["standoff"]["depth_m"], dtype=self.dtype, device=dv)
        if self.any_jit:
            u = torch.rand(n, 2 * L, 5, generator=self.gen, dtype=torch.float64, device=dv); j = self.jit[:, None]; J = JITTER
            lo3, hi3 = (torch.tensor([x[i] for x in J["target"]], dtype=torch.float64, device=dv) for i in (0, 1))
            if self.reach_mode != "D": tgt = torch.where(j[..., None], tgt + (lo3 + (hi3 - lo3) * u[..., :3]).to(self.dtype), tgt)
            cs = torch.where(j, J["commit_s"][0] + (J["commit_s"][1] - J["commit_s"][0]) * u[..., 3], cs)
            dep = torch.where(j, (J["depth_m"][0] + (J["depth_m"][1] - J["depth_m"][0]) * u[..., 4]).to(self.dtype), dep)
        W(self.valid, ok); W(self.committed, torch.zeros_like(ok)); W(self.ik_done, torch.zeros_like(ok)); W(self.seq, self.nseq[:, None] + rank)
        W(self.lib, s_ix); W(self.limb, g(self.lib_limb)); W(self.key, key); W(self.gated, gate[:, None].expand(n, 2 * L)); W(self.nerr, torch.zeros_like(rank))
        W(self.t_on, sim(self.lib_on)); W(self.t_apex, sim(self.lib_ap)); W(self.t_end, sim(self.lib_en)); W(self.commit_s, cs)
        W(self.tgt_r1, tgt); W(self.reach, g(self.lib_reach).to(self.dtype)); W(self.depth, dep); W(self.closest, torch.full_like(dep, 9.9))
        W(self.ik_res, torch.full_like(dep, float("nan"))); W(self.err2, torch.zeros_like(dep)); W(self.target_w, torch.zeros_like(tgt))
        W(self.target_wn, torch.zeros_like(tgt)); W(self.committed_n, torch.zeros_like(ok)); W(self.g_warp, torch.zeros_like(rank)); W(self.g_steps, torch.zeros_like(rank)); W(self.g_s, torch.zeros_like(dep)); W(self.g_conv, torch.zeros_like(rank))
        self.valid[:, K] = False; self.nseq += ok.sum(1)
        upd = self.valid & ~self.committed & (now[:, None] <= self.t_end + 1.0)
        self.target_w = torch.where(upd[..., None], def_x[:, None] + _mv(def_R[:, None], self.tgt_r1), self.target_w)
        self.committed |= upd & (now[:, None] >= self.t_apex - self.commit_s)
        if self.any_jit:   # the nominal intent's target: unjittered target, nominal commit time (bit-equal to a T0 run's)
            upn = self.valid & ~self.committed_n & (now[:, None] <= self.t_end + 1.0)
            self.target_wn = torch.where(upn[..., None], def_x[:, None] + _mv(def_R[:, None], self.target.expand(n, K + 1, 3)), self.target_wn)
            self.committed_n |= upn & (now[:, None] >= self.t_apex - p["commit_s"])
        aimed = self.valid & (self.gated == 0)
        # StandOff
        sl = aimed & (self.t_on - p["lead_s"] <= now[:, None]) & (now[:, None] <= self.t_end)
        v = torch.minimum(torch.maximum(self.reach - self.depth, torch.full_like(self.reach, p["standoff"]["min_m"])), self.home[:, None].expand_as(self.reach))
        dist = torch.where(self.aimed, torch.minimum(self.home, torch.where(sl, v, torch.full_like(v, float("inf"))).amin(1)), self.home)
        # ArmWarp, vz
        w = self.weight(now); act = aimed & (self.t_on - p["lead_s"] <= now[:, None]) & (now[:, None] <= self.t_end + a["tail_s"] + p["window_tail_s"]) & (w > 0)
        wl = torch.where(act[:, None, :] & (self.limb[:, None, :] == torch.arange(2, device=dv)[None, :, None]), w[:, None, :], torch.full_like(w[:, None, :], -1.0))
        wmax = wl.amax(-1); has = (wmax > 0) & self.aimed[:, None]
        j = torch.where((wl == wmax[..., None]) & (wl > 0), self.seq[:, None, :], torch.full_like(self.seq[:, None, :], 1 << 62)).argmin(-1)
        rq5 = rq[:, self.ix]; tw = self.target_w.gather(1, j[..., None].expand(n, 2, 3))
        p_t = torch.einsum("nji,nkj->nki", opp_R, tw - opp_x[:, None]); p_t = torch.cat([self.kin.fk(rq5)[..., :1], p_t[..., 1:]], -1)
        qa = rq5 if self.skip_idle and not bool(has.any()) else self.ik(rq5, p_t)   # skip_idle (CPU replica, a host sync): no limb warps, d5 is 0 either way
        d5 = torch.where(has[..., None], wmax.to(self.dtype)[..., None] * (qa - rq5), torch.zeros_like(rq5))
        if self.debug: self.dbg = dict(rq5=rq5, p_t=p_t, qa=qa, has=has, w=wmax, slot=j)
        if self.any_jit and self.reach_mode != "raw": d5 = self._reach(rq5, p_t, d5, has, wmax, j, opp_x, opp_R)
        delta = torch.zeros_like(rq).index_copy(1, self.ix.reshape(-1), d5.reshape(n, 10))
        rq = rq + delta; rdq = rdq + (delta - self.prev) / dt; self.prev = delta
        if q_act is not None: self._log(now, rq, opp_x, opp_R, c(q_act))
        return rq.to(self.io), rdq.to(self.io), dist.to(self.io)

    def _reach(self, rq5, p_t, d5, has, wmax, j, opp_x, opp_R):
        """T1 rows: the nominal warp dn of the same slot (the T0 run's arithmetic); 'A3' F' (plan A3.1): the jittered target's x c moved
        by SECANT_IK safeguarded secant steps on g(c) = palm x - nominal palm x + SECANT_TOL / 2 (yz kept; steps capped at SECANT_STEP
        until a sign change, then kept inside the bracket or bisected), the kept iterate the smallest |g| (c0 included); then the guard (A2.1): where palm x still exceeds the nominal's, d -> dn + s (d - dn), s the largest feasible
        after GUARD_STEPS bisection steps per limb."""
        n, z = self.n, torch.zeros_like(rq5); w = wmax.to(self.dtype); on = self.jit[:, None] & has
        tn = self.target_wn.gather(1, j[..., None].expand(n, 2, 3)); p_n = torch.einsum("nji,nkj->nki", opp_R, tn - opp_x[:, None])
        p_n = torch.cat([p_t[..., :1], p_n[..., 1:]], -1); dn = torch.where(has[..., None], w[..., None] * (self.ik(rq5, p_n) - rq5), z)
        x = lambda d: self.kin.fk(rq5 + d)[..., 0]; xn, xj = x(dn), x(d5); d, xb, c0 = d5, xj, p_t[..., 0]
        if self.reach_mode == "A3":   # F': safeguarded secant on the target's x, root g = palm x - nominal x + SECANT_TOL / 2 = 0
            D = lambda c: torch.where(has[..., None], w[..., None] * (self.ik(rq5, torch.cat([c[..., None], p_t[..., 1:]], -1)) - rq5), z)
            nan = torch.full_like(c0, float("nan")); cp, gp = c0, xj - xn + SECANT_TOL / 2; cb, gb = c0, gp
            cl, ch = torch.where(gp < 0, c0, nan), torch.where(gp > 0, c0, nan); c = c0 - (gp / w.clamp(min=1e-9)).clamp(-SECANT_STEP, SECANT_STEP)
            for _ in range(SECANT_IK):
                dk = D(c); xk = x(dk); gk = xk - xn + SECANT_TOL / 2; b = on & (gk.abs() < gb.abs())
                d, xb, gb, cb = torch.where(b[..., None], dk, d), torch.where(b, xk, xb), torch.where(b, gk, gb), torch.where(b, c, cb)
                cl, ch = torch.where(gk < 0, c, cl), torch.where(gk > 0, c, ch); br = ~(cl.isnan() | ch.isnan())
                q = gk - gp; cs = c - gk * (c - cp) / torch.where(q != 0, q, torch.ones_like(q)); cs = torch.where(q != 0, cs, torch.full_like(cs, float("inf")))
                ins = (cs - cl) * (cs - ch) < 0   # strictly inside the bracket (NaN and inf compare False)
                cs = torch.where(br, torch.where(ins, cs, (cl + ch) / 2), c + torch.nan_to_num(cs - c, nan=0.0, posinf=0.0, neginf=0.0).clamp(-SECANT_STEP, SECANT_STEP))
                cp, gp, c = c, gk, cs
            conv = on & ((xb - xn).abs() <= SECANT_TOL)
        else: conv, cb = torch.zeros_like(on), c0
        dd = d - dn; g = on & (xb > xn); lo, hi = torch.zeros_like(xn), torch.ones_like(xn)
        for _ in range(GUARD_STEPS):
            s = (lo + hi) / 2; f = x(dn + s[..., None] * dd) <= xn; lo, hi = torch.where(f, s, lo), torch.where(f, hi, s)
        out = torch.where(g[..., None], dn + lo[..., None] * dd, d)
        self.g_warp.scatter_add_(1, j, on.long()); self.g_steps.scatter_add_(1, j, g.long()); self.g_s.scatter_add_(1, j, torch.where(g, lo, torch.zeros_like(lo))); self.g_conv.scatter_add_(1, j, conv.long())
        if self.debug: self.dbg.update(d_jit=d5, d_nom=dn, d_cor=d, x_nom=xn, x_jit=xj, x_cor=xb, shift=torch.where(on, cb - c0, torch.zeros_like(c0)), conv=conv, guard=g, s=torch.where(g, lo, torch.ones_like(lo)))
        return out

    def _log(self, now, rq, opp_x, opp_R, q_act):
        """Aim._log for every intent in its window [t_on, t_apex + 0.15]: closest palm to target, arm tracking error and (log_ik)
        the full-target IK residual at the apex (from the warped reference, as the eval)."""
        n = self.n; qa = q_act[:, self.ix]; palm = opp_x[:, None] + _mv(opp_R[:, None], self.kin.fk(qa)); lim = self.limb[..., None].expand(n, self.K + 1, 3)
        win = self.valid & (self.t_on <= now[:, None]) & (now[:, None] <= self.t_apex + self.p["window_tail_s"])
        self.closest = torch.where(win, torch.minimum(self.closest, torch.linalg.vector_norm(palm.gather(1, lim) - self.target_w, dim=-1)), self.closest)
        e2 = ((qa - rq[:, self.ix]) ** 2).mean(-1); self.err2 += torch.where(win, e2.gather(1, self.limb), torch.zeros_like(self.err2)); self.nerr += win.long()
        if self.log_ik:
            need = win & ~self.ik_done & (now[:, None] >= self.t_apex); e, s = need.nonzero(as_tuple=True)
            if len(e):
                l = self.limb[e, s]; p_t = torch.einsum("mji,mj->mi", opp_R[e], self.target_w[e, s] - opp_x[e]); m = torch.arange(len(e), device=self.device)
                q = self.ik(rq[e][:, self.ix], p_t[:, None].expand(-1, 2, 3).clone())
                self.ik_res[e, s] = torch.linalg.vector_norm(p_t - self.kin.fk(q)[m, l], dim=-1); self.ik_done[e, s] = True

    def intents(self, i):
        """Env i's intents in the order made (valid slots by seq; complete only while nothing wrapped), as dicts."""
        G = ("", "tilt", "touch"); s = [int(k) for k in torch.argsort(self.seq[i, :self.K]) if bool(self.valid[i, k])]
        f = lambda x, k: x[i, k].item(); v = lambda x, k: x[i, k].double().cpu().numpy()
        return [dict(limb=LIMBS[f(self.limb, k)], lib=f(self.lib, k), key=f(self.key, k), t_on=f(self.t_on, k), t_apex=f(self.t_apex, k), t_end=f(self.t_end, k),
                     gated=G[f(self.gated, k)], committed=bool(self.committed[i, k]), reach_m=f(self.reach, k), target_r1=v(self.tgt_r1, k), target_w=v(self.target_w, k),
                     closest_m=f(self.closest, k), ik_res_m=f(self.ik_res, k), arm_rmse=float(np.sqrt(f(self.err2, k) / f(self.nerr, k))) if f(self.nerr, k) else None,
                     commit_s=f(self.commit_s, k), depth_m=f(self.depth, k), guard_warped_steps=f(self.g_warp, k), guard_steps=f(self.g_steps, k),
                     guard_s_mean=f(self.g_s, k) / f(self.g_steps, k) if f(self.g_steps, k) else None, secant_conv_steps=f(self.g_conv, k)) for k in s]
