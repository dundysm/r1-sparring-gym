"""Fight-motion opponent: by default a second (kinematic) Unitree R1 replaying KungFuAthleteBot clips retargeted
from the G1 by joint name (shared Unitree naming; R1 lacks waist_pitch / wrist_pitch / wrist_yaw, those are dropped,
angles are clipped to R1 joint ranges, root height scaled by R1/G1 standing pelvis height). Optional raw G1 replay, KungFuAthleteBot clips kinematically in front of the R1.

The G1 is driven by writing qpos (+ finite-difference qvel) every physics substep, so its limbs are an
infinite-mass "fist" for the R1: every contact is a real MuJoCo contact force on a real R1 body, logged per body.
Clips: KungFuAthleteBot (MIT code / Apache-2.0 data, LuluCao/KungfuAthleteBot on HF), G1 29-DoF, 50 fps."""
from __future__ import annotations
import numpy as np, mujoco
from pathlib import Path
from . import ASSETS
from .model import opponent_spec

G1_JOINTS = ['left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw']
G1_PELVIS_Z, R1_PELVIS_Z = 0.793, 0.74

def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])

def _yaw(q):
    w, x, y, z = q; return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def list_clips(): return sorted(p.stem for p in (ASSETS / "motions").glob("kfab_*.npz"))

class FightReplay:
    """root="inplace" (default): the clip's slow root drift and heading changes are removed (1 s low-pass), so the
    opponent stays `distance` m in front of the R1, facing it, while lunges, steps and twists shorter than ~1 s are kept.
    root="clip": the original placement (clip mean xy at `distance`, frame-0 heading); the clips drift up to 2.4 m, so
    the opponent is mostly out of reach. Either way the loop is cross-faded over `blend_s` so it never teleports."""
    def __init__(self, m: mujoco.MjModel, clip="kfab_200", distance=0.62, speed=1.0, prefix="opp_", kind=None, root="inplace", blend_s=0.3):
        z = np.load(ASSETS / "motions" / f"{clip}.npz"); q = z["qpos"].astype(float); self.fps = float(z["fps"]) if "fps" in z else 50.0
        self.speed = speed; self.clip = clip; self.raw = q.copy(); self.distance = distance; self.root = root
        self.kind = kind or ("g1" if _has(m, prefix + "waist_pitch_joint") else "r1")
        q = self._retarget(q) if self.kind == "r1" else self._canon(q)
        self.q = _seamless(q, int(round(blend_s * self.fps))); self.T = len(self.q)
        j = m.joint(prefix + "floating_base_joint")
        self.qadr = m.jnt_qposadr[j.id]; self.vadr = m.jnt_dofadr[j.id]; self.nq = q.shape[1]
        self.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(prefix)}
        sub = _sub(self.kind); hinges = [sub.joint(i).name for i in range(1, sub.njnt)]
        self.puppet = self.kind == "r1" and _has_act(m, prefix + hinges[0].replace("_joint", ""))
        if self.puppet:   # actuators in the order of qpos[7:]
            self.act = np.array([m.actuator(prefix + h.replace("_joint", "")).id for h in hinges])
            self.pel = m.body(prefix + "pelvis").id; self.mass = float(m.body_subtreemass[self.pel])

    # root assist for the puppet: PD toward the clip's root pose, force-limited. Horizontal force is capped near the
    # friction limit of a 29 kg robot (mu*m*g ~ 280 N), so the opponent cannot bulldoze the R1 with the assist.
    ROOT_KP, ROOT_KD, ROOT_FXY, ROOT_FZ = 1500.0, 300.0, 250.0, 600.0
    ROT_KP, ROT_KD, ROT_TAU = 400.0, 40.0, 150.0

    def place(self, m, d, t):
        """Put the opponent exactly on the clip at time t (kinematic set, e.g. at reset)."""
        q0, q1 = self.qpos_at(t), self.qpos_at(t + m.opt.timestep)
        d.qpos[self.qadr:self.qadr + self.nq] = q0
        v = np.zeros(self.nq - 1); mujoco.mj_differentiatePos(_sub(self.kind), v, m.opt.timestep, q0, q1)
        d.qvel[self.vadr:self.vadr + self.nq - 1] = v

    def _canon(self, q):
        q = q.copy(); distance = self.distance
        if self.root == "inplace":
            yaw = np.unwrap([_yaw(x) for x in q[:, 3:7]]); n = int(round(self.fps))
            dyaw = np.pi - _lowpass(yaw, n); xy = q[:, :2] - _lowpass(q[:, :2], n)    # keep fast motion, drop drift
            for t in range(len(q)):
                c, s = np.cos(dyaw[t]), np.sin(dyaw[t])
                q[t, :2] = np.array([[c, -s], [s, c]]) @ xy[t] + [distance, 0.0]
                q[t, 3:7] = _quat_mul([np.cos(dyaw[t] / 2), 0, 0, np.sin(dyaw[t] / 2)], q[t, 3:7])
            return q
        # canonicalise: start heading -> face the R1 (-x), mean xy -> `distance` in front of R1
        yaw0 = _yaw(q[0, 3:7]); target = np.pi; dy = target - yaw0
        c, s = np.cos(dy), np.sin(dy); R = np.array([[c, -s], [s, c]]); rq = np.array([np.cos(dy / 2), 0, 0, np.sin(dy / 2)])
        xy = (q[:, :2] - q[:, :2].mean(0)) @ R.T
        q[:, :2] = xy + [distance, 0.0]
        for t in range(len(q)): q[t, 3:7] = _quat_mul(rq, q[t, 3:7])
        return q

    def _retarget(self, q):
        """Keypoint IK retarget G1 -> R1 (cached). Root pose copied (height scaled); R1 joints solved by damped
        least squares so knees/ankles/torso/elbows/wrists match the G1 keypoints (pelvis-relative, height-scaled)."""
        cache = ASSETS / "motions" / f"r1_{self.clip}.npy"
        if cache.exists():
            out = np.load(cache)
            if out.shape[0] == len(q): return self._canon(out)
        out = retarget_g1_to_r1(self.raw); np.save(cache, out); return self._canon(out)

    def qpos_at(self, t):
        f = (t * self.fps * self.speed) % self.T; i = int(f) % self.T; a = f - int(f); q1 = self.q[(i + 1) % self.T]
        if np.dot(self.q[i, 3:7], q1[3:7]) < 0: q1 = q1.copy(); q1[3:7] *= -1
        q = (1 - a) * self.q[i] + a * q1; q[3:7] /= np.linalg.norm(q[3:7]); return q

    def apply(self, m, d, t, dt):
        """Call once per physics substep, after any other xfrc_applied writes (it sets the opponent pelvis wrench)."""
        if not self.puppet: return self.place(m, d, t)
        q0, q1 = self.qpos_at(t), self.qpos_at(t + dt)
        d.ctrl[self.act] = q0[7:]
        v = np.zeros(self.nq - 1); mujoco.mj_differentiatePos(_sub(self.kind), v, dt, q0, q1)   # target vel (root: local ang)
        p, R = d.xpos[self.pel], d.xmat[self.pel].reshape(3, 3)
        vel = d.qvel[self.vadr:self.vadr + 3]; w = R @ d.qvel[self.vadr + 3:self.vadr + 6]
        F = self.ROOT_KP * (q0[:3] - p) + self.ROOT_KD * (v[:3] - vel); F[2] += self.mass * 9.81
        fxy = np.linalg.norm(F[:2]); F[:2] *= min(1.0, self.ROOT_FXY / (fxy + 1e-9)); F[2] = np.clip(F[2], 0.0, self.ROOT_FZ)
        err = np.zeros(3); mujoco.mju_subQuat(err, q0[3:7], d.xquat[self.pel])   # rotation to target, pelvis frame
        tau = self.ROT_KP * (R @ err) + self.ROT_KD * (R @ v[3:6] - w)
        tau *= min(1.0, self.ROT_TAU / (np.linalg.norm(tau) + 1e-9))
        d.xfrc_applied[self.pel, :3] = F; d.xfrc_applied[self.pel, 3:] = tau

def _lowpass(x, n):
    """Centred moving average over n frames (edge-padded), along axis 0."""
    x = np.asarray(x, float); pad = [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1)
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=0); c = np.concatenate([np.zeros_like(c[:1]), c])
    return (c[n:] - c[:-n]) / n

def _seamless(q, b):
    """Loopable clip: drop the last b frames and blend them into the first b, so frame T-1 -> frame 0 is a normal step."""
    if b < 1 or len(q) <= 2 * b: return q
    L = len(q) - b; out = q[:L].copy(); w = (np.arange(b) / b)[:, None]; tail = q[L:L + b].copy()
    tail[:, 3:7] *= np.sign(np.sum(tail[:, 3:7] * q[:b, 3:7], axis=1, keepdims=True) + 1e-12)   # same quaternion hemisphere
    out[:b] = (1 - w) * tail + w * q[:b]
    out[:b, 3:7] /= np.linalg.norm(out[:b, 3:7], axis=1, keepdims=True)
    return out

def _has(m, name):
    try: m.joint(name); return True
    except KeyError: return False

def _has_act(m, name):
    try: m.actuator(name); return True
    except KeyError: return False

_SUB = {}
def _sub(kind="g1"):
    """An opponent-only model so mj_differentiatePos works on the opponent qpos slice."""
    if kind not in _SUB: _SUB[kind] = opponent_spec(kind).compile()
    return _SUB[kind]

KEYS = ["torso_link", "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
        "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link"]

def retarget_g1_to_r1(q, iters=30, damp=1e-2, prior=0.05, max_step=0.25):
    """Per frame: start from the G1's own joint angles (both robots use Unitree joint names), then damped least squares
    on the keypoints with a pull toward that prior. The prior picks one arm solution, so the IK cannot flip between
    equivalent shoulder solutions from frame to frame; `max_step` caps any joint at 0.25 rad per frame (12.5 rad/s)."""
    g, r = _sub("g1"), _sub("r1"); dg, dr = mujoco.MjData(g), mujoco.MjData(r)
    sc = R1_PELVIS_Z / G1_PELVIS_Z; out = np.zeros((len(q), r.nq)); qr = r.qpos0.copy()
    kb = [r.body(k).id for k in KEYS]; gb = [g.body(k).id for k in KEYS]; J = np.zeros((3, r.nv))
    lo, hi = r.jnt_range[1:, 0], r.jnt_range[1:, 1]
    g_adr = {g.joint(i).name: g.jnt_qposadr[i] for i in range(1, g.njnt)}
    r_names = [r.joint(i).name for i in range(1, r.njnt)]
    prev = None
    for t in range(len(q)):
        dg.qpos[:] = q[t]; mujoco.mj_kinematics(g, dg)
        p0 = dg.xpos[g.body("pelvis").id]; tgt = [p0 + sc * (dg.xpos[b] - p0) for b in gb]
        qp = np.clip([q[t, g_adr[n]] if n in g_adr else 0.0 for n in r_names], lo, hi)
        qr[:7] = q[t, :7]; qr[2] = q[t, 2] * sc; qr[7:] = qp
        for _ in range(iters):
            dr.qpos[:] = qr; mujoco.mj_kinematics(r, dr)
            mujoco.mj_comPos(r, dr)
            rows, errs = [], []
            for b, tp in zip(kb, tgt):
                mujoco.mj_jacBody(r, dr, J, None, b); rows.append(J[:, 6:].copy()); errs.append(tp - dr.xpos[b])
            A = np.vstack(rows); e = np.concatenate(errs); n = A.shape[1]
            dq = np.linalg.solve(A.T @ A + (damp + prior) * np.eye(n), A.T @ e + prior * (qp - qr[7:]))
            qr[7:] = np.clip(qr[7:] + dq, lo, hi)
        if prev is not None: qr[7:] = prev + np.clip(qr[7:] - prev, -max_step, max_step)
        out[t] = qr; prev = qr[7:].copy()
    return out
