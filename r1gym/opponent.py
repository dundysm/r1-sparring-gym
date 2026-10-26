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
    def __init__(self, m: mujoco.MjModel, clip="kfab_200", distance=0.62, speed=1.0, prefix="opp_", kind=None):
        z = np.load(ASSETS / "motions" / f"{clip}.npz"); q = z["qpos"].astype(float); self.fps = float(z["fps"]) if "fps" in z else 50.0
        self.speed = speed; self.clip = clip; self.raw = q.copy(); self.distance = distance
        self.kind = kind or ("g1" if _has(m, prefix + "waist_pitch_joint") else "r1")
        q = self._retarget(q) if self.kind == "r1" else self._canon(q)
        self.q = q; self.T = len(q)
        j = m.joint(prefix + "floating_base_joint")
        self.qadr = m.jnt_qposadr[j.id]; self.vadr = m.jnt_dofadr[j.id]; self.nq = q.shape[1]
        self.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(prefix)}

    def _canon(self, q):
        q = q.copy(); distance = self.distance
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
        f = (t * self.fps * self.speed) % (self.T - 1); i = int(f); a = f - i
        q = (1 - a) * self.q[i] + a * self.q[i + 1]; q[3:7] /= np.linalg.norm(q[3:7]); return q

    def apply(self, m, d, t, dt):
        q0, q1 = self.qpos_at(t), self.qpos_at(t + dt)
        d.qpos[self.qadr:self.qadr + self.nq] = q0
        v = np.zeros(self.nq - 1); mujoco.mj_differentiatePos(_sub(self.kind), v, dt, q0, q1)
        d.qvel[self.vadr:self.vadr + self.nq - 1] = v

def _has(m, name):
    try: m.joint(name); return True
    except KeyError: return False

_SUB = {}
def _sub(kind="g1"):
    """An opponent-only model so mj_differentiatePos works on the opponent qpos slice."""
    if kind not in _SUB: _SUB[kind] = opponent_spec(kind).compile()
    return _SUB[kind]

KEYS = ["torso_link", "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
        "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link"]

def retarget_g1_to_r1(q, iters=30, damp=1e-2):
    g, r = _sub("g1"), _sub("r1"); dg, dr = mujoco.MjData(g), mujoco.MjData(r)
    sc = R1_PELVIS_Z / G1_PELVIS_Z; out = np.zeros((len(q), r.nq)); qr = r.qpos0.copy()
    kb = [r.body(k).id for k in KEYS]; gb = [g.body(k).id for k in KEYS]; J = np.zeros((3, r.nv))
    lo, hi = r.jnt_range[1:, 0], r.jnt_range[1:, 1]
    for t in range(len(q)):
        dg.qpos[:] = q[t]; mujoco.mj_kinematics(g, dg)
        p0 = dg.xpos[g.body("pelvis").id]; tgt = [p0 + sc * (dg.xpos[b] - p0) for b in gb]
        qr[:7] = q[t, :7]; qr[2] = q[t, 2] * sc
        for _ in range(iters):
            dr.qpos[:] = qr; mujoco.mj_kinematics(r, dr)
            mujoco.mj_comPos(r, dr)
            rows, errs = [], []
            for b, tp in zip(kb, tgt):
                mujoco.mj_jacBody(r, dr, J, None, b); rows.append(J[:, 6:].copy()); errs.append(tp - dr.xpos[b])
            A = np.vstack(rows); e = np.concatenate(errs)
            dq = np.linalg.solve(A.T @ A + damp * np.eye(A.shape[1]), A.T @ e)
            qr[7:] = np.clip(qr[7:] + dq, lo, hi)
        out[t] = qr
    return out
