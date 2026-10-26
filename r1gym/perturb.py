"""Procedural hits: strikes (short impulses), pushes (longer), grabs (sustained pull), at a body + local offset.
Applied as external wrenches through MjData.xfrc_applied at the hit point, so they also create torque."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, mujoco

@dataclass
class Hit:
    kind: str            # strike | push | grab
    body: str            # R1 body name, e.g. torso_link
    t0: float            # start time [s]
    duration: float      # [s]
    force: np.ndarray    # world-frame force [N]
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # point in body frame [m]
    @property
    def impulse(self): return float(np.linalg.norm(self.force) * self.duration)

KINDS = {"strike": (0.05, 0.12), "push": (0.15, 0.3), "grab": (0.4, 0.8)}   # duration ranges [s]
# where on the R1 an opponent can land a hit (body, local point): chest, upper chest/face height, shoulders, hips, knees
LOCATIONS = {"chest": ("torso_link", [0.08, 0, 0.15]), "head": ("torso_link", [0.06, 0, 0.35]),
             "l_shoulder": ("left_shoulder_roll_link", [0, 0, 0]), "r_shoulder": ("right_shoulder_roll_link", [0, 0, 0]),
             "pelvis": ("pelvis", [0.05, 0, 0]), "l_knee": ("left_knee_link", [0.03, 0, -0.1]), "r_knee": ("right_knee_link", [0.03, 0, -0.1])}

def make_hit(kind, location, impulse, t0, yaw, rng=None, pitch=0.0):
    """Hit with a given impulse [N*s]; `yaw` = direction the force pushes the robot toward (0 = robot pushed backward,
    i.e. a frontal strike). A grab pulls (yaw+pi semantics are up to the caller)."""
    rng = rng or np.random.default_rng()
    lo, hi = KINDS[kind]; dur = float(rng.uniform(lo, hi))
    body, off = LOCATIONS[location]
    dirn = np.array([-np.cos(yaw) * np.cos(pitch), -np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    return Hit(kind, body, t0, dur, dirn * impulse / dur, np.array(off, float))

def random_schedule(rng, t_end, max_impulse, gap=(1.0, 2.5), first=0.6):
    hits, t = [], first + rng.uniform(0, 0.6)
    while t < t_end - 0.5:
        kind = rng.choice(["strike", "strike", "push", "grab"])
        loc = rng.choice(list(LOCATIONS))
        imp = rng.uniform(0.2, 1.0) * max_impulse * (0.6 if kind == "grab" else 1.0)
        h = make_hit(kind, loc, imp, t, rng.uniform(-np.pi, np.pi), rng, pitch=rng.uniform(-0.2, 0.2))
        hits.append(h); t += h.duration + rng.uniform(*gap)
    return hits

class HitApplier:
    def __init__(self, m: mujoco.MjModel):
        self.m = m; self.ids = {}
    def apply(self, d: mujoco.MjData, hits, t):
        """Write xfrc_applied for all hits active at time t. Returns active hit list."""
        d.xfrc_applied[:] = 0; active = []
        for h in hits:
            if h.t0 <= t < h.t0 + h.duration:
                b = self.ids.setdefault(h.body, self.m.body(h.body).id)
                R = d.xmat[b].reshape(3, 3); p = d.xpos[b] + R @ h.offset
                d.xfrc_applied[b, :3] += h.force
                d.xfrc_applied[b, 3:] += np.cross(p - d.xipos[b], h.force)   # torque about body CoM
                active.append(h)
        return active
