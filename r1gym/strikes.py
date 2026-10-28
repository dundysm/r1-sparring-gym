"""Strike library: the hand and foot strikes in the motion the opponent's tracker performs (FightReplay.q at distance 0,
i.e. after retarget, cleaning, tracker and style cuts and grounding, in place and facing -x, so 'forward' is toward the R1).

Rule (RULE; scripts/strike_library.py writes r1gym/assets/motions/strikes.json and audits it against hand labels):
  effector  the palm sites (wrist + 0.10 m) and foot sites of r1.xml
  onset     forward speed (-x, toward the R1) rises through v_on = 1.5 m/s
  apex      the most forward point within 0.4 s of onset; it must be >= 0.25 m ahead of the spot (the in-place clip's centre)
  end       the first frame from the apex on whose forward speed is <= 0
  kicks     apex >= 0.30 m above the floor (every lower foot event is a step)
Per strike: frames, rise time, speeds, the apex in the heading frame (forward, the opponent's left, height), the approach
direction over the last 0.1 s before the apex (straight / hook / uppercut / downward: its largest component), the band of
the R1 it is level with, the opponent's torso height at the apex, and the unwarped reach: the largest bout distance
(pelvis to spot) at which the apex still meets the front of the R1's head / torso shell, the R1 standing at home."""
from __future__ import annotations
import json, numpy as np, mujoco
from dataclasses import dataclass, asdict
from functools import lru_cache
from . import ASSETS

EFFECTORS = {"l_hand": "left_palm", "r_hand": "right_palm", "l_foot": "left_foot", "r_foot": "right_foot"}
RULE = dict(trigger="forward", v_on_mps=1.5, apex_within_s=0.4, apex_min_fwd_m=0.25, apex_ref="spot", kick_min_z_m=0.30, approach_s=0.1,
            bands_m=dict(head=[1.02, 1.23], torso=[0.79, 1.02]))   # head first where they overlap (1.02-1.08): the face sits in front
PATH = ASSETS / "motions" / "strikes.json"

@dataclass(frozen=True)
class Strike:
    clip: str; split: str; limb: str; onset: int; apex: int; end: int; fps: float
    rise_s: float; peak_fwd_mps: float; peak_speed_mps: float
    apex_m: tuple            # (forward of the spot, the opponent's left, height above the floor)
    approach: tuple          # unit direction over the last approach_s before the apex, same axes (up for the third)
    type: str; band: str; torso_z_m: float
    reach_head_m: float; reach_torso_m: float   # bout distance at which the apex meets the head / torso front
    @property
    def t_on(self): return self.onset / self.fps
    @property
    def t_apex(self): return self.apex / self.fps
    @property
    def t_end(self): return self.end / self.fps
    @property
    def hand(self): return self.limb.endswith("hand")

@lru_cache(None)
def r1_surfaces():
    """Front x, half-width and z range of the R1's head (head_pitch + head_yaw meshes) and torso (waist_yaw shell), standing at
    home with its pelvis at the origin facing +x: the hull vertices' extents (MuJoCo collides the hull)."""
    from .model import HOME
    from .opponent import _sub
    r = _sub("r1"); d = mujoco.MjData(r); d.qpos[:3] = [0, 0, 0.74]; d.qpos[3] = 1; d.qpos[7:] = HOME; mujoco.mj_kinematics(r, d)
    zones = dict(head=("head_pitch_link", "head_yaw_link"), torso=("waist_yaw_link",)); out = {}
    for z, names in zones.items():
        V = []
        for g in range(r.ngeom):
            if r.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and r.mesh(r.geom_dataid[g]).name in names:
                a, n = r.mesh_vertadr[r.geom_dataid[g]], r.mesh_vertnum[r.geom_dataid[g]]
                V.append(d.geom_xpos[g] + r.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T)
        V = np.concatenate(V); out[z] = dict(front_m=round(float(V[:, 0].max()), 3), half_width_m=round(float(np.abs(V[:, 1]).max()), 3),
                                             z_m=[round(float(V[:, 2].min()), 3), round(float(V[:, 2].max()), 3)])
    return out

def effector_paths(q):
    """Effector sites (EFFECTORS order) and the torso origin per frame of an R1 qpos trajectory."""
    from .opponent import _sub
    r = _sub("r1"); d = mujoco.MjData(r); sid = [r.site(s).id for s in EFFECTORS.values()]; tor = r.body("torso_link").id
    P, Tq = np.zeros((len(q), len(sid), 3)), np.zeros((len(q), 3))
    for t in range(len(q)): d.qpos[:] = q[t]; mujoco.mj_kinematics(r, d); P[t] = d.site_xpos[sid]; Tq[t] = d.xpos[tor]
    return P, Tq

def _band(z, bands):
    return "head" if bands["head"][0] <= z <= bands["head"][1] else "torso" if bands["torso"][0] <= z < bands["head"][0] else "high" if z > bands["head"][1] else "low"

def detect(q, fps, clip="", split="", rule=None, paths=None, stats=None):
    """Strikes in an in-place R1 clip at distance 0 (FightReplay(..., distance=0).q). The clip loops seamlessly, so speeds use
    circular differences; a strike that would end across the loop seam is dropped (every window lies inside its clip) and
    counted in stats['dropped_at_seam']. rule overrides RULE (scripts/strike_library.py uses it to explain the two design
    probes' counts)."""
    R = {**RULE, **(rule or {})}; P, Tq = paths if paths is not None else effector_paths(q); T = len(q); S = []; fr = r1_surfaces()
    H = P * [-1.0, -1.0, 1.0]                                                           # heading frame: forward (-x), left (-y), up
    V = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) * fps / 2; sp = np.linalg.norm(V, axis=2)
    w, k = int(round(R["apex_within_s"] * fps)), max(1, int(round(R["approach_s"] * fps)))
    for j, limb in enumerate(EFFECTORS):
        v = V[:, j, 0] if R["trigger"] == "forward" else sp[:, j]; t = 0
        while t < T:
            if not (v[t] > R["v_on_mps"] >= v[t - 1]): t += 1; continue                 # rising edge (t - 1 wraps: the loop)
            a = t; ap = a + int(np.argmax(H[a:min(T, a + w + 1), j, 0])); e = ap
            while e < T and V[e, j, 0] > 0: e += 1
            t = max(e, a + 1)
            fwd = H[ap, j, 0] - (0.0 if R["apex_ref"] == "spot" else -Tq[ap, 0])               # ahead of the spot, or of its own torso
            if fwd < R["apex_min_fwd_m"] or (limb.endswith("foot") and H[ap, j, 2] < R["kick_min_z_m"]): continue
            if e >= T:
                if stats is not None: stats["dropped_at_seam"] = stats.get("dropped_at_seam", 0) + 1
                continue
            u = H[ap, j] - H[max(a, ap - k), j]; u = u / (np.linalg.norm(u) + 1e-12)
            typ = ("straight", "hook", "uppercut", "downward")[int(np.argmax([u[0], abs(u[1]), u[2], -u[2]]))]
            S.append(Strike(clip, split, limb, int(a), int(ap), int(e), float(fps), round((ap - a) / fps, 3), round(float(V[a:ap + 1, j, 0].max()), 3),
                            round(float(sp[a:ap + 1, j].max()), 3), tuple(round(float(x), 3) for x in H[ap, j]), tuple(round(float(x), 3) for x in u), typ,
                            _band(float(H[ap, j, 2]), R["bands_m"]), round(float(Tq[ap, 2]), 3),
                            round(float(H[ap, j, 0] + fr["head"]["front_m"]), 3), round(float(H[ap, j, 0] + fr["torso"]["front_m"]), 3)))
    return sorted(S, key=lambda s: (s.onset, s.limb))

def proposals(q, fps, v_min=1.0, sep_s=0.2, paths=None):
    """Detector-independent candidates for the hand-label audit: per effector, local maxima of its total speed (any direction)
    >= v_min, at least sep_s apart. Every strike under RULE has one (forward speed > 1.5 m/s implies total speed > 1)."""
    P, _ = paths if paths is not None else effector_paths(q); sp = np.linalg.norm((np.roll(P, -1, 0) - np.roll(P, 1, 0)) * fps / 2, axis=2); out = []
    for j, limb in enumerate(EFFECTORS):
        x = sp[:, j]; pk = [t for t in range(1, len(x) - 1) if x[t] >= v_min and x[t] >= x[t - 1] and x[t] > x[t + 1]]
        keep = []
        for t in sorted(pk, key=lambda t: -x[t]):                                         # strongest first, then non-maximum suppression
            if all(abs(t - u) >= sep_s * fps for u in keep): keep.append(t)
        out += [(limb, int(t), round(float(x[t]), 3)) for t in sorted(keep)]
    return out

# hand-labelled candidates: every val proposal, a seeded sample of 36 train ones, then (round 2, added because round 1 held only
# 13 clear strikes, before any precision / recall was computed) 36 more train ones drawn from the hands only: hands get aimed
AUDIT = dict(val="all", train=36, seed=0, train_hands=36, seed2=1)

def split_of(clip):
    from .env import TRAIN_CLIPS, VAL_CLIPS, TEST_CLIPS
    return "train" if clip in TRAIN_CLIPS else "val" if clip in VAL_CLIPS else "test" if clip in TEST_CLIPS else "unused"

@lru_cache(None)
def motion(clip):
    """(q, fps) the tracker performs for a clip, at distance 0 (in place, facing -x): FightReplay.q."""
    from .model import load_cfg
    from .opponent import FightReplay
    r = FightReplay(_kin_model(), clip, distance=0.0, cfg={**load_cfg(), "opponent_mode": "kinematic"}); return r.q, r.fps

@lru_cache(None)
def _kin_model():
    from .model import build_spec, load_cfg
    return build_spec(opponent=True, cfg={**load_cfg(), "opponent_mode": "kinematic"}).compile()

def audit_candidates(clips):
    """The hand-label sample: (clip, split, limb, speed-peak frame, speed) for every val proposal and AUDIT['train'] train
    proposals drawn without replacement (seeded), in clip/frame order."""
    allp = [(c, split_of(c), *p) for c in clips for p in proposals(*motion(c))]
    tr = [p for p in allp if p[1] == "train"]; pick = set(np.random.default_rng(AUDIT["seed"]).choice(len(tr), min(AUDIT["train"], len(tr)), replace=False).tolist())
    rest = [i for i, p in enumerate(tr) if i not in pick and p[2].endswith("hand")]
    pick2 = {rest[i] for i in np.random.default_rng(AUDIT["seed2"]).choice(len(rest), min(AUDIT["train_hands"], len(rest)), replace=False)}
    return [p for p in allp if p[1] == "val"] + [p for i, p in enumerate(tr) if i in pick] + [p for i, p in enumerate(tr) if i in pick2]

def matches(s: Strike, limb, frame, tol_s=0.1):
    """A labelled candidate (limb, speed-peak frame) is the detector's strike s if it is the same effector and the peak lies
    within tol_s of the strike's [onset, end]."""
    k = tol_s * s.fps; return s.limb == limb and s.onset - k <= frame <= s.end + k

@lru_cache(None)
def _lib(path=str(PATH)):
    return json.load(open(path))

def load(split=None, clip=None, path=PATH):
    """Library strikes, optionally of one split ('train' | 'val' | 'test') or one clip."""
    keys = set(Strike.__dataclass_fields__)
    S = [Strike(**{k: tuple(v) if isinstance(v, list) else v for k, v in s.items() if k in keys}) for s in _lib(str(path))["strikes"]]
    return [s for s in S if (split is None or s.split == split) and (clip is None or s.clip == clip)]

def clip_frames(clip, path=PATH):
    """Frame count of the clip the library was detected on (FightReplay.T); FightReplay.strikes checks it."""
    return _lib(str(path))["per_clip"][clip]["frames"]

def windows_for(clip, lag_s=0.0, hands_only=False, path=PATH):
    """(t0, t1, limb) in clip seconds per library strike: [onset, end] shifted by the tracker's lag (the strike the opponent
    actually throws comes that much after its reference). limb as HitMeter.windows reads it ('l_hand', 'r_foot', ...)."""
    return [(round(s.t_on + lag_s, 4), round(s.t_end + lag_s, 4), s.limb) for s in load(clip=clip, path=path) if s.hand or not hands_only]

def as_dict(s: Strike): return asdict(s)
