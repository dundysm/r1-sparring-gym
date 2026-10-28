"""Starting a bout, and the demo / demo-check scenario (scripts/demo.py, scripts/demo_check.py, scripts/eval_opponent.py).

The opponent (a second, physical R1 performing a fight clip on its own motors, kept in front of the R1) is what hits the
R1 from the front: every contact is a real MuJoCo contact with force bounded by the opponent's motors. On top of that,
three scripted pushes come from directions the opponent can't reach (side, side, behind); the demo draws them as arrows."""
from __future__ import annotations
import itertools
import numpy as np
from .env import R1HitEnv
from .model import build_spec, HOME, CTRL_JOINTS
from .opponent import FightReplay
from .perturb import make_hit

START_STEP, START_SPAN = 0.02, 6.0   # clean-start grid over the first 6 s of the clip [s]
# The R1's default poses the clean-start check uses, fixed so that a bout's start never depends on which controllers are
# in the run: Unitree's (HOME: PD, v1-v3 and the shipped arms-down stepping base) and the guard variant (gpu/r1_spar_task.py).
GUARD = dict(left_shoulder_pitch_joint=-0.85, left_shoulder_roll_joint=0.15, left_shoulder_yaw_joint=0.2, left_elbow_joint=-0.7, left_wrist_roll_joint=0.0,
             right_shoulder_pitch_joint=-0.85, right_shoulder_roll_joint=-0.15, right_shoulder_yaw_joint=-0.2, right_elbow_joint=-0.7, right_wrist_roll_joint=0.0)
REF_POSES = {"home": np.array(HOME, float), "guard": np.array([GUARD.get(j, h) for j, h in zip(CTRL_JOINTS, HOME)])}

def clean_starts(cfg, clips, distances, poses=REF_POSES, yaw=0.0):
    """For each (clip, distance), a mask over start times: True where no part of the opponent starts inside the R1 in any
    of the given default poses. A bout shouldn't begin with a foot or fist already inside the other robot. yaw: the R1's
    initial yaw [rad] (R1HitEnv.reset init_yaw), or several: clean at every one of them (off-axis bouts share a start)."""
    ok = {}; m = build_spec(opponent=True, cfg=cfg).compile()
    for pose, y in itertools.product(poses.values(), np.atleast_1d(yaw)):
        e = R1HitEnv(seed=0, cfg={**cfg, "home": list(pose), "perception": "truth"}, model=m, episode_s=1, max_impulse=0, train=False)   # contacts only: no camera needed
        for clip, dist in itertools.product(clips, distances):
            e.opp = FightReplay(m, clip, distance=dist, cfg=cfg); opp = e.opp.bodies
            good = ok.setdefault((clip, dist), np.ones(int(round(START_SPAN / START_STEP)), bool))
            for i in range(len(good)):
                e.opp_t0 = i * START_STEP; e.reset(hits=[], delay=0, init_yaw=float(y) if y else None); d = e.d
                for j in range(d.ncon):
                    b1, b2 = m.geom_bodyid[d.contact[j].geom1], m.geom_bodyid[d.contact[j].geom2]
                    if 0 not in (b1, b2) and (b1 in opp) != (b2 in opp): good[i] = False; break
    return ok

def pick_start(good, u):
    """Clean start time for a uniform draw u in [0, 6): uniform over the clean grid points."""
    idx = np.flatnonzero(good)
    if not len(idx): raise ValueError("no clean start time for this clip and distance")
    return round(float(idx[min(int(u / START_SPAN * len(idx)), len(idx) - 1)] * START_STEP), 3)

# The demo clip: the longest after the style cuts (14 s, so 12 s of video never loops). The stepping policies trained on
# Unitree's velocity task with pushes, never against a clip, so every clip is new to them.
DEMO_CLIP, DEMO_DISTANCE, DEMO_SECONDS = "kfab_201", 0.65, 12.0
# (kind, location, impulse N*s, t0 s, yaw): yaw = direction the R1 is pushed toward (0 = backward)
SCRIPTED = [("push", "l_shoulder", 5, 3.0, np.pi / 2), ("grab", "r_shoulder", 4, 6.0, -np.pi / 2), ("push", "pelvis", 6, 8.5, np.pi)]

def demo_env(m, cfg, hit_seed=3, delay=2, distance=DEMO_DISTANCE, clip=DEMO_CLIP, seconds=DEMO_SECONDS, t0=None, init_yaw=None, setup=None):
    """Env with the opponent and the scripted pushes, reset and ready; returns (env, first obs). t0=None: the first clip
    time at which the opponent doesn't start inside the R1 in any reference pose (plus this controller's own, if it isn't
    one), so every controller gets the same start, as in scripts/eval_opponent.py. init_yaw: start off-axis (the clean
    start is then judged at that yaw). setup(env): called before the reset (e.g. percept.attach)."""
    if t0 is None:
        own = np.array(cfg.get("home", HOME), float); poses = dict(REF_POSES)
        if not any(np.allclose(own, p, atol=1e-3) for p in poses.values()): poses["own"] = own
        t0 = pick_start(clean_starts(cfg, [clip], [distance], poses, yaw=init_yaw or 0.0)[(clip, distance)], 0.0)
    e = R1HitEnv(seed=1, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
    e.opp = FightReplay(m, clip, distance=distance, cfg=cfg); e.opp_t0 = t0
    if setup: setup(e)
    rng = np.random.default_rng(hit_seed)   # one stream per run: each hit gets its own duration draw
    hits = [make_hit(k, loc, imp, t, yaw, rng) for k, loc, imp, t, yaw in SCRIPTED]
    return e, e.reset(hits=hits, delay=delay, init_yaw=init_yaw)

def variant_kwargs(s):
    """Demo variant s: re-sampled hit durations, obs delay 0-2 steps, opponent distance 0.55-0.70 m."""
    return dict(hit_seed=100 + s, delay=s % 3, distance=[0.55, 0.6, 0.65, 0.7][s % 4])

# Off-axis bouts (phase 3, M1): the R1 starts turned so the opponent, placed on the bout's +x axis as always, sits at a
# bearing of +-30 or +-50 deg; the steered opponent keeps to that world axis (Tracker fixes it at the first step), so it
# stays off-axis unless the R1 turns. Every bearing of a (clip, distance, start draw) shares one start time, clean at all of
# them; init_yaw = -bearing. Each episode carries its own perception seed, the same for every perception source.
OFFAXIS_BEARINGS_DEG, OFFAXIS_STARTS, OFFAXIS_SEED = (30.0, -30.0, 50.0, -50.0), 6, 30_000

def offaxis_episodes(cfg, clips, distances, bearings_deg=OFFAXIS_BEARINGS_DEG, starts=OFFAXIS_STARTS, good=None):
    """[{clip, distance, start, t0, bearing_deg, init_yaw, pseed}]: clips x distances x start draws x bearings."""
    good = good or clean_starts(cfg, clips, distances, yaw=[-np.radians(b) for b in bearings_deg]); out = []
    for clip, dist, (k, u) in itertools.product(clips, distances, enumerate(np.linspace(0, 5.9, starts))):
        t0 = pick_start(good[(clip, dist)], u)
        for b in bearings_deg: out.append(dict(clip=clip, distance=dist, start=k, t0=t0, bearing_deg=b, init_yaw=-np.radians(b), pseed=OFFAXIS_SEED + len(out)))
    return out
