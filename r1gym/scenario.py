"""The demo / demo-check scenario, shared by scripts/demo.py and scripts/demo_check.py.

The opponent (a second, physical R1 replaying a held-out fight clip in place) is what hits the R1 from the front: every
contact is a real MuJoCo contact with force bounded by the opponent's motors. On top of that, three scripted pushes come
from directions the opponent can't reach (side, side, behind); the demo draws them as arrows."""
from __future__ import annotations
import numpy as np
from .env import R1HitEnv
from .opponent import FightReplay
from .perturb import make_hit

DEMO_CLIP, DEMO_DISTANCE, DEMO_SECONDS = "kfab_202", 0.55, 12.0
# (kind, location, impulse N*s, t0 s, yaw): yaw = direction the R1 is pushed toward (0 = backward)
SCRIPTED = [("push", "l_shoulder", 5, 3.0, np.pi / 2), ("grab", "r_shoulder", 4, 6.0, -np.pi / 2), ("push", "pelvis", 6, 8.5, np.pi)]

def demo_env(m, cfg, hit_seed=3, delay=2, distance=DEMO_DISTANCE, clip=DEMO_CLIP, seconds=DEMO_SECONDS):
    """Env with the opponent and the scripted pushes, reset and ready; returns (env, first obs)."""
    e = R1HitEnv(seed=1, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
    e.opp = FightReplay(m, clip, distance=distance, cfg=cfg)
    rng = np.random.default_rng(hit_seed)   # one stream per run: each hit gets its own duration draw
    hits = [make_hit(k, loc, imp, t, yaw, rng) for k, loc, imp, t, yaw in SCRIPTED]
    return e, e.reset(hits=hits, delay=delay)

def variant_kwargs(s):
    """Demo variant s: re-sampled hit durations, obs delay 0-2 steps, opponent distance 0.50-0.65 m."""
    return dict(hit_seed=100 + s, delay=s % 3, distance=[0.5, 0.55, 0.6, 0.65][s % 4])
