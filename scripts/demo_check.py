"""Headless check of the demo scenario (no rendering): fall time per controller for the fixed demo hit schedule,
plus N variants (hit durations re-sampled, obs delay 0-2, opponent distance 0.80-0.95 m) so the demo isn't a cherry-pick."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import PDStand, TorchPolicy
SCHED = [("strike", "chest", 4, 1.5, 0.0), ("push", "l_shoulder", 5, 3.0, np.pi / 2), ("strike", "head", 4, 4.6, 0.0),
         ("grab", "r_shoulder", 4, 6.0, -np.pi / 2), ("strike", "pelvis", 7, 7.8, np.pi), ("push", "chest", 8, 9.5, 0.0)]

def run(ctrl, m, cfg, hit_seed=3, delay=2, seconds=12.0, clip="kfab_202", distance=0.85):
    e = R1HitEnv(seed=1, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
    e.opp = FightReplay(m, clip, distance=distance)
    hits = [make_hit(k, l, i, t, y, np.random.default_rng(hit_seed)) for k, l, i, t, y in SCHED]
    o = e.reset(hits=hits, delay=delay)
    for _ in range(int(seconds / e.dt)):
        o, r, d, info = e.step(ctrl.act(o[None])[0])
        if info["fallen"]: return round(e.t * e.dt, 2)
    return None   # survived

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="+", default=["v2=results/policy/policy.pt"], help="label=path ..."); ap.add_argument("--variants", type=int, default=8)
    a = ap.parse_args(); cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile()
    out = {}
    cs = [PDStand()]
    for lp in a.policy:
        lab, path = lp.split("=", 1); c = TorchPolicy(path); c.name = f"rl_{lab}"; cs.append(c)
    for c in cs:
        out[c.name] = {"demo": run(c, m, cfg), "variants": [run(c, m, cfg, hit_seed=100 + s, delay=s % 3, distance=[0.8, 0.85, 0.9, 0.95][s % 4]) for s in range(a.variants)]}
    for k, v in out.items():   # survival time per run (None = survived the full clip)
        t = [x if x is not None else 12.0 for x in v["variants"]]; v["variants_mean_survival_s"] = round(float(np.mean(t)), 2)
        v["variants_survived_12s"] = sum(x is None for x in v["variants"])
    print(json.dumps(out))
