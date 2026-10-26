"""How much exploration noise can the stiff PD stand take with no hits at all?
White Gaussian residual actions (std in action units; joint target = HOME + 0.25 * EMA(a)) for 5 s, 16 episodes per level.
This caps PPO's useful exploration std for the residual-on-stand policy (docs/MODEL_AUDIT.md)."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stds", type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.15, 0.25]); ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args(); cfg = load_cfg(); m = build_spec(cfg=cfg).compile(); out = {}
    for std in a.stds:
        falls = 0; rng = np.random.default_rng(0)
        for k in range(a.n):
            e = R1HitEnv(seed=k, cfg=cfg, model=m, episode_s=5, max_impulse=0); e.reset(hits=[])
            for t in range(250):
                if e.step(rng.normal(0, std, 24))[3]["fallen"]: falls += 1; break
        out[std] = falls / a.n; print(f"std {std:.2f} (~{cfg['action_scale'] * std:.3f} rad): {falls}/{a.n} fall in 5 s, no hits", flush=True)
    print(json.dumps(out))
