"""Cross-simulator check for a unitree_rl_mjlab policy: trained in MuJoCo Warp (mjlab), run here in CPU MuJoCo under
configs/r1_mjlab.yaml. If the obs/action contract were wrong, it would not stand or track velocity commands here.
Reports, per command, mean base velocity over the last 6 of 8 s (world frame, heading-aligned) and falls over N seeds."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r1gym.env import R1HitEnv
from r1gym.model import build_spec
from r1gym.policy import load_controller, controller_cfg

COMMANDS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.5)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("policy"); ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--out", default="")
    a = ap.parse_args(); c = load_controller(f"p={a.policy}"); cfg = controller_cfg(c); m = build_spec(cfg=cfg).compile(); out = []
    for cmd in COMMANDS:
        vel, falls = [], 0
        for s in range(a.seeds):
            e = R1HitEnv(seed=s, cfg=cfg, model=m, episode_s=8, max_impulse=0, train=False); o = e.reset(hits=[], delay=s % 3); e.command = np.array(cmd)
            o = e._raw_obs(); vs = []
            for t in range(400):
                o, r, d, info = e.step(c.act(o[None])[0])
                if info["fallen"]: falls += 1; break
                if t >= 100:
                    R = e.d.xmat[e.r.pel].reshape(3, 3); v = R.T @ e.d.qvel[e.r.free_v:e.r.free_v + 3]   # body-frame lin vel
                    vs.append([v[0], v[1], e.d.qvel[e.r.free_v + 5]])
            if vs: vel.append(np.mean(vs, 0))
        mv = np.mean(vel, 0).round(3).tolist() if vel else None
        out.append(dict(command=cmd, measured_vx_vy_wz=mv, falls=falls, seeds=a.seeds)); print(out[-1], flush=True)
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
