"""How well the tracking-policy opponent performs its clips here (CPU MuJoCo), alone, before it ever meets the R1.
For each clip, N start times spread over the clip, 10 s each: does the opponent fall (pelvis tilt > 60 deg or z < 0.25 m),
when, and how far its joints and torso are from the reference (RMS joint error, torso height error) while it is up."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r1gym.model import build_spec, load_cfg
from r1gym.env import R1HitEnv
from r1gym.opponent import FightReplay, list_clips

def run(m, cfg, clip, t0, seconds=10.0):
    e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
    e.opp = FightReplay(m, clip, distance=4.0, cfg=cfg); e.opp_t0 = t0; e.reset(hits=[], delay=0); op = e.opp
    jerr, zerr = [], []
    for k in range(int(seconds / e.dt)):
        e.step(np.zeros(24)); d = e.d
        if d.xmat[op.pel][8] < 0.5 or d.xpos[op.pel][2] < 0.25: return round(k * e.dt, 2), jerr, zerr
        ref = op.qpos_at(d.time + t0)
        jerr.append(np.sqrt(np.mean((d.qpos[op.qadr + 7:op.qadr + op.nq] - ref[7:]) ** 2))); zerr.append(abs(d.qpos[op.qadr + 2] - ref[2]))
    return None, jerr, zerr

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default=""); ap.add_argument("--starts", type=int, default=8); ap.add_argument("--out", default="")
    a = ap.parse_args(); cfg = load_cfg()
    if a.policy: cfg.update(opponent_mode="policy", opponent_policy=a.policy)
    m = build_spec(opponent=True, cfg=cfg).compile(); out = {}
    for clip in list_clips():
        T = FightReplay(m, clip, distance=4.0, cfg=cfg).T / 50.0; rows = [run(m, cfg, clip, s * T / a.starts) for s in range(a.starts)]
        fell = [r[0] for r in rows]; j = np.concatenate([r[1] for r in rows if r[1]]) if any(r[1] for r in rows) else np.array([np.nan])
        z = np.concatenate([r[2] for r in rows if r[2]]) if any(r[2] for r in rows) else np.array([np.nan])
        out[clip] = dict(clip_s=round(T, 1), stays_up=f"{sum(f is None for f in fell)}/{len(fell)}", fall_times=fell,
                         joint_rmse_rad=round(float(np.mean(j)), 3), pelvis_z_err_m=round(float(np.mean(z)), 3))
        print(clip, json.dumps(out[clip]), flush=True)
    up = sum(int(v["stays_up"].split("/")[0]) for v in out.values()); n = sum(int(v["stays_up"].split("/")[1]) for v in out.values())
    print(f"overall: opponent stays up {up}/{n} runs ({up / n:.0%})")
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
