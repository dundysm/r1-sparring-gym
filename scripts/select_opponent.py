"""Opponent checkpoint comparison -> results/opponent_tracker/selection.json.

Train + val clips only (never test): 2 distances x 6 clean starts per clip, 10 s bouts, no scripted hits, against the
arms-down stepping base (it stands its ground: its turn command is off, and hold-your-spot is off too, overriding its gym.json). Per bout: does the opponent fall, does the R1 fall,
the opponent's bearing off the R1's starting axis (0 = straight in front; median and 90th percentile), and its distance
error. --opponent label=path.onnx[:nostyle] (the motion_index.json next to the ONNX; nostyle = the clips without the style
cuts, i.e. the motion the stage-3 tracker was trained on: git show 5787b83:results/opponent_tracker/policy.onnx and
motion_index.json)."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.opponent import FightReplay
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start

def compare(onnx, style=True, r1="results/policy_mjlab/policy.onnx", seconds=10.0):
    c = load_controller(f"r1={r1}"); cfg = {**controller_cfg(c), "hold_position": False, "opponent_mode": "policy", "opponent_policy": str(Path(onnx).resolve()), "opponent_style_cuts": style}
    m = build_spec(opponent=True, cfg=cfg).compile(); clips, dists = TRAIN_CLIPS + VAL_CLIPS, [0.55, 0.65]; good = clean_starts(cfg, clips, dists)
    eps = [(cl, d, pick_start(good[(cl, d)], u)) for cl in clips for d in dists for u in np.linspace(0, 5.9, 6)]
    def run(ep):
        cl, d, t0 = ep; e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
        e.opp = FightReplay(m, cl, distance=d, cfg=cfg); e.opp_t0 = t0; o = e.reset(hits=[], delay=1); op = e.opp; b, dd = [], []
        for _ in range(int(seconds / e.dt)):
            o, r, dn, info = e.step(c.act(o[None])[0])
            if info["fallen"]: return dict(fell=True, ofell=False, b=b, dd=dd)
            if e.d.xmat[op.pel][8] < 0.5 or e.d.xpos[op.pel][2] < 0.25: return dict(fell=False, ofell=True, b=b, dd=dd)
            v = e.d.xpos[op.pel][:2] - e.d.xpos[e.r.pel][:2]; b.append(abs(np.degrees(np.arctan2(v[1], v[0])))); dd.append(abs(np.linalg.norm(v) - d))
        return dict(fell=False, ofell=False, b=b, dd=dd)
    rows = [r for r in ThreadPoolExecutor(8).map(run, eps)]; ok = [r for r in rows if r["b"]]
    return dict(bouts=len(rows), opponent_fell=sum(r["ofell"] for r in rows), r1_fell=sum(r["fell"] for r in rows),
                bearing_median_deg=round(float(np.median([np.median(r["b"]) for r in ok])), 1), bearing_p90_median_deg=round(float(np.median([np.percentile(r["b"], 90) for r in ok])), 1),
                bearing_p90_worst_deg=round(float(max(np.percentile(r["b"], 90) for r in ok)), 1), dist_err_median_m=round(float(np.median([np.median(r["dd"]) for r in ok])), 3))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--opponent", nargs="+", required=True); ap.add_argument("--out", default="results/opponent_tracker/selection.json"); a = ap.parse_args()
    res = {"about": __doc__}
    for spec in a.opponent:
        lab, rest = spec.split("=", 1); path, flag = (rest[:-8], "nostyle") if rest.endswith(":nostyle") else (rest, "")   # Windows paths have a colon
        res[lab] = compare(path, style=flag != "nostyle"); print(lab, json.dumps(res[lab]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
