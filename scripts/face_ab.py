"""Turn-toward-the-opponent A/B for the stepping policies -> results/face_opponent_ab.json (the evidence behind gym.json).

1. Turn response: yaw-rate command held from standing for 6 s, mean measured rate after 2 s, 3 seeds.
2. Bouts with the shipped opponent on train + val clips (never test): 2 distances x 6 clean starts per clip, 12 s, the demo's
   scripted pushes, obs delay 1 step. Each policy runs with the command off, on, and on with a deadband (turn once the
   bearing error passes 20 deg, stop below 6 deg). Facing error = |bearing to the opponent - pelvis heading|, median per
   bout, then median over the bouts it survived; drift = pelvis distance from its start at the end; ring-out = got more
   than 1 m from its start at any time (rates over survived bouts).
--hold: the same bouts for the hold-your-spot outer loop (off / hold / hold_tight) -> results/hold_ab.json.
3. --val: survival (and up-and-within-1 m, drift) on the headline eval's validation episodes (scripts/eval_opponent.py, standard and hard tier) for each
   setting, merged into the same file. Settings are chosen on these, never on test."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start, SCRIPTED

SETTINGS = {"off": dict(face_opponent=False), "on": dict(face_opponent=True), "deadband": dict(face_opponent=True, face_deadband=[0.35, 0.1])}
# --hold: the hold-your-spot outer loop (r1gym/env.py _hold) instead, turn command off throughout
HOLD = {"off": dict(hold_position=False), "hold": dict(hold_position=True, hold_deadband=[0.25, 0.08]), "hold_tight": dict(hold_position=True, hold_deadband=[0.12, 0.05])}

def turn_response(c, cfg):
    out = {}; m = build_spec(cfg=cfg).compile()   # no opponent in the scene
    for wz in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        r = []
        for seed in range(3):
            e = R1HitEnv(seed=seed, cfg=cfg, model=m, episode_s=8, max_impulse=0, train=False); o = e.reset(hits=[], delay=1); e.command = np.array([0, 0, wz]); ys = []
            for _ in range(300):
                o, *_ = e.step(c.act(o[None])[0]); R = e.d.xmat[e.r.pel].reshape(3, 3); ys.append(np.arctan2(R[1, 0], R[0, 0]))
            r.append(np.mean(np.diff(np.unwrap(ys[100:]))) / e.dt)
        out[f"{wz:.1f}"] = round(float(np.mean(r)), 2)
    return out

def bouts(c, cfg, m, eps, seconds=12.0, settings=None):
    def run(ep):
        cl, d, t0 = ep; e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
        e.opp = FightReplay(m, cl, distance=d, cfg=cfg); e.opp_t0 = t0; rng = np.random.default_rng(3)
        o = e.reset(hits=[make_hit(k, l, i, t, y, rng) for k, l, i, t, y in SCRIPTED], delay=1); q = e.r.free_q; start = e.d.qpos[q:q + 2].copy(); f = []; maxr = 0.0
        for _ in range(int(seconds / e.dt)):
            o, r, dn, info = e.step(c.act(o[None])[0])
            if info["fallen"]: return dict(fell=True)
            v = e.d.xpos[min(e.opp.bodies)][:2] - e.d.xpos[e.r.pel][:2]; R = e.d.xmat[e.r.pel].reshape(3, 3)
            f.append(abs(np.degrees(np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.arctan2(R[1, 0], R[0, 0])))))))
            maxr = max(maxr, float(np.linalg.norm(e.d.qpos[q:q + 2] - start)))
        return dict(fell=False, face=float(np.median(f)), face90=float(np.percentile(f, 90)), drift=float(np.linalg.norm(e.d.qpos[q:q + 2] - start)), out=maxr > 1.0)
    rows = list(ThreadPoolExecutor(8).map(run, eps)); ok = [r for r in rows if not r["fell"]]
    return dict(bouts=len(rows), falls=sum(r["fell"] for r in rows), facing_err_median_deg=round(float(np.median([r["face"] for r in ok])), 1),
                facing_err_p90_median_deg=round(float(np.median([r["face90"] for r in ok])), 1), drift_mean_m=round(float(np.mean([r["drift"] for r in ok])), 2),
                ring_out=round(float(np.mean([r["out"] for r in ok])), 3))

def val_survival(c, base, m, settings=SETTINGS):
    import eval_opponent as eo
    from r1gym.model import load_cfg
    out = {}
    for tier in ("standard", "hard"):
        T = eo.TIERS[tier]; good = clean_starts({**load_cfg(), "opponent_mode": "policy"}, VAL_CLIPS, T["distances"]); eps = eo.episodes(VAL_CLIPS, 16, tier, good)
        out[tier] = {}
        for k, v in settings.items():
            rows = eo.run(c, eps, m, {**base, "face_deadband": None, **v}, T["seconds"], dr=T["dr"])
            out[tier][k] = dict(survival=round(float(np.mean([r["fell_s"] is None for r in rows])), 3),
                                in_ring=round(float(np.mean([r["fell_s"] is None and not r["ring_out"] for r in rows])), 3),
                                drift_m=round(float(np.mean([r["drift_m"] for r in rows if r["fell_s"] is None] or [0])), 2))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="+", default=["arms_down=results/policy_mjlab/policy.onnx", "guard=results/policy_mjlab_guard/policy.onnx"])
    ap.add_argument("--out", default=None); ap.add_argument("--val", action="store_true"); ap.add_argument("--hold", action="store_true"); a = ap.parse_args()
    S = HOLD if a.hold else SETTINGS; a.out = a.out or ("results/hold_ab.json" if a.hold else "results/face_opponent_ab.json")
    if a.val:   # merge validation survival into the existing file
        sys.path.insert(0, str(Path(__file__).resolve().parent)); res = json.load(open(a.out)); res["about"] = __doc__; res.setdefault("val_survival", {})
        for spec in a.policy:
            lab, path = spec.split("=", 1); c = load_controller(f"{lab}={path}"); base = {**controller_cfg(c), "opponent_mode": "policy"}
            res["val_survival"][lab] = val_survival(c, base, build_spec(opponent=True, cfg=base).compile(), S); print(lab, res["val_survival"][lab], flush=True)
        json.dump(res, open(a.out, "w"), indent=1); sys.exit(0)
    res = {"about": __doc__, "turn_response_cmd_to_measured_rad_s": {}, "bouts": {}}
    for spec in a.policy:
        lab, path = spec.split("=", 1); c = load_controller(f"{lab}={path}"); base = {**controller_cfg(c), "opponent_mode": "policy"}
        m = build_spec(opponent=True, cfg=base).compile()
        if not a.hold: res["turn_response_cmd_to_measured_rad_s"][lab] = turn_response(c, base); print(lab, res["turn_response_cmd_to_measured_rad_s"][lab], flush=True)
        clips, dists = TRAIN_CLIPS + VAL_CLIPS, [0.55, 0.65]; good = clean_starts(base, clips, dists)
        eps = [(cl, d, pick_start(good[(cl, d)], u)) for cl in clips for d in dists for u in np.linspace(0, 5.9, 6)]
        res["bouts"][lab] = {k: bouts(c, {**base, **v}, m, eps) for k, v in S.items()}
        print(lab, json.dumps(res["bouts"][lab]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
