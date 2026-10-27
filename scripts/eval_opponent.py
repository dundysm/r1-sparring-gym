"""Headline eval: survive a physical R1 opponent (r1gym/opponent.py, puppet mode) replaying held-out fight clips.

Episodes: clips x opponent distances 0.50-0.70 m x reps; each rep samples the clip start time, the obs delay (0-2 steps) and,
with --pushes (default on), two scripted side/back pushes the opponent can't deliver (4-6 N*s). 10 s per episode.
Every controller sees exactly the same episodes, so differences are paired. --split test (default) uses TEST_CLIPS,
--split val uses VAL_CLIPS: choose candidates on val, report on test."""
import sys, json, time, argparse, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.env import R1HitEnv, VAL_CLIPS, TEST_CLIPS
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import PDStand, load_controller, controller_cfg
from r1gym.stats import wilson, paired_diff

DISTANCES = [0.5, 0.55, 0.6, 0.65, 0.7]

def episodes(clips, reps, pushes):
    out = []
    for k, (clip, dist, r) in enumerate(itertools.product(clips, DISTANCES, range(reps))):
        rng = np.random.default_rng(10_000 + k); hits = []
        if pushes:
            for t in sorted(rng.uniform(1.0, 9.0, 2)):
                yaw = rng.choice([np.pi / 2, -np.pi / 2, np.pi]) + rng.uniform(-0.3, 0.3)   # side or from behind
                hits.append(("push", str(rng.choice(["l_shoulder", "r_shoulder", "pelvis", "chest"])), float(rng.uniform(4, 6)), float(t), float(yaw)))
        out.append(dict(clip=clip, distance=dist, rep=r, t0=float(rng.uniform(0, 6)), delay=int(rng.integers(0, 3)), hits=hits, seed=10_000 + k))
    return out

def run(ctrl, eps, m, cfg, T=10.0, threads=8):
    reps = {}; envs = []
    for ep in eps:
        key = (ep["clip"], ep["distance"])
        if key not in reps: reps[key] = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg)
        e = R1HitEnv(seed=ep["seed"], cfg=cfg, model=m, episode_s=T, max_impulse=0, train=False)
        e.opp = reps[key]; e.opp_t0 = ep["t0"]; rng = np.random.default_rng(ep["seed"])
        hits = [make_hit(k, l, i, t, y, rng) for k, l, i, t, y in ep["hits"]]
        envs.append([e, e.reset(hits=hits, delay=ep["delay"]), None])
    pool = ThreadPoolExecutor(threads); n = len(envs)
    for t in range(int(T / envs[0][0].dt)):
        live = [i for i in range(n) if envs[i][2] is None]
        if not live: break
        A = ctrl.act(np.stack([envs[i][1] for i in live]))
        def st(j):
            x = envs[live[j]]; o, r, d, info = x[0].step(A[j]); x[1] = o
            e = x[0]; b = min(e.opp.bodies)   # opponent pelvis; deep stances reach z ~0.37, so judge by tilt
            if e.d.xmat[b][8] < 0.5 or e.d.xpos[b][2] < 0.25: e.opp_fell = True   # an opponent on the floor is no threat
            if info["fallen"]: x[2] = x[0].t * x[0].dt; x.append(x[0].d.qpos[:2].copy())   # where it was when it fell
        list(pool.map(st, range(len(live))))
    pool.shutdown()
    # drift: how far the R1's pelvis moved from its start (xy). The in-place opponent doesn't follow, so stepping back
    # also takes the R1 out of reach; report it next to survival.
    return [dict(clip=ep["clip"], distance=ep["distance"], fell_s=x[2], opp_fell=bool(getattr(x[0], "opp_fell", False)), drift_m=round(float(np.linalg.norm((x[3] if len(x) > 3 else x[0].d.qpos[:2]))), 3),
                 back_m=round(float(-(x[3] if len(x) > 3 else x[0].d.qpos[:2])[0]), 3)) for ep, x in zip(eps, envs)]

def summarise(rows, T):
    s = [x["fell_s"] is None for x in rows]; n = len(s); k = int(sum(s))
    ttf = [x["fell_s"] if x["fell_s"] is not None else T for x in rows]
    by_d = {d: round(float(np.mean([x["fell_s"] is None for x in rows if x["distance"] == d])), 3) for d in DISTANCES}
    surv = [x for x in rows if x["fell_s"] is None]
    return dict(episodes=n, survival=round(k / n, 3), survival_ci95=wilson(k, n), mean_time_to_fall_s=round(float(np.mean(ttf)), 2), survival_by_distance=by_d,
                survivors_mean_drift_m=round(float(np.mean([x["drift_m"] for x in surv])), 3) if surv else None,
                survivors_mean_back_m=round(float(np.mean([x["back_m"] for x in surv])), 3) if surv else None,
                survivors_back_over_0p3m=round(float(np.mean([x["back_m"] > 0.3 for x in surv])), 3) if surv else None,
                opponent_fell=round(float(np.mean([x["opp_fell"] for x in rows])), 3))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="*", default=["v1=results/policy_v1/policy.pt", "v2=results/policy/policy.pt", "v3=results/policy_v3_candidate/policy.pt"],
                    help="label=path (.pt residual policy, .onnx unitree_rl_mjlab policy)")
    ap.add_argument("--split", choices=["test", "val"], default="test"); ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--opponent", choices=["tracked", "puppet"], default="tracked",
                    help="tracked: the R1 on its own motors (default, realistic); puppet: assisted replay (stress test, up to 250 N horizontal assist)")
    ap.add_argument("--no_pushes", action="store_true"); ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--out", default="results/eval_opponent")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(); models = {}; tag = "" if a.opponent == "tracked" else "_puppet"
    if a.opponent == "puppet": cfg["opponent_mode"] = "puppet"
    clips = TEST_CLIPS if a.split == "test" else VAL_CLIPS; eps = episodes(clips, a.reps, not a.no_pushes)
    ctrls = [PDStand()]
    for lp in a.policy:
        lab, path = lp.split("=", 1)
        if Path(path).exists(): ctrls.append(load_controller(f"RL {lab}={path}"))
    raw, summ = {}, {}
    for c in ctrls:
        cc = {**controller_cfg(c), "opponent_mode": cfg["opponent_mode"]}; key = str(getattr(c, "cfg_path", "default"))   # own contract per controller
        if key not in models: models[key] = build_spec(opponent=True, cfg=cc).compile()
        t0 = time.time(); raw[c.name] = run(c, eps, models[key], cc, a.seconds); summ[c.name] = summarise(raw[c.name], a.seconds)
        print(c.name, json.dumps(summ[c.name]), f"({time.time() - t0:.0f} s)", flush=True)
    base = [x["fell_s"] is None for x in raw["pd_stand"]]
    for name in raw:
        if name != "pd_stand":
            d, ci = paired_diff([x["fell_s"] is None for x in raw[name]], base); summ[name]["survival_minus_pd"] = d; summ[name]["survival_minus_pd_ci95"] = ci
    meta = dict(opponent=a.opponent, split=a.split, clips=clips, distances=DISTANCES, reps=a.reps, pushes=not a.no_pushes, seconds=a.seconds, opponent_mode=cfg.get("opponent_mode", "puppet"))
    json.dump(dict(meta=meta, summary=summ), open(out / f"summary_{a.split}{tag}.json", "w"), indent=1)
    json.dump(dict(meta=meta, episodes=eps, raw=raw), open(out / f"raw_{a.split}{tag}.json", "w"))
    print(json.dumps(summ, indent=1))

if __name__ == "__main__": main()
