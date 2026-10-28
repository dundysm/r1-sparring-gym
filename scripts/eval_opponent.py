"""Headline eval: survive a physical R1 opponent (r1gym/opponent.py; default: the tracking-policy opponent on its own motors)
performing held-out fight clips.

Tiers (--tier):
  standard  10 s; opponent 0.50-0.70 m away; two scripted side/back pushes the opponent can't deliver (4-6 N*s).
  hard      20 s; opponent 0.40-0.55 m away; six scripted strikes and pushes from any direction at any hit location
            (8-14 N*s); the R1's mass, COM, foot friction, motor strength, armature and command latency randomised per
            episode (the training ranges, r1gym/env.py). Set on the validation clip once the stepping policies
            saturated the standard tier (guard 1.000, arms-down 0.994 on test).
Each episode samples the clip start time, the obs delay (0-2 steps) and the hits. The start time is drawn uniformly from
the clean ones: those where no part of the opponent begins inside the R1 in either reference pose (REF_POSES), which
skips 7-85% of starts in the stored (shell-contact) runs, by clip and distance (meta.start_times_skipped). Every controller sees exactly the same
episodes, so differences are paired, and the episodes don't depend on which controllers are in the run. --split test
(default) uses TEST_CLIPS, --split val uses VAL_CLIPS: choose candidates on val, report on test."""
import sys, json, time, argparse, itertools, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.env import R1HitEnv, VAL_CLIPS, TEST_CLIPS, _DR_NOMINAL
from r1gym.model import build_spec, load_cfg, HOME
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit, LOCATIONS
from r1gym.policy import PDStand, load_controller, controller_cfg
from r1gym.stats import wilson, paired_diff
from r1gym.scenario import clean_starts, pick_start, REF_POSES, START_STEP, START_SPAN

TIERS = {"standard": dict(distances=[0.5, 0.55, 0.6, 0.65, 0.7], seconds=10.0, hits=2, impulse=(4, 6), dr=False, seed=10_000),
         "hard": dict(distances=[0.4, 0.45, 0.5, 0.55], seconds=20.0, hits=6, impulse=(8, 14), dr=True, seed=20_000)}
RING_M = 1.0       # the R1's pelvis got more than 1 m from where it started (pushed or stepped): out of the sparring ring
def episodes(clips, reps, tier, good, hits_on=True, seconds=None):
    T = TIERS[tier]; out = []; S = seconds or T["seconds"]
    for k, (clip, dist, r) in enumerate(itertools.product(clips, T["distances"], range(reps))):
        rng = np.random.default_rng(T["seed"] + k); hits = []; ep = {}
        if hits_on and tier == "standard":
            for t in sorted(rng.uniform(1.0, 9.0, 2)):
                yaw = rng.choice([np.pi / 2, -np.pi / 2, np.pi]) + rng.uniform(-0.3, 0.3)   # side or from behind
                hits.append(("push", str(rng.choice(["l_shoulder", "r_shoulder", "pelvis", "chest"])), float(rng.uniform(4, 6)), float(t), float(yaw)))
        elif hits_on:   # strikes and pushes, any direction (yaw = where the R1 is pushed toward), any location incl. head and knees
            for t in sorted(rng.uniform(1.0, S - 1.0, T["hits"])):
                hits.append((str(rng.choice(["strike", "push"])), str(rng.choice(list(LOCATIONS))), float(rng.uniform(*T["impulse"])), float(t), float(rng.uniform(-np.pi, np.pi))))
        if T["dr"]:   # the env (DR, sensor noise) and the hit durations draw from their own streams, independent of the hits
            ep = dict(env_seed=int(np.random.SeedSequence([T["seed"] + k, 1]).generate_state(1)[0]), hit_seed=int(np.random.SeedSequence([T["seed"] + k, 2]).generate_state(1)[0]))
        t0 = pick_start(good[(clip, dist)], float(rng.uniform(0, 6)))
        out.append(dict(clip=clip, distance=dist, rep=r, t0=t0, delay=int(rng.integers(0, 3)), hits=hits, seed=T["seed"] + k, **ep))
    return out

def run(ctrl, eps, m, cfg, T=10.0, threads=8, dr=False):
    # DR envs write their sampled parameters into the model before each step, so with DR each thread steps its own copy
    ms = [copy.deepcopy(m) for _ in range(threads)] if dr else [m] * threads
    reps = {}; envs = []
    for i, ep in enumerate(eps):
        key = (ep["clip"], ep["distance"])
        if key not in reps: reps[key] = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg)
        e = R1HitEnv(seed=ep.get("env_seed", ep["seed"]), cfg=cfg, model=ms[i % threads], episode_s=T, max_impulse=0, train=False, dr=dr)
        e.opp = reps[key]; e.opp_t0 = ep["t0"]; rng = np.random.default_rng(ep.get("hit_seed", ep["seed"]))
        hits = [make_hit(k, l, imp, t, y, rng) for k, l, imp, t, y in ep["hits"]]
        o = e.reset(hits=hits, delay=ep["delay"]); q = e.r.free_q
        envs.append(dict(e=e, o=o, fell=None, at=None, start=e.d.qpos[q:q + 2].copy(), maxr=0.0, opp_fell=False))
    groups = [list(range(g, len(envs), threads)) for g in range(threads)]; pool = ThreadPoolExecutor(threads)
    for _ in range(int(T / envs[0]["e"].dt)):
        live = [i for i in range(len(envs)) if envs[i]["fell"] is None]
        if not live: break
        A = dict(zip(live, ctrl.act(np.stack([envs[i]["o"] for i in live]))))
        def st(g):   # one thread per model copy: its envs step one after another
            for i in g:
                if i not in A: continue
                x = envs[i]; e = x["e"]; x["o"], r, d, info = e.step(A[i]); q = e.r.free_q; xy = e.d.qpos[q:q + 2]
                x["maxr"] = max(x["maxr"], float(np.linalg.norm(xy - x["start"])))
                b = min(e.opp.bodies)   # opponent pelvis; deep stances reach z ~0.37, so judge by tilt
                if e.d.xmat[b][8] < 0.5 or e.d.xpos[b][2] < 0.25: x["opp_fell"] = True   # an opponent on the floor is no threat
                if info["fallen"]: x["fell"] = e.t * e.dt; x["at"] = xy.copy()   # where it was when it fell
        list(pool.map(st, groups))
    pool.shutdown()
    if dr:   # the copies die here; drop their cached nominals so a later model can't pick up a stale entry by a reused id()
        for x in ms: _DR_NOMINAL.pop(id(x), None)
    # drift: how far the R1's pelvis is from its start (xy) at the end (or where it fell), and the ring-out rate. The tracked
    # opponent is steered after the R1; the puppet stays where it was placed, so there stepping back also leaves its reach.
    rows = []
    for ep, x in zip(eps, envs):
        q = x["e"].r.free_q; p = (x["at"] if x["at"] is not None else x["e"].d.qpos[q:q + 2]) - x["start"]
        rows.append(dict(clip=ep["clip"], distance=ep["distance"], fell_s=x["fell"], opp_fell=x["opp_fell"], drift_m=round(float(np.linalg.norm(p)), 3),
                         back_m=round(float(-p[0]), 3), max_disp_m=round(x["maxr"], 3), ring_out=x["maxr"] > RING_M))
    return rows

def summarise(rows, T):
    s = [x["fell_s"] is None for x in rows]; n = len(s); k = int(sum(s))
    ring = [x["fell_s"] is None and not x["ring_out"] for x in rows]
    ttf = [x["fell_s"] if x["fell_s"] is not None else T for x in rows]
    by_d = {d: round(float(np.mean([x["fell_s"] is None for x in rows if x["distance"] == d])), 3) for d in sorted({x["distance"] for x in rows})}
    surv = [x for x in rows if x["fell_s"] is None]
    return dict(episodes=n, survival=round(k / n, 3), survival_ci95=wilson(k, n), mean_time_to_fall_s=round(float(np.mean(ttf)), 2), survival_by_distance=by_d,
                survival_in_ring=round(float(np.mean(ring)), 3), survival_in_ring_ci95=wilson(int(sum(ring)), n),
                survivors_ring_out=round(float(np.mean([x["ring_out"] for x in surv])), 3) if surv else None,
                survivors_mean_drift_m=round(float(np.mean([x["drift_m"] for x in surv])), 3) if surv else None,
                survivors_mean_back_m=round(float(np.mean([x["back_m"] for x in surv])), 3) if surv else None,
                survivors_back_over_0p3m=round(float(np.mean([x["back_m"] > 0.3 for x in surv])), 3) if surv else None,
                opponent_fell=round(float(np.mean([x["opp_fell"] for x in rows])), 3))

def pairs(raw, summ):
    """Paired differences on the same episodes between every two controllers (and vs PD): survival, and up and in the ring."""
    up = {n: [x["fell_s"] is None for x in r] for n, r in raw.items()}
    ring = {n: [x["fell_s"] is None and not x["ring_out"] for x in r] for n, r in raw.items()}
    for name in raw:
        summ[name]["survival_minus"] = {o: dict(zip(["diff", "ci95"], paired_diff(up[name], up[o]))) for o in raw if o != name}
        summ[name]["survival_in_ring_minus"] = {o: dict(zip(["diff", "ci95"], paired_diff(ring[name], ring[o]))) for o in raw if o != name}
        if name != "pd_stand": summ[name]["survival_minus_pd"], summ[name]["survival_minus_pd_ci95"] = paired_diff(up[name], up["pd_stand"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="*", default=["v1=results/policy_v1/policy.pt", "v2=results/policy/policy.pt", "v3=results/policy_v3_candidate/policy.pt"],
                    help="label=path (.pt residual policy, .onnx unitree_rl_mjlab policy)")
    ap.add_argument("--split", choices=["test", "val"], default="test"); ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--tier", choices=list(TIERS), default="standard")
    ap.add_argument("--opponent", choices=["tracked", "puppet"], default="tracked",
                    help="tracked: the R1 on its own motors (default, realistic); puppet: assisted replay (stress test, up to 250 N horizontal assist)")
    ap.add_argument("--no_pushes", action="store_true"); ap.add_argument("--seconds", type=float, default=None); ap.add_argument("--out", default="results/eval_opponent")
    ap.add_argument("--from_raw", action="store_true", help="recompute summary_*.json from the stored raw_*.json (no simulation)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tier = TIERS[a.tier]; T = a.seconds or tier["seconds"]
    cfg = load_cfg(); models = {}; tag = ("" if a.tier == "standard" else f"_{a.tier}") + ("" if a.opponent == "tracked" else "_puppet") + ("_nohits" if a.no_pushes else "")
    if a.from_raw:
        st = json.load(open(out / f"raw_{a.split}{tag}.json")); raw = st["raw"]; summ = {n: summarise(r, st["meta"]["seconds"]) for n, r in raw.items()}; pairs(raw, summ)
        json.dump(dict(meta=st["meta"], summary=summ), open(out / f"summary_{a.split}{tag}.json", "w"), indent=1); return
    if a.opponent == "puppet": cfg["opponent_mode"] = "puppet"
    clips = TEST_CLIPS if a.split == "test" else VAL_CLIPS
    ctrls = [PDStand()]
    for lp in a.policy:
        lab, path = lp.split("=", 1)
        if Path(path).exists(): ctrls.append(load_controller(f"RL {lab}={path}"))
    ccfg = {c.name: {**controller_cfg(c), "opponent_mode": cfg["opponent_mode"]} for c in ctrls}   # own contract per controller
    poses = dict(REF_POSES)
    for n, cc in ccfg.items():   # a pose outside REF_POSES changes every controller's episodes: allowed, but loudly
        h = np.array(cc.get("home", HOME), float)
        if not any(np.allclose(h, p, atol=1e-3) for p in poses.values()):
            print(f"WARNING: {n}'s default pose is not in REF_POSES; adding it changes the episodes, so this run is not paired with stored results", flush=True)
            poses[n] = h
    good = clean_starts({**cfg}, clips, tier["distances"], poses); eps = episodes(clips, a.reps, a.tier, good, not a.no_pushes, T)
    raw, summ = {}, {}
    for c in ctrls:
        cc = ccfg[c.name]; key = str(getattr(c, "cfg_path", "default"))
        if key not in models: models[key] = build_spec(opponent=True, cfg=cc).compile()
        t0 = time.time(); raw[c.name] = run(c, eps, models[key], cc, T, dr=tier["dr"]); summ[c.name] = summarise(raw[c.name], T)
        print(c.name, json.dumps(summ[c.name]), f"({time.time() - t0:.0f} s)", flush=True)
    pairs(raw, summ)
    skipped = {f"{c}@{d}": round(1 - float(np.mean(g)), 3) for (c, d), g in good.items()}
    meta = dict(tier=a.tier, tier_cfg=tier, opponent=a.opponent, split=a.split, clips=clips, reps=a.reps, pushes=not a.no_pushes, seconds=T,
                opponent_mode=cfg.get("opponent_mode", "puppet"), ring_m=RING_M, start_poses=list(poses), start_times_skipped=skipped)
    json.dump(dict(meta=meta, summary=summ), open(out / f"summary_{a.split}{tag}.json", "w"), indent=1)
    json.dump(dict(meta=meta, episodes=eps, raw=raw), open(out / f"raw_{a.split}{tag}.json", "w"))
    print(json.dumps(summ, indent=1))

if __name__ == "__main__": main()
