"""Hit-robustness eval: fall rate vs impulse x hit location, and recovery time, PD stand vs RL policy.
One hit per episode at t=1.0 s, 4 directions, strikes and pushes, observation delay sampled 0-40 ms, sensor noise on."""
import sys, json, time, argparse, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg
from r1gym.perturb import make_hit, LOCATIONS
from r1gym.policy import PDStand, TorchPolicy

IMPULSES = [2, 4, 6, 8, 10, 12, 15, 20]   # N*s  (R1 = 28.9 kg)
DIRS = {"front": 0.0, "left": np.pi / 2, "back": np.pi, "right": -np.pi / 2}   # where the hit comes FROM -> pushed away

def run(ctrl, scenarios, m, cfg, t_hit=1.0, T=4.0, threads=8):
    envs = []
    for k, (kind, loc, imp, dname, seed) in enumerate(scenarios):
        e = R1HitEnv(seed=seed, cfg=cfg, model=m, episode_s=T, max_impulse=0); rng = np.random.default_rng(seed)
        h = make_hit(kind, loc, imp, t_hit, DIRS[dname], rng)
        o = e.reset(hits=[h]); envs.append([e, o, False, None, None, h])   # env, obs, done, fell_t, recover_t, hit
    pool = ThreadPoolExecutor(threads); n = len(envs); steps = int(T / envs[0][0].dt); calm = np.zeros(n)
    for t in range(steps):
        A = ctrl.act(np.stack([x[1] for x in envs]))
        def st(i):
            x = envs[i]
            if x[2]: return
            o, r, d, info = x[0].step(A[i]); x[1] = o
            if info["fallen"]: x[2] = True; x[3] = x[0].t * x[0].dt; return
            e = x[0]; dd = e.d; R = dd.xmat[e.r.pel].reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))); w = np.linalg.norm(dd.qvel[e.r.free_v + 3:e.r.free_v + 6])
            tt = e.t * e.dt; hend = x[5].t0 + x[5].duration
            if tt > hend and x[4] is None:
                calm[i] = calm[i] + 1 if (tilt < 5 and w < 0.3) else 0
                if calm[i] >= 10: x[4] = max(0.0, tt - 10 * e.dt - hend)   # settled for 0.2 s
        list(pool.map(st, range(n)))
    return [dict(kind=s[0], loc=s[1], impulse=s[2], dir=s[3], fell=bool(x[2]), recovery_s=(None if x[2] else x[4])) for s, x in zip(scenarios, envs)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="results/policy/policy.pt"); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="results/eval"); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(); m = build_spec(cfg=cfg).compile()
    sc = [(k, l, i, dn, 1000 * r + j) for j, (k, l, i, dn, r) in enumerate(itertools.product(["strike", "push"], LOCATIONS, IMPULSES, DIRS, range(a.reps)))]
    res = {}
    for ctrl in [PDStand(), TorchPolicy(a.policy)]:
        t0 = time.time(); rows = run(ctrl, sc, m, cfg); print(ctrl.name, len(rows), "episodes in", round(time.time() - t0, 1), "s", flush=True)
        res[ctrl.name] = rows
    json.dump(res, open(out / "eval_raw.json", "w"))
    summary = {}
    for name, rows in res.items():
        S = {"fall_rate_vs_impulse": {}, "fall_rate_vs_location": {}, "fall_rate_vs_location_impulse": {}, "recovery_s_median_vs_impulse": {}}
        for i in IMPULSES:
            r = [x for x in rows if x["impulse"] == i]; S["fall_rate_vs_impulse"][i] = round(np.mean([x["fell"] for x in r]), 3)
            rc = [x["recovery_s"] for x in r if x["recovery_s"] is not None]; S["recovery_s_median_vs_impulse"][i] = round(float(np.median(rc)), 3) if rc else None
        for l in LOCATIONS:
            r = [x for x in rows if x["loc"] == l]; S["fall_rate_vs_location"][l] = round(np.mean([x["fell"] for x in r]), 3)
            S["fall_rate_vs_location_impulse"][l] = {i: round(np.mean([x["fell"] for x in r if x["impulse"] == i]), 3) for i in IMPULSES}
        f = [x["fell"] for x in rows]; S["overall_fall_rate"] = round(float(np.mean(f)), 3); S["episodes"] = len(rows)
        S["max_impulse_50pct_survival"] = max([i for i in IMPULSES if S["fall_rate_vs_impulse"][i] <= 0.5], default=0)
        summary[name] = S
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps({k: {kk: v[kk] for kk in ["overall_fall_rate", "fall_rate_vs_impulse", "max_impulse_50pct_survival"]} for k, v in summary.items()}, indent=1))
    plot(summary, out)

def plot(summary, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, S in summary.items():
        ax[0].plot(IMPULSES, [1 - S["fall_rate_vs_impulse"][i] for i in IMPULSES], "o-", label=name)
        ax[2].plot(IMPULSES, [S["recovery_s_median_vs_impulse"][i] or np.nan for i in IMPULSES], "o-", label=name)
    ax[0].set(xlabel="hit impulse [N·s]", ylabel="survival rate", title="R1 survival vs impulse (strikes+pushes, all locations)", ylim=(-0.02, 1.02)); ax[0].legend(); ax[0].grid(alpha=.3)
    locs = list(LOCATIONS); w = 0.38
    for k, (name, S) in enumerate(summary.items()):
        ax[1].bar(np.arange(len(locs)) + k * w, [1 - S["fall_rate_vs_location"][l] for l in locs], w, label=name)
    ax[1].set_xticks(np.arange(len(locs)) + w / 2, locs, rotation=30); ax[1].set(title="survival vs hit location (all impulses)", ylim=(0, 1)); ax[1].legend()
    ax[2].set(xlabel="hit impulse [N·s]", ylabel="median recovery time [s]", title="recovery time (survivors)"); ax[2].grid(alpha=.3); ax[2].legend()
    fig.tight_layout(); fig.savefig(out / "survival.png", dpi=120)

if __name__ == "__main__": main()
