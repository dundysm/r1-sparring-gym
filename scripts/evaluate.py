"""Hit-robustness eval: fall rate vs impulse x hit location, and recovery time, PD stand vs RL policy.
Single-hit sweep: one hit per episode at t=1.0 s, 4 directions, strikes and pushes, observation delay sampled 0-40 ms, sensor noise on.
Multi-hit sweep (v3): 3 hits 0.5-1.0 s apart (random kind/location/direction, each at the given impulse), nominal model and
domain-randomised model (mass/COM/friction/motor strength/armature/latency, same ranges as training)."""
import sys, json, time, argparse, itertools, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg
from r1gym.perturb import make_hit, LOCATIONS
from r1gym.policy import PDStand, load_controller, controller_cfg
from r1gym.stats import wilson, paired_diff

IMPULSES = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
MULTI = [4, 6, 8, 10, 12]   # N*s  (R1 = 28.9 kg)
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
    pool.shutdown(); gc.collect()
    return [dict(kind=s[0], loc=s[1], impulse=s[2], dir=s[3], fell=bool(x[2]), recovery_s=(None if x[2] else x[4])) for s, x in zip(scenarios, envs)]

def run_multi(ctrl, imp, n, m, cfg, dr=False, T=5.0, threads=8):
    return _multi(ctrl, imp, n, m, cfg, dr, T, threads, 0)

def _multi(ctrl, imp, n, m, cfg, dr, T, threads, k0):
    envs = []
    for k in range(k0, k0 + n):
        seed = 50000 + 100 * imp + k; rng = np.random.default_rng(seed); t = 1.0; hs = []
        for _ in range(3):
            hs.append(make_hit(rng.choice(["strike", "push"]), rng.choice(list(LOCATIONS)), imp, t, rng.uniform(-np.pi, np.pi), rng)); t += rng.uniform(0.5, 1.0)
        e = R1HitEnv(seed=seed, cfg=cfg, model=m, episode_s=T, max_impulse=0, dr=dr); envs.append([e, e.reset(hits=hs), False])   # DR envs share m -> stepped on 1 thread
    pool = ThreadPoolExecutor(1 if dr else threads)
    for _ in range(int(T / envs[0][0].dt)):
        A = ctrl.act(np.stack([x[1] for x in envs]))
        def st(i):
            x = envs[i]
            if x[2]: return
            o, r, d, info = x[0].step(A[i]); x[1] = o; x[2] = bool(info["fallen"])
        list(pool.map(st, range(len(envs))))
    out = round(float(np.mean([not x[2] for x in envs])), 3); pool.shutdown()
    if dr:   # restore the nominal model for later (nominal) evals
        from r1gym.env import _DR_NOMINAL; n0 = _DR_NOMINAL[id(m)]
        m.body_mass[:] = n0["mass"]; m.body_ipos[:] = n0["ipos"]; m.geom_friction[:] = n0["fric"]; m.actuator_gainprm[:] = n0["gain"]
        m.actuator_biasprm[:] = n0["bias"]; m.actuator_forcerange[:] = n0["frc"]; m.dof_armature[:] = n0["arm"]
    envs.clear(); gc.collect(); return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="+", default=["v1=results/policy_v1/policy.pt", "v2=results/policy/policy.pt", "v3=results/policy_v3_candidate/policy.pt"], help="label=path ...")
    ap.add_argument("--multi_n", type=int, default=96); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="results/eval"); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = {}
    sc = [(k, l, i, dn, 1000 * r + j) for j, (k, l, i, dn, r) in enumerate(itertools.product(["strike", "push"], LOCATIONS, IMPULSES, DIRS, range(a.reps)))]
    res = {}
    ctrls = [PDStand()]
    for lp in a.policy:
        lab, path = lp.split("=", 1)
        if Path(path).exists(): ctrls.append(load_controller(f"RL {lab}={path}"))
    multi = {}
    for ctrl in ctrls:
        cfg = controller_cfg(ctrl); key = str(getattr(ctrl, "cfg_path", "default"))   # own contract per controller
        if key not in models: models[key] = build_spec(cfg=cfg).compile()
        m = models[key]; t0 = time.time(); rows = run(ctrl, sc, m, cfg); print(ctrl.name, len(rows), "episodes in", round(time.time() - t0, 1), "s", flush=True)
        res[ctrl.name] = rows
        multi[ctrl.name] = {"nominal": {i: run_multi(ctrl, i, a.multi_n, m, cfg) for i in MULTI},
                            "domain_randomised": {i: run_multi(ctrl, i, a.multi_n, m, cfg, dr=True) for i in MULTI}}
        print(ctrl.name, "3-hit survival", multi[ctrl.name], flush=True)
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
        S["overall_fall_rate_ci95"] = wilson(int(sum(f)), len(f))
        S["fall_rate_vs_impulse_ci95"] = {i: wilson(int(sum(x["fell"] for x in rows if x["impulse"] == i)), sum(x["impulse"] == i for x in rows)) for i in IMPULSES}
        if name != "pd_stand":   # same scenarios for every controller -> paired difference
            d, ci = paired_diff(f, [x["fell"] for x in res["pd_stand"]]); S["fall_rate_minus_pd"] = d; S["fall_rate_minus_pd_ci95"] = ci
        S["max_impulse_50pct_survival"] = max([i for i in IMPULSES if S["fall_rate_vs_impulse"][i] <= 0.5], default=0)
        S["three_hit_survival"] = multi[name]; summary[name] = S
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps({k: {kk: v[kk] for kk in ["overall_fall_rate", "fall_rate_vs_impulse", "max_impulse_50pct_survival", "three_hit_survival"]} for k, v in summary.items()}, indent=1))
    plot(summary, out)

def plot(summary, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.5))
    for name, S in summary.items():
        ax[0].plot(IMPULSES, [1 - S["fall_rate_vs_impulse"][i] for i in IMPULSES], "o-", label=name)
        ax[2].plot(IMPULSES, [S["recovery_s_median_vs_impulse"][i] or np.nan for i in IMPULSES], "o-", label=name)
    ax[0].set(xlabel="hit impulse [N·s]", ylabel="survival rate", title="R1 survival vs impulse (strikes+pushes, all locations)", ylim=(-0.02, 1.02)); ax[0].legend(); ax[0].grid(alpha=.3)
    locs = list(LOCATIONS); w = 0.8 / len(summary)
    for name, S in summary.items():
        for key, ls in [("nominal", "o-"), ("domain_randomised", "s--")]:
            ax[3].plot(MULTI, [S["three_hit_survival"][key][i] for i in MULTI], ls, label=f"{name} ({'DR' if key[0] == 'd' else 'nominal'})")
    ax[3].set(xlabel="impulse of EACH of 3 hits [N·s]", ylabel="survival rate", title="3 hits 0.5-1 s apart", ylim=(-0.02, 1.02)); ax[3].legend(fontsize=7); ax[3].grid(alpha=.3)
    for k, (name, S) in enumerate(summary.items()):
        ax[1].bar(np.arange(len(locs)) + k * w, [1 - S["fall_rate_vs_location"][l] for l in locs], w, label=name)
    ax[1].set_xticks(np.arange(len(locs)) + w * (len(summary) - 1) / 2, locs, rotation=30); ax[1].set(title="survival vs hit location (all impulses)", ylim=(0, 1)); ax[1].legend()
    ax[2].set(xlabel="hit impulse [N·s]", ylabel="median recovery time [s]", title="recovery time (survivors)"); ax[2].grid(alpha=.3); ax[2].legend()
    fig.tight_layout(); fig.savefig(out / "survival.png", dpi=120)

if __name__ == "__main__": main()
