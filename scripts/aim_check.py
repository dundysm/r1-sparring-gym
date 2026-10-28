"""Paired aim check on train + val (M1.4) -> results/opponent_aim/aim_check.json (+ aim_check_raw.json), and the M1 operator
chosen by the preregistered rule -> results/opponent_tracker/aim.json (docs/AIMED_OPPONENT.md: decision rules, amendment 1).
Test is never run here.

Grid: TRAIN_CLIPS + VAL_CLIPS x 0.40-0.65 m x 3 start times (clean at every distance in both reference poses) x 10 s; obs
delay 1 step, no scripted hits, nominal physics. Every operator runs the identical episodes; the unaimed side is 'shadow'
(the scheduler logs its intents but leaves the reference alone: bit-identical to the shipped opponent).
Targets: mannequin (the R1 welded at the pelvis, PD stand at home: the aim alone), arms-down and guard (the shipped stepping
bases with their gym.json). Shell solref 0.02 is re-run on arms-down: peak forces depend on it, impulses shouldn't.
Per bout: the HitMeter (rates, every sensitivity rule, outcome of each intent window), pelvis spacing, R1 and opponent falls;
per intent: outcome, closest palm-to-target distance, full-target IK residual at the apex, arm tracking error, gating.
Clusters for every bootstrap: (clip, start); each runs at six distances under every operator."""
import sys, json, time, argparse, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec, load_cfg, SHELL_BIT
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.opponent import FightReplay
from r1gym.hits import HitMeter, landed_block, sensitivity_block, RULES
from r1gym.policy import PDStand, load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start
from r1gym.stats import cluster_paired_diff, wilson

CANDIDATES = ["standoff", "armwarp", "armwarp_vz", "standoff+armwarp_vz", "select"]   # rule 4's order of simplicity
DISTANCES, SECONDS, STARTS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65], 10.0, (1.0, 3.0, 5.0)
BUCKETS = [(0.0, 0.03, "<3cm"), (0.03, 0.08, "3-8cm"), (0.08, 9.0, ">=8cm")]

def grid(cfg):
    clips = TRAIN_CLIPS + VAL_CLIPS; good = clean_starts(cfg, clips, DISTANCES)
    out = []
    for c in clips:
        g = np.logical_and.reduce([good[(c, d)] for d in DISTANCES])   # one start time serves every distance: a cluster
        for u in STARTS: t0 = pick_start(g, u); out += [dict(clip=c, t0=t0, distance=d) for d in DISTANCES]
    return out

class Target:
    def __init__(self, name, ctrl, cfg, m): self.name, self.ctrl, self.cfg, self.m, self.reps = name, ctrl, cfg, m, {}
    def replay(self, clip, dist):
        k = (clip, dist)
        if k not in self.reps: self.reps[k] = FightReplay(self.m, clip, distance=dist, cfg=self.cfg)
        return self.reps[k]

def bout(tg, ep, mode):
    cfg = {**tg.cfg, "opponent_aim": mode}; e = R1HitEnv(seed=0, cfg=cfg, model=tg.m, episode_s=SECONDS + 1, max_impulse=0, train=False)
    e.opp = tg.replay(ep["clip"], ep["distance"]); e.opp_t0 = ep["t0"]; hm = HitMeter(e.m, cfg["hit_metric"]); e.substep_hooks.append(hm)
    o = e.reset(hits=[], delay=1); fell, ofell, sp = None, False, []; q = e.r.free_q
    for k in range(int(SECONDS / e.dt)):
        o, r, dn, info = e.step(tg.ctrl.act(o[None])[0]); d = e.d
        sp.append(float(np.linalg.norm(d.qpos[e.opp.qadr:e.opp.qadr + 2] - d.qpos[q:q + 2])))   # actual pelvis spacing
        if d.xmat[e.opp.pel][8] < 0.5 or d.xpos[e.opp.pel][2] < 0.25: ofell = True
        if info["fallen"]: fell = round((k + 1) * e.dt, 2); break
    aim = e.opp_state["aim"]; secs = fell or SECONDS; its = [it for it in aim.intents if it.t_apex + aim.p["window_tail_s"] <= secs]   # whole windows only
    wins = [it.window(aim.p["window_tail_s"]) for it in its]; outc = hm.windows(wins)
    I = [dict(limb=it.limb, gated=it.gated, outcome=o, closest_cm=round(100 * it.closest_m, 1), ik_res_cm=None if it.ik_res_m is None else round(100 * it.ik_res_m, 1),
              arm_rmse=it.as_dict()["arm_rmse"], reach_m=it.reach_m) for it, o in zip(its, outc)]
    return dict(clip=ep["clip"], t0=ep["t0"], distance=ep["distance"], mode=mode, fell_s=fell, opp_fell=ofell, splices=aim.splices, hit_meter=hm.summary(secs, sp, wins), intents=I)

def block(rows):
    """One (target, mode): falls, the landed and sensitivity blocks (exposure-weighted, cluster CIs), window outcomes, aim accuracy."""
    cl = [(r["clip"], r["t0"]) for r in rows]; hm = [r["hit_meter"] for r in rows]; I = [i for r in rows for i in r["intents"]]
    W = {k: sum(i["outcome"] == k for i in I) for k in ("head", "torso", "blocked", "miss")}; n = len(I); A = [i for i in I if not i["gated"]]
    med = lambda x: round(float(np.median(x)), 2) if len(x) else None
    by_ik = {lab: dict(n=len(s), head_share=round(sum(i["outcome"] == "head" for i in s) / len(s), 3) if s else None)
             for lo, hi, lab in BUCKETS for s in [[i for i in I if i["ik_res_cm"] is not None and lo <= i["ik_res_cm"] / 100 < hi]]}
    return dict(bouts=len(rows), r1_fell=sum(r["fell_s"] is not None for r in rows), opp_fell=sum(r["opp_fell"] for r in rows), opp_fell_rate=round(float(np.mean([r["opp_fell"] for r in rows])), 3),
                landed=landed_block(hm, cl, None), sensitivity=sensitivity_block(hm, cl),
                windows=dict(n=n, **W, head_share=round(W["head"] / n, 3) if n else None, gated_share=round((n - len(A)) / n, 3) if n else None,
                             head_share_aimed=round(sum(i["outcome"] == "head" for i in A) / len(A), 3) if A else None),
                closest_cm_median=med([i["closest_cm"] for i in I]), closest_cm_median_aimed=med([i["closest_cm"] for i in A]), ik_res_cm_median=med([i["ik_res_cm"] for i in I if i["ik_res_cm"] is not None]),
                by_ik_residual=by_ik, arm_rmse_median=med([i["arm_rmse"] for i in I if i["arm_rmse"] is not None]), splices_per_bout=med([r["splices"] for r in rows]))

def paired(a, b):
    """a minus b on identical episodes (same order): primary and every rule (exposure-weighted, cluster bootstrap), window head
    share (weighted by windows), closest distance, falls."""
    cl = [(r["clip"], r["t0"]) for r in a]; wa = [r["hit_meter"]["secs"] / 60 for r in a]; wb = [r["hit_meter"]["secs"] / 60 for r in b]
    diff = lambda f, ua=wa, ub=wb: dict(zip(("diff", "ci95"), cluster_paired_diff([f(r) for r in a], [f(r) for r in b], cl, wa=ua, wb=ub)))
    hs = lambda r: sum(i["outcome"] == "head" for i in r["intents"]) / max(1, len(r["intents"])); nw = lambda rs: [max(1, len(r["intents"])) for r in rs]
    cm = lambda r: float(np.mean([i["closest_cm"] for i in r["intents"]])) if r["intents"] else 0.0
    return dict(head_clean_pm=diff(lambda r: r["hit_meter"]["head_clean_pm"]), head_landed_pm=diff(lambda r: r["hit_meter"]["head_landed_pm"]),
                head_impulse_pm=diff(lambda r: r["hit_meter"]["head_impulse_pm"]), torso_clean_pm=diff(lambda r: r["hit_meter"]["torso_clean_pm"]),
                rules={k: {z: diff(lambda r, k=k, i=i: r["hit_meter"]["rules"][k][i]) for i, z in enumerate(("head", "torso"))} for k in RULES},
                window_head_share=diff(hs, nw(a), nw(b)), closest_cm=diff(cm, None, None),
                opp_fell_pp=dict(zip(("diff", "ci95"), cluster_paired_diff([float(r["opp_fell"]) for r in a], [float(r["opp_fell"]) for r in b], cl))),
                r1_fell_pp=dict(zip(("diff", "ci95"), cluster_paired_diff([float(r["fell_s"] is not None) for r in a], [float(r["fell_s"] is not None) for r in b], cl))),
                head_clean_pm_by_distance={f"{d:.2f}": round(float(np.sum([r["hit_meter"]["n"]["head_clean"] for r in a if r["distance"] == d]) / np.sum([r["hit_meter"]["secs"] / 60 for r in a if r["distance"] == d])
                                                          - np.sum([r["hit_meter"]["n"]["head_clean"] for r in b if r["distance"] == d]) / np.sum([r["hit_meter"]["secs"] / 60 for r in b if r["distance"] == d])), 2) for d in DISTANCES})

def choose(summ, raw, tol):
    """docs/AIMED_OPPONENT.md decision rules 1-7 (amendment 1)."""
    cand = [c for c in CANDIDATES if c != "select" or tol["long_runs"]["select"]["passes"]]
    pooled = lambda mode: float(np.mean([r["opp_fell"] for t in ("mannequin", "arms_down", "guard") for r in raw[t][mode]]))
    base = pooled("shadow"); rep = {}
    for c in CANDIDATES:
        if c not in cand: rep[c] = dict(considered=False, why="splice tolerance failed (tolerance.json long_runs.select)"); continue
        p = summ["arms_down"]["paired"][c]["head_clean_pm"]; f = pooled(c)
        rep[c] = dict(considered=c in cand, opp_fell_pooled=round(f, 4), opp_fell_unaimed=round(base, 4), rejected_falls=f > base + 0.02, delta=p["diff"], ci95=p["ci95"],
                      ci_above_0=p["ci95"][0] > 0, window_head_share_delta=summ["arms_down"]["paired"][c]["window_head_share"]["diff"])
    ok = sorted([c for c in cand if not rep[c]["rejected_falls"] and rep[c]["ci_above_0"]], key=lambda c: -rep[c]["delta"])
    if not ok:
        pick, why = "standoff", ("no candidate's paired head clean strikes/min (arms-down, aimed minus unaimed) has a 95% CI above 0 "
                                 "(rule 5): M1 ships StandOff as distance management, not as aiming, and M2 goes ahead")
    elif len(ok) == 1 or rep[ok[0]]["delta"] - rep[ok[1]]["delta"] >= 0.5: pick, why = ok[0], f"largest paired delta with its CI above 0 (rule 3), {rep[ok[0]]['delta']}/min {rep[ok[0]]['ci95']}"
    else:
        top = ok[:2]; hs = {c: rep[c]["window_head_share_delta"] for c in top}
        pick = max(top, key=lambda c: (hs[c], -CANDIDATES.index(c))); why = f"top two within 0.5/min ({top}); higher head-landed share per window, then the simpler (rule 4): {hs}"
    flips = {k: summ["arms_down"]["paired"][pick]["rules"][k]["head"]["diff"] for k in RULES}
    sign = np.sign(rep[pick]["delta"]); flip = [k for k, v in flips.items() if np.sign(v) != sign and v != 0]
    return pick, why, rep, flips, flip

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default="results/opponent_aim"); a = ap.parse_args()
    import eval_opponent as eo
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tol = json.load(open("results/opponent_tracker/tolerance.json")); modes = ["shadow"] + [c for c in CANDIDATES if c != "select" or tol["long_runs"]["select"]["passes"]]
    base = {**load_cfg(), "opponent_mode": "policy"}; tg = {}
    tg["mannequin"] = Target("mannequin", PDStand(), base, build_spec(opponent=True, cfg=base, mannequin=True).compile())
    for name, path in (("arms_down", "results/policy_mjlab/policy.onnx"), ("guard", "results/policy_mjlab_guard/policy.onnx")):
        c = load_controller(f"{name}={path}"); cc = {**controller_cfg(c), "opponent_mode": "policy"}; tg[name] = Target(name, c, cc, build_spec(opponent=True, cfg=cc).compile())
    m2 = copy.deepcopy(tg["arms_down"].m); sh = (m2.geom_contype | m2.geom_conaffinity) & SHELL_BIT > 0; m2.geom_solref[sh] = [0.02, 1.0]
    tg["arms_down_solref0.02"] = Target("arms_down_solref0.02", tg["arms_down"].ctrl, tg["arms_down"].cfg, m2)
    eps = grid(base); pool = ThreadPoolExecutor(a.threads); raw = {}
    for name, t in tg.items():
        for k in [(c, d) for c in TRAIN_CLIPS + VAL_CLIPS for d in DISTANCES]: t.replay(*k).strikes   # built before the threads share them
        raw[name] = {}
        for mode in modes:
            raw[name][mode] = list(pool.map(lambda ep: bout(t, ep, mode), eps)); print(name, mode, f"({time.time() - t0:.0f} s)", flush=True)
    summ = {}
    for name in tg:
        summ[name] = dict(by_mode={md: block(raw[name][md]) for md in modes}, paired={md: paired(raw[name][md], raw[name]["shadow"]) for md in modes if md != "shadow"},
                          shell_solref=eo.shell_solref(tg[name].m))
    pick, why, rep, flips, flip = choose(summ, raw, tol)
    sens = f"; under the other hit rules its head delta is {flips}" + (f": the sign flips under {flip}" if flip else " (same sign under every rule)")
    meta = dict(eo.provenance(tg["arms_down"].m, tg["arms_down"].cfg), threads=a.threads, wall_s=round(time.time() - t0, 1), grid=dict(clips=TRAIN_CLIPS + VAL_CLIPS, distances=DISTANCES, starts=STARTS, seconds=SECONDS, bouts_per_cell=len(eps)),
                episodes=eps, modes=modes, select_tolerance=tol["long_runs"]["select"])
    res = dict(about=__doc__, meta=meta, decision=dict(operator=pick, why=why + sens, candidates=rep, rule_head_deltas=flips, sign_flips_under=flip), summary=summ)
    json.dump(res, open(out / "aim_check.json", "w"), indent=1); json.dump(dict(meta=meta, raw=raw), open(out / "aim_check_raw.json", "w"))
    lag = tol["lag"]["train_median_s"]
    json.dump(dict(operator=pick, _why=why + sens, lag_s=lag, _why_lag="median reference-to-tracked apex delay over train hand strikes, opponent alone (tolerance.json); shifts library windows",
                   decided_on="train + val (results/opponent_aim/aim_check.json), before any aimed test run", candidates=rep,
                   meta={k: meta[k] for k in ("git_hash", "git_dirty", "mujoco", "config_hash", "prereg_sha", "shell_solref", "hit_metric")}),
              open("results/opponent_tracker/aim.json", "w"), indent=1)
    print(json.dumps(res["decision"], indent=1))

if __name__ == "__main__": main()
