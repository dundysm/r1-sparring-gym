"""M2 gates on train + val (results/opponent_tracker_aim/prereg.json) -> results/opponent_tracker_aim/gates.json (+ aim_check_raw.json).

  alone      scripts/eval_tracker.py's 48 runs for the M2 tracker (eval_tracker.json, run first) against the shipped
             results/opponent_tracker/eval_tracker.json: stays up >= 45/48, mean per-clip joint RMSE <= shipped + 0.02 rad.
  parity     the pod's gpu/parity_mjlab.py fall times (parity_mjlab_side.json) against eval_tracker.json, same 48 starts: every
             fall both have happens within 0.1 s.
  tolerance  scripts/tracker_tolerance.py --split val --skip_long (tolerance_val.json): 'any20' within 5 cm >= 0.65, and the
             20 cm / 1 m/s lunge adds >= 12 cm median reach.
  aim_check  scripts/aim_check.py's grid (train + val clips, 0.40-0.65 m, 3 clean starts, 10 s, no scripted hits) on the
             arms-down and guard targets (the stepping bases aim_check used, with their gym.json as in this tree). The M2
             operator is picked on the M2 tracker alone by aim_check's rule 3 (largest paired delta, aimed minus shadow, arms
             down, CI above 0; falls <= shadow + 2 pp) among standoff+armwarp_vz and standoff+armwarp. Then, same episodes:
             M2 + its operator minus the shipped tracker + M1's operator (standoff+armwarp_vz), re-run here on this tree:
             head clean strikes/min and head-landed share per window, both CIs above 0 on both targets, and opponent falls at
             most the shipped tracker's (with M1's operator) + 2 pp. The shipped runs are also checked against the stored
             results/opponent_aim/aim_check_raw.json (same tree state as M1 or not: reported, not assumed)."""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec, load_cfg
from r1gym.env import TRAIN_CLIPS, VAL_CLIPS
from r1gym.policy import load_controller, controller_cfg
import aim_check as AC

D = Path("results/opponent_tracker_aim"); SHIPPED = "results/opponent_tracker/policy.onnx"; M1_OP = "standoff+armwarp_vz"; M2_CANDIDATES = ["standoff+armwarp_vz", "standoff+armwarp"]

def gate_alone():
    m2, sh = json.load(open(D / "eval_tracker.json")), json.load(open("results/opponent_tracker/eval_tracker.json"))
    up = lambda d: sum(sum(x is None for x in v["fall_times"]) for k, v in d.items() if k.startswith("kfab"))
    n = lambda d: sum(len(v["fall_times"]) for k, v in d.items() if k.startswith("kfab"))
    rm = lambda d: float(np.mean([v["joint_rmse_rad"] for k, v in d.items() if k.startswith("kfab")]))
    return dict(stays_up=f"{up(m2)}/{n(m2)}", shipped=f"{up(sh)}/{n(sh)}", joint_rmse_rad=round(rm(m2), 4), shipped_joint_rmse_rad=round(rm(sh), 4),
                passes=up(m2) >= 45 and rm(m2) <= rm(sh) + 0.02)

def gate_parity():
    mj, ours = json.load(open(D / "parity_mjlab_side.json")), json.load(open(D / "eval_tracker.json")); shared, gaps = 0, []
    for clip, xs in mj.items():
        for a, b in zip(xs, ours[clip]["fall_times"]):
            if a is not None and b is not None: shared += 1; gaps.append(abs(a - b))
    up = lambda d: sum(x is None for k, v in d.items() for x in (v if isinstance(v, list) else v["fall_times"]) if k.startswith("kfab"))
    return dict(mjlab_stays_up=up(mj), this_repo_stays_up=up({k: v for k, v in ours.items() if k.startswith("kfab")}), shared_falls=shared,
                max_gap_s=round(max(gaps), 3) if gaps else None, passes=all(g <= 0.1 for g in gaps))

def gate_tolerance():
    t = json.load(open(D / "tolerance_val.json")); s = json.load(open(D / "tolerance_val_shipped.json"))
    a20, l = t["arm_warps"]["any20"]["within_5cm"], t["lunges"]["l20_v10"]["extra_reach_cm_median"]
    return dict(any20_within_5cm=a20, shipped_any20_within_5cm=s["arm_warps"]["any20"]["within_5cm"], n=t["arm_warps"]["any20"]["n"],
                lunge_20cm_1mps_extra_reach_cm=l, shipped_lunge_cm=s["lunges"]["l20_v10"]["extra_reach_cm_median"], falls=t["arm_warps"]["any20"]["fell"],
                passes=a20 is not None and a20 >= 0.65 and l >= 12.0)

def aim_runs(threads):
    base = {**load_cfg(), "opponent_mode": "policy"}; eps = AC.grid(base)
    stored = json.load(open("results/opponent_aim/aim_check_raw.json")); assert [dict(clip=e["clip"], t0=e["t0"], distance=e["distance"]) for e in stored["meta"]["episodes"]] == eps, "grid changed"
    pool = ThreadPoolExecutor(threads); raw, t0 = {}, time.time()
    for tracker, onnx, modes in (("shipped", SHIPPED, ["shadow", M1_OP]), ("m2", str(D / "policy.onnx"), ["shadow"] + M2_CANDIDATES)):
        for name, path in (("arms_down", "results/policy_mjlab/policy.onnx"), ("guard", "results/policy_mjlab_guard/policy.onnx")):
            c = load_controller(f"{name}={path}"); cc = {**controller_cfg(c), "opponent_mode": "policy", "opponent_policy": onnx}
            tg = AC.Target(name, c, cc, build_spec(opponent=True, cfg=cc).compile())
            for k in [(cl, d) for cl in TRAIN_CLIPS + VAL_CLIPS for d in AC.DISTANCES]: tg.replay(*k).strikes
            for mode in modes:
                raw.setdefault(tracker, {}).setdefault(name, {})[mode] = list(pool.map(lambda ep: AC.bout(tg, ep, mode), eps)); print(tracker, name, mode, f"({time.time() - t0:.0f} s)", flush=True)
    same = {f"{n}/{md}": raw["shipped"][n][md] == stored["raw"][n][md] for n in ("arms_down", "guard") for md in ("shadow", M1_OP)}
    return raw, eps, same

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
    import eval_opponent as eo
    raw, eps, same = aim_runs(a.threads); R = raw["m2"]; S = raw["shipped"]
    rule3 = {}
    for md in M2_CANDIDATES:
        p = AC.paired(R["arms_down"][md], R["arms_down"]["shadow"])["head_clean_pm"]; f = float(np.mean([r["opp_fell"] for t in ("arms_down", "guard") for r in R[t][md]]))
        f0 = float(np.mean([r["opp_fell"] for t in ("arms_down", "guard") for r in R[t]["shadow"]]))
        rule3[md] = dict(delta=p["diff"], ci95=p["ci95"], opp_fell=round(f, 4), opp_fell_shadow=round(f0, 4), eligible=p["ci95"][0] > 0 and f <= f0 + 0.02)
    ok = sorted([md for md in M2_CANDIDATES if rule3[md]["eligible"]], key=lambda md: -rule3[md]["delta"]); op = ok[0] if ok else M1_OP
    comp = {}
    for t in ("arms_down", "guard"):
        p = AC.paired(R[t][op], S[t][M1_OP]); fm, fs = float(np.mean([r["opp_fell"] for r in R[t][op]])), float(np.mean([r["opp_fell"] for r in S[t][M1_OP]]))
        comp[t] = dict(head_clean_pm=p["head_clean_pm"], window_head_share=p["window_head_share"], head_landed_pm=p["head_landed_pm"], opp_fell=round(fm, 4), shipped_opp_fell=round(fs, 4),
                       shipped_unaimed_opp_fell=round(float(np.mean([r["opp_fell"] for r in S[t]["shadow"]])), 4), rules={k: v["head"] for k, v in p["rules"].items()},
                       m2=AC.block(R[t][op]), shipped=AC.block(S[t][M1_OP]),
                       passes=p["head_clean_pm"]["ci95"][0] > 0 and p["window_head_share"]["ci95"][0] > 0 and fm <= fs + 0.02)
    gates = dict(alone=gate_alone(), parity=gate_parity(), tolerance=gate_tolerance(),
                 aim_check=dict(operator=op, operator_rule3=rule3, vs_shipped_m1=comp, passes=all(c["passes"] for c in comp.values()), shipped_reproduces_stored=same))
    res = dict(about=__doc__, prereg="results/opponent_tracker_aim/prereg.json", meta=dict(eo.provenance(build_spec(opponent=True).compile(), load_cfg()), threads=a.threads, episodes=len(eps)),
               gates=gates, all_pass=all(g["passes"] for g in gates.values()))
    json.dump(res, open(D / "gates.json", "w"), indent=1); json.dump(raw, open(D / "aim_check_raw.json", "w"))
    print(json.dumps({k: (v if k != "aim_check" else {kk: vv for kk, vv in v.items() if kk != "vs_shipped_m1"} | {t: {x: c[x] for x in ("head_clean_pm", "window_head_share", "opp_fell", "shipped_opp_fell", "passes")} for t, c in v["vs_shipped_m1"].items()}) for k, v in gates.items()}, indent=1, default=str))
    print("ALL PASS" if res["all_pass"] else "NOT ALL PASS")

if __name__ == "__main__": main()
