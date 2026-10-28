"""Turning candidates (gpu/r1_spar_task.py stage 4) -> results/turn_selection.json, by the rule fixed in
results/turn_prereg.json before they were trained. Criterion 1 (CPU MuJoCo turn response, scripts/face_ab.py turn_response)
is checked first; criteria 2-3 (hard / standard validation, with the turn setting chosen by the face_ab rule) run only for a
candidate that passes it, since all three must hold. Test is never used here. The prereg also said test would be reported
once for both candidates after the decision; that is only possible for a candidate with a turn setting chosen on validation,
so when criterion 1 fails it is not run, and the output records the deviation."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1gym.policy import load_controller, controller_cfg
from r1gym.model import build_spec
import face_ab as fa

CANDIDATES = {"turn_arms": "results/policy_mjlab_turn/policy.onnx", "turn_guard": "results/policy_mjlab_guard_turn/policy.onnx"}

def within_one_bout(x, ref, n):
    """x and ref are fractions of n bouts stored to 3 dp (face_ab.val_survival, turn_prereg.json): compare whole bouts."""
    return round(x * n) >= round(ref * n) - 1

def choose(val):
    """face_ab's rule on hard validation: most up-and-within-1 m among the settings within one bout of the best survival."""
    h = val["hard"]; best = max(x["survival"] for x in h.values())
    ok = {k: x for k, x in h.items() if within_one_bout(x["survival"], best, 64)}
    return max(ok, key=lambda k: (ok[k]["in_ring"], ok[k]["survival"]))

if __name__ == "__main__":
    pre = json.load(open("results/turn_prereg.json")); ref = pre["shipped_reference_validation"]; out = {"about": __doc__, "prereg": "results/turn_prereg.json", "candidates": {}}
    for lab, path in CANDIDATES.items():
        pose = pre["candidates"][lab]["replaces"]; s = ref[pose]
        c = load_controller(f"{lab}={path}"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
        tr = fa.turn_response(c, cfg); mn = min(abs(tr["0.8"]), abs(tr["-0.8"]))
        r = dict(policy=path, replaces=s["policy"], turn_response_cmd_to_measured_rad_s=tr, turn_min_abs_0p8=mn,
                 criterion_1=dict(need=f">= 0.25 and >= 1.5 x {s['turn_min_abs_0p8']}", passed=bool(mn >= 0.25 and mn >= 1.5 * s["turn_min_abs_0p8"])))
        if r["criterion_1"]["passed"]:
            val = fa.val_survival(c, cfg, build_spec(opponent=True, cfg=cfg).compile()); k = choose(val); h, st = val["hard"][k], val["standard"][k]
            r.update(val_survival=val, turn_setting=k,
                     criterion_2=dict(passed=bool(within_one_bout(h["survival"], s["hard_val_survival"], 64) and within_one_bout(h["in_ring"], s["hard_val_in_ring"], 64))),
                     criterion_3=dict(passed=bool(within_one_bout(st["survival"], s["standard_val_survival"], 80))))
            r["ships"] = r["criterion_2"]["passed"] and r["criterion_3"]["passed"]
            r["test"] = "to run once now, at turn_setting (scripts/eval_opponent.py --split test), and report"
        else:
            r.update(criterion_2="not run: criterion 1 failed", criterion_3="not run: criterion 1 failed", ships=False,
                     test="not run: criterion 1 failed, so no turn setting was chosen on validation to test with (a deviation from turn_prereg.json's test report)")
        out["candidates"][lab] = r; print(lab, json.dumps({k: v for k, v in r.items() if k != "val_survival"}), flush=True)
    json.dump(out, open("results/turn_selection.json", "w"), indent=1)
