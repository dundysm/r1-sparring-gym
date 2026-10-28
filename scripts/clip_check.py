"""Clipping check -> results/clip_check.json: how far the two robots' visible shells pass into each other.

Each step, the bout's state is copied into a model where the shells (each link's visual mesh, as MuJoCo's convex hull)
collide robot-vs-robot, and the deepest shell-shell penetration is read from MuJoCo's collision check. The hull holds the
mesh, so this is an upper bound on how far the visible robots overlap. Run for robot_contact: capsule (Unitree's capsules
carry the contact, as before) and shell (the shells do), with both shipped stepping policies (default: the footwork ones,
each with its gym.json outer loops), on the demo and on 48 bouts on train + val clips (2 distances x 6 clean starts per clip,
12 s, the demo's scripted pushes).

--crosscheck OLD_RAW NEW_RAW: the clean-start filter also uses the shells, so a shell run starts its bouts at other clip
times than a capsule run. This runs the hard-tier validation bouts of two stored runs (OLD_RAW from capsule contact:
git show 97a22e2:results/eval_opponent/raw_val_hard.json; NEW_RAW the current one) under both contact models, so the
contact model and the start times can be told apart. Its default policies are PR #13's stage-2 bases, run with
hold-your-spot off as PR #13 ran them; rows merge per policy into crosscheck_val_hard."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec, SHELL_BIT
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start, demo_env, SCRIPTED

def overlap(ms, ds, qpos):
    """Deepest shell-shell penetration [m] for this state (0 if the shells don't touch)."""
    ds.qpos[:] = qpos; mujoco.mj_kinematics(ms, ds); mujoco.mj_collision(ms, ds); p = 0.0
    for j in range(ds.ncon):
        c = ds.contact[j]
        if ms.geom_contype[c.geom1] | ms.geom_contype[c.geom2] == SHELL_BIT: p = max(p, -c.dist)
    return p

def bout(c, cfg, m, ms, e, o, steps):
    ds = mujoco.MjData(ms); P = []
    for _ in range(steps):
        o, r, dn, info = e.step(c.act(o[None])[0])
        if info["fallen"]: return dict(fell=True, overlap=P)
        P.append(overlap(ms, ds, e.d.qpos))
    return dict(fell=False, overlap=P)

def summary(rows):
    mx = np.array([max(r["overlap"]) if r["overlap"] else 0.0 for r in rows]) * 100; allp = np.concatenate([r["overlap"] for r in rows if r["overlap"]]) * 100
    return dict(bouts=len(rows), falls=sum(r["fell"] for r in rows), max_overlap_cm_median=round(float(np.median(mx)), 2), max_overlap_cm_p95=round(float(np.percentile(mx, 95)), 2),
                max_overlap_cm_worst=round(float(mx.max()), 2), steps_over_1cm=round(float(np.mean(allp > 1)), 4), steps_over_3cm=round(float(np.mean(allp > 3)), 4))

def crosscheck(policies, old_raw, new_raw):
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import eval_opponent as eo
    out = {}
    for spec in policies:
        lab, path = spec.split("=", 1); c = load_controller(f"{lab}={path}"); base = {**controller_cfg(c), "opponent_mode": "policy", "hold_position": False}   # PR #13 ran the stage-2 bases without the loop
        for mode in ("capsule", "shell"):
            cfg = {**base, "robot_contact": mode}; m = build_spec(opponent=True, cfg=cfg).compile()
            for starts, f in (("capsule_starts", old_raw), ("shell_starts", new_raw)):
                eps = json.load(open(f))["episodes"]; rows = eo.run(c, eps, m, cfg, 20.0, dr=True)
                out.setdefault(lab, {})[f"{mode}_contact_{starts}"] = dict(survival=round(float(np.mean([r["fell_s"] is None for r in rows])), 3),
                                                                          opponent_fell=round(float(np.mean([r["opp_fell"] for r in rows])), 3))
                print(lab, mode, starts, out[lab][f"{mode}_contact_{starts}"], flush=True)
    return out

FW = ["footwork_arms_down=results/policy_mjlab_footwork/policy.onnx", "footwork_guard=results/policy_mjlab_guard_footwork/policy.onnx"]   # shipped
S2 = ["arms_down=results/policy_mjlab/policy.onnx", "guard=results/policy_mjlab_guard/policy.onnx"]   # the stage-2 bases PR #13 crosschecked

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="+", help="label=path; default the footwork policies, with --crosscheck the stage-2 bases")
    ap.add_argument("--out", default="results/clip_check.json"); ap.add_argument("--crosscheck", nargs=2, metavar=("OLD_RAW", "NEW_RAW")); a = ap.parse_args()
    if a.crosscheck:   # merged into the existing file, per policy
        res = json.load(open(a.out)); res["about"] = __doc__; res.setdefault("crosscheck_val_hard", {}).update(crosscheck(a.policy or S2, *a.crosscheck))
        json.dump(res, open(a.out, "w"), indent=1); sys.exit(0)
    res = {"about": __doc__}
    if Path(a.out).exists() and "crosscheck_val_hard" in json.load(open(a.out)): res["crosscheck_val_hard"] = json.load(open(a.out))["crosscheck_val_hard"]   # kept (its own --crosscheck run)
    for spec in a.policy or FW:
        lab, path = spec.split("=", 1); c = load_controller(f"{lab}={path}"); base = {**controller_cfg(c), "opponent_mode": "policy"}
        ms = build_spec(opponent=True, cfg={**base, "robot_contact": "shell"}).compile()   # the measuring model
        for mode in ("capsule", "shell"):
            cfg = {**base, "robot_contact": mode}; m = build_spec(opponent=True, cfg=cfg).compile()
            e, o = demo_env(m, cfg); demo = bout(c, cfg, m, ms, e, o, 600)
            clips, dists = TRAIN_CLIPS + VAL_CLIPS, [0.55, 0.65]; good = clean_starts(cfg, clips, dists)
            eps = [(cl, d, pick_start(good[(cl, d)], u)) for cl in clips for d in dists for u in np.linspace(0, 5.9, 6)]
            def run(ep):
                cl, d, t0 = ep; e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=13, max_impulse=0, train=False)
                e.opp = FightReplay(m, cl, distance=d, cfg=cfg); e.opp_t0 = t0; rng = np.random.default_rng(3)
                o = e.reset(hits=[make_hit(k, l, i, t, y, rng) for k, l, i, t, y in SCRIPTED], delay=1); return bout(c, cfg, m, ms, e, o, 600)
            rows = list(ThreadPoolExecutor(8).map(run, eps))
            res.setdefault(lab, {})[mode] = dict(demo=summary([demo]), bouts=summary(rows)); print(lab, mode, json.dumps(res[lab][mode]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
