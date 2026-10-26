"""Headless check of the demo scenario (no rendering): fall time per controller for the demo run, plus N variants
(hit durations re-sampled, obs delay 0-2, opponent distance 0.50-0.65 m) so the demo isn't a cherry-pick.
Also reports the opponent's real contacts on the R1 (count, peak force) so the demo's falls can be attributed."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r1gym.model import build_spec, load_cfg
from r1gym.scenario import demo_env, variant_kwargs, DEMO_SECONDS
from r1gym.contacts import per_body_force
from r1gym.policy import PDStand, load_controller, controller_cfg

def run(ctrl, m, cfg, **kw):
    e, o = demo_env(m, cfg, **kw); peak = 0.0; touching = 0; events = 0; prev = False
    for _ in range(int(DEMO_SECONDS / e.dt)):
        o, r, d, info = e.step(ctrl.act(o[None])[0])
        f = sum(per_body_force(m, e.d, e.r.r1_bodies).values()); now = f > 1.0
        touching += now; events += now and not prev; prev = now; peak = max(peak, f)
        if info["fallen"]: return dict(fell_s=round(e.t * e.dt, 2), contact_events=events, contact_steps=touching, peak_contact_N=round(peak))
    return dict(fell_s=None, contact_events=events, contact_steps=touching, peak_contact_N=round(peak))   # survived

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="+", default=["v2=results/policy/policy.pt"], help="label=path ...")
    ap.add_argument("--variants", type=int, default=8); ap.add_argument("--out", default="")
    a = ap.parse_args(); out = {}; models = {}
    cs = [PDStand()]
    for lp in a.policy:
        lab, path = lp.split("=", 1); cs.append(load_controller(f"rl_{lab}={path}"))
    for c in cs:
        cfg = controller_cfg(c); key = str(getattr(c, "cfg_path", "default"))
        if key not in models: models[key] = build_spec(opponent=True, cfg=cfg).compile()
        m = models[key]
        out[c.name] = {"demo": run(c, m, cfg), "variants": [run(c, m, cfg, **variant_kwargs(s)) for s in range(a.variants)]}
        t = [x["fell_s"] if x["fell_s"] is not None else DEMO_SECONDS for x in out[c.name]["variants"]]
        out[c.name]["variants_mean_survival_s"] = round(float(np.mean(t)), 2)
        out[c.name]["variants_survived_full"] = sum(x["fell_s"] is None for x in out[c.name]["variants"])
        print(c.name, json.dumps(out[c.name]), flush=True)
    if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
