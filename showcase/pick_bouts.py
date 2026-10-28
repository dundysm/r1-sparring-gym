"""Pick the showcase bouts from a stored eval run by a fixed rule, so the video isn't cherry-picked by eye.

python showcase/pick_bouts.py --raw results/eval_opponent/raw_test_hard.json --hero "RL guard_fw" --base pd_stand > showcase/bouts.json

Candidates: bouts where the hero survived and never left the 1 m ring and the base controller fell. Ranked by the
hardest single strike to the head, then total scripted impulse. Also reports the hero's survival over the whole run, so
the picked bout can be put in context on screen ("one of N bouts it survived")."""
import json, argparse
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--hero", default="RL guard_fw"); ap.add_argument("--base", default="pd_stand")
    ap.add_argument("--n", type=int, default=5); a = ap.parse_args()
    st = json.load(open(a.raw)); eps = st["episodes"]; H, B = st["raw"][a.hero], st["raw"][a.base]
    rows = []
    for i, (ep, h, b) in enumerate(zip(eps, H, B)):
        if h["fell_s"] is not None or h["ring_out"] or b["fell_s"] is None: continue
        head = max([imp for k, loc, imp, t, y in ep["hits"] if loc == "head"] or [0.0])
        rows.append(dict(episode=i, clip=ep["clip"], distance=ep["distance"], head_max_ns=round(head, 1), total_ns=round(sum(x[2] for x in ep["hits"]), 1),
                         base_fell_s=b["fell_s"], hero_drift_m=h["drift_m"], hero_max_disp_m=h["max_disp_m"], hits=ep["hits"]))
    rows.sort(key=lambda r: (-r["head_max_ns"], -r["total_ns"]))
    up = sum(x["fell_s"] is None for x in H); inr = sum(x["fell_s"] is None and not x["ring_out"] for x in H)
    print(json.dumps(dict(raw=a.raw, hero=a.hero, base=a.base, rule="hero up and in the ring, base fell; by hardest head strike, then total impulse",
                          hero_survived=f"{up}/{len(H)}", hero_in_ring=f"{inr}/{len(H)}", candidates=len(rows), top=rows[:a.n]), indent=1))
