"""Strike library -> r1gym/assets/motions/strikes.json (r1gym/strikes.py RULE on the motion the opponent's tracker performs).

Also audits the detector against hand labels (r1gym/assets/motions/strike_labels.json; how they were made is in that file):
precision and recall with Wilson 95% intervals, by limb and labelling round, for the rule and for rule variants that show
where the two design probes' counts came from (design A: 70 events, 60 hands; design B: 107, 83 hands). --check regenerates
in memory and compares with the stored file (everything but meta's git fields must be byte-identical)."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from r1gym import strikes as S
from r1gym.opponent import list_clips
from r1gym.stats import wilson

LABELS = S.ASSETS / "motions" / "strike_labels.json"
DISTANCES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
# One change at a time from RULE, toward each design probe's rule (aimed_opponent/p2_strikes.py: forward speed > 1.5 m/s along
# its heading, apex >= 0.25 m ahead of its own torso, fist / ankle points, no kick height; rg/p2_strikes.py: total speed above
# 1.0 m/s peaking >= 2.0, apex >= 0.25 m ahead of the spot, palm / foot sites, no kick height)
VARIANTS = {"rule": {}, "no_kick_height": dict(kick_min_z_m=-9.0), "A_like: apex ahead of its torso, no kick height": dict(apex_ref="torso", kick_min_z_m=-9.0),
            "B_like: total speed >= 1.0, no kick height": dict(trigger="total", v_on_mps=1.0, kick_min_z_m=-9.0)}
DESIGNS = dict(A=dict(events=70, hands=60, feet=10, probe="scratchpad/design/aimed_opponent/p2_strikes.py"), B=dict(events=107, hands=83, feet=24, probe="scratchpad/design/rg/p2_strikes.py"))

def audit(lib_by_clip, labels):
    """The detector against the certain labels.
    recall     a labelled strike is found if a strike of the same effector spans its speed peak within 0.1 s (strikes.matches)
    precision  per detection: a true strike if a candidate labelled 1 peaks inside its [onset, end], a false one if every labelled
               candidate inside it is 0; detections with no labelled candidate inside (unsampled train time) are left out
    precision_per_candidate  the share of matched candidates labelled 1 (first scoring, kept for the record: a wind-up peaking
               just before a real strike counts against it)"""
    cert = [l for l in labels if l["strike"] is not None]; rows = [(l, any(S.matches(s, l["limb"], l["frame"]) for s in lib_by_clip[l["clip"]])) for l in cert]
    dets = []
    for c, ss in lib_by_clip.items():
        for s in ss:
            ins = [l for l in cert if l["clip"] == c and S.matches(s, l["limb"], l["frame"], tol_s=0.0)]
            dets.append((s, None if not ins else int(any(l["strike"] for l in ins)), ins[0] if ins else None))
    def pr(keep, keep_d):
        sel = [r for r in rows if keep(r[0])]; d = [x for x in dets if x[1] is not None and keep_d(x)]
        tp = sum(1 for l, f in sel if f and l["strike"]); fn = sum(1 for l, f in sel if not f and l["strike"]); fpc = sum(1 for l, f in sel if f and not l["strike"])
        tpd, fpd = sum(x[1] for x in d), sum(1 - x[1] for x in d)
        return dict(labelled=len(sel), strikes=tp + fn, found=tp, recall=round(tp / (tp + fn), 3) if tp + fn else None, recall_ci95=wilson(tp, tp + fn) if tp + fn else None,
                    detections_scored=len(d), true_detections=tpd, precision=round(tpd / len(d), 3) if d else None, precision_ci95=wilson(tpd, len(d)) if d else None,
                    precision_per_candidate=round(tp / (tp + fpc), 3) if tp + fpc else None)
    hand = lambda limb: limb.endswith("hand"); sp = lambda c: S.split_of(c)
    out = dict(all=pr(lambda l: True, lambda x: True), hands=pr(lambda l: hand(l["limb"]), lambda x: x[0].hand), feet=pr(lambda l: not hand(l["limb"]), lambda x: not x[0].hand),
               val=pr(lambda l: sp(l["clip"]) == "val", lambda x: x[0].split == "val"), train=pr(lambda l: sp(l["clip"]) == "train", lambda x: x[0].split == "train"))
    def why(l):   # what the rule saw around a missed strike's speed peak
        q, fps = S.motion(l["clip"]); P, _ = S.effector_paths(q); j = list(S.EFFECTORS).index(l["limb"]); H = P[:, j] * [-1.0, -1.0, 1.0]
        V = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) * fps / 2; w = [(l["frame"] + k) % len(q) for k in range(-12, 13)]
        return f"max forward speed {V[w, 0].max():.1f} m/s, most forward {H[w, 0].max():.2f} m at {H[w[int(np.argmax(H[w, 0]))], 2]:.2f} m high"
    out["misses"] = [f'{l["sheet"]} {l["clip"]} {l["limb"]} f{l["frame"]}: {l["note"]} [{why(l)}]' for l, f in rows if l["strike"] and not f]
    out["false_detections"] = [f'{s.clip} {s.limb} frames {s.onset}-{s.end}: {l["sheet"]} {l["note"]}' for s, t, l in dets if t == 0]
    return out

def coverage(lib, split):
    """(zone, bout distance) cells with at least one hand strike of the split, level with the zone (its band) and within its
    half-width + 5 cm sideways, that reaches it unwarped (reach >= distance)."""
    fr = S.r1_surfaces(); out = {}
    for z in ("head", "torso"):
        hs = [s for s in lib if s.split == split and s.hand and s.band == z and abs(s.apex_m[1]) <= fr[z]["half_width_m"] + 0.05]
        per = {f"{d:.2f}": sum((s.reach_head_m if z == "head" else s.reach_torso_m) >= d for s in hs) for d in DISTANCES}
        out[z] = dict(aligned_strikes=len(hs), reaching_at=per, cells_covered=f"{sum(v > 0 for v in per.values())}/{len(per)}")
    return out

def build():
    import eval_opponent as eo
    from r1gym.model import build_spec, load_cfg
    clips = list_clips(); labels = json.load(open(LABELS, encoding="utf-8"))["labels"]; lib, per, var = [], {}, {}
    for v, r in VARIANTS.items():
        byc = {}
        for c in clips:
            q, fps = S.motion(c); st = {}; byc[c] = S.detect(q, fps, c, S.split_of(c), rule=r, stats=st)
            if v == "rule":
                lib += byc[c]; ss = byc[c]
                per[c] = dict(split=S.split_of(c), frames=len(q), seconds=round(len(q) / fps, 2), strikes=len(ss), hands=sum(s.hand for s in ss), feet=sum(not s.hand for s in ss),
                              by_band={b: sum(s.band == b for s in ss) for b in ("head", "torso", "low", "high")}, by_type={t: sum(s.type == t for s in ss) for t in ("straight", "hook", "uppercut", "downward")},
                              dropped_at_seam=st.get("dropped_at_seam", 0))
        n = [s for ss in byc.values() for s in ss]; a = audit(byc, labels)
        var[v] = dict(events=len(n), hands=sum(s.hand for s in n), feet=sum(not s.hand for s in n), **{f"{k}_{m}": a[k][m] for k in ("all", "hands") for m in ("precision", "recall")})
        if v == "rule": aud = a
    splits = {sp: dict(seconds=round(sum(p["seconds"] for p in per.values() if p["split"] == sp), 2), strikes=sum(p["strikes"] for p in per.values() if p["split"] == sp),
                       hands=sum(p["hands"] for p in per.values() if p["split"] == sp), head_band=sum(s.band == "head" for s in lib if s.split == sp),
                       head_band_hands=sum(s.band == "head" and s.hand for s in lib if s.split == sp), coverage=coverage(lib, sp)) for sp in ("train", "val", "test")}
    hands = [s for s in lib if s.hand]; q = lambda x, p: round(float(np.percentile(x, p)), 3)
    summary = dict(strikes=len(lib), hands=len(hands), feet=len(lib) - len(hands), splits=splits,
                   hand_apex_fwd_m={f"p{p}": q([s.apex_m[0] for s in hands], p) for p in (10, 50, 90)}, hand_apex_z_m={f"p{p}": q([s.apex_m[2] for s in hands], p) for p in (10, 50, 90)},
                   hand_peak_fwd_mps={f"p{p}": q([s.peak_fwd_mps for s in hands], p) for p in (10, 50, 90)}, hand_rise_s={f"p{p}": q([s.rise_s for s in hands], p) for p in (10, 50, 90)},
                   hand_torso_z_m={f"p{p}": q([s.torso_z_m for s in hands], p) for p in (10, 50, 90)}, hand_reach_head_m={f"p{p}": q([s.reach_head_m for s in hands], p) for p in (10, 50, 90)},
                   test_note=f"test (kfab_202, kfab_205) is {splits['test']['seconds']} s with {splits['test']['strikes']} strikes, {splits['test']['head_band']} of them level with the head: "
                             "aiming on test rests on a handful of strikes")
    cfg = load_cfg(); meta = {k: v for k, v in eo.provenance(build_spec(opponent=True, cfg=cfg).compile(), cfg).items() if k in ("git_hash", "git_dirty", "mujoco")}
    designs = dict(DESIGNS, why=("Counts per rule variant (same clips, same labels) are in 'variants'. The kick-height floor removes the steps both probes "
                                 "counted as foot strikes; B's total-speed trigger also fires on retractions, wind-ups and swings back, which adds events at "
                                 "the cost of precision; A measured the apex from its own torso, which moves with the strike, so lunging strikes pass with "
                                 "less reach toward the R1. Neither probe audited its events."))
    return dict(about="Hand and foot strikes in the motion the opponent's tracker performs (r1gym/strikes.py RULE). Generated by scripts/strike_library.py; "
                      "audited against strike_labels.json (precision is per detection, scored only where a labelled candidate peaks inside it: in val every "
                      "candidate is labelled, in train a sample, so a wind-up inside a long window whose drive wasn't sampled counts as a false detection).",
                rule={**S.RULE, "effectors": S.EFFECTORS, "r1_surfaces": S.r1_surfaces(), "reach": "bout distance (pelvis to spot) at which the apex meets the zone's front, the R1 at home"},
                meta=dict(meta, generated_by="scripts/strike_library.py"), per_clip=per, summary=summary,
                audit=dict(labels="r1gym/assets/motions/strike_labels.json", labelled=len(json.load(open(LABELS, encoding="utf-8"))["labels"]),
                           unsure=sum(l["strike"] is None for l in labels), **aud),
                variants=var, designs=designs, strikes=[S.as_dict(s) for s in lib])

def dumps(d): return json.dumps(d, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args(); d = build()
    if a.check:
        old = json.load(open(S.PATH, encoding="utf-8")); strip = lambda x: dumps({**x, "meta": {k: v for k, v in x["meta"].items() if not k.startswith("git")}})
        ok = strip(old) == strip(d); print("strikes.json regenerates byte-identical (meta git fields aside):", ok); sys.exit(0 if ok else 1)
    open(S.PATH, "w", encoding="utf-8").write(dumps(d)); print(dumps({k: v for k, v in d.items() if k != "strikes"}))
