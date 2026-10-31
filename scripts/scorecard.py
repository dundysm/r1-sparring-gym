"""Sparring scorecard: per controller and tier, the numbers the defender is now judged by, read from stored eval summaries
(scripts/eval_opponent.py) without simulating anything.

Headline: clean head strikes RECEIVED per minute (landed.head_clean_pm: total count / total minutes until the R1 fell, with its
(clip, start) cluster-bootstrap 95% CI); survival saturated at 0.98-1.0 on the hard tier, so it is a guard rail now, not the
score. Also: clean torso strikes received per minute, survival and up-and-within-1 m (survival_in_ring, both with Wilson CIs),
the opponent's falls (share, and the count from the raw file beside the summary), the blocked share of the library's strike
windows, on aimed tiers the blocked and head shares of the aimed opponent's own strike windows (intent_windows), and, in
summaries made after the sparring additions, the rates over minutes both robots were up and the opponent-fall log
(opponent_falls). Every value is copied from its file unchanged (no rounding, no recomputation) except opponent_fell_n (counted
from raw) and intent_blocked_share (blocked / n); missing ones are None. meta: git commit and dirty flag, this script's sha256,
every file read's sha256.
Tier label: split, tier, then 'nohits' (no scripted strikes), 'puppet' (assisted opponent), 'aimed' (opponent aimed at the head).
The hard tier comes from results/eval_opponent/aim/ (hit meter on; its unaimed bouts match results/eval_opponent/raw_*_hard*
field for field, per its README), the standard tier from results/eval_opponent/ (no hit meter there).

python scripts/scorecard.py                     # print
python scripts/scorecard.py --write             # results/sparring/scorecard_baseline.json"""
import sys, json, glob, hashlib, argparse, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["results/eval_opponent/aim/summary_*.json", "results/eval_opponent/summary_test.json", "results/eval_opponent/summary_val.json",
           "results/eval_opponent/summary_test_puppet.json", "results/eval_opponent/summary_val_puppet.json"]
OUT = "results/sparring/scorecard_baseline.json"

def script_sha():
    """sha256[:16] of this script, LF-normalised (the same on every checkout)."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]

def tier_label(meta):
    aimed = meta.get("opponent_aim") not in (None, "none", "shadow")
    return " ".join([meta["split"], meta["tier"]] + (["nohits"] if not meta.get("pushes", True) else []) + (["puppet"] if meta.get("opponent") == "puppet" else []) + (["aimed"] if aimed else []))

def row(s, raw=None, aimed=False):
    """One controller's scorecard row from its summary block (and its raw rows, for the opponent-fall count). aimed: also the
    windows of the strikes the aimed opponent actually threw (summary.intent_windows): their count, head share (stored) and
    blocked share (blocked / n, rounded to 3 as landed.windows.blocked_share is); blocked_share stays the library windows'."""
    L = s.get("landed") or {}; g = lambda k: L.get(k, {}).get("rate") if L.get(k) else None; ci = lambda k: L.get(k, {}).get("ci95") if L.get(k) else None
    out = dict(episodes=s["episodes"], head_clean_pm=g("head_clean_pm"), head_clean_pm_ci95=ci("head_clean_pm"), torso_clean_pm=g("torso_clean_pm"), torso_clean_pm_ci95=ci("torso_clean_pm"),
               survival=s["survival"], survival_ci95=s["survival_ci95"], up_within_1m=s["survival_in_ring"], up_within_1m_ci95=s["survival_in_ring_ci95"],
               opponent_fell=s["opponent_fell"], opponent_fell_n=sum(bool(x["opp_fell"]) for x in raw) if raw is not None else None,
               blocked_share=(L.get("windows") or {}).get("blocked_share"), head_clean_n=(L.get("counts") or {}).get("head_clean"))
    if "head_clean_pm_both_up" in L:
        out.update(head_clean_pm_both_up=g("head_clean_pm_both_up"), head_clean_pm_both_up_ci95=ci("head_clean_pm_both_up"), torso_clean_pm_both_up=g("torso_clean_pm_both_up"), minutes=L.get("minutes"))
    if "opponent_falls" in s: out["opponent_falls"] = s["opponent_falls"]
    W = s.get("intent_windows")
    if aimed and W: out.update(intent_n=W["n"], intent_head_share=W["head_share"], intent_blocked_share=round(W["blocked"] / W["n"], 3) if W["n"] else None)
    return out

def build(sources=SOURCES, root=ROOT):
    """{tier label: {source, raw, hit_meter, controllers: {name: row}}} and the sha256 of every file read."""
    tiers, sha = {}, {}
    for f in sorted({Path(p).resolve() for pat in sources for p in glob.glob(str(root / pat))}):
        b = f.read_bytes(); d = json.loads(b); rel = f.relative_to(root).as_posix(); sha[rel] = hashlib.sha256(b).hexdigest()[:16]
        rf = f.with_name(f.name.replace("summary_", "raw_", 1)); raw = None
        if rf.exists(): rb = rf.read_bytes(); raw = json.loads(rb)["raw"]; sha[rf.relative_to(root).as_posix()] = hashlib.sha256(rb).hexdigest()[:16]
        lab = tier_label(d["meta"]); assert lab not in tiers, f"{rel} and {tiers[lab]['source']} are both '{lab}'"
        tiers[lab] = dict(source=rel, raw=rf.relative_to(root).as_posix() if raw is not None else None, hit_meter=bool(d["meta"].get("hit_meter")), seconds=d["meta"]["seconds"],
                          opponent_aim=d["meta"].get("opponent_aim"), controllers={n: row(s, raw.get(n) if raw else None, lab.endswith(" aimed")) for n, s in d["summary"].items()})
    return dict(sorted(tiers.items())), sha

def table(tiers):
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
    lines = []
    for lab, t in tiers.items():
        lines += [f"\n{lab}  ({t['source']})", f"  {'controller':14s} {'head/min':>8s} {'95% CI':>15s} {'torso/min':>9s} {'surv':>6s} {'up<1m':>6s} {'opp fell':>9s} {'blocked':>7s} {'int.blk':>7s}"]
        for n, r in t["controllers"].items():
            c = r["head_clean_pm_ci95"]; of = f"{r['opponent_fell_n']}/{r['episodes']}" if r["opponent_fell_n"] is not None else f(r["opponent_fell"], 3)
            lines.append(f"  {n:14s} {f(r['head_clean_pm']):>8s} {('(' + f(c[0]) + ', ' + f(c[1]) + ')') if c else '-':>15s} {f(r['torso_clean_pm']):>9s} {f(r['survival'], 3):>6s} {f(r['up_within_1m'], 3):>6s} {of:>9s} {f(r['blocked_share'], 3):>7s} {f(r.get('intent_blocked_share'), 3):>7s}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true", help=f"write {OUT}"); ap.add_argument("--out", default=OUT); ap.add_argument("--sources", nargs="*", default=SOURCES)
    a = ap.parse_args(); tiers, sha = build(a.sources); print(table(tiers))
    if a.write:
        git = lambda *x: (lambda r: r.stdout.strip() if r.returncode == 0 else None)(subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True))   # None: unknown
        st = git("status", "--porcelain")   # as eval_opponent.provenance(): untracked files count
        meta = dict(made_by="python scripts/scorecard.py --write" + (f" --sources {' '.join(a.sources)}" if a.sources != SOURCES else ""), script="scripts/scorecard.py",
                    script_sha256_16=script_sha(), git_hash=git("rev-parse", "HEAD"), git_dirty=None if st is None else bool(st), sources_sha256_16=sha, primary="head_clean_pm",
                    definitions=dict(head_clean_pm="clean head strikes received per minute until the R1 fell (r1gym/hits.py, preregistered in docs/AIMED_OPPONENT.md); ci95: (clip, start) cluster bootstrap",
                                     up_within_1m="survival_in_ring: up at the end and the pelvis never more than 1 m from its start", opponent_fell_n="bouts where the opponent fell (raw opp_fell)",
                                     blocked_share="library strike windows where the limb touched the R1's arms before its head or torso (landed.windows)",
                                     intent_blocked_share="aimed tiers: the same outcome over the aimed opponent's own strike windows (intent_windows.blocked / intent_windows.n)",
                                     intent_head_share="aimed tiers: intent windows that landed on the head (intent_windows.head_share)"))
        o = ROOT / a.out; o.parent.mkdir(parents=True, exist_ok=True); json.dump(dict(meta=meta, tiers=tiers), open(o, "w"), indent=1); print(f"\nwrote {a.out}")

if __name__ == "__main__": main()
