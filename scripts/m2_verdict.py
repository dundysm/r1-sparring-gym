"""Sparring M2 selection and verdict, exactly as preregistered (results/sparring/prereg_m2.json, docs/SPARRING_M1M2.md
section 5), from the raw files scripts/eval_opponent.py writes into its --out (no simulation).

select   reads only the val runs: raw_val_hard_nohits_aim.json (aimed, --aim chosen) and raw_val_hard_nohits.json (its
         unaimed pair: same episodes, reported). Among A and B, those with aimed survival >= 0.95 and survival_in_ring at
         most 0.05 below guard_fw_r's; of those (of both if none qualifies) the lower exposure-weighted head clean
         strikes/min; a difference under 0.1/min picks A. -> results/sparring/m2_selection.json (B: train Blind-B first).
verdict  reads the test runs (raw_test_hard_nohits_aim.json, raw_test_hard_nohits.json) and the selection. Rates: sum of
         counts / sum of minutes until the R1 fell (hit_meter.n.<zone>_clean, hit_meter.secs / 60). R = rate(candidate) /
         rate(reference) on all bouts; its 95% CI the 2.5 / 97.5 percentiles over 4000 (clip, t0) cluster-bootstrap
         resamples (seed 0, r1gym/stats.py's draw), every controller's bouts resampled with the same clusters (paired: one
         index matrix for every ratio); a zero reference rate counts as R = inf, in a resample and in the point estimate.
         The five rule parts, and the preregistration's reported-regardless items the raw files hold: every HitMeter rule
         (exposure-weighted), library and intent window outcomes with blocked share, impulse per head hit, the unaimed rows;
         beside them each run's scorecard rows and sensitivity block (scripts/scorecard.py row, from the summary files, when
         present). -> results/sparring/m2_verdict.json
Controllers are matched by label with any 'RL ' prefixes stripped (eval_opponent adds one; doubled ones occur), case, '-' and
spaces ignored: guard_fw_r, A, B, blind (A's twin), blind_b (B's twin), or --label ROLE=NAME (replaces that role's aliases).
Every run must be the preregistered one or the script exits: split, hard tier (20 s bouts, its distances, DR, seed), the tracker
opponent, no scripted hits, hit meter, keypoints truth, --aim chosen, reps (val 32, test 16), 128 bouts, aimed episodes equal
to the unaimed pair's, and kp_mode per controller (the reference 83-d, candidates sighted, twins blind). Decisions use unrounded values; inf is written as "inf".

python scripts/m2_verdict.py select  --out results/eval_opponent/m2
python scripts/m2_verdict.py verdict --out results/eval_opponent/m2"""
import sys, json, math, hashlib, argparse, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from r1gym.stats import _cluster_ix
from scorecard import tier_label, row

PREREG = ROOT / "results" / "sparring" / "prereg_m2.json"
SELECTION, VERDICT = ROOT / "results" / "sparring" / "m2_selection.json", ROOT / "results" / "sparring" / "m2_verdict.json"
N_BOOT, SEED, BOUTS, REPS = 4000, 0, 128, dict(val=32, test=16)
HARD = dict(distances=[0.4, 0.45, 0.5, 0.55], seconds=20.0, dr=True, seed=20_000)   # eval_opponent.TIERS["hard"] as preregistered (pinned here)
SURV_MIN, RING_DROP_MAX, TIE_PM = 0.95, 0.05, 0.1                                       # selection, and rule parts 4 and 5
HEAD_WARM_MAX, HEAD_WARM_HI, HEAD_BLIND_HI, TORSO_WARM_MAX = 0.60, 0.80, 0.90, 1.5      # rule parts 1-3 (CI bounds strict)
TOL = 1e-9   # float slack on the inclusive thresholds (fractions of 128 bouts never sit on them; a 0.1/min tie can)
ALIASES = dict(ref=("guard_fw_r",), A=("a", "spar_a", "block_a"), B=("b", "spar_b", "block_b"),
               blind=("blind", "blind_a", "spar_blind", "block_blind"), blind_b=("blind_b", "spar_blind_b", "block_blind_b"))
TWIN = dict(A="blind", B="blind_b")
KP_MODE = dict(ref=None, A="obs", B="obs", cand="obs", blind="zeros", blind_b="zeros", twin="zeros")

def read(path):
    """(parsed JSON, sha256[:16]) of a file: every file this script reads goes through here."""
    b = Path(path).read_bytes(); return json.loads(b), hashlib.sha256(b).hexdigest()[:16]

def rel(p):
    p = Path(p).resolve(); return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)

def norm(label):
    """'RL RL Blind-B' -> 'blind_b': eval_opponent's 'RL ' prefix (doubled when the label already had one), case, '-', spaces."""
    s = label.strip()
    while s[:3].lower() == "rl ": s = s[3:].lstrip()
    return s.lower().replace("-", "_").replace(" ", "_")

def find(names, role, labels=None):
    """The one controller in names playing role (ALIASES, or labels[role] alone: that exact name if present); exits on none or several."""
    if labels and role in labels and labels[role] in names: return labels[role]
    want = {norm(labels[role])} if labels and role in labels else set(ALIASES[role]); hit = [n for n in names if norm(n) in want]
    if len(hit) != 1: sys.exit(f"{role}: {len(hit)} controllers labelled {sorted(want)} among {sorted(names)} (--label {role}=NAME)")
    return hit[0]

def load(out, split, aimed, files):
    f = Path(out) / f"raw_{split}_hard_nohits{'_aim' if aimed else ''}.json"
    if not f.exists(): sys.exit(f"{f} missing: scripts/eval_opponent.py --split {split} --tier hard --no_pushes --keypoints truth" + (" --aim chosen" if aimed else "") + f" --out {out}")
    d, files[rel(f)] = read(f); return d

def protocol(U, A, split, who):
    """Departures from the preregistered runs ([] if none): U the unaimed raw file, A the aimed one, who {role: label}."""
    bad = []
    for d, aimed in ((U, False), (A, True)):
        m = d["meta"]; lab = tier_label(m); want = f"{split} hard nohits" + (" aimed" if aimed else ""); k = (m.get("keypoints") or {})
        if lab != want: bad.append(f"{'aimed' if aimed else 'unaimed'} run is '{lab}', not '{want}'")
        if not m.get("hit_meter"): bad.append(f"{lab}: no hit meter (--hits 1)")
        if m.get("seconds") != HARD["seconds"]: bad.append(f"{lab}: {m.get('seconds')} s bouts, the hard tier's are {HARD['seconds']} s")
        tc = m.get("tier_cfg") or {}
        if any(tc.get(x) != v for x, v in HARD.items()): bad.append(f"{lab}: tier_cfg {dict((x, tc.get(x)) for x in HARD)}, the hard tier's is {HARD}")
        if m.get("opponent") != "tracked" or m.get("opponent_mode") != "policy": bad.append(f"{lab}: opponent {m.get('opponent')!r} / {m.get('opponent_mode')!r}, not the tracker ('tracked' / 'policy')")
        if k.get("source") != "truth": bad.append(f"{lab}: keypoints {k.get('source')!r}, not 'truth'")
        if m.get("reps") != REPS[split] or len(d["episodes"]) != BOUTS: bad.append(f"{lab}: reps {m.get('reps')}, {len(d['episodes'])} bouts; preregistered {REPS[split]}, {BOUTS}")
        if aimed and not (m.get("aim") or {}).get("from_aim_json"): bad.append(f"{lab}: not --aim chosen")
        for r, n in who.items():
            if n not in d["raw"]: bad.append(f"{lab}: no {r} ({n})"); continue
            if len(d["raw"][n]) != len(d["episodes"]): bad.append(f"{lab}: {n} has {len(d['raw'][n])} rows for {len(d['episodes'])} episodes")
            if k.get("kp_mode", {}).get(n) != KP_MODE[r]: bad.append(f"{lab}: {n} kp_mode {k.get('kp_mode', {}).get(n)!r}, a {r} needs {KP_MODE[r]!r}")
            if any("hit_meter" not in x for x in d["raw"][n]): bad.append(f"{lab}: {n} rows without hit_meter")
    if U["episodes"] != A["episodes"]: bad.append("the aimed run's episodes differ from its unaimed pair's")
    return bad

def counts(rows):
    """Per bout (eval_opponent rows): head and torso clean strikes received, minutes until the R1 fell, up at the end, up and within 1 m."""
    H = [x["hit_meter"] for x in rows]; a = lambda v: np.array(v, float)
    return dict(head=a([h["n"]["head_clean"] for h in H]), torso=a([h["n"]["torso_clean"] for h in H]), mins=a([h["secs"] / 60 for h in H]),
                up=a([x["fell_s"] is None for x in rows]), ring=a([x["fell_s"] is None and not x["ring_out"] for x in rows]))

rate = lambda C, z: float(C[z].sum() / C["mins"].sum())   # exposure-weighted: sum of counts / sum of minutes

def div(a, b): return float(a / b) if b > 0 else math.inf   # a zero reference rate: R = inf (preregistered)

def pct(x, q):
    """np.percentile (linear), also when x holds inf: numpy gives nan between two infs, here the answer is inf."""
    v = np.sort(np.asarray(x, float))
    if np.isfinite(v).all(): return float(np.percentile(v, q))
    h = (len(v) - 1) * q / 100; lo = int(math.floor(h)); f = h - lo
    if f == 0 or math.isinf(v[lo + 1]): return float(v[lo]) if f == 0 else math.inf
    return float(v[lo] + (v[lo + 1] - v[lo]) * f)

def ratio(num, den, z, clusters):
    """R = rate(num) / rate(den) of zone z on all bouts, and its 95% percentile CI over N_BOOT (clip, t0) cluster resamples
    (stats.cluster_paired_diff's draw, seed SEED: the same index matrix every call, so all ratios share their resamples, and
    both controllers' bouts come from the same clusters); a resample with a zero reference rate counts as inf."""
    c = _cluster_ix(clusters); k = int(c.max()) + 1; i = np.random.default_rng(SEED).integers(0, k, (N_BOOT, k))
    S = lambda x: np.bincount(c, x, k)[i].sum(1); rn, rd = S(num[z]) / S(num["mins"]), S(den[z]) / S(den["mins"])
    with np.errstate(divide="ignore", invalid="ignore"): bs = np.where(rd > 0, rn / np.where(rd > 0, rd, 1), np.inf)
    return dict(R=div(rate(num, z), rate(den, z)), ci95=[pct(bs, 2.5), pct(bs, 97.5)], rate=rate(num, z), ref_rate=rate(den, z), resamples_inf=int(np.isinf(bs).sum()))

def report(rows):
    """One controller's raw rows as the preregistration reports them regardless: survival, up within 1 m, head / torso clean
    strikes/min (sum of counts / sum of minutes), every HitMeter rule (per-bout rates weighted by minutes), outcomes and blocked
    share of the library strike windows and of the opponent's intent windows, impulse per head hit (pooled: n, median, p90)."""
    C = counts(rows); H = [x["hit_meter"] for x in rows]; w = C["mins"]; imp = [v for h in H for v in h["head_hit_imp_Ns"]]
    def win(key):
        if key not in H[0]: return None
        W = {o: sum(h[key][o] for h in H) for o in H[0][key]}; n = sum(W.values()); return dict(W, n=n, blocked_share=W["blocked"] / n if n else None)
    return dict(episodes=len(rows), survival=float(C["up"].mean()), survival_in_ring=float(C["ring"].mean()), minutes=float(w.sum()),
                head_clean_n=int(C["head"].sum()), head_clean_pm=rate(C, "head"), torso_clean_n=int(C["torso"].sum()), torso_clean_pm=rate(C, "torso"),
                rules={r: {z: float(np.dot([h["rules"][r][j] for h in H], w) / w.sum()) for j, z in enumerate(("head", "torso"))} for r in H[0]["rules"]},
                windows=win("windows"), intent_windows=win("intent_windows"),
                impulse_per_head_hit_Ns=dict(n=len(imp), **({"median": float(np.median(imp)), "p90": float(np.percentile(imp, 90))} if imp else {})))

def scorecard_rows(out, raw_name, d, files):
    """scripts/scorecard.py's rows and the sensitivity block per controller, from the summary beside a raw file (None if absent)."""
    f = Path(out) / raw_name.replace("raw_", "summary_", 1)
    if not f.exists(): return None
    s, files[rel(f)] = read(f); aimed = tier_label(s["meta"]).endswith(" aimed")
    return {n: dict(row=row(v, d["raw"].get(n), aimed), sensitivity=v.get("sensitivity")) for n, v in s["summary"].items()}

def meta(mode, files, made_by):
    git = lambda *x: (lambda r: r.stdout.strip() if r.returncode == 0 else None)(subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True))
    st = git("status", "--porcelain"); _, files[rel(PREREG)] = read(PREREG)
    return dict(mode=mode, made_by=made_by, script="scripts/m2_verdict.py", script_sha256_16=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16],
                git_hash=git("rev-parse", "HEAD"), git_dirty=None if st is None else bool(st), prereg="results/sparring/prereg_m2.json", files_sha256_16=dict(sorted(files.items())),
                n_boot=N_BOOT, seed=SEED, rate="sum of hit_meter.n.<zone>_clean / sum of hit_meter.secs / 60", inf='"inf": a zero reference rate')

def jsonable(x):
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)): return "inf" if math.isinf(x) else float(x)
    return x.item() if isinstance(x, np.generic) else x

def write(res, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True); json.dump(jsonable(res), open(dest, "w"), indent=1, allow_nan=False); return res

def select(out, dest=SELECTION, labels=None, made_by=None):
    """The preregistered val selection between A and B (module docstring); reads the val raw files only."""
    files = {}; U, A = load(out, "val", False, files), load(out, "val", True, files)
    who = {r: find(list(A["raw"]), r, labels) for r in ("ref", "A", "B")}; bad = protocol(U, A, "val", who)
    if bad: sys.exit("not the preregistered val runs:\n  " + "\n  ".join(bad))
    C = {r: counts(A["raw"][n]) for r, n in who.items()}; ring_ref = float(C["ref"]["ring"].mean()); cand = {}
    for r in ("A", "B"):
        s, ring = float(C[r]["up"].mean()), float(C[r]["ring"].mean())
        cand[r] = dict(label=who[r], survival=s, survival_in_ring=ring, ring_drop_vs_ref=ring_ref - ring, head_clean_pm=rate(C[r], "head"),
                       qualifies=bool(s >= SURV_MIN - TOL and ring_ref - ring <= RING_DROP_MAX + TOL))
    pool = [r for r in ("A", "B") if cand[r]["qualifies"]] or ["A", "B"]; d = cand["A"]["head_clean_pm"] - cand["B"]["head_clean_pm"]
    pick = pool[0] if len(pool) == 1 else "B" if d >= TIE_PM - TOL else "A"   # two in the pool: B only when lower by 0.1/min or more
    return write(dict(meta=meta("select", files, made_by), rule=read(PREREG)[0]["selection"]["rule"], reference=dict(label=who["ref"], survival_in_ring=ring_ref),
                      candidates=cand, pool=pool, head_clean_pm_A_minus_B=d, tie_break_to_A=len(pool) == 2 and abs(d) < TIE_PM - TOL,
                      selected=pick, selected_label=who[pick], twin=TWIN[pick], train_blind_b=pick == "B",
                      aimed_rows={n: report(r) for n, r in A["raw"].items()}, unaimed_rows={n: report(r) for n, r in U["raw"].items()}), dest)

def verdict(out, selection=SELECTION, dest=VERDICT, labels=None, made_by=None):
    """The preregistered test verdict (module docstring) for the selected candidate."""
    files = {}; S, files[rel(selection)] = read(selection); pick = S["selected"]
    if pick not in TWIN: sys.exit(f"{selection}: selected {pick!r}, not A or B")
    U, A = load(out, "test", False, files), load(out, "test", True, files)
    who = dict(ref=find(list(A["raw"]), "ref", labels), cand=find(list(A["raw"]), pick, labels), twin=find(list(A["raw"]), TWIN[pick], labels))
    bad = protocol(U, A, "test", who)
    if bad: sys.exit("not the preregistered test runs:\n  " + "\n  ".join(bad))
    C = {r: counts(A["raw"][n]) for r, n in who.items()}; cl = [(e["clip"], e["t0"]) for e in A["episodes"]]
    Rw, Rb, Rt = ratio(C["cand"], C["ref"], "head", cl), ratio(C["cand"], C["twin"], "head", cl), ratio(C["cand"], C["ref"], "torso", cl)
    surv, ring_c, ring_r = float(C["cand"]["up"].mean()), float(C["cand"]["ring"].mean()), float(C["ref"]["ring"].mean())
    parts = dict(head_vs_warm=dict(R_warm=Rw["R"], ci95_hi=Rw["ci95"][1], need=f"R_warm <= {HEAD_WARM_MAX} and ci95_hi < {HEAD_WARM_HI}", ok=Rw["R"] <= HEAD_WARM_MAX + TOL and Rw["ci95"][1] < HEAD_WARM_HI),
                 head_vs_blind=dict(ci95_hi=Rb["ci95"][1], need=f"ci95_hi of R_blind < {HEAD_BLIND_HI}", ok=Rb["ci95"][1] < HEAD_BLIND_HI),
                 torso_vs_warm=dict(R_warm_torso=Rt["R"], need=f"R_warm(torso) <= {TORSO_WARM_MAX}", ok=Rt["R"] <= TORSO_WARM_MAX + TOL),
                 survival=dict(survival=surv, need=f">= {SURV_MIN}", ok=surv >= SURV_MIN - TOL),
                 ring=dict(ref_minus_candidate=ring_r - ring_c, need=f"<= {RING_DROP_MAX}", ok=ring_r - ring_c <= RING_DROP_MAX + TOL))
    return write(dict(meta=meta("verdict", files, made_by), selected=pick, roles=who, selection=dict(file=rel(selection), selected_label=S.get("selected_label"), candidates=S.get("candidates")),
                      ratios=dict(R_warm_head=Rw, R_blind_head=Rb, R_warm_torso=Rt), rule=parts, passed=all(p["ok"] for p in parts.values()),
                      reported_regardless=dict(aimed_rows={n: report(r) for n, r in A["raw"].items()}, unaimed_rows={n: report(r) for n, r in U["raw"].items()},
                                               scorecard=dict(aimed=scorecard_rows(out, "raw_test_hard_nohits_aim.json", A, files), unaimed=scorecard_rows(out, "raw_test_hard_nohits.json", U, files)))), dest)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0]); ap.add_argument("mode", choices=["select", "verdict"])
    ap.add_argument("--out", required=True, help="scripts/eval_opponent.py --out holding the raw files"); ap.add_argument("--dest", default=None, help="output JSON (default results/sparring/m2_{selection,verdict}.json)")
    ap.add_argument("--selection", default=str(SELECTION), help="verdict: the selection file"); ap.add_argument("--label", nargs="*", default=[], help="ROLE=NAME (ref, A, B, blind, blind_b)")
    a = ap.parse_args(); labels = dict(x.split("=", 1) for x in a.label); bad = set(labels) - set(ALIASES)
    if bad: sys.exit(f"--label roles {sorted(bad)}: not in {list(ALIASES)}")
    mb = "python scripts/m2_verdict.py " + " ".join(sys.argv[1:])
    r = select(a.out, a.dest or SELECTION, labels, mb) if a.mode == "select" else verdict(a.out, Path(a.selection), a.dest or VERDICT, labels, mb)
    if a.mode == "select": print(f"selected {r['selected']} ({r['selected_label']}); pool {r['pool']}; head/min A - B {r['head_clean_pm_A_minus_B']:.3f}" + ("; train Blind-B before the test" if r["train_blind_b"] else ""))
    else: print(json.dumps(jsonable(dict(passed=r["passed"], rule=r["rule"])), indent=1))

if __name__ == "__main__": main()
