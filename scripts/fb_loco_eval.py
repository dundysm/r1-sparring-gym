"""P3 LOCO pilot evaluation on the pod CPU (M2V2_PLAN_v3 section 4 P3 bars, 3.1 'P3 LOCO', B5, B6; A5.2, A8.2, A9.1, A9.2).
Candidates  the pilot pair's final checkpoints, sighted and blind (run names loco_s42[_blind]_{A0,A,B}, pod_fb.sh P3): B's newest
            model_*.pt; a sighted B stopped by WATCH6 (an 'EARLY_STOP loco_s42_B' line in progress_fb.txt) or absent falls back to A's newest
            (plan 4: 'a stopped stage B falls back to the stage-A final'); each exported by export_hybrid_fb (float64 graph, A9.1) with the
            blind flag read from the checkpoint (A9.2), never an argument. start = the exported start H(A3s)/M3s (A9.1: LOCO, selection
            and the test use the exported start through the same path); G = its shipped ONNX.
Episodes    held-out kfab_203 88000-89151 (role loco_heldout: 4 distances x 288) and seen kfab_200/201 89500-90651 (role loco_seen: 2 x 4 x
            144), fb_r0eo's construction (eval_opponent.episodes, tier hard, aimed, 20 s).
Platform    A8.2: every controller, the start and G included, runs here, on the same machine as the pilots (the plan's 'precomputed
            locally' rows would be another platform); EO.run unchanged, as R0-EO runs a cell (threads 1, DR, meter, falls, keypoints truth,
            aim.json lag), rows in fb_r0eo's format; <work>/<ctrl>/<set>_<chunk>.json.gz (32 bouts, gzip mtime 0, atomic), resumable.
Bars        fb_eq.loco_bars(pilot vs start on the same episodes): held-out R <= 0.90 and upper < 1.0; seen R <= 0.85 and upper < 0.95;
            survival diff >= -0.015 and lower >= -0.04; g = log R_ho / log R_seen >= 0.4. Decision: proceed iff all four; seen passes but
            held-out or g fails -> stop and report (A5.2 replaced 'Track 2 first'); seen or survival fails -> stop and diagnose.
            Reported: R(pilot vs blind pilot) per clip set (the blind pilot against the start too), g's bootstrap 90% CI (episodes resampled
            within each set, paired), G and the start's own rates.
--dry       the plumbing on the M10 smoke block (role 'smoke', --seeds; kfab_203 seeds as the held-out set, the rest as seen) with the M10
            plumbing checkpoints as the pilots (ckpt_B.pt sighted, ckpt_blind_twin.pt blind): no LOCO seed is simulated.
Usage: python -I scripts/fb_loco_eval.py --out loco.json [--workers N] [--sighted CKPT] [--blind CKPT] [--work DIR] [--dry --seeds 69000-69011 --ckdir DIR]"""
import sys, os, json, gzip, time, math, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np

FMT, CHUNK, NBOOT = "loco/1", 32, 2000
FB_ROOT = Path(os.environ.get("FB_ROOT", "/root")); RUNS = FB_ROOT / "unitree_rl_mjlab" / "logs" / "rsl_rl" / "r1_fullbody"; PROGRESS = FB_ROOT / "progress_fb.txt"
SETS = dict(heldout=("loco_heldout", 88000, 288, ("kfab_203",)), seen=("loco_seen", 89500, 144, ("kfab_200", "kfab_201")))
CTRLS = ("pilot", "pilot_blind", "start", "G")

def newest(run, runs=RUNS):
    """The newest model_*.pt of the run dirs named *_<run> (mjlab: '<timestamp>_<run name>'), or None."""
    ds = sorted((d for d in Path(runs).glob(f"*_{run}") if d.is_dir()), key=lambda d: d.stat().st_mtime)
    for d in reversed(ds):
        ck = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if ck: return ck[-1]
    return None

def stopped(name, progress=PROGRESS):
    p = Path(progress); return p.exists() and any(l.startswith(f"EARLY_STOP {name} ") for l in p.read_text(encoding="utf-8", errors="replace").splitlines())

def pilot_ckpt(blind, runs=RUNS, progress=PROGRESS, tag="loco_s42"):
    """(checkpoint, stage, note) of a LOCO pilot (module docstring); the blind twin is never stopped (WATCH6)."""
    nm = tag + ("_blind" if blind else ""); b, a = newest(f"{nm}_B", runs), newest(f"{nm}_A", runs)
    if b is not None and not (not blind and stopped(f"{nm}_B", progress)): return b, "B", ""
    if a is None: raise FileNotFoundError(f"no checkpoint of {nm}_A / _B under {runs}")
    return a, "A", "B stopped by WATCH6: the A final" if b is not None else "no B run: the A final" + (" (A stopped: _stopped)" if stopped(f"{nm}_A", progress) else "")

def episodes(dry=False, seeds=None):
    """{set: eval-format episodes} (module docstring), each set checked by fb_seeds in its role."""
    from fb_seeds import check
    if dry:
        import fb_cpu_smoke as SM, fb_t1c as TC
        eps = [TC._eo_episode(SM.episode(s, SM.PLUMB["block"][0])) for s in seeds]
        for e in eps: check([e["seed"]], "smoke", {e["clip"]})
        return dict(heldout=[e for e in eps if e["clip"] == "kfab_203"], seen=[e for e in eps if e["clip"] != "kfab_203"]), dict(heldout="smoke", seen="smoke")
    import fb_r0eo as R0
    out, roles = {}, {}
    for k, (role, s0, reps, clips) in SETS.items():
        e = R0.episodes(s0, reps, clips=clips); assert len(e) == 1152; check([x["seed"] for x in e], role, {x["clip"] for x in e}); out[k], roles[k] = e, role
    return out, roles

def export_all(work, cks):
    """{ctrl: ONNX path}: the pilots (blind flag from their checkpoints), the start (exported, A9.1), G (shipped)."""
    import export_hybrid_fb as EX, hybrid_actor as HA
    out = {}
    for c, ck in (("pilot", cks["pilot"]), ("pilot_blind", cks["pilot_blind"])):
        b = EX.ckpt_blind(ck); assert b == (c == "pilot_blind"), f"{c}: checkpoint {ck} stores blind = {b}"
        EX.export(Path(work) / "onnx" / c, blind=b, ckpt=ck); out[c] = Path(work) / "onnx" / c / "policy.onnx"
    EX.export(Path(work) / "onnx" / "start", blind=False); out["start"] = Path(work) / "onnx" / "start" / "policy.onnx"; out["G"] = ROOT / HA.G_ONNX
    return out

def run_cell(spec, path, eps, role, seconds=20.0):
    """EO.run (unchanged, fb_r0eo's probe installed) of one controller on eps -> fb_r0eo rows; the seeds checked in `role` first."""
    import eval_opponent as EO, fb_r0eo as R0
    from fb_seeds import check
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import build_spec
    check([e["seed"] for e in eps], role, {e["clip"] for e in eps})
    c = load_controller(f"{spec}={path}"); cfg = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": R0.AIM}; m = build_spec(opponent=True, cfg=cfg).compile()
    aim = json.load(open(ROOT / "results" / "opponent_tracker" / "aim.json")); assert aim["operator"] == R0.AIM; lag = float(aim.get("lag_s", 0.0))
    base = EO.R1HitEnv; assert not getattr(base, "_fb_probe", False); EO.R1HitEnv = R0.probe_cls(EO); R0._ENVS.clear()
    try:
        eo = EO.run(c, eps, m, cfg, seconds, threads=1, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")
        assert len(R0._ENVS) == len(eps); rows = [R0.bout_row(ep, r, e._fb, seconds) for ep, r, e in zip(eps, eo, R0._ENVS)]
    finally: EO.R1HitEnv = base; R0._ENVS.clear()
    return rows

def _job(args):
    ctrl, path, st, j, eps, role, work, plat = args
    import torch, fb_t1c as TC
    torch.set_num_threads(1); p = Path(work) / ctrl / f"{st}_{j:03d}.json.gz"
    meta = dict(fmt=FMT, ctrl=ctrl, set=st, chunk=j, seeds=[e["seed"] for e in eps], onnx_sha256=TC.sha(path), platform=plat)
    if p.exists():
        try:
            if json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))["meta"] == meta: return dict(ctrl=ctrl, set=st, chunk=j, cached=True)
        except Exception: pass
    t0 = time.time(); rows = run_cell(ctrl, path, eps, role)
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_bytes(gzip.compress(json.dumps(dict(meta=meta, rows=rows), sort_keys=True, default=float).encode("utf-8"), mtime=0)); os.replace(tmp, p)
    return dict(ctrl=ctrl, set=st, chunk=j, n=len(rows), falls=sum(not r["survived"] for r in rows), head_clean=sum(r["n"]["head_clean"] for r in rows), wall_s=round(time.time() - t0, 1))

def load(work, ctrl, st, eps, path, plat):
    import fb_t1c as TC
    rows = []
    for j in range(0, len(eps), CHUNK):
        S = json.loads(gzip.decompress((Path(work) / ctrl / f"{st}_{j // CHUNK:03d}.json.gz").read_bytes()).decode("utf-8")); m = S["meta"]
        assert m["seeds"] == [e["seed"] for e in eps[j:j + CHUNK]] and m["onnx_sha256"] == TC.sha(path) and m["platform"] == plat, f"{ctrl}/{st} chunk {j // CHUNK}: meta mismatch"
        rows += S["rows"]
    return rows

def g_boot(ho_p, ho_s, se_p, se_s, n=NBOOT, seed=0):
    """g's bootstrap 90% CI: episodes resampled with replacement within each set, pilot and start rows together."""
    import fb_eq as EQ
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i, k = rng.integers(len(ho_p), size=len(ho_p)), rng.integers(len(se_p), size=len(se_p))
        lh = EQ.rate_u(ho_p[i])[0] - EQ.rate_u(ho_s[i])[0]; ls = EQ.rate_u(se_p[k])[0] - EQ.rate_u(se_s[k])[0]
        if ls != 0: out.append(lh / ls)
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))] if out else None

def ratio(a, b):
    """R(a vs b): exp of the log head-clean-per-up-minute difference, with the paired delta-method 90% CI (fb_eq's arithmetic)."""
    import fb_eq as EQ
    la, ua = EQ.rate_u(a); lb, ub = EQ.rate_u(b); u = ua - ub; se = u.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
    return dict(R=math.exp(la - lb), ci90=[math.exp(la - lb - EQ.Z90 * se), math.exp(la - lb + EQ.Z90 * se)])

def decide(b):
    if b["proceed"]: return "proceed"
    if b["seen"] and b["survival"]: return "stop and report (seen passes, held-out or g fails; A5.2)"
    return "stop and diagnose (seen or survival fails)"

def run(out, cks=None, work=None, workers=None, dry=False, seeds=None, ckdir=None, log=print):
    import fb_eq as EQ, fb_t1c as TC
    cap = int(os.environ.get("FB_MAX_WORKERS", 4 if dry else os.cpu_count())); workers = workers or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 2); assert workers <= cap, f"at most {cap} workers"
    work = Path(work or Path(out).with_suffix("").as_posix() + "_work"); work.mkdir(parents=True, exist_ok=True); plat = TC.platform(); (work / "platform.json").write_text(json.dumps(plat, indent=1), encoding="utf-8")
    notes = {}
    if cks is None or not all(cks.values()):
        if dry: d = Path(ckdir); cks = dict(pilot=d / "ckpt_B.pt", pilot_blind=d / "ckpt_blind_twin.pt"); notes = dict(pilot="dry: M10 ckpt_B", pilot_blind="dry: M10 blind twin")
        else:
            (s, ss, sn), (b, bs, bn) = pilot_ckpt(False), pilot_ckpt(True); cks = dict(pilot=s, pilot_blind=b); notes = dict(pilot=f"{ss} {sn}".strip(), pilot_blind=f"{bs} {bn}".strip())
    E, roles = episodes(dry, seeds); paths = export_all(work, cks); t00 = time.time()
    jobs = [(c, str(paths[c]), st, j // CHUNK, eps[j:j + CHUNK], roles[st], str(work), plat) for c in CTRLS for st, eps in E.items() for j in range(0, len(eps), CHUNK)]
    import multiprocessing as mp
    res = []
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs, chunksize=1): res.append(r); log(json.dumps(dict(r, done=len(res), of=len(jobs), elapsed_s=round(time.time() - t00))), flush=True)
    X = {c: {st: EQ.arrays(load(work, c, st, eps, paths[c], plat))[0] for st, eps in E.items()} for c in CTRLS}
    bars = EQ.loco_bars(X["pilot"]["heldout"], X["start"]["heldout"], X["pilot"]["seen"], X["start"]["seen"])
    rep = dict(plan="M2V2_PLAN_v3 section 4 P3 bars (B5), A5.2, A8.2, A9.1, A9.2", dry=dry, roles=roles, n={st: len(e) for st, e in E.items()}, platform=plat,
               checkpoints={k: str(v) for k, v in cks.items()}, checkpoint_notes=notes, onnx={k: str(v) for k, v in paths.items()}, onnx_sha256={k: TC.sha(v) for k, v in paths.items()},
               bars=bars, decision=decide(bars), g_boot90=g_boot(X["pilot"]["heldout"], X["start"]["heldout"], X["pilot"]["seen"], X["start"]["seen"]),
               R_pilot_vs_blind={st: ratio(X["pilot"][st], X["pilot_blind"][st]) for st in E}, R_blind_vs_start={st: ratio(X["pilot_blind"][st], X["start"][st]) for st in E},
               R_pilot_vs_G={st: ratio(X["pilot"][st], X["G"][st]) for st in E},
               rates={c: {st: dict(survival=float(1 - x[:, 0].mean()), kd=float(x[:, 1].mean()), head_clean_pm=float(x[:, 2].sum() / (x[:, 3].sum() / 60))) for st, x in X[c].items()} for c in CTRLS},
               wall_s=round(time.time() - t00, 1))
    Path(out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8"); return rep

def _rng(s): a, b = (s.split("-") + [s])[:2]; return list(range(int(a), int(b) + 1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--sighted"); ap.add_argument("--blind"); ap.add_argument("--work"); ap.add_argument("--workers", type=int)
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--seeds", default="69000-69011"); ap.add_argument("--ckdir")
    a = ap.parse_args(); cks = dict(pilot=a.sighted, pilot_blind=a.blind) if a.sighted and a.blind else None
    r = run(a.out, cks, a.work, a.workers, a.dry, _rng(a.seeds) if a.dry else None, a.ckdir)
    print(json.dumps(dict(decision=r["decision"], bars=r["bars"], g_boot90=r["g_boot90"], wall_s=r["wall_s"]), default=float), flush=True)
    sys.exit(0 if r["bars"]["proceed"] or a.dry else 3)
