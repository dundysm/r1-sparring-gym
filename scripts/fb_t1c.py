"""P2's T1-c on the pod CPU (M2V2_PLAN_v3 2.2 'T1-a..e', 3.1 T1-c, section 4 P2; A5.3, A7, A8): the replica's G (fb_m6_run.REQ_CONF,
batched rowwise as R-EQ judged it) against the opponent driven by T0 and by T1 on the same episodes, fb_eq.t1c_set's 6144 (the R-EQ
block's shards 0-31, role 'eq'), judged by fb_eq.t1c_gate: one-sided lower 90% bound of log kd(T1) / kd(T0) > log 0.8 and the point in
[0.8, 1.25]. Pass -> T1 rows train; fail -> 100% T0 (plan 2.2), which pod_fb.sh logs.
Platform (A8.2: references on the same platform): bout outcomes diverge chaotically between CPUs, so the T0 arm the ratio is paired
against runs here, on this machine, beside the T1 arm. --t0-arm DIR reuses a stored T0 arm only when its platform.json equals this
machine's fingerprint (platform()) and its episodes and conf are these; anything else is refused, never mixed.
T1: an rsl_rl checkpoint of the tracking task (actor 132 -> 512-256-128 -> 24, read by fb_opponent.TrackerMLP), by default the newest
model_*.pt of the R1-Track-Spar-Reangle run (run name 't1' under logs/rsl_rl/r1_tracking_spar, pod_fb.sh P2). The T1 arm swaps only the
driver's network (motion, aim, placement, the account unchanged; fb_replica.opp_modules wrapped, the conf carries the tracker's sha256 so
the batch cache never mixes arms).
Work: <work>/<arm>/<chunk>.json.gz (64 bouts, gzip mtime 0, atomic), resumable, a spawn pool of --workers processes (FB_MAX_WORKERS cap).
--dry: the plumbing on the M10 smoke block (role 'smoke', --seeds, fb_cpu_smoke's episode map) with T1 = T0's checkpoint, so both arms
are bit identical and the ratio is exactly 1 (asserted); no R-EQ seed is simulated.
Usage: python -I scripts/fb_t1c.py --out t1c.json [--workers 12] [--t1 model.pt] [--t0-arm DIR] [--work DIR] [--dry --seeds 69000-69007]"""
import sys, os, json, gzip, time, hashlib, platform as _pf, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np

FMT, CHUNK = "t1c/1", 64
FB_ROOT = Path(os.environ.get("FB_ROOT", "/root")); T1_RUNS = FB_ROOT / "unitree_rl_mjlab" / "logs" / "rsl_rl" / "r1_tracking_spar"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def platform():
    """This machine's fingerprint: CPU model, ISA, OS, and the package versions the bouts depend on."""
    import mujoco, torch
    cpu = _pf.processor() or ""
    try: cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception: pass
    return dict(cpu=cpu, machine=_pf.machine(), system=_pf.system(), python=_pf.python_version(), numpy=np.__version__, mujoco=mujoco.__version__, torch=torch.__version__)

def find_t1(runs=T1_RUNS):
    """The newest model_*.pt of the T1 run (run dirs named *_t1 first, else the newest run dir but t0 and *_t1_loco)."""
    runs = Path(runs); dirs = [d for d in runs.glob("*") if d.is_dir() and d.name != "t0" and not d.name.endswith("t1_loco")]
    named = [d for d in dirs if d.name.endswith("_t1") or d.name == "t1"]; dirs = named or dirs
    if not dirs: raise FileNotFoundError(f"no T1 run under {runs}")
    d = max(dirs, key=lambda x: x.stat().st_mtime); ck = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ck: raise FileNotFoundError(f"no model_*.pt in {d}")
    return ck[-1]

def episodes(dry=False, seeds=None):
    """The T1-c episodes: fb_eq.t1c_set of the R-EQ block (role 'eq'); dry: fb_cpu_smoke's map of the smoke seeds (role 'smoke')."""
    from fb_seeds import check
    if dry:
        import fb_cpu_smoke as SM
        s0 = SM.PLUMB["block"][0]; eps = []
        for s in seeds: e = SM.episode(s, s0); check([s], "smoke", {e["clip"]}); eps.append(_eo_episode(e))
        return eps, "smoke"
    import fb_r0eo as R0, fb_eq as EQ
    allx = R0.episodes(); keys = [(e["seed"], e["clip"], e["distance"]) for e in allx]; ix = EQ.t1c_set(keys); eps = [allx[i] for i in ix]
    assert len(eps) == EQ.T1C_N; check([e["seed"] for e in eps], "eq", {e["clip"] for e in eps}); return eps, "eq"

def _eo_episode(e):
    """An eval-format episode (t0 on the clean grid, delay, env/hit seeds) for a smoke seed: eval_opponent.episodes on its (clip, distance)."""
    import eval_opponent as EO, fb_r0eo as R0
    EO.TIERS["fb_t1c_dry"] = dict(EO.TIERS["hard"], seed=e["seed"], distances=[e["distance"]]); g = R0.good_starts((e["clip"],), (e["distance"],))
    out = EO.episodes([e["clip"]], 1, "fb_t1c_dry", g, hits_on=False, seconds=20.0); assert out[0]["seed"] == e["seed"]; return out[0]

def with_tracker(pt):
    """fb_replica.opp_modules whose driver runs the checkpoint `pt` (TrackerMLP in the conf's dtype) in place of T0."""
    import fb_replica as RP, fb_opponent as FO
    base = RP.opp_modules
    def om(n, dt, c):
        M, drv, aim, pol = base(n, dt, c); assert c.get("tracker") == sha(pt), "conf / tracker mismatch"
        drv.mlp = FO.TrackerMLP(pt, dt); return M, drv, aim, pol
    return om

def _job(args):
    """One chunk of one arm in a worker: replica G rows (fb_r0eo format) on the chunk's episodes."""
    arm, j, eps, pt, conf, work, role = args
    import torch, fb_replica as RP, fb_m6_run as M6
    torch.set_num_threads(1); p = Path(work) / arm / f"{j:04d}.json.gz"
    meta = dict(fmt=FMT, arm=arm, chunk=j, seeds=[e["seed"] for e in eps], conf=conf, tracker=sha(pt), platform=platform())
    if p.exists():
        try:
            S = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
            if S["meta"] == meta: return dict(arm=arm, chunk=j, cached=True)
        except Exception: pass
    c = RP.conf(**dict(conf, tracker=sha(pt))); RP.opp_modules = with_tracker(pt); b, rw = M6.BATCH, M6.ROWWISE
    t0 = time.time(); rows = RP.run("G", eps, c, batch=min(b, len(eps)), rowwise=rw)
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_bytes(gzip.compress(json.dumps(dict(meta=meta, rows=rows), sort_keys=True, default=float).encode("utf-8"), mtime=0)); os.replace(tmp, p)
    return dict(arm=arm, chunk=j, n=len(rows), kd=sum(r["kd"] for r in rows), wall_s=round(time.time() - t0, 1))

def load_arm(work, arm, eps, conf, pt, plat):
    """Rows of an arm in episode order, every chunk's meta asserted (seeds, conf, tracker, platform)."""
    rows = []
    for j in range(0, len(eps), CHUNK):
        S = json.loads(gzip.decompress((Path(work) / arm / f"{j // CHUNK:04d}.json.gz").read_bytes()).decode("utf-8")); m = S["meta"]
        assert m["seeds"] == [e["seed"] for e in eps[j:j + CHUNK]] and m["conf"] == conf and m["tracker"] == sha(pt) and m["platform"] == plat, f"{arm} chunk {j // CHUNK}: meta mismatch"
        rows += S["rows"]
    return rows

def run(out, t1=None, t0_arm=None, work=None, workers=12, dry=False, seeds=None, log=print):
    import fb_m6_run as M6, fb_eq as EQ, fb_opponent as FO
    cap = int(os.environ.get("FB_MAX_WORKERS", 4 if dry else os.cpu_count())); assert workers <= cap, f"at most {cap} worker processes (FB_MAX_WORKERS)"
    plat = platform()
    if t0_arm:   # A8.2: a stored T0 arm only from this platform, these episodes and this conf
        P = json.loads((Path(t0_arm) / "platform.json").read_text(encoding="utf-8")) if (Path(t0_arm) / "platform.json").exists() else None
        assert P == plat, f"--t0-arm {t0_arm}: platform {P} != this machine's {plat} (A8.2: the T0 arm runs here)"
        work_t0 = Path(t0_arm)
    t1 = Path(t1) if t1 else (Path(FO.TRACKER_PT) if dry else find_t1()); t0 = Path(FO.TRACKER_PT); eps, role = episodes(dry, seeds); arms = {"t0": t0, "t1": t1}
    work = Path(work or Path(out).with_suffix("").as_posix() + "_work"); conf = {k: list(v) if isinstance(v, tuple) else v for k, v in M6.REQ_CONF.items()}
    if not t0_arm: work_t0 = work
    work.mkdir(parents=True, exist_ok=True); (work / "platform.json").write_text(json.dumps(plat, indent=1), encoding="utf-8")
    jobs = [(a, j // CHUNK, eps[j:j + CHUNK], str(pt), conf, str(work_t0 if a == "t0" else work), role) for a, pt in arms.items() for j in range(0, len(eps), CHUNK)]
    t00 = time.time(); res = []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs, chunksize=1): res.append(r); log(json.dumps(dict(r, done=len(res), of=len(jobs), elapsed_s=round(time.time() - t00))), flush=True)
    R = {a: load_arm(work_t0 if a == "t0" else work, a, eps, conf, pt, plat) for a, pt in arms.items()}
    key = lambda r: (r["seed"], r["clip"], r["distance"], r["t0"], r["k"], r["act_delay"]); assert [key(r) for r in R["t0"]] == [key(r) for r in R["t1"]], "arms not on the same episodes"
    X = {a: EQ.arrays(R[a])[0] for a in R}; g = EQ.t1c_gate(X["t0"], X["t1"]) if X["t0"][:, 1].sum() > 0 else dict(est=None, pass_=False, reason="no T0 knockdowns")
    rep = dict(plan="M2V2_PLAN_v3 3.1 T1-c (P2), A5.3 shards 0-31, A8.2 same platform", dry=dry, role=role, n=len(eps), seeds=[eps[0]["seed"], eps[-1]["seed"]], conf=conf, platform=plat,
               t0=str(t0), t0_sha256=sha(t0), t1=str(t1), t1_sha256=sha(t1), t0_arm=str(work_t0), kd=dict(t0=int(X["t0"][:, 1].sum()), t1=int(X["t1"][:, 1].sum())),
               falls=dict(t0=int(X["t0"][:, 0].sum()), t1=int(X["t1"][:, 0].sum())), gate=g, pass_=bool(g.get("pass_")), wall_s=round(time.time() - t00, 1))
    if dry: rep["arms_identical"] = bool(np.array_equal(X["t0"], X["t1"]) and sha(t0) == sha(t1))
    Path(out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8"); return rep

def _rng(s): a, b = (s.split("-") + [s])[:2]; return list(range(int(a), int(b) + 1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--t1"); ap.add_argument("--t0-arm"); ap.add_argument("--work"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--seeds", default="69000-69007")
    a = ap.parse_args(); r = run(a.out, a.t1, a.t0_arm, a.work, a.workers, a.dry, _rng(a.seeds) if a.dry else None)
    print(json.dumps({k: v for k, v in r.items() if k not in ("platform", "conf")}, default=float), flush=True)
    sys.exit(0 if (r["pass_"] or a.dry and r.get("arms_identical")) else 1)
