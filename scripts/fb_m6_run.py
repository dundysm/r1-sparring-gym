"""M6 driver for the CPU stages after R0-EO (M2V2_PLAN_v3 3 M6, 3.1, 3.2): one resumable runner, fb_replica.py bouts in a spawn pool of at
most 4 processes (2 while R0-EO runs), rows in fb_r0eo's shard format (<out>/<stage>/<tag>/<ctrl>/<chunk>.json.gz, gzip mtime 0, atomic
rename), a manifest that refuses a mismatching resume, and <out>/STOP (stop after the running tasks).
Stages:
  req   R-EQ: the replica's final configuration (REQ_CONF, the 3.2 fallback list's current entry) on the R-EQ block 70000-82287 (role 'eq',
        R0-EO's episodes) for G, H(A3s)/M3s and A3, shard by shard (192 balanced bouts) in the order fb_eq.LOOK_RULE needs them; after each
        look's shards (all controllers) fb_eq.sequential on that look against R0-EO's rows (fb_r0eo.load); stops at the first passing look.
  r2    R2 diagnostics (report, B2): each ladder step alone vs step 0 (EO = R4-pre's rows, r4pre/runs/*_main.pkl), 768 bouts per controller
        on 67000-67767 (role 'r2_diag'): (a) torch float32 modules, (b) hulls 2.1, (c) solver 10/20, (d) the training reset (torch64),
        (e) FullBodyHitAccount scoring (same-state counts vs the HitMeter on every bout, and the distributional statistics).
  t1c   the T1-c T0 arm: replica G (REQ_CONF) vs T0 on fb_eq.T1C_RULE's 6144 episodes; shares the req stage's G shards when the conf is the same.
  topp  T-opp (E6 baseline, no simulation): opponent first falls with no R1 contact in the 0.5 s before (kd_contact False) per bout and per
        up-minute, replica (req rows) and EO (R0-EO rows), per controller.
  r6    R6 (report) on 67000-67767 vs step 0: defender self-collision on (eval config + self_collision), the eval solver vs the chosen one
        (eval config with REQ_CONF's solver; = R2 (c) when that is 10/20).
Batched (A7.1, A7.2): every torch-backend conf runs BATCH bouts in lockstep per process (fb_replica.Batch, ROWWISE: the tracker MLP one row
at a time); the batch is part of each chunk's meta and of the manifest, so a resume never mixes batch configurations. The numpy (eval)
conf runs one bout at a time, as M6 R1 proved it.
--dry: every stage on 8 bouts (60100-60107, role 'identity'), step 0 / the EO side = the eval-mode replica (bit identical to EO.run, M6
R1), looks (4, 8): plumbing only, no gate meaning.
Usage: python -I scripts/fb_m6_run.py {req|r2|t1c|topp|r6|all} --out DIR [--workers 2] [--r0eo DIR] [--conf JSON] [--seconds 20] [--dry]
       python -I scripts/fb_m6_run.py status --out DIR"""
import sys, os, json, gzip, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np

FMT = "m6run/1"
SCR = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody")
R0EO, R4PRE = SCR / "r0eo" / "out", SCR / "r4pre" / "runs"
CTRLS = ("G", "H", "A3"); SECONDS = 20.0
DEPLOYED_VMODE = "site"   # A4.3 as frozen, read literally (fb_m6_check g011: pos within 1 cm of 88.3% of contacts < 99%); 'pos' on force-bearing contacts
REQ_CONF = dict(hulls="mix", solver=(10, 20), backend="torch32", reset="eo", scorer="account", vmode=DEPLOYED_VMODE)   # 3.2's list: first entry
R2_STEPS = dict(a_torch32=dict(backend="torch32"), b_hulls=dict(hulls="mix"), c_solver=dict(solver=(10, 20)), d_reset=dict(backend="torch64", reset="train"),
                e_scorer=dict(scorer="account", vmode=DEPLOYED_VMODE))
BATCH, ROWWISE = 64, True   # A7.1: K = 64 by measurement (fb_m6b_check speed: 1.0-1.2x EO per 20-s bout per worker; 128 no faster, twice the memory); A7.2: rowwise = the
# batch-variant ops (tracker MLP sgemm, facing atan2) one row at a time, the rest batched (fb_m6b_check invariance; M6b_NOTES)
DRY = dict(role="identity", seeds=tuple(range(60100, 60108)), looks=(4, 8))

def write_atomic(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def read(p): return json.loads(gzip.decompress(Path(p).read_bytes()).decode("utf-8"))
def cpath(out, stage, tag, ctrl, chunk): return Path(out) / stage / tag / ctrl / f"{chunk}.json.gz"
def batch_of(conf):
    """(batch, rowwise) a conf runs with: BATCH / ROWWISE for the torch backends, (1, False) for the numpy (eval) conf."""
    import fb_replica as RP
    return (BATCH, ROWWISE) if RP.conf(**conf)["backend"] != "numpy" else (1, False)

def cmeta(stage, tag, ctrl, chunk, conf, seeds, seconds):
    b, rw = batch_of(conf); return dict(fmt=FMT, stage=stage, tag=tag, ctrl=ctrl, chunk=chunk, conf=conf, seeds=list(seeds), seconds=seconds, batch=b, rowwise=rw)

def chunk_ok(out, stage, tag, ctrl, chunk, conf, seeds, seconds):
    p = cpath(out, stage, tag, ctrl, chunk)
    if not p.exists(): return False
    try: S = read(p)
    except Exception: return False
    return S.get("meta") == json.loads(json.dumps(cmeta(stage, tag, ctrl, chunk, conf, seeds, seconds))) and [r["seed"] for r in S["rows"]] == list(seeds)

# ---- episodes per block
_EPS = {}
def block_eps(role, dry=False):
    """{seed: episode} of a role's block (fb_replica.episodes: fb_seeds-checked), cached per process."""
    import fb_r0eo as R0, fb_replica as RP
    if (role, dry) in _EPS: return _EPS[(role, dry)]
    if dry: E = {e["seed"]: e for e in RP.episodes(DRY["role"], DRY["seeds"])}
    else:
        blk = dict(eq=(70000, R0.REPS), r2_diag=(67000, 64))[role]; eps = R0.episodes(blk[0], blk[1], R0.HARD); RP.episodes(role, [e["seed"] for e in eps]); E = {e["seed"]: e for e in eps}
    _EPS[(role, dry)] = E; return E

def shards(eps, reps_per=16):
    """Balanced chunks: shard j = reps reps_per*j .. +reps_per-1 of every (clip, distance) stratum, in seed order (fb_r0eo.shard_eps)."""
    J = {}
    for e in sorted(eps.values(), key=lambda e: e["seed"]): J.setdefault(e["rep"] // reps_per, []).append(e["seed"])
    return J

# ---- tasks
def work(args):
    stage, tag, ctrl, chunk, conf, seeds, seconds, out, p1, role, dry = args
    if (Path(out) / "STOP").exists(): return dict(stage=stage, tag=tag, ctrl=ctrl, chunk=chunk, skipped="STOP")
    import torch, fb_replica as RP
    torch.set_num_threads(1); c = RP.conf(**conf); eps = block_eps(role, dry); t0 = time.time(); b, rw = batch_of(conf); rows = RP.run(ctrl, [eps[s] for s in seeds], c, seconds, p1=p1, batch=b, rowwise=rw)
    write_atomic(cpath(out, stage, tag, ctrl, chunk), gzip.compress(json.dumps(dict(meta=cmeta(stage, tag, ctrl, chunk, conf, seeds, seconds), rows=rows), sort_keys=True).encode("utf-8"), mtime=0))
    return dict(stage=stage, tag=tag, ctrl=ctrl, chunk=chunk, n=len(rows), falls=sum(not r["survived"] for r in rows), kd=sum(r["kd"] for r in rows), wall_s=round(time.time() - t0, 1), pid=os.getpid())

def run_tasks(tasks, out, workers, log=print):
    """Every task whose chunk is not finished, on a spawn pool; returns the summaries."""
    assert workers <= 4, "at most 4 worker processes"
    todo = [t for t in tasks if not chunk_ok(t[7], t[0], t[1], t[2], t[3], t[4], t[5], t[6])]
    log(json.dumps(dict(event="tasks", todo=len(todo), of=len(tasks))), flush=True); res, t0 = [], time.time()
    if not todo: return res
    if workers == 1: it = map(work, todo)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(min(workers, len(todo))); it = pool.imap_unordered(work, todo, chunksize=1)
    for r in it:
        res.append(r); el = time.time() - t0; log(json.dumps(dict(r, done=len(res), of=len(todo), elapsed_h=round(el / 3600, 3))), flush=True)
    if workers > 1: pool.close(); pool.join()
    return res

def load_rows(out, stage, tag, ctrl, chunks):
    rows = []
    for ch in chunks: rows += read(cpath(out, stage, tag, ctrl, ch))["rows"]
    return sorted(rows, key=lambda r: r["seed"])

def conf_tag(conf):
    import fb_replica as RP
    c = RP.conf(**conf); return "_".join(f"{k}-{'x'.join(map(str, v)) if isinstance(v, (list, tuple)) else v}" for k, v in sorted(c.items()) if v != RP.EVAL[k]) or "eval"

def manifest(out, key, val):
    """<out>/manifest.json: one value per key for the life of the out dir (a resume with another conf / rule refuses)."""
    p = Path(out) / "manifest.json"; M = json.loads(p.read_text(encoding="utf-8")) if p.exists() else dict(fmt=FMT, runs=[])
    if key in M: assert M[key] == json.loads(json.dumps(val)), f"{p}: {key} was {M[key]}, now {val}"
    M[key] = val; M["runs"].append(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid(), key=key)); Path(out).mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(M, indent=1), encoding="utf-8")

# ---- the EO side and step 0
def eo_rows(r0eo, chunks_by_shard, dry=False, out=None, workers=2, p1=None, seconds=SECONDS):
    """{ctrl: R0-EO rows} over the shards (fb_r0eo.load); dry: the eval-mode replica on the dry episodes (stage 'eo')."""
    if dry:
        tasks = [("eo", "eval", c, "dry", {}, list(DRY["seeds"]), seconds, str(out), p1, DRY["role"], True) for c in CTRLS]; run_tasks(tasks, out, workers)
        return {c: load_rows(out, "eo", "eval", c, ["dry"]) for c in CTRLS}
    import fb_r0eo as R0
    return R0.load(r0eo, CTRLS, sorted(chunks_by_shard))

def step0(dry=False, out=None, workers=2, p1=None, seconds=SECONDS):
    """R2 / R6 step 0: R4-pre's main-block rows (EO = p1_eval's bout, 67000-67767) as {ctrl: rows} in fb_eq's format; dry: the eval-mode replica."""
    if dry: return eo_rows(None, None, True, out, workers, p1, seconds)
    import pickle
    return {c: sorted(pickle.load(open(R4PRE / f"{c}_main.pkl", "rb"))["rows"], key=lambda r: r["seed"]) for c in CTRLS}

def diag(base, rows, paired=True):
    """A ladder step vs step 0 (report): fb_eq's R-EQ statistics (paired on the same episodes, or P1-R4g's unpaired form)."""
    import fb_eq as Q
    A = {c: Q.arrays(base[c])[0] for c in CTRLS}; B = {c: Q.arrays(rows[c])[0] for c in CTRLS}; act = Q.arrays(rows["G"])[1]
    if paired:
        for c in CTRLS: assert [r["seed"] for r in base[c]] == [r["seed"] for r in rows[c]]
        g = Q.req_gates(A, B, act)
    else: g = Q.r4g_gates(A, B)
    return dict(n=len(act), paired=paired, stats=g, failing=[k for k, v in g.items() if not v["pass_"]], report=Q.report(A, B, act) if paired else None)

def same_state(rows):
    """R2 (e): the account vs the HitMeter on the same bouts: landed / clean per bout and in total."""
    out = {}
    for c, rs in rows.items():
        A = [r["acct"]["n"] for r in rs]; H = [r["hm"]["n"] for r in rs]
        out[c] = dict(bouts=len(rs), **{k: [sum(a[k] for a in A), sum(h[k] for h in H)] for k in ("head_landed", "head_clean", "torso_landed", "torso_clean")},
                      bouts_landed_equal=sum(a["head_landed"] == h["head_landed"] and a["torso_landed"] == h["torso_landed"] for a, h in zip(A, H)))
    return out

# ---- stages
def stage_req(a, conf):
    import fb_eq as Q
    eps = block_eps("eq", a.dry); tag = conf_tag(conf); manifest(a.out, "req_conf", dict(tag=tag, conf=conf))
    if a.dry: J = {"dry": list(DRY["seeds"])}; looks = [(n, ["dry"]) for n in DRY["looks"]]
    else:
        rule = Q.LOOK_RULE; keys = [(s, eps[s]["clip"], eps[s]["distance"]) for s in sorted(eps)]; J = shards(eps)
        sets = Q.look_sets(keys, Q.LOOKS, rule); manifest(a.out, "look_rule", rule); need = [sorted({eps[keys[i][0]]["rep"] // 16 for i in ix}) for ix in sets]; looks = list(zip(Q.LOOKS, [[j for j in nd] for nd in need]))
    res = []
    for i, (n, chunks) in enumerate(looks):
        tasks = [("req", tag, c, f"shard_{j:03d}" if j != "dry" else "dry", conf, J[j], a.seconds, a.out, a.p1, "eq", a.dry) for j in chunks for c in CTRLS]; run_tasks(tasks, a.out, a.workers)
        if (Path(a.out) / "STOP").exists(): return dict(stopped=True, looks=res)
        ch = [t[3] for t in tasks if t[2] == "G"]; rep = {c: load_rows(a.out, "req", tag, c, ch) for c in CTRLS}
        eo = eo_rows(a.r0eo, chunks, a.dry, a.out, a.workers, a.p1, a.seconds)
        if a.dry: eo = {c: [r for r in eo[c] if r["seed"] in set(DRY["seeds"][:n])] for c in CTRLS}; rep = {c: [r for r in rep[c] if r["seed"] in set(DRY["seeds"][:n])] for c in CTRLS}
        sq = Q.sequential(eo, rep, rule=None if a.dry else Q.LOOK_RULE, looks=Q.LOOKS[:i + 1], sets=None if not a.dry else [np.arange(k) for k in DRY["looks"][:i + 1]])
        res.append(dict(look=n, chunks=len(chunks), passed=sq["passed"], passed_at=sq["passed_at"], fail_classes=sq["fail_classes"], last=sq["looks"][-1], report=sq["report"]))
        if sq["passed"]: break
    return dict(conf=conf, tag=tag, looks=res, passed=bool(res and res[-1]["passed"]))

def stage_r2(a, steps=R2_STEPS):
    eps = block_eps("r2_diag", a.dry); J = {"dry": list(DRY["seeds"])} if a.dry else shards(eps); base = step0(a.dry, a.out, a.workers, a.p1, a.seconds); out = {}
    if a.dry: base = {c: [r for r in base[c]] for c in CTRLS}
    for k, conf in steps.items():
        tag = conf_tag(conf); tasks = [("r2", tag, c, f"shard_{j:03d}" if j != "dry" else "dry", conf, J[j], a.seconds, a.out, a.p1, "r2_diag", a.dry) for j in J for c in CTRLS]; run_tasks(tasks, a.out, a.workers)
        if (Path(a.out) / "STOP").exists(): return dict(stopped=True, **out)
        rows = {c: load_rows(a.out, "r2", tag, c, [t[3] for t in tasks if t[2] == c]) for c in CTRLS}
        out[k] = dict(conf=conf, tag=tag, vs_step0=diag(base, rows, paired=conf.get("reset", "eo") == "eo"), **({"same_state": same_state(rows)} if conf.get("scorer") == "account" else {}))
    return out

def stage_t1c(a, conf):
    import fb_eq as Q
    eps = block_eps("eq", a.dry); tag = conf_tag(conf)
    if a.dry: sel = list(DRY["seeds"]); chunks = {"dry": sel}
    else:
        rule = Q.T1C_RULE; keys = [(s, eps[s]["clip"], eps[s]["distance"]) for s in sorted(eps)]; sel = [keys[i][0] for i in Q.t1c_set(keys, rule)]; manifest(a.out, "t1c_rule", rule)
        J = shards(eps); chunks = {f"shard_{j:03d}": J[j] for j in sorted({eps[s]["rep"] // 16 for s in sel})} if rule == "shards" else \
            {f"t1c_{j:03d}": [s for s in J[j] if s in set(sel)] for j in sorted({eps[s]["rep"] // 16 for s in sel})}
    stage = "req" if (a.dry or Q.T1C_RULE == "shards") else "t1c"   # the req stage's G shards are the T0 arm when the conf is the same
    run_tasks([(stage, tag, "G", ch, conf, s, a.seconds, a.out, a.p1, "eq", a.dry) for ch, s in chunks.items()], a.out, a.workers)
    rows = [r for r in load_rows(a.out, stage, tag, "G", list(chunks)) if r["seed"] in set(sel)]; x = Q.arrays(rows)[0]
    return dict(conf=conf, tag=tag, rule=None if a.dry else Q.T1C_RULE, n=len(rows), kd=int(x[:, 1].sum()), kd_rate=float(x[:, 1].mean()), falls=int(x[:, 0].sum()), stored=f"{stage}/{tag}/G")

def stage_topp(a, conf):
    """No simulation: the req rows present (replica) and the R0-EO rows of the same shards."""
    tag = conf_tag(conf); base = Path(a.out) / "req" / tag; out = {}
    if not base.exists(): return dict(missing=str(base))
    chunks = sorted(p.name[:-8] for p in (base / "G").glob("*.json.gz")); chunks = [ch for ch in chunks if all(cpath(a.out, "req", tag, c, ch).exists() for c in CTRLS)]
    rep = {c: load_rows(a.out, "req", tag, c, chunks) for c in CTRLS}; eo = eo_rows(a.r0eo, [int(ch[6:]) for ch in chunks if ch.startswith("shard_")], a.dry, a.out, a.workers, a.p1, a.seconds)
    if a.dry: s = set(r["seed"] for r in rep["G"]); eo = {c: [r for r in eo[c] if r["seed"] in s] for c in CTRLS}
    f = lambda rs: dict(bouts=len(rs), up_min=round(sum(r["up_min"] for r in rs), 3), kd=sum(r["kd"] for r in rs), kd_no_contact=sum(bool(r["kd"]) and r["kd_contact"] is False for r in rs),
                        no_contact_per_bout=round(sum(bool(r["kd"]) and r["kd_contact"] is False for r in rs) / max(len(rs), 1), 5),
                        no_contact_per_up_min=round(sum(bool(r["kd"]) and r["kd_contact"] is False for r in rs) / max(sum(r["up_min"] for r in rs), 1e-9), 5))
    return dict(tag=tag, chunks=len(chunks), replica={c: f(rep[c]) for c in CTRLS}, eo={c: f(eo[c]) for c in CTRLS},
                definition="opponent first falls (EO:86) with no robot-robot contact in the 0.5 s before (ContactTail: kd_contact False)")

def stage_r6(a, conf):
    steps = dict(selfcoll=dict(self_collision=True))
    if tuple(conf["solver"]) != (10, 20): steps["solver_chosen"] = dict(solver=tuple(conf["solver"]))
    out = stage_r2(a, steps)
    if tuple(conf["solver"]) == (10, 20): out["solver_chosen"] = "= R2 (c) c_solver (chosen solver 10/20)"
    return out

def status(out):
    S = {}
    for p in sorted(Path(out).glob("*/*/*/*.json.gz")):
        st, tag, c = p.parts[-4], p.parts[-3], p.parts[-2]; S.setdefault(f"{st}/{tag}", {}).setdefault(c, 0); S[f"{st}/{tag}"][c] += 1
    print(json.dumps(S, indent=1)); return S

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["req", "r2", "t1c", "topp", "r6", "all", "status"]); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--r0eo", default=str(R0EO)); ap.add_argument("--conf", default=None); ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--p1", default=None); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    if a.cmd == "status": status(a.out); return
    conf = dict(REQ_CONF, **(json.loads(a.conf) if a.conf else {})); Path(a.out).mkdir(parents=True, exist_ok=True); manifest(a.out, "batch", dict(batch=BATCH, rowwise=ROWWISE)); (Path(a.out) / "run.pid").write_text(str(os.getpid())); res = {}
    for st in (("req", "t1c", "topp", "r2", "r6") if a.cmd == "all" else (a.cmd,)):
        t0 = time.time(); r = dict(req=lambda: stage_req(a, conf), r2=lambda: stage_r2(a), t1c=lambda: stage_t1c(a, conf), topp=lambda: stage_topp(a, conf), r6=lambda: stage_r6(a, conf))[st]()
        r = dict(r, wall_s=round(time.time() - t0, 1), dry=a.dry, seconds=a.seconds); res[st] = r
        (Path(a.out) / f"result_{st}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8"); print(json.dumps({st: {k: v for k, v in r.items() if k in ("passed", "wall_s", "n", "kd", "replica", "eo")}}, default=str), flush=True)
    return res

if __name__ == "__main__": main()
