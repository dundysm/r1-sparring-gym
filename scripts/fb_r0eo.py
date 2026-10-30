"""M6 R0-EO reference (M2V2_PLAN_v3 3 M6, 3.1; A4.4): the eval's own bout for G, H(A3s)/M3s and the full run-3 A3 on the R-EQ
episodes 70000-82287 (fb_seeds role 'eq', 12288 bouts per controller), the EO side of every 3.1 gate, of P1-R4g, of T1-c (70000-76143)
and the T-opp baseline.
Bout: eval_opponent.run unchanged, as eval_opponent.main runs a hard_r2 cell with --aim: R1HitEnv + the tracker opponent on its motors,
aim standoff+armwarp_vz (results/opponent_tracker/aim.json, its lag_s for the library windows), 20 s, DR on, obs delay k and act_delay as
drawn, no pushes, keypoints truth, meter (HitMeter) and falls (opponent fall log, ContactTail, both-up intervals) on, rev_hits off,
threads 1. Controllers as in R4-pre (r4pre/r4pre_eval.py): G = guard_fw_r, H = P1's p1_eval.make_ctrl('guard_fw_r+a3s', 'm3s') with
its scale / clip set from the env's (action_scale(cfg), action_clip), A3 = the run-3 A3 ONNX; cfg = controller_cfg + opponent policy.
Episodes: R4-pre's construction (r4pre_eval.episodes): EO.episodes(train clips, 1024 reps, tier hard with seed 70000, hits off, 20 s)
over product(clips, distances 0.40-0.55, reps): seed 70000 + k, k = (clip * 4 + distance) * 1024 + rep; clean starts fb_motion.npz's
(= R4-pre's good_*.npz = the eval's clean_starts under G's cfg, M1 (3)). Paired: every controller runs the identical list.
Parallelism: R4-pre split one list i % W over W spawn processes; here a task is one (controller, shard), shard j = reps 16j..16j+15 of
all 12 (clip, distance) strata (192 bouts, every shard balanced), on a spawn pool of W processes, shard-major over controllers; a
shard runs as EO.run calls of CHUNK bouts (memory). A row depends on its episode only (checked: shard 0 run as one 192-bout call
equals a direct EO.run of 70000-70015 bit for bit, and equals the CHUNK-bout calls byte for byte).
Read-only probe: R1HitEnv subclassed in the worker (as fb_record / fb_m5_check do) reads act_delay after reset and the EO:86 opponent
down test after every step (non-sticky), for the knockdown edges; asserted equal to EO's own opp_fall and both_up_iv on every bout.
Per bout: seed, clip, distance, rep, t0, k, act_delay, env/hit seeds, fell_s, survived, secs / up_min, kd (= EO's opp_fell, the R4-pre
knockdown), kd_t, kd_edges (every rising edge, s), down_iv ((t_a, t_b] s), kd_n, opp_up_end, fall_after_kd, kd_contact (an R1 contact
in the 0.5 s before the first flag: T-opp), n / n_both_up / both_up_s (HitMeter), and eo = the EO.run row verbatim.
Output: <out>/<ctrl>/shard_NNN.json.gz (gzip mtime 0, written to a temp file and renamed), <out>/manifest.json (refuses a mismatching
resume), <out>/run.pid. Resumable: finished, valid shards are skipped. <out>/STOP stops after the running tasks.
Usage: python -I scripts/fb_r0eo.py run --out DIR [--workers 4] [--ctrls G,H,A3] [--shards 0-63] [--p1 p1_eval.py]
       python -I scripts/fb_r0eo.py check --out DIR --report r0e_check.json [--workers 2]   (shard 0 + direct EO.run on 70000-70015)
       python -I scripts/fb_r0eo.py status --out DIR;  load(out) -> {ctrl: rows in seed order} for scripts/fb_eq.py"""
import sys, os, json, gzip, time, hashlib, argparse, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

FMT = "r0eo/1"
BLOCK = (70000, 82287); REPS = 1024; SHARD_REPS = 16; NSHARD = REPS // SHARD_REPS
CHUNK = 32   # bouts per EO.run call: each live env commits ~40-60 MB (MjData arena, keypoint twin); 192 per call x 4 workers committed ~32 GB
TRAIN = ("kfab_200", "kfab_201", "kfab_203"); HARD = (0.4, 0.45, 0.5, 0.55); AIM = "standoff+armwarp_vz"; SECONDS = 20.0
CTRLS = {"G": ("guard_fw_r", "plain", "results/policy_mjlab_guard_footwork_r/policy.onnx"),
         "H": ("guard_fw_r+a3s", "m3s", None),                                   # P1's M3sHybrid (G legs, A3s arms)
         "A3": ("a3", "plain", "results/policy_mjlab_spar3_a3/policy.onnx")}
SCR = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2")
P1 = SCR / "gap" / "run4pre" / "P1" / "p1_eval.py"; R4PRE = SCR / "fullbody" / "r4pre"
MOTION = ROOT / "gpu" / "motions" / "fb" / "fb_motion.npz"

def sha256(b): return hashlib.sha256(b).hexdigest()
def load_p1(path):
    spec = importlib.util.spec_from_file_location("p1_eval", str(path)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

# ---- episodes (R4-pre's construction)
def good_starts(clips=TRAIN, dists=HARD):
    with np.load(MOTION) as z: cl, D, clean = z["clips"].tolist(), z["distances"], z["clean"]
    return {(c, d): clean[cl.index(c), int(np.flatnonzero(np.isclose(D, d))[0])] for c in clips for d in dists}

def episodes(seed0=BLOCK[0], reps=REPS, dists=HARD, clips=TRAIN):
    """r4pre_eval.episodes(seed0, reps, dists): EO.episodes over product(clips, dists, reps), tier hard with seed seed0, hits off."""
    import eval_opponent as EO
    EO.TIERS["r0eo"] = dict(EO.TIERS["hard"], seed=seed0, distances=[float(d) for d in dists])
    eps = EO.episodes(list(clips), reps, "r0eo", good_starts(clips, dists), hits_on=False, seconds=SECONDS)
    assert [e["seed"] for e in eps] == list(range(seed0, seed0 + len(eps))) and {e["clip"] for e in eps} <= set(TRAIN); return eps

def shard_eps(eps, j):
    """Shard j: reps SHARD_REPS*j .. +SHARD_REPS-1 of every (clip, distance) stratum, in seed order."""
    return [e for e in eps if SHARD_REPS * j <= e["rep"] < SHARD_REPS * (j + 1)]

def eps_sha(eps): return sha256(json.dumps(eps, sort_keys=True).encode())

# ---- controllers and the read-only probe (worker side)
_CELLS, _ENVS = {}, []
def cell(name, p1=P1):
    """(controller, cfg, model, lag) of a controller, cached per process."""
    if name in _CELLS: return _CELLS[name]
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import build_spec, action_scale
    spec, variant, path = CTRLS[name]
    c = load_controller(f"{spec}={ROOT / path}") if path else load_p1(p1).make_ctrl(spec, variant)
    cfg = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": AIM}
    if variant == "m3s": c.set_scale(action_scale(cfg), float(cfg.get("action_clip", 4.0)))
    aim = json.load(open(ROOT / "results" / "opponent_tracker" / "aim.json")); assert aim["operator"] == AIM, aim["operator"]
    _CELLS[name] = (c, cfg, build_spec(opponent=True, cfg=cfg).compile(), float(aim.get("lag_s", 0.0))); return _CELLS[name]

_PROBE = {}
def probe_cls(EO):
    """A subclass of EO.R1HitEnv (run_cell installs it for one EO.run, then restores the base): per env, act_delay after reset and
    the opponent-down step runs (the EO:86 test after each step, non-sticky)."""
    base = EO.R1HitEnv
    if base not in _PROBE:
        class ProbeEnv(base):
            _fb_probe = True
            def __init__(self, *a, **k): super().__init__(*a, **k); _ENVS.append(self)
            def reset(self, *a, **k):
                o = super().reset(*a, **k); self._fb = dict(act_delay=int(self.act_delay), dt=float(self.dt), down=[]); return o
            def step(self, a):
                out = super().step(a); d, b = self.d, min(self.opp.bodies)
                if bool(d.xmat[b][8] < EO.OPP_TILT) or bool(d.xpos[b][2] < EO.OPP_Z):
                    r = self._fb["down"]
                    if r and r[-1][1] == self.t - 1: r[-1][1] = self.t
                    else: r.append([self.t, self.t])
                return out
        _PROBE[base] = ProbeEnv
    return _PROBE[base]

def bout_row(ep, eo, fb, T):
    """The stored row; asserts the probe's down runs against EO's own opp_fell, opp_fall.t_s and both_up_iv (steps 1..n partitioned)."""
    dt, runs, fell = fb["dt"], fb["down"], eo["fell_s"]; n = int(round(fell / dt)) if fell is not None else int(T / dt)
    up = {k for a, b in eo["both_up_iv"] for k in range(int(round(a / dt)) + 1, int(round(b / dt)) + 1)}; dn = {k for a, b in runs for k in range(a, b + 1)}
    assert up | dn == set(range(1, n + 1)) and not up & dn, f"{ep['seed']}: down runs disagree with both_up_iv"
    assert bool(runs) == bool(eo["opp_fell"]) and (not runs or round(runs[0][0] * dt, 3) == eo["opp_fall"]["t_s"]), f"{ep['seed']}: first down run != opp_fall"
    secs = fell if fell is not None else T; hm = eo["hit_meter"]; kd = bool(runs)
    return dict(seed=ep["seed"], clip=ep["clip"], distance=ep["distance"], rep=ep["rep"], t0=ep["t0"], k=ep["delay"], act_delay=fb["act_delay"],
                env_seed=ep["env_seed"], hit_seed=ep["hit_seed"], fell_s=fell, survived=fell is None, secs=secs, up_min=secs / 60, kd=kd,
                kd_t=eo["opp_fall"]["t_s"] if kd else None, kd_edges=[round(a * dt, 3) for a, _ in runs], down_iv=[[round((a - 1) * dt, 3), round(b * dt, 3)] for a, b in runs],
                kd_n=len(runs), opp_up_end=not runs or runs[-1][1] < n, fall_after_kd=kd and fell is not None,
                kd_contact=eo["opp_fall"]["r1_contact_0p5s"]["contact"] if kd else None, n=hm["n"], n_both_up=hm["n_both_up"], both_up_s=eo["both_up_s"], eo=eo)

def run_cell(name, eps, seconds=SECONDS, p1=P1):
    """EO.run (unchanged, probe installed) of one controller on eps -> stored rows, in eps order."""
    import eval_opponent as EO
    from fb_seeds import check
    check([e["seed"] for e in eps], "eq", {e["clip"] for e in eps})
    c, cfg, m, lag = cell(name, p1); base = EO.R1HitEnv; assert not getattr(base, "_fb_probe", False); EO.R1HitEnv = probe_cls(EO); _ENVS.clear()
    try:
        eo = EO.run(c, eps, m, cfg, seconds, threads=1, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")
        assert len(_ENVS) == len(eps); rows = [bout_row(ep, r, e._fb, seconds) for ep, r, e in zip(eps, eo, _ENVS)]
    finally: EO.R1HitEnv = base; _ENVS.clear()
    return rows

def direct(name, eps, seconds=SECONDS, p1=P1, threads=1):
    """A plain EO.run (no probe) of the same cell: the reference of the bit-identity check."""
    import eval_opponent as EO
    from fb_seeds import check
    check([e["seed"] for e in eps], "eq", {e["clip"] for e in eps}); assert not getattr(EO.R1HitEnv, "_fb_probe", False), "probe installed"
    c, cfg, m, lag = cell(name, p1)
    return EO.run(c, eps, m, cfg, seconds, threads=threads, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")

# ---- shards
def shard_path(out, name, j): return Path(out) / name / f"shard_{j:03d}.json.gz"
def shard_meta(name, j, eps, seconds): return dict(fmt=FMT, ctrl=name, spec=list(CTRLS[name][:2]), shard=j, seconds=seconds, seeds=[e["seed"] for e in eps])

def read_shard(p):
    return json.loads(gzip.decompress(Path(p).read_bytes()).decode("utf-8"))

def shard_ok(out, name, j, eps, seconds):
    p = shard_path(out, name, j)
    if not p.exists(): return False
    try: S = read_shard(p)
    except Exception: return False
    return S.get("meta") == shard_meta(name, j, eps, seconds) and [r["seed"] for r in S.get("rows", [])] == S["meta"]["seeds"]

def write_atomic(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def work(args):
    name, j, out, seconds, p1 = args
    if (Path(out) / "STOP").exists(): return dict(ctrl=name, shard=j, skipped="STOP")
    eps = shard_eps(episodes(), j); t0 = time.time(); rows = [r for i in range(0, len(eps), CHUNK) for r in run_cell(name, eps[i:i + CHUNK], seconds, p1)]
    S = dict(meta=shard_meta(name, j, eps, seconds), rows=rows)
    write_atomic(shard_path(out, name, j), gzip.compress(json.dumps(S, sort_keys=True).encode("utf-8"), mtime=0))
    return dict(ctrl=name, shard=j, n=len(rows), falls=sum(not r["survived"] for r in rows), kd=sum(r["kd"] for r in rows),
                head_clean=sum(r["n"]["head_clean"] for r in rows), wall_s=round(time.time() - t0, 1), pid=os.getpid())

def manifest(eps, ctrls, seconds, p1):
    import mujoco, eval_opponent as EO
    c = {n: cell(n, p1) for n in ctrls}
    return dict(fmt=FMT, plan="M2V2_PLAN_v3 3 M6 R0-EO; A4.4", block=list(BLOCK), reps=REPS, shard_reps=SHARD_REPS, n_shards=NSHARD, clips=list(TRAIN),
                distances=list(HARD), seconds=seconds, aim=AIM, lag_s=next(iter(c.values()))[3], ctrls={n: list(CTRLS[n][:2]) for n in ctrls},
                cfg_sha={n: sha256(json.dumps(x[1], sort_keys=True, default=str).encode())[:16] for n, x in c.items()},
                episodes_sha256=eps_sha(eps), fb_motion_sha256=sha256(MOTION.read_bytes()), p1=str(p1), p1_sha256=sha256(Path(p1).read_bytes()),
                eo_run="eval_opponent.run(dr=True, meter=True, falls=True, rev=False, keypoints='truth', threads=1)", chunk=CHUNK, mujoco=mujoco.__version__,
                provenance=EO.provenance(next(iter(c.values()))[2], next(iter(c.values()))[1]))

MKEYS = ("fmt", "block", "reps", "shard_reps", "clips", "distances", "seconds", "aim", "lag_s", "cfg_sha", "episodes_sha256", "fb_motion_sha256", "p1_sha256")

def run(out, workers=4, ctrls=tuple(CTRLS), shards=range(NSHARD), seconds=SECONDS, p1=P1, log=print):
    """Every (controller, shard) not finished yet, on a spawn pool of `workers` processes; returns the task summaries."""
    assert workers <= 4, "at most 4 worker processes"
    out = Path(out); out.mkdir(parents=True, exist_ok=True); eps = episodes(); (out / "run.pid").write_text(str(os.getpid()))
    M = manifest(eps, ctrls, seconds, p1); mp_ = out / "manifest.json"
    if mp_.exists():
        old = json.loads(mp_.read_text(encoding="utf-8")); bad = [k for k in MKEYS if old.get(k) != M.get(k) and not (k == "cfg_sha" and all(old[k].get(n) == v for n, v in M[k].items()))]
        assert not bad, f"{mp_} was written for another configuration: {bad}"
        M["ctrls"] = {**old.get("ctrls", {}), **M["ctrls"]}; M["cfg_sha"] = {**old.get("cfg_sha", {}), **M["cfg_sha"]}; M["runs"] = old.get("runs", [])
    M.setdefault("runs", []).append(dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid(), workers=workers, ctrls=list(ctrls), shards=[min(shards), max(shards)]))
    mp_.write_text(json.dumps(M, indent=1), encoding="utf-8")
    todo = [(n, j) for j in shards for n in ctrls if not shard_ok(out, n, j, shard_eps(eps, j), seconds)]
    log(json.dumps(dict(event="start", out=str(out), pid=os.getpid(), workers=workers, todo=len(todo), of=len(ctrls) * len(shards))), flush=True)
    res, t0 = [], time.time()
    if todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            for r in pool.imap_unordered(work, [(n, j, str(out), seconds, str(p1)) for n, j in todo], chunksize=1):
                res.append(r); el = time.time() - t0; done = sum("n" in x for x in res)
                log(json.dumps(dict(r, done=len(res), of=len(todo), elapsed_h=round(el / 3600, 2), eta_h=round(el / max(done, 1) * (len(todo) - len(res)) / 3600, 2) if done else None)), flush=True)
    left = [(n, j) for j in shards for n in ctrls if not shard_ok(out, n, j, shard_eps(eps, j), seconds)]
    log(json.dumps(dict(event="end", left=len(left), wall_h=round((time.time() - t0) / 3600, 2))), flush=True); return res

def load(out, ctrls=tuple(CTRLS), shards=range(NSHARD)):
    """{ctrl: rows in seed order} over the given shards; asserts every shard present and the controllers paired on the same episodes."""
    R = {}
    for n in ctrls:
        rows = []
        for j in shards:
            S = read_shard(shard_path(out, n, j)); assert S["meta"]["fmt"] == FMT and S["meta"]["ctrl"] == n and S["meta"]["shard"] == j; rows += S["rows"]
        R[n] = sorted(rows, key=lambda r: r["seed"])
    key = lambda r: (r["seed"], r["clip"], r["distance"], r["t0"], r["k"], r["act_delay"])
    ks = [[key(r) for r in R[n]] for n in ctrls]; assert all(k == ks[0] for k in ks), "controllers not paired"; return R

# ---- check (R0E: construction = R4-pre's, rows = a direct EO.run bit for bit)
def _direct_task(args): name, seeds, p1, threads = args; eps = [e for e in episodes() if e["seed"] in set(seeds)]; return name, direct(name, eps, SECONDS, p1, threads)

def _actdelay_task(args):
    """act_delay of each episode from a fresh R1HitEnv reset (DR draw only, no step), independent of the probe."""
    name, eps, p1 = args
    import copy
    from r1gym.env import R1HitEnv, _DR_NOMINAL
    from r1gym.opponent import FightReplay
    c, cfg, m, _ = cell(name, p1); mm = copy.deepcopy(m); reps, out = {}, []
    for ep in eps:
        k = (ep["clip"], ep["distance"]); reps[k] = reps.get(k) or FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg)
        e = R1HitEnv(seed=ep["env_seed"], cfg=cfg, model=mm, episode_s=SECONDS, max_impulse=0, train=False, dr=True)
        e.opp = reps[k]; e.opp_t0 = ep["t0"]; e.reset(hits=[], delay=ep["delay"]); out.append(int(e.act_delay))
    _DR_NOMINAL.pop(id(mm), None); return out

def check(out, report, workers=2, p1=P1, r4pre=R4PRE):
    import pickle, multiprocessing as mp
    from fb_seeds import check as seed_check
    assert workers <= 2, "the check uses at most 2 worker processes"
    R = {}; eps = episodes(); seeds = [e["seed"] for e in eps]
    # (1) the block: seeds, clips, role, shards
    seed_check(seeds, "eq", {e["clip"] for e in eps}); sh = [shard_eps(eps, j) for j in range(NSHARD)]
    R["block"] = dict(n=len(eps), seeds=[seeds[0], seeds[-1]], exact=seeds == list(range(BLOCK[0], BLOCK[1] + 1)), role_eq="ok",
                      shards_partition=sorted(e["seed"] for s in sh for e in s) == seeds, shard_sizes=sorted({len(s) for s in sh}),
                      shards_balanced=all(len({(e["clip"], e["distance"]) for e in s}) == 12 and all(sum(e["clip"] == c and e["distance"] == d for e in s) == SHARD_REPS for c in TRAIN for d in HARD) for s in sh),
                      k_counts={int(v): int(sum(e["delay"] == v for e in eps)) for v in (0, 1, 2)}, episodes_sha256=eps_sha(eps))
    # (2) R4-pre's construction: its own episodes() (seed asserts widened to this block only) = ours; its runs' keys = ours on its blocks
    spec = importlib.util.spec_from_file_location("r4pre_eval", str(Path(r4pre) / "r4pre_eval.py")); RE = importlib.util.module_from_spec(spec); spec.loader.exec_module(RE)
    RE.ALLOWED = [BLOCK]; theirs = RE.episodes(BLOCK[0], REPS, "hard")
    with np.load(Path(r4pre) / "good_kfab_200_kfab_201_kfab_203.npz") as z: masks_eq = all(np.array_equal(z[f"{c}@{d}"], v) for (c, d), v in good_starts().items())
    key = lambda r: (r["seed"], r["clip"], r["distance"], r["t0"], r["delay"])
    blk = {}
    for tag, s0, reps, ds in (("main", 67000, 64, HARD), ("close", 68000, 128, (0.4, 0.45))):
        ours = [key(e) for e in episodes(s0, reps, ds)]
        blk[tag] = {c: ours == [key(r) for r in pickle.load(open(Path(r4pre) / "runs" / f"{c}_{tag}.pkl", "rb"))["rows"]] for c in ("G", "H", "A3")}
    R["r4pre_construction"] = dict(r4pre_episodes_equal=json.dumps(theirs, sort_keys=True) == json.dumps(eps, sort_keys=True), clean_masks_equal=masks_eq, r4pre_run_keys_equal=blk)
    # (3) shard 0 through the runner (production path, 192-bout shards), then a plain EO.run of 70000-70015 in fresh processes
    t0 = time.time(); run(out, workers=workers, shards=range(1), p1=p1); R["shard0_wall_s"] = round(time.time() - t0, 1)
    rows = {n: read_shard(shard_path(out, n, 0))["rows"] for n in CTRLS}; sub = list(range(70000, 70016))
    with mp.get_context("spawn").Pool(workers) as pool:
        D = dict(pool.map(_direct_task, [(n, sub, str(p1), 2) for n in CTRLS], chunksize=1))
        AD = pool.map(_actdelay_task, [("G", sh[0][i::workers], str(p1)) for i in range(workers)], chunksize=1)
    ad = [None] * len(sh[0])
    for i, a in enumerate(AD): ad[i::workers] = a
    J = lambda x: json.dumps(x, sort_keys=True)
    R["direct_identity"] = {}
    for n in CTRLS:
        mine = [r for r in rows[n] if r["seed"] in set(sub)]; assert [r["seed"] for r in mine] == sub
        R["direct_identity"][n] = dict(bouts=len(sub), rows_equal=sum(J(a["eo"]) == J(b) for a, b in zip(mine, D[n])), bit_identical=all(J(a["eo"]) == J(b) for a, b in zip(mine, D[n])),
                                       fell=sum(b["fell_s"] is not None for b in D[n]), opp_fell=sum(bool(b["opp_fell"]) for b in D[n]),
                                       head_clean=sum(b["hit_meter"]["n"]["head_clean"] for b in D[n]), act_delay_equal_across_ctrls=[r["act_delay"] for r in mine] == [r["act_delay"] for r in rows["G"] if r["seed"] in set(sub)])
    R["act_delay_vs_fresh_reset"] = dict(bouts=len(ad), equal=[r["act_delay"] for r in rows["G"]] == ad, ones=int(sum(ad)))
    R["shard0"] = {n: dict(n=len(rows[n]), falls=sum(not r["survived"] for r in rows[n]), kd=sum(r["kd"] for r in rows[n]), kd_edges=sum(r["kd_n"] for r in rows[n]),
                           fall_after_kd=sum(r["fall_after_kd"] for r in rows[n]), head_clean=sum(r["n"]["head_clean"] for r in rows[n]),
                           up_min=round(sum(r["up_min"] for r in rows[n]), 3), act_delay1=sum(r["act_delay"] for r in rows[n])) for n in CTRLS}
    R["pass"] = bool(R["block"]["exact"] and R["block"]["shards_partition"] and R["block"]["shards_balanced"] and R["r4pre_construction"]["r4pre_episodes_equal"]
                     and R["r4pre_construction"]["clean_masks_equal"] and all(all(v.values()) for v in blk.values())
                     and all(v["bit_identical"] and v["act_delay_equal_across_ctrls"] for v in R["direct_identity"].values()) and R["act_delay_vs_fresh_reset"]["equal"])
    Path(report).parent.mkdir(parents=True, exist_ok=True); Path(report).write_text(json.dumps(R, indent=1), encoding="utf-8"); print(json.dumps(R, indent=1), flush=True); return R

def status(out):
    eps = episodes(); done = {n: [j for j in range(NSHARD) if shard_ok(out, n, j, shard_eps(eps, j), SECONDS)] for n in CTRLS}
    print(json.dumps({n: dict(done=len(v), of=NSHARD, missing=[j for j in range(NSHARD) if j not in v][:16]) for n, v in done.items()})); return done

def _rng(s): a, b = (s.split("-") + [s])[:2]; return range(int(a), int(b) + 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "check", "status"]); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None); ap.add_argument("--ctrls", default="G,H,A3"); ap.add_argument("--shards", default=f"0-{NSHARD - 1}")
    ap.add_argument("--p1", default=str(P1)); ap.add_argument("--report"); a = ap.parse_args()
    if a.cmd == "run": run(a.out, a.workers or 4, tuple(a.ctrls.split(",")), _rng(a.shards), SECONDS, a.p1)
    elif a.cmd == "check": check(a.out, a.report or str(Path(a.out) / "r0e_check.json"), a.workers or 2, a.p1)
    else: status(a.out)
