"""M6b checks of the batched CPU replica (M2V2_PLAN_v3 A7.1, A7.2; scripts/fb_replica.py Batch):
  speed       s per bout and per 20-s bout per worker on the same machine load, interleaved: the eval's own bout (eval_opponent.run as
              R0-EO runs it: 32-bout calls, threads 1) and the replica at R-EQ's configuration (fb_m6_run.REQ_CONF) at batch K for each
              K given, rowwise and not, on the 64 episodes of 60200-60263 (role 'hit_scoring'; K > 64 repeats them), one controller per worker process.
              EO is timed before and after the replica (the load drifts); the ratio uses their mean.
  invariance  A7.2: rows at batch K vs N = 1 (run_bout) on 64 bouts per controller (60100-60115, role 'identity', and 60200-60247, role
              'hit_scoring'), G, H(A3s)/M3s and A3, REQ_CONF, 20 s; two batched variants: 'batch' (the deployed modules as they are)
              and 'rowwise' (the tracker MLP and the facing's atan2 one row at a time, fb_replica.Batch rowwise). Every differing row is listed with the fields that differ and the
              first control step whose quantities differ, in step order: the defender's action (controller on the delayed obs), the
              defender's joint targets, the opponent's joint targets (AimOp + TrackerDriver), qpos after the physics. Same-state module
              probes inside the batched run name the source: each batched call of the tracker MLP, of TrackerDriver.face (vs its row-wise atan2
              form) and of the controller, and sampled calls
              of AimOp.ik, fb_hits.zone_forces / closing_speeds, fb_obs.proprio83 / kp_state, are recomputed one row at a time on the same
              inputs and compared bit for bit.
Run in the repo: python -I scripts/fb_m6b_check.py {speed|invariance} --out DIR [--k 64] [--ks 32,64,96] [--workers 2] [--ctrls G,H,A3]"""
import sys, os, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, torch

INV_SEEDS = dict(identity=tuple(range(60100, 60116)), hit_scoring=tuple(range(60200, 60248)))   # 16 + 48 = 64 per controller
SPEED_SEEDS = tuple(range(60200, 60264)); CTRLS = ("G", "H", "A3")
J = lambda x: json.dumps(x, sort_keys=True, default=str)

def _eps(role_seeds):
    import fb_replica as RP
    return [e for role, s in role_seeds.items() for e in RP.episodes(role, s)]

# ---- speed
def _speed(args):
    ctrl, ks, seconds = args; torch.set_num_threads(1)
    import fb_replica as RP, fb_m6_run as D, fb_r0eo as R0, eval_opponent as EO
    c = RP.conf(**D.REQ_CONF); base = _eps(dict(hit_scoring=SPEED_SEEDS)); out = []
    def eo():
        cc, cfg, m, lag = R0.cell(ctrl); t0 = time.time()
        rows = [r for i in range(0, len(base), R0.CHUNK) for r in EO.run(cc, base[i:i + R0.CHUNK], m, cfg, seconds, threads=1, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")]
        w = time.time() - t0; ss = sum(r["fell_s"] or seconds for r in rows); return dict(kind="EO", ctrl=ctrl, n=len(rows), wall_s=round(w, 1), s_per_bout=round(w / len(rows), 3), s_per_20s_bout=round(20 * w / ss, 3))
    out.append(eo())
    for k in ks:
        for rw in (True, False):
            eps = (base * (k // len(base) + 1))[:max(k, len(base))]; RP._BATCH.clear(); t0 = time.time(); rows = RP.run(ctrl, eps, c, seconds, batch=k, rowwise=rw); w = time.time() - t0
            B = next(iter(RP._BATCH.values())); ss = sum(r["secs"] for r in rows)
            out.append(dict(kind=f"{'rowwise' if rw else 'batch'}{k}", ctrl=ctrl, n=len(rows), wall_s=round(w, 1), s_per_bout=round(w / len(rows), 3), s_per_20s_bout=round(20 * w / ss, 3),
                            phases_s={a: round(b, 1) for a, b in B.tm.items()}))
            RP._BATCH.clear()
    out.append(eo()); return out

def speed(ks, seconds=20.0, workers=2, ctrls=CTRLS):
    res = sum(_pool(_speed, [(c, list(ks), seconds) for c in ctrls], workers), []); per = {}
    for c in ctrls:
        R = [r for r in res if r["ctrl"] == c]; e = [r for r in R if r["kind"] == "EO"]; eo20 = float(np.mean([r["s_per_20s_bout"] for r in e])); eob = float(np.mean([r["s_per_bout"] for r in e]))
        per[c] = dict(eo_s_per_20s_bout=round(eo20, 3), eo_s_per_bout=round(eob, 3), eo_runs=[r["s_per_20s_bout"] for r in e],
                      batch={r["kind"]: dict(s_per_20s_bout=r["s_per_20s_bout"], s_per_bout=r["s_per_bout"], x_eo_20s=round(r["s_per_20s_bout"] / eo20, 3), x_eo_bout=round(r["s_per_bout"] / eob, 3),
                                             phases_s=r["phases_s"]) for r in R if r["kind"] != "EO"})
    return dict(seconds=seconds, ks=list(ks), workers=workers, per_ctrl=per, runs=res)

# ---- invariance
class _N1Trace:
    """run_bout probe: per step the action, d.ctrl (defender then opponent targets) and qpos."""
    def __init__(self): self.T = []
    def __call__(self, B, kind, a):
        if kind == "reset": self.T.append([])
        elif kind == "step": self.T[-1].append((np.asarray(a, np.float64).copy(), B.d.ctrl.copy(), B.d.qpos.copy()))

class _Probes:
    """Same-state module probes in a Batch: the batched call vs the same call one row at a time, bit for bit."""
    EVERY = dict(mlp=1, face=1, ctrl=1, ik=20, zone_forces=10, closing_speeds=10, proprio83=10, kp_state=10)
    def __init__(self): self.S = {k: dict(calls=0, checked=0, rows=0, rows_diff=0, calls_diff=0, max_abs=0.0) for k in self.EVERY}
    def cmp(self, k, full, rows):
        s = self.S[k]; full = np.asarray(full, np.float64); rows = np.asarray(rows, np.float64); d = full != rows
        if d.ndim > 1: d = d.reshape(len(d), -1).any(1)
        s["checked"] += 1; s["rows"] += len(d); s["rows_diff"] += int(d.sum()); s["calls_diff"] += int(d.any())
        if d.any(): s["max_abs"] = max(s["max_abs"], float(np.nanmax(np.abs(full - rows))))
    def due(self, k): self.S[k]["calls"] += 1; return (self.S[k]["calls"] - 1) % self.EVERY[k] == 0
    def install(self, Bt):
        import fb_obs as OB, fb_hits as HT
        P = self; drv, aim = Bt.drv, Bt.aim; mlp, ik, ctrl = drv.mlp, aim.ik, Bt.ctrl
        def mlp_w(x):
            y = mlp(x)
            if len(x) > 1 and P.due("mlp"): P.cmp("mlp", y.double().numpy(), torch.cat([mlp(x[i:i + 1]) for i in range(len(x))]).double().numpy())
            return y
        def ik_w(rq5, p_t):
            y = ik(rq5, p_t)
            if len(rq5) > 1 and P.due("ik"): P.cmp("ik", y.numpy(), torch.cat([ik(rq5[i:i + 1], p_t[i:i + 1]) for i in range(len(rq5))]).numpy())
            return y
        class Ctrl:
            def __getattr__(s, a): return getattr(ctrl, a)
            def act(s, o):
                y = ctrl.act(o)
                if len(o) > 1 and P.due("ctrl"): P.cmp("ctrl", y, np.concatenate([ctrl.act(o[i:i + 1]) for i in range(len(o))]))
                return y
        face = drv.face
        def face_w(target, pel_xy):
            from types import SimpleNamespace
            import fb_replica as RP
            ns = SimpleNamespace(yaw=drv.yaw.clone(), yaw_set=drv.yaw_set.clone(), dt=drv.dt); y = face(target, pel_xy)
            if len(target) > 1 and P.due("face"): P.cmp("face", y.double().numpy(), RP.rowwise_face(ns)(target, pel_xy).double().numpy())
            return y
        drv.mlp, drv.face, aim.ik, Bt.ctrl = mlp_w, face_w, ik_w, Ctrl()
        def wrap(mod, name, split):
            f = getattr(mod, name)
            def g(*a, **k):
                y = f(*a, **k); n = len(a[0])
                if n > 1 and P.due(name):
                    R = [f(*split(a, i), **k) for i in range(n)]; Y = y if isinstance(y, tuple) else (y,)
                    for j, yy in enumerate(Y): P.cmp(name, yy.double().numpy() if yy.dtype != torch.bool else yy.numpy(), torch.cat([(r if isinstance(r, tuple) else (r,))[j] for r in R]).double().numpy() if yy.dtype != torch.bool else torch.cat([(r if isinstance(r, tuple) else (r,))[j] for r in R]).numpy())
                return y
            setattr(mod, name, g); return f
        rows = lambda a, i: tuple(x[i:i + 1] if torch.is_tensor(x) and x.dim() and x.shape[0] == len(a[0]) else x for x in a)
        self.orig = [(HT, "zone_forces", wrap(HT, "zone_forces", lambda a, i: (a[0][i:i + 1],) + a[1:2] + tuple(x[i:i + 1] if x is not None else None for x in a[2:]))),
                     (HT, "closing_speeds", wrap(HT, "closing_speeds", rows)), (OB, "proprio83", wrap(OB, "proprio83", rows)), (OB, "kp_state", wrap(OB, "kp_state", rows))]
    def remove(self):
        for mod, name, f in self.orig: setattr(mod, name, f)

def _first_diff(a, b):
    """First step (1-based) and quantity in step order where two traces differ (None: equal over the shorter)."""
    for t, (x, y) in enumerate(zip(a, b)):
        for name, i in (("defender_action", 0), ("defender_targets", (1, slice(0, 24))), ("opponent_targets", (1, slice(24, 48))), ("qpos", 2)):
            u, v = (x[i], y[i]) if isinstance(i, int) else (x[i[0]][i[1]], y[i[0]][i[1]])
            if not np.array_equal(u, v): return dict(step=t + 1, quantity=name, max_abs=float(np.abs(u - v).max()))
    return None if len(a) == len(b) else dict(step=min(len(a), len(b)) + 1, quantity="length", max_abs=None)

def _diff_fields(x, y, pre=""):
    out = []
    for k in sorted(set(x) | set(y)):
        if k in ("_info",): continue
        u, v = x.get(k), y.get(k)
        if J(u) == J(v): continue
        if isinstance(u, dict) and isinstance(v, dict) and k in ("eo", "hit_meter", "acct", "hm", "n"): out += _diff_fields(u, v, pre + k + ".")
        else: out.append(pre + k)
    return out

def _invariance(args):
    ctrl, k, seconds, p1 = args; torch.set_num_threads(1)
    import fb_replica as RP, fb_m6_run as D
    c = RP.conf(**D.REQ_CONF); eps = _eps(INV_SEEDS); ctrl_, cfg, lag, m, names = RP.world(ctrl, c, p1); t0 = time.time()
    tr = _N1Trace(); B1 = RP.Bout(ctrl_, cfg, c, m, native=names, seconds=seconds, lag=lag); ref = [RP.rep_row(e, *RP.run_bout(B1, e, tr), seconds, c) for e in eps]; T1 = tr.T
    out = dict(ctrl=ctrl, bouts=len(eps), n1_wall_s=round(time.time() - t0, 1), variants={})
    for var in ("batch", "rowwise"):
        t0 = time.time(); Bt = RP.Batch(ctrl_, cfg, c, m, k, native=names, seconds=seconds, lag=lag, rowwise=var == "rowwise"); Bt._modules(min(k, len(eps)))
        P = _Probes(); P.install(Bt); TK = [[] for _ in eps]; rows = []
        try:
            for j in range(0, len(eps), k):
                Bt.probe = lambda i, S, a, j=j: TK[j + i].append((np.asarray(a, np.float64).copy(), S.d.ctrl.copy(), S.d.qpos.copy()))
                rows += [RP.rep_row(e, row, info, seconds, c) for e, (row, info) in zip(eps[j:j + k], Bt.run(eps[j:j + k]))]
        finally: P.remove()
        diffs = []
        for e, x, y, a, b in zip(eps, ref, rows, T1, TK):
            if J(x) == J(y) and len(a) == len(b) and all(np.array_equal(p, q) for u, v in zip(a, b) for p, q in zip(u, v)): continue
            diffs.append(dict(seed=e["seed"], row_equal=J(x) == J(y), fields=_diff_fields(x, y), first=_first_diff(a, b), steps=[len(a), len(b)],
                              n1=dict(fell_s=x["fell_s"], kd=x["kd"], head_clean=x["n"]["head_clean"]), batched=dict(fell_s=y["fell_s"], kd=y["kd"], head_clean=y["n"]["head_clean"])))
        fq = {}
        for d in diffs:
            q = (d["first"] or {}).get("quantity", "none"); fq[q] = fq.get(q, 0) + 1
        out["variants"][var] = dict(k=k, wall_s=round(time.time() - t0, 1), rows_equal=len(eps) - sum(not d["row_equal"] for d in diffs), traces_equal=len(eps) - len(diffs),
                                    first_diff_quantity=fq, probes=P.S, diffs=diffs)
    return out

def invariance(k=64, seconds=20.0, workers=2, ctrls=CTRLS, p1=None):
    res = _pool(_invariance, [(c, k, seconds, p1) for c in ctrls], workers)
    return dict(k=k, seconds=seconds, seeds={r: [s[0], s[-1], len(s)] for r, s in INV_SEEDS.items()}, per_ctrl={r["ctrl"]: r for r in res},
                summary={r["ctrl"]: {v: dict(rows_equal=x["rows_equal"], traces_equal=x["traces_equal"], of=r["bouts"], first_diff_quantity=x["first_diff_quantity"],
                                             probes_with_diff={p: s for p, s in x["probes"].items() if s["rows_diff"]}) for v, x in r["variants"].items()} for r in res})

def _pool(fn, jobs, workers):
    assert workers <= 2, "at most 2 worker processes"
    if workers == 1: return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool: return pool.map(fn, jobs, chunksize=1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["speed", "invariance"]); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--ks", default="32,64,96"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--ctrls", default=",".join(CTRLS)); ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--p1", default=None); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); ctrls = tuple(a.ctrls.split(","))
    r = speed([int(x) for x in a.ks.split(",")], a.seconds, a.workers, ctrls) if a.cmd == "speed" else invariance(a.k, a.seconds, a.workers, ctrls, a.p1)
    r["wall_s"] = round(time.time() - t0, 1); (out / f"m6b_{a.cmd}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    print(json.dumps(r.get("per_ctrl") if a.cmd == "speed" else r["summary"], default=str), flush=True)

if __name__ == "__main__": main()
