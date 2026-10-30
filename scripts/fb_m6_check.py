"""M6 part 1 checks of the CPU replica (M2V2_PLAN_v3 3 M6 R1, M5 G0-11 on the replica; A4.3), against the eval's own code:
  identity  R1: scripts/fb_replica.py in eval mode (numpy backend, full hulls, Newton 100/50, EO's episodes, HitMeter) for G and P1's
            H(A3s)/M3s on 60100-60115 (fb_seeds role 'identity'; fb_record's seed -> episode map): every row bit identical to
            eval_opponent.run (threads 1, DR, meter, falls, keypoints truth, aim.json's lag) on the same episodes (fell_s and every
            HitMeter count, the whole row).
  f64       the torch float64 modules (TrackerDriver + AimOp, EvalActorObs, HybridDelayedAction, fb_dr, fb_events) as deployed vs the
            numpy replica on the same bouts: per-step qpos max |diff| <= 1e-9 over the steps before the first defender-opponent shell
            contact (either run); then same-state checks on the torch run's own states (a shadow runs the eval's code on them each step:
            Aim.reference, Tracker.act's 132-d obs and targets, R1HitEnv._raw_obs + DelayedNoisyObs + KeypointSource with the eval's
            rng, R1HitEnv's action arithmetic, R1.fallen / EO:86, _dr_sample_into_model) <= 1e-9. Diagnostics (reported): the same
            run with the eval's own tracker session (actor 'onnx') and reference arrays (ref 'eval') in the torch driver; the
            tracker MLP float64 vs the ONNX graph in float64 on the eval's obs rows (G0-1's reference, scripts/fb_m2_check.onnx_f64).
  g011      M5 G0-11 on the replica: eval mode, scorer 'account' with native contact sensors (found / force / pos, netforce, as mjlab
            builds them), 60200-60263 (role 'hit_scoring'), G and H. A4.3 probe on every substep: the sensor's pos within 1 cm of the
            MuJoCo contact position on >= 99% of contacts (strike-limb secondaries vs defender shells) -> deployed closing-speed
            input 'pos' (velocities at that point), else 'site' (palm / foot sites). Gate (the deployed input, float64 as M5):
            landed counts per (limb, zone) exact vs the HitMeter on every bout, clean-vs-not agreement >= 0.95, median landed head
            impulse within 10% (fb_m5_check.compare); reported: float32 and the other inputs, windows and pay vs the HitMeter and
            pay_ref, the rows vs the M5 recording's EO rows (bit identity of the eval-mode replica on these bouts).
Run in the repo: python -I scripts/fb_m6_check.py {identity|f64|g011|all} --out DIR [--workers 2] [--p1 p1_eval.py] [--m5rec npz]"""
import sys, json, time, copy, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, torch, mujoco
import fb_replica as RP

TOL = dict(qpos=1e-9, same=1e-9, pos_m=0.01, pos_share=0.99, clean_agree=0.95, med_imp=0.10)
ID_SEEDS, HIT_SEEDS, CTRLS = tuple(range(60100, 60116)), tuple(range(60200, 60264)), ("G", "H")
ANCHOR = slice(48, 57)   # the tracker obs' anchor position (3) and orientation (6): fb_motion's export vs the ONNX initializers (M2 G0-2 (b))
VMODES = ("pos", "site", "com", "point")
M5REC = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody\build\verify_m5\rec\fb_m5rec_60200-60363.npz")
J = lambda x: json.dumps(x, sort_keys=True, default=str)

def _eo(name, eps, p1, seconds=20.0):
    """eval_opponent.run on eps, as R0-EO runs it (threads 1)."""
    import eval_opponent as EO, fb_r0eo as R0
    c, cfg, m, lag = R0.cell(name, p1 or R0.P1); return EO.run(c, eps, m, cfg, seconds, threads=1, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")

# ---- R1 identity
def _identity(args):
    name, seeds, p1 = args; torch.set_num_threads(1); eps = RP.episodes("identity", seeds); ref = _eo(name, eps, p1); rows = RP.run(name, eps, RP.conf(), p1=p1)
    return [dict(ctrl=name, seed=e["seed"], equal=J(r["eo"]) == J(x), fell=[r["eo"]["fell_s"], x["fell_s"]], n=[r["eo"]["hit_meter"]["n"], x["hit_meter"]["n"]],
                 opp_fell=[r["eo"]["opp_fell"], x["opp_fell"]]) for e, r, x in zip(eps, rows, ref)]

def identity(workers=2, p1=None, seeds=ID_SEEDS):
    jobs = [(n, list(seeds[i::workers]), p1) for n in CTRLS for i in range(workers)]; res = sorted(sum(_pool(_identity, jobs, workers), []), key=lambda x: (x["ctrl"], x["seed"]))
    per = {n: dict(bouts=sum(x["ctrl"] == n for x in res), identical=sum(x["equal"] for x in res if x["ctrl"] == n), falls=sum(x["fell"][1] is not None for x in res if x["ctrl"] == n),
                   opp_fell=sum(bool(x["opp_fell"][1]) for x in res if x["ctrl"] == n), head_clean=sum(x["n"][1]["head_clean"] for x in res if x["ctrl"] == n),
                   torso_landed=sum(x["n"][1]["torso_landed"] for x in res if x["ctrl"] == n)) for n in CTRLS}
    return dict(per_ctrl=per, mismatches=[x for x in res if not x["equal"]], ok=bool(all(v["identical"] == v["bouts"] == len(seeds) for v in per.values())))

# ---- f64: qpos traces and the same-state shadow
class Contact:
    """Substep hook: any defender-opponent shell contact this control step."""
    def __init__(self, m):
        from r1gym.model import SHELL_BIT
        self.a = np.array([(m.geom_contype[g], m.geom_conaffinity[g]) == (0, SHELL_BIT) for g in range(m.ngeom)]); self.b = np.array([(m.geom_contype[g], m.geom_conaffinity[g]) == (SHELL_BIT, 0) for g in range(m.ngeom)]); self.hit = False
    def reset(self): self.hit = False
    def substep(self, m, d):
        n = d.ncon; g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]; self.hit |= bool(np.any((self.a[g1] & self.b[g2]) | (self.a[g2] & self.b[g1])))

class Trace:
    """probe: qpos after reset and after every step, and the first step with a shell contact."""
    def __init__(self): self.q, self.first = [], None
    def __call__(self, B, kind, a):
        if kind == "reset": self.c = Contact(B.m); B.hooks.append(self.c); self.q = [B.d.qpos.copy()]; self.first = None; self.ctrl = []
        elif kind == "step":
            self.q.append(B.d.qpos.copy()); self.ctrl.append(B.d.ctrl.copy())
            if self.c.hit and self.first is None: self.first = len(self.q) - 1
            self.c.hit = False
            if B.shadow: B.shadow.post(B)

class Shadow:
    """The eval's own code on the torch replica's states, every step (module outputs vs the eval's, max |diff|)."""
    KEYS = ("dr_fields", "act_delay", "aim_rq", "aim_rdq", "aim_dist", "obs132_other", "obs132_anchor", "mlp_vs_graph64", "targets_vs_onnx32", "proprio83", "delayed83_noise",
            "keypoints", "action_targets", "applied", "fell", "opp_down", "yaw_spot_axis")
    def __init__(self, f64graph): self.g64 = f64graph; self.mx = {k: 0.0 for k in self.KEYS}; self.n = 0
    def _u(self, k, a, b): a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); self.mx[k] = max(self.mx[k], float(np.abs(a - b).max()) if a.size else 0.0)
    def reset(self, B):
        from r1gym.env import R1HitEnv as E
        from r1gym.sensors import DelayedNoisyObs
        from r1gym.keypoints import KeypointSource
        from r1gym.aim import Aim
        from types import SimpleNamespace
        m2 = copy.deepcopy(B.m); import fb_dr as DR
        F2 = DR.model_fields(m2)[0]
        for k in DR.FIELDS: F2[k][0] = B.nom[k]
        sh = SimpleNamespace(m=m2, r=B.r, rng=np.random.default_rng(B.ep.get("env_seed", B.ep["seed"]))); E._dr_init(sh); E._dr_sample_into_model(sh)
        from r1gym.env import _DR_NOMINAL; _DR_NOMINAL.pop(id(m2), None)
        for k in DR.FIELDS: self._u("dr_fields", getattr(m2, k), getattr(B.m, k))
        self._u("act_delay", sh.act_delay, B.act_delay); self.rng = sh.rng
        self.dno = DelayedNoisyObs(B.k, B.noise, self.rng, uniform=True); self.dz = DelayedNoisyObs(B.k, np.zeros(83), np.random.default_rng(0), uniform=True)
        self.src = KeypointSource(B.m, "truth", delay=B.k) if B.kpm == "obs" else None; self.t = 0
        rp = B.opp_rp; self.es = {"last": np.zeros(24, np.float32), "r1_torso": B.r.torso, "aim": Aim(B.m, rp, B.mode, B.cfg.get("aim"), B.r.torso)}
        a = self.es["aim"]; base = type(a).reference.__get__(a); cap = self.cap = {}
        def ref(*x, **k): out = base(*x, **k); cap["rq"], cap["rdq"] = np.array(out[0], np.float64), np.array(out[1], np.float64); return out
        a.reference = ref; self.a_prev = np.zeros(24, np.float32); self.filt = np.zeros(24)
    def act(self, B, a):
        a = np.clip(a, -B.clip, B.clip).astype(np.float32)
        if B.act_delay: a, self.a_prev = self.a_prev, a
        self.filt = B.alpha * self.filt + (1 - B.alpha) * a; self._u("action_targets", B.home + B.scale * self.filt, B.d.ctrl[B.r.act]); self._u("applied", a, B.act.applied[0].numpy())
    def opp_before(self, B, t):
        import r1gym.opponent as OP, fb_record as REC
        d = B.d; self.es["target"] = d.xpos[B.r.pel][:2].copy(); self.es["last"] = B.drv.last[0].double().numpy().copy()
        npr = REC._NpRec(OP.np); OP.np = npr
        try: B.opp_rp.tracker.act(B.m, d, B.opp_rp, self.es, t, B.dt)
        finally: OP.np = npr._np
        self.obs64 = npr.last.copy(); self.ctrl32 = d.ctrl[B.oi.act].copy()
    def opp_after(self, B, rq, rdq, dist, tg, info):
        self._u("aim_rq", rq[0].numpy(), self.cap["rq"]); self._u("aim_rdq", rdq[0].numpy(), self.cap["rdq"]); self._u("aim_dist", float(dist[0]), self.es.get("dist", B.opp_rp.distance))
        o = info["obs"][0].numpy(); m = np.ones(132, bool); m[ANCHOR] = False
        self._u("obs132_other", o[m], self.obs64[m]); self._u("obs132_anchor", o[ANCHOR], self.obs64[ANCHOR])
        x = torch.from_numpy(self.obs64[None]); self._u("mlp_vs_graph64", B.drv.mlp(x)[0].numpy(), self.g64(self.obs64[None])[0]); self._u("targets_vs_onnx32", tg[0].numpy(), self.ctrl32)
        self._u("yaw_spot_axis", [float(B.drv.yaw[0]), *B.drv.spot[0].tolist(), *B.drv.axis[0].tolist()], [self.es["yaw"], *self.es["spot"], *self.es["axis"]])
    def obs(self, B, p, o):
        raw = B._raw_obs(); self._u("proprio83", p[0].numpy(), raw)
        if self.t == 0: self.dno.reset(raw); kp = self.src.reset(B.d) if self.src else None
        else: kp = self.src.observe(B.d) if self.src else None
        self._u("delayed83_noise", o[0, :83].double().numpy(), self.dno(raw))
        if kp is not None: self._u("keypoints", o[0, 83:].double().numpy(), kp)
        self.t += 1; self.n += 1
    def post(self, B):
        import eval_opponent as EO, fb_events as EV
        d, b = B.d, B.oi.pel; self._u("fell", float(bool(B.r.fallen(d))), float(bool(EV.fell_eval(torch.tensor(d.xpos[B.r.pel][2]), EV.up_z(torch.from_numpy(d.xquat[B.r.pel].copy()))))))
        self._u("opp_down", float(bool(d.xmat[b][8] < EO.OPP_TILT) or bool(d.xpos[b][2] < EO.OPP_Z)), float(bool(B.od.now[0])))

def _f64(args):
    name, seeds, p1 = args; torch.set_num_threads(1)
    import fb_m2_check as M2
    g64, _ = M2.onnx_f64(M2.onnx_path()); eps = RP.episodes("identity", seeds); out = []; BB = {}
    for tag, kw in (("numpy", {}), ("torch64", dict(backend="torch64")), ("torch64_onnx_evalref", dict(backend="torch64", actor="onnx", ref="eval"))):
        c = RP.conf(**kw); ctrl, cfg, lag, m, names = RP.world(name, c, p1); BB[tag] = RP.Bout(ctrl, cfg, c, m, seconds=20.0, lag=lag)
    for ep in eps:
        rec = {}
        for tag, B in BB.items():
            tr = Trace(); B.shadow = Shadow(g64) if tag == "torch64" else None
            row, info = RP.run_bout(B, ep, tr); rec[tag] = dict(q=np.array(tr.q), first=tr.first, ctrl=np.array(tr.ctrl), row=row, shadow=B.shadow)
        n0 = rec["numpy"]; r = dict(ctrl=name, seed=ep["seed"], first_contact_step=n0["first"], fell=n0["row"]["fell_s"])
        for tag in ("torch64", "torch64_onnx_evalref"):
            x = rec[tag]; k = min(v for v in (n0["first"], x["first"], len(n0["q"]), len(x["q"])) if v is not None); D = np.abs(n0["q"][:k] - x["q"][:k]).max(1) if k else np.zeros(0)
            over = np.flatnonzero(D > TOL["qpos"]); C = np.abs(n0["ctrl"][:1] - x["ctrl"][:1]).max() if len(x["ctrl"]) else 0.0
            r[tag] = dict(steps_compared=int(k), first_contact_step=x["first"], qpos_max=float(D.max()) if len(D) else 0.0, first_step_over=int(over[0]) if len(over) else None,
                          qpos_at=[float(D[i]) for i in (1, 2, 5, 10, 25) if i < len(D)], ctrl_step1_max=float(C), opp_ctrl_step1_max=float(np.abs(n0["ctrl"][0][-24:] - x["ctrl"][0][-24:]).max()),
                          def_ctrl_step1_max=float(np.abs(n0["ctrl"][0][:24] - x["ctrl"][0][:24]).max()))
        r["same_state"] = dict(rec["torch64"]["shadow"].mx, steps=rec["torch64"]["shadow"].n); out.append(r)
    return out

def f64(workers=2, p1=None, seeds=ID_SEEDS):
    jobs = [(n, list(seeds[i::workers]), p1) for n in CTRLS for i in range(workers)]; res = sorted(sum(_pool(_f64, jobs, workers), []), key=lambda x: (x["ctrl"], x["seed"]))
    ss = {k: max(x["same_state"][k] for x in res) for k in Shadow.KEYS}; f = lambda tag, k: max(x[tag][k] for x in res)
    imp = ("obs132_anchor", "mlp_vs_graph64", "targets_vs_onnx32")   # reported against their own gates (M1/M2), see the notes
    return dict(bouts=len(res), per_bout=res, qpos=dict(deployed=dict(max=f("torch64", "qpos_max"), bouts_over=sum(x["torch64"]["qpos_max"] > TOL["qpos"] for x in res),
                                                                         opp_ctrl_step1_max=f("torch64", "opp_ctrl_step1_max"), def_ctrl_step1_max=f("torch64", "def_ctrl_step1_max")),
                                                          onnx_evalref=dict(max=f("torch64_onnx_evalref", "qpos_max"), bouts_over=sum(x["torch64_onnx_evalref"]["qpos_max"] > TOL["qpos"] for x in res),
                                                                            opp_ctrl_step1_max=f("torch64_onnx_evalref", "opp_ctrl_step1_max"))),
                same_state=ss, same_state_steps=sum(x["same_state"]["steps"] for x in res),
                ok_qpos_deployed=bool(all(x["torch64"]["qpos_max"] <= TOL["qpos"] for x in res)), ok_qpos_onnx_evalref=bool(all(x["torch64_onnx_evalref"]["qpos_max"] <= TOL["qpos"] for x in res)),
                ok_same_state_all=bool(all(v <= TOL["same"] for v in ss.values())), ok_same_state_but_tracker_actor_ref=bool(all(v <= TOL["same"] for k, v in ss.items() if k not in imp)))

# ---- G0-11 on the replica + A4.3 (sensor rows logged in the replica, then fb_m5_check.replay / compare unchanged)
def _post(B, out):
    """probe: fb_m5_check.POST per step (the post-step state the eval reads)."""
    def f(B, kind, a):
        if kind != "step": return
        d, r, b = B.d, B.r, B.oi.pel; q, v = r.free_q, r.free_v; R = d.xmat[r.torso]
        out.append([d.time, d.xpos[b][2], d.xmat[b][8], d.qpos[q], d.qpos[q + 1], d.qvel[v], d.qvel[v + 1], d.qvel[v + 2], d.xpos[b][0], d.xpos[b][1], R[2], R[5], R[8], float(r.fallen(d))])
    return f

def _g011(args):
    name, seeds, p1 = args; torch.set_num_threads(1)
    import fb_m5_check as M5, fb_hits as HT
    from r1gym.hits import STRIKE_LIMBS
    eps = RP.episodes("hit_scoring", seeds); c = RP.conf(); ctrl, cfg, lag, m, names = RP.world(name, c, p1, sensors=True)
    B = RP.Bout(ctrl, cfg, c, m, native=names, seconds=20.0, lag=0.0, log_vmodes=VMODES); out = []   # lag 0: as the M5 recording's EO rows
    for ep in eps:
        post = []; row, info = RP.run_bout(B, ep, _post(B, post)); hm = B.hm; secs = row["fell_s"] or 20.0; tail = HT.WIN_TAIL_S
        its = [it for it in B.intents() if it.t_on >= 0 and it.t_apex + tail <= secs]; oc = hm.windows([it.window(tail) for it in its]); evs = hm.events(by_side=True); bt = B.sens.block_touch
        ab = lambda sd, g, t0: (HT.SIDES.index(sd) * 4 + HT.LIMBS.index(g), round(t0, 4)) in bt
        ints = sorted((dict(limb=it.limb, t_on=it.t_on, t_apex=it.t_apex, gated=it.gated, outcome=o, pay=M5.pay_ref(evs, it.t_on, it.t_apex, it.limb, hm.p, ab),
                            pay_any_arm=M5.pay_ref(evs, it.t_on, it.t_apex, it.limb, hm.p)) for it, o in zip(its, oc)), key=lambda x: x["t_on"])
        ev = [dict(limb=x["limb"], zone=x["zone"], t0=x["t0"], t1=x["t1"], impulse_Ns=x["impulse_Ns"], peak_N=x["peak_N"], v_close=x["v_close"], dur_s=x["dur_s"],
                   landed=bool(hm.landed(x)), clean=bool(hm.clean(x))) for x in hm.events() if x["limb"] in STRIKE_LIMBS]
        A = dict(hit=np.array(B.sens.rows["hit"], np.float64).reshape(-1, 7 + 3 * len(VMODES)), body=np.array(B.sens.rows["body"], np.float64).reshape(-1, M5.BODY_COLS), post=np.array(post, np.float64).reshape(-1, len(M5.POST)))
        out.append((dict(ctrl=name, role="hit_scoring", episode=ep, row=row, events=ev, intents=ints, probe=dict(B.sens.probe)), A))
    return out

def g011(workers=2, p1=None, seeds=HIT_SEEDS, m5rec=M5REC, save=None, BA=None):
    """save: an npz of the replica's logged bouts (fb_m5_check.save format, velocity columns per VMODES); BA: reuse such bouts (no simulation)."""
    import fb_m5_check as M5
    if BA is None: BA = sorted(sum(_pool(_g011, [(n, list(seeds[i::workers]), p1) for n in CTRLS for i in range(workers)], workers), []), key=lambda b: (CTRLS.index(b[0]["ctrl"]), b[0]["episode"]["seed"]))
    if save: M5.save(dict(meta=dict(bouts=[b for b, _ in BA], vmodes=list(VMODES)), **{f"{n}/{k}": v for n, (_, A) in enumerate(BA) for k, v in A.items()}), save)
    B = [b for b, _ in BA]; P = {k: int(sum(b["probe"][k] for b in B)) for k in B[0]["probe"] if k != "max_err_nonzero"}; P["max_err_nonzero_m"] = max(b["probe"]["max_err_nonzero"] for b in B)
    share = P["within_1cm"] / max(P["contacts"], 1); deployed = "pos" if share >= TOL["pos_share"] else "site"
    P.update(share_within_1cm=round(share, 5), share_within_1cm_nonzero_force=round(P["within_1cm_nonzero"] / max(P["contacts_nonzero"], 1), 5),
             share_slots_strongest_1cm=round(P["slots_strongest_1cm"] / max(P["slots"], 1), 5), passes=bool(share >= TOL["pos_share"]), deployed=deployed)
    ctrl, cfg, lag, m, names = RP.world("G", RP.conf(), p1, sensors=True); res = {}
    for i, vm in enumerate(VMODES):   # the input in fb_m5_check's 'com' slot (cols 7-9), the HitMeter point in its 'point' slot
        R = dict(meta=dict(bouts=B))
        for n, (_, A) in enumerate(BA):
            h = A["hit"]; R[f"{n}/hit"] = np.concatenate([h[:, :7], h[:, 7 + 3 * i:10 + 3 * i], h[:, 7 + 3 * VMODES.index("point"):10 + 3 * VMODES.index("point")]], 1); R[f"{n}/body"], R[f"{n}/post"] = A["body"], A["post"]
        for dt in (torch.float64, torch.float32):
            k = f"{vm}/{str(dt).split('.')[-1]}"; res[k] = {}
            for c in CTRLS:
                idx = [j for j, b in enumerate(B) if b["ctrl"] == c]; rp = M5.replay(R, idx, m, "com", dtype=dt)
                if dt == torch.float32: rp["log"] = [dict(x, t0=round(x["t0"], 4), t1=round(x["t1"], 4)) if x["kind"] == "episode" else x for x in rp["log"]]   # float32 times on the HitMeter's 4-dp grid (compare matches t0 within 1e-6)
                res[k][c] = M5.compare(R, idx, rp)
    gk = f"{deployed}/float64"; ok = all(res[gk][n]["ok"] for n in CTRLS); eo = {}
    if m5rec and Path(m5rec).exists():   # the eval-mode replica's rows vs the M5 recording's EO rows on the same bouts (recorded with rev_hits on)
        meta = M5.load(m5rec)["meta"]; ref = {(b["ctrl"], b["episode"]["seed"]): b["row"] for b in meta["bouts"] if b["role"] == "hit_scoring"}
        eq = [J(b["row"]) == J({k: v for k, v in ref[(b["ctrl"], b["episode"]["seed"])].items() if k != "rev_meter"}) for b in B if (b["ctrl"], b["episode"]["seed"]) in ref]
        eo = dict(bouts=len(eq), rows_equal=sum(eq))
    sm = lambda r: {k: r[k] for k in ("bouts", "landed_hm", "landed_ours", "count_mismatch_bouts", "clean_agree", "clean_hm", "clean_ours", "head_impulse_median", "windows", "ok")}
    alt = "site" if deployed == "pos" else "pos"
    return dict(a43_probe=P, deployed_vmode=deployed, gated=gk, ok=bool(ok), ok_other_reading=bool(all(res[f"{alt}/float64"][n]["ok"] for n in CTRLS)),
                per_input={k: {n: sm(v[n]) for n in CTRLS} for k, v in res.items()}, mismatches={n: res[gk][n]["count_mismatches"] for n in CTRLS}, vs_m5_recording_rows=eo)


def _pool(fn, jobs, workers):
    assert workers <= 2, "at most 2 worker processes (R0-EO holds 4)"
    jobs = [j for j in jobs if j[1]]
    if workers == 1: return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool: return pool.map(fn, jobs, chunksize=1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["identity", "f64", "g011", "all"]); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--p1", default=None); ap.add_argument("--m5rec", default=str(M5REC)); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res = {}
    for cmd in (("identity", "f64", "g011") if a.cmd == "all" else (a.cmd,)):
        t0 = time.time(); r = dict(identity=lambda: identity(a.workers, a.p1), f64=lambda: f64(a.workers, a.p1), g011=lambda: g011(a.workers, a.p1, m5rec=a.m5rec, save=str(out / "g011_bouts.npz")))[cmd]()
        r["wall_s"] = round(time.time() - t0, 1); res[cmd] = r; (out / f"m6_{cmd}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        print(json.dumps({cmd: {k: v for k, v in r.items() if k in ("ok", "per_ctrl", "qpos", "same_state", "ok_qpos_deployed", "ok_qpos_onnx_evalref", "ok_same_state_all",
                                                                       "ok_same_state_but_tracker_actor_ref", "a43_probe", "deployed_vmode", "wall_s", "vs_m5_recording_rows")}}, default=str), flush=True)
    return res

if __name__ == "__main__": main()
