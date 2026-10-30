"""M3 checks of the torch aim (M2V2_PLAN_v3 3 M3, G0-4; gpu/fb_opponent.py ArmKinematics, AimOp) on CPU against the eval's own
r1gym/aim.py, in float64 and float32 (m12):
  record  the 16 record bouts (seeds 60000-60015, role 'record': scripts/fb_record.py's episodes, G, aim standoff+armwarp_vz, DR,
          no pushes, 20 s) re-run through eval_opponent.run with a read-only hook on Aim.reference: per control step its inputs
          (time, clock, the strike-limb touch flag Aim._touching, both torso poses, the bout distance, the opponent's qpos and palm
          sites, the clip reference) and outputs (warped rq / rdq, the stand-off distance, the warp), and per bout every intent.
          Each bout also runs without the hook (rows identical, asserted); with --rec, the opponent's qpos per step must equal
          M2's recording bit for bit (same bouts).
  FK      ArmKinematics vs Aim._palm_b + Rt^T mj_jacSite on the opponent sub-model: the train clips' reference frames, 20k uniform
          joint vectors, the eval's warped references; and the palm in the world from the recorded joints vs d.site_xpos.
          <= 1e-6 (float64) / 1e-4 (float32).
  AimOp   batched over the 16 bouts on the recorded states vs Aim.reference: intents (limb, onset, times, gates, commit, reach)
          identical; stand-off distance <= 1e-6; closest palm-to-target per intent <= 1 mm. float64: warped q <= 1e-4 rad. float32
          as deployed (states handed over rounded to float32, outputs float32, the aim's arithmetic float64), warped q per plan
          Amendment A1: (i) vs the float64 port on the same float32-rounded states <= 1e-4 rad on every row, (ii) vs the eval
          <= 1e-4 rad on >= 99.9% of rows. Reported: float32 arithmetic throughout, and on every row the float32 run misses the eval
          by 1e-4, the eval's own Aim.ik on the IK target from the exact and from the float32-rounded states. Also the ring
          (8 slots) = 64 slots bit for bit with no overflow.
  jitter  AimOp with JITTER, the target-space reach correction F' and the A2.1 guard (plan A3.1, reach 'A3') on every bout over
          generator seeds 0-19 (no episode seed; the same recorded states, no new simulation) vs the nominal aim, per A3.2: A1 (i)
          at every warped step of every seed the IK's x target (the vz target F' starts from) is the reference palm's own and the
          nominal one, bit for bit; A2 (ii') the pooled mean of (jitter - nominal) forward distance (opponent torso frame) of the
          warped reference palm at the aimed apexes has a bout-cluster bootstrap 90% CI inside +-5 mm; A2 (iii') at every warped
          step of every seed jittered palm x - nominal palm x <= 1e-6 m (float64) / 1e-5 m (float32); A1 (iv) mixed mask (odd
          bouts jittered, seed 0): T0 rows bit-equal to the no-jitter run, jittered rows differ; (vi) the mean yz apex spread >=
          50% of the raw jitter's (same seeds); (vii) per-step arm q change on T1 warped limb-steps, q99.9 <= 1.25x and max <= T0
          max + 0.2 rad of the T0 rows'. Reported: (v) the guard's binding share and mean s, F' convergence (1e-6 m) share, the IK
          calls and wall per step; A1's replaced (ii) t CI and (iii) apex excess per seed, the frozen v0 measure, the eval's own
          warp vs the clip, the IK residual, the raw jitter at seed 0. If F' fails (ii'), (vi) or (vii) in either dtype, A3.3's
          fallback D (reach 'D': no arm-target offset, commit and depth jitter kept, the A2.1 guard) is gated on A1 (i), (ii'),
          (iii'), A1 (iv) with (v) reported, and adopted if it passes.
res 'superseded_by_A1' keeps the two frozen gates A1 replaced (warped q f32 on every row vs the eval; jitter forward distance
<= 1 mm), as measured; they no longer enter res 'ok' or the exit code. The jitter results keep A1's replaced (ii), (iii) per seed
under 'superseded_by_A2'. res 'adopted' names the reach mode that passed (F' or D), else None.
Run in the repo: python scripts/fb_m3_check.py --out m3_check.json [--aimrec aimrec.npz] [--rec fb_rec_60000-60015.npz] [--workers 4]"""
import sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
import fb_opponent as FO

TOL = dict(fk_f64=1e-6, fk_f32=1e-4, dist=1e-6, q=1e-4, closest_m=1e-3, q_rows_frac=0.999, jitter_ci_m=5e-3, jitter_ci=0.90, guard_x_f64=1e-6, guard_x_f32=1e-5,
           guard_bind_report_frac=0.5, yz_keep_frac=0.5, smooth_q999_ratio=1.25, smooth_max_add_rad=0.2)
V0 = dict(q=1e-4, jitter_x_m=1e-3)   # the frozen gates A1 replaced (reported only)
V1 = dict(jitter_excess_m=5e-3)      # A1 jitter (iii) (and (ii) as a t CI at one seed), replaced by A2 (reported only)
JITTER_SEEDS, BOOT = tuple(range(20)), 4000   # A2: jitter generator seeds; bout-cluster bootstrap resamples
SEEDS = tuple(range(60000, 60016)); TRAIN = FO.TRAIN
STEP_KEYS = ("now", "time", "t", "fi", "touch", "opp_x", "opp_R", "def_x", "def_R", "home", "qopp", "palm", "rq0", "rdq0", "rq", "rdq", "dist", "prev")

# ---- recording the eval's Aim
class AimHook:
    """Read-only hook on r1gym.aim.Aim.reference (class level, this process): inputs and outputs per call, the Aim objects seen."""
    def install(self):
        import r1gym.aim as A
        self.A, self.orig, self.rows, self.aims = A, A.Aim.reference, [], []; hook = self
        def reference(aim, tr, m, d, replay, fi, state, t, dt):
            if not any(a is aim for a in hook.aims): hook.aims.append(aim)
            rq0, rdq0, _, _ = tr.reference(replay, tr.frame(replay.clip, fi), state)
            x = dict(now=d.time + dt, time=d.time, t=float(t), fi=int(fi), touch=bool(aim._touching(d)), opp_x=d.xpos[replay.torso].copy(), opp_R=d.xmat[replay.torso].reshape(3, 3).copy(),
                     def_x=d.xpos[aim.r1_torso].copy(), def_R=d.xmat[aim.r1_torso].reshape(3, 3).copy(), home=float(replay.distance), qopp=d.qpos[replay.qadr:replay.qadr + replay.nq].copy(),
                     palm=np.stack([d.site_xpos[aim.palm[k]] for k in FO.LIMBS]), rq0=np.array(rq0, np.float64), rdq0=np.array(rdq0, np.float64))
            out = hook.orig(aim, tr, m, d, replay, fi, state, t, dt)
            x.update(rq=np.array(out[0], np.float64), rdq=np.array(out[1], np.float64), dist=float(state.get("dist", np.nan)), prev=aim.prev.copy()); hook.rows.append(x)
            return out
        A.Aim.reference = reference; return self
    def remove(self): self.A.Aim.reference = self.orig

def _intent(it):
    return dict(limb=it.limb, zone=it.zone, onset=int(it.onset), t_on=it.t_on, t_apex=it.t_apex, t_end=it.t_end, gated=it.gated, reach_m=float(it.reach_m),
                target_r1=np.asarray(it.target_r1, float).tolist(), target_w=None if it.target_w is None else np.asarray(it.target_w, float).tolist(), committed=bool(it.committed),
                closest_m=float(it.closest_m), ik_res_m=None if it.ik_res_m is None else float(it.ik_res_m),
                arm_rmse=float(np.sqrt(np.mean(np.square(it.arm_err)))) if it.arm_err else None)

def work(args):
    seeds, seconds = args
    torch.set_num_threads(1)
    import eval_opponent as EO, fb_record as REC
    from fb_seeds import check
    check(seeds, "record", TRAIN); c, cfg, m = REC.setup(); out = []
    for ep in REC.episodes(seeds, seconds):
        plain = EO.run(c, [ep], m, cfg, seconds, threads=1, dr=True, meter=False, falls=True)[0]; h = AimHook().install()
        try: row = EO.run(c, [ep], m, cfg, seconds, threads=1, dr=True, meter=False, falls=True)[0]
        finally: h.remove()
        assert json.dumps(row, sort_keys=True, default=str) == json.dumps(plain, sort_keys=True, default=str), f"the hook changed bout {ep['seed']}"
        assert len(h.aims) == 1, f"{len(h.aims)} Aim objects in one bout"
        a = h.aims[0]; out.append((ep, row, h.rows, [_intent(it) for it in a.intents], json.loads(json.dumps(a.p)), a.mode))
    return out

def record(seeds=SEEDS, seconds=20.0, workers=4, rec=None):
    from fb_seeds import check
    check(seeds, "record", TRAIN); assert workers <= 4; t0 = time.time()
    sl = [list(seeds[i::workers]) for i in range(workers) if seeds[i::workers]]
    if len(sl) == 1: parts = [work((sl[0], seconds))]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(sl)) as pool: parts = pool.map(work, [(s, seconds) for s in sl], chunksize=1)
    A = assemble(sorted((b for p in parts for b in p), key=lambda b: b[0]["seed"]), seeds, seconds, t0); meta = A["meta"]
    if rec:   # the same bouts as M2's recording: opponent qpos per step bit-equal
        with np.load(rec, allow_pickle=False) as z: Q, Bm, m2 = z["qpos"], z["bout"], json.loads(str(z["meta"]))
        assert [e["seed"] for e in m2["episodes"]] == list(seeds) and m2["steps"] == meta["steps"], "not the M2 recording's bouts"
        qa = _opp_adr(); meta["same_as_m2_recording"] = bool(np.array_equal(Q[:, qa:qa + 31], A["qopp"]) and np.array_equal(Bm, A["bout"]))
        assert meta["same_as_m2_recording"], "the re-run differs from M2's recording"
    A["meta"] = np.array(json.dumps(meta, default=str)); return A

def assemble(bouts, seeds, seconds, t0=None):
    """work()'s bouts -> per-row arrays and meta (the dict form load() returns, meta still a dict)."""
    A = {k: np.stack([np.asarray(r[k]) for b in bouts for r in b[2]]) for k in STEP_KEYS}
    A["bout"] = np.concatenate([np.full(len(b[2]), i, np.int32) for i, b in enumerate(bouts)])
    A["meta"] = dict(seeds=list(seeds), role="record", seconds=seconds, episodes=[b[0] for b in bouts], rows=[b[1] for b in bouts], intents=[b[3] for b in bouts],
                     aim_p=bouts[0][4], aim_mode=bouts[0][5], steps=[len(b[2]) for b in bouts], wall_s=None if t0 is None else round(time.time() - t0, 1))
    assert all(b[4] == A["meta"]["aim_p"] and b[5] == "standoff+armwarp_vz" for b in bouts); return A

def _opp_adr():
    from r1gym.model import build_spec, load_cfg
    m = build_spec(opponent=True, cfg=load_cfg()).compile(); return int(m.jnt_qposadr[m.joint("opp_floating_base_joint").id])

def load(path):
    with np.load(path, allow_pickle=False) as z: A = {k: z[k] for k in z.files}
    A["meta"] = json.loads(str(A["meta"])); return A

# ---- FK and Jacobian
def mj_palm(Q):
    """The eval's way (Aim._palm_b, Aim.ik): palm in the torso frame and Rt^T mj_jacSite on the arm dofs, Q [N, 24] -> [N, 2, 3], [N, 2, 3, 5]."""
    import mujoco
    from r1gym.opponent import _sub
    from r1gym.aim import ARM, PALM
    m = _sub("r1"); d = mujoco.MjData(m); d.qpos[3] = 1.0; J = np.zeros((3, m.nv)); tor = m.body("torso_link").id; site = [m.site(PALM[l]).id for l in FO.LIMBS]
    P, A = np.zeros((len(Q), 2, 3)), np.zeros((len(Q), 2, 3, 5))
    for i, q in enumerate(Q):
        d.qpos[7:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); Rt = d.xmat[tor].reshape(3, 3)
        for l, limb in enumerate(FO.LIMBS):
            P[i, l] = Rt.T @ (d.site_xpos[site[l]] - d.xpos[tor]); mujoco.mj_jacSite(m, d, J, None, site[l]); A[i, l] = Rt.T @ J[:, [6 + k for k in ARM[limb]]]
    return P, A

def g0_4_fk(A):
    from r1gym.opponent import _sub
    M = FO.Motion(); m = _sub("r1"); lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    sets = dict(reference_frames=M.ref_q.double().numpy(), uniform_20k=np.random.default_rng(0).uniform(lo, hi, size=(20000, 24)), eval_warped=A["rq"])
    ix = list(FO.ARM_IX); res = {}
    for dt, tol in ((torch.float64, TOL["fk_f64"]), (torch.float32, TOL["fk_f32"])):
        k = FO.ArmKinematics.from_model(dtype=dt); r = {}
        for name, Q in sets.items():
            P, J = mj_palm(Q); p, j = k.fk(torch.from_numpy(Q[:, ix]), jac=True)
            r[name] = dict(n=len(Q), fk=float(np.abs(p.double().numpy() - P).max()), jac=float(np.abs(j.double().numpy() - J).max()))
        q = torch.from_numpy(A["qopp"][:, 7:][:, ix]); R, x = torch.from_numpy(A["opp_R"]).to(dt), torch.from_numpy(A["opp_x"]).to(dt)
        pw = x[:, None] + FO._mv(R[:, None], k.fk(q)); r["palm_world_vs_site_xpos"] = dict(n=len(q), fk=float(np.abs(pw.double().numpy() - A["palm"]).max()))
        r["ok"] = all(max(v.get("fk", 0), v.get("jac", 0)) <= tol for v in r.values() if isinstance(v, dict)); res[str(dt).split(".")[-1]] = r
    res["ok"] = res["float64"]["ok"] and res["float32"]["ok"]; return res

# ---- AimOp on the recorded states
def replay(A, dtype, compute=torch.float64, slots=64, jitter=False, seed=0, log_ik=True, debug=False, round_to=None, reach="A3"):
    """AimOp over every recorded bout at once (one env per bout); finished bouts repeat their last row. The states are handed over
    in `dtype` (float32: rounded as a float32 sim holds them), the clock and time in float64; round_to=float32 with dtype float64
    hands over the float32-rounded states in float64 (A1 (i): the float64 port on the float32 run's inputs). jitter: bool or a
    per-env bool tensor; reach the AimOp reach mode on jittered rows (FO.REACH: 'A3' F' + guard, 'A2' guard, 'D', 'raw' pre-A2).
    Per-row outputs, each bout's intents at its last step, the IK calls and wall per step, and with debug the IK's target and result
    per row (with jitter and a reach step also the guard's flag and s, the F' convergence flag, palm x residual and target shift per
    limb)."""
    eps = A["meta"]["episodes"]; B = len(eps); lens = np.bincount(A["bout"], minlength=B); start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    M = FO.Motion(); op = FO.AimOp(B, dtype, motion=M, slots=slots, seed=seed, log_ik=log_ik, compute=compute, debug=debug, reach=reach)
    clip = torch.tensor([TRAIN.index(e["clip"]) for e in eps]); op.reset(None, clip, torch.tensor([e["distance"] for e in eps], dtype=torch.float64), aimed=True, jitter=jitter)
    n = len(A["bout"]); out = dict(rq=np.zeros((n, 24)), rdq=np.zeros((n, 24)), dist=np.zeros(n), base_equal=True); intents = [None] * B
    if debug: out.update(p_t=np.zeros((n, 2, 3)), qa=np.zeros((n, 2, 5)), rq5=np.zeros((n, 2, 5)), has=np.zeros((n, 2), bool))
    if debug and op.any_jit and reach != "raw": out.update(guard=np.zeros((n, 2), bool), s=np.ones((n, 2)), conv=np.zeros((n, 2), bool), xres=np.zeros((n, 2)), shift=np.zeros((n, 2)))
    T = lambda k, i: torch.from_numpy(np.ascontiguousarray(A[k][i])); R = (lambda x: x.to(round_to).to(dtype)) if round_to is not None else (lambda x: x.to(dtype)); X = lambda k, i: R(T(k, i))
    t0 = time.perf_counter()
    for j in range(int(lens.max())):
        i = start + np.minimum(j, lens - 1); fi = T("fi", i); rq0, rdq0 = M.ref_q[M.row(clip, fi)], M.ref_dq[M.row(clip, fi)]
        out["base_equal"] &= bool(np.array_equal(rq0.double().numpy(), A["rq0"][i]) and np.array_equal(rdq0.double().numpy(), A["rdq0"][i]))
        rq, rdq, dist = op.step(fi, T("now", i), R(rq0), R(rdq0), X("opp_x", i), X("opp_R", i), X("def_x", i), X("def_R", i), T("touch", i), q_act=X("qopp", i)[:, 7:])
        live = j < lens; r = i[live]
        out["rq"][r], out["rdq"][r], out["dist"][r] = rq[live].double().numpy(), rdq[live].double().numpy(), dist[live].double().numpy()
        if debug:
            if "xres" in out: op.dbg["xres"] = op.dbg["x_cor"] - op.dbg["x_nom"]
            for k in ("p_t", "qa", "rq5", "has", "guard", "s", "conv", "xres", "shift"):
                if k in out: out[k][r] = op.dbg[k][live].numpy() if op.dbg[k].dtype == torch.bool else op.dbg[k][live].double().numpy()
        for b in np.flatnonzero(lens - 1 == j): intents[b] = op.intents(int(b))
    assert slots < 64 or all(int(op.nseq[b]) <= slots for b in range(B)), "history slots wrapped"
    out.update(intents=intents, overflow=op.overflow.tolist(), nseq=op.nseq.tolist(), lib_onset=op.lib_on.numpy(), ik_calls_per_step=op.ik_calls / int(lens.max()),
               wall_ms_per_step=1e3 * (time.perf_counter() - t0) / int(lens.max())); return out

def q_gate_a1(e_eval, e_same):
    """A1 float32 warped q: per-row max |q| error vs the eval and vs the float64 port on the same float32-rounded states."""
    within = float((e_eval <= TOL["q"]).mean())
    return dict(same_input_q_maxdiff=float(e_same.max()), same_input_ok=bool(e_same.max() <= TOL["q"]), eval_rows_within_tol_frac=within,
                eval_rows_ok=bool(within >= TOL["q_rows_frac"]))

def compare(A, O, name, port=None):
    """O vs the eval's recording. port (the float64 port on O's float32-rounded states) selects A1's float32 warped-q gate (i)-(ii);
    without it warped q <= 1e-4 rad on every row (float64)."""
    E = A["meta"]["intents"]; eps = A["meta"]["episodes"]
    eq, mis, dt_t, d_tw, d_cl, d_ik, d_arm, d_reach, n_int, gates = True, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, {"": 0, "tilt": 0, "touch": 0}
    for b, (ev, tv) in enumerate(zip(E, O["intents"])):
        c = TRAIN.index(eps[b]["clip"])
        if len(ev) != len(tv): eq = False; mis.append(dict(bout=b, eval=len(ev), torch=len(tv))); continue
        for e, t in zip(ev, tv):
            n_int += 1; gates[e["gated"]] += 1
            same = e["limb"] == t["limb"] and e["zone"] == "head" and e["onset"] == int(O["lib_onset"][c, t["lib"]]) and e["gated"] == t["gated"] and e["committed"] == t["committed"]
            same &= e["t_on"] == t["t_on"] and e["t_apex"] == t["t_apex"] and e["t_end"] == t["t_end"]
            if not same: eq = False; mis.append(dict(bout=b, eval={k: e[k] for k in ("limb", "onset", "gated", "t_on", "committed")}, torch={k: t[k] for k in ("limb", "lib", "gated", "t_on", "committed")}))
            dt_t = max(dt_t, abs(e["t_on"] - t["t_on"]), abs(e["t_apex"] - t["t_apex"]), abs(e["t_end"] - t["t_end"])); d_reach = max(d_reach, abs(e["reach_m"] - t["reach_m"]))
            d_tw = max(d_tw, float(np.abs(np.array(e["target_w"]) - t["target_w"]).max())); d_cl = max(d_cl, abs(e["closest_m"] - t["closest_m"]))
            if e["ik_res_m"] is not None: d_ik = max(d_ik, abs(e["ik_res_m"] - t["ik_res_m"]))
            if e["arm_rmse"] is not None: d_arm = max(d_arm, abs(e["arm_rmse"] - t["arm_rmse"]))
    dist_ok = bool(np.isfinite(A["dist"]).all()); w = A["rq"] - A["rq0"]; eq_ = np.abs(O["rq"] - A["rq"]).max(1)
    res = dict(run=name, bouts=len(E), steps=len(A["bout"]), intents=n_int, gates=gates, warped_steps=int((np.abs(w).max(1) > 0).sum()), warp_absmax=float(np.abs(w).max()),
               base_reference_bit_equal=bool(O["base_equal"]), intents_identical=bool(eq), mismatches=mis[:10], intent_time_maxdiff=dt_t, reach_maxdiff=d_reach,
               dist_maxdiff=float(np.abs(O["dist"] - A["dist"]).max()) if dist_ok else None, q_maxdiff=float(eq_.max()), q_rows_over_tol=int((eq_ > TOL["q"]).sum()),
               q_quantiles_99_999=[float(np.quantile(eq_, 0.99)), float(np.quantile(eq_, 0.999))], closest_maxdiff_m=d_cl,
               info_rdq_maxdiff=float(np.abs(O["rdq"] - A["rdq"]).max()), info_target_w_maxdiff=d_tw, info_ik_res_maxdiff=d_ik, info_arm_rmse_maxdiff=d_arm,
               overflow=int(sum(O["overflow"])), standoff_steps=int((A["dist"] < A["home"]).sum()))
    if port is None: res["q_gate"] = "every row vs the eval"; q_ok = res["q_maxdiff"] <= TOL["q"]
    else:
        res["q_gate"] = "A1 (i) same-input vs the float64 port, every row; (ii) vs the eval on >= 99.9% of rows"
        res["A1"] = q_gate_a1(eq_, np.abs(O["rq"] - port["rq"]).max(1)); q_ok = res["A1"]["same_input_ok"] and res["A1"]["eval_rows_ok"]
        res["A1"].update(info_same_input_rdq_maxdiff=float(np.abs(O["rdq"] - port["rdq"]).max()), info_same_input_dist_maxdiff=float(np.abs(O["dist"] - port["dist"]).max()),
                         info_same_input_port_vs_eval_q_maxdiff=float(np.abs(port["rq"] - A["rq"]).max()))
    res["ok"] = bool(dist_ok and eq and O["base_equal"] and res["dist_maxdiff"] <= TOL["dist"] and q_ok and d_cl <= TOL["closest_m"])
    return res

def eval_ik_floor(A, O64, O32, top=6):
    """The rows where the float32-state run misses 1e-4 rad: the eval's own Aim.ik on the IK target from the exact states and on
    the one from the float32-rounded states (same reference arm). Its answer moves as much as AimOp's: the operator, not the port."""
    import fb_record as REC
    from r1gym.aim import Aim
    from r1gym.opponent import FightReplay
    c, cfg, m = REC.setup(); e = np.abs(O32["rq"] - A["rq"]).max(1); rows = [int(r) for r in np.argsort(-e)[:max(top, int((e > TOL["q"]).sum()))] if e[r] > TOL["q"]]; out = []
    for r in rows:
        ep = A["meta"]["episodes"][int(A["bout"][r])]; aim = Aim(m, FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg), "standoff+armwarp_vz", cfg.get("aim"))
        for l in np.flatnonzero(O64["has"][r] & O32["has"][r]):
            ix = list(FO.ARM_IX[l]); q64, _ = aim.ik(A["rq0"][r], FO.LIMBS[l], O64["p_t"][r, l]); q32, _ = aim.ik(A["rq0"][r], FO.LIMBS[l], O32["p_t"][r, l])
            q0 = A["rq0"][r][ix]; lo, hi = np.minimum(np.maximum(aim.lo[ix], q0 - 0.8), q0), np.maximum(np.minimum(aim.hi[ix], q0 + 0.8), q0)
            bound = lambda q: (np.isclose(q, lo, rtol=0, atol=1e-9) | np.isclose(q, hi, rtol=0, atol=1e-9)).astype(int).tolist()
            out.append(dict(row=r, bout=int(A["bout"][r]), limb=FO.LIMBS[l], q_err_rad=float(e[r]), p_t_diff_m=float(np.abs(O64["p_t"][r, l] - O32["p_t"][r, l]).max()),
                            eval_ik_moves_rad=float(np.abs(q64 - q32).max()), port_ik_moves_rad=float(np.abs(O64["qa"][r, l] - O32["qa"][r, l]).max()),
                            eval_ik_vs_port_f64=float(np.abs(q64 - O64["qa"][r, l]).max()), at_bound_exact=bound(q64), at_bound_rounded=bound(q32)))
    return out

def ring(A, O64):
    """The deployed ring (8 slots) vs the 64-slot history run: per-row outputs bit-equal, no live intent overwritten."""
    O = replay(A, torch.float64, slots=8)
    eq = all(np.array_equal(O[k], O64[k]) for k in ("rq", "rdq", "dist"))
    return dict(slots=8, outputs_bit_equal=bool(eq), overflow=int(sum(O["overflow"])), intents_made=int(sum(O["nseq"])), ok=bool(eq and sum(O["overflow"]) == 0))

def apex_rows(A, intents):
    """(row, limb) of each aimed intent's apex step: its first control step with now >= t_apex (the eval's apex log step)."""
    out = []
    for b, ev in enumerate(intents):
        ix = np.flatnonzero(A["bout"] == b); now = A["now"][ix]
        for e in ev:
            if e["gated"]: continue
            k = np.flatnonzero((now >= e["t_apex"]) & (now <= e["t_apex"] + FO.AIM_P["window_tail_s"]))
            if len(k): out.append((int(ix[k[0]]), FO.LIMBS.index(e["limb"])))
    return out

def t_quantile(p, df):
    """Student-t quantile, p > 0.5 (no scipy in this venv): bisection on Simpson's rule of the density."""
    from math import lgamma, exp, pi
    c = exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / np.sqrt(df * pi); n = 4001; w = np.ones(n); w[1:-1:2], w[2:-1:2] = 4, 2
    cdf = lambda x: 0.5 + c * (x / (n - 1) / 3) * (w * (1 + np.linspace(0, x, n) ** 2 / df) ** (-(df + 1) / 2)).sum()
    lo, hi = 0.0, 100.0
    for _ in range(80): mid = (lo + hi) / 2; lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2

def mean_ci(x, level=TOL["jitter_ci"]):
    """Two-sided t interval of the mean of x: [lo, hi] (nan with fewer than 2 values)."""
    if len(x) < 2: return [float("nan")] * 2
    h = t_quantile(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return [float(x.mean() - h), float(x.mean() + h)]

def cluster_ci(x, g, level=TOL["jitter_ci"], n=4000, seed=0):
    """Percentile bootstrap CI of the mean of x, resampling the clusters g (bouts): A2 (ii') (gated), and reported elsewhere."""
    u = np.unique(g); s = np.array([x[g == k].sum() for k in u]); c = np.array([(g == k).sum() for k in u]); ix = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    m = s[ix].sum(1) / c[ix].sum(1); return [float(np.quantile(m, (1 - level) / 2)), float(np.quantile(m, (1 + level) / 2))]

def mixed_mask(A, O_nom, dtype, seed=0, reach="A3"):
    """A1 jitter (iv): odd bouts jittered, even bouts T0 in one op. T0 rows (rq, rdq, dist) bit-equal to the no-jitter run
    O_nom; jittered rows differ (the probe is live)."""
    B = len(A["meta"]["episodes"]); mask = torch.arange(B) % 2 == 1; O = replay(A, dtype, jitter=mask, seed=seed, log_ik=False, reach=reach); t0 = ~mask.numpy()[A["bout"]]
    eq = {k: bool(np.array_equal(O[k][t0], O_nom[k][t0])) for k in ("rq", "rdq", "dist")}; diff = bool(not np.array_equal(O["rq"][~t0], O_nom["rq"][~t0]))
    ti = all(len(a) == len(b) and all(np.array_equal(x["target_r1"], y["target_r1"]) and x["commit_s"] == y["commit_s"] and x["depth_m"] == y["depth_m"] for x, y in zip(a, b))
             for k, (a, b) in enumerate(zip(O["intents"], O_nom["intents"])) if not mask[k])
    return dict(bouts_t0=int((~mask).sum()), bouts_t1=int(mask.sum()), rows_t0=int(t0.sum()), t0_rows_bit_equal=eq, t1_rows_differ=diff, info_t0_intents_equal=bool(ti),
                ok=bool(all(eq.values()) and diff and mask.any() and (~mask).any()))

_JA = None
def _jinit(A):
    global _JA; _JA = A; torch.set_num_threads(1); FO.pin_precision()

def _jrun(args):
    dtype, seed, reach = args; O = replay(_JA, dtype, jitter=True, seed=seed, log_ik=False, debug=True, reach=reach)
    return {k: O[k] for k in ("rq", "rdq", "dist", "p_t", "qa", "rq5", "has", "guard", "s", "conv", "xres", "shift", "intents", "ik_calls_per_step", "wall_ms_per_step") if k in O}

def jitter_runs(A, tasks, workers=1):
    """_jrun over tasks (dtype, generator seed, reach mode) on the recorded states: serial, or a spawn pool of <= 4 workers."""
    assert workers <= 4
    if workers <= 1 or len(tasks) == 1:
        global _JA; _JA = A
        try: return [_jrun(t) for t in tasks]
        finally: _JA = None
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, len(tasks)), initializer=_jinit, initargs=(A,)) as pool: return pool.map(_jrun, tasks, chunksize=1)

def jitter_test(A, O_nom, dtype, seeds=JITTER_SEEDS, workers=1, reach="A3"):
    """Jitter (reach-corrected) vs the nominal aim (O_nom, same dtype) over the generator seeds, plan Amendment A3.2 (reach 'A3', F'
    then the A2.1 guard) or A3.3 (reach 'D'): A1 (i) at every warped step of every seed the IK's x target is the reference palm's own
    and the nominal one, bit for bit (the vz target F' starts from); A2 (ii') the pooled mean of (jitter - nominal) forward distance
    (opponent torso frame) of the warped reference palm at the aimed apexes has a bout-cluster bootstrap 90% CI inside +-5 mm; A2
    (iii') at every warped step of every seed jittered palm x - nominal palm x <= 1e-6 m (float64) / 1e-5 m (float32); A1 (iv)
    mixed_mask (seed 0); 'A3' only: (vi) the mean yz apex spread (jitter vs nominal) >= 50% of the raw jitter's (reach 'raw', same
    seeds), (vii) the per-step change of the commanded arm q (max over the limb's 5 joints) on T1 warped limb-steps: q99.9 <= 1.25x
    and max <= + 0.2 rad of the same on the T0 rows (the nominal run; T0 rows are seed-free by A1 (iv)). Reported: (v) the share of
    jittered intents and limb-steps the guard binds on and the mean s, the share of limb-steps where F' converged within 1e-6 m;
    (vi), (vii) for 'D'; IK calls and wall per step; A1's replaced (ii) t CI and (iii) apex excess per seed, the frozen v0 measure,
    the eval's own warp vs the clip, the IK residual (seed 0), and the raw jitter at seed 0 (the M3A run)."""
    f64 = dtype == torch.float64; tol = TOL["guard_x_f64" if f64 else "guard_x_f32"]
    runs = jitter_runs(A, [(dtype, s, reach) for s in seeds] + [(dtype, s, "raw") for s in seeds], workers); raws = runs[len(seeds):]; runs = runs[:len(seeds)]
    kin = FO.ArmKinematics.from_model(dtype=torch.float64)
    rows = apex_rows(A, A["meta"]["intents"]); r = np.array([x[0] for x in rows], int); l = np.array([x[1] for x in rows], int); m = np.arange(len(r)); ax = list(sum(FO.ARM_IX, ()))
    F = lambda Q: kin.fk(torch.from_numpy(np.ascontiguousarray(Q[:, ax]).reshape(-1, 2, 5))).numpy(); P = lambda Q: F(Q[r])[m, l]; Fx = lambda Q: F(Q)[..., 0]
    pn, p0, pe = P(O_nom["rq"]), P(A["rq0"]), P(A["rq"]); hn = O_nom["has"]; x0n = kin.fk(torch.from_numpy(O_nom["rq5"])).numpy()[..., 0]; xn, x_clip = Fx(O_nom["rq"]), Fx(A["rq0"])
    mx = lambda x: float(x.max()) if len(x) else float("nan"); q = lambda x: [round(float(v), 4) for v in np.quantile(np.abs(x), [0.5, 0.9, 1.0])] if len(x) else None
    yz = lambda pj: np.linalg.norm(pj[:, 1:] - pn[:, 1:], axis=1)
    nxt = np.r_[False, A["bout"][1:] == A["bout"][:-1]][1:, None]   # row t (>= 1) has a predecessor in its bout
    dq = lambda O, h, b=0: np.stack([np.abs(np.diff((O["rq"] - b)[:, list(FO.ARM_IX[k])], axis=0)).max(1) for k in (0, 1)], 1)[nxt & h[1:]]   # b = A["rq0"]: the warp alone
    def ik_x(O):   # A1 (i)
        h = O["has"]; x0 = kin.fk(torch.from_numpy(O["rq5"])).numpy()[..., 0]
        d = dict(warped_steps_jitter=int(h.sum()), warped_steps_nominal=int(hn.sum()), warped_steps_same=bool(np.array_equal(h, hn)), reference_arm_same=bool(np.array_equal(O["rq5"], O_nom["rq5"])),
                 jitter_is_reference_palm_x=bool(np.array_equal(O["p_t"][..., 0][h], x0[h])), nominal_is_reference_palm_x=bool(np.array_equal(O_nom["p_t"][..., 0][hn], x0n[hn])),
                 jitter_equals_nominal=bool(np.array_equal(O["p_t"][..., 0][h & hn], O_nom["p_t"][..., 0][h & hn])))
        d["ok"] = bool(h.any() and all(v for v in d.values() if isinstance(v, bool))); return d
    per, DX, EX, I, YZ, YZr, DQ, SG = [], [], [], [], [], [], [], [0, 0, 0, 0.0, 0]   # SG: T1 warped (row, limb), guarded, guarded with s == 0, sum of s, F' converged
    XR, SH = [], []
    DW = []
    for s, O, Or in zip(seeds, runs, raws):
        i_ = ik_x(O); pj = P(O["rq"]); dx = pj[:, 0] - pn[:, 0]; hw = O["has"] & hn; ex = (Fx(O["rq"]) - xn)[hw]; DX.append(dx); EX.append(ex); YZ.append(yz(pj)); YZr.append(yz(P(Or["rq"]))); DQ.append(dq(O, O["has"])); DW.append(dq(O, O["has"], A["rq0"]))
        it = [t for ev in O["intents"] for t in ev if not t["gated"] and t["guard_warped_steps"]]; I += it; h = O["has"]; g = O["guard"] & h
        SG[0] += int(h.sum()); SG[1] += int(g.sum()); SG[2] += int((O["s"][g] == 0).sum()); SG[3] += float(O["s"][g].sum()); SG[4] += int((O["conv"] & h).sum()); XR.append(O["xres"][h]); SH.append(O["shift"][h])
        ci = mean_ci(dx); a1iii = mx(pj[:, 0] - p0[:, 0]) <= mx(pn[:, 0] - p0[:, 0]) + V1["jitter_excess_m"]
        per.append(dict(seed=s, i_ok=i_["ok"], apex_mean_m=float(dx.mean()), apex_max_m=mx(dx), guard_x_excess_max_m=mx(ex), steps_over_0=int((ex > 0).sum()), intents=len(it),
                        guard_intents=sum(t["guard_steps"] > 0 for t in it), secant_conv_frac=float((O["conv"] & h).sum() / h.sum()) if h.any() else None, yz_spread_ratio=float(YZ[-1].mean() / YZr[-1].mean()),
                        a1_ii_t_ci90_m=ci, a1_ii_ok=bool(-TOL["jitter_ci_m"] < ci[0] and ci[1] < TOL["jitter_ci_m"]), a1_iii_ok=bool(a1iii)))
        if s == seeds[0]: O0, Or0, i0, pj0, dx0 = O, Or, i_, pj, dx
    # A1 (i) on every seed
    i_ = dict(seeds=len(seeds), ok_seeds=sum(p["i_ok"] for p in per), seed0=i0); i_["ok"] = bool(i_["ok_seeds"] == len(seeds))
    # A2 (ii'): pooled apexes, bout clusters
    dxp = np.concatenate(DX); gp = np.tile(A["bout"][r], len(seeds)); ci = cluster_ci(dxp, gp, n=BOOT) if len(dxp) else [float("nan")] * 2; bm = [float(dxp[gp == b].mean()) for b in np.unique(gp)]
    ii = dict(apexes=len(dxp), apexes_per_seed=len(r), bouts=len(bm), seeds=len(seeds), mean_m=float(dxp.mean()) if len(dxp) else float("nan"), cluster_bootstrap_ci90_m=ci, bootstrap=dict(resamples=BOOT, seed=0),
              per_seed_mean_range_m=[min(p["apex_mean_m"] for p in per), max(p["apex_mean_m"] for p in per)], info_bout_means_m=[round(v, 5) for v in bm], info_naive_t_ci90_m=mean_ci(dxp),
              info_share_below_0=float((dxp < 0).mean()), info_share_above_0=float((dxp > 0).mean()), info_absmean_m=float(np.abs(dxp).mean()) if len(dxp) else None, info_min_m=-mx(-dxp))
    ii["ok"] = bool(-TOL["jitter_ci_m"] < ci[0] and ci[1] < TOL["jitter_ci_m"])
    # A2 (iii'): every warped step of every seed
    exa = np.concatenate(EX); iii = dict(tol_m=tol, warped_steps=len(exa), excess_max_m=mx(exa), steps_over_tol=int((exa > tol).sum()), steps_over_0=int((exa > 0).sum()), excess_min_m=-mx(-exa))
    iii["ok"] = bool(len(exa) and exa.max() <= tol)
    # (v): how often the guard binds after F', and how often F' converged
    gi = [t for t in I if t["guard_steps"] > 0]; fr = len(gi) / len(I) if I else float("nan"); xr, sh = np.concatenate(XR), np.concatenate(SH)
    v = dict(jittered_intents=len(I), guard_intents=len(gi), guard_intents_frac=fr, binds_over_half=bool(fr > TOL["guard_bind_report_frac"]), mean_s_guarded_steps=SG[3] / SG[1] if SG[1] else None,
             info_mean_s_per_intent=float(np.mean([t["guard_s_mean"] for t in gi])) if gi else None, t1_warped_limb_steps=SG[0], guarded_limb_steps=SG[1], guarded_steps_frac=SG[1] / SG[0] if SG[0] else None,
             guarded_steps_s0_frac=SG[2] / SG[1] if SG[1] else None, info_guarded_share_of_steps_per_guard_intent=float(np.mean([t["guard_steps"] / t["guard_warped_steps"] for t in gi])) if gi else None,
             secant=dict(tol_m=FO.SECANT_TOL, extra_ik_calls=FO.SECANT_IK if reach == "A3" else 0, converged_limb_steps=SG[4], converged_frac=SG[4] / SG[0] if SG[0] else None,
                         x_residual_q_0_001_50_999_100_m=[float(v_) for v_ in np.quantile(xr, [0, 0.001, 0.5, 0.999, 1])] if len(xr) else None,
                         target_shift_abs_q50_q99_max_m=[float(v_) for v_ in np.quantile(np.abs(sh), [0.5, 0.99, 1])] if len(sh) else None),
             ik_calls_per_step=dict(jitter=runs[0]["ik_calls_per_step"], raw=raws[0]["ik_calls_per_step"]), wall_ms_per_step_16_envs=dict(jitter=float(np.mean([o["wall_ms_per_step"] for o in runs])),
             raw=float(np.mean([o["wall_ms_per_step"] for o in raws]))),
             info_rdq_jitter_minus_nominal_q999_max_seed0=[float(np.quantile(np.abs(O0["rdq"] - O_nom["rdq"]).max(1), 0.999)), float(np.abs(O0["rdq"] - O_nom["rdq"]).max())])
    # (vi) yz diversity kept; (vii) smoothness vs T0
    yzc, yzr = float(np.concatenate(YZ).mean()), float(np.concatenate(YZr).mean())
    vi = dict(yz_apex_spread_mean_m=yzc, raw_yz_apex_spread_mean_m=yzr, ratio=yzc / yzr if yzr else float("nan"), min_ratio=TOL["yz_keep_frac"], per_seed_ratio_range=[min(p["yz_spread_ratio"] for p in per), max(p["yz_spread_ratio"] for p in per)])
    vi["ok"] = bool(yzr > 0 and yzc >= TOL["yz_keep_frac"] * yzr)
    d1, d0 = np.concatenate(DQ), dq(O_nom, hn); Q = lambda x, p_: float(np.quantile(x, p_)) if len(x) else float("nan")
    vii = dict(t1_limb_steps=len(d1), t0_limb_steps=len(d0), t1_q999_rad=Q(d1, 0.999), t0_q999_rad=Q(d0, 0.999), t1_max_rad=mx(d1), t0_max_rad=mx(d0), q999_ratio_max=TOL["smooth_q999_ratio"],
               max_add_rad=TOL["smooth_max_add_rad"], info_q99_t1_t0_rad=[Q(d1, 0.99), Q(d0, 0.99)], info_q50_t1_t0_rad=[Q(d1, 0.5), Q(d0, 0.5)],
               info_warp_delta_change_q999_max_t1_t0_rad=[Q(np.concatenate(DW), 0.999), mx(np.concatenate(DW)), Q(dq(O_nom, hn, A["rq0"]), 0.999), mx(dq(O_nom, hn, A["rq0"]))],
               info_raw_q999_max_rad=[Q(np.concatenate([dq(o, o["has"]) for o in raws]), 0.999), mx(np.concatenate([dq(o, o["has"]) for o in raws]))])
    vii["ok"] = bool(len(d1) and len(d0) and vii["t1_q999_rad"] <= TOL["smooth_q999_ratio"] * vii["t0_q999_rad"] and vii["t1_max_rad"] <= vii["t0_max_rad"] + TOL["smooth_max_add_rad"])
    iv = mixed_mask(A, O_nom, dtype, reach=reach)
    gated = ("A1_i_ik_x_target", "ii_cluster_ci", "iii_guard_every_step", "A1_iv_mixed_mask") + (("vi_yz_kept", "vii_smoothness") if reach == "A3" else ())
    res = dict(dtype=str(dtype).split(".")[-1], reach=reach, seeds=list(seeds), apexes=len(r), gated=list(gated),
               A3={"A1_i_ik_x_target": i_, "ii_cluster_ci": ii, "iii_guard_every_step": iii, "A1_iv_mixed_mask": iv, "v_guard_binding": v, "vi_yz_kept": vi, "vii_smoothness": vii},
               per_seed=per, superseded_by_A2=dict(A1_ii_t_ci_pass_seeds=sum(p["a1_ii_ok"] for p in per), A1_iii_apex_excess_pass_seeds=sum(p["a1_iii_ok"] for p in per), seed0_A1_ii_t_ci90_m=per[0]["a1_ii_t_ci90_m"],
                                               seed0_A1_iii_margin_m=mx(pj0[:, 0] - p0[:, 0]) - mx(pn[:, 0] - p0[:, 0])))
    # seed 0 (as the A1 run): v0, the eval's warp vs the clip, the IK residual, the intents' draws
    resid = lambda o: np.linalg.norm(o["p_t"][r, l] - kin.fk(torch.from_numpy(o["qa"][r])).numpy()[m, l], axis=1); rj, rn = resid(O0), resid(O_nom)
    reached = lambda rr, pp: float(np.abs(pp[rr <= 1e-3, 0] - p0[rr <= 1e-3, 0]).max(initial=0)); tj = [t for ev in O0["intents"] for t in ev]; tn = [t for ev in O_nom["intents"] for t in ev]
    dtgt = np.array([np.array(a["target_r1"]) - np.array(b["target_r1"]) for a, b in zip(tj, tn)])
    res.update(v0_x_jitter_vs_nominal_maxabs_m=mx(np.abs(dx0)), v0_apexes_jitter_vs_nominal_over_1mm=int((np.abs(dx0) > V0["jitter_x_m"]).sum()), x_jitter_vs_nominal_q50_q90_max_m=q(dx0),
               eval_nominal_x_vs_clip=dict(over_1mm=int((np.abs(pe[:, 0] - p0[:, 0]) > 1e-3).sum()), q50_q90_max_m=q(pe[:, 0] - p0[:, 0]), min_m=-mx(-(pe[:, 0] - p0[:, 0])), max_m=mx(pe[:, 0] - p0[:, 0])),
               eval_every_warped_step_x_over_clip_max_m=mx((Fx(A["rq"]) - x_clip)[hn]),
               ik_reached=dict(nominal=int((rn <= 1e-3).sum()), jitter=int((rj <= 1e-3).sum()), x_vs_clip_maxabs_where_reached_m=max(reached(rn, pn), reached(rj, pj0))),
               info_target_offset_absmax=np.abs(dtgt).max(0).round(4).tolist(), info_commit_s=[round(min(t["commit_s"] for t in tj), 3), round(max(t["commit_s"] for t in tj), 3)],
               info_depth_m=[round(min(t["depth_m"] for t in tj), 3), round(max(t["depth_m"] for t in tj), 3)],
               info_dist_jitter_minus_nominal_range_m=[float((O0["dist"] - O_nom["dist"]).min()), float((O0["dist"] - O_nom["dist"]).max())])
    if len(r): res["ik_reached"]["residual_q50_max_m"] = dict(nominal=[round(float(np.median(rn)), 4), round(float(rn.max()), 4)], jitter=[round(float(np.median(rj)), 4), round(float(rj.max()), 4)])
    po = P(Or0["rq"]); do = po[:, 0] - pn[:, 0]; eo = (Fx(Or0["rq"]) - xn)[Or0["has"] & hn]   # the raw (pre-A2) jitter at seed 0: the M3A run's numbers
    res["info_raw_jitter_seed0"] = dict(apex_mean_m=float(do.mean()), a1_t_ci90_m=mean_ci(do), cluster_ci90_m=cluster_ci(do, A["bout"][r], n=BOOT), a1_iii_margin_m=mx(po[:, 0] - p0[:, 0]) - mx(pn[:, 0] - p0[:, 0]),
                                        every_step_x_excess_max_m=mx(eo), yz_at_apex_mean_m=float(yz(po).mean()) if len(r) else None,
                                        rdq_jitter_minus_nominal_q999_max=[float(np.quantile(np.abs(Or0["rdq"] - O_nom["rdq"]).max(1), 0.999)), float(np.abs(Or0["rdq"] - O_nom["rdq"]).max())])
    res["ok"] = bool(all(res["A3"][k]["ok"] for k in gated)); res["fprime_fail_triggers_D"] = bool(reach == "A3" and not (ii["ok"] and vi["ok"] and vii["ok"])); return res

def params_ok(A):
    """The recorded Aim's parameters (merged config) = AIM_P for zone 'head'."""
    p, q = A["meta"]["aim_p"], FO.AIM_P
    return p["zone"] == "head" and tuple(p["targets"]["head"]) == q["target"] and p["arm"] == q["arm"] and p["standoff"] == q["standoff"] and \
        all(p[k] == q[k] for k in ("lead_s", "commit_s", "window_tail_s", "upright_gate_deg", "gate_after_touch_s"))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--aimrec", default=""); ap.add_argument("--rec", default="")
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args(); assert a.workers <= 4
    FO.pin_precision(); assert FO.precision_ok(); t0 = time.time()
    if a.aimrec and Path(a.aimrec).exists(): A = load(a.aimrec); print("aim recording", a.aimrec, flush=True)
    else:
        A = record(workers=a.workers, rec=a.rec or None)
        if a.aimrec: Path(a.aimrec).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.aimrec, **{k: v for k, v in A.items()}); A = load(a.aimrec)
    mt = A["meta"]; res = dict(tol=TOL, seeds=mt["seeds"], role=mt["role"], aim_mode=mt["aim_mode"], steps=len(A["bout"]), record_wall_s=mt["wall_s"],
                               same_as_m2_recording=mt.get("same_as_m2_recording"), falls=sum(r["fell_s"] is not None for r in mt["rows"]),
                               opp_fell=sum(bool(r["opp_fell"]) for r in mt["rows"]), params_equal=params_ok(A))
    print(json.dumps(res), flush=True)
    res["FK"] = g0_4_fk(A); print("FK", json.dumps(res["FK"]), flush=True)
    O64 = replay(A, torch.float64, debug=True); res["AimOp f64"] = compare(A, O64, "float64"); print("AimOp f64", json.dumps(res["AimOp f64"]), flush=True)
    O32 = replay(A, torch.float32, debug=True); O32s = replay(A, torch.float64, round_to=torch.float32, debug=True)
    res["AimOp f32"] = compare(A, O32, "float32 states and outputs, float64 aim arithmetic (as deployed)", port=O32s); print("AimOp f32", json.dumps(res["AimOp f32"]), flush=True)
    res["info AimOp f32 arithmetic"] = compare(A, replay(A, torch.float32, compute=torch.float32), "float32 states, outputs and arithmetic", port=O32s); print("info f32 arithmetic", json.dumps(res["info AimOp f32 arithmetic"]), flush=True)
    res["f32 floor: eval Aim.ik"] = eval_ik_floor(A, O64, O32); print("eval ik floor", json.dumps(res["f32 floor: eval Aim.ik"]), flush=True)
    res["ring"] = ring(A, O64); print("ring", json.dumps(res["ring"]), flush=True)
    res["jitter f64"] = jitter_test(A, O64, torch.float64, workers=a.workers); print("jitter f64", json.dumps(res["jitter f64"]), flush=True)
    res["jitter f32"] = jitter_test(A, O32, torch.float32, workers=a.workers); print("jitter f32", json.dumps(res["jitter f32"]), flush=True)
    jk, res["adopted"] = ("jitter f64", "jitter f32"), None
    if any(res[k]["fprime_fail_triggers_D"] for k in jk):   # A3.3: fallback D, re-gated
        res["jitter D f64"] = jitter_test(A, O64, torch.float64, workers=a.workers, reach="D"); print("jitter D f64", json.dumps(res["jitter D f64"]), flush=True)
        res["jitter D f32"] = jitter_test(A, O32, torch.float32, workers=a.workers, reach="D"); print("jitter D f32", json.dumps(res["jitter D f32"]), flush=True)
        jk = ("jitter D f64", "jitter D f32"); res["adopted"] = "D" if all(res[k]["ok"] for k in jk) else None
    elif all(res[k]["ok"] for k in jk): res["adopted"] = "F'"
    res["ok"] = dict(params=res["params_equal"], FK=res["FK"]["ok"], AimOp_f64=res["AimOp f64"]["ok"], AimOp_f32=res["AimOp f32"]["ok"], ring=res["ring"]["ok"],
                     jitter_f64=res[jk[0]]["ok"], jitter_f32=res[jk[1]]["ok"])
    # the two frozen gates A1 replaced, as measured (not in 'ok'); eval_explains: the eval's own operator does the same
    a32, fl = res["AimOp f32"], res["f32 floor: eval Aim.ik"]; mv = {}
    for x in fl: mv[x["row"]] = max(mv.get(x["row"], 0.0), x["eval_ik_moves_rad"])
    res["superseded_by_A1"] = dict(
        f32_warped_q_every_row_vs_eval=dict(tol=V0["q"], q_maxdiff=a32["q_maxdiff"], rows_over=a32["q_rows_over_tol"], steps=a32["steps"], passes=a32["q_maxdiff"] <= V0["q"],
                                            eval_explains=bool(len(mv) == a32["q_rows_over_tol"] and all(mv[x["row"]] >= x["q_err_rad"] - 1e-6 for x in fl))),
        jitter_x_at_apex_1mm={d: dict(tol=V0["jitter_x_m"], maxabs_m=res[f"jitter {d}"]["v0_x_jitter_vs_nominal_maxabs_m"], passes=res[f"jitter {d}"]["v0_x_jitter_vs_nominal_maxabs_m"] <= V0["jitter_x_m"],
                                      eval_explains=bool(res[f"jitter {d}"]["eval_nominal_x_vs_clip"]["over_1mm"] > 0 and res[f"jitter {d}"]["ik_reached"]["nominal"] == 0)) for d in ("f64", "f32")})
    res["wall_s"] = round(time.time() - t0, 1); print(json.dumps(res["ok"]), json.dumps(res["superseded_by_A1"]), res["wall_s"], flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1, default=str)
    sys.exit(0 if all(res["ok"].values()) else 1)

if __name__ == "__main__": main()
