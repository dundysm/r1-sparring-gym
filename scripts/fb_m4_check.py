"""M4 checks of the defender's observation, action, latency, fall and DR modules (M2V2_PLAN_v3 3 M4, G0-5..G0-10; gpu/fb_obs.py,
fb_actions.py, fb_events.py, fb_dr.py) on CPU against the eval's own code:
  record  the 16 record bouts (seeds 60000-60015, role 'record': scripts/fb_record.py's episodes, aim standoff+armwarp_vz, DR, delays
          as drawn, no pushes, 20 s) through eval_opponent.run for G and for P1's H(A3s)/M3s (p1_eval.py M3sHybrid, --p1; keypoints
          truth), with read-only hooks: R1HitEnv subclass (the clean 83 of _raw_obs, the step's action in, the applied action, the
          joint targets, act_delay, the fall flag, the post-step state and both pelvises' derived quantities as the eval reads them),
          DelayedNoisyObs subclass (the delayed row and the exact noise term, same arithmetic and rng draws), KeypointSource
          subclass (undelayed and delivered blocks), the controller's act (output; its input is asserted = the delivered obs). Each
          bout also runs without hooks (rows identical, asserted); G's states must equal M2's recording (--rec) bit for bit.
  G0-5    keypoint block (EvalActorObs, keypoint part) vs KeypointSource('truth', delay=k), k in {0, 1, 2}, on 2000 recorded states
          (8 segments x 250, each with a reset at its start and in its middle: pre-fill included), and vs the eval's delivered
          block on every H step: positions, masks and yaw pair bit-equal (float32), velocities <= 1e-6; glove-elbow 0.194862 m.
  G0-6    EvalActorObs on every recorded step (32 bouts, batched): the eval's own noise replayed -> the delivered 111 (H) / 83 (G)
          bit for bit; noise 0 -> r1gym's DelayedNoisyObs(k, zeros) on the same clean rows bit for bit; out(t) = clean(max(0, t - k))
          in all 111 dims; zero k switches. Reported: its own uniform noise (bounds, keypoints exactly 0).
  G0-6b   B3 in the replica (two-R1 MjModel of the eval, recorded bouts as previous episodes, 64 batched envs): 64 resets (after a
          defender fall, after opponent knockdowns, the rest at random steps) in mjlab's order (fb_events.B3_ORDER): every ring row
          at the first obs = the fresh row (R1HitEnv._raw_obs + KeypointSource.reset on the placed, forwarded state), keypoints =
          those of the placed opponent, k and act_delay = the event's draw, no row of the previous episode, zero velocities,
          a_prev 0. Negative controls (each must fail): eager pre-fill at obs reset, latency drawn after the manager resets, no
          pre-fill, obs computed before the forward.
  G0-7    HybridDelayedAction on every recorded step: applied = the eval's last_a (a(t-1) when delayed, 0 at the first delayed step),
          targets = d.ctrl bit for bit, the next obs' actions slot = applied; the term reads no env obs (stub env + source scan).
  G0-8    fell_eval / opp_down_now vs R1.fallen and the EO:86 test on the recorded post-step states (>= 1e4) and 10,000 random
          states; the non-sticky flag rebuilds EO's both-up intervals and its first opponent fall exactly.
  G0-9    DR: the writer with the eval's draws (sample_np, numpy seeds = the 16 bouts' env seeds and 0-199) = R1HitEnv.
          _dr_sample_into_model field for field, act_delay included; torch draws inside the ranges and reaching their ends; only
          FIELDS change, only on defender entries; dof_invweight0, body_invweight0, body_subtreemass nominal; efc_R / efc_D /
          efc_diagApprox on a contact state = the eval's DR model's (exactly as the eval); mj_setConst (negative control) changes them.
  G0-10   fb_actions.compose with the same ONNX actors = P1's M3sHybrid.act on every recorded H obs, bit for bit; m3s_input = g_in.
Run in the repo: python scripts/fb_m4_check.py --out m4_check.json --m4rec m4rec.npz --rec fb_rec_60000-60015.npz --p1 p1_eval.py [--workers 4]"""
import sys, json, time, argparse, copy, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
import fb_opponent as FO, fb_obs as OB, fb_actions as AC, fb_events as EV, fb_dr as DR

TOL = dict(kp_vel=1e-6, glove_elbow_m=0.194862, glove_elbow_tol=5e-7, g05_states=2000, g05_segments=8, resets=64, g08_states=10000, g08_random=10000,
           dr_seeds=200, range_edge_frac=0.01)
SEEDS = tuple(range(60000, 60016)); TRAIN = FO.TRAIN; CTRLS = ("G", "H")
G_SPEC = "guard_fw_r=results/policy_mjlab_guard_footwork_r/policy.onnx"; H_SPEC = "guard_fw_r+a3s"

def load_p1(path):
    spec = importlib.util.spec_from_file_location("p1_eval", str(path)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def setup(ctrl, p1=None):
    """(controller, cfg, model) as the record bouts run them: G, or P1's H(A3s)/M3s with its scale set from the env's."""
    import fb_record as REC
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import build_spec, action_scale
    aim = REC.AIM
    if ctrl == "G": lab, p = G_SPEC.split("="); c = load_controller(f"{lab}={ROOT / p}")
    else: c = load_p1(p1).make_ctrl(H_SPEC, "m3s")
    cfg = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": aim}
    if ctrl == "H": c.set_scale(action_scale(cfg), float(cfg.get("action_clip", 4.0)))
    return c, cfg, build_spec(opponent=True, cfg=cfg).compile()

# ---- recording (read-only hooks on the eval's own bout)
def _hooks(S):
    import r1gym.env as RE, eval_opponent as EO
    base_env, base_obs, base_kp = RE.R1HitEnv, RE.DelayedNoisyObs, EO.KeypointSource
    class RecObs(base_obs):
        def __call__(self, obs):   # DelayedNoisyObs.__call__'s arithmetic and rng draws, with the noise term kept
            self.buf.append(obs); o = self.buf[0]; n = (self.rng.uniform(-1, 1, o.shape) if self.uniform else self.rng.normal(0, 1, o.shape)) * self.std
            S["sens_o"].append(np.array(o, np.float32)); S["sens_n"].append(np.array(n, np.float64)); return o + n
    class RecEnv(base_env):
        def _raw_obs(self):
            r = super()._raw_obs(); S["raw"].append(r.copy()); return r
        def reset(self, *a, **k):
            o = super().reset(*a, **k); d = self.d
            S["reset"] = dict(act_delay=int(self.act_delay), k=int(self.sensor.k), scale=self.scale.tolist(), clip=self.clip, home=self.home.tolist(), alpha=self.alpha)
            S["qpos"].append(d.qpos.copy()); S["qvel"].append(d.qvel.copy()); return o
        def step(self, a):
            S["a_in"].append(np.array(a, np.float32)); out = super().step(a); d, r = self.d, self.r; b = min(self.opp.bodies)
            S["last_a"].append(self.last_a.copy()); S["ctrl"].append(d.ctrl[r.act].copy()); S["fallen"].append(bool(out[3]["fallen"]))
            S["qpos"].append(d.qpos.copy()); S["qvel"].append(d.qvel.copy())
            S["post"].append(np.concatenate([d.xpos[r.pel], d.xmat[r.pel], d.xquat[r.pel], d.xpos[b], d.xmat[b], d.xquat[b]])); return out
    class RecKP(base_kp):
        def block(self, d):
            b = super().block(d); S["kp_block"].append(b.copy()); S["kp_qpos"].append(d.qpos.copy()); return b
        def reset(self, d): o = super().reset(d); S["kp_out"].append(o.copy()); return o
        def observe(self, d): o = super().observe(d); S["kp_out"].append(o.copy()); return o
    def install(): RE.DelayedNoisyObs, EO.R1HitEnv, EO.KeypointSource = RecObs, RecEnv, RecKP
    def remove(): RE.DelayedNoisyObs, EO.R1HitEnv, EO.KeypointSource = base_obs, base_env, base_kp
    return install, remove

# 'post' columns: defender pelvis xpos, xmat, xquat, then the opponent pelvis' (as read after mj_step: one substep stale)
PO = dict(pz=2, pup=11, pquat=slice(12, 16), oz=18, oup=27, oquat=slice(28, 32))
KEYS = ("raw", "sens_o", "sens_n", "kp_block", "kp_out", "kp_qpos", "qpos", "qvel", "a_in", "a_out", "obs_in", "last_a", "ctrl", "fallen", "post")

def work(args):
    jobs, seconds, p1 = args
    torch.set_num_threads(1)
    import eval_opponent as EO, fb_record as REC
    from fb_seeds import check
    out = []
    for ctrl, seeds in jobs:
        check(seeds, "record", TRAIN); c, cfg, m = setup(ctrl, p1); kp = "truth" if ctrl == "H" else None
        for ep in REC.episodes(seeds, seconds):
            plain = EO.run(c, [ep], m, cfg, seconds, threads=1, dr=True, meter=False, falls=True, keypoints=kp)[0]
            S = {k: [] for k in KEYS}; install, remove = _hooks(S); act0 = c.act
            def act(obs, S=S): a = act0(obs); S["obs_in"].append(np.array(obs, np.float64)); S["a_out"].append(np.array(a, np.float32)); return a
            c.act = act; install()
            try: row = EO.run(c, [ep], m, cfg, seconds, threads=1, dr=True, meter=False, falls=True, keypoints=kp)[0]
            finally: remove(); c.act = act0
            assert json.dumps(row, sort_keys=True, default=str) == json.dumps(plain, sort_keys=True, default=str), f"the hooks changed bout {ctrl} {ep['seed']}"
            T = len(S["a_in"]); assert len(S["raw"]) == len(S["sens_o"]) == len(S["qpos"]) == T + 1 and len(S["a_out"]) == T
            assert all(x.shape == (1, 24) and np.array_equal(y, x[0]) for x, y in zip(S["a_out"], S["a_in"])), "the step's action is not the controller's"
            D = np.stack(S["sens_o"]).astype(np.float64) + np.stack(S["sens_n"])
            if kp:
                assert len(S["kp_out"]) == T + 1 and all(np.array_equal(q, p) for q, p in zip(S["kp_qpos"], S["qpos"])), "keypoints read another state"
                D = np.concatenate([D, np.stack(S["kp_out"]).astype(np.float64)], 1)
            assert np.array_equal(np.concatenate(S["obs_in"]), D[:T]), "the controller's input is not the delivered obs"
            A = {k: (np.concatenate if k == "a_out" else np.stack)(S[k]) for k in KEYS if S[k] and k not in ("kp_qpos", "obs_in", "a_in")}
            out.append((ctrl, ep, row, S["reset"], A))
    return out

def record(seeds=SEEDS, seconds=20.0, workers=4, p1=None, rec=None, ctrls=CTRLS):
    from fb_seeds import check
    check(seeds, "record", TRAIN); assert workers <= 4; t0 = time.time()
    jobs = [(c, [s]) for c in ctrls for s in seeds]; sl = [jobs[i::workers] for i in range(workers) if jobs[i::workers]]
    if len(sl) == 1: parts = [work((sl[0], seconds, p1))]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(sl)) as pool: parts = pool.map(work, [(s, seconds, p1) for s in sl], chunksize=1)
    bouts = sorted((b for p in parts for b in p), key=lambda b: (ctrls.index(b[0]), b[1]["seed"]))
    R = dict(meta=dict(seeds=list(seeds), role="record", seconds=seconds, ctrls=list(ctrls), bouts=[dict(ctrl=b[0], episode=b[1], row=b[2], reset=b[3], steps=len(b[4]["a_out"])) for b in bouts],
                       wall_s=round(time.time() - t0, 1)))
    for i, b in enumerate(bouts):
        for k, v in b[4].items(): R[f"{i}/{k}"] = v
    if rec and "G" in ctrls:   # G's bouts are M2's recording: every state before each act bit-equal
        with np.load(rec, allow_pickle=False) as z: Q, Bm, m2 = z["qpos"], z["bout"], json.loads(str(z["meta"]))
        assert [e["seed"] for e in m2["episodes"]] == list(seeds), "not the M2 recording's bouts"
        same = [bool(np.array_equal(Q[Bm == j], R[f"{i}/qpos"][:-1])) for j, i in enumerate(i for i, b in enumerate(bouts) if b[0] == "G")]
        R["meta"]["same_as_m2_recording"] = all(same); assert all(same), f"G's states differ from M2's recording: {same}"
    return R

def save(R, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **{k: (np.array(json.dumps(v, default=str)) if k == "meta" else v) for k, v in R.items()})

def load(path):
    with np.load(path, allow_pickle=False) as z: R = {k: z[k] for k in z.files}
    R["meta"] = json.loads(str(R["meta"])); return R

def bouts(R, ctrl=None):
    """[(index, bout meta, {key: array})] in recording order."""
    return [(i, b, {k.split("/", 1)[1]: v for k, v in R.items() if k.startswith(f"{i}/")}) for i, b in enumerate(R["meta"]["bouts"]) if ctrl in (None, b["ctrl"])]

def model_cfg():
    """The eval's two-R1 model under G's cfg (H's is the same: G's legs, cfg and home), its R1Robot and the 6 keypoint bodies."""
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import build_spec, R1Robot
    import fb_record as REC
    lab, p = G_SPEC.split("="); cfg = {**controller_cfg(load_controller(f"{lab}={ROOT / p}")), "opponent_mode": "policy", "opponent_aim": REC.AIM}
    m = build_spec(opponent=True, cfg=cfg).compile(); return m, cfg, R1Robot(m, cfg), [m.body(n).id for n in OB.body_names()]

def poses(m, ids6, Q):
    """Fresh kinematics (what mjlab's forward leaves, and KeypointSource's twin computes) of qpos rows Q -> xpos [n, 6, 3], xquat [n, 6, 4]."""
    import mujoco
    d = mujoco.MjData(m); X, W = np.zeros((len(Q), 6, 3)), np.zeros((len(Q), 6, 4))
    for i, q in enumerate(Q): d.qpos[:] = q; mujoco.mj_kinematics(m, d); X[i], W[i] = d.xpos[ids6], d.xquat[ids6]
    return torch.from_numpy(X), torch.from_numpy(W)

def _eq(a, b): return bool(np.array_equal(np.asarray(a), np.asarray(b)))

# ---- G0-5
def g0_5(R, m, ids6, pipe=None):
    from r1gym.keypoints import KeypointSource
    import mujoco
    B = bouts(R); L = TOL["g05_states"] // TOL["g05_segments"]; rng = np.random.default_rng(0)
    dn = [i for i, b, _ in B if b["row"]["opp_fell"] or b["row"]["fell_s"] is not None]; rest = [i for i, b, _ in B if i not in dn]
    pick = (dn + list(rng.permutation(rest)))[:TOL["g05_segments"]]; segs = []
    for i in pick:
        T = B[i][1]["steps"] + 1; lo = 0 if T <= L else int(rng.integers(0, T - L + 1))
        if B[i][1]["row"]["fell_s"] is not None: lo = max(0, T - L)   # the last 250 states up to the fall
        segs.append((int(i), lo))
    N = 3 * len(segs); lat = EV.Latency(N); lat.k[:] = torch.tensor([k for _ in segs for k in (0, 1, 2)])
    obs = OB.EvalActorObs(N, lat, np.zeros(OB.OBS_DIM), out_dtype=torch.float64); srcs = [KeypointSource(m, "truth", delay=k) for _ in segs for k in (0, 1, 2)]
    QS = [B[i][2]["qpos"][lo:lo + L] for i, lo in segs]; P_all = [poses(m, ids6, Q) for Q in QS]; d = mujoco.MjData(m)
    e = dict(pos=0, mask=0, yaw=0, vel=0.0, n=0); ge = []
    for t in range(L):
        if t in (0, L // 2): obs.reset(torch.arange(N))
        X = torch.stack([P_all[s][0][t] for s in range(len(segs)) for _ in range(3)]); W = torch.stack([P_all[s][1][t] for s in range(len(segs)) for _ in range(3)])
        P, y, v = OB.kp_state(X, W); out = obs.compute(torch.zeros(N, 83), P, y, v, noise=torch.zeros(N, OB.OBS_DIM, dtype=torch.float64))[:, 83:].float().numpy()
        ge += [torch.linalg.vector_norm(P[:, 0] - P[:, 2], dim=1), torch.linalg.vector_norm(P[:, 1] - P[:, 3], dim=1)]
        for j, s in enumerate(srcs):
            d.qpos[:] = QS[j // 3][t]; ref = s.reset(d) if t in (0, L // 2) else s.observe(d)
            e["pos"] += int((out[j, OB.POS] != ref[OB.POS]).sum()); e["mask"] += int((out[j, OB.MASK] != ref[OB.MASK]).sum()); e["yaw"] += int((out[j, OB.YAW] != ref[OB.YAW]).sum())
            e["vel"] = max(e["vel"], float(np.abs(out[j, OB.VEL].astype(np.float64) - ref[OB.VEL]).max())); e["n"] += 1
    ge = torch.cat(ge).numpy(); res = dict(states=len(segs) * L, segments=[dict(bout=i, ctrl=B[i][1]["ctrl"], seed=B[i][1]["episode"]["seed"], start=lo) for i, lo in segs], ks=[0, 1, 2],
                                       rows_compared=e["n"], pos_mismatch=e["pos"], mask_mismatch=e["mask"], yaw_mismatch=e["yaw"], vel_maxdiff=e["vel"],
                                       glove_elbow_min=float(ge.min()), glove_elbow_max=float(ge.max()))
    if pipe is not None: res["eval_bouts"] = pipe["kp_vs_eval"]
    ev = res.get("eval_bouts", dict(pos_mismatch=0, mask_mismatch=0, yaw_mismatch=0, vel_maxdiff=0.0))
    res["ok"] = bool(e["pos"] == e["mask"] == e["yaw"] == 0 and e["vel"] <= TOL["kp_vel"] and max(abs(ge.max() - TOL["glove_elbow_m"]), abs(ge.min() - TOL["glove_elbow_m"])) <= TOL["glove_elbow_tol"]
                     and ev["pos_mismatch"] == ev["mask_mismatch"] == ev["yaw_mismatch"] == 0 and ev["vel_maxdiff"] <= TOL["kp_vel"] and res["states"] >= TOL["g05_states"])
    return res

# ---- the batched pipeline over every recorded step (G0-5 eval part, G0-6, G0-7)
def pipeline(R, m, cfg, r, ids6, noise="eval"):
    """EvalActorObs + HybridDelayedAction over all bouts at once: our proprio83 (command from the eval row) and keypoints from the
    recorded states, the recorded controller outputs. noise: 'eval' (the recorded noise terms) or 'zero'."""
    from r1gym.sensors import DelayedNoisyObs
    B = bouts(R); N = len(B); Ts = [b["steps"] for _, b, _ in B]; Tm = max(Ts) + 1; rs = B[0][1]["reset"]
    assert all(b["reset"]["scale"] == rs["scale"] and b["reset"]["home"] == rs["home"] and b["reset"]["alpha"] == 0.0 for _, b, _ in B)
    lat = EV.Latency(N); lat.k[:] = torch.tensor([b["reset"]["k"] for _, b, _ in B]); lat.act_delay[:] = torch.tensor([bool(b["reset"]["act_delay"]) for _, b, _ in B])
    obs = OB.EvalActorObs(N, lat, OB.noise_halfwidths(cfg), out_dtype=torch.float64); act = AC.HybridDelayedAction(N, lat, rs["home"], rs["scale"], rs["clip"])
    obs.reset(torch.arange(N)); act.reset(torch.arange(N)); home = torch.tensor(rs["home"], dtype=torch.float64)
    PW = [poses(m, ids6, A["qpos"]) for _, _, A in B]; pad = lambda x, t: x[min(t, len(x) - 1)]
    fq, qa, va = r.free_q, r.qadr, r.vadr; fv = r.free_v
    st = dict(proprio=0, out_eval83=0, out_eval_kp=0, kp=dict(pos_mismatch=0, mask_mismatch=0, yaw_mismatch=0, vel_maxdiff=0.0, steps=0), applied=0, targets=0, slot=0, first_delayed=[], steps=0, kp_steps=0)
    clean = np.zeros((N, Tm, OB.OBS_DIM), np.float32); out_all = np.zeros((N, Tm, OB.OBS_DIM))
    for t in range(Tm):
        if t > 0:
            a = torch.from_numpy(np.stack([pad(A["a_out"], t - 1) for _, _, A in B])); tg = act.process(a).numpy()
            for j, (_, b, A) in enumerate(B):
                if t - 1 < Ts[j]:
                    st["applied"] += int(not _eq(act.applied[j].numpy(), A["last_a"][t - 1])); st["targets"] += int(not _eq(tg[j], A["ctrl"][t - 1]))
                    st["slot"] += int(not _eq(A["raw"][t][OB.ACT], act.applied[j].numpy()))
                    if t == 1 and b["reset"]["act_delay"]: st["first_delayed"].append(bool(float(act.applied[j].abs().max()) == 0.0))
        Q = np.stack([pad(A["qpos"], t) for _, _, A in B]); V = np.stack([pad(A["qvel"], t) for _, _, A in B]); raw = np.stack([pad(A["raw"], t) for _, _, A in B])
        p = OB.proprio83(torch.from_numpy(Q[:, fq + 3:fq + 7]), torch.from_numpy(V[:, fv + 3:fv + 6]), torch.from_numpy(Q[:, qa]), torch.from_numpy(V[:, va]), home,
                         torch.from_numpy(raw[:, OB.CMD].astype(np.float64)), torch.full((N,), t), act.applied)
        X = torch.stack([pad(P[0], t) for P in PW]); W = torch.stack([pad(P[1], t) for P in PW]); P_T, y, v = OB.kp_state(X, W)
        nz = torch.zeros(N, OB.OBS_DIM, dtype=torch.float64)
        if noise == "eval": nz[:, :83] = torch.from_numpy(np.stack([pad(A["sens_n"], t) for _, _, A in B]))
        o = obs.compute(p, P_T, y, v, noise=nz).numpy(); clean[:, t] = obs.clean.numpy(); out_all[:, t] = o
        for j, (_, b, A) in enumerate(B):
            if t > Ts[j]: continue
            st["steps"] += 1; st["proprio"] += int(not _eq(p[j].numpy(), A["raw"][t]))
            if noise == "eval":
                st["out_eval83"] += int(not _eq(o[j, :83], A["sens_o"][t].astype(np.float64) + A["sens_n"][t]))
                if "kp_out" in A:
                    ref, oo = A["kp_out"][t], o[j, 83:].astype(np.float32); st["kp_steps"] += 1; k = st["kp"]; k["steps"] += 1
                    st["out_eval_kp"] += int(not _eq(o[j, 83:], ref.astype(np.float64)))
                    k["pos_mismatch"] += int((oo[OB.POS] != ref[OB.POS]).sum()); k["mask_mismatch"] += int((oo[OB.MASK] != ref[OB.MASK]).sum())
                    k["yaw_mismatch"] += int((oo[OB.YAW] != ref[OB.YAW]).sum()); k["vel_maxdiff"] = max(k["vel_maxdiff"], float(np.abs(oo[OB.VEL].astype(np.float64) - ref[OB.VEL]).max()))
    res = dict(bouts=N, steps=st["steps"], proprio83_vs_raw_obs_mismatch=st["proprio"], k_switches=int(obs.k_switches), act_switches=int(act.switches))
    if noise == "eval":
        res.update(out_vs_eval_delivered83_mismatch=st["out_eval83"], out_vs_eval_delivered_kp_mismatch=st["out_eval_kp"], kp_steps=st["kp_steps"], kp_vs_eval=st["kp"],
                   applied_vs_eval_last_a_mismatch=st["applied"], targets_vs_eval_ctrl_mismatch=st["targets"], eval_actions_slot_vs_applied_mismatch=st["slot"],
                   delayed_bouts=int(lat.act_delay.sum()), first_delayed_step_zero=st["first_delayed"])
    else:   # r1gym's DelayedNoisyObs(k, zeros) on the same clean rows; and out(t) = clean(max(0, t - k)) in every dim
        dno = lag = 0
        for j, (_, b, _) in enumerate(B):
            k = b["reset"]["k"]; D = DelayedNoisyObs(k, np.zeros(OB.OBS_DIM), np.random.default_rng(0), uniform=True); D.reset(clean[j, 0])
            for t in range(Ts[j] + 1):
                ref = D(clean[j, t]); dno += int(not _eq(out_all[j, t], ref)); lag += int(not _eq(out_all[j, t], clean[j, max(0, t - k)].astype(np.float64)))
        res.update(out_vs_DelayedNoisyObs_mismatch=dno, out_vs_clean_t_minus_k_mismatch=lag, ks=sorted({b["reset"]["k"] for _, b, _ in B}))
    return res

def own_noise(cfg, n=4096, seed=0):
    """EvalActorObs's own uniform noise: inside the half-widths, exactly 0 where the half-width is 0 (command, phase, actions, keypoints)."""
    lat = EV.Latency(n); hw = OB.noise_halfwidths(cfg); obs = OB.EvalActorObs(n, lat, hw, out_dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    z = torch.zeros(n, 83); P = torch.zeros(n, 5, 3, dtype=torch.float64); obs.compute(z, P, torch.zeros(n, dtype=torch.float64), torch.zeros(n, 5, dtype=torch.bool))
    o = obs.compute(z, P, torch.zeros(n, dtype=torch.float64), torch.zeros(n, 5, dtype=torch.bool)).numpy(); H = np.asarray(hw)
    return dict(inside=bool((np.abs(o) <= H).all()), zero_where_zero=bool((o[:, H == 0] == 0).all()), reach_frac=float((np.abs(o).max(0)[H > 0] / H[H > 0]).min()))

def g0_6(pe, p0, cfg):
    on = own_noise(cfg); res = dict(eval_noise=dict({k: v for k, v in pe.items() if k in ("bouts", "steps", "proprio83_vs_raw_obs_mismatch", "out_vs_eval_delivered83_mismatch",
                                                     "out_vs_eval_delivered_kp_mismatch", "kp_steps", "k_switches")}), zero_noise=p0, own_noise=on)
    res["ok"] = bool(pe["proprio83_vs_raw_obs_mismatch"] == pe["out_vs_eval_delivered83_mismatch"] == pe["out_vs_eval_delivered_kp_mismatch"] == pe["k_switches"] == 0 and pe["kp_steps"] > 0
                     and p0["out_vs_DelayedNoisyObs_mismatch"] == p0["out_vs_clean_t_minus_k_mismatch"] == p0["k_switches"] == 0 and p0["ks"] == [0, 1, 2] and on["inside"] and on["zero_where_zero"])
    return res

def reads_no_obs():
    """m11: the action term never reads env observations. (a) its only env input is a stub latency buffer whose every attribute
    but act_delay raises; (b) no identifier in its code (names, attributes, arguments; docstrings aside) mentions an observation."""
    import inspect, ast, textwrap
    class Stub:
        def __init__(self): self.act_delay = torch.tensor([True, False])
        def __getattr__(self, k): raise AssertionError(f"action term read env.{k}")
    lat = Stub(); a = AC.HybridDelayedAction(2, lat, np.zeros(24), np.ones(24)); a.reset(torch.arange(2)); x = torch.ones(2, 24); a.process(x); a.process(2 * x)
    tree = ast.parse(textwrap.dedent(inspect.getsource(AC.HybridDelayedAction)))
    ids = {getattr(n, f) for n in ast.walk(tree) for f in ("id", "attr", "arg") if isinstance(getattr(n, f, None), str)}; hit = sorted(i for i in ids if "obs" in i.lower())
    return dict(stub_ok=bool(_eq(a.applied[0].numpy(), np.ones(24)) and _eq(a.applied[1].numpy(), 2 * np.ones(24))), identifiers=len(ids), source_obs_names=hit, ok=not hit)

def g0_7(pe):
    ro = reads_no_obs(); res = dict({k: pe[k] for k in ("steps", "applied_vs_eval_last_a_mismatch", "targets_vs_eval_ctrl_mismatch", "eval_actions_slot_vs_applied_mismatch", "delayed_bouts", "act_switches")},
                                    first_delayed_step_zero=pe["first_delayed_step_zero"], reads_no_obs=ro)
    res["ok"] = bool(pe["applied_vs_eval_last_a_mismatch"] == pe["targets_vs_eval_ctrl_mismatch"] == pe["eval_actions_slot_vs_applied_mismatch"] == pe["act_switches"] == 0
                     and 0 < pe["delayed_bouts"] < pe["bouts"] and all(pe["first_delayed_step_zero"]) and ro["ok"] and ro["stub_ok"])
    return res

# ---- G0-6b: B3 in the replica
def resets(R, n=TOL["resets"], seed=0):
    """(bout, cut, kind): after each defender fall (its fall step), at each opponent knockdown (first step down) and 10 steps later,
    the rest at random steps."""
    out = []
    for i, b, A in bouts(R):
        if b["row"]["fell_s"] is not None: out.append((i, b["steps"], "fall"))
        dn = np.flatnonzero((A["post"][:, PO["oup"]] < EV.OPP_TILT) | (A["post"][:, PO["oz"]] < EV.OPP_Z))
        if len(dn): out += [(i, int(dn[0]) + 1, "knockdown"), (i, min(int(dn[0]) + 11, b["steps"]), "knockdown+10")]
    rng = np.random.default_rng(seed); B = bouts(R)
    while len(out) < n: i = int(rng.integers(len(B))); out.append((i, int(rng.integers(3, B[i][1]["steps"] + 1)), "random"))
    return out[:n]

def g0_6b(R, m, cfg, r, ids6, bug=None, seed=0, n=TOL["resets"], need_kinds=True):
    import mujoco, types
    from r1gym.keypoints import KeypointSource
    from r1gym.env import R1HitEnv
    import fb_record as REC
    B = bouts(R); RS = resets(R, n=n, seed=seed); N = len(RS); rs = B[0][1]["reset"]; home = np.array(rs["home"]); cut = np.array([c for _, c, _ in RS])
    lat = EV.Latency(N); lat.k[:] = torch.tensor([B[i][1]["reset"]["k"] for i, _, _ in RS]); lat.act_delay[:] = torch.tensor([bool(B[i][1]["reset"]["act_delay"]) for i, _, _ in RS])
    obs = OB.EvalActorObs(N, lat, np.zeros(OB.OBS_DIM), out_dtype=torch.float64); act = AC.HybridDelayedAction(N, lat, home, rs["scale"], rs["clip"])
    drv = FO.TrackerDriver(N, torch.float64); obs.reset(torch.arange(N)); act.reset(torch.arange(N))   # the first episode: the recorded eval bout (its k, act_delay)
    D = [mujoco.MjData(m) for _ in range(N)]; gen = torch.Generator().manual_seed(seed); prng = np.random.default_rng(seed); fq, fv, qa, va = r.free_q, r.free_v, r.qadr, r.vadr
    opp = FightReplayIdx(m); done = np.zeros(N, bool); prev = [[] for _ in range(N)]; t_ep = np.zeros(N, int); chk = []
    for g in range(int(cut.max()) + 1):
        if g > 0:
            act.process(torch.from_numpy(np.stack([B[i][2]["a_out"][g - 1] if not done[j] else np.zeros(24, np.float32) for j, (i, _, _) in enumerate(RS)])))
        Rg = [j for j in range(N) if cut[j] == g]
        for j in range(N):   # live envs: the recorded post-step state of step g (derived quantities fresh, as mj_forward leaves them)
            if not done[j]: D[j].qpos[:] = B[RS[j][0]][2]["qpos"][g]; D[j].qvel[:] = B[RS[j][0]][2]["qvel"][g]; mujoco.mj_forward(m, D[j])
        log = EV.ResetLog(); draw = None
        if Rg:
            Rt = torch.tensor(Rg); prev_last = {j: list(prev[j][-3:]) for j in Rg}
            if bug != "late_draw": draw = EV.draw_latency(lat, Rt, gen); log("event:fb_draw_latency")
            for j in Rg:   # defender reset event: qpos/qvel written, no forward (derived quantities stay the previous episode's)
                d = D[j]; d.qpos[fq:fq + 3] = (0.0, 0.0, 0.74); d.qpos[fq + 3:fq + 7] = (1.0, 0.0, 0.0, 0.0); d.qpos[qa] = home; d.qvel[fv:fv + 6] = 0; d.qvel[va] = 0; d.time = 0.0
            log("event:reset_defender")
            if bug != "no_prefill": obs.reset(Rt)
            log("obs.reset")
            if bug == "eager":   # pre-fill now, from the stale derived quantities and the unplaced opponent
                P, y, v = OB.kp_state(*_poses_of(D, Rg, ids6))
                row = torch.cat([_proprio(D, Rg, r, home, act, torch.zeros(len(Rg), dtype=torch.long)), OB.pack(P, torch.zeros_like(P[:, :2]), y, v)], -1)
                obs.ring[Rt] = row[:, None].expand(-1, OB.RING, -1); obs.prefill[Rt] = False; obs.k_ep[Rt] = lat.k[Rt]
            act.reset(Rt); log("action.reset")
            if bug == "late_draw": draw = EV.draw_latency(lat, Rt, gen)
            c = prng.integers(0, 3, len(Rg)); di = prng.integers(0, 4, len(Rg)); dist = np.array(REC.HARD)[di]; t0 = np.array([_clean_t0(drv.M, int(ci), float(dd), prng) for ci, dd in zip(c, dist)])
            qp, qv = drv.reset(Rt, c, t0, dist)
            for k, j in enumerate(Rg): D[j].qpos[opp.qa:opp.qa + 31] = qp[k].numpy(); D[j].qvel[opp.va:opp.va + 30] = qv[k].numpy()
            log("command.reset")
            if bug != "no_forward":
                for j in Rg: mujoco.mj_forward(m, D[j])
            log("forward")
            tg = _command_compute(drv, D, Rg, r, opp, t0); log("command.compute")
            for j in Rg: done[j] = True; t_ep[j] = 0
        # obs compute for every env: live envs from the recording (command from the eval row), reset / parked envs from their MjData
        P, y, v = OB.kp_state(*_poses_of(D, range(N), ids6)); tstep = torch.tensor([t_ep[j] if done[j] else g for j in range(N)])
        cmd = torch.from_numpy(np.stack([np.zeros(3) if done[j] else B[RS[j][0]][2]["raw"][g][OB.CMD].astype(np.float64) for j in range(N)]))
        p = _proprio(D, range(N), r, home, act, tstep, cmd)
        o = obs.compute(p, P, y, v, noise=torch.zeros(N, OB.OBS_DIM, dtype=torch.float64)).numpy()
        if Rg: log("obs.compute")
        for j in Rg:
            d = D[j]; shim = types.SimpleNamespace(d=d, r=r, contract="mjlab", command=np.zeros(3), t=0, dt=FO.CTRL_DT, home=home, last_a=np.zeros(24, np.float32))
            kp_ref = KeypointSource(m, "truth", delay=int(lat.k[j])).reset(d); fresh = np.concatenate([R1HitEnv._raw_obs(shim), kp_ref]).astype(np.float32)
            ring = obs.ring[j].numpy(); i_d = Rg.index(j)
            chk.append(dict(kind=RS[j][2], bout=RS[j][0], cut=int(g), k=int(lat.k[j]), act_delay=bool(lat.act_delay[j]),
                            ring_fresh=all(_eq(ring[q], fresh) for q in range(OB.RING)), out_fresh=_eq(o[j], fresh.astype(np.float64)), kp_placed=_eq(o[j, 83:].astype(np.float32), kp_ref),
                            draw_ok=bool(draw is not None and int(obs.k_ep[j]) == int(draw[0][i_d]) == int(lat.k[j]) and bool(act.delay[j]) == bool(draw[1][i_d]) == bool(lat.act_delay[j])),
                            no_prev_rows=not any(_eq(ring[q], pr) for q in range(OB.RING) for pr in prev_last[j]), vel_zero=bool((o[j, 83 + 15:83 + 21] == 0).all()),
                            a_prev_zero=bool(float(act.a_prev[j].abs().max()) == 0.0 and float(act.applied[j].abs().max()) == 0.0),
                            targets_ok=bool(np.isfinite(tg[i_d]).all() and np.abs(tg[i_d]).max() > 0), order_ok=log.ok()))
        for j in range(N):
            if done[j]: t_ep[j] += 1
            else: prev[j] = (prev[j] + [obs.clean[j].numpy().copy()])[-3:]
    keys = ("ring_fresh", "out_fresh", "kp_placed", "draw_ok", "no_prev_rows", "vel_zero", "a_prev_zero", "targets_ok", "order_ok")
    res = dict(resets=len(chk), by_kind={k: sum(c["kind"] == k for c in chk) for k in ("fall", "knockdown", "knockdown+10", "random")},
               ks=sorted({c["k"] for c in chk}), act_delay_share=round(float(np.mean([c["act_delay"] for c in chk])), 3), **{k: sum(c[k] for c in chk) for k in keys}, bug=bug)
    res["ok"] = bool(all(res[k] == len(chk) for k in keys) and len(chk) == n and (not need_kinds or (res["by_kind"]["fall"] > 0 and res["by_kind"]["knockdown"] > 0)))
    res["failures"] = [c for c in chk if not all(c[k] for k in keys)][:4]; return res

class FightReplayIdx:
    """The opponent's free-joint addresses and bodies in the eval model (FightReplay's)."""
    def __init__(self, m):
        j = m.joint("opp_floating_base_joint").id; self.qa, self.va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.pel, self.torso = m.body("opp_pelvis").id, m.body("opp_torso_link").id
        from r1gym.model import CTRL_JOINTS
        self.act = np.array([m.actuator("opp_" + h.replace("_joint", "")).id for h in CTRL_JOINTS])

def _poses_of(D, js, ids6):
    js = list(js); return torch.from_numpy(np.stack([D[j].xpos[ids6] for j in js])), torch.from_numpy(np.stack([D[j].xquat[ids6] for j in js]))

def _proprio(D, js, r, home, act, tstep, cmd=None):
    js = list(js); Q = np.stack([D[j].qpos for j in js]); V = np.stack([D[j].qvel for j in js]); fq, fv = r.free_q, r.free_v
    cmd = torch.zeros(len(js), 3, dtype=torch.float64) if cmd is None else cmd
    return OB.proprio83(torch.from_numpy(Q[:, fq + 3:fq + 7]), torch.from_numpy(V[:, fv + 3:fv + 6]), torch.from_numpy(Q[:, r.qadr]), torch.from_numpy(V[:, r.vadr]),
                        torch.as_tensor(home), cmd, tstep, act.applied[torch.tensor(js)])

def _clean_t0(M, c, dist, rng):
    """A clean start at the bout distance (fb_motion's masks = the eval's clean_starts), uniform over the clean grid (pick_start)."""
    j = int(np.flatnonzero(np.isclose(M.distances.numpy(), dist))[0]); idx = np.flatnonzero(M.clean[c, j].numpy()); return round(float(idx[int(rng.integers(len(idx)))] * 0.02), 3)

def _command_compute(drv, D, Rg, r, opp, t0):
    """The opponent command's compute on the forwarded state (TrackerDriver.step for the reset envs; others hold their state)."""
    import copy as _c
    sub = _c.copy(drv); n = len(Rg); ix = torch.tensor(Rg)
    for k in ("clip", "offset", "t0", "dist", "yaw", "spot", "axis", "last", "yaw_set", "axis_set"): setattr(sub, k, getattr(drv, k)[ix].clone())
    sub.n = n; T = lambda f: torch.from_numpy(np.stack([f(D[j]) for j in Rg]))
    tg, _ = sub.step(torch.as_tensor(t0 + FO.CTRL_DT), T(lambda d: d.xpos[r.pel][:2]), T(lambda d: d.xpos[opp.pel][:2]), T(lambda d: d.xpos[opp.torso]), T(lambda d: d.xquat[opp.torso]),
                     T(lambda d: d.qvel[opp.va + 3:opp.va + 6]), T(lambda d: d.qpos[opp.qa + 7:opp.qa + 31]), T(lambda d: d.qvel[opp.va + 6:opp.va + 30]))
    for k in ("yaw", "spot", "axis", "last", "yaw_set", "axis_set"): getattr(drv, k)[ix] = getattr(sub, k)
    for k, j in enumerate(Rg): D[j].ctrl[opp.act] = tg[k].numpy()
    return tg.numpy()

# ---- G0-8
def _quat(rng, tilt_max):
    """A random orientation: tilt U(0, tilt_max) about a random horizontal axis, then a random yaw (wxyz)."""
    t, ph, yw = rng.uniform(0, tilt_max), rng.uniform(0, 2 * np.pi), rng.uniform(-np.pi, np.pi)
    a = torch.tensor([np.cos(t / 2), np.sin(t / 2) * np.cos(ph), np.sin(t / 2) * np.sin(ph), 0.0], dtype=torch.float64); z = torch.tensor([np.cos(yw / 2), 0.0, 0.0, np.sin(yw / 2)], dtype=torch.float64)
    return FO.qmul(z, a).numpy()

def g0_8(R, m, cfg, r, n_rand=TOL["g08_random"], seed=0):
    import mujoco, eval_opponent as EO
    const = dict(fall_z=bool(EV.FALL_Z == cfg["fall"]["min_pelvis_z"]), up_min=bool(EV.UP_MIN == np.cos(np.deg2rad(cfg["fall"]["max_tilt_deg"]))), opp=bool((EV.OPP_TILT, EV.OPP_Z) == (EO.OPP_TILT, EO.OPP_Z)))
    c = dict(states=0, fell_mismatch=0, fell_via_xmat_mismatch=0, upz_vs_xmat_mismatch=0, opp_down_mismatch=0, both_up_iv_mismatch=0, first_opp_fall_mismatch=0,
             opp_fell_mismatch=0, edges=0, fell_true=0, opp_down_true=0, opp_up_again=0)
    for i, b, A in bouts(R):
        P = torch.from_numpy(A["post"]); up, oup = EV.up_z(P[:, PO["pquat"]]), EV.up_z(P[:, PO["oquat"]]); c["states"] += len(P)
        c["upz_vs_xmat_mismatch"] += int((up.numpy() != A["post"][:, PO["pup"]]).sum() + (oup.numpy() != A["post"][:, PO["oup"]]).sum())
        c["fell_mismatch"] += int((EV.fell_eval(P[:, PO["pz"]], up).numpy() != A["fallen"]).sum()); c["fell_true"] += int(A["fallen"].sum())
        c["fell_via_xmat_mismatch"] += int((EV.fell_eval(P[:, PO["pz"]], P[:, PO["pup"]]).numpy() != A["fallen"]).sum())
        ref = (A["post"][:, PO["oup"]] < EO.OPP_TILT) | (A["post"][:, PO["oz"]] < EO.OPP_Z)   # eval_opponent.run's test, on the same derived quantities
        od = EV.OppDown(1); now = []
        for t in range(len(P)): c["edges"] += int(od.update(P[t:t + 1, PO["oz"]], oup[t:t + 1])[0]); now.append(bool(od.now[0]))
        now = np.array(now, bool); c["opp_down_mismatch"] += int((now != ref).sum()); c["opp_down_true"] += int(now.sum()); c["opp_up_again"] += int((now[:-1] & ~now[1:]).sum())
        iv = []
        for t in np.flatnonzero(~now) + 1:   # EO: control step e.t (1-based) both up -> extend or open an interval
            if iv and iv[-1][1] == t - 1: iv[-1][1] = int(t)
            else: iv.append([int(t), int(t)])
        c["both_up_iv_mismatch"] += int([[round((k0 - 1) * 0.02, 3), round(k1 * 0.02, 3)] for k0, k1 in iv] != b["row"]["both_up_iv"])
        ff, first = b["row"]["opp_fall"], np.flatnonzero(now)
        c["first_opp_fall_mismatch"] += int((ff is None) != (len(first) == 0) or (ff is not None and round(int(first[0] + 1) * 0.02, 3) != ff["t_s"]))
        c["opp_fell_mismatch"] += int(bool(b["row"]["opp_fell"]) != bool(now.any()))
    from r1gym.model import R1Robot
    rng = np.random.default_rng(seed); d = mujoco.MjData(m); rb = R1Robot(m, cfg); op = FightReplayIdx(m); home = np.array(cfg["home"]); oq = op.qa + 7
    Z, U, OZ, OU, F, O = [], [], [], [], [], []
    for _ in range(n_rand):
        q = m.qpos0.copy()
        for qa, x0 in ((r.free_q, 0.0), (op.qa, 0.5)): q[qa:qa + 3] = (x0 + rng.normal() * 0.05, rng.normal() * 0.05, rng.uniform(0.15, 0.85)); q[qa + 3:qa + 7] = _quat(rng, np.radians(100))
        q[r.qadr] = home + rng.normal(0, 0.3, 24); q[oq:oq + 24] = FO.DEFAULT + rng.normal(0, 0.3, 24); d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        Z.append(d.xpos[rb.pel][2]); U.append(d.xquat[rb.pel].copy()); OZ.append(d.xpos[op.pel][2]); OU.append(d.xquat[op.pel].copy())
        F.append(bool(rb.fallen(d))); O.append(bool(d.xmat[op.pel][8] < EO.OPP_TILT or d.xpos[op.pel][2] < EO.OPP_Z))
    T = lambda x: torch.from_numpy(np.array(x)); f = EV.fell_eval(T(Z), EV.up_z(T(U))).numpy(); o = EV.opp_down_now(T(OZ), EV.up_z(T(OU))).numpy()
    rnd = dict(states=n_rand, fell_mismatch=int((f != np.array(F)).sum()), opp_down_mismatch=int((o != np.array(O)).sum()), fell_true=int(np.sum(F)), opp_down_true=int(np.sum(O)))
    res = dict(constants=const, recorded=c, random=rnd)
    res["ok"] = bool(all(const.values()) and c["states"] + rnd["states"] >= TOL["g08_states"] and all(c[k] == 0 for k in c if k.endswith("mismatch"))
                     and rnd["fell_mismatch"] == rnd["opp_down_mismatch"] == 0 and 0 < rnd["fell_true"] < n_rand and 0 < rnd["opp_down_true"] < n_rand and c["fell_true"] > 0 and c["opp_down_true"] > 0)
    return res

# ---- G0-9
def _arrays(m):
    out = {}
    for k in dir(m):
        if k.startswith("_"): continue
        try: v = getattr(m, k)
        except Exception: continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub": out[k] = v.copy()
    return out

def g0_9(R, m, cfg, n_seeds=TOL["dr_seeds"]):
    import mujoco
    from r1gym.env import R1HitEnv, _DR_NOMINAL
    from r1gym.model import CTRL_JOINTS
    m0 = copy.deepcopy(m); ix = DR.DRIndex(m0, CTRL_JOINTS); me = copy.deepcopy(m); e = R1HitEnv(seed=0, cfg=cfg, model=me, train=False, dr=True)
    assert (np.array_equal(ix.bodies, e._r1b) and np.array_equal(ix.feet, e._feet) and np.array_equal(ix.dofs, e._r1dof) and np.array_equal(ix.act, e.r.act)), "index sets differ from the eval's"
    G = bouts(R, "G"); env_seeds = [int(b["episode"]["env_seed"]) for _, b, _ in G]; rec_ad = [bool(b["reset"]["act_delay"]) for _, b, _ in G]
    mo = copy.deepcopy(m); F, nom = DR.model_fields(mo); fm = adm = 0; snap = {}
    for s in env_seeds + list(range(n_seeds)):
        e.rng = np.random.default_rng(s); e._dr_sample_into_model(); p, ad = DR.sample_np(np.random.default_rng(s), ix); DR.write(F, nom, ix, p)
        fm += sum(not np.array_equal(getattr(mo, k), getattr(me, k)) for k in DR.FIELDS); adm += int(ad != bool(e.act_delay))
        if s == env_seeds[0]: snap = dict(eval=copy.deepcopy(me), ours=copy.deepcopy(mo))
    rec_adm = sum(DR.sample_np(np.random.default_rng(s), ix)[1] != a for s, a in zip(env_seeds, rec_ad))
    # torch draws: ranges and their ends
    P = DR.sample(100000, ix, generator=torch.Generator().manual_seed(0)); rg = {}
    for k, (lo, hi) in DR.RANGES.items():
        x = P[k]; rg[k] = dict(min=float(x.min()), max=float(x.max()), inside=bool(float(x.min()) >= lo and float(x.max()) <= hi),
                               ends=bool(float(x.min()) - lo <= TOL["range_edge_frac"] * (hi - lo) and hi - float(x.max()) <= TOL["range_edge_frac"] * (hi - lo)))
    # what a write changes: only FIELDS, only defender entries (MjModel, our write with the first env seed's draws)
    A0, A1 = _arrays(m0), _arrays(snap["ours"]); changed = sorted(k for k in A0 if not np.array_equal(A0[k], A1[k]))
    rows = dict(body_mass=ix.bodies, body_ipos=ix.bodies, geom_friction=ix.feet, actuator_gainprm=ix.act, actuator_biasprm=ix.act, actuator_forcerange=ix.act, dof_armature=ix.dofs)
    off_rows = {k: [int(i) for i in np.flatnonzero((A0[k] != A1[k]).reshape(len(A0[k]), -1).any(1)) if i not in set(rows[k].tolist())] for k in changed if k in rows}
    # torch per-world fields: envs not in ids untouched, opponent entries nominal
    F4 = {k: v[None].repeat(4, *([1] * v.dim())) for k, v in nom.items()}; DR.write(F4, nom, ix, DR.sample(2, ix, generator=torch.Generator().manual_seed(1)), ids=[1, 3])
    opp_b = [b for b in range(m0.nbody) if m0.body(b).name.startswith("opp_")]; opp_a = [a for a in range(m0.nu) if (m0.actuator(a).name or "").startswith("opp_")]
    opp_g = [g for g in range(m0.ngeom) if m0.geom_bodyid[g] in set(opp_b)]; opp_d = [v for v in range(m0.nv) if m0.dof_bodyid[v] in set(opp_b)]
    orow = dict(body_mass=opp_b, body_ipos=opp_b, geom_friction=opp_g, actuator_gainprm=opp_a, actuator_biasprm=opp_a, actuator_forcerange=opp_a, dof_armature=opp_d)
    t4 = dict(untouched_envs=all(torch.equal(F4[k][j], nom[k]) for k in DR.FIELDS for j in (0, 2)), opponent_nominal=all(torch.equal(F4[k][j][orow[k]], nom[k][orow[k]]) for k in DR.FIELDS for j in range(4)),
              written_envs_changed=all(not torch.equal(F4["body_mass"][j], nom["body_mass"]) for j in (1, 3)))
    # no set_const: invweight0 / subtreemass nominal; the constraint regularisation on a contact state = the eval's DR model's
    i, b, A = max(G, key=lambda x: x[1]["row"]["opp_fell"]); t = int(np.flatnonzero((A["post"][:, PO["oup"]] < EV.OPP_TILT) | (A["post"][:, PO["oz"]] < EV.OPP_Z))[0]) if b["row"]["opp_fell"] else 100
    def fwd(mm):
        d = mujoco.MjData(mm); d.qpos[:] = A["qpos"][t]; d.qvel[:] = A["qvel"][t]; mujoco.mj_forward(mm, d); return d
    de, do, dn = fwd(snap["eval"]), fwd(snap["ours"]), fwd(m0); sc = copy.deepcopy(snap["ours"]); mujoco.mj_setConst(sc, mujoco.MjData(sc)); ds = fwd(sc)
    CON = ("dof_invweight0", "body_invweight0", "body_subtreemass", "dof_M0", "actuator_acc0"); EFC = ("efc_R", "efc_D", "efc_KBIP", "efc_diagA")
    reg = dict(state=dict(bout=i, seed=b["episode"]["seed"], step=t, ncon=int(do.ncon), nefc=int(do.nefc)),
               const_nominal={k: _eq(getattr(snap["ours"], k), getattr(m0, k)) for k in CON}, eval_const_nominal={k: _eq(getattr(snap["eval"], k), getattr(m0, k)) for k in CON},
               efc_equal_eval={k: bool(de.nefc == do.nefc and _eq(getattr(de, k), getattr(do, k))) for k in EFC},
               set_const_changes={k: not _eq(getattr(sc, k), getattr(m0, k)) for k in CON}, set_const_efc_R_differs=bool(ds.nefc != do.nefc or not _eq(ds.efc_R, do.efc_R)),
               info_efc_R_vs_nominal_model_maxrel=float(np.max(np.abs(do.efc_R - dn.efc_R) / np.maximum(np.abs(dn.efc_R), 1e-300))) if dn.nefc == do.nefc else None)
    _DR_NOMINAL.pop(id(me), None)
    res = dict(seeds=dict(env_seeds=env_seeds, numpy=f"0-{n_seeds - 1}"), field_mismatch_vs_eval=fm, act_delay_mismatch_vs_eval=adm, act_delay_mismatch_vs_recorded_bouts=int(rec_adm),
               ranges=rg, changed_fields=changed, off_defender_rows=off_rows, torch_per_world=t4, regularisation=reg)
    res["ok"] = bool(fm == adm == rec_adm == 0 and all(v["inside"] and v["ends"] for v in rg.values()) and set(changed) <= set(DR.FIELDS) and not any(off_rows.values())
                     and all(t4.values()) and all(reg["const_nominal"].values()) and all(reg["eval_const_nominal"].values()) and all(reg["efc_equal_eval"].values())
                     and all(reg["set_const_changes"].values()) and reg["set_const_efc_R_differs"] and reg["state"]["nefc"] > 0)
    return res

# ---- G0-10
def g0_10(R, p1):
    c, cfg, _ = setup("H", p1); assert getattr(c.arms, "kp_mode", None) == "obs" and getattr(c.base, "kp_mode", None) is None
    legs = lambda x: torch.from_numpy(c.base.act(x.numpy())); arms = lambda x: torch.from_numpy(c.arms.act(x.numpy())); n = mis = gin = 0
    for i, b, A in bouts(R, "H"):
        T = b["steps"]; O = np.concatenate([A["sens_o"][:T].astype(np.float64) + A["sens_n"][:T], A["kp_out"][:T].astype(np.float64)], 1)
        for t in range(T):   # batch 1, as recorded
            o = torch.from_numpy(O[t:t + 1]); mis += int(not _eq(AC.compose(o, legs, arms, c.scale, c.clip, 111).numpy()[0], A["a_out"][t]))
            gin += int(not _eq(AC.m3s_input(o, c.scale, c.clip).numpy(), c.g_in(O[t:t + 1]))); n += 1
    o32 = torch.from_numpy(O.astype(np.float32)); d32 = float((AC.m3s_input(o32, c.scale, c.clip).double() - torch.from_numpy(c.g_in(O))).abs().max())
    res = dict(rows=n, compose_vs_p1_mismatch=mis, m3s_input_vs_g_in_mismatch=gin, scale=list(map(float, c.scale)), clip=c.clip, info_float32_rows_m3s_maxdiff=d32)
    res["ok"] = bool(n > 0 and mis == gin == 0); return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--m4rec", default=""); ap.add_argument("--rec", default="")
    ap.add_argument("--p1", required=True); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seconds", type=float, default=20.0); a = ap.parse_args()
    assert a.workers <= 4; FO.pin_precision(); torch.set_num_threads(4); t0 = time.time()
    if a.m4rec and Path(a.m4rec).exists(): R = load(a.m4rec); print("M4 recording", a.m4rec, flush=True)
    else:
        R = record(seconds=a.seconds, workers=a.workers, p1=a.p1, rec=a.rec or None)
        if a.m4rec: save(R, a.m4rec); R = load(a.m4rec)
    mt = R["meta"]; m, cfg, r, ids6 = model_cfg()
    res = dict(tol=TOL, seeds=mt["seeds"], role=mt["role"], ctrls=mt["ctrls"], record_wall_s=mt["wall_s"], same_as_m2_recording=mt.get("same_as_m2_recording"),
               bouts=[dict(ctrl=b["ctrl"], seed=b["episode"]["seed"], steps=b["steps"], k=b["reset"]["k"], act_delay=b["reset"]["act_delay"], fell_s=b["row"]["fell_s"],
                           opp_fell=b["row"]["opp_fell"]) for b in mt["bouts"]])
    print(json.dumps({k: v for k, v in res.items() if k != "bouts"}), flush=True)
    pe = pipeline(R, m, cfg, r, ids6, "eval"); p0 = pipeline(R, m, cfg, r, ids6, "zero")
    for k, f in (("G0-5", lambda: g0_5(R, m, ids6, pe)), ("G0-6", lambda: g0_6(pe, p0, cfg)), ("G0-6b", lambda: g0_6b(R, m, cfg, r, ids6)), ("G0-7", lambda: g0_7(pe)),
                 ("G0-8", lambda: g0_8(R, m, cfg, r)), ("G0-9", lambda: g0_9(R, m, cfg)), ("G0-10", lambda: g0_10(R, a.p1))):
        t1 = time.time(); res[k] = f(); res[k]["wall_s"] = round(time.time() - t1, 1); print(k, json.dumps(res[k], default=str), flush=True)
    neg = {}
    for bug in ("eager", "late_draw", "no_prefill", "no_forward"):
        x = g0_6b(R, m, cfg, r, ids6, bug=bug); neg[bug] = dict(fails=not x["ok"], **{k: x[k] for k in ("resets", "ring_fresh", "out_fresh", "kp_placed", "draw_ok", "no_prev_rows", "vel_zero")})
    res["G0-6b negative controls"] = neg; print("G0-6b negative", json.dumps(neg), flush=True)
    res["ok"] = {k.replace("-", ""): res[k]["ok"] for k in ("G0-5", "G0-6", "G0-6b", "G0-7", "G0-8", "G0-9", "G0-10")}
    res["ok"]["G06b_negatives_fail"] = all(v["fails"] for v in neg.values()); res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res["ok"]), res["wall_s"], flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1, default=str)
    sys.exit(0 if all(res["ok"].values()) else 1)

if __name__ == "__main__": main()
