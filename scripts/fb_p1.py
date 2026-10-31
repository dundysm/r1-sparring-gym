"""P1 parity on the GPU pod (M2V2_PLAN_v3 section 4 P1; 3.1 P1-R4g, 3.2; B3, B4, m12; A4.3, A5.1, A8, A10, A11). Each item writes its
part of --out; every item must pass (3.2's fail handling: one fix and re-smoke; P1-R4g's reserve is a second set of GPU worlds, --seed).
  (1) t0_alone   T0 alone, 24 starts x 3 train clips (t0 = s x T / 24), 10 s at 4 m, no aim, defender idle (zero actions, fell_eval off):
                 the task's GPU worlds vs the eval's own tracker on this pod's CPU (fb_m2_check._alone 'eval' = scripts/eval_tracker.run):
                 fall times within 0.1 s, or both stay up (M2 G0-3b's rule).
  (2) restricted tracker-alone trajectory parity over the contact-free first 0.3 s (B4): 32 of those starts, the opponent pelvis (env
                 frame, defender heading 0) per step for 15 steps: max |GPU - CPU| <= 2 cm.
  (3) r4g        P1-R4g: G, H(A3s)/M3s and full A3 as the defender in the task (play cfg: no pushes, no ring_out; T0 nominal rows, eval
                 distances, aimed, eval fall rule, delays on; the population's T0-only window re-armed per batch), 12288 worlds x 1000
                 steps per controller, the first episode of every world -> rows (fell, knockdown = OppDown's edge, the account's clean head
                 hits, up seconds, act_delay); fb_eq.r4g_gates against R0-EO (unpaired; reference = the R0-EO rows of A8.2's pod platform,
                 --r0eo; the laptop R0-EO is reported beside it when given, --r0eo-laptop). All gates but kd_A3 (report) must pass.
  (4) hit        P1-Hit: 32 worlds x 1000 steps (H); every substep the GPU's contacts between the opponent's collision geoms and the
                 defender's zoned geoms (Warp's contact list and efc force) and the involved bodies' cvel / xipos / subtree_com are dumped;
                 r1gym's HitMeter runs on exactly those contacts on the replica's model (geoms mapped by fb_m9_check's keys; mujoco's
                 mj_contactForce / mj_objectVelocity fed the dump) and is compared with the GPU FullBodyHitAccount's own episode log, same
                 state: clean calls agree >= 0.95 over landed episodes; landed counts per (limb, zone) exact on >= 90% of worlds (the first
                 episode of each world).
  (5) b3         B3 asserts during (3): at every episode's first observation every ring row equals the fresh row and its keypoint block
                 equals one built from the placed opponent (velocities 0); command_counter == 1 per episode (w_cmd_bad 0); act_delay share
                 0.50 +- 0.05 and each k share 1/3 +- 0.05 (WATCH6 E0's tolerances); no within-episode lag switch.
  (6) runner     a 3-iteration rsl_rl smoke of the real runner per stage, chained as the pod trains (A0 from start_sighted, A from A0, B
                 from A; FB_STAGE, fullbody_task.py, cwd the unitree checkout): exit 0, each checkpoint's blind flag False, residual keys in
                 B only, legs = G's checkpoint in A0 / A.
--dry (this CPU, no GPU, no mjlab env): every item's CPU half and comparator on stand-ins: (1)/(2) the torch32 driver (MjTracker) in the
eval world as the 'GPU' side, 2 starts x 3 clips; (3) fb_replica at REQ_CONF with the training reset on the M10 smoke seeds as the 'GPU'
worlds vs the laptop R0-EO rows (read only); (4) a replica bout's own contacts dumped by the same recorder fields and replayed (must equal
the live HitMeter exactly) and compared with its account; the mjlab-to-replica geom map on mjlab's real Scene compiled with stubs; (5) the
B3 asserts on fb_cpu_smoke's env; (6) the commands printed and the M10 checkpoints checked as (6) checks them.
Usage (pod, cwd the repo): python -I scripts/fb_p1.py --all --out p1.json [--items t0_alone,r4g,...] [--r0eo DIR] [--worlds 4096] [--seed 0]
       (CPU):              python -I scripts/fb_p1.py --dry --out p1_dry.json [--ckdir build\\m10\\plumbing] [--r0eo-laptop DIR]"""
import sys, os, json, time, math, types, argparse, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np

ITEMS = ("t0_alone", "restricted", "r4g", "hit", "b3", "runner")
TOL = dict(alone_s=0.1, traj_m=0.02, clean_agree=0.95, exact_worlds=0.90, share=0.05)
ONSET_TOL = 0.0025   # onsets lie on the 5 ms substep grid; the float32 account's t0 is off by up to 1 ulp (1.9e-6 s at 16-20 s), so 1e-6 split same events (M10 verify)
STARTS, ALONE_S, ALONE_D, TRAJ_N, TRAJ_STEPS, DT = 24, 10.0, 4.0, 32, 15, 0.02
R4G_N, R4G_STEPS, HIT_N = 12288, 1000, 32
TRAIN = ("kfab_200", "kfab_201", "kfab_203")
FB_ROOT = Path(os.environ.get("FB_ROOT", "/root")); UNITREE = Path(os.environ.get("FB_UNITREE", str(FB_ROOT / "unitree_rl_mjlab")))
A3_PT = "results/policy_mjlab_spar3_a3/model_8995.pt"

# ============================================================================================ comparators (pure)
def alone_compare(ref, got, tol=TOL["alone_s"]):
    """{(clip, start): fall time or None} x2 -> M2 G0-3b's verdict: both up, or both fell within tol."""
    bad, dts, up, fell = [], [], 0, 0
    for k in sorted(ref):
        a, b = ref[k], got.get(k, "missing")
        if a is None and b is None: up += 1
        elif a is not None and b not in (None, "missing") and abs(a - b) <= tol + 1e-9: fell += 1; dts.append(abs(a - b))
        else: bad.append(dict(clip=k[0], start=k[1], cpu=a, gpu=b))
    return dict(runs=len(ref), both_up=up, both_fell_within=fell, max_dt_s=max(dts) if dts else None, mismatches=bad, ok=not bad and len(got) == len(ref))

def traj_compare(ref, got, tol=TOL["traj_m"]):
    """{key: [steps, 3]} x2 (opponent pelvis, env frame) -> max |difference| per key and the verdict."""
    e = {f"{k[0]}/{k[1]}": float(np.abs(np.asarray(ref[k]) - np.asarray(got[k])).max()) for k in ref}
    return dict(worlds=len(ref), max_m=max(e.values()) if e else None, per_world=e, ok=bool(e) and max(e.values()) <= tol and set(ref) == set(got))

def rows_from_first(fell, kd, hc, steps, act_delay, k=None):
    """Per-world first-episode tensors -> fb_eq.arrays-compatible rows."""
    out = []
    for i in range(len(steps)):
        s = float(steps[i]) * DT; out.append(dict(seed=i, clip="", distance=0.0, t0=0.0, k=int(k[i]) if k is not None else 0, act_delay=int(act_delay[i]),
                                                   fell_s=s if bool(fell[i]) else None, kd=bool(kd[i]), n=dict(head_clean=int(hc[i])), secs=s))
    return out

def r4g_verdict(eo, gpu):
    """fb_eq.r4g_gates on {ctrl: rows} (EO, GPU worlds): every gate but kd_A3 (report) passes."""
    import fb_eq as EQ
    E = {c: EQ.arrays(eo[c])[0] for c in ("G", "H", "A3")}; G = {c: EQ.arrays(gpu[c])[0] for c in ("G", "H", "A3")}; g = EQ.r4g_gates(E, G)
    rate = lambda X: {c: dict(n=len(x), survival=float(1 - x[:, 0].mean()), kd=float(x[:, 1].mean()), head_clean_pm=float(x[:, 2].sum() / max(x[:, 3].sum() / 60, 1e-9))) for c, x in X.items()}
    return dict(gates=g, gated_pass={k: v["pass_"] for k, v in g.items() if k != "kd_A3"}, ok=all(v["pass_"] for k, v in g.items() if k != "kd_A3"), eo=rate(E), gpu=rate(G))

def shares_ok(n_eps, ad, kc, tol=TOL["share"]):
    """act_delay share 0.50 and each k share 1/3, within tol (WATCH6 E0)."""
    s = dict(act_delay=ad / max(n_eps, 1), **{f"k{i}": kc[i] / max(n_eps, 1) for i in range(3)})
    return dict(episodes=int(n_eps), shares=s, ok=bool(n_eps > 0 and abs(s["act_delay"] - 0.5) <= tol and all(abs(s[f"k{i}"] - 1 / 3) <= tol for i in range(3))))

# ============================================================================================ (4) the HitMeter on dumped contacts
class _MjShim:
    """r1gym.hits' `mujoco` during a replay: mj_contactForce from the dump (normal force; the hit contacts are shell-shell, condim 1),
    mj_objectVelocity from the real module on the replay's MjData (cvel / xipos / subtree_com written from the dump); the rest real."""
    def __init__(self, real): self.real = real
    def __getattr__(self, k): return getattr(self.real, k)
    def mj_contactForce(self, m, d, i, f6): f6[:] = 0.0; f6[0] = d.fn[i]
    def mj_objectVelocity(self, m, d, ot, b, res, flg): self.real.mj_objectVelocity(m, d.real, ot, b, res, flg)

def hit_replay(m, subs, params=None):
    """r1gym HitMeter on dumped substeps of one episode, on the replica's model m. subs: [dict(t, geom (C, 2) replica ids, pos (C, 3),
    frame (C, 9), fn (C,), bodies {replica body id: (cvel 6, xipos 3, subtree_com 3 of its root)})] in time order -> the HitMeter."""
    import mujoco, r1gym.hits as H
    hm = H.HitMeter(m, params); d = mujoco.MjData(m); shim = _MjShim(mujoco); real = H.mujoco; H.mujoco = shim
    try:
        for s in subs:
            for b, (cv, xi, sc) in s["bodies"].items(): d.cvel[b] = cv; d.xipos[b] = xi; d.subtree_com[m.body_rootid[b]] = sc
            g = np.asarray(s["geom"], np.int64).reshape(-1, 2)
            D = types.SimpleNamespace(ncon=len(g), contact=types.SimpleNamespace(geom1=g[:, 0], geom2=g[:, 1], pos=np.asarray(s["pos"], float).reshape(-1, 3),
                                      frame=np.asarray(s["frame"], float).reshape(-1, 9)), fn=np.asarray(s["fn"], float), xipos=d.xipos, time=float(s["t"]), real=d)
            hm.substep(m, D)
    finally: H.mujoco = real
    return hm

def dump_rows(m, d, hm, t):
    """The recorder (CPU twin of the GPU dump): the HitMeter-relevant contacts of MjData d after a substep (geom pair, pos, frame, normal
    force) and the involved bodies' cvel / xipos / root subtree_com, time t (episode time)."""
    import mujoco
    n = d.ncon
    if n == 0: return dict(t=t, geom=[], pos=[], frame=[], fn=[], bodies={})
    g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]; ix = np.flatnonzero((hm.is_r1[g1] & hm.is_opp[g2]) | (hm.is_r1[g2] & hm.is_opp[g1])); f6 = np.zeros(6); fn, bodies = [], {}
    for i in ix:
        mujoco.mj_contactForce(m, d, int(i), f6); fn.append(float(f6[0]))
        for g in (g1[i], g2[i]): b = int(m.geom_bodyid[g]); bodies[b] = (d.cvel[b].copy(), d.xipos[b].copy(), d.subtree_com[m.body_rootid[b]].copy())
    return dict(t=t, geom=np.stack([g1[ix], g2[ix]], 1), pos=d.contact.pos[ix].copy(), frame=d.contact.frame[ix].copy(), fn=np.array(fn), bodies=bodies)

def hit_compare(acct_logs, meters, hm_p=None):
    """Per world: the GPU account's landed episodes (its log, kind 'episode') vs the replayed HitMeter's: counts per (limb, zone) exact,
    clean agreement over landed episodes (fb_m5_check.compare's matching: same limb / zone, onset within ONSET_TOL; unmatched = disagreement)."""
    import fb_hits as HT
    keys = [(l, z) for l in HT.LIMBS for z in ("head", "torso")]; exact = n_land = agree = 0; bad = []
    for w, (log, hm) in enumerate(zip(acct_logs, meters)):
        mine = [e for e in log if e["kind"] == "episode" and e["landed"]]; ev = hm.events(); ref = [dict(limb=e["limb"], zone=e["zone"], t0=e["t0"], clean=hm.clean(e)) for e in ev if hm.landed(e)]
        cm = {k: sum((e["limb"], e["zone"]) == k for e in mine) for k in keys}; cr = {k: sum((e["limb"], e["zone"]) == k for e in ref) for k in keys}
        exact += cm == cr
        if cm != cr: bad.append(dict(world=w, gpu={f"{a}/{b}": v for (a, b), v in cm.items() if v}, cpu={f"{a}/{b}": v for (a, b), v in cr.items() if v}))
        used = set()
        for e in ref:
            j = next((j for j, x in enumerate(mine) if j not in used and (x["limb"], x["zone"]) == (e["limb"], e["zone"]) and abs(x["t0"] - e["t0"]) < ONSET_TOL), None); n_land += 1
            if j is not None: used.add(j); agree += int(mine[j]["clean"] == e["clean"])
        n_land += len(mine) - len(used)
    W = len(meters); ca = agree / n_land if n_land else float("nan")
    return dict(worlds=W, exact_worlds=exact, exact_share=exact / W if W else None, landed_union=n_land, clean_agree=ca, mismatches=bad[:10],
                ok=bool(W and exact / W >= TOL["exact_worlds"] and n_land and ca >= TOL["clean_agree"]))

def geom_map(mj, rp):
    """{mjlab geom id: replica geom id} by fb_m9_check's geom keys (entity, body, name or mesh, type, ordinal within the body)."""
    import fb_m9_check as M9, mujoco
    def G(m, mjf):
        out, seen = {}, {}
        for g in range(m.ngeom):
            k = M9.geom_key(m, g, mjf)
            if k[0] == "world": continue
            i = seen[k] = seen.get(k, -1) + 1; out[k + (i,)] = g
        return out
    a, b = G(mj, True), G(rp, False); return {a[k]: b[k] for k in a if k in b}, sorted(set(a) ^ set(b), key=str)

def body_map(mj, rp):
    """{mjlab body id: replica body id} by name (robot/ -> '', opponent/ -> opp_)."""
    import fb_m9_check as M9
    key = lambda m, b, f: M9._ent_of(m.body(b).name, f); B = {key(rp, b, False): b for b in range(1, rp.nbody)}
    return {b: B[key(mj, b, True)] for b in range(1, mj.nbody) if key(mj, b, True) in B}

# ============================================================================================ CPU references (the pod's CPU)
def cpu_alone(starts=STARTS, workers=4, clips=TRAIN, modes=("eval",)):
    """M2 G0-3b's eval runs (scripts/eval_tracker.run verbatim, 10 s at 4 m) -> {mode: {(clip, start): fall time}} and the start times."""
    import fb_m2_check as M2, multiprocessing as mp
    jobs = [(c, s, starts, md) for c in clips for s in range(starts) for md in modes]
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool: out = pool.map(M2._alone, jobs, chunksize=1)
    return {md: {(r["clip"], r["start"]): r["fell"] for r in out if r["mode"] == md} for md in modes}, {(r["clip"], r["start"]): r["t0"] for r in out}

def _traj_job(args):
    """The opponent pelvis for TRAJ_STEPS steps of eval_tracker's world (R1HitEnv at 4 m, zero actions) with the eval Tracker or the driver."""
    clip, t0, mode = args
    import torch, fb_opponent as FO
    torch.set_num_threads(1)
    from r1gym.model import build_spec, load_cfg
    from r1gym.env import R1HitEnv
    from r1gym.opponent import FightReplay
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile(); e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=ALONE_S + 1, max_impulse=0, train=False)
    r = FightReplay(m, clip, distance=ALONE_D, cfg=cfg)
    if mode != "eval": FO.MjTracker(r, torch.float32 if mode == "torch32" else torch.float64).install()
    e.opp = r; e.opp_t0 = t0; e.reset(hits=[], delay=0); out = []
    for _ in range(TRAJ_STEPS): e.step(np.zeros(24)); out.append(e.d.xpos[r.pel].copy())
    return (clip, t0, mode), np.array(out)

def cpu_traj(keys, workers=4, mode="eval"):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, len(keys))) as pool: res = pool.map(_traj_job, [(c, t0, mode) for c, t0 in keys], chunksize=1)
    return {(k[0], k[1]): v for k, v in res}

def traj_keys(t0s, n=TRAJ_N):
    """The first n (clip, t0) of the 24 x 3 grid, clips interleaved."""
    ks = sorted(t0s, key=lambda k: (k[1], TRAIN.index(k[0]))); return [(c, t0s[(c, s)]) for c, s in ks[:n]]

# ============================================================================================ GPU (pod only: mjlab, mujoco_warp, CUDA)
def gpu_env(n, edit=None, device="cuda:0", seed=0):
    """The task's env (play cfg) with n worlds; edit(cfg) before it is built."""
    for p in (UNITREE, UNITREE / "scripts"):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    import torch, mjlab.tasks, src.tasks  # noqa: F401
    import fullbody_task as FT
    from mjlab.envs import ManagerBasedRlEnv
    torch.manual_seed(seed); cfg = FT.env_cfg("A", play=True); cfg.scene.num_envs = n; cfg.seed = seed
    if edit: edit(cfg)
    return ManagerBasedRlEnv(cfg=cfg, device=device)

def gpu_policy(name, dev):
    import torch, hybrid_actor as HA, fullbody_task as FT
    if name == "H":
        sc, clip, _ = FT.eval_contract(); a = HA.HybridActor.from_checkpoints(ROOT, sc, clip, False).to(dev).eval(); return lambda o: a.mean(o)
    n = HA.Net.from_actor_sd(HA.load_actor_sd(ROOT / (HA.G_PT if name == "G" else A3_PT))).to(dev).eval(); w = n.n_in
    return lambda o: n(o[:, :w])

def _t0_only(env):
    """Re-arm the population's T0-only window (A11.2: this process's iterations 1-100) before a batch's first reset."""
    env.command_manager.get_term("opp").nstep = 0

def gpu_r4g(ctrls=("G", "H", "A3"), N=R4G_N, worlds=4096, steps=R4G_STEPS, seed=0, b3=True, log=print):
    """(3) and (5): first-episode rows per controller from batches of `worlds`, and the B3 asserts on every reset."""
    import torch, fb_obs as OB
    env = gpu_env(worlds, seed=seed); dev = env.device; c = env.command_manager.get_term("opp"); rows = {}; B = dict(ring_bad=0, kp_bad=0, first_obs=0, cmd_bad=0, eps=0, ad=0, k=[0, 0, 0], switches=0)
    base_reset = c.reset; WK = dict(w_cmd_bad="cmd_bad", w_eps="eps", w_ad="ad", w_switch="switches")
    def reset(env_ids):
        out = base_reset(env_ids)
        for k, v in out.items():   # FbOppCommand.reset emits its WATCH6 sums since the last reset batch (fullbody_task)
            if k in ("w_k0", "w_k1", "w_k2"): B["k"][int(k[-1])] += float(v)
            elif k in WK: B[WK[k]] += float(v)
        return out
    c.reset = reset
    def b3_check(ids):
        """Every ring row of the reset envs = the fresh (clean) row; the keypoint block = one built from the placed opponent now."""
        if not len(ids): return
        ob = env.fb_obs_term; ring, clean = ob.obs.ring[ids], ob.obs.clean[ids]; B["first_obs"] += len(ids); B["ring_bad"] += int((ring != clean[:, None]).any(-1).any(-1).sum())
        r, o = env.scene["robot"].data, env.scene["opponent"].data; X = torch.cat([o.body_link_pos_w[ids][:, ob.kb], r.body_link_pos_w[ids][:, ob.tb][:, None]], 1)
        Q = torch.cat([o.body_link_quat_w[ids][:, ob.kb], r.body_link_quat_w[ids][:, ob.tb][:, None]], 1); P_T, y, v = OB.kp_state(X, Q)
        h = OB.KeypointHistory(len(ids), torch.float32, dev); blk = h.block(P_T, y, v); B["kp_bad"] += int((blk != clean[:, OB.PROPRIO_DIM:]).any(-1).sum())
    for name in ctrls:
        pol = gpu_policy(name, dev); R = []
        for j in range(math.ceil(N / worlds)):
            _t0_only(env); obs, _ = env.reset(); n = env.num_envs
            if b3: b3_check(torch.arange(n, device=dev))
            first = torch.ones(n, dtype=torch.bool, device=dev); fell, kd = torch.zeros_like(first), torch.zeros_like(first); hc = torch.zeros(n, device=dev); st = torch.zeros(n, dtype=torch.long, device=dev)
            ad, kk = env.fb_latency.act_delay.clone(), env.fb_latency.k.clone(); t0 = time.time()
            for _ in range(steps):
                with torch.no_grad(): a = pol(obs["actor"])
                obs, _, term, tout, _ = env.step(a); st += first.long(); hc += c.acct.step["head_clean"] * first; kd |= c.edge & first; fell |= term & first; done = term | tout
                if b3: b3_check(torch.nonzero(done).flatten())
                first &= ~done
            R += rows_from_first(fell.cpu(), kd.cpu(), hc.cpu(), st.cpu(), ad.cpu(), kk.cpu()); log(json.dumps(dict(item="r4g", ctrl=name, batch=j, worlds=n, wall_s=round(time.time() - t0, 1))), flush=True)
        rows[name] = R[:N]
    c.reset = base_reset; sh = shares_ok(B["eps"], B["ad"], B["k"])
    b3r = dict(first_obs=B["first_obs"], ring_bad=B["ring_bad"], kp_bad=B["kp_bad"], cmd_bad=B["cmd_bad"], switches=B["switches"], shares=sh,
               ok=bool(B["first_obs"] > 0 and B["ring_bad"] == 0 and B["kp_bad"] == 0 and B["cmd_bad"] == 0 and B["switches"] == 0 and sh["ok"]))
    return rows, b3r

def _alone_edit(grid):
    """cfg edits for the tracker-alone worlds: the population draw scripted per world (clip, t0, 4 m, no aim, T0), the defender at heading 0,
    fell_eval off (the eval_tracker loop never terminates), episodes ALONE_S + 1."""
    import torch, fullbody_task as FT
    def draw(env, env_ids, steps_per_env=24):
        c = env.command_manager.get_term("opp"); n = len(env_ids); ids = env_ids.cpu()
        d = dict(row=torch.zeros(n, dtype=torch.long), clip=torch.tensor([TRAIN.index(grid[i][0]) for i in ids.tolist()]), dist_ix=torch.zeros(n, dtype=torch.long),
                 distance=torch.full((n,), ALONE_D, dtype=torch.float64), t0=torch.tensor([grid[i][1] for i in ids.tolist()], dtype=torch.float64), aimed=torch.zeros(n, dtype=torch.bool),
                 jitter=torch.zeros(n, dtype=torch.bool), conj=torch.zeros(n, dtype=torch.bool), t1=torch.zeros(n, dtype=torch.bool), opp_dr=torch.zeros(n, dtype=torch.bool), push=torch.zeros(n, dtype=torch.bool))
        if not hasattr(env, "fb_pop"): env.fb_pop = {k: torch.zeros(env.num_envs, dtype=v.dtype) for k, v in d.items()}
        for k, v in d.items(): env.fb_pop[k][ids] = v
    def reset_def(env, env_ids):
        r = env.scene[FT.ENT]; n = len(env_ids); dev = env.device; pos = env.scene.env_origins[env_ids].clone(); pos[:, 2] += FT.DEF_Z
        quat = torch.zeros(n, 4, device=dev); quat[:, 0] = 1.0; r.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(n, 6, device=dev)], 1), env_ids=env_ids)
        jid = torch.as_tensor(r.find_joints(list(FT.FO_CTRL()), preserve_order=True)[0], device=dev); home = torch.as_tensor(FT.eval_contract()[2]["home"], dtype=torch.float32, device=dev)
        r.write_joint_state_to_sim(home.expand(n, -1).clone(), torch.zeros(n, 24, device=dev), joint_ids=jid, env_ids=env_ids)
    def edit(cfg):
        cfg.events["fb_draw_population"].func = draw; cfg.events["fb_reset_defender"].func = reset_def; cfg.events.pop("fb_dr", None)
        cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("fb_update", "time_out", "nan")}; cfg.episode_length_s = ALONE_S + 1
    return edit

def gpu_alone(grid, steps, device="cuda:0"):
    """(1)/(2) GPU half: worlds on grid [(clip, t0)], zero defender actions, the aim bypassed (eval_tracker: opponent_aim none) -> (fall
    times {(clip, idx): t or None} as eval_tracker reports them (round(k x dt, 2), k the 0-based step), pelvis paths [steps, 3] env frame)."""
    import torch
    env = gpu_env(len(grid), _alone_edit(grid), device); c = env.command_manager.get_term("opp"); c.aim.step = lambda fi, now, rq, rdq, *a, **k: (rq, rdq, None)
    obs, _ = env.reset(); n = env.num_envs; fall = [None] * n; P = []; o = env.scene["opponent"].data
    for k in range(steps):
        obs, *_ = env.step(torch.zeros(n, 24, device=env.device)); P.append((o.body_link_pos_w[:, c.o_pel] - env.scene.env_origins).cpu().numpy())
        for i in torch.nonzero(c.od.now).flatten().tolist():
            if fall[i] is None: fall[i] = round(k * DT, 2)
    return fall, np.stack(P, 1)

def gpu_hit(N=HIT_N, steps=R4G_STEPS, ctrl="H", seed=0):
    """(4) GPU half: per world its first episode's dumped substeps (replica ids) and the account's episode log; then the replay."""
    import torch, warp as wp, mujoco, fb_replica as RP, fb_m6_run as M6, export_hybrid_fb as EX, fullbody_task as FT
    env = gpu_env(N, seed=seed); dev = env.device; c = env.command_manager.get_term("opp"); c.acct.log = []; sim = env.sim; mj = sim.mj_model
    _, _, cfg = EX.env_scale(); rp, _ = RP.build_model({**cfg, "opponent_mode": "policy", "opponent_aim": EX.AIM}, RP.conf(hulls=FT.hulls_cfg(), solver=FT.solver_cfg()))
    gm, miss = geom_map(mj, rp); bm = body_map(mj, rp); import r1gym.hits as H; hm0 = H.HitMeter(rp)
    rel = torch.zeros(mj.ngeom, dtype=torch.bool, device=dev); r1 = torch.zeros_like(rel); op = torch.zeros_like(rel)
    for a, b in gm.items(): r1[a] = bool(hm0.is_r1[b]); op[a] = bool(hm0.is_opp[b])
    G2R = torch.full((mj.ngeom,), -1, dtype=torch.long, device=dev); G2R[list(gm)] = torch.as_tensor(list(gm.values()), device=dev)
    root = torch.as_tensor(mj.body_rootid, device=dev); gb = torch.as_tensor(mj.geom_bodyid, device=dev); subs = [[] for _ in range(N)]; live = torch.ones(N, dtype=torch.bool, device=dev); j, on = [0], [False]
    D = sim.wp_data; F = lambda *names: next(getattr(o, k) for o in (D, getattr(D, "contact", None), getattr(D, "efc", None)) if o is not None for k in names if hasattr(o, k))
    def dump():
        n = int(wp.to_torch(D.nacon)[0]); C = D.contact; g = wp.to_torch(C.geom)[:n].long(); w = wp.to_torch(C.worldid)[:n].long()
        m_ = ((r1[g[:, 0]] & op[g[:, 1]]) | (r1[g[:, 1]] & op[g[:, 0]])) & live[w]
        if not bool(m_.any()): return
        # the normal force as mj_contactForce gives it: one efc row (condim 1, elliptic), the sum of the 2 (dim - 1) pyramid edges (pyramidal
        # condim 3: the feet, priority 1, so foot contacts with the shells are condim 3; M10 verify)
        ea = wp.to_torch(C.efc_address)[:n].long(); dm = wp.to_torch(C.dim)[:n].long(); nr = torch.where(dm > 1, 2 * (dm - 1), 1) if int(mj.opt.cone) == 0 else torch.ones_like(dm)
        if ea.dim() == 1: ea = ea[:, None] + torch.arange(int(nr.max()), device=dev)[None]
        use = torch.arange(ea.shape[1], device=dev)[None] < nr[:, None]; assert bool((ea.shape[1] >= nr).all()) and bool((ea[use] >= 0).all()), "efc rows of a contact missing"
        ef = wp.to_torch(F("force", "efc_force")); fr_ = ef[w[:, None], ea.clamp(min=0)] if ef.dim() == 2 else ef[ea.clamp(min=0)]
        fn = (fr_ * use).sum(1)[m_]; g, w = g[m_], w[m_]; pos, fr = wp.to_torch(C.pos)[:n][m_], wp.to_torch(C.frame)[:n][m_].reshape(-1, 9)
        cv, xi, sc = sim.data.cvel, sim.data.xipos, sim.data.subtree_com; t = (env.episode_length_buf.double() * DT + (j[0] + 1) * 0.005)
        for wi in torch.unique(w).tolist():
            k = w == wi; gg = g[k]; bb = torch.unique(gb[gg.flatten()]).tolist()
            subs[wi].append(dict(t=round(float(t[wi]), 4), geom=G2R[gg].cpu().numpy(), pos=pos[k].cpu().numpy(), frame=fr[k].cpu().numpy(), fn=fn[k].cpu().numpy(),
                                 bodies={bm[b]: (cv[wi, b].cpu().numpy(), xi[wi, b].cpu().numpy(), sc[wi, int(root[b])].cpu().numpy()) for b in bb}))
    base = c.opp.hook
    def hook():   # substep j[0] of the current env.step only (counted from the step's start; resets' and forwards' updates are not dumped; M10 verify)
        base()
        if on[0] and j[0] < FT.DECIM: dump()
        j[0] += 1
    c.opp.hook = hook; pol = gpu_policy(ctrl, dev); _t0_only(env); obs, _ = env.reset(); cut = [None] * N
    for _ in range(steps):
        with torch.no_grad(): a = pol(obs["actor"])
        j[0] = 0; on[0] = True; obs, _, term, tout, _ = env.step(a); on[0] = False; assert j[0] >= FT.DECIM, f"{j[0]} substep hooks in a step"; done = (term | tout) & live
        for i in torch.nonzero(done).flatten().tolist(): cut[i] = len(c.acct.log)
        live &= ~done
    c.opp.hook = base; L = len(c.acct.log); logs = [[e for e in c.acct.log[:cut[i] if cut[i] is not None else L] if e["env"] == i] for i in range(N)]
    meters = [hit_replay(rp, subs[i]) for i in range(N)]; res = hit_compare(logs, meters); res["unmapped_geoms"] = [str(x) for x in miss][:10]; return res

def runner_smoke(iters=3, ne=64, log=print):
    """(6) the real runner, A0 -> A -> B, `iters` iterations each, chained from start_sighted (pod_fb.sh's layout under $U/logs)."""
    import torch, hybrid_actor as HA
    runs = UNITREE / "logs" / "rsl_rl" / "r1_fullbody"; out = {}; prev = ("start_sighted", "model_0.pt"); G = HA.load_actor_sd(ROOT / HA.G_PT)
    for st, task in (("A0", "R1-Fb-A"), ("A", "R1-Fb-A"), ("B", "R1-Fb-B")):
        nm = f"p1_smoke_{st}"; cmd = [sys.executable, str(ROOT / "gpu" / "fullbody_task.py"), task, "--env.scene.num-envs", str(ne), "--agent.max-iterations", str(iters),
                                      "--agent.resume", "True", "--agent.load-run", prev[0], "--agent.load-checkpoint", prev[1], "--agent.run-name", nm]
        r = subprocess.run(cmd, cwd=str(UNITREE), env=dict(os.environ, FB_STAGE=st), capture_output=True, text=True); ck = sorted(runs.glob(f"*_{nm}/model_*.pt"), key=lambda p: p.stat().st_mtime)
        o = dict(rc=r.returncode, tail=r.stdout[-400:] + r.stderr[-400:], ckpt=str(ck[-1]) if ck else None)
        if ck:
            sd = torch.load(ck[-1], map_location="cpu", weights_only=False)["actor_state_dict"]; o.update(blind=HA.stored_blind(sd), residual=any(k.startswith("residual.") for k in sd),
                                                                                                legs_equal_G=all(torch.equal(sd["legs." + k], v) for k, v in G.items() if not k.startswith("distribution.")))
            prev = (ck[-1].parent.name, ck[-1].name)
        o["ok"] = bool(r.returncode == 0 and ck and o["blind"] is False and o["residual"] == (st == "B") and (o["legs_equal_G"] or st == "B")); out[st] = o; log(json.dumps(dict(item="runner", stage=st, rc=r.returncode)), flush=True)
    return dict(stages=out, ok=all(v["ok"] for v in out.values()))

# ============================================================================================ drivers
def run_pod(out, items, r0eo, r0eo_laptop=None, worlds=4096, seed=0, workers=4, log=print):
    import fb_r0eo as R0
    res, t00 = {}, time.time(); save = lambda: Path(out).write_text(json.dumps(dict(res, ok=all(v.get("ok") for v in res.values()), wall_s=round(time.time() - t00, 1)), indent=1, default=float), encoding="utf-8")
    if "t0_alone" in items or "restricted" in items:
        ref, t0s = cpu_alone(workers=workers); grid = [(c, t0s[(c, s)]) for c in TRAIN for s in range(STARTS)]; idx = {(c, s): i for i, (c, s) in enumerate((c, s) for c in TRAIN for s in range(STARTS))}
        fall, P = gpu_alone(grid, int(ALONE_S / DT))
        if "t0_alone" in items: res["t0_alone"] = alone_compare(ref["eval"], {k: fall[i] for k, i in idx.items()}); save()
        if "restricted" in items:
            keys = traj_keys(t0s); cr = cpu_traj(keys, workers); inv = {(c, t0s[(c, s)]): i for (c, s), i in idx.items()}
            res["restricted"] = traj_compare(cr, {k: P[inv[k], :TRAJ_STEPS] for k in cr}); save()
    if "r4g" in items or "b3" in items:
        rows, b3 = gpu_r4g(seed=seed, worlds=worlds, b3="b3" in items, log=log)
        if "r4g" in items:
            eo = R0.load(r0eo); res["r4g"] = dict(reference=str(r0eo), **r4g_verdict(eo, rows), seed=seed)
            if r0eo_laptop: res["r4g"]["vs_laptop_reference"] = r4g_verdict(R0.load(r0eo_laptop), rows)["gates"]
        if "b3" in items: res["b3"] = b3
        save()
    if "hit" in items: res["hit"] = gpu_hit(seed=seed); save()
    if "runner" in items: res["runner"] = runner_smoke(log=log); save()
    save(); return res

def run_dry(out, ckdir=None, r0eo_laptop=None, seeds=range(69000, 69008), log=print):
    """Every item's CPU half and comparator on CPU stand-ins (module docstring)."""
    import torch, mujoco, fb_replica as RP, fb_m6_run as M6, fb_r0eo as R0, fb_cpu_smoke as SM
    res, t00 = {}, time.time(); torch.set_num_threads(1)
    # (1) / (2): the eval vs the torch32 driver, 2 starts x 3 clips
    ref, t0s = cpu_alone(starts=2, workers=2, modes=("eval", "torch32")); res["t0_alone"] = alone_compare(ref["eval"], ref["torch32"])
    keys = [(c, t0s[(c, s)]) for c in TRAIN for s in range(2)]; res["restricted"] = traj_compare(cpu_traj(keys, 2, "eval"), cpu_traj(keys, 2, "torch32"))
    # (3): replica rows (REQ_CONF, the training reset, T0 nominal) on smoke seeds as the 'GPU' worlds vs the laptop R0-EO (read only)
    from fb_seeds import check
    eps = [SM.episode(s, SM.PLUMB["block"][0]) for s in seeds]
    for e in eps: check([e["seed"]], "smoke", {e["clip"]})
    import fb_t1c as TC
    eps = [TC._eo_episode(e) for e in eps]; c = RP.conf(**dict(M6.REQ_CONF, reset="train"))
    gpu = {n: RP.run(n, eps, c, batch=len(eps), rowwise=True) for n in ("G", "H", "A3")}
    ref = Path(r0eo_laptop or R0.SCR / "fullbody" / "r0eo" / "out"); eo = R0.load(ref, shards=range(2)) if ref.exists() else None
    res["r4g"] = dict(r4g_verdict(eo, gpu), reference=str(ref), note="dry: 8 replica bouts per controller vs 2 laptop shards (384 EO rows each): plumbing only, no verdict") if eo else dict(ok=None, note="no R0-EO rows")
    # (4): a replica bout's contacts dumped and replayed through the HitMeter shim; vs its live HitMeter (exact) and its account (G0-11-like)
    res["hit"] = dry_hit(eps[:4])
    # (5): B3 asserts on the smoke env
    res["b3"] = dry_b3(seeds)
    # (6): commands and the M10 checkpoints as the runner check reads them
    res["runner"] = dry_runner(ckdir)
    res["hit"]["geom_map"] = gm = dry_geom_map(); res["hit"]["ok"] = bool(res["hit"]["ok"] and gm["ok"])   # last: it stubs warp / scipy in this process
    res["ok"] = all(v.get("ok") for k, v in res.items() if isinstance(v, dict) and k != "r4g"); res["wall_s"] = round(time.time() - t00, 1)
    Path(out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8"); return res

def dry_hit(eps):
    """The recorder on replica bouts (scorer 'account', H, REQ_CONF): replay == live HitMeter bit for bit (events), and the account vs the
    replay through hit_compare (P1-Hit's comparator)."""
    import torch, mujoco, fb_replica as RP, fb_m6_run as M6
    c = RP.conf(**M6.REQ_CONF); ctrl, cfg, lag, m, names = RP.world("H", c); logs, meters, same = [], [], 0
    for ep in eps:
        B = RP.Bout(ctrl, cfg, c, m, native=names, lag=lag); subs = []
        class Rec:
            def reset(self): pass
            def substep(self, m_, d): subs.append(dump_rows(m_, d, B.hm, round(float(d.time), 4)))
        B_reset = B.reset
        def reset(ep_, B=B, base=B_reset):
            o = base(ep_); B.hooks.append(Rec()); return o
        B.reset = reset; row, info = RP.run_bout(B, ep); live = B.hm; rp = hit_replay(m, subs)
        same += json.dumps(live.events(), sort_keys=True, default=float) == json.dumps(rp.events(), sort_keys=True, default=float)
        logs.append([dict(x, env=0) for x in info["acct"]["log"]] if "acct" in info else []); meters.append(rp)
    out = hit_compare(logs, meters); out["replay_equals_live"] = f"{same}/{len(eps)}"; out["ok"] = bool(out["ok"] and same == len(eps)); return out

def dry_geom_map():
    """mjlab's real Scene compiled on CPU with stubs (fb_m9_check) vs the replica: every HitMeter-relevant geom maps one to one."""
    import fb_m9_check as M9, fullbody_task as FT, r1gym.hits as H
    M9.stub_mjlab(); mj, _ = M9.mjlab_model(FT.hulls_cfg(), FT.solver_cfg()); rp = M9.replica_model(FT.hulls_cfg(), FT.solver_cfg()); gm, miss = geom_map(mj, rp)
    hm = H.HitMeter(rp); need = {g for g in range(rp.ngeom) if hm.is_r1[g] or hm.is_opp[g]}; got = set(gm.values()); bm = body_map(mj, rp)
    return dict(mapped=len(gm), unmapped=len(miss), hit_geoms=len(need), hit_geoms_mapped=len(need & got), bodies_mapped=len(bm), bodies=rp.nbody - 1, ok=bool(need <= got and len(set(gm.values())) == len(gm) and len(bm) == rp.nbody - 1))

def dry_b3(seeds):
    """B3 asserts on fb_cpu_smoke's env (the replica's reset order): at every first observation the ring rows equal the fresh row and the
    keypoint velocities are 0; act_delay / k constant (switch counters)."""
    import torch, fb_cpu_smoke as SM, fb_obs as OB
    S = SM.Seeds("smoke", min(seeds), max(seeds), SM.PLUMB["block"][0]); env = SM.SmokeEnv(4, S); a = SM.make_actor("A").eval(); bad = first = 0
    def chk(ids):
        nonlocal bad, first
        for i in ids:
            first += 1; R, cl = env.B.obs.ring[i], env.B.obs.clean[i]; bad += int(not bool((R == cl[None]).all())) + int(not bool((cl[OB.PROPRIO_DIM:][OB.VEL] == 0).all()))
    chk(range(4))
    with torch.no_grad():
        for _ in range(50):
            o, r, d, t, info = env.step(a({"actor": env.obs, "critic": env.critic_obs()}).numpy()); chk(torch.nonzero(d).flatten().tolist())
    sw = int(env.B.obs.k_switches) + int(env.B.act.switches) + env.log["switch_ep"]
    return dict(first_obs=first, bad=bad, switches=sw, ok=bool(first > 0 and bad == 0 and sw == 0), note="replica order; the GPU asserts run on the pod")

def dry_runner(ckdir):
    import torch, hybrid_actor as HA
    cmds = [f"FB_STAGE={st} python {ROOT / 'gpu' / 'fullbody_task.py'} {t} --env.scene.num-envs 64 --agent.max-iterations 3 --agent.resume True --agent.load-run <prev> --agent.run-name p1_smoke_{st}"
            for st, t in (("A0", "R1-Fb-A"), ("A", "R1-Fb-A"), ("B", "R1-Fb-B"))]
    out = dict(commands=cmds)
    if ckdir:
        G = HA.load_actor_sd(ROOT / HA.G_PT); chk = {}
        for st in ("A0", "A", "B"):
            sd = torch.load(Path(ckdir) / f"ckpt_{st}.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
            chk[st] = dict(blind=HA.stored_blind(sd), residual=any(k.startswith("residual.") for k in sd), legs_equal_G=all(torch.equal(sd["legs." + k], v) for k, v in G.items() if not k.startswith("distribution.")))
        out["m10_ckpts"] = chk; out["ok"] = all(v["blind"] is False and v["residual"] == (s == "B") and v["legs_equal_G"] for s, v in chk.items())
    else: out["ok"] = True
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--all", action="store_true"); ap.add_argument("--items", default=",".join(ITEMS))
    ap.add_argument("--r0eo", default=os.environ.get("FB_R0EO", str(FB_ROOT / "r0eo_pod"))); ap.add_argument("--r0eo-laptop"); ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--dry", action="store_true"); ap.add_argument("--ckdir")
    a = ap.parse_args()
    if a.dry: r = run_dry(a.out, a.ckdir, a.r0eo_laptop)
    else:
        items = ITEMS if a.all else tuple(x for x in a.items.split(",") if x)
        assert not ({"r4g"} & set(items)) or Path(a.r0eo).exists(), f"P1-R4g needs the R0-EO pod reference (A8.2) at {a.r0eo} (--r0eo / FB_R0EO)"
        r = run_pod(a.out, items, a.r0eo, a.r0eo_laptop, a.worlds, a.seed, a.workers)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in r.items()}), flush=True); sys.exit(0 if r.get("ok") or all(v.get("ok") for v in r.values() if isinstance(v, dict)) else 1)
