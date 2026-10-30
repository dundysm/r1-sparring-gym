"""M5 checks of the full-body hit account and reward terms (M2V2_PLAN_v3 3 M5; gpu/fb_hits.py, gpu/fb_rewards.py) on CPU against the
eval's own code:
  record  EO bouts with contact logs: scripts/fb_record.py's episodes (seed s -> clip TRAIN[(s - 60000) % 3], distance HARD[((s - 60000)
          // 3) % 4]; aim standoff+armwarp_vz, DR, delays as drawn, no pushes, 20 s) through eval_opponent.run (meter, falls, rev_hits on)
          for G and P1's H(A3s)/M3s (keypoints truth), blocks 60200-60263 (role 'hit_scoring', G0-11) and 60300-60363 ('w0_calib',
          G0-12). Read-only hooks: an R1HitEnv subclass appends a substep logger after the eval's own hooks (fb_hits.MjContactSensors
          both ways: hit and body sensors, sparse) and logs the post-step pelvis / torso state; after the bout the HitMeter's episodes
          and intent-window outcomes (HitMeter.windows on the aim's intents, as meter_row scores them) and a python HIT_R2 + m2 pay (block-shell-first: run 2 bz) on
          the HitMeter's own per-side episodes. Each bout also runs without hooks (rows identical, asserted).
  replay  FullBodyHitAccount (float64, batched over a block's bouts) on the logged sensors, intents in AimOp-like slots (K 8), the
          closing speed from body COM velocities ('com', the GPU's input) or the HitMeter's contact-point velocities ('point');
          FbRewards on the logged states (fb_events.OppDown on the post-step opponent pelvis, as EO:86).
  G0-11   same-state scoring on 60200-60263 (EO states; the plan's replica-mode run is M6 R1's: run again there), 'com': landed counts
          per (limb, zone) exact vs the HitMeter on every bout; clean-vs-not agreement >= 0.95 over landed episodes (unmatched count as
          disagreements); median landed head impulse within 10%. Reported: 'point', window outcomes and pay vs the HitMeter /
          python reference, all-zone counts, the reverse direction (R1 -> opponent) vs HitMeter(reverse=True).
  G0-12   scripted events (scripted()) and the recorded EO bouts: W0 (A4.1) = the 10th percentile of W over G's knockdowns with contact
          (a defender torso / pelvis / leg shell touched the opponent in the last 1 s) on the R4-pre G knockdown bouts re-recorded with
          contact logs (role 'w0_r4pre': fb_seeds.W0_R4PRE, re-derived from r4pre/runs/G_*.pkl and asserted equal; R4-pre's own episodes,
          fb_r0eo.episodes; G only; each bout hooked = plain), at least W0_MIN_N such knockdowns (else stop: no W0), frozen in
          fb_rewards.W0 (asserted equal); every listed bout has exactly one knockdown edge at EO's opp_fall time; the 60300-60363 value
          is report only. On every recorded bout (60200, 60300, w0_r4pre): each charged knockdown is one step of -2.4 at a rising edge
          with W >= W0 and no landed strike within 0.3 s; fb_down_neutral nonzero only while opp_down_now (and after the opponent got up,
          zero again); fb_down_approach / fb_down_contact zero while both are up; the tangle impulse = the per-substep sum of |F| dt over
          the pelvis + leg body-sensor entries (independent numpy). Reported: the re-recorded rows vs R4-pre's G rows, and (A4.2) the
          knockdowns before EARLY_S.
Run in the repo: python scripts/fb_m5_check.py --out m5_check.json --rec m5rec.npz --rec-w0 w0rec.npz --p1 p1_eval.py [--r4pre DIR]
                 [--workers 4] [--reuse]"""
import sys, gc, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
import fb_hits as HT, fb_rewards as RW, fb_events as EV

TOL = dict(clean_agree=0.95, med_imp=0.10, w0_pct=10.0, slots=8)
BLOCKS = dict(hit_scoring=(60200, 60263), w0_calib=(60300, 60363)); TRAIN = ("kfab_200", "kfab_201", "kfab_203"); CTRLS = ("G", "H")
W0_ROLE, W0_MIN_N, EARLY_S = "w0_r4pre", 30, 0.5   # A4.1 (W0 on the R4-pre G knockdown bouts, >= 30 with contact), A4.2 (report)
R4PRE_BLOCKS = dict(main=(67000, 64, (0.4, 0.45, 0.5, 0.55)), close=(68000, 128, (0.4, 0.45)))   # r4pre_eval.episodes(seed0, reps, dists)
R4PRE = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody\r4pre")
HIT_COLS = 13   # k (substep), sensor, primary, found, F (3), v_com (3), v_pt (3)
BODY_COLS = 8   # k (substep), primary, F (3), v (3)
POST = ("time", "opp_z", "opp_up", "def_x", "def_y", "def_vx", "def_vy", "def_vz", "opp_x", "opp_y", "up_x", "up_y", "up_z", "fallen")

# ---- recording
class Logger:
    """Substep hook (after the eval's own): both sensor sets, sparse; never writes d."""
    def __init__(self, m, S):
        self.f, self.r, self.S, self.k = HT.MjContactSensors(m), HT.MjContactSensors(m, reverse=True), S, 0
    def reset(self): pass
    def substep(self, m, d):
        k = self.k; self.k += 1
        for sens, hk, bk in ((self.f, "hit", "body"), (self.r, "rhit", "rbody")):
            F, fo, vc, vp, Fb, vb = sens.read(d)
            for l, p in zip(*np.nonzero(fo)): self.S[hk].append([k, l, p, fo[l, p], *F[l, p], *vc[l, p], *vp[l, p]])
            for q in np.flatnonzero(np.any(Fb != 0, 1)): self.S[bk].append([k, q, *Fb[q], *vb[q]])

def _hooks(S):
    import eval_opponent as EO
    base = EO.R1HitEnv
    class RecEnv(base):
        def reset(self, *a, **k):
            o = super().reset(*a, **k); S["env"] = self; self.substep_hooks.append(Logger(self.m, S)); return o   # after the meter: hooks[0] stays the HitMeter
        def step(self, a):
            out = super().step(a); d, r = self.d, self.r; b = min(self.opp.bodies); q, v = r.free_q, r.free_v; R = d.xmat[r.torso]
            S["post"].append([d.time, d.xpos[b][2], d.xmat[b][8], d.qpos[q], d.qpos[q + 1], d.qvel[v], d.qvel[v + 1], d.qvel[v + 2], d.xpos[b][0], d.xpos[b][1], R[2], R[5], R[8], float(out[3]["fallen"])])
            return out
    def install(): EO.R1HitEnv = RecEnv
    def remove(): EO.R1HitEnv = base
    return install, remove

def pay_ref(ev_side, t_on, t_apex, limb, p, arm_block=None):
    """HIT_R2 + m2 on the HitMeter's per-side episodes (python): blocked as HitMeter.windows, the arm episode opened on a block shell
    (arm_block(side, limb group, t0) -> bool: an elbow / wrist shell touched at that substep, run 2's bz; None: always), by apex +
    0.05 s, and no landed head / torso episode of the side's limb or its partner that opened in the window."""
    from r1gym.hits import _landed
    side, g = limb.split("_", 1); part = HT.LIMBS[HT.PARTNER[HT.LIMBS.index(g)]]; t0w, t1w = round(t_on, 4), round(t_apex + HT.WIN_TAIL_S, 4)
    inw = lambda e: side in e["side"] and t0w <= e["t0"] <= t1w
    mine = sorted((e for e in ev_side if e["limb"] == g and inw(e)), key=lambda e: e["t0"])
    hit = next((e for e in mine if _landed(e, p)), None); arm = next((e for e in mine if e["zone"] == "arm"), None)
    bl = bool(arm and (hit is None or arm["t0"] < hit["t0"])); anyl = any(_landed(e, p) for e in ev_side if e["limb"] in (g, part) and inw(e))
    return bool(bl and (arm_block is None or arm_block(side, g, arm["t0"])) and arm["t0"] <= round(t_apex + HT.BLOCK_BY_APEX_S, 4) + 1e-9 and not anyl)

def block_touches(hit, post, block_idx):
    """{(sensor, t)}: the logged substeps (HitMeter's rounded time) a limb sensor touched a block shell; t of substep k = the post-step
    time of step k // 4 minus (3 - k % 4) substeps."""
    bk = np.asarray(block_idx, bool); h = hit[bk[hit[:, 2].astype(int)] & (hit[:, 3] > 0)] if len(hit) else hit
    return {(int(r[1]), round(float(post[int(r[0]) // 4, 0]) - (3 - int(r[0]) % 4) * HT.DT, 4)) for r in h}

_R4E = {}
def r4pre_episodes(seeds):
    """R4-pre's own episodes (fb_r0eo.episodes = r4pre_eval.episodes on both blocks, R0E check (2)) of the given seeds."""
    if not _R4E:
        import fb_r0eo as R0
        for s0, reps, ds in R4PRE_BLOCKS.values(): _R4E.update({e["seed"]: e for e in R0.episodes(s0, reps, ds)})
    return [_R4E[int(s)] for s in seeds]

def r4pre_list(r4pre=R4PRE):
    """A4.1's list from R4-pre's G run files: (sorted seeds where G's bout had a knockdown, {seed: R4-pre G row}, {file: sha256})."""
    import pickle, hashlib
    rows, sha = {}, {}
    for t in R4PRE_BLOCKS:
        b = (Path(r4pre) / "runs" / f"G_{t}.pkl").read_bytes(); sha[f"G_{t}.pkl"] = hashlib.sha256(b).hexdigest(); d = pickle.loads(b)
        assert d["meta"]["ctrl"] == "guard_fw_r" and d["meta"]["variant"] == "plain"; rows.update({r["seed"]: r for r in d["rows"]})
    return sorted(s for s, r in rows.items() if r["knockdown"]), rows, sha

def work(args):
    jobs, seconds, p1 = args
    torch.set_num_threads(1)
    import eval_opponent as EO, fb_record as REC, fb_m4_check as C4
    from fb_seeds import check
    from r1gym.hits import STRIKE_LIMBS
    out = []
    for ctrl, role, seeds in jobs:
        check(seeds, role, TRAIN, ctrl=ctrl); c, cfg, m = C4.setup(ctrl, p1); kp = "truth" if ctrl == "H" else None
        for ep in (r4pre_episodes(seeds) if role == W0_ROLE else REC.episodes(seeds, seconds)):
            kw = dict(threads=1, dr=True, meter=True, falls=True, rev=True, keypoints=kp)
            plain = EO.run(c, [ep], m, cfg, seconds, **kw)[0]
            S = dict(hit=[], body=[], rhit=[], rbody=[], post=[]); install, remove = _hooks(S); install()
            try: row = EO.run(c, [ep], m, cfg, seconds, **kw)[0]
            finally: remove()
            assert json.dumps(row, sort_keys=True, default=str) == json.dumps(plain, sort_keys=True, default=str), f"the hooks changed bout {ctrl} {ep['seed']}"
            e = S["env"]; hm = e.substep_hooks[0]; assert type(hm).__name__ == "HitMeter" and not hm.reverse
            secs = row["fell_s"] or seconds; aim = e.opp_state["aim"]; tail = aim.p["window_tail_s"]; assert tail == HT.WIN_TAIL_S
            its = [it for it in aim.intents if it.t_on >= 0 and it.t_apex + tail <= secs]; oc = hm.windows([it.window(tail) for it in its]); evs = hm.events(by_side=True)
            lg = [h for h in e.substep_hooks if isinstance(h, Logger)][0]; bt = block_touches(np.array(S["hit"], np.float64).reshape(-1, HIT_COLS), np.array(S["post"], np.float64).reshape(-1, len(POST)), lg.f.block_idx)
            ab = lambda sd, g, t0: (HT.SIDES.index(sd) * 4 + HT.LIMBS.index(g), round(t0, 4)) in bt
            ints = sorted((dict(limb=it.limb, t_on=it.t_on, t_apex=it.t_apex, gated=it.gated, outcome=o, pay=pay_ref(evs, it.t_on, it.t_apex, it.limb, hm.p, ab),
                                pay_any_arm=pay_ref(evs, it.t_on, it.t_apex, it.limb, hm.p)) for it, o in zip(its, oc)), key=lambda x: x["t_on"])
            ev = [dict(limb=x["limb"], zone=x["zone"], t0=x["t0"], t1=x["t1"], impulse_Ns=x["impulse_Ns"], peak_N=x["peak_N"], v_close=x["v_close"], dur_s=x["dur_s"],
                       landed=bool(hm.landed(x)), clean=bool(hm.clean(x))) for x in hm.events() if x["limb"] in STRIKE_LIMBS]
            rv = [h for h in e.substep_hooks[1:] if type(h).__name__ == "HitMeter" and h.reverse][0]
            rev = [dict(limb=x["limb"], zone=x["zone"], t0=x["t0"], landed=bool(rv.landed(x)), clean=bool(rv.clean(x))) for x in rv.events() if x["limb"] in STRIKE_LIMBS]
            A = dict(post=np.array(S["post"], np.float64).reshape(-1, len(POST)))
            for k, w in (("hit", HIT_COLS), ("body", BODY_COLS), ("rhit", HIT_COLS), ("rbody", BODY_COLS)): A[k] = np.array(S[k], np.float64).reshape(-1, w)
            out.append((ctrl, role, ep, row, dict(events=ev, intents=ints, rev_events=rev), A))
            e.substep_hooks.clear(); S.clear(); del e, hm, rv, aim; gc.collect()   # env <-> logger cycle: free the bout now
    return out

def record(blocks=tuple(BLOCKS), seconds=20.0, workers=4, p1=None, ctrls=CTRLS, seeds=None):
    """seeds: {role: [seeds]} (default: every seed of each block; W0_ROLE: fb_seeds.W0_R4PRE)."""
    from fb_seeds import check, W0_R4PRE
    assert workers <= 4; t0 = time.time(); seeds = seeds or {r: list(W0_R4PRE) if r == W0_ROLE else list(range(BLOCKS[r][0], BLOCKS[r][1] + 1)) for r in blocks}
    for r, s in seeds.items():
        for c in ctrls: check(s, r, TRAIN, ctrl=c)
    jobs = [(c, r, [s]) for r in seeds for c in ctrls for s in seeds[r]]; sl = [jobs[i::workers] for i in range(workers) if jobs[i::workers]]
    if len(sl) == 1: parts = [work((sl[0], seconds, p1))]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(sl)) as pool: parts = pool.map(work, [(s, seconds, p1) for s in sl], chunksize=1)
    bouts = sorted((b for p in parts for b in p), key=lambda b: (b[1], ctrls.index(b[0]), b[2]["seed"]))
    R = dict(meta=dict(seeds=seeds, seconds=seconds, ctrls=list(ctrls), wall_s=round(time.time() - t0, 1),
                       bouts=[dict(ctrl=b[0], role=b[1], episode=b[2], row=b[3], **b[4]) for b in bouts]))
    for i, b in enumerate(bouts):
        for k, v in b[5].items(): R[f"{i}/{k}"] = v
    return R

def save(R, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **{k: (np.array(json.dumps(v, default=str)) if k == "meta" else v) for k, v in R.items()})

def load(path):
    with np.load(path, allow_pickle=False) as z: R = {k: z[k] for k in z.files}
    R["meta"] = json.loads(str(R["meta"])); return R

def merge(*Rs):
    """One recording from several: bouts concatenated in order, arrays renumbered."""
    out = dict(meta=dict(seeds={k: v for R in Rs for k, v in R["meta"]["seeds"].items()}, seconds=Rs[0]["meta"]["seconds"], wall_s=sum(R["meta"]["wall_s"] for R in Rs),
                         ctrls=sorted({c for R in Rs for c in R["meta"]["ctrls"]}, key=lambda c: CTRLS.index(c)), bouts=[]))
    assert all(R["meta"]["seconds"] == out["meta"]["seconds"] for R in Rs); n = 0
    for R in Rs:
        out["meta"]["bouts"] += R["meta"]["bouts"]
        for k, v in R.items():
            if k != "meta": i, f = k.split("/", 1); out[f"{int(i) + n}/{f}"] = v
        n += len(R["meta"]["bouts"])
    return out

def model():
    import fb_m4_check as C4
    m, cfg, r, _ = C4.model_cfg(); return m, cfg

# ---- replay
def replay(R, idx, m, vmode="com", reverse=False, w0=1.0, dtype=torch.float64, slots=TOL["slots"]):
    """FullBodyHitAccount + FbRewards over bouts idx of the recording, batched. -> dict(log, values {term: (N, T)}, live (N, T), edges, ...)."""
    sens = HT.MjContactSensors(m, reverse=reverse); P, Pb = len(sens.prim), len(sens.bprim); B = [R["meta"]["bouts"][i] for i in idx]; N = len(idx)
    hk, bk = ("rhit", "rbody") if reverse else ("hit", "body"); H = [R[f"{i}/{hk}"] for i in idx]; Bd = [R[f"{i}/{bk}"] for i in idx]; Po = [R[f"{i}/post"] for i in idx]
    T = max(len(p) for p in Po); log = []
    acc = HT.FullBodyHitAccount(N, sens.zone_idx, sens.bzone_idx, slots=0 if reverse else slots, dtype=dtype, log=log, block=sens.block_idx)
    rew = RW.FbRewards(N, w0=w0, dtype=dtype); od = EV.OppDown(N); vcol = slice(7, 10) if vmode == "com" else slice(10, 13)
    hs = [np.searchsorted(h[:, 0] // 4, np.arange(T + 1)) for h in H]; bs = [np.searchsorted(b[:, 0] // 4, np.arange(T + 1)) for b in Bd]
    its = [[] if reverse else b["intents"] for b in B]; vis = [0] * N; last = [len(p) - 1 for p in Po]
    sid, son, sap, ssd = (np.full((N, slots), -1, np.int64), np.zeros((N, slots)), np.zeros((N, slots)), np.zeros((N, slots), np.int64))
    vals, live, edges = {}, np.zeros((N, T), bool), []
    for s in range(T):
        F, fo, V = np.zeros((N, 8, P, 4, 3)), np.zeros((N, 8, P, 4)), np.zeros((N, 8, P, 4, 3)); Fb, vb = np.zeros((N, Pb, 4, 3)), np.zeros((N, Pb, 4, 3))
        post = np.stack([Po[i][min(s, last[i])] for i in range(N)]); t_end = np.array([Po[i][min(s, last[i])][0] + 0.02 * max(0, s - last[i]) for i in range(N)])
        for i in range(N):
            if s > last[i]: continue
            live[i, s] = True; h = H[i][hs[i][s]:hs[i][s + 1]]; b = Bd[i][bs[i][s]:bs[i][s + 1]]
            if len(h):
                j, l, p = (h[:, 0] % 4).astype(int), h[:, 1].astype(int), h[:, 2].astype(int); F[i, l, p, j] = h[:, 4:7]; fo[i, l, p, j] = h[:, 3]; V[i, l, p, j] = h[:, vcol]
            if len(b): j, q = (b[:, 0] % 4).astype(int), b[:, 1].astype(int); Fb[i, q, j] = b[:, 2:5]; vb[i, q, j] = b[:, 5:8]
            while vis[i] < len(its[i]) and its[i][vis[i]]["t_on"] - 0.8 <= t_end[i] + 1e-9:
                k = vis[i]; it = its[i][k]; sl = k % slots; sid[i, sl], son[i, sl], sap[i, sl], ssd[i, sl] = k, it["t_on"], it["t_apex"], HT.SIDES.index(it["limb"][0]); vis[i] += 1
        tt = lambda x: torch.from_numpy(x).to(dtype); win = None if reverse else dict(id=torch.from_numpy(sid.copy()), on=tt(son), apex=tt(sap), side=torch.from_numpy(ssd.copy()))
        acc.process(tt(F), tt(t_end), v_rel=tt(V), found=tt(fo), win=win, body=(tt(Fb), tt(vb)))
        edge = od.update(tt(post[:, 1]), tt(post[:, 2]))
        v = rew.step(acc, edge, od.now, tt(t_end), torch.zeros(N, dtype=torch.long), v_pel=tt(post[:, 5:8]), def_xy=tt(post[:, 3:5]), opp_xy=tt(post[:, 8:10]), torso_up=tt(post[:, 10:13]))
        for k, x in v.items(): vals.setdefault(k, np.zeros((N, T)))[:, s] = x.numpy()
        vals.setdefault("down_now", np.zeros((N, T)))[:, s] = od.now.numpy(); vals.setdefault("W", np.zeros((N, T)))[:, s] = rew.attr.W().numpy()
        vals.setdefault("tangle_Ns", np.zeros((N, T)))[:, s] = acc.step["tangle"].numpy()
        for i in torch.nonzero(edge & torch.from_numpy(live[:, s])).flatten().tolist():
            wz = [z in HT.WORK_ZONES for z in (HT.ZONES[x] for x in sens.bzone_idx)]
            lo = 4 * max(0, s - rew.attr.L + 1); bb = Bd[i]; win_b = bb[(bb[:, 0] >= lo) & (bb[:, 0] < 4 * (s + 1))]
            edges.append(dict(n=i, bout=idx[i], ctrl=B[i]["ctrl"], seed=B[i]["episode"]["seed"], step=s, t=float(t_end[i]), W=float(rew.attr.W()[i]), charged=bool(v["fb_knockdown"][i] > 0),
                              contact=bool(len(win_b) and any(wz[int(q)] for q in win_b[:, 1])), land_dt=float(t_end[i] - acc.land_t[i]), opp_fall=B[i]["row"].get("opp_fall")))
        done = [i for i in range(N) if last[i] == s]
        if done: acc.close_envs(done)
    return dict(log=log, values=vals, live=live, edges=edges, P=P, Pb=Pb, zone_idx=sens.zone_idx, bzone_idx=sens.bzone_idx)

# ---- G0-11
def g0_11(R, m, role="hit_scoring", vmodes=("com", "point")):
    out = {}
    for vm in vmodes:
        res = {}
        for ctrl in CTRLS:
            idx = [i for i, b in enumerate(R["meta"]["bouts"]) if b["role"] == role and b["ctrl"] == ctrl]
            if not idx: continue
            rp = replay(R, idx, m, vm); res[ctrl] = compare(R, idx, rp)
        out[vm] = dict(per_ctrl=res, ok=bool(res and all(x["ok"] for x in res.values())))
    rv = {}
    for ctrl in CTRLS:
        idx = [i for i, b in enumerate(R["meta"]["bouts"]) if b["role"] == role and b["ctrl"] == ctrl]
        if idx: rv[ctrl] = compare_rev(R, idx, replay(R, idx, m, "com", reverse=True))
    out["reverse_report"] = rv; out["ok"] = out[vmodes[0]]["ok"]; out["gated_vmode"] = vmodes[0]; return out

def compare(R, idx, rp):
    B = [R["meta"]["bouts"][i] for i in idx]; E = [x for x in rp["log"] if x["kind"] == "episode"]; Wl = [x for x in rp["log"] if x["kind"] == "window"]
    keys = [(l, z) for l in HT.LIMBS for z in ("head", "torso")]; bad, n_land, match, agree, ours_imp, hm_imp, ours_all, hm_all = [], 0, 0, 0, [], [], [], []
    for n, b in enumerate(B):
        mine = [e for e in E if e["env"] == n and e["landed"]]; ref = [e for e in b["events"] if e["landed"]]
        cm = {k: sum((e["limb"], e["zone"]) == k for e in mine) for k in keys}; cr = {k: sum((e["limb"], e["zone"]) == k for e in ref) for k in keys}
        if cm != cr: bad.append(dict(seed=b["episode"]["seed"], ours={f"{k[0]}/{k[1]}": v for k, v in cm.items() if v}, hm={f"{k[0]}/{k[1]}": v for k, v in cr.items() if v}))
        used = set()
        for e in ref:
            j = next((j for j, x in enumerate(mine) if j not in used and (x["limb"], x["zone"]) == (e["limb"], e["zone"]) and abs(x["t0"] - e["t0"]) < 1e-6), None)
            n_land += 1
            if j is not None: used.add(j); match += 1; agree += int(mine[j]["clean"] == e["clean"])
        n_land += len(mine) - len(used)
        ours_imp += [e["impulse_Ns"] for e in mine if e["zone"] == "head"]; hm_imp += [e["impulse_Ns"] for e in ref if e["zone"] == "head"]
        ours_all += [e["impulse_Ns"] for e in mine]; hm_all += [e["impulse_Ns"] for e in ref]
    mo, mh = (float(np.median(x)) if x else float("nan") for x in (ours_imp, hm_imp))
    # windows: outcome and pay vs the HitMeter / the python reference, per intent in order
    wo = {}; wn = wp = wn_ok = wp_ok = 0
    for n, b in enumerate(B):
        mine = {x["id"]: x for x in Wl if x["env"] == n}
        for k, it in enumerate(b["intents"]):
            x = mine.get(k); wn += 1; wp += 1
            if x is None: continue
            wn_ok += int(x["outcome"] == it["outcome"]); wp_ok += int(x["paid"] == it["pay"]); wo[it["outcome"]] = wo.get(it["outcome"], 0) + 1
    clean_agree = agree / n_land if n_land else float("nan"); rel = abs(mo / mh - 1) if mh == mh and mh > 0 else float("nan")
    res = dict(bouts=len(B), landed_hm=sum(len([e for e in b["events"] if e["landed"]]) for b in B), landed_ours=sum(e["landed"] for e in E), count_mismatch_bouts=len(bad), count_mismatches=bad[:10],
               landed_union=n_land, matched=match, clean_agree=round(clean_agree, 4), clean_hm=sum(e["clean"] for b in B for e in b["events"]), clean_ours=sum(e["clean"] for e in E),
               head_impulse_median=dict(ours=mo, hm=mh, rel=rel, n_ours=len(ours_imp), n_hm=len(hm_imp)),
               landed_impulse_median=dict(ours=float(np.median(ours_all)) if ours_all else None, hm=float(np.median(hm_all)) if hm_all else None),
               windows=dict(intents=wn, outcome_agree=wn_ok, pay_agree=wp_ok, hm_outcomes=wo, paid_ours=sum(x["paid"] for x in Wl), paid_ref=sum(it["pay"] for b in B for it in b["intents"]),
                            paid_ref_any_arm=sum(it.get("pay_any_arm", it["pay"]) for b in B for it in b["intents"])))
    res["ok"] = bool(not bad and clean_agree >= TOL["clean_agree"] and rel <= TOL["med_imp"]); return res

def compare_rev(R, idx, rp):
    B = [R["meta"]["bouts"][i] for i in idx]; E = [x for x in rp["log"] if x["kind"] == "episode"]; bad = 0; n = [0, 0]
    for k, b in enumerate(B):
        a = sorted((e["limb"], e["zone"]) for e in E if e["env"] == k and e["landed"]); r = sorted((e["limb"], e["zone"]) for e in b["rev_events"] if e["landed"])
        bad += int(a != r); n[0] += len(a); n[1] += len(r)
    return dict(bouts=len(B), landed_ours=n[0], landed_hm=n[1], count_mismatch_bouts=bad)

# ---- G0-12: W0 and the recorded bouts
def calibrate_w0(R, m, role=W0_ROLE):
    """W0 = P10 of W over role's G knockdown edges with contact; n_ok iff >= W0_MIN_N of them; edges_eo: every bout has one edge iff EO's
    opp_fell, at EO's opp_fall time."""
    idx = [i for i, b in enumerate(R["meta"]["bouts"]) if b["role"] == role and b["ctrl"] == "G"]; rp = replay(R, idx, m, "com")
    E = rp["edges"]; Wc = np.array([e["W"] for e in E if e["contact"]]); w0 = float(np.percentile(Wc, TOL["w0_pct"])) if len(Wc) else None
    W = np.array([e["W"] for e in E if e["W"] > 0]); B = R["meta"]["bouts"]; ne = {i: sum(e["bout"] == i for e in E) for i in idx}
    eo_t = {i: (B[i]["row"].get("opp_fall") or {}).get("t_s") for i in idx}
    edges_eo = all(ne[i] == int(eo_t[i] is not None) for i in idx) and all(abs(e["t"] - eo_t[e["bout"]]) < 1e-6 for e in E)
    return dict(role=role, bouts=len(idx), knockdowns=len(E), with_contact=int(len(Wc)), with_work=int(len(W)), n_min=W0_MIN_N, n_ok=bool(len(Wc) >= W0_MIN_N),
                bouts_with_kd=sum(v > 0 for v in ne.values()), edges_equal_eo=bool(edges_eo), W_all=[round(e["W"], 4) for e in E], w0=w0, w0_frozen=RW.W0,
                w0_variant_work_only=float(np.percentile(W, TOL["w0_pct"])) if len(W) else None,
                W_contact_quantiles={q: round(float(np.percentile(Wc, q)), 4) for q in (0, 5, 10, 25, 50, 75, 90, 100)} if len(Wc) else None,
                contact_W0_share=round(float((Wc == 0).mean()), 4) if len(Wc) else None,
                edges=[{k: v for k, v in e.items() if k != "opp_fall"} | dict(eo_contact_0p5s=(e["opp_fall"] or {}).get("r1_contact_0p5s", {}).get("contact")) for e in E])

def r4pre_match(R, rows, role=W0_ROLE):
    """The re-recorded EO rows vs R4-pre's G rows (p1_eval = EO bout): per field, how many bouts are equal."""
    B = [b for b in R["meta"]["bouts"] if b["role"] == role]; J = lambda x: json.dumps(x, sort_keys=True); f = dict(fell_s=0, opp_fell_t=0, hit_counts=0, max_disp_m=0, r1_contact=0, act_delay_key=0)
    for b in B:
        e, r = b["row"], rows[b["episode"]["seed"]]; of = e.get("opp_fall") or {}
        f["fell_s"] += e["fell_s"] == r["fell_s"]; f["opp_fell_t"] += of.get("t_s") == (r["opp_fell"] or {}).get("t"); f["hit_counts"] += J(e["hit_meter"]["n"]) == J(r["n"])
        f["max_disp_m"] += e["max_disp_m"] == r["max_disp_m"]; f["r1_contact"] += J(of.get("r1_contact_0p5s")) == J((r["opp_fell"] or {}).get("r1_contact"))
        f["act_delay_key"] += all(b["episode"][k] == r[k] for k in ("clip", "distance", "t0", "delay"))
    return dict(bouts=len(B), equal=f, all_equal=bool(B) and all(v == len(B) for v in f.values()))

def early_kd(R, cal, rows, early=EARLY_S, role=W0_ROLE):
    """A4.2 (report): the knockdowns before `early` s among the re-recorded R4-pre G bouts: W, contact, the first R1-opponent contact
    (any logged sensor: opponent strike limbs on the R1, R1 shells on the opponent, both ways) and whether it was at the first substep."""
    B = R["meta"]["bouts"]; out = []
    for e in cal["edges"]:
        if e["t"] >= early: continue
        i = e["bout"]; ks = [int(R[f"{i}/{k}"][0, 0]) for k in ("hit", "body", "rhit", "rbody") if len(R[f"{i}/{k}"])]; k0 = min(ks) if ks else None; post = R[f"{i}/post"]
        t_first = None if k0 is None else round(float(post[k0 // 4, 0]) - (3 - k0 % 4) * HT.DT, 4); r = rows[e["seed"]]
        out.append(dict(seed=e["seed"], clip=B[i]["episode"]["clip"], distance=B[i]["episode"]["distance"], t0=B[i]["episode"]["t0"], act_delay=r["act_delay"], t=e["t"], W=round(e["W"], 4),
                        contact=e["contact"], first_contact_s=t_first, contact_at_first_substep=k0 == 0, eo_contact_0p5s=e["eo_contact_0p5s"], cause=(B[i]["row"].get("opp_fall") or {}).get("cause"),
                        opp_up_z_step1=round(float(post[0, 2]), 4), fell_s=B[i]["row"]["fell_s"]))
    w0 = cal["w0"]; ch = lambda e: w0 is not None and e["W"] >= w0 and e["W"] > 0 and not e["land_dt"] <= RW.LAND_GATE_S + 1e-9   # the charge rule at the calibrated W0
    return dict(early_s=early, knockdowns=len(cal["edges"]), early=len(out), early_with_contact=sum(x["contact"] for x in out), charged=sum(map(ch, cal["edges"])),
                early_charged=sum(ch(e) for e in cal["edges"] if e["t"] < early), early_contact_at_first_substep=sum(x["contact_at_first_substep"] for x in out), bouts=out)

def recorded_checks(R, m, w0):
    res = {}; dt = 0.02; he = RW.WEIGHTS["A"]["fb_knockdown"] * dt
    for role in (*BLOCKS, W0_ROLE):
        for ctrl in CTRLS:
            idx = [i for i, b in enumerate(R["meta"]["bouts"]) if b["role"] == role and b["ctrl"] == ctrl]
            if not idx: continue
            rp = replay(R, idx, m, "com", w0=w0); V, L = rp["values"], rp["live"]; dn = V["down_now"] > 0
            kd = V["fb_knockdown"] * L; E = rp["edges"]; ev_steps = {(e["n"], e["step"]) for e in E}
            ch = [(n, s) for n, s in zip(*np.nonzero(kd))]
            kd_ok = all((n, s) in ev_steps for n, s in ch) and all(e["charged"] == (e["W"] >= w0 and e["W"] > 0 and not e["land_dt"] <= RW.LAND_GATE_S + 1e-9) for e in E) \
                and np.all(kd[kd != 0] == 1.0) and np.allclose(RW.WEIGHTS["A"]["fb_knockdown"] * kd[kd != 0] * dt, -2.4)
            dnv = V["fb_down_neutral"] * L; up = L & ~dn
            neutral_ok = bool(np.all(dnv[up] == 0) and np.all(dnv[L & dn] > 0))
            got_up = sum(bool(np.any(dn[n][L[n]]) and np.any(~dn[n][L[n]][np.argmax(dn[n][L[n]]):])) for n in range(len(idx)))
            ac_ok = bool(np.all(V["fb_down_approach"][up] == 0) and np.all(V["fb_down_contact"][up] == 0))
            # tangle: independent numpy per-substep sum over the pelvis + leg body entries
            tz = np.array([HT.ZONES[z] in HT.TANGLE_ZONES for z in rp["bzone_idx"]]); terr = 0.0
            for n, i in enumerate(idx):
                b = R[f"{i}/body"]; ref = np.zeros(L.shape[1])
                if len(b): np.add.at(ref, (b[:, 0] // 4).astype(int), np.linalg.norm(b[:, 2:5], axis=1) * HT.DT * tz[b[:, 1].astype(int)])
                terr = max(terr, float(np.abs(ref - V["tangle_Ns"][n] * L[n]).max()))
            res[f"{role}/{ctrl}"] = dict(bouts=len(idx), knockdown_edges=len(E), charged=int((kd != 0).sum()), charged_reward=round(float(he), 4), kd_ok=bool(kd_ok),
                                        down_steps=int((L & dn).sum()), got_up_bouts=int(got_up), neutral_ok=neutral_ok, approach_contact_zero_when_up=ac_ok,
                                        approach_when_down_s=round(float((V["fb_down_approach"] * L).sum() * dt), 3), contact_when_down_Ns=round(float((V["fb_down_contact"] * L).sum() * dt), 3),
                                        tangle_Ns_per_min=round(float((V["tangle_Ns"] * L).sum() / (L.sum() * dt / 60)), 2), tangle_maxdiff=terr, tangle_ok=terr <= 1e-9,
                                        lean_away_mean_deg=round(float((V["fb_lean_away"] * L).sum() / L.sum()), 3))
    res["ok"] = all(v["kd_ok"] and v["neutral_ok"] and v["approach_contact_zero_when_up"] and v["tangle_ok"] for v in res.values() if isinstance(v, dict)); return res

# ---- G0-12: scripted events (N = 1, float64)

class Script:
    """A tiny synthetic world for the account: 32 target primaries with the eval's zones, the body sensor on torso / pelvis / leg."""
    def __init__(self, n=1, w0=1.0, slots=2, m=None):
        if m is None: m, _ = model()
        self.s = HT.MjContactSensors(m); self.n = n; self.P, self.Pb = len(self.s.prim), len(self.s.bprim)
        self.acc = HT.FullBodyHitAccount(n, self.s.zone_idx, self.s.bzone_idx, slots=slots, dtype=torch.float64, log=[], block=self.s.block_idx); self.rew = RW.FbRewards(n, w0=w0, dtype=torch.float64); self.od = EV.OppDown(n)
        self.t = 0.0; self.z = lambda: (torch.zeros(n, 8, self.P, 4, 3, dtype=torch.float64), torch.zeros(n, self.Pb, 4, 3, dtype=torch.float64))
    def prim(self, zone, k=0): return [i for i, z in enumerate(self.s.zone_idx) if HT.ZONES[z] == zone][k]
    def bprim(self, zone, k=0): return [i for i, z in enumerate(self.s.bzone_idx) if HT.ZONES[z] == zone][k]
    def step(self, F=None, Fb=None, vb=None, v=None, down=False, win=None, t_push=None, v_pel=None):
        self.t = round(self.t + 0.02, 10); n = self.n; F0, Fb0 = self.z(); F = F0 if F is None else F; Fb = Fb0 if Fb is None else Fb; vb = torch.zeros_like(Fb) if vb is None else vb
        T = lambda x: torch.full((n,), x, dtype=torch.float64); self.acc.process(F, T(self.t), v_rel=v, win=win, body=(Fb, vb))
        dn = torch.full((n,), bool(down)); edge = self.od.update(T(0.1 if down else 0.7), T(0.1 if down else 1.0)); assert bool((self.od.now == dn).all())
        xy = torch.zeros(n, 2, dtype=torch.float64); oxy = xy + torch.tensor([0.5, 0.0], dtype=torch.float64)
        vals = self.rew.step(self.acc, edge, self.od.now, T(self.t), torch.zeros(n, dtype=torch.long), t_push=None if t_push is None else T(t_push),
                             v_pel=torch.zeros(n, 3, dtype=torch.float64) if v_pel is None else v_pel, def_xy=xy, opp_xy=oxy)
        tot, r = self.rew.total(vals, "A"); return vals, r

def _strike(sc, side, limb, zone, f=200.0, v=3.0, p=None):
    """One control step of a strike limb on a zone: force f N on one primary every substep, closing at v m/s (FORCE_SIGN: the force on
    the limb points back along -x, the limb moves +x)."""
    F, _ = sc.z(); l = HT.SIDES.index(side) * 4 + HT.LIMBS.index(limb); p = sc.prim(zone) if p is None else p; F[:, l, p, :, 0] = -f
    V = torch.zeros_like(F); V[:, l, p, :, 0] = v; return F, V

def scripted(m=None):
    out = {}; rw = lambda r, k: float(r.get(k, torch.zeros(1))[0]) if k in r else 0.0
    def push_bout(sc, steps):   # the defender's leg pushes the opponent with positive work: 100 N at 0.2 m/s
        _, Fb = sc.z(); vb = torch.zeros_like(Fb); q = sc.bprim("leg"); Fb[:, q, :, 0] = 100.0; vb[:, q, :, 0] = 0.2
        for _ in range(steps): sc.step(Fb=Fb, vb=vb)
    # 1. an attributed knockdown: exactly one -2.4 event
    sc = Script(w0=1.0, m=m); push_bout(sc, 30); seq = [sc.step(down=True)[1] for _ in range(10)] + [sc.step(down=False)[1] for _ in range(5)] + [sc.step(down=True)[1] for _ in range(5)]
    kd = [rw(r, "fb_knockdown") for r in seq]; out["attributed_knockdown"] = dict(rewards=[x for x in kd if x], W=float(sc.rew.attr.W()[0]), ok=kd.count(-2.4) == 1 and sum(1 for x in kd if x) == 1 and abs(kd[0] + 2.4) < 1e-12)
    # 2. a tracker-alone fall (no contact): 0
    sc = Script(w0=1.0, m=m); [sc.step() for _ in range(30)]; r = sc.step(down=True)[1]; out["tracker_alone_fall"] = dict(reward=rw(r, "fb_knockdown"), ok=rw(r, "fb_knockdown") == 0.0)
    # 3. a knockdown right after an opponent strike landed (< 0.3 s): 0; the same strike > 0.3 s earlier: charged
    for name, gap in (("kd_after_landed_strike", 6), ("kd_strike_0p36s_before", 18)):
        sc = Script(w0=1.0, m=m); push_bout(sc, 20); F, V = _strike(sc, "r", "hand", "head"); _, Fb = sc.z(); vb = torch.zeros_like(Fb); q = sc.bprim("leg"); Fb[:, q, :, 0] = 100.0; vb[:, q, :, 0] = 0.2
        for _ in range(3): sc.step(F=F, v=V, Fb=Fb, vb=vb)
        for _ in range(gap - 1): sc.step(Fb=Fb, vb=vb)
        r = sc.step(down=True, Fb=Fb, vb=vb)[1]; dtl = sc.t - float(sc.acc.land_t[0])
        out[name] = dict(reward=rw(r, "fb_knockdown"), since_landed_s=round(dtl, 4), ok=(rw(r, "fb_knockdown") == 0.0) if gap == 6 else (abs(rw(r, "fb_knockdown") + 2.4) < 1e-12))
    # 4. a blocked strike followed by a stumble with only arm-shell contact: 0 (arm shells are not in the work set)
    sc = Script(w0=1.0, m=m); F, V = _strike(sc, "l", "hand", "arm"); sc.step(F=F, v=V); sc.step(F=F, v=V)
    acc2 = HT.FullBodyHitAccount(1, sc.s.zone_idx, list(sc.s.bzone_idx) + [HT.ARM], dtype=torch.float64); sc.acc = acc2   # a body sensor that also sees an arm shell
    Fb = torch.zeros(1, sc.Pb + 1, 4, 3, dtype=torch.float64); vb = torch.zeros_like(Fb); Fb[:, -1, :, 0] = 300.0; vb[:, -1, :, 0] = 0.5
    for _ in range(20): sc.step(Fb=Fb, vb=vb)
    r = sc.step(down=True, Fb=Fb, vb=vb)[1]; out["blocked_then_arm_only_stumble"] = dict(reward=rw(r, "fb_knockdown"), W=float(sc.rew.attr.W()[0]), ok=rw(r, "fb_knockdown") == 0.0 and float(sc.rew.attr.W()[0]) == 0.0)
    # 5. down-neutral accrues only while opp_down_now and stops when the opponent is up again (non-sticky)
    sc = Script(w0=1.0, m=m); pat = [0] * 10 + [1] * 20 + [0] * 10 + [1] * 5 + [0] * 5; v = [float(sc.step(down=bool(d))[0]["fb_down_neutral"][0]) for d in pat]
    out["down_neutral_nonsticky"] = dict(steps_charged=sum(x > 0 for x in v), steps_down=sum(pat), per_s_charge=round(-RW.HE * max(v), 5), ok=all((x > 0) == bool(d) for x, d in zip(v, pat)))
    # 6. fb_down_approach / fb_down_contact zero while both are up; when down, -2 per (m/s s) and -0.1 per N s
    sc = Script(w0=1e9, m=m); _, Fb = sc.z(); q = sc.bprim("leg"); Fb[:, q, :, 0] = 50.0; vp = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
    a_up, r_up = sc.step(Fb=Fb, v_pel=vp); a_dn, r_dn = sc.step(Fb=Fb, v_pel=vp, down=True)
    out["down_terms_zero_when_up"] = dict(up=[float(a_up["fb_down_approach"][0]), float(a_up["fb_down_contact"][0])], down_reward=[rw(r_dn, "fb_down_approach"), rw(r_dn, "fb_down_contact")],
                                          ok=float(a_up["fb_down_approach"][0]) == 0.0 == float(a_up["fb_down_contact"][0]) and abs(rw(r_dn, "fb_down_approach") + 2.0 * 0.3 * 0.02) < 1e-12
                                          and abs(rw(r_dn, "fb_down_contact") + 0.1 * 50.0 * 4 * HT.DT) < 1e-12)
    # 7. HIT_R2 + m2: the glove touches a forearm shell before the apex; a same-side forearm landing on the head after it cancels the pay;
    # an upper-arm (shoulder) first touch is blocked but pays nothing (run 2 bz)
    def r2(case):
        sc = Script(w0=1.0, m=m, slots=2); win = dict(id=torch.tensor([[0, -1]]), on=torch.tensor([[0.10, 0.0]], dtype=torch.float64), apex=torch.tensor([[0.40, 0.0]], dtype=torch.float64), side=torch.tensor([[0, 0]]))
        paid = 0.0
        for k in range(40):
            t = round(0.02 * (k + 1), 10); F, V = sc.z()[0], None
            if 0.2 < t <= 0.24: F, V = _strike(sc, "l", "hand", "arm", f=50.0, p=sc.prim("arm") if case == "upper_arm" else sc.s.block_idx.index(True))   # a shoulder / a forearm shell
            if case in ("l_forearm", "r_forearm", "hand") and 0.3 < t <= 0.36:
                side, limb = ("l", "forearm") if case == "l_forearm" else ("r", "forearm") if case == "r_forearm" else ("l", "hand"); F2, V2 = _strike(sc, side, limb, "head"); F = F + F2; V = V2 if V is None else V + V2
            paid += float(sc.step(F=F, v=V, win=win)[0]["spar_block"][0])
        sc.acc.close_envs([0]); return paid + 0.0, [x for x in sc.acc.log if x["kind"] == "window"]
    p = {c: r2(c) for c in ("none", "l_forearm", "r_forearm", "hand", "upper_arm")}
    out["hit_r2_forearm"] = dict(paid={c: v[0] for c, v in p.items()}, outcomes={c: v[1][0]["outcome"] if v[1] else None for c, v in p.items()},
                                 ok=p["none"][0] == 1.0 and p["l_forearm"][0] == 0.0 and p["r_forearm"][0] == 1.0 and p["hand"][0] == 0.0
                                 and p["upper_arm"][0] == 0.0 and p["upper_arm"][1][0]["outcome"] == "blocked")
    # 8. tangle: per-substep |F| dt summed (direction flips do not cancel), pelvis + leg only, the reward <= 0
    sc = Script(w0=1e9, m=m); _, Fb = sc.z(); ql, qp, qt = sc.bprim("leg"), sc.bprim("pelvis"), sc.bprim("torso"); Fb[:, ql, :, 0] = torch.tensor([40.0, -40.0, 40.0, -40.0], dtype=torch.float64)
    Fb[:, qp, :, 1] = 10.0; Fb[:, qt, :, 2] = 99.0; v, r = sc.step(Fb=Fb); want = (4 * 40.0 + 4 * 10.0) * HT.DT
    out["tangle_per_substep"] = dict(tangle_Ns=float(sc.acc.step["tangle"][0]), want=want, reward=rw(r, "fb_leg_tangle"), ok=abs(float(sc.acc.step["tangle"][0]) - want) < 1e-12 and abs(rw(r, "fb_leg_tangle") + 0.05 * want) < 1e-12)
    # 9. every hit term masked 0.5 s after a push (a strike, the tangle, a knockdown inside the mask); fb_down_neutral not
    vp = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64); res = []; tp = None
    def leg(sc):
        _, Fb = sc.z(); vb = torch.zeros_like(Fb); q = sc.bprim("leg"); Fb[:, q, :, 0] = 100.0; vb[:, q, :, 0] = 0.2; return Fb, vb
    sc = Script(w0=1.0, m=m); push_bout(sc, 10); tp = sc.t
    for k in range(50):
        F, V = _strike(sc, "r", "hand", "head") if k in (0, 1, 2, 10, 11, 12, 35, 36, 37) else (None, None); Fb, vb = leg(sc)
        v, r = sc.step(F=F, v=V, Fb=Fb, vb=vb, down=k >= 30, t_push=tp, v_pel=vp); res.append((sc.t, v))
    msk = [(t, v) for t, v in res if t - tp <= RW.PUSH_MASK_S]; aft = [(t, v) for t, v in res if t - tp > RW.PUSH_MASK_S]
    zero = all(float(v[k][0]) == 0.0 for _, v in msk for k in RW.HIT_TERMS if k in v); some = {k: any(float(v[k][0]) != 0 for _, v in aft if k in v) for k in RW.HIT_TERMS}
    sc = Script(w0=1.0, m=m); push_bout(sc, 30); Fb, vb = leg(sc); v_in, _ = sc.step(Fb=Fb, vb=vb, down=True, t_push=sc.t - 0.1, v_pel=vp)
    kd_in = dict(knockdown=float(v_in["fb_knockdown"][0]), W=float(sc.rew.attr.W()[0]), down_neutral=float(v_in["fb_down_neutral"][0]))
    out["push_mask"] = dict(masked_steps=len(msk), after_nonzero=some, knockdown_inside_mask=kd_in,
                            ok=zero and all(some[k] for k in ("spar_head_impulse", "spar_head_clean", "fb_leg_tangle", "fb_down_approach", "fb_down_contact", "fb_knockdown"))
                            and kd_in["knockdown"] == 0.0 and kd_in["W"] >= 1.0 and kd_in["down_neutral"] > 0)
    # 10. the frozen W0 (A4.1): a tracker-alone fall and a knockdown with contact but no positive work give 0; any positive work, -2.4
    def frozen(F, v):
        sc = Script(w0=RW.W0, m=m); _, Fb = sc.z(); vb = torch.zeros_like(Fb); q = sc.bprim("leg"); Fb[:, q, :, 0] = F; vb[:, q, :, 0] = v
        for _ in range(30): sc.step(Fb=Fb, vb=vb)
        return rw(sc.step(down=True, Fb=Fb, vb=vb)[1], "fb_knockdown"), float(sc.rew.attr.W()[0])
    fz = dict(alone=frozen(0.0, 0.0), no_work=frozen(100.0, 0.0), backing=frozen(100.0, -0.2), tiny_work=frozen(1.0, 1e-3))
    want = -2.4 if fz["tiny_work"][1] >= RW.W0 else 0.0   # W0 = 0: any positive work is charged
    out["frozen_w0"] = dict(w0=RW.W0, cases=fz, ok=fz["alone"][0] == fz["no_work"][0] == fz["backing"][0] == 0.0 and fz["no_work"][1] == fz["backing"][1] == 0.0
                            and fz["tiny_work"][1] > 0 and abs(fz["tiny_work"][0] - want) < 1e-12)
    # weights and stages
    out["stages"] = dict(ok=("spar_clearance" in RW.WEIGHTS["A"] and "spar_clearance" not in RW.WEIGHTS["B"] and RW.res_l2_weight(0) == -0.5 and abs(RW.res_l2_weight(2000) + 0.1) < 1e-12
                             and RW.WEIGHTS["A"]["fb_knockdown"] * 0.02 == -2.4 and RW.WEIGHTS["A"]["spar_head_clean"] * 0.02 == -RW.HE))
    out["ok"] = all(v["ok"] for v in out.values()); return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--rec", required=True); ap.add_argument("--rec-w0", required=True); ap.add_argument("--p1", required=True)
    ap.add_argument("--r4pre", default=str(R4PRE)); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--reuse", action="store_true"); a = ap.parse_args()
    import fb_seeds as SD
    t0 = time.time(); lst, rows, src = r4pre_list(a.r4pre)   # A4.1: the list, re-derived from R4-pre's outputs, equals the frozen one before anything runs
    wl = dict(n=len(lst), derived_equals_frozen=lst == sorted(SD.W0_R4PRE), sha256=SD.list_sha(lst), frozen_sha256=SD.W0_R4PRE_SHA256, sources_sha256=src)
    wl["ok"] = bool(wl["derived_equals_frozen"] and wl["sha256"] == wl["frozen_sha256"]); assert wl["ok"], wl
    if a.reuse and Path(a.rec).exists(): R = load(a.rec)
    else: R = record(seconds=a.seconds, workers=a.workers, p1=a.p1); save(R, a.rec)
    if a.reuse and Path(a.rec_w0).exists(): Rw = load(a.rec_w0)
    else: Rw = record(blocks=(W0_ROLE,), seconds=a.seconds, workers=a.workers, p1=a.p1, ctrls=("G",)); save(Rw, a.rec_w0)
    R = merge(R, Rw); m, _ = model(); torch.set_num_threads(min(4, a.workers))
    cal = calibrate_w0(R, m, W0_ROLE); w0 = cal["w0"] if cal["n_ok"] and cal["edges_equal_eo"] else None   # fewer than W0_MIN_N: stop, no W0
    res = dict(meta=dict(rec=a.rec, rec_w0=a.rec_w0, wall_record_s=R["meta"]["wall_s"], bouts=len(R["meta"]["bouts"]), seeds={k: [min(v), max(v), len(v)] for k, v in R["meta"]["seeds"].items()}),
               w0_list=wl, w0=cal, w0_60300_report=calibrate_w0(R, m, "w0_calib"), r4pre_match=r4pre_match(R, rows))
    res["early_kd_report"] = early_kd(R, cal, rows)
    res["g0_11"] = g0_11(R, m); res["g0_12_scripted"] = scripted(m)
    res["g0_12_recorded"] = recorded_checks(R, m, w0) if w0 is not None else dict(ok=False, why=f"W0 not calibrated: {cal['with_contact']} G knockdowns with contact (need >= {W0_MIN_N}), edges = EO: {cal['edges_equal_eo']}")
    res["w0"]["frozen_ok"] = RW.W0 is not None and w0 is not None and abs(RW.W0 - w0) <= 1e-4 * max(1.0, w0)
    res["g0_12_ok"] = bool(res["g0_12_scripted"]["ok"] and res["g0_12_recorded"]["ok"] and res["w0"]["frozen_ok"])
    res["ok"] = dict(G011_on_EO_states=res["g0_11"]["ok"], G012=res["g0_12_ok"]); res["wall_s"] = round(time.time() - t0, 1)
    s = json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print(json.dumps(res["ok"]))
    print(json.dumps(dict(w0=cal["w0"], knockdowns=cal["knockdowns"], with_contact=cal["with_contact"], with_work=cal["with_work"], n_ok=cal["n_ok"], edges_equal_eo=cal["edges_equal_eo"])))
    if a.out: Path(a.out).write_text(s, encoding="utf-8")
    return 0 if all(res["ok"].values()) else 1

if __name__ == "__main__":
    sys.exit(main())
