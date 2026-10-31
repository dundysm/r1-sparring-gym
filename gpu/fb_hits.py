"""Full-body M2 v2 hit accounting (M2V2_PLAN_v3 2.4, build order M5): r1gym/hits.py's HitMeter rules on contact-sensor data for a
whole striker body against a whole target body, symmetric (opponent -> defender drives rewards, defender -> opponent is logged), plus
the inputs of the knockdown attribution (m1). Pure torch (float32 on the GPU, float64 on CPU); MjContactSensors is the CPU sensor.

Sensors (mjlab ContactSensorCfg, fields found/force, reduce 'netforce', num_slots 1, history = the 4 substeps; oldest first here)
  hit    8 (SENSORS: side x strike limb), primaries = the target's 32 shells, secondary = the limb's bodies by the HitMeter's own map
         (opp_limb): hand = wrist_roll_link, forearm = elbow_link, shin = knee_link (body); foot = ankle_pitch_link (subtree: both ankle
         bodies are 'foot'; D2 7.1's ankle_roll subtree would miss the ankle_pitch shell). LIMB_SECONDARY.
  body   primaries = the target's shells of BODY_ZONES (torso, pelvis, leg; arm shells may be added, they are masked), secondary = the
         striker's root subtree: tangle impulse (pelvis + leg) and the target's work on the striker (torso + pelvis + leg).
  MuJoCo's netforce is the force on the SECONDARY in the world frame (MjContactSensors = mjSENS_CONTACT exactly, MuJoCo 3.17): a hit
  sensor reads the target's push back on the limb (FORCE_SIGN, gpu/sparring_task.py's measured -1), the body sensor reads the
  target's push on the striker (the work W uses it as is).
Per (limb, zone) and substep: force = sum over the zone's primaries of |F| (as gpu/sparring_task.py HitAccount), touch = any primary
found, closing speed = FORCE_SIGN (v_striker - v_target) . F_hat at the zone's strongest primary (HitMeter: the relative velocity along
the normal at the strongest contact; v_rel is the caller's: body COM velocities on the GPU, contact-point ones in MjContactSensors).
Episodes (HitMeter.events): substeps at most merge_gap_s apart merged; impulse = sum F dt, peak, closing speed of the first substep;
landed / clean as HIT:51-60 on HitMeter's rounded values (peak 0.1 N, impulse 1e-4 N s, v_close 1e-3 m/s). Two key sets: merged over
sides per (limb, zone) (HitMeter's counts, rewards) and per (side, limb, zone) (its windows).
Windows (aim intent slots, AimOp's ring): [t_on, t_apex + 0.15 s] of a side's strike limb; outcome as HitMeter.windows (blocked iff an
arm-zone episode of that limb opens in the window before the first landed head / torso one); spar_block pays (HIT_R2, m2) iff blocked,
the arm episode opened on a block shell (run 2's 'block' zone, BLOCK_LINKS: elbow / wrist; an upper-arm-first touch pays nothing,
gpu/sparring_task.py _finish bz), the arm episode opened by apex + 0.05 s, and no head / torso episode of the same side's limb or its partner (hand <-> forearm,
foot <-> shin) that opened in the window landed. A window is decided once past and no unlanded head / torso episode that opened in it
is still open; a new intent in its slot, close_envs and finish decide it on what is known."""
import numpy as np, torch

ZONES = ("head", "torso", "pelvis", "arm", "leg")       # r1gym/hits.py ZONES (HIT:29-36); asserted in tests/test_fb_hits.py
LIMBS = ("hand", "forearm", "foot", "shin")             # r1gym/hits.py STRIKE_LIMBS
SIDES = ("l", "r")
SENSORS = tuple(f"{s}_{l}" for s in SIDES for l in LIMBS)          # hit sensor order [side][limb] -> (2, 4)
LIMB_SECONDARY = dict(hand=("body", "wrist_roll_link"), forearm=("body", "elbow_link"), foot=("subtree", "ankle_pitch_link"), shin=("body", "knee_link"))
PARTNER = (1, 0, 3, 2)                                  # hand <-> forearm, foot <-> shin (m2: same-side strike limbs)
HIT_P = dict(merge_gap_s=0.1, min_impulse_Ns=0.5, min_peak_N=5.0, v_close_min=1.0, max_strike_s=0.15)   # HitMeter DEFAULTS = cfg hit_metric
DT = 0.005
FORCE_SIGN = -1.0
T_TOL, W_TOL = 1e-4, 5e-5                               # gpu/sparring_task.py T_TOL / W_TOL
WIN_TAIL_S, BLOCK_BY_APEX_S = 0.15, 0.05                # r1gym/aim.py window_tail_s; HIT_R2 block_by_apex_s
HEAD, TORSO, ARM = 0, 1, 3
BODY_ZONES, TANGLE_ZONES, WORK_ZONES = ("torso", "pelvis", "leg"), ("pelvis", "leg"), ("torso", "pelvis", "leg")
OUTCOMES = ("head", "torso", "blocked", "miss")
BLOCK_LINKS = ("elbow", "wrist")                        # gpu/sparring_task.py SHELLS 'block' (elbow_link, wrist_roll_link); 'upper_arm' = shoulders
INF = float("inf")

def zone_of(link, mesh=""):
    """r1gym/hits.py r1_zone: link and mesh names without the entity prefix."""
    if mesh in ("head_pitch_link", "head_yaw_link") or mesh == "head_collision": return "head"
    if link == "torso_link": return "torso"
    if link in ("pelvis", "waist_roll_link"): return "pelvis"
    if any(k in link for k in ("shoulder", "elbow", "wrist")): return "arm"
    if any(k in link for k in ("hip", "knee", "ankle")): return "leg"
    return None

def limb_of(link):
    """r1gym/hits.py opp_limb: (side, limb group)."""
    side = "l" if link.startswith("left") else "r" if link.startswith("right") else ""
    for k, g in (("wrist", "hand"), ("elbow", "forearm"), ("shoulder", "upper_arm"), ("ankle", "foot"), ("knee", "shin"), ("hip", "thigh")):
        if k in link: return side, g
    return side, "trunk"

def _r(x, k): return torch.round(x * k) / k             # HitMeter rounds before it compares (round(x, 1), round(x, 4), round(x, 3))

class Episodes:
    """HitMeter.events for keys (N, *K): open, t0, t1, impulse, peak, closing speed at the episode's first substep."""
    def __init__(self, n, shape, p, dtype, device):
        z = lambda dt=dtype: torch.zeros(n, *shape, dtype=dt, device=device)
        self.p, self.k = p, len(shape); self.open = z(torch.bool); self.t0, self.t1, self.imp, self.peak, self.vc = z(), z(), z(), z(), z()

    def struck(self): return (_r(self.peak, 10) >= self.p["min_peak_N"]) & (_r(self.imp, 1e4) >= self.p["min_impulse_Ns"])
    def clean(self): return self.struck() & (_r(self.vc, 1e3) >= self.p["v_close_min"]) & (self.t1 - self.t0 + DT <= self.p["max_strike_s"] + T_TOL)

    def close(self, tj):
        """Substep at time tj (N,): the episodes that close before its contacts (fields still hold until update)."""
        self.tj = tj.view(-1, *[1] * self.k); self.C = self.open & (self.tj - self.t1 > self.p["merge_gap_s"] + T_TOL); return self.C

    def update(self, F, touch, vc):
        """The substep's contacts: -> episodes opened now."""
        tj = self.tj; self.open = self.open & ~self.C; nw = touch & ~self.open; self.open = self.open | nw
        self.t0 = torch.where(nw, tj, self.t0); self.imp = torch.where(nw, torch.zeros_like(F), self.imp); self.peak = torch.where(nw, torch.zeros_like(F), self.peak)
        self.vc = torch.where(nw, vc, self.vc); self.t1 = torch.where(touch, tj, self.t1)
        self.imp = self.imp + torch.where(touch, F, torch.zeros_like(F)) * DT; self.peak = torch.maximum(self.peak, torch.where(touch, F, torch.zeros_like(F)))
        return nw

    def reset(self, ids):
        self.open[ids] = False
        for x in (self.t0, self.t1, self.imp, self.peak, self.vc): x[ids] = 0

def zone_forces(F, zone_idx, found=None):
    """Hit sensors F (N, L, P, S, 3) (force on the limb), found (N, L, P, S) or None, zone_idx (P,) -> zone force (N, L, Z, S) = sum of
    |F| over the zone's primaries, touch (N, L, Z, S), the strongest primary per zone (N, L, Z, S) (-1: none)."""
    Z = len(ZONES); zi = torch.as_tensor(zone_idx, device=F.device); M = torch.nn.functional.one_hot(zi, Z).to(F.dtype)   # (P, Z)
    a = torch.linalg.vector_norm(F, dim=-1); Fz = torch.einsum("nlps,pz->nlzs", a, M)
    t = (a > 0) if found is None else (found > 0); Tz = torch.einsum("nlps,pz->nlzs", t.to(F.dtype), M) > 0
    s = torch.where(M.T[None, None, :, :, None] > 0, a[:, :, None], torch.full_like(a[:, :, None], -1.0))   # (N, L, Z, P, S)
    return Fz, Tz, torch.where(Tz, s.argmax(3), torch.full_like(Fz, -1, dtype=torch.long))

def closing_speeds(F, v_rel, strongest, signed=True):
    """FORCE_SIGN (v_striker - v_target) . F_hat at each zone's strongest primary: F (N, L, P, S, 3), v_rel (N, L, P or 1, S, 3),
    strongest (N, L, Z, S) -> (N, L, Z, S); 0 where nothing touches. signed=False: |.| (the sign of a simulator's netforce unchecked)."""
    ix = strongest.clamp(min=0); g = lambda X: torch.gather(X, 2, ix[..., None].expand(*ix.shape, 3)) if X.shape[2] > 1 else X.expand(-1, -1, ix.shape[2], -1, -1)
    Fs, vs = g(F), g(v_rel); c = (vs * torch.nn.functional.normalize(Fs, dim=-1)).sum(-1); c = FORCE_SIGN * c if signed else c.abs()
    return torch.where(strongest >= 0, c, torch.zeros_like(c))

class FullBodyHitAccount:
    """N envs. zone_idx (P,): zone of each hit-sensor primary (the target's shells, sensor order); body_zone_idx (Pb,): of the body
    sensor's primaries (None: no body sensor). slots K: intent window slots (0: no windows). log: a list collecting closed merged
    episodes and decided windows (host sync, CPU checks only). Per step (process) `step`: head_imp / torso_imp (N s of the strike
    limbs, landed or not), head_clean / torso_clean / head_landed / torso_landed (episodes closed this step), block (paid windows
    decided), tangle (N s on the body sensor's pelvis + leg shells), work (J: the target's positive work on the striker, torso +
    pelvis + leg shells: sum of max(0, F . v_link) dt per primary and substep).
    land_t (N,): the last substep a strike limb's landed episode touched the target's head or torso (m1's 0.3 s exclusion).
    ep: per episode counters (landed / clean per (limb, zone in head, torso), window outcomes, paid, the step sums)."""
    STEP = ("head_imp", "torso_imp", "head_clean", "torso_clean", "head_landed", "torso_landed", "block", "tangle", "work")

    def __init__(self, n, zone_idx, body_zone_idx=None, p=None, slots=0, dtype=torch.float32, device="cpu", log=None, block=None):
        """block (P,) bool: the hit-sensor primaries of run 2's 'block' zone (arm shells of BLOCK_LINKS; MjContactSensors.block_idx);
        None: every arm-zone primary."""
        self.n, self.p, self.dtype, self.dev, self.log = n, {**HIT_P, **(p or {})}, dtype, device, log; Z = len(ZONES)
        self.zi = torch.as_tensor(zone_idx, device=device); self.bz = None if body_zone_idx is None else torch.as_tensor(body_zone_idx, device=device)
        self.blk = (self.zi == ARM) if block is None else torch.as_tensor(block, dtype=torch.bool, device=device); assert bool((~self.blk | (self.zi == ARM)).all())
        self.A = Episodes(n, (4, Z), self.p, dtype, device); self.B = Episodes(n, (2, 4, Z), self.p, dtype, device)
        self.ht = torch.zeros(4, Z, dtype=torch.bool, device=device); self.ht[:, :2] = True   # strike limb on head / torso
        z = lambda *s, dt=dtype: torch.zeros(n, *s, dtype=dt, device=device)
        self.step = {k: z() for k in self.STEP}; self.land_t = torch.full((n,), -INF, dtype=dtype, device=device)
        self.ep = dict(landed=z(4, 2), clean=z(4, 2), windows=z(len(OUTCOMES)), paid=z(), **{k: z() for k in self.STEP})
        if self.bz is not None:
            self.tmask = torch.tensor([ZONES[i] in TANGLE_ZONES for i in self.bz.tolist()], device=device); self.wmask = torch.tensor([ZONES[i] in WORK_ZONES for i in self.bz.tolist()], device=device)
        self.K = slots
        if slots:
            self.sid = torch.full((n, slots), -1, dtype=torch.long, device=device); self.w_act = z(slots, dt=torch.bool); self.w_side, self.w_limb = z(slots, dt=torch.long), z(slots, dt=torch.long)
            self.w_on, self.w_off, self.w_apex, self.w_arm, self.w_hit, self.w_any = (z(slots) for _ in range(6)); self.w_hz = z(slots, dt=torch.long); self.ewin = z(slots, 5, dt=torch.bool)
            self.w_armb = z(slots, dt=torch.bool)   # the arm episode opened with a block shell touching (run 2's bz)
            self.EZ = torch.tensor([HEAD, TORSO, ARM, HEAD, TORSO], device=device); self.PART = torch.tensor(PARTNER, device=device)

    # ---- one control step
    def process(self, F, t_end, v_rel=None, found=None, win=None, body=None, signed=True):
        """F (N, 8, P, S, 3) hit sensors (MuJoCo netforce, oldest substep first), t_end (N,) episode time at the step's end, v_rel
        (N, 8, P or 1, S, 3) striker minus target velocity per substep (None: closing speed 0), found (N, 8, P, S) or None; win:
        dict(id, on, apex, side[, limb]) (N, K) from the aim's intent slots (id -1 = empty); body: (Fb (N, Pb, S, 3), vb (N, Pb, S, 3))
        the body sensor and the target links' velocities. Returns self.step."""
        for v in self.step.values(): v.zero_()
        N, S = self.n, F.shape[3]; Fz, Tz, st = zone_forces(F, self.zi, found)
        vc = closing_speeds(F, v_rel, st, signed) if v_rel is not None else torch.zeros_like(Fz)
        Fz, Tz, vc = (x.reshape(N, 2, 4, len(ZONES), S) for x in (Fz, Tz, vc))
        if self.K:
            self._new_windows(win); tp = (torch.linalg.vector_norm(F, dim=-1) > 0) if found is None else (found > 0)
            tb = (tp & self.blk[None, None, :, None]).any(2).reshape(N, 8, S)   # a block shell touched, per limb sensor and substep
        for j in range(S):
            tj = (t_end - (S - 1 - j) * DT).to(self.dtype); f, t, v = Fz[..., j], Tz[..., j], vc[..., j]
            fa, ta = f.sum(1), t.any(1); va = torch.where(t, v, torch.full_like(v, -INF)).amax(1); va = torch.where(ta, va, torch.zeros_like(va))
            self._score(self.A.close(tj)); self.A.update(fa, ta, va); self.B.close(tj); nwB = self.B.update(f, t, v)
            lt = (self.A.struck() & self.A.open & ta & self.ht).flatten(1).any(1); self.land_t = torch.where(lt, tj, self.land_t)
            self.step["head_imp"] += fa[..., HEAD].sum(1) * DT; self.step["torso_imp"] += fa[..., TORSO].sum(1) * DT
            if self.K: self._win_substep(tj, nwB, tb[..., j])
        if body is not None:
            Fb, vb = body; a = torch.linalg.vector_norm(Fb, dim=-1) * DT
            self.step["tangle"] += (a * self.tmask[None, :, None]).sum((1, 2))
            self.step["work"] += ((Fb * vb).sum(-1).clamp(min=0) * DT * self.wmask[None, :, None]).sum((1, 2))
        for k in ("head_imp", "torso_imp", "tangle", "work"): self.ep[k] += self.step[k]
        return self.step

    def _score(self, C):
        """Count the merged episodes C (N, 4, Z) closed now (their fields still hold)."""
        A = self.A; L = C & A.struck() & self.ht; Cl = L & A.clean()
        for k, x in (("head_landed", L[..., HEAD]), ("torso_landed", L[..., TORSO]), ("head_clean", Cl[..., HEAD]), ("torso_clean", Cl[..., TORSO])): self.step[k] += x.sum(1).to(self.dtype)
        self.ep["landed"] += L[..., :2].to(self.dtype); self.ep["clean"] += Cl[..., :2].to(self.dtype)
        if self.log is not None and bool(C.any()):
            S = A.struck()
            for e, l, z in C.nonzero().tolist():
                self.log.append(dict(kind="episode", env=e, limb=LIMBS[l], zone=ZONES[z], t0=float(A.t0[e, l, z]), t1=float(A.t1[e, l, z]), impulse_Ns=float(A.imp[e, l, z]),
                                     peak_N=float(A.peak[e, l, z]), v_close=float(A.vc[e, l, z]), landed=bool(L[e, l, z]), clean=bool(Cl[e, l, z]), struck=bool(S[e, l, z])))

    # ---- windows
    def _new_windows(self, win):
        if win is None: return
        new = win["id"] != self.sid; self._decide(self.w_act & new)
        lb = win.get("limb", torch.zeros_like(win["id"]))
        self.sid = torch.where(new, win["id"], self.sid); self.w_act = torch.where(new, win["id"] >= 0, self.w_act)
        for x, y in ((self.w_on, win["on"]), (self.w_apex, win["apex"]), (self.w_off, win["apex"] + WIN_TAIL_S)): x.copy_(torch.where(new, y.to(self.dtype), x))
        self.w_side = torch.where(new, win["side"], self.w_side); self.w_limb = torch.where(new, lb, self.w_limb)
        for x in (self.w_arm, self.w_hit, self.w_any): x.masked_fill_(new, INF)
        self.ewin &= ~new[..., None]; self.w_armb &= ~new

    def _keys(self):
        L = torch.stack([self.w_limb] * 3 + [self.PART[self.w_limb]] * 2, -1); return (self.w_side[..., None] * 4 + L) * len(ZONES) + self.EZ   # (N, K, 5) into (2, 4, Z)

    def _win_substep(self, tj, nwB, tb):
        B, k = self.B, self._keys(); g = lambda X: torch.gather(X.flatten(1), 1, k.flatten(1)).view(*k.shape)
        op, nw, t0, st = g(B.open), g(nwB), g(B.t0), g(B.struck()); tj = tj[:, None]
        inw = self.w_act & (tj >= self.w_on - W_TOL) & (tj <= self.w_off + W_TOL); self.ewin = torch.where(nw, inw[..., None], self.ewin)
        arm = nw[..., 2] & inw & torch.isinf(self.w_arm); self.w_arm = torch.where(arm, tj.expand_as(self.w_arm), self.w_arm)
        self.w_armb = torch.where(arm, torch.gather(tb, 1, self.w_side * 4 + self.w_limb), self.w_armb)   # run 2: block zone wins a tie
        hit = st & self.ewin & op; tt = torch.where(hit, t0, torch.full_like(t0, INF))
        for e, zz in ((0, HEAD), (1, TORSO)):
            b = tt[..., e] < self.w_hit; self.w_hit = torch.where(b, tt[..., e], self.w_hit); self.w_hz = torch.where(b, torch.full_like(self.w_hz, zz), self.w_hz)
        self.w_any = torch.minimum(self.w_any, tt[..., [0, 1, 3, 4]].amin(-1))
        pend = (op & self.ewin & ~st)[..., [0, 1, 3, 4]].any(-1)
        self._decide(self.w_act & (tj > self.w_off + W_TOL) & ~pend)

    def _decide(self, D):
        if not self.K or not bool(D.any()): return
        bl = D & torch.isfinite(self.w_arm) & (self.w_arm < self.w_hit); hit = D & ~bl & torch.isfinite(self.w_hit)
        oc = torch.stack([hit & (self.w_hz == HEAD), hit & (self.w_hz == TORSO), bl, D & ~bl & ~hit], -1)
        pay = bl & self.w_armb & (self.w_arm <= self.w_apex + BLOCK_BY_APEX_S + W_TOL) & ~torch.isfinite(self.w_any)
        self.step["block"] += pay.sum(1).to(self.dtype); self.ep["paid"] += pay.sum(1).to(self.dtype); self.ep["windows"] += oc.sum(1).to(self.dtype); self.w_act = self.w_act & ~D
        if self.log is not None:
            for e, s in D.nonzero().tolist():
                self.log.append(dict(kind="window", env=e, id=int(self.sid[e, s]), outcome=OUTCOMES[int(oc[e, s].int().argmax())], paid=bool(pay[e, s]), arm_block=bool(self.w_armb[e, s]),
                                     t_arm=float(self.w_arm[e, s]), t_hit=float(self.w_hit[e, s]), t_any=float(self.w_any[e, s])))

    # ---- episode ends
    def close_envs(self, ids):
        """Score the open episodes and decide the open windows of envs ids (an episode ending; HitMeter scores open episodes at the bout's
        end as they stand). Then reset them."""
        m = torch.zeros(self.n, dtype=torch.bool, device=self.dev); m[ids] = True
        self._score(self.A.open & m[:, None, None])
        if self.K: self._decide(self.w_act & m[:, None])
        self.reset(ids)

    def reset(self, ids):
        self.A.reset(ids); self.B.reset(ids); self.land_t[ids] = -INF
        for v in self.ep.values(): v[ids] = 0
        if self.K:
            self.sid[ids] = -1; self.w_act[ids] = False; self.ewin[ids] = False; self.w_armb[ids] = False
            for x in (self.w_arm, self.w_hit, self.w_any): x[ids] = INF

LIMB_COM = dict(hand="wrist_roll_link", forearm="elbow_link", foot="ankle_roll_link", shin="knee_link")   # the limb body whose COM velocity the GPU reads

class MjContactSensors:
    """CPU stand-in for the hit and body sensors on an r1gym two-R1 MjModel (striker = the opponent, 'opp_' bodies; reverse=True: the
    R1 strikes): read(d) after mj_step -> F (8, P, 3), found (8, P), v_com / v_pt (8, P, 3) (striker minus target velocity: LIMB_COM's
    COM minus the primary link's COM, or both bodies at the strongest contact's point as the HitMeter takes it), Fb (Pb, 3), vb (Pb, 3)
    (the target link's COM velocity). F and Fb are MuJoCo's contact-sensor netforce (the force on the secondary, world frame), equal to
    mjSENS_CONTACT built as mjlab builds it (tests). Shells = target geoms that can collide with a striker geom, in geom order; zones
    by zone_of on the link and mesh names, limbs by limb_of."""
    def __init__(self, m, reverse=False, body_zones=BODY_ZONES):
        import mujoco
        self.mj, self.m = mujoco, m; nb = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        opp = np.array([m.geom_bodyid[g] != 0 and nb[g].startswith("opp_") for g in range(m.ngeom)]); r1 = np.array([m.geom_bodyid[g] != 0 for g in range(m.ngeom)]) & ~opp
        st, tg = (r1, opp) if reverse else (opp, r1); spf, tpf = ("", "opp_") if reverse else ("opp_", "")
        hit = lambda a, b: bool((m.geom_contype[a] & m.geom_conaffinity[b]) | (m.geom_contype[b] & m.geom_conaffinity[a]))
        mesh = lambda g: m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else m.geom(g).name
        strip = lambda s, pf: s[len(pf):] if pf and s.startswith(pf) else s
        S = np.flatnonzero(st); self.prim = [int(g) for g in np.flatnonzero(tg) if any(hit(g, h) for h in S)]
        self.zone_idx = [ZONES.index(zone_of(strip(nb[g], tpf), strip(mesh(g), tpf))) for g in self.prim]
        self.limb = np.full(m.ngeom, -1); self.pix = np.full(m.ngeom, -1); self.pix[self.prim] = np.arange(len(self.prim))
        for g in S:
            sd, lb = limb_of(strip(nb[g], spf))
            if lb in LIMBS and sd: self.limb[g] = SIDES.index(sd) * 4 + LIMBS.index(lb)
        self.block_idx = [ZONES[z] == "arm" and any(k in strip(nb[g], tpf) for k in BLOCK_LINKS) for g, z in zip(self.prim, self.zone_idx)]
        keep = [i for i, z in enumerate(self.zone_idx) if ZONES[z] in body_zones]; self.bprim = [self.prim[i] for i in keep]; self.bzone_idx = [self.zone_idx[i] for i in keep]
        self.bix = np.full(m.ngeom, -1); self.bix[self.bprim] = np.arange(len(self.bprim)); self.is_st, self.is_tg = st, tg
        self.limb_body = [m.body(f"{spf}{'left' if s == 'l' else 'right'}_{LIMB_COM[l]}").id for s in SIDES for l in LIMBS]
        self.f6, self.v6 = np.zeros(6), np.zeros(6)

    def _vel(self, d, b, p=None):
        self.mj.mj_objectVelocity(self.m, d, self.mj.mjtObj.mjOBJ_BODY, int(b), self.v6, 0)
        return self.v6[3:].copy() if p is None else self.v6[3:] + np.cross(self.v6[:3], p - d.xipos[b])

    def read(self, d):
        """d after mj_step (contacts, forces, cvel and xipos of the state the contacts were found in)."""
        m, P, Pb = self.m, len(self.prim), len(self.bprim); F, fo, vcom, vpt, best = np.zeros((8, P, 3)), np.zeros((8, P)), np.zeros((8, P, 3)), np.zeros((8, P, 3)), np.full((8, P), -1.0)
        Fb, vb = np.zeros((Pb, 3)), np.zeros((Pb, 3)); n = d.ncon
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            for i in np.flatnonzero((self.is_tg[g1] & self.is_st[g2]) | (self.is_tg[g2] & self.is_st[g1])):
                a, b = int(g1[i]), int(g2[i]); t, s = (a, b) if self.is_tg[a] else (b, a); p = self.pix[t]
                if p < 0: continue
                self.mj.mj_contactForce(m, d, int(i), self.f6); f = (1.0 if t == a else -1.0) * (self.f6[:3] @ d.contact.frame[i].reshape(3, 3))   # force on the striker
                q = self.bix[t]
                if q >= 0: Fb[q] += f
                l = self.limb[s]
                if l < 0: continue
                F[l, p] += f; fo[l, p] += 1; fn = abs(float(self.f6[0]))
                if fn >= best[l, p]:   # HitMeter: the strongest contact's point and bodies
                    best[l, p] = fn; bs, bt = m.geom_bodyid[s], m.geom_bodyid[t]; pt = d.contact.pos[i]
                    vpt[l, p] = self._vel(d, bs, pt) - self._vel(d, bt, pt); vcom[l, p] = self._vel(d, self.limb_body[l]) - self._vel(d, bt)
        for q, g in enumerate(self.bprim):
            if Fb[q].any(): vb[q] = self._vel(d, m.geom_bodyid[g])
        return F, fo, vcom, vpt, Fb, vb
