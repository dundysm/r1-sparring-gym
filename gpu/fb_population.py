"""Full-body M2 v2 opponent population and library wiring (M2V2_PLAN_v3 2.2, design_learned-tracker 3.7 / 4.2; A6.2, A9.3, A9.4).
Pure torch + numpy; the mjlab task (gpu/fullbody_task.py) and the CPU replica (scripts/fb_replica.py) both drive it.

FbMotion     one fb_opponent.Motion over the train originals (fb_motion.npz, the tracker's own reference) followed by the L1 rows
             (fb_l1.npz, or fb_l1_loco.npz: plan A6.3), each row a 'clip' of its own: q for placement, the reference its driver is
             fed (npz ref_*: a conjugated row's is mirror_ref of T1's training reference, M8b), the strike table (L1 strike_row ->
             clip), clean-start masks, conj (A6.2), family ('original' for the originals). loco: kfab_203 and every row hosted on it
             are left out (P3: removed from every training tensor).
Population   per episode and env (D2 3.7): row t0_nominal 40% (T0, originals, nominal eval aim, eval distance grid {0.40, 0.45,
             0.50, 0.55}), t0_shadow 10% (T0, originals, shadow aim), t1 35% (L1, aim + jitter, distance uniform over the 0.35-0.65
             grid, pushes), t1_dr 15% (t1 + opponent DR). t0_only (stage-A iterations 1-100, WATCH6 E0): 100% t0_nominal. Start t0
             uniform over the (clip, distance) clean-start grid (START_STEP), as the replica's training reset. Controller (A9.3): T1
             drives the L1 rows of T1_FAMILIES (reangle: the one family below T1-b's 90% bar with T0 alone); T0 drives every other
             row, conjugated on mirrored rows (A6.2). t1_ok False (T1 failed a gate, or not trained): T1_FAMILIES rows are never
             drawn and T0 drives all (plan 2.2: "T1 fails a gate -> 100% T0").
PolicyMux    the two trackers as one callable: both run on every env, a per-env select (design 3.7 'Implementation'); wrapped by
             fb_mirror.conjugate for the mirrored rows (FM.conjugate(drv, conj) after drv.mlp = PolicyMux(...)).
SwapAim      design 4.2's p 0.2 mirrored-strike swap (A9.4) around an AimOp, on jitter (T1) rows only: a new hand-strike intent is
             swapped with p SWAP_P; while a swapped intent's arm weight w (AimOp.weight) is on, the shown reference's upper body
             (waist and both arms, UPPER) is blended toward its L/R mirror (fb_mirror joint map: the other hand throws the same
             strike, mirrored, at the same onset; legs, root and anchor untouched), rq + w (M rq - rq), rdq + w (M rdq - rdq) +
             dw/dt (M rq - rq); the intent's limb is flipped, so the aim's StandOff and vz warp, the hit windows and the logs all
             follow the striking hand. The flip lands right after the intent is made (0.78-0.8 s before its onset, w = 0 there);
             an intent made inside its own arm lead (only at an episode's first steps) keeps its limb for that one step.
             T0 rows are untouched (bit-equal to the AimOp alone; tests).
OppDR        the t1_dr rows' opponent DR (masses x U(0.9, 1.1) per opponent body, kp x U(0.9, 1.1) per opponent actuator (gain and
             bias), foot mu U(0.5, 1.25)), written after the defender's fb_dr.write (which restores every field's nominal first), no
             set_const; other rows keep the opponent nominal."""
from pathlib import Path
import numpy as np, torch
import fb_opponent as FO, fb_mirror as FM

GPU = Path(__file__).resolve().parent; FB = GPU / "motions" / "fb"
ROWS = ("t0_nominal", "t0_shadow", "t1", "t1_dr")         # = fb_rewards.ROWS (the down-neutral EMA's keys; asserted in tests)
SHARES = (0.40, 0.10, 0.35, 0.15)
T0_ONLY_ITERS = 100                                      # stage-A iterations 1-100: 100% t0_nominal, no pushes (m6)
T0_DIST, T1_DIST = (0.40, 0.45, 0.50, 0.55), (0.35, 0.65)   # eval hard grid (EO TIERS hard); U(0.35, 0.65) on the 0.01 m clean grid
T1_FAMILIES = ("reangle",)                               # A9.3: T1 trains on the reangle family only
SWAP_P = 0.2                                             # design 4.2
UPPER = tuple(range(12, 24))                             # waist roll / yaw and both arms in CTRL_JOINTS order
OPP_DR = dict(mass=(0.9, 1.1), kp=(0.9, 1.1), foot_mu=(0.5, 1.25))
LOCO_DROP = "kfab_203"

class FbMotion(FO.Motion):
    def __init__(self, loco=False, device="cpu", orig=FO.MOTION, lib=None):
        lib = lib or FB / ("fb_l1_loco.npz" if loco else "fb_l1.npz")
        with np.load(orig, allow_pickle=False) as z: O = {k: z[k] for k in z.files}
        with np.load(lib, allow_pickle=False) as z: L = {k: z[k] for k in z.files}
        assert tuple(O["clips"].tolist()) == FO.TRAIN and set(L["host"].tolist()) <= set(FO.TRAIN), "train clips only"
        assert np.array_equal(O["distances"], L["distances"]) and float(O["fps"]) == float(L["fps"]) == FO.FPS
        keep_o = [i for i, c in enumerate(FO.TRAIN) if not (loco and c == LOCO_DROP)]; keep_l = [i for i, h in enumerate(L["host"].tolist()) if not (loco and h == LOCO_DROP)]
        self.clips = tuple(FO.TRAIN[i] for i in keep_o) + tuple(str(L["names"][i]) for i in keep_l)
        self.family = ("original",) * len(keep_o) + tuple(str(L["family"][i]) for i in keep_l); self.host = tuple(FO.TRAIN[i] for i in keep_o) + tuple(str(L["host"][i]) for i in keep_l)
        self.n_orig = len(keep_o); segs = [(O, int(O["row0"][i]), int(O["frames"][i])) for i in keep_o] + [(L, int(L["row0"][i]), int(L["frames"][i])) for i in keep_l]
        cat = lambda k: np.concatenate([A[k][r:r + T] for A, r, T in segs]); fr = np.array([T for _, _, T in segs])
        t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dt)
        self.frames, self.row0 = t(fr, torch.long), t(np.concatenate([[0], np.cumsum(fr)[:-1]]), torch.long); self.q = t(cat("q"), torch.float64)
        self.ref_q, self.ref_dq, self.ref_apos, self.ref_aquat = (t(cat(k)) for k in ("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"))
        self.clean = t(np.concatenate([O["clean"][keep_o], L["clean"][keep_l]])); self.distances = t(O["distances"], torch.float64)
        self.conj = t(np.r_[np.zeros(len(keep_o), bool), L["conj"][keep_l].astype(bool)]); self.start_step = float(O["start_step"])
        so, sl = np.isin(O["strike_clip"], keep_o), np.isin(L["strike_row"], keep_l); ro, rl = {c: j for j, c in enumerate(keep_o)}, {c: len(keep_o) + j for j, c in enumerate(keep_l)}
        keys = sorted({k[7:] for k in O if k.startswith("strike_")} & {k[7:] for k in L if k.startswith("strike_")})
        self.strikes = {k: np.concatenate([O["strike_" + k][so], L["strike_" + k][sl]]) for k in keys}
        self.strikes["clip"] = np.r_[[ro[c] for c in O["strike_clip"][so]], [rl[c] for c in L["strike_row"][sl]]].astype(np.int32)
        self.device = device
    def index(self, name): return self.clips.index(name)

class Population:
    """Draws per env and episode (module docstring). M: FbMotion; t1_ok: T1 present and gated in."""
    def __init__(self, M, t1_ok=True, shares=SHARES, t1_families=T1_FAMILIES):
        self.M, self.t1_ok, self.shares, self.t1_families = M, bool(t1_ok), torch.tensor(shares, dtype=torch.float64), tuple(t1_families)
        assert abs(float(self.shares.sum()) - 1) < 1e-12; dist = M.distances.cpu().numpy()
        self.orig = torch.arange(M.n_orig); fam = np.array(M.family)
        self.lib = torch.as_tensor(np.flatnonzero((fam != "original") & (self.t1_ok | ~np.isin(fam, self.t1_families))))
        self.t1_clip = torch.as_tensor(np.isin(fam, self.t1_families) & self.t1_ok)   # clips T1 drives
        self.d_t0 = torch.as_tensor([int(np.flatnonzero(np.isclose(dist, d))[0]) for d in T0_DIST])
        self.d_t1 = torch.as_tensor(np.flatnonzero((dist >= T1_DIST[0] - 1e-9) & (dist <= T1_DIST[1] + 1e-9)))

    def draw(self, n, gen=None, t0_only=False):
        """n episodes -> dict of [n] tensors (cpu): row, clip, dist_ix, distance, t0, aimed, jitter, conj, t1 (T1 drives), opp_dr, push."""
        u = lambda: torch.rand(n, generator=gen, dtype=torch.float64); pick = lambda pool: pool[torch.randint(len(pool), (n,), generator=gen)]
        row = torch.zeros(n, dtype=torch.long) if t0_only else torch.bucketize(u(), torch.cumsum(self.shares, 0)[:-1], right=True)
        t1 = row >= 2; clip = torch.where(t1, pick(self.lib), pick(self.orig)); di = torch.where(t1, pick(self.d_t1), pick(self.d_t0))
        cl = self.M.clean.cpu()
        for _ in range(64):   # a (clip, distance) with no clean start: redraw its distance (rare; asserted below)
            bad = ~cl[clip, di].any(-1)
            if not bool(bad.any()): break
            di = torch.where(bad, torch.where(t1, pick(self.d_t1), pick(self.d_t0)), di)
        assert not bool((~cl[clip, di].any(-1)).any()), "no clean start"
        g = cl[clip, di]; k = (u() * g.sum(-1)).long().clamp(max=g.shape[-1] - 1); idx = torch.argmax((torch.cumsum(g.long(), -1) > k[:, None]).long(), -1)
        return dict(row=row, clip=clip, dist_ix=di, distance=self.M.distances.cpu()[di], t0=torch.round(idx.double() * self.M.start_step * 1000) / 1000,
                    aimed=row != 1, jitter=t1, conj=self.M.conj.cpu()[clip], t1=t1 & self.t1_clip[clip], opp_dr=row == 3, push=t1)

class PolicyMux:
    """t0, t1: callables obs [n, 132] -> actions [n, 24]; sel [n] bool (True: T1). t1 None: T0 everywhere."""
    def __init__(self, t0, t1=None, sel=None):
        self.t0, self.t1, self.sel = t0, t1, sel; self.dtype = getattr(t0, "dtype", None)
    @torch.no_grad()
    def __call__(self, obs):
        if self.t1 is None: return self.t0(obs)
        return torch.where(self.sel[:, None], self.t1(obs), self.t0(obs))

def driver(n, M, dtype=torch.float32, device="cpu", t1_pt=None):
    """TrackerDriver over M with T0 (and T1 if t1_pt) muxed per env and conjugated per env: returns (drv, mux, mirror policy).
    At reset: mux.sel[ids] = draw['t1'], pol.set(ids, draw['conj'])."""
    drv = FO.TrackerDriver(n, dtype, device, motion=M); sel = torch.zeros(n, dtype=torch.bool, device=device)
    mux = PolicyMux(drv.mlp, FO.TrackerMLP(t1_pt, dtype, device) if t1_pt else None, sel); drv.mlp = mux
    pol = FM.conjugate(drv, torch.zeros(n, dtype=torch.bool)); return drv, mux, pol

class SwapAim:
    """An AimOp with design 4.2's mirrored-strike swap on its jitter rows (module docstring). Attribute access falls through to the
    AimOp (valid, limb, seq, intents, ...). gen: the swap draws' torch Generator (on the op's device)."""
    def __init__(self, aim, p=SWAP_P, gen=None):
        self.aim, self.p, self.gen = aim, float(p), gen; n, K1, dv = aim.n, aim.K + 1, aim.device
        self.swap = torch.zeros(n, K1, dtype=torch.bool, device=dv); self.w_prev = torch.zeros(n, dtype=torch.float64, device=dv)
        self.up = torch.as_tensor(UPPER, device=dv); self.swaps = torch.zeros(n, dtype=torch.long, device=dv)
    def __getattr__(self, k): return getattr(self.aim, k)
    def reset(self, ids, clip, distance, aimed=True, jitter=False):
        self.aim.reset(ids, clip, distance, aimed, jitter); ids = self.aim._ids(ids); self.swap[ids] = False; self.w_prev[ids] = 0; self.swaps[ids] = 0
    def weight(self, now):
        """The live swapped intents' arm weight per env (max over slots), float64."""
        A = self.aim; now = torch.as_tensor(now, dtype=torch.float64, device=A.device).expand(A.n)
        return torch.where(self.swap & A.valid, A.weight(now), torch.zeros_like(A.t_on)).amax(1)
    @torch.no_grad()
    def step(self, fi, now, rq, rdq, *args, dt=FO.CTRL_DT, **kw):
        A = self.aim; w = self.weight(now); u = self.up
        if A.any_jit:   # no host sync: the blend runs on every env once any is a jitter row, adding exact zeros where w = w_prev = 0
            mq, mdq = FM.mirror_joints(rq), FM.mirror_joints(rdq); wd = w.to(rq.dtype)[:, None]; dw = ((w - self.w_prev) / dt).to(rq.dtype)[:, None]
            rq, rdq = rq.clone(), rdq.clone(); e = mq[:, u] - rq[:, u]
            rdq[:, u] = rdq[:, u] + wd * (mdq[:, u] - rdq[:, u]) + dw * e; rq[:, u] = rq[:, u] + wd * e
        self.w_prev = w; s0 = A.nseq.clone(); out = A.step(fi, now, rq, rdq, *args, dt=dt, **kw)
        new = A.valid & (A.seq >= s0[:, None]) & (A.seq < A.nseq[:, None]); new[:, A.K] = False
        if A.any_jit:
            d = (torch.rand(new.shape, generator=self.gen, device=A.device) < self.p) & new & A.jit[:, None]
            self.swap = torch.where(new, d, self.swap); A.limb = torch.where(d, 1 - A.limb, A.limb); self.swaps += d.sum(1)
        return out

class OppDR:
    """Opponent DR index sets from a compiled model (prefix: 'opp_' in the eval model, 'opponent/' in mjlab's)."""
    def __init__(self, m, ctrl_joints, prefix="opp_"):
        nb = [m.body(b).name for b in range(m.nbody)]; ob = {b for b in range(1, m.nbody) if nb[b].startswith(prefix)}
        self.bodies = np.array(sorted(ob)); self.feet = np.array([g for g in range(m.ngeom) if "foot" in (m.geom(g).name or "") and m.geom_bodyid[g] in ob], np.int64)
        names = [m.actuator(a).name for a in range(m.nu)]
        self.act = np.array([next(a for a, x in enumerate(names) if x in (prefix + j, prefix + j.replace("_joint", ""))) for j in ctrl_joints])
        self.nb, self.na = len(self.bodies), len(self.act)
    def sample(self, n, generator=None, dtype=torch.float64, device="cpu"):
        u = lambda r, s: r[0] + (r[1] - r[0]) * torch.rand(s, dtype=dtype, device=device, generator=generator)
        return dict(mass=u(OPP_DR["mass"], (n, self.nb)), kp=u(OPP_DR["kp"], (n, self.na)), foot_mu=u(OPP_DR["foot_mu"], (n,)))
    def write(self, fields, ix_ids, p):
        """fields as fb_dr.write's (already reset to nominal for these envs by fb_dr.write); ix_ids: the envs with opponent DR (rows of p)."""
        I = torch.as_tensor(ix_ids, dtype=torch.long)[:, None]; b, a = torch.as_tensor(self.bodies), torch.as_tensor(self.act); f = fields
        if len(I) == 0: return
        f["body_mass"][I, b] = f["body_mass"][I, b] * p["mass"].to(f["body_mass"].dtype)
        f["actuator_gainprm"][I, a, 0] = f["actuator_gainprm"][I, a, 0] * p["kp"].to(f["actuator_gainprm"].dtype)
        f["actuator_biasprm"][I, a, 1] = f["actuator_biasprm"][I, a, 1] * p["kp"].to(f["actuator_biasprm"].dtype)
        if len(self.feet): f["geom_friction"][I, torch.as_tensor(self.feet), 0] = p["foot_mu"][:, None].to(f["geom_friction"].dtype)
