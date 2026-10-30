"""M6 CPU replica (M2V2_PLAN_v3 3 M6, 3.1-3.2; A4.3): r1gym's two-R1 world (build_spec(opponent=True), the eval's MjModel) stepped
in the eval's order by its own loop, one bout at a time (N = 1 per process), with the M2-M5 modules plugged in.
Switches (conf, EVAL = the eval):
  hulls     'full' (the eval's mesh shells) | 'mix' (fb_hulls.npz: 256-point hulls on head, waist_yaw, elbows, wrists, 64 elsewhere,
            plan 2.1) | 'all256' (256 everywhere, export_attacks.fps_sample of the full hull, as export_fb_hulls): both robots' shells; a hull mesh takes
            its original's name (the original is renamed fborig_<mesh>; A10: r1_zone / zone_of key on mesh names)
  solver    (iterations, ls_iterations): (100, 50) eval; the 3.2 fallback list 10/20 -> 25/30 -> 50/50 -> 100/50 (Newton, cone,
            impratio, ccd_iterations 35 stay the eval's)
  reset     'eo' (the episode as EO.run resets it: R1HitEnv's rng from env_seed draws DR, act_delay and the obs noise; t0 and k from
            the episode) | 'train' (the T0-nominal training reset, torch backends: per episode a torch Generator(seed) draws k and
            act_delay (fb_events, first), DR (fb_dr.sample), t0 uniform over the clean grid of the episode's (clip, distance), a bout
            heading U(-pi, pi) turning both robots about the origin (TrackerDriver.reset), and the obs noise (EvalActorObs))
  scorer    'hitmeter' (r1gym HitMeter, the eval's row) | 'account' (FullBodyHitAccount on the contact sensors, primary; the HitMeter
            still rides along read-only for same-state comparisons and the windows; window slots from the aim's intents)
  backend   'numpy' (r1gym's own Tracker / Aim / DelayedNoisyObs / KeypointSource / R1HitEnv action and DR arithmetic) | 'torch64' |
            'torch32' (TrackerDriver + AimOp (compute float64), EvalActorObs + kp_state + proprio83, HybridDelayedAction, fb_dr,
            fb_events.fell_eval / OppDown, in that dtype)
  self_collision  False (eval) | True (R6: defender capsules collide with each other, bit 8; r1.xml's excludes and parent filtering)
  vmode     the account's closing-speed input: 'pos' (velocity of the limb body and of the struck link at the contact sensor's
            netforce 'pos', A4.3 deployed if it passes) | 'site' (palm / foot site velocities) | 'com' (body COMs) | 'point' (the HitMeter's
            strongest contact point)
  actor     'mlp' (TrackerMLP in the backend's dtype) | 'onnx' (diagnostic: the eval's own tracker session on the driver's obs cast to
            float32, as Tracker.act feeds it)
  ref       'fb' (the driver's reference from fb_motion.npz) | 'eval' (diagnostic: the eval Tracker's own reference arrays, its ONNX
            initializers at Tracker.frame, in place of fb_motion's)
  motion    (optional key, torch backends) 'fb' (default: fb_motion.npz, the train originals) | 'l1' (fb_population.FbMotion: the
            originals then the L1 rows; an episode's 'row' names the L1 row, 'clip' its host; M8b contract, A9.4: the driver is
            fb_mirror-conjugated with mask = the row's npz conj, set at reset, and fed the row's npz reference; the aim is
            fb_population.SwapAim, inert unless the episode sets 'jitter')
Outer loops (face the opponent's pelvis bearing, hold the spot), the 83's arithmetic for the numpy backend and the eval's DR draws are
R1HitEnv's own functions (borrowed, not copied); the per-step bookkeeping (falls, opponent fall log via EO.ContactTail, both-up
intervals, ring, spacing) and the row are EO.run's. Each conf compiles one model per process (cached).
run(name, eps, conf) -> rows in fb_r0eo's format (bout_row: kd, kd_edges, down_iv, act_delay, n, ...; eo = the EO.run-format row) plus
'acct' (the account's summary when scorer 'account') and 'hm' (the HitMeter's n when the account is primary).
probe(step_info) per step and per bout (checks: qpos traces, first shell contact, same-state module outputs, sensor logs)."""
import sys, json, copy, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, mujoco, torch
import fb_opponent as FO, fb_obs as OB, fb_actions as AC, fb_events as EV, fb_dr as DR, fb_hits as HT, fb_population as POP

HULLS, RESETS, SCORERS, BACKENDS, VMODES, ACTORS, REFS = ("full", "mix", "all256"), ("eo", "train"), ("hitmeter", "account"), ("numpy", "torch64", "torch32"), ("pos", "site", "com", "point"), ("mlp", "onnx"), ("fb", "eval")
EVAL = dict(hulls="full", solver=(100, 50), reset="eo", scorer="hitmeter", backend="numpy", self_collision=False, vmode="pos", actor="mlp", ref="fb")
FALLBACK = dict(hulls=("mix", "all256", "full"), solver=((10, 20), (25, 30), (50, 50), (100, 50)))   # 3.2, in order
HULLS_NPZ = ROOT / "gpu" / "motions" / "fb" / "fb_hulls.npz"
SELF_BIT = 8                                       # R6: defender capsule self-collision bit (0/1 -> 8/9)
K_SLOTS, LEAD_S = 8, 0.8                           # account window slots (AimOp's ring); intents visible from t_on - lead_s
SITE_OF = dict(hand="palm", forearm=None, foot="foot", shin=None)   # vmode 'site': the limb's site (else its body COM)
DT_T = {"torch64": torch.float64, "torch32": torch.float32}
MOTIONS = ("fb", "l1")

def conf(**kw):
    c = dict(EVAL, **kw); c["solver"] = tuple(int(x) for x in c["solver"])
    for k, opts in (("hulls", HULLS), ("reset", RESETS), ("scorer", SCORERS), ("backend", BACKENDS), ("vmode", VMODES), ("actor", ACTORS), ("ref", REFS)): assert c[k] in opts, (k, c[k])
    assert not (c["reset"] == "train" and c["backend"] == "numpy"), "the training reset needs a torch backend (bout heading)"
    assert not ((c["actor"] == "onnx" or c["ref"] == "eval") and c["backend"] == "numpy"), "actor onnx / ref eval are torch-backend diagnostics"
    assert c.get("motion", "fb") in MOTIONS and not (c.get("motion", "fb") == "l1" and (c["backend"] == "numpy" or c["actor"] != "mlp" or c["ref"] != "fb")), "motion l1: torch backends, mlp actor, fb ref"
    return c

def conf_key(c): return json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in sorted(c.items())})

# ---- the world
def shell_geoms(sp):
    """Shell mesh geoms of both robots in a build_spec MjSpec: defender (0, SHELL_BIT), opponent (SHELL_BIT, 0)."""
    from r1gym.model import SHELL_BIT
    return [g for g in sp.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype, g.conaffinity) in ((0, SHELL_BIT), (SHELL_BIT, 0))]

def hull_points(kind, m_eval):
    """{mesh name (defender's): body-frame points [n, 3]} of the reduced hulls ('mix': fb_hulls.npz; 'all256': 256 farthest-point samples)."""
    import export_fb_hulls as XH
    from export_attacks import fps_sample
    with np.load(HULLS_NPZ, allow_pickle=False) as z: names, n, P = z["meshes"].tolist(), z["n"], z["points"]
    if kind == "mix": return {k: P[i, :n[i]] for i, k in enumerate(names)}
    G = XH.mesh_geoms(m_eval); out = {}
    for k in names:
        V, _ = XH.hull(m_eval, G[k]); out[k] = V if len(V) <= XH.N_FINE else fps_sample(V, XH.N_FINE)
    return out

def build_model(ccfg, c, sensors=False):
    """The replica's compiled model for conf c (ccfg: the controller's cfg, as EO builds it). sensors: native contact sensors
    (mjSENS_CONTACT, reduce netforce, num 1, fields found / force / pos, as mjlab builds them: one per primary shell and field) for the
    8 opponent strike limbs on every defender shell, named '{sensor}/{mesh}/{field}' (A4.3 probe and vmode 'pos')."""
    from r1gym.model import build_spec
    sp = build_spec(opponent=True, cfg=ccfg); m0 = None
    if c["hulls"] != "full":
        m0 = sp.compile(); P = hull_points(c["hulls"], m0); sp = build_spec(opponent=True, cfg=ccfg)
        for g in shell_geoms(sp):   # A10: the hull takes the original mesh name (r1_zone / zone_of key on it), the original becomes fborig_<mesh>
            mn = g.meshname; base = mn[4:] if mn.startswith("opp_") else mn; sp.mesh(mn).name = f"fborig_{mn}"
            sp.add_mesh(name=mn, uservert=np.asarray(P[base], np.float64).ravel().tolist())
            g.meshname = mn; g.pos = [0.0, 0.0, 0.0]; g.quat = [1.0, 0.0, 0.0, 0.0]
    if c["self_collision"]:
        for g in sp.geoms:
            if g.name.endswith("_collision") and not g.name.startswith("opp_") and (g.contype, g.conaffinity) == (0, 1): g.contype, g.conaffinity = SELF_BIT, 1 | SELF_BIT
    names = None
    if sensors:
        for g in shell_geoms(sp):
            if (g.contype, g.conaffinity) != (0, 4): continue
            g.name = g.meshname + "_shell"
        mm = sp.compile(); s = HT.MjContactSensors(mm); names = [mm.mesh(mm.geom_dataid[g]).name + "_shell" for g in s.prim]
        for sd in HT.SIDES:
            for l in HT.LIMBS:
                kind, b = HT.LIMB_SECONDARY[l]; rt = mujoco.mjtObj.mjOBJ_XBODY if kind == "subtree" else mujoco.mjtObj.mjOBJ_BODY
                for nm in names:
                    for f, bit in (("found", 0), ("force", 1), ("pos", 4)):
                        sp.add_sensor(name=f"{sd}_{l}/{nm}/{f}", type=mujoco.mjtSensor.mjSENS_CONTACT, objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=nm,
                                      reftype=rt, refname=f"opp_{'left' if sd == 'l' else 'right'}_{b}", intprm=[1 << bit, 3, 1])
    m = sp.compile(); m.opt.iterations, m.opt.ls_iterations = c["solver"]
    return m, names

class OppIdx:
    """The opponent's addresses in the eval model (FightReplay's): free joint qpos / dof, pelvis, torso, actuators, bodies."""
    def __init__(self, m):
        from r1gym.model import CTRL_JOINTS
        j = m.joint("opp_floating_base_joint").id; self.qadr, self.vadr, self.nq = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), 31
        self.pel, self.torso = m.body("opp_pelvis").id, m.body("opp_torso_link").id; self.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("opp_")}
        self.act = np.array([m.actuator("opp_" + h.replace("_joint", "")).id for h in CTRL_JOINTS])

class _It:
    """An intent as meter_row reads it (StrikeIntent's fields and window())."""
    def __init__(self, x): self.__dict__.update(x)
    def window(self, tail): return (round(self.t_on, 4), round(self.t_apex + tail, 4), self.limb)

class _AimShim:
    def __init__(self, intents): self.p = dict(window_tail_s=HT.WIN_TAIL_S); self.intents = intents

class Sensors:
    """Per substep (a hook after mj_step): the 8 forward limb sensors and the body sensor (fb_hits.MjContactSensors), the account's
    closing-speed input by vmode, the block-shell touches (run 2's bz, for pay_ref) and, with native sensors, the A4.3 pos probe."""
    def __init__(self, m, vmodes, native=None, log=False):
        self.s, self.m, self.vmodes = HT.MjContactSensors(m), m, tuple(vmodes); s = self.s; self.P, self.Pb = len(s.prim), len(s.bprim)
        assert "pos" not in self.vmodes or native is not None, "vmode 'pos' reads the native contact sensors"
        self.site = {l: [m.site(f"opp_{'left' if sd == 'l' else 'right'}_{SITE_OF[l]}").id if SITE_OF[l] else None for sd in HT.SIDES] for l in HT.LIMBS}
        self.native = None
        if native is not None:
            self.native = np.array([[m.sensor(f"{k}/{nm}/pos").adr[0] for nm in native] for k in HT.SENSORS]); self.nfound = np.array([[m.sensor(f"{k}/{nm}/found").adr[0] for nm in native] for k in HT.SENSORS])
            self.nforce = np.array([[m.sensor(f"{k}/{nm}/force").adr[0] for nm in native] for k in HT.SENSORS])
        self.v6 = np.zeros(6); self.f6 = np.zeros(6); self.blk = np.asarray(s.block_idx, bool); self.log = log; self.reset()
    def reset(self):
        P, Pb = self.P, self.Pb; self.F, self.fo = np.zeros((8, P, 4, 3)), np.zeros((8, P, 4)); self.V = {v: np.zeros((8, P, 4, 3)) for v in self.vmodes}
        self.Fb, self.vb = np.zeros((Pb, 4, 3)), np.zeros((Pb, 4, 3)); self.j = 0; self.k = 0; self.block_touch = set(); self.rows = dict(hit=[], body=[])
        self.probe = dict(contacts=0, within_1cm=0, contacts_nonzero=0, within_1cm_nonzero=0, slots=0, slots_strongest_1cm=0, slots_pos_zero=0, max_err_nonzero=0.0, found_mismatch=0, force_mismatch=0)
    def begin(self):
        self.F[:] = 0; self.fo[:] = 0; self.Fb[:] = 0; self.vb[:] = 0; self.j = 0
        for v in self.V.values(): v[:] = 0
    def _vel(self, d, b, p):
        """Velocity of body b at point p (np.cross's arithmetic, written out: the same products and differences)."""
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, int(b), self.v6, 0); w0, w1, w2 = self.v6[:3].tolist(); r0, r1, r2 = (p - d.xipos[b]).tolist()
        return self.v6[3:] + np.array([w1 * r2 - w2 * r1, w2 * r0 - w0 * r2, w0 * r1 - w1 * r0])
    def _read(self, d):
        """MjContactSensors.read (its loop, operation for operation) computing the strongest-contact velocities only for vmodes 'com' /
        'point' (A7.1 profiling: they were most of the sensor time); F, found, Fb, vb bit-equal to read's (tests)."""
        s, m, P, Pb = self.s, self.m, self.P, self.Pb; F, fo, Fb, vb = np.zeros((8, P, 3)), np.zeros((8, P)), np.zeros((Pb, 3)), np.zeros((Pb, 3)); n = d.ncon
        strong = "com" in self.vmodes or "point" in self.vmodes; vcom, vpt, best = np.zeros((8, P, 3)), np.zeros((8, P, 3)), np.full((8, P), -1.0)
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]; f6 = s.f6
            for i in np.flatnonzero((s.is_tg[g1] & s.is_st[g2]) | (s.is_tg[g2] & s.is_st[g1])):
                a, b = int(g1[i]), int(g2[i]); t, st = (a, b) if s.is_tg[a] else (b, a); p = s.pix[t]
                if p < 0: continue
                mujoco.mj_contactForce(m, d, int(i), f6); f = (1.0 if t == a else -1.0) * (f6[:3] @ d.contact.frame[i].reshape(3, 3)); q = s.bix[t]
                if q >= 0: Fb[q] += f
                l = s.limb[st]
                if l < 0: continue
                F[l, p] += f; fo[l, p] += 1
                if strong and abs(float(f6[0])) >= best[l, p]:
                    best[l, p] = abs(float(f6[0])); bs, bt = m.geom_bodyid[st], m.geom_bodyid[t]; pt = d.contact.pos[i]
                    vpt[l, p] = s._vel(d, bs, pt) - s._vel(d, bt, pt); vcom[l, p] = s._vel(d, s.limb_body[l]) - s._vel(d, bt)
        for q in np.flatnonzero(np.any(Fb != 0, 1)): vb[q] = s._vel(d, m.geom_bodyid[s.bprim[q]])
        return F, fo, vcom, vpt, Fb, vb
    def substep(self, m, d):
        s, j = self.s, self.j; F, fo, vcom, vpt, Fb, vb = self._read(d); self.F[:, :, j], self.fo[:, :, j], self.Fb[:, j], self.vb[:, j] = F, fo, Fb, vb
        for vm, V in self.V.items():
            if vm == "com": V[:, :, j] = vcom
            elif vm == "point": V[:, :, j] = vpt
            else:   # the limb body (LIMB_COM) and the struck link, both at the sensor's pos ('pos') or at the limb's site ('site')
                for l, p in zip(*np.nonzero(fo)):
                    tb = m.geom_bodyid[s.prim[p]]; sb = s.limb_body[l]; si = self.site[HT.LIMBS[l % 4]][l // 4]
                    x = d.sensordata[self.native[l, p]:self.native[l, p] + 3] if vm == "pos" else d.site_xpos[si] if si is not None else d.xipos[sb]
                    V[l, p, j] = self._vel(d, sb, x) - self._vel(d, tb, x)
        t = round(float(d.time), 4)
        for l, p in zip(*np.nonzero(fo)):
            if self.blk[p]: self.block_touch.add((int(l), t))
        if self.native is not None: self._probe(m, d, fo)
        if self.log:   # sparse rows as fb_m5_check's Logger: k (substep of the bout), sensor, primary, found, F, then V per vmode
            for l, p in zip(*np.nonzero(fo)): self.rows["hit"].append([self.k, l, p, fo[l, p], *F[l, p], *[x for vm in self.vmodes for x in self.V[vm][l, p, j]]])
            for q in np.flatnonzero(np.any(Fb != 0, 1)): self.rows["body"].append([self.k, q, *Fb[q], *vb[q]])
        self.j += 1; self.k += 1
    def _probe(self, m, d, fo):
        """A4.3: every MuJoCo contact between a strike-limb secondary and a defender shell: the native sensor's pos within 1 cm of the
        contact's position; per found slot also the strongest contact (the HitMeter's point)."""
        s, n = self.s, d.ncon; pr = self.probe
        if not n: return
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]; best = {}
        for i in np.flatnonzero((s.is_tg[g1] & s.is_st[g2]) | (s.is_tg[g2] & s.is_st[g1])):
            a, b = int(g1[i]), int(g2[i]); t, st = (a, b) if s.is_tg[a] else (b, a); p, l = s.pix[t], s.limb[st]
            if p < 0 or l < 0: continue
            mujoco.mj_contactForce(m, d, int(i), self.f6); x = d.sensordata[self.native[l, p]:self.native[l, p] + 3]; e = float(np.linalg.norm(x - d.contact.pos[i]))
            pr["contacts"] += 1; pr["within_1cm"] += e <= 0.01; nz = abs(self.f6[0]) > 0; pr["contacts_nonzero"] += nz; pr["within_1cm_nonzero"] += nz and e <= 0.01
            if not np.all(x == 0): pr["max_err_nonzero"] = max(pr["max_err_nonzero"], e)
            if abs(self.f6[0]) >= best.get((l, p), (-1.0, 0))[0]: best[(l, p)] = (abs(self.f6[0]), e)
        for (l, p), (f, e) in best.items():
            pr["slots"] += 1; pr["slots_strongest_1cm"] += e <= 0.01; x = d.sensordata[self.native[l, p]:self.native[l, p] + 3]; pr["slots_pos_zero"] += bool(np.all(x == 0))
            pr["found_mismatch"] += int(d.sensordata[self.nfound[l, p]] != fo[l, p]); fa = self.nforce[l, p]; pr["force_mismatch"] += int(not np.array_equal(d.sensordata[fa:fa + 3], self.F[l, p, self.j]))

# ---- one bout
from r1gym.env import R1HitEnv as _E

def opp_modules(n, dt, c):
    """(Motion, TrackerDriver, AimOp, mirror policy or None) for n bouts: motion 'fb' as M6 built them; 'l1' FbMotion, the driver
    conjugated per bout (fb_population.driver) and SwapAim around the AimOp (A9.4)."""
    if c.get("motion", "fb") == "fb":
        M = FO.Motion(); return M, FO.TrackerDriver(n, dt, motion=M), FO.AimOp(n, dt, motion=M, slots=K_SLOTS, skip_idle=True), None
    M = POP.FbMotion(); drv, _, pol = POP.driver(n, M, dt)
    return M, drv, POP.SwapAim(FO.AimOp(n, dt, motion=M, slots=K_SLOTS, skip_idle=True), gen=torch.Generator().manual_seed(0)), pol

def opp_clip(M, pol, ids, ep):
    """The episode's clip index in M; on L1 (pol set) the bouts' conjugation mask = the row's npz conj (M8b contract)."""
    if pol is None: return FO.TRAIN.index(ep["clip"])
    ci = M.index(ep.get("row", ep["clip"])); assert M.host[ci] == ep["clip"], (ep, M.host[ci]); pol.set(ids, bool(M.conj[ci])); return ci

class Bout:
    """One bout at a time in a private MjData of the conf's model. Attribute names follow R1HitEnv so its outer loops, _raw_obs and DR
    draws run unchanged (borrowed below)."""
    FACE_GAIN, FACE_MAX, HOLD_VX, HOLD_VY = _E.FACE_GAIN, _E.FACE_MAX, _E.HOLD_VX, _E.HOLD_VY
    _raw_obs, _face, _hold, true_bearing, _dr_init, _dr_sample_into_model = _E._raw_obs, _E._face, _E._hold, _E.true_bearing, _E._dr_init, _E._dr_sample_into_model

    def __init__(self, ctrl, ccfg, c, m, native=None, seconds=20.0, lag=0.0, vmodes_report=(), acct_dtypes=None, log_vmodes=(), slot=None):
        """slot: a Batch's bout (its own MjData, rng, outer loops, sensors and meters; the torch modules and accounts are the Batch's;
        slot = the Batch's shared kinematics twin)."""
        from r1gym.model import R1Robot, HOME, action_scale, CTRL_JOINTS
        self.ctrl, self.cfg, self.c, self.m, self.native, self.seconds, self.lag = ctrl, ccfg, c, m, native, seconds, lag
        self.d = mujoco.MjData(m); self.dk = mujoco.MjData(m) if slot is None else slot; self.r = R1Robot(m, ccfg); cfg = ccfg
        self.nsub = int(round(cfg["model"]["control_dt"] / cfg["model"]["timestep"])); self.dt = cfg["model"]["control_dt"]
        n = cfg["sensors"]["noise"]; self.contract = cfg.get("contract", "residual"); assert self.contract == "mjlab"
        self.noise = np.concatenate([np.full(3, n["ang_vel"]), np.full(3, n["gravity"]), np.zeros(5), np.full(24, n["joint_pos"]), np.full(24, n["joint_vel"]), np.zeros(24)])
        self.home = np.array(cfg.get("home", HOME), float); self.scale = action_scale(cfg); self.clip = float(cfg.get("action_clip", 4.0)); self.alpha = cfg.get("action_filter", 0.3)
        assert self.alpha == 0.0, "mjlab contract: no action EMA (HybridDelayedAction has none)"
        self.command = np.zeros(3); self.face = bool(cfg.get("face_opponent", False)); self.perception = cfg.get("perception", "truth"); self.percepts = None
        self.face_band = cfg.get("face_deadband"); self.hold = bool(cfg.get("hold_position", False)); self.hold_band, self.hold_gain = cfg.get("hold_deadband", [0.25, 0.08]), float(cfg.get("hold_gain", 1.5))
        self.kpm = getattr(ctrl, "kp_mode", None); self.oi = OppIdx(m); self.replays = {}; self.torch = c["backend"] != "numpy"; self.shadow = None   # shadow: same-state reference hooks (checks)
        self.mode = cfg.get("opponent_aim", "none"); assert self.mode in ("standoff+armwarp_vz", "shadow"), self.mode
        if self.torch:
            dt = DT_T[c["backend"]]; self.tdt = dt
            if slot is None:
                self.M, self.drv, self.aim, self.pol = opp_modules(1, dt, c); self.lat = EV.Latency(1)
                self.obs = OB.EvalActorObs(1, self.lat, OB.noise_halfwidths(cfg), kp_dtype=dt, out_dtype=torch.float32 if dt == torch.float32 else torch.float64)
                self.act = AC.HybridDelayedAction(1, self.lat, self.home, self.scale, self.clip, dtype=dt); self.od = EV.OppDown(1)
            self.ids6 = [m.body(nm).id for nm in OB.body_names()]; self.ix = DR.DRIndex(m, CTRL_JOINTS); self.F = DR.model_fields(m)[0]; self.nom = NOMINAL[id(m)]
            from r1gym.aim import _gate_geoms
            self.g_opp, self.g_r1 = _gate_geoms(m)
        else: self._dr_init()
        self.acct, self.accts = None, {}
        if c["scorer"] == "account":   # the primary account (conf vmode, first dtype) and report-only ones on the same contacts
            vms = (c["vmode"],) + tuple(v for v in vmodes_report if v != c["vmode"]); self.sens = Sensors(m, vms, native); s = self.sens.s
            if slot is not None: return
            dts = tuple(acct_dtypes or (self.tdt if self.torch else torch.float64,))
            for vm in vms:
                for dt in dts: self.accts[(vm, str(dt).split(".")[-1])] = HT.FullBodyHitAccount(1, s.zone_idx, s.bzone_idx, slots=K_SLOTS, dtype=dt, log=[], block=s.block_idx)
            self.acct = self.accts[(c["vmode"], str(dts[0]).split(".")[-1])]
        elif native is not None or log_vmodes: self.sens = Sensors(m, log_vmodes, native, log=bool(log_vmodes))   # probe and / or sparse sensor rows (G0-11 replay)
        else: self.sens = None

    def replay(self, clip, dist):
        from r1gym.opponent import FightReplay
        k = (clip, dist)
        if k not in self.replays: self.replays[k] = FightReplay(self.m, clip, distance=dist, cfg=self.cfg)
        return self.replays[k]

    # ---- reset (R1HitEnv.reset's order; mjlab's B3 order for the training reset)
    def reset(self, ep):
        import eval_opponent as EO
        from r1gym.sensors import DelayedNoisyObs
        from r1gym.hits import HitMeter
        m, d, c = self.m, self.d, self.c; self.ep = ep; train = c["reset"] == "train"
        if train: g = self.gen = torch.Generator().manual_seed(int(ep["seed"])); EV.draw_latency(self.lat, [0], g)
        self.rng = np.random.default_rng(ep.get("env_seed", ep["seed"]))
        self.r.reset(m, d)
        if train:   # bout heading about the origin (the opponent is turned with it by TrackerDriver.reset)
            self.heading = float((2 * torch.rand(1, generator=g, dtype=torch.float64) - 1) * np.pi); d.qpos[self.r.free_q + 3:self.r.free_q + 7] = [np.cos(self.heading / 2), 0, 0, np.sin(self.heading / 2)]; mujoco.mj_forward(m, d)
        self.t = 0; self.last_a = np.zeros(24, np.float32); self.filt = np.zeros(24); self.k = int(self.lat.k[0]) if train else int(ep["delay"])
        if not self.torch: self.sensor = DelayedNoisyObs(self.k, self.noise, self.rng, uniform=True)
        if not self.torch: self._dr_sample_into_model()
        elif train: p = DR.sample(1, self.ix, generator=g); DR.write(self.F, self.nom, self.ix, p); self.act_delay = int(self.lat.act_delay[0])
        else: p, ad = DR.sample_np(self.rng, self.ix); DR.write(self.F, self.nom, self.ix, p); self.lat.k[0] = self.k; self.lat.act_delay[0] = ad; self.act_delay = int(ad)
        self.a_prev = np.zeros(24, np.float32); rp = self.replay(ep["clip"], ep["distance"]); self.opp_rp = rp
        if train:
            from r1gym.scenario import START_STEP
            M = self.M; j = int(np.flatnonzero(np.isclose(M.distances.numpy(), ep["distance"]))[0]); ci = opp_clip(M, self.pol, [0], ep); idx = np.flatnonzero(M.clean[ci, j].numpy())
            self.opp_t0 = round(float(idx[int(torch.randint(len(idx), (1,), generator=g))] * START_STEP), 3)
        else: self.opp_t0 = ep["t0"]
        if self.torch:
            ci = opp_clip(self.M, self.pol, [0], ep); o = self.oi
            qp, qv = self.drv.reset(None, ci, self.opp_t0, ep["distance"], *((torch.zeros(1, 2, dtype=torch.float64), self.heading) if train else ()))
            d.qpos[o.qadr:o.qadr + 31] = qp[0].numpy(); d.qvel[o.vadr:o.vadr + 30] = qv[0].numpy(); mujoco.mj_forward(m, d)
            self.aim.reset(None, ci, ep["distance"], aimed=self.mode != "shadow", jitter=bool(ep.get("jitter", False))); self.alog = {}
            self.od.reset([0]); self.obs.reset([0]); self.act.reset([0]); self.opp = o; self.opp_state = {}; self.obs.gen = self.gen if train else None
            if c["actor"] == "onnx": self.drv.mlp = self._onnx_actor()
            if c["ref"] == "eval": self.drv.reference = self._eval_ref
        else:
            rp.place(m, d, self.opp_t0); mujoco.mj_forward(m, d); self.opp = rp
            self.opp_state = {"last": np.zeros(24, np.float32), "r1_torso": self.r.torso}
            if self.mode != "none":
                from r1gym.aim import Aim
                self.opp_state["aim"] = Aim(m, rp, self.mode, self.cfg.get("aim"), self.r.torso)
        self.turning = False; self.returning = False; self.spot = d.qpos[self.r.free_q:self.r.free_q + 2].copy()
        if self.shadow: self.shadow.reset(self)
        self.hm = HitMeter(m, self.cfg.get("hit_metric")); self.tail = EO.ContactTail(m); self.hooks = [self.hm, self.tail] + ([self.sens] if self.sens is not None else [])
        for h in self.hooks: h.reset()
        for A in self.accts.values(): A.reset([0]); A.log.clear()
        if self.acct is not None: self.wvis = 0; self.ws = (np.full((1, K_SLOTS), -1, np.int64), np.zeros((1, K_SLOTS)), np.zeros((1, K_SLOTS)), np.zeros((1, K_SLOTS), np.int64))
        self._face(); self._hold()
        if not self.torch:
            o = self._raw_obs(); self.sensor.reset(o); out = self.sensor(o)
            if self.kpm == "obs": from r1gym.keypoints import KeypointSource; self.src = KeypointSource(m, "truth", delay=self.k); return np.concatenate([out, self.src.reset(d)])
            return np.concatenate([out, np.zeros(OB.KP_DIM, np.float32)]) if self.kpm == "zeros" else out
        return self._obs_t()

    def _onnx_actor(self):
        """Diagnostic actor: the eval's tracker session (r1gym Tracker, ONNX through the eval config) on the driver's obs cast to float32."""
        tr, drv, rp = self.opp_rp.tracker, self.drv, self.opp_rp; assert tr is not None
        def f(obs):
            fr = tr.frame(rp.clip, int(drv.fi_last)); a = tr.s.run(["actions"], {"obs": obs.double().numpy().astype(np.float32), "time_step": np.array([[fr]], np.float32)})[0]
            return torch.from_numpy(a).to(drv.dtype)
        return f

    def _eval_ref(self, fi):
        """Diagnostic: Tracker.reference (the ONNX initializers) at Tracker.frame(clip, fi), in the driver's dtype."""
        tr, rp, dt = self.opp_rp.tracker, self.opp_rp, self.drv.dtype; f = tr.frame(rp.clip, int(fi[0])); c = lambda x: torch.from_numpy(np.asarray(x)[None].copy()).to(dt)
        return c(tr.ref_q[f]), c(tr.ref_dq[f]), c(tr.ref_anchor_pos[f]), c(tr.ref_anchor[f])

    def _obs_t(self):
        """EvalActorObs on the fresh kinematics (mjlab's forward; KeypointSource's twin), noise: the eval's rng draws ('eo') or its own."""
        d, dk, r = self.d, self.dk, self.r; dt = self.tdt; T = lambda x: torch.from_numpy(np.asarray(x, np.float64)[None]).to(dt)
        p = OB.proprio83(T(d.qpos[r.free_q + 3:r.free_q + 7]), T(d.qvel[r.free_v + 3:r.free_v + 6]), T(d.qpos[r.qadr]), T(d.qvel[r.vadr]), self.home, T(self.command), torch.tensor([self.t]), self.act.applied)
        if self.kpm == "obs":
            dk.qpos[:] = d.qpos; mujoco.mj_kinematics(self.m, dk); P_T, y, v = OB.kp_state(T(dk.xpos[self.ids6]), T(dk.xquat[self.ids6]))
        else: P_T, y, v = torch.zeros(1, 5, 3, dtype=dt), torch.zeros(1, dtype=dt), torch.zeros(1, 5, dtype=torch.bool)
        nz = None
        if self.c["reset"] == "eo": nz = torch.zeros(1, OB.OBS_DIM, dtype=self.obs.out_dtype); nz[0, :83] = torch.from_numpy(self.rng.uniform(-1, 1, 83) * self.noise).to(self.obs.out_dtype)
        o = self.obs.compute(p, P_T, y, v, noise=nz if nz is not None else None, blind=self.kpm != "obs")
        if self.shadow: self.shadow.obs(self, p, o)
        o = o[0].double().numpy(); return o if self.kpm in ("obs", "zeros") else o[:83]

    # ---- one control step (R1HitEnv.step's order)
    def step(self, a):
        m, d, r = self.m, self.d, self.r
        if self.torch:
            tg = self.act.process(torch.as_tensor(np.asarray(a, np.float32)[None])); d.ctrl[r.act] = tg[0].double().numpy()
            if self.shadow: self.shadow.act(self, a)
        else:
            a = np.clip(a, -self.clip, self.clip).astype(np.float32)
            if self.act_delay: a, self.a_prev = self.a_prev, a
            self.filt = self.alpha * self.filt + (1 - self.alpha) * a; d.ctrl[r.act] = self.home + self.scale * self.filt
        mujoco.mj_kinematics(m, d)
        if self.torch: self._opp_t()
        else:
            if self.cfg.get("opponent_face", True): self.opp_state["target"] = d.xpos[r.pel][:2].copy()
            self.opp.tracker.act(m, d, self.opp, self.opp_state, d.time + self.opp_t0 + self.dt, self.dt)
        if self.sens is not None: self.sens.begin()
        for _ in range(self.nsub):
            d.xfrc_applied[:] = 0; mujoco.mj_step(m, d)
            for h in self.hooks: h.substep(m, d)
        d.xfrc_applied[:] = 0; self.t += 1
        if self.torch: fallen = bool(EV.fell_eval(torch.tensor(d.xpos[r.pel][2]), EV.up_z(torch.from_numpy(d.xquat[r.pel].copy()))))
        else: fallen = bool(r.fallen(d))
        self.last_a = a if not self.torch else self.act.applied[0].numpy()
        if self.acct is not None: self._score()
        self._face(); self._hold()
        if self.torch: return self._obs_t(), fallen
        o = self.sensor(self._raw_obs())
        if self.kpm == "obs": return np.concatenate([o, self.src.observe(d)]), fallen
        return (np.concatenate([o, np.zeros(OB.KP_DIM, np.float32)]) if self.kpm == "zeros" else o), fallen

    def _touching(self):
        d = self.d
        if d.ncon == 0: return False
        g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]; return bool(np.any((self.g_opp[g1] & self.g_r1[g2]) | (self.g_opp[g2] & self.g_r1[g1])))

    def _opp_t(self):
        """TrackerDriver + AimOp on the kinematics just computed (Tracker.act's inputs), targets into d.ctrl."""
        d, o, r, dt = self.d, self.oi, self.r, self.tdt; t = d.time + self.opp_t0 + self.dt; f64 = lambda x: torch.from_numpy(np.asarray(x, np.float64).copy()[None])
        if self.shadow: self.shadow.opp_before(self, t)
        fi = self.drv.clock(t); self.drv.fi_last = int(fi[0]); rq, rdq, _, _ = self.drv.reference(fi)
        rq, rdq, dist = self.aim.step(fi, torch.tensor([d.time + self.dt], dtype=torch.float64), rq, rdq, f64(d.xpos[o.torso]), f64(d.xmat[o.torso].reshape(3, 3)), f64(d.xpos[r.torso]),
                                      f64(d.xmat[r.torso].reshape(3, 3)), torch.tensor([self._touching()]), q_act=f64(d.qpos[o.qadr + 7:o.qadr + 31]))
        tg, info = self.drv.step(t, f64(d.xpos[r.pel][:2]), f64(d.xpos[o.pel][:2]), f64(d.xpos[o.torso]), f64(d.xquat[o.torso]), f64(d.qvel[o.vadr + 3:o.vadr + 6]),
                                 f64(d.qpos[o.qadr + 7:o.qadr + 31]), f64(d.qvel[o.vadr + 6:o.vadr + 30]), torso_mat=f64(d.xmat[o.torso].reshape(3, 3)), ref=(rq, rdq), dist=dist)
        d.ctrl[o.act] = tg[0].double().numpy(); self.opp_info = info; self._aim_log()
        if self.shadow: self.shadow.opp_after(self, rq, rdq, dist, tg, info)

    def _aim_log(self):
        """Snapshot of the aim's live intent slots by sequence number (the 8-slot ring wraps over a bout)."""
        A, K = self.aim, self.aim.K
        for k in range(K):
            if not bool(A.valid[0, k]): continue
            s = int(A.seq[0, k]); self.alog[s] = dict(limb=FO.LIMBS[int(A.limb[0, k])], t_on=float(A.t_on[0, k]), t_apex=float(A.t_apex[0, k]), t_end=float(A.t_end[0, k]),
                                                      gated=("", "tilt", "touch")[int(A.gated[0, k])], closest_m=float(A.closest[0, k]), slot=k)
        assert int(A.overflow[0]) == 0, "aim slot overflow"

    def intents(self):
        """The bout's intents in the order made, as meter_row reads them."""
        if self.torch: return [_It(self.alog[s]) for s in sorted(self.alog)]
        return self.opp_state["aim"].intents if "aim" in self.opp_state else []

    def _win(self):
        """Window slots for the account: id, t_on, t_apex, side (AimOp's ring, or Aim.intents placed as AimOp would)."""
        K = K_SLOTS; sid, son, sap, ssd = np.full((1, K), -1, np.int64), np.zeros((1, K)), np.zeros((1, K)), np.zeros((1, K), np.int64); t_end = self.d.time
        if self.torch:
            A = self.aim
            for k in range(K):
                if bool(A.valid[0, k]): sid[0, k], son[0, k], sap[0, k], ssd[0, k] = int(A.seq[0, k]), float(A.t_on[0, k]), float(A.t_apex[0, k]), int(A.limb[0, k])
        else:   # Aim.intents in the order made, slot k % K once visible (fb_m5_check.replay's placement)
            its = self.intents(); sid, son, sap, ssd = self.ws
            while self.wvis < len(its) and its[self.wvis].t_on - LEAD_S <= t_end + 1e-9:
                k = self.wvis; it = its[k]; sl = k % K; sid[0, sl], son[0, sl], sap[0, sl], ssd[0, sl] = k, it.t_on, it.t_apex, HT.SIDES.index(it.limb[0]); self.wvis += 1
        return dict(id=torch.from_numpy(sid), on=torch.from_numpy(son), apex=torch.from_numpy(sap), side=torch.from_numpy(ssd))

    def _score(self):
        S, w, t = self.sens, self._win(), self.d.time
        for (vm, _), A in self.accts.items():
            dt = A.dtype; tt = lambda x: torch.from_numpy(x[None]).to(dt)
            A.process(tt(S.F), torch.tensor([t], dtype=dt), v_rel=tt(S.V[vm]), found=tt(S.fo), win=dict(w, on=w["on"].to(dt), apex=w["apex"].to(dt)), body=(tt(S.Fb), tt(S.vb)))

    def acct_summary(self, secs, both_up, A=None):
        """An account's per-bout counts (closed episodes after close_envs), as HitMeter.summary's n / n_both_up; its log."""
        A = A or self.acct; A.close_envs([0]); E = [x for x in A.log if x["kind"] == "episode" and x["landed"]]; W = [x for x in A.log if x["kind"] == "window"]
        cl = [x for x in E if x["clean"]]; inb = lambda t: any(a + 1e-9 < t <= b + 1e-9 for a, b in both_up)
        n = dict(head_landed=sum(x["zone"] == "head" for x in E), head_clean=sum(x["zone"] == "head" for x in cl), torso_landed=sum(x["zone"] == "torso" for x in E),
                 torso_clean=sum(x["zone"] == "torso" for x in cl))
        return dict(secs=round(secs, 3), n=n, n_both_up={f"{z}_clean": sum(x["zone"] == z and inb(x["t0"]) for x in cl) for z in ("head", "torso")},
                    windows={o: sum(x["outcome"] == o for x in W) for o in HT.OUTCOMES}, paid=sum(x["paid"] for x in W), log=list(A.log))

# ---- a run: EO.run's per-bout loop and row, one bout at a time
_W, NOMINAL = {}, {}
def world(name, c, p1=None, sensors=False):
    """(controller, cfg, lag, model, native sensor names) per process, cached per (controller, conf, sensors)."""
    import fb_r0eo as R0
    k = (name, conf_key(c), sensors)
    if k not in _W:
        from types import SimpleNamespace
        from r1gym.model import R1Robot
        ctrl, cfg, _, lag = R0.cell(name, p1 or R0.P1); m, names = build_model(cfg, c, sensors); _W[k] = (ctrl, cfg, lag, m, names)
        NOMINAL[id(m)] = DR.model_fields(m)[1]; _E._dr_init(SimpleNamespace(m=m, r=R1Robot(m, cfg)))   # nominal DR fields, before any bout writes them
    return _W[k]

class _Book:
    """EO.run's per-bout bookkeeping (displacement, spacing, falls, opponent fall log, both-up and down runs) after each step."""
    def __init__(self, B): q = B.r.free_q; self.start = B.d.qpos[q:q + 2].copy(); self.maxr, self.sp, self.fell, self.at, self.opp_fell, self.ofall, self.bu, self.down = 0.0, [], None, None, False, None, [], []
    def step(self, B, fallen, dn=None):
        """dn: OppDown's flag this step (torch backends), asserted equal to EO:86."""
        import eval_opponent as EO
        d = B.d; q = B.r.free_q; xy = d.qpos[q:q + 2]
        self.maxr = max(self.maxr, float(np.linalg.norm(xy - self.start))); self.sp.append(float(np.linalg.norm(d.qpos[B.oi.qadr:B.oi.qadr + 2] - xy)))
        b = B.oi.pel; tl, ht = bool(d.xmat[b][8] < EO.OPP_TILT), bool(d.xpos[b][2] < EO.OPP_Z); assert dn is None or dn == (tl or ht)
        if tl or ht:
            self.opp_fell = True
            if self.ofall is None: self.ofall = dict(t_s=round(B.t * B.dt, 3), cause="tilt+height" if tl and ht else "tilt" if tl else "height", up_z=round(float(d.xmat[b][8]), 3),
                                                     pelvis_z=round(float(d.xpos[b][2]), 3), r1_contact_0p5s=B.tail.snap(float(d.time))); B.tail.done = True
            if self.down and self.down[-1][1] == B.t - 1: self.down[-1][1] = B.t
            else: self.down.append([B.t, B.t])
        else:
            if self.bu and self.bu[-1][1] == B.t - 1: self.bu[-1][1] = B.t
            else: self.bu.append([B.t, B.t])
        if fallen: self.fell = B.t * B.dt; self.at = xy.copy()

def _finish(B, ep, k, acct=None):
    """The bout's EO-format row and info from its bookkeeping k; acct(secs, both_up, A) -> an account's summary (B.acct_summary)."""
    import eval_opponent as EO
    q, T = B.r.free_q, B.seconds; p = (k.at if k.at is not None else B.d.qpos[q:q + 2]) - k.start; secs = k.fell or T; dt = B.dt; biv = [((k0 - 1) * dt, k1 * dt) for k0, k1 in k.bu]
    sh = type("E", (), {})(); sh.substep_hooks, sh.opp, sh.opp_state = [B.hm], B.opp_rp, {"aim": _AimShim(B.intents()) if B.torch else B.opp_state.get("aim")}
    row = dict(clip=ep["clip"], distance=ep["distance"], fell_s=k.fell, opp_fell=k.opp_fell, drift_m=round(float(np.linalg.norm(p)), 3), back_m=round(float(-p[0]), 3),
               max_disp_m=round(k.maxr, 3), ring_out=k.maxr > EO.RING_M, hit_meter=EO.meter_row(sh, ep, secs, k.sp, B.lag, biv), opp_fall=k.ofall,
               both_up_s=round(sum(k1 - k0 + 1 for k0, k1 in k.bu) * dt, 3), both_up_iv=[[round(a, 3), round(c, 3)] for a, c in biv])
    info = dict(act_delay=int(B.act_delay), dt=float(dt), down=k.down, t0=B.opp_t0, k=B.k)
    if acct is not None: info["acct"], info["accts"] = acct(secs, biv)
    return row, info

def run_bout(B, ep, probe=None):
    """One bout through B (Bout) -> (EO-format row, probe info)."""
    o = B.reset(ep); k = _Book(B)
    if probe: probe(B, "reset", None)
    for _ in range(int(B.seconds / B.dt)):
        a = B.ctrl.act(o[None])[0]; o, fallen = B.step(a); d, b = B.d, B.oi.pel; dn = None
        if B.torch: B.od.update(torch.tensor([d.xpos[b][2]]), EV.up_z(torch.from_numpy(d.xquat[b].copy()))[None]); dn = bool(B.od.now[0])
        k.step(B, fallen, dn)
        if probe: probe(B, "step", a)
        if k.fell is not None: break
    acct = None if B.acct is None else lambda secs, biv: (B.acct_summary(secs, biv), {f"{vm}/{dt}": B.acct_summary(secs, biv, A) for (vm, dt), A in B.accts.items() if A is not B.acct})
    row, info = _finish(B, ep, k, acct)
    if probe: probe(B, "end", None)
    return row, info

# ---- K bouts in lockstep (A7.1)
def _atan2_rows(y, x): return torch.cat([torch.atan2(y[i:i + 1], x[i:i + 1]) for i in range(len(y))])

def rowwise_face(drv):
    """TrackerDriver.face (fb_opponent, operation for operation: want = bearing - pi, yaw += wrap(want - yaw) x min(1, dt / FACE_TAU)) with its
    atan2 calls one row at a time, as at batch 1 (tests: equal to the module's face at n = 1)."""
    def face(target, pel_xy):
        to = target - pel_xy; want = _atan2_rows(to[:, 1], to[:, 0]) - np.pi; y0 = torch.where(drv.yaw_set, drv.yaw, want); x = want - y0
        drv.yaw = y0 + _atan2_rows(torch.sin(x), torch.cos(x)) * min(1.0, drv.dt / FO.FACE_TAU); drv.yaw_set[:] = True; return drv.yaw
    return face

class Batch:
    """K bouts in lockstep in one process, torch backends (A7.1): each bout is a Bout slot (its own MjData, rng, R1HitEnv outer loops,
    sensors, HitMeter, ContactTail, bookkeeping); the torch modules (TrackerDriver, AimOp, EvalActorObs, HybridDelayedAction, OppDown,
    fell_eval, the accounts) run once per step on the batch, as on the GPU; the controller acts on the live bouts' stacked obs (as
    EO.run). The bouts share the model: each writes its DR draws into it before its own MuJoCo calls (EO's _dr_apply), and one
    kinematics twin serves the keypoints (xpos / xquat do not read the DR fields). A finished bout keeps its frozen state in the batch
    (its outputs unread, its account closed and summarised at its end). Rows equal run_bout's in form; bitwise equality with N = 1 is
    reported (fb_m6b_check.py), not assumed. rowwise: the batch-variant ops one row at a time, the tracker MLP (float32 sgemm at batch K
    differs from batch 1 in every call) and the facing's float32 atan2 (vectorised at batch K, scalar at batch 1: 1-ulp differences),
    which fb_m6b_check found to be the only sources (rowwise_face)."""
    def __init__(self, ctrl, ccfg, c, m, k, native=None, seconds=20.0, lag=0.0, rowwise=False):
        assert c["backend"] != "numpy" and c["actor"] == "mlp" and c["ref"] == "fb", "Batch runs the deployed torch modules"
        self.ctrl, self.cfg, self.c, self.m, self.K, self.seconds, self.lag, self.rowwise = ctrl, ccfg, c, m, k, seconds, lag, rowwise
        self.dk = mujoco.MjData(m); self.S = [Bout(ctrl, ccfg, c, m, native, seconds, lag, slot=self.dk) for _ in range(k)]; B = self.S[0]
        self.tdt = dt = B.tdt; self.mode = B.mode; self.kpm = B.kpm; rp = {}
        for s in self.S: s.replays = rp
        self.mods = {}; self.tm = {}; self.probe = None   # tm: wall s per phase; probe(i, slot, action) after each live step

    def _modules(self, n):
        """The torch modules for n bouts (cached per n)."""
        if n not in self.mods:
            dt, B = self.tdt, self.S[0]; lat = EV.Latency(n); M, drv, aim, pol = opp_modules(n, dt, self.c)
            if self.rowwise: mlp = drv.mlp; drv.mlp = lambda x: torch.cat([mlp(x[i:i + 1]) for i in range(len(x))]); drv.face = rowwise_face(drv)
            obs = OB.EvalActorObs(n, lat, OB.noise_halfwidths(self.cfg), kp_dtype=dt, out_dtype=torch.float32 if dt == torch.float32 else torch.float64)
            act = AC.HybridDelayedAction(n, lat, B.home, B.scale, B.clip, dtype=dt); accts = {}
            if self.c["scorer"] == "account":
                s = B.sens.s; accts[(self.c["vmode"], str(dt).split(".")[-1])] = HT.FullBodyHitAccount(n, s.zone_idx, s.bzone_idx, slots=K_SLOTS, dtype=dt, log=[], block=s.block_idx)
            self.mods[n] = dict(M=M, lat=lat, drv=drv, aim=aim, pol=pol, obs=obs, act=act, od=EV.OppDown(n), accts=accts)
        self.__dict__.update(self.mods[n]); self.acct = next(iter(self.accts.values()), None)

    @staticmethod
    def _dr(B):
        """The bout's DR fields into the shared model (EO's _dr_apply)."""
        for k, v in B.drp.items(): getattr(B.m, k)[:] = v

    def _reset(self, i, B, ep):
        """Bout.reset's torch path for bout i (module rows i)."""
        m, d, c = B.m, B.d, B.c; B.ep = ep; train = c["reset"] == "train"
        if train: g = B.gen = torch.Generator().manual_seed(int(ep["seed"])); EV.draw_latency(self.lat, [i], g)
        B.rng = np.random.default_rng(ep.get("env_seed", ep["seed"]))
        B.r.reset(m, d)
        if train: B.heading = float((2 * torch.rand(1, generator=g, dtype=torch.float64) - 1) * np.pi); d.qpos[B.r.free_q + 3:B.r.free_q + 7] = [np.cos(B.heading / 2), 0, 0, np.sin(B.heading / 2)]; mujoco.mj_forward(m, d)
        B.t = 0; B.k = int(self.lat.k[i]) if train else int(ep["delay"])
        if train: p = DR.sample(1, B.ix, generator=g); DR.write(B.F, B.nom, B.ix, p); B.act_delay = int(self.lat.act_delay[i])
        else: p, ad = DR.sample_np(B.rng, B.ix); DR.write(B.F, B.nom, B.ix, p); self.lat.k[i] = B.k; self.lat.act_delay[i] = ad; B.act_delay = int(ad)
        B.drp = {k: getattr(m, k).copy() for k in DR.FIELDS}; rp = B.replay(ep["clip"], ep["distance"]); B.opp_rp = rp
        if train:
            from r1gym.scenario import START_STEP
            M = self.M; j = int(np.flatnonzero(np.isclose(M.distances.numpy(), ep["distance"]))[0]); ci = opp_clip(M, self.pol, [i], ep); idx = np.flatnonzero(M.clean[ci, j].numpy())
            B.opp_t0 = round(float(idx[int(torch.randint(len(idx), (1,), generator=g))] * START_STEP), 3)
        else: B.opp_t0 = ep["t0"]
        ci = opp_clip(self.M, self.pol, [i], ep); o = B.oi
        qp, qv = self.drv.reset([i], ci, B.opp_t0, ep["distance"], *((torch.zeros(1, 2, dtype=torch.float64), B.heading) if train else ()))
        d.qpos[o.qadr:o.qadr + 31] = qp[0].numpy(); d.qvel[o.vadr:o.vadr + 30] = qv[0].numpy(); mujoco.mj_forward(m, d)
        self.aim.reset([i], ci, ep["distance"], aimed=self.mode != "shadow", jitter=bool(ep.get("jitter", False))); B.alog = {}
        self.od.reset([i]); self.obs.reset([i]); self.act.reset([i]); B.opp = o; B.opp_state = {}
        B.turning = False; B.returning = False; B.spot = d.qpos[B.r.free_q:B.r.free_q + 2].copy()
        import eval_opponent as EO
        from r1gym.hits import HitMeter
        B.hm = HitMeter(m, self.cfg.get("hit_metric")); B.tail = EO.ContactTail(m); B.hooks = [B.hm, B.tail] + ([B.sens] if B.sens is not None else [])
        for h in B.hooks: h.reset()
        B._face(); B._hold()

    def _obs(self, live):
        """Bout._obs_t for the batch: proprio83, keypoints (the shared twin), the eval's rng noise ('eo') or the bout generator's, one
        EvalActorObs.compute; -> the live bouts' obs rows."""
        S, dt, n = self.S, self.tdt, self.n; T = lambda x: torch.from_numpy(np.asarray(x, np.float64)).to(dt); B0 = S[0]; f, v = B0.r.free_q, B0.r.free_v
        p = OB.proprio83(T([S[i].d.qpos[f + 3:f + 7] for i in range(n)]), T([S[i].d.qvel[v + 3:v + 6] for i in range(n)]), T([S[i].d.qpos[S[i].r.qadr] for i in range(n)]),
                         T([S[i].d.qvel[S[i].r.vadr] for i in range(n)]), B0.home, T([S[i].command for i in range(n)]), torch.tensor([S[i].t for i in range(n)]), self.act.applied)
        if self.kpm == "obs":
            X, Q, dk = np.zeros((n, 6, 3)), np.zeros((n, 6, 4)), self.dk
            for i in range(n): dk.qpos[:] = S[i].d.qpos; mujoco.mj_kinematics(self.m, dk); X[i], Q[i] = dk.xpos[B0.ids6], dk.xquat[B0.ids6]
            P_T, y, vv = OB.kp_state(T(X), T(Q))
        else: P_T, y, vv = torch.zeros(n, 5, 3, dtype=dt), torch.zeros(n, dtype=dt), torch.zeros(n, 5, dtype=torch.bool)
        od = self.obs.out_dtype; nz = torch.zeros(n, OB.OBS_DIM, dtype=od)
        for i in live:
            B = S[i]
            if self.c["reset"] == "eo": nz[i, :83] = torch.from_numpy(B.rng.uniform(-1, 1, 83) * B.noise).to(od)
            else: nz[i] = (2 * torch.rand(1, OB.OBS_DIM, dtype=od, generator=B.gen) - 1)[0] * self.obs.noise
        o = self.obs.compute(p, P_T, y, vv, noise=nz, blind=self.kpm != "obs").double().numpy()
        return {i: o[i] if self.kpm in ("obs", "zeros") else o[i, :83] for i in live}

    def _opp(self, live):
        """Bout._opp_t for the batch: TrackerDriver + AimOp on the kinematics just computed, targets into each live bout's d.ctrl."""
        S, n = self.S, self.n; st = lambda f: torch.from_numpy(np.stack([f(S[i]) for i in range(n)]).astype(np.float64))
        t = np.array([S[i].d.time + S[i].opp_t0 + S[i].dt for i in range(n)]); fi = self.drv.clock(t); rq, rdq, _, _ = self.drv.reference(fi)
        o, r = S[0].oi, S[0].r
        rq, rdq, dist = self.aim.step(fi, torch.tensor([S[i].d.time + S[i].dt for i in range(n)], dtype=torch.float64), rq, rdq, st(lambda B: B.d.xpos[o.torso]), st(lambda B: B.d.xmat[o.torso].reshape(3, 3)),
                                      st(lambda B: B.d.xpos[r.torso]), st(lambda B: B.d.xmat[r.torso].reshape(3, 3)), torch.tensor([S[i]._touching() for i in range(n)]), q_act=st(lambda B: B.d.qpos[o.qadr + 7:o.qadr + 31]))
        tg, info = self.drv.step(t, st(lambda B: B.d.xpos[r.pel][:2]), st(lambda B: B.d.xpos[o.pel][:2]), st(lambda B: B.d.xpos[o.torso]), st(lambda B: B.d.xquat[o.torso]), st(lambda B: B.d.qvel[o.vadr + 3:o.vadr + 6]),
                                 st(lambda B: B.d.qpos[o.qadr + 7:o.qadr + 31]), st(lambda B: B.d.qvel[o.vadr + 6:o.vadr + 30]), torso_mat=st(lambda B: B.d.xmat[o.torso].reshape(3, 3)), ref=(rq, rdq), dist=dist)
        tg = tg.double().numpy(); A = self.aim; V = A.valid[:, :A.K].numpy(); assert not A.overflow[list(live)].any(), "aim slot overflow"
        for i in live: S[i].d.ctrl[o.act] = tg[i]
        ix = [(i, k) for i, k in zip(*np.nonzero(V)) if i in live]
        if ix:   # Bout._aim_log for the live bouts: the live intent slots by sequence number
            I, Kk = (np.array(x) for x in zip(*ix)); g = lambda x: x[I, Kk].tolist(); seq, lb, on, ap, en, ga, cl = g(A.seq), g(A.limb), g(A.t_on), g(A.t_apex), g(A.t_end), g(A.gated), g(A.closest.double())
            for j, (i, k) in enumerate(ix): S[i].alog[seq[j]] = dict(limb=FO.LIMBS[lb[j]], t_on=on[j], t_apex=ap[j], t_end=en[j], gated=("", "tilt", "touch")[ga[j]], closest_m=cl[j], slot=int(k))

    def _score(self, live):
        S, n, A = self.S, self.n, self.aim; K = A.K; v = A.valid[:, :K]
        w = dict(id=torch.where(v, A.seq[:, :K], -1), on=torch.where(v, A.t_on[:, :K], 0.0), apex=torch.where(v, A.t_apex[:, :K], 0.0), side=torch.where(v, A.limb[:, :K], 0))
        Z = lambda x: np.zeros_like(x); t = [S[i].d.time for i in range(n)]
        for (vm, _), Ac in self.accts.items():
            dt = Ac.dtype; st = lambda f: torch.from_numpy(np.stack([f(S[i].sens) if i in live else Z(f(S[i].sens)) for i in range(n)])).to(dt)
            Ac.process(st(lambda s: s.F), torch.tensor(t, dtype=dt), v_rel=st(lambda s: s.V[vm]), found=st(lambda s: s.fo), win=dict(w, on=w["on"].to(dt), apex=w["apex"].to(dt)),
                       body=(st(lambda s: s.Fb), st(lambda s: s.vb)))

    def _summary(self, i, secs, biv):
        """Bout.acct_summary for bout i: close its episodes and windows now, its log entries only."""
        out = []
        for Ac in self.accts.values():
            Ac.close_envs([i]); L = [x for x in Ac.log if x["env"] == i]; E = [x for x in L if x["kind"] == "episode" and x["landed"]]; W = [x for x in L if x["kind"] == "window"]
            cl = [x for x in E if x["clean"]]; inb = lambda t: any(a + 1e-9 < t <= b + 1e-9 for a, b in biv)
            out.append(dict(secs=round(secs, 3), n=dict(head_landed=sum(x["zone"] == "head" for x in E), head_clean=sum(x["zone"] == "head" for x in cl), torso_landed=sum(x["zone"] == "torso" for x in E),
                                                        torso_clean=sum(x["zone"] == "torso" for x in cl)),
                            n_both_up={f"{z}_clean": sum(x["zone"] == z and inb(x["t0"]) for x in cl) for z in ("head", "torso")},
                            windows={o: sum(x["outcome"] == o for x in W) for o in HT.OUTCOMES}, paid=sum(x["paid"] for x in W), log=[dict(x, env=0) for x in L]))
        return out[0], {}

    def run(self, eps):
        """[(row, info)] of eps in order, K at a time in lockstep (torch inference mode: no autograd bookkeeping, the same kernels)."""
        out = []
        with torch.inference_mode():
            for j in range(0, len(eps), self.K): out += self._chunk(eps[j:j + self.K])
        return out

    def _chunk(self, eps):
        n = self.n = len(eps); self._modules(n); S = self.S
        for Ac in self.accts.values(): Ac.reset(list(range(n))); Ac.log.clear()
        for i, ep in enumerate(eps): self._reset(i, S[i], ep)
        live = list(range(n)); bk = [_Book(S[i]) for i in range(n)]; res = [None] * n; o = self._obs(live); f, b = S[0].r.free_q, S[0].oi.pel
        tm = self.tm; c = [time.perf_counter()]
        def tick(k): t = time.perf_counter(); tm[k] = tm.get(k, 0.0) + t - c[0]; c[0] = t
        tick("reset")
        for _ in range(int(self.seconds / S[0].dt)):
            if not live: break
            A = dict(zip(live, self.ctrl.act(np.stack([o[i] for i in live])))); Af = torch.zeros(n, 24); tick("ctrl")
            for i in live: Af[i] = torch.as_tensor(np.asarray(A[i], np.float32))
            tg = self.act.process(Af).double().numpy()
            for i in live: B = S[i]; self._dr(B); B.d.ctrl[B.r.act] = tg[i]; mujoco.mj_kinematics(B.m, B.d)
            tick("act_kin"); self._opp(live); tick("opp")
            for i in live:
                B = S[i]; d = B.d; self._dr(B)
                if B.sens is not None: B.sens.begin()
                for _ in range(B.nsub):
                    d.xfrc_applied[:] = 0; mujoco.mj_step(B.m, d)
                    for h in B.hooks: h.substep(B.m, d)
                d.xfrc_applied[:] = 0; B.t += 1
            tick("physics_hooks"); P = lambda x: torch.from_numpy(np.stack([x(S[i].d) for i in range(n)]).copy())
            fallen = EV.fell_eval(torch.tensor([S[i].d.xpos[S[i].r.pel][2] for i in range(n)]), EV.up_z(P(lambda d: d.xquat[S[0].r.pel]))).tolist()
            if self.acct is not None: self._score(live)
            tick("score")
            for i in live: S[i]._face(); S[i]._hold()
            o = self._obs(live); tick("obs")
            self.od.update(torch.tensor([S[i].d.xpos[b][2] for i in range(n)]), EV.up_z(P(lambda d: d.xquat[b]))); dn = self.od.now.tolist()
            for i in list(live):
                bk[i].step(S[i], fallen[i], dn[i])
                if self.probe: self.probe(i, S[i], A[i])
                if bk[i].fell is not None: res[i] = self._end(i, eps[i], bk[i]); live.remove(i)
            tick("book")
        for i in live: res[i] = self._end(i, eps[i], bk[i])
        tick("end"); return res

    def _end(self, i, ep, k):
        return _finish(self.S[i], ep, k, None if self.acct is None else lambda secs, biv: self._summary(i, secs, biv))

def rep_row(ep, row, info, T, c):
    """fb_r0eo.bout_row (asserts the down runs against the row's both-up intervals) with the replica's own t0 / k (training reset), the
    account's counts as the primary n when it scores, and the HitMeter's kept as 'hm'."""
    import fb_r0eo as R0
    ep2 = dict(ep, t0=info["t0"], delay=info["k"]); eo = row
    if "acct" in info:
        A = info["acct"]; eo = dict(row, hit_meter=dict(row["hit_meter"], n=dict(row["hit_meter"]["n"], **A["n"]), n_both_up=A["n_both_up"]))
    out = R0.bout_row(ep2, eo, info, T); out["eo"] = row; out["conf"] = conf_key(c)
    if "acct" in info: out["acct"] = {k: v for k, v in info["acct"].items() if k != "log"}; out["hm"] = dict(n=row["hit_meter"]["n"], n_both_up=row["hit_meter"]["n_both_up"])
    return out

_BATCH = {}
def run(name, eps, c, seconds=20.0, p1=None, sensors=False, probe=None, keep_info=False, batch=1, rowwise=False):
    """Rows (fb_r0eo format) of controller `name` on eps under conf c, one bout at a time (batch 1) or `batch` bouts in lockstep
    (Batch, torch backends; cached per process)."""
    torch.set_num_threads(1); sensors = sensors or (c["scorer"] == "account" and c["vmode"] == "pos"); ctrl, cfg, lag, m, names = world(name, c, p1, sensors); out = []
    if batch > 1:
        assert probe is None, "probes run at batch 1"; k = (name, conf_key(c), sensors, batch, seconds, rowwise)
        if k not in _BATCH: _BATCH.clear(); _BATCH[k] = Batch(ctrl, cfg, c, m, batch, native=names, seconds=seconds, lag=lag, rowwise=rowwise)
        res = _BATCH[k].run(eps)
    else: B = Bout(ctrl, cfg, c, m, native=names, seconds=seconds, lag=lag); res = (run_bout(B, ep, probe) for ep in eps)
    for ep, (row, info) in zip(eps, res):
        r = rep_row(ep, row, info, seconds, c)
        if keep_info: r["_info"] = info
        out.append(r)
    return out

def episodes(role, seeds):
    """The episodes a role's seeds name: 'identity' / 'hit_scoring' as fb_record (seed s -> clip TRAIN[(s - 60000) % 3], distance
    HARD[((s - 60000) // 3) % 4]); 'eq' / 'r2_diag' / 'reserve' fb_r0eo's construction of their blocks; checked through fb_seeds."""
    import fb_r0eo as R0, fb_record as REC
    from fb_seeds import check
    seeds = [int(s) for s in seeds]; check(seeds, role, set(R0.TRAIN))   # refused before any episode is built
    if role in ("identity", "hit_scoring"): eps = REC.episodes(seeds, 20.0)
    else:
        blk = dict(eq=(70000, 1024, R0.HARD), r2_diag=(67000, 64, R0.HARD), reserve=(100000, 1024, R0.HARD))[role]; allx = {e["seed"]: e for e in R0.episodes(*blk)}; eps = [allx[s] for s in seeds]
    check(seeds, role, {e["clip"] for e in eps}); return eps

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ctrl", default="G"); ap.add_argument("--role", default="identity"); ap.add_argument("--seeds", default="60100-60101")
    ap.add_argument("--conf", default="{}", help="JSON overrides of EVAL"); ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--out")
    a = ap.parse_args(); lo, hi = (int(x) for x in (a.seeds.split("-") + [a.seeds])[:2]); c = conf(**json.loads(a.conf)); t0 = time.time()
    rows = run(a.ctrl, episodes(a.role, range(lo, hi + 1)), c, a.seconds)
    print(json.dumps(dict(conf=c, n=len(rows), falls=sum(not r["survived"] for r in rows), kd=sum(r["kd"] for r in rows), head_clean=sum(r["n"]["head_clean"] for r in rows), wall_s=round(time.time() - t0, 1))))
    if a.out: Path(a.out).write_text(json.dumps(rows), encoding="utf-8")
