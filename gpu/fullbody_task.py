"""Full-body M2 v2 mjlab task (M2V2_PLAN_v3 2.1-2.5, section 3 M9; A4-A9): the R1 defender against the eval's own full-body opponent,
a second R1 driven by the frozen tracker (T0; T1 on the reangle rows), aimed by the eval's standoff+armwarp_vz, on mjlab 1.2.0.

Scene (2.1)  entities `robot` (the defender) then `opponent`, both get_r1_robot_cfg() (actuators 100/40/100/40/20, kd 2/2/2/2/1, caps
             60/50/60/60/33, armature 0.01, the eval's, C1). Capsules (CollisionCfg): defender 0/1, opponent 2/2, feet condim 3, priority
             1, mu 0.6, others condim 1; opponent capsules solref / solimp OPP_* (r1gym.model). Shells: every non-collision mesh geom of
             both robots, named '<mesh>_shell' in the entity's own spec (sensors resolve names before the scene's spec_fn), reduced to
             the hull set HULLS (fb_hulls.npz 'mix' | 'all256' | 'full', the 3.2 fallback list; FB_HULLS) in the entity spec too; their
             bits in the scene's spec_fn (scene_edit): defender 0/4, opponent 4/0, condim 1, SHELL solref / solimp; floor `terrain`
             conaffinity 1 -> 3 (m9: mjlab compiles it 1/1, C1; asserted on mjlab's compiled model by scripts/fb_m9_check.py).
             Sim options SIM (B4: 5 ms, implicitfast, pyramidal, impratio 1, ccd_iterations 35); solver (iterations, ls) from FB_SOLVER
             (M6's choice from SOLVER_FALLBACK); njmax 600, nconmax 128.
             Sensors (mjlab ContactSensorCfg, found / force, netforce, 1 slot, history 4): fb_hit_<side>_<limb> (8: the defender's 32
             shells x the opponent's strike-limb bodies, fb_hits.LIMB_SECONDARY), fb_body (the defender's torso / pelvis / leg shells x
             the opponent's pelvis subtree: tangle, W), fb_rev_<side>_<limb> (the defender striking, logged).
Opponent     command 'opp' (FbOppCommand, a CommandTerm with resampling_time_range (1e9, 1e9): placement only from command reset,
             B3): fb_population's FbMotion (originals + L1 rows), TrackerDriver with PolicyMux (T0 / T1) conjugated per env by fb_mirror
             (mask = the row's npz conj at reset, the row's npz reference fed: M8b), AimOp wrapped by SwapAim (A9.4), OppDown,
             FullBodyHitAccount, FbRewards. _update_command runs after the step's single forward (the eval's order: the defender's
             target written, kinematics, the tracker acts, 4 substeps). The population row, latency and DR are drawn by reset events
             before any manager reset; placement reads the defender root from qpos (written by the reset event), never derived data.
             First-ever reset: env.reset() runs no command compute, so the actor obs term calls command_manager.compute(dt=0.0) once
             when it finds placed but undriven envs (only right after env.reset()).
Defender     actor group: one class term FbActorObs = fb_obs.EvalActorObs (111: the warm start's 83 then the 28-d keypoint block of
             the opponent's real links), computed once per control step (the observation manager's single compute; rsl_rl's
             get_observations reads the cache); blind: the 28 zeros after the delay (the ring keeps the true block for the critic).
             Action term FbAction = fb_actions.HybridDelayedAction (one-step motor delay per episode, no EMA, targets home + scale x
             applied with R1HitEnv's home / scale). Twist command FbTwistCommand: the eval's outer loops on every env (face the
             opponent's pelvis bearing, gain 2.0, cap 0.8; hold the start spot, 0.25 / 0.08, gain 1.5), computed after the forward,
             so the reset obs carries the real reset command.
Order (M4_VERIFY 4, M5_VERIFY)  reset events: fb_draw_latency first (no min_step_count), fb_reset_defender, fb_draw_population,
             fb_dr (defender, fb_dr.write into mjlab's per-world model fields, no set_const; DRIndex with explicit prefix / is_defender;
             then the t1_dr rows' opponent DR); termination phase: fb_update first (one per step: OppDown.update on the one-substep-stale
             quantities, the account on the 4 substeps' sensors and velocity ring, FbRewards.step), then fell_eval (R1.fallen), ring_out
             (training), nan, time_out; rewards read fb_update's values; the critic reads FbActorObs.clean / kp_clean of the same step.
             The eval's reads after the last mj_step are kept: the aim's touch gate (d.contact) and _face's opponent pelvis (xpos) come
             from that substep (fb_update stores them), fresh only for envs placed this step (R1HitEnv.reset's mj_forward). The
             account's touch is the sensors' found per substep (sensordata recorded by the hook; mjlab keeps no found history).
Closing speed (A4.3 / A6.6)  the palm / foot site velocities (vmode 'site'): per substep the opponent entity's update hook records the
             strike limbs' site (hand, foot) or body-COM (forearm, shin) points and velocities and the defender primary links' COM
             velocity at those points.
Critic       run 3's 138 inputs in order (the base 98, the clean keypoint block, 10 intent features in the strike_phase slot, ring_home)
             then PRIV (29): the opponent root relative to the defender, opp_down_now and its age, act_delay, k, the DR scalars, tangle
             impulse, W. Zero-initialised columns: start_checkpoint().
Policy       FbActorModel: the rsl_rl model factory (class_name 'fullbody_task:FbActorModel'): HybridActor.from_checkpoints (A0: nothing
             trains; A: arms; B: to_stage_b + residual) on the task's actor group; `blind_flag` is in every checkpoint (A9.2). res_l2 is a
             reward term reading the actor's residual from the forward that produced the action (one process).
WATCH6       counters emitted at every reset batch under Metrics/opp/<WATCH_KEYS> (sums since the last emission, so a window's ratio of
             means is the ratio of totals) and the actor's std / leg_dev / residual; read by scripts/fb_watch.py --rule 6.
Tasks        R1-Fb-A / -B, -Blind, -LOCO (P3: kfab_203 rows removed); stages A0 / A / B by FB_STAGE in the actor cfg (A0 and A share
             the env). Run from a unitree_rl_mjlab checkout: python fullbody_task.py R1-Fb-A [train.py flags]; gpu/pod_fb.sh drives it.
Pure parts (no mjlab import): the spec edits (entity_spec, scene_edit), shell tables, intent features, critic privileged block,
watch keys; the mjlab glue lives in _classes() and the cfg builders (imports inside functions, as sparring_task.py)."""
import os, sys, json, math, hashlib, functools
from pathlib import Path
import numpy as np, mujoco, torch

GPU = Path(__file__).resolve().parent; ROOT = GPU.parent
for p in (GPU, ROOT, ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import fb_opponent as FO, fb_obs as OB, fb_actions as AC, fb_events as EV, fb_dr as DR, fb_hits as HT, fb_rewards as RW, fb_population as POP

ENT, OPP = "robot", "opponent"; PFX = {ENT: "robot/", OPP: "opponent/"}
SIM = dict(timestep=0.005, integrator="implicitfast", cone="pyramidal", impratio=1.0, ccd_iterations=35)
SOLVER_FALLBACK, HULL_FALLBACK = ((10, 20), (25, 30), (50, 50), (100, 50)), ("mix", "all256", "full")   # 3.2, in order
NJMAX, NCONMAX, DECIM, EP_S, CTRL_DT, DEF_Z = 600, 128, 4, 20.0, 0.02, 0.74
SHELL_BIT, SHELL_SOLREF, SHELL_SOLIMP = 4, (0.01, 1.0), (0.95, 0.99, 0.001, 0.5, 2.0)        # r1gym.model SHELL_*
OPP_SOLREF, OPP_SOLIMP = (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)                           # r1gym.model OPP_*
FOOT = r"^(left|right)_foot[1-7]_collision$"; FLOOR = "terrain"; FLOOR_BITS = (1, 3)
HULLS_NPZ = GPU / "motions" / "fb" / "fb_hulls.npz"
RING = dict(free=0.25, out=1.0)
FACE_GAIN, FACE_MAX, HOLD_BAND, HOLD_GAIN, HOLD_VX, HOLD_VY = 2.0, 0.8, (0.25, 0.08), 1.5, (-0.5, 0.8), 0.5   # R1HitEnv, gym.json
PUSH = dict(vmax=0.3, interval_s=(3.0, 7.0))     # T1 rows: velocity pushes <= 0.3 m/s at half stage 2's rate (1.5-3.5 s), m5
DEF_HEAD_T = (0.01, 0.0, 0.357)                  # the defender's head point in its torso frame (clearance)
SITE_OF = dict(hand="palm", forearm=None, foot="foot", shin=None)   # vmode 'site' (fb_replica.SITE_OF)
INTENT_DIM = 10
PRIV = ("opp_rel_x", "opp_rel_y", "opp_rel_z", "opp_rel_yaw_s", "opp_rel_yaw_c", "opp_vx", "opp_vy", "opp_vz", "opp_down_now", "opp_down_age",
        "act_delay", "k0", "k1", "k2", "dr_mass", "dr_torso_kg", "dr_com_0", "dr_com_1", "dr_com_2", "dr_com_3", "dr_com_4", "dr_com_5", "dr_foot_mu",
        "dr_kp", "dr_kd", "dr_cap", "dr_armature", "tangle", "work_w")
WATCH_KEYS = ("w_live_s", "w_live_s_np", "w_term_np", "w_kd_np", "w_head_clean", "w_eps", "w_t0", "w_ad", "w_k0", "w_k1", "w_k2", "w_switch",
              "w_cmd_bad", "w_opp_nc_fall", "w_lean_deg_s", "w_tangle", "w_down_s", "w_down_app", "w_down_con", "w_manifest_ok")
ACTOR_KEYS = ("fb_arm_std", "fb_leg_std", "fb_leg_dev_u", "fb_res_abs_u")   # metrics terms (std; leg_dev and |residual| in micro units)
STAGES = ("A0", "A", "B")

def solver_cfg(): return tuple(int(x) for x in os.environ.get("FB_SOLVER", "%d,%d" % SOLVER_FALLBACK[0]).split(","))
def hulls_cfg(): h = os.environ.get("FB_HULLS", HULL_FALLBACK[0]); assert h in HULL_FALLBACK, h; return h

# =============================================================================================== pure spec edits
def shell_name(mesh): return f"{mesh}_shell"

def hull_points(kind, spec):
    """{mesh: body-frame points} of the reduced hulls ('mix': fb_hulls.npz; 'all256': 256 farthest-point samples of the full hull, as
    fb_replica.hull_points), from an R1 spec's own meshes (compiled once)."""
    with np.load(HULLS_NPZ, allow_pickle=False) as z: names, n, P = z["meshes"].tolist(), z["n"], z["points"]
    if kind == "mix": return {k: P[i, :n[i]] for i, k in enumerate(names)}
    import export_fb_hulls as XH
    from export_attacks import fps_sample
    c = spec.copy()
    for g in c.geoms:   # MuJoCo builds the hull graph only for colliding meshes (the eval's shells collide)
        if g.type == mujoco.mjtGeom.mjGEOM_MESH: g.contype, g.conaffinity = SHELL_BIT, 0
    m = c.compile(); G = {m.mesh(m.geom_dataid[g]).name: g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and not m.geom(g).name.endswith("_collision")}; out = {}
    for k in names: V, _ = XH.hull(m, G[k]); out[k] = V if len(V) <= XH.N_FINE else fps_sample(V, XH.N_FINE)
    return out

def entity_spec(base, hulls="mix"):
    """An R1 entity spec (base(): unitree's get_spec) with every non-collision mesh geom named '<mesh>_shell' and, for hulls != 'full',
    its mesh replaced by the reduced hull (body-frame points, geom pose identity, as fb_replica.build_model) under the original mesh
    name, the original renamed fborig_<mesh> (A10: the zone maps key on mesh names)."""
    s = base(); P = None if hulls == "full" else hull_points(hulls, s)
    for g in list(s.geoms):
        if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.name.endswith("_collision"): continue
        mesh = g.meshname; g.name = shell_name(mesh)
        if P is not None:   # A10: the hull takes the original mesh name (zone maps key on it), the original becomes fborig_<mesh>
            s.mesh(mesh).name = f"fborig_{mesh}"; s.add_mesh(name=mesh, uservert=np.asarray(P[mesh], np.float64).ravel().tolist())
            g.meshname = mesh; g.pos = [0.0, 0.0, 0.0]; g.quat = [1.0, 0.0, 0.0, 0.0]
    return s

def scene_edit(spec, prev=None):
    """SceneCfg.spec_fn: the shells' bits (defender 0/4, opponent 4/0; condim 1, SHELL solref / solimp), the opponent capsules' OPP
    solref / solimp, and the floor's conaffinity 3 (m9)."""
    if prev is not None: prev(spec)
    fl = [g for g in spec.geoms if g.name == FLOOR]; assert len(fl) == 1 and fl[0].contype == FLOOR_BITS[0], "mjlab's plane terrain"
    fl[0].conaffinity = FLOOR_BITS[1]
    for g in spec.geoms:
        for ent, bits in ((ENT, (0, SHELL_BIT)), (OPP, (SHELL_BIT, 0))):
            if not g.name.startswith(PFX[ent]): continue
            if g.name.endswith("_shell"): g.contype, g.conaffinity = bits; g.condim = 1; g.solref = list(SHELL_SOLREF); g.solimp = list(SHELL_SOLIMP)
            elif ent == OPP and (g.contype or g.conaffinity): g.solref = list(OPP_SOLREF); g.solimp = list(OPP_SOLIMP)

def collision_kwargs(ent):
    """CollisionCfg fields of an entity's capsules (r1gym.model.build_spec: defender FULL_COLLISION_WITHOUT_SELF's bits 0/1 with r1.xml's
    priority 1 kept, M9 model diff; the tracked opponent 2/2, priority 0 but the feet)."""
    ct, ca = (0, 1) if ent == ENT else (2, 2)
    pr = 1 if ent == ENT else {FOOT: 1}   # the eval keeps r1.xml's priority 1 on every defender capsule; the tracked opponent: feet 1, rest 0
    return dict(geom_names_expr=(".*_collision",), contype=ct, conaffinity=ca, condim={FOOT: 3, ".*_collision": 1}, priority=pr, friction={FOOT: (0.6,)})

def strip(name, ent): p = PFX[ent]; return name[len(p):] if name.startswith(p) else name

def shell_tables(m, target=ENT):
    """From a compiled scene model: the target entity's shells in geom order (mjlab's primary order) with zone (fb_hits.ZONES index),
    block flag (BLOCK_LINKS on arm), link body id; and the body-sensor subset (BODY_ZONES)."""
    G = [g for g in range(m.ngeom) if m.geom(g).name.startswith(PFX[target]) and m.geom(g).name.endswith("_shell")]
    link = [strip(m.body(m.geom_bodyid[g]).name, target) for g in G]; mesh = [strip(m.geom(g).name, target)[:-len("_shell")] for g in G]
    zone = [HT.ZONES.index(HT.zone_of(l, x)) for l, x in zip(link, mesh)]
    block = [HT.ZONES[z] == "arm" and any(k in l for k in HT.BLOCK_LINKS) for z, l in zip(zone, link)]
    keep = [i for i, z in enumerate(zone) if HT.ZONES[z] in HT.BODY_ZONES]
    return dict(geoms=G, names=[m.geom(g).name for g in G], snames=[strip(m.geom(g).name, target) for g in G], zone=zone, block=block, body=[int(m.geom_bodyid[g]) for g in G], link=link,
                b_geoms=[G[i] for i in keep], b_names=[strip(m.geom(G[i]).name, target) for i in keep], b_zone=[zone[i] for i in keep], b_body=[int(m.geom_bodyid[G[i]]) for i in keep])

def sensor_adr(m, sensors, field="found"):
    """[[sensordata address]] (one row per mjlab ContactSensor, its primaries in slot order) of `field` (dim 1, num_slots 1)."""
    return [[int(m.sensor(s.sensor_name).adr[0]) for s in c._slots if s.field_name == field] for c in sensors]

def limb_bodies(striker=OPP):
    """The 8 hit sensors' secondaries (fb_hits.SENSORS order): (name, mode, body pattern) per (side, limb)."""
    out = []
    for s in HT.SIDES:
        for l in HT.LIMBS:
            kind, b = HT.LIMB_SECONDARY[l]; out.append((f"{s}_{l}", kind, f"{'left' if s == 'l' else 'right'}_{b}"))
    return out

# =============================================================================================== pure: critic features
def intent_features(aim, swap, now):
    """[N, 10]: per hand (l, r) of its highest-weight live intent: live, arm weight, time to apex, gated, swapped (A9.4); times clipped
    to [-0.5, 1.0] s. aim: AimOp (slots valid / limb / t_apex / gated), swap [N, K + 1] bool, now [N] float64."""
    w = aim.weight(now); live = aim.valid & (now[:, None] <= aim.t_end + aim.p["arm"]["tail_s"] + aim.p["window_tail_s"]); out = []
    for lb in (0, 1):
        m = live & (aim.limb == lb); wl = torch.where(m, w + 1e-3, torch.full_like(w, -1.0)); j = wl.argmax(1, keepdim=True); g = lambda x: x.gather(1, j)[:, 0]
        on = m.any(1); out += [on, torch.where(on, g(w), 0 * now), torch.where(on, (g(aim.t_apex) - now).clamp(-0.5, 1.0), 0 * now), on & (g(aim.gated) > 0), on & g(swap)]
    return torch.stack([x.to(torch.float32) for x in out], 1)

def ring_home(spot, pel_xy, yaw):
    """Run 2's critic ring_home: the start xy minus the pelvis xy in the pelvis heading frame."""
    e = spot - pel_xy; c, s = torch.cos(yaw), torch.sin(yaw); return torch.stack([c * e[:, 0] + s * e[:, 1], -s * e[:, 0] + c * e[:, 1]], -1)

def priv_block(def_xy, def_yaw, opp_pos, opp_yaw, opp_vel, down_now, down_age_s, act_delay, k, dr, tangle, work):
    """The PRIV block [N, 29] (float32): the opponent root in the defender's heading frame (x, y, z rel, yaw sin / cos, velocity), the
    down flag and age (s, clipped 5), act_delay, k one-hot, DR scalars (mean mass scale, torso kg, pelvis / torso COM offsets, foot mu,
    mean kp / kd / cap / armature scales), this step's tangle impulse and the 1 s defender work W."""
    c, s = torch.cos(def_yaw), torch.sin(def_yaw); d = opp_pos[:, :2] - def_xy; ry = opp_yaw - def_yaw
    rot = lambda v: torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], -1)
    kk = torch.nn.functional.one_hot(k.long(), 3).to(torch.float32)
    drs = torch.stack([dr["mass"].mean(-1), dr["torso_kg"], *dr["com_m"].reshape(len(c), 6).unbind(-1), dr["foot_mu"], dr["kp"].mean(-1), dr["kd"].mean(-1), dr["cap"].mean(-1), dr["armature"].mean(-1)], 1)
    x = [rot(d), opp_pos[:, 2:3], torch.sin(ry)[:, None], torch.cos(ry)[:, None], rot(opp_vel[:, :2]), opp_vel[:, 2:3], down_now[:, None].float(), down_age_s.clamp(max=5.0)[:, None],
         act_delay[:, None].float(), kk, drs, tangle[:, None], work[:, None]]
    out = torch.cat([v.to(torch.float32) for v in x], 1); assert out.shape[1] == len(PRIV); return out

# =============================================================================================== actor registry (one process)
_ACTOR = {}
def actor_stats():
    """The live actor's stats for the metrics terms and res_l2 (zeros before an actor exists)."""
    a = _ACTOR.get("a"); return None if a is None else a

# =============================================================================================== mjlab glue
_C, _E = {}, {}
def _ent_classes():
    """FbEntity(Cfg) (mjlab.entity only: the scene alone compiles with warp stubbed, fb_m9_check)."""
    if _E: return _E
    from dataclasses import dataclass
    from mjlab.entity import Entity, EntityCfg

    class FbEntity(Entity):
        """An entity with a per-substep hook after mj_step (scene.update): the opponent's records the closing-speed ring."""
        hook = None
        def update(self, dt):
            super().update(dt)
            if self.hook is not None: self.hook()

    @dataclass
    class FbEntityCfg(EntityCfg):
        def build(self): return FbEntity(self)
    _E.update(FbEntity=FbEntity, FbEntityCfg=FbEntityCfg); return _E

def _classes():
    """mjlab classes, built once on first use (mjlab imported only here)."""
    if _C: return _C
    from dataclasses import dataclass
    from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
    from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
    from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

    class FbOppCommand(CommandTerm):
        """The opponent, its aim, the hit account, attribution and rewards (module docstring)."""
        def __init__(self, cfg, env):
            super().__init__(cfg, env); n, dev = self.num_envs, self.device; FO.pin_precision(); self.dt = env.step_dt
            self.robot, self.opp = env.scene[ENT], env.scene[OPP]; m = env.sim.mj_model; J = list(FO_CTRL())
            self.M = POP.FbMotion(loco=cfg.loco, device=dev); self.pop = POP.Population(self.M, t1_ok=bool(cfg.t1_pt))
            self.drv, self.mux, self.pol = POP.driver(n, self.M, torch.float32, dev, cfg.t1_pt or None)
            self.aim = POP.SwapAim(FO.AimOp(n, torch.float32, dev, motion=self.M, slots=cfg.slots), gen=torch.Generator(device=dev).manual_seed(cfg.seed))
            self.od = EV.OppDown(n, dev); self.ojid = torch.as_tensor(self.opp.find_joints(J, preserve_order=True)[0], device=dev)
            self.rjid = torch.as_tensor(self.robot.find_joints(J, preserve_order=True)[0], device=dev)
            b = lambda e, nm: e.find_bodies(nm)[0][0]
            self.r_pel, self.r_torso, self.o_pel, self.o_torso = b(self.robot, "pelvis"), b(self.robot, "torso_link"), b(self.opp, "pelvis"), b(self.opp, "torso_link")
            T = shell_tables(m, ENT); self.T = T; R = shell_tables(m, OPP)
            names = lambda nm: list(dict.fromkeys(s.primary_name for s in env.scene[nm]._slots))   # entity-relative names, geom order (one slot per field)
            for k, _, _ in limb_bodies(): assert names(f"fb_hit_{k}") == T["snames"] and names(f"fb_rev_{k}") == R["snames"], (k, "primary order")
            assert names("fb_body") == T["b_names"]
            self.acct = HT.FullBodyHitAccount(n, T["zone"], T["b_zone"], slots=cfg.slots, dtype=torch.float32, device=dev, block=T["block"])
            self.racct = HT.FullBodyHitAccount(n, R["zone"], None, dtype=torch.float32, device=dev, block=R["block"])
            self.rw = RW.FbRewards(n, dtype=torch.float32, device=dev, down=RW.DownNeutral(device=dev))
            ht = torch.as_tensor([HT.ZONES[z] in ("head", "torso") for z in T["zone"]], device=dev); self.ht = ht
            # closing-speed ring (vmode 'site'): limb points / velocities and the defender primaries' COM state, per substep
            ob = {nm: self.opp.find_bodies(nm)[0][0] for nm in ("left_wrist_roll_link", "right_wrist_roll_link", "left_elbow_link", "right_elbow_link", "left_knee_link", "right_knee_link")}
            os_ = {nm: self.opp.find_sites(nm)[0][0] for nm in ("left_palm", "right_palm", "left_foot", "right_foot")}
            self.limb_pt = []
            for s in HT.SIDES:
                side = "left" if s == "l" else "right"
                for l in HT.LIMBS:
                    st = SITE_OF[l]; self.limb_pt.append(("site", os_[f"{side}_{st}"]) if st else ("body", ob[f"{side}_{HT.LIMB_COM[l]}"]))
            rb = {int(x): i for i, x in enumerate(self.robot.indexing.body_ids.tolist())}
            self.p_body = torch.as_tensor([rb[x] for x in T["body"]], device=dev); self.b_body = torch.as_tensor([rb[x] for x in T["b_body"]], device=dev)
            P, Pb = len(T["geoms"]), len(T["b_geoms"])
            self.vring = torch.zeros(n, 8, P, DECIM, 3, device=dev); self.bring = torch.zeros(n, Pb, DECIM, 3, device=dev); self.j = 0
            # found per substep (mjlab keeps no found history): the account's touch is found > 0 as the replica / HitMeter count it (a
            # zero-net-force contact still touches: 11% of found slots, M6a); fadr / radr: the hit / reverse sensors' found addresses
            sens = lambda p: [env.scene[f"{p}_{k}"] for k, _, _ in limb_bodies()]
            self.fadr, self.radr = (torch.as_tensor(sensor_adr(m, sens(p)), device=dev) for p in ("fb_hit", "fb_rev"))
            self.fring = torch.zeros(n, 8, P, DECIM, device=dev); self.rfring = torch.zeros(n, 8, len(R["geoms"]), DECIM, device=dev)
            # the eval's staleness after the last mj_step (R1HitEnv.step): the aim's touch gate reads d.contact and _face the opponent
            # pelvis xpos of the last substep; fresh (post-forward) only for envs placed this step (R1HitEnv.reset's mj_forward)
            self.touch = torch.zeros(n, dtype=torch.bool, device=dev); self.o_pel_xy = torch.zeros(n, 2, device=dev); self.placed = torch.ones(n, dtype=torch.bool, device=dev)
            self.opp.hook = self._substep
            self.t_push = torch.full((n,), -1e9, device=dev); self.undriven = False   # placed since the last _update_command (a host flag: no device sync)
            self.row = torch.zeros(n, dtype=torch.long, device=dev); self.pushable = torch.zeros(n, dtype=torch.bool, device=dev)
            self.rv = {}; self.edge = torch.zeros(n, dtype=torch.bool, device=dev); self._upd = -1
            self.nstep = 0   # control steps of this process (never restored from a checkpoint: 'iterations 1-100 of every run', res_l2's B iteration)
            self.W = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in WATCH_KEYS}; self.W["w_manifest_ok"] += float(manifest_ok(cfg.loco))
            self.head_p = torch.tensor(DEF_HEAD_T, device=dev)
        @property
        def command(self): return intent_features(self.aim.aim, self.aim.swap, self._now())
        def _t_ep(self): return self._env.episode_length_buf.to(torch.float64) * self.dt
        def _now(self): return self._t_ep() + self.dt
        def _update_metrics(self): pass
        # ---- per substep (after mj_step): the 'site' closing-speed inputs
        def _substep(self):
            rd, od = self.robot.data, self.opp.data; j = self.j % DECIM
            X = torch.stack([od.site_pos_w[:, i] if k == "site" else od.body_com_pos_w[:, i] for k, i in self.limb_pt], 1)          # (N, 8, 3)
            V = torch.stack([od.site_lin_vel_w[:, i] if k == "site" else od.body_com_lin_vel_w[:, i] for k, i in self.limb_pt], 1)
            c, v, w = rd.body_com_pos_w[:, self.p_body], rd.body_com_lin_vel_w[:, self.p_body], rd.body_com_ang_vel_w[:, self.p_body]   # (N, P, 3)
            vt = v[:, None] + torch.cross(w[:, None].expand(-1, 8, -1, -1), X[:, :, None] - c[:, None], dim=-1)
            self.vring[:, :, :, j] = V[:, :, None] - vt; self.bring[:, :, j] = rd.body_com_lin_vel_w[:, self.b_body]
            sd = self._env.sim.data.sensordata; self.fring[..., j] = sd[:, self.fadr]; self.rfring[..., j] = sd[:, self.radr]; self.j += 1   # this mj_step's sensors
        # ---- placement (command reset, after the reset events)
        def _resample_command(self, env_ids):
            e, ids = self._env, env_ids; d = {k: v[ids.cpu()].to(self.device) for k, v in e.fb_pop.items()}
            q = e.sim.data.qpos[ids][:, self.robot.indexing.free_joint_q_adr]; w, x, y, z = q[:, 3:7].unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).double()
            qp, qv = self.drv.reset(ids, d["clip"], d["t0"].double(), d["distance"].double(), q[:, :2].double(), yaw)
            R = FO.quat2mat(qp[:, 3:7]); wv = torch.einsum("nij,nj->ni", R, qv[:, 3:6])
            self.opp.write_root_state_to_sim(torch.cat([qp[:, :7], qv[:, :3], wv], 1).float(), env_ids=ids)
            self.opp.write_joint_state_to_sim(qp[:, 7:].float(), qv[:, 6:].float(), joint_ids=self.ojid, env_ids=ids)
            self.aim.reset(ids, d["clip"], d["distance"].float(), aimed=d["aimed"], jitter=d["jitter"]); self.mux.sel[ids] = d["t1"]; self.pol.set(ids, d["conj"])
            self.od.reset(ids); self.acct.reset(ids); self.racct.reset(ids); self.rw.reset(ids); self.row[ids] = d["row"]; self.pushable[ids] = d["push"]
            self.t_push[ids] = -1e9; self.undriven = True; self.placed[ids] = True
            W = self.W; W["w_eps"] += len(ids); W["w_t0"] += (d["row"] < 2).sum(); W["w_ad"] += e.fb_latency.act_delay[ids].sum()
            for i in range(3): W[f"w_k{i}"] += (e.fb_latency.k[ids] == i).sum()
        def reset(self, env_ids):
            if isinstance(env_ids, torch.Tensor) and len(env_ids):
                self.acct.close_envs(env_ids); self.W["w_cmd_bad"] += (self.command_counter[env_ids] != 1).sum()
            out = super().reset(env_ids); out.update(zip(self.W, torch.stack(list(self.W.values())).tolist()))   # one host sync per reset batch
            ok = self.W["w_manifest_ok"].clone()
            for v in self.W.values(): v.zero_()
            self.W["w_manifest_ok"] += ok; return out
        # ---- after the forward: the tracker acts on this state (targets for the next 4 substeps)
        def _update_command(self):
            rd, od = self.robot.data, self.opp.data; t = self._t_ep() + self.drv.t0 + self.dt; now = self._now()
            ot, oR = od.body_link_pos_w[:, self.o_torso], FO.quat2mat(od.body_link_quat_w[:, self.o_torso]); rt, rR = rd.body_link_pos_w[:, self.r_torso], FO.quat2mat(rd.body_link_quat_w[:, self.r_torso])
            fi = self.drv.clock(t); rq, rdq, _, _ = self.drv.reference(fi)
            touch = torch.where(self.placed, self._touching(), self.touch); qa = od.joint_pos[:, self.ojid]
            rq, rdq, dist = self.aim.step(fi, now, rq, rdq, ot, oR, rt, rR, touch, q_act=None)
            tg, _ = self.drv.step(t, rd.body_link_pos_w[:, self.r_pel, :2], od.body_link_pos_w[:, self.o_pel, :2], ot, od.body_link_quat_w[:, self.o_torso],
                                  od.root_link_ang_vel_b, qa, od.joint_vel[:, self.ojid], torso_mat=oR, ref=(rq, rdq), dist=dist)
            self.opp.set_joint_position_target(tg.float(), joint_ids=self.ojid); self.undriven = False
        def _touching(self, found=None):
            """A strike limb's sensor finds the defender's head or torso shells (r1gym.aim._gate_geoms): found (N, 8, P), default the
            sensordata now (after the step's forward: the placed envs' reset state)."""
            f = self._env.sim.data.sensordata[:, self.fadr] if found is None else found
            return f.gt(0).logical_and(self.ht[None, None]).flatten(1).any(1)
        # ---- termination phase (once per step): OppDown, the account, attribution and the reward values
        def update(self):
            e = self._env
            if self._upd == e.common_step_counter: return
            self._upd = e.common_step_counter; self.nstep += 1; rd, od = self.robot.data, self.opp.data; n = self.num_envs; t_end = self._t_ep()
            H = lambda nm: e.scene[nm].data.force_history.flip(2)                  # (N, P, S, 3), oldest first
            F = torch.stack([H(f"fb_hit_{k}") for k, _, _ in limb_bodies()], 1); Fb = H("fb_body")
            rot = (self.j - DECIM) % DECIM; R = lambda x, d: x.roll(-rot, d) if rot else x; vr, br, fo, rfo = R(self.vring, 3), R(self.bring, 2), R(self.fring, 3), R(self.rfring, 3)
            A = self.aim.aim; v = A.valid[:, :A.K]; win = dict(id=torch.where(v, A.seq[:, :A.K], -1), on=A.t_on[:, :A.K].float(), apex=A.t_apex[:, :A.K].float(), side=A.limb[:, :A.K])
            self.acct.process(F, t_end.float(), v_rel=vr, found=fo, win=win, body=(Fb, br))
            Fr = torch.stack([H(f"fb_rev_{k}") for k, _, _ in limb_bodies()], 1); self.racct.process(Fr, t_end.float(), found=rfo)
            self.touch = self._touching(fo[..., -1]); self.o_pel_xy = od.body_link_pos_w[:, self.o_pel, :2].clone(); self.placed[:] = False   # the eval's stale reads (__init__)
            pq = od.body_link_quat_w[:, self.o_pel]; self.edge = self.od.update(od.body_link_pos_w[:, self.o_pel, 2], EV.up_z(pq))
            tq = FO.quat2mat(rd.body_link_quat_w[:, self.r_torso]); up = tq[:, :, 2]
            res = None
            a = actor_stats()
            if a is not None and a.stage == "B" and a.r is not None and a.r.shape[0] == n: res = a.r.detach().to(self.device)
            clear = self._clearance(t_end)
            self.rv = self.rw.step(self.acct, self.edge, self.od.now, t_end.float(), self.row, t_push=self.t_push, v_pel=rd.root_link_lin_vel_w,
                                   def_xy=rd.body_link_pos_w[:, self.r_pel, :2], opp_xy=od.body_link_pos_w[:, self.o_pel, :2], torso_up=up, residual=res, clear=clear)
            if res is None: self.rv["res_l2"] = torch.zeros(n, device=self.device)
            # WATCH6 counters (device tensors, read at the reset batch; no-push envs: never pushed this episode)
            W = self.W; dt = self.dt; np_ = self.t_push < -1e8; dn = self.od.now.float(); s = lambda x: x.sum().double()
            W["w_live_s"] += dt * n; W["w_live_s_np"] += dt * s(np_); W["w_kd_np"] += s(self.edge & np_); W["w_head_clean"] += s(self.acct.step["head_clean"])
            W["w_opp_nc_fall"] += s(self.edge & (self.rw.attr.W() <= 0)); W["w_tangle"] += s(self.acct.step["tangle"])
            W["w_down_s"] += dt * s(dn); W["w_down_app"] += dt * s(self.rv.get("fb_down_approach", 0 * dn)); W["w_down_con"] += s(dn * self.acct.step["tangle"])
            if "fb_lean_away" in self.rv: W["w_lean_deg_s"] += dt * s(RW.lean_away_deg(up, rd.body_link_pos_w[:, self.r_pel, :2], od.body_link_pos_w[:, self.o_pel, :2]))
            sw = (e.fb_obs_term.obs.k_switches + e.action_manager.get_term("joint_pos").hda.switches).double(); W["w_switch"] += sw - getattr(self, "_sw0", 0.0 * sw); self._sw0 = sw
        def _clearance(self, t_end):
            A = self.aim.aim; rd, od = self.robot.data, self.opp.data; now = t_end.double()
            x_T, R_T = rd.body_link_pos_w[:, self.r_torso], FO.quat2mat(rd.body_link_quat_w[:, self.r_torso]); hp = x_T + torch.einsum("nij,j->ni", R_T, self.head_p)
            palm = torch.stack([od.site_pos_w[:, self.limb_pt[0][1]], od.site_pos_w[:, self.limb_pt[4][1]]], 1)   # l / r palm
            pl = palm.gather(1, A.limb[..., None].expand(-1, -1, 3).clamp(max=1)); d = torch.linalg.norm(pl - hp[:, None], dim=-1)
            win = A.valid & (A.gated == 0) & A.aimed[:, None] & (now[:, None] >= A.t_on - 0.15) & (now[:, None] <= A.t_apex + 0.15)
            return RW.clearance_value(d, win.float())
        def note_terminations(self, fell, ring_out):
            np_ = self.t_push < -1e8; self.W["w_term_np"] += float(((fell | ring_out) & np_).sum())

    @dataclass(kw_only=True)
    class FbOppCommandCfg(CommandTermCfg):
        loco: bool = False; t1_pt: str = ""; slots: int = 8; seed: int = 0
        def build(self, env): return FbOppCommand(self, env)

    class FbTwistCommand(UniformVelocityCommand):
        """The eval's outer loops on every env (R1HitEnv._face / _hold with gym.json's face_opponent / hold_position, no face
        deadband): after the forward, from the defender root and the opponent's pelvis."""
        def __init__(self, cfg, env):
            super().__init__(cfg, env); n = self.num_envs; self.spot = torch.zeros(n, 2, device=self.device); self.returning = torch.zeros(n, dtype=torch.bool, device=self.device)
        def reset(self, env_ids):
            out = super().reset(env_ids)
            if isinstance(env_ids, torch.Tensor) and len(env_ids):
                self.returning[env_ids] = False; self.spot[env_ids] = self._env.sim.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr[:2]]
            return out
        def _update_command(self):
            super()._update_command(); d = self.robot.data; o = self._env.scene[OPP]; pel = d.root_link_pos_w[:, :2]; w, x, y, z = d.root_link_quat_w.unbind(-1)
            oc = self._env.command_manager.get_term("opp"); yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            v = torch.where(oc.placed[:, None], o.data.body_link_pos_w[:, oc.o_pel, :2], oc.o_pel_xy) - pel   # R1HitEnv.true_bearing: the opponent pelvis xpos of the last mj_step
            wz = (FACE_GAIN * FO.wrap(torch.atan2(v[:, 1], v[:, 0]) - yaw)).clamp(-FACE_MAX, FACE_MAX)
            e = self.spot - pel; dist = torch.linalg.norm(e, dim=-1); self.returning = torch.where(self.returning, dist > HOLD_BAND[1], dist > HOLD_BAND[0])
            c, s = torch.cos(yaw), torch.sin(yaw); r = self.returning.float()
            vx = (HOLD_GAIN * (c * e[:, 0] + s * e[:, 1])).clamp(*HOLD_VX) * r; vy = (HOLD_GAIN * (-s * e[:, 0] + c * e[:, 1])).clamp(-HOLD_VY, HOLD_VY) * r
            self.vel_command_b = torch.stack([vx, vy, wz], -1)

    @dataclass(kw_only=True)
    class FbTwistCommandCfg(UniformVelocityCommandCfg):
        def build(self, env): return FbTwistCommand(self, env)

    class FbAction(ActionTerm):
        """fb_actions.HybridDelayedAction on the defender's 24 joints (CTRL_JOINTS order); reads no observation (m11)."""
        def __init__(self, cfg, env):
            super().__init__(cfg, env); sc, clip, ccfg = eval_contract(); latency(env)
            self.jid = torch.as_tensor(self._entity.find_joints(list(FO_CTRL()), preserve_order=True)[0], device=self.device)
            self.hda = AC.HybridDelayedAction(self.num_envs, env.fb_latency, ccfg["home"], sc, clip, dtype=torch.float32, device=self.device)
            self._raw = torch.zeros(self.num_envs, 24, device=self.device); self._tg = torch.zeros_like(self._raw)
        @property
        def action_dim(self): return 24
        @property
        def raw_action(self): return self._raw
        def process_actions(self, actions): self._raw[:] = actions; self._tg = self.hda.process(actions).float()
        def apply_actions(self): self._entity.set_joint_position_target(self._tg, joint_ids=self.jid)
        def reset(self, env_ids=None):
            ids = slice(None) if env_ids is None else env_ids; self.hda.reset(ids); self._raw[ids] = 0; return {}

    @dataclass(kw_only=True)
    class FbActionCfg(ActionTermCfg):
        def build(self, env): return FbAction(self, env)

    _C.update(**_ent_classes(), FbOppCommand=FbOppCommand, FbOppCommandCfg=FbOppCommandCfg, FbTwistCommand=FbTwistCommand,
              FbTwistCommandCfg=FbTwistCommandCfg, FbAction=FbAction, FbActionCfg=FbActionCfg)
    return _C

def FO_CTRL():
    from r1gym.model import CTRL_JOINTS
    return CTRL_JOINTS

@functools.lru_cache(maxsize=1)
def eval_contract():
    """(scale [24], clip, controller cfg) of R1HitEnv under G's controller cfg (export_hybrid_fb.env_scale): home, scale, noise."""
    import export_hybrid_fb as EX
    return EX.env_scale()

def task_input_files():
    """Every file the task reads (fb_m9_check (2)): motions and hulls (M1 / M8 manifests), the tracker, the start's donors and G's
    controller files (eval_contract)."""
    import hybrid_actor as HA
    fb = GPU / "motions" / "fb"
    return [fb / "fb_motion.npz", fb / "fb_l1.npz", fb / "fb_l1_loco.npz", fb / "fb_hulls.npz", FO.TRACKER_PT, ROOT / HA.G_PT, *(ROOT / v for v in HA.ARM_PT.values()),
            ROOT / HA.G_ONNX, (ROOT / HA.G_ONNX).with_name("gym.json"), ROOT / "r1gym" / "assets" / "r1" / "r1.xml"]

def manifest_ok(loco=False):
    """The motion inputs equal their manifests (WATCH6 E0 'clip-manifest hash mismatch'): fb_motion.npz and fb_l1(_loco).npz sha256."""
    import export_fb_library as L
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); f_npz, _, _, f_man = L.names_for(loco)
    man = json.load(open(GPU / "motions" / "fb" / f_man, encoding="utf-8")); mm = json.load(open(GPU / "motions" / "fb" / "fb_motion_manifest.json", encoding="utf-8"))
    return sha(GPU / "motions" / "fb" / f_npz) == man["outputs"][f_npz] and sha(FO.MOTION) == mm["outputs"]["fb_motion.npz"]

# ---- observation terms
class FbActorObs:
    """Actor class term (module docstring): EvalActorObs on the env; registers itself as env.fb_obs_term for the critic terms."""
    def __init__(self, cfg, env):
        self.env = env; n, dev = env.num_envs, env.device; sc, clip, ccfg = eval_contract(); latency(env)
        self.obs = OB.EvalActorObs(n, env.fb_latency, OB.noise_halfwidths(ccfg), device=dev, kp_dtype=torch.float32, out_dtype=torch.float32)
        self.home = torch.as_tensor(ccfg["home"], dtype=torch.float32, device=dev); r, o = env.scene[ENT], env.scene[OPP]
        self.jid = torch.as_tensor(r.find_joints(list(FO_CTRL()), preserve_order=True)[0], device=dev)
        self.kb = torch.as_tensor([o.find_bodies(b)[0][0] for b in OB.KP_BODIES], device=dev); self.tb = r.find_bodies(OB.DEF_BODY)[0][0]
        self.clean = torch.zeros(n, OB.OBS_DIM, device=dev); env.fb_obs_term = self
    def reset(self, env_ids=None): self.obs.reset(slice(None) if env_ids is None else env_ids)
    def __call__(self, env, blind=False):
        if env.command_manager.get_term("opp").undriven: env.command_manager.compute(dt=0.0)   # first-ever reset: env.reset() runs no command compute
        r, o = env.scene[ENT].data, env.scene[OPP].data; act = env.action_manager.get_term("joint_pos").hda
        p = OB.proprio83(r.root_link_quat_w, r.root_link_ang_vel_b, r.joint_pos[:, self.jid], r.joint_vel[:, self.jid], self.home,
                         env.command_manager.get_term("twist").vel_command_b, env.episode_length_buf, act.applied)
        X = torch.cat([o.body_link_pos_w[:, self.kb], r.body_link_pos_w[:, self.tb][:, None]], 1); Q = torch.cat([o.body_link_quat_w[:, self.kb], r.body_link_quat_w[:, self.tb][:, None]], 1)
        P_T, y, v = OB.kp_state(X, Q); out = self.obs.compute(p, P_T, y, v); self.clean = self.obs.clean
        if blind: out = out.clone(); out[:, OB.PROPRIO_DIM:] = 0
        return out

def kp_truth(env): return env.fb_obs_term.clean[:, OB.PROPRIO_DIM:]
def intent_obs(env): return env.command_manager.get_term("opp").command
def ring_home_obs(env):
    tw = env.command_manager.get_term("twist"); d = tw.robot.data; w, x, y, z = d.root_link_quat_w.unbind(-1)
    return ring_home(tw.spot, d.root_link_pos_w[:, :2], torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
def priv_obs(env):
    c = env.command_manager.get_term("opp"); rd, od = c.robot.data, c.opp.data; q = rd.root_link_quat_w; w, x, y, z = q.unbind(-1)
    oq = od.root_link_quat_w; ow, ox, oy, oz = oq.unbind(-1); lat = latency(env); n = env.num_envs
    dr = getattr(env, "fb_dr_p", None) or {k: torch.zeros(n, *s, device=env.device) for k, s in (("mass", (1,)), ("torso_kg", ()), ("com_m", (2, 3)), ("foot_mu", ()), ("kp", (1,)), ("kd", (1,)), ("cap", (1,)), ("armature", (1,)))}
    return priv_block(rd.root_link_pos_w[:, :2], torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), od.root_link_pos_w, torch.atan2(2 * (ow * oz + ox * oy), 1 - 2 * (oy * oy + oz * oz)),
                      od.root_link_lin_vel_w, c.od.now, c.od.age.float() * c.dt, lat.act_delay, lat.k, dr, c.acct.step["tangle"], c.rw.attr.W())

# ---- events (reset mode; fb_draw_latency first)
def latency(env):
    """env.fb_latency, the env-level latency buffers (fb_events.Latency), made by whichever manager needs them first."""
    if not hasattr(env, "fb_latency"): env.fb_latency = EV.Latency(env.num_envs, env.device)
    return env.fb_latency

def fb_draw_latency(env, env_ids): latency(env); EV.fb_draw_latency(env, env_ids)

def fb_reset_defender(env, env_ids):
    """The eval's R1Robot.reset (pelvis 0.74 m, upright, home, at rest), turned by a bout heading U(-pi, pi) about the env origin (the
    replica's training reset)."""
    r = env.scene[ENT]; n = len(env_ids); dev = env.device; h = (2 * torch.rand(n, device=dev) - 1) * math.pi
    pos = env.scene.env_origins[env_ids].clone(); pos[:, 2] += DEF_Z; quat = torch.stack([torch.cos(h / 2), 0 * h, 0 * h, torch.sin(h / 2)], 1)
    r.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(n, 6, device=dev)], 1), env_ids=env_ids)
    jid = torch.as_tensor(r.find_joints(list(FO_CTRL()), preserve_order=True)[0], device=dev); home = torch.as_tensor(eval_contract()[2]["home"], dtype=torch.float32, device=dev)
    r.write_joint_state_to_sim(home.expand(n, -1).clone(), torch.zeros(n, 24, device=dev), joint_ids=jid, env_ids=env_ids)

def fb_draw_population(env, env_ids, steps_per_env=24):
    """The episode's population row (fb_population.Population.draw), on CPU, into env.fb_pop (read by the placement); iterations
    1-100 of a run: t0_nominal only."""
    c = env.command_manager.get_term("opp"); t0_only = c.nstep // steps_per_env < POP.T0_ONLY_ITERS   # this run's iterations 1-100 (mjlab restores common_step_counter on resume)
    d = c.pop.draw(len(env_ids), t0_only=t0_only)
    if not hasattr(env, "fb_pop"): env.fb_pop = {k: torch.zeros(env.num_envs, dtype=v.dtype) for k, v in d.items()}
    for k, v in d.items(): env.fb_pop[k][env_ids.cpu()] = v

def fb_dr(env, env_ids):
    """The defender's DR (fb_dr: nominal, then the eval's draws, no set_const) into mjlab's per-world fields, then the t1_dr rows'
    opponent DR (fb_population.OppDR)."""
    m = env.sim.mj_model
    if not hasattr(env, "fb_dr_ix"):
        env.fb_dr_ix = DR.DRIndex(m, FO_CTRL(), is_defender=lambda nm: nm.startswith(PFX[ENT]), prefix=PFX[ENT]); env.fb_opp_ix = POP.OppDR(m, FO_CTRL(), PFX[OPP])
        env.fb_dr_nom = {k: env.sim.get_default_field(k) for k in DR.FIELDS}; env.fb_dr_p = DR.sample(env.num_envs, env.fb_dr_ix, dtype=torch.float32, device=env.device)
        for k in env.fb_dr_p: env.fb_dr_p[k][:] = torch.as_tensor(1.0 if k not in ("torso_kg", "com_m") else 0.0)
    F = {k: getattr(env.sim.model, k) for k in DR.FIELDS}; p = DR.sample(len(env_ids), env.fb_dr_ix, dtype=torch.float32, device=env.device)
    DR.write(F, env.fb_dr_nom, env.fb_dr_ix, p, env_ids)
    for k, v in p.items(): env.fb_dr_p[k][env_ids] = v
    od = env.fb_pop["opp_dr"][env_ids.cpu()].to(env.device); ids = env_ids[od]
    if len(ids): env.fb_opp_ix.write(F, ids, env.fb_opp_ix.sample(len(ids), dtype=torch.float32, device=env.device))
fb_dr.model_fields = DR.FIELDS

def fb_push(env, env_ids):
    """Interval event (T1 rows only, m5): a velocity push of the defender root, xy <= PUSH vmax; its time masks the hit terms 0.5 s."""
    c = env.command_manager.get_term("opp"); ids = env_ids[c.pushable[env_ids]]
    if not len(ids): return
    r = env.scene[ENT]; v = r.data.root_link_vel_w[ids].clone(); dv = (2 * torch.rand(len(ids), 2, device=env.device) - 1) * PUSH["vmax"]
    dv = dv * (PUSH["vmax"] / torch.linalg.norm(dv, dim=-1, keepdim=True).clamp(min=PUSH["vmax"])); v[:, :2] += dv
    r.write_root_link_velocity_to_sim(v, env_ids=ids); c.t_push[ids] = c._t_ep()[ids].float()

# ---- termination / reward / metrics terms
def fb_update(env):
    """Termination term listed first: runs the step's single update (never terminates)."""
    env.command_manager.get_term("opp").update(); return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def fell_eval(env):
    c = env.command_manager.get_term("opp"); d = c.robot.data; fell = EV.fell_eval(d.body_link_pos_w[:, c.r_pel, 2], EV.up_z(d.body_link_quat_w[:, c.r_pel]))
    c.note_terminations(fell, torch.zeros_like(fell)); return fell

def ring_r(env):
    tw = env.command_manager.get_term("twist"); return torch.linalg.norm(tw.robot.data.root_link_pos_w[:, :2] - tw.spot, dim=-1)

def ring_out(env, radius=RING["out"]):
    out = ring_r(env) > radius; c = env.command_manager.get_term("opp"); c.note_terminations(torch.zeros_like(out), out); return out

def spar_ring(env, free=RING["free"], out=RING["out"]): return 1.0 - ((ring_r(env) - free) / (out - free)).clamp(0.0, 1.0)
def spar_alive(env): return (~env.termination_manager.terminated).float()

def fb_term(env, name):
    """A reward value of this step (fb_update's FbRewards.step), by name; res_l2 annealed to res_l2_weight(B iteration) / -0.5."""
    c = env.command_manager.get_term("opp"); v = c.rv.get(name)
    if v is None: return torch.zeros(env.num_envs, device=env.device)
    if name == "res_l2": v = v * (RW.res_l2_weight(c.nstep // getattr(env.cfg, "fb_steps_per_env", 24)) / RW.RES_L2[0])   # B iteration = this run's
    return v.float()

def actor_metric(env, key):
    a = actor_stats(); z = torch.zeros(env.num_envs, device=env.device)
    if a is None or a._mu is None: return z
    s = a.std().detach(); legs = a._mu.detach()[:, :14]
    val = dict(fb_arm_std=s[14:].mean(), fb_leg_std=s[:14].mean() if a.stage == "B" else s[:14].mean() * 0, fb_leg_dev_u=a.leg_dev * 1e6, fb_res_abs_u=(a.r.abs().max() if a.r is not None else s.new_zeros(())) * 1e6)[key]
    return z + float(val)

# =============================================================================================== cfg builders
def scene_cfg(hulls=None, base_sensors=(), num_envs=1):
    """mjlab SceneCfg of the task: plane terrain, robot, opponent (FbEntityCfg), the fb contact sensors, scene_edit."""
    from mjlab.scene import SceneCfg
    from mjlab.terrains.terrain_entity import TerrainEntityCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.entity import EntityCfg
    from src.assets.robots.unitree_r1.r1_constants import get_r1_robot_cfg
    hulls = hulls or hulls_cfg(); C = _ent_classes(); ents = {}
    home = dict(zip(FO_CTRL(), [float(x) for x in eval_contract()[2]["home"]])) if "FB_NO_HOME" not in os.environ else None
    for ent in (ENT, OPP):
        r = get_r1_robot_cfg(); kw = {f.name: getattr(r, f.name) for f in __import__("dataclasses").fields(r)}
        kw["collisions"] = (CollisionCfg(**collision_kwargs(ent)),); kw["spec_fn"] = functools.partial(entity_spec, r.spec_fn, hulls)
        if ent == ENT and home: kw["init_state"] = EntityCfg.InitialStateCfg(pos=(0.0, 0.0, DEF_Z), joint_pos=home, joint_vel={".*": 0.0})
        if ent == OPP: kw["init_state"] = EntityCfg.InitialStateCfg(pos=(0.5, 0.0, 0.76), rot=(0.0, 0.0, 0.0, 1.0), joint_pos=dict(r.init_state.joint_pos), joint_vel={".*": 0.0})
        ents[ent] = C["FbEntityCfg"](**kw)
    shells = lambda zones=None: tuple(shell_name(n) for n in _shell_meshes(zones))
    sens = []
    for (k, kind, body), (rk, _, rbody) in zip(limb_bodies(), limb_bodies()):
        hk = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=DECIM)
        sens.append(ContactSensorCfg(name=f"fb_hit_{k}", primary=ContactMatch(mode="geom", entity=ENT, pattern=shells()), secondary=ContactMatch(mode=kind, entity=OPP, pattern=body), **hk))
    sens.append(ContactSensorCfg(name="fb_body", primary=ContactMatch(mode="geom", entity=ENT, pattern=shells(HT.BODY_ZONES)), secondary=ContactMatch(mode="subtree", entity=OPP, pattern="pelvis"),
                                 fields=("found", "force"), reduce="netforce", num_slots=1, history_length=DECIM))
    for k, kind, body in limb_bodies():
        sens.append(ContactSensorCfg(name=f"fb_rev_{k}", primary=ContactMatch(mode="geom", entity=OPP, pattern=shells()), secondary=ContactMatch(mode=kind, entity=ENT, pattern=body),
                                     fields=("found", "force"), reduce="netforce", num_slots=1, history_length=DECIM))
    return SceneCfg(num_envs=num_envs, terrain=TerrainEntityCfg(terrain_type="plane"), entities=ents, sensors=tuple(base_sensors) + tuple(sens), spec_fn=scene_edit)

@functools.lru_cache(maxsize=4)
def _shell_meshes(zones=None):
    """The R1's shell meshes in geom order (r1gym's r1.xml = unitree's, asserted by fb_m9_check), optionally only those of zones."""
    s = mujoco.MjSpec.from_file(str(ROOT / "r1gym" / "assets" / "r1" / "r1.xml")); out = []
    for g in s.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.name.endswith("_collision"): continue
        link = g.parent.name
        if zones is None or HT.zone_of(link, g.meshname) in zones: out.append(g.meshname)
    return tuple(out)

def sim_cfg(solver=None):
    from mjlab.sim import MujocoCfg, SimulationCfg
    it, ls = solver or solver_cfg()
    return SimulationCfg(njmax=NJMAX, nconmax=NCONMAX, mujoco=MujocoCfg(**SIM, iterations=it, ls_iterations=ls))

def env_cfg(stage="A", blind=False, loco=False, play=False, t1_pt=None, steps_per_env=24):
    """The task's ManagerBasedRlEnvCfg on run 3's base (sparring_task._base: Unitree's R1 flat cfg, guard, footwork): scene, sim,
    observations, actions, commands, events, terminations, rewards, metrics replaced or extended as the module docstring says."""
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.envs.mdp.terminations import nan_detection
    import dataclasses, sparring_task as ST
    assert stage in STAGES
    cfg = ST._base()(play); C = _classes(); base_sens = tuple(s for s in (cfg.scene.sensors or ()) if s.name in ("feet_ground_contact", "self_collision"))
    cfg.scene = scene_cfg(None, base_sens); cfg.sim = sim_cfg(); cfg.decimation = DECIM; cfg.episode_length_s = EP_S
    cfg.observations["actor"] = ObservationGroupCfg(terms={"fb_actor": ObservationTermCfg(func=FbActorObs, params={"blind": blind})}, concatenate_terms=True, enable_corruption=False)
    cr = dict(cfg.observations["critic"].terms); assert "height_scan" not in cr
    assert list(cfg.observations)[:2] == ["actor", "critic"], "the actor group computes first: the critic reads FbActorObs.clean of the same step (M4_VERIFY 4.1)"
    cr.update(opp_keypoints_truth=ObservationTermCfg(func=kp_truth), strike_phase=ObservationTermCfg(func=intent_obs), ring_home=ObservationTermCfg(func=ring_home_obs), fb_priv=ObservationTermCfg(func=priv_obs))
    cfg.observations["critic"] = dataclasses.replace(cfg.observations["critic"], terms=cr)
    cfg.actions = {"joint_pos": C["FbActionCfg"](entity_name=ENT)}
    tw = cfg.commands["twist"]; tw_kw = {f.name: getattr(tw, f.name) for f in dataclasses.fields(tw)}
    cfg.commands = {"opp": C["FbOppCommandCfg"](resampling_time_range=(1e9, 1e9), loco=loco, t1_pt=t1_pt or os.environ.get("FB_T1_PT", "")), "twist": C["FbTwistCommandCfg"](**tw_kw)}
    cfg.events = {"fb_draw_latency": EventTermCfg(func=fb_draw_latency, mode="reset"), "fb_reset_defender": EventTermCfg(func=fb_reset_defender, mode="reset"),
                  "fb_draw_population": EventTermCfg(func=fb_draw_population, mode="reset", params={"steps_per_env": steps_per_env}),
                  "fb_dr": EventTermCfg(func=fb_dr, mode="reset")}
    if not play: cfg.events["fb_push"] = EventTermCfg(func=fb_push, mode="interval", interval_range_s=PUSH["interval_s"])
    cfg.terminations = {"fb_update": TerminationTermCfg(func=fb_update), "time_out": cfg.terminations["time_out"], "fell_eval": TerminationTermCfg(func=fell_eval),
                        "nan": TerminationTermCfg(func=nan_detection)}
    if not play: cfg.terminations["ring_out"] = TerminationTermCfg(func=ring_out)
    cfg.curriculum = {}
    W = RW.WEIGHTS["B" if stage == "B" else "A"]
    for k, w in W.items():
        f = {"spar_ring": spar_ring, "spar_alive": spar_alive}.get(k)
        cfg.rewards[k] = RewardTermCfg(func=f, weight=w) if f else RewardTermCfg(func=fb_term, weight=w, params={"name": k})
    cfg.metrics = {**(cfg.metrics or {}), **{k: MetricsTermCfg(func=actor_metric, params={"key": k}) for k in ACTOR_KEYS}}
    return cfg

# =============================================================================================== rsl_rl actor factory (A9.2)
class FbActorModel:
    """rsl_rl 5.0.1 model factory (MLPModel's constructor signature): obs, obs_groups, obs_set, output_dim, **cfg (fb_stage, fb_blind,
    fb_root, fb_seed; MLPModel's own keys ignored) -> a HybridActor (A0: nothing trains, A: arms + std, B: to_stage_b), registered as
    the task's live actor (res_l2, metrics). The blind flag rides in its state dict (blind_flag)."""
    def __new__(cls, obs, obs_groups, obs_set, output_dim, fb_stage="A", fb_blind=False, fb_root=str(ROOT), fb_seed=0, **_):
        import hybrid_actor as HA
        assert output_dim == 24 and fb_stage in STAGES; sc, clip, _c = eval_contract()
        a = HA.HybridActor.from_checkpoints(fb_root, sc, clip, bool(fb_blind)); a.obs_groups = tuple(obs_groups[obs_set])
        if fb_stage == "A0": a.arms.requires_grad_(False); a.std_param.requires_grad_(False)
        if fb_stage == "B": a.to_stage_b(seed=fb_seed); a.load_state_dict = functools.partial(load_into_b, a, a.load_state_dict)
        a.load_state_dict = functools.partial(load_same_blind, a, a.load_state_dict)
        a._mu, a.leg_dev = None, 0.0; f0 = a.forward
        def fwd(obs, masks=None, hidden_state=None, stochastic_output=False):
            out = f0(obs, masks, hidden_state, stochastic_output)
            if not stochastic_output: a._mu = out
            else: a.leg_dev = max(a.leg_dev, float((out[:, :14] - a._mu[:, :14]).abs().max())) if a.stage != "B" else 0.0
            return out
        a.forward = fwd; _ACTOR["a"] = a; return a

def load_same_blind(a, load, sd, strict=True, **kw):
    """A9.2: a checkpoint loads only into its own twin (its stored blind flag = this run's); else the loaded buffer would relabel the run."""
    import hybrid_actor as HA
    assert HA.stored_blind(sd) == a.blind, f"checkpoint blind_flag {HA.stored_blind(sd)} != this run's blind {a.blind} (A9.2)"
    return load(sd, strict=strict, **kw)

def load_into_b(a, load, sd, strict=True, **kw):
    """A stage-B actor loading a stage-A checkpoint (the B restart; the optimizer is built on B's parameters before any load): the
    residual keeps its zero-output init and res_norm becomes the loaded arm normaliser, leg std LEG_STD0 (= to_stage_b on the loaded A,
    tests); a B checkpoint loads as is."""
    import hybrid_actor as HA
    if not any(k.startswith("residual.") for k in sd):
        sd = dict(sd); cur = a.state_dict()
        for k in cur:
            if k.startswith("residual."): sd[k] = cur[k]
            elif k.startswith("res_norm."): sd[k] = sd["arms.obs_normalizer." + k[len("res_norm."):]]
        sd["std_param"] = sd["std_param"].clone(); sd["std_param"][:HA.NL] = HA.LEG_STD0
    return load(sd, strict=strict, **kw)

def start_checkpoint(out, blind=False, critic_src=None):
    """The warm start for A0 (2.5): actor = the start HybridActor (blind flag stored), critic = the arm donor's 138-input critic widened by
    len(PRIV) zero-weight columns (normaliser mean 0, std 1 on them)."""
    import hybrid_actor as HA
    sc, clip, _c = eval_contract(); a = HA.HybridActor.from_checkpoints(ROOT, sc, clip, blind)
    src = torch.load(critic_src or ROOT / HA.ARM_PT["blind" if blind else "sighted"], map_location="cpu", weights_only=False); cs = dict(src["critic_state_dict"])
    w = cs["mlp.0.weight"]; assert w.shape[1] == 138, ("run 3's critic has 138 inputs", w.shape); k = len(PRIV)
    cs["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], k, dtype=w.dtype)], 1)
    for nm, v in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        if nm in cs: cs[nm] = torch.cat([cs[nm], torch.full((*cs[nm].shape[:-1], k), v, dtype=cs[nm].dtype)], -1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)   # the pod writes it into a fresh run dir ($RUNS/start_<twin>)
    torch.save({"actor_state_dict": a.state_dict(), "critic_state_dict": cs, "iter": 0, "infos": dict(fb_start=True, blind=bool(blind))}, out); return out

# =============================================================================================== registration
TASKS = {"R1-Fb-A": dict(stage="A"), "R1-Fb-B": dict(stage="B"), "R1-Fb-A-Blind": dict(stage="A", blind=True), "R1-Fb-B-Blind": dict(stage="B", blind=True),
         "R1-Fb-LOCO-A": dict(stage="A", loco=True), "R1-Fb-LOCO-B": dict(stage="B", loco=True), "R1-Fb-LOCO-A-Blind": dict(stage="A", loco=True, blind=True),
         "R1-Fb-LOCO-B-Blind": dict(stage="B", loco=True, blind=True)}
RL = dict(A=dict(learning_rate=2e-4, entropy_coef=0.003), B=dict(learning_rate=2e-4, entropy_coef=0.003))

_R = {}
def runner_cls():
    """FbRunner: mjlab's MjlabOnPolicyRunner (env_state kept) without the velocity runner's ONNX export at every save (the HybridActor
    has no as_onnx; export_hybrid_fb exports it, asserting the blind flag); load skips the optimizer state across a stage boundary or
    from a start checkpoint (A0 start: none stored; A -> B: B's optimizer has the residual's parameters), keeping the iteration."""
    if _R: return _R["cls"]
    from mjlab.rl.runner import MjlabOnPolicyRunner
    class FbRunner(MjlabOnPolicyRunner):
        def load(self, path, load_cfg=None, strict=True, map_location=None):
            d = torch.load(path, map_location="cpu", weights_only=False); a = self.alg.get_policy()
            same = "optimizer_state_dict" in d and any(k.startswith("residual.") for k in d["actor_state_dict"]) == (getattr(a, "stage", "A") == "B")
            return super().load(path, load_cfg or dict(actor=True, critic=True, optimizer=same, iteration=True), strict, map_location)
    _R["cls"] = FbRunner; return FbRunner

def register():
    import dataclasses
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.rl import RslRlModelCfg
    from src.tasks.velocity.config.r1.rl_cfg import unitree_r1_ppo_runner_cfg
    @dataclasses.dataclass
    class FbModelCfg(RslRlModelCfg):
        fb_stage: str = "A"; fb_blind: bool = False; fb_root: str = str(ROOT); fb_seed: int = 0
    for tid, kw in TASKS.items():
        stage = kw["stage"] if kw["stage"] == "B" else os.environ.get("FB_STAGE", "A"); assert stage in ("A0", "A") or kw["stage"] == "B", (tid, stage)
        rl = unitree_r1_ppo_runner_cfg(); rl.experiment_name = "r1_fullbody"; spe = rl.num_steps_per_env
        rl.actor = FbModelCfg(class_name="fullbody_task:FbActorModel", fb_stage=stage, fb_blind=kw.get("blind", False), fb_seed=int(os.environ.get("FB_SEED", "0")))
        for k, v in RL[kw["stage"]].items(): setattr(rl.algorithm, k, v)
        mk = lambda play: _with_steps(env_cfg(kw["stage"], kw.get("blind", False), kw.get("loco", False), play, steps_per_env=spe), spe)
        register_mjlab_task(task_id=tid, env_cfg=mk(False), play_env_cfg=mk(True), rl_cfg=rl, runner_cls=runner_cls())

def _with_steps(cfg, spe):
    cfg.fb_steps_per_env = spe; return cfg

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--out", required=True); ap.add_argument("--blind", action="store_true"); a = ap.parse_args()
        print(start_checkpoint(a.out, a.blind)); sys.exit(0)
    sys.modules.setdefault("fullbody_task", sys.modules["__main__"])   # rsl_rl imports class_name 'fullbody_task:FbActorModel': this module, not a 2nd copy (_ACTOR)
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts"))
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
