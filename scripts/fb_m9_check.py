"""M9 checks (M2V2_PLAN_v3 section 3 row M9, m9 / m10 / m13; A9): the fullbody mjlab task's scene, inputs, WATCH6 and pod script on CPU.
  (1) model  mjlab 1.2.0's REAL Scene of gpu/fullbody_task.py (scene_cfg: both entities, prefixes robot/ and opponent/, CollisionCfg,
             actuators, the entity spec_fn (shell names, hulls), the fb contact sensors, scene_edit) compiled on CPU with warp,
             mujoco_warp, mediapy and scipy stubbed in sys.modules (CRITIQUE_M2V2 C1, no install), the task's MujocoCfg applied as
             mjlab's Simulation does; diffed by geom, body, joint and actuator name against scripts/fb_replica.py's model of the same
             (hulls, solver): geoms contype / conaffinity / condim / solref / solimp / solmix / priority / friction / margin / gap /
             type / size / pos / quat and the hull vertex sets (exact, as row sets), bodies (order, mass, inertia, ipos, iquat, pos,
             quat), joints (armature, damping, frictionloss, range, limited), actuators (gain / bias / force range / limits / gear),
             opt (timestep, integrator, cone, impratio, ccd_iterations, iterations, ls_iterations, solver, jacobian, tolerances,
             gravity, flags). Zone maps (A10.3, zone_maps): the task's shell zones / blocks / body subset on mjlab's model = the
             replica's account sensors on the same hulls = the full-hull replica's, both entities, geom for geom by link. Floor `terrain` contype 1 / conaffinity 3 asserted on mjlab's compiled model (m9) and conaffinity 1
             without scene_edit (C1's finding reproduced). DRIndex with explicit is_defender / prefix on mjlab's model: a fb_dr.write
             changes only defender rows; OppDR only opponent rows (M4_VERIFY 4.2). The mjlab source used is checked file by file
             against mjlab-1.2.0-py3-none-any.whl.
  (2) inputs every motion / hull input of the task (fb_population, fullbody_task) equals its M1 / M8 manifest's output sha256; the
             banned-name scan (fb_seeds.scan) over every task input is clean.
  (3) watch  WATCH6 replays (scripts/fb_watch.py): synthetic logs that must stop do, each for its own rule; a healthy synthetic log, the
             run-3 training logs and the blind twin's logs trip no load check (E0) (the run-3 logs predate the fb counters: E0 is
             evaluated on the keys they carry; the others are reported as not applicable).
  (4) pod    gpu/pod_fb.sh: bash -n, the asserts present (sim options, tf32 off, md5, versions, the M9 model diff rerun), a taskset plan
             with disjoint training / background-eval cores, and a dry run with FB_FAKE_GPU=1 FB_DRY=1 (no GPU, no network, no training).
Run in the repo: python -I scripts/fb_m9_check.py --out m9_check.json [--src <run2b\\src>] [--logs <run-3 log dir, default RUN3>] [--only model,inputs,watch,pod]"""
import sys, os, json, time, types, hashlib, argparse, zipfile, subprocess, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, mujoco

SRC = Path(os.environ.get("FB_MJLAB_SRC", r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\run2b\src"))
RUN3 = Path(os.environ.get("FB_RUN3_LOGS", str(SRC.parents[1] / "dl5")))   # run 3's training logs (spar3_A3s / A3 / blind3), as tests/test_fb_watch.py
STUBS = ("warp", "warp.context", "warp.types", "mujoco_warp", "mujoco_warp._src", "mujoco_warp._src.types", "mujoco_warp._src.io", "tyro", "mediapy", "scipy", "scipy.interpolate")

# ---- (1) stubbed mjlab
class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return _Dummy()
    def __getattr__(self, n): return _Dummy()
    def __getitem__(self, k): return _Dummy()
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
    def __or__(self, o): return self
    __ror__ = __or__

class _StubMod(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Dummy()

def stub_mjlab(src=SRC):
    """FB_POD=1 (the pod, P0): the installed mjlab, nothing stubbed. Otherwise C1's pattern: stub the GPU / media modules, put mjlab 1.2.0's and unitree_rl_mjlab's sources first on sys.path, import what
    the scene needs (any other missing third-party module is stubbed and named in the result)."""
    if os.environ.get("FB_POD"):
        import mjlab, importlib.metadata as md; return dict(pod=True, mjlab=md.version("mjlab"), mjlab_path=str(Path(mjlab.__file__).parent))
    for nm in STUBS: m = _StubMod(nm); m.__path__ = []; sys.modules[nm] = m
    for p in (src / "unitree_rl_mjlab", src / "mjlab_x"):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    stubbed = []
    for _ in range(40):
        try:
            for mod in ("mjlab.scene.scene", "mjlab.terrains.terrain_entity", "mjlab.sim", "mjlab.sensor", "src.assets.robots.unitree_r1.r1_constants"): importlib.import_module(mod)
            break
        except ModuleNotFoundError as e:
            nm = e.name
            if nm.startswith(("mjlab", "src")): raise
            for k in [x for x in list(sys.modules) if x.startswith(("mjlab", "src"))]: del sys.modules[k]
            parts = nm.split(".")
            for i in range(1, len(parts) + 1):
                q = ".".join(parts[:i])
                if not isinstance(sys.modules.get(q), _StubMod): mm = _StubMod(q); mm.__path__ = []; sys.modules[q] = mm
            stubbed.append(nm)
    import mjlab
    return dict(stubbed_extra=stubbed, mjlab_path=str(Path(mjlab.__file__).parent))

def wheel_identity(src=SRC):
    """FB_POD=1: the installed mjlab's version is 1.2.0. Otherwise mjlab_x (the source on sys.path) vs mjlab-1.2.0-py3-none-any.whl: every wheel .py / .xml file byte-equal."""
    if os.environ.get("FB_POD"):
        import importlib.metadata as md; v = md.version("mjlab"); return dict(installed=v, ok=v == "1.2.0")
    whl = src / "mjlab_whl" / "mjlab-1.2.0-py3-none-any.whl"; bad, n = [], 0
    with zipfile.ZipFile(whl) as z:
        for nm in z.namelist():
            if not nm.startswith("mjlab/") or not nm.endswith((".py", ".xml")): continue
            n += 1; f = src / "mjlab_x" / nm
            if not f.exists() or f.read_bytes() != z.read(nm): bad.append(nm)
    return dict(wheel=whl.name, files=n, mismatched=bad, ok=n > 0 and not bad)

def mjlab_model(hulls, solver, edit=True):
    """mjlab's Scene(scene_cfg) compiled on CPU, the task's MujocoCfg applied (Simulation.__init__'s cfg.mujoco.apply)."""
    import fullbody_task as FT
    from mjlab.scene.scene import Scene
    cfg = FT.scene_cfg(hulls)
    if not edit: cfg.spec_fn = None
    sc = Scene(cfg, device="cpu"); m = sc.compile(); FT.sim_cfg(solver).mujoco.apply(m); return m, sc

def replica_model(hulls, solver):
    import fb_replica as RP, export_hybrid_fb as EX
    _, _, cfg = EX.env_scale(); ccfg = {**cfg, "opponent_mode": "policy", "opponent_aim": EX.AIM}
    m, _ = RP.build_model(ccfg, RP.conf(hulls=hulls, solver=solver)); return m

def old_names_replica(hulls, solver):
    """Negative control (A10.1): the replica's model with the pre-A10 hull names ('fbhull_<mesh>', the originals kept)."""
    import fb_replica as RP, export_hybrid_fb as EX
    from r1gym.model import build_spec
    _, _, cfg = EX.env_scale(); ccfg = {**cfg, "opponent_mode": "policy", "opponent_aim": EX.AIM}; sp = build_spec(opponent=True, cfg=ccfg); P = RP.hull_points(hulls, sp.compile())
    for g in RP.shell_geoms(sp):
        mn = g.meshname; sp.add_mesh(name=f"fbhull_{mn}", uservert=np.asarray(P[mn[4:] if mn.startswith("opp_") else mn], np.float64).ravel().tolist())
        g.meshname = f"fbhull_{mn}"; g.pos = [0.0, 0.0, 0.0]; g.quat = [1.0, 0.0, 0.0, 0.0]
    m = sp.compile(); m.opt.iterations, m.opt.ls_iterations = solver; return m

def _ent_of(name, mj):
    """(entity, stripped name) of a model element name: mjlab robot/ | opponent/; the eval model '' | opp_."""
    if mj:
        for e, p in (("def", "robot/"), ("opp", "opponent/")):
            if name.startswith(p): return e, name[len(p):]
        return "world", name
    return ("opp", name[4:]) if name.startswith("opp_") else ("def", name)

def _mesh(m, g):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: return ""
    s = m.mesh(m.geom_dataid[g]).name
    for p in ("robot/", "opponent/"): s = s[len(p):] if s.startswith(p) else s
    return s[4:] if s.startswith("opp_") else s   # A10: hulls carry the original mesh names (no prefix stripped but the entity's)

def geom_key(m, g, mj):
    b = m.geom_bodyid[g]
    if m.geom(g).name in ("floor", "terrain") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE: return ("world", "floor")
    if b == 0 or m.body(b).name == "terrain": return ("world", m.geom(g).name)
    e, body = _ent_of(m.body(b).name, mj); nm = m.geom(g).name; nm = _ent_of(nm, mj)[1] if nm else ""
    if nm.endswith("_shell"): nm = ""
    return (e, body, nm or "mesh:" + _mesh(m, g), int(m.geom_type[g]))

GF = ("contype", "conaffinity", "condim", "solref", "solimp", "solmix", "priority", "friction", "margin", "gap", "size", "pos", "quat")
BF = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat")
JF = ("dof_armature", "dof_damping", "dof_frictionloss", "jnt_range", "jnt_limited", "jnt_stiffness")
AF = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_forcelimited", "actuator_ctrlrange", "actuator_ctrllimited", "actuator_gear",
      "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_trntype", "actuator_dynprm", "actuator_actlimited")
OPT = ("timestep", "integrator", "cone", "impratio", "ccd_iterations", "iterations", "ls_iterations", "solver", "jacobian", "tolerance", "ls_tolerance", "gravity",
       "noslip_iterations", "ccd_tolerance", "o_margin", "density", "viscosity", "wind", "magnetic", "disableflags", "enableflags")

def _eq(a, b): return np.array_equal(np.asarray(a), np.asarray(b))

def diff(mj, rp):
    """By name: every mismatch [(kind, key, field, mjlab, replica)] and the counts compared."""
    out, cnt, inert = [], {}, []   # inert: ctrlrange where neither model limits ctrl (MuJoCo ignores it), reported
    def G(m, mjf):   # key + its ordinal among equal keys (geom order within the body), world geoms other than the floor left out
        out, seen = {}, {}
        for g in range(m.ngeom):
            k = geom_key(m, g, mjf)
            if k[0] == "world" and k != ("world", "floor") or (k == ("world", "floor") and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE): continue
            i = seen[k] = seen.get(k, -1) + 1; out[k + (i,)] = g
        return out
    gm, gr = G(mj, True), G(rp, False)
    for k in sorted(set(gm) ^ set(gr), key=str): out.append(("geom", k, "missing", k in gm, k in gr))
    hv = 0
    for k in sorted(set(gm) & set(gr), key=str):
        a, b = gm[k], gr[k]
        for f in GF:
            x, y = getattr(mj, "geom_" + f)[a], getattr(rp, "geom_" + f)[b]
            if f == "size" and k[:2] == ("world", "floor") and _eq(x[:2], y[:2]): inert.append((k, "plane size[2] (grid spacing, rendering only)", np.asarray(x).tolist(), np.asarray(y).tolist())); continue
            if not _eq(x, y): out.append(("geom", k, f, np.asarray(x).tolist(), np.asarray(y).tolist()))
        if mj.geom_type[a] == mujoco.mjtGeom.mjGEOM_MESH:
            va, vb = (m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]] for m, g in ((mj, a), (rp, b)))
            hv += 1
            if not _eq(np.unique(va, axis=0), np.unique(vb, axis=0)): out.append(("geom", k, "hull_vertices", len(va), len(vb)))
    cnt["geoms"], cnt["mesh_geoms"] = len(set(gm) & set(gr)), hv
    for kind, n_m, n_r, name, fields in (("body", mj.nbody, rp.nbody, lambda m, i: m.body(i).name, BF), ("joint", mj.njnt, rp.njnt, lambda m, i: m.joint(i).name, JF)):
        nm_m = [(_ent_of(name(mj, i), True), i) for i in range(n_m)]; nm_r = [(_ent_of(name(rp, i), False) if name(rp, i) != "world" else ("world", "world"), i) for i in range(n_r)]
        for e in ("def", "opp"):
            om, orr = [k[1] for k, _ in nm_m if k[0] == e], [k[1] for k, _ in nm_r if k[0] == e]
            if om != orr: out.append((kind, e, "order", om, orr))
        dm, dr = {k: i for k, i in nm_m if k[0] != "world"}, {k: i for k, i in nm_r if k[0] != "world"}
        for k in sorted(set(dm) & set(dr)):
            for f in fields:
                arr_m, arr_r = getattr(mj, f), getattr(rp, f)
                if f.startswith("dof_"):
                    if mj.jnt_type[dm[k]] == mujoco.mjtJoint.mjJNT_FREE: continue
                    ia, ib = mj.jnt_dofadr[dm[k]], rp.jnt_dofadr[dr[k]]
                else: ia, ib = dm[k], dr[k]
                if not _eq(arr_m[ia], arr_r[ib]): out.append((kind, k, f, np.asarray(arr_m[ia]).tolist(), np.asarray(arr_r[ib]).tolist()))
        cnt[kind + "s"] = len(set(dm) & set(dr))
    jt = lambda m, a: m.joint(m.actuator_trnid[a, 0]).name
    am, ar = {_ent_of(jt(mj, a), True): a for a in range(mj.nu)}, {_ent_of(jt(rp, a), False): a for a in range(rp.nu)}
    for k in sorted(set(am) ^ set(ar)): out.append(("actuator", k, "missing", k in am, k in ar))
    for k in sorted(set(am) & set(ar)):
        for f in AF:
            x, y = getattr(mj, f)[am[k]], getattr(rp, f)[ar[k]]
            if f == "actuator_ctrlrange" and not mj.actuator_ctrllimited[am[k]] and not rp.actuator_ctrllimited[ar[k]]: inert.append((k, np.asarray(x).tolist(), np.asarray(y).tolist())); continue
            if not _eq(x, y): out.append(("actuator", k, f, np.asarray(x).tolist(), np.asarray(y).tolist()))
    cnt["actuators"] = len(set(am) & set(ar)); cnt["inert"] = len(inert); cnt["inert_kinds"] = sorted({str(x[1]) if isinstance(x[1], str) else "ctrlrange (ctrllimited 0 in both)" for x in inert})
    for f in OPT:
        x, y = getattr(mj.opt, f), getattr(rp.opt, f)
        if not _eq(x, y): out.append(("opt", "", f, np.asarray(x).tolist(), np.asarray(y).tolist()))
    return out, cnt

def zone_maps(mj, rp, rp_full):
    """A10.3: per target entity (defender: the opponent striking; opponent: reverse), each shell's (mesh name, zone, block, body subset,
    body-sensor zone) keyed by (link, ordinal among the link's shells): the task's shell_tables on mjlab's model, the replica's account
    sensors (fb_hits.MjContactSensors, its zone_idx / block_idx / bprim) on the replica's model of the same hulls and on the full-hull
    replica; all three equal, head shells present."""
    import fullbody_task as FT, fb_hits as HT
    def keyed(rows):
        out, seen = {}, {}
        for link, v in rows: i = seen[link] = seen.get(link, -1) + 1; out[f"{link}#{i}"] = v
        return out
    def rep(m, rev):
        s = HT.MjContactSensors(m, reverse=rev); st = lambda x: x[4:] if rev and x.startswith("opp_") else x; bset = set(s.bprim); bz = dict(zip(s.bprim, s.bzone_idx))
        return keyed([(st(m.body(m.geom_bodyid[g]).name), (st(m.mesh(m.geom_dataid[g]).name), int(z), bool(b), g in bset, bz.get(g, -1))) for g, z, b in zip(s.prim, s.zone_idx, s.block_idx)])
    res = {}
    for ent, rev in ((FT.ENT, False), (FT.OPP, True)):
        T = FT.shell_tables(mj, ent); bset = set(T["b_geoms"]); bz = dict(zip(T["b_geoms"], T["b_zone"]))
        task = keyed([(l, (FT.strip(mj.mesh(mj.geom_dataid[g]).name, ent), int(z), bool(b), g in bset, bz.get(g, -1))) for g, l, z, b in zip(T["geoms"], T["link"], T["zone"], T["block"])])
        a, f = rep(rp, rev), rep(rp_full, rev); bad = sorted(k for k in set(task) | set(a) | set(f) if not (task.get(k) == a.get(k) == f.get(k)))
        heads = sum(1 for v in task.values() if v[1] == HT.ZONES.index("head")); blocks = sum(1 for v in task.values() if v[2])
        res["defender" if ent == FT.ENT else "opponent"] = dict(n=len(task), mismatched=bad[:20], n_mismatch=len(bad), head_shells=heads, block_shells=blocks,
                                                                 zones=sorted({HT.ZONES[v[1]] for v in task.values()}), ok=not bad and len(task) == 32 and heads == 2 and blocks == 4)
    res["ok"] = all(v["ok"] for v in res.values()); return res

def dr_check(m):
    """fb_dr.DRIndex(is_defender = robot/, prefix robot/) and OppDR(prefix opponent/) on mjlab's model: a write of random draws changes
    only the right entity's rows (bodies, dofs, actuators, foot geoms); the defender's index sets equal the whole robot/ entity's."""
    import torch, fb_dr as DR, fb_population as POP, fullbody_task as FT
    from r1gym.model import CTRL_JOINTS
    ix = DR.DRIndex(m, CTRL_JOINTS, is_defender=lambda nm: nm.startswith("robot/"), prefix="robot/"); ox = POP.OppDR(m, CTRL_JOINTS, "opponent/")
    F, nom = DR.model_fields(m); F = {k: v.clone() for k, v in F.items()}; g = torch.Generator().manual_seed(0)
    p = DR.sample(1, ix, generator=g); DR.write(F, nom, ix, p)
    ent = lambda nm: "def" if nm.startswith("robot/") else "opp" if nm.startswith("opponent/") else "world"
    owner = dict(body_mass=lambda i: ent(m.body(i).name), body_ipos=lambda i: ent(m.body(i).name), geom_friction=lambda i: ent(m.body(m.geom_bodyid[i]).name),
                 actuator_gainprm=lambda i: ent(m.joint(m.actuator_trnid[i, 0]).name), actuator_biasprm=lambda i: ent(m.joint(m.actuator_trnid[i, 0]).name),
                 actuator_forcerange=lambda i: ent(m.joint(m.actuator_trnid[i, 0]).name), dof_armature=lambda i: ent(m.body(m.dof_bodyid[i]).name))
    rows = lambda F: {k: sorted({owner[k](int(i)) for i in np.flatnonzero(np.any((F[k][0] != nom[k]).reshape(len(nom[k]), -1).numpy(), 1))}) for k in DR.FIELDS}
    d_rows = rows(F); F2 = {k: v.clone() for k, v in DR.model_fields(m)[0].items()}; DR.write(F2, nom, ix, DR.sample(1, ix, generator=g), ids=[0]); ox.write(F2, [0], ox.sample(1, generator=g)); both = rows(F2)
    F3 = {k: nom[k][None].clone() for k in DR.FIELDS}; ox.write(F3, [0], ox.sample(1, generator=g)); o_rows = rows(F3)
    rb = sorted(b for b in range(m.nbody) if m.body(b).name.startswith("robot/") and b != 0); feet = [g for g in range(m.ngeom) if m.geom(g).name.startswith("robot/") and "foot" in m.geom(g).name]
    res = dict(defender_write_rows=d_rows, opponent_write_rows=o_rows, both_rows=both, defender_bodies=ix.nb, robot_bodies=len(rb), defender_feet=len(ix.feet), robot_feet=len(feet),
               defender_dofs=ix.nd, defender_acts=ix.na, opp_bodies=ox.nb, opp_acts=ox.na, opp_feet=len(ox.feet))
    res["ok"] = (all(v in ([], ["def"]) for v in d_rows.values()) and all(v in ([], ["opp"]) for v in o_rows.values()) and ix.nb == len(rb) and list(ix.bodies) == rb
                 and len(ix.feet) == len(feet) == 14 and ix.na == 24 and ix.nd == 24 + 6 and ox.na == 24 and ox.nb == len(rb) and len(ox.feet) == 14
                 and all(v in ([], ["def"], ["def", "opp"], ["opp"]) for v in both.values()) and "world" not in sum(both.values(), []))
    return res

def sensor_check(sc, m):
    """The fb contact sensors as mjlab resolved them: hit / reverse sensors' primaries = the target's 32 shells in geom order (FbOppCommand
    asserts the same at construction: the account's zone map), fb_body's = the BODY_ZONES shells; each sensor's secondary resolved."""
    import fullbody_task as FT
    T, R = FT.shell_tables(m, FT.ENT), FT.shell_tables(m, FT.OPP); out = {}
    prim = lambda nm: list(dict.fromkeys(s.primary_name for s in sc.sensors[nm]._slots))
    for k, _, _ in FT.limb_bodies():
        out[f"fb_hit_{k}"] = prim(f"fb_hit_{k}") == T["snames"]; out[f"fb_rev_{k}"] = prim(f"fb_rev_{k}") == R["snames"]
    out["fb_body"] = prim("fb_body") == T["b_names"]
    return dict(per_sensor=out, n_primary=len(T["names"]), n_body_primary=len(T["b_names"]), zones=sorted({__import__("fb_hits").ZONES[z] for z in T["zone"]}), ok=all(out.values()) and len(out) == 17)

def found_check(sc, m, n_states=24, seed=0):
    """The account's per-substep found inputs (fullbody_task.sensor_adr, read from sensordata by FbOppCommand's hook) on mjlab's
    compiled model: every address is the '<sensor>_<shell>_found' mjSENS_CONTACT (dim 1, found bit, netforce, 1 slot) on the target's
    shells in shell order with the sensor's secondary (the striker's limb body, or its subtree); and on random overlapping two-robot
    states, after mj_forward, its sensordata = the number of MuJoCo contacts between that shell and that secondary (match count)."""
    import fullbody_task as FT
    sid = {int(m.sensor_adr[i]): i for i in range(m.nsensor)}; bad, rows = [], []
    sub = lambda b: {x for x in range(m.nbody) if any(y == b for y in _anc(m, x))}
    for pfx, tgt, st in (("fb_hit", FT.ENT, FT.OPP), ("fb_rev", FT.OPP, FT.ENT)):
        T = FT.shell_tables(m, tgt); A = FT.sensor_adr(m, [sc.sensors[f"{pfx}_{k}"] for k, _, _ in FT.limb_bodies()])
        for (k, kind, body), adr in zip(FT.limb_bodies(), A):
            rb = m.body(FT.PFX[st] + body).id; sec = sub(rb) if kind == "subtree" else {rb}
            if len(adr) != len(T["geoms"]): bad.append((pfx, k, "count", len(adr))); continue
            for p, a in enumerate(adr):
                i = sid[a]; ok = (m.sensor(i).name == f"{pfx}_{k}_{T['snames'][p]}_found" and m.sensor_type[i] == mujoco.mjtSensor.mjSENS_CONTACT and m.sensor_dim[i] == 1
                                  and list(m.sensor_intprm[i][:3]) == [1, 3, 1] and m.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_GEOM and m.sensor_objid[i] == T["geoms"][p]
                                  and m.sensor_reftype[i] == (mujoco.mjtObj.mjOBJ_XBODY if kind == "subtree" else mujoco.mjtObj.mjOBJ_BODY) and m.sensor_refid[i] == rb)
                if not ok: bad.append((pfx, k, p, m.sensor(i).name))
                rows.append((a, T["geoms"][p], sec))
    d = mujoco.MjData(m); rng = np.random.default_rng(seed); fj = {e: next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.joint(j).name.startswith(FT.PFX[e])) for e in (FT.ENT, FT.OPP)}
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]; mism, matched, found_pos = 0, 0, 0
    for s in range(n_states):
        mujoco.mj_resetData(m, d); dx, h = rng.uniform(0.12, 0.4), rng.uniform(-0.4, 0.4)
        for e, pos, yaw in ((FT.ENT, (0.0, 0.0, 0.74), 0.0), (FT.OPP, (dx, rng.uniform(-0.1, 0.1), 0.74 + rng.uniform(-0.05, 0.05)), np.pi + h)):
            a = m.jnt_qposadr[fj[e]]; d.qpos[a:a + 7] = [*pos, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for j in hinge:
            lo, hi = m.jnt_range[j]; d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi) * 0.6
        mujoco.mj_forward(m, d); g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]; bid = m.geom_bodyid
        for a, g, sec in rows:
            ref = int(sum(1 for x, y in zip(g1, g2) if (x == g and bid[y] in sec) or (y == g and bid[x] in sec))); got = float(d.sensordata[a])
            mism += got != ref; matched += ref; found_pos += got > 0
    return dict(sensors=len(rows), name_mismatch=bad[:10], n_bad=len(bad), states=n_states, value_mismatch=int(mism), contacts_matched=int(matched), slots_found=int(found_pos),
                ok=not bad and len(rows) == 2 * 8 * 32 and mism == 0 and matched > 0)

def _anc(m, b):
    while True:
        yield b
        if b == 0: return
        b = int(m.body_parentid[b])

def check_model(src=SRC, hulls_list=("mix", "all256", "full"), solvers=None):
    import fullbody_task as FT
    solvers = solvers or FT.SOLVER_FALLBACK; res = dict(stub=stub_mjlab(src), wheel=wheel_identity(src), runs=[]); rp_full = replica_model("full", FT.SOLVER_FALLBACK[0])
    for h, s in [(h, FT.SOLVER_FALLBACK[0]) for h in hulls_list] + [("mix", s) for s in solvers[1:]]:
        mj, sc = mjlab_model(h, s); rp = replica_model(h, s); bad, cnt = diff(mj, rp); fl = mj.geom(FT.FLOOR).id; zm = zone_maps(mj, rp, rp_full)
        r = dict(hulls=h, solver=list(s), counts=cnt, mismatches=[list(map(str, x)) for x in bad[:50]], n_mismatch=len(bad),
                 floor=dict(contype=int(mj.geom_contype[fl]), conaffinity=int(mj.geom_conaffinity[fl])), opt=dict(timestep=mj.opt.timestep, integrator=int(mj.opt.integrator),
                 cone=int(mj.opt.cone), impratio=mj.opt.impratio, ccd_iterations=int(mj.opt.ccd_iterations), iterations=int(mj.opt.iterations), ls_iterations=int(mj.opt.ls_iterations)),
                 sizes=dict(nq=mj.nq, nv=mj.nv, nu=mj.nu, ngeom=mj.ngeom, nbody=mj.nbody, nmesh=mj.nmesh, nsensor=mj.nsensor),
                 sensors=sorted({mj.sensor(i).name.split("/")[0] for i in range(mj.nsensor) if mj.sensor(i).name.startswith("fb_")}), zones=zm)
        r["ok"] = (len(bad) == 0 and r["floor"] == dict(contype=1, conaffinity=3) and mj.opt.ccd_iterations == 35 and abs(mj.opt.timestep - 0.005) < 1e-15
                   and (mj.opt.iterations, mj.opt.ls_iterations) == tuple(s) and mj.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST and mj.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
                   and mj.opt.impratio == 1.0 and cnt["geoms"] > 0 and zm["ok"])
        res["runs"].append(r)
        if h == "mix" and s == FT.SOLVER_FALLBACK[0]:
            res["zones_negative_control"] = {k: v for k, v in zone_maps(mj, old_names_replica(h, s), rp_full).items()}   # the pre-A10 names must fail
            res["dr"] = dr_check(mj); res["shells"] = dict(defender=len(FT.shell_tables(mj, FT.ENT)["geoms"]), opponent=len(FT.shell_tables(mj, FT.OPP)["geoms"])); res["sensors"] = sensor_check(sc, mj); res["found"] = found_check(sc, mj)
    m0, _ = mjlab_model("mix", FT.SOLVER_FALLBACK[0], edit=False); f0 = m0.geom(FT.FLOOR).id; bad0, _ = diff(m0, replica_model("mix", FT.SOLVER_FALLBACK[0]))
    res["floor_without_scene_edit"] = dict(contype=int(m0.geom_contype[f0]), conaffinity=int(m0.geom_conaffinity[f0]))
    res["negative_control_without_scene_edit"] = dict(mismatches=len(bad0), fields=sorted({b[2] for b in bad0}))   # the diff is not vacuous
    res["ok"] = not res["zones_negative_control"]["ok"] and res["wheel"]["ok"] and all(r["ok"] for r in res["runs"]) and res["dr"]["ok"] and res["sensors"]["ok"] and res["found"]["ok"] and res["negative_control_without_scene_edit"]["mismatches"] > 0 and res["shells"] == dict(defender=32, opponent=32) and res["floor_without_scene_edit"] == dict(contype=1, conaffinity=1)
    return res

# ---- (2) inputs
def check_inputs():
    import fb_seeds, fullbody_task as FT
    import export_fb_motion as EM
    sha = EM.sha256; fb = ROOT / "gpu" / "motions" / "fb"; out = {}   # the manifests' own sha256 (text LF-normalised: a CRLF checkout and the pod's LF one agree)
    for f, man in (("fb_motion.npz", "fb_motion_manifest.json"), ("fb_l1.npz", "fb_l1_manifest.json"), ("fb_l1_loco.npz", "fb_l1_loco_manifest.json"), ("fb_hulls.npz", "fb_hulls_manifest.json"),
                   ("fb_l1_track.csv", "fb_l1_manifest.json"), ("fb_l1_track_index.json", "fb_l1_manifest.json"), ("fb_l1_track_loco.csv", "fb_l1_loco_manifest.json"), ("fb_l1_track_loco_index.json", "fb_l1_loco_manifest.json")):
        want = json.load(open(fb / man, encoding="utf-8"))["outputs"].get(f); got = sha(fb / f); out[f] = dict(manifest=man, sha256=got[:16], ok=got == want)
    used = sorted({str(p.relative_to(ROOT)).replace("\\", "/") for p in FT.task_input_files()})
    unlisted = [u for u in used if u.startswith("gpu/motions/fb/") and Path(u).name not in out]
    hits = fb_seeds.scan(fb_seeds.task_inputs()); return dict(files=out, task_inputs=used, unlisted_motion_inputs=unlisted, banned_hits=[list(h) for h in hits],
                                                                n_scanned=len(fb_seeds.task_inputs()), manifest_ok_fn=bool(FT.manifest_ok(False) and FT.manifest_ok(True)),
                                                                ok=all(v["ok"] for v in out.values()) and not unlisted and not hits and FT.manifest_ok(False) and FT.manifest_ok(True))

# ---- (3) WATCH6 replays
def check_watch(logs=None):
    import fb_watch as WT
    return WT.replays(logs)

# ---- (4) pod script
def check_pod():
    sh = ROOT / "gpu" / "pod_fb.sh"; txt = sh.read_text(encoding="utf-8"); bash = os.environ.get("FB_BASH", r"C:\Program Files\Git\bin\bash.exe")
    lint = subprocess.run([bash, "-n", str(sh)], capture_output=True, text=True)
    need = dict(sim_options="assert_sim", tf32="precision_ok", md5="md5sum -c", versions="mujoco-warp", model_diff="fb_m9_check.py", taskset="taskset -c",
                stages=" P0 ", watch="fb_watch.py", fake_gpu="FB_FAKE_GPU", blind="--blind", start="fullbody_task.py start")
    present = {k: v in txt for k, v in need.items()}
    import tempfile; tmp = tempfile.mkdtemp(prefix="fb_pod_").replace("\\", "/")
    env = {**os.environ, "FB_FAKE_GPU": "1", "FB_DRY": "1", "FB_CPUS": "16", "FB_REPO": str(ROOT).replace("\\", "/"), "TMPDIR": tmp}
    dry = subprocess.run([bash, str(sh), "plan"], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    out = dry.stdout; pin = [l for l in out.splitlines() if l.startswith("PIN ")]
    train = {int(c) for l in pin if " train " in l for c in _cpus(l.split()[-1])}; bg = {int(c) for l in pin if " bg " in l for c in _cpus(l.split()[-1])}
    res = dict(lint_rc=lint.returncode, lint_err=lint.stderr[-500:], present=present, dry_rc=dry.returncode, dry_tail=out[-1500:], dry_err=dry.stderr[-800:],
               pin_train=sorted(train), pin_bg=sorted(bg), stages=[l for l in out.splitlines() if l.startswith("STAGE ")])
    res["ok"] = lint.returncode == 0 and all(present.values()) and dry.returncode == 0 and train and bg and not (train & bg) and "DRY OK" in out
    return res

def _cpus(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b or a) + 1))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--src", default=str(SRC)); ap.add_argument("--logs", default=str(RUN3))
    ap.add_argument("--only", default="model,inputs,watch,pod"); a = ap.parse_args(); t0 = time.time(); res = {}
    import torch; torch.set_num_threads(2)
    for k in a.only.split(","):
        t = time.time(); res[k] = dict(model=lambda: check_model(Path(a.src)), inputs=check_inputs, watch=lambda: check_watch(a.logs), pod=check_pod)[k](); res[k]["wall_s"] = round(time.time() - t, 1)
    res["ok"] = all(res[k]["ok"] for k in a.only.split(",")); res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8"); print(json.dumps({k: (v["ok"] if isinstance(v, dict) else v) for k, v in res.items()}))
    sys.exit(0 if res["ok"] else 1)

if __name__ == "__main__": main()
