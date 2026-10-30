"""Full-body M2 v2 domain randomisation of the defender (M2V2_PLAN_v3 2.3 m7, build order M4): the eval's ranges (ENV:187-199),
written the way the eval writes them, straight into the model fields with no set_const. Pure torch + numpy.

The eval (R1HitEnv._dr_sample_into_model, then _dr_apply before every step) resets each field to its nominal value and then:
mass x U(0.85, 1.15) per defender body, torso + U(-0.5, 2.0) kg, pelvis and torso COM (body_ipos) + U(-2.5, 2.5) cm per axis, one
foot slide friction U(0.5, 1.25) for all defender foot geoms, kp and kd x U(0.8, 1.2) per actuator (gainprm[0], biasprm[1] by kp,
biasprm[2] by kd), force caps x U(0.8, 1.05), armature x U(0.5, 1.5) per defender dof. It never calls mj_setConst, so
dof_invweight0, body_invweight0, body_subtreemass and the constraint regularisation built from them stay nominal; mjlab's
dr.body_mass / dr.body_com_offset would recompute them (set_const), so this writer replaces them. The opponent is never touched.
act_delay is drawn by fb_events.fb_draw_latency, not here (the eval draws it last from the same stream: sample_np consumes it).

DRIndex       defender index sets from a compiled model, as R1HitEnv._dr_init: bodies (sorted), pelvis, torso, foot geoms, dofs,
              actuators in CTRL_JOINTS order.
sample        torch draws for N envs; sample_np: one env's draws from a numpy Generator in the eval's order (replay checks).
write         fields: {name: tensor [N, ...]} (mjlab's per-world model fields, or an MjModel's arrays wrapped as N = 1); nominal:
              {name: tensor [...]}; writes only FIELDS, only for envs ids."""
import numpy as np, torch

RANGES = dict(mass=(0.85, 1.15), torso_kg=(-0.5, 2.0), com_m=(-0.025, 0.025), foot_mu=(0.5, 1.25), kp=(0.8, 1.2), kd=(0.8, 1.2), cap=(0.8, 1.05), armature=(0.5, 1.5))
FIELDS = ("body_mass", "body_ipos", "geom_friction", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "dof_armature")

class DRIndex:
    def __init__(self, m, ctrl_joints, is_defender=lambda name: not name.startswith("opp_"), prefix=""):
        nb = [m.body(b).name for b in range(m.nbody)]; r1 = {b for b in range(1, m.nbody) if is_defender(nb[b])}
        self.bodies = np.array(sorted(r1)); self.pel, self.torso = m.body(prefix + "pelvis").id, m.body(prefix + "torso_link").id
        self.feet = np.array([g for g in range(m.ngeom) if "foot" in (m.geom(g).name or "") and m.geom_bodyid[g] in r1], dtype=np.int64)
        self.dofs = np.flatnonzero(np.isin(m.dof_bodyid, list(r1))); self.act = np.array([m.actuator(prefix + j.replace("_joint", "")).id for j in ctrl_joints])
        self.nb, self.nd, self.na = len(self.bodies), len(self.dofs), len(self.act)

def _u(lo_hi, shape, gen, dtype, device): lo, hi = lo_hi; return lo + (hi - lo) * torch.rand(shape, dtype=dtype, device=device, generator=gen)

def sample(n, ix, generator=None, dtype=torch.float64, device="cpu"):
    """Draws for n envs: {mass [n, nb], torso_kg [n], com_m [n, 2, 3], foot_mu [n], kp, kd, cap [n, na], armature [n, nd]}."""
    R, g = RANGES, generator
    return dict(mass=_u(R["mass"], (n, ix.nb), g, dtype, device), torso_kg=_u(R["torso_kg"], (n,), g, dtype, device), com_m=_u(R["com_m"], (n, 2, 3), g, dtype, device),
                foot_mu=_u(R["foot_mu"], (n,), g, dtype, device), kp=_u(R["kp"], (n, ix.na), g, dtype, device), kd=_u(R["kd"], (n, ix.na), g, dtype, device),
                cap=_u(R["cap"], (n, ix.na), g, dtype, device), armature=_u(R["armature"], (n, ix.nd), g, dtype, device))

def sample_np(rng, ix):
    """One env's draws from a numpy Generator in _dr_sample_into_model's order (n = 1 tensors); also returns its act_delay draw."""
    R = RANGES; p = dict(mass=rng.uniform(*R["mass"], ix.nb), torso_kg=rng.uniform(*R["torso_kg"]), com_m=rng.uniform(*R["com_m"], (2, 3)))
    p["foot_mu"] = rng.uniform(*R["foot_mu"]) if len(ix.feet) else np.nan
    p.update(kp=rng.uniform(*R["kp"], ix.na), kd=rng.uniform(*R["kd"], ix.na), cap=rng.uniform(*R["cap"], ix.na), armature=rng.uniform(*R["armature"], ix.nd))
    act_delay = bool(rng.random() < 0.5); return {k: torch.as_tensor(np.asarray(v, np.float64))[None] for k, v in p.items()}, act_delay

def write(fields, nominal, ix, p, ids=None):
    """Nominal, then the defender's draws p (rows aligned with ids), for envs ids (default all); the eval's order of operations."""
    f = fields; n = f["body_mass"].shape[0]; ids = torch.arange(n) if ids is None else torch.as_tensor(ids, dtype=torch.long)
    for k in FIELDS: f[k][ids] = nominal[k].to(f[k].dtype)
    b, a, dv = (torch.as_tensor(x, dtype=torch.long) for x in (ix.bodies, ix.act, ix.dofs)); I = ids[:, None]
    f["body_mass"][I, b] = f["body_mass"][I, b] * p["mass"]; f["body_mass"][ids, ix.torso] = f["body_mass"][ids, ix.torso] + p["torso_kg"]
    pt = torch.as_tensor([ix.pel, ix.torso]); f["body_ipos"][I, pt] = f["body_ipos"][I, pt] + p["com_m"]
    if len(ix.feet): f["geom_friction"][I, torch.as_tensor(ix.feet), 0] = p["foot_mu"][:, None].to(f["geom_friction"].dtype)
    f["actuator_gainprm"][I, a, 0] = f["actuator_gainprm"][I, a, 0] * p["kp"]; f["actuator_biasprm"][I, a, 1] = f["actuator_biasprm"][I, a, 1] * p["kp"]
    f["actuator_biasprm"][I, a, 2] = f["actuator_biasprm"][I, a, 2] * p["kd"]; f["actuator_forcerange"][I, a] = f["actuator_forcerange"][I, a] * p["cap"][..., None]
    f["dof_armature"][I, dv] = f["dof_armature"][I, dv] * p["armature"]

def model_fields(m):
    """An MjModel's FIELDS as N = 1 torch views (writes go into the model) and their nominal copies."""
    F = {k: torch.from_numpy(getattr(m, k))[None] for k in FIELDS}; return F, {k: v[0].clone() for k, v in F.items()}
