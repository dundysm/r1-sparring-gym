"""Full-body M2 v2 defender DR (M2V2_PLAN_v3 M4 G0-9, m7; gpu/fb_dr.py): with the eval's draws the writer equals
R1HitEnv._dr_sample_into_model field for field (act_delay included), the index sets are the eval's, torch draws stay in ENV:187-199's
ranges, a write changes only FIELDS and only defender entries (per-world fields: other envs untouched), and no set_const runs:
dof_invweight0 / body_invweight0 / body_subtreemass stay nominal, unlike after mj_setConst. The recorded-state gate (constraint
regularisation on a contact state) runs in scripts/fb_m4_check.py."""
import sys, copy
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_dr as DR, fb_m4_check as C
from r1gym.env import R1HitEnv, _DR_NOMINAL
from r1gym.model import CTRL_JOINTS

@pytest.fixture(scope="module")
def eval_env():
    m, cfg, _, _ = C.model_cfg(); me = copy.deepcopy(m); e = R1HitEnv(seed=0, cfg=cfg, model=me, train=False, dr=True)
    yield m, e; _DR_NOMINAL.pop(id(me), None)

def test_index_sets_are_the_evals(eval_env):
    m, e = eval_env; ix = DR.DRIndex(m, CTRL_JOINTS)
    assert np.array_equal(ix.bodies, e._r1b) and np.array_equal(ix.feet, e._feet) and np.array_equal(ix.dofs, e._r1dof) and np.array_equal(ix.act, e.r.act) and len(ix.feet) > 0

def test_writer_with_the_evals_draws_equals_the_eval(eval_env):
    m, e = eval_env; ix = DR.DRIndex(m, CTRL_JOINTS); mo = copy.deepcopy(m); F, nom = DR.model_fields(mo)
    for s in range(25):
        e.rng = np.random.default_rng(s); e._dr_sample_into_model(); p, ad = DR.sample_np(np.random.default_rng(s), ix); DR.write(F, nom, ix, p)
        assert all(np.array_equal(getattr(mo, k), getattr(e.m, k)) for k in DR.FIELDS) and ad == bool(e.act_delay), s

def test_torch_draws_inside_the_ranges():
    ix = DR.DRIndex(C.model_cfg()[0], CTRL_JOINTS); P = DR.sample(20000, ix, generator=torch.Generator().manual_seed(0))
    for k, (lo, hi) in DR.RANGES.items(): assert lo <= float(P[k].min()) < lo + 0.01 * (hi - lo) and hi - 0.01 * (hi - lo) < float(P[k].max()) <= hi, k
    assert P["mass"].shape == (20000, ix.nb) and P["com_m"].shape == (20000, 2, 3) and P["armature"].shape == (20000, ix.nd) and P["kp"].shape == (20000, 24)

def test_only_defender_entries_change_and_no_set_const(eval_env):
    m, _ = eval_env; ix = DR.DRIndex(m, CTRL_JOINTS); mo = copy.deepcopy(m); F, nom = DR.model_fields(mo); DR.write(F, nom, ix, DR.sample(1, ix, generator=torch.Generator().manual_seed(1)))
    A0, A1 = C._arrays(m), C._arrays(mo); changed = {k for k in A0 if not np.array_equal(A0[k], A1[k])}; assert changed == set(DR.FIELDS), changed
    own = dict(body_mass=ix.bodies, body_ipos=ix.bodies, geom_friction=ix.feet, actuator_gainprm=ix.act, actuator_biasprm=ix.act, actuator_forcerange=ix.act, dof_armature=ix.dofs)
    for k in DR.FIELDS: assert set(np.flatnonzero((A0[k] != A1[k]).reshape(len(A0[k]), -1).any(1))) <= set(own[k].tolist()), k
    for k in ("dof_invweight0", "body_invweight0", "body_subtreemass", "dof_M0", "actuator_acc0"): assert np.array_equal(getattr(mo, k), getattr(m, k)), k
    mujoco.mj_setConst(mo, mujoco.MjData(mo)); assert not np.array_equal(mo.body_subtreemass, m.body_subtreemass) and not np.array_equal(mo.dof_invweight0, m.dof_invweight0)

def test_per_world_fields_write_only_their_envs():
    m = C.model_cfg()[0]; ix = DR.DRIndex(m, CTRL_JOINTS); _, nom = DR.model_fields(copy.deepcopy(m)); F = {k: v[None].repeat(3, *([1] * v.dim())) for k, v in nom.items()}
    DR.write(F, nom, ix, DR.sample(1, ix, generator=torch.Generator().manual_seed(2)), ids=[1])
    assert all(torch.equal(F[k][j], nom[k]) for k in DR.FIELDS for j in (0, 2)) and not torch.equal(F["dof_armature"][1], nom["dof_armature"])
    opp = [b for b in range(m.nbody) if m.body(b).name.startswith("opp_")]; assert torch.equal(F["body_mass"][1][opp], nom["body_mass"][opp])
