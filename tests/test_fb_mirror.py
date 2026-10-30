"""Mirror conjugation of the opponent's tracker (M2V2_PLAN_v3 A6.2; gpu/fb_mirror.py): the joint map = export_fb_library's, the maps
are exact involutions, the qpos mirror is the world reflection, the driver's obs in a mirrored world = M_o(obs) and the conjugated
tracker's action there = M_a(action) on synthetic two-robot states, the per-env mask, re-wrapping. The same proof on all 15,412
recorded eval rows runs in scripts/fb_m8_check.py (5) (here with FB_REC set to the M2 recording)."""
import os, sys
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import fb_mirror as FM, fb_opponent as FO, export_fb_library as L
from r1gym.model import CTRL_JOINTS, build_spec, load_cfg

def test_joint_map_is_the_librarys_and_the_maps_are_involutions():
    assert FM.JOINTS == tuple(CTRL_JOINTS); p, s = L.mirror_map(); assert np.array_equal(FM.JP, p) and np.array_equal(FM.JS, s)
    for ps, n in ((FM.OBS, 132), (FM.ACT, 24), (FM.QPOS, 31), (FM.QVEL, 30)):
        x = torch.randn(5, n, dtype=torch.float64); assert torch.equal(FM._apply(FM._apply(x, ps), ps), x) and sorted(ps[0]) == list(range(n)) and set(np.abs(ps[1])) == {1.0}
    x = torch.randn(3, 132, dtype=torch.float32); assert FM.mirror_obs(x).dtype == torch.float32 and FM.mirror_qpos(np.zeros((2, 31), np.float32)).dtype == np.float32
    assert np.array_equal(FO.DEFAULT, FM.mirror_joints(np.array(FO.DEFAULT))) and np.array_equal(FO.SCALE, np.abs(FM.mirror_joints(np.array(FO.SCALE))))   # targets mirror with actions

def test_qpos_mirror_is_the_librarys_reflection():
    q = L.base_motion()["kfab_203"]; assert np.array_equal(FM.mirror_qpos(q), L.mirror(q, clip=False))
    ref = tuple(np.asarray(x, np.float32) for x in (q[:5, 7:], q[:5, 7:] * 2, q[:5, :3], q[:5, 3:7])); m = FM.mirror_ref(*ref)
    assert np.array_equal(m[0], FM.mirror_joints(ref[0])) and np.array_equal(m[2], ref[2] * np.float32([1, -1, 1])) and np.array_equal(m[3], ref[3] * np.float32([1, -1, 1, -1])) and m[3].dtype == np.float32

def _states(n=48, seed=0):
    """n two-robot states: the opponent on fb_motion frames 0.45 m away, perturbed, the defender perturbed at home; random qvel,
    reference, driver state."""
    rng = np.random.default_rng(seed); Q = np.concatenate(list(L.base_motion().values())); m = build_spec(opponent=True, cfg=load_cfg()).compile()
    qa = int(m.jnt_qposadr[m.joint("opp_floating_base_joint").id]); qp = np.tile(m.qpos0, (n, 1)); j = rng.integers(len(Q), size=n)
    qp[:, qa:qa + 31] = Q[j]; qp[:, qa] += 0.45; qp[:, qa + 7:] += rng.normal(0, 0.05, (n, 24)); qp[:, 7:31] += rng.normal(0, 0.1, (n, 24)); qp[:, :2] += rng.normal(0, 0.05, (n, 2))
    qv = rng.normal(0, 0.5, (n, m.nv)); ref = Q[rng.integers(len(Q), size=n)]
    return m, qa, qp, qv, dict(rq=ref[:, 7:], rdq=rng.normal(0, 1, (n, 24)), apos=ref[:, :3] + [0, 0, 0.2], aori=ref[:, 3:7], yaw=rng.uniform(-3, 3, n),
                               spot=np.c_[0.45 + rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)], axis=np.c_[np.ones(n), rng.normal(0, 0.1, n)], last=rng.normal(0, 1, (n, 24)), dist=0.45 + 0 * rng.random(n))

def _world(m, qa, qp, qv, st, dtype, mirrored):
    d = mujoco.MjData(m); n = len(qp); va = qa - 1; V = np.array([1.0, -1.0, 1.0])
    if mirrored:
        qp, qv = qp.copy(), qv.copy()
        for a, v in ((0, 0), (qa, va)): qp[:, a:a + 31] = FM.mirror_qpos(qp[:, a:a + 31]); qv[:, v:v + 30] = FM.mirror_qvel(qv[:, v:v + 30])
        st = dict(st, rq=FM.mirror_joints(st["rq"]), rdq=FM.mirror_joints(st["rdq"]), apos=st["apos"] * V, aori=st["aori"] * [1, -1, 1, -1], yaw=-st["yaw"], spot=st["spot"] * V[:2], axis=st["axis"] * V[:2], last=FM.mirror_joints(st["last"]))
    K = {k: np.zeros((n, w)) for k, w in (("tp", 3), ("tq", 4), ("pp", 3), ("dp", 3))}
    for i in range(n):
        d.qpos[:] = qp[i]; mujoco.mj_kinematics(m, d)
        for k, b in (("tp", "opp_torso_link"), ("pp", "opp_pelvis"), ("dp", "pelvis")): K[k][i] = d.xpos[m.body(b).id]
        K["tq"][i] = d.xquat[m.body("opp_torso_link").id]
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64).to(dtype); drv = FO.TrackerDriver(n, dtype)
    if mirrored: FM.conjugate(drv)
    drv.yaw, drv.spot, drv.axis, drv.last = t(st["yaw"]), t(st["spot"]), t(st["axis"]), t(st["last"]); drv.yaw_set[:] = True; drv.axis_set[:] = True
    drv.face(t(K["dp"][:, :2]), t(K["pp"][:, :2])); drv.steer(t(K["dp"][:, :2]), t(st["dist"]))
    o = drv.observation(t(st["rq"]), t(st["rdq"]), t(st["apos"]), t(st["aori"]), t(K["tp"]), t(K["tq"]), t(qv[:, va + 3:va + 6]), t(qp[:, qa + 7:qa + 31]), t(qv[:, va + 6:va + 30]))
    return o, drv.mlp(o), drv

def test_obs_and_action_conjugation_on_synthetic_states():
    m, qa, qp, qv, st = _states(); assert int(m.jnt_dofadr[m.joint("opp_floating_base_joint").id]) == qa - 1
    for dt, tol in ((torch.float64, 1e-9), (torch.float32, 1e-5)):
        o, a, w = _world(m, qa, qp, qv, st, dt, False); om, am, wm = _world(m, qa, qp, qv, st, dt, True)
        assert float((om - FM.mirror_obs(o)).abs().max()) <= tol and float((am - FM.mirror_act(a)).abs().max()) <= tol, dt
        assert float(FO.wrap(wm.yaw + w.yaw).abs().max()) <= tol and torch.allclose(wm.spot, w.spot * torch.tensor([1.0, -1.0], dtype=dt), atol=tol)

def test_mask_per_env_and_rewrap():
    drv = FO.TrackerDriver(4, torch.float64); mlp = drv.mlp; o = torch.randn(4, 132, dtype=torch.float64)
    pol = FM.conjugate(drv, [True, False, True, False]); a = drv.mlp(o); eq = lambda x, y: torch.allclose(x, y, rtol=0, atol=1e-12)   # rows independent up to BLAS blocking
    assert eq(a[[0, 2]], FM.mirror_act(mlp(FM.mirror_obs(o[[0, 2]])))) and eq(a[[1, 3]], mlp(o[[1, 3]])) and not eq(a[[0, 2]], mlp(o[[0, 2]]))
    pol.set([1], True); assert eq(drv.mlp(o)[1], FM.mirror_act(mlp(FM.mirror_obs(o[1:2])))[0])
    pol.set(slice(None), False); assert torch.equal(drv.mlp(o), mlp(o))
    FM.conjugate(drv); assert drv.mlp.policy is mlp and torch.equal(drv.mlp(o), FM.mirror_act(mlp(FM.mirror_obs(o))))   # no double wrap
    assert drv.mlp.dtype == torch.float64

@pytest.mark.skipif(not os.environ.get("FB_REC"), reason="FB_REC not set (the M2 recording fb_rec_60000-60015.npz lives outside the repo)")
def test_conjugation_on_the_recorded_rows():
    import fb_m8_check as K
    r = K.check_conjugation(os.environ["FB_REC"]); assert r["ok"] and r["fk_mirror"] == 0.0 and r["float64"]["obs"] <= 1e-9, r
