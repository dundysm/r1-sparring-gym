"""Full-body M2 v2 M6 CPU replica (scripts/fb_replica.py; M2V2_PLAN_v3 3 M6, A4.3): the switches build what they say (reduced hulls
are exactly fb_hulls' hulls on both robots' shells and change nothing else, the solver iterations, defender self-collision, native contact
sensors whose found / force equal fb_hits.MjContactSensors and whose netforce pos is the force-weighted contact centroid, 0 without
force); in eval mode one 1 s bout (60100, G, role 'identity') is bit identical to eval_opponent.run; the torch float64 modules with the
eval's own tracker session and reference arrays keep qpos bit-equal to it before the first shell contact, while the deployed float64
driver (TrackerMLP, fb_motion's reference) moves the opponent by > 1e-9 at the first step (M6a's impossible-as-specified finding); the
training reset is deterministic per seed; the account scorer's rows carry the account and the HitMeter; roles are seed-guarded.
The full checks are scripts/fb_m6_check.py."""
import sys, json
from pathlib import Path
import numpy as np, pytest, mujoco
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts", ROOT / "tests"): sys.path.insert(0, str(p))
import fb_replica as RP, fb_hits as HT
from fb_seeds import SeedRefused

@pytest.fixture(scope="module")
def ccfg(): return RP.world("G", RP.conf())[1]

@pytest.fixture(scope="module")
def m_eval(ccfg): return RP.build_model(ccfg, RP.conf())[0]

def test_conf_validates():
    assert RP.conf() == dict(RP.EVAL, solver=(100, 50)) and RP.conf(solver=[10, 20])["solver"] == (10, 20)
    for bad in (dict(hulls="x"), dict(backend="numpy", reset="train"), dict(backend="numpy", actor="onnx"), dict(scorer="both")):
        with pytest.raises(AssertionError): RP.conf(**bad)
    assert RP.FALLBACK["solver"][0] == (10, 20) and RP.FALLBACK["solver"][-1] == (100, 50) and RP.FALLBACK["hulls"] == ("mix", "all256", "full")

def test_reduced_hulls_and_switches(ccfg, m_eval):
    import export_fb_hulls as XH
    m, _ = RP.build_model(ccfg, RP.conf(hulls="mix", solver=(10, 20))); assert (m.opt.iterations, m.opt.ls_iterations, m.opt.ccd_iterations) == (10, 20, m_eval.opt.ccd_iterations)
    assert (m_eval.opt.iterations, m_eval.opt.ls_iterations, m_eval.opt.solver) == (100, 50, mujoco.mjtSolver.mjSOL_NEWTON)
    for f in ("body_mass", "body_inertia", "body_ipos", "geom_contype", "geom_conaffinity", "geom_condim", "geom_solref", "geom_solimp", "geom_friction", "actuator_gainprm", "dof_armature"):
        assert np.array_equal(getattr(m, f), getattr(m_eval, f)), f
    P = RP.hull_points("mix", m_eval); G = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g], m.geom_conaffinity[g]) in ((0, 4), (4, 0))]
    assert len(G) == 64
    for g in G[::5]:
        name = m.mesh(m.geom_dataid[g]).name; base = name[len("fbhull_"):]; base = base[4:] if base.startswith("opp_") else base
        V, _ = XH.hull(m, g); W, _ = XH.hull_of(P[base]); d = np.abs(np.sort(V.round(9), 0) - np.sort(W.round(9), 0)).max(); assert len(V) == len(W) and d < 1e-9, (name, d)
    m6, _ = RP.build_model(ccfg, RP.conf(self_collision=True)); ch = [g for g in range(m6.ngeom) if m6.geom_contype[g] != m_eval.geom_contype[g] or m6.geom_conaffinity[g] != m_eval.geom_conaffinity[g]]
    assert ch and all((m6.geom_contype[g], m6.geom_conaffinity[g]) == (8, 9) and (m_eval.geom_contype[g], m_eval.geom_conaffinity[g]) == (0, 1) and not m6.body(m6.geom_bodyid[g]).name.startswith("opp_") for g in ch)

def test_native_sensors_found_force_pos(ccfg, m_eval):
    from test_fb_hits import _contact_states
    M, names = RP.build_model(ccfg, RP.conf(), sensors=True); s = HT.MjContactSensors(M); d = mujoco.MjData(M); f6 = np.zeros(6); n = 0
    for q, v in _contact_states(m_eval, n=10, seed=3):
        d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(M, d); F, fo = s.read(d)[:2]
        for l, k in enumerate(HT.SENSORS):
            for p, nm in enumerate(names):
                assert d.sensor(f"{k}/{nm}/found").data[0] == fo[l, p] and np.array_equal(d.sensor(f"{k}/{nm}/force").data, F[l, p])
                if not fo[l, p]: continue
                cs = [i for i in range(d.ncon) if s.prim[p] in (d.contact.geom1[i], d.contact.geom2[i]) and s.limb[d.contact.geom2[i] if d.contact.geom1[i] == s.prim[p] else d.contact.geom1[i]] == l]
                fn = np.array([mujoco.mj_contactForce(M, d, i, f6) or abs(f6[0]) for i in cs]); pos = d.sensor(f"{k}/{nm}/pos").data
                if fn.sum() > 0: assert np.allclose(pos, (fn[:, None] * d.contact.pos[cs]).sum(0) / fn.sum(), atol=1e-9); n += 1
                else: assert not pos.any()   # no net force: pos 0 (A4.3: such contacts are not within 1 cm)
    assert n >= 3

def test_eval_mode_is_eo_and_f64_diagnostics():
    import eval_opponent as EO, fb_r0eo as R0, fb_m6_check as C6
    eps = RP.episodes("identity", [60100]); c, cfg, m, lag = R0.cell("G"); ref = EO.run(c, eps, m, cfg, 1.0, threads=1, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")
    rows = RP.run("G", eps, RP.conf(), 1.0); assert json.dumps(rows[0]["eo"], sort_keys=True) == json.dumps(ref[0], sort_keys=True) and rows[0]["act_delay"] in (0, 1)
    Q = {}
    for tag, kw in (("numpy", {}), ("deployed", dict(backend="torch64")), ("diag", dict(backend="torch64", actor="onnx", ref="eval"))):
        cc = RP.conf(**kw); ctrl, ccfg, lag, mm, _ = RP.world("G", cc); B = RP.Bout(ctrl, ccfg, cc, mm, seconds=0.2, lag=lag); tr = C6.Trace(); RP.run_bout(B, eps[0], tr); Q[tag] = (np.array(tr.q), tr.first)
    k = min(x for x in (Q["numpy"][1], Q["diag"][1], len(Q["numpy"][0])) if x is not None); assert k >= 2
    assert np.array_equal(Q["numpy"][0][:k], Q["diag"][0][:k])          # eval's actor and reference in the torch driver: bit-equal before contact
    assert np.abs(Q["numpy"][0][1] - Q["deployed"][0][1]).max() > 1e-9   # the float64 MLP vs the eval's float32 ONNX (and fb_motion's anchors)

def test_train_reset_deterministic_and_account_rows():
    eps = RP.episodes("identity", [60102])
    a = RP.run("G", eps, RP.conf(backend="torch64", reset="train"), 0.1); b = RP.run("G", eps, RP.conf(backend="torch64", reset="train"), 0.1)
    assert (a[0]["t0"], a[0]["k"], a[0]["act_delay"]) == (b[0]["t0"], b[0]["k"], b[0]["act_delay"]) and a[0]["eo"] == b[0]["eo"]
    r = RP.run("G", eps, RP.conf(scorer="account", vmode="site"), 0.5)[0]; assert set(r["acct"]["n"]) >= {"head_clean", "head_landed"} and "n" in r["hm"]

def test_roles_are_guarded():
    for role, s in (("identity", 60116), ("identity", 21000), ("hit_scoring", 60100), ("eq", 22000), ("r2_diag", 68000)):
        with pytest.raises((SeedRefused, KeyError)): RP.episodes(role, [s])

def test_aim_skip_idle_changes_nothing():
    """AimOp(skip_idle) (the replica's: no IK on steps without a live warp) gives the same opponent trajectory as the full path."""
    import fb_m6_check as C6
    eps = RP.episodes("identity", [60103]); cc = RP.conf(backend="torch64"); ctrl, ccfg, lag, mm, _ = RP.world("G", cc); Q = []
    for skip in (True, False):
        B = RP.Bout(ctrl, ccfg, cc, mm, seconds=1.5, lag=lag); B.aim.skip_idle = skip; tr = C6.Trace(); RP.run_bout(B, eps[0], tr); Q.append(np.array(tr.q))
    assert np.array_equal(Q[0], Q[1]) and len(Q[0]) == 76

def test_sensor_read_equals_contact_sensors(ccfg, m_eval):
    """Sensors._read (A7.1's leaner loop) = MjContactSensors.read on F, found, Fb, vb, and on the velocities it computes; _vel = np.cross's."""
    from test_fb_hits import _contact_states
    S = RP.Sensors(m_eval, ("site",)); S2 = RP.Sensors(m_eval, ("com", "point")); d = mujoco.MjData(m_eval); n = 0
    for q, v in _contact_states(m_eval, n=10, seed=4):
        d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(m_eval, d); ref = S.s.read(d); a, b = S._read(d), S2._read(d); n += int(ref[1].sum() > 0)
        for k in (0, 1, 4, 5): assert np.array_equal(a[k], ref[k]) and np.array_equal(b[k], ref[k])
        assert np.array_equal(b[2], ref[2]) and np.array_equal(b[3], ref[3])
        for bd in (3, 17):
            p = d.xpos[bd] + 0.1; mujoco.mj_objectVelocity(m_eval, d, mujoco.mjtObj.mjOBJ_BODY, bd, S.v6, 0); w = S.v6.copy()
            assert np.array_equal(S._vel(d, bd, p), w[3:] + np.cross(w[:3], p - d.xipos[bd]))
    assert n >= 3

def test_batch_rows_equal_n1_rowwise():
    """Batch (K bouts in lockstep, A7.1) with the tracker MLP row by row gives the N = 1 rows bit for bit (account scorer, torch32, 1 s);
    finished bouts keep the batch running (one bout falls or not, rows equal either way)."""
    import fb_m6_run as D
    c = RP.conf(**D.REQ_CONF); eps = RP.episodes("identity", [60104, 60105, 60106])
    a = RP.run("G", eps, c, 1.0); b = RP.run("G", eps, c, 1.0, batch=3, rowwise=True); c2 = RP.run("G", eps[:2], c, 1.0, batch=2)
    J = lambda r: json.dumps(r, sort_keys=True); assert [J(x) for x in a] == [J(x) for x in b] and len(c2) == 2 and all("acct" in r for r in b)

def test_rowwise_face_is_the_drivers_face():
    """Batch's rowwise face = TrackerDriver.face at n = 1, row by row and over consecutive steps (yaw set and unset), bit for bit."""
    import torch, fb_opponent as FO
    g = torch.Generator().manual_seed(0); n = 37; A = FO.TrackerDriver(n, torch.float32); A.face = RP.rowwise_face(A); ones = [FO.TrackerDriver(1, torch.float32, motion=A.M) for _ in range(n)]
    for _ in range(4):
        t, p = torch.randn(n, 2, generator=g), torch.randn(n, 2, generator=g); y = A.face(t, p)
        assert torch.equal(y, torch.cat([d.face(t[i:i + 1], p[i:i + 1]) for i, d in enumerate(ones)]))
