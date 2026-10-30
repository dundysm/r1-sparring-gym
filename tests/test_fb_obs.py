"""Full-body M2 v2 defender observation (M2V2_PLAN_v3 M4 G0-5, G0-6; gpu/fb_obs.py): the keypoint contract = r1gym/keypoints.py,
exact FMA and numpy's matmul / wrap arithmetic, kp_state = KeypointSource.true_state bit for bit on random two-R1 states, the
delayed block with resets = KeypointSource('truth', delay=k), EvalActorObs = DelayedNoisyObs (one lag per episode, all 111 dims,
lazy pre-fill, noise after the delay), proprio83 = R1HitEnv._raw_obs. The recorded-bout gates run in scripts/fb_m4_check.py."""
import sys, types
from fractions import Fraction
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu"))
import fb_obs as OB, fb_events as EV
from r1gym import keypoints as KP
from r1gym.sensors import DelayedNoisyObs
from r1gym.model import build_spec, load_cfg, R1Robot, HOME
from r1gym.env import R1HitEnv
from r1gym import CONFIGS

@pytest.fixture(scope="module")
def world():
    cfg = load_cfg(CONFIGS / "r1_mjlab.yaml"); m = build_spec(opponent=True, cfg=cfg).compile(); return m, cfg, R1Robot(m, cfg), [m.body(n).id for n in OB.body_names()]

def states(m, r, n, seed=0):
    """Random two-R1 states facing each other: the defender at the origin, the opponent 0.3-0.7 m ahead, tilted and turned."""
    rng = np.random.default_rng(seed); Q = np.tile(m.qpos0, (n, 1)); j = m.joint("opp_floating_base_joint").id; oa = m.jnt_qposadr[j]
    for i in range(n):
        for qa, x0, yaw0 in ((r.free_q, 0.0, 0.0), (oa, rng.uniform(0.3, 0.7), np.pi)):
            t, ph, yw = rng.uniform(0, 0.6), rng.uniform(0, 2 * np.pi), yaw0 + rng.uniform(-1.2, 1.2)
            a = np.array([np.cos(t / 2), np.sin(t / 2) * np.cos(ph), np.sin(t / 2) * np.sin(ph), 0.0]); z = np.array([np.cos(yw / 2), 0, 0, np.sin(yw / 2)]); q = np.zeros(4)
            mujoco.mju_mulQuat(q, z, a); Q[i, qa:qa + 7] = [x0 + rng.normal() * 0.05, rng.normal() * 0.1, rng.uniform(0.5, 0.8), *q]
        Q[i, r.qadr] += rng.normal(0, 0.4, 24); Q[i, oa + 7:oa + 31] += rng.normal(0, 0.6, 24)
    return Q

def poses(m, ids6, Q):
    d = mujoco.MjData(m); X, W = np.zeros((len(Q), 6, 3)), np.zeros((len(Q), 6, 4))
    for i, q in enumerate(Q): d.qpos[:] = q; mujoco.mj_kinematics(m, d); X[i], W[i] = d.xpos[ids6], d.xquat[ids6]
    return torch.from_numpy(X), torch.from_numpy(W)

def test_contract_equals_r1gym_keypoints(world):
    m, cfg, _, _ = world
    assert OB.KP_NAMES == KP.KP_NAMES and all(KP.KP_BODIES[k] == ("opp_" + b, o) for k, b, o in zip(KP.KP_NAMES, OB.KP_BODIES, OB.KP_OFF))
    assert np.array_equal(OB.CAM_C, KP.CAM_C) and np.array_equal(OB.CAM_R, KP.CAM_R) and (OB.CAM_F, OB.CAM_CX, OB.CAM_CY, OB.CAM_ZMIN) == (KP.CAM_F, KP.CAM_CX, KP.CAM_CY, KP.CAM_ZMIN)
    assert (OB.N_KP, OB.KP_DIM, OB.PROPRIO_DIM, OB.CTRL_DT, OB.POS, OB.VEL, OB.YAW, OB.MASK) == (KP.N_KP, KP.KP_DIM, KP.PROPRIO_DIM, KP.CTRL_DT, KP.POS, KP.VEL, KP.YAW, KP.MASK)
    assert OB.DEF_BODY == KP.DEF_BODY and KP.OPP_YAW_BODY == "opp_" + OB.KP_BODIES[OB.OPP_YAW_KP] and OB.OBS_DIM == OB.PROPRIO_DIM + OB.KP_DIM
    e = R1HitEnv(cfg=cfg, model=m, train=False); hw = OB.noise_halfwidths(cfg)
    assert e.contract == "mjlab" and np.array_equal(hw[:83], e.noise) and not hw[83:].any()

def test_fma_exact_and_rowmat_is_numpys_matmul():
    rng = np.random.default_rng(0); a, b, c = (rng.normal(size=3000) * 10.0 ** rng.uniform(-4, 2, 3000) for _ in range(3))
    got = OB.fma(*(torch.from_numpy(x) for x in (a, b, c))).numpy()
    assert all(got[i] == float(Fraction(a[i]) * Fraction(b[i]) + Fraction(c[i])) for i in range(3000))
    D, R = rng.normal(size=(500, 5, 3)), rng.normal(size=(500, 3, 3))
    assert np.array_equal(OB.rowmat(torch.from_numpy(D), torch.from_numpy(R)[:, None]).numpy(), np.stack([D[i] @ R[i] for i in range(500)]))
    assert np.array_equal(OB.rowmat(torch.from_numpy(D), torch.tensor(OB.CAM_R, dtype=torch.float64)).numpy(), D @ np.array(OB.CAM_R))   # in_view's camera matmul

def test_np_wrap_equals_numpy():
    x = np.concatenate([np.random.default_rng(1).uniform(-20, 20, 20000), [0.0, -0.0, np.pi, -np.pi, 2 * np.pi, -2 * np.pi, 3 * np.pi, 1e-300, -1e-300]])
    assert np.array_equal(OB.np_wrap(torch.from_numpy(x)).numpy(), KP.wrap(x))

def test_kp_state_equals_keypoint_source_bit_for_bit(world):
    m, _, r, ids6 = world; Q = states(m, r, 400); src = KP.KeypointSource(m, "truth"); d = mujoco.MjData(m); P, y, v = OB.kp_state(*poses(m, ids6, Q))
    for i, q in enumerate(Q):
        d.qpos[:] = q; P0, y0, v0 = src.true_state(d)
        assert np.array_equal(P[i].numpy(), P0) and y[i].item() == y0 and np.array_equal(v[i].numpy(), v0), i
    assert 0 < float(v.float().mean()) < 1   # both in and out of view
    ge = torch.linalg.vector_norm(P[:, :2] - P[:, 2:4], dim=-1); assert float((ge - 0.194862).abs().max()) <= 5e-7   # G0-5 glove-elbow

def test_float32_keypoints_close(world):
    m, _, r, ids6 = world; Q = states(m, r, 100, seed=3); X, W = poses(m, ids6, Q)
    P64, y64, _ = OB.kp_state(X, W); P32, y32, _ = OB.kp_state(X.float(), W.float())
    assert (P32.double() - P64).abs().max() < 1e-5 and (y32.double() - y64).abs().max() < 1e-5

@pytest.mark.parametrize("k", [0, 1, 2])
def test_delayed_block_with_resets_equals_keypoint_source(world, k):
    """G0-5 on random sequences: two envs, resets at 0 and mid-way (the lazy pre-fill), bit-equal positions, masks, yaw; velocities."""
    m, _, r, ids6 = world; Q = states(m, r, 60, seed=10 + k); Q2 = states(m, r, 60, seed=20 + k); X, W = poses(m, ids6, Q); X2, W2 = poses(m, ids6, Q2)
    lat = EV.Latency(2); lat.k[:] = k; obs = OB.EvalActorObs(2, lat, np.zeros(111), out_dtype=torch.float64)
    srcs = [KP.KeypointSource(m, "truth", delay=k) for _ in range(2)]; d = mujoco.MjData(m)
    for t in range(60):
        if t in (0, 31): obs.reset(torch.arange(2))
        P, y, v = OB.kp_state(torch.stack([X[t], X2[t]]), torch.stack([W[t], W2[t]])); o = obs.compute(torch.zeros(2, 83), P, y, v, noise=torch.zeros(2, 111, dtype=torch.float64))
        for j, (s, QQ) in enumerate(zip(srcs, (Q, Q2))):
            d.qpos[:] = QQ[t]; ref = s.reset(d) if t in (0, 31) else s.observe(d); got = o[j, 83:].float().numpy()
            assert np.array_equal(got[OB.POS], ref[OB.POS]) and np.array_equal(got[OB.MASK], ref[OB.MASK]) and np.array_equal(got[OB.YAW], ref[OB.YAW]), (t, j)
            assert np.abs(got[OB.VEL] - ref[OB.VEL]).max() <= 1e-6

def test_eval_actor_obs_is_delayed_noisy_obs_with_one_lag_per_episode():
    """G0-6 on random rows: per env and episode one k for all 111 dims, resets at random steps, = DelayedNoisyObs(k, 0)."""
    n, T = 6, 80; rng = np.random.default_rng(0); lat = EV.Latency(n); gen = torch.Generator().manual_seed(0)
    obs = OB.EvalActorObs(n, lat, np.zeros(111), out_dtype=torch.float64); resets = rng.random((T, n)) < 0.05; resets[0] = True; ref = [None] * n
    for t in range(T):
        ids = torch.from_numpy(np.flatnonzero(resets[t]))
        if len(ids): EV.draw_latency(lat, ids, gen); obs.reset(ids)
        p = torch.from_numpy(rng.normal(size=(n, 83)).astype(np.float32)); P = torch.from_numpy(rng.normal(size=(n, 5, 3)) * 0.3); P[..., 2] += 2.0
        o = obs.compute(p, P, torch.from_numpy(rng.uniform(-3, 3, n)), torch.from_numpy(rng.random((n, 5)) < 0.7), noise=torch.zeros(n, 111, dtype=torch.float64)).numpy()
        for j in range(n):
            row = obs.clean[j].numpy().copy()
            if resets[t, j]: ref[j] = DelayedNoisyObs(int(lat.k[j]), np.zeros(111), np.random.default_rng(0), uniform=True); ref[j].reset(row)
            assert np.array_equal(o[j], ref[j](row)), (t, j)
            if resets[t, j]: assert all(np.array_equal(obs.ring[j, q].numpy(), row) for q in range(3)) and not row[83 + 15:83 + 21].any()
    assert int(obs.k_switches) == 0 and set(lat.k.tolist()) <= {0, 1, 2}
    lat.k[0] = (lat.k[0] + 1) % 3; obs.compute(p, P, torch.zeros(n, dtype=torch.float64), torch.ones(n, 5, dtype=torch.bool)); assert int(obs.k_switches) == 1   # a mid-episode change is counted

def test_reset_only_flags_and_noise_after_the_delay():
    n = 512; lat = EV.Latency(n); lat.k[:] = 2; hw = OB.noise_halfwidths(load_cfg(CONFIGS / "r1_mjlab.yaml"))
    obs = OB.EvalActorObs(n, lat, hw, out_dtype=torch.float64, generator=torch.Generator().manual_seed(0)); z = torch.zeros(n, 83); P = torch.zeros(n, 5, 3, dtype=torch.float64)
    f = lambda: obs.compute(z + 1, P, torch.zeros(n, dtype=torch.float64), torch.zeros(n, 5, dtype=torch.bool))
    f(); ring = obs.ring.clone(); hist = obs.hist.prev.clone(); obs.reset(torch.arange(0, n, 2))
    assert torch.equal(obs.ring, ring) and bool(obs.prefill[::2].all()) and not bool(obs.prefill[1::2].any()) and bool(obs.hist.first[::2].all())
    nz = (f() - obs.ring[:, 2].double()).numpy(); assert (np.abs(nz) <= hw).all() and not nz[:, hw == 0].any() and (np.abs(nz).max(0)[hw > 0] > 0.9 * hw[hw > 0]).all()
    o = obs.compute(z, P, torch.zeros(n, dtype=torch.float64), torch.ones(n, 5, dtype=torch.bool), noise=torch.zeros(n, 111, dtype=torch.float64), blind=True)
    assert not o[:, 83:].any() and not obs.clean[:, 83:].any()

def test_proprio83_equals_raw_obs(world):
    m, cfg, r, _ = world; Q = states(m, r, 200, seed=5); rng = np.random.default_rng(5); home = np.array(cfg.get("home", HOME), float); d = mujoco.MjData(m)
    V = rng.normal(size=(200, m.nv)); C = rng.normal(size=(200, 3)) * rng.choice([0.0, 0.03, 1.0], (200, 1)); t = rng.integers(0, 1000, 200); A = rng.normal(size=(200, 24)).astype(np.float32)
    got = OB.proprio83(torch.from_numpy(Q[:, r.free_q + 3:r.free_q + 7]), torch.from_numpy(V[:, r.free_v + 3:r.free_v + 6]), torch.from_numpy(Q[:, r.qadr]), torch.from_numpy(V[:, r.vadr]),
                       torch.from_numpy(home), torch.from_numpy(C), torch.from_numpy(t), torch.from_numpy(A)).numpy()
    for i in range(200):
        d.qpos[:] = Q[i]; d.qvel[:] = V[i]; sh = types.SimpleNamespace(d=d, r=r, contract="mjlab", command=C[i], t=int(t[i]), dt=0.02, home=home, last_a=A[i])
        assert np.array_equal(got[i], R1HitEnv._raw_obs(sh)), i
    assert (np.abs(got[:, OB.PHASE]).sum(1) > 0).sum() == (np.linalg.norm(C, axis=1) >= 0.1).sum()
