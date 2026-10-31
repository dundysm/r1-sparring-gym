"""Opponent keypoints for the sparring defender (r1gym/keypoints.py; docs/SPARRING_M1M2.md 1.1-1.4 and 4): the pinned constants
agree with the camera config and the render twin's keypoints; frames (world rotations and translations leave the block alone,
the defender's yaw moves only the relative yaw), order and dims; the FOV mask on known points; zero-plus-mask; velocities;
the delay line equals DelayedNoisyObs; the env's MjData is never written. MjlabPolicy loads a synthetic widened ONNX (a Slice
to the first 83 inputs in front of the warm start's graph) and a reader ONNX shows the 28 inputs arrive in order; through
eval_opponent.run the widened sighted and blind twins reproduce the warm start's rows exactly, and the 83-d rows don't change
with the flag on. (The camera source is a stub.) Against the GPU (gpu/sparring_task.py, when present): its torch keypoint_block
equals assemble, the yaw pair present iff any keypoint is on both sides, and its per-env velocity bookkeeping (KeypointHistory)
driven by a recorded CPU bout reproduces the CPU blocks, resets and a defender-only turn included."""
import sys, json, shutil, importlib.util
from pathlib import Path
import numpy as np, mujoco, pytest, onnx
from onnx import helper, TensorProto, numpy_helper
from r1gym import keypoints as K
from r1gym.keypoints import KeypointSource, assemble, in_view, KP_DIM, POS, VEL, YAW, MASK, KP_NAMES
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay
from r1gym.sensors import DelayedNoisyObs
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
WARM = ROOT / "results" / "policy_mjlab_guard_footwork_r" / "policy.onnx"

@pytest.fixture(scope="module")
def world():
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller(f"warm={WARM}"); cc = {**controller_cfg(c), "opponent_mode": load_cfg()["opponent_mode"], "opponent_aim": "none"}
    return c, cc, build_spec(opponent=True, cfg=cc).compile()

def env(world, delay=1, t0=3.0, dr=False, seed=0):
    c, cc, m = world; e = R1HitEnv(seed=seed, cfg=cc, model=m, episode_s=10, max_impulse=0, train=False, dr=dr)
    e.opp = FightReplay(m, "kfab_204", distance=0.45, cfg=cc); e.opp_t0 = t0; o = e.reset(hits=[], delay=delay); return e, o

# ---- constants and layout
def test_layout_and_pinned_constants():
    from r1gym.perception.camera import HeadCameraSpec
    from r1gym.perception.render_scene import KEYPOINTS
    sl = [POS, VEL, YAW, MASK]; assert [s.start for s in sl] == [0, 15, 21, 23] and [s.stop for s in sl] == [15, 21, 23, 28] == [15, 21, 23, KP_DIM]
    assert KP_NAMES == ("l_glove", "r_glove", "l_elbow", "r_elbow", "head") and K.PROPRIO_DIM + KP_DIM == 111
    assert K.OBS_NAMES_SIGHTED.split(",")[-1] == "opp_keypoints" and K.OBS_NAMES_BLIND.split(",")[-1] == "opp_keypoints_blind"
    assert K.OBS_NAMES == "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"
    s = HeadCameraSpec(); p, R = s.mount_torso   # the doc's numbers are the camera config's, to its 4 decimals
    assert np.allclose(p, K.CAM_C, atol=1e-4) and np.allclose(R, K.CAM_R, atol=1e-4) and np.allclose(s.K(), [K.CAM_F, K.CAM_F, K.CAM_CX, K.CAM_CY], atol=1e-3)
    for k, src in (("l_glove", "l_fist"), ("r_glove", "r_fist"), ("l_elbow", "l_elbow"), ("r_elbow", "r_elbow"), ("head", "head")):
        b, off = K.KP_BODIES[k]; assert b == "opp_" + KEYPOINTS[src][0] and np.allclose(off, KEYPOINTS[src][1])

def test_in_view_known_points():
    c, R = K.CAM_C, K.CAM_R; at = lambda qx, qy, qz: c + R @ np.array([qx, qy, qz])   # a point given in camera axes, back in T
    tx, ty = K.CAM_CX / K.CAM_F, K.CAM_CY / K.CAM_F
    P = np.array([at(0, 0, 1), at(0, 0, -1), at(0.99 * tx, 0, 1), at(1.01 * tx, 0, 1), at(0, -0.99 * ty * 2, 2), at(0, -1.01 * ty * 2, 2), at(0, 0, 0.04), at(0, 0, 0.06),
                  [0.5, 0.0, -0.3], [0.4, 0.0, 0.6], [-0.5, 0.0, 0.0]])   # T: ahead and below the head (seen), above the head, behind
    assert in_view(P).tolist() == [True, False, True, False, True, False, False, True, True, False, False]
    assert in_view(P.reshape(1, 11, 3)).shape == (1, 11)

# ---- assemble: zero-plus-mask, velocities, batching
def test_assemble_zero_plus_mask_and_velocity():
    g = np.random.default_rng(0); P, P0 = g.normal(size=(5, 3)), g.normal(size=(5, 3)); y = 2.5
    b = assemble(P, P0, y, np.ones(5)); assert b.dtype == np.float32 and b.shape == (KP_DIM,)
    assert np.allclose(b[POS], P.ravel(), atol=1e-6) and np.allclose(b[VEL], ((P[:2] - P0[:2]) / 0.02).ravel(), rtol=1e-6)
    assert np.allclose(b[YAW], [np.sin(y), np.cos(y)], atol=1e-7) and (b[MASK] == 1).all()
    assert (assemble(P, None, y, np.ones(5))[VEL] == 0).all()   # first step of an episode
    mk = np.array([0, 1, 1, 0, 0.0]); b = assemble(P, P0, y, mk)
    assert (b[0:3] == 0).all() and (b[9:15] == 0).all() and (b[15:18] == 0).all() and b[MASK].tolist() == mk.tolist()
    assert np.allclose(b[YAW], [np.sin(y), np.cos(y)], atol=1e-7)   # the yaw pair is present when any keypoint is (doc 8), head absent or not
    assert (assemble(P, P0, y, np.zeros(5)) == 0).all()   # nothing in view: the whole block is zero
    assert np.allclose(b[3:9], P[1:3].ravel(), atol=1e-6) and np.allclose(b[18:21], (P[1] - P0[1]) / 0.02, rtol=1e-6)
    assert not np.signbit(b[b == 0]).any()   # absent entries are +0.0
    PB, P0B, yB, mB = g.normal(size=(7, 5, 3)), g.normal(size=(7, 5, 3)), g.uniform(-3, 3, 7), g.random((7, 5)) < 0.5
    assert np.array_equal(assemble(PB, P0B, yB, mB), np.stack([assemble(PB[i], P0B[i], yB[i], mB[i]) for i in range(7)]))

# ---- the CPU source on the real tracker opponent
def test_frames_order_and_world_invariance(world):
    _, _, m = world; e, _ = env(world); s = KeypointSource(m); P, y, vis = s.true_state(e.d)
    d = e.d   # after reset the env ran mj_forward: its xpos are current, so the definitions can be checked directly
    tb = m.body("torso_link").id; RT = d.xmat[tb].reshape(3, 3)
    for i, k in enumerate(KP_NAMES):
        b, off = K.KP_BODIES[k]; j = m.body(b).id; assert np.allclose(P[i], RT.T @ (d.xpos[j] + d.xmat[j].reshape(3, 3) @ off - d.xpos[tb]), atol=1e-9)
    assert P[0, 1] < P[1, 1] and P[2, 1] < P[3, 1]   # facing the defender, the opponent's own left is on the defender's right (-y)
    assert 0.2 < P[4, 0] < 1.0 and np.cos(y) < -0.5   # its head is ahead, and it faces back
    # rotate and translate the whole world about z: the block doesn't move; turn only the defender: only the relative yaw does
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos; a = 0.7; qa = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]); Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    for fq in (e.r.free_q, e.opp.qadr):
        d2.qpos[fq:fq + 3] = Rz @ d.qpos[fq:fq + 3] + [0.3, -1.2, 0.0]; q = np.zeros(4); mujoco.mju_mulQuat(q, qa, d.qpos[fq + 3:fq + 7]); d2.qpos[fq + 3:fq + 7] = q
    P2, y2, v2 = s.true_state(d2); assert np.allclose(P2, P, atol=1e-9) and abs(K.wrap(y2 - y)) < 1e-9 and (v2 == vis).all()
    d3 = mujoco.MjData(m); d3.qpos[:] = d.qpos; fq = e.r.free_q; q = np.zeros(4); mujoco.mju_mulQuat(q, np.array([np.cos(0.1), 0, 0, np.sin(0.1)]), d.qpos[fq + 3:fq + 7]); d3.qpos[fq + 3:fq + 7] = q
    assert abs(K.wrap(s.true_state(d3)[1] - (y - 0.2))) < 1e-9

def test_source_reads_current_state_never_writes(world):
    """The env's xpos lag one substep after mj_step; the source reads qpos on its own twin, and leaves the env's MjData bit-for-bit."""
    c, _, m = world; e, o = env(world); s = KeypointSource(m); s.reset(e.d)
    for _ in range(10): o, *_ = e.step(c.act(o[None])[0])
    F = ("qpos", "qvel", "act", "ctrl", "qacc", "xpos", "xmat", "xquat", "qfrc_applied", "xfrc_applied", "sensordata", "time", "ncon", "efc_force")
    snap = {f: np.array(getattr(e.d, f)).tobytes() for f in F}; b = s.observe(e.d)
    assert all(np.array(getattr(e.d, f)).tobytes() == snap[f] for f in F)
    d2 = mujoco.MjData(m); d2.qpos[:] = e.d.qpos; mujoco.mj_kinematics(m, d2); j = m.body("opp_torso_link").id; tb = m.body("torso_link").id
    head = d2.xmat[tb].reshape(3, 3).T @ (d2.xpos[j] + d2.xmat[j].reshape(3, 3) @ [0.01, 0, 0.357] - d2.xpos[tb])
    assert np.allclose(s.true_state(e.d)[0][4], head, atol=1e-9) and not np.allclose(e.d.xpos[j], d2.xpos[j], atol=1e-6)   # stale vs current

def test_bout_velocity_mask_and_delay(world):
    """Over a bout: velocities are one-step differences of the true T positions (0 at reset); a masked keypoint is zero; the
    delayed block equals a zero-noise DelayedNoisyObs of the undelayed one (delay k: reset copies, then k steps late)."""
    c, _, m = world; e, o = env(world, delay=2); s0, s2 = KeypointSource(m, delay=0), KeypointSource(m, delay=2)
    B0, B2, P = [s0.reset(e.d)], [s2.reset(e.d)], [s0.prev]; seen = np.zeros(5, bool); hidden = np.zeros(5, bool)
    for _ in range(80):
        o, *_ = e.step(c.act(o[None])[0]); B0.append(s0.observe(e.d)); B2.append(s2.observe(e.d)); P.append(s0.prev)
    B0, B2 = np.array(B0), np.array(B2); M = B0[:, MASK] > 0.5; seen, hidden = M.any(0), (~M).any(0)
    assert (B0[0, VEL] == 0).all() and seen[:2].all() and hidden.any()   # both gloves seen at some point, something hidden at some point
    for t in range(1, len(B0)):
        v = (P[t][:2] - P[t - 1][:2]) / 0.02
        for g in (0, 1):
            assert np.allclose(B0[t, 15 + 3 * g:18 + 3 * g], v[g] if M[t, g] else 0, atol=1e-4) and np.allclose(B0[t, 3 * g:3 * g + 3], P[t][g] if M[t, g] else 0, atol=1e-6)
        assert ((B0[t, :15].reshape(5, 3) == 0) | M[t][:, None]).all() and M[t].any() == (np.abs(B0[t, YAW]).sum() > 0.5)
        assert np.array_equal(M[t], in_view(P[t]))
    line = DelayedNoisyObs(2, np.zeros(KP_DIM), np.random.default_rng(5), uniform=True); line.reset(B0[0])
    assert np.array_equal(B2, np.array([line(b) for b in B0], np.float32)) and np.array_equal(B2[2:], B0[:-2])

def test_options_corrupt_fov_and_camera_stub(world):
    _, _, m = world; e, _ = env(world); a = KeypointSource(m).reset(e.d)
    assert np.array_equal(a, KeypointSource(m).reset(e.d))   # truth is deterministic
    assert (KeypointSource(m, fov=False).reset(e.d)[MASK] == 1).all()
    b = [KeypointSource(m, fov=False, corrupt=True, rng=np.random.default_rng(i)).reset(e.d) for i in range(40)]
    assert not np.array_equal(b[0], b[1]) and 0.03 < np.mean([x[MASK] == 0 for x in b]) < 0.2   # dropout p = 0.10
    on = np.array([x[MASK] for x in b]) > 0.5; P = np.array([x[POS].reshape(5, 3) for x in b]); P_true = KeypointSource(m, fov=False).true_state(e.d)[0]
    assert np.abs(P - P_true)[on].max() <= 0.04 + 1e-6   # per-step 0.02 + per-episode bias 0.02
    with pytest.raises(ValueError): KeypointSource(m, corrupt=True)
    with pytest.raises(ValueError): KeypointSource(m, mode="nope")
    with pytest.raises(NotImplementedError): KeypointSource(m, "camera", delay=1).reset(e.d)

# ---- widened policies
def widened(tmp, names=K.OBS_NAMES_SIGHTED, reader=False, tag="spar"):
    """A synthetic 111-input policy: a Slice to the first 83 inputs in front of the warm start's graph (it acts exactly as the
    warm start), or reader=True: actions = inputs 83:107 (the first 24 of the keypoint block). Metadata copied, names set;
    the warm start's gym.json beside it, as the M2 results will have."""
    mo = onnx.load(str(WARM)); g = mo.graph; inp = g.input[0].name
    cst = lambda n, v: numpy_helper.from_array(np.array(v, np.int64), n)
    if reader:
        del g.node[:]; del g.initializer[:]; g.initializer.extend([cst("kp_s", [83]), cst("kp_e", [107]), cst("kp_a", [1])])
        g.node.append(helper.make_node("Slice", [inp, "kp_s", "kp_e", "kp_a"], [g.output[0].name]))
    else:
        for n in g.node: n.input[:] = ["obs83" if x == inp else x for x in n.input]
        g.initializer.extend([cst("kp_s", [0]), cst("kp_e", [83]), cst("kp_a", [1])]); g.node.insert(0, helper.make_node("Slice", [inp, "kp_s", "kp_e", "kp_a"], ["obs83"]))
    g.input[0].type.tensor_type.shape.dim[1].dim_value = 111
    for p in mo.metadata_props:
        if p.key == "observation_names": p.value = names
    onnx.checker.check_model(mo); d = Path(tmp) / tag; d.mkdir(); onnx.save(mo, str(d / "policy.onnx")); shutil.copy(WARM.with_name("gym.json"), d / "gym.json")
    return d / "policy.onnx"

def test_policy_loads_widened_and_reads_the_block(world, tmp_path):
    from r1gym.policy import load_controller, controller_cfg
    warm = world[0]; assert warm.kp_mode is None and warm.obs_dim == 83
    sp = load_controller(f"A={widened(tmp_path)}"); bl = load_controller(f"blind={widened(tmp_path, K.OBS_NAMES_BLIND, tag='blind')}")
    assert (sp.kp_mode, sp.obs_dim, bl.kp_mode, bl.obs_dim) == ("obs", 111, "zeros", 111) and controller_cfg(sp) == controller_cfg(warm)
    X = np.random.default_rng(1).normal(size=(3, 111)).astype(np.float32)
    assert np.array_equal(sp.act(X), warm.act(X[:, :83])) and np.array_equal(bl.act(X), warm.act(X[:, :83]))
    with pytest.raises(AssertionError): sp.act(X[:, :83])
    rd = load_controller(f"rd={widened(tmp_path, reader=True, tag='reader')}"); assert np.array_equal(rd.act(X), X[:, 83:107])
    bad = widened(tmp_path, K.OBS_NAMES + ",opp_kp", tag="bad")
    with pytest.raises(AssertionError): load_controller(f"bad={bad}")

def test_eval_rows_with_keypoints(world, tmp_path):
    """Through eval_opponent.run on 2 short DR bouts (delays 0 and 2): the 83-d warm start's rows are the same with the flag on;
    the widened sighted policy (with --keypoints truth) and its blind twin (no flag) reproduce them exactly; a sighted policy
    without the flag is an error; the reader policy gets, at every step, the source's block for that bout."""
    import eval_opponent as eo
    from r1gym.policy import load_controller
    warm, cc, m = world; T = 1.0
    eps = [dict(clip="kfab_204", distance=0.45, rep=0, t0=3.0, delay=0, hits=[], seed=0), dict(clip="kfab_204", distance=0.5, rep=1, t0=4.0, delay=2, hits=[], seed=1)]
    kw = dict(threads=1, dr=True, meter=True, falls=True)
    ref = eo.run(warm, eps, m, cc, T, **kw); assert eo.run(warm, eps, m, cc, T, keypoints="truth", **kw) == ref
    sp = load_controller(f"A={widened(tmp_path)}"); bl = load_controller(f"blind={widened(tmp_path, K.OBS_NAMES_BLIND, tag='blind')}")
    assert eo.run(sp, eps, m, cc, T, keypoints="truth", **kw) == ref and eo.run(bl, eps, m, cc, T, **kw) == ref
    with pytest.raises(ValueError): eo.run(sp, eps, m, cc, T, **kw)
    class Spy:   # the warm start's actions, recording what a widened policy is given
        name, kp_mode, obs_dim = "spy", "obs", 111
        def __init__(self): self.rows = []
        def act(self, X): self.rows.append(np.array(X)); return warm.act(X[:, :83])
    spy = Spy(); assert eo.run(spy, eps, m, cc, T, keypoints="truth", **kw) == ref
    import copy
    for j, ep in enumerate(eps):   # replay bout j alone, as run() builds it, with its own source: the rows are [the env's obs, that source's block]
        e = R1HitEnv(seed=ep["seed"], cfg=cc, model=copy.deepcopy(m), episode_s=T, max_impulse=0, train=False, dr=True)
        e.opp = FightReplay(e.m, ep["clip"], distance=ep["distance"], cfg=cc); e.opp_t0 = ep["t0"]; o = e.reset(hits=[], delay=ep["delay"])
        s = KeypointSource(e.m, "truth", delay=ep["delay"]); R = [np.concatenate([o, s.reset(e.d)])]
        for _ in range(len(spy.rows) - 1): o, *_ = e.step(warm.act(R[-1][None, :83])[0]); R.append(np.concatenate([o, s.observe(e.d)]))
        X = np.array([r[j] for r in spy.rows]); assert X.shape[1] == 111 and np.array_equal(X, np.array(R))
        assert np.array_equal(X[:ep["delay"] + 1, 83:], np.repeat(X[:1, 83:], ep["delay"] + 1, 0)) and (X[:, 83:][:, MASK] == 1).any()   # the line starts full of the reset block

def _gpu():
    p = ROOT / "gpu" / "sparring_task.py"
    if not p.exists(): pytest.skip("gpu/sparring_task.py not written yet")
    spec = importlib.util.spec_from_file_location("sparring_task", p); st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st); return st

def test_contract_matches_gpu_keypoint_block():
    """The GPU's torch keypoint_block(P_T, P_T_prev, rel_yaw, mask) equals assemble to 1e-6, including the yaw rule (doc 8: the
    pair is present iff any keypoint is): rows with the head out and others in carry it, rows with nothing in view are zero."""
    import torch
    st = _gpu(); g = np.random.default_rng(3); P, P0, y, mk = g.normal(size=(64, 5, 3)), g.normal(size=(64, 5, 3)), g.uniform(-4, 4, 64), (g.random((64, 5)) < 0.7).astype(np.float32)
    mk[:8] = 0; mk[8:16, 4] = 0; mk[8:16, 0] = 1   # nothing in view; head out, a glove in
    out = st.keypoint_block(*(torch.as_tensor(v, dtype=torch.float32) for v in (P, P0, y, mk))).numpy(); ref = assemble(P, P0, y, mk)
    assert out.shape == (64, KP_DIM) and np.allclose(out, ref, atol=1e-5, rtol=1e-6)
    for b in (out, ref):
        assert (b[:8] == 0).all() and not np.signbit(b[:8]).any() and (np.abs(b[8:16, YAW]).sum(-1) > 0.5).all() and np.allclose(b[8:16, YAW], np.stack([np.sin(y[8:16]), np.cos(y[8:16])], -1), atol=1e-6)

def test_contract_gpu_velocity_bookkeeping_on_a_recorded_bout(world):
    """The velocity frame. A recorded CPU bout (KeypointSource on the tracker opponent: a reset, 15 steps, one step where only
    the defender turns 0.3 rad about its pelvis, 5 steps, a second reset, 15 steps) is fed through the GPU's per-env bookkeeping
    (KeypointHistory + keypoint_block, mask = in_view_t) in two envs whose resets fall on different steps: every block equals
    the CPU's (1.3: velocity = (p_T(t) - p_T(t - 0.02)) / 0.02 of T positions, each in its own step's T; 0 after a reset), with
    the FOV mask and with every keypoint present. On the turn step a bookkeeping that kept the previous WORLD positions and
    re-expressed them in the current T (the world velocity, which leaves the turn out) differs from the contract by > 1 m/s,
    so this test tells the two apart."""
    import torch
    st = _gpu(); c, _, m = world; rec = []
    def record(e, d, reset):
        S = (KeypointSource(m), KeypointSource(m, fov=False)) if reset else src
        blocks = [s.reset(d) if reset else s.observe(d) for s in S]; P, y, vis = S[0].true_state(d); k = S[0].dk
        P_w = k.xpos[S[0].kb] + np.einsum("kij,kj->ki", k.xmat[S[0].kb].reshape(-1, 3, 3), S[0].koff)
        rec.append(dict(reset=reset, P=P, y=y, vis=vis, blocks=blocks, P_w=P_w, x_T=k.xpos[S[0].tb].copy(), R_T=k.xmat[S[0].tb].reshape(3, 3).copy())); return S
    e, o = env(world, delay=0, t0=3.0); src = record(e, e.d, True)
    for _ in range(15): o, *_ = e.step(c.act(o[None])[0]); record(e, e.d, False)
    d2 = mujoco.MjData(m); d2.qpos[:] = e.d.qpos; d2.mocap_pos[:] = e.d.mocap_pos; d2.mocap_quat[:] = e.d.mocap_quat; fq = e.r.free_q; q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.array([np.cos(0.15), 0, 0, np.sin(0.15)]), e.d.qpos[fq + 3:fq + 7]); d2.qpos[fq + 3:fq + 7] = q; record(e, d2, False); turn = len(rec) - 1
    for _ in range(5): o, *_ = e.step(c.act(o[None])[0]); record(e, e.d, False)
    e, o = env(world, delay=0, t0=5.0, seed=1); src = record(e, e.d, True); second = len(rec) - 1
    for _ in range(15): o, *_ = e.step(c.act(o[None])[0]); record(e, e.d, False)
    L = len(rec); order = (list(range(L)), [(second + i) % L for i in range(L)])   # env 1 starts at the second reset
    kh = st.KeypointHistory(2); err = 0.0; f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    for i in range(L):
        R = [rec[order[j][i]] for j in (0, 1)]; ids = [j for j in (0, 1) if R[j]["reset"]]
        if ids: kh.reset(torch.tensor(ids))
        P, prev = kh.step(f32([r["P"] for r in R])); y = f32([r["y"] for r in R])
        for mk, b in ((st.in_view_t(P), 0), (torch.ones(2, 5, dtype=torch.bool), 1)):
            gb = st.keypoint_block(P, prev, y, mk).numpy(); cb = np.array([r["blocks"][b] for r in R]); err = max(err, float(np.abs(gb - cb).max()))
            assert np.array_equal(gb[:, MASK], cb[:, MASK])
        for j in ids: assert (gb[j, VEL] == 0).all()
    assert err < 1e-4
    r0, r1 = rec[turn - 1], rec[turn]; v = r1["blocks"][1][VEL].reshape(2, 3)
    v_world = (r1["P"][:2] - K.to_torso(r0["P_w"][:2], r1["x_T"], r1["R_T"])) / 0.02   # the previous world points in the current T: ~0 here
    assert np.allclose(v, (r1["P"][:2] - r0["P"][:2]) / 0.02, atol=1e-4) and np.abs(v - v_world).max() > 1.0 and np.abs(v_world).max() < 0.1
