"""Full-body M2 v2 opponent driver (M2V2_PLAN_v3 M2; gpu/fb_opponent.py, scripts/fb_record.py, scripts/fb_m2_check.py): the
tracker's training contract, MuJoCo's quaternion and differentiatePos arithmetic, placement = FightReplay.place, the actor vs the
tracker ONNX (float32 and float64), the obs / facing / steering against a recorded eval bout (seed 60001, role 'record', 1 s),
the recorder leaving the bout unchanged, tracker-alone fall parity on two starts, no ONNX read by the driver, m12 precision
flags. The full gates (16 bouts, 72 tracker-alone starts) run in scripts/fb_m2_check.py."""
import sys, builtins
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "gpu"))
import fb_opponent as FO, fb_m2_check as C, fb_record as REC
from fb_seeds import SeedRefused
from r1gym.model import build_spec, load_cfg, HOME, CTRL_JOINTS, torque_caps
from r1gym.opponent import FightReplay, Tracker, _sub

@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    """One recorded eval bout, 1 s (seed 60001: kfab_201, 0.40 m)."""
    out = tmp_path_factory.mktemp("rec") / "rec.npz"
    eps = REC.work(([60001], 1.0)); assert len(eps) == 1 and eps[0][0]["clip"] == "kfab_201"
    ep, row, rows = eps[0]; A = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}; A["bout"] = np.zeros(len(rows), np.int32)
    import json; A["meta"] = np.array(json.dumps(dict(episodes=[ep], rows=[row], seeds=[60001], aim=REC.AIM, seconds=1.0, steps=[len(rows)]), default=str))
    np.savez(out, **A); return C.load_rec(out)

def test_tracker_contract_constants():
    tr = Tracker(C.onnx_path()); m = build_spec(opponent=True, cfg=load_cfg()).compile(); act = [m.actuator("opp_" + j.replace("_joint", "")).id for j in CTRL_JOINTS]
    assert np.array_equal(FO.DEFAULT, HOME) and np.array_equal(FO.DEFAULT, tr.default)
    assert np.array_equal(FO.KP, m.actuator_gainprm[act, 0]) and np.array_equal(FO.KP, tr.kp_meta) and np.array_equal(FO.CAP, torque_caps({**load_cfg(), "torque_cap_policy": "mjlab"}))
    assert np.array_equal(FO.SCALE, tr.scale) and tr.steerable and (FO.FACE_TAU, FO.STEER_TAU, FO.STEER_MAX) == (Tracker.FACE_TAU, Tracker.STEER_TAU, Tracker.STEER_MAX)

def test_precision_pinned():
    torch.backends.cuda.matmul.allow_tf32 = True; FO.TrackerDriver(1); assert FO.precision_ok()

def test_quaternion_ops_match_mujoco():
    rng = np.random.default_rng(0); a, b = rng.normal(size=(64, 4)), rng.normal(size=(64, 4)); a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
    M, R = np.zeros((64, 4)), np.zeros((64, 9)); N = np.zeros((64, 4))
    for i in range(64): mujoco.mju_mulQuat(M[i], a[i], b[i]); mujoco.mju_quat2Mat(R[i], a[i]); mujoco.mju_negQuat(N[i], a[i])
    A, B = torch.from_numpy(a), torch.from_numpy(b)
    assert np.abs(FO.qmul(A, B).numpy() - M).max() < 1e-15 and np.abs(FO.quat2mat(A).reshape(64, 9).numpy() - R).max() < 1e-15 and np.array_equal(FO.qconj(A).numpy(), N)

def test_differentiate_pos_matches_mujoco():
    r = _sub("r1"); rng = np.random.default_rng(1); q0 = np.tile(r.qpos0, (32, 1)); q0[:, :3] = rng.normal(size=(32, 3)); q0[:, 3:7] = rng.normal(size=(32, 4))
    q0[:, 3:7] /= np.linalg.norm(q0[:, 3:7], axis=1, keepdims=True); q0[:, 7:] = rng.normal(size=(32, 24)) * 0.3; q1 = q0.copy(); q1[:, 7:] += rng.normal(size=(32, 24)) * 0.01
    q1[:, :3] += rng.normal(size=(32, 3)) * 0.01; q1[:, 3:7] += rng.normal(size=(32, 4)) * 0.02; q1[:, 3:7] /= np.linalg.norm(q1[:, 3:7], axis=1, keepdims=True)
    q1[::4, 3:7] *= -1   # the other hemisphere
    want = np.zeros((32, 30))
    for i in range(32): mujoco.mj_differentiatePos(r, want[i], 0.005, q0[i], q1[i])
    got = FO.differentiate_pos(torch.from_numpy(q0), torch.from_numpy(q1), 0.005).numpy(); assert np.abs(got - want).max() < 1e-9 * np.abs(want).max()

@pytest.mark.parametrize("clip,d", [("kfab_200", 0.4), ("kfab_201", 0.55), ("kfab_203", 0.47)])
def test_placement_equals_fightreplay_place(clip, d):
    m = build_spec(opponent=True, cfg={**load_cfg(), "opponent_mode": "kinematic"}).compile(); rep = FightReplay(m, clip, distance=d, cfg={**load_cfg(), "opponent_mode": "kinematic"})
    drv = FO.TrackerDriver(1, torch.float64); d0 = mujoco.MjData(m); j = m.joint("opp_floating_base_joint").id; qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
    for t0 in (0.0, 0.06, 1.234567, 4.98, rep.T / 50.0 - 0.001, 11.37):
        rep.place(m, d0, t0); qp, qv = drv.reset(None, FO.TRAIN.index(clip), t0, d)
        assert np.abs(qp[0].numpy() - d0.qpos[qa:qa + 31]).max() <= 1e-15 and np.abs(qv[0].numpy() - d0.qvel[va:va + 30]).max() <= 1e-12, t0
    assert drv.spot[0].tolist() == [d, 0.0] and not drv.yaw_set.any() and not drv.axis_set.any() and float(drv.last.abs().max()) == 0.0

def test_placement_bout_frame_rotation():
    drv = FO.TrackerDriver(2, torch.float64); h = 0.7; o = np.array([[0.0, 0.0], [1.5, -2.0]])
    q0, v0 = drv.reset(None, [0, 0], 1.0, 0.45); q1, v1 = drv.reset(None, [0, 0], 1.0, 0.45, origin=o, heading=h)
    c, s = np.cos(h), np.sin(h); Rz = np.array([[c, -s], [s, c]])
    assert np.allclose(q1[:, :2].numpy(), o + q0[:, :2].numpy() @ Rz.T, atol=1e-12) and np.allclose(v1[:, :2].numpy(), v0[:, :2].numpy() @ Rz.T, atol=1e-9)
    assert np.allclose(v1[:, 3:].numpy(), v0[:, 3:].numpy(), atol=1e-9) and np.allclose(drv.spot.numpy(), o + np.array([0.45 * c, 0.45 * s]), atol=1e-12)
    assert np.allclose(FO.quat2mat(q1[:, 3:7]).numpy(), np.einsum("ij,njk->nik", np.block([[Rz, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]), FO.quat2mat(q0[:, 3:7]).numpy()), atol=1e-12)

def test_actor_matches_the_onnx_float32_and_float64(rec):
    r = C.g0_1(rec, n_min=40); assert r["ok"] and all(r["weights_bit_equal"].values()), r
    assert r["f32_torch_vs_onnxruntime"] <= 1e-5 and r["f64_torch_vs_onnx_graph_f64"] <= 1e-9

def test_obs_facing_steering_on_a_recorded_bout(rec):
    r = C.g0_2_3(rec, torch.float64); assert r["g02_ok"] and r["g03_ok"] and r["clock_and_frame_equal"], r
    assert r["g02a_builder"] == 0.0 and r["placement_qpos"] <= 1e-15 and r["reference_joint_vs_eval"] == 0.0

def test_recorder_refuses_other_blocks():
    for s in ([60016], [21000], [22000], [67000]):
        with pytest.raises(SeedRefused): REC.work((s, 0.1))
    assert REC.assign(range(60000, 60016))[:4] == [("kfab_200", 0.4, 60000), ("kfab_201", 0.4, 60001), ("kfab_203", 0.4, 60002), ("kfab_200", 0.45, 60003)]
    S = list(range(60000, 60016)); assert sorted([x for w in range(4) for x in REC.assign(S[w::4])], key=lambda x: x[2]) == REC.assign(S)   # split-independent

@pytest.mark.parametrize("clip,s", [("kfab_200", 4), ("kfab_201", 0)])
def test_tracker_alone_parity_short(clip, s, monkeypatch):
    import eval_tracker as ET
    monkeypatch.setattr(ET, "FightReplay", FightReplay); ev = C._alone((clip, s, 8, "eval", 3.0))
    for mode in ("torch32", "torch64"):
        r = C._alone((clip, s, 8, mode, 3.0)); assert ET.FightReplay is FightReplay
        assert (r["fell"] is None) == (ev["fell"] is None) and (r["fell"] is None or abs(r["fell"] - ev["fell"]) <= 0.1), (ev, r)
        assert r["place_qpos"] <= 1e-15 and r["place_qvel"] <= 1e-12 and abs(r["joint_rmse"] - ev["joint_rmse"]) < 0.02, (ev, r)

def test_driver_reads_no_onnx(monkeypatch):
    seen = []; op = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (seen.append(str(f)), op(f, *a, **k))[1])
    drv = FO.TrackerDriver(3, torch.float32); drv.reset(None, [0, 1, 2], [0.1, 0.2, 0.3], 0.45)
    z = torch.zeros(3, 3); drv.step(1.0, torch.zeros(3, 2), torch.ones(3, 2), z, torch.tensor([[1.0, 0, 0, 0]] * 3), z, torch.zeros(3, 24), torch.zeros(3, 24))
    assert seen and not any(s.lower().endswith(".onnx") for s in seen) and any(s.endswith("model_13749.pt") for s in seen), seen

def test_mjtracker_serves_one_env():
    """eval_opponent.run shares a FightReplay between envs of one (clip, distance); the driver's state is per MjTracker, so acting
    on an env other than the one it last placed is refused."""
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile(); mt = FO.MjTracker(FightReplay(m, "kfab_200", distance=4.0, cfg=cfg)).install()
    d0, d1 = mujoco.MjData(m), mujoco.MjData(m); mt.place(m, d0, 0.5); mujoco.mj_forward(m, d0); st = {"last": np.zeros(24, np.float32), "target": np.zeros(2)}
    mt.act(m, d0, mt.replay, st, 0.52); assert np.abs(d0.ctrl[mt.replay.act]).max() > 0
    with pytest.raises(AssertionError, match="serves the one env"): mt.act(m, d1, mt.replay, st, 0.52)
