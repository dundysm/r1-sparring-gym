import numpy as np, mujoco, pytest
from r1gym.model import build_spec, R1Robot, load_cfg, torque_caps, CTRL_JOINTS
from r1gym.check_model import stand
from r1gym.perturb import make_hit, HitApplier, random_schedule, LOCATIONS
from r1gym.sensors import DelayedNoisyObs
from r1gym.env import R1HitEnv, BatchEnv, OBS_DIM, ACT_DIM
from r1gym.opponent import FightReplay, list_clips

CFG = load_cfg()

def test_model_compiles_with_24_capped_actuators():
    m = build_spec(cfg=CFG).compile(); assert m.nu == 24
    caps = torque_caps(CFG); assert np.allclose(m.actuator_forcerange[:, 1], caps)
    assert caps.max() <= 60 and abs(m.body_subtreemass[m.body("pelvis").id] - 28.93) < 0.1

def test_stand_pd_stands_and_stock_gains_fall():
    assert stand(CFG, "stand_pd")[0] is True
    assert stand(CFG, "pd")[0] is False      # documented: Unitree deploy gains cannot hold a passive stand

def test_hit_impulse_and_wrench():
    rng = np.random.default_rng(0); h = make_hit("strike", "chest", 10.0, 0.5, 0.0, rng)
    assert abs(h.impulse - 10.0) < 1e-6 and h.force[0] < 0       # frontal strike pushes robot backward (-x)
    m = build_spec(cfg=CFG).compile(); d = mujoco.MjData(m); R1Robot(m, CFG).reset(m, d)
    HitApplier(m).apply(d, [h], 0.55); assert np.linalg.norm(d.xfrc_applied[m.body("torso_link").id, :3]) > 0
    assert all(sum(x.impulse for x in random_schedule(rng, 10, 5)) > 0 for _ in range(3))

def test_delay_buffer():
    s = DelayedNoisyObs(2, np.zeros(1), np.random.default_rng(0)); s.reset(np.array([0.0]))
    outs = [s(np.array([float(i)]))[0] for i in range(1, 5)]; assert outs == [0.0, 0.0, 1.0, 2.0]

def test_env_shapes_and_pd_survives_no_hits():
    e = R1HitEnv(seed=0, train=False); o = e.reset(hits=[]); assert o.shape == (OBS_DIM,)
    for _ in range(150): o, r, d, info = e.step(np.zeros(ACT_DIM)); assert not info["fallen"]

def test_big_strike_knocks_pd_over():
    e = R1HitEnv(seed=0, train=False); e.reset(hits=[make_hit("strike", "chest", 20.0, 0.3, 0.0, np.random.default_rng(0))], delay=0)
    assert any(e.step(np.zeros(ACT_DIM))[3]["fallen"] for _ in range(150))

def test_batch_env_shares_model():
    b = BatchEnv(4, threads=2); o = b.reset(); assert o.shape == (4, OBS_DIM) and all(e.m is b.envs[0].m for e in b.envs)
    o, r, d, i = b.step(np.zeros((4, ACT_DIM))); assert r.shape == (4,)

def test_opponent_replay_moves_and_touches():
    assert len(list_clips()) >= 1
    m = build_spec(opponent=True, cfg=CFG).compile(); op = FightReplay(m, list_clips()[0])
    q0, q1 = op.qpos_at(0.0), op.qpos_at(1.0); assert q0.shape == (31,) and np.abs(q0[7:] - q1[7:]).max() > 0.05
    assert 0.3 < op.q[:, 0].mean() < 1.0                          # in front of the R1


def test_verify_r1_specs(tmp_path):
    import subprocess, sys, json, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]; out = tmp_path / "r1v.json"
    r = subprocess.run([sys.executable, str(root / "scripts" / "verify_r1.py"), "--out", str(out), "--policies"], capture_output=True, text=True)
    res = json.load(open(out))
    failed = [c["check"] for c in res["checks"] if not c["pass"] and "reference available" not in c["check"]]
    assert not failed, failed


def test_r1_opponent_retarget():
    from r1gym.model import build_spec
    from r1gym.opponent import FightReplay, list_clips
    m = build_spec(opponent=True).compile()
    op = FightReplay(m, list_clips()[0]); assert op.kind == "r1" and op.q.shape[1] == 31


def test_opponent_loop_is_seamless_and_retarget_has_no_flips():
    m = build_spec(opponent=True, cfg=CFG).compile()
    for clip in list_clips():
        op = FightReplay(m, clip, distance=0.6); q = np.vstack([op.q, op.q[:1]])   # includes the wrap back to frame 0
        assert np.abs(np.diff(q[:, :2], axis=0)).max() < 0.05, clip              # no root teleport
        assert np.abs(np.diff(q[:, 7:], axis=0)).max() < 0.4, clip               # 0.25 rad/frame cap (+~0.1 inside the loop cross-fade)
        assert abs(np.median(q[:, 0]) - 0.6) < 0.1 and abs(np.median(q[:, 1])) < 0.1, clip   # stays in front, in range


def test_puppet_opponent_is_a_capped_physical_r1():
    m = build_spec(opponent=True, cfg=CFG).compile(); assert m.nu == 48
    caps = torque_caps(CFG); opp = [m.actuator("opp_" + j.replace("_joint", "")).id for j in CTRL_JOINTS]
    assert np.allclose(m.actuator_forcerange[opp, 1], caps)
    from r1gym.scenario import demo_env
    from r1gym.contacts import per_body_force
    e, o = demo_env(m, CFG); op = e.opp; peak = 0.0
    for _ in range(150):   # 3 s, PD stand
        e.step(np.zeros(ACT_DIM)); peak = max(peak, sum(per_body_force(m, e.d, e.r.r1_bodies).values()))
        assert np.linalg.norm(e.d.qpos[op.qadr:op.qadr + 3] - op.qpos_at(e.d.time)[:3]) < 0.25   # tracks its clip
    assert peak < 3000.0
