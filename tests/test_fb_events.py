"""Full-body M2 v2 reset events and fall rules (M2V2_PLAN_v3 M4 G0-6b, G0-8; gpu/fb_events.py): latency draws (shares, only the
reset envs), the B3 call order, the fall constants = configs/r1_assumptions.yaml and eval_opponent.py, boundaries, up_z =
mju_quat2Mat's [2, 2] bit for bit, fell_eval = R1.fallen and opp_down_now = EO:86 on random states, the non-sticky flag."""
import sys
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_events as EV, fb_m4_check as C
import eval_opponent as EO
from r1gym.model import load_cfg, R1Robot

def test_latency_draws_only_reset_envs_with_the_eval_shares():
    lat = EV.Latency(60000); g = torch.Generator().manual_seed(0); EV.draw_latency(lat, torch.arange(60000), g)
    sh = [float((lat.k == k).float().mean()) for k in (0, 1, 2)]; assert max(abs(s - 1 / 3) for s in sh) < 0.01 and abs(float(lat.act_delay.float().mean()) - 0.5) < 0.01
    k0, d0 = lat.k.clone(), lat.act_delay.clone(); ids = torch.tensor([3, 7, 11]); k, d = EV.draw_latency(lat, ids, g)
    keep = torch.ones(60000, dtype=torch.bool); keep[ids] = False
    assert torch.equal(lat.k[keep], k0[keep]) and torch.equal(lat.act_delay[keep], d0[keep]) and torch.equal(k, lat.k[ids]) and int(lat.draws.sum()) == 60003
    env = type("E", (), dict(num_envs=4, fb_latency=EV.Latency(4)))(); EV.fb_draw_latency(env, None, g); assert int(env.fb_latency.draws.sum()) == 4

def test_b3_order():
    assert EV.B3_ORDER[0] == "event:fb_draw_latency" and EV.B3_ORDER.index("obs.reset") < EV.B3_ORDER.index("action.reset") < EV.B3_ORDER.index("command.reset") \
        < EV.B3_ORDER.index("forward") < EV.B3_ORDER.index("command.compute") < EV.B3_ORDER.index("obs.compute")
    log = EV.ResetLog(); [log(c) for c in EV.B3_ORDER]; assert log.ok(); log("obs.compute"); assert not log.ok()

def test_fall_constants_and_boundaries():
    f = load_cfg()["fall"]; assert EV.FALL_Z == f["min_pelvis_z"] and EV.UP_MIN == np.cos(np.deg2rad(f["max_tilt_deg"])) and (EV.OPP_TILT, EV.OPP_Z) == (EO.OPP_TILT, EO.OPP_Z)
    T = lambda *x: torch.tensor(x, dtype=torch.float64); below = lambda v: float(np.nextafter(v, -1))
    assert EV.fell_eval(T(0.45, below(0.45), 0.7, 0.7), T(1.0, 1.0, EV.UP_MIN, below(EV.UP_MIN))).tolist() == [False, True, False, True]
    assert EV.opp_down_now(T(0.25, below(0.25), 0.7, 0.7), T(1.0, 1.0, 0.5, below(0.5))).tolist() == [False, True, False, True]

def test_up_z_is_xmat_22():
    q = np.random.default_rng(0).normal(size=(2000, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True); R = np.zeros(9)
    want = np.array([(mujoco.mju_quat2Mat(R, x), R[8])[1] for x in q]); assert np.array_equal(EV.up_z(torch.from_numpy(q)).numpy(), want)

def test_rules_equal_r1_fallen_and_eo_on_random_states():
    m, cfg, r, _ = C.model_cfg(); rng = np.random.default_rng(1); d = mujoco.MjData(m); op = C.FightReplayIdx(m); rb = R1Robot(m, cfg); n = 0
    for _ in range(2000):
        q = m.qpos0.copy()
        for qa in (r.free_q, op.qa): q[qa:qa + 3] = (0.0, 0.0, rng.uniform(0.15, 0.85)); q[qa + 3:qa + 7] = C._quat(rng, np.radians(100))
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); T = lambda x: torch.tensor([x], dtype=torch.float64)
        assert bool(EV.fell_eval(T(d.xpos[rb.pel][2]), EV.up_z(torch.from_numpy(d.xquat[rb.pel][None])))[0]) == bool(rb.fallen(d))
        o = bool(d.xmat[op.pel][8] < EO.OPP_TILT or d.xpos[op.pel][2] < EO.OPP_Z); assert bool(EV.opp_down_now(T(d.xpos[op.pel][2]), EV.up_z(torch.from_numpy(d.xquat[op.pel][None])))[0]) == o; n += o
    assert 0 < n < 2000

def test_opp_down_is_non_sticky():
    od = EV.OppDown(2); z = lambda *x: torch.tensor(x, dtype=torch.float64)
    seq = [(z(0.7, 0.7), z(1.0, 1.0)), (z(0.2, 0.7), z(1.0, 1.0)), (z(0.2, 0.7), z(1.0, 0.3)), (z(0.7, 0.7), z(1.0, 1.0)), (z(0.2, 0.7), z(1.0, 1.0))]
    edges, now, ever = [], [], []
    for zz, uu in seq: e = od.update(zz, uu); edges.append(e.tolist()); now.append(od.now.tolist()); ever.append(od.ever.tolist())
    assert [n[0] for n in now] == [False, True, True, False, True] and [n[1] for n in now] == [False, False, True, False, False]
    assert [e[0] for e in edges] == [False, True, False, False, False] and ever[-1] == [True, True] and od.age.tolist() == [0, 2]
    od.reset(torch.tensor([0])); assert od.ever.tolist() == [False, True] and not bool(od.now[0])
