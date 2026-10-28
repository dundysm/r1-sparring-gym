"""Defence oracle (scripts/defence_oracle.py): 'off' is bit-identical to the plain env under both DR command latencies (so the
unmodified variant reproduces the stored eval), targets reach the policy's action space exactly, decisions land delay_steps
later, the IK reaches what is reachable within its limits, the slip rolls the head away from the threat's side; v2 plans one
guard per head intent on its apex line, holds it from t_on - lead to the apex + tail and releases it, over-commands as
specified; the placement distance is a true segment-to-segment distance."""
import sys, numpy as np, pytest, mujoco
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import defence_oracle as DO
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay
from r1gym.aim import StrikeIntent

@pytest.fixture(scope="module")
def world():
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller(DO.POLICY); cfg = {**controller_cfg(c), "opponent_mode": load_cfg()["opponent_mode"], "opponent_aim": "standoff+armwarp_vz"}
    return c, cfg, build_spec(opponent=True, cfg=cfg).compile()

def bout(world, cls, ops=None, steps=120, dr=True, seed=7, v2=None):
    c, cfg, m = world; cfg = cfg if ops is None else {**cfg, "oracle": {"ops": list(ops), **({"v2": v2} if v2 else {})}}
    e = cls(seed=seed, cfg=cfg, model=m, episode_s=steps * 0.02 + 1, max_impulse=0, train=False, dr=dr); e.opp = FightReplay(m, "kfab_204", distance=0.45, cfg=cfg); e.opp_t0 = 1.0
    o = e.reset(hits=[], delay=1); Q, O = [], []
    for _ in range(steps): o, r, d, info = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy()); O.append(o)
    return np.array(Q), np.array(O), e

def test_off_is_bit_identical_under_both_latencies_and_on_changes_the_bout(world):
    seen = set()
    for seed in range(7, 40):
        q0, o0, e0 = bout(world, R1HitEnv, seed=seed, steps=60); q1, o1, e1 = bout(world, DO.OracleEnv, (), seed=seed, steps=60)
        assert e0.act_delay == e1.act_delay and np.abs(q0 - q1).max() == 0 and np.abs(o0 - o1).max() == 0 and e1.log["applied"] == 0 and e1.log["edited"] == 0
        seen.add(e1.act_delay)
        if seen == {0, 1}: break
    assert seen == {0, 1}   # both DR command latencies covered (the policy's last-action feedback differs between them)
    q0, _, _ = bout(world, R1HitEnv); q2, _, e2 = bout(world, DO.OracleEnv, ("arm",)); _, _, e1 = bout(world, DO.OracleEnv, ())
    assert e1.log["threat"] > 0 and e2.log["applied"] > 0 and np.abs(q2 - q0).max() > 1e-3
    q3, _, e3 = bout(world, DO.OracleEnv, ("arm2",), v2=dict(lead_s=0.4))
    assert e3.log["applied"] > 0 and e3.log["edited"] > 0 and e3.plans and np.abs(q3 - q0).max() > 1e-3

def test_targets_map_to_actions_and_land_two_steps_later(world):
    """edit() puts home + scale * a exactly on the decided arm angles; a decision made at step k drives d.ctrl at step k + 2."""
    c, cfg, m = world; k = {"n": 0}
    class Marked(DO.OracleEnv):
        def decide(self):
            k["n"] += 1; return dict(hand="l_hand", side="left", away=1.0, q=self.home[DO.ARM_IX["left"]] + 0.01 * k["n"], res=0.0)
    _, _, e = bout(world, Marked, ("arm",), steps=6, dr=False)
    ctrl = e.d.ctrl[e.r.act[DO.ARM_IX["left"]]]
    assert np.allclose(ctrl, e.home[DO.ARM_IX["left"]] + 0.01 * (6 - 2), atol=1e-5)   # the 6th step applied the 4th decision
    a = e.edit(np.zeros(24, np.float32), dict(side="right", away=-1.0, q=np.array([-0.5, -0.4, 0.1, -0.3])))
    assert np.allclose(e.home + e.scale * a, np.r_[e.home[:19], [-0.5, -0.4, 0.1, -0.3], e.home[23]], atol=1e-6)
    R = np.array(e.rec); assert R.shape == (6, 58) and np.allclose(R[-1, DO.REC["ctrl"]][:4], e.home[DO.ARM_IX["left"]] + 0.01 * 3, atol=1e-5)   # the record holds the target the last step tracked
    assert R[-1, DO.REC["edited"]].tolist() == [1.0, 0.0] and R[1, DO.REC["edited"]].tolist() == [0.0, 0.0]

def test_ik_reaches_and_respects_limits(world):
    c, cfg, m = world; e = DO.OracleEnv(seed=0, cfg={**cfg, "oracle": {"ops": ["arm"]}}, model=m, episode_s=1, max_impulse=0, train=False)
    e.opp = FightReplay(m, "kfab_204", distance=0.45, cfg=cfg); e.reset(hits=[], delay=0); kd = e.kd
    for side, dy in (("left", 1), ("right", -1)):
        kd.qpos[:] = e.d.qpos; mujoco.mj_kinematics(m, kd); x0 = e.forearm(kd, side)
        q, res = e.ik(side, x0 + np.array([0.05, -0.04 * dy, 0.03])); assert res < 0.01   # 7 cm away in front of the face: reached
        kd.qpos[:] = e.d.qpos; q, res = e.ik(side, x0 + np.array([1.5, 0, 0])); ix = DO.ARM_IX[side]   # out of reach: clamped
        assert res > 0.5 and np.all(q >= e.lo[ix] - 1e-12) and np.all(q <= e.hi[ix] + 1e-12) and np.all(np.abs(q - e.home[ix]) <= e.p["max_dq"] + 1e-12)

def test_slip_rolls_the_head_away_and_slews(world):
    """away=+1 (the threat on the R1's left): the waist roll target rises at slip_rate toward home + 0.3, and a positive roll
    moves the head to the R1's right (-y); with no decision the offsets slew back to exactly zero (the policy's action again)."""
    c, cfg, m = world; e = DO.OracleEnv(seed=0, cfg={**cfg, "oracle": {"ops": ["slip"]}}, model=m, episode_s=1, max_impulse=0, train=False)
    e.opp = FightReplay(m, "kfab_204", distance=0.45, cfg=cfg); e.reset(hits=[], delay=0); w = DO.WAIST_ROLL; a0 = np.zeros(24, np.float32)
    rolls = [e.home[w] + e.scale[w] * e.edit(a0, dict(side="left", away=1.0))[w] for _ in range(10)]
    assert np.allclose(np.diff(rolls[:3]), e.p["slip_rate"] * e.dt) and abs(rolls[-1] - (e.home[w] + 0.3)) < 1e-6
    for _ in range(20): a = e.edit(a0, None)
    assert np.array_equal(a, a0) and not e.slip.any()
    kd = e.kd; kd.qpos[:] = e.d.qpos; mujoco.mj_kinematics(m, kd); y0 = e.head(kd)[1]
    kd.qpos[e.r.qadr[w]] = 0.3; mujoco.mj_kinematics(m, kd); assert e.head(kd)[1] < y0 - 0.05

def test_v2_plans_on_the_apex_line_holds_and_releases(world):
    """One head intent from the opponent's left hand (on the R1's right): the right arm guards it; its planned point is block_d
    from the target toward that shoulder (R1 torso frame) and the IK reaches it; the guard is due from t_on - lead_s (plan time
    = now + delay), held to t_apex + tail_s, half released release_s / 2 later, gone after; torso and (with gated False) gated
    intents are ignored; edit() commands q* + over * (q* - the policy's target), blended by the weight."""
    c, cfg, m = world; v2 = dict(lead_s=0.3, over=0.5); e = DO.OracleEnv(seed=0, cfg={**cfg, "oracle": {"ops": ["arm2"], "v2": v2}}, model=m, episode_s=1, max_impulse=0, train=False)
    e.opp = FightReplay(m, "kfab_204", distance=0.45, cfg=cfg); e.reset(hits=[], delay=0); p2 = e.p2; dl = e.p["delay_steps"] * e.dt
    it = StrikeIntent("l_hand", "head", "kfab_204", 0, 1.0, 1.3, 1.5, np.array([0.057, 0.0, 0.36]), 0.5)
    e.opp_state["aim"] = SimpleNamespace(intents=[it, StrikeIntent("r_hand", "torso", "kfab_204", 1, 1.0, 1.3, 1.5, np.array([0.04, 0.0, 0.15]), 0.5)])
    plan = lambda t: (setattr(e.d, "time", t - dl), e.plan_v2())[1]
    assert plan(1.0 - 0.3 - 0.01) is None and not e.plans
    d = plan(1.0 - 0.3 + 1e-6); assert set(d) == {"right"} and d["right"]["w"] == 1.0 and len(e.plans) == 1   # the torso intent: ignored
    pl = next(iter(e.plans.values())); kd = e.kd; kd.qpos[:] = e.d.qpos; mujoco.mj_kinematics(m, kd); t = e.r.torso; R, x = kd.xmat[t].reshape(3, 3), kd.xpos[t]
    s_t = R.T @ (kd.xpos[e.shoulder["l_hand"]] - x); u = (s_t - it.target_r1) / np.linalg.norm(s_t - it.target_r1)
    assert s_t[1] < 0 and np.allclose(pl["P_t"], it.target_r1 + p2["block_d"] * u) and pl["res"] < 0.02 and np.all(np.abs(pl["q"] - e.home[DO.ARM_IX["right"]]) <= p2["max_dq"] + 1e-12)
    assert plan(1.3 + 0.15 - 1e-6)["right"]["w"] == 1.0 and abs(plan(1.3 + 0.15 + 0.05)["right"]["w"] - 0.5) < 1e-6 and plan(1.3 + 0.15 + 0.1 + 0.01) is None
    assert len(e.plans) == 1 and np.array_equal(plan(1.2)["right"]["q"], pl["q"])   # planned once, held (no chasing)
    it.gated = "tilt"; assert plan(1.2) is not None; e.p2 = {**p2, "gated": False}; assert plan(1.2) is None; e.p2 = p2
    ix = DO.ARM_IX["right"]; a = np.zeros(24, np.float32); a[ix] = 0.2; qp = e.home[ix] + e.scale[ix] * 0.2; qs = pl["q"]
    for w in (1.0, 0.5):
        b = e.edit(a, {"right": dict(w=w, q=qs)}); want = w * np.clip(qs + 0.5 * (qs - qp), e.lo[ix], e.hi[ix]) + (1 - w) * qp
        assert np.allclose(e.home[ix] + e.scale[ix] * b[ix], want, atol=1e-5) and np.array_equal(np.delete(b, ix), np.delete(a, ix))

def test_segment_distance():
    rng = np.random.default_rng(0); s = np.linspace(0, 1, 801)
    assert abs(DO.seg_dist(np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([0.5, -1, 0.3]), np.array([0.5, 1, 0.3])) - 0.3) < 1e-12   # crossing, 0.3 apart
    assert abs(DO.seg_dist(np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([2., 0, 0]), np.array([3., 0, 0])) - 1.0) < 1e-12        # collinear, end to end
    for _ in range(50):
        p1, q1, p2, q2 = rng.normal(size=(4, 3)); A = p1 + s[:, None] * (q1 - p1); B = p2 + s[:, None] * (q2 - p2)
        brute = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1)).min(); d = DO.seg_dist(p1, q1, p2, q2); assert d <= brute + 1e-9 and brute - d < 5e-3
