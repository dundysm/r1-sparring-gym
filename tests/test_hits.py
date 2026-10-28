"""HitMeter (r1gym/hits.py): read-only, impulse matches the momentum a fist loses, every shell has a zone, bounces merge."""
import numpy as np, mujoco, pytest
from r1gym.model import build_spec, load_cfg, R1Robot, HOME, SHELL_BIT, SHELL_SOLREF, SHELL_SOLIMP
from r1gym.hits import HitMeter, ZONES, LIMBS, RULES
from r1gym.env import R1HitEnv
from r1gym.opponent import FightReplay

CFG = load_cfg()

@pytest.fixture(scope="module")
def m():
    return build_spec(opponent=True, cfg=CFG).compile()

def test_meter_is_read_only(m):
    """(a) The same bout with and without the meter is bit-identical (tracked opponent at 0.40 m, stepping base, 3 s)."""
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller("step=results/policy_mjlab/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    mm = build_spec(opponent=True, cfg=cfg).compile(); out = []
    for on in (False, True):
        e = R1HitEnv(seed=0, cfg=cfg, model=mm, episode_s=3, max_impulse=0, train=False); e.opp = FightReplay(mm, "kfab_200", distance=0.4, cfg=cfg); e.opp_t0 = 1.0
        hm = HitMeter(mm, cfg["hit_metric"]); e.substep_hooks += [hm] if on else []; o = e.reset(hits=[], delay=1); Q = []
        for _ in range(150): o, *_ = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy())
        out.append((np.array(Q), hm))
    assert np.abs(out[0][0] - out[1][0]).max() == 0 and not out[0][1].log
    ev = out[1][1].events(); assert len(ev) > 0 and {e["zone"] for e in ev} <= set(ZONES) and {e["limb"] for e in ev} <= set(LIMBS)

def _fist_world():
    """The R1 standing at home in zero gravity, and a free 0.5 kg fist (the opponent's wrist shell) with its body origin
    at its centre of mass, so its momentum is exactly m * qvel."""
    s = build_spec(opponent=True, cfg=CFG); b = s.worldbody.add_body(name="opp_test_wrist"); b.add_freejoint()
    g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="opp_left_wrist_roll_link", mass=0.5, contype=SHELL_BIT, conaffinity=0, condim=1)
    g.solref, g.solimp = SHELL_SOLREF, SHELL_SOLIMP; m = s.compile(); g.pos = -m.geom_pos[m.ngeom - 1]; m = s.compile(); m.opt.gravity[:] = 0
    j = m.body("opp_test_wrist").jntadr[0]; return m, m.jnt_qposadr[j], m.jnt_dofadr[j]

def test_fist_on_head_impulse_matches_momentum():
    """(b) A fist thrown at the face at known speeds: impulse within 10% of the momentum it lost, zone 'head', limb 'hand',
    closing speed at first touch within 5% of the throw. A touch under 5 N, or under 0.5 N*s, is not a landed hit."""
    m, fq, fv = _fist_world(); d = mujoco.MjData(m); r = R1Robot(m, CFG); oq = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]
    for v0 in [0.02, 0.5, 1.5, 3.0]:
        r.reset(m, d); d.ctrl[r.act] = HOME; d.qpos[oq:oq + 3] = [5, 0, 0.74]   # the opponent out of the way
        d.qpos[fq:fq + 3] = [0.3, 0.0, 1.13]; d.qvel[fv:fv + 3] = [-v0, 0, 0]; mujoco.mj_forward(m, d); p0 = 0.5 * d.qvel[fv:fv + 3].copy()
        hm = HitMeter(m, CFG["hit_metric"])
        for _ in range(int((8.0 if v0 < 0.1 else 0.3) / m.opt.timestep)): mujoco.mj_step(m, d); hm.substep(m, d)
        dp = np.linalg.norm(0.5 * d.qvel[fv:fv + 3] - p0); ev = hm.events(); imp = sum(e["impulse_Ns"] for e in ev)
        assert ev and all(e["zone"] == "head" and e["limb"] == "hand" for e in ev) and abs(imp - dp) < 0.1 * dp, (v0, imp, dp, ev)
        if v0 < 0.1: assert max(e["peak_N"] for e in ev) < 5 and not any(hm.landed(e) for e in ev); continue
        assert len(ev) == 1 and abs(ev[0]["v_close"] - v0) < 0.05 * v0 and ev[0]["peak_N"] > 5 and hm.landed(ev[0]) == hm.clean(ev[0]) == (v0 > 1), ev

def test_every_shell_has_a_zone(m):
    """(c) Every R1 shell maps to a zone (head = exactly the two head meshes), every opponent shell to a limb group."""
    hm = HitMeter(m); r1 = [g for g in range(m.ngeom) if m.geom_conaffinity[g] & SHELL_BIT]; opp = [g for g in range(m.ngeom) if m.geom_contype[g] & SHELL_BIT]
    assert len(r1) == len(opp) > 20 and {hm.zone_of.get(g) for g in r1} == set(ZONES) and {hm.limb_of.get(g) for g in opp} == set(LIMBS)
    assert {m.mesh(m.geom_dataid[g]).name for g in r1 if hm.zone_of[g] == "head"} == {"head_pitch_link", "head_yaw_link"}
    assert all(m.body(m.geom_bodyid[g]).name.startswith("opp_") for g in opp) and not any(m.body(m.geom_bodyid[g]).name.startswith("opp_") for g in r1)
    mc = build_spec(opponent=True, cfg={**CFG, "robot_contact": "capsule"}).compile(); zc = HitMeter(mc).zone_of   # capsule contact: the head is still 'head'
    assert zc[mc.geom("head_collision").id] == "head" and zc[mc.geom("torso_collision").id] == "torso"

def _touch(k, t0, n, f, v=None, dt=0.005):
    return [(round(t0 + i * dt, 4), {k: f}, {k: v} if i == 0 and v is not None else {}) for i in range(n)]

def test_bounce_merges_and_rules(m):
    """(d) A bounce under 0.1 s is one episode (two under design A's 20 ms merge); landed / clean / press / rules / windows."""
    hm = HitMeter(m, CFG["hit_metric"]); L, R = ("l", "hand", "head"), ("r", "hand", "torso")
    hm.log = sorted(_touch(L, 1.0, 6, 100.0, 2.0) + _touch(L, 1.08, 4, 50.0) + _touch(L, 1.4, 60, 20.0, 1.5)   # strike, bounce, then a 0.3 s press
                    + _touch(R, 2.0, 4, 40.0, 0.5) + _touch(("r", "hand", "arm"), 3.0, 4, 80.0, 2.0) + _touch(("r", "hand", "head"), 3.05, 4, 80.0, 2.0)
                    + _touch(("l", "foot", "leg"), 4.0, 2, 3.0, 0.2), key=lambda x: x[0])
    ev = hm.events(); head = [e for e in ev if e["zone"] == "head"]
    assert len(head) == 3 and head[0]["impulse_Ns"] == pytest.approx((6 * 100 + 4 * 50) * 0.005) and head[0]["peak_N"] == 100 and head[0]["v_close"] == 2.0
    assert head[0]["dur_s"] == pytest.approx(0.1) and hm.clean(head[0]) and hm.landed(head[1]) and not hm.clean(head[1])   # head[1]: the press
    assert len([e for e in hm.events(gap=0.02) if e["zone"] == "head"]) == 4
    torso = next(e for e in ev if e["zone"] == "torso"); assert hm.landed(torso) and not hm.clean(torso)                 # too slow to be a strike
    assert hm.count("clean") == {"head": 2} and hm.count("landed") == {"head": 3, "torso": 1} and hm.count("A_peak20") == {"head": 4, "torso": 1, "arm": 1}
    assert hm.windows([(0.9, 1.2, "l_hand"), (2.9, 3.2, "r_hand"), (1.9, 2.2, "r_hand"), (1.9, 2.2, "l_hand"), (3.9, 4.2, "l_foot")]) == ["head", "blocked", "torso", "miss", "miss"]
    s = hm.summary(60.0, spacing=[0.5, 0.55, 0.6], wins=[(0.9, 1.2, "l_hand"), (2.9, 3.2, "r_hand")])
    assert s["head_clean_pm"] == 2 and s["head_landed_pm"] == 3 and s["arm_clean_pm"] == 1 and s["windows"] == dict(head=1, torso=0, blocked=1, miss=0)
    assert set(s["rules"]) == set(RULES) and s["spacing_m"][1] == 0.55 and len(s["head_hit_imp_Ns"]) == 3
    hm.reset(); assert hm.events() == [] and hm.summary(10.0)["head_clean_pm"] == 0
    for gap, n in ((0.1, 1), (0.105, 2)):   # merge_gap_s is between contact substeps: 0.1 s merges a contact-free 95 ms, not 100 ms
        hm.log = _touch(L, 1.0, 1, 50.0, 2.0) + _touch(L, 1.0 + gap, 1, 50.0); hm._ev = {}; assert len(hm.events()) == n
    for k, clean in ((30, True), (31, False)):   # clean includes both ends (30 substeps fit 0.15 s), B_strike as the probe (31)
        hm.log = _touch(L, 1.0, k, 100.0, 2.0); hm._ev = {}; assert hm.count("clean") == ({"head": 1} if clean else {}) and hm.count("B_strike") == {"head": 1}

def test_rate_ci_clusters():
    """rate_ci: one bout per cluster is the plain bootstrap; bouts repeated inside their cluster give the same interval.
    Weighted by minutes it is total count / total minutes, which a bout the R1 fell early in can't dominate."""
    from r1gym.stats import rate_ci
    x = np.random.default_rng(1).poisson(6, 40).astype(float); r = rate_ci(x)
    assert r == rate_ci(x, list(range(40))) and r[1][0] < x.mean() < r[1][1]
    assert rate_ci(np.repeat(x, 3), np.repeat(np.arange(40), 3)) == r and rate_ci(np.repeat(x, 3))[1][1] - rate_ci(np.repeat(x, 3))[1][0] < r[1][1] - r[1][0]
    n, mins = np.r_[x[:39], 1.0], np.r_[np.full(39, 1 / 6), 0.4 / 60]   # 39 full 10 s bouts, one that fell at 0.4 s after one hit
    w, (lo, hi) = rate_ci(n / mins, w=mins); assert w == round(n.sum() / mins.sum(), 3) and lo < w < hi
    assert abs(w - 6 * x[:39].mean()) < 0.5 and (n / mins).mean() - 6 * x[:39].mean() > 2   # plain mean: that one hit is 150/min
