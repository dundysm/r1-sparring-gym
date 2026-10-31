"""HitMeter(reverse=True) (r1gym/hits.py): the R1's strikes on the opponent, the forward meter mirrored; both-up exposure keys;
eval_opponent's ContactTail and opponent-fall log."""
import sys
from pathlib import Path
import numpy as np, mujoco, pytest
from r1gym.model import build_spec, load_cfg, R1Robot, HOME, SHELL_BIT, SHELL_SOLREF, SHELL_SOLIMP
from r1gym.hits import HitMeter, ZONES, LIMBS, landed_block
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
CFG = load_cfg()

@pytest.fixture(scope="module")
def m():
    return build_spec(opponent=True, cfg=CFG).compile()

def _name(m, g):
    return m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else m.geom(g).name

def test_reverse_mirrors_forward(m):
    """(a) Off by default; reversed, every opponent shell has the zone its R1 twin has forward and every R1 shell the limb
    group its opponent twin has forward (head = exactly the opponent's two head meshes)."""
    f, r = HitMeter(m), HitMeter(m, reverse=True); assert not f.reverse and r.reverse
    assert set(r.zone_of) <= set(f.keys) and set(r.keys) >= set(f.zone_of)   # forward attackers are reverse targets and back
    gid = {_name(m, g): g for g in range(m.ngeom) if m.geom_bodyid[g] and (m.geom_contype[g] or m.geom_conaffinity[g])}; n = 0
    for g in r.zone_of:
        tw = gid[_name(m, g)[4:]]; assert f.zone_of[tw] == r.zone_of[g]; n += 1
    for g in r.limb_of:
        if gid.get("opp_" + _name(m, g)) in f.limb_of: assert f.limb_of[gid["opp_" + _name(m, g)]] == r.limb_of[g]; n += 1
    assert n > 40 and set(r.zone_of.values()) == set(ZONES) and set(r.limb_of.values()) <= set(LIMBS)
    assert {_name(m, g) for g, z in r.zone_of.items() if z == "head" and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH} == {"opp_head_pitch_link", "opp_head_yaw_link"}
    assert all(m.body(m.geom_bodyid[g]).name.startswith("opp_") for g in r.zone_of) and not any(m.body(m.geom_bodyid[g]).name.startswith("opp_") for g in r.keys)

def _r1_fist_world():
    """Zero gravity, the opponent at the origin (held still by the caller: a welded target), the R1 out of the way, and a free
    0.5 kg R1-side fist (the R1's wrist shell, body origin at its centre of mass) aimed at the opponent's face."""
    s = build_spec(opponent=True, cfg=CFG); b = s.worldbody.add_body(name="left_wrist_test"); b.add_freejoint()
    g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="left_wrist_roll_link", mass=0.5, contype=0, conaffinity=SHELL_BIT, condim=1)
    g.solref, g.solimp = SHELL_SOLREF, SHELL_SOLIMP; m = s.compile(); g.pos = -m.geom_pos[m.ngeom - 1]; m = s.compile(); m.opt.gravity[:] = 0
    j = m.body("left_wrist_test").jntadr[0]; return m, m.jnt_qposadr[j], m.jnt_dofadr[j]

def test_r1_fist_on_opponent_head():
    """(b) The R1's fist thrown at the opponent's face: reversed, one episode, zone 'head', limb 'hand', side 'l', impulse within
    10% of the momentum the fist lost, closing speed within 5% of the throw, landed and clean above 1 m/s (0.5 m/s: under 0.5 N*s); the forward meter sees only the
    opponent's head (trunk) on the R1's arm zone, never a landed hit. eval_opponent's ContactTail, labelling by geom, puts the
    touch on the opponent's head (its shell sits on opp_torso_link) as an R1 strike; once done it logs nothing."""
    import eval_opponent as eo
    m, fq, fv = _r1_fist_world(); d = mujoco.MjData(m); r = R1Robot(m, CFG); oq = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]
    hg = [g for g in range(m.ngeom) if _name(m, g) == "opp_head_pitch_link"][0]
    J = [j for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith("opp_")]; qa = np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + (7 if m.jnt_type[j] == 0 else 1)) for j in J])
    va = np.concatenate([np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + (6 if m.jnt_type[j] == 0 else 1)) for j in J])
    for v0 in [0.5, 1.5, 3.0]:
        mujoco.mj_resetData(m, d); r.reset(m, d); d.qpos[r.free_q:r.free_q + 3] = [-5, 0, 0.74]; d.qpos[oq:oq + 3] = [0, 0, 0.74]; d.qpos[fq:fq + 3] = [5, 5, 5]
        mujoco.mj_forward(m, d); h = d.geom_xpos[hg].copy(); u = np.array([-1.0, 0, 0])   # from +x toward the face
        d.qpos[fq:fq + 3] = h - 0.3 * u; d.qvel[fv:fv + 3] = v0 * u; mujoco.mj_forward(m, d); p0 = 0.5 * d.qvel[fv:fv + 3].copy(); q0 = d.qpos[qa].copy()
        rv, fw, ct, cd = HitMeter(m, CFG["hit_metric"], reverse=True), HitMeter(m, CFG["hit_metric"]), eo.ContactTail(m, keep_s=1.0), eo.ContactTail(m); cd.done = True
        for _ in range(int((0.4 / v0 + 0.15) / m.opt.timestep)):
            d.qpos[qa] = q0; d.qvel[va] = 0; mujoco.mj_step(m, d); rv.substep(m, d); fw.substep(m, d); ct.substep(m, d); cd.substep(m, d)   # the opponent welded in place
        sn = ct.snap(float(d.time)); assert sn["r1_strike"] and {(x["r1"], x["r1_limb"], x["opp"], x["opp_zone"]) for x in sn["pairs"]} == {("left_wrist_test", "hand", "torso_link", "head")}, sn
        assert abs(sn["impulse_Ns"] - sum(e["impulse_Ns"] for e in rv.events())) < 1e-3 and cd.log == []
        dp = np.linalg.norm(0.5 * d.qvel[fv:fv + 3] - p0); ev = rv.events(); imp = sum(e["impulse_Ns"] for e in ev)
        assert ev and all(e["zone"] == "head" and e["limb"] == "hand" for e in ev) and abs(imp - dp) < 0.1 * dp, (v0, imp, dp, ev)
        assert len(ev) == 1 and abs(ev[0]["v_close"] - v0) < 0.05 * v0 and rv.landed(ev[0]) == rv.clean(ev[0]) == (v0 > 1), ev
        assert rv.events(by_side=True)[0]["side"] == "l" and rv.summary(0.4)["n"]["head_clean"] == int(v0 > 1)
        assert all(e["zone"] == "arm" and e["limb"] == "trunk" for e in fw.events()) and not any(fw.landed(e) for e in fw.events())

def _touch(k, t0, n, f, v=None, dt=0.005):
    return [(round(t0 + i * dt, 4), {k: f}, {k: v} if i == 0 and v is not None else {}) for i in range(n)]

def test_both_up_exposure(m):
    """(c) both_up: clean strikes whose onset falls inside the (t_a, t_b] intervals, per minute of their total (a number x:
    (0, x]); without it the summary has no new keys (old rows and landed blocks unchanged); landed_block adds minutes and the
    both-up rates only when every bout has them."""
    hm = HitMeter(m, CFG["hit_metric"]); H, T = ("l", "hand", "head"), ("r", "hand", "torso")
    hm.log = sorted(_touch(H, 1.0, 6, 100.0, 2.0) + _touch(H, 20.0, 6, 100.0, 2.0) + _touch(T, 40.0, 6, 100.0, 2.0), key=lambda x: x[0])
    a, b = hm.summary(60.0), hm.summary(60.0, both_up=30.0)
    assert not any("both_up" in k for k in a) and {k: v for k, v in b.items() if k in a} == a
    assert b["n_both_up"] == dict(head_clean=2, torso_clean=0) and b["head_clean_pm_both_up"] == 4.0 and b["torso_clean_pm_both_up"] == 0.0 and b["head_clean_pm"] == 2.0
    L0, L1 = landed_block([a, a], [0, 1]), landed_block([b, b], [0, 1]); assert {k: v for k, v in L1.items() if k in L0} == L0 and "minutes" not in L0
    assert L1["minutes"] == dict(r1_up=2.0, both_up=1.0) and L1["head_clean_pm_both_up"]["rate"] == 4.0 and landed_block([a, b], [0, 1]).keys() == L0.keys()
    c = hm.summary(60.0, both_up=[(0.0, 10.0), (19.0, 29.0), (39.0, 49.0)]); assert c["both_up_s"] == 30.0 and c["n_both_up"] == dict(head_clean=2, torso_clean=1)   # an opponent down 10-19 s and 29-39 s, back up after
    c = hm.summary(60.0, both_up=[(1.0, 20.0)]); assert c["n_both_up"] == dict(head_clean=1, torso_clean=0) and c["both_up_s"] == 19.0   # (t_a, t_b]: a strike at t_a is out, one at t_b in

def test_contact_tail_and_fall_block(m):
    """(d) ContactTail keeps the last 0.5 s and sums impulse per (body, group, body, zone), the zone per geom (the opponent's
    head shells on opp_torso_link are 'head', its torso shell 'torso'); fall_block pools the opponent-fall logs: the contact
    impulse median over falls with contact only, and the falls the opponent got up from (a both-up interval after the flag)."""
    import eval_opponent as eo
    ct = eo.ContactTail(m); nm = {_name(m, g): ct.lab[g] for g in ct.lab}
    assert nm["opp_head_pitch_link"] == nm["opp_head_yaw_link"] == ("torso_link", "head") and nm["opp_waist_yaw_link"] == ("torso_link", "torso") and nm["left_wrist_roll_link"] == ("left_wrist_roll_link", "hand")
    W, P = ("left_wrist_roll_link", "hand", "torso_link", "head"), ("pelvis", "trunk", "pelvis", "pelvis")
    ct.dt = 0.01; ct.log = [(1.0, {W: 50.0}), (1.4, {W: 100.0, P: 10.0})]
    s = ct.snap(1.45); p = {(x["r1"], x["opp"]): x for x in s["pairs"]}
    assert s["contact"] and s["r1_strike"] and s["n_pairs"] == 2 and p[("left_wrist_roll_link", "torso_link")]["impulse_Ns"] == 1.5 and p[("pelvis", "pelvis")]["opp_zone"] == "pelvis" and p[("left_wrist_roll_link", "torso_link")]["opp_zone"] == "head"
    assert p[("left_wrist_roll_link", "torso_link")]["r1_limb"] == "hand" and p[("left_wrist_roll_link", "torso_link")]["last_s_before"] == 0.05
    s2 = ct.snap(1.85); assert s2["n_pairs"] == 2 and s2["impulse_Ns"] == 1.1 and not ct.snap(1.95)["contact"]
    rows = [dict(fell_s=None, both_up_s=12.0, both_up_iv=[[0.0, 12.0]], opp_fall=dict(t_s=12.0, cause="tilt", r1_contact_0p5s=s)), dict(fell_s=6.0, both_up_s=6.0, both_up_iv=[[0.0, 6.0]], opp_fall=None)]
    fb = eo.fall_block(rows, 20.0); assert fb["n"] == 1 and fb["cause"]["tilt"] == 1 and fb["after_r1_strike_0p5s"] == 1 and fb["r1_up_min"] == round(26 / 60, 3) and fb["both_up_min"] == 0.3
    assert fb["got_up"] == 0 and fb["contact_impulse_Ns_median"] == s["impulse_Ns"] and fb["both_up_def"] == eo.BOTH_UP_DEF
    rows += [dict(fell_s=None, both_up_s=15.0, both_up_iv=[[0.0, 0.4], [2.0, 16.6]], opp_fall=dict(t_s=0.42, cause="tilt", r1_contact_0p5s=ct.snap(9.0)))]   # flagged at 0.42 s, up again from 2 s, no contact
    fb = eo.fall_block(rows, 20.0); assert fb["n"] == 2 and fb["got_up"] == 1 and fb["after_contact_0p5s"] == 1 and fb["contact_impulse_Ns_median"] == s["impulse_Ns"]

def test_eval_rows_log_opponent_falls():
    """(e) A short tracked-opponent bout through eval_opponent.run: falls off (run's default) gives the earlier row; on, the fall
    log is consistent with opp_fell, both-up time is at most the R1's, and the reverse meter rides along without touching the rest."""
    import eval_opponent as eo
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller("RL step=results/policy_mjlab/policy.onnx"); cc = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": "shadow"}
    mm = build_spec(opponent=True, cfg=cc).compile(); ep = dict(clip="kfab_200", distance=0.4, rep=0, t0=1.0, delay=1, hits=[], seed=0)
    o = eo.run(c, [ep], mm, cc, 2.0, threads=1, meter=True); a = eo.run(c, [ep], mm, cc, 2.0, threads=1, meter=True, falls=True); b = eo.run(c, [ep], mm, cc, 2.0, threads=1, meter=True, rev=True, falls=True)
    for r in (a[0], b[0]):
        iv = r["both_up_iv"]; secs = r["fell_s"] or 2.0
        assert (r["opp_fall"] is not None) == r["opp_fell"] and r["both_up_s"] <= secs and r["hit_meter"]["both_up_s"] == r["both_up_s"] and abs(sum(c - a for a, c in iv) - r["both_up_s"]) < 1e-6
        assert all(0 <= a < c <= secs + 1e-9 for a, c in iv) and all(iv[i][1] < iv[i + 1][0] for i in range(len(iv) - 1)) and (r["opp_fell"] or iv == [[0.0, secs]])
    assert {k: v for k, v in b[0].items() if k != "rev_meter"} == a[0] and "rev_meter" in b[0] and "rev_meter" not in a[0]
    strip = lambda r: {k: ({j: w for j, w in v.items() if "both_up" not in j} if k == "hit_meter" else v) for k, v in r.items() if k not in ("opp_fall", "both_up_s", "both_up_iv")}
    assert strip(a[0]) == o[0] and "opp_fall" not in o[0] and not any("both_up" in k for k in o[0]["hit_meter"])   # off (the default): the earlier rows
