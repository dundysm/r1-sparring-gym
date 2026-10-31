"""Full-body M2 v2 torch aim (M2V2_PLAN_v3 M3; gpu/fb_opponent.py ArmKinematics, AimOp; scripts/fb_m3_check.py): the aim's contract
(arm joints, palm sites, parameters = the eval's merged config, the strike library per clip with the eval's repeated-onset rule),
FK and Jacobian vs mj_jacSite (float64 and float32, any model prefix), Aim.weight, AimOp vs Aim.reference on a recorded eval bout
(seed 60001, role 'record', 4 s; float32 by plan Amendment A1: same-input vs the float64 port on every row, vs the eval on >= 99.9%
of rows), the slot ring and its overflow count, reset, shadow rows, and the jitter (bounds; the reach guard of plan Amendment A2
alone: the 8-step bisection on s, the nominal warp computed alongside = the T0 run's, its negative control; Amendment A3's F': the
corrected target keeps the jittered yz, its x moved with at most 4 extra IK calls, the guard on what remains, IK calls per step;
fallback D's draws; A1 (i) the IK's x target stays the reference palm's own and the nominal one, (iv) the mixed-mask probe and its
negative control; A2 (iii') no forward excess at any warped step; the (ii') bout-cluster CI, (vi) yz kept and (vii) smoothness
rules, (v) reported). The full G0-4 (16 bouts x jitter seeds 0-19, where (ii'), (vi) and (vii) are gated) runs in
scripts/fb_m3_check.py."""
import sys, json
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "gpu"))
import fb_opponent as FO, fb_m3_check as C
from fb_seeds import SeedRefused
from r1gym.aim import ARM, PALM, AIM, OPERATORS, merged, StrikeIntent, Aim
from r1gym.model import build_spec, load_cfg, CTRL_JOINTS
from r1gym.opponent import FightReplay, _sub

@pytest.fixture(scope="module")
def bouts():
    """One recorded eval bout with the Aim hook, 4 s (seed 60001: kfab_201, 0.40 m)."""
    b = C.work(([60001], 4.0)); assert len(b) == 1 and b[0][0]["clip"] == "kfab_201"; return b

def _assemble(b, k=1):
    A = C.assemble(b * k, [60001] * k, 4.0); A["meta"] = json.loads(json.dumps(A["meta"], default=str)); return A

@pytest.fixture(scope="module")
def rec(bouts): return _assemble(bouts)

@pytest.fixture(scope="module")
def rec2(bouts):
    """The same bout twice (two envs): the mixed jitter mask needs a T0 and a T1 env."""
    return _assemble(bouts, 2)

def test_aim_contract():
    assert {k: tuple(v) for k, v in ARM.items()} == dict(zip(FO.LIMBS, FO.ARM_IX)) and tuple(PALM[k] for k in FO.LIMBS) == FO.PALMS
    assert [CTRL_JOINTS[i] for i in FO.ARM_IX[0]] == [f"left_{j}_joint" for j in FO.ARM_JOINTS]
    p = merged(load_cfg().get("aim")); q = FO.AIM_P
    assert p["zone"] == "head" and tuple(p["targets"]["head"]) == q["target"] and p["arm"] == q["arm"] and p["standoff"] == q["standoff"]
    assert all(p[k] == q[k] for k in ("lead_s", "commit_s", "window_tail_s", "upright_gate_deg", "gate_after_touch_s")) and AIM["arm"]["damp"] == q["arm"]["damp"]
    assert OPERATORS["standoff+armwarp_vz"] == ("standoff", "arm", "vz")
    assert json.load(open(ROOT / "results" / "opponent_tracker" / "aim.json"))["operator"] == "standoff+armwarp_vz"

@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-6), (torch.float32, 1e-4)])
def test_fk_jacobian_vs_mj_jacsite(dtype, tol):
    m = _sub("r1"); Q = np.random.default_rng(3).uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1], size=(300, 24)); P, J = C.mj_palm(Q)
    p, j = FO.ArmKinematics.from_model(dtype=dtype).fk(torch.from_numpy(Q[:, list(FO.ARM_IX)]), jac=True)
    assert p.dtype == dtype and np.abs(p.double().numpy() - P).max() <= tol and np.abs(j.double().numpy() - J).max() <= tol

def test_chain_constants_any_prefix():
    """The same chain read from the eval's two-R1 model ('opp_') and from the opponent sub-model."""
    a = FO.ArmKinematics.chain(_sub("r1")); b = FO.ArmKinematics.chain(build_spec(opponent=True, cfg=load_cfg()).compile(), "opp_")
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)

def test_strike_library_per_clip():
    """AimOp's library = Aim.lib (the clip's hand strikes in library order), minus a strike whose onset repeats an earlier one:
    the eval's key set makes the first only (kfab_203 frame 0: l_hand made, r_hand never)."""
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile(); op = FO.AimOp(1, torch.float64)
    for c, clip in enumerate(FO.TRAIN):
        lib = Aim(m, FightReplay(m, clip, distance=0.45, cfg=cfg), "standoff+armwarp_vz", cfg.get("aim")).lib; keep, seen = [], set()
        for s in lib:
            if s.onset not in seen: keep.append(s); seen.add(s.onset)
        n = int(op.lib_ok[c].sum()); assert n == len(keep) and (clip != "kfab_203" or len(keep) == len(lib) - 1)
        assert op.lib_on[c, :n].tolist() == [s.onset for s in keep] and op.lib_ap[c, :n].tolist() == [s.apex for s in keep] and op.lib_en[c, :n].tolist() == [s.end for s in keep]
        assert [FO.LIMBS[i] for i in op.lib_limb[c, :n].tolist()] == [s.limb for s in keep] and op.lib_reach[c, :n].tolist() == [s.reach_head_m for s in keep]

def test_weight_equals_aim_weight():
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile(); aim = Aim(m, FightReplay(m, "kfab_200", distance=0.45, cfg=cfg), "standoff+armwarp_vz", cfg.get("aim"))
    ints = [(1.0, 1.26, 1.30), (2.0, 2.0, 2.0), (3.0, 3.08, 3.5), (4.02, 4.3, 4.31)]; op = FO.AimOp(1, torch.float64, slots=4)
    op.valid[0, :4] = True; op.t_on[0, :4], op.t_apex[0, :4], op.t_end[0, :4] = (torch.tensor([x[i] for x in ints], dtype=torch.float64) for i in range(3))
    for now in np.arange(0.5, 5.0, 0.005):
        w = op.weight(torch.tensor([now], dtype=torch.float64))[0, :4].tolist()
        assert w == [aim.weight(StrikeIntent("l_hand", "head", "kfab_200", 0, *x, np.zeros(3), 0.5), now) for x in ints], now

@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_aimop_equals_aim_reference_on_a_recorded_bout(rec, dtype):
    """float64: every row vs the eval. float32 (A1): vs the float64 port on the same float32-rounded states (output rounding only)."""
    f32 = dtype == torch.float32; O = C.replay(rec, dtype); r = C.compare(rec, O, str(dtype), port=C.replay(rec, torch.float64, round_to=torch.float32) if f32 else None)
    assert r["ok"] and r["intents"] >= 4 and r["warped_steps"] > 0 and r["intents_identical"] and r["base_reference_bit_equal"], r
    if not f32: assert r["q_maxdiff"] <= 1e-9 and r["dist_maxdiff"] == 0.0 and r["intent_time_maxdiff"] == 0.0 and r["closest_maxdiff_m"] <= 1e-12 and "A1" not in r, r
    else: assert r["A1"]["same_input_ok"] and r["A1"]["same_input_q_maxdiff"] <= 1e-6 and r["A1"]["eval_rows_ok"], r

def test_a1_float32_q_rule():
    """A1 (i) every row vs the port <= 1e-4 rad; (ii) vs the eval <= 1e-4 rad on >= 99.9% of rows (1 of 1000 over passes, 2 fail)."""
    z = np.zeros(1000); one, two = z.copy(), z.copy(); one[0] = 0.1; two[:2] = 0.1; g = C.q_gate_a1
    assert g(one, z)["eval_rows_ok"] and g(one, z)["same_input_ok"] and not g(two, z)["eval_rows_ok"]
    s = z.copy(); s[5] = 1.0001e-4; assert not g(z, s)["same_input_ok"] and g(z, s * 0.99)["same_input_ok"] and g(z, z)["eval_rows_within_tol_frac"] == 1.0
    assert C.TOL["q"] == 1e-4 and C.TOL["q_rows_frac"] == 0.999 and C.TOL["jitter_ci_m"] == 5e-3 and C.TOL["jitter_ci"] == 0.90
    assert C.TOL["guard_x_f64"] == 1e-6 and C.TOL["guard_x_f32"] == 1e-5 and C.TOL["guard_bind_report_frac"] == 0.5 and C.JITTER_SEEDS == tuple(range(20)) and FO.GUARD_STEPS == 8
    assert C.TOL["yz_keep_frac"] == 0.5 and C.TOL["smooth_q999_ratio"] == 1.25 and C.TOL["smooth_max_add_rad"] == 0.2 and FO.SECANT_IK <= 4 and FO.SECANT_TOL == 1e-6   # A3.2
    assert FO.JITTER == dict(target=((-0.03, 0.03), (-0.04, 0.04), (-0.05, 0.03)), commit_s=(0.10, 0.20), depth_m=(0.0, 0.06))   # A2, A3: ranges unchanged

def test_t_quantile_and_mean_ci():
    """The 90% t interval of A1 jitter (ii) (no scipy here): t_0.95 at df 1, 4, 10, 30, 217 and the normal limit."""
    for df, t in ((1, 6.3138), (4, 2.1318), (10, 1.8125), (30, 1.6973), (217, 1.6519), (10 ** 6, 1.6449)): assert abs(C.t_quantile(0.95, df) - t) < 1e-4, df
    x = np.random.default_rng(0).normal(0.001, 0.01, 50); lo, hi = C.mean_ci(x); h = C.t_quantile(0.95, 49) * x.std(ddof=1) / np.sqrt(50)
    assert np.isclose(lo, x.mean() - h) and np.isclose(hi, x.mean() + h) and np.isnan(C.mean_ci(x[:1])).all()
    a, b = C.cluster_ci(x, np.arange(50) % 5); assert a < x.mean() < b

def test_ring_slots_and_overflow(rec):
    """8 slots (deployed) = 64 bit for bit with nothing overwritten; 1 slot overwrites live intents and says so."""
    a, b = C.replay(rec, torch.float64), C.replay(rec, torch.float64, slots=8)
    assert all(np.array_equal(a[k], b[k]) for k in ("rq", "rdq", "dist")) and sum(b["overflow"]) == 0
    assert sum(C.replay(rec, torch.float64, slots=1, log_ik=False)["overflow"]) > 0

def test_recorder_refuses_other_blocks():
    for s in ([60016], [21000], [22000], [23000]):
        with pytest.raises(SeedRefused): C.work((s, 0.1))

def _drive(op, rec, rows):
    M = FO.Motion(); clip = torch.tensor([FO.TRAIN.index(rec["meta"]["episodes"][0]["clip"])]); out = []
    for i in rows:
        T = lambda k: torch.from_numpy(np.ascontiguousarray(rec[k][i:i + 1])); fi = T("fi"); rq0, rdq0 = M.ref_q[M.row(clip, fi)], M.ref_dq[M.row(clip, fi)]
        out.append(op.step(fi, T("now"), rq0, rdq0, T("opp_x"), T("opp_R"), T("def_x"), T("def_R"), T("touch"), q_act=T("qopp")[:, 7:]) + (rq0, rdq0))
    return out

def test_reset_and_shadow(rec):
    """reset() gives a fresh Aim (outputs equal a new op); a shadow row makes and logs intents but shows the clip unchanged."""
    ep = rec["meta"]["episodes"][0]; c = FO.TRAIN.index(ep["clip"]); n = len(rec["bout"]); rows = range(n)
    a = FO.AimOp(1, torch.float64); a.reset(None, c, ep["distance"]); _drive(a, rec, rows[: n // 2]); a.reset(None, c, ep["distance"]); x = _drive(a, rec, rows)
    b = FO.AimOp(1, torch.float64); b.reset(None, c, ep["distance"]); y = _drive(b, rec, rows)
    assert all(torch.equal(u, v) for p, q in zip(x, y) for u, v in zip(p, q))
    s = FO.AimOp(1, torch.float64); s.reset(None, c, ep["distance"], aimed=False); z = _drive(s, rec, rows)
    assert all(torch.equal(rq, rq0.double()) and torch.equal(rdq, rdq0.double()) and float(d) == ep["distance"] for rq, rdq, d, rq0, rdq0 in z)
    assert int(s.nseq[0]) == int(b.nseq[0]) > 0 and [t["t_on"] for t in s.intents(0)] == [t["t_on"] for t in b.intents(0)]

def test_jitter_bounds_and_ik_x_target(rec):
    """Jittered intents draw inside JITTER; the IK's x target is the reference palm's own (vz) with jitter on; outputs are
    float32 when the states are."""
    O = C.replay(rec, torch.float32, jitter=True, debug=True); J = FO.JITTER; t = [x for ev in O["intents"] for x in ev]; d = np.array([x["target_r1"] for x in t]) - FO.AIM_P["target"]
    assert len(t) >= 4 and all(lo - 1e-7 <= v <= hi + 1e-7 for x in d for v, (lo, hi) in zip(x, J["target"])) and np.abs(d).max() > 0
    assert all(J["commit_s"][0] <= x["commit_s"] <= J["commit_s"][1] and J["depth_m"][0] - 1e-7 <= x["depth_m"] <= J["depth_m"][1] + 1e-7 for x in t)
    kin = FO.ArmKinematics.from_model(); h = O["has"]; assert h.any() and np.array_equal(O["p_t"][..., 0][h], kin.fk(torch.from_numpy(O["rq5"])).numpy()[..., 0][h])
    op = FO.AimOp(2, torch.float32); assert op.dtype == torch.float64 and op.io == torch.float32 and FO.precision_ok()


def test_reach_guard_bisection(rec):
    """A2.1 alone (reach 'A2') on a jittered env, step by step: where the jittered palm lies further forward than the nominal warp's, the arm delta is
    nominal + s (jittered - nominal) with s on the 8-step grid, feasible (x <= nominal x) and s + 1/256 not; elsewhere the jittered
    delta. The nominal warp alongside is the T0 run's, bit for bit; with the guard off the jitter does add reach (negative control)."""
    ep = rec["meta"]["episodes"][0]; c = FO.TRAIN.index(ep["clip"]); M = FO.Motion(); clip = torch.tensor([c]); kin = FO.ArmKinematics.from_model(); ix = torch.tensor(FO.ARM_IX)
    op = FO.AimOp(1, torch.float64, motion=M, debug=True, reach="A2"); op.reset(None, c, ep["distance"], jitter=True); N = C.replay(rec, torch.float64); ng = gs = 0
    x = lambda q: kin.fk(q[None])[0, :, 0]
    for i in range(len(rec["bout"])):
        T = lambda k: torch.from_numpy(np.ascontiguousarray(rec[k][i:i + 1])); fi = T("fi"); rq0, rdq0 = M.ref_q[M.row(clip, fi)], M.ref_dq[M.row(clip, fi)]
        rq = op.step(fi, T("now"), rq0, rdq0, T("opp_x"), T("opp_R"), T("def_x"), T("def_R"), T("touch"))[0]; d = op.dbg; h, g, s = d["has"][0], d["guard"][0], d["s"][0]
        rq5, dn, dj = d["rq5"][0], d["d_nom"][0], d["d_jit"][0]
        assert torch.equal(rq[0, ix], rq5 + torch.where(g[:, None], dn + s[:, None] * (dj - dn), dj)) and torch.equal(g, h & (d["x_jit"][0] > d["x_nom"][0]))
        for l in torch.nonzero(h).flatten().tolist():
            assert torch.equal((rq5 + dn)[l], torch.from_numpy(N["rq"][i][list(FO.ARM_IX[l])])) and x(rq[0, ix])[l] <= d["x_nom"][0, l]
            if g[l]:
                ng += 1; gs += float(s[l]); assert 0 <= s[l] < 1 and float(s[l] * 256) == int(s[l] * 256)
                assert x(rq5 + dn + (s[:, None] + 1 / 256) * (dj - dn))[l] > d["x_nom"][0, l]
            else: assert torch.equal(rq[0, ix][l], (rq5 + dj)[l])
    t = op.intents(0); assert ng > 0 and sum(v["guard_steps"] for v in t) == ng and abs(sum(v["guard_s_mean"] * v["guard_steps"] for v in t if v["guard_steps"]) - gs) < 1e-12
    O = C.replay(rec, torch.float64, jitter=True, reach="raw")
    F = lambda Q: kin.fk(torch.from_numpy(np.ascontiguousarray(Q[:, list(sum(FO.ARM_IX, ()))]).reshape(-1, 2, 5))).numpy()[..., 0]
    assert (F(O["rq"]) - F(N["rq"])).max() > 1e-6 and (F(C.replay(rec, torch.float64, jitter=True, reach="A2")["rq"]) - F(N["rq"])).max() <= 0.0

def test_fprime_reach_correction(rec):
    """A3.1 F' on a jittered env, step by step: the corrected delta is the weighted IK on the jittered target with only its x moved
    (yz kept bit for bit), its palm x and convergence flag as reported; the guard (A2.1) acts on what remains (output dn + s (dc - dn)
    where the corrected palm still lies ahead of the nominal, else dc) and no step lies ahead; 2 + SECANT_IK IK calls per step (T0:
    1); the intents count the converged steps; F' keeps more yz at the apex than the guard alone and lands nearer the nominal x."""
    ep = rec["meta"]["episodes"][0]; c = FO.TRAIN.index(ep["clip"]); M = FO.Motion(); clip = torch.tensor([c]); kin = FO.ArmKinematics.from_model(); ix = torch.tensor(FO.ARM_IX)
    op = FO.AimOp(1, torch.float64, motion=M, debug=True); op.reset(None, c, ep["distance"], jitter=True); t0 = FO.AimOp(1, torch.float64, motion=M); t0.reset(None, c, ep["distance"])
    ref = FO.AimOp(1, torch.float64, motion=M); nw = nc = ng = 0; x = lambda q: kin.fk(q[None])[0, :, 0]
    for i in range(len(rec["bout"])):
        T = lambda k: torch.from_numpy(np.ascontiguousarray(rec[k][i:i + 1])); fi = T("fi"); rq0, rdq0 = M.ref_q[M.row(clip, fi)], M.ref_dq[M.row(clip, fi)]; n0, m0 = op.ik_calls, t0.ik_calls
        a = (fi, T("now"), rq0, rdq0, T("opp_x"), T("opp_R"), T("def_x"), T("def_R"), T("touch")); rq = op.step(*a)[0]; t0.step(*a)
        assert op.ik_calls - n0 == 2 + FO.SECANT_IK and t0.ik_calls - m0 == 1
        d = op.dbg; h, g, s, cv = d["has"][0], d["guard"][0], d["s"][0], d["conv"][0]; rq5, dn, dc, w = d["rq5"][0], d["d_nom"][0], d["d_cor"][0], d["w"][0]
        assert torch.equal(rq[0, ix], rq5 + torch.where(g[:, None], dn + s[:, None] * (dc - dn), dc)) and torch.equal(g, h & (d["x_cor"][0] > d["x_nom"][0]))
        for l in torch.nonzero(h).flatten().tolist():
            nw += 1; p = d["p_t"][0].clone(); p[l, 0] += d["shift"][0, l]; q = ref.ik(rq5[None], p[None])[0]
            assert torch.equal(p[l, 1:], d["p_t"][0, l, 1:]) and torch.allclose(dc[l], w[l] * (q - rq5)[l], rtol=0, atol=1e-9) and abs(float(x(rq5 + dc)[l] - d["x_cor"][0, l])) <= 1e-12
            assert bool(cv[l]) == bool((d["x_cor"][0, l] - d["x_nom"][0, l]).abs() <= FO.SECANT_TOL) and x(rq[0, ix])[l] <= d["x_nom"][0, l]
            nc += int(cv[l]); ng += int(g[l])
    t = op.intents(0); assert nw > 0 and nc > 0 and sum(v["secant_conv_steps"] for v in t) == nc and sum(v["guard_steps"] for v in t) == ng
    N = C.replay(rec, torch.float64); F = lambda Q: kin.fk(torch.from_numpy(np.ascontiguousarray(Q[:, list(sum(FO.ARM_IX, ()))]).reshape(-1, 2, 5))).numpy()
    rows = C.apex_rows(rec, rec["meta"]["intents"]); r, l = np.array([v[0] for v in rows]), np.array([v[1] for v in rows]); P = lambda Q: F(Q[r])[np.arange(len(r)), l]; pn = P(N["rq"])
    A3, A2 = (P(C.replay(rec, torch.float64, jitter=True, reach=k)["rq"]) for k in ("A3", "A2")); yz = lambda p: np.linalg.norm(p[:, 1:] - pn[:, 1:], axis=1).mean()
    assert yz(A3) > yz(A2) and np.abs(A3[:, 0] - pn[:, 0]).mean() < np.abs(A2[:, 0] - pn[:, 0]).mean(), (yz(A3), yz(A2))

def test_fallback_d_draws(rec):
    """A3.3 D: no arm-target offset (target_r1 = the nominal target), commit time and depth jittered with the same draws as A3, and
    the guard keeps every step at or below the nominal palm x."""
    D, J = (C.replay(rec, torch.float64, jitter=True, reach=k) for k in ("D", "A3")); t, u = [x for ev in D["intents"] for x in ev], [x for ev in J["intents"] for x in ev]
    assert len(t) == len(u) >= 4 and all(np.array_equal(x["target_r1"], FO.AIM_P["target"]) for x in t) and any(not np.array_equal(x["target_r1"], FO.AIM_P["target"]) for x in u)
    assert all(x["commit_s"] == y["commit_s"] and x["depth_m"] == y["depth_m"] for x, y in zip(t, u)) and len({x["commit_s"] for x in t}) > 1
    kin = FO.ArmKinematics.from_model(); F = lambda Q: kin.fk(torch.from_numpy(np.ascontiguousarray(Q[:, list(sum(FO.ARM_IX, ()))]).reshape(-1, 2, 5))).numpy()[..., 0]
    assert (F(D["rq"]) - F(C.replay(rec, torch.float64)["rq"])).max() <= 0.0

@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_jitter_a3_on_a_recorded_bout(rec2, dtype):
    """A3.2 jitter gates (F' then the guard) on two envs of the same recorded bout over generator seeds 0-1: A1 (i) the IK's x target
    at every warped step is the reference palm's own and the nominal one, bit for bit; (iii') no forward excess over the nominal warp
    at any warped step (gated here too: it holds by construction); A1 (iv) T0 rows bit-equal to the no-jitter run while T1 rows
    differ, and the probe fails when the T0 rows are compared to a jittered run; (ii'), (vi) and (vii) follow their rules (one bout
    in two envs here: gated on the 16 bouts only); (v) reported, with F' convergence and the IK calls per step."""
    O = C.replay(rec2, dtype, debug=True); r = C.jitter_test(rec2, O, dtype, seeds=(0, 1)); a = r["A3"]
    i, ii, iii, iv, v = (a[k] for k in ("A1_i_ik_x_target", "ii_cluster_ci", "iii_guard_every_step", "A1_iv_mixed_mask", "v_guard_binding"))
    assert i["ok"] and i["ok_seeds"] == 2 and i["seed0"]["warped_steps_jitter"] > 0 and i["seed0"]["warped_steps_same"] and i["seed0"]["jitter_equals_nominal"], i
    assert iii["ok"] and iii["warped_steps"] == 2 * i["seed0"]["warped_steps_jitter"] and iii["excess_max_m"] <= (1e-6 if dtype == torch.float64 else 1e-5), iii
    if dtype == torch.float64: assert iii["steps_over_0"] == 0 and iii["tol_m"] == 1e-6
    assert iv["ok"] and iv["bouts_t0"] == iv["bouts_t1"] == 1 and all(iv["t0_rows_bit_equal"].values()) and iv["t1_rows_differ"], iv
    assert not C.mixed_mask(rec2, C.replay(rec2, dtype, jitter=True), dtype)["ok"]
    lo, hi = ii["cluster_bootstrap_ci90_m"]; assert ii["apexes"] == 2 * ii["apexes_per_seed"] >= 4 and ii["bouts"] == 2 and lo <= ii["mean_m"] <= hi
    assert ii["ok"] == (-5e-3 < lo and hi < 5e-3) and r["ok"] == (i["ok"] and ii["ok"] and iii["ok"] and iv["ok"] and a["vi_yz_kept"]["ok"] and a["vii_smoothness"]["ok"])
    assert 0 < v["guard_intents"] <= v["jittered_intents"] and v["binds_over_half"] == (v["guard_intents_frac"] > 0.5) and 0 <= v["mean_s_guarded_steps"] < 1, v
    assert sum(p["guard_intents"] for p in r["per_seed"]) == v["guard_intents"] and len(r["per_seed"]) == 2
    assert 0 < v["secant"]["converged_frac"] <= 1 and v["secant"]["extra_ik_calls"] == FO.SECANT_IK and v["ik_calls_per_step"] == dict(jitter=2 + FO.SECANT_IK, raw=1), v
    vi, vii = a["vi_yz_kept"], a["vii_smoothness"]; assert vi["ok"] == (vi["yz_apex_spread_mean_m"] >= 0.5 * vi["raw_yz_apex_spread_mean_m"] > 0), vi
    assert vii["ok"] == (vii["t1_q999_rad"] <= 1.25 * vii["t0_q999_rad"] and vii["t1_max_rad"] <= vii["t0_max_rad"] + 0.2) and vii["t1_limb_steps"] == 2 * vii["t0_limb_steps"] > 0, vii
    assert r["gated"] == ["A1_i_ik_x_target", "ii_cluster_ci", "iii_guard_every_step", "A1_iv_mixed_mask", "vi_yz_kept", "vii_smoothness"]
    assert r["fprime_fail_triggers_D"] == (not (ii["ok"] and vi["ok"] and vii["ok"]))
