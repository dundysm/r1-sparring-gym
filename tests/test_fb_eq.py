"""Full-body M2 v2 M6 gate statistics (scripts/fb_eq.py; M2V2_PLAN_v3 3.1, 0.R, 3.2): the delta-method TOST arithmetic on a hand-checked
example; R-EQ decisions equal critique_checks/gate_power_v3.py's own req_gates on its resampled R4-pre cells (when the scratch files exist);
the look and T1-c episode sets of every rule (balanced where the rule says so, refused while unfrozen); the group-sequential stop; 3.2's
fail classes; pairing is asserted; a correct synthetic replica passes and the wrong replicas W1 and W5 fail. reproduce() and dry_run()
run in full from the CLI (M6a notes)."""
import sys, math
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"): sys.path.insert(0, str(p))
import fb_eq as Q

HAVE_GP = Q.GP_PATH.exists() and (Q.GP_PATH.parent.parent / "r4pre" / "runs" / "G_main.pkl").exists()

def test_tost_arithmetic():
    u = np.array([0.0, 1.0, 2.0, 3.0]); g = Q.tost(u, 1.5, 0.0, 3.0); se = u.std(ddof=1) / 2
    assert math.isclose(g["se"], se) and math.isclose(g["ci90"][0], 1.5 - Q.Z90 * se) and g["pass_"] == (1.5 - Q.Z90 * se >= 0 and 1.5 + Q.Z90 * se <= 3)
    assert Q.Z90 == pytest.approx(1.6448536269514722, abs=0) and Q.M["surv"] == 0.015 and Q.M["R"] == 0.0625 and Q.M["gap"] == 0.03 and Q.M["pfk"] == 0.20
    assert Q.M["hc"] == (math.log(0.9), math.log(1.1)) and Q.M["kd"] == (math.log(0.8), math.log(1.25)) and Q.LOOKS == (4096, 8192, 12288)
    assert set(Q.GATED) == {"surv_G", "surv_H", "hc_G", "hc_H", "kd_G", "R_HvsG", "gap_A3_G", "gap_sign", "pfk_contrast", "R5_latency"} and Q.REPORT == ("kd_A3",)

def test_classify():
    g = dict(surv_G=dict(est=0.03, se=0.005, ci90=[0.022, 0.038], margin=[-0.015, 0.015], pass_=False), surv_H=dict(est=0.01, se=0.005, ci90=[0.002, 0.018], margin=[-0.015, 0.015], pass_=False),
             gap_sign=dict(est=0.001, se=0.01, lower90=-0.015, pass_=False))
    assert Q.classify(g) == dict(surv_G="conclusive", surv_H="noise-sized", gap_sign="noise-sized")

def _keys():
    import fb_r0eo as R0
    return [(e["seed"], e["clip"], e["distance"]) for e in R0.episodes()]

def test_look_and_t1c_sets(monkeypatch):
    K = _keys(); strata = lambda ix: {(K[i][1], K[i][2]) for i in ix}
    assert (Q.LOOK_RULE, Q.T1C_RULE) == ("shards_ceil", "shards")   # plan A5.3: the first 22 / 43 / 64 shards; T1-c = shards 0-31
    assert tuple(len(s) for s in Q.look_sets(K)) == (4224, 8256, 12288) and np.array_equal(Q.t1c_set(K), Q.t1c_set(K, "shards"))
    monkeypatch.setattr(Q, "LOOK_RULE", None); monkeypatch.setattr(Q, "T1C_RULE", None)
    with pytest.raises(ValueError): Q.look_sets(K)
    with pytest.raises(ValueError): Q.t1c_set(K)
    want = dict(shards_floor=(4032, 8064, 12288), shards_ceil=(4224, 8256, 12288), reps=(4092, 8184, 12288), seed_prefix=(4096, 8192, 12288))
    for rule, ns in want.items():
        S = Q.look_sets(K, rule=rule); assert tuple(len(s) for s in S) == ns and all(set(a) <= set(b) for a, b in zip(S, S[1:]))
        if rule != "seed_prefix":
            for s in S: c = np.bincount([(i // 1024) for i in s], minlength=12); assert len(strata(s)) == 12 and len(set(c)) == 1
    assert strata(Q.look_sets(K, rule="seed_prefix")[0]) == {("kfab_200", d) for d in (0.4, 0.45, 0.5, 0.55)}   # clip-major: the unbalanced reading
    a, b = Q.t1c_set(K, "shards"), Q.t1c_set(K, "seed_range"); assert len(a) == len(b) == 6144 and len(strata(a)) == 12 and len(strata(b)) == 6

def test_paired_refuses_other_episodes():
    r = lambda s, ad=0: dict(seed=s, clip="kfab_200", distance=0.4, t0=1.0, k=0, act_delay=ad, fell_s=None, kd=False, n=dict(head_clean=1), secs=20.0)
    eo = {c: [r(1), r(2)] for c in ("G", "H", "A3")}; rep = {c: [r(1), r(2)] for c in ("G", "H", "A3")}; Q.paired(eo, rep)
    rep["H"] = [r(1), r(2, 1)]
    with pytest.raises(AssertionError): Q.paired(eo, rep)

@pytest.mark.skipif(not HAVE_GP, reason="critique_checks/gate_power_v3.py and the R4-pre rows (scratch) not present")
def test_decisions_equal_gate_power_v3():
    GP = Q._gp(); pool = GP.load("main"); GP.rng = np.random.default_rng(7); n = 0
    for N, q in ((768, 0.1), (2304, 0.0), (4096, 0.3)):
        for _ in range(8):
            eo, rep, I = GP.cells(pool, N, q); act = pool["act"][I]
            if n % 3 == 0: GP.WRONG["W3_kd_x1.6"](rep)
            assert GP.req_gates(eo, rep, act) == Q.passes(Q.req_gates(eo, rep, act)); n += 1
            est, u = Q.log_ratio_u(rep["G"][:, 1], eo["G"][:, 1]); se = u.std(ddof=1) / math.sqrt(N)
            assert Q.t1c_gate(eo["G"], rep["G"])["pass_"] == ((est - Q.Z90 * se > math.log(0.8)) and (math.log(0.8) <= est <= math.log(1.25)))

@pytest.mark.skipif(not HAVE_GP, reason="critique_checks/gate_power_v3.py and the R4-pre rows (scratch) not present")
def test_sequential_on_synthetic_rows():
    import fb_r0eo as R0, pickle
    GP = Q._gp(); pool = GP.load("main"); K = _keys(); P = pickle.load(open(Q.GP_PATH.parent.parent / "r4pre" / "runs" / "G_main.pkl", "rb"))["rows"]
    pool["dist"] = np.array([r["distance"] for r in P]); pool["by_cd"] = {(c, d): np.flatnonzero((pool["clip"] == c) & (pool["dist"] == d)) for c in R0.TRAIN for d in R0.HARD}
    ok = Q.sequential(*Q.synth(GP, pool, K, 1.0, None, np.random.default_rng(3)), rule="shards_ceil"); assert ok["passed"] and ok["passed_at"] == 4096
    for w in ("W1_surv-0.03", "W5_no_A3_gap"):
        bad = Q.sequential(*Q.synth(GP, pool, K, 0.1, w, np.random.default_rng(3)), rule="shards_ceil"); assert not bad["passed"] and len(bad["looks"]) == 3 and bad["fail_classes"]
