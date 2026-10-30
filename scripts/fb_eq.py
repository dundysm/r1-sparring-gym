"""M6 gate statistics (M2V2_PLAN_v3 3.1, 0.R, 3.2): the paired delta-method TOST of critique_checks/gate_power_v3.py on real rows.
Every gate is a 90% CI from the delta method (est +- Z90 se, se = sd(u) / sqrt(n) of the per-episode linearisation u); TOST passes iff the
CI lies inside the margin. Rows: fb_r0eo's (EO) and fb_replica's (replica) format, or R4-pre's run rows (arrays(): fell = fell_s set,
kd = kd / knockdown, hc = n.head_clean, secs = minutes up x 60; act_delay per episode).
  R-EQ (replica vs EO, same episodes; gated GATED, report kd_A3):
    surv_G / surv_H   survival, replica - EO, +-0.015
    hc_G / hc_H       log head clean per up-second, replica - EO, in log [0.90, 1.10]
    kd_G              log knockdowns, replica / EO, in log [0.80, 1.25]   (kd_A3: report)
    R_HvsG            (log R(H vs G)) replica - EO, +-0.0625
    gap_A3_G          (A3 - G fall rate) replica - EO, +-0.03             0.R frozen R4 contrasts
    gap_sign          the replica's A3 - G fall rate: one-sided 90% lower bound > 0
    pfk_contrast      (P(fall | kd) A3 - G) replica - EO, +-0.20
    R5_latency        the replica's A3 fall rate, act_delay 1 - 0 (two independent groups): one-sided 90% lower bound > 0
  sequential(): group-sequential looks LOOKS (4096 / 8192 / 12288 per controller): pass at the first look where every GATED test passes,
    else the next look; fail after the last. Each look is a balanced episode subset defined by LOOK_RULE (look_sets): plan A5.3 froze
    'shards_ceil' (the first 22 / 43 / 64 whole shards, 4224 / 8256 / 12288 bouts); None refuses. classify(): 3.2's fail classes per gate
    (noise-sized: the CI overlaps the margin; conclusive: outside).
  T1-c (t1c_gate): kd(T1) / kd(T0) on the same episodes: one-sided 90% lower bound > 0.8 and the point in [0.8, 1.25]; T1C_RULE picks the
    6144 episodes: A5.3 froze 'shards' (shards 0-31, balanced; replaces "70000-76143"); None refuses.
  P1-R4g (r4g_gates): GPU worlds vs EO, unpaired (variances added), surv / hc / kd / gap_A3_G at the R-EQ margins.
  P3 LOCO (loco_bars): pilot vs start on the same episodes: held-out R <= 0.90 and upper < 1.0; seen R <= 0.85 and upper < 0.95;
    survival diff >= -0.015 and lower >= -0.04; g = log R_ho / log R_seen >= 0.4.
reproduce(): gate_power_v3.py's own simulation (its RNG, its resampling of the R4-pre rows, every table of gate_power_v3_all.json) with
its gate decisions replaced by this module's: the table must come out equal. dry_run(): synthetic replica rows (gate_power_v3's q-model
on the R4-pre rows, laid on the R-EQ episode keys) through rows -> looks -> gates, correct and the wrong replicas W1-W6 (W2b, W3b).
CLI: python -I scripts/fb_eq.py reproduce --out power.json | dry --out dry.json [--trials 20] | gate --eo DIR --rep DIR --out req.json"""
import sys, json, math, time, argparse, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np

Z90 = 1.6448536269514722
M = dict(surv=0.015, hc=(math.log(0.9), math.log(1.1)), kd=(math.log(0.8), math.log(1.25)), R=0.0625, gap=0.03, pfk=0.20)
GATED = ("surv_G", "surv_H", "hc_G", "hc_H", "kd_G", "R_HvsG", "gap_A3_G", "gap_sign", "pfk_contrast", "R5_latency"); REPORT = ("kd_A3",)
LOOKS = (4096, 8192, 12288)
LOOK_RULES = ("shards_floor", "shards_ceil", "reps", "seed_prefix")
LOOK_RULE = "shards_ceil"   # frozen by plan A5.3 (before any replica or gate data): looks at the first 22 / 43 / 64 shards; None refuses
T1C_RULES = ("shards", "seed_range"); T1C_RULE = "shards"; T1C_N = 6144; T1C_LO, T1C_POINT = math.log(0.8), (math.log(0.8), math.log(1.25))   # A5.3: T1-c = shards 0-31
SHARD, NSTRATA, BLOCK0 = 192, 12, 70000
SCR = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody")
GP_PATH = SCR / "critique_checks" / "gate_power_v3.py"

# ---- rows -> arrays
def arrays(rows):
    """(x (N, 4) float [fell, kd, head clean, up seconds], act_delay (N,), keys [(seed, clip, distance)])."""
    x = np.array([[r["fell_s"] is not None, bool(r["kd"] if "kd" in r else r["knockdown"]), r["n"]["head_clean"], r["secs"]] for r in rows], float)
    return x, np.array([int(r["act_delay"]) for r in rows]), [(r["seed"], r["clip"], r["distance"]) for r in rows]

def paired(eo, rep, ctrls=("G", "H", "A3")):
    """{ctrl: rows} x2 -> (eo arrays, rep arrays, act): same episodes in the same order (seed, clip, distance, t0, k, act_delay)."""
    key = lambda r: (r["seed"], r["clip"], r["distance"], r["t0"], r.get("k", r.get("delay")), int(r["act_delay"]))
    k0 = [key(r) for r in eo[ctrls[0]]]
    for c in ctrls: assert [key(r) for r in eo[c]] == k0 and [key(r) for r in rep[c]] == k0, f"{c}: EO and replica rows are not on the same episodes"
    E = {c: arrays(eo[c])[0] for c in ctrls}; R = {c: arrays(rep[c])[0] for c in ctrls}; return E, R, arrays(eo[ctrls[0]])[1]

# ---- delta-method pieces (gate_power_v3's arithmetic, operation for operation)
def tost(u, est, lo, hi):
    se = u.std(ddof=1) / math.sqrt(len(u)); a, b = est - Z90 * se, est + Z90 * se
    return dict(est=float(est), se=float(se), ci90=[float(a), float(b)], margin=[float(lo), float(hi)], pass_=bool((a >= lo) and (b <= hi)))
def log_ratio_u(a, b):
    A, B = a.sum(), b.sum(); return math.log(max(A, 1e-9) / max(B, 1e-9)), a / max(A / len(a), 1e-9) - b / max(B / len(b), 1e-9)
def rate_u(x):
    H, S = x[:, 2].mean(), x[:, 3].mean(); return math.log(max(H, 1e-9) / S), x[:, 2] / max(H, 1e-9) - x[:, 3] / S
def pfk_u(x):
    K = x[:, 1].mean(); fk = (x[:, 0] * x[:, 1]); P = fk.mean() / max(K, 1e-9); return P, (fk - P * x[:, 1]) / max(K, 1e-9)
def lower(u, est=None):
    se = u.std(ddof=1) / math.sqrt(len(u)); est = u.mean() if est is None else est; return dict(est=float(est), se=float(se), lower90=float(est - Z90 * se), pass_=bool(est - Z90 * se > 0))

def req_gates(eo, rep, act=None):
    """R-EQ on paired arrays {ctrl: (N, 4)} (replica vs EO, same episodes): {gate: dict(est, se, ci90 | lower90, margin, pass_)}."""
    g = {}
    for c in ("G", "H"):
        d = rep[c][:, 0] - eo[c][:, 0]; g[f"surv_{c}"] = tost(-d, -d.mean(), -M["surv"], M["surv"])
        lr, ur = rate_u(rep[c]); le, ue = rate_u(eo[c]); g[f"hc_{c}"] = tost(ur - ue, lr - le, *M["hc"])
    for c in ("G", "A3"):
        est, u = log_ratio_u(rep[c][:, 1], eo[c][:, 1]); g[f"kd_{c}"] = tost(u, est, *M["kd"])
    lHr, uHr = rate_u(rep["H"]); lGr, uGr = rate_u(rep["G"]); lHe, uHe = rate_u(eo["H"]); lGe, uGe = rate_u(eo["G"])
    g["R_HvsG"] = tost((uHr - uGr) - (uHe - uGe), (lHr - lGr) - (lHe - lGe), -M["R"], M["R"])
    u = (rep["A3"][:, 0] - rep["G"][:, 0]) - (eo["A3"][:, 0] - eo["G"][:, 0]); g["gap_A3_G"] = tost(u, u.mean(), -M["gap"], M["gap"])
    ur_ = rep["A3"][:, 0] - rep["G"][:, 0]; s = ur_.std(ddof=1) / math.sqrt(len(ur_)); g["gap_sign"] = dict(est=float(ur_.mean()), se=float(s), lower90=float(ur_.mean() - Z90 * s), pass_=bool(ur_.mean() - Z90 * s > 0))
    PAr, uAr = pfk_u(rep["A3"]); PGr, uGr2 = pfk_u(rep["G"]); PAe, uAe = pfk_u(eo["A3"]); PGe, uGe2 = pfk_u(eo["G"])
    g["pfk_contrast"] = tost((uAr - uGr2) - (uAe - uGe2), (PAr - PGr) - (PAe - PGe), -M["pfk"], M["pfk"])
    if act is not None:
        f = rep["A3"][:, 0]; a1, a0 = f[act == 1], f[act == 0]; d = a1.mean() - a0.mean(); se = math.sqrt(a1.var(ddof=1) / len(a1) + a0.var(ddof=1) / len(a0))
        g["R5_latency"] = dict(est=float(d), se=float(se), lower90=float(d - Z90 * se), pass_=bool(d - Z90 * se > 0))
    return g

def passes(g): return {k: v["pass_"] for k, v in g.items()}
def all_gated(g): return all(g[k]["pass_"] for k in GATED)

def classify(g):
    """3.2 (3) per failed gate: 'noise-sized' (the 90% CI overlaps the margin) or 'conclusive' (it lies entirely outside); one-sided: the
    lower bound vs 0 (conclusive iff the upper bound est + Z90 se is also <= 0)."""
    out = {}
    for k in GATED:
        v = g.get(k)
        if v is None or v["pass_"]: continue
        if "ci90" in v: (a, b), (lo, hi) = v["ci90"], v["margin"]; out[k] = "conclusive" if (b < lo or a > hi) else "noise-sized"
        else: out[k] = "conclusive" if v["est"] + Z90 * v["se"] <= 0 else "noise-sized"
    return out

def report(eo, rep, act):
    """Report-only (0.R): A3 / G knockdown ratio and the H - G survival difference, EO and replica; G's latency interaction."""
    out = {}
    for tag, X in (("eo", eo), ("rep", rep)):
        est, u = log_ratio_u(X["A3"][:, 1], X["G"][:, 1]); se = u.std(ddof=1) / math.sqrt(len(u)); d = (1 - X["H"][:, 0]) - (1 - X["G"][:, 0])
        f = X["G"][:, 0]; a1, a0 = f[act == 1], f[act == 0]
        out[tag] = dict(kd_A3_over_G=[math.exp(est), math.exp(est - Z90 * se), math.exp(est + Z90 * se)], surv_H_minus_G=[float(d.mean()), float(d.mean() - Z90 * d.std(ddof=1) / math.sqrt(len(d))), float(d.mean() + Z90 * d.std(ddof=1) / math.sqrt(len(d)))],
                        G_fall_act1_minus_act0=float(a1.mean() - a0.mean()), survival={c: float(1 - X[c][:, 0].mean()) for c in X}, kd_rate={c: float(X[c][:, 1].mean()) for c in X},
                        head_clean_pm={c: float(X[c][:, 2].sum() / (X[c][:, 3].sum() / 60)) for c in X})
    return out

# ---- looks
def look_sets(keys, looks=LOOKS, rule=None):
    """Index sets (into keys = [(seed, clip, distance)] of the R-EQ block, any order) of each look. rule (LOOK_RULE when None):
    'shards_floor' whole fb_r0eo shards (192 balanced bouts: reps 16j..16j+15 of the 12 strata), floor(n / 192) of them (4032 / 8064 / 12288);
    'shards_ceil' ceil (4224 / 8256 / 12288); 'reps' reps 0..floor(n / 12) - 1 of every stratum (4092 / 8184 / 12288); 'seed_prefix' the
    first n seeds (seed order is clip-major: look 1 = kfab_200 only)."""
    rule = rule or LOOK_RULE
    if rule is None: raise ValueError("LOOK_RULE is not frozen: the main session sets it before any R-EQ look (fb_eq.LOOK_RULES)")
    assert rule in LOOK_RULES, rule; seed = np.array([k[0] for k in keys]); rep = (seed - BLOCK0) % 1024; out = []
    for n in looks:
        if rule == "shards_floor": m = rep < SHARD // NSTRATA * (n // SHARD)
        elif rule == "shards_ceil": m = rep < SHARD // NSTRATA * -(-n // SHARD)
        elif rule == "reps": m = rep < n // NSTRATA
        else: m = seed < BLOCK0 + n
        out.append(np.flatnonzero(m))
    return out

def sequential(eo, rep, rule=None, looks=LOOKS, sets=None):
    """Group-sequential R-EQ on {ctrl: rows} (EO, replica): the first look whose GATED tests all pass, else fail after the last. sets: explicit
    index sets per look (dry runs); default look_sets(rule)."""
    E, R, act = paired(eo, rep); keys = arrays(eo["G"])[2]; sets = look_sets(keys, looks, rule) if sets is None else sets; res = []
    for n, ix in zip(looks, sets):
        g = req_gates({c: v[ix] for c, v in E.items()}, {c: v[ix] for c, v in R.items()}, act[ix]); res.append(dict(look=n, n=int(len(ix)), gates=g, all_gated=all_gated(g)))
        if all_gated(g): break
    last = res[-1]; return dict(rule=rule or LOOK_RULE, looks=res, passed=last["all_gated"], passed_at=last["look"] if last["all_gated"] else None,
                                fail_classes=None if last["all_gated"] else classify(last["gates"]), report=report({c: v[sets[len(res) - 1]] for c, v in E.items()}, {c: v[sets[len(res) - 1]] for c, v in R.items()}, act[sets[len(res) - 1]]))

# ---- T1-c, P1-R4g, P3 LOCO
def t1c_gate(t0, t1):
    """T1-c on (N, 4) arrays of G vs T0 and G vs T1 on the same episodes: lower 90% bound of log kd(T1)/kd(T0) > log 0.8 and the point in [0.8, 1.25]."""
    est, u = log_ratio_u(t1[:, 1], t0[:, 1]); se = u.std(ddof=1) / math.sqrt(len(t0))
    return dict(est=float(est), ratio=math.exp(est), se=float(se), lower90=float(est - Z90 * se), pass_=bool((est - Z90 * se > T1C_LO) and (T1C_POINT[0] <= est <= T1C_POINT[1])))

def t1c_set(keys, rule=None):
    """The 6144 T1-c episodes of the R-EQ block: 'shards' fb_r0eo shards 0-31 (reps 0-511 of every stratum); 'seed_range' 70000-76143 literally
    (clip-major: kfab_200 and kfab_201 at 0.40 / 0.45 m only)."""
    rule = rule or T1C_RULE
    if rule is None: raise ValueError("T1C_RULE is not frozen (fb_eq.T1C_RULES)")
    seed = np.array([k[0] for k in keys]); return np.flatnonzero((seed - BLOCK0) % 1024 < T1C_N // NSTRATA if rule == "shards" else (seed >= BLOCK0) & (seed < BLOCK0 + T1C_N))

def _tost2(est, va, vb, lo, hi): se = math.sqrt(va + vb); return dict(est=float(est), se=float(se), ci90=[float(est - Z90 * se), float(est + Z90 * se)], margin=[float(lo), float(hi)], pass_=bool((est - Z90 * se >= lo) and (est + Z90 * se <= hi)))

def r4g_gates(eo, gpu):
    """P1-R4g on {ctrl: (N, 4)} EO and GPU-world arrays (unpaired, any sizes)."""
    g = {}; Ne, Ng = len(eo["G"]), len(gpu["G"])
    for c in ("G", "H"):
        g[f"surv_{c}"] = _tost2(eo[c][:, 0].mean() - gpu[c][:, 0].mean(), eo[c][:, 0].var() / Ne, gpu[c][:, 0].var() / Ng, -M["surv"], M["surv"])
        lr, ur = rate_u(gpu[c]); le, ue = rate_u(eo[c]); g[f"hc_{c}"] = _tost2(lr - le, ur.var() / Ng, ue.var() / Ne, *M["hc"])
    for c in ("G", "A3"):
        A, B = gpu[c][:, 1], eo[c][:, 1]; est = math.log(max(A.mean(), 1e-9) / max(B.mean(), 1e-9)); g[f"kd_{c}"] = _tost2(est, (A / A.mean()).var() / Ng, (B / B.mean()).var() / Ne, *M["kd"])
    gr = gpu["A3"][:, 0].mean() - gpu["G"][:, 0].mean(); ge = eo["A3"][:, 0].mean() - eo["G"][:, 0].mean()
    g["gap_A3_G"] = _tost2(gr - ge, (gpu["A3"][:, 0] - gpu["G"][:, 0]).var() / Ng, (eo["A3"][:, 0] - eo["G"][:, 0]).var() / Ne, -M["gap"], M["gap"]); return g

def loco_bars(ho_pilot, ho_start, seen_pilot, seen_start):
    """P3 bars on (N, 4) arrays, pilot vs start on the same episodes per clip set."""
    res = []; ds = []
    for pl, st in ((ho_pilot, ho_start), (seen_pilot, seen_start)):
        lp, up = rate_u(pl); ls, us = rate_u(st); u = up - us; se = u.std(ddof=1) / math.sqrt(len(pl)); res.append((lp - ls, se)); ds.append(st[:, 0] - pl[:, 0])
    (lh, sh), (lsn, ssn) = res; d = np.concatenate(ds); sd = d.std(ddof=1) / math.sqrt(len(d))
    c = dict(heldout=bool(lh <= math.log(0.90) and lh + Z90 * sh < 0), seen=bool(lsn <= math.log(0.85) and lsn + Z90 * ssn < math.log(0.95)),
             survival=bool(-d.mean() >= -0.015 and -d.mean() - Z90 * sd >= -0.04), generalisation=bool(lsn < 0 and lh / lsn >= 0.4))
    return dict(R_heldout=math.exp(lh), R_seen=math.exp(lsn), surv_diff=float(-d.mean()), g=float(lh / lsn) if lsn else None, **c, proceed=all(c.values()))

# ---- reproduce gate_power_v3 with this module's decisions
def _gp(path=GP_PATH):
    spec = importlib.util.spec_from_file_location("gate_power_v3", str(path)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def reproduce(path=GP_PATH, which="all"):
    """gate_power_v3's __main__ for `which`, its RNG and draws untouched, every decision from this module -> its output dict."""
    GP = _gp(path); GP.req_gates = lambda eo, rep, act=None: passes(req_gates(eo, rep, act)); rng = GP.rng; cells, scale_kd, redraw, scale_hc, more_falls, rate_u_gp = GP.cells, GP.scale_kd, GP.redraw, GP.scale_hc, GP.more_falls, GP.rate_u
    def sim_t1c(pool, N, ratio, q=0.0):
        ok = 0
        for _ in range(GP.TR):
            eo, rep, _ = cells(pool, N, q, ("G",)); t0, t1 = eo["G"], rep["G"]
            if ratio != 1.0: scale_kd(t1, ratio)
            g = t1c_gate(t0, t1); ok += (np.float64(g["lower90"]) > T1C_LO) and (T1C_POINT[0] <= g["est"] <= T1C_POINT[1])   # = pass_; gate_power_v3's operand types, so its count turns numpy and round() reports the same digit
        return round(ok / GP.TR, 3)
    def sim_loco(pool, N_ho, N_seen, R_ho, R_seen, dsurv=0.0, q=0.2):
        ho = np.flatnonzero(pool["clip"] == "kfab_203"); se_ = np.flatnonzero(pool["clip"] != "kfab_203"); ok = 0; parts = np.zeros(4)
        for _ in range(GP.TR):
            arr = []
            for idxs, n, R in ((ho, N_ho, R_ho), (se_, N_seen, R_seen)):
                I = rng.choice(idxs, n); st = pool["H"][I]; J = np.where(rng.random(n) < q, I, redraw(pool, I)); pl = pool["H"][J].copy(); scale_hc(pl, R)
                if dsurv: more_falls(pl, -dsurv) if dsurv < 0 else None
                arr += [pl, st]
            b = loco_bars(*arr); c = [b["heldout"], b["seen"], b["survival"], b["generalisation"]]; parts += c; ok += all(c)
        return dict(ALL=round(ok / GP.TR, 3), ho=round(parts[0] / GP.TR, 3), seen=round(parts[1] / GP.TR, 3), surv=round(parts[2] / GP.TR, 3), gen=round(parts[3] / GP.TR, 3))
    def sim_r4g(pool, N_gpu, N_eo):
        acc = None; keys = None
        for _ in range(GP.TR):
            eo, _, _ = cells(pool, N_eo, 1.0); _, rep, _ = cells(pool, N_gpu, 0.0); g = passes(r4g_gates(eo, rep))
            if acc is None: keys = list(g); acc = np.zeros(len(keys) + 1)
            v = np.array([g[k] for k in keys], float); acc[:-1] += v; acc[-1] += all(g[k] for k in keys if k != "kd_A3")
        return {**{k: round(a / GP.TR, 3) for k, a in zip(keys, acc[:-1])}, "ALL_gated": round(acc[-1] / GP.TR, 3)}
    main = GP.load("main"); out = {"assumptions": dict(margins=dict(GP.M, hc=[0.9, 1.1], kd=[0.8, 1.25]), trials=GP.TR, z90=GP.Z90)}
    out["calibration_q"] = {str(q): GP.calib(main, q) for q in (0.0, 0.1, 0.2, 0.3)}
    Q = 0.1; out["q_used"] = Q; SEQ = (4096, 8192, 12288); out["looks"] = SEQ
    if which in ("all", "req"):
        out["R_EQ_correct_fixedN"] = {str(N): GP.sim_req(main, N, Q) for N in (2304, 4096, 6144, 8192, 12288)}
        out["R_EQ_correct_sequential"] = GP.sim_req(main, 0, Q, looks=SEQ)
        out["R_EQ_wrong_sequential"] = {w: GP.sim_req(main, 0, Q, w, looks=SEQ) for w in GP.WRONG}
        out["R_EQ_correct_sequential_q0"] = GP.sim_req(main, 0, 0.0, looks=SEQ)
    if which in ("all", "t1c"): out["T1c_pass"] = {str(N): {str(r): sim_t1c(main, N, r) for r in (1.0, 0.8, 0.6)} for N in (768, 1536, 3072, 4096, 4608, 6144)}
    if which in ("all", "loco"):
        out["LOCO"] = {str(n): dict(good=sim_loco(main, n, n, 0.80, 0.72), bad_no_transfer=sim_loco(main, n, n, 0.98, 0.72), bad_no_learning=sim_loco(main, n, n, 0.95, 0.95),
                                    start_equal=sim_loco(main, n, n, 1.0, 1.0)) for n in (576, 768, 1152)}
    if which in ("all", "r4g"): out["P1_R4g_correct"] = {f"gpu{ng}_eo{ne}": sim_r4g(main, ng, ne) for ng, ne in ((8192, 4096), (12288, 12288), (24576, 12288))}
    return json.loads(json.dumps(out))

def diff(a, b, path=""):
    """Leaves where two json trees differ."""
    if isinstance(a, dict) and isinstance(b, dict): return sum((diff(a[k], b[k], f"{path}/{k}") if k in b else [f"{path}/{k} missing"] for k in a), []) + [f"{path}/{k} extra" for k in b if k not in a]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b): return sum((diff(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [f"{path}: {a} != {b}"]

# ---- dry run: synthetic replica rows on the R-EQ keys
def synth(GP, pool, keys, q, wrong, rng):
    """EO and replica rows {ctrl: rows} on keys [(seed, clip, distance)] (the R-EQ block): EO = a pool row of the key's (clip, distance)
    stratum; replica = that row with prob q, else a fresh one of its (clip, distance, act_delay) stratum (gate_power_v3.cells), then the
    wrong replica's effect (gate_power_v3.WRONG on the arrays; GP.rng = rng)."""
    GP.rng = rng; cl = pool["clip"]; dist = pool["dist"]; N = len(keys)
    I = np.array([rng.choice(pool["by_cd"][(c, d)]) for _, c, d in keys]); eo = {c: pool[c][I] for c in ("G", "H", "A3")}; rep = {}
    for c in ("G", "H", "A3"): J = np.where(rng.random(N) < q, I, GP.redraw(pool, I)); rep[c] = pool[c][J].copy()
    if wrong == "W5_no_A3_gap": rep["A3"] = rep["G"].copy(); rep["A3"][:, 1] = pool["A3"][GP.redraw(pool, I), 1]
    elif wrong == "W6_no_pfk_contrast":
        pG = pool["G"][:, 0][pool["G"][:, 1] == 1].mean(); k = rep["A3"][:, 1] == 1; rep["A3"][k, 0] = rng.random(k.sum()) < pG
    elif wrong: GP.WRONG[wrong](rep)
    act = pool["act"][I]; row = lambda k, x, a: dict(seed=int(k[0]), clip=k[1], distance=float(k[2]), t0=0.0, k=0, act_delay=int(a), fell_s=float(x[3]) if x[0] else None,
                                                     kd=bool(x[1]), n=dict(head_clean=int(x[2])), secs=float(x[3]))
    return ({c: [row(k, x, a) for k, x, a in zip(keys, eo[c], act)] for c in eo}, {c: [row(k, x, a) for k, x, a in zip(keys, rep[c], act)] for c in rep})

def dry_run(trials=20, seed=20261030, path=GP_PATH, rules=LOOK_RULES):
    import fb_r0eo as R0
    GP = _gp(path); pool = GP.load("main"); keys = [(e["seed"], e["clip"], e["distance"]) for e in R0.episodes()]
    P = __import__("pickle").load(open(Path(path).parent.parent / "r4pre" / "runs" / "G_main.pkl", "rb"))["rows"]; pool["dist"] = np.array([r["distance"] for r in P])
    pool["by_cd"] = {(c, d): np.flatnonzero((pool["clip"] == c) & (pool["dist"] == d)) for c in R0.TRAIN for d in R0.HARD}
    out = {}
    for rule in rules:
        out[rule] = {}
        for w in (None, *GP.WRONG):
            rng = np.random.default_rng(seed); res = [sequential(*synth(GP, pool, keys, 0.1, w, rng), rule=rule) for _ in range(trials)]
            out[rule][w or "correct"] = dict(passed=sum(r["passed"] for r in res), of=trials, mean_n=float(np.mean([r["looks"][-1]["n"] for r in res])),
                                             look_ns=[x["n"] for x in res[0]["looks"]], gates_failed_last={k: sum(not r["looks"][-1]["gates"][k]["pass_"] for r in res) for k in GATED},
                                             example_fail_classes=next((r["fail_classes"] for r in res if not r["passed"]), None))
    ex = sequential(*synth(GP, pool, keys, 0.1, None, np.random.default_rng(seed + 1)), rule=rules[0])
    return dict(trials=trials, q=0.1, rules=list(rules), results=out, example_correct=dict(passed_at=ex["passed_at"], looks=[dict(look=x["look"], n=x["n"], gates={k: dict(v) for k, v in x["gates"].items()}) for x in ex["looks"]]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["reproduce", "dry", "gate"]); ap.add_argument("--out", required=True); ap.add_argument("--which", default="all")
    ap.add_argument("--trials", type=int, default=20); ap.add_argument("--gp", default=str(GP_PATH)); ap.add_argument("--eo"); ap.add_argument("--rep"); ap.add_argument("--rule", default=None)
    a = ap.parse_args(); t0 = time.time()
    if a.cmd == "reproduce":
        ours = reproduce(a.gp, a.which); ref = json.load(open(Path(a.gp).with_name(f"gate_power_v3_{a.which}.json"))); d = diff(ref, ours)
        res = dict(which=a.which, equal=not d, diffs=d[:50], wall_s=round(time.time() - t0, 1), ours=ours)
    elif a.cmd == "dry": res = dict(dry_run(a.trials, path=a.gp), wall_s=round(time.time() - t0, 1))
    else:
        import fb_r0eo as R0
        res = sequential(R0.load(a.eo), R0.load(a.rep), rule=a.rule)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8"); print(json.dumps({k: v for k, v in res.items() if k not in ("ours", "results", "example_correct", "looks")}, default=str))
