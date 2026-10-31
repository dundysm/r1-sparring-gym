"""The GPU sparring task's pure parts on CPU (gpu/sparring_task.py; docs/SPARRING_M1M2.md 2.1-2.5, 3.1, 8): no mjlab needed.
AttackSchedule: the aim is latched at the strike's onset (a target moved after it does not move the strike; moved before it,
it does), the guard blend has no target jumps at strike start / end / return, |Delta| <= 0.5 m, the glove reaches 5 mm inside
the head at the apex. HitAccount: the review's blk.py case (a contact going on before the window opens is never its first
contact: a clean head hit, not a block), pelvis is no candidate, an arm and a head episode opening on the same substep is a
hit as in HitMeter, the float32 episode clock late in an episode keeps HitMeter's 0.1 s merge and 0.15 s clean boundaries,
blocked windows equal r1gym HitMeter.windows on random contact sequences, and on short recorded CPU bouts (the calibration's own path, _cal_bout) both meters count the same hits
and windows; the closing speed is sign-free. The fists touch only the proxies, never the floor, in the scene composed as mjlab
composes it. widen leaves the warm start's actor and critic outputs unchanged (skipped without model_7496.pt). The export
files keep their schema. The pod's gates: calibration_check (the calibration must have chosen ATTACK, or a recorded
deviation) and gate()'s deviation for a speed pass at 3072 envs."""
import sys, json, math, importlib.util
from pathlib import Path
import numpy as np, mujoco, pytest
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
WARM_DIR = ROOT / "results" / "policy_mjlab_guard_footwork_r"; MOTIONS = ROOT / "gpu" / "motions"

def _load():
    spec = importlib.util.spec_from_file_location("sparring_task", ROOT / "gpu" / "sparring_task.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
T = _load()

@pytest.fixture(scope="module")
def lib(): return T.load_library()

def pose(n, yaw=0.3, xy=(1.0, -2.0)):
    """A standing defender: pelvis xy, yaw, torso_link origin 3 cm ahead at 0.82 m, its rotation."""
    pel = torch.tensor(xy).expand(n, 2).clone(); yw = torch.full((n,), yaw); c, s = math.cos(yaw), math.sin(yaw)
    RT = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]).expand(n, 3, 3).clone(); xT = torch.cat([pel, torch.full((n, 1), 0.82)], 1) + torch.tensor([0.03 * c, 0.03 * s, 0])
    return pel, yw, xT, RT

def forced(lib, hand=0, sig=1.0, seed=1, move=None, move_from=None, far=False, steps=120):
    """One strike of `hand` at the head, forced at step 1 (sigma `sig`) on a still defender. move: the defender's torso slips
    that far sideways (y) once the strike has latched, or from step `move_from` on. Per step: (glove target, latched, tau, Delta)."""
    sc = T.AttackSchedule(lib, 1, "cpu", 0.02, gen=torch.Generator().manual_seed(seed)); pl, yy, xt, rt = pose(1)
    if far: xt = xt + torch.tensor([2.0, 0.0, 0.0])
    sc.reset(torch.tensor([0]), pl, yy); sc.t_next[:] = 1e9; sid = int(sc.hseg[hand, 0]); out = []
    for k in range(steps):
        t = torch.tensor([k * 0.02])
        if k == 1:
            sc.phase[0, hand] = 1; sc.seg_id[0, hand] = sid; sc.t0[0, hand] = t; sc.sig[0, hand] = sig; sc.zone[0, hand] = 0; sc.S0[0, hand] = sc.S[0]; sc.f0[0, hand] = sc.f[0]
            sc.latched[0, hand] = False; sc.delta[0, hand] = 0; sc.t_apex[0, hand] = t + float(lib["seg"][sid, 3]) / (50 * sig)
        x2 = xt.clone()
        if move is not None and (bool(sc.latched[0, hand]) if move_from is None else k >= move_from): x2 = x2 + torch.tensor([0.0, move, 0.0])
        g, *_ = sc.step(t, pl, yy, x2, rt); out.append((g[0, hand].clone(), bool(sc.latched[0, hand]), float(sc.tau[0, hand]), sc.delta[0, hand].clone()))
    return sc, out, sid

# ---- AttackSchedule (2.4)
def test_latching_at_onset(lib):
    """Delta follows the defender until the strike's onset, then is world-fixed: a head that slips 0.3 m after onset doesn't
    move the strike at all; one that slips before it (from step 5, onset at step 13) changes Delta from then on and moves the
    apex by about the slip. The right hand's first segment: its Delta stays under the 0.5 m cap."""
    _, a, sid = forced(lib, 1); _, b, _ = forced(lib, 1, move=0.3); _, c, _ = forced(lib, 1, move=0.3, move_from=5)
    lat = [i for i, x in enumerate(a) if x[1] and x[2] > 0]; on = float(lib["seg"][sid, 2])   # latched and still striking
    assert len(lat) > 5 and min(a[i][2] for i in lat) >= on - 1e-6 and not a[lat[0] - 1][1] and a[lat[0] - 1][2] < on   # latched at the first tau >= onset
    assert max(float((a[i][0] - b[i][0]).abs().max()) for i in lat) < 1e-6 and [x[1] for x in a] == [x[1] for x in b]
    assert torch.equal(a[4][3], c[4][3]) and float((a[lat[0]][3] - c[lat[0]][3]).abs().max()) > 0.2 and float(torch.linalg.norm(a[lat[0]][3])) < 0.5 - 1e-3
    assert all(torch.equal(c[i][3], c[lat[0]][3]) for i in lat)   # and fixed after the onset
    ap = float(lib["seg"][sid, 3]); k = min(lat, key=lambda i: abs(a[i][2] - ap))
    assert 0.2 < float(c[k][0][1] - a[k][0][1]) < 0.35   # aimed before the onset: the apex follows the slip (less than 0.3: the aim
    # point sits TIP - delta back along the approach, which turns with the slip; 0.260 at delta +5 mm, 0.246 at -2 cm)

def test_aim_reaches_5mm_inside_the_head(lib):
    """At the apex (w = 1) the glove target is 5 mm inside the head's aim point along the strike's approach, or short where
    |Delta| hit the 0.5 m cap (the left hand's first segment, a downward torso-band strike, is capped; the right hand's isn't)."""
    capped = []
    for hand in (0, 1):
        sc, out, sid = forced(lib, hand); ap = float(lib["seg"][sid, 3]); k = min(range(len(out)), key=lambda i: abs(out[i][2] - ap) if out[i][2] > 0 else 1e9)
        pl, yy, xt, rt = pose(1); surf = xt[0] + rt[0] @ torch.tensor(T.SCHED["aim_head"]); uh = torch.nn.functional.normalize(surf[:2] - sc.S0[0, hand], dim=0)
        aim = surf - (T.TIP - T.SCHED["delta"]) * torch.cat([uh, torch.zeros(1)]); capped.append(float(torch.linalg.norm(out[k][3])) >= 0.5 - 1e-5)
        assert float(torch.linalg.norm(out[k][0] - aim)) < 0.01 or capped[-1]
    assert not all(capped)

def test_delta_capped_at_half_a_metre(lib):
    """A defender 2 m away: the bend Delta saturates at 0.5 m (the strike falls short rather than flying across the env)."""
    _, out, _ = forced(lib, far=True); d = [float(torch.linalg.norm(x[3])) for x in out]
    assert max(d) <= 0.5 + 1e-5 and max(d) > 0.5 - 1e-3

def test_guard_blend_continuity(lib):
    """Free-running schedule, 64 envs x 20 s: strikes start, end and return; no glove target moves faster than the library's
    fastest frame x sigma 1.3 (+ the frame's own drift), and the transition steps move no more than the strikes themselves."""
    N = 64; s = T.AttackSchedule(lib, N, "cpu", 0.02, gen=torch.Generator().manual_seed(0)); pel, yw, xT, RT = pose(N); s.reset(torch.arange(N), pel, yw)
    G, PH = [], []
    for k in range(1, 1000):
        g, *_ = s.step(torch.full((N,), k * 0.02), pel, yw, xT, RT); G.append(g.clone()); PH.append(s.phase.clone())
    G, PH = torch.stack(G), torch.stack(PH); jump = torch.linalg.norm(G[1:] - G[:-1], dim=-1) / 0.02
    libv = max(float(np.linalg.norm(np.diff(lib["kp"][a:a + n, :2], axis=0), axis=-1).max()) for a, n, *_ in lib["seg"]) * 50
    assert float(jump.max()) < 1.3 * libv + 2.5
    st, en, rt = ((PH[1:] == b) & (PH[:-1] == a) for a, b in ((0, 1), (1, 2), (2, 0))); assert st.any() and en.any() and rt.any()
    normal = float(jump[(PH[1:] == 1) & (PH[:-1] == 1)].quantile(0.999))
    assert all(float(jump[m].max()) <= max(normal, 1.0) for m in (st, en, rt))

# ---- HitAccount and closing speeds (2.5)
ZI = T.zone_index([l + "_shell" for l in T.LINKS])

def run(ha, forces, wins, vcs=2.0):
    """forces: per control step (N, 2, 5, 4) zone forces; wins {step: (hand, on, off)} opens a strike window at that step."""
    sid = torch.zeros(ha.n, 2, dtype=torch.long); won, woff = torch.zeros(ha.n, 2), torch.zeros(ha.n, 2); blk = []
    for k, F in enumerate(forces):
        if k in wins: h, a, b = wins[k]; sid[:, h] += 1; won[:, h] = a; woff[:, h] = b
        ha.process(F, torch.full((ha.n,), (k + 1) * 0.02), sid.clone(), won.clone(), woff.clone(), torch.full((ha.n, 2, 5, 4), vcs)); blk.append(ha.step["block"].clone())
    return torch.stack(blk)

def test_hitaccount_ignores_contact_from_before_the_window():
    """The review's repro (m1m2rev/blk.py): a fist resting on the forearm from t = 0, its window opens at 0.2 s, a clean 300 N
    head hit at 0.3 s: one window, a clean head hit, no block credited (spar_block 0)."""
    hb = T.HitAccount(1, "cpu", ZI); blk = []
    for k in range(1, 30):
        t = k * 0.02; Fz = torch.zeros(1, 2, 5, 4); Fz[0, 0, 3] = 10.0
        if 0.30 <= t < 0.32: Fz[0, 0, 0] = 300.0
        hb.process(Fz, torch.tensor([t]), torch.tensor([[1 if t > 0.1 else 0, 0]]), torch.tensor([[0.2, 0.0]]), torch.tensor([[0.45, 0.0]]), torch.full((1, 2, 5, 4), 3.0))
        blk.append(float(hb.step["block"][0]))
    hb.flush(torch.tensor([0.6]))
    assert sum(blk) == 0 and float(hb.ep["blocked"][0]) == 0 and float(hb.ep["head_clean"][0]) == 1 and float(hb.ep["windows"][0]) == 1

def test_hitaccount_block_rules():
    """A forearm block first then a head hit: blocked, spar_block paid once. Pelvis first, then the forearm: still blocked
    (pelvis is no candidate). Head lands first, forearm after: a hit, not blocked. A window with no contact: a miss."""
    ha = T.HitAccount(4, "cpu", ZI); F = [torch.zeros(4, 2, 5, 4) for _ in range(30)]
    F[3][0, 0, 3, :] = 30.0; F[5][0, 0, 0, :] = 60.0   # env 0: block zone at 0.065 s, head at 0.105 s
    F[3][1, 0, 2, :] = 30.0; F[5][1, 0, 3, :] = 30.0   # env 1: pelvis, then block zone
    F[3][2, 0, 0, :] = 60.0; F[5][2, 0, 4, :] = 30.0   # env 2: head lands, then upper arm
    blk = run(ha, F, {1: (0, 0.03, 0.3)}); ha.flush(torch.full((4,), 1.0))
    assert ha.ep["windows"].tolist() == [1.0] * 4 and ha.ep["blocked"].tolist() == [1.0, 1.0, 0.0, 0.0] and ha.ep["block_first"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert blk.sum(0).tolist() == [1.0, 1.0, 0.0, 0.0] and ha.ep["head_landed"].tolist() == [1.0, 0.0, 1.0, 0.0]

def test_hitaccount_equal_onset_is_a_hit():
    """The review's r1_simul.py: window [0.2025, 0.3525]; a block-zone touch and a 40 N head press open on the same substep
    (t = 0.35 s), the head press lands only after the window's end. HitMeter: 'head' (blocked needs the arm strictly first);
    HitAccount waits for the head episode and agrees: not blocked, no spar_block, one clean head hit."""
    ha = T.HitAccount(1, "cpu", ZI); F = [torch.zeros(1, 2, 5, 4) for _ in range(40)]
    F[69 // 4][0, 0, 3, 69 % 4] = 10.0
    for q in range(69, 75): F[q // 4][0, 0, 0, q % 4] = 40.0
    blk = run(ha, F, {5: (0, 0.2025, 0.3525)}); ha.flush(torch.full((1,), 1.0))
    assert float(blk.sum()) == 0 and ha.ep["windows"].tolist() == [1.0] and ha.ep["blocked"].tolist() == [0.0] and ha.ep["head_clean"].tolist() == [1.0]

def test_hitaccount_float32_clock_late_in_an_episode():
    """The episode clock is float32 (episode_length_buf.float() * step_dt, as AttackCommand._t_ep). After 12 s: a 30-substep
    60 N head contact (0.15 s: clean for HitMeter) is clean; two 4-substep 1.2 N*s presses exactly 0.1 s apart (HitMeter
    merges them) are one landed episode, not two; 0.105 s apart they are two."""
    def go(bursts, steps=700):
        ha = T.HitAccount(1, "cpu", ZI); F = torch.zeros(steps, 1, 2, 5, 4)
        for s0, L in bursts:
            for q in range(s0, s0 + L): F[q // 4, 0, 0, 0, q % 4] = 60.0
        z, zf = torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2)
        for k in range(steps): ha.process(F[k], torch.full((1,), k + 1, dtype=torch.long).float() * 0.02, z, zf, zf, torch.full((1, 2, 5, 4), 2.0))
        ha.flush(torch.full((1,), steps * 0.02)); return ha.ep["head_landed"].item(), ha.ep["head_clean"].item()
    s0 = 4 * 615 + 1
    assert go([(s0, 30)]) == (1.0, 1.0) and go([(s0, 31)]) == (1.0, 0.0)
    assert go([(s0, 4), (s0 + 23, 4)]) == (1.0, 1.0) and go([(s0, 4), (s0 + 24, 4)]) == (2.0, 2.0)

def test_hitaccount_blocked_windows_equal_hitmeter_on_random_contacts():
    """48 envs x 8 s of random contact bursts on every zone (both fists, 1-90 N, 1-12 substeps) and strike windows: per env,
    HitAccount's window count and blocked windows equal r1gym HitMeter.windows on the same contacts (its 'arm' = block +
    upper_arm)."""
    from r1gym.hits import HitMeter, DEFAULTS
    g = np.random.default_rng(7); NE, KS = 48, 400; HZ = ("head", "torso", "pelvis", "arm", "arm")
    F = np.zeros((KS, NE, 2, 5, 4), np.float32); sid = np.zeros((KS, NE, 2), np.int64); won, woff = np.zeros((KS, NE, 2), np.float32), np.zeros((KS, NE, 2), np.float32); wins = [[] for _ in range(NE)]
    for i in range(NE):
        for h in (0, 1):
            k, c = int(g.integers(5, 30)), 0
            while k < KS - 20:   # strikes >= 0.6 s apart, as the schedule's
                a = k * 0.02 + float(g.uniform(0.0, 0.05)) + 0.0025; b = a + float(g.uniform(0.08, 0.3)); c += 1; wins[i].append((round(a, 4), round(b, 4), ("l_hand", "r_hand")[h]))
                sid[k:, i, h] = c; won[k:, i, h] = a; woff[k:, i, h] = b; k += int(g.integers(30, 60))
            for _ in range(int(g.integers(15, 40))):
                z, s0, L, f = int(g.integers(0, 5)), int(g.integers(0, KS * 4 - 20)), int(g.integers(1, 13)), float(g.uniform(1, 90))
                for q in range(s0, s0 + L): F[q // 4, i, h, z, q % 4] += f
    ha = T.HitAccount(NE, "cpu", ZI)
    for k in range(KS): ha.process(torch.as_tensor(F[k]), torch.full((NE,), (k + 1) * 0.02), torch.as_tensor(sid[k]), torch.as_tensor(won[k]), torch.as_tensor(woff[k]), torch.full((NE, 2, 5, 4), 2.0))
    ha.flush(torch.full((NE,), KS * 0.02)); bl, nw = [], []
    for i in range(NE):
        hm = object.__new__(HitMeter); hm.dt, hm.p, hm._ev, hm.log = 0.005, dict(DEFAULTS), {}, []
        for k in range(KS):
            for j in range(4):
                acc = {}
                for h in (0, 1):
                    for z in range(5):
                        if F[k, i, h, z, j] > 0: key = ("lr"[h], "hand", HZ[z]); acc[key] = acc.get(key, 0.0) + float(F[k, i, h, z, j])
                if acc: hm.log.append((round(k * 0.02 + 0.005 * (j + 1), 4), acc, {}))
        o = hm.windows(wins[i]); bl.append(o.count("blocked")); nw.append(len(o))
    assert ha.ep["blocked"].numpy().tolist() == [float(x) for x in bl] and ha.ep["windows"].numpy().tolist() == [float(x) for x in nw]
    assert 50 < sum(bl) < sum(nw)   # a mix of blocked and other windows

def test_hitaccount_matches_hitmeter_on_recorded_bouts():
    """The calibration's own path (_cal_bout: R1HitEnv with the frozen warm start, the schedule driving the attacker chain on
    the proxies, HitMeter on MuJoCo's contacts and HitAccount on the hit sensors), 3 bouts x 6 s: the same head / torso hits,
    clean head hits, strike windows and blocked windows, and the same head-hit impulses."""
    onnx = WARM_DIR / "policy.onnx"
    if not onnx.exists(): pytest.skip("warm start policy.onnx absent")
    nt = torch.get_num_threads(); H, A, IH, IA = [], [], [], []
    try:
        for seed in (1, 2, 3):
            r = T._cal_bout((seed, T.ATTACK["mass"], T.ATTACK["fmax"], T.ATTACK["delta"], "proxy", 6.0, str(onnx))); n, w = r["hm"]["n"], r["hm"]["windows"]
            H.append([n["head_landed"], n["head_clean"], n["torso_landed"], sum(w.values()), w["blocked"]]); IH += r["hm"]["head_hit_imp_Ns"]
            A.append([r["ha"][k] for k in ("head_landed", "head_clean", "torso_landed", "windows", "blocked")]); IA += r["ha_imp"]
    finally: torch.set_num_threads(nt)
    H, A = np.array(H, float), np.array(A, float)
    assert H[:, 0].sum() >= 2 and H[:, 3].sum() >= 6   # a non-trivial case: head hits and strike windows
    assert np.abs(H - A).max() <= 1 and np.abs(H.sum(0) - A.sum(0)).max() <= 1   # hits differ only where HitMeter logs a 0 N contact (doc 8)
    if len(IH) == len(IA): assert np.allclose(sorted(IH), sorted(IA), rtol=0.05, atol=0.01)

def test_closing_speed_is_sign_free():
    """|v . n|: the same closing speed whichever way the sensor's net force points (Warp's sign is unchecked); signed=True
    applies FORCE_SIGN, and flips with the force. HitAccount's counts don't move when every force vector is negated."""
    g = torch.Generator().manual_seed(0); vr = torch.randn(8, 2, 3, 4, generator=g) * 3; Fv = torch.randn(8, 2, 5, 4, 3, generator=g) * 50
    a, b = T.closing_speeds(vr, Fv), T.closing_speeds(vr, -Fv); s, s2 = T.closing_speeds(vr, Fv, signed=True), T.closing_speeds(vr, -Fv, signed=True)
    assert a.shape == (8, 2, 5, 4) and torch.allclose(a, b) and (a >= 0).all() and torch.allclose(s, -s2) and torch.allclose(s.abs(), a)
    v1 = torch.zeros(1, 2, 3, 4); v1[0, 0, 0] = 3.0; F1 = torch.zeros(1, 2, 5, 4, 3); F1[0, 0, 0, :, 0] = -50.0
    assert abs(float(T.closing_speeds(v1, F1, signed=True)[0, 0, 0, 0]) - 3.0) < 1e-5 and abs(float(T.closing_speeds(v1, -F1)[0, 0, 0, 0]) - 3.0) < 1e-5
    ep = []; Fm = Fv.norm(dim=-1) * (torch.rand(8, 2, 5, 4, generator=g) < 0.3)
    for sgn in (1.0, -1.0):
        ha = T.HitAccount(8, "cpu", ZI); vc = T.closing_speeds(vr, sgn * Fv)
        for k in range(20): ha.process(Fm if k % 7 == 2 else torch.zeros_like(Fm), torch.full((8,), (k + 1) * 0.02), torch.zeros(8, 2, dtype=torch.long), torch.zeros(8, 2), torch.zeros(8, 2), vc)
        ha.flush(torch.full((8,), 1.0)); ep.append({k: v.clone() for k, v in ha.ep.items()})
    assert all(torch.equal(ep[0][k], ep[1][k]) for k in ("head_landed", "head_clean", "torso_landed", "torso_clean", "head_imp_landed")) and float(ep[0]["head_landed"].sum()) > 0

# ---- the pod's gates (prereg_m2.json m1_gate_before_training)
def test_calibration_check(tmp_path, monkeypatch):
    """Passes only when the calibration chose ATTACK's row (constants_match); otherwise only with SPAR_CALIB_DEVIATION set,
    which comes back as the recorded deviation; an unreadable file never passes."""
    att = dict(mass=T.ATTACK["mass"], fmax=T.ATTACK["fmax"], delta=T.ATTACK["delta"]); row = dict(att, geometry="proxy", **{"pass": False}, hitmeter=dict(impulse_per_head_hit_Ns=dict(median=3.6)))
    def write(chosen, match):
        f = tmp_path / "cal.json"; f.write_text(json.dumps(dict(rows=[row], chosen=chosen, constants_match=match))); return f
    monkeypatch.delenv("SPAR_CALIB_DEVIATION", raising=False)
    assert T.calibration_check(write(att, True))["pass"] is True
    r = T.calibration_check(write(dict(mass=1.5, fmax=80.0, delta=-0.02), False)); assert r["pass"] is False and r["attack_row"]["median_Ns"] == 3.6 and r["deviation"] is None
    assert T.calibration_check(write(None, False))["pass"] is False and T.calibration_check(write(att, False))["pass"] is False
    monkeypatch.setenv("SPAR_CALIB_DEVIATION", "kept on purpose")
    r = T.calibration_check(write(dict(mass=1.5, fmax=80.0, delta=-0.02), False)); assert r["pass"] is True and r["match"] is False and r["deviation"] == "kept on purpose"
    assert T.calibration_check(tmp_path / "absent.json")["pass"] is False

def test_gate_records_a_3072_env_pass_as_a_deviation(tmp_path):
    """gate(): mean iteration time over 20-200, nan terminations, mean collection time; a pass at 3072 envs is listed under
    'deviations' (the preregistration says 4096), one at 4096 is not."""
    log = tmp_path / "smoke.log"; log.write_text("".join(f"Collection time: 1.{k % 10}s \nIteration time: 1.5s\nEpisode_Termination/nan: 0.0000\n" for k in range(200)))
    for ne, dev in ((4096, False), (3072, True)):
        m1 = tmp_path / f"m1_{ne}.json"; m1.write_text(json.dumps({"m1": {"pass": True}}))
        j = T.gate(str(m1), str(log), "mesh", ne); s = j["smoke"][-1]
        assert j["pass"] and s["mean_iter_s"] == 1.5 and abs(s["mean_collection_s"] - 1.45) < 0.01 and s["iter_s_at_first"] == 1.5 and bool(j.get("deviations")) == dev

# ---- the scene (2.1): fists touch only the proxies
def _scene(proxy):
    """The R1 with its proxies and the attacker composed as mjlab composes them (robot/ and attacker/ prefixes, the attacker under
    a mocap base, CollisionCfg(FULL_COLLISION, disable_other_geoms) on the robot, a terrain plane, SceneCfg.spec_fn's proxy bits)."""
    from r1gym import ASSETS
    def r1():
        s = mujoco.MjSpec.from_file(str(ASSETS / "r1" / "r1.xml")); s.meshdir = str(ASSETS / "r1" / "assets"); names = {b.name for b in s.bodies}
        for e in list(s.excludes):
            if e.bodyname1 not in names or e.bodyname2 not in names: s.delete(e)
        return s
    rs = T._robot_spec(r1, proxy)
    for g in rs.geoms:
        if g.name.endswith("_collision"): g.contype, g.conaffinity, g.condim = 1, 1, (3 if "foot" in g.name else 1)
        else: g.contype, g.conaffinity = 0, 0
    sc = mujoco.MjSpec(); sc.option.timestep = T.SUB_DT; sc.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    sc.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05]); sc.attach(rs, prefix="robot/", frame=sc.worldbody.add_frame())
    w = mujoco.MjSpec(); mb = w.worldbody.add_body(name="mocap_base", mocap=True); w.attach(T.attacker_spec(actuators=True), prefix="", frame=mb.add_frame())
    sc.attach(w, prefix="attacker/", frame=sc.worldbody.add_frame()); T.add_contact_sensors_cpu(sc, "robot/", "attacker/"); T._scene_spec(None, sc)
    return sc.compile()

@pytest.mark.parametrize("proxy", ["mesh", "capsule"])
def test_fists_touch_only_proxies_never_the_floor(proxy):
    """Left fist driven into the head, right fist driven 0.4 m below the floor, defender's root held: the fists touch proxies
    only (no floor, no robot collision geom), the right one passes through the floor plane, the head sensor fires."""
    m = _scene(proxy); d = mujoco.MjData(m); fl = m.geom("terrain").id; caps = {m.geom(f"attacker/att_{s}_capsule").id for s in "lr"}
    prox = {m.geom(f"robot/{l}_shell").id for l in T.LINKS}; qa = [m.jnt_qposadr[m.joint("attacker/" + j).id] for j in T.ATT_JOINTS]
    act = [m.actuator("attacker/" + j).id for j in T.ATT_JOINTS]; d.qpos[2] = 0.76; d.qpos[3] = 1.0; mujoco.mj_forward(m, d)
    tb = m.body("robot/torso_link").id; head = d.xpos[tb] + d.xmat[tb].reshape(3, 3) @ np.array([0.09, 0, 0.36])
    q0 = np.array([0.6, 0.0, 1.1, np.pi, 0.0, 0.3, -0.3, 0.05, 0.0, 0.0]); d.qpos[qa] = q0; tgt = q0.copy(); tgt[:3] = head; tgt[5:8] = [0.3, -0.3, -0.4]; d.ctrl[act] = tgt
    n_floor = n_prox = n_other = 0; zmin, fh = 1.0, 0.0; rz = m.jnt_qposadr[m.joint("attacker/att_r_z").id]
    sh = [m.sensor(f"hit_l_{l}_shell_force").adr[0] for l in ("head_yaw_link", "head_pitch_link")]
    for _ in range(200):
        d.qpos[:7] = [0, 0, 0.76, 1, 0, 0, 0]; d.qvel[:6] = 0; mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            pair = {c.geom1, c.geom2}
            if pair & caps: n_floor += fl in pair; n_prox += bool(pair & prox); n_other += not (fl in pair or pair & prox)
        zmin = min(zmin, float(d.qpos[rz])); fh = max(fh, sum(float(np.linalg.norm(d.sensordata[a:a + 3])) for a in sh))
    assert n_floor == 0 and n_other == 0 and n_prox > 0 and zmin < -0.2 and fh > 0

# ---- widen (3.1) and the export files (2.3)
def test_widen_leaves_outputs_unchanged(tmp_path):
    src = WARM_DIR / "model_7496.pt"
    if not src.exists(): pytest.skip("model_7496.pt absent")
    dst = tmp_path / "w" / "model_7496.pt"; T.widen(str(src), str(dst))
    c0, c1 = (torch.load(p, weights_only=False, map_location="cpu") for p in (src, dst)); a1, s1 = c1["actor_state_dict"], c1["optimizer_state_dict"]["state"]
    assert a1["mlp.0.weight"].shape == (512, 111) and c1["critic_state_dict"]["mlp.0.weight"].shape == (512, 136)
    assert s1[1]["exp_avg"].shape == (512, 111) and float(s1[1]["exp_avg"][:, 83:].abs().max()) == 0 and torch.equal(s1[1]["exp_avg"][:, :83], c0["optimizer_state_dict"]["state"][1]["exp_avg"])
    g = torch.Generator().manual_seed(0); x, xn = torch.randn(512, 83, generator=g) * 2, torch.randn(512, 28, generator=g) * 5
    assert float((T.load_actor(str(src))(x) - T.load_actor(str(dst))(torch.cat([x, xn], 1))).abs().max()) < 1e-4
    def critic(sd, x):
        sd = {k: v.double() if v.is_floating_point() else v for k, v in sd.items()}; z = (x.double() - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
        for i in (0, 2, 4, 6): z = z @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]; z = torch.nn.functional.elu(z) if i < 6 else z
        return z
    xc = torch.randn(512, 98, generator=g) * 2
    assert float((critic(c0["critic_state_dict"], xc) - critic(c1["critic_state_dict"], torch.cat([xc, torch.randn(512, 38, generator=g) * 5], 1))).abs().max()) < 1e-9
    mn, sd = a1["obs_normalizer._mean"][0, 83:], a1["obs_normalizer._std"][0, 83:]
    assert mn[21:23].tolist() == [0.0, -1.0] and mn[23:].tolist() == [0.5] * 5 and sd[15:21].tolist() == [2.0] * 6

def test_export_schema():
    from r1gym.env import TRAIN_CLIPS
    L = np.load(MOTIONS / "attacks_train.npz"); Sh = np.load(MOTIONS / "attacks_shells.npz"); j = json.loads((MOTIONS / "attacks_train.json").read_text()); seg = L["seg"]
    assert L["kp"].dtype == np.float32 and L["kp"].shape[1:] == (5, 3) and L["yaw"].shape == (len(L["kp"]),) and L["yaw"].dtype == np.float32 and float(L["fps"]) == 50.0
    assert seg.shape == (37, 6) and seg.dtype == np.int32 and (seg[:, 5] == 0).sum() == 17 and L["guard"].shape == (2, 5, 3)
    assert (seg[1:, 0] == seg[:-1, 0] + seg[:-1, 1]).all() and seg[-1, 0] + seg[-1, 1] == len(L["kp"]) and (seg[:, 2] == 12).all() and (seg[:, 4] + 12 == seg[:, 1] - 1).all()
    assert ((seg[:, 2] <= seg[:, 3]) & (seg[:, 3] <= seg[:, 4])).all()
    assert np.allclose(np.linalg.norm(L["kp"][:, :2] - L["kp"][:, 2:4], axis=-1), T.ELBOW_TO_FIST, atol=1e-3)   # the GPU elbow = glove - 0.195 x axis
    assert j["n"] == 37 == len(j["segments"]) and {x["clip"] for x in j["segments"]} <= set(TRAIN_CLIPS) and "git" in j and "rule" in j
    assert Sh["points"].shape == (16, 64, 3) and Sh["points"].dtype == np.float32 and Sh["capsule"].shape == (16, 7) and tuple(Sh["links"].tolist()) == T.LINKS
    assert [str(z) for z in Sh["zones"]] == [T.ZONE_OF_LINK[l] for l in T.LINKS] and tuple(str(z) for z in Sh["zone_names"]) == T.ZONES
    sch = T.AttackSchedule(T.load_library(), 1); assert sch.kp.shape == L["kp"].shape   # load_library reads it as the task does
