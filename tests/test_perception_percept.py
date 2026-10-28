"""M1: the R1 turns toward what its own head camera sees (r1gym/perception/percept.py, hub.py, detect_colorkey.py, env.py's
perception key and init_yaw, scenario.py's off-axis episodes). Tests that draw need an offscreen GL context and skip
without one (CI has none); the truth, timing and firewall tests run anywhere."""
import sys, json, numpy as np, mujoco, pytest
from r1gym import CONFIGS, ROOT
from r1gym.env import R1HitEnv, VAL_CLIPS
from r1gym.model import build_spec, load_cfg, R1Robot
from r1gym.opponent import FightReplay
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import demo_env, clean_starts, offaxis_episodes, REF_POSES, OFFAXIS_BEARINGS_DEG
from r1gym.perception.camera import HeadCameraSpec
from r1gym.perception.percept import attach, Percept, PerceptStream, SCHEMA, wrap
from r1gym.perception.render_scene import gl_available

MJ = load_cfg(CONFIGS / "r1_mjlab.yaml"); GL = gl_available(); needs_gl = pytest.mark.skipif(not GL, reason="no offscreen GL context here")
GUARD = "guard=results/policy_mjlab_guard/policy.onnx"

def guard(**over):
    c = load_controller(GUARD); cfg = {**controller_cfg(c), "opponent_mode": "policy", **over}; return c, cfg, build_spec(opponent=True, cfg=cfg).compile()

def legacy_face(e):   # r1gym/env.py _face before M1 (commit cdddb94), verbatim but for self -> e
    if not e.face or e.opp is None: return
    d, r = e.d, e.r; w, x, y, z = d.qpos[r.free_q + 3:r.free_q + 7]; v = d.xpos[min(e.opp.bodies)][:2] - d.qpos[r.free_q:r.free_q + 2]
    err = np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))))
    if e.face_band is not None:
        e.turning = abs(err) > e.face_band[1] if e.turning else abs(err) > e.face_band[0]
        if not e.turning: e.command[2] = 0.0; return
    e.command[2] = float(np.clip(e.FACE_GAIN * err, -e.FACE_MAX, e.FACE_MAX))

def test_truth_mode_is_the_legacy_face_and_hold():
    """perception: truth (the default) commands exactly what _face did before M1, on 200 random states: any pose, opponent
    anywhere, deadband on or off, either turning state; _hold is untouched."""
    c = {**MJ, "face_opponent": True, "hold_position": True}; m = build_spec(cfg=c, opponent=True).compile(); rng = np.random.default_rng(0)
    e = R1HitEnv(seed=0, cfg=c, model=m, episode_s=1, max_impulse=0, train=False); e.opp = FightReplay(m, "kfab_200", distance=0.6, cfg={**c, "opponent_mode": "puppet"}); e.reset(hits=[], delay=0)
    assert e.perception == "truth" and e.percepts is None; adr = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]; q = e.r.free_q; diff = 0.0
    for k in range(200):
        quat = rng.normal(size=4) * [1, 0.1, 0.1, 1]; e.d.qpos[q:q + 7] = np.r_[rng.uniform(-1, 1, 2), 0.74, quat / np.linalg.norm(quat)]
        e.d.qpos[adr:adr + 2] = rng.uniform(-2, 2, 2); mujoco.mj_kinematics(m, e.d); e.face_band = [None, [0.35, 0.1]][k % 2]
        state = (rng.random() < 0.5, rng.random() < 0.5, rng.uniform(-1, 1, 3)); out = []
        for f in (legacy_face, lambda e: e._face()):
            e.turning, e.returning, e.command = state[0], state[1], state[2].copy(); f(e); e._hold(); out.append((e.command.copy(), e.turning, e.returning))
        diff = max(diff, float(np.abs(out[0][0] - out[1][0]).max())); assert out[0][1:] == out[1][1:]
    assert diff <= 1e-12, diff

@pytest.mark.skipif(sys.platform != "win32", reason="the stored results were generated on Windows; bouts are chaotic, so another platform's float/ONNX differences diverge them")
def test_stored_val_episodes_rerun_bit_identical():
    """16 stored validation bouts per shipped policy (results/eval_opponent/raw_val.json) re-run bit-identically (every row
    field) under the gym.json they were run with (face off, hold as shipped), perception truth."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("eval_opponent", ROOT.parent / "scripts" / "eval_opponent.py"); eo = importlib.util.module_from_spec(sp); sp.loader.exec_module(eo)
    r = json.load(open(ROOT.parent / "results" / "eval_opponent" / "raw_val.json")); eps = r["episodes"][:16]; assert {e["clip"] for e in eps} == set(VAL_CLIPS)
    for lab, path in [("RL guard", "results/policy_mjlab_guard/policy.onnx"), ("RL step", "results/policy_mjlab/policy.onnx")]:
        c = load_controller(f"{lab}={path}"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
        assert cfg["perception"] == "truth" and not cfg["face_opponent"]
        assert eo.run(c, eps, build_spec(opponent=True, cfg=cfg).compile(), cfg, 10.0, threads=4) == r["raw"][lab][:16], lab

def test_init_yaw_and_offaxis_episodes():
    """reset(init_yaw) turns the R1 about its pelvis, so the opponent (placed on +x) sits at -init_yaw; None changes nothing.
    Off-axis episodes: every bearing of a (clip, distance, start) shares one clean start time; perception seeds are unique."""
    _, cfg, m = guard()
    E = [R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=1, max_impulse=0, train=False) for _ in range(2)]
    for x in E: x.opp = FightReplay(m, "kfab_200", distance=0.6, cfg=cfg); x.opp_t0 = 1.0
    o0, o1 = E[0].reset(hits=[], delay=0), E[1].reset(hits=[], delay=0, init_yaw=None); assert np.array_equal(E[0].d.qpos, E[1].d.qpos) and np.array_equal(o0, o1); a = E[0]
    b0 = a.true_bearing()[0]   # the clip's in-place pose isn't exactly on +x: init_yaw shifts whatever bearing it has
    for b in (30.0, -50.0): a.reset(hits=[], delay=0, init_yaw=-np.radians(b)); assert abs(wrap(a.true_bearing()[0] - b0 - np.radians(b))) < 1e-9
    g0 = clean_starts(cfg, ["kfab_204"], [0.55]); g1 = clean_starts(cfg, ["kfab_204"], [0.55], yaw=0.0); assert np.array_equal(g0[("kfab_204", 0.55)], g1[("kfab_204", 0.55)])
    eps = offaxis_episodes(cfg, ["kfab_204"], [0.55], starts=2); assert len(eps) == 2 * len(OFFAXIS_BEARINGS_DEG) and len({e["pseed"] for e in eps}) == len(eps)
    for k in range(2): assert len({e["t0"] for e in eps if e["start"] == k}) == 1
    assert sorted({e["bearing_deg"] for e in eps}) == sorted(OFFAXIS_BEARINGS_DEG) and all(np.isclose(e["init_yaw"], -np.radians(e["bearing_deg"])) for e in eps)

def _bout(source, steps=150, spec=None, hub=None, phase=None, filter_s=0.0, seed=7, deg=40.0, **over):
    """Guard vs the tracked opponent from 40 deg off-axis, turn command on with the val grid's widest deadband."""
    c, cfg, m = guard(face_opponent=True, face_deadband=[0.35, 0.1], perception=source, percept_filter_s=filter_s, **over); s = []
    def setup(e):
        if source != "truth": st = attach(e, source, spec=spec, seed=seed, hub=hub); st.phase = phase; s.append(st)
    e, o = demo_env(m, cfg, t0=1.0, init_yaw=-np.radians(deg), setup=setup); Q, W = [], []
    for _ in range(steps): o, *_ = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy()); W.append(e.command[2])
    return e, (s[0] if s else None), np.array(Q), np.array(W)

def test_truth_lag_at_zero_latency_is_truth():
    """On a camera running at the control rate with frames on the ticks, zero latency, no jitter and no drops, truth_lag is
    truth: the same commands and a bit-identical bout. So the pipeline adds nothing but the camera's timing."""
    sp = HeadCameraSpec(gym_cfg=MJ); sp.rate, sp.latency, sp.jitter, sp.drop = 1.0 / MJ["model"]["control_dt"], np.zeros(2), 0.0, (0.0, 0.0)
    _, _, Qt, Wt = _bout("truth"); _, s, Ql, Wl = _bout("truth_lag", spec=sp, phase=0.0)
    assert np.abs(Wt).max() > 0.3 and np.array_equal(Wt, Wl) and np.array_equal(Qt, Ql) and len(s.delivered) == 151

def test_delivery_is_causal_and_lost_tracks_stand_still():
    """No percept is delivered before t_capture + latency, or used before it is delivered; a dropped frame is never estimated.
    With every later frame dropped the R1 keeps turning toward the last bearing (gyro-compensated) for percept_lost_s, then
    stands still."""
    e, s, _, _ = _bout("truth_lag", steps=100); sp = s.spec; lo = sp.latency[0]
    assert all(p.t_avail >= p.t_capture + lo - 1e-12 and t >= p.t_avail - 1e-9 for t, p in s.delivered) and len(s.delivered) > 50
    assert all(age >= lo - 1e-9 for t, b, age in s.used if b is not None) and all(sl[1] is None for _, ta, _, sl in s.captures if ta is None)
    s.queue.p = 1.0; c = load_controller(GUARD); o = e._raw_obs(); none_at = None
    for k in range(80):
        o, *_ = e.step(c.act(o[None])[0]); t, b, age = s.used[-1]
        if b is None: none_at = none_at or t; assert e.command[2] == 0.0 and not e.turning
        else: assert none_at is None and age <= s.lost_s + 1e-9 and np.isclose(b, wrap(s.track[0] - (s.yaw - s.track[1])))
    assert none_at is not None and abs(none_at - (s.t_seen + s.lost_s)) <= e.dt + 1e-9

class _Firewall:
    def __getattr__(self, k): raise AssertionError(f"perception read MjData.{k}")

class _StubTwin:   # GL-free stand-in for the render twin: an empty grey image whatever the state
    spec = HeadCameraSpec(gym_cfg=MJ)
    def sync(self, qpos): self.q = np.array(qpos)
    def frame(self, cam="head"): return np.full((120, 160, 3), 128, np.uint8)

def test_camera_estimates_never_read_mjdata():
    """Firewall: with the physics MjData swapped for an object that raises on any read, the hub renders and estimates the
    pending frames and env._face turns on them: they come from the snapshot (the render twin's input) and the sensed
    proprioception only. The renderer draws the snapshot, not the live state."""
    from r1gym.perception.hub import PerceptHub
    from r1gym.perception.detect_colorkey import ColorKeyDetector
    if GL:
        from r1gym.perception.render_scene import RenderTwin
        _, cfg, m = guard(); spec = HeadCameraSpec(gym_cfg=cfg); twin = RenderTwin(cfg, physics=m, spec=spec)
    else: spec = _StubTwin.spec; twin = _StubTwin()
    hub = PerceptHub(twin, ColorKeyDetector(spec, MJ)); e, s, _, _ = _bout("camera", steps=1, spec=spec); c = load_controller(GUARD); o = e._raw_obs(); seen = 0
    for k in range(60):
        o, *_ = e.step(c.act(o[None])[0]); d = e.d; e.d = _Firewall()
        try: seen += hub.process([s]); e._face()
        finally: e.d = d
    assert seen >= 30 and len(s.delivered) >= 25 and (not GL or np.mean([p.presence for _, p in s.delivered]) > 0.9)
    if GL:   # a snapshot taken, then the live opponent moved 5 m behind the R1: the frame still shows it where it was
        while not s.pending: o, *_ = e.step(c.act(o[None])[0])
        sn = s.pending[-1]; adr = e.opp.qadr; d.qpos[adr:adr + 2] = d.qpos[:2] - [5.0, 0.0]; hub.process([s]); assert sn.slot[1]["presence"]
        twin.close()

@needs_gl
def test_inline_and_batched_hub_give_the_same_bout():
    """Rendering inline (one env: the stream calls the hub) or after each tick (a runner's batch) gives bit-identical bouts:
    a frame is never needed in the tick that captured it."""
    from r1gym.perception.render_scene import RenderTwin
    from r1gym.perception.hub import PerceptHub
    from r1gym.perception.detect_colorkey import ColorKeyDetector
    c, cfg, m = guard(face_opponent=True, face_deadband=[0.35, 0.1], perception="camera"); spec = HeadCameraSpec(gym_cfg=cfg)
    twin = RenderTwin(cfg, physics=m, spec=spec); hub = PerceptHub(twin, ColorKeyDetector(spec, cfg)); out = []
    for inline in (True, False):
        s = []; e, o = demo_env(m, cfg, t0=1.0, init_yaw=-np.radians(40), setup=lambda e: s.append(attach(e, "camera", spec=spec, seed=7, hub=hub if inline else None))); Q = []
        for _ in range(120):
            o, *_ = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy())
            if not inline: hub.process(s)
        out.append((np.array(Q), np.array([p[:8] for _, p in s[0].delivered], float)))
    twin.close(); assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1], equal_nan=True) and len(out[0][1]) > 60

@needs_gl
def test_colorkey_finds_the_opponent_and_only_it():
    """A standing opponent (arms down or in a guard) placed at a known bearing and range, facing the R1 in its guard: the
    colour key's mask lies on the opponent (within the 1 px the render's antialiasing spreads it; never on the R1's own red
    or blue ankle links). Seen whole, its bearing is within 2 deg (the lens's 12.5 deg parallax corrected); at 45-50 deg the
    R1's own forearm hides part of it and the visible part's median pulls the bearing inward, by up to ~12 deg (the colour
    key has no notion of occlusion). Range is the nearest floor contact, the toes of a square stance: up to ~0.22 m short.
    An opponent behind the R1 gives no detection."""
    from r1gym.perception.render_scene import RenderTwin
    from r1gym.perception.detect_colorkey import ColorKeyDetector
    from r1gym.perception.camera import area_down
    cfg = {**MJ, "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=cfg).compile(); d = mujoco.MjData(m); r = R1Robot(m, cfg); spec = HeadCameraSpec(gym_cfg=cfg)
    tw = RenderTwin(cfg, physics=m, spec=spec); det = ColorKeyDetector(spec, cfg); oq = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]; errs = []
    for b, rng_, pose in [(0.0, 0.6, "home"), (30.0, 0.6, "guard"), (-30.0, 0.7, "home"), (45.0, 0.55, "guard"), (-50.0, 0.65, "guard"), (-15.0, 0.9, "home")]:
        r.reset(m, d); d.qpos[r.qadr] = REF_POSES["guard"]; a = np.radians(b); y = a + np.pi
        d.qpos[oq:oq + 7] = [rng_ * np.cos(a), rng_ * np.sin(a), 0.74, np.cos(y / 2), 0, 0, np.sin(y / 2)]; d.qpos[oq + 7:oq + 31] = REF_POSES[pose]; mujoco.mj_forward(m, d)
        tw.sync(d.qpos); img = tw.frame("head"); _, opp = tw.masks(tw.render("head", "seg")); mk = det.mask(img)
        est = det(img, d.qpos[r.qadr], d.xmat[r.pel].reshape(3, 3).T @ [0, 0, -1.0]); v = d.xpos[m.body("opp_pelvis").id][:2] - d.qpos[:2]
        o = np.pad(area_down(opp.astype(float), 2) > 0, 1); near = np.any([o[1 + i:121 + i, 1 + j:161 + j] for i in (-1, 0, 1) for j in (-1, 0, 1)], 0)   # 3x3 dilation
        assert mk.sum() > 300 and near[mk].all(), b
        errs.append((round(float(np.degrees(abs(wrap(est["bearing"] - np.arctan2(v[1], v[0]))))), 2), round(abs(est["range"] - float(np.hypot(*v))), 3))); assert np.sign(est["bearing"]) == np.sign(b) or b == 0
    assert max(errs[k][0] for k in (0, 1, 5)) < 2.0 and max(e for e, _ in errs) < 13.0 and max(x for _, x in errs) < 0.25, errs
    d.qpos[oq:oq + 2] = [-1.5, 0.0]; mujoco.mj_forward(m, d); tw.sync(d.qpos); g = d.xmat[r.pel].reshape(3, 3).T @ [0, 0, -1.0]
    assert not det(tw.frame("head"), d.qpos[r.qadr], g)["presence"]; tw.close()

def test_shuffled_replays_another_bouts_estimates():
    """The shuffled control gets each frame's estimate from a donor bout's stream at the same time since the start (none
    before the donor's first frame), never its own; derange() never maps a bout to itself."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("face_ab", ROOT.parent / "scripts" / "face_ab.py"); fa = importlib.util.module_from_spec(sp); sp.loader.exec_module(fa)
    for n in (2, 5, 96): d = fa.derange(n); assert sorted(d) == list(range(n)) and (d != np.arange(n)).all()
    est = lambda b: dict(presence=True, bearing=b, bearing_sd=0.0, range=0.6, range_sd=0.0, rel_heading=np.nan)
    donor = [(0.05, est(0.3)), (0.10, est(-0.2)), (0.15, est(0.1))]; c, cfg, m = guard(perception="shuffled")
    e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=1, max_impulse=0, train=False); e.opp = FightReplay(m, "kfab_200", distance=0.6, cfg=cfg); e.opp_t0 = 1.0
    s = attach(e, "shuffled", donor=donor, seed=3); e.reset(hits=[], delay=0)
    for _ in range(12): e.step(np.zeros(24))
    for tc, ta, truth, slot in s.captures:
        if ta is None: continue
        i = max([k for k, (t, _) in enumerate(donor) if t <= tc + 1e-9], default=None); assert slot[1] is donor[i][1] if i is not None else not slot[1]["presence"]

def test_percept_schema():
    assert SCHEMA == 1 and Percept._fields == ("t_capture", "t_avail", "presence", "bearing", "bearing_sd", "range", "range_sd", "rel_heading", "source")
    assert wrap(0.3) == 0.3 and np.isclose(wrap(np.pi + 0.1), -np.pi + 0.1)
    with pytest.raises(RuntimeError, match="PerceptStream"):
        _, cfg, m = guard(face_opponent=True, perception="camera"); e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=1, max_impulse=0, train=False)
        e.opp = FightReplay(m, "kfab_200", distance=0.6, cfg=cfg); e.reset(hits=[], delay=0)

def test_m1_results_carry_run_meta_and_recompute():
    """The M1 results carry r1gym.perception.run_meta (GL strings where they rendered); the test run is the preregistered
    one (its sha, choice, size) and every stored pass/fail recomputes from its stored bouts with the preregistered rules."""
    import importlib.util
    from r1gym.perception import run_meta
    sp = importlib.util.spec_from_file_location("face_ab", ROOT.parent / "scripts" / "face_ab.py"); fa = importlib.util.module_from_spec(sp); sp.loader.exec_module(fa)
    P = ROOT.parent / "results" / "perception"; R = {f: json.loads((P / f"{f}.json").read_text()) for f in ("m1_offaxis_val", "m1_offaxis_test", "preregistration_m1", "camera_turn")}
    for f, r in R.items():
        assert set(run_meta()) <= set(r) and len(r["git"]) == 40 and isinstance(r["git_dirty"], list), f
        if f != "preregistration_m1": assert r["gl_renderer"] and not r["gl_renderer"].startswith("unavailable"), f
    v, t, pre = R["m1_offaxis_val"], R["m1_offaxis_test"], R["preregistration_m1"]
    assert t["preregistration_sha"] == fa.sha(P / "preregistration_m1.json") and pre["val_file_sha"] == fa.sha(P / "m1_offaxis_val.json") and pre["criteria"] == fa.M1_CRITERIA
    assert t["choice"] == pre["choice"] == v["choice"] and t["starts"] == pre["test"]["starts"] and t["clips"] == pre["test"]["clips"] and len(t["episodes"]) == 192
    for r in (v, t):   # val predates (e)'s presence/no_track/scope fields (it stays byte-identical: the preregistration pins its sha)
        c, s = fa.m1_criteria(r["raw"]["main"], r["raw"]["empty_ring"]), r["criteria"]; assert c.keys() == s.keys() and all(c[k] == s[k] for k in c if k != "e") and s["e"].items() <= c["e"].items()
    assert t["criteria"]["all_passed"] and R["camera_turn"]["tune"] == {k: pre["choice"][k] for k in ("percept_filter_s", "face_deadband")}
    assert t["criteria"]["e"]["presence"] == 0 and t["criteria"]["e"]["no_track"] == 1 and t["pre_test_smoke_run"]["clips"] == pre["test"]["clips"]   # (e)'s scope and the dry run, disclosed
    assert max(R["camera_turn"]["media_bytes"].values()) < 5_000_000 and all((ROOT.parent / "media" / f).stat().st_size == n for f, n in R["camera_turn"]["media_bytes"].items())

def test_m1_guard_hashes_the_whole_package():
    """The test run's code pin covers every module a bout imports (policy.py, perturb.py, sensors.py were missing from the hand list)."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("face_ab", ROOT.parent / "scripts" / "face_ab.py"); fa = importlib.util.module_from_spec(sp); sp.loader.exec_module(fa)
    f = fa.m1_files(); assert {"r1gym/policy.py", "r1gym/perturb.py", "r1gym/sensors.py", "r1gym/perception/__init__.py", "configs/r1_camera.yaml", "scripts/face_ab.py"} <= set(f)
    assert all("\\" not in p for p in f) and fa.M1_SPLITS["smoke"][0] in fa.TRAIN_CLIPS and not set(fa.M1_SPLITS["smoke"]) & set(fa.VAL_CLIPS + fa.TEST_CLIPS)
