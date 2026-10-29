"""Sparring M2 run 2 (docs/SPARRING_M1M2.md section 9, results/sparring/prereg_m2_run2.json) in gpu/sparring_task.py, on CPU:
run 1 stays bit for bit (AttackSchedule and HitAccount with default p against tests/fixtures/sparring_task_576727a.py, a frozen
copy of the file run 1 trained with); SCHED_R2's commit latch (Delta follows the defender until max(onset, apex - 0.15 s));
the reach gate (a strike thrown at a defender out of reach pays no clearance); HIT_R2 (block pay only for intercepts, the
blocked share unchanged); the ring terms (bonus, ring_out, ring_home); the R1-Spar-Block2-* configs on a minimal fake mjlab
(the run-1 configs untouched); widen2 (critic 98 -> 138, outputs unchanged; skipped without model_7496.pt); gate() reading
iterations 20-200 of a training log."""
import sys, json, math, types, importlib.util, dataclasses, re
from pathlib import Path
import numpy as np, pytest
torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
WARM_DIR = ROOT / "results" / "policy_mjlab_guard_footwork_r"; FROZEN = Path(__file__).resolve().parent / "fixtures" / "sparring_task_576727a.py"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
T = _load("sparring_task", ROOT / "gpu" / "sparring_task.py")
O = _load("sparring_task_576727a", FROZEN)   # run 1's file (git show 576727a:gpu/sparring_task.py), never edited

@pytest.fixture(scope="module")
def lib(): return T.load_library()

def frame(pel, yaw):
    """Torso origin 3 cm ahead of the pelvis at 0.79 m and the yaw rotation (N, 3, 3)."""
    n = len(yaw); c, s = torch.cos(yaw), torch.sin(yaw); R = torch.zeros(n, 3, 3); R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    return torch.cat([pel + 0.03 * torch.stack([c, s], -1), torch.full((n, 1), 0.79)], -1), R

def drive(mod, lib, p, steps=500, n=128, seed=3, v=(0.25, 0.1), place=None):
    """A scripted moving defender (pelvis drifting at v, yaw wobbling) against mod.AttackSchedule(p). place(s, pel) -> pel lets
    a test put the pelvis relative to the live spot before each step. Returns per-step outputs, the schedule and per-step
    records (tau before, latched before, strike_id before, delta before, delta after, latched after)."""
    s = mod.AttackSchedule(lib, n, "cpu", 0.02, gen=torch.Generator().manual_seed(seed), p=p); pel = torch.zeros(n, 2); yaw = torch.zeros(n)
    s.reset(torch.arange(n), pel.clone(), yaw.clone()); outs, rec = [], []
    for k in range(steps):
        t = torch.full((n,), k * 0.02); yaw = yaw + 0.01 * math.sin(k * 0.05); pel = pel + 0.02 * torch.tensor(v) * math.cos(k * 0.01)
        if place is not None: pel = place(s, pel)
        xT, R = frame(pel, yaw); b = (s.latched.clone(), s.strike_id.clone(), s.delta.clone())
        g, e, h, y = s.step(t, pel, yaw, xT, R); outs.append(torch.cat([g.flatten(1), e.flatten(1), h, y[:, None]], 1).clone())
        rec.append(dict(lat0=b[0], sid0=b[1], d0=b[2], d1=s.delta.clone(), lat1=s.latched.clone(), tau=s.tau.clone(), pel=pel.clone()))
    return torch.stack(outs), s, rec

# ---- 1. run-1 parity
def test_run1_schedule_bit_identical(lib):
    """Default p (and AttackCommand's p={"delta": cfg.delta} with an empty sched): every target, spot and latch equals run 1's file."""
    for p in (None, {"delta": T.ATTACK["delta"]}):
        a, sa, _ = drive(T, lib, p); b, sb, _ = drive(O, lib, p)
        assert torch.equal(a, b) and torch.equal(sa.S, sb.S) and torch.equal(sa.latched, sb.latched) and torch.equal(sa.delta, sb.delta)
        assert not bool(sa.reach_ok.any())   # no reach gate: never set

def test_run1_hit_account_bit_identical():
    """HitAccount with default p (and an empty hit dict) counts exactly as run 1's on random contact sequences."""
    g = torch.Generator().manual_seed(7); zi = T.zone_index([l + "_shell" for l in T.LINKS]); n = 48
    a, a2, b = T.HitAccount(n, "cpu", zi), T.HitAccount(n, "cpu", zi, p={}), O.HitAccount(n, "cpu", zi)
    sid = torch.zeros(n, 2, dtype=torch.long); won, woff = torch.zeros(n, 2), torch.zeros(n, 2); steps = []
    for k in range(400):
        t = (k + 1) * 0.02
        if k % 20 == 0: sid += 1; won[:] = t + 0.02; woff[:] = t + 0.3
        F = (torch.rand(n, 2, 5, 4, generator=g) < 0.04).float() * 40.0 * torch.rand(n, 2, 5, 4, generator=g); vc = torch.rand(n, 2, 5, 4, generator=g) * 3
        outs = []
        for h in (a, a2, b): h.process(F, torch.full((n,), t), sid.clone(), won.clone(), woff.clone(), vc); outs.append({k_: v.clone() for k_, v in h.step.items()})
        steps.append(outs)
    for h in (a, a2, b): h.flush(torch.full((n,), 8.1))
    assert all(torch.equal(x[0][k_], x[2][k_]) and torch.equal(x[1][k_], x[2][k_]) for x in steps for k_ in x[2])
    assert all(torch.equal(a.ep[k_], b.ep[k_]) and torch.equal(a2.ep[k_], b.ep[k_]) for k_ in b.ep) and float(b.ep["block_first"].sum()) > 0

def test_run1_clearance_formula():
    """clearance_value without a reach mask is run 1's sum of clip((d - 0.10) / 0.15) over the windows."""
    g = torch.Generator().manual_seed(1); d = torch.rand(64, 2, generator=g) * 0.4; win = torch.rand(64, 2, generator=g) < 0.5
    assert torch.equal(T.clearance_value(d, win), (((d - 0.10) / 0.15).clamp(0.0, 1.0) * win).sum(1))
    assert T.reach_mask(types.SimpleNamespace(p=dict(T.SCHED)), None) is None

# ---- 2. commit latch
def run_latch(lib, p, v=(0.6, 0.0)):
    """Per newly latched strike: (tau at the latch step, the threshold, tau one step earlier, onset); and how often Delta moved
    after the onset but before the latch, and after the latch (same strike)."""
    s = T.AttackSchedule(lib, 256, "cpu", 0.02, gen=torch.Generator().manual_seed(5), p=p); n = s.n; pel = torch.zeros(n, 2); yaw = torch.zeros(n)
    s.reset(torch.arange(n), pel.clone(), yaw.clone()); lat, moved_pre, moved_post, cs = [], 0, 0, p.get("commit_s", 0.0)
    for k in range(1500):
        t = torch.full((n,), k * 0.02); pel = pel + 0.02 * torch.tensor(v) * math.cos(k * 0.01); xT, R = frame(pel, yaw)
        l0, sid0, d0 = s.latched.clone(), s.strike_id.clone(), s.delta.clone()
        s.step(t, pel, yaw, xT, R)
        sg = s.seg[s.seg_id]; on, ap = sg[..., 2].float(), sg[..., 3].float(); th = torch.maximum(on, ap - cs * s.fps * s.sig) if cs > 0 else on
        same = (s.phase == 1) & (s.strike_id == sid0); changed = (s.delta - d0).norm(dim=-1) > 1e-7
        moved_pre += int((same & ~s.latched & (s.tau > on + 1) & changed).sum()); moved_post += int((same & l0 & changed).sum())
        new = s.latched & ~l0 & (s.phase == 1)
        if bool(new.any()): lat.append(torch.stack([s.tau[new], th[new], (s.tau - s.fps * s.sig * 0.02)[new], on[new]], -1))
    return torch.cat(lat), moved_pre, moved_post

def test_commit_latch(lib):
    """commit_s 0.15: Delta keeps following a moving defender after the onset until tau >= apex - 0.15 s x fps x sigma (the
    first step past it latches), then never moves; commit_s 0: the latch is the onset's, as run 1."""
    L, pre, post = run_latch(lib, dict(T.SCHED_R2))
    assert len(L) > 200 and bool((L[:, 0] >= L[:, 1] - 1e-4).all()) and bool((L[:, 2] < L[:, 1] + 1e-4).all())
    assert bool((L[:, 1] > L[:, 3] + 0.5).any()) and pre > 0 and post == 0
    L0, pre0, post0 = run_latch(lib, {"commit_s": 0.0})
    assert len(L0) > 200 and bool((L0[:, 0] >= L0[:, 3] - 1e-4).all()) and bool((L0[:, 2] < L0[:, 3] + 1e-4).all()) and post0 == 0

# ---- 3. reach gate
def placed(extra):
    """Put the pelvis `extra` beyond the stand-off D from the live spot (along the current spot -> pelvis direction)."""
    def f(s, pel):
        u = torch.nn.functional.normalize(pel - s.S, dim=-1); return s.S + (s.D + extra)[:, None] * u
    return f

def test_reach_gate(lib):
    """A defender standing at the stand-off is in reach at every latch; one 0.2 m beyond D is out of reach at every latch, and
    its latched head strikes pay no clearance in their windows (run 1, no gate: they would)."""
    for extra, want in ((0.0, True), (0.2, False)):
        s = T.AttackSchedule(lib, 64, "cpu", 0.02, gen=torch.Generator().manual_seed(3), p=dict(T.SCHED_R2)); n = s.n; pel = torch.zeros(n, 2); yaw = torch.zeros(n)
        s.reset(torch.arange(n), pel.clone(), yaw.clone()); seen, masked = [], 0
        for k in range(600):
            t = torch.full((n,), k * 0.02); pel = placed(extra)(s, pel); xT, R = frame(pel, yaw); l0 = s.latched.clone()
            s.step(t, pel, yaw, xT, R); new = s.latched & ~l0 & (s.phase == 1); seen.append(s.reach_ok[new])
            m = T.reach_mask(s, pel); lat = (s.phase == 1) & s.latched
            assert torch.equal(m[lat], s.reach_ok[lat])
            if not want:
                win = lat & (s.zone == 0); c = T.clearance_value(torch.full((n, 2), 0.4), win, m); masked += int(win.sum())
                assert float(c.abs().max()) == 0.0 and (not bool(win.any()) or float(T.clearance_value(torch.full((n, 2), 0.4), win).max()) > 0)
        seen = torch.cat(seen); assert len(seen) > 50 and bool((seen == want).all()) and (want or masked > 50)
    s0 = T.AttackSchedule(lib, 8, "cpu", 0.02, p={}); assert T.reach_mask(s0, torch.zeros(8, 2)) is None

def test_reach_mask_before_latch_uses_live_excess(lib):
    """Before the latch the gate reads the live spacing excess (pelvis - S distance - D <= reach_tol)."""
    s = T.AttackSchedule(lib, 3, "cpu", 0.02, p=dict(T.SCHED_R2)); s.S = torch.zeros(3, 2); s.D = torch.full((3,), 0.5); s.latched[:] = False
    pel = torch.tensor([[0.5, 0.0], [0.549, 0.0], [0.56, 0.0]]); m = T.reach_mask(s, pel)
    assert m.tolist() == [[True, True], [True, True], [False, False]]

# ---- 4. block pay by apex
def run_hits(p, opens, heads=None):
    """One window (hand 0) [0.03, 0.30] s, apex 0.15 s; env i's left fist first touches the forearm (block zone) at opens[i] s for
    a substep; nothing lands, unless heads[i] (s) is set: then the same fist lands a head hit (200 N for 0.02 s, 4 N*s) from
    heads[i]. Returns (spar_block pay per env, ep blocked per env)."""
    zi = T.zone_index([l + "_shell" for l in T.LINKS]); n = len(opens); h = T.HitAccount(n, "cpu", zi, p=p); pay = torch.zeros(n)
    sid = torch.zeros(n, 2, dtype=torch.long); won, woff = torch.zeros(n, 2), torch.zeros(n, 2)
    for k in range(30):
        t_end = (k + 1) * 0.02; F = torch.zeros(n, 2, 5, 4)
        if k == 0: sid[:, 0] += 1; won[:, 0] = 0.03; woff[:, 0] = 0.30
        for i, to in enumerate(opens):
            for j in range(4):
                if abs(t_end - (3 - j) * 0.005 - to) < 1e-6: F[i, 0, 3, j] = 30.0
                th = (heads or {}).get(i)
                if th is not None and -1e-6 < t_end - (3 - j) * 0.005 - th < 0.02 - 1e-6: F[i, 0, 0, j] = 200.0
        h.process(F, torch.full((n,), t_end), sid.clone(), won.clone(), woff.clone(), torch.full((n, 2, 5, 4), 2.0)); pay += h.step["block"]
    h.flush(torch.full((n,), 0.7)); pay += h.step["block"]; return pay.tolist(), h.ep["blocked"].tolist()

def test_block_by_apex():
    """HIT_R2: an arm contact opening at apex + 0.03 s pays spar_block, at apex + 0.08 s it does not (apex + 0.05 s exactly
    does); ep 'blocked' counts all three either way, and run 1 pays all three."""
    opens = (0.18, 0.23, 0.20)
    p2, b2 = run_hits(dict(T.HIT_R2), opens); p1, b1 = run_hits(None, opens)
    assert p2 == [1.0, 0.0, 1.0] and p1 == [1.0, 1.0, 1.0] and b1 == b2 == [1.0, 1.0, 1.0]

def test_graze_then_landed_hit_pays_nothing():
    """Exploit review blocker 1: under HIT_R2 a forearm touch at 0.12 s followed by a head hit landing from 0.14 s pays no
    spar_block (the strike was not stopped); the touch alone still pays. ep 'blocked' / 'block_first' keep run 1's
    definition (arm first: blocked), and run 1 still pays the graze."""
    opens = (0.12, 0.12)
    p2, b2 = run_hits(dict(T.HIT_R2), opens, heads={1: 0.14}); p1, b1 = run_hits(None, opens, heads={1: 0.14})
    assert p2 == [1.0, 0.0] and p1 == [1.0, 1.0] and b1 == b2 == [1.0, 1.0]

# ---- 5. ring terms
def ring_env(xy, spot, yaw=0.0):
    xy = torch.tensor(xy, dtype=torch.float32); n = len(xy)
    data = types.SimpleNamespace(root_link_pos_w=torch.cat([xy, torch.full((n, 1), 0.78)], -1), heading_w=torch.full((n,), yaw))
    tw = types.SimpleNamespace(robot=types.SimpleNamespace(data=data), spot=torch.tensor(spot, dtype=torch.float32).expand(n, 2).clone())
    return types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name: tw), num_envs=n, device="cpu")

def test_ring_terms():
    env = ring_env([[0.2, 0.0], [0.0, 0.625], [-1.1, 0.0], [1.0, 0.0], [0.0, -1.01]], [0.0, 0.0])
    b = T.spar_ring(env); assert torch.allclose(b, torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0]), atol=1e-6) and T.RING["weight"] > 0 and float(b.min()) >= 0
    assert T.spar_ring_out(env).tolist() == [False, False, True, False, True] and torch.allclose(T.spar_ring_r(env), torch.tensor([0.2, 0.625, 1.1, 1.0, 1.01]))
    h = T.ring_home(ring_env([[1.0, 2.0]], [1.0, 3.0], yaw=math.pi / 2)); assert torch.allclose(h, torch.tensor([[1.0, 0.0]]), atol=1e-6)   # home straight ahead
    h = T.ring_home(ring_env([[1.0, 2.0]], [0.0, 2.0], yaw=math.pi / 2)); assert torch.allclose(h, torch.tensor([[0.0, 1.0]]), atol=1e-6)   # home to the left
    assert T.WEIGHTS["A2"] == dict(spar_head_impulse=-20.0, spar_head_clean=-40.0, spar_torso_impulse=-5.0, spar_block=40.0, spar_clearance=0.5, spar_ring=2.5)

# ---- 6. task configs on a minimal fake mjlab
@pytest.fixture(scope="module")
def fake_mjlab():
    """Just enough of mjlab for the cfg builders (dataclass cfgs that store their arguments); removed again afterwards."""
    from dataclasses import dataclass, field
    class NS:
        def __init__(self, *a, **k): self.args = a; self.__dict__.update(k)
    @dataclass(kw_only=True)
    class TermCfg: func: object = None; params: dict = field(default_factory=dict)
    @dataclass(kw_only=True)
    class ObservationTermCfg(TermCfg): delay_min_lag: int = 0; delay_max_lag: int = 0
    @dataclass(kw_only=True)
    class ObservationGroupCfg: terms: dict = field(default_factory=dict)
    @dataclass(kw_only=True)
    class RewardTermCfg(TermCfg): weight: float = 0.0
    @dataclass(kw_only=True)
    class TerminationTermCfg(TermCfg): time_out: bool = False
    @dataclass(kw_only=True)
    class MetricsTermCfg(TermCfg): pass
    @dataclass
    class SceneEntityCfg: name: str; joint_names: object = None; body_names: object = None
    class CommandTerm: pass
    @dataclass(kw_only=True)
    class CommandTermCfg: resampling_time_range: tuple = (1.0, 1.0); debug_vis: bool = False
    class UniformVelocityCommand(CommandTerm): pass
    @dataclass(kw_only=True)
    class UniformVelocityCommandCfg(CommandTermCfg): entity_name: str = "robot"; rel_standing_envs: float = 0.0; ranges: object = None
    class Entity: pass
    @dataclass(kw_only=True)
    class EntityCfg:
        InitialStateCfg = NS
        init_state: object = None; spec_fn: object = None; articulation: object = None
    mods = {"mjlab": dict(), "mjlab.managers": dict(), "mjlab.managers.observation_manager": dict(ObservationTermCfg=ObservationTermCfg, ObservationGroupCfg=ObservationGroupCfg),
            "mjlab.managers.reward_manager": dict(RewardTermCfg=RewardTermCfg), "mjlab.managers.termination_manager": dict(TerminationTermCfg=TerminationTermCfg),
            "mjlab.managers.metrics_manager": dict(MetricsTermCfg=MetricsTermCfg), "mjlab.managers.scene_entity_config": dict(SceneEntityCfg=SceneEntityCfg),
            "mjlab.managers.command_manager": dict(CommandTerm=CommandTerm, CommandTermCfg=CommandTermCfg), "mjlab.tasks": dict(), "mjlab.tasks.velocity": dict(),
            "mjlab.tasks.velocity.mdp": dict(UniformVelocityCommand=UniformVelocityCommand, UniformVelocityCommandCfg=UniformVelocityCommandCfg),
            "mjlab.entity": dict(Entity=Entity, EntityCfg=EntityCfg, EntityArticulationInfoCfg=NS), "mjlab.actuator": dict(BuiltinPositionActuatorCfg=NS),
            "mjlab.sensor": dict(ContactMatch=NS, ContactSensorCfg=NS), "mjlab.envs": dict(), "mjlab.envs.mdp": dict(), "mjlab.envs.mdp.terminations": dict(nan_detection=lambda env: None)}
    old = {k: sys.modules.get(k) for k in mods}
    for k, a in mods.items(): m = types.ModuleType(k); m.__dict__.update(a); sys.modules[k] = m
    saved_C = dict(T._C); T._C.clear()
    try: yield types.SimpleNamespace(ObservationTermCfg=ObservationTermCfg, ObservationGroupCfg=ObservationGroupCfg, RewardTermCfg=RewardTermCfg, SceneEntityCfg=SceneEntityCfg,
                                     UniformVelocityCommandCfg=UniformVelocityCommandCfg, NS=NS)
    finally:
        T._C.clear(); T._C.update(saved_C)
        for k, v in old.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v

ACT = ("base_ang_vel", "projected_gravity", "command", "phase", "joint_pos", "joint_vel", "actions")
def make_factory(M):
    def make(play=False):
        actor = {k: M.ObservationTermCfg(func=k, delay_max_lag=0 if play else 2) for k in ACT}
        critic = {**actor, **{k: M.ObservationTermCfg(func=k) for k in ("base_lin_vel", "foot_height", "foot_air_time", "foot_contact", "foot_contact_forces")}}
        walk = {r".*hip_pitch.*": 0.5, r".*waist_roll.*": 0.1, r".*elbow.*": 0.1}
        return types.SimpleNamespace(scene=types.SimpleNamespace(entities={"robot": types.SimpleNamespace(spec_fn=lambda: None)}, sensors=("feet", "self"), spec_fn=None),
            commands={"twist": M.UniformVelocityCommandCfg(resampling_time_range=(3.0, 8.0), rel_standing_envs=0.3)},
            observations={"actor": M.ObservationGroupCfg(terms=actor), "critic": M.ObservationGroupCfg(terms=critic)},
            rewards={"pose": M.RewardTermCfg(func="posture", weight=1.0, params={"std_standing": {".*": 0.05}, "std_walking": dict(walk), "std_running": dict(walk)}),
                     "stand_still": M.RewardTermCfg(func="stand_still", weight=-1.0, params={"command_name": "twist", "command_threshold": 0.1, "asset_cfg": M.SceneEntityCfg("robot", joint_names=".*")})},
            events={"push_robot": object(), "shove": object(), "foot_friction": object()}, terminations={"time_out": 0, "fell_over": 1})
    return make

def test_block2_configs(fake_mjlab):
    M = fake_mjlab; make = make_factory(M)
    for blind in (False, True):
        c = T._block2(make, blind=blind)(); a = list(c.observations["actor"].terms); cr = list(c.observations["critic"].terms)
        assert a == list(ACT) + (["opp_keypoints_blind"] if blind else ["opp_keypoints"])   # actor 83 + 28 = 111, unchanged from run 1
        assert cr[-3:] == ["opp_keypoints_truth", "strike_phase", "ring_home"] and len(cr) == 15 and c.observations["critic"].terms["ring_home"].func is T.ring_home
        tw = c.commands["twist"]; assert type(tw).__name__ == "OuterLoopTwistCommandCfg" and tw.rel_outer_envs == 1.0 and tw.ring_metric is True
        assert c.commands["attack"].sched == T.SCHED_R2 and c.commands["attack"].hit == T.HIT_R2 and list(c.commands) == ["twist", "attack"]
        assert {k: c.rewards[k].weight for k in T.WEIGHTS["A2"]} == T.WEIGHTS["A2"] and c.rewards["spar_ring"].func is T.spar_ring
        assert c.rewards["spar_ring"].params == {"command_name": "twist", "free": 0.25, "out": 1.0} and c.rewards["spar_block"].params == {"command_name": "attack"}
        ro = c.terminations["ring_out"]; assert ro.func is T.spar_ring_out and ro.params == {"command_name": "twist", "radius": 1.0} and ro.time_out is False
        assert c.rewards["stand_still"].params["asset_cfg"].joint_names == T.STAND_STILL_JOINTS and c.rewards["stand_still"].params["command_threshold"] == 0.1
        assert list(c.metrics) == ["spar_hits", "spar_ring_r", "spar_arm_idle_touch"] and "push_robot" in c.events
        cp = T._block2(make, blind=blind)(play=True); assert "ring_out" not in cp.terminations and "spar_ring" in cp.rewards and "nan" in cp.terminations
    # run 1's tasks, built after (and from the same make): untouched
    for w, blind in (("A", False), ("B", False), ("A", True)):
        c = T._block(make, w, blind)(); tw = c.commands["twist"]
        assert tw.rel_outer_envs == 0.5 and tw.ring_metric is False and c.commands["attack"].sched == {} and c.commands["attack"].hit == {}
        assert "spar_ring" not in c.rewards and "ring_out" not in c.terminations and "ring_home" not in c.observations["critic"].terms
        assert {k: c.rewards[k].weight for k in T.WEIGHTS["A"]} == T.WEIGHTS[w] and c.rewards["stand_still"].params["asset_cfg"].joint_names == ".*"
        assert list(c.metrics) == ["spar_hits"] and len(c.observations["critic"].terms) == 14
    ck = T._attack(make, outer=1.0, check=True, sched=T.SCHED_R2)(); assert ck.commands["attack"].sched == T.SCHED_R2 and "push_robot" not in ck.events
    assert {"R1-Spar-Block2-A", "R1-Spar-Block2-Blind", "R1-Spar-Block-A", "R1-Spar-Block-B", "R1-Spar-Block-Blind", "R1-Spar-Block-Blind-B", "R1-Spar-Attack"} == set(T.TASKS)

def test_stand_still_joints():
    from r1gym.model import CTRL_JOINTS
    hit = [j for j in CTRL_JOINTS if any(re.fullmatch(p, j) for p in T.STAND_STILL_JOINTS)]
    assert len(hit) == 14 and not any(re.search("shoulder|elbow|wrist", j) for j in hit)   # 12 leg + 2 waist joints

# ---- 7. widen2
def test_widen2(tmp_path):
    src = WARM_DIR / "model_7496.pt"
    if not src.exists(): pytest.skip("model_7496.pt absent")
    dst = tmp_path / "w2" / "model_7496.pt"; T.widen(str(src), str(dst), critic_extra=T.HOME_DIM)
    c0, c2 = (torch.load(p, weights_only=False, map_location="cpu") for p in (src, dst)); cs = c2["critic_state_dict"]
    assert c2["actor_state_dict"]["mlp.0.weight"].shape == (512, 111) and cs["mlp.0.weight"].shape == (512, 138) and float(cs["mlp.0.weight"][:, 98:].abs().max()) == 0
    assert c2["optimizer_state_dict"]["state"][9]["exp_avg"].shape == (512, 138) and cs["obs_normalizer._mean"][0, 136:].tolist() == [0.0, 0.0] and cs["obs_normalizer._var"][0, 136:].tolist() == [0.25, 0.25]
    def critic(sd, x):
        sd = {k: v.double() if v.is_floating_point() else v for k, v in sd.items()}; z = (x.double() - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
        for i in (0, 2, 4, 6): z = z @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]; z = torch.nn.functional.elu(z) if i < 6 else z
        return z
    g = torch.Generator().manual_seed(0); xc = torch.randn(512, 98, generator=g) * 2
    assert float((critic(c0["critic_state_dict"], xc) - critic(cs, torch.cat([xc, torch.randn(512, 40, generator=g) * 5], 1))).abs().max()) < 1e-9
    x, xn = torch.randn(512, 83, generator=g) * 2, torch.randn(512, 28, generator=g) * 5
    assert float((T.load_actor(str(src))(x) - T.load_actor(str(dst))(torch.cat([x, xn], 1))).abs().max()) < 1e-4
    assert T.widen_norm(True, 2)[1][-2:] == [0.5, 0.5] and T.widen_norm(True) == T.widen_norm(True, 0) and len(T.widen_norm(True, 2)[0]) == 40

# ---- gate 2 from A2's own log
def test_gate_reads_iterations_20_to_200(tmp_path):
    """Run 2's gate 2: mean iteration time over iterations 20-200 of the 1500-iteration A2 log (later ones ignored), <= 2.2 s,
    no nan termination anywhere; 3072 envs is no deviation when preregistered."""
    log = tmp_path / "train_A2.log"; lines = []
    for i in range(1500): lines += [f"Learning iteration {7496 + i}/8996", "  Episode_Termination/nan: 0.0000", f"   Iteration time: {5.0 if i < 19 else (2.1 if i < 200 else 9.0):.2f}s", "  Collection time: 1.50s"]
    log.write_text("\n".join(lines)); m1 = tmp_path / "g.json"; m1.write_text(json.dumps({"m1": {"pass": True}}))
    j = T.gate(str(m1), str(log), "capsule", 3072, max_iter_s=2.2, last=200, prereg_envs=3072); s = j["smoke"][-1]
    assert j["pass"] and s["mean_iter_s"] == 2.1 and s["iterations_gated"] == [20, 200] and not j.get("deviations")
    assert not T.gate(str(m1), str(log), "capsule", 3072, max_iter_s=2.0, last=200, prereg_envs=3072)["smoke"][-1]["pass_speed"]
    log.write_text("\n".join(lines + ["  Episode_Termination/nan: 0.0100"])); assert not T.gate(str(m1), str(log), "capsule", 3072, max_iter_s=2.2, last=200, prereg_envs=3072)["pass"]
