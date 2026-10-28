"""Aim operators on the shipped tracker (r1gym/aim.py): off by default and bit-identical when idle, warps only inside strike
windows and smoothly, IK reaches what is reachable, the stand-off stays inside the tracker's drift, per-env state under threads."""
import numpy as np, pytest
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec, load_cfg
from r1gym.env import R1HitEnv
from r1gym.opponent import FightReplay
from r1gym.aim import Aim, ARM, StrikeIntent, merged

@pytest.fixture(scope="module")
def world():
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller("step=results/policy_mjlab/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    return c, cfg, build_spec(opponent=True, cfg=cfg).compile()

def bout(world, mode, clip="kfab_201", dist=0.45, t0=1.0, steps=200, rep=None):
    c, cfg, m = world; cfg = {**cfg, "opponent_aim": mode}
    e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=steps * 0.02 + 1, max_impulse=0, train=False); e.opp = rep or FightReplay(m, clip, distance=dist, cfg=cfg); e.opp_t0 = t0
    o = e.reset(hits=[], delay=1); Q, I = [], []
    for _ in range(steps):
        o, r, d, info = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy()); I.append(info.get("strike_intent"))
    return np.array(Q), e, I

def test_default_is_off_and_shadow_is_bit_identical(world):
    """configs: opponent_aim none (no Aim at all). 'shadow' logs intents but leaves the reference alone: the same bout, bit for bit."""
    assert load_cfg()["opponent_aim"] == "none"
    q0, e0, i0 = bout(world, "none"); q1, e1, i1 = bout(world, "shadow")
    assert "aim" not in e0.opp_state and np.abs(q0 - q1).max() == 0 and len(e1.opp_state["aim"].intents) > 2
    assert all(x is None for x in i0) and any(x is not None for x in i1) and {"limb", "zone", "t_on", "t_apex"} <= set(next(x for x in i1 if x))

def test_warp_is_zero_outside_windows_and_c1():
    """The blend weight is 0 outside [t_on - 0.15 s, end + 0.1 s] and C1 (smoothstep ramps, a hold at the apex); in a bout the
    warp is exactly 0 at every step no intent's weight covers."""
    from r1gym.aim import AIM
    a = Aim.__new__(Aim); a.p = merged(None); it = StrikeIntent("l_hand", "head", "x", 0, 1.0, 1.2, 1.22, np.zeros(3), 0.5)
    t = np.arange(0.5, 2.0, 1e-4); w = np.array([a.weight(it, x) for x in t]); dw = np.gradient(w, t)
    assert w[t <= 0.85].max() == 0 and w[t >= 1.2 + 0.1 + 0.1].max() == 0 and w.max() == 1.0
    assert np.abs(np.diff(w)).max() < 1e-3 and np.abs(np.diff(dw)).max() < 0.05 and abs(dw[0]) + abs(dw[-1]) == 0   # continuous value and slope

def test_armwarp_bout_warps_only_inside_windows(world):
    q, e, _ = bout(world, "armwarp", steps=250); a = e.opp_state["aim"]
    on = [(now, wmax) for now, wmax, _ in a.trace]; covered = lambda now: any(a.weight(it, now) > 0 for it in a.intents if not it.gated)
    assert any(wmax > 0.05 for _, wmax in on) and all(wmax == 0 for now, wmax in on if not covered(now))
    assert all(wmax <= 0.8 + 1e-9 for _, wmax in on)   # max_dq

def test_ik_reaches_reachable_targets(world):
    """From a strike's reference arm, a palm target made by moving the arm's 5 joints up to 0.3 rad is met within 1 cm."""
    c, cfg, m = world; rep = FightReplay(m, "kfab_201", distance=0.5, cfg=cfg); tr = rep.tracker; a = Aim(m, rep, "armwarp", cfg.get("aim")); rng = np.random.default_rng(0); res = []
    for s in [s for s in rep.strikes if s.hand]:
        rq = tr.ref_q[tr.frame(rep.clip, s.apex)].copy(); ix = ARM[s.limb]; qt = rq.copy(); qt[ix] = np.clip(qt[ix] + rng.uniform(-0.3, 0.3, 5), a.lo[ix], a.hi[ix])
        res.append(a.ik(rq, s.limb, a._palm_b(qt, s.limb)[0])[1])
    assert len(res) >= 15 and max(res) < 0.01, max(res)

def test_standoff_stays_inside_the_trained_drift(world):
    """At 0.65 m the stand-off closes in for strikes that fall short, never below 0.35 m nor beyond the bout distance, and the
    steered spot never moves faster than 0.3 m/s."""
    q, e, _ = bout(world, "standoff", dist=0.65, steps=300); a = e.opp_state["aim"]; dist = [x for _, _, x in a.trace]
    assert all(0.35 - 1e-9 <= x <= 0.65 + 1e-9 for x in dist) and min(dist) < 0.6
    c, cfg, m = world; e2 = R1HitEnv(seed=0, cfg={**cfg, "opponent_aim": "standoff"}, model=m, episode_s=7, max_impulse=0, train=False)
    e2.opp = FightReplay(m, "kfab_201", distance=0.65, cfg=cfg); e2.opp_t0 = 1.0; o = e2.reset(hits=[], delay=1); sp = []
    for _ in range(300):
        o, *_ = e2.step(c.act(o[None])[0]); sp.append(e2.opp_state["spot"].copy())
    assert np.linalg.norm(np.diff(sp, axis=0), axis=1).max() / 0.02 <= 0.3 + 1e-6

def test_envs_sharing_replay_and_tracker_keep_their_own_aim_state(world):
    """Eight envs on one model, one FightReplay and the class-cached Tracker, stepped by 8 threads, end exactly where each
    ends when run alone: all aim state is per env."""
    c, cfg, m = world; rep = FightReplay(m, "kfab_201", distance=0.45, cfg=cfg); t0s = [0.5 + 0.37 * i for i in range(8)]
    alone = [bout(world, "armwarp", t0=t, steps=120, rep=rep)[0] for t in t0s]
    with ThreadPoolExecutor(8) as ex: together = list(ex.map(lambda t: bout(world, "armwarp", t0=t, steps=120, rep=rep)[0], t0s))
    assert all(np.abs(a - b).max() == 0 for a, b in zip(alone, together)) and np.abs(alone[0] - alone[1]).max() > 0

def test_select_splices_inside_the_clip(world):
    """'select' moves the clip clock to the clip's head-level strikes, cross-fading the reference; the bout clip is unchanged."""
    q, e, _ = bout(world, "select", steps=250); a = e.opp_state["aim"]
    assert a.splices > 0 and a.clip == "kfab_201" and all(s.apex_m[2] >= 1.02 for s in a.high) and len(a.intents) > 0

def test_mannequin_is_welded_at_home():
    """build_spec(mannequin=True): the R1 can't fall or be pushed off its spot (aim_check's target for the aim alone)."""
    import mujoco
    from r1gym.model import R1Robot
    from r1gym.perturb import make_hit, HitApplier
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg, mannequin=True).compile(); d = mujoco.MjData(m); r = R1Robot(m, cfg); r.reset(m, d)
    oq = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]; d.qpos[oq:oq + 3] = [5, 0, 0.74]; mujoco.mj_forward(m, d); ha, mx = HitApplier(m), 0.0
    h = make_hit("strike", "head", 10.0, 0.1, 0.0, np.random.default_rng(0))
    for _ in range(400): ha.apply(d, [h], d.time); mujoco.mj_step(m, d); mx = max(mx, float(np.linalg.norm(d.xpos[r.pel] - [0, 0, 0.74])))
    assert m.neq == 1 and mx < 0.03 and d.xmat[r.pel][8] > 0.999 and not r.fallen(d)
