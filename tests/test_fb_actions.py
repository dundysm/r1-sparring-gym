"""Full-body M2 v2 action term and composition (M2V2_PLAN_v3 M4 G0-7, G0-10; gpu/fb_actions.py): applied = a(t-1) when delayed
and 0 at the first delayed step, targets = R1HitEnv.step's arithmetic (clip, cast, no EMA), act_delay read once per episode, the
term reads no observation (m11), the M3s rewiring and the legs/arms routing = P1's M3sHybrid (FB_P1_EVAL=<p1_eval.py> also runs
it against P1's own class; skipped without it). The recorded-bout gates run in scripts/fb_m4_check.py."""
import os, sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_actions as AC, fb_events as EV, fb_m4_check as C
from r1gym.model import load_cfg, action_scale, HOME
from r1gym import CONFIGS

def term(n, delays):
    cfg = load_cfg(CONFIGS / "r1_mjlab.yaml"); lat = EV.Latency(n); lat.act_delay[:] = torch.tensor(delays)
    a = AC.HybridDelayedAction(n, lat, np.array(HOME, float), action_scale(cfg), float(cfg["action_clip"])); a.reset(torch.arange(n)); return a, lat, cfg

def test_applied_is_the_previous_action_when_delayed():
    a, lat, cfg = term(2, [True, False]); rng = np.random.default_rng(0); A = rng.normal(size=(30, 2, 24)).astype(np.float32); home, sc = np.array(HOME, float), action_scale(cfg)
    for t in range(30):
        tg = a.process(torch.from_numpy(A[t])).numpy()
        want0 = np.zeros(24, np.float32) if t == 0 else A[t - 1, 0]
        assert np.array_equal(a.applied[0].numpy(), want0) and np.array_equal(a.applied[1].numpy(), A[t, 1])
        for j in range(2):   # R1HitEnv.step (mjlab contract): clip, float32, filt = 0 x filt + 1 x a, target = home + scale x filt
            filt = 0.0 * np.zeros(24) + (1 - 0.0) * a.applied[j].numpy(); assert np.array_equal(tg[j], home + sc * filt)
    a.reset(torch.tensor([0])); assert not a.a_prev[0].any() and not a.applied[0].any() and a.a_prev[1].any()
    a.process(torch.ones(2, 24)); assert not a.applied[0].any()   # the first delayed step of the new episode applies 0

def test_act_delay_is_read_at_reset_only_and_clip_cast():
    a, lat, _ = term(2, [False, False]); a.clip = 0.5; lat.act_delay[0] = True   # a mid-episode change: ignored, counted
    a.process(torch.full((2, 24), 2.0, dtype=torch.float64)); assert a.applied.dtype == torch.float32 and float(a.applied.max()) == 0.5 and int(a.switches) == 1
    a.reset(torch.tensor([0])); a.process(torch.ones(2, 24)); assert not a.applied[0].any()

def test_action_term_reads_no_observation():
    r = C.reads_no_obs(); assert r["ok"] and r["stub_ok"] and r["identifiers"] > 10, r

def _p1_reference(obs, legs, arms, scale, clip):
    """P1's M3sHybrid.act, transcribed (p1_eval.py lines 37-43; resim.Hybrid._o for an 83-d base and a 111-d arm net)."""
    g = np.array(obs[:, :83], copy=True); g[:, 73:83] = np.clip(obs[:, 25:35] / scale[14:24], -clip, clip)
    a = legs(g).copy(); b = arms(obs); a[:, 14:] = b[:, 14:]; return a

def _nets(seed):
    rng = np.random.default_rng(seed); W1, W2 = rng.normal(size=(83, 24)).astype(np.float32), rng.normal(size=(111, 24)).astype(np.float32)
    return (lambda x: np.asarray(x, np.float32) @ W1), (lambda x: np.asarray(x, np.float32) @ W2)

def test_compose_equals_p1_m3s():
    legs, arms = _nets(0); scale = action_scale(load_cfg(CONFIGS / "r1_mjlab.yaml")); rng = np.random.default_rng(1); O = rng.normal(size=(64, 111)) * 2
    for clip in (1e9, 0.5):
        got = AC.compose(torch.from_numpy(O), lambda x: torch.from_numpy(legs(x.numpy())), lambda x: torch.from_numpy(arms(x.numpy())), scale, clip).numpy()
        assert np.array_equal(got, _p1_reference(O, legs, arms, np.asarray(scale, float), clip))
    g = AC.m3s_input(torch.from_numpy(O), scale, 1e9).numpy(); assert np.array_equal(g[:, :73], O[:, :73]) and not np.array_equal(g[:, 73:83], O[:, 73:83])
    assert AC.arm_input(torch.from_numpy(O), "zeros")[:, 83:].abs().sum() == 0 and AC.arm_input(torch.from_numpy(O), 83).shape == (64, 83)

@pytest.mark.skipif(not os.environ.get("FB_P1_EVAL"), reason="FB_P1_EVAL not set (P1's p1_eval.py lives outside the repo)")
def test_compose_equals_p1s_own_class():
    p1 = C.load_p1(os.environ["FB_P1_EVAL"]); legs, arms = _nets(2)
    base = type("B", (), dict(act=lambda self, x: legs(x), kp_mode=None))(); arm = type("A", (), dict(act=lambda self, x: arms(x), kp_mode="obs"))()
    h = p1.M3sHybrid(base, arm, "t"); scale = action_scale(load_cfg(CONFIGS / "r1_mjlab.yaml")); h.set_scale(scale, 1e9); O = np.random.default_rng(3).normal(size=(64, 111))
    got = AC.compose(torch.from_numpy(O), lambda x: torch.from_numpy(legs(x.numpy())), lambda x: torch.from_numpy(arms(x.numpy())), scale, 1e9).numpy()
    assert np.array_equal(got, h.act(O)) and np.array_equal(AC.m3s_input(torch.from_numpy(O), scale, 1e9).numpy(), h.g_in(O))
