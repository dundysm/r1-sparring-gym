"""Full-body M2 v2 reward terms (M2V2_PLAN_v3 M5 G0-12; gpu/fb_rewards.py): the scripted events of scripts/fb_m5_check.py scripted()
(an attributed knockdown is exactly one -2.4 event; a tracker-alone fall, a knockdown < 0.3 s after a landed strike, a blocked strike
then an arm-only stumble give 0; down-neutral only while opp_down_now; approach / contact 0 while both are up; HIT_R2 pays 0 when a
same-side forearm lands after the glove touch; tangle summed per substep; every hit term masked 0.5 s after a push), the weights of
the plan's 2.4 table, W0 frozen (A4.1; a knockdown with no positive work never charged), W_EPS and the 0.5 s start window (A5.1: work at or
below the float32 floor and an edge in the episode's first 0.5 s never charged), the attribution ring, the per-row EMA, the lean sign and the
push-mask boundary."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_rewards as RW, fb_m5_check as C

@pytest.fixture(scope="module")
def scripted(): return C.scripted(C.model()[0])

@pytest.mark.parametrize("name", ["attributed_knockdown", "tracker_alone_fall", "kd_after_landed_strike", "kd_strike_0p36s_before", "blocked_then_arm_only_stumble",
                                  "down_neutral_nonsticky", "down_terms_zero_when_up", "hit_r2_forearm", "tangle_per_substep", "push_mask", "frozen_w0", "w_eps_floor",
                                  "start_window", "stages"])
def test_scripted_events(scripted, name):
    assert scripted[name]["ok"], scripted[name]

def test_weights_are_the_plan_table():
    A = dict(spar_head_clean=-40, spar_head_impulse=-20, spar_torso_impulse=-5, spar_block=40, spar_clearance=0.5, spar_ring=2.5, spar_alive=4,
             fb_knockdown=-120, fb_down_neutral=-0.8, fb_down_approach=-2.0, fb_down_contact=-0.1, fb_leg_tangle=-0.05)
    assert RW.WEIGHTS["A"] == A and RW.WEIGHTS["B"] == {**{k: v for k, v in A.items() if k != "spar_clearance"}, "fb_lean_away": -0.01, "res_l2": -0.5}
    assert RW.res_l2_weight(-5) == -0.5 and abs(RW.res_l2_weight(1000) + 0.3) < 1e-12 and abs(RW.res_l2_weight(9999) + 0.1) < 1e-12
    assert set(RW.HIT_TERMS) <= set(RW.WEIGHTS["A"]) and "fb_down_neutral" not in RW.HIT_TERMS

def test_w0_frozen():
    # A4.1: P10 of W over the 98 R4-pre G knockdowns with contact (32 with W = 0) is 0; the charge still needs W > 0
    assert RW.W0 == 0.0; RW.FbRewards(2); a = RW.KnockdownAttribution(1, RW.W0); T = lambda x: torch.tensor([x], dtype=torch.float64); E = torch.tensor([True])
    assert not bool(a.charge(E, T(1.0), T(-np.inf))); a.push(T(2 * RW.W_EPS)); assert bool(a.charge(E, T(1.0), T(-np.inf)))

def test_w_eps_and_start_window():
    """A5.1: W_EPS frozen (10 x the measured float32 / float64 gap of W, 1.16094e-5 J); the charge needs W > W_EPS and t > 0.5 s."""
    assert RW.W_EPS == 0.00011609440647930569 and RW.START_S == 0.5; T = lambda x: torch.tensor([x], dtype=torch.float64); E = torch.tensor([True])
    a = RW.KnockdownAttribution(1, RW.W0, dtype=torch.float64); a.push(T(RW.W_EPS)); assert not bool(a.charge(E, T(1.0), T(-np.inf)))
    a.push(T(1e-9)); assert bool(a.charge(E, T(1.0), T(-np.inf)))
    for t, ok in ((0.02, False), (0.48, False), (0.5, False), (0.5 + 1e-12, False), (0.52, True)): assert bool(a.charge(E, T(t), T(-np.inf))) == ok, t
    with pytest.raises(AssertionError): RW.KnockdownAttribution(1, 0.0, eps=0.0)

def test_attribution_ring_forgets_after_one_second():
    a = RW.KnockdownAttribution(1, 1.0); a.push(torch.tensor([5.0])); T = lambda x: torch.tensor([x])
    for _ in range(a.L - 1): a.push(torch.tensor([0.0]))
    assert a.L == 50 and float(a.W()) == 5.0 and bool(a.charge(T(True), T(1.0), T(-np.inf)))
    a.push(torch.tensor([0.0])); assert float(a.W()) == 0.0 and not bool(a.charge(T(True), T(1.0), T(-np.inf)))
    a.push(torch.tensor([2.0])); assert not bool(a.charge(T(True), T(1.0), T(0.7))) and bool(a.charge(T(True), T(1.0), T(0.69))) and not bool(a.charge(T(False), T(1.0), T(-np.inf)))

def test_down_neutral_rows_are_separate():
    d = RW.DownNeutral(prior_pm=5.0); r0 = d.rate_pm().clone(); d.update(torch.tensor([3.0, 0.0]), torch.tensor([0.02, 0.02]), torch.tensor([2, 2]))
    r = d.rate_pm(); assert torch.equal(r[[0, 1, 3]], r0[[0, 1, 3]]) and r[2] > r0[2]   # rows without exposure are untouched
    for _ in range(20000): d.update(torch.zeros(2), torch.full((2,), 0.02), torch.tensor([1, 1]))
    assert float(d.rate_pm()[1]) < 0.01 and float(d.rate_pm()[0]) == 5.0

def test_lean_sign_and_push_mask_boundary():
    up = torch.tensor([[-np.sin(np.radians(10)), 0.0, np.cos(np.radians(10))], [np.sin(np.radians(10)), 0.0, np.cos(np.radians(10))]], dtype=torch.float64)
    lean = RW.lean_away_deg(up, torch.zeros(2, 2, dtype=torch.float64), torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(lean, torch.tensor([10.0, -10.0], dtype=torch.float64))   # tilting back, away from an opponent at +x, is positive
    t = torch.tensor([1.5, 1.5 + 1e-9, 2.0], dtype=torch.float64); assert RW.push_mask(t, torch.full((3,), 1.0, dtype=torch.float64)).tolist() == [0.0, 1.0, 1.0]
    assert RW.push_mask(t, torch.full((3,), -np.inf, dtype=torch.float64)).tolist() == [1.0, 1.0, 1.0]
