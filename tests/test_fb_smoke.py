"""M10 CPU training smoke and the pod evaluation scripts (M2V2_PLAN_v3 3 M10, section 4 P1 / P2 / P3; A8.2, A9.2, A11): the minimal
PPO = rsl_rl 5.0.1's PPO.compute_returns / update on the same data (the source in the scratchpad, tensordict stubbed), the PPO config =
unitree's R1 runner cfg + fullbody_task.RL, the smoke env's seeds and plumbing on a 2-env iteration, the scripted knockdown classes, the
magnitude table arithmetic, and the comparators / discovery / platform rules of fb_p1, fb_t1c and fb_loco_eval. The blocking smoke itself
runs in scripts/fb_cpu_smoke.py plumbing."""
import sys, json, copy, types, re, math
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import fb_cpu_smoke as SM, fb_rewards as RW, hybrid_actor as HA, fullbody_task as FT, fb_p1 as P1, fb_t1c as TC, fb_loco_eval as LE, fb_eq as EQ, fb_m7_check as M7
from fb_seeds import SeedRefused

UNITREE_RL = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\unitree_rl_mjlab\src\tasks\velocity\config\r1\rl_cfg.py")

def rsl_ppo():
    for n in ("tensordict", "git"):
        if n not in sys.modules: m = types.ModuleType(n); m.TensorDict = dict; sys.modules[n] = m
    if str(M7.RSL) not in sys.path: sys.path.insert(0, str(M7.RSL))
    from rsl_rl.algorithms.ppo import PPO
    return PPO

@pytest.mark.skipif(not UNITREE_RL.exists(), reason="unitree_rl_mjlab source not in the scratchpad")
def test_ppo_cfg_is_unitree_r1_and_task_overrides():
    src = UNITREE_RL.read_text(encoding="utf-8")
    for k, v in SM.PPO_CFG.items():
        m = re.search(rf"\b{k}=([^,\n]+)", src); assert m, k; got = m.group(1).strip().strip('"')
        assert (got == v) if isinstance(v, str) else (got == str(v) if isinstance(v, bool) else float(got) == float(v)), (k, got, v)
    assert re.search(r"num_steps_per_env=24\b", src) and SM.SPE == 24 and FT.RL["A"] == dict(learning_rate=2e-4, entropy_coef=0.003)

class _Obs(dict):
    def __init__(self, d, n): super().__init__(d); self.batch_size = (n,)

class _Critic(torch.nn.Module):
    """rsl_rl's critic call signature around the smoke's critic net (the same parameters)."""
    is_recurrent = False
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, obs, masks=None, hidden_state=None): return self.net(obs["critic"])

def _filled(n=4, nc=12, seed=0):
    """A smoke PPO (stage A actor from the M9 factory, a fresh critic) with one full rollout of synthetic rows."""
    torch.manual_seed(seed); a = SM.make_actor("A"); c = HA.Net(nc, 1); ppo = SM.PPO(a, c, n, nc, 2e-4, 0.003); sd = HA.load_actor_sd(ROOT / HA.ARM_PT["sighted"])
    g = torch.Generator().manual_seed(seed)
    for k in range(SM.SPE):
        oa = (sd["obs_normalizer._mean"] + sd["obs_normalizer._std"] * torch.randn(n, 111, generator=g)).float(); oc = torch.randn(n, nc, generator=g)
        with torch.no_grad():
            ppo.act(oa, oc); r = torch.randn(n, generator=g) * 0.1; d = torch.rand(n, generator=g) < 0.1; to = d & (torch.rand(n, generator=g) < 0.5); ppo.process(oa, oc, r, d, to)
    return ppo, torch.randn(n, nc, generator=g)

@pytest.mark.skipif(not M7.RSL.exists(), reason="rsl_rl 5.0.1 source not in the scratchpad")
def test_returns_and_update_are_rsl_rl_ppo():
    """The same rollout through rsl_rl 5.0.1's PPO.compute_returns and PPO.update (a storage shim that yields the same mini-batches):
    returns, advantages, losses, the adaptive lr and every parameter after the update bit-equal."""
    PPO = rsl_ppo(); ppo, oc_last = _filled(); n, nc = ppo.st.n, ppo.st.oc.shape[-1]
    aR = SM.make_actor("A"); aR.load_state_dict(ppo.actor.state_dict()); cR = copy.deepcopy(ppo.critic); st = ppo.st   # the factory's actor (its forward is a closure: no deepcopy)
    S = types.SimpleNamespace(num_transitions_per_env=st.T, values=st.val.clone(), dones=st.done.clone(), rewards=st.rew.clone(), returns=torch.zeros_like(st.val), advantages=None)
    def gen(nmb, ne):
        T = types.SimpleNamespace(oa=st.oa, oc=st.oc, act=st.act, val=S.values, ret=S.returns, logp=st.logp, adv=S.advantages, mu=st.mu, sd=st.sd, n=n, T=st.T)
        for oa, oc, a, v, adv, ret, lp, old in SM.Storage.batches(T, nmb, ne):
            yield types.SimpleNamespace(observations=_Obs({"actor": oa, "critic": oc}, len(oa)), actions=a, values=v, advantages=adv, returns=ret, old_actions_log_prob=lp,
                                        old_distribution_params=old, masks=None, hidden_states=(None, None))
    S.mini_batch_generator = gen; S.clear = lambda: None
    c = SM.PPO_CFG; R = PPO(aR, _Critic(cR), S, num_learning_epochs=c["num_learning_epochs"], num_mini_batches=c["num_mini_batches"], clip_param=c["clip_param"], gamma=c["gamma"],
                            lam=c["lam"], value_loss_coef=c["value_loss_coef"], entropy_coef=0.003, learning_rate=2e-4, max_grad_norm=c["max_grad_norm"],
                            use_clipped_value_loss=c["use_clipped_value_loss"], schedule=c["schedule"], desired_kl=c["desired_kl"])
    ppo.returns(oc_last); R.compute_returns({"critic": oc_last}); assert torch.equal(ppo.st.ret, S.returns) and torch.equal(ppo.st.adv, S.advantages)
    torch.manual_seed(5); L1 = ppo.update(); torch.manual_seed(5); L2 = R.update()
    assert L1["value"] == L2["value"] and L1["surrogate"] == L2["surrogate"] and L1["entropy"] == L2["entropy"] and L1["lr"] == R.learning_rate
    for (k, p), (_, q) in zip(list(ppo.actor.named_parameters()) + list(ppo.critic.named_parameters()), list(aR.named_parameters()) + list(cR.named_parameters())): assert torch.equal(p, q), k

def test_stage_a_update_leaves_legs_and_their_grads_alone():
    ppo, oc_last = _filled(n=2, nc=8, seed=1); ref = SM.g_ref(); ppo.returns(oc_last); arms0 = copy.deepcopy(ppo.actor.arms.state_dict()); ppo.update()
    assert SM.legs_frozen(ppo.actor, ref) and any(not torch.equal(v, arms0[k]) for k, v in ppo.actor.arms.state_dict().items() if "mlp" in k)

def test_a0_trains_the_critic_only():
    torch.manual_seed(0); a = SM.make_actor("A0"); assert not any(p.requires_grad for p in a.parameters())

def test_seeds_and_episode_map():
    s = SM.Seeds("smoke", 69000, 69002, 69000); eps = [s() for _ in range(3)]
    assert [e["clip"] for e in eps] == list(SM.TRAIN) and [e["distance"] for e in eps] == [0.4, 0.4, 0.4] and SM.episode(69003, 69000)["distance"] == 0.45
    with pytest.raises(AssertionError): s()
    with pytest.raises(SeedRefused): SM.Seeds("smoke", 69100, 69101, 69000)()   # canary block, smoke role
    with pytest.raises(SeedRefused): SM.Seeds("canary", 69000, 69001, 69100)()
    assert SM.PLUMB["train"][1] < SM.PLUMB["scripted"][0] and SM.PLUMB["scripted"][1] <= SM.PLUMB["block"][1] and SM.CANARY["block"] == (69100, 69999)

def test_kd_class_follows_the_final_rule():
    e = lambda **k: dict(dict(charged=False, t=1.0, W=1.0, land_t=-math.inf), **k)
    assert SM.kd_class(e(charged=True)) == "charged" and SM.kd_class(e(t=0.5)) == "start_window" and SM.kd_class(e(W=RW.W_EPS)) == "w_floor"
    assert SM.kd_class(e(land_t=0.75)) == "land_gate" and SM.kd_class(e()) == "other"

def test_magnitudes_arithmetic():
    q = SM.SmokeEnv._zq(); q.update(alive_s=10.0, ring=5.0, kd_expected=2.0); R = {k: 0.0 for k in SM.TERMS}; R.update(spar_alive=40.0, spar_ring=12.5, fb_knockdown=-4.8 * 3)
    m = SM.magnitudes([("A", R, q)])
    assert m["spar_alive"]["ok"] and m["spar_ring"]["ok"] and m["fb_knockdown"]["x"] == pytest.approx(3.0) and not m["fb_knockdown"]["within_2x"] and not m["spar_head_clean"]["fires"]
    assert not m["fb_lean_away"]["fires"]   # B-only term: an A group never counts for it

def test_scripts_cover_every_class():
    names = [s.name for r in SM.SCRIPTS for s in r]; assert len(set(names)) == len(names) == 16 and all(len(r) == 8 for r in SM.SCRIPTS)
    inj = [s.inject for r in SM.SCRIPTS for s in r if s.inject is not None]; assert sum(x <= RW.W_EPS for x in inj) == 3 and sum(x > RW.W_EPS for x in inj) == 3
    assert any(s.k * SM.DT <= RW.START_S and s.vd > 0 for r in SM.SCRIPTS for s in r) and any(s.k * SM.DT > RW.START_S and s.vd > 0 for r in SM.SCRIPTS for s in r)

def test_smoke_env_one_iteration_plumbing():
    """2 envs, one stage-A iteration: finite rows, the reward = the terms' sum, critic width, legs frozen, no switches."""
    env = SM.SmokeEnv(2, SM.Seeds("smoke", 69000, 69009, 69000)); nc = env.critic_obs().shape[1]; ppo = SM.restart("A", None, 2, nc)
    assert nc == 111 + FT.INTENT_DIM + len(FT.PRIV) + 2 and env.obs.shape == (2, 111)
    rows = SM.run_stage(env, ppo, "A", 1, log=lambda *a, **k: None); r = rows[0]
    assert r["nan"] == 0 and r["legs_frozen"] and r["leg_dev"] == 0.0 and r["reward"] * 2 * SM.SPE == pytest.approx(sum(r["terms"].values()), rel=1e-4, abs=1e-4)
    assert int(env.B.obs.k_switches) == 0 and int(env.B.act.switches) == 0 and env.log["switch_ep"] == 0

# ---- fb_p1
def test_p1_comparators():
    a = P1.alone_compare({("c", 0): None, ("c", 1): 1.20}, {("c", 0): None, ("c", 1): 1.28}); assert a["ok"] and a["both_up"] == 1 and a["both_fell_within"] == 1
    assert not P1.alone_compare({("c", 0): None}, {("c", 0): 3.0})["ok"] and not P1.alone_compare({("c", 1): 1.0}, {("c", 1): 1.2})["ok"]
    t = P1.traj_compare({("c", 0.5): np.zeros((15, 3))}, {("c", 0.5): np.full((15, 3), 0.019)}); assert t["ok"]
    assert not P1.traj_compare({("c", 0.5): np.zeros((15, 3))}, {("c", 0.5): np.full((15, 3), 0.021)})["ok"]
    assert P1.shares_ok(300, 150, [100, 100, 100])["ok"] and not P1.shares_ok(300, 200, [100, 100, 100])["ok"] and not P1.shares_ok(0, 0, [0, 0, 0])["ok"]
    rows = P1.rows_from_first(torch.tensor([True, False]), torch.tensor([False, True]), torch.tensor([2.0, 3.0]), torch.tensor([150, 1000]), torch.tensor([1, 0]))
    x, ad, _ = EQ.arrays(rows); assert x.tolist() == [[1, 0, 2, 3.0], [0, 1, 3, 20.0]] and ad.tolist() == [1, 0]

def test_p1_hit_compare_matches_float32_onsets():
    """A float32 account onset late in the episode (1 ulp from the HitMeter's grid time) is the same event, not two disagreements."""
    t = 17.345; t32 = float(np.float32(t)); assert abs(t32 - t) > 1e-7
    hm = types.SimpleNamespace(events=lambda: [dict(limb="hand", zone="head", t0=t)], landed=lambda e: True, clean=lambda e: True)
    log = [dict(kind="episode", landed=True, limb="hand", zone="head", t0=t32 + 1.2e-6, clean=True)]
    r = P1.hit_compare([log], [hm]); assert r["landed_union"] == 1 and r["clean_agree"] == 1.0 and r["exact_worlds"] == 1

def test_p1_r4g_verdict_on_equal_distributions():
    rng = np.random.default_rng(0); mk = lambda n: [dict(seed=i, clip="", distance=0.0, t0=0.0, k=0, act_delay=int(rng.random() < .5), fell_s=None if rng.random() > .03 else 5.0,
                                                      kd=bool(rng.random() < .07), n=dict(head_clean=int(rng.poisson(1.1))), secs=20.0) for i in range(n)]
    eo = {c: mk(12288) for c in ("G", "H", "A3")}; gpu = {c: mk(12288) for c in ("G", "H", "A3")}
    v = P1.r4g_verdict(eo, gpu); assert set(v["gated_pass"]) == {"surv_G", "surv_H", "hc_G", "hc_H", "kd_G", "gap_A3_G"} and v["gates"]["surv_G"]["pass_"]

def test_p1_hit_replay_equals_the_live_hitmeter():
    """The recorder's fields replayed through the shim give the live HitMeter's events exactly (one 20-s replica bout, smoke seed)."""
    ep = TC._eo_episode(SM.episode(69000, 69000)); r = P1.dry_hit([ep]); assert r["replay_equals_live"] == "1/1" and r["landed_union"] > 0

# ---- fb_t1c
def test_t1c_platform_and_t0_arm_refusal(tmp_path):
    p = TC.platform(); assert {"cpu", "machine", "system", "numpy", "mujoco", "torch"} <= set(p)
    (tmp_path / "platform.json").write_text(json.dumps(dict(p, cpu="AMD EPYC 9655 (another box)")), encoding="utf-8")
    with pytest.raises(AssertionError, match="A8.2"): TC.run(str(tmp_path / "x.json"), t0_arm=str(tmp_path), dry=True, seeds=[69000], workers=1)

def test_t1c_finds_the_t1_run(tmp_path):
    for d in ("t0", "2026-11-01_10-00-00_t1", "2026-11-01_12-00-00_t1_loco"):
        (tmp_path / d).mkdir(); (tmp_path / d / "model_250.pt").write_bytes(b"x"); (tmp_path / d / "model_1000.pt").write_bytes(b"x")
    assert TC.find_t1(tmp_path) == tmp_path / "2026-11-01_10-00-00_t1" / "model_1000.pt"
    with pytest.raises(FileNotFoundError): TC.find_t1(tmp_path / "none")

def test_t1c_with_tracker_checks_the_conf():
    import fb_opponent as FO, fb_replica as RP
    om = TC.with_tracker(FO.TRACKER_PT); c = RP.conf(backend="torch32", tracker="0" * 64)
    with pytest.raises(AssertionError): om(1, torch.float32, c)
    M, drv, aim, pol = om(1, torch.float32, dict(c, tracker=TC.sha(FO.TRACKER_PT))); assert isinstance(drv.mlp, FO.TrackerMLP)

# ---- fb_loco_eval
def test_loco_pilot_checkpoint_rules(tmp_path):
    runs, prog = tmp_path / "runs", tmp_path / "progress_fb.txt"
    for d in ("x_loco_s42_A", "x_loco_s42_B", "x_loco_s42_blind_A", "x_loco_s42_blind_B"): (runs / d).mkdir(parents=True); (runs / d / "model_99.pt").write_bytes(b"x")
    prog.write_text("watch loco_s42_B at 1000 rc 0\n", encoding="utf-8")
    assert LE.pilot_ckpt(False, runs, prog)[1] == "B" and LE.pilot_ckpt(True, runs, prog)[1] == "B"
    prog.write_text("EARLY_STOP loco_s42_B at 1000 x\nEARLY_STOP loco_s42_blind_B at 1000 x\n", encoding="utf-8")
    s, b = LE.pilot_ckpt(False, runs, prog), LE.pilot_ckpt(True, runs, prog)
    assert s[1] == "A" and "A final" in s[2] and b[1] == "B"   # a stopped sighted B falls back to A's final; the blind twin is never stopped
    assert LE.decide(dict(proceed=True, seen=True, survival=True)) == "proceed" and LE.decide(dict(proceed=False, seen=True, survival=True)).startswith("stop and report")
    assert LE.decide(dict(proceed=False, seen=False, survival=True)).startswith("stop and diagnose")

def test_loco_survival_bar_direction():
    """P3: 'survival pilot - start point >= -0.015 and 90% lower >= -0.04': a pilot that falls more fails, one that falls less passes."""
    rng = np.random.default_rng(2)
    def mk(n, pf, hc): f = (rng.random(n) < pf).astype(float); return np.stack([f, np.zeros(n), rng.poisson(hc, n), np.where(f > 0, 10.0, 20.0)], 1)
    worse = EQ.loco_bars(mk(1152, 0.13, 0.5), mk(1152, 0.03, 1.0), mk(1152, 0.13, 0.5), mk(1152, 0.03, 1.0))
    better = EQ.loco_bars(mk(1152, 0.01, 0.5), mk(1152, 0.05, 1.0), mk(1152, 0.01, 0.5), mk(1152, 0.05, 1.0))
    assert worse["surv_diff"] < -0.05 and not worse["survival"] and not worse["proceed"] and better["surv_diff"] > 0.02 and better["survival"] and better["proceed"]

def test_loco_ratio_and_boot():
    rng = np.random.default_rng(1); mk = lambda n, hc: np.stack([np.zeros(n), np.zeros(n), rng.poisson(hc, n), np.full(n, 20.0)], 1).astype(float)
    hp, hs, sp, ss = mk(400, 0.8), mk(400, 1.0), mk(400, 0.7), mk(400, 1.0); r = LE.ratio(hp, hs); assert 0.6 < r["R"] < 1.0 and r["ci90"][0] < r["R"] < r["ci90"][1]
    lo, hi = LE.g_boot(hp, hs, sp, ss, n=200); assert lo < hi
