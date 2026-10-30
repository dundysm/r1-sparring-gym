"""Full-body M2 v2 defender actor and export (M2V2_PLAN_v3 M7; gpu/hybrid_actor.py, scripts/export_hybrid_fb.py): stage A = H(arms)/M3s
(fb_actions.compose with the same nets; P1's M3sHybrid arithmetic with the donors' ONNX plugged in), leg dims without gradient, stage B
at init = stage A bit for bit, the anchor's gradient on the residual only, the ONNX = the actor's float64 reference bit for bit through
r1gym's MjlabPolicy, checkpoint round trip, t9b's seed guard and a 1 s t9b bout (60400). Rows: the donors' normalizer mean + std x
N(0, 1) (no simulation outside the t9b bout). The recorded-row gates run in scripts/fb_m7_check.py."""
import sys, json, copy
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import hybrid_actor as HA, fb_actions as AC, export_hybrid_fb as EX, fb_m7_check as M7
from fb_seeds import SeedRefused

@pytest.fixture(scope="module")
def env():
    sc, clip, cfg = EX.env_scale(); sd = HA.load_actor_sd(ROOT / HA.ARM_PT["sighted"]); g = torch.Generator().manual_seed(0)
    x = sd["obs_normalizer._mean"] + sd["obs_normalizer._std"] * torch.randn(256, 111, generator=g, dtype=torch.float64)
    return sc, clip, x

def actor(env, blind=False): sc, clip, _ = env; return HA.HybridActor.from_checkpoints(ROOT, sc, clip, blind).eval()
def eq(a, b): return np.array_equal(*(EX.bits(v.detach().numpy() if torch.is_tensor(v) else np.asarray(v)) for v in (a, b)))

def test_scale_is_the_envs_and_nets_load_strict(env):
    from r1gym.model import load_cfg, action_scale
    from r1gym import CONFIGS
    sc, clip, _ = env; assert np.array_equal(sc, np.asarray(action_scale(load_cfg(CONFIGS / "r1_mjlab.yaml")), np.float64)) and clip == 1e9
    a = actor(env); assert a.legs.n_in == 83 and a.arms.n_in == 111 and a.std_param.shape == (24,) and not any(p.requires_grad for p in a.legs.parameters())
    assert set(a.trainable()) == {k for k, _ in a.named_parameters() if k.startswith("arms.")} | {"std_param"}

@pytest.mark.skipif(not M7.RSL.exists(), reason="rsl_rl 5.0.1 source not in the scratchpad")
def test_nets_are_rsl_rl_bit_for_bit(env):
    a = actor(env); mk = M7.rsl_nets(); g, u = a.inputs(env[2])
    with torch.no_grad(): assert eq(a.legs(g), mk(HA.load_actor_sd(ROOT / HA.G_PT))(g)) and eq(a.arms(u), mk(HA.load_actor_sd(ROOT / HA.ARM_PT["sighted"]))(u))

@pytest.mark.parametrize("blind", [False, True])
def test_stage_a_is_the_composition(env, blind):
    """= fb_actions.compose (M4 G0-10) with the same nets on float64 and float32 rows; = P1's M3sHybrid arithmetic with the donors' ONNX."""
    from r1gym.policy import load_controller
    sc, clip, x = env; a = actor(env, blind); L = lambda v: a.legs(v.float()); R = lambda v: a.arms(v.float())
    with torch.no_grad():
        for o in (x, x.float()): assert eq(a.mean(o), AC.compose(o.double(), L, R, sc, clip, a.width))
        G, A = load_controller(f"g={ROOT / HA.G_ONNX}"), load_controller(f"a={ROOT / HA.ARM_ONNX['blind' if blind else 'sighted']}")
        a.plug(lambda v: torch.from_numpy(G.act(v.numpy())), lambda v: torch.from_numpy(A.act(v.numpy()))); X = x[:64].numpy()
        g = np.array(X[:, :83], copy=True); g[:, 73:83] = np.clip(X[:, 25:35] / sc[14:24], -clip, clip)   # P1 M3sHybrid.act (p1_eval.py 37-43)
        ref = G.act(g).copy(); ref[:, 14:] = A.act(X if not blind else np.concatenate([X[:, :83], np.zeros((64, 28), np.float32)], 1))[:, 14:]
        assert eq(a.mean(x[:64]), ref)

def test_stage_a_leg_dims_have_no_gradient(env):
    a = actor(env).train(); x = env[2].float(); mu = a.mean(x); T = list(a.trainable().values())
    assert all(v is None or not v.any() for v in torch.autograd.grad(mu[:, :14].sum(), T, allow_unused=True))
    act = a(x, stochastic_output=True); assert eq(act[:, :14], mu[:, :14].detach())
    lp = a.get_output_log_prob(act); b = act.clone(); b[:, :14] += 3.0; assert torch.equal(lp, a.get_output_log_prob(b))
    (-(lp.mean()) - 0.003 * a.output_entropy.mean() + ((a.output_mean - 1.0) ** 2).mean()).backward()
    assert all(p.grad is None for p in a.legs.parameters()) and not a.std_param.grad[:14].any() and a.std_param.grad[14:].all()
    assert not a.arms.mlp[-1].weight.grad[:14].any() and not a.arms.mlp[-1].bias.grad[:14].any()
    kl = a.get_kl_divergence(a.output_distribution_params, (a.output_mean + 1.0, a.output_std)); assert torch.allclose(kl, torch.full_like(kl, 0.5 * float((1 / a.std_param[14:].detach() ** 2).sum())))

def test_normalization_updates_arms_only(env):
    a = actor(env).train(); g0, a0 = copy.deepcopy(a.legs.state_dict()), a.arms.obs_normalizer._mean.clone()
    a.update_normalization(env[2].float()); assert all(torch.equal(v, g0[k]) for k, v in a.legs.state_dict().items()) and not torch.equal(a.arms.obs_normalizer._mean, a0)

@pytest.mark.parametrize("blind", [False, True])
def test_stage_b_at_init_is_stage_a(env, blind):
    A = M7.perturbed(actor(env, blind), 1); B = copy.deepcopy(A).to_stage_b(seed=5).eval(); x = env[2]
    with torch.no_grad():
        for o in (x, x.float()):
            assert eq(B.mean(o), A.mean(o)) and not B.r.any() and eq(B.mean(o, ref=True), A.mean(o, ref=True))
            assert all(eq(B.mean(o[t:t + 1]), A.mean(o[t:t + 1])) for t in range(0, 256, 17))
    s = B.std(); assert eq(s[14:], A.std()[14:]) and torch.all(s[:14] == torch.tensor(0.15))
    with torch.no_grad(): B.std_param[:14] = 0.9
    assert torch.all(B.std()[:14] == torch.tensor(0.35)) and set(B.trainable()) == set(A.trainable()) | {k for k, _ in B.named_parameters() if k.startswith("residual.")}

def test_anchor_gradient_reaches_only_the_residual(env):
    a = actor(env).to_stage_b(seed=5).train(); x = env[2].float()
    with torch.no_grad(): a.residual[-1].weight.normal_(0, 1e-2, generator=torch.Generator().manual_seed(2))
    a(x, stochastic_output=True); a.res_l2().mean().backward()
    for k, p in a.named_parameters():
        if k.startswith("residual."): assert p.grad is not None and p.grad.any(), k
        else: assert p.grad is None or not p.grad.any(), k

@pytest.mark.parametrize("blind,stage", [(False, "A"), (True, "A"), (False, "B"), (True, "B")])
def test_onnx_equals_the_reference_through_the_eval_loader(env, tmp_path, blind, stage):
    from r1gym.policy import load_controller
    a = M7.perturbed(actor(env, blind), 3)
    if stage == "B":
        a.to_stage_b(seed=4)
        with torch.no_grad(): a.residual[-1].weight.normal_(0, 1e-2, generator=torch.Generator().manual_seed(6))
    a.eval(); man = EX.export(tmp_path, actor=a); X = env[2].numpy(); c = load_controller(f"x={tmp_path / 'policy.onnx'}")
    with torch.no_grad(): ref = a.mean(torch.from_numpy(X.astype(np.float32)), ref=True).numpy()
    assert eq(c.act(X), ref) and (stage == "A" or np.abs(a.r.numpy()).max() > 1e-3)
    assert c.kp_mode == ("zeros" if blind else "obs") and c.batch1 and (tmp_path / "gym.json").read_bytes() == (ROOT / HA.G_ONNX).with_name("gym.json").read_bytes()
    import onnx
    meta = {p.key: p.value for p in onnx.load(str(tmp_path / "policy.onnx")).metadata_props}; fb = json.loads(meta["fb_hybrid"])
    assert meta["observation_names"].endswith(HA.OBS_END["blind" if blind else "sighted"]) and fb["stage"] == stage and fb["dtype"] == "float64" and man["stage"] == stage

def test_checkpoint_round_trip(env, tmp_path):
    a = M7.perturbed(actor(env), 8).to_stage_b(seed=9)
    with torch.no_grad(): a.residual[-1].bias.fill_(0.01)
    torch.save({"actor_state_dict": a.state_dict()}, tmp_path / "model_1.pt"); b = EX.build(ckpt=tmp_path / "model_1.pt"); x = env[2]
    with torch.no_grad(): assert b.stage == "B" and eq(b.mean(x), a.eval().mean(x))

def test_blind_flag_is_stored_and_asserted(env, tmp_path):
    """A9.2: every state dict carries blind_flag; build() refuses a --blind that disagrees with it and a checkpoint without it;
    ckpt_blind reads it (selection / test read the flag, never an argument)."""
    for blind in (False, True):
        a = actor(env, blind); torch.save({"actor_state_dict": a.state_dict()}, tmp_path / f"m{int(blind)}.pt"); assert HA.stored_blind(a.state_dict()) is blind
        assert EX.ckpt_blind(tmp_path / f"m{int(blind)}.pt") is blind and EX.build(blind, tmp_path / f"m{int(blind)}.pt").blind is blind
        with pytest.raises(AssertionError): EX.build(not blind, tmp_path / f"m{int(blind)}.pt")
    sd = actor(env).state_dict(); sd.pop("blind_flag"); torch.save({"actor_state_dict": sd}, tmp_path / "old.pt")
    with pytest.raises(AssertionError): EX.build(False, tmp_path / "old.pt")

def test_t9b_refuses_other_seeds_before_any_bout():
    for s in ([22000], [21000], [60399], [60416], [60000]):
        with pytest.raises(SeedRefused): EX.t9b_episodes(s)

def test_t9b_one_second_bout_is_identical(tmp_path):
    """t9b end to end on 60400 for 1 s (role 't9', in role): EO.run with the start ONNX = the replica with the torch composition, rows
    and served controller rows; a --blind that disagrees with the export's manifest is refused."""
    EX.export(tmp_path, actor=EX.build()); r = EX.t9b(tmp_path / "policy.onnx", workers=1, seconds=1.0, seeds=(60400,))
    assert r["ok"] and r["identical"] == 1 and r["ctrl_rows"]["onnx"] == r["ctrl_rows"]["torch"] == r["same_state"]["rows"] == 50 and r["same_state"]["bit_mismatch"] == 0
    with pytest.raises(AssertionError): EX.t9b(tmp_path / "policy.onnx", blind=True, workers=1, seconds=1.0, seeds=(60400,))
