"""M7 checks of the defender's actor and its export (M2V2_PLAN_v3 3 M7, 2.5; gpu/hybrid_actor.py, scripts/export_hybrid_fb.py) on CPU, on
the recorded eval obs of P1's H(A3s)/M3s (M4 recording, fb_m4rec_60000-60015.npz: every H step, the 83 + noise and the delivered
keypoint block as the controller received them, and its action). No simulation.
  (1) Stage A forward = H(A3s)/M3s bit for bit; leg dims have no gradient.
      a  P1 identity: HybridActor (stage A) with the donors' own ONNX sessions plugged in (P1's c.base / c.arms) = P1's M3sHybrid.act =
         the recorded action, every row at batch 1 (as recorded); the blind twin vs P1's H(blind3)/M3s on the same rows.
      b  deployed torch nets: HybridActor.mean (float32) = fb_actions.compose (M4 G0-10's composition) with the same torch nets, on the
         float64 rows and on their float32 casts, batched and at batch 1, sighted and blind; the torch nets = rsl_rl 5.0.1's own
         EmpiricalNormalization + MLP (the training stack's source, scratchpad run2b/src/rsl_x) bit for bit. Reported: torch vs the
         donors' ONNX (float32 GEMM order) and vs P1's action.
      c  gradients: a stage-A PPO-style loss (arm-only log-prob x advantage, entropy, plus an MSE on all 24 mean dims) leaves every G
         parameter without a gradient, the mean's leg dims without a graph, std_param[0:14] and the arm net's leg-output rows at exactly
         0; leg dims of a sample = G's mean; the log-prob ignores the leg dims; an Adam step on the actor's parameters leaves G and the
         leg actions bit-identical.
  (2) Stage B at init = stage A output bit for bit: from the start and from an emulated stage-A final (arm net and std perturbed,
      seeded), to_stage_b: mean bit-equal (float64 / float32 rows, batched and batch 1, ref path), residual exactly 0, arm std = A's,
      leg std 0.15; the exported B-init ONNX = the A ONNX bit for bit.
  (3) The anchor-loss (res_l2) gradient reaches only the residual: with a non-zero residual (seeded), every residual tensor has a
      non-zero gradient and every other parameter none or exactly 0; at init (r = 0) the anchor gradient is 0 everywhere.
  (4a) T9: the exported ONNX through r1gym's MjlabPolicy = HybridActor.mean(ref=True) on >= 1e4 recorded rows, <= 1e-6, for the start
      and a trained-like stage B (sighted, blind); the loader contract (kp_mode, batch 1, cfg = P1's H). Reported: the deployed float32
      torch path vs the ONNX, a float32-graph export, and the donors' own shipped ONNX vs their checkpoints (float32 vs float32).
  (4b) is the Join step (export_hybrid_fb.py t9b), not run here.
Run in the repo: python -I scripts/fb_m7_check.py --out m7_check.json --m4rec fb_m4rec_60000-60015.npz --p1 p1_eval.py --dir <scratch>"""
import sys, json, copy, time, types, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, torch
import hybrid_actor as HA, fb_actions as AC, export_hybrid_fb as EX

RSL = Path(r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\run2b\src\rsl_x")
TOL = EX.TOL; PERTURB = 1e-2
def bits(x): return EX.bits(x.numpy() if torch.is_tensor(x) else x)
def neq(a, b): return int((bits(a) != bits(b)).sum())
def ort1(p):
    """An onnxruntime session on 1 intra-op thread, as the eval's loader (MjlabPolicy) opens it."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

def rows(m4rec):
    """(obs [N, 111] float64, P1's recorded actions [N, 24]) of every H step, in recording order."""
    import fb_m4_check as M4
    R = M4.load(m4rec); O, A = [], []
    for _, b, X in M4.bouts(R, "H"):
        T = b["steps"]; O.append(np.concatenate([X["sens_o"][:T].astype(np.float64) + X["sens_n"][:T], X["kp_out"][:T].astype(np.float64)], 1)); A.append(X["a_out"][:T])
    return np.concatenate(O), np.concatenate(A)

def p1_ctrl(p1, arms="a3s"):
    import fb_m4_check as M4
    sc, clip, _ = EX.env_scale(); c = M4.load_p1(p1).make_ctrl(f"guard_fw_r+{arms}", "m3s"); c.set_scale(sc, clip); return c

def rsl_nets():
    """rsl_rl 5.0.1's EmpiricalNormalization and MLP from the training stack's source (tensordict / git stubbed: unused here)."""
    if not RSL.exists(): return None
    for n in ("tensordict", "git"):
        if n not in sys.modules: m = types.ModuleType(n); m.TensorDict = dict; sys.modules[n] = m
    sys.path.insert(0, str(RSL)); from rsl_rl.modules import MLP, EmpiricalNormalization
    def make(sd):
        w = sd["mlp.0.weight"].shape[1]; norm, net = EmpiricalNormalization(w), MLP(w, 24, list(HA.HID), "elu")
        norm.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("obs_normalizer.")}); net.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("mlp.")})
        return lambda x: net(norm(x))
    return make

def perturbed(a, seed, scale=PERTURB):
    """A trained-like copy: every trainable tensor + scale x N(0, 1) (seeded)."""
    a = copy.deepcopy(a); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(a.trainable().items()): p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return a

# ---- (1)
def c1(O, A, p1):
    res = {}; sc, clip, _ = EX.env_scale(); o64 = torch.from_numpy(O); o32 = o64.float()
    for kind, arms in (("sighted", "a3s"), ("blind", "blind3")):
        c = p1_ctrl(p1, arms); a = HA.HybridActor.from_checkpoints(ROOT, sc, clip, kind == "blind").eval()
        f = lambda m: (lambda x: torch.from_numpy(m.act(x.numpy())))
        a.plug(f(c.base), f(c.arms)); mis_p1 = mis_rec = 0; P = []
        with torch.no_grad():
            for t in range(len(O)):
                y = a.mean(o64[t:t + 1]); P.append(c.act(O[t:t + 1])); mis_p1 += neq(y, P[-1])
                if kind == "sighted": mis_rec += neq(y[0], A[t])
        P = np.concatenate(P)
        a.plug(None); res[f"a_{kind}"] = dict(rows=len(O), vs_p1_bit_mismatch=mis_p1, **({"vs_recorded_bit_mismatch": mis_rec} if kind == "sighted" else {}))
        # b: torch nets vs fb_actions.compose (M4 G0-10) with the same nets, and vs rsl_rl's own modules
        L = lambda x: a.legs(x.to(torch.float32)); R = lambda x: a.arms(x.to(torch.float32)); w = a.width; b = {}
        with torch.no_grad():
            for nm, x in (("f64", o64), ("f32", o32)):
                ref = AC.compose(x.double(), L, R, sc, clip, w); b[f"{nm}_batched"] = neq(a.mean(x), ref)
                y1 = torch.cat([a.mean(x[t:t + 1]) for t in range(0, len(x), 7)]); b[f"{nm}_batch1"] = sum(neq(a.mean(x[t:t + 1]), AC.compose(x[t:t + 1].double(), L, R, sc, clip, w)) for t in range(0, len(x), 7))
                b[f"info_{nm}_batch1_vs_batched_max_abs"] = float((y1 - ref[::7]).abs().max())
            mk = rsl_nets()
            if mk is not None:
                g, u = a.inputs(o64); b["rsl_rl_legs"] = neq(a.legs(g), mk(HA.load_actor_sd(ROOT / HA.G_PT))(g)); b["rsl_rl_arms"] = neq(a.arms(u), mk(HA.load_actor_sd(ROOT / HA.ARM_PT[kind]))(u))
            b["info_vs_p1_max_abs"] = float(np.abs(a.mean(o64).double().numpy() - P).max())
            g, u = a.inputs(o64); b["info_torch_vs_donor_onnx_legs"] = float(np.abs(a.legs(g).numpy() - c.base.act(g.numpy())).max())
            b["info_torch_vs_donor_onnx_arms"] = float(np.abs(a.arms(u).numpy() - c.arms.act(u.numpy())).max())
        b["rows"] = len(O); b["batch1_rows"] = len(range(0, len(O), 7)); res[f"b_{kind}"] = b
    res["c"] = c1_grad(O); bad = [k for k, v in res.items() if k[0] in "ab" for kk, vv in v.items() if ("mismatch" in kk or kk.startswith(("f64", "f32", "rsl"))) and vv]
    res["ok"] = bool(not bad and res["c"]["ok"] and all("rsl_rl_legs" in res[f"b_{k}"] for k in ("sighted", "blind"))); res["nonzero"] = bad; return res

def c1_grad(O, n=4096, seed=0):
    sc, clip, _ = EX.env_scale(); a = HA.HybridActor.from_checkpoints(ROOT, sc, clip).train(); x = torch.from_numpy(O[:n]).float(); g = torch.Generator().manual_seed(seed)
    a0 = copy.deepcopy(a); out = {}
    mu_det = a.mean(x); T = list(a.trainable().values())   # d(leg dims of the mean) / d(every trained parameter): none or exactly 0
    gl = torch.autograd.grad(mu_det[:, HA.LEGS].sum(), T, allow_unused=True, retain_graph=True); ga = torch.autograd.grad(mu_det[:, HA.ARMS].sum(), T, allow_unused=True)
    out["leg_mean_grad_nonzero"] = sum(int(v is not None and bool((v != 0).any())) for v in gl); out["arm_mean_grad_nonzero"] = sum(int(v is not None and bool((v != 0).any())) for v in ga)
    act = a(x, stochastic_output=True); out["sample_legs_eq_mean_mismatch"] = neq(act[:, HA.LEGS], mu_det[:, HA.LEGS].detach())
    lp = a.get_output_log_prob(act); act2 = act.clone(); act2[:, HA.LEGS] += 1.0; out["logprob_ignores_legs"] = bool(torch.equal(lp, a.get_output_log_prob(act2)))
    adv = torch.randn(n, generator=g); mu = a.output_mean
    loss = -(lp * adv).mean() - 0.003 * a.output_entropy.mean() + 0.5 * ((mu - torch.randn(mu.shape, generator=g)) ** 2).mean()   # the MSE touches all 24 dims
    loss.backward(); P = dict(a.named_parameters())
    out["legs_params_with_grad"] = [k for k, p in P.items() if k.startswith("legs.") and p.grad is not None]
    out["legs_params_requiring_grad"] = [k for k, p in P.items() if k.startswith("legs.") and p.requires_grad]
    out["std_leg_grad_nonzero"] = int((a.std_param.grad[HA.LEGS] != 0).sum()); out["std_arm_grad_nonzero"] = int((a.std_param.grad[HA.ARMS] != 0).sum())
    W = a.arms.mlp[-1]; out["arm_net_leg_rows_grad_nonzero"] = int((W.weight.grad[HA.LEGS] != 0).sum() + (W.bias.grad[HA.LEGS] != 0).sum())
    out["arm_net_arm_rows_grad_nonzero"] = int((W.weight.grad[HA.ARMS] != 0).sum())
    opt = torch.optim.Adam(a.parameters(), lr=2e-4); opt.step()
    with torch.no_grad(): m1, m0 = a.mean(x), a0.mean(x)
    out["legs_state_unchanged"] = all(torch.equal(v, a0.state_dict()[k]) for k, v in a.state_dict().items() if k.startswith("legs."))
    out["leg_actions_after_step_mismatch"] = neq(m1[:, HA.LEGS], m0[:, HA.LEGS]); out["arm_actions_changed"] = neq(m1[:, HA.ARMS], m0[:, HA.ARMS]) > 0
    out["ok"] = bool(out["leg_mean_grad_nonzero"] == 0 and out["arm_mean_grad_nonzero"] > 0 and out["sample_legs_eq_mean_mismatch"] == 0 and out["logprob_ignores_legs"]
                     and not out["legs_params_with_grad"] and not out["legs_params_requiring_grad"] and out["std_leg_grad_nonzero"] == 0 and out["std_arm_grad_nonzero"] > 0
                     and out["arm_net_leg_rows_grad_nonzero"] == 0 and out["arm_net_arm_rows_grad_nonzero"] > 0 and out["legs_state_unchanged"]
                     and out["leg_actions_after_step_mismatch"] == 0 and out["arm_actions_changed"])
    return out

# ---- (2)
def c2(O, d):
    sc, clip, _ = EX.env_scale(); o64 = torch.from_numpy(O); o32 = o64.float(); res = {}
    for kind in ("sighted", "blind"):
        start = HA.HybridActor.from_checkpoints(ROOT, sc, clip, kind == "blind").eval()
        for nm, A in (("start", start), ("a_final", perturbed(start, 7))):
            B = copy.deepcopy(A).to_stage_b(seed=42).eval(); r = {}
            with torch.no_grad():
                for xn, x in (("f64", o64), ("f32", o32)):
                    r[f"{xn}_batched"] = neq(B.mean(x), A.mean(x)); r[f"{xn}_residual_nonzero"] = int((B.r != 0).sum())
                    r[f"{xn}_batch1"] = sum(neq(B.mean(x[t:t + 1]), A.mean(x[t:t + 1])) for t in range(0, len(x), 7))   # both at batch 1
                    r[f"{xn}_ref"] = neq(B.mean(x, ref=True), A.mean(x, ref=True))
                r["arm_std_mismatch"] = neq(B.std()[HA.ARMS], A.std()[HA.ARMS]); r["leg_std"] = sorted(set(map(float, B.std()[HA.LEGS])))
                pa, pb = d / f"c2_{kind}_{nm}_A", d / f"c2_{kind}_{nm}_B"; EX.export(pa, actor=A); EX.export(pb, actor=B)
                sa, sb = (ort1(p / "policy.onnx") for p in (pa, pb))
                ya = np.concatenate([sa.run(None, {"obs": q[None]})[0] for q in O.astype(np.float32)]); yb = np.concatenate([sb.run(None, {"obs": q[None]})[0] for q in O.astype(np.float32)])
                r["onnx_B_vs_A"] = neq(yb, ya)
            res[f"{kind}_{nm}"] = r
    res["ok"] = bool(all(v[k] == 0 for v in res.values() for k in v if k != "leg_std") and all(v["leg_std"] == [float(torch.tensor(0.15))] for v in res.values())); return res

# ---- (3)
def c3(O, seed=3):
    sc, clip, _ = EX.env_scale(); x = torch.from_numpy(O[:4096]).float(); res = {}
    for nm in ("init", "nonzero"):
        a = perturbed(HA.HybridActor.from_checkpoints(ROOT, sc, clip), 7).to_stage_b(seed=42).train()
        if nm == "nonzero":
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for p in a.residual[-1].parameters(): p.copy_(1e-2 * torch.randn(p.shape, generator=g))
        a.zero_grad(set_to_none=True); a(x, stochastic_output=True); loss = 0.5 * a.res_l2().mean(); loss.backward()
        P = dict(a.named_parameters()); res_p = {k: p for k, p in P.items() if k.startswith("residual.")}; other = {k: p for k, p in P.items() if not k.startswith("residual.")}
        r = dict(res_l2_mean=float(a.res_l2().mean().detach()), residual_tensors=len(res_p), residual_tensors_nonzero_grad=sum(int(p.grad is not None and bool((p.grad != 0).any())) for p in res_p.values()),
                 other_params_nonzero_grad=[k for k, p in other.items() if p.grad is not None and bool((p.grad != 0).any())],
                 other_params_with_grad_tensor=sum(p.grad is not None for p in other.values()), residual_abs_grad_sum=float(sum(p.grad.abs().sum() for p in res_p.values() if p.grad is not None)))
        res[nm] = r
    res["ok"] = bool(res["nonzero"]["residual_tensors_nonzero_grad"] == res["nonzero"]["residual_tensors"] == 6 and not res["nonzero"]["other_params_nonzero_grad"]
                     and not res["init"]["other_params_nonzero_grad"] and res["init"]["residual_abs_grad_sum"] == 0.0); return res

# ---- (4a)
def c4(O, d, p1):
    sc, clip, _ = EX.env_scale(); res = {}; O32 = torch.from_numpy(O.astype(np.float32))
    for kind in ("sighted", "blind"):
        start = HA.HybridActor.from_checkpoints(ROOT, sc, clip, kind == "blind").eval(); B = perturbed(start, 7).to_stage_b(seed=42)
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for p in B.residual[-1].parameters(): p.copy_(1e-2 * torch.randn(p.shape, generator=g))
        for nm, a in (("start", start), ("trained_B", B.eval())):
            p = d / f"t9_{kind}_{nm}"; EX.export(p, actor=a); r = EX.t9a(a, p / "policy.onnx", O); r["contract"] = EX.contract(p / "policy.onnx", p1)
            with torch.no_grad(): r["residual_abs_max"] = float(a.r.abs().max()) if a.stage == "B" else 0.0
            p32 = d / f"t9_{kind}_{nm}_f32"; EX.export(p32, actor=a, dtype="float32"); r32 = EX.t9a(a, p32 / "policy.onnx", O)
            with torch.no_grad(): f = a.mean(O32).numpy()
            from r1gym.policy import load_controller
            y32 = load_controller(f"x={p32 / 'policy.onnx'}").act(O); r["info_f32_graph"] = dict(vs_ref_max_abs=r32["max_abs"], vs_f32_torch_max_abs=float(np.abs(y32 - f).max()))
            r["contract_ok"] = bool(r["contract"]["kp_mode"] == ("zeros" if kind == "blind" else "obs") and r["contract"]["obs_dim"] == 111 and r["contract"]["batch1"]
                                    and r["contract"]["cfg_equals_G"] and r["contract"].get("cfg_equals_P1_H", True))
            if nm == "start" and kind == "sighted": r["info_start_onnx_vs_p1_recorded_max_abs"] = float(np.abs(load_controller(f"x={p / 'policy.onnx'}").act(O) - rows_actions[0]).max())
            res[f"{kind}_{nm}"] = r
    don = {}
    for nm, pt, ox, w in (("G", HA.G_PT, HA.G_ONNX, 83), ("A3s", HA.ARM_PT["sighted"], HA.ARM_ONNX["sighted"], 111), ("blind3", HA.ARM_PT["blind"], HA.ARM_ONNX["blind"], 111)):
        net = HA.Net.from_actor_sd(HA.load_actor_sd(ROOT / pt)).eval(); s = ort1(ROOT / ox)
        x = O[:, :w].astype(np.float32); y = np.concatenate([s.run(None, {"obs": q[None]})[0] for q in x])
        with torch.no_grad(): t = net(torch.from_numpy(x)).numpy(); t64 = net(torch.from_numpy(x), torch.float64).numpy()
        don[nm] = dict(onnx_vs_torch32_max_abs=float(np.abs(y - t).max()), onnx_vs_torch64_max_abs=float(np.abs(y - t64).max()), bit_equal_share=float((bits(y) == bits(t)).mean()), act_abs_max=float(np.abs(y).max()))
    res["info_donors_shipped_onnx_vs_checkpoint"] = don
    res["ok"] = bool(all(v["ok"] and v["contract_ok"] for k, v in res.items() if not k.startswith("info"))); return res

rows_actions = [None]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--m4rec", required=True); ap.add_argument("--p1", required=True); ap.add_argument("--dir", required=True)
    ap.add_argument("--only", default="1,2,3,4"); a = ap.parse_args()
    torch.set_num_threads(2); t0 = time.time(); d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    O, A = rows(a.m4rec); rows_actions[0] = A; res = dict(rows=len(O), m4rec=a.m4rec, tol=TOL, perturb=PERTURB)
    print("rows", O.shape, flush=True)
    for k, f in (("1", lambda: c1(O, A, a.p1)), ("2", lambda: c2(O, d)), ("3", lambda: c3(O)), ("4", lambda: c4(O, d, a.p1))):
        if k not in a.only.split(","): continue
        t1 = time.time(); res[f"M7-{k}"] = f(); res[f"M7-{k}"]["wall_s"] = round(time.time() - t1, 1); print(f"M7-{k}", json.dumps(res[f"M7-{k}"], default=str), flush=True)
    res["ok"] = {k: res[k]["ok"] for k in res if k.startswith("M7-")}; res["wall_s"] = round(time.time() - t0, 1); print(json.dumps(res["ok"]), res["wall_s"], flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1, default=str)
    sys.exit(0 if all(res["ok"].values()) else 1)

if __name__ == "__main__": main()
