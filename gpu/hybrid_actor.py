"""Full-body M2 v2 defender actor (M2V2_PLAN_v3 2.5, M7; design_learned-tracker 8.1-8.3): the start H(arms)/M3s as one torch module
with rsl_rl 5.0.1's actor-model interface, its two training stages, and its one-file ONNX export. Pure torch (+ onnx for the export).
Composition (P1's p1_eval.py M3sHybrid, fb_actions.compose; G0-10):
  legs   G = guard_fw_r model_7496.pt, frozen (no parameter of it trains): G(f32(m3s(obs))), m3s = fb_actions.m3s_input computed in
         float64 with the env's float64 scale (R1HitEnv.scale, action_clip), then rounded to float32: P1's arithmetic for a float64
         row (g_in, then MjlabPolicy's cast) and for a float32 row alike. Dims 0-13 of G's output.
  arms   the arm net (A3s or blind3 model_8995.pt, trained) on f32(arm_input(obs)): the 111-d row, or its 83 + 28 zeros (blind twin).
         Dims 14-23 of its output.
  stage A  mean = [G legs, arm-net arms]; ArmOnlyGaussian: std_param stays 24-d (the donor's), sample / log-prob / entropy / KL over
         dims 14-23 only; the leg dims of a sample are G's mean exactly (no noise, no gradient).
  stage B  (to_stage_b) legs = G legs + r(obs), r = MLP(111 -> 256 -> 128 -> 14) ELU on the arm net's view, with its own
         EmpiricalNormalization (initialised as a copy of the arm net's, so it starts on the same scale and barely moves); last layer
         exactly zero, so B at init = A bit for bit. Gaussian over all 24 dims: leg std 0.15 (std_param[0:14], rsl_rl's 'scalar' std
         as the donors'), clamped at <= 0.35; arm std continues. res_l2() = |r|^2 per env on the residual's mean (the anchor).
Nets are rsl_rl 5.0.1 MLPModel actors (state-dict keys obs_normalizer._mean/_var/_std/count, mlp.{0,2,4,6}): (x - mean) / (std + 0.01)
-> 512-256-128 ELU -> 24, in float32. mean(obs, ref=True) evaluates the same function with every net in float64 on the float32
weights (std + 0.01 summed in float32: the constant the ONNX folds) and rounds the action to float32: the function export_onnx writes.
export_onnx: one stateless graph, input 'obs' [1, 111] float32 (the donors' fixed batch 1: MjlabPolicy runs it row by row, so the action
of a row never depends on the batch), output 'actions' [1, 24] float32, G + M3s + arm net (+ residual), compute dtype float64 (default)
or float32, donor metadata (the arm donor's, observation_names ending ',opp_keypoints' / ',opp_keypoints_blind') plus 'fb_hybrid'.
Blind flag (A9.2): the buffer blind_flag rides in every state dict, so every rsl_rl checkpoint states whether its arms were trained on
the 28 zeros; scripts/export_hybrid_fb.build asserts it equals --blind, and stored_blind() is what selection / test scripts read."""
import copy, json
import numpy as np, torch
from torch import nn
from torch.distributions import Normal
import fb_actions as AC

ACT_DIM, LEGS, ARMS, NL = AC.ACT_DIM, AC.LEGS, AC.ARMS, 14
EPS, HID, RES_HID = 1e-2, (512, 256, 128), (256, 128)        # rsl_rl EmpiricalNormalization eps; run-3 actor; plan 2.5 residual
LEG_STD0, LEG_STD_MAX = 0.15, 0.35
G_PT = "results/policy_mjlab_guard_footwork_r/model_7496.pt"
ARM_PT = dict(sighted="results/policy_mjlab_spar3_a3s/model_8995.pt", blind="results/policy_mjlab_spar3_blind3/model_8995.pt")
ARM_ONNX = dict(sighted="results/policy_mjlab_spar3_a3s/policy.onnx", blind="results/policy_mjlab_spar3_blind3/policy.onnx")
G_ONNX = "results/policy_mjlab_guard_footwork_r/policy.onnx"
OBS_END = dict(sighted=",opp_keypoints", blind=",opp_keypoints_blind")

class Normalizer(nn.Module):
    """rsl_rl 5.0.1 EmpiricalNormalization (buffers, forward, update), with a float64 evaluation of the same constants."""
    def __init__(self, n, eps=EPS):
        super().__init__(); self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, n)); self.register_buffer("_var", torch.ones(1, n)); self.register_buffer("_std", torch.ones(1, n))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
    def den(self): return self._std + self.eps               # float32 sum, as rsl_rl's forward and the exported Div constant
    def forward(self, x, dtype=None):
        if dtype is None: return (x - self._mean) / (self._std + self.eps)
        return (x.to(dtype) - self._mean.to(dtype)) / self.den().to(dtype)
    @torch.no_grad()
    def update(self, x):
        if not self.training: return
        n = x.shape[0]; self.count += n; rate = n / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True); mean_x = torch.mean(x, dim=0, keepdim=True); dm = mean_x - self._mean
        self._mean += rate * dm; self._var += rate * (var_x - self._var + dm * (mean_x - self._mean)); self._std = torch.sqrt(self._var)

def mlp(n_in, n_out, hidden):
    """rsl_rl MLP layout: Linear, ELU, ..., Linear (keys 0, 2, 4, ...)."""
    L, d = [], n_in
    for h in hidden: L += [nn.Linear(d, h), nn.ELU()]; d = h
    return nn.Sequential(*L, nn.Linear(d, n_out))

def run_mlp(seq, z, dtype=None):
    if dtype is None: return seq(z)
    for m in seq: z = nn.functional.linear(z, m.weight.to(dtype), m.bias.to(dtype)) if isinstance(m, nn.Linear) else nn.functional.elu(z)
    return z

class Net(nn.Module):
    """One rsl_rl actor model without its distribution: obs_normalizer + mlp."""
    def __init__(self, n_in, n_out=ACT_DIM, hidden=HID):
        super().__init__(); self.obs_normalizer = Normalizer(n_in); self.mlp = mlp(n_in, n_out, hidden); self.n_in = n_in
    def forward(self, x, dtype=None): return run_mlp(self.mlp, self.obs_normalizer(x, dtype), dtype)
    @classmethod
    def from_actor_sd(cls, sd):
        n = cls(sd["mlp.0.weight"].shape[1]); n.load_state_dict({k: v for k, v in sd.items() if not k.startswith("distribution.")}, strict=True); return n

def load_actor_sd(path):
    return torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]

class HybridActor(nn.Module):
    """legs + arms (+ residual); rsl_rl 5.0.1 actor-model interface (forward(obs, stochastic_output), get_output_log_prob, output_*,
    get_kl_divergence, update_normalization, reset, hidden-state no-ops). obs: [N, 111] tensor or a dict of groups (obs_groups)."""
    is_recurrent = False
    def __init__(self, g_sd, arm_sd, scale, clip=1e9, blind=False, obs_groups=("actor",)):
        super().__init__()
        self.legs = Net.from_actor_sd(g_sd).requires_grad_(False); self.arms = Net.from_actor_sd(arm_sd)
        assert self.legs.n_in == 83 and self.arms.n_in == 111, (self.legs.n_in, self.arms.n_in)
        self.std_param = nn.Parameter(arm_sd["distribution.std_param"].detach().clone())     # 24-d, the arm donor's
        self.register_buffer("scale", torch.as_tensor(np.asarray(scale, np.float64))); self.clip = float(clip)
        self.blind, self.width, self.obs_groups = bool(blind), ("zeros" if blind else 111), tuple(obs_groups)
        self.register_buffer("blind_flag", torch.tensor(bool(blind)))   # A9.2: stored in every checkpoint (actor_state_dict), asserted by export
        self.stage, self.residual, self.res_norm, self._plug, self._dist, self.r = "A", None, None, None, None, None

    @classmethod
    def from_checkpoints(cls, root, scale, clip=1e9, blind=False, g_pt=G_PT, arm_pt=None):
        from pathlib import Path
        root = Path(root); return cls(load_actor_sd(root / g_pt), load_actor_sd(root / (arm_pt or ARM_PT["blind" if blind else "sighted"])), scale, clip, blind)

    def train(self, mode=True):
        super().train(mode); self.legs.eval(); return self                 # G is frozen: its normalizer never updates

    # ---- stages
    def to_stage_b(self, seed=0, leg_std=LEG_STD0):
        """Stage B: a zero-output residual on legs and waist; leg std = leg_std (log 0.15 per plan 2.5); arm std continues."""
        assert self.stage == "A", "already stage B"
        dev = self.std_param.device
        with torch.random.fork_rng(devices=[]): torch.manual_seed(seed); res = mlp(111, NL, RES_HID)   # nn.Linear's default init, seeded
        nn.init.zeros_(res[-1].weight); nn.init.zeros_(res[-1].bias)
        self.residual = res.to(dev); self.res_norm = copy.deepcopy(self.arms.obs_normalizer)
        with torch.no_grad(): self.std_param[:NL] = leg_std
        self.stage = "B"; return self

    def trainable(self):
        """name -> parameter of the parameters a stage trains (the optimizer's set): arm net + std (A); + residual (B)."""
        return {k: p for k, p in self.named_parameters() if p.requires_grad and not k.startswith("legs.")}

    # ---- composition
    def plug(self, legs=None, arms=None):
        """Diagnostic: replace the nets by callables f32 rows -> [N, 24] (e.g. the donors' own ONNX sessions); None: the torch nets."""
        self._plug = None if legs is None else (legs, arms); return self

    def inputs(self, obs):
        """(G's input, the arm net's input), float32: f32(m3s in float64), f32(arm view)."""
        g = AC.m3s_input(obs.to(torch.float64), self.scale.to(obs.device), self.clip).to(torch.float32)
        return g, AC.arm_input(obs, self.width).to(torch.float32)

    def mean(self, obs, ref=False):
        """The deterministic action [N, 24] (float32 rows of the net dtype; ref: every net in float64, action rounded to float32)."""
        dt = torch.float64 if ref else None; g, u = self.inputs(obs)
        if self._plug is not None: L, A = self._plug[0](g), self._plug[1](u)
        else: L, A = self.legs(g, dt), self.arms(u, dt)
        legs = L[:, LEGS]
        if self.stage == "B":
            self.r = run_mlp(self.residual, self.res_norm(u, dt), dt); legs = legs + self.r
        out = torch.cat([legs, A[:, ARMS]], 1); return out.to(torch.float32) if ref else out

    def _cat(self, obs): return obs if torch.is_tensor(obs) else torch.cat([obs[g] for g in self.obs_groups], -1)

    # ---- distribution (ArmOnlyGaussian in A, Gaussian over 24 in B)
    def dims(self): return ARMS if self.stage == "A" else slice(0, ACT_DIM)
    def std(self):
        s = self.std_param
        return s if self.stage == "A" else torch.cat([s[:NL].clamp(max=LEG_STD_MAX), s[NL:]])

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mu = self.mean(self._cat(obs))
        if not stochastic_output: return mu
        d = self.dims(); self._mu, self._sd = mu, self.std().expand_as(mu); self._dist = Normal(mu[:, d], self._sd[:, d], validate_args=False)
        a = mu.detach().clone(); a[:, d] = self._dist.sample(); return a

    def get_output_log_prob(self, outputs): return self._dist.log_prob(outputs[:, self.dims()]).sum(-1)
    @property
    def output_entropy(self): return self._dist.entropy().sum(-1)
    @property
    def output_mean(self): return self._mu
    @property
    def output_std(self): return self._sd
    @property
    def output_distribution_params(self): return (self._mu, self._sd)
    def get_kl_divergence(self, old, new):
        d = self.dims(); o, n = Normal(old[0][:, d], old[1][:, d]), Normal(new[0][:, d], new[1][:, d])
        return torch.distributions.kl_divergence(o, n).sum(-1)

    def res_l2(self):
        """The anchor term's per-env value |r|^2 on the residual's mean from the last forward (0 in stage A)."""
        assert self.stage == "B" and self.r is not None, "res_l2: stage B, after a forward"
        return (self.r ** 2).sum(-1)

    def update_normalization(self, obs):
        u = self.inputs(self._cat(obs))[1]; self.arms.obs_normalizer.update(u)
        if self.res_norm is not None: self.res_norm.update(u)

    def reset(self, dones=None, hidden_state=None): pass
    def get_hidden_state(self): return None
    def detach_hidden_state(self, dones=None): pass

# ---- export
def stored_blind(sd):
    """The blind flag an actor state dict carries (A9.2); a state dict without it (pre-A9 or not a HybridActor) is refused."""
    assert "blind_flag" in sd, "no stored blind flag (A9.2): not a fullbody HybridActor checkpoint"
    return bool(sd["blind_flag"])

def _np(t, dt): return t.detach().cpu().numpy().astype(dt)

def onnx_model(actor, meta, dtype="float64", ir_version=None, opset=18):
    """The actor's deterministic mean as one ONNX graph (see the module doc); meta: metadata_props (dict, str values)."""
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as TP
    D, ndt = (TP.DOUBLE, np.float64) if dtype == "float64" else (TP.FLOAT, np.float32)
    nodes, inits, k = [], [], [0]
    def c(name, arr): inits.append(nh.from_array(np.asarray(arr), name)); return name
    def n(op, ins, **kw): k[0] += 1; o = f"{op.lower()}_{k[0]}"; nodes.append(h.make_node(op, ins, [o], **kw)); return o
    def sl(x, a, b): return n("Slice", [x, c(f"s{a}_{b}_{k[0]}", np.array([a], np.int64)), c(f"e{a}_{b}_{k[0]}", np.array([b], np.int64)), c(f"ax_{k[0]}", np.array([1], np.int64))])
    def to(x, t): return n("Cast", [x], to=t)
    def net(x, mod, norm, tag):
        z = n("Div", [n("Sub", [x, c(f"{tag}_mean", _np(norm._mean, ndt))]), c(f"{tag}_den", _np(norm.den(), ndt))])
        lin = [m for m in mod if isinstance(m, nn.Linear)]
        for i, m in enumerate(lin):
            z = n("Gemm", [z, c(f"{tag}_W{i}", _np(m.weight, ndt)), c(f"{tag}_b{i}", _np(m.bias, ndt))], transB=1)
            if i < len(lin) - 1: z = n("Elu", [z], alpha=1.0)
        return z
    x64 = to("obs", TP.DOUBLE)                                              # m3s in float64, rounded to float32 (P1)
    m3 = n("Clip", [n("Div", [sl(x64, 25, 35), c("m3s_scale", _np(actor.scale[ARMS].reshape(1, -1), np.float64))]),
                    c("m3s_lo", np.array(-actor.clip, np.float64)), c("m3s_hi", np.array(actor.clip, np.float64))])
    g = to(n("Concat", [sl(x64, 0, 73), m3], axis=1), TP.FLOAT)
    u = "obs" if not actor.blind else n("Concat", [sl("obs", 0, 83), c("kp_zeros", np.zeros((1, 28), np.float32))], axis=1)
    if D == TP.DOUBLE: g, u = to(g, D), to(u, D)
    L, A = net(g, actor.legs.mlp, actor.legs.obs_normalizer, "legs"), net(u, actor.arms.mlp, actor.arms.obs_normalizer, "arms")
    legs = sl(L, 0, NL)
    if actor.stage == "B": legs = n("Add", [legs, net(u, actor.residual, actor.res_norm, "res")])
    out = n("Concat", [legs, sl(A, 14, 24)], axis=1)
    nodes.append(h.make_node("Cast", [out], ["actions"], to=TP.FLOAT))
    gr = h.make_graph(nodes, "fb_hybrid", [h.make_tensor_value_info("obs", TP.FLOAT, [1, 111])], [h.make_tensor_value_info("actions", TP.FLOAT, [1, ACT_DIM])], inits)
    m = h.make_model(gr, opset_imports=[h.make_opsetid("", opset)], producer_name="r1-sparring-gym hybrid_actor")
    if ir_version: m.ir_version = ir_version
    for kk, v in meta.items(): e = m.metadata_props.add(); e.key, e.value = kk, str(v)
    onnx.checker.check_model(m); return m

def donor_meta(arm_onnx):
    """(metadata_props, ir_version) of the arm donor's ONNX (A3s / blind3: G's joints, gains, home and scale; its observation_names)."""
    import onnx
    m = onnx.load(str(arm_onnx)); return {p.key: p.value for p in m.metadata_props}, m.ir_version

def export_onnx(actor, path, arm_onnx, info=None, dtype="float64"):
    """Write the actor's ONNX with the donor's metadata (+ 'fb_hybrid': stage, blind, dtype, scale, clip, info). Returns the path."""
    import onnx
    meta, irv = donor_meta(arm_onnx); assert meta["observation_names"].endswith(OBS_END["blind" if actor.blind else "sighted"]), meta["observation_names"]
    meta["fb_hybrid"] = json.dumps(dict(stage=actor.stage, blind=actor.blind, dtype=dtype, scale=[float(s) for s in actor.scale], clip=actor.clip, **(info or {})), sort_keys=True)
    onnx.save(onnx_model(actor, meta, dtype, irv), str(path)); return path
