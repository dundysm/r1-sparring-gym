"""Controllers with one interface: act(obs_batch[N,78]) -> actions[N,24]. PD baseline = zero residual."""
from __future__ import annotations
import numpy as np
from pathlib import Path

class PDStand:
    name = "pd_stand"
    def act(self, obs): return np.zeros((len(obs), 24), np.float32)

class TorchPolicy:
    name = "rl_policy"
    def __init__(self, path):
        import torch, sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train import Actor
        ck = torch.load(path, weights_only=False); self.actor = Actor(); self.actor.load_state_dict(ck["actor"]); self.actor.eval()
        self.mean, self.std = ck["mean"].astype(np.float32), np.sqrt(ck["var"] + 1e-8).astype(np.float32); self.torch = torch
    def act(self, obs):
        o = np.clip((obs - self.mean) / self.std, -5, 5).astype(np.float32)
        with self.torch.no_grad(): return self.actor(self.torch.from_numpy(o)).numpy()

class OnnxPolicy:
    """Deployment path: normaliser is baked into the graph; input raw obs [N,78] float32."""
    name = "rl_policy_onnx"
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    def act(self, obs): return self.s.run(None, {"obs": obs.astype(np.float32)})[0]

class MjlabPolicy:
    """unitree_rl_mjlab velocity policy (ONNX with mjlab metadata), run under configs/r1_mjlab.yaml. The command is
    zero (stand / spar in place). Checks the ONNX metadata against this gym's contract before use. kp_mode: None (83-d),
    "obs" or "zeros" (a widened sparring defender, obs_dim 111: the caller appends the keypoint block or 28 zeros)."""
    name = "mjlab_policy"
    def __init__(self, path):
        import json, onnx, onnxruntime as ort
        from .model import CTRL_JOINTS, HOME, load_cfg, action_scale
        from .keypoints import OBS_NAMES, OBS_NAMES_SIGHTED, OBS_NAMES_BLIND, PROPRIO_DIM, KP_DIM
        from . import CONFIGS
        self.cfg_path = CONFIGS / "r1_mjlab.yaml"; cfg = load_cfg(self.cfg_path)
        meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
        assert meta["joint_names"].split(",") == CTRL_JOINTS, "joint order differs from the gym"
        self.cfg_overrides = {"home": [float(x) for x in meta["default_joint_pos"].split(",")]}   # e.g. a guard pose; HOME otherwise
        g = Path(path).with_name("gym.json")   # per-policy gym settings measured for this policy (e.g. face_opponent)
        if g.exists(): self.cfg_overrides.update({k: v for k, v in json.load(open(g)).items() if not k.startswith("_")})
        assert np.allclose([float(x) for x in meta["action_scale"].split(",")], action_scale(cfg), atol=2e-3)
        # the 83 of every shipped policy, or a sparring defender's widened 111 (docs/SPARRING_M1M2.md 1.4): kp_mode "obs" takes
        # the 28-d opponent keypoint block (r1gym/keypoints.py) after the 83, "zeros" (the blind twin) 28 zeros there
        kp = {OBS_NAMES: None, OBS_NAMES_SIGHTED: "obs", OBS_NAMES_BLIND: "zeros"}
        assert meta["observation_names"] in kp, f"observation_names {meta['observation_names']!r}: not this gym's contract"
        self.kp_mode = kp[meta["observation_names"]]; self.obs_dim = PROPRIO_DIM + (KP_DIM if self.kp_mode else 0)
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]); self.inp = self.s.get_inputs()[0].name
        self.batch1 = self.s.get_inputs()[0].shape[0] == 1
        w = self.s.get_inputs()[0].shape[-1]
        if self.kp_mode: assert w in (self.obs_dim, None) or isinstance(w, str), f"widened policy's input is {w} wide, not {self.obs_dim}"
    def act(self, obs):
        if self.kp_mode: assert obs.shape[-1] == self.obs_dim, f"widened policy: rows must be {self.obs_dim}-d (83 + keypoints), got {obs.shape[-1]}"
        obs = obs.astype(np.float32)
        if self.batch1: return np.concatenate([self.s.run(None, {self.inp: o[None]})[0] for o in obs])
        return self.s.run(None, {self.inp: obs})[0]

def load_controller(spec):
    """'label=PD' | 'label=path/policy.pt' (residual) | 'label=path/policy.onnx' (mjlab metadata -> MjlabPolicy)."""
    lab, path = spec.split("=", 1)
    c = PDStand() if path == "PD" else MjlabPolicy(path) if path.endswith(".onnx") else TorchPolicy(path)
    c.name = lab; return c

def controller_cfg(ctrl):
    from .model import load_cfg
    return {**load_cfg(getattr(ctrl, "cfg_path", None)), **getattr(ctrl, "cfg_overrides", {})}
