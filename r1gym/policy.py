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
    zero (stand / spar in place). Checks the ONNX metadata against this gym's contract before use."""
    name = "mjlab_policy"
    def __init__(self, path):
        import onnx, onnxruntime as ort
        from .model import CTRL_JOINTS, HOME, load_cfg, action_scale
        from . import CONFIGS
        self.cfg_path = CONFIGS / "r1_mjlab.yaml"; cfg = load_cfg(self.cfg_path)
        meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
        assert meta["joint_names"].split(",") == CTRL_JOINTS, "joint order differs from the gym"
        assert np.allclose([float(x) for x in meta["default_joint_pos"].split(",")], HOME, atol=1e-3)
        assert np.allclose([float(x) for x in meta["action_scale"].split(",")], action_scale(cfg), atol=2e-3)
        assert meta["observation_names"] == "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]); self.inp = self.s.get_inputs()[0].name
        self.batch1 = self.s.get_inputs()[0].shape[0] == 1
    def act(self, obs):
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
    return load_cfg(getattr(ctrl, "cfg_path", None))
