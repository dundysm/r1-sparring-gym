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
