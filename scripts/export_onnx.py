"""Export the actor (with the obs normaliser baked in) to ONNX and measure inference latency (onnxruntime, CPU, 1 thread)."""
import sys, time, json, argparse, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch, torch.nn as nn
from train import Actor
from r1gym.env import OBS_DIM

class Deploy(nn.Module):
    def __init__(s, actor, mean, var):
        super().__init__(); s.a = actor
        s.register_buffer("mean", torch.tensor(mean, dtype=torch.float32)); s.register_buffer("std", torch.tensor(np.sqrt(var + 1e-8), dtype=torch.float32))
    def forward(s, obs): return s.a(torch.clamp((obs - s.mean) / s.std, -5, 5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="results/policy/policy.pt"); ap.add_argument("--out", default="results/policy/policy.onnx"); a = ap.parse_args()
    ck = torch.load(a.ckpt, weights_only=False); act = Actor(); act.load_state_dict(ck["actor"]); act.eval()
    m = Deploy(act, ck["mean"], ck["var"]).eval(); x = torch.zeros(1, OBS_DIM)
    torch.onnx.export(m, x, a.out, input_names=["obs"], output_names=["action"], dynamic_axes={"obs": {0: "n"}, "action": {0: "n"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0); xs = rng.normal(size=(2000, 1, OBS_DIM)).astype(np.float32)
    err = float(np.abs(s.run(None, {"obs": xs[0]})[0] - m(torch.from_numpy(xs[0])).detach().numpy()).max())
    for i in range(200): s.run(None, {"obs": xs[i]})
    t = []
    for i in range(2000):
        t0 = time.perf_counter(); s.run(None, {"obs": xs[i]}); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e6
    xb = rng.normal(size=(4096, OBS_DIM)).astype(np.float32); t0 = time.perf_counter()
    for _ in range(20): s.run(None, {"obs": xb})
    batch_ms = (time.perf_counter() - t0) / 20 * 1e3
    import os
    res = {"onnx": a.out, "bytes": os.path.getsize(a.out), "max_abs_err_vs_torch": err, "latency_us_batch1": {"p50": round(float(np.percentile(t, 50)), 1),
           "p99": round(float(np.percentile(t, 99)), 1), "mean": round(float(t.mean()), 1)}, "batch4096_ms": round(batch_ms, 2),
           "hardware": f"{platform.processor() or platform.machine()} ({os.cpu_count()} vCPU box, 1 ORT thread, shared with other jobs)", "onnxruntime": ort.__version__}
    print(json.dumps(res, indent=1)); json.dump(res, open(Path(a.out).parent / "latency.json", "w"), indent=1)
    from r1gym.deploy import write; write(Path(a.out).parent / "deploy_contract.json")   # everything outside the graph

if __name__ == "__main__": main()
