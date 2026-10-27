"""Parity check, mjlab side: run the opponent's tracking policy inside mjlab (the training sim) from fixed starts and record
when each run falls. The same starts in this repo: scripts/eval_tracker.py. results/opponent_tracker/parity_mjlab.json
holds both.

On the GPU box, from the unitree_rl_mjlab checkout, with r1_track_task.py and the motion index next to it:
  python parity_mjlab.py <run dir>/<run>.onnx /root/motion_index.json --out parity_mjlab_side.json
8 starts per clip spread over the clip (as eval_tracker.py), 10 s each, no terminations, steering drift off; a run falls when
the pelvis tilts past 60 deg or drops below 0.25 m."""
import sys, json, argparse
sys.path.insert(0, "/root"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
import onnxruntime as ort
import r1_track_task as t
from mjlab.envs import ManagerBasedRlEnv

ap = argparse.ArgumentParser(); ap.add_argument("onnx"); ap.add_argument("index"); ap.add_argument("--motion", default="motions/r1_kungfu.npz")
ap.add_argument("--out", default=""); a = ap.parse_args()
ix = json.load(open(a.index)); blend = ix["blend_frames"]
def frame(clip, f):   # same mapping as r1gym/opponent.py Tracker.frame
    c = ix["clips"][clip]; f %= c["frames"]; return c["start_frame"] + (c["frames"] + f if f <= blend else f)
runs = [(clip, s, frame(clip, int(round(s * c["frames"] / 50.0 / 8 * 50)))) for clip, c in ix["clips"].items() for s in range(8)]
cfg = t.env_cfg(play=True, motion_file=a.motion, relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True); cfg.commands["motion"].steer_speed = 0.0
cfg.scene.num_envs = len(runs); cfg.commands["motion"].joint_position_range = (0.0, 0.0); cfg.commands["motion"].sampling_mode = "uniform"
cfg.terminations = {"time_out": cfg.terminations["time_out"]}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); cmd = env.command_manager.get_term("motion")
start = torch.tensor([r[2] for r in runs], device=env.device)
cmd._uniform_sampling = lambda env_ids: cmd.time_steps.__setitem__(env_ids, start[env_ids])
obs, _ = env.reset(); sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
robot = env.scene["robot"]; pel = robot.body_names.index("pelvis"); fell = [None] * len(runs)
for k in range(500):
    o = obs["actor"].cpu().numpy().astype(np.float32); ts = cmd.time_steps.cpu().numpy()
    act = np.stack([sess.run(["actions"], {"obs": o[i:i + 1], "time_step": np.array([[ts[i]]], np.float32)})[0][0] for i in range(len(runs))])
    obs, *_ = env.step(torch.tensor(act, device=env.device))
    q = robot.data.body_link_quat_w[:, pel].cpu().numpy(); z = robot.data.body_link_pos_w[:, pel, 2].cpu().numpy(); up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    for i in range(len(runs)):
        if fell[i] is None and (up[i] < 0.5 or z[i] < 0.25): fell[i] = round((k + 1) * 0.02, 2)
out = {}
for (clip, s, f), x in zip(runs, fell): out.setdefault(clip, []).append(x)
for clip, xs in out.items(): print(clip, "stays up", f"{sum(x is None for x in xs)}/8", xs)
print("overall", sum(x is None for x in fell), "/", len(fell))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
