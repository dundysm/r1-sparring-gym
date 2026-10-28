"""Does a stepping policy turn (and step) both ways in mjlab (MuJoCo Warp), where it trained? Holds fixed velocity commands
for 300 control steps (no pushes, no standing envs) and prints the mean measured rates after step 100. Run on the GPU box from
a unitree_rl_mjlab checkout with r1_spar_task.py in /root:
  python turn_check_mjlab.py <policy.onnx> [guard] [--key <name> --out turn_check_mjlab.json]
--key merges that policy's rows into --out. One env per command, no seeds. A row with "fell": true terminated (and
auto-reset) during the run, so its rates are not valid. results/turn_check_mjlab.json holds the four stepping policies' rows
from the 2026-10-27 run, copied from the printed output (this version, which writes the file and checks for falls, came after it)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "/root"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import mjlab.tasks, src.tasks  # noqa: F401
import r1_spar_task as t
import onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
make = t._guard(t._cfg(2)) if "guard" in sys.argv[2:] else t._cfg(2); cfg = make(play=True)
T = torch.tensor([[0, 0, -0.8], [0, 0, -0.4], [0, 0, 0.0], [0, 0, 0.4], [0, 0, 0.8], [0.5, 0, 0], [0, 0.3, 0], [0, -0.3, 0], [-0.3, 0, 0]], dtype=torch.float32)
cfg.scene.num_envs = len(T); tw = cfg.commands["twist"]; tw.heading_command = False; tw.ranges.heading = None; tw.rel_standing_envs = 0.0; tw.resampling_time_range = (1e9, 1e9)
cfg.events.pop("push_robot", None)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); cmd = env.command_manager.get_term("twist"); T = T.to(env.device)
obs, _ = env.reset(); s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"]); name = s.get_inputs()[0].name
robot = env.scene["robot"]; w, v = [], []; fell = np.zeros(len(T), bool)
for k in range(300):
    cmd.vel_command_b[:] = T
    o = obs["actor"].cpu().numpy().astype(np.float32)
    a = np.concatenate([s.run(None, {name: o[i:i + 1]})[0] for i in range(len(T))])
    out = env.step(torch.tensor(a, device=env.device)); obs = out[0]; fell |= (out[2] | out[3]).cpu().numpy().astype(bool)   # terminated | truncated
    if k >= 100: w.append(robot.data.root_link_ang_vel_w[:, 2].cpu().numpy()); v.append(robot.data.root_link_lin_vel_b[:, :2].cpu().numpy())
w, v = np.mean(w, 0), np.mean(v, 0)
rows = []
for c, ww, vv, f in zip(T.cpu().numpy(), w, v, fell):
    print(f"cmd {c} -> wz {ww:+.2f} vx {vv[0]:+.2f} vy {vv[1]:+.2f}" + ("  FELL (auto-reset; rates invalid)" if f else ""))
    rows.append(dict(command_vx_vy_wz=[round(float(x), 2) for x in c], measured_wz=round(float(ww), 2), measured_vx=round(float(vv[0]), 2), measured_vy=round(float(vv[1]), 2), fell=bool(f)))
A = sys.argv[2:]
if "--key" in A:
    path = A[A.index("--out") + 1] if "--out" in A else "turn_check_mjlab.json"
    d = json.load(open(path)) if os.path.exists(path) else {"about": __doc__, "policies": {}}
    d["policies"][A[A.index("--key") + 1]] = dict(policy=sys.argv[1] + (" (guard)" if "guard" in A else ""), rows=rows)
    json.dump(d, open(path, "w"), indent=1)
