"""Does a stepping policy turn (and step) both ways in mjlab (MuJoCo Warp), where it trained? Holds fixed velocity commands
for 300 control steps (no pushes, no standing envs) and prints the mean measured rates after step 100. Run on the GPU box from
a unitree_rl_mjlab checkout with r1_spar_task.py in /root:
  python turn_check_mjlab.py <policy.onnx> [guard] [--reps 8] [--seed N] [--key <name> --out turn_check_mjlab.json]
Each command runs in --reps envs. On the flat plane the reset randomisation (xy +-0.5 m, yaw) changes neither the dynamics nor
the observations, so the reps differ by the training randomisation Unitree's play cfg keeps (foot friction U(0.3, 1.6), encoder
bias +-0.015 rad per joint, torso COM +-5 cm), drawn per env and fresh each run unless --seed is set. No obs noise (play mode).
Rates are the mean over the reps that didn't fall, with the spread across reps. A rep that terminated (and auto-reset) is counted in "falls" and left out.
--key merges that policy's rows into --out. results/turn_check_mjlab.json: see its "about"."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "/root"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import mjlab.tasks, src.tasks  # noqa: F401
import r1_spar_task as t
import onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
A = sys.argv[2:]; R = int(A[A.index("--reps") + 1]) if "--reps" in A else 8
make = t._guard(t._cfg(2)) if "guard" in A else t._cfg(2); cfg = make(play=True)
if "--seed" in A: cfg.seed = int(A[A.index("--seed") + 1])
C = torch.tensor([[0, 0, -0.8], [0, 0, -0.4], [0, 0, 0.0], [0, 0, 0.4], [0, 0, 0.8], [0.5, 0, 0], [0, 0.3, 0], [0, -0.3, 0], [-0.3, 0, 0]], dtype=torch.float32)
T = C.repeat_interleave(R, 0)   # env i runs command i // R
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
w, v = np.mean(w, 0), np.mean(v, 0)   # per env
rows = []
for j, c in enumerate(C.numpy()):
    idx = [i for i in range(j * R, (j + 1) * R) if not fell[i]]; nf = R - len(idx)
    m = lambda x: round(float(np.mean(x[idx])), 2) if idx else None
    sd = round(float(np.std(w[idx])), 2) if idx else None
    rows.append(dict(command_vx_vy_wz=[round(float(x), 2) for x in c], measured_wz=m(w), measured_vx=m(v[:, 0]), measured_vy=m(v[:, 1]), wz_std_over_reps=sd, reps=R, falls=nf))
    print(f"cmd {c} -> wz {rows[-1]['measured_wz']} (sd {sd}) vx {rows[-1]['measured_vx']} vy {rows[-1]['measured_vy']}  falls {nf}/{R}")
if "--key" in A:
    path = A[A.index("--out") + 1] if "--out" in A else "turn_check_mjlab.json"
    d = json.load(open(path)) if os.path.exists(path) else {"about": __doc__, "policies": {}}
    d["policies"][A[A.index("--key") + 1]] = dict(policy=sys.argv[1] + (" (guard)" if "guard" in A else ""), rows=rows)
    json.dump(d, open(path, "w"), indent=1)
