"""Deployment contract for the balance policy: everything outside the ONNX graph that the robot-side loop must reproduce.

The ONNX file holds only the MLP and the observation normaliser. The joint order, the observation layout and scales,
the action filter, the action -> joint-target mapping and the PD gains live here, are exported next to the ONNX as
deploy_contract.json (scripts/export_onnx.py), and are checked against the training env step by step in
tests/test_gym.py::test_deploy_loop_matches_env."""
from __future__ import annotations
import json
import numpy as np
from .model import CTRL_JOINTS, HOME, load_cfg, torque_caps, _group

def contract(cfg=None, gains="stand_pd"):
    cfg = cfg or load_cfg()
    kp = [cfg[gains][_group(j)]["kp"] for j in CTRL_JOINTS]; kd = [cfg[gains][_group(j)]["kd"] for j in CTRL_JOINTS]
    return {
        "control_hz": round(1.0 / cfg["model"]["control_dt"]),
        "joint_order": CTRL_JOINTS,                       # Unitree deploy order (unitree_rl_mjlab deploy.yaml joint_ids_map)
        "home_rad": HOME.tolist(),
        "obs_layout": [                                   # concatenated in this order; 78 floats, float32
            {"name": "base_ang_vel", "size": 3, "source": "IMU gyro, pelvis frame [rad/s]", "scale": 1.0},
            {"name": "projected_gravity", "size": 3, "source": "R_pelvis^T @ (0, 0, -1) from the IMU quaternion", "scale": 1.0},
            {"name": "joint_pos_minus_home", "size": 24, "source": "q - home_rad, joint_order [rad]", "scale": 1.0},
            {"name": "joint_vel", "size": 24, "source": "dq, joint_order [rad/s]", "scale": 0.05},
            {"name": "last_action", "size": 24, "source": "previous raw policy output (after clip, before filter)", "scale": 1.0},
        ],
        "action": {"clip": 4.0, "ema_alpha": cfg.get("action_filter", 0.3), "scale_rad": cfg["action_scale"],
                   "target": "q_target = home_rad + scale_rad * f,  f <- ema_alpha * f + (1 - ema_alpha) * clip(a)  (f starts at 0)"},
        "pd": {"kp": kp, "kd": kd, "torque_cap_nm": torque_caps(cfg).tolist(), "law": "tau = kp * (q_target - q) - kd * dq, clipped to the cap"},
        "trained_with": {"obs_delay_ms": [0, 40], "notes": "stand_pd gains (kp 300 on legs/ankles) are 3-7.5x Unitree's deploy gains and "
                         "have not been validated on hardware; see docs/MODEL_AUDIT.md"},
    }

class DeployLoop:
    """Reference robot-side loop: raw sensors in, joint targets out. `policy` maps obs[1, 78] -> action[1, 24]
    (r1gym.policy.OnnxPolicy or TorchPolicy)."""
    def __init__(self, policy, c=None):
        self.p = policy; self.c = c or contract(); self.home = np.array(self.c["home_rad"], np.float32)
        self.last = np.zeros(24, np.float32); self.f = np.zeros(24)
    def obs(self, gyro, quat_wxyz, q, dq):
        w, x, y, z = quat_wxyz
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return np.concatenate([gyro, R.T @ np.array([0, 0, -1.0]), np.asarray(q) - self.home, 0.05 * np.asarray(dq), self.last]).astype(np.float32)
    def step(self, gyro, quat_wxyz, q, dq):
        a = np.clip(self.p.act(self.obs(gyro, quat_wxyz, q, dq)[None])[0], -self.c["action"]["clip"], self.c["action"]["clip"]).astype(np.float32)
        al = self.c["action"]["ema_alpha"]; self.f = al * self.f + (1 - al) * a; self.last = a
        return self.home + self.c["action"]["scale_rad"] * self.f

def write(path, cfg=None):
    json.dump(contract(cfg), open(path, "w"), indent=1)
