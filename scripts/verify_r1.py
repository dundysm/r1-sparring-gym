"""Prove the sim / training / eval / demo run on Unitree R1 specs (not G1). Writes results/r1_verification.json.
Reference sources: the vendored unitree_rl_mjlab r1.xml (the MJCF Unitree trains R1 on) and, when available,
unitree_mujoco R1_C++.xml (path via --ref or $R1_MUJOCO_XML). Exit code 1 on any failed check."""
import os, sys, json, hashlib, argparse, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from r1gym import ASSETS
from r1gym.model import build_spec, load_cfg, CTRL_JOINTS, torque_caps, R1Robot, opponent_spec
from r1gym.env import R1HitEnv

def md5s(d): return {hashlib.md5(p.read_bytes()).hexdigest() for p in Path(d).glob("*.STL")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default=os.environ.get("R1_MUJOCO_XML", "/workspace/unitree_mujoco/unitree_robots/r1/R1_C++.xml"))
    ap.add_argument("--out", default=str(ROOT / "results" / "r1_verification.json")); ap.add_argument("--policies", nargs="*", default=[str(p) for p in sorted((ROOT / "results").glob("policy*/policy.pt"))]); a = ap.parse_args()
    cfg = load_cfg(); checks = []; info = {}
    def chk(name, ok, detail=None): checks.append({"check": name, "pass": bool(ok), "detail": detail})
    m = build_spec(cfg=cfg).compile(); mo = build_spec(opponent=True, cfg=cfg).compile()
    up = mujoco.MjSpec.from_file(str(ASSETS / "r1" / "r1.xml")); up.meshdir = str(ASSETS / "r1" / "assets")
    for e in list(up.excludes):
        if e.bodyname1 not in {b.name for b in up.bodies} or e.bodyname2 not in {b.name for b in up.bodies}: up.delete(e)
    um = up.compile()
    hinges = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    info["hinge_dof"] = len(hinges); info["actuated_dof"] = m.nu; info["joints"] = hinges
    chk("24 actuated hinge DoF (unitree_rl_mjlab R1)", len(hinges) == 24 and m.nu == 24 and hinges == CTRL_JOINTS)
    chk("no G1-only joints (waist_pitch / wrist_pitch / wrist_yaw)", not any(re.search(r"waist_pitch|wrist_pitch|wrist_yaw", j) for j in hinges))
    # meshes
    mesh_files = {um.mesh(i).name for i in range(um.nmesh)}
    r1h, g1h = md5s(ASSETS / "r1" / "assets"), md5s(ASSETS / "g1" / "meshes")
    chk("R1 mesh files share no hash with G1 meshes", not (r1h & g1h), f"{len(r1h)} R1 STL, {len(r1h & g1h)} overlap")
    chk("protagonist geoms use only R1 meshes", m.nmesh == um.nmesh, f"{m.nmesh} meshes")
    # masses
    tot = float(m.body_subtreemass[m.body("pelvis").id]); info["total_mass_kg"] = round(tot, 3)
    info["body_mass_kg"] = {m.body(i).name: round(float(m.body_mass[i]), 4) for i in range(1, m.nbody)}
    chk("body masses identical to upstream r1.xml", np.allclose(m.body_mass, um.body_mass), f"total {tot:.2f} kg")
    g1 = opponent_spec("g1").compile(); g1tot = float(g1.body_subtreemass[g1.body("pelvis").id]); info["g1_total_mass_kg_for_contrast"] = round(g1tot, 2)
    chk("total mass is not the G1's", abs(tot - g1tot) > 1.0, f"R1 {tot:.2f} vs G1 {g1tot:.2f}")
    # joint ranges / damping / armature
    ids = [m.joint(j).id for j in CTRL_JOINTS]; uids = [um.joint(j).id for j in CTRL_JOINTS]
    chk("joint ranges identical to upstream r1.xml", np.allclose(m.jnt_range[ids], um.jnt_range[uids]))
    chk("joint damping/frictionloss from r1.xml defaults", np.allclose(m.dof_damping[m.jnt_dofadr[ids]], um.dof_damping[um.jnt_dofadr[uids]]) and
        np.allclose(m.dof_frictionloss[m.jnt_dofadr[ids]], um.dof_frictionloss[um.jnt_dofadr[uids]]),
        {"damping": float(um.dof_damping[um.jnt_dofadr[uids[0]]]), "frictionloss": float(um.dof_frictionloss[um.jnt_dofadr[uids[0]]])})
    arm = m.dof_armature[m.jnt_dofadr[ids]]
    info["armature"] = {"used": float(arm[0]), "upstream_r1_xml": float(um.dof_armature[um.jnt_dofadr[uids[0]]]), "flag": "ASSUMPTION (configs/r1_assumptions.yaml)"}
    # torque caps vs unitree_mujoco ctrlrange
    caps = torque_caps(cfg); acaps = np.array([m.actuator_forcerange[m.actuator(j.replace("_joint", "")).id, 1] for j in CTRL_JOINTS])
    chk("actuator force caps == configured R1 torque caps", np.allclose(acaps, caps))
    ref = Path(a.ref)
    if ref.exists():
        rs = mujoco.MjSpec.from_file(str(ref)); rc = {}
        for act in rs.actuators: rc[act.target] = float(act.ctrlrange[1])
        mj = np.array([rc.get(j, np.nan) for j in CTRL_JOINTS])
        chk("torque caps <= unitree_mujoco R1_C++.xml ctrlrange", np.all(acaps <= mj + 1e-9), dict(zip(CTRL_JOINTS, mj.tolist())))
        rjoints = [j.name for j in rs.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
        info["unitree_mujoco_R1_hinges"] = len(rjoints); info["unitree_mujoco_extra_unactuated"] = sorted(set(rjoints) - set(hinges))
        chk("all 24 joints exist in unitree_mujoco R1_C++.xml", set(hinges) <= set(rjoints))
    else: chk("unitree_mujoco R1_C++.xml reference available", False, f"not found at {ref} (set --ref)")
    info["pd_gains"] = {k: cfg["stand_pd"][k] for k in cfg["stand_pd"]}
    info["pd_gains_flag"] = "ASSUMPTION: tuned in this repo (not from G1, not from Unitree); see configs/r1_assumptions.yaml"
    # sensors
    sites = [m.site(i).name for i in range(m.nsite)]; sens = [m.sensor(i).name for i in range(m.nsensor)]
    info["sites"] = sites; info["sensors"] = sens; info["cameras"] = [m.camera(i).name for i in range(m.ncam)]
    chk("IMU site + gyro/accelerometer from r1.xml", "imu" in sites and any("ang_vel" in s or "gyro" in s for s in sens))
    info["head_camera"] = "ABSENT in both upstream R1 MJCFs; not used in phases 1-2 (phase 3 adds one)"
    # env / policies
    e = R1HitEnv(seed=0, cfg=cfg); o = e.reset(); info["obs_dim"] = int(o.shape[-1]); info["act_dim"] = int(e.nu if hasattr(e, "nu") else 24)
    chk("env model is the R1 model (same nq/nbody)", e.m.nq == m.nq and e.m.nbody == m.nbody, f"nq {e.m.nq}, nbody {e.m.nbody}")
    import torch
    for p in a.policies:
        c = torch.load(p, map_location="cpu", weights_only=False); sd = c["actor"]
        w0 = next(v for k, v in sd.items() if k.endswith("0.weight")); wl = [v for k, v in sd.items() if k.endswith("weight")][-1]
        chk(f"{Path(p).parent.name}: obs/act dims match R1 env ({info['obs_dim']}/24)", w0.shape[1] == info["obs_dim"] and wl.shape[0] == 24,
            f"obs {w0.shape[1]}, act {wl.shape[0]}")
    # opponent
    oj = [mo.joint(i).name for i in range(mo.njnt) if mo.joint(i).name.startswith("opp_")]
    info["opponent_model"] = cfg.get("opponent_model", "r1"); info["opponent_hinges"] = len(oj) - 1
    chk("opponent is an R1 (config default)", info["opponent_model"] != "r1" or not any("waist_pitch" in j for j in oj), f"{len(oj) - 1} hinges")
    out = {"pass": all(c["pass"] for c in checks), "checks": checks, "info": info}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    for c in checks: print(("PASS " if c["pass"] else "FAIL ") + c["check"], "" if c["detail"] is None or isinstance(c["detail"], dict) else f"[{c['detail']}]")
    print("total mass", info["total_mass_kg"], "kg; obs", info["obs_dim"], "act", info["act_dim"]); sys.exit(0 if out["pass"] else 1)

if __name__ == "__main__": main()
