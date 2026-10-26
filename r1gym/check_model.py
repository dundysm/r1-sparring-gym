"""Print and verify everything the sim assumes about the R1: joints, limits, torque caps, mass, stand test.
python -m r1gym.check_model  [--json results/model_check.json]"""
import sys, json, argparse
import numpy as np, mujoco
from .model import build_spec, R1Robot, load_cfg, CTRL_JOINTS, HOME, torque_caps

def stand(cfg, gains, seconds=5.0, noise=0.0, seed=0):
    m = build_spec(cfg=cfg, gains=gains).compile(); d = mujoco.MjData(m); r = R1Robot(m, cfg); r.reset(m, d)
    d.qpos[r.qadr] += np.random.default_rng(seed).normal(0, noise, 24); mujoco.mj_forward(m, d)
    for k in range(int(seconds / m.opt.timestep)):
        r.pd(d, HOME); mujoco.mj_step(m, d)
        if r.fallen(d): return False, k * m.opt.timestep
    return True, seconds

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    cfg = load_cfg(); m = build_spec(cfg=cfg).compile(); caps = torque_caps(cfg); rep = {"joints": []}
    print(f"{'joint':28s} {'range [rad]':>18s} {'cap mjlab':>9s} {'cap mujoco':>10s} {'cap used':>8s}")
    allc = {k: torque_caps({**cfg, 'torque_cap_policy': k}) for k in ("mjlab", "mujoco")}
    for i, j in enumerate(CTRL_JOINTS):
        jj = m.joint(j); lo, hi = m.jnt_range[jj.id]
        assert jj.limited or m.jnt_limited[jj.id], f"{j} has no limits"; assert lo <= HOME[i] <= hi, f"HOME outside range for {j}"
        print(f"{j:28s} [{lo:+.3f},{hi:+.3f}] {allc['mjlab'][i]:9.0f} {allc['mujoco'][i]:10.0f} {caps[i]:8.0f}")
        rep["joints"].append(dict(joint=j, range=[float(lo), float(hi)], cap=float(caps[i])))
    rep["mass_kg"] = float(m.body_subtreemass[m.body("pelvis").id]); rep["nu"] = m.nu
    for g in ("stand_pd", "pd"):
        ok, t = stand(cfg, g); rep[f"stand_{g}"] = {"stands_5s": ok, "t": t}
        print(f"stand test, gains={g:9s}: {'STANDS 5 s' if ok else f'FALLS at {t:.2f}s'}")
    print(f"total mass {rep['mass_kg']:.2f} kg, {m.nu} actuators, timestep {m.opt.timestep}, integrator implicitfast")
    if a.json: json.dump(rep, open(a.json, "w"), indent=1)

if __name__ == "__main__": main()
