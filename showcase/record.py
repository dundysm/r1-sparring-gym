"""Record bouts for the Blender showcase (showcase/README.md): every physics substep (5 ms) of a bout, replayed exactly as
the headline eval ran it.

python showcase/record.py --raw results/eval_opponent/raw_test_hard.json --episode 17 --policy guard_fw=results/policy_mjlab_guard_footwork/policy.onnx --out showcase/rec/hero.npz
python showcase/record.py --demo --policy guard_fw=... --out showcase/rec/demo.npz      # the README demo scenario instead
python showcase/record.py --scene showcase/rec/scene                                      # meshes + geoms for Blender (once)

--raw/--episode replays episode i of a stored eval run (its clip, distance, start time, obs delay, hits and, in the hard
tier, randomised physics) with one controller, and checks the replay against the stored outcome for that controller
(fell_s, ring_out): a mismatch is an error, so a recorded bout is the bout the eval scored. --policy PD replays the PD stand.

The .npz holds, per substep (t = 0 at reset): body world poses `xpos` (T, nbody, 3) and `xquat` (T, nbody, 4, wxyz), the
R1 pelvis start `spot` (xy), robot-robot contacts (`con_t`, `con_pos`, `con_f`: time, world point, normal force > 5 N), the
scripted hits (`hits_json`), and `meta_json` (source, controller, fell_s, ring_out, opponent fell, ...). Body names in
`body_names`; the opponent's start with "opp_"."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
import r1gym.env as E
from r1gym.env import R1HitEnv
from r1gym.model import build_spec
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import PDStand, load_controller, controller_cfg
from r1gym.scenario import demo_env

RING_M = 1.0

class _Rec:
    """Stands in for the mujoco module inside r1gym.env, so every mj_step the env takes is recorded (the env steps 4x per
    control step; nothing else changes)."""
    def __init__(self): self.on = None
    def __getattr__(self, k): return getattr(mujoco, k)
    def mj_step(self, m, d, *a):
        mujoco.mj_step(m, d, *a)
        if self.on is not None: self.on(m, d)

def make_env(a, c, cfg):
    if a.demo:
        m = build_spec(opponent=True, cfg=cfg).compile(); e, o = demo_env(m, cfg); return e, o, dict(source="demo scenario (r1gym/scenario.py)"), None
    st = json.load(open(a.raw)); ep = st["episodes"][a.episode]; T = st["meta"]["seconds"]; dr = st["meta"]["tier_cfg"]["dr"]
    m = build_spec(opponent=True, cfg=cfg).compile()
    e = R1HitEnv(seed=ep.get("env_seed", ep["seed"]), cfg=cfg, model=m, episode_s=T, max_impulse=0, train=False, dr=dr)   # as scripts/eval_opponent.py run()
    e.opp = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg); e.opp_t0 = ep["t0"]; rng = np.random.default_rng(ep.get("hit_seed", ep["seed"]))
    hits = [make_hit(k, l, imp, t, y, rng) for k, l, imp, t, y in ep["hits"]]
    o = e.reset(hits=hits, delay=ep["delay"])
    stored = st["raw"].get(a.label)
    meta = dict(source=f"{a.raw} episode {a.episode}", tier=st["meta"]["tier"], split=st["meta"]["split"], clip=ep["clip"], distance=ep["distance"], t0=ep["t0"],
                delay=ep["delay"], dr=dr, seconds=T)
    return e, o, meta, (stored[a.episode] if stored else None)

def export_scene(out):
    """Visual meshes (group 2) of both robots, once: meshes/<name>.obj (MuJoCo's compiled vertices, mesh frame) and
    scene_geoms.json in the mj2blender schema (grokbot/mujoco2blender/blender_assets/_src/mj2blender.py)."""
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller("x=results/policy_mjlab_footwork/policy.onnx"); m = build_spec(opponent=True, cfg={**controller_cfg(c), "opponent_mode": "policy"}).compile()
    out = Path(out); (out / "meshes").mkdir(parents=True, exist_ok=True); geoms = []
    for i in range(m.ngeom):
        if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[i] != 2: continue
        mid = m.geom_dataid[i]; mname = m.mesh(mid).name.removeprefix("opp_"); f = out / "meshes" / f"{mname}.obj"
        if not f.exists():
            va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
            V = m.mesh_vert[va:va + vn]; F = m.mesh_face[fa:fa + fn]
            with open(f, "w") as fh:
                fh.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V)); fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))
        b = int(m.geom_bodyid[i]); geoms.append(dict(name=m.geom(i).name or f"geom{i}", type="mesh", mesh=mname, body=b, body_name=m.body(b).name,
                                                     size=[0, 0, 0], rgba=m.geom_rgba[i].round(4).tolist(), material=None,
                                                     pos=m.geom_pos[i].round(6).tolist(), quat=m.geom_quat[i].round(6).tolist()))
    json.dump(dict(geoms=geoms, body_names=[m.body(b).name for b in range(m.nbody)]), open(out / "scene_geoms.json", "w"), indent=1)
    print("exported", len(geoms), "visual geoms to", out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw"); ap.add_argument("--episode", type=int); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--policy", help="label=path.onnx|pt, or PD"); ap.add_argument("--label", help="the controller's row in the raw file (default 'RL <label>', 'pd_stand' for PD)")
    ap.add_argument("--after", type=float, default=0.0, help="seconds to keep simulating after a fall (picture only; outcome fields stay the eval's)")
    ap.add_argument("--out"); ap.add_argument("--scene"); a = ap.parse_args()
    if a.scene: export_scene(a.scene); return
    lab, path = (a.policy.split("=", 1) + [None])[:2] if a.policy != "PD" else ("pd_stand", None)
    c = PDStand() if path is None else load_controller(f"RL {lab}={path}")
    a.label = a.label or ("pd_stand" if path is None else f"RL {lab}")
    cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    E.mujoco = rec = _Rec()
    e, o, meta, stored = make_env(a, c, cfg)
    m, d = e.m, e.d; q = e.r.free_q; spot = d.qpos[q:q + 2].copy()
    xpos, xquat, ts, ct, cp, cf = [], [], [], [], [], []
    opp = set(range(m.nbody)) & {b for b in range(m.nbody) if m.body(b).name.startswith("opp_")}
    def on(m, d):
        xpos.append(d.xpos.astype(np.float32).copy()); xquat.append(d.xquat.astype(np.float32).copy()); ts.append(d.time)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c_ = d.contact[i]; b1, b2 = m.geom_bodyid[c_.geom1], m.geom_bodyid[c_.geom2]
            if b1 == 0 or b2 == 0 or ((b1 in opp) == (b2 in opp)): continue   # robot-vs-robot only
            mujoco.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) > 5.0: ct.append(d.time); cp.append(c_.pos.copy()); cf.append(abs(f6[0]))
    on(m, d); rec.on = on
    fell, maxr, opp_fell, after = None, 0.0, False, 0
    for _ in range(e.T):
        o, r, done, info = e.step(c.act(o[None])[0])
        if fell is not None:   # --after: keep simulating past the fall, for the picture only (the eval stops at the fall)
            after += 1
            if after >= round(a.after / e.dt): break
            continue
        xy = d.qpos[q:q + 2]; maxr = max(maxr, float(np.linalg.norm(xy - spot)))
        b = min(e.opp.bodies)
        if d.xmat[b][8] < 0.5 or d.xpos[b][2] < 0.25: opp_fell = True
        if info["fallen"]:
            fell = e.t * e.dt
            if a.after <= 0: break
    rec.on = None
    meta.update(controller=a.label, policy=path, fell_s=fell, ring_out=maxr > RING_M, max_disp_m=round(maxr, 3), opp_fell=opp_fell, substep_s=float(m.opt.timestep),
                after_fall_s=round(after * e.dt, 3))
    if stored is not None:   # the replay must be the bout the eval scored
        ok = stored["fell_s"] == fell and stored["ring_out"] == (maxr > RING_M)
        meta["matches_stored"] = ok; meta["stored"] = stored
        if not ok: raise SystemExit(f"replay differs from the stored eval row: stored {stored}, replay fell_s={fell} ring_out={maxr > RING_M}")
    hits = [dict(kind=h.kind, body=h.body, t0=h.t0, duration=h.duration, force=h.force.tolist(), offset=h.offset.tolist(), impulse=h.impulse) for h in e.hits]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, t=np.array(ts), xpos=np.array(xpos), xquat=np.array(xquat), spot=spot, body_names=np.array([m.body(b).name for b in range(m.nbody)]),
                        con_t=np.array(ct), con_pos=np.array(cp).reshape(-1, 3), con_f=np.array(cf), hits_json=json.dumps(hits), meta_json=json.dumps(meta))
    print("recorded", a.out, len(ts), "substeps;", json.dumps(meta))

if __name__ == "__main__": main()
