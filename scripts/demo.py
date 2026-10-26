"""Headless demo: Unitree R1 vs a second, physical R1 replaying a held-out KungFuAthleteBot clip (retargeted G1->R1),
plus three scripted side/back pushes. PD stand (left) vs RL policy (right).
What you see is what hits the robot: yellow spheres = real opponent contacts (size ~ force), orange arrows = scripted pushes.
Writes media/demo.mp4, media/demo.gif, media/still_*.png and results/demo_contacts.json (per-body contact forces).
Rendering needs OpenGL: MUJOCO_GL=egl on a headless Linux box, glfw (default) on a desktop."""
import os, sys, json, argparse
if sys.platform.startswith("linux"): os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from r1gym.model import build_spec, load_cfg
from r1gym.scenario import demo_env, DEMO_CLIP, DEMO_DISTANCE, DEMO_SECONDS
from r1gym.contacts import body_contacts, per_body_force
from r1gym.policy import PDStand, TorchPolicy

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def add_geom(scn, kind, rgba, size=None, a=None, b=None, pos=None, width=0.02):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array(size if size is not None else [0.0, 0.0, 0.0], float), np.array(pos if pos is not None else [0.0, 0.0, 0.0], float),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    if a is not None: mujoco.mjv_connector(g, kind, width, np.array(a, float), np.array(b, float))
    scn.ngeom += 1

def overlay(scn, m, e):
    """Real opponent->R1 contacts as yellow spheres; active scripted hits as orange arrows pointing into the hit point."""
    for b1, b2, f, pos in body_contacts(m, e.d):
        if (b1.startswith("opp_") != b2.startswith("opp_")) and "world" not in (b1, b2) and f > 1.0:
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [1.0, 0.85, 0.1, 0.85], size=[0.02 + min(f, 600.0) / 600.0 * 0.06] * 3, pos=pos)
    for h in e.hits:
        if h.t0 <= e.d.time < h.t0 + h.duration + 0.15:
            bid = m.body(h.body).id; R = e.d.xmat[bid].reshape(3, 3); p = e.d.xpos[bid] + R @ h.offset
            u = h.force / (np.linalg.norm(h.force) + 1e-9); L = 0.15 + 0.03 * h.impulse
            add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [1.0, 0.45, 0.1, 1.0], a=p - u * L, b=p, width=0.025)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--left", default="PD stand (baseline)=PD", help="label=PD or label=path/to/policy.pt")
    ap.add_argument("--right", default="RL hit-robust policy (v2)=results/policy/policy.pt"); ap.add_argument("--seconds", type=float, default=DEMO_SECONDS)
    ap.add_argument("--clip", default=DEMO_CLIP); ap.add_argument("--distance", type=float, default=DEMO_DISTANCE)
    ap.add_argument("--w", type=int, default=640); ap.add_argument("--h", type=int, default=480); ap.add_argument("--name", default="demo"); a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; (root / "media").mkdir(exist_ok=True)
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile()
    labels = [x.split("=", 1)[0] for x in (a.left, a.right)]
    ctrls = [PDStand() if p == "PD" else TorchPolicy(p) for p in (x.split("=", 1)[1] for x in (a.left, a.right))]
    envs = []; rens = []
    for c in ctrls:
        e, o = demo_env(m, cfg, distance=a.distance, clip=a.clip, seconds=a.seconds); e.contact_log = []
        envs.append([e, o, None]); rens.append(mujoco.Renderer(m, a.h, a.w))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [a.distance / 2, 0, 0.6]; cam.distance = 2.6; cam.azimuth = 100; cam.elevation = -8   # side-on
    fps = 25; every = int(round(1 / (fps * envs[0][0].dt))); frames = []; F = font(22); Fs = font(17); Ft = font(13)
    for t in range(int(a.seconds / envs[0][0].dt)):
        for c, x in zip(ctrls, envs):
            e = x[0]
            act = c.act(x[1][None])[0] if x[2] is None else np.zeros(24)   # after a fall: zero residual, physics keeps going
            o, r, d, info = e.step(act); x[1] = o
            if info["fallen"] and x[2] is None: x[2] = e.t * e.dt
        if t % every == 0:
            row = []
            for k, (c, x, rn) in enumerate(zip(ctrls, envs, rens)):
                e = x[0]; rn.update_scene(e.d, cam); overlay(rn.scene, m, e); img = Image.fromarray(rn.render()); dr = ImageDraw.Draw(img)
                dr.text((14, 10), "Unitree R1 · " + labels[k], font=F, fill=(255, 255, 255))
                f = sum(per_body_force(m, e.d, e.r.r1_bodies).values())
                if f > 1.0: dr.text((14, 40), f"OPPONENT CONTACT {f:.0f} N", font=Fs, fill=(255, 220, 40))
                act = [h for h in e.hits if h.t0 <= e.d.time < h.t0 + h.duration + 0.25]
                if act: h = act[0]; dr.text((14, 62), f"SCRIPTED {h.kind.upper()} {h.impulse:.0f} N·s → {h.body.replace('_link', '')}", font=Fs, fill=(255, 140, 50))
                dr.text((14, a.h - 52), "silver: Unitree R1 (24-DoF, official MJCF)  |  orange: physical R1 opponent, same motors", font=Ft, fill=(200, 200, 210))
                dr.text((14, a.h - 30), f"t={e.d.time:4.1f}s  obs delay 40 ms  " + ("FELL at %.1fs" % x[2] if x[2] else "standing"), font=Fs,
                        fill=(255, 90, 90) if x[2] else (140, 255, 140))
                row.append(np.asarray(img))
            frames.append(np.concatenate(row, 1))
    out = root / "media" / f"{a.name}.mp4"; imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for s in [3.1, 5.0, 8.6]:
        i = min(len(frames) - 1, int(s * fps)); Image.fromarray(frames[i]).save(root / "media" / f"still_{s:.1f}s.png")
    imageio.mimwrite(root / "media" / f"{a.name}.gif", [np.asarray(Image.fromarray(f).resize((640, 240))) for f in frames[::2]], fps=fps // 2, loop=0)
    json.dump({c.name if hasattr(c, "name") else labels[k]: {"fell_at_s": x[2], "contacts": x[0].contact_log} for k, (c, x) in enumerate(zip(ctrls, envs))},
              open(root / "results" / f"{a.name}_contacts.json", "w"))
    print("wrote", out, len(frames), "frames;", {l: x[2] for l, x in zip(labels, envs)})

if __name__ == "__main__": main()
