"""Headless demo: Unitree R1 vs a second, physical R1 performing a KungFuAthleteBot clip (retargeted G1->R1) on its own
motors, steered to stay in front of it, plus three scripted side/back pushes. Default: the two shipped stepping policies
side by side (arms down, guard), each against its own copy of the opponent; --left none for one panel.
What you see is what hits the robot: yellow spheres = real opponent contacts (size ~ force), orange arrows = scripted pushes.
Writes media/demo.mp4, media/demo.gif, media/still_*.png and results/demo_contacts.json (per-body contact forces).
--headcam --offaxis 40: the R1 turning toward what its own head camera sees (phase 3, M1), with a head-camera inset ->
media/camera_turn.mp4 and .gif.
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
from r1gym.policy import PDStand, load_controller, controller_cfg

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

def headcam(a):
    """--headcam (phase 3, M1): one bout of the --right policy starting --offaxis deg off-axis, turning toward what its own head
    camera sees (perception: camera, the colour key; filter and deadband as preregistered for M1). Fight view with rays on the
    floor from the R1's pelvis (white: the opponent's true bearing, green: the camera estimate it acts on, grey: its heading),
    a head-camera inset (the frame in use, the detector's mask tinted, crosses where the estimate and the truth put the
    opponent's pelvis) and a strip chart of both bearings. Writes media/<name>.mp4/.gif (name camera_turn) and
    results/perception/<name>.json (the bout's numbers, run metadata)."""
    from r1gym.perception import run_meta
    from r1gym.perception.camera import HeadCameraSpec, mount_fk, project_pinhole
    from r1gym.perception.render_scene import RenderTwin
    from r1gym.perception.detect_colorkey import ColorKeyDetector
    from r1gym.perception.hub import PerceptHub
    from r1gym.perception.percept import attach
    root = Path(__file__).resolve().parents[1]; name = a.name if a.name != "demo" else "camera_turn"; W, H, CH = 960, 540, 148   # 688 rows: a multiple of the encoder's 8
    src = next(p for p in (root / "results/perception/preregistration_m1.json", root / "results/perception/m1_offaxis_val.json") if p.exists())
    tune = {k: json.load(open(src))["choice"][k] for k in ("percept_filter_s", "face_deadband")}; lab = a.right.split("=", 1)[0]; c = load_controller(a.right)
    cfg = {**controller_cfg(c), "opponent_mode": "policy", "face_opponent": True, "perception": "camera", **tune}; m = build_spec(opponent=True, cfg=cfg).compile()
    spec = HeadCameraSpec(gym_cfg=cfg); tw = RenderTwin(cfg, physics=m, spec=spec); det = ColorKeyDetector(spec, cfg); hub = PerceptHub(tw, det, keep=12); S = []
    e, o = demo_env(m, cfg, distance=a.distance, clip=a.clip, seconds=a.seconds, init_yaw=-np.radians(a.offaxis), setup=lambda e: S.append(attach(e, "camera", spec=spec, seed=a.seed, hub=hub)))
    s = S[0]; rn = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [a.distance / 2, 0, 0.7]; cam.distance, cam.azimuth, cam.elevation = 2.6, 100, -20
    opel = m.body("opp_pelvis").id; fps = 25; every = int(round(1 / (fps * e.dt))); F, Fs, Ft = font(22), font(16), font(13); frames, T, B, U, Z = [], [], [], [], []; fell = None
    yaw = lambda q: np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    def ray(scn, ang, length, rgba, z=0.03):
        p = e.d.xpos[e.r.pel].copy(); p[2] = z; d_ = np.array([np.cos(ang), np.sin(ang), 0.0]); add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, rgba, a=p, b=p + length * d_, width=0.012)
    def inset():
        if not s.delivered or id(s) not in hub.images: return Image.new("RGB", (320, 240))
        tc = s.delivered[-1][1].t_capture; got = [x for x in hub.images[id(s)] if abs(x[0] - tc) < 1e-9]
        if not got: return Image.new("RGB", (320, 240))
        _, img, sn = got[0]; mk = det.mask(img); vis = img.astype(float); vis[mk] = 0.55 * vis[mk] + 0.45 * np.array([60, 255, 90]); im = Image.fromarray(vis.astype(np.uint8)).resize((320, 240), Image.NEAREST)
        dr = ImageDraw.Draw(im); est = sn.slot[1]; tw.sync(sn.qpos); q = sn.qpos; Rp = np.zeros(9); mujoco.mju_quat2Mat(Rp, q[3:7]); Rp = Rp.reshape(3, 3)
        hy = yaw(q[3:7]); Hm = np.array([[np.cos(hy), np.sin(hy), 0], [-np.sin(hy), np.cos(hy), 0], [0, 0, 1]]); X = Hm @ (tw.d.xpos[opel] - q[:3])   # the truth, in the heading frame
        K = spec.K(2); pt, Rt = mount_fk(q[e.r.qadr], Rp.T @ [0, 0, -1.0], spec); ps, Rs = mount_fk(sn.q, sn.g, spec)
        marks = [((255, 255, 255), pt, Rt, X)]
        if est["presence"]: r_ = est["range"] if np.isfinite(est["range"]) else np.hypot(*X[:2]); marks.append(((60, 255, 90), ps, Rs, np.array([r_ * np.cos(est["bearing"]), r_ * np.sin(est["bearing"]), X[2]])))
        for col, p_, R_, P in marks:
            c_ = R_.T @ (P - p_)
            if c_[2] > 0.05:
                u, v = project_pinhole(c_, K); dr.line([(u - 9, v), (u + 9, v)], fill=col, width=3); dr.line([(u, v - 9), (u, v + 9)], fill=col, width=3)
        dr.rectangle([0, 0, 319, 239], outline=(200, 200, 210)); return im
    def chart():
        im = Image.new("RGB", (W, CH), (18, 20, 26)); dr = ImageDraw.Draw(im); x = lambda t: 50 + (W - 70) * t / a.seconds; y = lambda deg: CH / 2 - (CH / 2 - 14) * np.clip(deg, -60, 60) / 60
        for dg in (-45, 0, 45): dr.line([(50, y(dg)), (W - 20, y(dg))], fill=(70, 72, 80) if dg else (110, 112, 120)); dr.text((8, y(dg) - 8), f"{dg:+d}°" if dg else "0°", font=Ft, fill=(150, 150, 160))
        for vals, col in ((B, (255, 255, 255)), (U, (60, 255, 90))):
            pts = [(x(t), y(v)) for t, v in zip(T, vals) if v is not None]
            if len(pts) > 1: dr.line(pts, fill=col, width=2)
        dr.text((W - 470, 6), "bearing of the opponent:  white = true   green = camera estimate in use", font=Ft, fill=(200, 200, 210)); return im
    for k in range(int(a.seconds / e.dt) + 1):
        if k:
            o, r, d_, info = e.step(c.act(o[None])[0] if fell is None else np.zeros(24))
            if info["fallen"] and fell is None: fell = e.t * e.dt
        b_true = float(e.true_bearing()[0]); b_use = s.used[-1][1] if s.used else None; T.append(e.d.time); B.append(np.degrees(b_true)); U.append(None if b_use is None else np.degrees(b_use)); Z.append(float(e.command[2]))
        if k % every: continue
        p, op = e.d.xpos[e.r.pel], e.d.xpos[opel]; mid = (p + op) / 2; v = op[:2] - p[:2]; az = 100 + np.degrees(np.arctan2(v[1], v[0]))
        cam.azimuth += 0.08 * ((az - cam.azimuth + 180) % 360 - 180); cam.lookat[:2] += 0.08 * (mid[:2] - cam.lookat[:2]); cam.distance += 0.08 * (max(2.4, 1.9 + 1.3 * np.linalg.norm(v)) - cam.distance)
        rn.update_scene(e.d, cam); overlay(rn.scene, m, e); hy = yaw(e.d.qpos[3:7]); ray(rn.scene, hy, 0.45, [0.75, 0.78, 0.85, 0.9]); ray(rn.scene, hy + b_true, float(np.hypot(*v)), [1, 1, 1, 0.95])
        if b_use is not None: ray(rn.scene, hy + b_use, float(np.hypot(*v)), [0.25, 1.0, 0.35, 0.95], z=0.05)
        img = Image.fromarray(rn.render()); dr = ImageDraw.Draw(img); img.paste(inset(), (W - 334, 14))
        dr.text((14, 10), "Unitree R1 · " + lab, font=F, fill=(255, 255, 255)); dr.text((14, 40), "turns toward what its own head camera sees (sim, no learning)", font=Fs, fill=(255, 255, 255))
        dr.text((14, 62), f"true bearing {np.degrees(b_true):+5.1f}°   camera estimate " + ("  --  " if b_use is None else f"{np.degrees(b_use):+5.1f}°") + f"   turn command {e.command[2]:+.2f} rad/s", font=Fs, fill=(220, 225, 235))
        for i, t_ in enumerate([f"head camera (sim): 160x120, 30 Hz, pitch {spec.pitch_deg:g}°", f"{1e3 * s.queue.base:.0f} ms latency · colour key, no learning"]): dr.text((W - 334, 258 + 17 * i), t_, font=Ft, fill=(200, 200, 210))
        x_ = W - 334
        for t_, col in (("tint: what it sees   ", (200, 200, 210)), ("+ estimate   ", (60, 255, 90)), ("+ truth", (255, 255, 255))): dr.text((x_, 292), t_, font=Ft, fill=col); x_ += dr.textlength(t_, font=Ft)
        act = [h for h in e.hits if h.t0 <= e.d.time < h.t0 + h.duration + 0.25]
        if act: h = act[0]; dr.text((14, 86), f"SCRIPTED {h.kind.upper()} {h.impulse:.0f} N·s → {h.body.replace('_link', '')}", font=Fs, fill=(255, 140, 50))
        dr.text((14, H - 52), f"silver: Unitree R1 (official MJCF), starts {a.offaxis:+g}° off-axis  |  orange: R1 opponent on its own motors, no assist", font=Ft, fill=(200, 200, 210))
        dr.text((14, H - 30), f"t={e.d.time:4.1f}s  " + ("FELL at %.1fs" % fell if fell else "standing"), font=Fs, fill=(255, 90, 90) if fell else (140, 255, 140))
        frames.append(np.concatenate([np.asarray(img), np.asarray(chart())], 0))
    out = root / "media" / f"{name}.mp4"; imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    g = [Image.fromarray(f).resize((456, int(round(456 * f.shape[0] / f.shape[1])) // 2 * 2), Image.LANCZOS).quantize(colors=104, method=Image.Quantize.MEDIANCUT) for f in frames[::2]]   # < 5 MB with headroom
    g[0].save(root / "media" / f"{name}.gif", save_all=True, append_images=g[1:], duration=int(2000 / fps), loop=0, optimize=True)
    err = np.abs(B); late = err[np.array(T) > a.seconds - 6]; ttf = next((t for t, x in zip(T, err) if x <= 15), None)
    res = dict(about=headcam.__doc__, **run_meta(gl=tw.gl()), policy=a.right, clip=a.clip, distance=a.distance, offaxis_deg=a.offaxis, seconds=a.seconds, seed=a.seed, tune=tune, tune_from=src.relative_to(root).as_posix(),
               fell_s=fell, facing_err_last6_med_deg=round(float(np.median(late)), 2), ttf15_s=None if ttf is None else round(ttf, 2), percept=s.stats(), hub=hub.cost(),
               media_bytes={p: (root / "media" / p).stat().st_size for p in (f"{name}.mp4", f"{name}.gif")})   # bytes: no MB/MiB ambiguity
    json.dump(res, open(root / "results" / "perception" / f"{name}.json", "w"), indent=1); rn.close(); tw.close()
    assert max(res["media_bytes"].values()) < 5_000_000, res["media_bytes"]   # the design's 5 MB, decimal
    print("wrote", out, len(frames), "frames;", json.dumps({k: res[k] for k in ("fell_s", "facing_err_last6_med_deg", "ttf15_s", "media_bytes")}))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--left", default="Stepping base=results/policy_mjlab/policy.onnx", help="label=PD | label=path (.pt/.onnx) | none (single panel)")
    ap.add_argument("--right", default="Stepping base + guard=results/policy_mjlab_guard/policy.onnx"); ap.add_argument("--seconds", type=float, default=DEMO_SECONDS)
    ap.add_argument("--clip", default=DEMO_CLIP); ap.add_argument("--distance", type=float, default=DEMO_DISTANCE)
    ap.add_argument("--w", type=int, default=640); ap.add_argument("--h", type=int, default=400); ap.add_argument("--name", default="demo")
    ap.add_argument("--headcam", action="store_true", help="the --right policy turning on its own head camera (phase 3, M1)")
    ap.add_argument("--offaxis", type=float, default=40.0, help="--headcam: the opponent's starting bearing [deg]"); ap.add_argument("--seed", type=int, default=0, help="--headcam: perception seed"); a = ap.parse_args()
    if a.headcam: return headcam(a)
    root = Path(__file__).resolve().parents[1]; (root / "media").mkdir(exist_ok=True)
    specs = [a.right] if a.left == "none" else [a.left, a.right]   # --left none: one panel (the right controller), larger
    if a.left == "none": a.w, a.h = 960, 540
    labels = [x.split("=", 1)[0] for x in specs]
    ctrls = [load_controller(x) for x in specs]
    envs = []; rens = []; models = {}
    for c in ctrls:   # each controller runs under its own contract (residual or mjlab); the opponent is identical
        cfg = controller_cfg(c); key = str(getattr(c, "cfg_path", "default"))
        if key not in models: models[key] = build_spec(opponent=True, cfg=cfg).compile()
        m = models[key]; e, o = demo_env(m, cfg, distance=a.distance, clip=a.clip, seconds=a.seconds); e.contact_log = []
        envs.append([e, o, None]); rens.append(mujoco.Renderer(m, a.h, a.w))
    cams = []   # one fight camera per panel: side-on to the line between the two robots, centred and zoomed on them
    for _ in envs:
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [a.distance / 2, 0, 0.7]; cam.distance = 2.6
        cam.azimuth = 100; cam.elevation = -8; cams.append(cam)
    def follow(cam, e, alpha=0.08):
        p, o = e.d.xpos[e.r.pel], e.d.xpos[min(e.opp.bodies)]; mid = (p + o) / 2; v = o[:2] - p[:2]
        az = 100 + np.degrees(np.arctan2(v[1], v[0])); daz = (az - cam.azimuth + 180) % 360 - 180   # shortest turn
        cam.azimuth += alpha * daz; cam.lookat[:2] += alpha * (mid[:2] - cam.lookat[:2])
        cam.distance += alpha * (max(2.4, 1.9 + 1.3 * np.linalg.norm(v)) - cam.distance)
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
                e = x[0]; cam = cams[k]; follow(cam, e); rn.update_scene(e.d, cam); overlay(rn.scene, e.m, e); img = Image.fromarray(rn.render()); dr = ImageDraw.Draw(img)
                dr.text((14, 10), "Unitree R1 · " + labels[k], font=F, fill=(255, 255, 255))
                f = sum(per_body_force(e.m, e.d, e.r.r1_bodies).values())
                if f > 1.0: dr.text((14, 40), f"OPPONENT CONTACT {f:.0f} N", font=Fs, fill=(255, 220, 40))
                act = [h for h in e.hits if h.t0 <= e.d.time < h.t0 + h.duration + 0.25]
                if act: h = act[0]; dr.text((14, 62), f"SCRIPTED {h.kind.upper()} {h.impulse:.0f} N·s → {h.body.replace('_link', '')}", font=Fs, fill=(255, 140, 50))
                dr.text((14, a.h - 52), "silver: Unitree R1 (official MJCF)  |  orange: R1 opponent on its own motors, no assist", font=Ft, fill=(200, 200, 210))
                dr.text((14, a.h - 30), f"t={e.d.time:4.1f}s  obs delay 40 ms  " + ("FELL at %.1fs" % x[2] if x[2] else "standing"), font=Fs,
                        fill=(255, 90, 90) if x[2] else (140, 255, 140))
                row.append(np.asarray(img))
            frames.append(np.concatenate(row, 1))
    out = root / "media" / f"{a.name}.mp4"; imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for s in [3.1, 5.0, 8.6]:
        i = min(len(frames) - 1, int(s * fps)); Image.fromarray(frames[i]).save(root / "media" / f"still_{s:.1f}s.png")
    gw = 360 * len(specs) if len(specs) > 1 else 480   # README GIF: 360 px per panel (480 single), 12 fps, 128 colours (<5 MB)
    gh = int(round(gw * frames[0].shape[0] / frames[0].shape[1])) // 2 * 2
    g = [Image.fromarray(f).resize((gw, gh), Image.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames[::2]]
    g[0].save(root / "media" / f"{a.name}.gif", save_all=True, append_images=g[1:], duration=int(2000 / fps), loop=0, optimize=True)
    json.dump({c.name if hasattr(c, "name") else labels[k]: {"fell_at_s": x[2], "contacts": x[0].contact_log} for k, (c, x) in enumerate(zip(ctrls, envs))},
              open(root / "results" / f"{a.name}_contacts.json", "w"))
    print("wrote", out, len(frames), "frames;", {l: x[2] for l, x in zip(labels, envs)})

if __name__ == "__main__": main()
