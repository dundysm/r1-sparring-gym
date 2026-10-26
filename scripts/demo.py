"""Headless demo: Unitree R1 vs an R1 opponent replaying KungFuAthleteBot clips (retargeted G1->R1) + procedural strikes/pushes/grabs. PD stand (left) vs RL policy (right).
Writes media/demo.mp4, media/still_*.png and results/demo_contacts.json (per-body contact forces)."""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from r1gym.env import R1HitEnv
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import PDStand, TorchPolicy

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="results/policy/policy.pt"); ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--clip", default="kfab_202"); ap.add_argument("--distance", type=float, default=0.85); ap.add_argument("--w", type=int, default=640); ap.add_argument("--h", type=int, default=480); a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; (root / "media").mkdir(exist_ok=True)
    cfg = load_cfg(); m = build_spec(opponent=True, cfg=cfg).compile()
    rng = np.random.default_rng(7)
    sched = [("strike", "chest", 4, 1.5, 0.0), ("push", "l_shoulder", 5, 3.0, np.pi / 2), ("strike", "head", 4, 4.6, 0.0),
             ("grab", "r_shoulder", 4, 6.0, -np.pi / 2), ("strike", "pelvis", 7, 7.8, np.pi), ("push", "chest", 8, 9.5, 0.0)]
    ctrls = [PDStand(), TorchPolicy(a.policy)]; envs = []; rens = []
    for c in ctrls:
        e = R1HitEnv(seed=1, cfg=cfg, model=m, episode_s=a.seconds + 1, max_impulse=0, train=False)
        e.opp = FightReplay(m, a.clip, distance=a.distance); e.contact_log = []
        hits = [make_hit(k, l, i, t, y, np.random.default_rng(3)) for k, l, i, t, y in sched]
        o = e.reset(hits=hits, delay=2); envs.append([e, o, None])
        rens.append(mujoco.Renderer(m, a.h, a.w))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [0.05, 0, 0.62]; cam.distance = 2.1; cam.azimuth = 140; cam.elevation = -8   # framed on the R1
    fps = 25; every = int(round(1 / (fps * envs[0][0].dt))); frames = []; F = font(22); Fs = font(17)
    for t in range(int(a.seconds / envs[0][0].dt)):
        for k, (c, x) in enumerate(zip(ctrls, envs)):
            e = x[0]
            if x[2] is None:
                o, r, d, info = e.step(c.act(x[1][None])[0]); x[1] = o
                if info["fallen"]: x[2] = e.t * e.dt
            else:   # fallen: keep physics + opponent going with zero action so the fall is visible
                for _ in range(e.nsub): e.opp.apply(m, e.d, e.d.time, m.opt.timestep); mujoco.mj_step(m, e.d)
        if t % every == 0:
            row = []
            for k, (c, x, rn) in enumerate(zip(ctrls, envs, rens)):
                e = x[0]; rn.update_scene(e.d, cam); img = Image.fromarray(rn.render()); dr = ImageDraw.Draw(img)
                lab = ["PD stand (baseline)", "RL hit-robust policy"][k]; dr.text((14, 10), "Unitree R1 · " + lab, font=F, fill=(255, 255, 255))
                dr.text((14, a.h - 52), "silver: Unitree R1 (24-DoF, official MJCF)  |  orange: R1 sparring partner", font=font(13), fill=(200, 200, 210))
                act = [h for h in e.hits if h.t0 <= e.d.time < h.t0 + h.duration + 0.25]
                if act: h = act[0]; dr.text((14, 40), f"{h.kind.upper()} {h.impulse:.0f} N·s → {h.body.replace('_link','')}", font=Fs, fill=(255, 200, 60))
                dr.text((14, a.h - 30), f"t={e.d.time:4.1f}s  obs delay 40 ms  " + ("FELL at %.1fs" % x[2] if x[2] else "standing"), font=Fs,
                        fill=(255, 90, 90) if x[2] else (140, 255, 140))
                row.append(np.asarray(img))
            frames.append(np.concatenate(row, 1))
    out = root / "media" / "demo.mp4"; imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for s in [1.6, 4.7, 9.7]:
        i = min(len(frames) - 1, int(s * fps)); Image.fromarray(frames[i]).save(root / "media" / f"still_{s:.1f}s.png")
    imageio.mimwrite(root / "media" / "demo.gif", [np.asarray(Image.fromarray(f).resize((640, 270))) for f in frames[::2]], fps=fps // 2, loop=0)
    json.dump({c.name: {"fell_at_s": x[2], "contacts": x[0].contact_log} for c, x in zip(ctrls, envs)}, open(root / "results" / "demo_contacts.json", "w"))
    print("wrote", out, len(frames), "frames;", {c.name: x[2] for c, x in zip(ctrls, envs)},
          "opponent contact events:", {c.name: len(x[0].contact_log) for c, x in zip(ctrls, envs)})

if __name__ == "__main__": main()
