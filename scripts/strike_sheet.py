"""Strike library contact sheets (rendered from the motion the opponent's tracker performs, r1gym/strikes.py).

default        media/strikes_sheet.png: the apex frame of every library strike (r1gym/assets/motions/strikes.json), the striking
               effector marked, captioned clip / limb / band / type / unwarped head reach.
--proposals D  the hand-label audit sheets (strikes.audit_candidates: every val candidate plus a seeded train sample): per
               candidate five frames 0.12 s apart around its speed peak, the effector marked, and its forward extension over
               time plus its top-down path. Nothing on them comes from the detector, so labelling them is blind to it.
               Writes D/sheet_NN.png and D/index.json (row -> candidate); the labels go in r1gym/assets/motions/strike_labels.json.
The opponent stands at the spot (x = 0) facing -x; a translucent stand-in marks where the R1 would stand at 0.60 m."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import list_clips
from r1gym import strikes as S

COL = {"l_hand": [1.0, 0.85, 0.1, 1], "r_hand": [0.2, 0.8, 1.0, 1], "l_foot": [1.0, 0.4, 0.8, 1], "r_foot": [0.5, 1.0, 0.4, 1]}
R1_AT = -0.60   # stand-in R1 pelvis x (bout distance 0.60 m)

def font(sz):
    for p in ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists(): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def gl_info(rn):
    """GL strings of the renderer's context (what rasterised the sheet)."""
    try:
        rn._gl_context.make_current(); from OpenGL import GL
        return {k: GL.glGetString(e).decode() for k, e in (("gl_vendor", GL.GL_VENDOR), ("gl_renderer", GL.GL_RENDERER), ("gl_version", GL.GL_VERSION))}
    except Exception as ex: return dict(gl_renderer=f"unavailable ({type(ex).__name__}: {ex})")

class Cam:
    def __init__(self, w, h):
        self.m = build_spec(cfg=load_cfg()).compile(); self.d = mujoco.MjData(self.m); self.rn = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE; self.cam.lookat[:] = [-0.22, 0, 0.78]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.75, 115, -10   # side-on from the opponent's left-front
        self.sid = {k: self.m.site(v).id for k, v in S.EFFECTORS.items()}
    def __call__(self, q, limb):
        d, scn = self.d, self.rn.scene; d.qpos[:] = q; mujoco.mj_forward(self.m, d); self.rn.update_scene(d, self.cam)
        def add(kind, size, pos, rgba, a=None, b=None, w=0.0):
            if scn.ngeom >= scn.maxgeom: return
            g = scn.geoms[scn.ngeom]; mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(pos, float), np.eye(3).ravel(), np.array(rgba, np.float32))
            if a is not None: mujoco.mjv_connector(g, kind, w, np.array(a, float), np.array(b, float))
            scn.ngeom += 1
        add(mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], [0, 0, 0], [0.9, 0.2, 0.2, 0.35], a=[R1_AT, 0, 0.62], b=[R1_AT, 0, 1.0], w=0.1)          # stand-in R1 trunk
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.07] * 3, [R1_AT + 0.05, 0, 1.14], [0.9, 0.2, 0.2, 0.45])                                         # and head
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.045] * 3, d.site_xpos[self.sid[limb]], COL[limb])
        return Image.fromarray(self.rn.render())

def plot(P, peak, fps, w, h, F):
    """Forward extension (-x) over +-0.5 s (peak marked) and the top-down path over the frames shown."""
    im = Image.new("RGB", (w, h), (24, 26, 30)); dr = ImageDraw.Draw(im); T = len(P); n = int(0.5 * fps)
    idx = [(peak + k) % T for k in range(-n, n + 1)]; x = -P[idx, 0]; h2 = h // 2 - 6
    sx = lambda k: 8 + (w - 16) * k / (2 * n); sy = lambda v: 4 + h2 - (h2 - 8) * (v + 0.2) / 1.0     # forward -0.2..0.8 m
    for v in (0.0, 0.25, 0.5): dr.line([(8, sy(v)), (w - 8, sy(v))], fill=(70, 70, 80)); dr.text((w - 34, sy(v) - 12), f"{v:.2f}", font=F, fill=(150, 150, 160))
    dr.line([(sx(k), sy(x[k])) for k in range(len(x))], fill=(255, 220, 60), width=2); dr.line([(sx(n), 4), (sx(n), h2)], fill=(200, 80, 80))
    dr.text((10, 4), "forward (m) vs time, +-0.5 s", font=F, fill=(200, 200, 210))
    k2 = [(peak + k) % T for k in range(-12, 13)]; top = P[k2]; y0 = h2 + 10                                       # top view: forward right->left
    tx = lambda p: w / 2 - 150 * (-p[0] - 0.25); ty = lambda p: y0 + (h - y0) / 2 - 150 * (-p[1])
    dr.ellipse([tx([R1_AT, 0]) - 6, ty([0, 0]) - 6, tx([R1_AT, 0]) + 6, ty([0, 0]) + 6], outline=(230, 80, 80))
    dr.text((tx([R1_AT, 0]) - 10, ty([0, 0]) + 8), "R1", font=F, fill=(230, 80, 80)); dr.ellipse([tx([0, 0]) - 4, ty([0, 0]) - 4, tx([0, 0]) + 4, ty([0, 0]) + 4], outline=(160, 160, 170))
    dr.line([(tx(p), ty(p)) for p in top], fill=(120, 200, 255), width=2); dr.ellipse([tx(top[12]) - 4, ty(top[12]) - 4, tx(top[12]) + 4, ty(top[12]) + 4], fill=(255, 220, 60))
    dr.text((10, y0), "top view, -0.24..+0.24 s", font=F, fill=(200, 200, 210)); return im

def proposals_sheets(out, rows=6, fw=220, skip=None):
    """skip: a labels file; candidates it already labels are left out (round 2 renders only the new ones)."""
    done = {(x["clip"], x["limb"], x["frame"]) for x in json.load(open(skip))["labels"]} if skip else set()
    out = Path(out); out.mkdir(parents=True, exist_ok=True); cands = [c for c in S.audit_candidates(list_clips()) if (c[0], c[2], c[3]) not in done]; cam = Cam(fw, fw); F, Fs = font(15), font(12)
    idx = []; sheets = [cands[i:i + rows] for i in range(0, len(cands), rows)]
    for n, part in enumerate(sheets):
        W = 5 * fw + 260; im = Image.new("RGB", (W, rows * (fw + 22)), (12, 12, 14)); dr = ImageDraw.Draw(im)
        for r, (clip, split, limb, f, v) in enumerate(part):
            q, fps = S.motion(clip); P, _ = S.effector_paths(q); y = r * (fw + 22); j = list(S.EFFECTORS).index(limb)
            for k, off in enumerate((-0.24, -0.12, 0.0, 0.12, 0.24)):
                im.paste(cam(q[(f + int(round(off * fps))) % len(q)], limb), (k * fw, y + 20))
            im.paste(plot(P[:, j], f, fps, 256, fw, Fs), (5 * fw + 2, y + 20))
            rid = f"{n:02d}.{r}"; dr.text((4, y + 2), f"[{rid}] {clip} ({split})  {limb}  speed peak frame {f} ({v:.1f} m/s)   frames: -0.24 -0.12 0 +0.12 +0.24 s", font=F, fill=(230, 230, 235))
            idx.append(dict(row=rid, clip=clip, split=split, limb=limb, frame=f, speed_mps=v))
        im.save(out / f"sheet_{n:02d}.png")
    json.dump(dict(candidates=idx, gl=gl_info(cam.rn)), open(out / "index.json", "w"), indent=1); print("wrote", len(sheets), "sheets,", len(idx), "candidates to", out)

def library_sheet(out, cols=10, fw=170):
    lib = S.load(); cam = Cam(fw, fw); F = font(11); rows = (len(lib) + cols - 1) // cols
    im = Image.new("RGB", (cols * fw, rows * (fw + 30) + 26), (12, 12, 14)); dr = ImageDraw.Draw(im)
    dr.text((6, 4), f"Strike library: apex frame of each of the {len(lib)} strikes (r1gym/assets/motions/strikes.json); red: the R1 at 0.60 m; marker: striking effector",
            font=font(14), fill=(235, 235, 240))
    for i, s in enumerate(lib):
        q, fps = S.motion(s.clip); x, y = (i % cols) * fw, 26 + (i // cols) * (fw + 30); im.paste(cam(q[s.apex], s.limb), (x, y))
        dr.text((x + 3, y + fw + 1), f"{s.clip[5:]} {s.split} {s.limb} f{s.apex}", font=F, fill=(220, 220, 225))
        dr.text((x + 3, y + fw + 14), f"{s.band} {s.type} z{s.apex_m[2]:.2f} reach {s.reach_head_m:.2f}", font=F, fill=(255, 210, 90) if s.band == "head" else (170, 170, 180))
    Path(out).parent.mkdir(parents=True, exist_ok=True); im.save(out); g = gl_info(cam.rn); print("wrote", out, len(lib), "strikes", g)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import eval_opponent as eo; from r1gym.model import load_cfg   # what rendered it, from which commit
    cfg = {**load_cfg(), "opponent_mode": "policy"}; meta = {k: v for k, v in eo.provenance(build_spec(opponent=True, cfg=cfg).compile(), cfg).items() if k in ("git_hash", "git_dirty", "mujoco")}
    Path("results/opponent_aim").mkdir(parents=True, exist_ok=True)
    json.dump(dict(image=str(Path(out).name), strikes=len(lib), library="r1gym/assets/motions/strikes.json", meta=dict(meta, **g)), open("results/opponent_aim/strikes_sheet.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--proposals", default=None); ap.add_argument("--skip", default=None); ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "media" / "strikes_sheet.png"))
    a = ap.parse_args()
    if a.proposals: proposals_sheets(a.proposals, skip=a.skip)
    else: library_sheet(a.out)
