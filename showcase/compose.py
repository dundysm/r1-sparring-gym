"""Compose the showcase cut (showcase/STORYBOARD.md): rendered shot frames + captions, labels and data cards.

    python showcase/compose.py                 # the cut -> showcase/frames/final/f_00001.png ... (1920x1080)
    python showcase/compose.py --loop          # README loop (the split, larger type) -> showcase/frames/loop/
    python showcase/compose.py --preview       # one still per overlay state -> showcase/preview/compose_*.png
    python showcase/compose.py --facts         # print every on-screen number and the file it came from
    --frames-dir / --build-dir                 # read shot frames / frame tables from elsewhere (e.g. placeholders)

Inputs: showcase/frames/<shot>/f_0001.png ... (Blender renders; split_pd and split_guard are 960x1080 halves) and
showcase/build/<shot>_frames.json (sim seconds per rendered frame, written by the bake step; if absent it is derived from
the speed map in showcase/shots.json). Captions and labels are keyed to SIM time through those tables, so they line up
with the action whatever the speed map does.

Every number on screen is read here, at compose time, from the recordings (showcase/rec/*.npz: hits, fall time, shove)
and the results files (results/eval_opponent/summary_test_hard.json, raw_test_hard.json, vs_3fcbb3b.json,
results/eval_hits/summary.json, eval_raw.json); cross-checks between them are asserted. Nothing is typed in.
Type is Inter (showcase/fonts, OFL). Paths are relative to the repo root; nothing machine-specific is baked in."""
import os, re, sys, json, math, argparse
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------------------------------
# Config: the cut (screen seconds, storyboard "Cut" table). A shot shorter than its slot holds its last frame.
CUT = [("hook", 4.6), ("title", 2.4), ("split", 7.5), ("hold", 7.5), ("stats", 8.0), ("end", 4.0)]
LOOP_S = 8.0          # README loop: the split with its labels on from the first frame
LOOP_UI = 1.7         # type scale for the loop (it is shown ~960 px wide)
W, H = 1920, 1080
FADE = 0.40           # text fade in/out (screen s)
RISE = 10             # text rises this many px while fading in
HIT_HOLD = 1.7        # a hit label stays this long (screen s), or until the next hit
HOOK_DIP = 0.45       # hook dips to black before the title
CARD_BLUR = 0.9       # sharp -> blurred background, centred on the hold/stats cut (screen s)
SPLIT_DIVIDER = 2     # px
REPO_URL = "github.com/dundysm/r1-sparring-gym"
TITLE = "R1 Sparring Gym"
SUBTITLE = "Teaching a Unitree R1 to take a hit, in simulation"
SPLIT_LABELS = ("Rigid stance (PD)", "Footwork policy")
DISCLAIMER = ["Simulation (MuJoCo). Scripted impulses stand in for strikes.", "The opponent is a second R1 on its own motors."]
# Captions per shot, in order; each shows from its anchor until the next one's (the last until the shot ends).
# Anchors: ("hit", i) the i-th scripted hit inside the shot (-1 = last; the caption is fully in as it lands),
# ("step", m) the first moment after that hit when a foot is m metres from where it was at the hit (from the poses),
# ("slowmo_end",) where the speed map returns to 1x, ("shove",) the moment the pelvis is furthest from its start spot,
# ("sim", t) a sim time.
CAPTIONS = {
    "hook": [("{hook_hit_imp} to the {hook_hit_loc}.", ("hit", -1)), ("It steps. It stays up.", ("step", 0.25))],
    "hold": [("Shoved {shove_m} off its spot…", ("shove",)), ("…it walks back.", ("sim", 18.0))],
}
HIT_LABEL_SHOTS = {"split": "bottom", "hold": "bottom"}   # hit labels share the captions' lower-third slot (H - HIT_BASE)
HIT_LABELS_YIELD = True   # a hit label stays off while a caption of the same shot is on (one message at a time)
CAPTION_PX, CAPTION_BASE = 52, 132   # caption size, baseline height above the bottom edge (px at 1080p)
HIT_BASE = 92        # hit-label baseline height above the bottom edge
SAFE = 120           # safe margin for every edge-anchored element (px at 1080p, times the UI scale)
TOP_BASE = 108       # baseline of the top row: split column labels, the slow-motion tag
CAPTION_NOTE = {"hook": {0: "scripted impulse"}}   # small label-caps line under a caption (shot -> caption index -> text)
STATS_OVERLINE = "Evaluation, in simulation"
LOC_LABEL = {"head": "head height", "chest": "chest", "pelvis": "pelvis", "l_knee": "left knee", "r_knee": "right knee",
             "l_shoulder": "left shoulder", "r_shoulder": "right shoulder"}       # eval location -> label
LOC_SHORT = {"head": "head", "l_knee": "left knee", "r_knee": "right knee", "l_shoulder": "left shoulder", "r_shoulder": "right shoulder"}
# palette
WHITE = (255, 255, 255); ORANGE = (255, 122, 26); GREEN = (126, 200, 150); RED = (236, 84, 84)
# fonts: Inter (variable, OFL) shipped in showcase/fonts, the same on Windows and Linux; system fallbacks only if it is missing
INTER = "Inter*opsz,wght*.ttf"
FONT_DIRS = [os.environ.get("SHOWCASE_FONT_DIR"), str(REPO / "showcase/fonts"), os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts"),
             "/usr/share/fonts", "/usr/local/share/fonts", os.path.join(os.environ["WINDIR"], "Fonts") if os.environ.get("WINDIR") else None]
FONTS = {  # role -> candidates: (file glob, {axis: value}) for variable fonts, or a static file
    "display": [(INTER, {"Weight": 600, "Optical size": 32}), ("Inter-SemiBold.ttf", None), ("seguisb.ttf", None), ("DejaVuSans-Bold.ttf", None)],
    "title":   [(INTER, {"Weight": 650, "Optical size": 32}), ("Inter-SemiBold.ttf", None), ("seguisb.ttf", None), ("DejaVuSans-Bold.ttf", None)],
    "medium":  [(INTER, {"Weight": 500, "Optical size": 28}), ("Inter-Medium.ttf", None), ("segoeui.ttf", None), ("DejaVuSans.ttf", None)],
    "light":   [(INTER, {"Weight": 300, "Optical size": 28}), ("Inter-Light.ttf", None), ("segoeuil.ttf", None), ("DejaVuSans.ttf", None)],
    "text":    [(INTER, {"Weight": 400, "Optical size": 18}), ("Inter-Regular.ttf", None), ("segoeui.ttf", None), ("DejaVuSans.ttf", None)],
    "label":   [(INTER, {"Weight": 600, "Optical size": 14}), ("Inter-SemiBold.ttf", None), ("seguisb.ttf", None), ("DejaVuSans-Bold.ttf", None)],
    "small":   [(INTER, {"Weight": 450, "Optical size": 14}), ("Inter-Regular.ttf", None), ("segoeui.ttf", None), ("DejaVuSans.ttf", None)],
}

# ---------------------------------------------------------------------------------------------------------------------
# Facts: every number on screen, with the file it came from.
def _pct(x, nd=0): return f"{100 * x:.{nd}f}%"

def load_facts(repo=REPO):
    rd = lambda p: json.load(open(repo / p, encoding="utf-8"))
    F = {}
    gd, pdr = np.load(repo / "showcase/rec/ep16_guard_fw.npz"), np.load(repo / "showcase/rec/ep16_pd.npz")
    gm, pm = json.loads(str(gd["meta_json"])), json.loads(str(pdr["meta_json"]))
    hits = json.loads(str(gd["hits_json"]))
    assert hits == json.loads(str(pdr["hits_json"])), "PD and guard bouts must have the same scripted hits"
    # location labels from the eval episode the recording replays (record.py checks the replay against it)
    m = re.match(r"(\S+) episode (\d+)", gm["source"]); raw = rd(m.group(1)); ep = raw["episodes"][int(m.group(2))]
    for h, (kind, loc, imp, t0, _y) in zip(hits, ep["hits"]):
        assert h["kind"] == kind and abs(h["t0"] - t0) < 1e-6 and abs(h["impulse"] - imp) < 1e-6, (h, kind, loc)
        h["loc"] = loc
    F["hits"] = [dict(t0=h["t0"], kind=h["kind"], loc=h["loc"], imp=f"{h['impulse']:.1f} N·s") for h in hits]
    F["pd_fell_s"] = pm["fell_s"]; F["guard_fell_s"] = gm["fell_s"]; F["bout_s"] = gm["seconds"]
    F["fell"] = f"{pm['fell_s']:.1f} s"
    # the shove: furthest the pelvis gets from its start spot (eval's max_disp_m) and when (from the recorded poses)
    names = [str(n) for n in gd["body_names"]]; p = names.index("pelvis")
    r = np.linalg.norm(gd["xpos"][:, p, :2] - gd["spot"], axis=1); i = int(r.argmax())
    assert abs(r[i] - gm["max_disp_m"]) < 0.005, (r[i], gm["max_disp_m"])
    F["shove_m"] = f"{gm['max_disp_m']:.2f} m"; F["shove_t"] = float(gd["t"][i])
    # survival on held-out hard bouts
    S = rd("results/eval_opponent/summary_test_hard.json"); g = S["summary"]["RL guard_fw"]; pds = S["summary"]["pd_stand"]
    n = g["episodes"]; up = round(g["survival"] * n)
    rows = raw["raw"]["RL guard_fw"]; assert m.group(1).endswith("raw_test_hard.json") and len(rows) == n
    assert up == sum(1 for x in rows if x["fell_s"] is None), "summary and raw survival disagree"
    pd_up = sum(1 for x in raw["raw"]["pd_stand"] if x["fell_s"] is None); assert pd_up == round(pds["survival"] * pds["episodes"])
    split = {"test": "held-out", "val": "validation"}[S["meta"]["split"]]
    F["survival"] = dict(up=up, n=n, pd_up=pd_up, pd_n=pds["episodes"], bout_s=f"{S['meta']['seconds']:.0f} s", split=split, tier=S["meta"]["tier"],
                         src="results/eval_opponent/summary_test_hard.json")
    # up and within 1 m, vs PR #13 (paired, same episodes)
    V = rd("results/eval_opponent/vs_3fcbb3b.json"); old, new = V["runs"]["raw_test_hard.json"]["RL guard_fw - RL guard@3fcbb3b"]["in_ring"]
    assert abs(new - g["survival_in_ring"]) < 1e-9, "vs_3fcbb3b and summary disagree"
    pr = re.search(r"PR #(\d+)", V["about"]).group(1); ring = S["meta"]["ring_m"]
    F["in_ring"] = dict(old=_pct(old), new=_pct(new), pr=pr, ring=f"{ring:g} m", ring_word=f"{ring:g} metre" if ring == 1 else f"{ring:g} metres",
                        src="results/eval_opponent/vs_3fcbb3b.json")
    # single-hit fall rate (no opponent), footwork guard vs the guard shipped in PR #13
    E = rd("results/eval_hits/summary.json"); er = rd("results/eval_hits/eval_raw.json")["RL guard_fw"]
    ne = E["RL guard_fw"]["episodes"]; assert ne == E["RL guard"]["episodes"]
    imps = sorted({x["impulse"] for x in er}); cum = [sum(1 for x in er if x["impulse"] <= v) for v in imps]
    hi = imps[cum.index(ne)]; sub = [x for x in er if x["impulse"] <= hi]
    assert abs(sum(x["fell"] for x in sub) / ne - E["RL guard_fw"]["overall_fall_rate"]) < 5e-4, "eval_raw and summary disagree"
    F["hits_fall"] = dict(new=_pct(E["RL guard_fw"]["overall_fall_rate"], 1), old=_pct(E["RL guard"]["overall_fall_rate"], 1), n=f"{ne:,}",
                          lo=f"{imps[0]:g}", hi=f"{hi:g}", pr=pr, src="results/eval_hits/summary.json")
    return F

# ---------------------------------------------------------------------------------------------------------------------
# Shots, frame tables, timeline
def speed_map_times(shot, fps):
    """Sim seconds for frame k = 0..n-1 at screen time k/fps through the piecewise speed map [[t, speed], ..., [t_end, null]]."""
    tm = shot["time"]; pieces = []; s0 = 0.0
    for (a, sp), (b, _) in zip(tm[:-1], tm[1:]):
        pieces.append((s0, a, b, sp)); s0 += (b - a) / sp
    out = []
    for k in range(int(math.floor(s0 * fps + 1e-6))):
        s = k / fps
        for j, (ss, a, b, sp) in enumerate(pieces):
            if s < ss + (b - a) / sp - 1e-9 or j == len(pieces) - 1:
                out.append(min(b, a + (s - ss) * sp)); break
    return out

class Shot:
    def __init__(self, spec, fps, frames_dir, build_dir, draft=False):
        self.spec, self.name = spec, spec["name"]; self.dir = Path(frames_dir) / self.name
        tab = Path(build_dir) / f"{self.name}_frames.json"
        if tab.exists():
            t = json.load(open(tab)); t = t if isinstance(t, list) else next(t[k] for k in ("frames", "t", "sim_s", "times") if k in t)
            self.t = [float(x) for x in t]; self.table_src = str(tab)
        else:
            self.t = speed_map_times(spec, fps); self.table_src = "shots.json speed map"
        self.draft, self.res = draft, tuple(spec["res"])
    def check(self):
        have = sum(1 for k in range(len(self.t)) if self.path(k).exists())
        if have < len(self.t) and not self.draft:
            raise SystemExit(f"{self.dir}: {have} of {len(self.t)} frames exist (render them, or pass --draft)")
    def path(self, k): return self.dir / f"f_{k + 1:04d}.png"
    def n(self): return len(self.t)
    def frame(self, k):
        k = min(max(k, 0), self.n() - 1); p = self.path(k)
        if not p.exists(): return Image.new("RGB", self.res, (40, 40, 44))
        im = Image.open(p).convert("RGB")
        return im if im.size == self.res else cover(im, self.res)
    def sim(self, k): return self.t[min(max(k, 0), self.n() - 1)]
    def screen_of(self, t_sim, fps):
        """Screen seconds (from the shot's first frame) of the first frame at or after sim time t_sim."""
        for k, t in enumerate(self.t):
            if t >= t_sim - 1e-9: return k / fps
        return self.n() / fps
    def slowmo_end(self):
        tm = self.spec["time"]
        for (a, sp), (b, _) in zip(tm[:-1], tm[1:]):
            if sp < 1: return b
        return tm[-1][0]
    def slowmo(self):
        tm = self.spec["time"]
        return next(((a, b, sp) for (a, sp), (b, _) in zip(tm[:-1], tm[1:]) if sp < 1), None)

def cover(im, size):
    w, h = size; s = max(w / im.width, h / im.height); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))

# ---------------------------------------------------------------------------------------------------------------------
# Type
def _find(pattern):
    for d in filter(None, FONT_DIRS):
        d = Path(d)
        if not d.is_dir(): continue
        hits = sorted(d.glob(pattern)) or sorted(d.rglob(pattern)) if "/" not in pattern else []
        if hits: return hits[0]
    return None

@lru_cache(None)
def font(role, size):
    size = max(6, int(round(size)))
    for fname, axes in FONTS[role]:
        p = _find(fname)
        if p is None: continue
        try:
            f = ImageFont.truetype(str(p), size)
            if axes:
                ax = f.get_variation_axes(); vals = []
                for a in ax:
                    nm = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
                    v = axes.get(nm, a["default"]); vals.append(min(max(v, a["minimum"]), a["maximum"]))
                f.set_variation_by_axes(vals)
            if "Inter" not in p.name and not _warned: print(f"compose.py: Inter not found, using {p.name} for {role}", file=sys.stderr); _warned.append(1)
            return f
        except Exception:
            continue
    return ImageFont.load_default(size)
_warned = []

class Sprite:
    """Pre-rendered text (+ soft shadow) with its baseline-left origin at (ox, oy) inside the image; w = advance width."""
    def __init__(self, img, ox, oy, w, asc): self.img, self.ox, self.oy, self.w, self.asc = img, ox, oy, w, asc

@lru_cache(4096)
def sprite(runs, shadow=0.5, blur=10, dy=2):
    """runs: tuple of (text, role, size, (r,g,b), alpha, tracking_em). Tracked runs are laid out glyph by glyph."""
    lay, x, asc, desc = [], 0.0, 0, 0
    for text, role, size, col, a, tr in runs:
        f = font(role, size); A, D = f.getmetrics(); asc, desc = max(asc, A), max(desc, D)
        if text == GAP:
            x += tr * size; continue
        if text == DOT:      # a round bullet centred on the cap height
            lay.append((x, DOT, size, col, a)); x += 0.40 * size; continue
        if tr:      # glyph by glyph, positions from prefix advances so kerning survives the tracking
            for i, ch in enumerate(text):
                lay.append((x + f.getlength(text[:i]) + i * tr * size, ch, f, col, a))
            x += f.getlength(text) + max(len(text) - 1, 0) * tr * size
        else:
            lay.append((x, text, f, col, a)); x += f.getlength(text)
    pad = int(blur * 2.5) + 4
    w, h = int(math.ceil(x)) + 2 * pad, asc + desc + 2 * pad
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0)); dr = ImageDraw.Draw(layer)
    for lx, t, f, col, a in lay:
        if t == DOT:
            cap = f * 0.70; r = f * 0.20; cx, cy = pad + lx + r, pad + asc - cap / 2
            ss = 4; big = Image.new("L", (int(2 * r * ss) + 4, int(2 * r * ss) + 4), 0)
            ImageDraw.Draw(big).ellipse((2, 2, 2 + 2 * r * ss, 2 + 2 * r * ss), fill=int(255 * a))
            m = big.resize((big.width // ss + 1, big.height // ss + 1), Image.LANCZOS)
            layer.paste(Image.new("RGBA", m.size, tuple(col) + (255,)), (int(round(cx - r)), int(round(cy - r))), m)
            continue
        dr.text((pad + lx, pad + asc), t, font=f, fill=tuple(col) + (int(255 * a),), anchor="ls")
    if shadow:
        m = layer.getchannel("A").filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * shadow))
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0)); sh.putalpha(m)
        sh = sh.transform(sh.size, Image.AFFINE, (1, 0, 0, 0, 1, -dy))
        layer = Image.alpha_composite(sh, layer)
    return Sprite(layer, pad, pad + asc, x, asc)

def put(frame, sp, x, y, alpha=1.0, align="left"):
    """Composite sprite sp onto RGBA frame with its baseline at y and its left/centre/right edge at x."""
    if alpha <= 0.003: return
    x = x - {"left": 0, "center": sp.w / 2, "right": sp.w}[align]
    im = sp.img
    if alpha < 0.997:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha + 0.5)))
    frame.alpha_composite(im, dest=(int(round(x - sp.ox)), int(round(y - sp.oy))))

def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def ease_out(x): x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3
def window(t, t0, t1, fin=FADE, fout=FADE):
    """Alpha of something shown from t0 to t1 (screen s) with fades in after t0 and out before t1."""
    if t < t0 or t > t1: return 0.0
    return min(ease((t - t0) / fin) if fin > 0 else 1.0, ease((t1 - t) / fout) if fout > 0 else 1.0)

GAP, DOT = "\x00gap", "\x00dot"
def caps(text, size, col=WHITE, a=1.0, tr=0.16, role="label"): return (text.upper(), role, size, col, a, tr)
def gap(em, size): return (GAP, "label", size, WHITE, 0.0, em)
def dot(size, col, a=1.0): return (DOT, "label", size, col, a, 0.0)
def plain(text, size, role="display", col=WHITE, a=1.0, tr=0.0): return (text, role, size, col, a, tr)

# ---------------------------------------------------------------------------------------------------------------------
# Image helpers
def to_np(im): return np.asarray(im, dtype=np.float32) / 255.0
def to_im(a): return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))

@lru_cache(None)
def vignette(w, h, strength=0.45):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2)) / math.sqrt(2)
    return (1 - strength * ease_np(r))[..., None]

def ease_np(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

@lru_cache(None)
def scrim(w, h, edge, depth, peak):
    """Black gradient (RGBA) from the bottom (edge='bottom') or top, `depth` px deep, `peak` max alpha."""
    y = np.arange(h, dtype=np.float32); d = (h - 1 - y) if edge == "bottom" else y
    a = peak * ease_np(1 - d / depth) ** 1.6
    A = np.repeat((a * 255).astype(np.uint8)[:, None], w, 1)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); im.putalpha(Image.fromarray(A)); return im

def add_scrim(frame, edge, alpha, ui=1.0):
    if alpha <= 0.003: return
    s = scrim(frame.width, frame.height, edge, int(420 * ui), 0.55)
    if alpha < 0.997: s = s.copy(); s.putalpha(s.getchannel("A").point(lambda v: int(v * alpha)))
    frame.alpha_composite(s)

@lru_cache(64)
def _soft_mask(w, h, feather):
    """Feathered rounded-rect mask (L), w x h inner box with `feather` px of falloff on every side."""
    m = Image.new("L", (w + 4 * feather, h + 4 * feather), 0)
    ImageDraw.Draw(m).rounded_rectangle((2 * feather, 2 * feather, 2 * feather + w, 2 * feather + h), radius=min(h // 2, 40), fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather))

def backdrop(frame, cx, cy, w, h, alpha, feather=40):
    """Darken a soft patch behind text centred at (cx, cy): invisible on the dark studio, keeps type legible on a lit floor."""
    if alpha <= 0.003: return
    w, h, feather = int(w) // 8 * 8 + 8, int(h) // 8 * 8 + 8, int(feather)
    m = _soft_mask(w, h, feather)
    if alpha < 0.997: m = m.point(lambda v: int(v * alpha))
    lay = Image.new("RGBA", m.size, (0, 0, 0, 0)); lay.putalpha(m)
    frame.alpha_composite(lay, dest=(int(round(cx - m.width / 2)), int(round(cy - m.height / 2))))

def blur(im, radius):
    if radius < 0.3: return im
    small = im.resize((im.width // 2, im.height // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius / 2))
    return small.resize(im.size, Image.BICUBIC)

def grade(im, dim=1.0, rad=0.0, vig=0.0):
    """Blur by rad px, multiply by dim, add vignette."""
    im = blur(im, rad)
    if dim >= 0.999 and vig <= 0: return im
    a = to_np(im) * dim
    if vig > 0: a = a * vignette(im.width, im.height, vig)
    return to_im(a)

def zoom(im, s):
    if abs(s - 1) < 1e-4: return im
    w, h = im.size; cw, ch = w / s, h / s; x0, y0 = (w - cw) / 2, (h - ch) / 2
    return im.transform((w, h), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BICUBIC)

# ---------------------------------------------------------------------------------------------------------------------
class Composer:
    def __init__(self, frames_dir, build_dir, draft=False, loop=False, repo=REPO):
        self.repo = repo; S = json.load(open(repo / "showcase/shots.json")); self.fps = S["fps"]
        self.shots = {s["name"]: Shot(s, self.fps, frames_dir, build_dir, draft) for s in S["shots"]}
        self.F = load_facts(repo); self.loop = loop; self.ui = LOOP_UI if loop else 1.0
        cut = [("split", LOOP_S)] if loop else CUT
        self.segs, t = [], 0.0
        for name, dur in cut:
            n = int(round(dur * self.fps)); self.segs.append(dict(name=name, start=t, dur=n / self.fps, n=n, f0=round(t * self.fps))); t += n / self.fps
        self.total = sum(s["n"] for s in self.segs)
        uses = dict(hook=["hook"], title=["hook", "split_pd", "split_guard"], split=["split_pd", "split_guard"], hold=["hold"], stats=["hold"], end=["hold"])
        for s in {x for sg in self.segs for x in uses[sg["name"]]}: self.shots[s].check()
        self._build_events()

    def seg(self, name): return next(s for s in self.segs if s["name"] == name)

    # -- events in screen seconds (relative to each segment's start) --------------------------------------------------
    def _anchor(self, shot, a):
        if a[0] == "sim": return shot.screen_of(a[1], self.fps)
        if a[0] == "slowmo_end": return shot.screen_of(shot.slowmo_end(), self.fps)
        if a[0] == "shove": return shot.screen_of(self.F["shove_t"], self.fps)
        inside = [h for h in self.F["hits"] if shot.t[0] <= h["t0"] <= shot.t[-1]]
        if a[0] == "hit": return max(0.0, shot.screen_of(inside[a[1]]["t0"], self.fps) - FADE)
        if a[0] == "step": return shot.screen_of(self.step_after(shot, inside[-1]["t0"], a[1]), self.fps)
        raise ValueError(a)

    def step_after(self, shot, t_hit, dist):
        """Sim time at which either foot of our R1 is first `dist` m (horizontally) from where it was at t_hit."""
        d = np.load(self.repo / shot.spec["rec"]); names = [str(n) for n in d["body_names"]]; t = d["t"]; k0 = int(np.searchsorted(t, t_hit))
        out = []
        for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
            xy = d["xpos"][k0:, names.index(foot), :2]; far = np.nonzero(np.linalg.norm(xy - xy[0], axis=1) >= dist)[0]
            if len(far): out.append(float(t[k0 + far[0]]))
        if not out: raise ValueError(f"no step of {dist} m after t = {t_hit:.2f} s")
        return min(out)

    def _build_events(self):
        F = self.F; self.captions = {}
        hook = self.shots["hook"]; hk = [h for h in F["hits"] if hook.t[0] <= h["t0"] <= hook.t[-1]][-1]
        fields = dict(hook_hit_imp=hk["imp"], hook_hit_loc=LOC_SHORT.get(hk["loc"], LOC_LABEL[hk["loc"]]), shove_m=F["shove_m"])
        for name, caps_ in CAPTIONS.items():
            sh = self.shots[name]; seg = self.seg(name) if any(s["name"] == name for s in self.segs) else None
            if seg is None: continue
            ts = [self._anchor(sh, a) for _, a in caps_]
            ends = [t - 0.08 for t in ts[1:]] + [seg["dur"] + FADE if name == "hook" else seg["dur"] - 0.35]   # the hook's dip fades its last one
            self.captions[name] = [(txt.format(**fields), t0, t1) for (txt, _), t0, t1 in zip(caps_, ts, ends)]
        self.hitlabels = {}
        for name, where in HIT_LABEL_SHOTS.items():
            if not any(s["name"] == name for s in self.segs): continue
            sh = self.shots["split_guard" if name == "split" else name]; seg = self.seg(name)
            inside = [h for h in F["hits"] if sh.t[0] <= h["t0"] <= sh.t[min(sh.n(), seg["n"]) - 1]]
            ts = [sh.screen_of(h["t0"], self.fps) for h in inside]
            ev = []
            for i, (h, t0) in enumerate(zip(inside, ts)):
                t1 = min(t0 + HIT_HOLD, ts[i + 1] - 0.12 if i + 1 < len(ts) else 1e9, seg["dur"] - 0.3)
                ev.append((h, t0, t1))
            self.hitlabels[name] = (where, ev)
        pd = self.shots["split_pd"]
        self.fell_t = pd.screen_of(F["pd_fell_s"], self.fps) if F["pd_fell_s"] is not None else None

    # -- rendering ---------------------------------------------------------------------------------------------------
    def render(self, f):
        """Final frame f (0-based) as an RGB image."""
        for s in self.segs:
            if s["f0"] <= f < s["f0"] + s["n"]:
                k = f - s["f0"]; t = k / self.fps
                return getattr(self, "r_" + s["name"])(k, t, s).convert("RGB")
        raise IndexError(f)

    def split_frame(self, k):
        L, R = self.shots["split_pd"].frame(k), self.shots["split_guard"].frame(k)
        im = Image.new("RGB", (W, H)); im.paste(L, (0, 0)); im.paste(R, (W // 2, 0)); return im

    def draw_divider(self, fr, alpha=1.0):
        d = Image.new("RGBA", (SPLIT_DIVIDER, H), (0, 0, 0, 0)); a = np.zeros((H, SPLIT_DIVIDER), np.float32)
        y = np.arange(H, dtype=np.float32); ui = self.ui
        prof = ease_np((y - 40 * ui) / (60 * ui)) * ease_np((H - 200 * ui - y) / (120 * ui))   # gone in the bottom 200 px: never under the lower-third slot
        a[:] = (0.30 * alpha * prof)[:, None]
        d = Image.fromarray(np.dstack([np.full((H, SPLIT_DIVIDER, 3), 255, np.uint8), (a * 255).astype(np.uint8)]), "RGBA")
        fr.alpha_composite(d, dest=(W // 2 - SPLIT_DIVIDER // 2, 0))

    def caption(self, fr, name, t):
        ui = self.ui
        for i, (txt, t0, t1) in enumerate(self.captions.get(name, [])):
            a = window(t, t0, t1)
            if a <= 0: continue
            sp = sprite((plain(txt, CAPTION_PX * ui, "display", WHITE, 1.0, -0.012),), shadow=0.5, blur=14)
            rise = RISE * (1 - ease_out((t - t0) / FADE)) if t - t0 < FADE else 0
            y = H - CAPTION_BASE * ui
            note = CAPTION_NOTE.get(name, {}).get(i)
            nh = 34 * ui if note else 0
            backdrop(fr, W / 2, y - 0.36 * CAPTION_PX * ui + nh / 2, sp.w + 60 * ui, 0.9 * CAPTION_PX * ui + nh, a * 0.55, 46 * ui)
            put(fr, sp, W / 2, y + rise, a, "center")
            if note:
                put(fr, sprite((caps(note, 18 * ui, WHITE, 0.60, 0.18),), shadow=0.6, blur=8), W / 2, y + 36 * ui + rise, a, "center")

    def hit_label(self, h, size):
        return sprite((dot(size, ORANGE), gap(0.62, size), caps(h["kind"], size, ORANGE), gap(0.55, size), ("·", "display", size * 1.1, WHITE, 0.55, 0.0),
                       gap(0.55, size), caps(LOC_LABEL[h["loc"]], size, ORANGE), gap(1.0, size), plain(h["imp"], size * 1.12, "display", WHITE, 0.95)),
                      shadow=0.6, blur=9)

    def hit_labels(self, fr, name, t):
        if name not in self.hitlabels: return
        where, ev = self.hitlabels[name]; ui = self.ui
        quiet = max([window(t, c0 - 0.3, c1 + 0.3, 0.3, 0.3) for _, c0, c1 in self.captions.get(name, [])] + [0.0]) if HIT_LABELS_YIELD else 0.0
        for h, t0, t1 in ev:
            a = window(t, t0, t1, fin=0.18, fout=0.45) * (1 - quiet)
            if a <= 0: continue
            sp = self.hit_label(h, 21 * ui)
            y = (H - HIT_BASE * ui) if where == "bottom" else TOP_BASE * ui
            backdrop(fr, W / 2, y - 8 * ui, sp.w + 40 * ui, 30 * ui, a * 0.5, 30 * ui)
            put(fr, sp, W / 2, y + 6 * (1 - ease_out((t - t0) / 0.25)), a, "center")

    # -- segments ------------------------------------------------------------------------------------------------------
    def r_hook(self, k, t, seg):
        sh = self.shots["hook"]; fr = sh.frame(k)
        dip = max(HOOK_DIP, seg["dur"] - sh.n() / self.fps)      # if the shot is shorter than its slot, the dip hides the held frame
        fade = min(ease(t / 0.3), ease((seg["dur"] - t) / dip))
        fr = fr.convert("RGBA")
        self.caption(fr, "hook", t)
        sm = sh.slowmo()
        if sm:   # speed tag while the shot runs slow
            a0, a1 = sh.screen_of(sm[0], self.fps), sh.screen_of(sm[1], self.fps)
            a = window(t, a0, a1, 0.3, 0.3)
            sp = sprite((caps(f"{1 / sm[2]:g}× slow motion", 18, WHITE, 0.82, 0.18),), shadow=0.6, blur=8)
            put(fr, sp, W - SAFE, TOP_BASE, a, "right")
        return dim_rgba(fr, fade)

    def r_title(self, k, t, seg):
        d = seg["dur"]; u = t / d
        if not hasattr(self, "_title_plate"):        # the hook's last frame (the hook dips to black on it), blurred and dimmed as one 1920 frame
            hk = self.shots["hook"]; self._title_plate = grade(hk.frame(min(hk.n(), self.seg("hook")["n"]) - 1), dim=0.42, rad=16, vig=0.40)
            self._split0 = self.split_frame(0)       # composited full frame first, so any blur has no seam at the divider
        out_u = ease((t - (d - 0.6)) / 0.6)               # last 0.6 s: cross into the split's first frame, un-blurring
        bg = zoom(self._title_plate, 1.0 + 0.045 * (1 - ease(u)))
        if out_u > 0.001:
            sp = grade(self._split0, dim=0.40 + 0.60 * out_u, rad=10 * (1 - out_u), vig=0.35 * (1 - out_u))
            bg = Image.blend(bg, sp, out_u)
        fr = dim_rgba(bg.convert("RGBA"), ease(t / 0.35))
        t1, t2 = 0.12, 0.30                                # title, then subtitle
        a1 = window(t, t1, d - 0.05, 0.5, 0.45); a2 = window(t, t2, d - 0.05, 0.5, 0.45)
        put(fr, sprite(((TITLE, "title", 112, WHITE, 1.0, -0.005),), shadow=0.45, blur=18),
            W / 2, 548 + RISE * (1 - ease_out((t - t1) / 0.6)), a1, "center")
        put(fr, sprite((plain(SUBTITLE, 36, "light", WHITE, 0.86, 0.01),), shadow=0.5, blur=12),
            W / 2, 618 + RISE * (1 - ease_out((t - t2) / 0.6)), a2, "center")
        return fr

    def r_split(self, k, t, seg):
        ui = self.ui; F = self.F
        fr = self.split_frame(k).convert("RGBA")
        lab_a = 1.0 if self.loop else window(t, 0.15, seg["dur"] - 0.15, 0.6, 0.5)
        add_scrim(fr, "top", lab_a * 0.55, ui * 0.75)
        self.draw_divider(fr, lab_a if not self.loop else 1.0)
        y = TOP_BASE * ui
        for i, txt in enumerate(SPLIT_LABELS):
            put(fr, sprite((caps(txt, 21 * ui, WHITE, 0.92, 0.16),), shadow=0.6, blur=10), W / 4 + i * W / 2, y, lab_a, "center")
        if self.fell_t is not None:
            a = window(t, self.fell_t, seg["dur"] - (0.0 if self.loop else 0.15), 0.25, 0.5 if not self.loop else 0.0)
            z = 21 * ui
            fell = sprite((dot(z, RED), gap(0.62, z), caps("fell", z, RED), gap(0.8, z), plain(F["fell"], z * 1.12, "display", WHITE, 0.95)), shadow=0.6, blur=9)
            put(fr, fell, W / 4, y + 46 * ui + 6 * (1 - ease_out((t - self.fell_t) / 0.25)), a, "center")
            if F["guard_fell_s"] is None:
                up = sprite((dot(z, GREEN), gap(0.62, z), caps("still up", z, GREEN)), shadow=0.6, blur=9)   # the split shows 7.5 s, not the whole bout
                put(fr, up, 3 * W / 4, y + 46 * ui + 6 * (1 - ease_out((t - self.fell_t) / 0.25)), a, "center")
        self.hit_labels(fr, "split", t)
        if self.loop:
            fr = dim_rgba(fr, min(ease(t / 0.2), ease((seg["dur"] - t) / 0.25)))
        return fr

    def card_bg(self):
        if not hasattr(self, "_card_bg"):
            sh = self.shots["hold"]; last = min(sh.n(), self.seg("hold")["n"]) - 1
            self._last_hold = sh.frame(last); self._card_bg = grade(self._last_hold, dim=0.34, rad=34, vig=0.45)
        return self._card_bg

    def blur_mix(self, im, u):
        """Partial card background: blur and dim ramped by u in [0, 1]."""
        if u <= 0.001: return im
        return grade(im, dim=1 - 0.66 * u, rad=34 * u, vig=0.45 * u)

    def r_hold(self, k, t, seg):
        sh = self.shots["hold"]; fr = sh.frame(k)
        u = ease((t - (seg["dur"] - CARD_BLUR / 2)) / CARD_BLUR)       # first half of the blur-in to the cards
        fr = self.blur_mix(fr, u).convert("RGBA")
        txt_a = 1 - ease((t - (seg["dur"] - CARD_BLUR / 2)) / 0.3)
        layer = Image.new("RGBA", fr.size, (0, 0, 0, 0))
        self.caption(layer, "hold", t); self.hit_labels(layer, "hold", t)
        if txt_a < 1: layer.putalpha(layer.getchannel("A").point(lambda v: int(v * txt_a)))
        fr.alpha_composite(layer)
        return fr

    def cards(self):
        F = self.F; s, r, h = F["survival"], F["in_ring"], F["hits_fall"]
        return [
            dict(label="Survival", big=((str(s["up"]), "display", 120, WHITE, 1.0, -0.015), ("/" + str(s["n"]), "light", 120, WHITE, 0.45, -0.015)),
                 sub=f"{s['split']} {s['tier']} bouts, {s['bout_s']} each", cmp=("Rigid stance (PD)", f"{s['pd_up']}/{s['pd_n']}"), src=s["src"]),
            dict(label=f"Up and within {r['ring']}", big=((r["new"], "display", 120, WHITE, 1.0, -0.015),),
                 sub=f"still standing, never more than {r['ring']} off its spot", cmp=(f"Previous guard (PR #{r['pr']})", r["old"]), src=r["src"]),
            dict(label="Falls from a single hit", big=((h["new"], "display", 120, WHITE, 1.0, -0.015),),
                 sub=f"{h['n']} single hits, {h['lo']}–{h['hi']} N·s, no opponent", cmp=(f"Previous guard (PR #{h['pr']})", h["old"]), src=h["src"]),
        ]

    def r_stats(self, k, t, seg):
        d = seg["dur"]
        u = ease((t + CARD_BLUR / 2) / CARD_BLUR)      # second half of the blur-in (the hold frame is frozen from here)
        if u < 1:
            self.card_bg(); fr = self.blur_mix(self._last_hold, u)
        else:
            fr = self.card_bg()
        fr = fr.convert("RGBA")
        col_w, col_gap = 520, 60; x0 = (W - 3 * col_w - 2 * col_gap) / 2
        top = 404
        oa = window(t, 0.30, d - 0.25, 0.7, 0.45)      # overline above the three cards, on the left safe margin
        put(fr, sprite((caps(STATS_OVERLINE, 18, WHITE, 0.50, 0.20),), shadow=0.4, blur=10), x0, top - 112 + 12 * (1 - ease_out((t - 0.30) / 0.8)), oa)
        for i, c in enumerate(self.cards()):     # staggered in, staggered out
            t0 = 0.45 + 0.28 * i; a = window(t, t0, d - 0.25 - 0.12 * (2 - i), 0.7, 0.45)
            if a <= 0: continue
            dy = 16 * (1 - ease_out((t - t0) / 0.8))
            x = x0 + i * (col_w + col_gap); y = top + dy
            rule = Image.new("RGBA", (40, 2), (255, 255, 255, 130))
            put(fr, sprite((caps(c["label"], 17, WHITE, 0.70, 0.18),), shadow=0.4, blur=10), x, y, a)
            put(fr, sprite(c["big"], shadow=0.3, blur=16), x - 5, y + 128, a)
            put(fr, sprite((plain(c["sub"], 22, "text", WHITE, 0.82),), shadow=0.4, blur=10), x, y + 182, a)
            put(fr, sprite((plain(c["cmp"][0], 20, "text", WHITE, 0.50), gap(0.6, 20), plain(c["cmp"][1], 20, "label", WHITE, 0.82)), shadow=0.4, blur=10), x, y + 220, a)
            put(fr, sprite((caps("source", 11.5, WHITE, 0.34, 0.20), gap(0.9, 11.5), plain(c["src"], 14, "small", WHITE, 0.42)), shadow=0.4, blur=10), x, y + 286, a)
            ra = rule.copy(); ra.putalpha(ra.getchannel("A").point(lambda v: int(v * a)))
            fr.alpha_composite(ra, dest=(int(x), int(y - 42)))
        return fr

    def r_end(self, k, t, seg):
        d = seg["dur"]; bg = self.card_bg()
        fr = grade(bg, dim=1 - 0.25 * ease(t / 0.8)).convert("RGBA")
        a = window(t, 0.25, d + 1, 0.7, 0.1)
        put(fr, sprite((caps(TITLE, 20, WHITE, 0.70, 0.22),), shadow=0.4, blur=10), W / 2, 432, a, "center")
        put(fr, sprite((plain(REPO_URL, 50),), shadow=0.35, blur=14), W / 2, 512, a, "center")
        for i, line in enumerate(DISCLAIMER):
            put(fr, sprite((plain(line, 26, "light", WHITE, 0.72),), shadow=0.4, blur=10), W / 2, 604 + 40 * i, window(t, 0.55, d + 1, 0.7, 0.1), "center")
        return dim_rgba(fr, ease((d - t) / 0.9))

    # -- preview stills ------------------------------------------------------------------------------------------------
    def preview_points(self):
        P = []; g = lambda name: self.seg(name)
        if self.loop:
            return [("loop_hit", g("split")["f0"] + round((self.hitlabels["split"][1][1][1] + 0.5) * self.fps)),
                    ("loop_fell", g("split")["f0"] + round((self.fell_t + 1.2) * self.fps))]
        hk = self.captions["hook"]
        P.append(("hook_caption1", g("hook")["f0"] + round((hk[0][1] + 0.9) * self.fps)))
        P.append(("hook_caption2", g("hook")["f0"] + round((hk[1][1] + 0.6) * self.fps)))
        P.append(("title", g("title")["f0"] + round(1.3 * self.fps)))
        ev = self.hitlabels["split"][1]
        P.append(("split_hit", g("split")["f0"] + round((ev[0][1] + 0.6) * self.fps)))
        P.append(("split_fell", g("split")["f0"] + round((self.fell_t + 1.0) * self.fps)))
        ev = self.hitlabels["hold"][1]; hc = self.captions["hold"]
        P.append(("hold_hit", g("hold")["f0"] + round((ev[0][1] + 0.6) * self.fps)))
        P.append(("hold_caption1", g("hold")["f0"] + round((ev[1][1] + 0.5) * self.fps)))
        P.append(("hold_caption2", g("hold")["f0"] + round((hc[1][1] + 0.8) * self.fps)))
        P.append(("stats", g("stats")["f0"] + round(4.0 * self.fps)))
        P.append(("end", g("end")["f0"] + round(2.0 * self.fps)))
        return P

def dim_rgba(fr, k):
    """Fade an RGBA frame toward black by factor k (1 = untouched)."""
    if k >= 0.999: return fr
    a = np.asarray(fr).copy(); a[..., :3] = (a[..., :3].astype(np.float32) * max(k, 0)).astype(np.uint8)
    return Image.fromarray(a, "RGBA")

# ---------------------------------------------------------------------------------------------------------------------
_C = None
def _init(args):
    global _C
    _C = Composer(*args)

def _job(fo):
    f, out = fo
    _C.render(f).save(out / f"f_{f + 1:05d}.png", compress_level=3)
    return f

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames-dir", default=str(REPO / "showcase/frames")); ap.add_argument("--build-dir", default=str(REPO / "showcase/build"))
    ap.add_argument("--out"); ap.add_argument("--loop", action="store_true"); ap.add_argument("--preview", action="store_true")
    ap.add_argument("--facts", action="store_true"); ap.add_argument("--draft", action="store_true", help="allow missing shot frames (grey)")
    ap.add_argument("--range", help="a:b final frames to render (0-based, half-open)"); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    if a.facts:
        F = load_facts(); print(json.dumps(F, indent=1, ensure_ascii=False)); return
    C = Composer(a.frames_dir, a.build_dir, a.draft, a.loop)
    print("cut:", ", ".join(f"{s['name']} {s['start']:.2f}-{s['start'] + s['dur']:.2f} s ({s['n']} f)" for s in C.segs), f"| {C.total} frames @ {C.fps} fps")
    for n, sh in C.shots.items(): print(f"  {n}: {sh.n()} frames, sim {sh.t[0]:.2f}-{sh.t[-1]:.2f} s, table: {sh.table_src}")
    if a.preview:
        out = Path(a.out or REPO / "showcase/preview"); out.mkdir(parents=True, exist_ok=True)
        for name, f in C.preview_points():
            p = out / f"compose_{name}.png"; C.render(f).save(p); print("preview", p.name, "frame", f + 1)
        return
    out = Path(a.out or REPO / ("showcase/frames/loop" if a.loop else "showcase/frames/final")); out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.range.split(":")) if a.range else (0, C.total)
    jobs = [(f, out) for f in range(lo, min(hi, C.total))]
    json.dump(dict(fps=C.fps, frames=C.total, segments=C.segs, loop=a.loop, moments=dict(C.preview_points())),   # moments: 0-based frames, e.g. poster picks
              open(out / "cut.json", "w"), indent=1)
    if a.workers <= 1:
        _init((a.frames_dir, a.build_dir, a.draft, a.loop)); [_job(j) for j in jobs]
    else:
        from multiprocessing import Pool
        with Pool(a.workers, initializer=_init, initargs=((a.frames_dir, a.build_dir, a.draft, a.loop),)) as p:
            for i, _ in enumerate(p.imap_unordered(_job, jobs, chunksize=4)):
                if i % 60 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    print("wrote", len(jobs), "frames to", out)

if __name__ == "__main__":
    main()
