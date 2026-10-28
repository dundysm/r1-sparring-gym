"""Encode the composed showcase (showcase/compose.py) with ffmpeg:

    showcase/frames/final/f_00001.png ...  ->  media/showcase.mp4         H.264 High, CRF 16, yuv420p, 30 fps, bt709, +faststart
                                           ->  media/showcase_poster.png  the title card (full size; --poster picks another moment)
    showcase/frames/loop/  (or the split   ->  media/showcase_loop.webp   README hero loop, ~960 px, animated WebP, loops forever
    segment of the cut, if not rendered)   ->  media/showcase_loop.gif    fallback, ~720 px, palettegen (diff) + paletteuse, under 10 MB

    python showcase/encode.py [--ffmpeg PATH] [--frames DIR] [--loop-frames DIR] [--out DIR] [--only mp4,webp,gif,poster] [--poster split_fell]

ffmpeg is --ffmpeg, else $FFMPEG, else the one on PATH. The build's encoders are checked first (libx264 required;
libwebp_anim for the WebP, else it is skipped with a note). Segment boundaries come from cut.json, which compose.py writes
next to the frames."""
import os, re, sys, json, shutil, argparse, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRF = 16
WEBP_W, WEBP_Q, WEBP_FPS = 960, 75, 30
GIF_TRIES = [(720, 20, "sierra2_4a", 256), (720, 20, "bayer:bayer_scale=3", 256), (720, 15, "sierra2_4a", 192),
             (640, 15, "bayer:bayer_scale=4", 128), (560, 12, "bayer:bayer_scale=4", 96)]   # (width, fps, dither, colours): first under GIF_MAX, else the smallest
GIF_MAX = 10e6            # bytes (README embeds stay under 10 MB)

def find_ffmpeg(arg):
    for c in (arg, os.environ.get("FFMPEG"), shutil.which("ffmpeg")):
        if c and Path(c).exists(): return str(c)
    raise SystemExit("ffmpeg not found: pass --ffmpeg or set FFMPEG")

def encoders(ff):
    out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return {m.group(1) for m in re.finditer(r"^\s*[VAS][\w.]{5}\s+(\S+)", out, re.M)}

def run(cmd):
    print(" ".join(f'"{c}"' if " " in c or ";" in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit(f"ffmpeg failed ({r.returncode})")
    return r

def size(p): return f"{Path(p).stat().st_size / 1e6:.2f} MB"

def seq(d, start=0, n=None, fps=30):
    """ffmpeg input args for frames d/f_%05d.png from 0-based frame `start`, `n` frames."""
    a = ["-framerate", str(fps), "-start_number", str(start + 1), "-i", str(Path(d) / "f_%05d.png")]
    return a, (["-frames:v", str(n)] if n else [])

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ffmpeg"); ap.add_argument("--frames", default=str(REPO / "showcase/frames/final"))
    ap.add_argument("--loop-frames", default=str(REPO / "showcase/frames/loop")); ap.add_argument("--out", default=str(REPO / "media"))
    ap.add_argument("--only", default="mp4,webp,gif,poster"); ap.add_argument("--crf", type=int, default=CRF)
    ap.add_argument("--gif-max-mb", type=float, default=GIF_MAX / 1e6)
    ap.add_argument("--poster", default="title", help="a moment name from cut.json (title, split_fell, stats, ...) or a 0-based frame")
    a = ap.parse_args(); only = set(a.only.split(",")); gif_max = a.gif_max_mb * 1e6
    ff = find_ffmpeg(a.ffmpeg); enc = encoders(ff)
    print("ffmpeg:", ff); print("encoders:", ", ".join(e for e in ("libx264", "libwebp_anim", "libwebp", "gif") if e in enc) or "none of the needed ones")
    if "libx264" not in enc: raise SystemExit("this ffmpeg has no libx264")
    fr = Path(a.frames); cut = json.load(open(fr / "cut.json")); fps = cut["fps"]; N = cut["frames"]
    have = len(list(fr.glob("f_*.png")))
    if have < N: raise SystemExit(f"{fr}: {have} of {N} frames (run showcase/compose.py first)")
    seg = {s["name"]: s for s in cut["segments"]}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); made = []

    if "mp4" in only:
        p = out / "showcase.mp4"; i, n = seq(fr, 0, N, fps)
        run([ff, "-y", "-hide_banner", "-loglevel", "error", *i, *n,
             "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p,"
                    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
             "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(a.crf),
             "-x264-params", "aq-mode=3:colorprim=bt709:transfer=bt709:colormatrix=bt709",   # aq-mode 3: fewer bits lost in the dark studio
             "-pix_fmt", "yuv420p", "-r", str(fps), "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
             "-movflags", "+faststart", str(p)])
        made.append(p)

    if "poster" in only:     # a named moment of the cut (compose.py's preview points; default the title card) or a frame number
        mom = cut.get("moments", {}); f = int(a.poster) if a.poster.isdigit() else mom.get(a.poster)
        if f is None: raise SystemExit(f"--poster {a.poster}: not a frame number or one of {sorted(mom)}")
        p = out / "showcase_poster.png"; shutil.copyfile(fr / f"f_{f + 1:05d}.png", p); made.append(p); print("poster: frame", f + 1)

    # the README loop: the dedicated loop render if there is one, else the split segment of the cut
    lf = Path(a.loop_frames)
    if (lf / "cut.json").exists():
        lc = json.load(open(lf / "cut.json")); ls, ln, ld = 0, lc["frames"], lf
        if len(list(lf.glob("f_*.png"))) < ln: raise SystemExit(f"{lf}: incomplete loop render")
    else:
        print("no loop render in", lf, "- using the split segment of the cut"); ls, ln, ld = seg["split"]["f0"], seg["split"]["n"], fr

    if "webp" in only:
        if "libwebp_anim" in enc:
            p = out / "showcase_loop.webp"; i, n = seq(ld, ls, ln, fps)
            run([ff, "-y", "-hide_banner", "-loglevel", "error", *i, *n, "-vf", f"fps={WEBP_FPS},scale={WEBP_W}:-2:flags=lanczos",
                 "-c:v", "libwebp_anim", "-lossless", "0", "-quality", str(WEBP_Q), "-compression_level", "6", "-preset", "picture", "-loop", "0", "-an", str(p)])
            made.append(p)
        else:
            print("this ffmpeg has no libwebp_anim: skipping the WebP loop")

    if "gif" in only:
        p = out / "showcase_loop.gif"; best = None
        for j, (w, gfps, dither, ncol) in enumerate(GIF_TRIES):
            q = out / f"_try{j}.gif"; i, n = seq(ld, ls, ln, fps)
            vf = (f"fps={gfps},scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff:max_colors={ncol}[pal];"
                  f"[b][pal]paletteuse=dither={dither}:diff_mode=rectangle")
            run([ff, "-y", "-hide_banner", "-loglevel", "error", *i, *n, "-filter_complex", vf, "-loop", "0", str(q)])
            print(f"  gif {w}px {gfps} fps {dither} {ncol} colours: {size(q)}")
            if best is None or q.stat().st_size < best.stat().st_size:
                if best is not None: best.unlink()
                best = q
            else:
                q.unlink()
            if best.stat().st_size <= gif_max: break
        else:
            print(f"  warning: no GIF try is under {a.gif_max_mb:g} MB; keeping the smallest")
        best.replace(p); made.append(p)

    for p in made: print(f"{p.relative_to(REPO) if p.is_relative_to(REPO) else p}: {size(p)}")

if __name__ == "__main__":
    main()
