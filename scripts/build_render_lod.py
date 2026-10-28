"""Render LODs for the render twin (phase 3, M0.2) -> r1gym/assets/r1/render_lod/*.stl + SOURCE, results/perception/lod_report.json.

Unitree's R1 visual meshes are ~435k drawn faces per robot, and offscreen rendering on this iGPU is triangle-bound
(results/perception/render_bench.json). Each mesh the R1 draws is welded (exact duplicate vertices merged) and
quadric-decimated (fast-simplification, MIT) to a share of a per-robot face budget proportional to its surface area (at
least 100 faces, never more than it has). The decimator's aggressiveness `agg` trades quality against reaching the target:
at agg 2-3 it stops early on the CAD meshes' open borders and non-manifold edges, so a candidate can end above its budget.
Candidates (budget x agg) are decimated first and gated in order of the faces they actually reach, smallest first; the
first that passes ships. The opponent R1 draws the same files. Physics never sees them (r1gym/perception/render_scene.py).

Gate (silhouette IoU of LOD vs full meshes, segmentation renders at 320x240, the render size):
- 200 poses from bouts of both shipped policies against the tracked opponent (train + val clips, 10 s, the demo's pushes,
  a pose every 0.4 s);
- x 4 views: the R1's head camera at -20 deg pitch, and 3 external views around the pair (side, behind, high);
- x both robots; silhouettes under 400 px in the full-mesh render are left out (and counted).
Pass: mean >= 0.98 and min >= 0.95. (Vertex clustering at 6-12 mm missed this gate, min IoU 0.80-0.90, in the design's
probe; so does the design's ~25k-face quadric target, below.)"""
import sys, json, time, struct, hashlib, shutil, tempfile, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from concurrent.futures import ThreadPoolExecutor
from r1gym import ASSETS
from r1gym.model import build_spec
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start, SCRIPTED
from r1gym.perception import run_meta
from r1gym.perception.render_scene import RenderTwin, LOD_DIR

SRC = ASSETS / "r1" / "assets"; STL = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
POLICIES = ["arms_down=results/policy_mjlab/policy.onnx", "guard=results/policy_mjlab_guard/policy.onnx"]
VIEWS = {"head": None, "ext_side": (90, -10), "ext_behind": (200, -20), "ext_high": (320, -45)}   # external: (azimuth, elevation) deg
GATE = dict(mean=0.98, min=0.95, min_px=400, size=[320, 240], head_pitch_deg=-20.0, poses=200)
CANDIDATES = [(b, agg) for agg in (7.0, 3.0, 2.0) for b in (25_000, 35_000, 50_000)]   # (faces per robot, fast-simplification agg; 7 is its default)

def read_stl(p):
    b = Path(p).read_bytes(); n = struct.unpack("<I", b[80:84])[0]
    if 84 + 50 * n == len(b): return np.frombuffer(b[84:], STL)["v"].reshape(-1, 3, 3).astype(float)
    return np.array([l.split()[1:4] for l in b.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")], float).reshape(-1, 3, 3)   # ASCII

def write_stl(p, v, f):
    t = v[f]; n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    r = np.zeros(len(f), STL); r["n"] = n; r["v"] = t.reshape(-1, 9)
    Path(p).write_bytes(b"r1gym render LOD (scripts/build_render_lod.py)".ljust(80) + struct.pack("<I", len(f)) + r.tobytes())

def weld(tris):
    """Triangle soup -> indexed mesh: exact duplicate vertices merged, degenerate and repeated faces dropped."""
    v, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True); f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, f[np.sort(np.unique(np.sort(f, 1), axis=0, return_index=True)[1])].astype(np.int32)

def visual_files(cfg):
    """The mesh files the R1 draws (geoms in the visible groups 0-2)."""
    s = build_spec(cfg=cfg); m = s.compile()
    used = {m.mesh(m.geom_dataid[g]).name for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] < 3}
    return sorted(Path(me.file).name for me in s.meshes if me.name in used)

def decimate(files, budget, out, agg=7.0):
    import fast_simplification as fs
    mesh = {f: weld(read_stl(SRC / f)) for f in files}; area = {}
    for f, (v, fc) in mesh.items(): t = v[fc]; area[f] = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
    A = sum(area.values()); rows = []
    for f in files:
        v, fc = mesh[f]; tgt = int(np.clip(round(budget * area[f] / A), min(100, len(fc)), len(fc)))
        v2, f2 = (v, fc) if tgt >= len(fc) else fs.simplify(v, fc, target_count=tgt, agg=agg)
        write_stl(out / (Path(f).stem + ".stl"), np.asarray(v2), np.asarray(f2)); rows.append(dict(file=f, faces_in=len(fc), target=tgt, faces_out=len(f2), area_cm2=round(area[f] * 1e4, 1)))
    return rows

def poses(n_per_bout=25, every=20):
    """qpos snapshots from bouts of both shipped policies vs the tracked opponent (train + val clips, demo pushes)."""
    out = []
    for spec in POLICIES:
        lab, path = spec.split("=", 1); c = load_controller(spec); cfg = {**controller_cfg(c), "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=cfg).compile()
        clips = TRAIN_CLIPS + VAL_CLIPS; dists = [0.45, 0.55, 0.65, 0.5]; good = {}
        for cl, d in zip(clips, dists): good.update(clean_starts(cfg, [cl], [d]))
        def run(k):
            cl, d = clips[k], dists[k]; e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=11, max_impulse=0, train=False)
            e.opp = FightReplay(m, cl, distance=d, cfg=cfg); e.opp_t0 = pick_start(good[(cl, d)], 1.0 + k); rng = np.random.default_rng(3)
            o = e.reset(hits=[make_hit(kk, l, i, t, y, rng) for kk, l, i, t, y in SCRIPTED], delay=1); Q = []
            for t in range(n_per_bout * every):
                o, r, dn, info = e.step(c.act(o[None])[0])
                if info["fallen"]: break
                if (t + 1) % every == 0: Q.append(e.d.qpos.copy())
            return lab, cl, d, Q
        for lab, cl, d, Q in ThreadPoolExecutor(4).map(run, range(len(clips))): out += [dict(policy=lab, clip=cl, distance=d, qpos=q) for q in Q]
    return out, cfg

def silhouettes(tw, P):
    """Per pose and view: (R1, opponent) masks, packed."""
    out = []
    for p in P:
        tw.sync(p["qpos"]); row = {}
        for v, ae in VIEWS.items():
            r1, op = tw.masks(tw.render("head" if ae is None else tw.ext_camera(*ae), "seg", size=GATE["size"])); row[v] = (np.packbits(r1), np.packbits(op))
        out.append(row)
    return out

def rgb_ms(tw, P, k=60):
    for p in P[:3]: tw.sync(p["qpos"]); tw.render("head")
    t0 = time.perf_counter()
    for p in P[:k]: tw.sync(p["qpos"]); tw.render("head")
    return round((time.perf_counter() - t0) / min(k, len(P)) * 1e3, 2)

def ious(full, lod):
    rows = []
    for i, (a, b) in enumerate(zip(full, lod)):
        for v in VIEWS:
            for j, robot in enumerate(("r1", "opponent")):
                x, y = np.unpackbits(a[v][j]).astype(bool), np.unpackbits(b[v][j]).astype(bool); n = int(x.sum())
                rows.append(dict(pose=i, view=v, robot=robot, px=n, iou=float((x & y).sum() / max(1, (x | y).sum()))))
    return rows

def stats(rows):
    g = [r["iou"] for r in rows if r["px"] >= GATE["min_px"]]
    return dict(n=len(g), mean=round(float(np.mean(g)), 4), min=round(float(np.min(g)), 4), p1=round(float(np.percentile(g, 1)), 4)) if g else dict(n=0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/perception/lod_report.json"); a = ap.parse_args()
    t0 = time.time(); P, cfg = poses(); P = P[:GATE["poses"]]; files = visual_files(cfg); print(len(P), "poses", f"{time.time() - t0:.0f}s", flush=True)
    full = RenderTwin(cfg, lod=False, room=False, head_pitch_deg=GATE["head_pitch_deg"]); F = silhouettes(full, P)
    from importlib.metadata import version
    res = dict(about=__doc__, **run_meta(gl=full.gl()), fast_simplification=version("fast-simplification"), gate=GATE, views=VIEWS,
               poses=dict(n=len(P), policies=[s.split("=")[0] for s in POLICIES], clips=TRAIN_CLIPS + VAL_CLIPS, per_policy={p: sum(x["policy"] == p for x in P) for p in ("arms_down", "guard")}),
               full=dict(faces_per_robot=int(sum(len(read_stl(SRC / f)) for f in files)), meshes_per_robot=len(files), rgb_ms_320x240_head=rgb_ms(full, P)), candidates={}, chosen=None)
    full.close(); print("full", res["full"], f"{time.time() - t0:.0f}s", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="r1lod_")); cands = {}
    for B, agg in CANDIDATES:
        d = tmp / f"{B}_agg{agg:g}"; d.mkdir(); rows = decimate(files, B, d, agg); cands[d.name] = dict(budget=B, agg=agg, faces_per_robot=int(sum(r["faces_out"] for r in rows)), dir=d, meshes=rows)
    for k, c in sorted(cands.items(), key=lambda kv: kv[1]["faces_per_robot"]):
        tw = RenderTwin(cfg, lod=True, lod_dir=c["dir"], room=False, head_pitch_deg=GATE["head_pitch_deg"])
        R = ious(F, silhouettes(tw, P)); ok = [r for r in R if r["px"] >= GATE["min_px"]]; s = stats(R)
        res["candidates"][k] = dict(budget=c["budget"], agg=c["agg"], faces_per_robot=c["faces_per_robot"], rgb_ms_320x240_head=rgb_ms(tw, P), iou=s,
            iou_by_view={v: stats([r for r in R if r["view"] == v]) for v in VIEWS}, iou_by_robot={x: stats([r for r in R if r["robot"] == x]) for x in ("r1", "opponent")},
            excluded_under_min_px=sum(r["px"] < GATE["min_px"] for r in R), worst=sorted(ok, key=lambda r: r["iou"])[:5],
            passed=bool(s["n"] and s["mean"] >= GATE["mean"] and s["min"] >= GATE["min"]))
        tw.close(); print(k, {x: v for x, v in res["candidates"][k].items() if x not in ("worst",)}, f"{time.time() - t0:.0f}s", flush=True)
        if res["candidates"][k]["passed"]: res["chosen"] = k; break
    for k, c in cands.items():
        if k not in res["candidates"]: res["candidates"][k] = dict(budget=c["budget"], agg=c["agg"], faces_per_robot=c["faces_per_robot"], gated=False)   # larger than the shipped one
    if res["chosen"] is not None:   # ship it: the LOD STLs + SOURCE (hashes of Unitree's meshes, so a mesh update shows as stale)
        c = cands[res["chosen"]]; res["shipped"] = dict(**res["candidates"][res["chosen"]], meshes=c["meshes"]); LOD_DIR.mkdir(parents=True, exist_ok=True)
        for p in LOD_DIR.glob("*.stl"): p.unlink()
        for p in c["dir"].glob("*.stl"): shutil.copy(p, LOD_DIR / p.name)
        commit = (ASSETS / "r1" / "SOURCE_COMMIT").read_text().split()[0]; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        lines = ["# Render LODs of Unitree's R1 visual meshes, for the render twin only (physics uses the originals in ../assets).",
                 f"# Derived from unitreerobotics/unitree_rl_mjlab @{commit} (Apache-2.0, r1gym/assets/LICENSE.unitree_rl_mjlab); modified by",
                 f"# scripts/build_render_lod.py: exact vertex weld + quadric decimation (fast-simplification {res['fast_simplification']}, MIT),",
                 f"# budget {c['budget']} faces per robot at agg {c['agg']:g} -> {c['faces_per_robot']}. Gate and numbers: results/perception/lod_report.json.",
                 "# lod_file source_file source_sha256 faces_in faces_out"]
        lines += [f"{Path(r['file']).stem}.stl {r['file']} {sha(SRC / r['file'])} {r['faces_in']} {r['faces_out']}" for r in c["meshes"]]
        (LOD_DIR / "SOURCE").write_text("\n".join(lines) + "\n")
    shutil.rmtree(tmp, ignore_errors=True); res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print("chosen", res["chosen"], f"{res['wall_s']}s")
