"""Reduced shell hulls for the full-body M2 v2 task (M2V2_PLAN_v3 2.1 and M1 (5), CRITIQUE_M2V2 m8), exported on CPU.

The eval collides every visual mesh of both R1s as its convex hull (robot_contact shell, r1gym/model.py). Per mesh, the
farthest-point samples (export_attacks.fps_sample) of that hull's vertices, in the body frame: 256 for the links that decide
head hits and blocks (head_pitch, head_yaw, waist_yaw, both elbows, both wrists), 64 elsewhere, the whole hull when it has
fewer vertices. Never maxhullvert.
Gate, on all 64 meshes of the eval's two-R1 model: the Hausdorff distance between the reduced hull (MuJoCo's hull of the
samples) and the full hull <= 3 mm on those seven links and <= 1 cm elsewhere. The reduced hull lies inside the full one
(its vertices are full-hull vertices), so the distance is the largest distance from a full-hull vertex to the reduced hull.
fb_hulls.npz   meshes, bodies [32] str (defender names; the opponent's are "opp_" + these), n [32] int32, points [32, 256, 3]
               float64 (rows n: on are zero), full_n [32] int32, tol_m [32].
fb_hulls_manifest.json   sha256 of the output and of every source file, the gate per mesh and robot, git hash, rules.
Run in the repo: python gpu/export_fb_hulls.py [--out gpu/motions/fb]"""
import sys, json, argparse
from pathlib import Path
GPU = Path(__file__).resolve().parent; ROOT = GPU.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(GPU))
import numpy as np, mujoco
from r1gym.model import build_spec, load_cfg
from export_attacks import fps_sample
from export_fb_motion import save_npz, sha256, git_hash, versions

FINE = ("head_pitch_link", "head_yaw_link", "waist_yaw_link", "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link")
N_FINE, N_COARSE, TOL_FINE, TOL_COARSE, PAD = 256, 64, 0.003, 0.01, 256
SOURCES = ["r1gym/assets/r1/r1.xml", "configs/r1_assumptions.yaml", "r1gym/model.py", "gpu/export_attacks.py", "gpu/export_fb_hulls.py", "gpu/export_fb_motion.py"]

def graph(m, mid):
    """MuJoCo's convex hull of mesh mid: (hull vertex ids into the mesh's vertices, faces [nf, 3] of mesh vertex ids)."""
    gr = m.mesh_graph[m.mesh_graphadr[mid]:]; nv, nf = int(gr[0]), int(gr[1]); assert m.mesh_graphadr[mid] >= 0, m.mesh(mid).name
    return gr[2 + nv:2 + 2 * nv], gr[2 + 3 * nv + 3 * nf:2 + 3 * nv + 6 * nf].reshape(nf, 3)   # numvert, numface, edgeadr, globalid, edge_localid, face_globalid

def body_frame(m, g, V):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); return m.geom_pos[g] + V @ R.reshape(3, 3).T

def hull(m, g):
    """(hull vertices [k, 3], triangles [nf, 3, 3]) of mesh geom g, in its body frame."""
    mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; vid, F = graph(m, mid); V = body_frame(m, g, m.mesh_vert[a:a + n].astype(np.float64))
    return V[vid], V[F]

def hull_of(P):
    """MuJoCo's convex hull of the points P (what a mesh made of them collides as), in P's frame."""
    s = mujoco.MjSpec(); s.add_mesh(name="h", uservert=np.asarray(P, np.float64).ravel().tolist())
    ge = s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="h"); ge.contype = ge.conaffinity = 1
    return hull(s.compile(), 0)

def point_triangle(P, T):
    """Distances [n, f] from points P [n, 3] to triangles T [f, 3, 3] (Ericson, Real-Time Collision Detection 5.1.5)."""
    a, b, c, p = T[None, :, 0], T[None, :, 1], T[None, :, 2], P[:, None]
    ab, ac = b - a, c - a; dot = lambda x, y: (x * y).sum(-1)
    d1, d2 = dot(ab, p - a), dot(ac, p - a); d3, d4 = dot(ab, p - b), dot(ac, p - b); d5, d6 = dot(ab, p - c), dot(ac, p - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    div = lambda x, y: x / np.where(y == 0, 1.0, y)
    seg = lambda x, y, t: x + (y - x) * np.clip(t, 0, 1)[..., None]
    den = va + vb + vc; q = a + ab * div(vb, den)[..., None] + ac * div(vc, den)[..., None]   # face interior
    for cond, val in ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), seg(b, c, div(d4 - d3, (d4 - d3) + (d5 - d6)))), \
                     ((vb <= 0) & (d2 >= 0) & (d6 <= 0), seg(a, c, div(d2, d2 - d6))), ((d6 >= 0) & (d5 <= d6), c), \
                     ((vc <= 0) & (d1 >= 0) & (d3 <= 0), seg(a, b, div(d1, d1 - d3))), ((d3 >= 0) & (d4 <= d3), b), ((d1 <= 0) & (d2 <= 0), a):
        q = np.where(cond[..., None], val, q)   # Ericson's order reversed: the earlier test of his wins
    return np.linalg.norm(p - q, axis=-1)

def hausdorff(V_full, T_red, chunk=256):
    """Largest distance from a full-hull vertex to the reduced hull (its surface triangles; every vertex is outside or on it)."""
    return max(float(point_triangle(V_full[i:i + chunk], T_red).min(1).max()) for i in range(0, len(V_full), chunk))

def eval_model():
    cfg = load_cfg(); assert cfg["robot_contact"] == "shell" and cfg["opponent_mode"] == "policy"
    return build_spec(opponent=True, cfg=cfg).compile()

def mesh_geoms(m):
    """{mesh name: geom} for the eval model's mesh geoms (both robots: 32 + 32 "opp_")."""
    G = {m.mesh(m.geom_dataid[g]).name: g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    assert len(G) == 64 and all("opp_" + k in G for k in G if not k.startswith("opp_")); return G

def gate(m, meshes, points, n):
    """Per (mesh, robot): Hausdorff [m] of the exported points' hull to the eval's full hull, its tolerance, the full hull size."""
    G = mesh_geoms(m); out = {}
    for k, name in enumerate(meshes):
        _, T_red = hull_of(points[k, :n[k]]); tol = TOL_FINE if name in FINE else TOL_COARSE
        for pre in ("", "opp_"):
            V, _ = hull(m, G[pre + name]); out[pre + name] = dict(hausdorff_m=hausdorff(V, T_red), tol_m=tol, full_n=len(V), n=int(n[k]))
    return out

def export(out=GPU / "motions" / "fb"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); m = eval_model(); G = mesh_geoms(m)
    meshes = sorted(k for k in G if not k.startswith("opp_")); pts = np.zeros((len(meshes), PAD, 3)); n, full_n, bodies = [], [], []
    for k, name in enumerate(meshes):
        V, _ = hull(m, G[name]); want = N_FINE if name in FINE else N_COARSE; P = V if len(V) <= want else fps_sample(V, want)
        assert (np.abs(P[:, None] - V[None]).max(-1) == 0).any(1).all()   # every sample is a full-hull vertex
        pts[k, :len(P)] = P; n.append(len(P)); full_n.append(len(V)); bodies.append(m.body(m.geom_bodyid[G[name]]).name)
    n = np.array(n, np.int32); res = gate(m, meshes, pts, n)
    bad = {k: v for k, v in res.items() if not v["hausdorff_m"] <= v["tol_m"]}
    assert not bad, f"Hausdorff gate failed: {bad}"
    save_npz(out / "fb_hulls.npz", meshes=np.array(meshes), bodies=np.array(bodies), n=n, points=pts, full_n=np.array(full_n, np.int32),
             tol_m=np.array([TOL_FINE if k in FINE else TOL_COARSE for k in meshes]))
    man = dict(outputs={"fb_hulls.npz": sha256(out / "fb_hulls.npz")}, sources={s: sha256(ROOT / s) for s in SOURCES}, git=git_hash(), versions=versions(),
               rules=dict(fine=list(FINE), n_fine=N_FINE, n_coarse=N_COARSE, tol_fine_m=TOL_FINE, tol_coarse_m=TOL_COARSE,
                          sample="export_attacks.fps_sample of the full hull's vertices (body frame); whole hull if smaller", maxhullvert="never", sha256="text sources LF-normalised, binaries raw"),
               gate={k: dict(v, hausdorff_m=round(v["hausdorff_m"], 6)) for k, v in sorted(res.items())})
    json.dump(man, open(out / "fb_hulls_manifest.json", "w", encoding="utf-8"), indent=1); return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(GPU / "motions" / "fb")); a = ap.parse_args()
    res = export(a.out); w = lambda fine: max((v["hausdorff_m"], k) for k, v in res.items() if (k.replace("opp_", "") in FINE) == fine)
    print(f"fb_hulls: {len(res)} meshes pass; worst fine {w(True)[1]} {w(True)[0] * 1e3:.2f} mm (<= 3), worst coarse {w(False)[1]} {w(False)[0] * 1e3:.2f} mm (<= 10)")
