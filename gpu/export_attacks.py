"""M1 attacker library and defender collision proxies (docs/SPARRING_M1M2.md 2.1, 2.3), exported on CPU for gpu/sparring_task.py.

attacks_train.npz  every TRAIN-split hand strike of r1gym/strikes.py (37: kfab_200/201/203; val and test clips never enter),
                   replayed from the in-place motion the tracker performs (strikes.motion: 50 fps, facing -x, wraps modulo the
                   clip), frames onset - 12 to end + 12 (0.24 s each side, both included). Per frame, forward kinematics on the
                   kinematic R1 (opponent._sub) give the five keypoints of the contract's 1.2 (opponent side) in the heading
                   frame H (forward toward the defender, the opponent's left, up; origin on the floor at the spot):
                     l_glove, r_glove  wrist_roll_link frame + (0.08, 0, 0)     l_elbow, r_elbow  elbow_link origin
                     head              torso_link frame + (0.01, 0, 0.357)
                   and the torso yaw in H (wrap(yaw_world - pi), 0 = facing the defender).
                     kp [F, 5, 3] float32 (segments concatenated), yaw [F] float32, fps 50.0,
                     seg [S, 6] int32: start row in kp, length n, onset, apex, end (segment-local frames), hand (0 left, 1 right),
                     guard [2, 5, 3] float32: per hand, the median over that hand's segments of their first and last frames.
attacks_train.json per segment: clip, onset frame, limb, band, type, rise_s, peak_speed; the rule; the git hash.
attacks_shells.npz the defender's 16 collision proxies: per link (2.1's table, zone order) the 64 farthest-point samples of the
                   full convex hull's vertices (the hull MuJoCo collides, mesh_graph) in the BODY frame (geom_pos + geom_mat @ v),
                   a PCA capsule fallback enclosing >= 95% of them, and the zone map.
                     links / bodies / zones [16] str, points [16, 64, 3] float32, capsule [16, 7] float32 (p0, p1, radius),
                     hull_n [16] int32, zone_names [5] str.
Asserts: train split only, 37 segments, 16 links x 64 points, the elbow-to-glove distance is 0.195 m on every frame (the GPU
elbow keypoint is the glove body origin - 0.195 x its axis), proxy extents within 1.5 cm of the full hull's.
Run in the repo: python gpu/export_attacks.py  [--out gpu/motions]"""
import sys, json, argparse, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from r1gym import strikes as S
from r1gym.env import TRAIN_CLIPS
from r1gym.opponent import _sub, opponent_spec

PAD = 12                    # frames each side of [onset, end] (0.24 s at 50 fps)
KP = (("l_glove", "left_wrist_roll_link", (0.08, 0.0, 0.0)), ("r_glove", "right_wrist_roll_link", (0.08, 0.0, 0.0)),
      ("l_elbow", "left_elbow_link", (0.0, 0.0, 0.0)), ("r_elbow", "right_elbow_link", (0.0, 0.0, 0.0)),
      ("head", "torso_link", (0.01, 0.0, 0.357)))
ELBOW_TO_FIST = 0.195
ZONE_NAMES = ("head", "torso", "pelvis", "block", "upper_arm")
SHELLS = (("head", ("head_pitch_link", "head_yaw_link")), ("torso", ("waist_yaw_link",)),
          ("pelvis", ("pelvis_link", "imu_in_pelvis_link", "waist_roll_link")),
          ("block", ("left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link")),
          ("upper_arm", tuple(f"{s}_shoulder_{j}_link" for s in ("left", "right") for j in ("pitch", "roll", "yaw"))))
N_PTS, CAP_SHARE, EXTENT_TOL = 64, 0.95, 0.015

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def strike_library():
    """(arrays dict for attacks_train.npz, per-segment metadata list)."""
    r = _sub("r1"); d = mujoco.MjData(r); bid = [r.body(b).id for _, b, _ in KP]; off = np.array([o for _, _, o in KP]); tor = r.body("torso_link").id
    L = [s for s in S.load(split="train") if s.hand]
    assert all(s.clip in TRAIN_CLIPS and s.split == "train" for s in L), "a non-train clip in the library"
    assert len(L) == 37, f"expected the 37 train hand strikes, got {len(L)}"
    kp, yaw, seg, meta, row = [], [], [], [], 0
    for s in L:
        q, fps = S.motion(s.clip); T = len(q); assert fps == 50.0
        P, Y = [], []
        for f in range(s.onset - PAD, s.end + PAD + 1):
            d.qpos[:] = q[f % T]; mujoco.mj_kinematics(r, d)
            P.append(d.xpos[bid] + np.einsum("kij,kj->ki", d.xmat[bid].reshape(-1, 3, 3), off))
            Y.append(wrap(np.arctan2(d.xmat[tor][3], d.xmat[tor][0]) - np.pi))
        P = np.array(P) * [-1.0, -1.0, 1.0]   # world (the in-place clip faces -x, spot at the origin) -> H
        for h in (0, 1):
            e2f = np.linalg.norm(P[:, h] - P[:, 2 + h], axis=1)
            assert np.abs(e2f - ELBOW_TO_FIST).max() < 1e-3, f"{s.clip} {s.limb}: elbow-to-glove {e2f.min():.4f}-{e2f.max():.4f}"
        n = len(P); kp.append(P); yaw.append(np.unwrap(Y))
        seg.append([row, n, PAD, PAD + s.apex - s.onset, PAD + s.end - s.onset, 0 if s.limb == "l_hand" else 1]); row += n
        meta.append(dict(clip=s.clip, onset=s.onset, limb=s.limb, band=s.band, type=s.type, rise_s=s.rise_s, peak_speed=s.peak_speed_mps))
    kp, yaw, seg = np.concatenate(kp).astype(np.float32), np.concatenate(yaw).astype(np.float32), np.array(seg, np.int32)
    guard = np.stack([np.median(np.concatenate([np.stack([kp[a], kp[a + n - 1]]) for a, n, *_, h in seg if h == hh]), 0) for hh in (0, 1)]).astype(np.float32)
    return dict(kp=kp, yaw=yaw, fps=np.float32(50.0), seg=seg, guard=guard), meta

def _hull_body_points(m, g):
    """Vertices of geom g's convex hull (MuJoCo's mesh_graph) in its body frame."""
    mid = m.geom_dataid[g]; a, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; ga = m.mesh_graphadr[mid]
    assert ga >= 0, f"mesh {m.mesh(mid).name}: no convex hull compiled"
    gr = m.mesh_graph[ga:]; hv = gr[2 + gr[0]:2 + 2 * gr[0]]   # numvert, numface, vert_edgeadr[numvert], vert_globalid[numvert]
    V = m.mesh_vert[a:a + nv][hv]; R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return m.geom_pos[g] + V @ R.reshape(3, 3).T

def fps_sample(V, k):
    """k farthest-point samples of V, starting from the point farthest from the centroid (deterministic)."""
    i = [int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))]; dmin = np.linalg.norm(V - V[i[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(dmin)); i.append(j); dmin = np.minimum(dmin, np.linalg.norm(V - V[j], axis=1))
    return V[i]

def pca_capsule(P, share=CAP_SHARE):
    """Capsule (p0, p1, r) along P's first principal axis: r = the share-quantile of the radial distance, the segment the
    shortest that still encloses every point within r of the axis (so >= share of the points)."""
    c = P.mean(0); ax = np.linalg.svd(P - c)[2][0]; t = (P - c) @ ax; rad = np.linalg.norm((P - c) - t[:, None] * ax, axis=1)
    r = float(np.sort(rad)[int(np.ceil(share * len(rad))) - 1]); ok = rad <= r + 1e-12; h = np.sqrt(np.maximum(r * r - rad[ok] ** 2, 0.0))
    # a point (axial t, radial rad <= r) is inside iff the segment [s0, s1] meets [t - h, t + h]: s1 >= max(t - h) = A and
    # s0 <= min(t + h) = B. A <= B: one axis point serves all (a sphere); else the segment [B, A]
    A, B = float(np.max(t[ok] - h)), float(np.min(t[ok] + h))
    s0, s1 = ((A + B) / 2, (A + B) / 2) if A <= B else (B, A)
    inside = np.mean((np.clip(t, s0, s1) - t) ** 2 + rad ** 2 <= r * r + 1e-9)
    assert inside >= share - 1e-9, inside
    return np.concatenate([c + s0 * ax, c + s1 * ax, [r]]).astype(np.float32)

def shell_proxies():
    """(arrays dict for attacks_shells.npz, per-link fidelity report)."""
    g = opponent_spec("r1")
    for ge in g.geoms:
        if ge.type == mujoco.mjtGeom.mjGEOM_MESH: ge.contype, ge.conaffinity = 0, 4   # collider -> MuJoCo compiles its hull
    m = g.compile(); by_mesh = {m.mesh(m.geom_dataid[i]).name: i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH}
    links, bodies, zones, pts, caps, hn, rep = [], [], [], [], [], [], {}
    for z, ls in SHELLS:
        for l in ls:
            gi = by_mesh[l]; H = _hull_body_points(m, gi); P = fps_sample(H, N_PTS)
            ext = float(np.max(np.abs(np.concatenate([H.min(0) - P.min(0), H.max(0) - P.max(0)]))))
            assert ext <= EXTENT_TOL, f"{l}: proxy extents {ext:.4f} m off the hull's"
            links.append(l); bodies.append(m.body(m.geom_bodyid[gi]).name); zones.append(z); pts.append(P); caps.append(pca_capsule(P)); hn.append(len(H))
            rep[l] = dict(body=bodies[-1], zone=z, hull_vertices=len(H), extent_err_m=round(ext, 4), capsule_r=round(float(caps[-1][6]), 4))
    assert len(links) == 16 and all(p.shape == (N_PTS, 3) for p in pts)
    return dict(links=np.array(links), bodies=np.array(bodies), zones=np.array(zones), points=np.array(pts, np.float32),
                capsule=np.array(caps, np.float32), hull_n=np.array(hn, np.int32), zone_names=np.array(ZONE_NAMES)), rep

def git_hash():
    root = Path(__file__).resolve().parents[1]
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "r1gym", "gpu/export_attacks.py"], cwd=root, stderr=subprocess.DEVNULL) != 0
        return h + ("+dirty" if dirty else "")
    except Exception: return "unknown"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "motions")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lib, meta = strike_library(); np.savez(out / "attacks_train.npz", **lib)
    rule = dict(split="train", limbs="hand", pad_frames=PAD, fps=50, frame="H: forward toward the defender, the opponent's left, up; "
                "origin on the floor at the spot (strikes.py: H = P * [-1, -1, 1] of the in-place motion)", keypoints={k: [b, list(o)] for k, b, o in KP},
                yaw="wrap(torso_link heading - pi), unwrapped per segment", seg="start, n, onset, apex, end (segment-local), hand (0 left, 1 right)",
                guard="per hand, median over its segments of their first and last frames", elbow_to_glove_m=ELBOW_TO_FIST)
    json.dump(dict(rule=rule, git=git_hash(), n=len(meta), frames=int(len(lib["kp"])), segments=meta), open(out / "attacks_train.json", "w"), indent=1)
    sh, rep = shell_proxies(); np.savez(out / "attacks_shells.npz", **sh)
    seg = lib["seg"]
    print(f"attacks_train: {len(seg)} segments ({int((seg[:, 5] == 0).sum())} left / {int((seg[:, 5] == 1).sum())} right), {len(lib['kp'])} frames, "
          f"n {seg[:, 1].min()}-{seg[:, 1].max()}; guard gloves (H) l {np.round(lib['guard'][0, 0], 3).tolist()} r {np.round(lib['guard'][1, 1], 3).tolist()}")
    print("attacks_shells:", json.dumps(rep))
