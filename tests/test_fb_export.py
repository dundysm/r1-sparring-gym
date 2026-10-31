"""Full-body M2 v2 exports (M2V2_PLAN_v3 M1; gpu/export_fb_motion.py, gpu/export_fb_hulls.py, scripts/fb_m1_check.py): train clips
only with sha256 manifests, the tracker's reference frame for frame, the eval's clean starts, the train strike table, the
hull Hausdorff gate, regeneration, and that no export opens a held-out file. The clean-start check runs here at 6 of the 31
distances; scripts/fb_m1_check.py runs all of them."""
import sys, builtins, re
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "gpu"))
import fb_m1_check as C, export_fb_motion as EM, export_fb_hulls as EH

def test_clips_are_the_train_clips_and_manifests_match():
    r = C.check_manifest(); assert r["clips_ok"] and r["outputs_sha_ok"] and r["stale_sources"] == [], r

def test_reference_equals_the_trackers_at_every_train_frame():
    r = C.check_tref(); assert r["ok"] and max(r["max"].values()) <= C.TOL_REF, r["max"]
    assert all(v["q_at_distance"] <= 1e-12 for v in r["per_clip"].values())

def test_clean_starts_equal_the_evals():
    r = C.check_clean(distances=(0.35, 0.4, 0.45, 0.5, 0.55, 0.65)); assert r["compared"] == 18 and r["mismatches"] == 0, r

def test_strike_table_is_the_train_rows():
    r = C.check_strikes(); assert r["ok"] and (r["hand"], r["kick"]) == (37, 1), r

def test_hull_gate_on_all_64_meshes():
    r = C.check_hulls(); assert r["ok"] and r["meshes"] == 64 and r["fine"] == 14, r
    assert all(n == (256 if k in EH.FINE else min(64, n)) for k, n in r["n"].items())

def test_point_triangle_distance_by_region():
    T = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]]); P = np.array([[0.2, 0.2, 0.5], [-1, -1, 0], [2, 0, 0], [0.5, -1, 0], [1, 1, 0], [0.1, 0.1, 0]])
    want = [0.5, np.sqrt(2), 1.0, 1.0, np.sqrt(0.5), 0.0]; assert np.allclose(EH.point_triangle(P, T)[:, 0], want, atol=1e-12)
    rng = np.random.default_rng(0); T = rng.normal(size=(20, 3, 3)); P = rng.normal(size=(50, 3)) * 2
    u = rng.random((4000, 2)); u = np.where(u.sum(1, keepdims=True) > 1, 1 - u, u); S = T[:, None, 0] + u[None, :, :1] * (T[:, None, 1] - T[:, None, 0]) + u[None, :, 1:] * (T[:, None, 2] - T[:, None, 0])
    brute = np.linalg.norm(P[:, None, None] - S[None], axis=-1).min(-1); exact = EH.point_triangle(P, T)
    assert (exact <= brute + 1e-12).all() and np.abs(exact - brute).max() < 0.1   # 4000 samples per triangle: spacing ~0.05

@pytest.mark.parametrize("mesh", ["waist_yaw_link", "head_yaw_link", "pelvis_link"])
def test_hausdorff_agrees_with_support_functions(mesh):
    """d_H of convex sets = max over directions of the support-function gap; 20k directions bound it from below."""
    m = EH.eval_model(); z = np.load(EH.GPU / "motions" / "fb" / "fb_hulls.npz"); k = z["meshes"].tolist().index(mesh); P = z["points"][k, :z["n"][k]]
    V, _ = EH.hull(m, EH.mesh_geoms(m)[mesh]); _, Tr = EH.hull_of(P); d = EH.hausdorff(V, Tr)
    u = np.random.default_rng(1).normal(size=(20000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True); gap = float(((V @ u.T).max(0) - (P @ u.T).max(0)).max())
    assert gap <= d + 1e-9 and gap >= 0.9 * d, (mesh, gap, d)

def test_exports_regenerate_and_open_no_held_out_file(tmp_path, monkeypatch):
    opened, real = [], builtins.open; EM.S.motion.cache_clear()   # strikes.motion caches the clips: make this export read them
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real(f, *a, **k))[1])
    A, _ = EM.export(tmp_path, clean=False); EH.export(tmp_path); monkeypatch.undo()
    bad = [p for p in opened if re.search(r"kfab_20[2456]|kfab_199|r1_kungfu|\.onnx", p)]; assert not bad and any("kfab_201.npz" in p for p in opened), bad
    old = dict(np.load(EM.GPU / "motions" / "fb" / "fb_motion.npz"))
    assert set(old) - set(A) == {"clean"} and all(np.array_equal(A[k], old[k]) for k in A)
    assert EM.sha256(tmp_path / "fb_hulls.npz") == EM.sha256(EM.GPU / "motions" / "fb" / "fb_hulls.npz")

def test_source_hashes_are_checkout_independent(tmp_path):
    """Text sources hash LF-normalised (a Windows autocrlf checkout and the pod's LF clone agree); binaries hash raw."""
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n"); (tmp_path / "b.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "a.npz").write_bytes(b"PK\r\n\x00"); (tmp_path / "b.npz").write_bytes(b"PK\n\x00")
    assert EM.sha256(tmp_path / "a.py") == EM.sha256(tmp_path / "b.py") and EM.sha256(tmp_path / "a.npz") != EM.sha256(tmp_path / "b.npz")
