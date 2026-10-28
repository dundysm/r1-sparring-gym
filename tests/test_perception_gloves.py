"""Gloves (sparring plan, piece C): the opponent's hands drawn in a glove colour in the render twin only, and the colour-key
detector's per-glove keypoints (r1gym/perception/render_scene.py set_gloves, detect_colorkey.py gloves=True, GloveTracker;
scripts/glove_check.py -> results/perception/gloves_check.json). Tests that draw need an offscreen GL context and skip
without one (CI has none)."""
import json, importlib.util, numpy as np, mujoco, pytest
from r1gym import CONFIGS, ROOT
from r1gym.model import build_spec, load_cfg, R1Robot
from r1gym.scenario import REF_POSES
from r1gym.perception import run_meta
from r1gym.perception.camera import HeadCameraSpec
from r1gym.perception.render_scene import RenderTwin, KEYPOINTS, GLOVE_RGBA, G_OPP, gl_available
from r1gym.perception.detect_colorkey import ColorKeyDetector, GloveTracker, _blobs, _dilate

CFG = {**load_cfg(CONFIGS / "r1_mjlab.yaml"), "opponent_mode": "policy"}
GL = gl_available(); needs_gl = pytest.mark.skipif(not GL, reason="no offscreen GL context here")
RESULT = ROOT.parent / "results" / "perception" / "gloves_check.json"

def arrays(m): return {k: np.array(getattr(m, k)).copy() for k in dir(m) if not k.startswith("_") and isinstance(getattr(m, k), np.ndarray)}

def staged(m, d, r, oq, b_deg, dist, pose):
    """The R1 in its guard at the origin, the opponent standing in a reference pose dist m away at bearing b_deg, facing it."""
    r.reset(m, d); d.qpos[r.qadr] = REF_POSES["guard"]; a = np.radians(b_deg); y = a + np.pi
    d.qpos[oq:oq + 7] = [dist * np.cos(a), dist * np.sin(a), 0.74, np.cos(y / 2), 0, 0, np.sin(y / 2)]; d.qpos[oq + 7:oq + 31] = REF_POSES[pose]; mujoco.mj_forward(m, d)

def test_gloves_change_only_the_twins_hand_colours():
    """Gloves are geom_rgba on the opponent's two hand meshes in the twin: every other twin array, and the physics model, stay
    bit-identical; off restores the nominal colours exactly, and nominal_appearance keeps them on."""
    m = build_spec(opponent=True, cfg=CFG).compile(); before = arrays(m)
    plain, gloved = RenderTwin(CFG, physics=m, room=False), RenderTwin(CFG, physics=m, room=False, gloves=True)
    ids = gloved.glove_geoms(); allg = np.concatenate([ids["left"], ids["right"]])
    assert all(len(v) == 1 for v in ids.values()) and (gloved.m.geom_group[allg] == G_OPP).all()
    assert [gloved.m.body(gloved.m.geom_bodyid[g]).name for g in allg] == ["opp_left_wrist_roll_link", "opp_right_wrist_roll_link"]
    A, B = arrays(plain.m), arrays(gloved.m); diff = {k for k in A if not np.array_equal(A[k], B[k])}
    assert diff == {"geom_rgba"} and set(np.flatnonzero((A["geom_rgba"] != B["geom_rgba"]).any(1))) == set(allg) and np.allclose(B["geom_rgba"][allg], GLOVE_RGBA)
    gloved.nominal_appearance(); assert np.allclose(gloved.m.geom_rgba[allg], GLOVE_RGBA)
    gloved.set_gloves(None); assert np.array_equal(gloved.m.geom_rgba, plain.m.geom_rgba) and gloved.gloves is None
    after = arrays(m); assert all(np.array_equal(before[k], after[k]) for k in before)
    assert all(np.array_equal(before[k], v) for k, v in arrays(build_spec(opponent=True, cfg=CFG).compile()).items())
    plain.close(); gloved.close()

def test_gloved_rendering_leaves_a_bout_bit_identical():
    """A 4 s guard bout gives bit-identical qpos with a gloved twin syncing (and, given GL, rendering) at 30 Hz and without it."""
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.scenario import demo_env
    c = load_controller("guard=results/policy_mjlab_guard/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    m = build_spec(opponent=True, cfg=cfg).compile(); tw = RenderTwin(cfg, physics=m, gloves=True); runs = []
    for active in (False, True):
        e, o = demo_env(m, cfg, t0=1.0, seconds=4.0); Q = []
        for k in range(200):
            o, *_ = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy())
            if active and k % 5 in (0, 2, 3):
                tw.sync(e.d.qpos)
                if GL: tw.render("head")
        runs.append(np.array(Q))
    tw.close(); assert np.array_equal(runs[0], runs[1])

def test_blobs_dilate_and_tracker():
    """8-connected blobs, largest first; Chebyshev dilation; the tracker starts from the image side and keeps identity when
    two gloves cross in the image, and falls back to the image side once a track is older than max_gap_s."""
    mk = np.zeros((10, 12), bool); mk[1:4, 1:4] = True; mk[4, 4] = True; mk[7:9, 8:11] = True; mk[0, 11] = True
    b = _blobs(mk); assert [len(x) for x in b] == [10, 6, 1] and (b[0] == [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3], [4, 4]]).all()
    dl = _dilate(np.array([[5, 5]]), 2, (10, 12)); assert dl.sum() == 25 and dl[3, 3] and dl[7, 7] and not dl[2, 5]
    def two(L, R):   # (bearing, elevation) deg of the true left and right fist; image labels: the one further right (smaller bearing) is "left"
        x = [dict(bearing=np.radians(b_), elevation=np.radians(e_), u=float(k), v=0.0) for k, (b_, e_) in enumerate((L, R))]
        return dict(left=x[0], right=x[1]) if L[0] < R[0] else dict(left=x[1], right=x[0])
    tk = GloveTracker()   # a left hook sweeps high across while the right fist stays low: they cross in azimuth, 20+ deg apart
    for k, b_ in enumerate((-20, -12, -4, 4, 12, 20)):
        out = tk(two((b_, 0.0), (0.0, -25.0)), 0.04 * k)
        assert np.isclose(np.degrees(out["left"]["bearing"]), b_) and np.isclose(np.degrees(out["right"]["elevation"]), -25.0), k
    assert out["left"]["side_image"] == "right"   # crossed: the image side says otherwise
    lone = dict(left=None, right=dict(bearing=np.radians(24.0), elevation=0.0, u=0.0, v=0.0)); out = tk(lone, 0.24)
    assert out["left"] is not None and out["right"] is None   # a lone fist near its fresh track keeps that track's side
    out = tk(lone, 0.24 + 0.5); assert out["right"] is not None and out["left"] is None   # stale tracks: the image side again
    assert tk({"left": None, "right": None}, 1.0) == {"left": None, "right": None}

@needs_gl
def test_gloves_found_on_the_fists_and_existing_outputs_unchanged():
    """Staged opponents: arms down (home), both gloves inside the image, are found on the correct sides within 1.5 deg of the
    fist keypoint (head frame) and 10% of its range; in a guard close up they touch the image border and are flagged edge.
    The existing outputs are bit-identical with gloves=True and gloves=False, and gloves=False adds no key. No glove is
    found with gloves off, nor with the opponent behind the R1."""
    m = build_spec(opponent=True, cfg=CFG).compile(); d = mujoco.MjData(m); r = R1Robot(m, CFG); spec = HeadCameraSpec(gym_cfg=CFG); Rm = spec.mount_torso[1]
    tw = RenderTwin(CFG, physics=m, spec=spec, gloves=True); det, det0 = ColorKeyDetector(spec, CFG, gloves=True), ColorKeyDetector(spec, CFG)
    oq = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]; g = lambda: d.xmat[r.pel].reshape(3, 3).T @ [0, 0, -1.0]
    for b, dist, pose in [(0.0, 0.6, "home"), (20.0, 0.6, "home"), (-25.0, 0.7, "home"), (0.0, 0.6, "guard")]:
        staged(m, d, r, oq, b, dist, pose); tw.sync(d.qpos); kp = tw.keypoints("opp_")
        for gl_on in (True, False):
            tw.set_gloves() if gl_on else tw.set_gloves(None); img = tw.frame("head"); e1, e0 = det(img, d.qpos[r.qadr], g()), det0(img, d.qpos[r.qadr], g())
            assert "gloves" not in e0 and all(np.array_equal(e1[k], e0[k], equal_nan=True) for k in e0), (b, pose)
            if not gl_on: assert e1["gloves"] == dict(left=None, right=None); continue
            for s in ("left", "right"):
                x = e1["gloves"][s]; c = Rm @ tw.to_camera(kp[list(KEYPOINTS).index(s[0] + "_fist")][None])[0]
                u = np.array([np.cos(x["elevation"]) * np.cos(x["bearing"]), np.cos(x["elevation"]) * np.sin(x["bearing"]), np.sin(x["elevation"])])
                err, rel = np.degrees(np.arccos(u @ c / np.linalg.norm(c))), abs(x["range"] - np.linalg.norm(c)) / np.linalg.norm(c)
                if pose == "home": assert not x["edge"] and err < 1.5 and rel < 0.1, (b, s, err, rel)
                else: assert x["edge"], (b, s)
    staged(m, d, r, oq, 180.0, 1.5, "home"); tw.sync(d.qpos); tw.set_gloves(); assert det(tw.frame("head"), d.qpos[r.qadr], g())["gloves"] == dict(left=None, right=None)
    tw.close()

def test_gloves_check_result():
    """gloves_check.json carries run_meta (GL strings: it rendered), the detector's constants as the code has them (GLOVE_AREA
    = the train fit), and its val summary recomputes from its raw rows."""
    r = json.loads(RESULT.read_text()); sp = importlib.util.spec_from_file_location("glove_check", ROOT.parent / "scripts" / "glove_check.py")
    gc = importlib.util.module_from_spec(sp); sp.loader.exec_module(gc)
    assert set(run_meta()) <= set(r) and len(r["git"]) == 40 and r["gl_renderer"] and not r["gl_renderer"].startswith("unavailable")
    assert r["train"]["fit"]["glove_area_m2"] == ColorKeyDetector.GLOVE_AREA == r["val"]["glove_area_m2_used"] == r["detector"]["glove_area_m2"]
    assert r["detector"]["tracker"] == dict(gate_deg=GloveTracker().gate, max_gap_s=GloveTracker().gap) and r["val"]["clips"] == gc.VAL_CLIPS
    v = r["val"]; R = gc.rows(v["raw"]); S = v["summary"]["all"]
    F = [None] * S["frames"]; got = gc.summarize(R, [dict(n_det_plain=0, n_det_empty=0, presence=True, presence_plain=True, body_shift_deg=None)] * len(F))
    for k in ("glove_frames", "in_fov", "visible", "detection_rate_in_fov", "detection_rate_visible", "detections", "side_correct", "side_correct_tracked",
              "fists_crossed_in_image", "errors", "errors_not_edge", "errors_edge"): assert json.loads(json.dumps(got[k])) == S[k], k
    assert got["false_detections"]["matched_to_invisible_glove"] == S["false_detections"]["matched_to_invisible_glove"]
    assert {r_["distance"] for r_ in R} == set(gc.DISTANCES) and 0.4 <= min(gc.DISTANCES) and max(gc.DISTANCES) <= 0.65 and any(r_["strike"] for r_ in R)
