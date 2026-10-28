"""M0.2: the render twin (r1gym/perception/render_scene.py), its LOD meshes, and physics staying untouched. Tests that draw
need an offscreen GL context and skip without one (CI has none); layout, LOD and camera-pose tests run anywhere."""
import hashlib, json, numpy as np, mujoco, pytest
from pathlib import Path
from r1gym import CONFIGS, ROOT
from r1gym.model import build_spec, load_cfg, R1Robot
from r1gym.opponent import FightReplay
from r1gym.scenario import REF_POSES
from r1gym.perception import run_meta
from r1gym.perception.camera import HeadCameraSpec
from r1gym.perception.render_scene import RenderTwin, LOD_DIR, KEYPOINTS, G_R1HEAD, gl_available, _ry

CFG = {**load_cfg(CONFIGS / "r1_mjlab.yaml"), "opponent_mode": "policy"}
GL = gl_available(); needs_gl = pytest.mark.skipif(not GL, reason="no offscreen GL context here")

def scene(opp_xy=(0.8, 0.0), pose="guard", t=2.0):
    """Physics model + state: the R1 at the origin in a reference pose, the opponent's clip pose moved to opp_xy."""
    m = build_spec(opponent=True, cfg=CFG).compile(); d = mujoco.MjData(m); r = R1Robot(m, CFG); r.reset(m, d); d.qpos[r.qadr] = REF_POSES[pose]
    op = FightReplay(m, "kfab_201", distance=0.6, cfg=CFG); op.place(m, d, t); d.qpos[op.qadr:op.qadr + 2] = opp_xy; mujoco.mj_forward(m, d); return m, d

def test_twin_matches_the_physics_layout_and_physics_gets_no_camera():
    m, d = scene(); tw = RenderTwin(CFG, physics=m)
    assert m.ncam == 0 and tw.m.ncam == 2 and tw.m.nq == m.nq and tw.m.nbody == m.nbody
    assert [tw.m.joint(j).name for j in range(tw.m.njnt)] == [m.joint(j).name for j in range(m.njnt)] and np.array_equal(tw.m.jnt_qposadr, m.jnt_qposadr)
    assert tw.m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT and not tw.m.geom_contype.any() and not tw.m.geom_conaffinity.any()
    assert tw.m.nmeshface < m.nmeshface / 4   # LOD meshes in (the unused collision/rod meshes stay full)
    with pytest.raises(AssertionError, match="qpos layout"): tw.check_layout(build_spec(opponent=False, cfg=CFG).compile())
    tw.sync(d.qpos); assert np.allclose(tw.d.xpos, d.xpos) and np.allclose(tw.d.xmat, d.xmat)   # same kinematics, LOD or not

def test_lod_meshes_match_their_source():
    """render_lod/SOURCE pins each LOD to the sha256 of the Unitree mesh it came from: an upstream mesh update shows here."""
    rows = [l.split() for l in (LOD_DIR / "SOURCE").read_text().splitlines() if l and not l.startswith("#")]
    src = LOD_DIR.parent / "assets"; assert len(rows) == 32
    for lod, f, sha, n_in, n_out in rows:
        assert hashlib.sha256((src / f).read_bytes()).hexdigest() == sha, f
        b = (LOD_DIR / lod).read_bytes(); n = int.from_bytes(b[80:84], "little"); assert n == int(n_out) <= int(n_in) and len(b) == 84 + 50 * n
    assert sum(int(r[4]) for r in rows) < 0.2 * sum(int(r[3]) for r in rows)

def test_head_pitch_turns_camera_and_shell_about_the_pivot():
    m, d = scene(); tw = RenderTwin(CFG, physics=m, head_pitch_deg=0.0); tw.sync(d.qpos); t = m.body("torso_link").id; R = tw.d.xmat[t].reshape(3, 3)
    piv, lens = tw.spec.pivot, np.array([0.088, 0.0, 0.355]); g0 = {g: tw.m.geom_pos[g].copy() for g, *_ in tw.heads[""]}
    for deg in (0.0, -30.0):
        tw.set_head_pitch(deg); c = tw.cams["head"]; p = R.T @ (tw.d.cam_xpos[c] - tw.d.xpos[t]); axis = R.T @ -tw.d.cam_xmat[c].reshape(3, 3)[:, 2]
        assert np.allclose(p, piv + _ry(-np.radians(deg)) @ (lens - piv), atol=1e-9) and np.allclose(p, tw.spec.mount_torso[0], atol=1e-9)   # = camera.py's mount
        assert abs(np.degrees(np.arcsin(axis[2])) - deg) < 1e-6 and abs(axis[1]) < 1e-9
        for g, p0, _ in tw.heads[""]: assert np.isclose(np.linalg.norm(tw.m.geom_pos[g] - piv), np.linalg.norm(p0 - piv))
    assert all(not np.allclose(tw.m.geom_pos[g], g0[g]) for g in g0)   # the shell pitched with the camera

def test_set_head_pitch_leaves_the_callers_spec_alone():
    """A spec passed in (shared with another twin, a test, or mount_fk) keeps its pitch: each twin pitches its own copy."""
    spec = HeadCameraSpec(gym_cfg=CFG, head_pitch_deg=-10.0); a, b = RenderTwin(CFG, spec=spec, room=False), RenderTwin(CFG, spec=spec, room=False)
    a.set_head_pitch(-30.0); c = b.cams["head"]
    assert spec.pitch_deg == -10.0 and b.spec.pitch_deg == b.pitch == -10.0 and a.spec.pitch_deg == a.pitch == -30.0
    assert np.allclose(b.m.cam_pos[c], spec.mount_torso[0]) and not np.allclose(a.m.cam_pos[c], spec.mount_torso[0])

def test_m02_results_share_run_meta_and_their_stated_rules():
    """lod_report, render_bench and coverage carry r1gym.perception.run_meta's block at the top level (as mount.json does),
    rendered ones with the GL strings; the bench gates on its median block; coverage picks the best-scoring pitch, an
    interior one (not the end of its sweep), and configs/r1_camera.yaml uses it."""
    R = {f: json.loads((ROOT.parent / "results" / "perception" / f"{f}.json").read_text()) for f in ("lod_report", "render_bench", "coverage")}
    for f, r in R.items():
        assert set(run_meta()) <= set(r) and "meta" not in r and isinstance(r["git_dirty"], list) and len(r["git"]) == 40, f
        assert r["gl_renderer"] and not r["gl_renderer"].startswith("unavailable") and r["gl_version"], f
    b = R["render_bench"]; g = b["gate"]; ms = b["single_process_ms"]["lod 320x240 shadows_off rgb"]
    assert g["ms_median_block"] == ms == round(float(np.median(b["single_process_block_ms"]["lod 320x240 shadows_off rgb"])), 2) and g["passed"] == (1e3 / ms >= 120)
    c, P = R["coverage"]["choice"], R["coverage"]["pitches"]; best = max(P, key=lambda p: c["scores"][f"{p:g}"])
    assert c["head_pitch_deg"] == best == HeadCameraSpec().pitch_deg and c["interior"] and P[0] != best != P[-1]

def test_rendering_leaves_physics_bit_identical():
    """A 10 s tracked bout (guard policy, demo pushes) gives bit-identical qpos with the twin syncing (and, given GL,
    rendering) every 30 Hz frame and without it."""
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.scenario import demo_env
    c = load_controller("guard=results/policy_mjlab_guard/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    m = build_spec(opponent=True, cfg=cfg).compile(); tw = RenderTwin(cfg, physics=m); runs = []
    for active in (False, True):
        e, o = demo_env(m, cfg, t0=1.0, seconds=10.0); Q = []; frames = 0
        for k in range(500):
            o, *_ = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy())
            if active and k % 5 in (0, 2, 3):   # 3 frames per 5 control steps = 30 Hz
                tw.sync(e.d.qpos); frames += 1
                if GL: tw.render("head")
        runs.append(np.array(Q))
    assert frames == 300 and np.array_equal(runs[0], runs[1])

@needs_gl
def test_opponent_pixels_in_front_and_none_behind():
    tw = RenderTwin(CFG); grp = tw.m.geom_group.copy()
    for xy, front in [((0.8, 0.0), True), ((-1.5, 0.0), False)]:
        m, d = scene(xy); tw.sync(d.qpos); seg = tw.render("head", "seg"); r1, opp = tw.masks(seg)
        assert (opp.sum() > 2000) if front else opp.sum() == 0, (xy, opp.sum())
        assert not np.isin(seg[..., 0][seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM], np.flatnonzero(grp == G_R1HEAD)).any()   # own head shell never drawn
    f = tw.frame(); assert f.shape == (120, 160, 3) and f.dtype == np.uint8

@needs_gl
def test_keypoint_projection_agrees_with_the_render():
    """The pinhole projection used by the coverage study lands each in-view opponent keypoint on the opponent's pixels,
    behind the visible surface by less than a body width."""
    m, d = scene((0.9, 0.0), pose="home"); tw = RenderTwin(CFG, physics=m, head_pitch_deg=-30); tw.sync(d.qpos)
    seg, dep = tw.render("head", "seg", size=(160, 120)), tw.render("head", "depth", size=(160, 120)); _, opp = tw.masks(seg)
    uv, z, ok = tw.project(tw.keypoints("opp_"), "head"); assert ok[[list(KEYPOINTS).index(k) for k in ("head", "chest", "pelvis")]].all()
    for (u, v), zz, i in zip(uv, z, ok):
        if i: assert opp[int(v), int(u)] and -0.01 < zz - dep[int(v), int(u)] < 0.25, (u, v, zz, dep[int(v), int(u)])

@needs_gl
def test_appearance_and_room_change_only_their_arrays_and_are_deterministic():
    m, d = scene(); tw = RenderTwin(CFG, physics=m); tw.sync(d.qpos); x0, s0 = tw.render("head"), tw.render("head", "seg")
    keep = {k: np.array(getattr(tw.m, k)).copy() for k in dir(tw.m) if not k.startswith("_") and isinstance(getattr(tw.m, k), np.ndarray)}
    rng = np.random.default_rng(0); tw.randomize_room(rng); tw.randomize_appearance(rng, robots=True)
    changed = {k for k, v in keep.items() if not np.array_equal(v, getattr(tw.m, k))}
    assert changed and changed <= {"geom_matid", "geom_rgba", "geom_pos", "geom_size", "mat_rgba"} | {k for k in keep if k.startswith("light_")}, changed
    x1 = tw.render("head"); r1a, oa = tw.masks(s0); r1b, ob = tw.masks(tw.render("head", "seg"))
    assert np.abs(x1.astype(int) - x0).mean() > 5 and np.array_equal(oa, ob)   # new look, same robot geometry
    tw.nominal_appearance(); tw.set_room(); assert np.array_equal(tw.render("head"), x0)
    other = RenderTwin(CFG, physics=m); other.sync(d.qpos); y = other.render("head"); other.close()   # a fresh context, then closed
    assert np.array_equal(y, x0) and np.array_equal(tw.render("head"), x0)   # closing another twin must not touch this one
