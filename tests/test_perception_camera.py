"""M0.1: head-camera spec, mount kinematics, lens models, rectifier and clocks (r1gym/perception/camera.py). CPU; the one
render test skips where no offscreen GL context can be made."""
import numpy as np, mujoco, pytest, yaml
from r1gym import CONFIGS
from r1gym.model import build_spec, R1Robot, load_cfg, CTRL_JOINTS
from r1gym.perception import run_meta
from r1gym.perception.camera import (HeadCameraSpec, mount_fk, sim_cam_heading, add_head_camera, own_head_geoms, project_pinhole,
    unproject_pinhole, project_ftheta, unproject_ftheta, area_down, FisheyeRectifier, CaptureClock, LatencyQueue, TORSO_IN_WAIST, rz)

CFG = load_cfg(); SPEC = HeadCameraSpec()
def ang(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1), -1, 1)))

def test_spec_numbers_and_chain_match_r1_xml():
    fx, fy, cx, cy = SPEC.K(); assert fx == fy and (cx, cy) == (80, 60) and abs(fx - 50.35) < 0.01
    assert abs(2 * np.degrees(np.arctan(cx / fx)) - 115.6) < 0.05                       # 160 x 120 at fovy 100: 115.6 deg H
    assert abs(SPEC.f - 640 / np.radians(73)) < 0.05                                       # f-theta fit to 146 deg over 1280 px
    m = build_spec(cfg=CFG).compile(); w, t = m.body("waist_roll_link").id, m.body("torso_link").id
    assert m.body_parentid[w] == m.body("pelvis").id and m.body_parentid[t] == w and m.ncam == 0   # the physics model gets no camera
    assert np.allclose(m.body_pos[w], 0) and np.allclose(m.body_pos[t], TORSO_IN_WAIST) and np.allclose(m.body_quat[[w, t]], [1, 0, 0, 0])
    assert np.allclose(m.jnt_axis[m.joint("waist_roll_joint").id], [1, 0, 0]) and np.allclose(m.jnt_axis[m.joint("waist_yaw_joint").id], [0, 0, 1])
    assert np.allclose(m.jnt_pos[[m.joint("waist_roll_joint").id, m.joint("waist_yaw_joint").id]], 0)
    assert np.allclose(m.site_pos[m.site("imu").id], 0) and np.allclose(m.site_quat[m.site("imu").id], [1, 0, 0, 0])   # IMU frame = pelvis frame

def test_latency_below_control_dt_is_refused(tmp_path):
    c = yaml.safe_load(open(CONFIGS / "r1_camera.yaml")); c["timing"]["latency_ms"] = [10, 60]; p = tmp_path / "cam.yaml"; yaml.safe_dump(c, open(p, "w"))
    with pytest.raises(AssertionError, match="causality"): HeadCameraSpec(p)
    with pytest.raises(KeyError): HeadCameraSpec(head_pitch=-20)

def test_pinhole_axis_left_and_round_trip():
    K = SPEC.K(SPEC.scale); assert np.allclose(project_pinhole([0, 0, 1.0], K), K[2:])   # optical axis -> principal point
    p, R = mount_fk(np.zeros(24), [0, 0, -1.0], SPEC)                                       # standing straight: heading = pelvis frame
    left = R.T @ (np.array([2.0, 0.5, p[2]]) - p); assert project_pinhole(left, K)[0] < K[2]   # the robot's left (+y) -> u < cx
    uv = np.random.default_rng(0).uniform(0, [320, 240], (500, 2)); assert np.abs(project_pinhole(unproject_pinhole(uv, K), K) - uv).max() < 1e-9

def test_ftheta_round_trip_and_fov_edges():
    rng = np.random.default_rng(0); th, ph = rng.uniform(0, np.radians(89), 2000), rng.uniform(-np.pi, np.pi, 2000)
    rays = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    assert np.radians(ang(unproject_ftheta(project_ftheta(rays, SPEC.f, SPEC.c), SPEC.f, SPEC.c), rays)).max() < 1e-6
    W, H = SPEC.lens_wh; ax = np.array([0, 0, 1.0])
    edge = lambda uv: float(ang(unproject_ftheta(np.array(uv, float), SPEC.f, SPEC.c), ax))
    assert abs(edge([0, H / 2]) - 73.0) < 0.05 and abs(edge([W, H / 2]) - 73.0) < 0.05   # 146 deg H, as the datasheet
    assert abs(edge([W / 2, 0]) - 61.6) < 0.05                                             # 123.2 deg V vs the datasheet's 124 (a pinhole: 140.2)

def test_crop_lies_inside_the_fisheye():
    s = SPEC.scale; K = SPEC.K(s); w, h = SPEC.net_wh[0] * s, SPEC.net_wh[1] * s; t = np.linspace(0, 1, 401)
    border = np.concatenate([np.stack([t * w, 0 * t], -1), np.stack([t * w, h + 0 * t], -1), np.stack([0 * t, t * h], -1), np.stack([w + 0 * t, t * h], -1)])
    uv = project_ftheta(unproject_pinhole(border, K), SPEC.f, SPEC.c); W, H = SPEC.lens_wh
    assert uv.min() > 0 and (uv[:, 0] < W).all() and (uv[:, 1] < H).all()
    r = lambda p: float(np.linalg.norm(project_ftheta(unproject_pinhole(np.array(p, float), K), SPEC.f, SPEC.c) - SPEC.c))
    assert abs(r([0, h / 2]) - 507) < 1 and abs(r([0, 0]) - 555) < 1                       # edge ray 57.8 deg, corner ray 63.3 deg

def test_rectifier_lut_matches_the_analytic_remap():
    r = FisheyeRectifier(SPEC); h, w = r.shape; K = SPEC.K(SPEC.scale); j, i = np.mgrid[0:h, 0:w]
    an = project_ftheta(unproject_pinhole(np.stack([i + 0.5, j + 0.5], -1), K), SPEC.f, SPEC.c)
    assert (h, w) == (240, 320) and np.abs(r.centroid() - an).max() < 0.1
    assert np.allclose(np.bincount(r.out, r.wt), 1, atol=1e-5)                            # weights of every output pixel sum to 1
    W, H = SPEC.lens_wh; v, u = np.mgrid[0:H, 0:W] + 0.5; ramp = np.stack([u, v], -1)     # a linear image is reproduced exactly
    assert np.abs(r(ramp) - an).max() < 0.1
    assert area_down(r(ramp), SPEC.scale).shape == (120, 160, 2)

def test_rectifier_refuses_a_crop_outside_the_lens():
    wide = HeadCameraSpec(); wide.fovy = 140                       # top edge ray 70 deg > the lens's 61.6 deg vertical half-FOV
    shifted = HeadCameraSpec(); shifted.c = shifted.c + [150, 0]   # principal point 150 px right: the crop's right edge had 133 px
    for sp in (wide, shifted):
        with pytest.raises(AssertionError, match="outside the lens"): FisheyeRectifier(sp, scale=1, sub=1)

def test_run_meta():
    r = run_meta(); assert len(r["git"]) == 40 and isinstance(r["git_dirty"], list) and r["mujoco"] == mujoco.__version__ and r["gl_renderer"] is None

def _fk_vs_sim(spec, n=200, seed=0):
    s = build_spec(cfg=CFG); add_head_camera(s, spec); m = s.compile(); d = mujoco.MjData(m); rb = R1Robot(m, CFG); rb.reset(m, d)
    lo, hi = m.jnt_range[[m.joint(j).id for j in CTRL_JOINTS]].T; rng = np.random.default_rng(seed); ep, er = [], []
    for _ in range(n):   # any base pose (upside down included), any joint angles in range
        q = rng.normal(size=4); d.qpos[rb.free_q:rb.free_q + 7] = np.r_[rng.normal(size=3), q / np.linalg.norm(q)]; d.qpos[rb.qadr] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d); g = d.xmat[rb.pel].reshape(3, 3).T @ [0, 0, -1.0] * rng.uniform(0.5, 2)   # gravity of any length
        p, R = mount_fk(d.qpos[rb.qadr], g, spec); pt, Rt = sim_cam_heading(d, m.camera("head_cam").id, rb.pel)
        ep.append(np.linalg.norm(p - pt)); er.append(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1))))
    return max(ep), max(er), m

def test_mount_fk_matches_the_sim_camera():
    for spec in (SPEC, HeadCameraSpec(head_pitch_deg=-20, lateral_offset_m=0.03)):
        ep, er, m = _fk_vs_sim(spec); assert ep < 1e-3 and er < 0.05, (ep, er)
    s = build_spec(cfg=CFG); add_head_camera(s, HeadCameraSpec(head_pitch_deg=0.0)); m = s.compile(); d = mujoco.MjData(m); R1Robot(m, CFG).reset(m, d)
    assert np.allclose(d.cam_xpos[0], [0.0325 + 0.088, 0, 0.789 + 0.355], atol=1e-3)     # lens 1.144 m up at home, head level (not the config's pitch)
    assert tuple(m.cam_resolution[0]) == (320, 240) and m.cam_fovy[0] == 100

def test_own_head_geoms():
    m = build_spec(opponent=True, cfg=CFG).compile()
    for pre in ("", "opp_"):
        g = own_head_geoms(m, SPEC, pre); assert len(g) == 2 and {m.mesh(m.geom_dataid[i]).name for i in g} == {pre + "head_pitch_link", pre + "head_yaw_link"}

def test_capture_clock_and_latency_queue_are_causal():
    rng = np.random.default_rng(0); late, lats, frames, pushed, dropped = [], [], [], 0, 0
    for ep in range(20):
        clk, lq = CaptureClock(SPEC, rng), LatencyQueue(SPEC, rng); n = 0; stamps = []
        assert SPEC.exposure[0] <= clk.exposure <= SPEC.exposure[1] and SPEC.latency[0] <= lq.base <= SPEC.latency[1]
        for k in range(1, 2001):   # 10 s of 5 ms substeps, popped at the 20 ms control rate
            t = k * SPEC.substep; tc = clk.poll(t)
            if tc is not None: stamps.append(tc); late.append(tc - clk.next + clk.period); lq.push(tc, n); n += 1
            if k % 4 == 0:
                for tc_, ta, item in lq.pop(t):
                    assert t >= ta >= tc_ + SPEC.latency[0] - 1e-12 and ta - tc_ <= SPEC.latency[1] + SPEC.jitter + 1e-12; lats.append(ta - tc_); frames.append(item)
        assert abs(len(stamps) / 10.0 - SPEC.rate) < 0.2 and np.all(np.diff(stamps) > 0); pushed += lq.pushed; dropped += lq.dropped
        assert frames == sorted(frames); frames = []                                       # FIFO: never out of order
    assert min(late) > -1e-9 and max(late) < SPEC.substep + 1e-9                      # stamped at the next substep, 0-5 ms late
    assert min(lats) >= SPEC.control_dt - 1e-12 and 0.005 < dropped / pushed < 0.03

def _gl():
    try: mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close(); return True
    except Exception: return False

@pytest.mark.skipif(not _gl(), reason="no offscreen GL context")
def test_render_puts_the_target_where_mount_fk_and_K_say():
    """Pins CV_MJ, K and the (w/2, h/2) principal point to what MuJoCo renders: a sphere placed by mount_fk + K alone on the
    ray through a chosen pixel renders there (segmentation centroid, 0.16 px worst seen; a 1 px principal-point shift fails),
    under random tilt, yaw, waist and mount."""
    rng = np.random.default_rng(0)
    for spec in (SPEC, HeadCameraSpec(head_pitch_deg=-20, lateral_offset_m=0.03)):
        s = build_spec(cfg=CFG); add_head_camera(s, spec); t = s.worldbody.add_body(name="target", mocap=True)
        t.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.08, 0, 0], group=4, contype=0, conaffinity=0)
        m = s.compile(); m.vis.quality.offsamples = 0   # no MSAA: a pixel is in the mask iff its centre is, so the centroid is unbiased
        d = mujoco.MjData(m); rb = R1Robot(m, CFG); rb.reset(m, d); K = spec.K(spec.scale); w, h = m.cam_resolution[0]; gid = m.geom(m.body("target").geomadr[0]).id
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[4] = 1   # the sphere alone: nothing occludes it
        rn = mujoco.Renderer(m, int(h), int(w)); rn.enable_segmentation_rendering(); wr = m.joint("waist_roll_joint").id; wy = m.joint("waist_yaw_joint").id
        for _ in range(6):   # any yaw, a 0-30 deg tilt about a random horizontal axis, waist anywhere in range
            a, th, yaw = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(0, 30)), rng.uniform(-np.pi, np.pi); q = np.zeros(4)
            mujoco.mju_mulQuat(q, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], [np.cos(th / 2), np.sin(th / 2) * np.cos(a), np.sin(th / 2) * np.sin(a), 0])
            d.qpos[rb.free_q:rb.free_q + 7] = np.r_[rng.uniform(-1, 1, 2), 1.0, q]
            for j in (wr, wy): d.qpos[m.jnt_qposadr[j]] = rng.uniform(*m.jnt_range[j])
            mujoco.mj_forward(m, d); Rp = d.xmat[rb.pel].reshape(3, 3); p, R = mount_fk(d.qpos[rb.qadr], Rp.T @ [0, 0, -1.0], spec)
            uv = rng.uniform([0.2 * w, 0.2 * h], [0.8 * w, 0.8 * h]); X = p + 2.0 * R @ unproject_pinhole(uv, K)   # 2 m out, heading frame
            d.mocap_pos[m.body("target").mocapid[0]] = d.xpos[rb.pel] + rz(np.arctan2(Rp[1, 0], Rp[0, 0])) @ X; mujoco.mj_forward(m, d)
            n = (uv - K[2:]) / K[0]; th, al = np.arctan(np.linalg.norm(n)), np.arcsin(0.08 / 2.0)   # perspective moves the silhouette ellipse's centre
            ctr = K[2:] + K[0] * n / np.linalg.norm(n) * (np.tan(th + al) + np.tan(th - al)) / 2   # outward (up to 0.4 px here): its radial extent is tan(th +- al)
            rn.update_scene(d, "head_cam", opt); seg = rn.render(); j, i = np.nonzero((seg[..., 0] == gid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
            assert len(i) > 30 and np.hypot(i.mean() + 0.5 - ctr[0], j.mean() + 0.5 - ctr[1]) < 0.3, (spec.pitch_deg, uv, ctr, len(i), i.mean() + 0.5, j.mean() + 0.5)
        rn.close()
