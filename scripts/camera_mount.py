"""Head-camera spec check (phase 3, M0.1) -> results/perception/mount.json. CPU, seconds; renders nothing, changes nothing.

1. Sources: configs/r1_camera.yaml's URLs. This script downloads nothing: rbtx_pdf_md5_checked_manually is a check done by
   hand on 2026-10-27 (the PDF at sources.rbtx vs the design's copy, byte-identical); --rbtx-pdf hashes a local copy here.
2. Lens fit: what 146 deg H over 1280 px implies for the vertical FOV with an f-theta (equidistant) lens vs a pinhole,
   against the datasheet's 124 deg; the same for mybotshop's 150 deg / 1088 rows.
3. Network camera and crop: the 160 x 120, fovy 100 virtual pinhole; where its border lands in the fisheye image (it must
   lie inside, so rectification never reads outside the lens) and how much of the lens image it keeps.
4. Rectifier: the lookup table's worst deviation from the analytic remap, its size and its cost.
5. Mount: the lens at home; mount_fk against the MuJoCo camera over random states (any base pose, joints in range) with
   noise-free encoders and gravity; then upright (tilt 0-30 deg) with one sample of the gym's obs noise (configs/r1_mjlab.yaml
   uniform half-widths: joint_pos 0.01 rad, gravity 0.05 per component; an upper bound, a real IMU filters gravity), head
   pitch 0 and -20 deg.
6. Clocks: 20 episodes x 60 s of capture + latency at 5 ms substeps: realised rate, stamp lag, latency, drops, early deliveries."""
import sys, json, time, argparse, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from r1gym import CONFIGS
from r1gym.model import build_spec, load_cfg, R1Robot, CTRL_JOINTS
from r1gym.perception import run_meta
from r1gym.perception.camera import (HeadCameraSpec, mount_fk, sim_cam_heading, add_head_camera, project_pinhole, unproject_pinhole,
    project_ftheta, unproject_ftheta, FisheyeRectifier, CaptureClock, LatencyQueue)

# done by hand, not by this script (it downloads nothing): the URL's PDF and the design's copy have this md5
RBTX_PDF = dict(file="rbtx_unitree_r1_variants_technical_datasheets_en-1.pdf", md5="a0fdbe4441328a4e35cb7e61e392170d", bytes=696747, date="2026-10-27",
                by="by hand: the PDF at sources.rbtx vs the design's copy dl_datasheet/r1.pdf")

def pdf_here(path):
    b = Path(path).read_bytes(); h = hashlib.md5(b).hexdigest(); return dict(file=Path(path).name, md5=h, bytes=len(b), matches_manual_check=h == RBTX_PDF["md5"])

def lens_fit(W, H, hfov, vfov):
    f = W / 2 / np.radians(hfov / 2); fp = W / 2 / np.tan(np.radians(hfov / 2))
    return dict(width=W, height=H, datasheet_hfov_deg=hfov, datasheet_vfov_deg=vfov, ftheta_f_px_per_rad=round(f, 1), ftheta_vfov_deg=round(float(np.degrees(H / f)), 1),
                pinhole_f_px=round(fp, 1), pinhole_vfov_deg=round(float(2 * np.degrees(np.arctan(H / 2 / fp))), 1))

def crop(spec):
    s = spec.scale; K = spec.K(s); w, h = spec.net_wh[0] * s, spec.net_wh[1] * s; W, H = spec.lens_wh; t = np.linspace(0, 1, 1001); z = 0 * t
    border = np.concatenate([np.stack([t * w, z], -1), np.stack([t * w, z + h], -1), np.stack([z, t * h], -1), np.stack([z + w, t * h], -1)])
    uv = project_ftheta(unproject_pinhole(border, K), spec.f, spec.c); r = lambda p: float(np.linalg.norm(project_ftheta(unproject_pinhole(np.array(p, float), K), spec.f, spec.c) - spec.c))
    th = lambda p: float(np.degrees(np.arccos(unproject_pinhole(np.array(p, float), K)[2])))
    v, u = np.mgrid[0:H, 0:W] + 0.5; ray = unproject_ftheta(np.stack([u, v], -1), spec.f, spec.c); P = project_pinhole(ray[ray[..., 2] > 0], K)
    seen = ((P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)).sum() / (W * H)
    return dict(edge_ray_deg=round(th([0, h / 2]), 1), edge_radius_px=round(r([0, h / 2]), 1), top_ray_deg=round(th([w / 2, 0]), 1), top_radius_px=round(r([w / 2, 0]), 1),
                corner_ray_deg=round(th([0, 0]), 1), corner_radius_px=round(r([0, 0]), 1), lens_half_extent_px=[W / 2, H / 2], lens_half_diagonal_px=round(float(np.hypot(W, H) / 2), 1),
                border_bbox_in_lens_px=[round(float(uv[:, 0].min()), 1), round(float(uv[:, 1].min()), 1), round(float(uv[:, 0].max()), 1), round(float(uv[:, 1].max()), 1)],
                margin_to_lens_edge_px=round(float(min(uv.min(), W - uv[:, 0].max(), H - uv[:, 1].max())), 1),
                inside=bool(uv.min() > 0 and uv[:, 0].max() < W and uv[:, 1].max() < H), lens_pixels_kept_frac=round(float(seen), 3))

def rectifier(spec):
    t = time.perf_counter(); r = FisheyeRectifier(spec); build = time.perf_counter() - t; h, w = r.shape; K = spec.K(spec.scale); j, i = np.mgrid[0:h, 0:w]
    an = project_ftheta(unproject_pinhole(np.stack([i + 0.5, j + 0.5], -1), K), spec.f, spec.c); img = np.random.default_rng(0).random((*spec.lens_wh[::-1], 3), np.float32); ms = []
    for _ in range(5): t = time.perf_counter(); r(img); ms.append(1e3 * (time.perf_counter() - t))
    return dict(out_wh=[w, h], sub_samples=r.sub, taps=int(len(r.src)), taps_per_pixel_mean=round(len(r.src) / (w * h), 1), max_centroid_err_px=float(f"{np.abs(r.centroid() - an).max():.2g}"),
                build_s=round(build, 2), apply_ms_rgb_numpy=round(float(np.median(ms)), 1))

def fk_check(spec, n, seed=0):
    cfg = load_cfg(); s = build_spec(cfg=cfg); add_head_camera(s, spec); m = s.compile(); d = mujoco.MjData(m); rb = R1Robot(m, cfg); rb.reset(m, d)
    cam = m.camera("head_cam").id; home = dict(world_m=d.cam_xpos[cam].round(4).tolist(), heading_frame_m=sim_cam_heading(d, cam, rb.pel)[0].round(4).tolist(),
                                               optical_axis_elevation_deg=round(float(np.degrees(np.arcsin(-d.cam_xmat[cam].reshape(3, 3)[2, 2]))), 2) + 0.0)
    noise = load_cfg(CONFIGS / "r1_mjlab.yaml")["sensors"]["noise"]; lo, hi = m.jnt_range[[m.joint(j).id for j in CTRL_JOINTS]].T
    rng = np.random.default_rng(seed); out = {"clean": [], "noisy": []}
    for k in ("clean", "noisy"):   # clean: any base pose (upside down included); noisy: upright, tilted 0-30 deg (heading is undefined near 90)
        for _ in range(n):
            if k == "clean": q = rng.normal(size=4); q /= np.linalg.norm(q)
            else:   # any yaw, then a 0-30 deg tilt about a random horizontal axis
                a, th, yaw = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(0, 30)), rng.uniform(-np.pi, np.pi); q = np.zeros(4)
                mujoco.mju_mulQuat(q, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], [np.cos(th / 2), np.sin(th / 2) * np.cos(a), np.sin(th / 2) * np.sin(a), 0])
            d.qpos[rb.free_q:rb.free_q + 7] = np.r_[rng.normal(size=3), q]; d.qpos[rb.qadr] = rng.uniform(lo, hi); mujoco.mj_forward(m, d)
            g = d.xmat[rb.pel].reshape(3, 3).T @ [0, 0, -1.0]; pt, Rt = sim_cam_heading(d, cam, rb.pel); qj = d.qpos[rb.qadr]
            if k == "noisy": qj, g = qj + rng.uniform(-1, 1, 24) * noise["joint_pos"], g + rng.uniform(-1, 1, 3) * noise["gravity"]
            p, R = mount_fk(qj, g, spec); out[k].append((1e3 * np.linalg.norm(p - pt), np.degrees(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1)))))
    st = lambda a: dict(median=float(f"{np.median(a):.3g}"), p95=float(f"{np.percentile(a, 95):.3g}"), max=float(f"{np.max(a):.3g}"))
    return dict(home=home, states=n, **{k: dict(pos_err_mm=st([e[0] for e in v]), rot_err_deg=st([e[1] for e in v])) for k, v in out.items()})

def clocks(spec, eps=20, seconds=60.0, seed=0):
    rng = np.random.default_rng(seed); n = int(round(seconds / spec.substep)); every = int(round(spec.control_dt / spec.substep))
    lag, lat, early, rates = [], [], 0, []; pushed = dropped = 0
    for _ in range(eps):
        clk, lq = CaptureClock(spec, rng), LatencyQueue(spec, rng)
        for k in range(1, n + 1):
            t = k * spec.substep; tc = clk.poll(t)
            if tc is not None: lag.append(tc - (clk.next - clk.period)); lq.push(tc, k)
            if k % every == 0:
                for tc_, ta, _ in lq.pop(t): lat.append(ta - tc_); early += ta - tc_ < spec.latency[0] - 1e-12 or ta > t
        rates.append(clk.k / seconds); pushed += lq.pushed; dropped += lq.dropped
    ms = lambda a: [round(1e3 * float(np.min(a)), 2), round(1e3 * float(np.max(a)), 2)]
    return dict(episodes=eps, seconds=seconds, rate_hz=[round(min(rates), 2), round(max(rates), 2)], stamp_lag_ms=ms(lag), latency_ms=ms(lat),
                latency_median_ms=round(1e3 * float(np.median(lat)), 1), drop_frac=round(dropped / pushed, 4), early_deliveries=int(early),
                latency_min_ge_control_dt=bool(spec.latency[0] >= spec.control_dt - 1e-12))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results" / "perception" / "mount.json"))
    ap.add_argument("--states", type=int, default=1000); ap.add_argument("--rbtx-pdf", help="a local copy of the RBTX datasheet to hash (optional)"); a = ap.parse_args()
    spec = HeadCameraSpec(); L = spec.cfg["lens"]; W, H = spec.lens_wh; ax = np.array([0, 0, 1.0]); K = spec.K()
    edge = lambda uv: round(float(np.degrees(np.arccos(unproject_ftheta(np.array(uv, float), spec.f, spec.c) @ ax))), 2)
    res = {"about": __doc__, **run_meta(), "gl_note": "M0.1 renders nothing (no GL context); render results record the renderer string",
           "config": "configs/r1_camera.yaml", "sources": spec.cfg["sources"], "rbtx_pdf_md5_checked_manually": RBTX_PDF, "rbtx_pdf_hashed_here": a.rbtx_pdf and pdf_here(a.rbtx_pdf),
           "lens_fit": {"rbtx": lens_fit(W, H, L["hfov_deg"], L["vfov_deg"]), "mybotshop": lens_fit(L["alt"]["width"], L["alt"]["height"], L["alt"]["hfov_deg"], L["alt"]["vfov_deg"]),
                        "used": dict(model=L["model"], f_px_per_rad=spec.f, half_fov_edges_deg=dict(horizontal=edge([0, H / 2]), vertical=edge([W / 2, 0])))},
           "net_camera": dict(wh=list(spec.net_wh), render_wh=[spec.net_wh[0] * spec.scale, spec.net_wh[1] * spec.scale], fovy_deg=spec.fovy, fx_px=round(float(K[0]), 2),
                              hfov_deg=round(float(2 * np.degrees(np.arctan(K[2] / K[0]))), 1), deg_per_px_centre=round(float(np.degrees(1 / K[0])), 3),
                              lens_px_per_net_px_centre=round(float(spec.f / K[0]), 2)),
           "crop": crop(spec), "rectifier": rectifier(spec), "mount_torso_link": {}, "mount_fk_vs_sim": {}, "clocks": clocks(spec)}
    for pitch in (0.0, -20.0):
        sp = HeadCameraSpec(head_pitch_deg=pitch); p, R = sp.mount_torso
        res["mount_torso_link"][f"pitch_{pitch:g}"] = dict(lens_m=p.round(4).tolist(), optical_axis=R[:, 2].round(4).tolist())
        res["mount_fk_vs_sim"][f"pitch_{pitch:g}"] = fk_check(sp, a.states); print(f"pitch {pitch:g}", json.dumps(res["mount_fk_vs_sim"][f"pitch_{pitch:g}"]), flush=True)
    for k in ("lens_fit", "net_camera", "crop", "rectifier", "clocks"): print(k, json.dumps(res[k]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
