"""The R1 head camera: spec, mount kinematics, lens models, fisheye rectifier, capture clock and latency (phase 3, M0.1).

Every number comes from configs/r1_camera.yaml. Nothing here touches the physics model or the env: a camera is only ever
added to a separate render model (add_head_camera, M0.2's render twin), so the gym behaves exactly as before.

Frames and conventions:
- torso_link: r1.xml's axes (x forward, y left, z up); the head is rigid on it.
- Pelvis heading frame: origin at the pelvis origin (= the IMU site), z up, x = the pelvis x-axis projected on the ground.
  It has no world yaw, so the robot gets it from IMU gravity alone; yaw is the same angle env._face uses.
- Camera axes: OpenCV (x right, y down, z = optical axis). A MuJoCo camera has y up and looks along -z (CV_MJ flips them).
- Pixels: continuous coordinates; pixel (i, j) covers [i, i+1) x [j, j+1), so its centre is (i + 0.5, j + 0.5) and a
  centred camera's principal point is (w / 2, h / 2).
- Clocks: sim time (d.time) in seconds."""
from __future__ import annotations
from collections import deque
import numpy as np, mujoco, yaml
from .. import CONFIGS
from ..model import CTRL_JOINTS, load_cfg

CV_MJ = np.diag([1.0, -1.0, -1.0])   # OpenCV camera axes <-> MuJoCo camera axes (columns), its own inverse
CV_TORSO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # head level: right = -y, down = -z, optical axis = +x
# r1.xml (unitree_rl_mjlab @1425b15) lines 163-169: waist_roll_link sits at the pelvis origin (roll about x), torso_link at
# (0.0325, 0, 0.049) in it (yaw about z), neither rotated. The robot has no MuJoCo, so mount_fk hard-codes the chain;
# tests/test_perception_camera.py checks it against the compiled model.
TORSO_IN_WAIST = np.array([0.0325, 0.0, 0.049])
WAIST = [CTRL_JOINTS.index("waist_roll_joint"), CTRL_JOINTS.index("waist_yaw_joint")]

def rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

class HeadCameraSpec:
    """configs/r1_camera.yaml as numbers. Keyword overrides replace mount or timing entries (e.g. head_pitch_deg=-20) for
    studies and randomisation; the physics config supplies the substep and control period the clocks run on."""
    def __init__(self, path=None, gym_cfg=None, **over):
        c = yaml.safe_load(open(path or CONFIGS / "r1_camera.yaml")); g = (gym_cfg or load_cfg())["model"]; self.cfg = c
        bad = set(over) - set(c["mount"]) - set(c["timing"])
        if bad: raise KeyError(f"not a mount/timing key: {sorted(bad)}")
        M = {**c["mount"], **{k: v for k, v in over.items() if k in c["mount"]}}; T = {**c["timing"], **{k: v for k, v in over.items() if k in c["timing"]}}
        L, N = c["lens"], c["net"]; self.lens_wh = (int(L["width"]), int(L["height"])); self.f = float(L["f_px_per_rad"]); self.c = np.array(self.lens_wh) / 2
        self.net_wh = (int(N["width"]), int(N["height"])); self.fovy = float(N["fovy_deg"]); self.scale = int(N["render_scale"])
        self.body = M["body"]; self.pos = np.array(M["pos"], float) + [0, M["lateral_offset_m"], 0]; self.pitch_deg = float(M["head_pitch_deg"])
        self.pivot = np.array(M["pitch_pivot"], float); self.hide_own = list(M["hide_own"])
        self.rate = float(T["rate_hz"]); self.exposure = np.array(T["exposure_ms"], float) / 1e3; self.latency = np.array(T["latency_ms"], float) / 1e3
        self.jitter = T["jitter_ms"] / 1e3; self.drop = tuple(T["drop"]); self.substep, self.control_dt = g["timestep"], g["control_dt"]
        # causality: a frame captured during control step k is then never due before the end of step k + 1, so the batched
        # pipeline (M1's PerceptHub) can render a whole step's snapshots after that step and is never late
        assert self.latency[0] >= self.control_dt - 1e-12, f"latency_min {self.latency[0]} s < control_dt {self.control_dt} s breaks causality"

    def K(self, scale=1):
        """Virtual-pinhole intrinsics (fx, fy, cx, cy) at `scale` x the network size (scale=self.scale: the render)."""
        w, h = self.net_wh[0] * scale, self.net_wh[1] * scale; f = h / 2 / np.tan(np.radians(self.fovy) / 2)
        return np.array([f, f, w / 2, h / 2])

    @property
    def mount_torso(self):
        """Lens position [m] and camera axes (columns, OpenCV) in torso_link: the head pitched by head_pitch_deg (elevation of
        the optical axis, + up) about the assumed pivot. Ry(-e) turns the forward axis up by e."""
        P = ry(-np.radians(self.pitch_deg)); return self.pivot + P @ (self.pos - self.pivot), P @ CV_TORSO

def heading_tilt(g):
    """Rotation from the pelvis frame to the pelvis heading frame, from IMU projected gravity g = R^T (0, 0, -1) (any
    length, e.g. noisy): rows are the heading frame's x, y, z in pelvis coordinates. Its y has no pelvis-x component, which
    is what makes the heading x the pelvis x projected on the ground."""
    u = -np.asarray(g, float) / np.linalg.norm(g); y = np.array([0.0, u[2], -u[1]]) / np.hypot(u[1], u[2])
    return np.stack([np.cross(y, u), y, u])

def mount_fk(q, g, spec):
    """Camera pose in the pelvis heading frame from sensed joint positions q (24, deploy order: model.CTRL_JOINTS) and IMU
    projected gravity g: (lens position [m] from the pelvis origin, camera axes as columns, OpenCV). Uses only the waist
    encoders and gravity, never world yaw, so it runs unchanged on the robot."""
    roll, yaw = np.asarray(q, float)[WAIST]; Rw = rx(roll); Rt = Rw @ rz(yaw); p, R = spec.mount_torso; T = heading_tilt(g)
    return T @ (Rw @ TORSO_IN_WAIST + Rt @ p), T @ Rt @ R

def sim_cam_heading(d, cam, pel):
    """Ground truth for mount_fk: MuJoCo camera `cam`'s pose in body `pel`'s heading frame (world yaw removed exactly).
    Needs current camera poses: mj_forward, or mj_kinematics + mj_camlight (mj_kinematics alone leaves cam_xpos stale)."""
    R = d.xmat[pel].reshape(3, 3); H = rz(-np.arctan2(R[1, 0], R[0, 0]))
    return H @ (d.cam_xpos[cam] - d.xpos[pel]), H @ d.cam_xmat[cam].reshape(3, 3) @ CV_MJ

def add_head_camera(s: mujoco.MjSpec, spec: HeadCameraSpec, prefix="", name="head_cam"):
    """Add the head camera to a RENDER model's spec on prefix + spec.body ("opp_" for the opponent). Never call it on the
    physics model: cameras live in the render twin (M0.2), so physics stays bit-identical."""
    p, R = spec.mount_torso; q = np.zeros(4); mujoco.mju_mat2Quat(q, (R @ CV_MJ).ravel())
    w, h = spec.net_wh
    return s.body(prefix + spec.body).add_camera(name=prefix + name, pos=p, quat=q, fovy=spec.fovy, resolution=[w * spec.scale, h * spec.scale])

def own_head_geoms(m, spec, prefix=""):
    """Geom ids of the head shell (spec.hide_own meshes on prefix + spec.body) to hide from that robot's own camera."""
    b = m.body(prefix + spec.body).id; names = {prefix + n for n in spec.hide_own}
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name in names]

# ---- lens models. P: points (..., 3) in camera axes (OpenCV); uv: (..., 2) pixels; rays come back unit length
def project_pinhole(P, K):
    """Pinhole projection (z > 0 only)."""
    fx, fy, cx, cy = K; P = np.asarray(P, float); return np.stack([cx + fx * P[..., 0] / P[..., 2], cy + fy * P[..., 1] / P[..., 2]], -1)

def unproject_pinhole(uv, K):
    fx, fy, cx, cy = K; uv = np.asarray(uv, float); r = np.stack([(uv[..., 0] - cx) / fx, (uv[..., 1] - cy) / fy, np.ones(uv.shape[:-1])], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)

def project_ftheta(P, f, c):
    """Equidistant (f-theta) fisheye: image radius = f * angle off the optical axis, valid out to 180 deg."""
    P = np.asarray(P, float); th = np.arctan2(np.hypot(P[..., 0], P[..., 1]), P[..., 2]); ph = np.arctan2(P[..., 1], P[..., 0])
    return np.stack([c[0] + f * th * np.cos(ph), c[1] + f * th * np.sin(ph)], -1)

def unproject_ftheta(uv, f, c):
    d = np.asarray(uv, float) - c; th = np.hypot(d[..., 0], d[..., 1]) / f; ph = np.arctan2(d[..., 1], d[..., 0])
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)

def area_down(img, k):
    """k x k box (area) downsample: the render (sim) or rectified (robot) image -> the network size."""
    h, w = img.shape[:2]; return np.asarray(img, np.float32).reshape(h // k, k, w // k, k, *img.shape[2:]).mean((1, 3))

class FisheyeRectifier:
    """f-theta lens image -> the virtual pinhole at `scale` x the network size, as a sparse lookup table. Each output pixel
    is the mean of the bilinearly interpolated fisheye over its own footprint (sub x sub samples): area weighting, so the
    up to ~5 x 5 lens pixels behind one render pixel are averaged, not aliased. The robot and the Cycles test (M5) rectify
    at the render scale and then area_down() to the network size, as the sim does with its 2x render."""
    def __init__(self, spec: HeadCameraSpec, scale=None, sub=None):
        s = spec.scale if scale is None else scale; K = spec.K(s); W, H = spec.lens_wh; w, h = spec.net_wh[0] * s, spec.net_wh[1] * s
        sub = sub or int(np.ceil(spec.f / K[0]))   # sample spacing <= 1 lens pixel at the centre, where lens pixels are densest
        o = (np.arange(sub) + 0.5) / sub; j, i, b, a = np.meshgrid(np.arange(h), np.arange(w), o, o, indexing="ij")
        src = project_ftheta(unproject_pinhole(np.stack([i + a, j + b], -1).reshape(-1, 2), K), spec.f, spec.c) - 0.5   # lens pixel-centre grid
        # the nominal crop keeps 101.7 px clear of the lens edge (camera_mount.py), but a randomised fovy or principal point
        # (M2) or the mybotshop lens may not: refuse rather than silently replicate edge pixels into the image
        assert (src > -0.5).all() and (src[:, 0] < W - 0.5).all() and (src[:, 1] < H - 0.5).all(), "virtual pinhole reads outside the lens image"
        x0 = np.floor(src).astype(np.int64); fr = src - x0; out = np.repeat(np.arange(w * h, dtype=np.int64), sub * sub); keys, wts = [], []
        for dx in (0, 1):
            for dy in (0, 1):   # border taps clamp: only within half a lens pixel of its edge (the assert above)
                keys.append(out * (W * H) + np.clip(x0[:, 1] + dy, 0, H - 1) * W + np.clip(x0[:, 0] + dx, 0, W - 1))
                wts.append((fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]))
        k, inv = np.unique(np.concatenate(keys), return_inverse=True)   # merge repeated taps: sorted by output, then source
        self.src = k % (W * H); out = k // (W * H); self.wt = (np.bincount(inv.ravel(), np.concatenate(wts)) / sub ** 2).astype(np.float32)
        self.starts = np.searchsorted(out, np.arange(w * h)); self.out = out; self.shape, self.lens_wh, self.sub = (h, w), (W, H), sub

    def __call__(self, img):
        """(H, W[, C]) lens image -> (h, w[, C]) float32 rectified image."""
        x = np.asarray(img, np.float32); f = x.reshape(self.lens_wh[0] * self.lens_wh[1], -1)
        return np.add.reduceat(f[self.src] * self.wt[:, None], self.starts).reshape(*self.shape, *x.shape[2:])

    def centroid(self):
        """Weighted mean lens coordinate behind each output pixel, (h, w, 2): equals the analytic remap of the pixel centre."""
        W = self.lens_wh[0]; n = self.shape[0] * self.shape[1]
        return np.stack([np.bincount(self.out, self.wt * (self.src % W + 0.5), n), np.bincount(self.out, self.wt * (self.src // W + 0.5), n)], -1).reshape(*self.shape, 2)

# ---- clocks
class CaptureClock:
    """Exposures at spec.rate with a random phase per episode. The sim can only snapshot at a physics substep, so a frame is
    stamped at the first substep at or after its nominal mid-exposure time (0-5 ms late); that stamp is t_capture.
    Exposure length is sampled per episode (auto-exposure drifts slowly) and kept for motion blur."""
    def __init__(self, spec: HeadCameraSpec, rng, t0=0.0): self.period = 1.0 / spec.rate; self.spec = spec; self.reset(rng, t0)
    def reset(self, rng, t0=0.0):
        self.t0 = t0 + rng.uniform(0, self.period); self.k = 0; self.exposure = float(rng.uniform(*self.spec.exposure))
    @property
    def next(self): return self.t0 + self.k * self.period   # no accumulated rounding over long episodes
    def poll(self, t):
        """Call once per substep with d.time after mj_step: the capture stamp if an exposure falls due, else None."""
        if t < self.next - 1e-9: return None   # 1e-9: d.time accumulates rounding from adding the 5 ms substep
        self.k += 1; return t

class LatencyQueue:
    """Camera -> estimate transport. Each frame is dropped with probability p (per episode, U(drop)) or delivered at
    t_capture + latency: latency = base (per episode, U(latency)) + per-frame jitter U(+-jitter), never below latency[0]
    (so the causality bound holds; this clamp is the simplest choice the design leaves open), and in order (FIFO)."""
    def __init__(self, spec: HeadCameraSpec, rng): self.spec = spec; self.reset(rng)
    def reset(self, rng):
        s = self.spec; self.rng = rng; self.base = float(rng.uniform(*s.latency)); self.p = float(rng.uniform(*s.drop))
        self.q = deque(); self.last = -np.inf; self.pushed = self.dropped = 0
    def push(self, t_capture, item):
        """Queue a frame; returns its delivery time, or None if it was dropped."""
        self.pushed += 1
        if self.rng.random() < self.p: self.dropped += 1; return None
        lat = max(self.base + self.rng.uniform(-self.spec.jitter, self.spec.jitter), self.spec.latency[0])
        t = self.last = max(t_capture + lat, self.last); self.q.append((t_capture, t, item)); return t
    def pop(self, t):
        """Frames delivered by time t, oldest first, as (t_capture, t_avail, item)."""
        out = []
        while self.q and self.q[0][1] <= t: out.append(self.q.popleft())
        return out
