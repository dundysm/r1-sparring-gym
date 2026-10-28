"""Colour-key opponent detector (phase 3, M1): no learning, no data. It finds the opponent by its colour in the nominal scene
(orange, rgba 0.85/0.35/0.25: hue 10 deg; the R1 is silver, the floor blue-grey, the walls grey), so it exploits the sim's
colours: it checks the camera chain (mount, clocks, latency, ego compensation, wiring), not perception. The learned
estimator (M3) replaces it.

From one network image (160x120 uint8 RGB) and the proprioception sensed at capture (joint encoders, IMU gravity):
- mask: hue 4-22 deg, saturation >= 0.4, value >= 0.08 (the R1's own red ankle links sit at hue 0, its blue ones at 240);
- presence: at least MIN_PX mask pixels;
- each mask pixel's ray in the pelvis heading frame: the nominal intrinsics and mount_fk (sensed waist and gravity);
- range: the opponent stands on the floor, so the nearest of its pixels' floor-plane hits (NEAR_PCT percentile; a raised
  part hits the plane farther away than it is) is its nearest floor contact, taken as the distance to its pelvis (fighting
  stances lean in over the front foot; the M1 results report the bias). The lens height above the floor = mount_fk's lens
  height above the pelvis + the pelvis height above the lower foot sole, by forward kinematics of the R1's own model from
  its encoders (the robot knows its own body; the sim state is never read);
- bearing: the median azimuth of the mask pixels' rays from the lens (a vertical body projects to one azimuth; the median
  shrugs off an extended arm), placed at that range and re-expressed from the pelvis: at head pitch -50 the lens sits
  16 cm ahead of the pelvis, 12.5 deg of parallax at 0.6 m and 40 deg.
sigmas are uncalibrated heuristics, not fitted (bearing: one pixel and half the silhouette's robust azimuth spread, since
the pelvis can sit anywhere across the visible body; range: 15%); the M1 results report their coverage. NEAR_PCT and the
thresholds are fixed from the geometry above; development used train-clip frames only."""
from __future__ import annotations
import numpy as np, mujoco
from ..model import build_spec, load_cfg, CTRL_JOINTS
from .camera import HeadCameraSpec, mount_fk, heading_tilt, unproject_pinhole

class ColorKeyDetector:
    HUE, SAT, VAL, MIN_PX, NEAR_PCT = (4.0, 22.0), 0.4, 0.08, 12, 3.0
    RANGE_SD = 0.15   # ASSUMPTION: relative

    def __init__(self, spec: HeadCameraSpec, cfg=None):
        self.spec = spec; w, h = spec.net_wh; j, i = np.mgrid[0:h, 0:w]
        self.rays = unproject_pinhole(np.stack([i + 0.5, j + 0.5], -1), spec.K(1))   # (h, w, 3), camera axes (OpenCV), at pixel centres
        self.px = 1.0 / spec.K(1)[0]   # one pixel at the image centre [rad]
        self.m = build_spec(cfg=cfg or load_cfg()).compile(); self.d = mujoco.MjData(self.m)   # the R1 alone: its own kinematic model
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j_).id] for j_ in CTRL_JOINTS]); self.feet = [self.m.site(s).id for s in ("left_foot", "right_foot")]

    def mask(self, img):
        x = np.asarray(img, np.float32) / 255.0; r, g, b = x[..., 0], x[..., 1], x[..., 2]; mx = x.max(-1); c = mx - x.min(-1)
        with np.errstate(divide="ignore", invalid="ignore"): hue = 60.0 * (g - b) / c   # the red sector (r is the max): hue in -60..60
        return (r >= mx) & (c > 0) & (hue >= self.HUE[0]) & (hue <= self.HUE[1]) & (c >= self.SAT * mx) & (mx >= self.VAL)

    def pelvis_height(self, q, g):
        """Pelvis origin above the lower foot sole (the r1.xml foot sites sit on the soles), from encoders and IMU gravity."""
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, heading_tilt(g).ravel())   # the pelvis's orientation in the heading frame
        d = self.d; d.qpos[:3] = 0; d.qpos[3:7] = quat; d.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, d)
        return float(-d.site_xpos[self.feet, 2].min())

    def __call__(self, img, q, g):
        mk = self.mask(img); n = int(mk.sum())
        if n < self.MIN_PX: return dict(presence=False, bearing=np.nan, bearing_sd=np.nan, range=np.nan, range_sd=np.nan, rel_heading=np.nan, n_px=n)
        p, R = mount_fk(q, g, self.spec); r = self.rays[mk] @ R.T   # rays in the pelvis heading frame
        az = np.arctan2(r[:, 1], r[:, 0]); a = float(np.median(az)); h = self.pelvis_height(q, g) + p[2]   # lens height above the floor
        dn = r[:, 2] < -1e-3; dist = h * np.hypot(r[dn, 0], r[dn, 1]) / -r[dn, 2]
        if dn.sum() >= self.MIN_PX and h > 0:
            rl = float(np.percentile(dist, self.NEAR_PCT)); xy = p[:2] + rl * np.array([np.cos(a), np.sin(a)])
            bearing, rng = float(np.arctan2(xy[1], xy[0])), float(np.hypot(*xy))
        else: bearing, rng = a, np.nan   # no floor hit (pitched up): the lens azimuth, no parallax correction
        mad = 1.4826 * float(np.median(np.abs(az - a)))
        return dict(presence=True, bearing=bearing, bearing_sd=float(np.hypot(self.px, 0.5 * mad)), range=rng, range_sd=self.RANGE_SD * rng,
                    rel_heading=np.nan, n_px=n)
