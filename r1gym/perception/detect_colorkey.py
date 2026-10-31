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
thresholds are fixed from the geometry above; development used train-clip frames only.

Gloves (gloves=True; render_scene.set_gloves paints the opponent's hands green, hue 120): the returned dict also carries
'gloves' = {"left": glove or None, "right": ...}, the opponent's own left and right hand. The existing keys are computed
exactly as before (from the orange mask alone, so a gloved opponent's hands no longer count toward its bearing). Per glove:
- mask: hue 95-145 deg, saturation >= 0.4, value >= 0.08; 8-connected blobs of at least GLOVE_MIN_PX pixels, largest first: a blob within
  GLOVE_MERGE_PX (Chebyshev) of a kept one is a fragment of that glove (cut by a forearm) and joins it, else it is kept if
  fewer than two are;
- side: the opponent faces the R1, so its left hand is on the image's right: of two blobs the one further right is "left";
  a lone blob is "left" if it lies right of the orange body's median column (the image centre without a body);
- u, v: the pixel centroid (network pixels, continuous coordinates), n_px its pixel count; edge: the blob touches the image
  border, so part of the glove may be cut off (its centroid and range are then biased);
- bearing, elevation [rad]: the centroid's ray in the head frame (torso_link axes from the lens, + left / + up), from the
  nominal intrinsics and mount only;
- range [m], lens to glove, from apparent size: the blob's solid angle (pixels weighted by cos^3 of their angle off the
  axis) is GLOVE_AREA / range^2, GLOVE_AREA fitted on train-clip frames (scripts/glove_check.py);
- xyz [m]: range along that ray placed in the pelvis heading frame (mount_fk: sensed waist and gravity), as 'bearing' is.
Each frame stands alone; GloveTracker (per stream) relabels sides by continuity. results/perception/gloves_check.json measures
both labels; gate and gap of the tracker were chosen on train frames."""
from __future__ import annotations
import numpy as np, mujoco
from ..model import build_spec, load_cfg, CTRL_JOINTS
from .camera import HeadCameraSpec, mount_fk, heading_tilt, unproject_pinhole

class ColorKeyDetector:
    HUE, SAT, VAL, MIN_PX, NEAR_PCT = (4.0, 22.0), 0.4, 0.08, 12, 3.0
    RANGE_SD = 0.15   # ASSUMPTION: relative
    GLOVE_HUE, GLOVE_MIN_PX, GLOVE_MERGE_PX = (95.0, 145.0), 4, 3
    GLOVE_AREA = 0.007045   # [m^2] effective glove area, fitted on train frames: results/perception/gloves_check.json train.fit (scripts/glove_check.py)

    def __init__(self, spec: HeadCameraSpec, cfg=None, gloves=False):
        self.spec = spec; self.gloves = bool(gloves); w, h = spec.net_wh; j, i = np.mgrid[0:h, 0:w]
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
        out = self._opponent(img, q, g)
        if self.gloves: out["gloves"] = self.find_gloves(img, q, g)
        return out

    def _opponent(self, img, q, g):
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

    def glove_mask(self, img):
        x = np.asarray(img, np.float32) / 255.0; r, g, b = x[..., 0], x[..., 1], x[..., 2]; mx = x.max(-1); c = mx - x.min(-1)
        with np.errstate(divide="ignore", invalid="ignore"): hue = 120.0 + 60.0 * (b - r) / c   # the green sector (g is the max): hue in 60..180
        return (g >= mx) & (c > 0) & (hue >= self.GLOVE_HUE[0]) & (hue <= self.GLOVE_HUE[1]) & (c >= self.SAT * mx) & (mx >= self.VAL)

    def find_gloves(self, img, q, g):
        """{"left": glove dict or None, "right": ...} (module docstring)."""
        main = []
        for b in (b for b in _blobs(self.glove_mask(img)) if len(b) >= self.GLOVE_MIN_PX):   # largest first
            near = [k for k, (grp, dil) in enumerate(main) if dil[b[:, 0], b[:, 1]].any()]
            if near: grp, _ = main[near[0]]; grp = np.concatenate([grp, b]); main[near[0]] = (grp, _dilate(grp, self.GLOVE_MERGE_PX, img.shape[:2]))   # a fragment
            elif len(main) < 2: main.append((b, _dilate(b, self.GLOVE_MERGE_PX, img.shape[:2])))
        out = dict(left=None, right=None); main = [grp for grp, _ in main]
        if not main: return out
        cen = [b.mean(0) + 0.5 for b in main]   # (v, u) pixel-centre coordinates
        if len(main) == 2: sides = ("left", "right") if cen[0][1] > cen[1][1] else ("right", "left")
        else:
            mk = self.mask(img); uc = float(np.median(np.nonzero(mk)[1]) + 0.5) if mk.sum() >= self.MIN_PX else self.spec.net_wh[0] / 2
            sides = ("left",) if cen[0][1] > uc else ("right",)
        p, R = mount_fk(q, g, self.spec); Rm = self.spec.mount_torso[1]; K = self.spec.K(1); sa = self.rays[..., 2] ** 3 / (K[0] * K[1])   # solid angle per pixel [sr]
        for b, (v, u), side in zip(main, cen, sides):
            ray = unproject_pinhole(np.array([u, v]), K); t = Rm @ ray; om = float(sa[b[:, 0], b[:, 1]].sum())
            rng = float(np.sqrt(self.GLOVE_AREA / om)) if self.GLOVE_AREA else np.nan
            edge = bool((b[:, 0] == 0).any() or (b[:, 0] == self.rays.shape[0] - 1).any() or (b[:, 1] == 0).any() or (b[:, 1] == self.rays.shape[1] - 1).any())
            out[side] = dict(u=float(u), v=float(v), n_px=int(len(b)), edge=edge, bearing=float(np.arctan2(t[1], t[0])), elevation=float(np.arctan2(t[2], np.hypot(t[0], t[1]))),
                             solid_angle=om, range=rng, xyz=(p + rng * (R @ ray)).tolist())
        return out

def _dilate(px, r, shape):
    """Mask of the pixels within Chebyshev distance r of the (n, 2) (row, col) pixels px."""
    mk = np.zeros(shape, bool); mk[px[:, 0], px[:, 1]] = True; out = mk.copy()
    for s in range(1, r + 1): out[s:] |= mk[:-s]; out[:-s] |= mk[s:]
    mk = out.copy()
    for s in range(1, r + 1): out[:, s:] |= mk[:, :-s]; out[:, :-s] |= mk[:, s:]
    return out

def _blobs(mk):
    """8-connected components of a boolean mask as (n, 2) int arrays of (row, col), largest first (ties: first pixel order)."""
    rows, cols = np.nonzero(mk); left = set(zip(rows.tolist(), cols.tolist())); out = []
    for px in zip(rows.tolist(), cols.tolist()):
        if px not in left: continue
        left.discard(px); st, comp = [px], []
        while st:
            y, x = st.pop(); comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (y + dy, x + dx)
                    if nb in left: left.discard(nb); st.append(nb)
        out.append(np.array(sorted(comp)))
    return sorted(out, key=len, reverse=True)

class GloveTracker:
    """Left / right by continuity, for one stream's frames in capture order (the detector itself stays per frame, so streams
    can share it). Each side keeps its last head-frame direction; a frame's 0-2 gloves go to the sides by least total cost:
    the angle [deg] to that side's track while it is fresh (seen within max_gap_s), else 0 if the image-side label agrees and
    gate_deg if not. So the image side starts a track and continuity keeps it while the fists cross in azimuth at different
    heights. No motion model: two fists passing close in the image can swap (a constant-rate prediction did no better on
    train frames). gate_deg and max_gap_s were chosen on train frames (scripts/glove_check.py)."""
    def __init__(self, gate_deg=30.0, max_gap_s=0.1): self.gate, self.gap = gate_deg, max_gap_s; self.reset()

    def reset(self): self.tr = dict(left=None, right=None)   # side -> (t, unit direction in the head frame)

    def __call__(self, gloves, t):
        """gloves: find_gloves' dict; returns the same gloves keyed by tracked side (each with 'side_image', its own label)."""
        D = [dict(v, side_image=k) for k, v in gloves.items() if v is not None]; out = dict(left=None, right=None)
        if not D: return out
        u = [np.array([np.cos(v["elevation"]) * np.cos(v["bearing"]), np.cos(v["elevation"]) * np.sin(v["bearing"]), np.sin(v["elevation"])]) for v in D]
        def cost(i, s):
            x = self.tr[s]
            if x is not None and t - x[0] <= self.gap: return float(np.degrees(np.arccos(np.clip(u[i] @ x[1], -1, 1))))
            return 0.0 if D[i]["side_image"] == s else self.gate
        opts = [((0, "left"), (1, "right")), ((0, "right"), (1, "left"))] if len(D) == 2 else [((0, "left"),), ((0, "right"),)]
        for i, s in min(opts, key=lambda o: sum(cost(i, s) for i, s in o)): out[s] = D[i]; self.tr[s] = (t, u[i])
        return out
