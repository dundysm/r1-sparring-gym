"""Render twin (phase 3, M0.2): a render-only MjModel of the sparring scene, kept apart from the physics model.

Built from the same build_spec(opponent=True) tree, so qpos layout and body names match the physics model (checked on
construction), with:
- the R1's visual meshes swapped for quadric LODs (r1gym/assets/r1/render_lod, scripts/build_render_lod.py): the full
  meshes are ~450k faces per robot and rendering on this iGPU is triangle-bound (35-40 ms per frame at any size);
- every contact off (it never steps);
- head cameras on torso_link and opp_torso_link (configs/r1_camera.yaml via camera.py). The physics model gets no camera;
- a room (walls 3-8 m, optional ceiling, up to 6 props), a bank of 36 builtin-texture materials and 4 lights.
sync(qpos) copies a physics state in and runs kinematics only, so rendering can never feed back into physics.
Appearance changes per frame touch only geom_matid, geom_rgba, mat_*, light_* and the headlight; the room layout per
episode only geom_pos/geom_size. Nothing recompiles and no texture is uploaded again.

Geom groups (MjvOption.geomgroup): 0 room and floor, 1 R1, 2 opponent, 3 collision shapes (never drawn), 4 R1 head shell,
5 opponent head shell. Each robot's own head shell is hidden from its own camera (the lens sits at the visor)."""
from __future__ import annotations
import copy, numpy as np, mujoco
from pathlib import Path
from .. import ASSETS
from ..model import build_spec, load_cfg
from .camera import HeadCameraSpec, add_head_camera, own_head_geoms, project_pinhole, project_ftheta, area_down, CV_MJ

LOD_DIR = ASSETS / "r1" / "render_lod"
G_ROOM, G_R1, G_OPP, G_COLL, G_R1HEAD, G_OPPHEAD = range(6)
CAMS = {"head": "head_cam", "opp_head": "opp_head_cam"}   # twin camera names (camera.add_head_camera, prefix "" / "opp_")
SEEN = {"head": (G_ROOM, G_R1, G_OPP, G_OPPHEAD), "opp_head": (G_ROOM, G_R1, G_OPP, G_R1HEAD), None: (G_ROOM, G_R1, G_OPP, G_R1HEAD, G_OPPHEAD)}
# 11 keypoints per robot: (body, offset in the body frame [m]). Joints sit at the body origin; the others at the link
# mesh's bounding-box centre (head and chest on torso_link, the fist on the wrist link, the foot on the ankle-roll link).
KEYPOINTS = {"head": ("torso_link", (0.01, 0.0, 0.357)), "chest": ("torso_link", (0.0, 0.0, 0.16)), "pelvis": ("pelvis", (0.02, 0.0, -0.05)),
             "l_elbow": ("left_elbow_link", (0.0, 0.0, 0.0)), "r_elbow": ("right_elbow_link", (0.0, 0.0, 0.0)),
             "l_fist": ("left_wrist_roll_link", (0.08, 0.0, 0.0)), "r_fist": ("right_wrist_roll_link", (0.08, 0.0, 0.0)),
             "l_knee": ("left_knee_link", (0.0, 0.0, 0.0)), "r_knee": ("right_knee_link", (0.0, 0.0, 0.0)),
             "l_foot": ("left_ankle_roll_link", (0.04, 0.0, -0.03)), "r_foot": ("right_ankle_roll_link", (0.04, 0.0, -0.03))}
# the links whose pixels count as the keypoint itself (coverage: a keypoint behind another of the opponent's own links is self-occluded)
KP_LINKS = {"head": ("head",), "chest": ("torso_link",), "pelvis": ("pelvis", "waist_roll_link"), "elbow": ("elbow_link", "shoulder_yaw_link"),
            "fist": ("wrist_roll_link", "elbow_link"), "knee": ("knee_link", "hip_yaw_link"), "foot": ("ankle_roll_link", "ankle_pitch_link")}
NOMINAL_ROOM = dict(size=(6.0, 6.0), center=(0.0, 0.0), height=3.0, ceiling=False, props=())   # the M1 colour-key scene
BANK = 36
HIDDEN_Z = -50.0   # hidden room geoms are parked under the floor (robust for rgb, depth and seg alike)

def _q(R): q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q
def _qmul(a, b): q = np.zeros(4); mujoco.mju_mulQuat(q, a, b); return q

class RenderTwin:
    def __init__(self, cfg=None, physics: mujoco.MjModel | None = None, spec: HeadCameraSpec | None = None, lod=True, lod_dir=None, room=True, **cam):
        """cfg: the physics config (same tree; G1 opponents keep full meshes). physics: the physics model to check the
        layout against. spec / **cam: the head camera (configs/r1_camera.yaml; overrides such as head_pitch_deg=-20).
        lod: swap in render_lod meshes (False: Unitree's full meshes, for the LOD gate and benchmark). A passed spec is
        copied: set_head_pitch writes the twin's own pitch, never the caller's (or another twin's) spec."""
        cfg = cfg or load_cfg(); self.spec = copy.copy(spec) if spec else HeadCameraSpec(gym_cfg=cfg, **cam); s = build_spec(opponent=True, cfg=cfg)
        lod_dir = Path(lod_dir or LOD_DIR); self.lod = bool(lod)
        for me in s.meshes if lod else []:
            p = lod_dir / (Path(me.file).stem + ".stl")
            if p.exists(): me.file = str(p)
        s.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # it never steps; contacts off anyway
        for g in s.geoms:
            g.contype = g.conaffinity = 0
            if g.group < 3: g.group = G_ROOM if g.parent.name == "world" else G_OPP if g.parent.name.startswith("opp_") else G_R1
        add_head_camera(s, self.spec, ""); add_head_camera(s, self.spec, "opp_")
        for n, p in (("fill", [-1.5, 2.0, 2.5]), ("rim", [2.5, -1.5, 2.2])):   # 4 lights with build_spec's two; off in the nominal scene
            s.worldbody.add_light(name=n, pos=p, dir=-np.array(p) / np.linalg.norm(p), diffuse=[0.3, 0.3, 0.3], castshadow=False, active=False)
        self._bank(s); self.room_geoms = self._room(s) if room else {}
        self.m = m = s.compile(); self.d = mujoco.MjData(m)
        m.vis.map.znear = 0.01 / m.stat.extent; m.vis.map.zfar = 40.0 / m.stat.extent   # near plane 1 cm: guard forearms are ~15 cm away
        for pre, grp in (("", G_R1HEAD), ("opp_", G_OPPHEAD)): m.geom_group[own_head_geoms(m, self.spec, pre)] = grp
        self.cams = {k: m.camera(v).id for k, v in CAMS.items()}
        self.heads = {pre: [(g, m.geom_pos[g].copy(), m.geom_quat[g].copy()) for g in own_head_geoms(m, self.spec, pre)] for pre in ("", "opp_")}
        bname = [m.body(b).name for b in range(m.nbody)]
        self.owner = np.array([0 if m.geom_bodyid[g] == 0 else 2 if bname[m.geom_bodyid[g]].startswith("opp_") else 1 for g in range(m.ngeom)])
        self.bank = np.array([m.material(f"bank{i}").id for i in range(BANK)]); self.room_ids = {k: m.geom(n).id for k, n in self.room_geoms.items()}
        self._nom = {k: getattr(m, k).copy() for k in ("geom_matid", "geom_rgba", "geom_pos", "geom_size", "mat_rgba", "mat_texid", "mat_texrepeat",
                     "mat_specular", "mat_shininess", "light_active", "light_diffuse", "light_ambient", "light_specular", "light_pos", "light_dir", "light_castshadow")}
        self._hl = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        self._opt = mujoco.MjvOption(); self._rn = {}; self.pitch = None
        if physics is not None: self.check_layout(physics)
        if room: self.set_room(**NOMINAL_ROOM)
        self.set_head_pitch(self.spec.pitch_deg)

    # ---- state
    def check_layout(self, m):
        """The twin must read a physics qpos verbatim: same joints, in the same order, at the same qpos addresses."""
        t = self.m; jn = lambda x: [x.joint(j).name for j in range(x.njnt)]; bn = lambda x: [x.body(b).name for b in range(x.nbody)]
        assert t.nq == m.nq and t.nv == m.nv and jn(t) == jn(m) and np.array_equal(t.jnt_qposadr, m.jnt_qposadr) and bn(t) == bn(m), \
            "render twin and physics model differ in qpos layout"

    def sync(self, qpos):
        """Copy a physics qpos in and run kinematics (+ cameras and lights: mj_kinematics alone leaves cam_xpos stale)."""
        self.d.qpos[:] = qpos; self._fwd()

    def _fwd(self): mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)

    def set_head_pitch(self, deg):
        """Pitch both heads by deg (optical-axis elevation, - down) about the assumed pivot: both cameras and both head
        shells move (the shells only matter to the other robot's camera). No recompile."""
        self.spec.pitch_deg = float(deg); p, R = self.spec.mount_torso; q = _q(R @ CV_MJ)
        for c in self.cams.values(): self.m.cam_pos[c] = p; self.m.cam_quat[c] = q
        P = _ry(-np.radians(deg)); qp = _q(P); piv = self.spec.pivot
        for heads in self.heads.values():
            for g, p0, q0 in heads: self.m.geom_pos[g] = piv + P @ (p0 - piv); self.m.geom_quat[g] = _qmul(qp, q0)
        self.pitch = float(deg); self._fwd()

    # ---- room, materials and lights
    def _bank(self, s):
        """36 materials on MuJoCo builtin textures (checker, gradient, flat + random dots) or none (flat colour), fixed seed."""
        rng = np.random.default_rng(0); B = mujoco.mjtBuiltin
        for i in range(BANK):
            kind = i % 4; c1, c2 = rng.uniform(0.05, 0.95, 3), rng.uniform(0.05, 0.95, 3); mat = s.add_material(name=f"bank{i}", rgba=[1, 1, 1, 1],
                texrepeat=[float(rng.choice([1, 2, 4, 8]))] * 2, texuniform=True, specular=float(rng.uniform(0, 0.5)), shininess=float(rng.uniform(0, 0.8)))
            if kind == 3: mat.rgba = list(c1) + [1.0]; continue
            t = s.add_texture(name=f"bank_tex{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, width=64, height=64, rgb1=c1, rgb2=c2,
                              builtin=[B.mjBUILTIN_CHECKER, B.mjBUILTIN_GRADIENT, B.mjBUILTIN_FLAT][kind])
            if kind == 2: t.mark = mujoco.mjtMark.mjMARK_RANDOM; t.random = float(rng.uniform(0.02, 0.3)); t.markrgb = c2; t.rgb2 = c1
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name

    def _room(self, s):
        G = mujoco.mjtGeom; out = {}; w = s.worldbody
        for k, (n, t) in enumerate([("wall0", G.mjGEOM_BOX), ("wall1", G.mjGEOM_BOX), ("wall2", G.mjGEOM_BOX), ("wall3", G.mjGEOM_BOX), ("ceiling", G.mjGEOM_BOX),
                                    ("prop0", G.mjGEOM_BOX), ("prop1", G.mjGEOM_BOX), ("prop2", G.mjGEOM_CYLINDER), ("prop3", G.mjGEOM_CYLINDER),
                                    ("prop4", G.mjGEOM_SPHERE), ("prop5", G.mjGEOM_CAPSULE)]):
            w.add_geom(name="room_" + n, type=t, size=[0.1, 0.1, 0.1], pos=[0, 0, HIDDEN_Z - k], rgba=[0.62, 0.62, 0.6, 1], group=G_ROOM, contype=0, conaffinity=0)
            out[n] = "room_" + n
        return out

    def set_room(self, size=(6.0, 6.0), center=(0.0, 0.0), height=3.0, ceiling=False, props=()):
        """Room layout (per episode): walls of a size[0] x size[1] m room, optional ceiling, props = [(x, y, half-size), ...]
        (up to 6, in the order box, box, cylinder, cylinder, sphere, capsule). geom_pos/geom_size only."""
        m, ids = self.m, self.room_ids; (lx, ly), (cx, cy), h = np.array(size, float) / 2, center, height; t = 0.05
        walls = [([cx + lx + t, cy, h / 2], [t, ly + 2 * t, h / 2]), ([cx - lx - t, cy, h / 2], [t, ly + 2 * t, h / 2]),
                 ([cx, cy + ly + t, h / 2], [lx, t, h / 2]), ([cx, cy - ly - t, h / 2], [lx, t, h / 2])]
        for k, (p, sz) in enumerate(walls): m.geom_pos[ids[f"wall{k}"]] = p; m.geom_size[ids[f"wall{k}"]] = sz
        m.geom_pos[ids["ceiling"]] = [cx, cy, h + t] if ceiling else [0, 0, HIDDEN_Z]; m.geom_size[ids["ceiling"]] = [lx + 2 * t, ly + 2 * t, t]
        for k in range(6):   # cubes, upright cylinders, a sphere and an upright capsule (2r half-height), standing on the floor
            g = ids[f"prop{k}"]
            if k < len(props): x, y, r = props[k]; m.geom_size[g] = [r, r, r]; m.geom_pos[g] = [x, y, 2 * r if k == 5 else r]
            else: m.geom_pos[g] = [0, 0, HIDDEN_Z - k]
        self._fwd()

    def randomize_room(self, rng):
        """Walls 3-8 m (the room up to 1.5 m off-centre, every wall >= 1.2 m from the bout's centre), ceiling with p 0.5,
        0-6 props inside the walls and outside a 1.2 m circle around the bout (where the robots are)."""
        L = rng.uniform(3.0, 8.0, 2); c = rng.uniform(-1, 1, 2) * np.minimum(1.5, L / 2 - 1.2); props = []
        for k in range(int(rng.integers(0, 7))):
            r = rng.uniform(0.12, 0.45)
            for _ in range(50):
                xy = c + rng.uniform(-1, 1, 2) * (L / 2 - r)
                if np.linalg.norm(xy) > 1.2 + r: props.append((xy[0], xy[1], r)); break
        self.set_room(size=L, center=c, height=rng.uniform(2.5, 3.5), ceiling=bool(rng.random() < 0.5), props=props)

    def randomize_appearance(self, rng, robots=False):
        """Per-frame appearance: bank materials (and tints) on the room and floor, 1-4 lights with random colour, position and
        shadow flag, headlight level; robots=True also recolours both robots (e.g. look-alike colours). No recompile."""
        m, n = self.m, self._nom; self.nominal_appearance()
        room = [g for g in range(m.ngeom) if self.owner[g] == 0]
        m.geom_matid[room] = rng.choice(self.bank, len(room)); m.geom_rgba[room] = [0.5, 0.5, 0.5, 1]   # default rgba: the material's colour shows
        m.mat_rgba[self.bank, :3] = np.clip(n["mat_rgba"][self.bank, :3] * rng.uniform(0.6, 1.4, (BANK, 3)), 0, 1)
        nl = m.nlight; on = rng.permutation(nl)[:int(rng.integers(1, nl + 1))]; m.light_active[:] = 0; m.light_active[on] = 1
        m.light_diffuse[:] = rng.uniform(0.15, 0.8, (nl, 1)) * rng.uniform(0.8, 1.0, (nl, 3)); m.light_ambient[:] = rng.uniform(0.0, 0.15, (nl, 1))
        m.light_pos[:] = np.c_[rng.uniform(-2.5, 2.5, (nl, 2)), rng.uniform(1.8, 3.2, nl)]
        dn = -m.light_pos + np.c_[rng.uniform(-0.5, 0.5, (nl, 2)), rng.uniform(0.3, 0.9, nl)]; m.light_dir[:] = dn / np.linalg.norm(dn, axis=1, keepdims=True)
        m.light_castshadow[:] = rng.random(nl) < 0.5
        m.vis.headlight.diffuse[:] = rng.uniform(0.0, 0.5); m.vis.headlight.ambient[:] = rng.uniform(0.0, 0.3)
        if robots:
            for k in (1, 2):
                gs = (self.owner == k) & (m.geom_group != G_COLL)
                m.geom_rgba[gs, :3] = rng.uniform(0.1, 0.95, 3); m.geom_matid[gs] = rng.choice(self.bank) if rng.random() < 0.3 else -1
        self._fwd()

    def nominal_appearance(self):
        """The physics scene's colours (silver R1, orange opponent, grey floor) and lights (build_spec's two)."""
        m = self.m
        for k in ("geom_matid", "geom_rgba", "mat_rgba", "mat_texid", "mat_texrepeat", "mat_specular", "mat_shininess",
                  "light_active", "light_diffuse", "light_ambient", "light_specular", "light_pos", "light_dir", "light_castshadow"): getattr(m, k)[:] = self._nom[k]
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = self._hl; self._fwd()

    # ---- rendering
    def renderer(self, w, h):
        if (w, h) not in self._rn: self._rn[(w, h)] = mujoco.Renderer(self.m, h, w)
        return self._rn[(w, h)]

    def gl(self):
        """GL vendor / renderer / version of the render-size context, for run_meta(gl=...) (r1gym.perception)."""
        return gl_info(self.renderer(self.spec.net_wh[0] * self.spec.scale, self.spec.net_wh[1] * self.spec.scale))

    def render(self, cam="head", kind="rgb", size=None, shadows=False, groups=None):
        """One image from the current state. cam: "head" (the R1's), "opp_head", or an MjvCamera (external view); kind: rgb
        (uint8), depth (float32, metres along the optical axis) or seg ((h, w, 2) int32 object id, type). size: (w, h),
        default the render size (network size x render_scale). groups: geom groups drawn (default: all but the camera's own head)."""
        w, h = size or (self.spec.net_wh[0] * self.spec.scale, self.spec.net_wh[1] * self.spec.scale); rn = self.renderer(w, h)
        rn.disable_depth_rendering(); rn.disable_segmentation_rendering()
        if kind == "depth": rn.enable_depth_rendering()
        elif kind == "seg": rn.enable_segmentation_rendering()
        o = self._opt; o.geomgroup[:] = 0; o.geomgroup[list(SEEN[cam if isinstance(cam, str) else None] if groups is None else groups)] = 1
        rn.update_scene(self.d, camera=self.cams[cam] if isinstance(cam, str) else cam, scene_option=o)
        f = rn.scene.flags; f[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(shadows); f[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0; f[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        return rn.render()

    def frame(self, cam="head", shadows=False):
        """The network image: rendered at render_scale x the network size, area-downsampled (uint8, (120, 160, 3))."""
        return np.round(area_down(self.render(cam, "rgb", shadows=shadows), self.spec.scale)).astype(np.uint8)

    def masks(self, seg):
        """(R1, opponent) pixel masks from a segmentation render."""
        ok = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] >= 0); o = np.where(ok, self.owner[np.maximum(seg[..., 0], 0)], 0)
        return o == 1, o == 2

    def ext_camera(self, azimuth, elevation, distance=2.6):
        """External free camera looking at the midpoint of the two pelvises (LOD gate, debugging)."""
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.azimuth, c.elevation, c.distance = azimuth, elevation, distance
        c.lookat[:] = 0.5 * (self.d.xpos[self.m.body("pelvis").id] + self.d.xpos[self.m.body("opp_pelvis").id]); c.lookat[2] = 0.6; return c

    # ---- geometry
    def keypoints(self, prefix="opp_"):
        """KEYPOINTS of one robot in world coordinates, (11, 3). The head keypoint pitches with the head."""
        d, P = self.d, _ry(-np.radians(self.pitch)); out = []
        for k, (b, off) in KEYPOINTS.items():
            off = np.array(off, float)
            if k == "head": off = self.spec.pivot + P @ (off - self.spec.pivot)
            i = self.m.body(prefix + b).id; out.append(d.xpos[i] + d.xmat[i].reshape(3, 3) @ off)
        return np.array(out)

    def keypoint_geoms(self, prefix="opp_"):
        """Per keypoint, the geom ids of its own links (KP_LINKS): a keypoint's pixel showing one of them is the keypoint itself."""
        m, grp = self.m, G_OPP if prefix else G_R1; out = {}
        for k in KEYPOINTS:
            part = k.split("_")[-1]; side = {"l": "left_", "r": "right_"}.get(k.split("_")[0], "") if "_" in k else ""
            if part == "head": out[k] = np.flatnonzero(m.geom_group == (G_OPPHEAD if prefix else G_R1HEAD)); continue
            b = [m.body(prefix + side + l).id for l in KP_LINKS[part]]; out[k] = np.flatnonzero(np.isin(m.geom_bodyid, b) & (m.geom_group == grp))
        return out

    def to_camera(self, P, cam="head"):
        """World points (N, 3) -> camera axes (OpenCV: x right, y down, z along the optical axis)."""
        c = self.cams[cam]; R = self.d.cam_xmat[c].reshape(3, 3); return (np.asarray(P, float) - self.d.cam_xpos[c]) @ R @ CV_MJ

    def project(self, P, cam="head", scale=1):
        """World points -> (uv pixels at scale x the network size, depth along the axis [m], in the pinhole view)."""
        Pc = self.to_camera(P, cam); K = self.spec.K(scale); w, h = self.spec.net_wh[0] * scale, self.spec.net_wh[1] * scale
        with np.errstate(divide="ignore", invalid="ignore"): uv = project_pinhole(Pc, K)
        inside = (Pc[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        return uv, Pc[:, 2], inside

    def in_fisheye(self, P, cam="head"):
        """Inside the physical lens image (146 x 124 deg f-theta, 1280 x 1080), FOV only."""
        uv = project_ftheta(self.to_camera(P, cam), self.spec.f, self.spec.c); W, H = self.spec.lens_wh
        return (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    def close(self):
        """Free the GL contexts, each made current first: mujoco.Renderer.close() frees its GL objects in whatever context is
        current, which corrupts another live renderer's images (mujoco 3.17, GLFW; caught by render_bench's determinism check)."""
        for r in self._rn.values():
            if r._gl_context: r._gl_context.make_current()
            r.close()
        self._rn = {}

    def __del__(self):   # a twin dropped without close() must not free its GL objects in another twin's context either
        try: self.close()
        except Exception: pass

def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def gl_info(rn: mujoco.Renderer):
    """GL vendor / renderer / version of a renderer's context (rasterisation can differ across GPUs and drivers)."""
    try:
        rn._gl_context.make_current(); from OpenGL import GL
        return {k: GL.glGetString(e).decode() for k, e in (("gl_vendor", GL.GL_VENDOR), ("gl_renderer", GL.GL_RENDERER), ("gl_version", GL.GL_VERSION))}
    except Exception as ex: return dict(gl_renderer=f"unavailable ({type(ex).__name__}: {ex})")

def gl_available():
    """True if an offscreen GL context can be made here (CI runners have none: render tests skip)."""
    try:
        s = mujoco.MjSpec(); s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0]); r = mujoco.Renderer(s.compile(), 8, 8); r.render(); r.close(); return True
    except Exception: return False
