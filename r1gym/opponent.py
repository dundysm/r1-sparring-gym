"""Fight-motion opponent: by default a second (kinematic) Unitree R1 replaying KungFuAthleteBot clips retargeted
from the G1 by joint name (shared Unitree naming; R1 lacks waist_pitch / wrist_pitch / wrist_yaw, those are dropped,
angles are clipped to R1 joint ranges, root height scaled by R1/G1 standing pelvis height). Optional raw G1 replay, KungFuAthleteBot clips kinematically in front of the R1.

The G1 is driven by writing qpos (+ finite-difference qvel) every physics substep, so its limbs are an
infinite-mass "fist" for the R1: every contact is a real MuJoCo contact force on a real R1 body, logged per body.
Clips: KungFuAthleteBot (MIT code / Apache-2.0 data, LuluCao/KungfuAthleteBot on HF), G1 29-DoF, 50 fps."""
from __future__ import annotations
import numpy as np, mujoco
from pathlib import Path
from . import ASSETS
from .model import opponent_spec

G1_JOINTS = ['left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw']
G1_PELVIS_Z, R1_PELVIS_Z = 0.793, 0.74

def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])

def _yaw(q):
    w, x, y, z = q; return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def _cuts(name="tracker_cuts"):
    import json
    p = ASSETS / "motions" / f"{name}.json"; return json.load(open(p))["cuts"] if p.exists() else {}

def list_clips():
    """Clips the opponent performs (those the cut list drops entirely are excluded)."""
    return sorted(p.stem for p in (ASSETS / "motions").glob("kfab_*.npz") if _cuts().get(p.stem) != "all")

class FightReplay:
    """root="inplace" (default): the clip's slow root drift and heading changes are removed (1 s low-pass), so the
    opponent stays `distance` m in front of the R1, facing it, while lunges, steps and twists shorter than ~1 s are kept.
    root="clip": the original placement (clip mean xy at `distance`, frame-0 heading); the clips drift up to 2.4 m, so
    the opponent is mostly out of reach. Either way the loop is cross-faded over `blend_s` so it never teleports."""
    def __init__(self, m: mujoco.MjModel, clip="kfab_200", distance=0.62, speed=1.0, prefix="opp_", kind=None, root="inplace", blend_s=0.3, cfg=None):
        z = np.load(ASSETS / "motions" / f"{clip}.npz"); q = z["qpos"].astype(float); self.fps = float(z["fps"]) if "fps" in z else 50.0
        self.speed = speed; self.clip = clip; self.raw = q.copy(); self.distance = distance; self.root = root
        self.kind = kind or ("g1" if _has(m, prefix + "waist_pitch_joint") else "r1")
        q = self._retarget(q) if self.kind == "r1" else q
        q, self.clean_report = clean_clip(q, self.fps)   # source glitches and flips out, before placement
        q = apply_cuts(q, self.fps, _cuts().get(clip, []))  # moves the R1 can't perform (gpu/README.md: tracker cuts)
        if (cfg or {}).get("opponent_style_cuts", True):   # idle stretches and posing (scripts/style_cuts.py); off = the stage-3 motion
            q = apply_cuts(q, self.fps, _cuts("style_cuts").get(clip, []), STYLE_MIN_KEEP_S)
        if self.kind == "r1": q = ground(q, self.fps)      # feet on the floor (retarget only scaled root height)
        q = self._canon(q)
        self.q = _seamless(q, int(round(blend_s * self.fps))); self.T = len(self.q)
        j = m.joint(prefix + "floating_base_joint")
        self.qadr = m.jnt_qposadr[j.id]; self.vadr = m.jnt_dofadr[j.id]; self.nq = q.shape[1]
        self.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(prefix)}
        sub = _sub(self.kind); hinges = [sub.joint(i).name for i in range(1, sub.njnt)]
        motors = self.kind == "r1" and _has_act(m, prefix + hinges[0].replace("_joint", ""))
        mode = (cfg or {}).get("opponent_mode", "puppet") if motors else "kinematic"; self.model = m
        if motors:   # actuators in the order of qpos[7:]
            self.act = np.array([m.actuator(prefix + h.replace("_joint", "")).id for h in hinges])
            self.pel = m.body(prefix + "pelvis").id; self.mass = float(m.body_subtreemass[self.pel])
            self.torso = m.body(prefix + "torso_link").id
        self.tracker = Tracker.get(cfg["opponent_policy"], self) if mode == "policy" else None
        self.puppet = motors and self.tracker is None

    # root assist for the puppet: PD toward the clip's root pose, force-limited. Horizontal force is capped near the
    # friction limit of a 29 kg robot (mu*m*g ~ 280 N), so the opponent cannot bulldoze the R1 with the assist.
    ROOT_KP, ROOT_KD, ROOT_FXY, ROOT_FZ = 1500.0, 300.0, 250.0, 600.0
    ROT_KP, ROT_KD, ROT_TAU = 400.0, 40.0, 150.0

    def place(self, m, d, t):
        """Put the opponent exactly on the clip at time t (kinematic set, e.g. at reset)."""
        q0, q1 = self.qpos_at(t), self.qpos_at(t + m.opt.timestep)
        d.qpos[self.qadr:self.qadr + self.nq] = q0
        v = np.zeros(self.nq - 1); mujoco.mj_differentiatePos(_sub(self.kind), v, m.opt.timestep, q0, q1)
        d.qvel[self.vadr:self.vadr + self.nq - 1] = v

    def _canon(self, q):
        q = q.copy(); distance = self.distance
        if self.root == "inplace":
            yaw = np.unwrap([_yaw(x) for x in q[:, 3:7]]); n = int(round(self.fps))
            dyaw = np.pi - _lowpass(yaw, n); xy = q[:, :2] - _lowpass(q[:, :2], n)    # keep fast motion, drop drift
            for t in range(len(q)):
                c, s = np.cos(dyaw[t]), np.sin(dyaw[t])
                q[t, :2] = np.array([[c, -s], [s, c]]) @ xy[t] + [distance, 0.0]
                q[t, 3:7] = _quat_mul([np.cos(dyaw[t] / 2), 0, 0, np.sin(dyaw[t] / 2)], q[t, 3:7])
            return q
        # canonicalise: start heading -> face the R1 (-x), mean xy -> `distance` in front of R1
        yaw0 = _yaw(q[0, 3:7]); target = np.pi; dy = target - yaw0
        c, s = np.cos(dy), np.sin(dy); R = np.array([[c, -s], [s, c]]); rq = np.array([np.cos(dy / 2), 0, 0, np.sin(dy / 2)])
        xy = (q[:, :2] - q[:, :2].mean(0)) @ R.T
        q[:, :2] = xy + [distance, 0.0]
        for t in range(len(q)): q[t, 3:7] = _quat_mul(rq, q[t, 3:7])
        return q

    def _retarget(self, q):
        """Keypoint IK retarget G1 -> R1 (cached). Root pose copied (height scaled); R1 joints solved by damped
        least squares so knees/ankles/torso/elbows/wrists match the G1 keypoints (pelvis-relative, height-scaled)."""
        cache = ASSETS / "motions" / f"r1_{self.clip}.npy"
        if cache.exists():
            out = np.load(cache)
            if out.shape[0] == len(q): return out
        out = retarget_g1_to_r1(self.raw); np.save(cache, out); return out

    def qpos_at(self, t):
        f = (t * self.fps * self.speed) % self.T; i = int(f) % self.T; a = f - int(f); q1 = self.q[(i + 1) % self.T]
        if np.dot(self.q[i, 3:7], q1[3:7]) < 0: q1 = q1.copy(); q1[3:7] *= -1
        q = (1 - a) * self.q[i] + a * q1; q[3:7] /= np.linalg.norm(q[3:7]); return q

    def apply(self, m, d, t, dt):
        """Call once per physics substep, after any other xfrc_applied writes (it sets the opponent pelvis wrench).
        Policy mode: nothing here; Tracker.act sets the motor targets once per control step."""
        if self.tracker is not None: return
        if not self.puppet: return self.place(m, d, t)
        q0, q1 = self.qpos_at(t), self.qpos_at(t + dt)
        d.ctrl[self.act] = q0[7:]
        v = np.zeros(self.nq - 1); mujoco.mj_differentiatePos(_sub(self.kind), v, dt, q0, q1)   # target vel (root: local ang)
        p, R = d.xpos[self.pel], d.xmat[self.pel].reshape(3, 3)
        vel = d.qvel[self.vadr:self.vadr + 3]; w = R @ d.qvel[self.vadr + 3:self.vadr + 6]
        F = self.ROOT_KP * (q0[:3] - p) + self.ROOT_KD * (v[:3] - vel); F[2] += self.mass * 9.81
        fxy = np.linalg.norm(F[:2]); F[:2] *= min(1.0, self.ROOT_FXY / (fxy + 1e-9)); F[2] = np.clip(F[2], 0.0, self.ROOT_FZ)
        err = np.zeros(3); mujoco.mju_subQuat(err, q0[3:7], d.xquat[self.pel])   # rotation to target, pelvis frame
        tau = self.ROT_KP * (R @ err) + self.ROT_KD * (R @ v[3:6] - w)
        tau *= min(1.0, self.ROT_TAU / (np.linalg.norm(tau) + 1e-9))
        d.xfrc_applied[self.pel, :3] = F; d.xfrc_applied[self.pel, 3:] = tau

# KungFuAthleteBot clips contain mocap glitches (root teleports up to 24 m/s, 60-84 rad/s spins in one frame) and aerial
# flips (kfab_204/205 go upside down). No robot can follow those, and they make the replay jerk. clean_clip repairs short
# glitches by interpolation and cuts flips/inversions and long glitches out, cross-fading the pieces together.
GLITCH_ROOT_SPEED, GLITCH_ANG_SPEED, GLITCH_JOINT_SPEED = 4.0, 15.0, 30.0   # m/s, rad/s, rad/s between consecutive frames
MAX_TILT_DEG, REPAIR_MAX_S, MIN_KEEP_S, JOIN_BLEND_S = 60.0, 0.3, 1.0, 0.3
STYLE_MIN_KEEP_S = 0.5   # style cuts keep shorter pieces: a 0.6 s strike between two pauses is what we want

def _sign_continuous(q):
    q = q.copy(); q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)   # source quaternions are 0.91-1.12 long
    for i in range(1, len(q)):
        if np.dot(q[i, 3:7], q[i - 1, 3:7]) < 0: q[i, 3:7] *= -1
    return q

def _runs(mask):
    """[start, end) index pairs of the True runs in a boolean array."""
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def clip_speeds(q, fps):
    q = _sign_continuous(q)
    root = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) * fps
    ang = 2 * np.arccos(np.clip(np.abs(np.sum(q[1:, 3:7] * q[:-1, 3:7], 1)), -1, 1)) * fps
    joint = np.abs(np.diff(q[:, 7:], axis=0)).max(1) * fps
    up = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)        # R[2,2] of the root: 1 upright, <0 upside down
    return root, ang, joint, up

def clean_clip(q, fps):
    q = _sign_continuous(q); root, ang, joint, up = clip_speeds(q, fps); n = len(q)
    jump = np.zeros(n, bool); jump[1:] |= (root > GLITCH_ROOT_SPEED) | (ang > GLITCH_ANG_SPEED) | (joint > GLITCH_JOINT_SPEED)
    jump = np.convolve(jump, np.ones(5), "same") > 0                                   # +-2 frames around each jump
    cut = up < np.cos(np.radians(MAX_TILT_DEG)); repaired = 0
    for a, b in _runs(jump):                                                             # short glitch: interpolate across
        if b - a <= REPAIR_MAX_S * fps and a > 0 and b < n and not cut[a - 1] and not cut[b]:
            w = ((np.arange(a, b) - (a - 1)) / (b - (a - 1)))[:, None]; q[a:b] = (1 - w) * q[a - 1] + w * q[b]
            q[a:b, 3:7] /= np.linalg.norm(q[a:b, 3:7], axis=1, keepdims=True); repaired += b - a
        else: cut[a:b] = True                                                            # long glitch: cut
    cut = np.convolve(cut, np.ones(int(0.3 * fps) * 2 + 1), "same") > 0                 # margin around cuts
    keep = [(a, b) for a, b in _runs(~cut) if b - a >= MIN_KEEP_S * fps]
    out = _join(q, keep, fps)
    return out, dict(frames_in=n, frames_out=len(out), repaired_frames=int(repaired), cut_frames=int(n - sum(b - a for a, b in keep)), pieces=len(keep))

def _join(q, keep, fps):
    """Concatenate the kept [a, b) pieces of q with a cross-fade. Each piece continues from where the previous one left
    off (xy) and is turned to the heading of the frame the cross-fade pairs it with, so a join never spins the root."""
    m = int(JOIN_BLEND_S * fps); out = q[keep[0][0]:keep[0][1]]
    for a, b in keep[1:]:
        seg = q[a:b].copy(); seg[:, :2] += out[-1, :2] - seg[0, :2]
        dy = _yaw(out[-m, 3:7]) - _yaw(seg[0, 3:7]); c, s_ = np.cos(dy), np.sin(dy)
        seg[:, :2] = (seg[:, :2] - seg[0, :2]) @ np.array([[c, -s_], [s_, c]]).T + seg[0, :2]
        rq = np.array([np.cos(dy / 2), 0, 0, np.sin(dy / 2)]); seg[:, 3:7] = np.array([_quat_mul(rq, x) for x in seg[:, 3:7]])
        out = _blend(out, seg, m)
    return out

def apply_cuts(q, fps, ranges, min_keep_s=None):
    """Remove [a, b) frame ranges (indices into the clip as it is at this stage) and join what is left."""
    if not ranges: return q
    cut = np.zeros(len(q), bool)
    for a, b in ranges: cut[max(0, a):min(len(q), b)] = True
    keep = [(a, b) for a, b in _runs(~cut) if b - a >= (MIN_KEEP_S if min_keep_s is None else min_keep_s) * fps]
    return _join(q, keep, fps)

def foot_lows(q):
    """Lowest point of the R1's foot capsules (both feet) and the xy of each ankle, per frame."""
    r = _sub("r1"); d = mujoco.MjData(r)
    feet = [g for g in range(r.ngeom) if "foot" in (r.geom(g).name or "") and r.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
    side = np.array([r.geom(g).name.startswith("left") for g in feet]); ank = [r.body("left_ankle_roll_link").id, r.body("right_ankle_roll_link").id]
    low, xy = np.zeros((len(q), 2)), np.zeros((len(q), 2, 2))
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(r, d)
        z = d.geom_xpos[feet, 2] - np.abs(r.geom_size[feet, 1] * d.geom_xmat[feet, 8]) - r.geom_size[feet, 0]
        low[t] = [z[side].min(), z[~side].min()]; xy[t] = d.xpos[ank, :2]
    return low, xy

def ground(q, fps, stance_speed=0.2):
    """Shift root height so the supporting foot sits on the floor. Stance = the lower foot moves < 0.2 m/s horizontally;
    the offset is taken on stance frames, interpolated across flight (jumps keep their height) and smoothed over 0.5 s."""
    low, xy = foot_lows(q); lower = low.argmin(1); sup = low.min(1)
    v = np.r_[0, np.linalg.norm(np.diff(xy[np.arange(len(q)), lower], axis=0), axis=1) * fps]
    stance = v < stance_speed
    if stance.sum() < 2: return q
    idx = np.arange(len(q)); dz = np.interp(idx, idx[stance], -sup[stance])
    q = q.copy(); q[:, 2] += _lowpass(dz, int(round(0.5 * fps))); return q

def _blend(a, b, n):
    """Cross-fade the last n frames of a into the first n frames of b."""
    w = (np.arange(n) / n)[:, None]; tail = a[-n:].copy()
    tail[:, 3:7] *= np.sign(np.sum(tail[:, 3:7] * b[:n, 3:7], axis=1, keepdims=True) + 1e-12)
    mid = (1 - w) * tail + w * b[:n]; mid[:, 3:7] /= np.linalg.norm(mid[:, 3:7], axis=1, keepdims=True)
    return np.concatenate([a[:-n], mid, b[n:]])

class Tracker:
    """Opponent driven by an R1 motion-tracking policy (gpu/r1_track_task.py): no root assist, only its own motors.
    The ONNX (mjlab export) bundles the reference motion: all clips, each twice, so frame start + (f mod frames) of a clip
    is always inside the trained motion. Obs: reference joint pos/vel, [steerable policies: the reference anchor (torso)
    position in the robot torso frame,] the reference anchor orientation in the robot torso frame (first two
    rotation-matrix columns), gyro, joint pos - default, joint vel, last action (129, or 132 when steerable)."""
    _cache = {}
    @classmethod
    def get(cls, path, replay):
        from pathlib import Path
        from . import ROOT
        path = str(path if Path(path).is_absolute() else ROOT.parent / path)
        if path not in cls._cache: cls._cache[path] = cls(path)
        t = cls._cache[path]; t.check(replay); return t
    def __init__(self, path):
        import json, onnx, onnxruntime as ort
        from onnx import numpy_helper
        from pathlib import Path
        g = onnx.load(path); meta = {p.key: p.value for p in g.metadata_props}
        init = {t.name.split(".")[0]: numpy_helper.to_array(t) for t in g.graph.initializer if t.name.split(".")[0] in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")}
        self.ref_q, self.ref_dq = init["joint_pos"], init["joint_vel"]; k = meta["body_names"].split(",").index(meta["anchor_body_name"])
        self.ref_anchor, self.ref_anchor_pos = init["body_quat_w"][:, k], init["body_pos_w"][:, k]   # motion exported at distance 0
        self.default = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
        from .model import torque_caps, load_cfg, CTRL_JOINTS
        assert meta["joint_names"].split(",") == list(CTRL_JOINTS)
        self.kp_meta = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
        self.scale = 0.25 * torque_caps({**load_cfg(), "torque_cap_policy": "mjlab"}) / self.kp_meta   # exact (metadata has 3 decimals)
        assert np.allclose(self.scale, [float(x) for x in meta["action_scale"].split(",")], atol=1e-3)
        assert meta["observation_names"] in ("command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions",
                                             "command,motion_anchor_pos_b,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions")
        self.steerable = "motion_anchor_pos_b" in meta["observation_names"]
        ix = json.load(open(Path(path).with_name("motion_index.json"))); self.index = ix["clips"]; self.blend = ix.get("blend_frames", 20)
        self.kp = np.array([float(x) for x in meta["joint_stiffness"].split(",")]); self.kd = np.array([float(x) for x in meta["joint_damping"].split(",")])
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def frame(self, clip, f):
        """Motion frame for clip frame f. Each clip is in the motion twice; the first blend_frames of the first copy are
        cross-faded with the previous clip, so those frames come from the second copy. Every frame used is pure."""
        c = self.index[clip]; f %= c["frames"]
        return c["start_frame"] + (c["frames"] + f if f <= self.blend else f)   # <=: frame `blend` has a blended neighbour
    def check(self, replay):
        """The replayed clip must be the motion the tracker was trained on (same retarget + cleaning)."""
        c = self.index[replay.clip]; assert c["frames"] == replay.T, (replay.clip, c["frames"], replay.T)
        m = replay.model
        assert np.allclose(m.actuator_gainprm[replay.act, 0], self.kp) and np.allclose(-m.actuator_biasprm[replay.act, 2], self.kd),             "opponent motors differ from the tracker's training contract"
        for f in (0, self.blend - 1, self.blend, replay.T - 1):
            assert np.abs(self.ref_q[self.frame(replay.clip, f)] - replay.q[f, 7:]).max() < 1e-3, (replay.clip, f)
    FACE_TAU = 0.5   # s; how quickly the heading correction follows the bearing to the target
    # Steering (steerable policies): the reference's spot (where the in-place clip is centred) follows the point `distance`
    # m from the target along the bout's starting axis, at most 0.3 m/s with a 0.5 s lag. The shipped tracker's last stage
    # (R1-Track-KungFu-Steer2) drifted the reference at up to 0.3 m/s (SteerMotionCommand), the same cap.
    STEER_TAU, STEER_MAX = 0.5, 0.3

    def act(self, m, d, replay, state, t, dt=0.02):
        """state: {"last": last action, "target": xy the opponent should face (or absent), "yaw": smoothed correction,
        "spot"/"axis": steering}. Facing: the tracker reads the reference orientation only relative to its own torso, so
        turning the whole reference about the vertical axis is the same as turning the world, which the policy saw in
        training (random reset yaw). Turning it toward the bearing to the target keeps the opponent facing the R1.
        Steering: a steerable policy also reads where its reference is, so moving the reference walks the opponent; it is
        kept in front of the R1 instead of wandering off to its side."""
        f = self.frame(replay.clip, int(round(t * replay.fps * replay.speed)))
        q, dq = d.qpos[replay.qadr + 7:replay.qadr + replay.nq], d.qvel[replay.vadr + 6:replay.vadr + replay.nq - 1]
        ref, yaw = self.ref_anchor[f], 0.0
        if state.get("target") is not None:
            to = np.asarray(state["target"]) - d.xpos[replay.pel][:2]; want = np.arctan2(to[1], to[0]) - np.pi   # reference faces -x
            y0 = state.setdefault("yaw", want); state["yaw"] = yaw = y0 + np.angle(np.exp(1j * (want - y0))) * min(1.0, dt / self.FACE_TAU)
            rz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]); ref = np.zeros(4); mujoco.mju_mulQuat(ref, rz, self.ref_anchor[f])
        inv, rel = np.zeros(4), np.zeros(4); mujoco.mju_negQuat(inv, d.xquat[replay.torso]); mujoco.mju_mulQuat(rel, inv, ref)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, rel); R = R.reshape(3, 3); pos = []
        if self.steerable:
            spot = state.setdefault("spot", np.array([replay.distance, 0.0]))   # where FightReplay placed the in-place clip
            if state.get("target") is not None:
                tgt = np.asarray(state["target"], float); axis = state.setdefault("axis", (spot - tgt) / (np.linalg.norm(spot - tgt) + 1e-9))
                v = (tgt + replay.distance * axis - spot) / self.STEER_TAU; sp = np.linalg.norm(v)
                state["spot"] = spot = spot + v * min(1.0, self.STEER_MAX / (sp + 1e-9)) * dt
            c, s_ = np.cos(yaw), np.sin(yaw); a = self.ref_anchor_pos[f]
            world = np.array([spot[0] + c * a[0] - s_ * a[1], spot[1] + s_ * a[0] + c * a[1], a[2]])
            Rt = d.xmat[replay.torso].reshape(3, 3); pos = [Rt.T @ (world - d.xpos[replay.torso])]
        obs = np.concatenate([self.ref_q[f], self.ref_dq[f], *pos, R[:, :2].reshape(-1), d.qvel[replay.vadr + 3:replay.vadr + 6],
                              q - self.default, dq, state["last"]]).astype(np.float32)
        a = self.s.run(["actions"], {"obs": obs[None], "time_step": np.array([[f]], np.float32)})[0][0]
        state["last"] = a.astype(np.float32); d.ctrl[replay.act] = self.default + self.scale * a

def _lowpass(x, n):
    """Centred moving average over n frames (edge-padded), along axis 0."""
    x = np.asarray(x, float); pad = [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1)
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=0); c = np.concatenate([np.zeros_like(c[:1]), c])
    return (c[n:] - c[:-n]) / n

def _seamless(q, b):
    """Loopable clip: drop the last b frames and blend them into the first b, so frame T-1 -> frame 0 is a normal step."""
    if b < 1 or len(q) <= 2 * b: return q
    L = len(q) - b; out = q[:L].copy(); w = (np.arange(b) / b)[:, None]; tail = q[L:L + b].copy()
    tail[:, 3:7] *= np.sign(np.sum(tail[:, 3:7] * q[:b, 3:7], axis=1, keepdims=True) + 1e-12)   # same quaternion hemisphere
    out[:b] = (1 - w) * tail + w * q[:b]
    out[:b, 3:7] /= np.linalg.norm(out[:b, 3:7], axis=1, keepdims=True)
    return out

def _has(m, name):
    try: m.joint(name); return True
    except KeyError: return False

def _has_act(m, name):
    try: m.actuator(name); return True
    except KeyError: return False

_SUB = {}
def _sub(kind="g1"):
    """An opponent-only model so mj_differentiatePos works on the opponent qpos slice."""
    if kind not in _SUB: _SUB[kind] = opponent_spec(kind).compile()
    return _SUB[kind]

KEYS = ["torso_link", "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
        "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link"]

def retarget_g1_to_r1(q, iters=30, damp=1e-2, prior=0.05, max_step=0.25):
    """Per frame: start from the G1's own joint angles (both robots use Unitree joint names), then damped least squares
    on the keypoints with a pull toward that prior. The prior picks one arm solution, so the IK cannot flip between
    equivalent shoulder solutions from frame to frame; `max_step` caps any joint at 0.25 rad per frame (12.5 rad/s)."""
    g, r = _sub("g1"), _sub("r1"); dg, dr = mujoco.MjData(g), mujoco.MjData(r)
    sc = R1_PELVIS_Z / G1_PELVIS_Z; out = np.zeros((len(q), r.nq)); qr = r.qpos0.copy()
    kb = [r.body(k).id for k in KEYS]; gb = [g.body(k).id for k in KEYS]; J = np.zeros((3, r.nv))
    lo, hi = r.jnt_range[1:, 0], r.jnt_range[1:, 1]
    g_adr = {g.joint(i).name: g.jnt_qposadr[i] for i in range(1, g.njnt)}
    r_names = [r.joint(i).name for i in range(1, r.njnt)]
    prev = None
    for t in range(len(q)):
        dg.qpos[:] = q[t]; mujoco.mj_kinematics(g, dg)
        p0 = dg.xpos[g.body("pelvis").id]; tgt = [p0 + sc * (dg.xpos[b] - p0) for b in gb]
        qp = np.clip([q[t, g_adr[n]] if n in g_adr else 0.0 for n in r_names], lo, hi)
        qr[:7] = q[t, :7]; qr[2] = q[t, 2] * sc; qr[7:] = qp
        for _ in range(iters):
            dr.qpos[:] = qr; mujoco.mj_kinematics(r, dr)
            mujoco.mj_comPos(r, dr)
            rows, errs = [], []
            for b, tp in zip(kb, tgt):
                mujoco.mj_jacBody(r, dr, J, None, b); rows.append(J[:, 6:].copy()); errs.append(tp - dr.xpos[b])
            A = np.vstack(rows); e = np.concatenate(errs); n = A.shape[1]
            dq = np.linalg.solve(A.T @ A + (damp + prior) * np.eye(n), A.T @ e + prior * (qp - qr[7:]))
            qr[7:] = np.clip(qr[7:] + dq, lo, hi)
        if prev is not None: qr[7:] = prev + np.clip(qr[7:] - prev, -max_step, max_step)
        out[t] = qr; prev = qr[7:].copy()
    return out
