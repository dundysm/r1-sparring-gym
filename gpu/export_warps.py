"""M2 warp library (docs/AIMED_OPPONENT.md, M2 run 1): whole-body aimed references built from TRAIN-clip hand strikes only,
appended to the tracker's motion as warped copies of the train clips -> gpu/motions/r1_kungfu_aim.csv, _index.json, _apex.json.

The motion is export_motion.py's (every clip twice, cross-faded; asserted equal to gpu/motions/r1_kungfu.csv), followed by
`copies` warped copies of each train clip (once each, cross-faded the same way). In a copy, a random set of the clip's hand
strikes whose windows don't overlap is aimed, each at its own target:
  target   head or torso (50/50) of an R1 standing at home, its pelvis D ~ U(0.42, 0.62) m in front of the spot (the in-place
           clip's centre, toward -x), +-0.10 m sideways, turned +-20 deg; the point is aim.py's (3 cm behind the face / chest
           front), moved +-0.05 m in height. A strike whose apex already goes past the target's depth keeps its depth and is
           aimed in height and sideways only (the strike lands earlier; it is never pulled back). The apex moves at most 0.25 m
           toward the target (the largest warp the tolerance gates test; 0.20 / 0.15 / 0.10 m on the 2nd-4th draw): a strike far off it
           is bent toward it, not onto it.
  IK       at the apex frame, damped least squares on the kinematic R1 (r1gym/opponent.py _sub): the striking arm (5 joints,
           +-0.8 rad, aim.py's limits), waist yaw (+-0.6) and roll (+-0.2), stance height (pelvis -3..+8 cm, legs re-solved so
           both feet keep their reference pose) and a spot lunge (the whole reference moved up to 0.25 m toward the R1 and
           0.10 m sideways, its feet with it, as the steering moves it), each pulled toward no change (the lunge hardest).
  timing   arm, waist and stance blend in and out with ArmWarp's weight (r1gym/aim.py: onset - 0.15 s to the apex, held
           0.05 s, out by the end + 0.1 s); the lunge ramps in no faster than 0.6 m/s and back out no faster than 0.4 m/s.
           Every frame in the window re-solves the legs (and, softly, the stance height, where a straight leg can't take it)
           so the feet stay where the reference (plus the lunge) has them.
  rejected (a new target is drawn, up to 4 times; else that strike stays unwarped in the copy) when the palm misses the
           target by more than 3 cm at the apex, a foot leaves its reference by more than 1 cm on any frame, a joint moves
           faster than both its p99 over the train clips and its own fastest in the unwarped window, or the centre of mass
           leaves the support polygon of the feet on the floor by 2 cm more than the unwarped reference does.
Run 2 (R1-Track-KungFu-Aim2): --head_share 1 --max_warp_m 0.35 --copies 48 --name r1_kungfu_aim2: every strike aimed at the head,
apexes moved up to 0.35 m (0.30 / 0.25 / 0.20 m on redraws).
_apex.json also marks, per frame, each train-clip hand strike's lunge window (0.3 s before onset to apex + 0.05 s), which
R1-Track-KungFu-Aim2 times its steering lunges to.
_apex.json lists, per motion frame, the striking hand at every train-clip strike's apex (+-2 frames; warped or not, in the
original part too): gpu/r1_track_task.py's strike_apex reward reads it. Val and test strikes are never warped or listed."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from r1gym.model import build_spec, CTRL_JOINTS, HOME
from r1gym.opponent import FightReplay, list_clips, _sub
from r1gym.env import TRAIN_CLIPS
from r1gym.aim import AIM, ARM, PALM, smoothstep
from r1gym import strikes as S
from export_motion import blend

W = dict(max_warp_m=0.25, D=(0.42, 0.62), lat=0.10, dz=0.05, yaw_deg=20.0, max_dq=0.8, waist_yaw=0.6, waist_roll=0.2, root_dz=(-0.03, 0.08),
         lunge_x=0.25, lunge_y=0.10, lunge_in_mps=0.6, lunge_out_mps=0.4, res_m=0.03, foot_m=0.01, lunge_lead_s=0.3, com_m=0.02, tries=4, apex_pad=2,
         prior=dict(arm=1e-3, waist=4e-3, root_z=8e-3, lunge=2e-2, legs=1e-3), iters=40)
WAIST = [CTRL_JOINTS.index("waist_roll_joint"), CTRL_JOINTS.index("waist_yaw_joint")]
LEGS = list(range(12)); FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

def hull_out(P, c):
    """Signed distance of point c outside the 2-D convex hull of P (monotone chain; max over edges of the outward offset)."""
    P = sorted(map(tuple, np.round(P, 9))); cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    H = np.array(lo[:-1] + up[:-1]); E = np.roll(H, -1, 0) - H; n = np.stack([E[:, 1], -E[:, 0]], 1) / np.linalg.norm(E, axis=1, keepdims=True)   # outward for a ccw hull
    return float(np.max(np.sum(n * (c - H), axis=1)))

class Kin:
    """Kinematic R1 (qpos: root xyz, quat wxyz, 24 joints in CTRL_JOINTS order, as FightReplay.q)."""
    def __init__(self):
        self.m = _sub("r1"); self.d = mujoco.MjData(self.m); m = self.m
        self.palm = {k: m.site(v).id for k, v in PALM.items()}; self.feet = [m.body(b).id for b in FEET]
        self.sole = {f: [g for g in range(m.ngeom) if m.geom_bodyid[g] == f and "foot" in m.geom(g).name] for f in self.feet}
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]; mid, half = (lo + hi) / 2, (hi - lo) / 2; fr = AIM["arm"]["range_frac"]
        self.lo, self.hi = mid - fr * half, mid + fr * half; self.Jp, self.Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    def fk(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d); return self.d
    def feet_pose(self, q):
        d = self.fk(q); return [(d.xpos[f].copy(), d.xquat[f].copy()) for f in self.feet]
    def support_out(self, q, qfeet, floor, sh=(0.0, 0.0)):
        """How far (m) q's centre of mass lies outside the convex hull of qfeet's foot geoms on the floor, moved by sh (<= 0:
        inside). The stance comes from the reference frame (qfeet), whose feet the warped frame keeps."""
        d = self.fk(qfeet); pts = []
        for f, gs in self.sole.items():
            if min(d.geom_xpos[g][2] - self.m.geom_size[g][0] for g in gs) < floor + 0.02: pts += [d.geom_xpos[g][:2] + np.asarray(sh) for g in gs]
        if len(pts) < 3: return 0.0   # no stance (a hop): nothing to check
        return hull_out(np.array(pts), self.fk(q).subtree_com[1][:2].copy())

def solve(k, q0, tasks, var, prior, bounds, iters, start=None):
    """Damped least squares on qpos-space variables `var` (dof indices: 0-2 root translation, 6+ joints). tasks: list of
    (kind, id, target, weight) with kind 'site' / 'body' (position) or 'rot' (body orientation, target quat)."""
    m = k.m; idx = np.array(var); qadr = np.array([v if v < 3 else v + 1 for v in var])   # dof -> qpos (free joint: 3 lin, 4 quat)
    lam = np.array([prior[i] for i in range(len(var))]); q = q0.copy() if start is None else start.copy()
    lo, hi = np.minimum(bounds[0], q0[qadr]), np.maximum(bounds[1], q0[qadr])   # a reference already past a limit may stay there
    for _ in range(iters):
        d = k.fk(q); A, e = [], []
        for kind, i, tgt, w in tasks:
            if kind == "site": mujoco.mj_jacSite(m, d, k.Jp, None, i); A.append(w * k.Jp[:, idx]); e.append(w * (tgt - d.site_xpos[i]))
            elif kind == "body": mujoco.mj_jacBody(m, d, k.Jp, None, i); A.append(w * k.Jp[:, idx]); e.append(w * (tgt - d.xpos[i]))
            else:
                mujoco.mj_jacBody(m, d, None, k.Jr, i); err = np.zeros(3); mujoco.mju_subQuat(err, tgt, d.xquat[i])
                A.append(w * k.Jr[:, idx]); e.append(w * d.xmat[i].reshape(3, 3) @ err)   # subQuat is in the body frame, the Jacobian in the world
        A, e = np.concatenate(A), np.concatenate(e); x = q[qadr]; x0 = q0[qadr]
        dx = np.linalg.solve(A.T @ A + np.diag(lam + 1e-6), A.T @ e - lam * (x - x0)); dx *= min(1.0, 0.1 / max(np.abs(dx).max(), 1e-12))   # trust region: 0.1 rad / m per step
        q[qadr] = np.clip(x + dx, lo, hi)
    return q

def weight(t, t_on, t_apex, t_end):
    """ArmWarp's blend (r1gym/aim.py Aim.weight) in clip seconds."""
    a = AIM["arm"]; t0, th = t_on - a["lead_s"], t_apex + a["hold_s"]; te = max(t_end, t_apex + a["min_end_s"]) + a["tail_s"]
    if t <= t0 or t >= te: return 0.0
    return smoothstep((t - t0) / max(t_apex - t0, 1e-6)) if t < t_apex else 1.0 if t <= th else smoothstep((te - t) / max(te - th, 1e-6))

def lunge_times(s, L):
    """(start, apex, hold end, end) of the lunge ramp: no slower in than ArmWarp, peak speed of a smoothstep is 1.5 L / T."""
    a = AIM["arm"]; tin = max(s.t_apex - (s.t_on - a["lead_s"]), 1.5 * L / W["lunge_in_mps"])
    tout = max(max(s.t_end, s.t_apex + a["min_end_s"]) + a["tail_s"] - (s.t_apex + a["hold_s"]), 1.5 * L / W["lunge_out_mps"])
    return s.t_apex - tin, s.t_apex, s.t_apex + a["hold_s"], s.t_apex + a["hold_s"] + tout

def ramp(t, t0, t1, t2, t3):
    if t <= t0 or t >= t3: return 0.0
    return smoothstep((t - t0) / (t1 - t0)) if t < t1 else 1.0 if t <= t2 else smoothstep((t3 - t) / (t3 - t2))

def r1_target(rng, zone):
    """World target and its draw: the R1 at home, pelvis D in front of the spot (toward -x), facing the opponent (+x)."""
    r = _sub("r1"); d = mujoco.MjData(r); d.qpos[:3] = [0, 0, 0.74]; d.qpos[3] = 1; d.qpos[7:] = HOME; mujoco.mj_kinematics(r, d)
    t = r.body("torso_link").id; p_t = d.xpos[t] + d.xmat[t].reshape(3, 3) @ np.array(AIM["targets"][zone])   # pelvis frame
    D, lat, dz, yaw = rng.uniform(*W["D"]), rng.uniform(-W["lat"], W["lat"]), rng.uniform(-W["dz"], W["dz"]), np.radians(rng.uniform(-W["yaw_deg"], W["yaw_deg"]))
    c, s_ = np.cos(yaw), np.sin(yaw); p = np.array([-D, lat, 0.0]) + np.array([c * p_t[0] - s_ * p_t[1], s_ * p_t[0] + c * p_t[1], p_t[2] + dz])
    return p, dict(zone=zone, D=round(D, 3), lat=round(lat, 3), dz=round(dz, 3), yaw_deg=round(float(np.degrees(yaw)), 1))

def warp_strike(k, q, s, target, vmax, cap=None):
    """Warp clip motion q (T x 31, in place at distance 0) for strike s toward world target. Returns (q', info) or (None, why)."""
    fps = s.fps; side = s.limb; arm = ARM[side]; fa = s.apex; qa = q[fa].copy(); target = np.array(target, float)
    p0 = k.fk(qa).site_xpos[k.palm[side]].copy(); target[0] = min(target[0], p0[0])   # a strike already past the target's depth keeps its depth
    v = target - p0; n = float(np.linalg.norm(v)); target = p0 + v * min(1.0, (cap or W["max_warp_m"]) / max(n, 1e-9))   # at most max_warp_m toward it
    feet = k.feet_pose(qa); floor = min(k.fk(qa).geom_xpos[g][2] - k.m.geom_size[g][0] for gs in k.sole.values() for g in gs)
    var = [0, 1, 2] + [6 + j for j in arm + WAIST + LEGS]; P = W["prior"]
    prior = [P["lunge"]] * 2 + [P["root_z"]] + [P["arm"]] * 5 + [P["waist"]] * 2 + [P["legs"]] * 12
    lo = np.r_[qa[0] - W["lunge_x"], qa[1] - W["lunge_y"], qa[2] + W["root_dz"][0], np.maximum(k.lo[arm], qa[7 + np.array(arm)] - W["max_dq"]),
               qa[7 + WAIST[0]] - W["waist_roll"], qa[7 + WAIST[1]] - W["waist_yaw"], k.lo[LEGS]]
    hi = np.r_[qa[0], qa[1] + W["lunge_y"], qa[2] + W["root_dz"][1], np.minimum(k.hi[arm], qa[7 + np.array(arm)] + W["max_dq"]),
               qa[7 + WAIST[0]] + W["waist_roll"], qa[7 + WAIST[1]] + W["waist_yaw"], k.hi[LEGS]]
    q1 = qa.copy()
    for _ in range(3):   # the feet move with the lunge: re-target them from the current lunge
        sh = np.r_[q1[0] - qa[0], q1[1] - qa[1], 0.0]
        tasks = [("site", k.palm[side], target, 1.0)] + [("body", f, p + sh, 3.0) for f, (p, _) in zip(k.feet, feet)] + [("rot", f, r, 1.0) for f, (_, r) in zip(k.feet, feet)]
        q1 = solve(k, qa, tasks, var, prior, (lo, hi), W["iters"], start=q1)
    res = float(np.linalg.norm(target - k.fk(q1).site_xpos[k.palm[side]]))
    if res > W["res_m"]: return None, f"residual {res:.3f}"
    L = np.r_[q1[0] - qa[0], q1[1] - qa[1]]; dz = q1[2] - qa[2]; dj = q1[7:] - qa[7:]; dj[LEGS] = 0.0
    lt = lunge_times(s, float(np.linalg.norm(L))); out = q.copy(); T = len(q); f0 = int(np.floor(min(lt[0], s.t_on - AIM["arm"]["lead_s"]) * fps)) - 1
    f1 = int(np.ceil(max(lt[3], max(s.t_end, s.t_apex + AIM["arm"]["min_end_s"]) + AIM["arm"]["tail_s"]) * fps)) + 1
    if f0 < 0 or f1 >= T: return None, "window crosses the loop seam"
    worst_foot, worst_com = 0.0, -9.9; prev = None
    for f in range(f0, f1 + 1):
        t = f / fps; w = weight(t, s.t_on, s.t_apex, s.t_end); wl = ramp(t, *lt)
        if w == 0 and wl == 0: prev = None; continue
        qf = q[f].copy(); qf[0:2] += wl * L; qf[2] += w * dz; qf[7:] += w * dj; ft = k.feet_pose(q[f]); sh = np.r_[wl * L, 0.0]
        st = qf.copy()
        if prev is not None: st[2] = qf[2] + prev[0]; st[7:19] = q[f][7:19] + prev[1]   # warm start: the previous frame's leg correction (smooth in time)
        tasks = [("body", fb, p + sh, 1.0) for fb, (p, _) in zip(k.feet, ft)] + [("rot", fb, r, 0.3) for fb, (_, r) in zip(k.feet, ft)]
        qf = solve(k, qf, tasks, [2] + [6 + j for j in LEGS], [W["prior"]["root_z"]] + [W["prior"]["legs"]] * 12,   # stance height gives way where a leg can't
                   (np.r_[q[f][2] + W["root_dz"][0], k.lo[LEGS]], np.r_[q[f][2] + W["root_dz"][1], k.hi[LEGS]]), 30, start=st)
        prev = (qf[2] - (q[f][2] + w * dz), qf[7:19] - q[f][7:19])
        worst_foot = max(worst_foot, max(float(np.linalg.norm(p + sh - k.fk(qf).xpos[fb])) for fb, (p, _) in zip(k.feet, ft)))
        worst_com = max(worst_com, k.support_out(qf, q[f], floor, wl * L) - k.support_out(q[f], q[f], floor)); out[f] = qf
    if worst_foot > W["foot_m"]: return None, f"foot {worst_foot:.3f}"
    if worst_com > W["com_m"]: return None, f"com {worst_com:.3f}"
    v = np.abs(np.diff(out[f0:f1 + 2, 7:], axis=0)) * fps; v0 = np.abs(np.diff(q[f0:f1 + 2, 7:], axis=0)) * fps
    over = v.max(0) > np.maximum(vmax, v0.max(0)) + 0.05   # 0.05 rad/s: the leg re-solve's numerical noise
    if over.any(): return None, "joint speed " + ",".join(CTRL_JOINTS[j] for j in np.where(over)[0])
    return out, dict(warp_m=round(float(np.linalg.norm(target - p0)), 3), residual_m=round(res, 4), lunge_m=[round(float(x), 3) for x in L], root_dz_m=round(float(dz), 3), arm_dq_max=round(float(np.abs(dj[arm]).max()), 3),
                     waist_dq=[round(float(dj[j]), 3) for j in WAIST], foot_m=round(worst_foot, 4), com_out_m=round(worst_com, 3), frames=[f0, f1])

def window(s):
    lt = lunge_times(s, W["lunge_x"]); a = AIM["arm"]
    return min(lt[0], s.t_on - a["lead_s"]), max(lt[3], max(s.t_end, s.t_apex + a["min_end_s"]) + a["tail_s"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="gpu/motions"); ap.add_argument("--copies", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head_share", type=float, default=0.5); ap.add_argument("--max_warp_m", type=float, default=W["max_warp_m"]); ap.add_argument("--name", default="r1_kungfu_aim")
    ap.add_argument("--blend_s", type=float, default=0.4); a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    m = build_spec(opponent=True).compile(); n = int(round(a.blend_s * 50)); k = Kin()
    clips = {c: FightReplay(m, c, distance=0.0).q for c in list_clips()}
    seq, index = None, {}
    for c, q in clips.items():   # exactly export_motion.py's motion
        two = np.concatenate([q, q]); start = 0 if seq is None else len(seq) - n; seq = two if seq is None else blend(seq, two, n)
        index[c] = {"start_frame": int(start), "frames": int(len(q))}
    lib = {c: [s for s in S.load(clip=c) if s.hand] for c in TRAIN_CLIPS}; assert all(s.split == "train" for v in lib.values() for s in v)
    vmax = np.percentile(np.abs(np.diff(np.concatenate([clips[c] for c in TRAIN_CLIPS])[:, 7:], axis=0)) * 50, 99, axis=0)
    apex = [(index[c]["start_frame"] + r * index[c]["frames"] + s.apex, s.limb) for c in TRAIN_CLIPS for r in (0, 1) for s in lib[c]]
    spans = [(index[c]["start_frame"] + r * index[c]["frames"] + s.onset, index[c]["start_frame"] + r * index[c]["frames"] + s.apex) for c in TRAIN_CLIPS for r in (0, 1) for s in lib[c]]
    copies, warps, rejects = [], [], {}
    for ci in range(a.copies):
        for c in TRAIN_CLIPS:
            q = clips[c].copy(); taken = []
            for s in sorted(lib[c], key=lambda s: rng.random()):
                w0, w1 = window(s)
                if any(not (w1 < u0 or w0 > u1) for u0, u1 in taken): continue
                for tr in range(W["tries"]):
                    zone = "head" if rng.random() < a.head_share else "torso"; tgt, draw = r1_target(rng, zone); q2, info = warp_strike(k, q, s, tgt, vmax, a.max_warp_m - 0.05 * tr)
                    if q2 is not None: q = q2; taken.append((w0, w1)); warps.append(dict(copy=ci, clip=c, onset=s.onset, apex=s.apex, limb=s.limb, **draw, **info)); break
                    rejects[info.split()[0]] = rejects.get(info.split()[0], 0) + 1
            start = len(seq) - n; seq = blend(seq, q, n); copies.append(dict(copy=ci, clip=c, start_frame=int(start), frames=int(len(q))))
            for w in warps:
                if w["copy"] == ci and w["clip"] == c and "frame" not in w: w["frame"] = int(start + w["apex"])
            apex += [(start + s.apex, s.limb) for s in lib[c]]; spans += [(start + s.onset, start + s.apex) for s in lib[c]]
            print(f"copy {ci} {c}: {sum(w['copy'] == ci and w['clip'] == c for w in warps)} of {len(lib[c])} strikes warped", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv = np.concatenate([seq[:, :3], seq[:, [4, 5, 6, 3]], seq[:, 7:]], 1)
    ref = np.loadtxt(out / "r1_kungfu.csv", delimiter=","); base = len(ref); assert np.allclose(csv[:base - n], ref[:base - n], atol=2e-6), "original part differs from export_motion.py"
    np.savetxt(out / f"{a.name}.csv", csv, delimiter=",", fmt="%.6f")
    hand = np.full(len(seq), -1, int)
    for f, limb in apex: hand[max(0, f - W["apex_pad"]):f + W["apex_pad"] + 1] = 0 if limb == "l_hand" else 1
    lw = np.zeros(len(seq), int); a0, a1 = int(round(W["lunge_lead_s"] * 50)), int(round(AIM["arm"]["hold_s"] * 50))
    for f0, fa in spans: lw[max(0, f0 - a0):fa + a1 + 1] = 1
    json.dump(dict(about="hand, per frame: -1 none, 0 left hand, 1 right hand at a train-clip strike apex (+-%d frames). lunge_window, per frame: 1 from "
                         "%.2f s before a train-clip hand strike's onset to its apex + %.2f s (R1-Track-KungFu-Aim2 times its lunges to these)" % (W["apex_pad"], W["lunge_lead_s"], AIM["arm"]["hold_s"]),
                   hand=hand.tolist(), lunge_window=lw.tolist()), open(out / f"{a.name}_apex.json", "w"))
    json.dump(dict(about=__doc__, fps=50, joints=CTRL_JOINTS, frames=int(len(seq)), blend_frames=n, clips=index, copies=copies, params=W, seed=a.seed, warps=warps, rejects=rejects,
                   n_warped=len(warps), n_strike_slots=a.copies * sum(len(v) for v in lib.values()), args=vars(a)), open(out / f"{a.name}_index.json", "w"), indent=1)
    print(len(seq), "frames,", round(len(seq) / 50, 1), "s;", len(warps), "warps; rejects", rejects)
