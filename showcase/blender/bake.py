"""Bake one shot of showcase/shots.json into an animated .blend: robots, camera, ring, trail, impact ripples.

blender -b showcase/build/base.blend -P showcase/blender/bake.py -- --shots showcase/shots.json --shot hook --out showcase/build/hook.blend
    [--ramp 0.1]   speed ramps: each speed change is eased over 2*ramp sim seconds inside the faster segment, so a
                   slow-motion stretch stays fully slow from its first sim second (0 = hard speed cuts)

Run from the repo root. Writes NAME.blend and NAME_frames.json (next to it): a JSON list, one sim time per rendered frame
(frame 1 = first entry); the same table is stored in the .blend as scene["frame_times"].

Timing: shot["time"] is a piecewise speed map [[t_sim, speed], ..., [t_end, null]]. Screen frames are taken at shots.fps;
the sim time of each frame comes from integrating dt_screen = dt_sim / speed (with a short smoothstep ramp between
speeds). Poses at a frame's sim time are interpolated between the two recorded 5 ms substeps around it (linear position,
slerp rotation), and every geom object's world transform is keyed every quarter frame with LINEAR interpolation (so
Cycles motion blur sees the true motion between frames). The shutter is keyed per frame: 0.5 at real time, 0.25 in the
5x slow motion (eased with the speed ramps); render.py keeps the keyed value.

Camera types (shot["camera"]):
  hero        low 3/4 angle on our R1, from the opponent's side. Azimuth = mean R1->opponent direction over the shot
              + side_deg (counter-clockwise seen from above). dist_m is the horizontal distance to the R1 torso, eased down
              by push_in (a fraction: 0.12 = 12 % closer by the end); camera height height_m; the camera follows the
              smoothed torso (heavy smoothing) and looks at the smoothed chest (light smoothing).
  fixed_side  static, side-on to the bout's starting axis (R1 -> opponent at t = 0 of the anchor shot: the shot itself, or
              the one named by same_as), at height_m, aimed at aim_z; yaw_deg swings it off the perpendicular, flip puts
              it on the other side of the axis (R1 on the right). The screen box of every body of both robots over every
              shot sharing the camera (the PD fall included) is centred horizontally by sliding along the axis, and the
              distance is set so the box fills `fill` of the frame width (same camera for split_pd and split_guard).
Lenses are full-frame equivalents on a 16:9 frame (36 mm sensor width): a narrower frame such as the 960x1080 split
halves is a crop of that 16:9 frame (sensor_width scaled down), so a 32 mm split half shows subjects at the same size as
a 32 mm full-width shot. Hero and orbit cameras then nudge their look-at point (fit_target) so the R1 stays inside the
frame (hero: head and arms; orbit: the whole robot, and the opponent if it can).
  orbit       around the R1's start spot. Azimuth eases from az0_deg to az1_deg; az 0 is straight in front of the R1 (its
              mean torso heading over the shot), positive towards the R1's right (clockwise from above). radius_m,
              height_m; looks at a smoothed mix of the spot and the R1 torso.
Depth of field focuses on the 'Focus' empty, keyed to the smoothed R1 chest; f-stop camera["fstop"] (default 4.0).
The light rig is turned so its key stays front-top and its rims behind the subjects for this shot's camera.
"""
import bpy, sys, os, json, math, argparse
import numpy as np
from mathutils import Vector, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mj2blender as mj

FX_DUR = 0.25          # sim seconds a ripple lives (1.25 screen s at 5x slow motion)
FX_GROW = 0.12         # sim seconds to reach full radius (ease-out)
FX_R0, FX_RMAX = 0.03, 0.20
FX_CORE_DUR, FX_CORE_R = 0.02, 0.07   # white-hot core flash: sim seconds (3 frames at 5x), disc radius [m]
SUBSTEP = 0.25         # robot pose keys every quarter frame (Cycles motion blur samples between frames)
TRAIL_WINDOW = 3.0     # sim seconds of pelvis path shown
TRAIL_HALF_W = 0.014   # ribbon half width [m]
LINEAR = 1             # KeyframeInterpolation enum value for 'LINEAR'


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--shots", default="showcase/shots.json"); p.add_argument("--shot", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ramp", type=float, default=0.1, help="speed-ramp half-width [sim s] at each speed change")
    p.add_argument("--fx-back", type=float, default=0.02, help="ripple centre moved this far off the struck surface, towards the striker [m]")
    return p.parse_args(a)


# ------------------------------------------------------------------------------------------------ timing
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def speed_at(tau, spec, ramp):
    knots = [k[0] for k in spec]; sp = [k[1] for k in spec[:-1]]
    s = np.full_like(tau, sp[0], dtype=float)
    for i in range(1, len(sp)):
        s = np.where(tau >= knots[i], sp[i], s)
    if ramp > 0:   # the ramp sits in the faster segment, so a slow-motion stretch stays fully slow (impacts included)
        for i in range(1, len(sp)):
            b, sa, sb = knots[i], sp[i - 1], sp[i]
            lo = b - 2 * ramp if sb < sa else b
            m = (tau >= lo) & (tau < lo + 2 * ramp)
            s = np.where(m, sa + (sb - sa) * smoothstep((tau - lo) / (2 * ramp)), s)
    return s


def frame_times(spec, fps, ramp):
    t0, t1 = spec[0][0], spec[-1][0]
    tau = np.linspace(t0, t1, int((t1 - t0) / 2e-4) + 2)
    inv = 1.0 / speed_at(tau, spec, ramp)
    screen = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(tau))])
    n = int(round(screen[-1] * fps))
    return np.interp(np.arange(n) / fps, screen, tau)


# ------------------------------------------------------------------------------------------------ poses
def interp_poses(rec, times):
    t = rec["t"]; i = np.clip(np.searchsorted(t, times, "right") - 1, 0, len(t) - 2)
    a = np.clip((times - t[i]) / (t[i + 1] - t[i]), 0.0, 1.0)
    xp = rec["xpos"][i] * (1 - a)[:, None, None] + rec["xpos"][i + 1] * a[:, None, None]
    xq = mj.slerp(rec["xquat"][i].astype(float), rec["xquat"][i + 1].astype(float), a[:, None])
    return xp, xq


def continuous(q):
    """Flip quaternion signs so consecutive keys never take the long way round."""
    q = q.copy()
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0: q[k] = -q[k]
    return q


def gsmooth(x, sigma):
    """Gaussian smoothing along axis 0 (edge-padded); sigma in samples."""
    if sigma <= 0: return x.copy()
    r = int(3 * sigma + 1); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    return np.stack([np.convolve(xp[:, j], k, "valid") for j in range(x.shape[1])], 1) if x.ndim > 1 else np.convolve(xp, k, "valid")


# ------------------------------------------------------------------------------------------------ keying
def key(id_, path, values, frames, interp=LINEAR, indices=None):
    """Write keyframes in bulk: values (N, k) for channels `indices` (default 0..k-1) of `path`."""
    values = np.asarray(values, float)
    if values.ndim == 1: values = values[:, None]
    ad = id_.animation_data or id_.animation_data_create()
    if ad.action is None: ad.action = bpy.data.actions.new(id_.name + "_bake")
    act = ad.action; n = len(frames)
    for j, idx in enumerate(indices if indices is not None else range(values.shape[1])):
        fc = act.fcurves.find(path, index=idx)
        if fc: act.fcurves.remove(fc)
        fc = act.fcurves.new(path, index=idx)
        fc.keyframe_points.add(n)
        co = np.empty(2 * n, np.float32); co[0::2] = frames; co[1::2] = values[:, j]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [interp] * n)
        fc.update()


def look_quat(cam, tgt):
    return np.array([list((Vector(t) - Vector(c)).to_track_quat("-Z", "Y")) for c, t in zip(cam, tgt)])


# ------------------------------------------------------------------------------------------------ camera
def sensor_width(res):
    """lens_mm is full-frame (36 mm across a 16:9 frame). A narrower frame (the 960x1080 split halves) is a crop of that
    16:9 frame, not a re-fit of the full sensor to its long side: same lens, same subject size, half the width."""
    return 36.0 * min(1.0, (res[0] / res[1]) / (16 / 9))


def fov_tans(lens, res):
    """tan(half fov) horizontally and vertically (sensor_fit HORIZONTAL, sensor_width(res))."""
    th = sensor_width(res) / 2 / lens; return th, th * res[1] / res[0]


def geom_samples(objs, every=12):
    """Subsampled local vertices per geom object (for framing checks)."""
    out = []
    for ob in objs:
        me = ob.data; v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", v); v = v.reshape(-1, 3)
        idx = np.unique(np.concatenate([np.arange(0, len(v), every), np.argmax(v, 0), np.argmin(v, 0)]))
        l = np.array(ob["mj_local"], float)
        out.append((int(ob["mj_body"]), mj.quat_rot(l[3:], v[idx]) + l[:3]))   # into the body frame
    return out


def world_points(rec, times, samples):
    xp, xq = interp_poses(rec, times); pts = []
    for b, V in samples:
        pts.append(xp[:, b, None, :] + mj.quat_rot(xq[:, b, None, :], V[None]))
    return np.concatenate(pts, 1).reshape(-1, 3)


def fixed_side(shots, S, cam, res, robot_objs, recs):
    """Static side-on camera shared by every shot of its group (the anchor shot and those naming it in same_as).
    Takes the screen-space box of every body of both robots over the whole range of every shot in the group (the PD
    fall included), slides camera + aim along the bout axis until that box is centred horizontally, and sets the
    distance so the box fills `fill` of the frame width (backing off if the top or bottom would clip). The camera sits
    at height_m and aims at aim_z on the bout's centre line; yaw_deg turns it off the perpendicular towards the opponent's
    side, which foreshortens the bout axis (the PD fall) and shows the R1 front-three-quarter."""
    anchor = cam.get("same_as") or S["name"]
    by = {s["name"]: s for s in shots["shots"]}
    group = [s for s in shots["shots"] if s["name"] == anchor or s["camera"].get("same_as") == anchor]
    acam = by[anchor]["camera"]                                  # every shot of the group uses the anchor's camera spec
    A = recs(by[anchor]["rec"]); bn = list(A["body_names"])
    p0 = A["xpos"][0, bn.index("pelvis"), :2].astype(float); o0 = A["xpos"][0, bn.index("opp_pelvis"), :2].astype(float)
    u = (o0 - p0) / np.linalg.norm(o0 - p0); mid = 0.5 * (p0 + o0)
    f = np.array([-u[1], u[0]])                                  # horizontal look direction: R1 left, opponent right
    yw = math.radians(acam.get("yaw_deg", 0.0))                  # > 0: camera swings towards the opponent's side (R1 front-3/4)
    f = math.cos(yw) * f - math.sin(yw) * u
    if acam.get("flip"): f = -f                                  # from the other side of the bout axis: R1 on the right
    h, az_ = acam["height_m"], acam.get("aim_z", 0.55); fill = acam.get("fill", 0.88); vmax = acam.get("v_margin", 0.94)
    th, tv = fov_tans(acam["lens_mm"], res)
    samples = geom_samples(robot_objs)
    pts = []
    for s in group:
        r = recs(s["rec"]); ts = np.arange(s["time"][0][0], min(s["time"][-1][0], float(r["t"][-1])) + 1e-9, 0.05)
        pts.append(world_points(r, ts, samples))
    pts = np.concatenate(pts, 0)
    U3 = np.array([u[0], u[1], 0.0])
    def view(D, sl):
        T = np.array([mid[0], mid[1], az_]) + sl * U3; C = np.array([mid[0] - f[0] * D, mid[1] - f[1] * D, h]) + sl * U3
        F = (T - C) / np.linalg.norm(T - C); R = np.cross(F, [0, 0, 1.0]); R /= np.linalg.norm(R); Up = np.cross(R, F)
        P = pts - C; z = P @ F
        return C, F, (P @ R) / z / th, (P @ Up) / z / tv
    D, sl = acam["dist_m"], 0.0
    for _ in range(40):
        for _ in range(6):                                       # centre the box horizontally (slide along the axis)
            C, F, nx, ny = view(D, sl); cx = 0.5 * (nx.min() + nx.max())
            Rr = np.cross(F, [0, 0, 1.0]); sl += cx * th * D / (Rr @ U3 / np.linalg.norm(Rr))   # screen-right vs the axis
        w = 0.5 * (nx.max() - nx.min())                          # half-width in ndc = fraction of the frame width
        need = max(w / fill, max(ny.max(), -ny.min()) / vmax)    # >1: too big, back off; <1: come closer
        if abs(need - 1) < 0.003: break
        D *= need ** 0.8
    C, F, nx, ny = view(D, sl)
    print(f"fixed_side: anchor {anchor}, group {[s['name'] for s in group]}, dist {D:.2f} m, slide {sl:+.3f} m along the axis, "
          f"aim z {az_:.2f} m; box ndc x [{nx.min():+.2f}, {nx.max():+.2f}] (fill {0.5 * (nx.max() - nx.min()):.2f}), "
          f"y [{ny.min():+.2f}, {ny.max():+.2f}]")
    q = Vector(F.tolist()).to_track_quat("-Z", "Y")
    return C, np.array(list(q)), math.degrees(math.atan2(-f[1], -f[0]))


def frame_points(rec, times, samples, ours=True):
    """(N, M, 3) world points of our R1's (ours) or the opponent's geoms at each time."""
    bn = list(rec["body_names"]); xp, xq = interp_poses(rec, times); pts = []
    for b, V in samples:
        if bn[b].startswith("opp_") == ours: continue
        pts.append(xp[:, b, None, :] + mj.quat_rot(xq[:, b, None, :], V[None]))
    return np.concatenate(pts, 1)


def fit_target(C, tgt, pts, th, tv, margin=0.88, top_only=False, sigma=6.0, iters=8, loose=None, loose_margin=0.97):
    """Nudge the look-at targets so the points (N, M, 3) stay inside the frame (|ndc| <= margin; with top_only, only
    the top edge and the sides are enforced, e.g. a medium shot that crops the legs on purpose). Corrections are
    smoothed over time (sigma frames), so the camera eases ahead of the motion instead of snapping. `loose` points
    (e.g. the opponent) are kept inside the wider loose_margin."""
    tgt = tgt.copy(); corr = np.zeros_like(tgt); M = pts.shape[1]
    if loose is not None: pts = np.concatenate([pts, loose], 1)
    scale = np.ones(pts.shape[1]); scale[M:] = margin / loose_margin      # loose points count as if nearer the centre
    for _ in range(iters):
        T = tgt + corr; F = T - C; dist = np.linalg.norm(F, axis=1, keepdims=True); F /= dist
        R = np.cross(F, [0, 0, 1.0]); R /= np.linalg.norm(R, axis=1, keepdims=True); U = np.cross(R, F)
        P = pts - C[:, None]; z = np.einsum("nmk,nk->nm", P, F)
        nx = np.einsum("nmk,nk->nm", P, R) / z / th * scale; ny = np.einsum("nmk,nk->nm", P, U) / z / tv * scale
        def need(lo, hi, enforce_lo=True):
            d = np.zeros(len(lo))
            over, under = hi > margin, (lo < -margin) & enforce_lo
            d = np.where(over, hi - margin, d); d = np.where(under, lo + margin, d)
            return np.where(over & under, 0.5 * (hi + lo), d)
        dx, dy = need(nx.min(1), nx.max(1)), need(ny.min(1), ny.max(1), not top_only)
        if np.abs(dx).max() < 0.01 and np.abs(dy).max() < 0.01: break
        step = (dx[:, None] * th * R + dy[:, None] * tv * U) * dist
        corr += gsmooth(step * 1.3, sigma)
    return tgt + corr


def xq_focus(rec, times, cam):
    _, xq = interp_poses(rec, times); return xq[:, list(rec["body_names"]).index(cam.get("focus", "torso_link"))]


def bake_camera(S, shots, times, frames, xp, rec, res, robot_objs, recs):
    th, tv = fov_tans(S["camera"]["lens_mm"], res); samples = geom_samples(robot_objs)
    cam = S["camera"]; co = bpy.data.objects["Camera"]; foc = bpy.data.objects["Focus"]; cd = co.data
    cd.lens = cam["lens_mm"]; cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = sensor_width(res)
    cd.dof.use_dof = True; cd.dof.aperture_fstop = cam.get("fstop", 4.0); cd.dof.focus_object = foc
    bn = list(rec["body_names"]); N = len(times)
    P = xp[:, bn.index(cam.get("focus", "torso_link"))].astype(float); O = xp[:, bn.index("opp_" + cam.get("focus", "torso_link"))].astype(float)
    chest = P + np.array([0, 0, 0.22])
    key(foc, "location", gsmooth(chest, 3), frames)
    e = smoothstep(np.linspace(0, 1, N))
    t = cam["type"]
    if t == "hero":
        # the bout axis at the shot's key moment: around its first scripted hit (else over the whole shot); the
        # opponent circles, so a whole-shot mean can put it straight between the camera and the R1
        hits = [h["t0"] for h in json.loads(str(rec["hits_json"])) if times[0] <= h["t0"] <= times[-1]]
        w = np.abs(times - hits[0]) <= 0.25 if hits else np.ones(N, bool)
        d = (O - P)[w, :2].mean(0); az = math.atan2(d[1], d[0]) + math.radians(cam.get("side_deg", 35))
        base = gsmooth(P[:, :2], 24); dist = cam["dist_m"] * (1 - cam.get("push_in", 0.0) * e)
        C = np.column_stack([base[:, 0] + dist * math.cos(az), base[:, 1] + dist * math.sin(az), np.full(N, cam["height_m"])])
        # aim between the two chests (mostly ours: look_mix of the way to the opponent) so the pair is framed, not just the R1
        lm = cam.get("look_mix", 0.3)
        tgt = gsmooth((1 - lm) * P + lm * O + np.array([0, 0, cam.get("look_dz", 0.18)]), 8)
        # a medium shot: the legs may leave the frame, the R1's head and arms may not
        tgt = fit_target(C, tgt, frame_points(rec, times, samples), th, tv, margin=0.9, top_only=True)
        key(co, "location", C, frames); key(co, "rotation_quaternion", continuous(look_quat(C, tgt)), frames)
        yaw = math.degrees(az)
        # framing report at the key moment: how much of the frame width the opponent covers, and whether the R1's head is clear
        if hits:
            k = int(np.argmin(np.abs(times - hits[0]))); Fv = tgt[k] - C[k]; Fv /= np.linalg.norm(Fv)
            Rv = np.cross(Fv, [0, 0, 1.0]); Rv /= np.linalg.norm(Rv)
            def ndcx(pp): P_ = pp - C[k]; return (P_ @ Rv) / (P_ @ Fv) / th
            ox = ndcx(frame_points(rec, times[k:k + 1], samples, ours=False)[0]); rx = ndcx(frame_points(rec, times[k:k + 1], samples)[0])
            vis = lambda a: max(0.0, min(a.max(), 1) - max(a.min(), -1)) / 2
            print(f"hero @ t {times[k]:.2f}: azimuth {yaw:.0f} deg, opponent covers {vis(ox) * 100:.0f} % of the frame width "
                  f"(ndc x {ox.min():+.2f}..{ox.max():+.2f}), R1 ndc x {rx.min():+.2f}..{rx.max():+.2f}")
    elif t == "fixed_side":
        C, q, yaw = fixed_side(shots, S, cam, res, robot_objs, recs)
        for path, v in (("location", C), ("rotation_quaternion", q)):
            key(co, path, np.repeat(v[None], N, 0), frames)
    elif t == "orbit":
        spot = rec["spot"].astype(float)
        # azimuth 0 = straight in front of the R1 (its mean torso heading over the shot), positive towards the R1's right
        # (clockwise from above): the orbit always shows our robot's front, whichever way the bout has turned it
        qT = xq_focus(rec, times, cam)
        yaw = np.arctan2(2 * (qT[:, 0] * qT[:, 3] + qT[:, 1] * qT[:, 2]), 1 - 2 * (qT[:, 2] ** 2 + qT[:, 3] ** 2))
        head = math.atan2(np.sin(yaw).mean(), np.cos(yaw).mean())
        az = head - np.radians(cam["az0_deg"] + (cam["az1_deg"] - cam["az0_deg"]) * e)
        print(f"orbit: R1 mean heading {math.degrees(head):.0f} deg, camera azimuth {math.degrees(az[0]):.0f} -> {math.degrees(az[-1]):.0f} deg (world)")
        R, H = cam["radius_m"], cam["height_m"]
        C = np.column_stack([spot[0] + R * np.cos(az), spot[1] + R * np.sin(az), np.full(N, H)])
        lz = cam.get("look_z", 0.62)
        tgt = gsmooth(0.4 * np.array([spot[0], spot[1], lz]) + 0.6 * np.column_stack([P[:, 0], P[:, 1], np.full(N, lz)]), 14)
        # the 1 m ring (32 points on the mat) is kept in with the opponent, inside the looser margin
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ringp = np.column_stack([spot[0] + 1.05 * np.cos(ang), spot[1] + 1.05 * np.sin(ang), np.zeros(32)])
        loose = np.concatenate([frame_points(rec, times, samples, ours=False), np.repeat(ringp[None], N, 0)], 1)
        tgt = fit_target(C, tgt, frame_points(rec, times, samples), th, tv, margin=0.86, sigma=10,   # all of the R1, feet included
                         loose=loose, loose_margin=cam.get("loose_margin", 0.95))                     # the opponent and the ring, if it can
        key(co, "location", C, frames); key(co, "rotation_quaternion", continuous(look_quat(C, tgt)), frames)
        Fv = tgt - C; Fv /= np.linalg.norm(Fv, axis=1, keepdims=True); Rv = np.cross(Fv, [0, 0, 1.0]); Rv /= np.linalg.norm(Rv, axis=1, keepdims=True)
        Uv = np.cross(Rv, Fv); Pr = ringp[None] - C[:, None]; zr = np.einsum("nmk,nk->nm", Pr, Fv)
        rx = np.einsum("nmk,nk->nm", Pr, Rv) / zr / th; ry = np.einsum("nmk,nk->nm", Pr, Uv) / zr / tv
        print(f"orbit: ring ndc over the shot x [{rx.min():+.2f}, {rx.max():+.2f}], y [{ry.min():+.2f}, {ry.max():+.2f}]")
        yaw = math.degrees(float(np.mean(az)))
    else:
        raise SystemExit(f"unknown camera type {t}")
    if t == "fixed_side": C = np.repeat(C[None], N, 0)
    return yaw, C


# ------------------------------------------------------------------------------------------------ overlays
def hide(ob, h=True):
    for o in [ob] + list(ob.children_recursive):
        o.hide_render = h; o.hide_viewport = h


def bake_trail(rec, times, frames, t_start, t_end):
    tr = bpy.data.objects["Trail"]; bn = list(rec["body_names"]); t = rec["t"]
    sel = np.where((t >= t_start - TRAIL_WINDOW - 0.2) & (t <= t_end + 0.05))[0][::2]
    p = rec["xpos"][sel, bn.index("pelvis"), :2].astype(float); ts = t[sel]
    p = gsmooth(p, 6)                                     # ~0.06 s: removes substep jitter, keeps the gait sway
    tang = np.gradient(gsmooth(p, 30), axis=0); nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    ln = np.linalg.norm(nrm, axis=1); last = np.array([0.0, 1.0])
    for k in range(len(nrm)):                              # hold the last direction while standing still
        if ln[k] > 1e-5: last = nrm[k] / ln[k]
        nrm[k] = last
    V = np.zeros((2 * len(p), 3)); V[0::2, :2] = p + TRAIL_HALF_W * nrm; V[1::2, :2] = p - TRAIL_HALF_W * nrm; V[:, 2] = 0.003
    F = [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(len(p) - 1)]
    me = tr.data; me.clear_geometry(); me.from_pydata(V.tolist(), [], F)
    for name, vals in (("t", np.repeat(ts, 2)), ("v", np.tile([1.0, -1.0], len(p)))):
        a = me.attributes.get(name) or me.attributes.new(name, "FLOAT", "POINT"); a.data.foreach_set("value", vals.astype(np.float32))
    me.update()
    mod = tr.modifiers["Trail"]; ids = {i.name: i.identifier for i in mod.node_group.interface.items_tree if getattr(i, "in_out", "") == "INPUT"}
    mod[ids["Window"]] = TRAIL_WINDOW
    key(tr, f'modifiers["Trail"]["{ids["Now"]}"]', times, frames)
    print(f"trail: {len(p)} samples, {ts[0]:.2f}..{ts[-1]:.2f} s")


def hit_anchor(rec, h, robot_objs):
    """The hit point on the struck robot's surface, in the struck body's frame: the recorded application point (body
    origin + offset) is pushed back along the force line onto the outermost surface the striker would touch first (the
    vertices of that body's geoms within 4 cm of the line, the one furthest towards the attacker)."""
    bn = list(rec["body_names"]); b = bn.index(h["body"]); xp, xq = interp_poses(rec, np.array([h["t0"]]))
    off = np.array(h["offset"], float); pt = xp[0, b] + mj.quat_rot(xq[0, b], off)
    fd = np.array(h["force"], float); fd /= np.linalg.norm(fd)
    V = [V for bb, V in geom_samples([o for o in robot_objs if int(o["mj_body"]) == b], every=1)]
    if not V: return off, fd
    W = xp[0, b] + mj.quat_rot(xq[0, b], np.concatenate(V, 0))            # world, at the hit
    d = W - pt; along = d @ fd; perp = np.linalg.norm(d - along[:, None] * fd, axis=1)
    near = perp < 0.04
    if not near.any(): return off, fd
    surf = pt + along[near].min() * fd                                   # furthest towards the attacker (-fd)
    qi = xq[0, b] * np.array([1, -1, -1, -1.0])
    return mj.quat_rot(qi, surf - xp[0, b]), fd                          # back into the body frame


def bake_fx(rec, times, frames, robot_objs, Ccam, back):
    """Per scripted hit in the shot: an orange shockwave ring at the hit point on the struck surface (normal along the
    impulse, so it reads as a wave leaving the impact, not a backdrop disc), radius eased out to FX_RMAX over FX_GROW sim
    seconds and faded to 0 by FX_DUR; plus a white-hot core flash (camera-facing) for the first FX_CORE_DUR sim seconds.
    Both follow the struck body (fixed world orientation)."""
    hits = json.loads(str(rec["hits_json"])); bn = list(rec["body_names"])
    tmpl = bpy.data.objects["FX_Ripple"]; ctmpl = bpy.data.objects["FX_Core"]
    coll = bpy.data.collections["FX"]; made = []
    for i, h in enumerate(hits):
        if not (times[0] - FX_DUR <= h["t0"] <= times[-1]): continue
        if h["t0"] > rec["t"][-1]: continue
        b = bn.index(h["body"]); loc, fd = hit_anchor(rec, h, robot_objs)
        xp, xq = interp_poses(rec, times)
        anc = xp[:, b] + mj.quat_rot(xq[:, b], loc[None]) - back * fd   # just off the surface, towards the striker
        age = times - h["t0"]; on = (age >= 0) & (age <= FX_DUR)
        ob = tmpl.copy(); ob.name = f"FX_Hit{i}_{h['kind']}_{h['body']}"; coll.objects.link(ob)
        ob.hide_render = False; ob.hide_viewport = False; ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(fd.tolist()))
        g = np.clip(age / FX_GROW, 0, 1); r = np.where(on, FX_R0 + (FX_RMAX - FX_R0) * (1 - (1 - g) ** 3), 1e-4)
        a = np.where(on, np.clip(age / 0.006, 0, 1) * np.clip(1 - age / FX_DUR, 0, 1) ** 1.3, 0.0)
        key(ob, "location", anc, frames); key(ob, "scale", np.column_stack([r, r, r]), frames); key(ob, "color", a, frames, indices=[3])
        # core flash: first frame at/after the hit at full strength, decaying over FX_CORE_DUR
        co = ctmpl.copy(); co.name = f"FX_Core{i}_{h['body']}"; coll.objects.link(co)
        co.hide_render = False; co.hide_viewport = False; co.rotation_mode = "QUATERNION"
        k0 = int(np.argmax(age >= 0)); toc = Ccam[k0] - anc[k0]
        co.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(toc.tolist()))
        ca = np.where((age >= 0) & (age <= FX_CORE_DUR), np.exp(-np.maximum(age - age[k0], 0) / (FX_CORE_DUR / 2.5)), 0.0)
        cpos = anc + 0.015 * toc / np.linalg.norm(toc)
        key(co, "location", cpos, frames); key(co, "scale", np.full((len(frames), 3), FX_CORE_R), frames); key(co, "color", ca, frames, indices=[3])
        made.append(dict(name=ob.name, t0=h["t0"], kind=h["kind"], body=h["body"], point=[round(float(x), 4) for x in anc[k0]],
                         core_frames=int((ca > 0.05).sum())))
    print("fx:", made)
    return made


def bake_backdrop(spec, xp, rec, Ccam):
    """A soft, dim light pool on the cyc behind the subjects, facing the camera: silhouette separation in the void."""
    if not spec: return
    bn = list(rec["body_names"]); P = xp[:, bn.index("torso_link")].mean(0); Cm = Ccam.mean(0)
    d = P - Cm; d[2] = 0; d /= np.linalg.norm(d)
    ob = bpy.data.objects["BG_Pool"].copy(); ob.data = ob.data.copy(); bpy.data.collections["Set"].objects.link(ob)
    ob.name = "BG_Pool_Shot"; ob.hide_render = False; ob.hide_viewport = False
    m = ob.data.materials[0].copy(); ob.data.materials[0] = m
    em = [n for n in m.node_tree.nodes if n.type == "EMISSION"][0]
    c = list(em.inputs["Color"].default_value); em.inputs["Color"].default_value = (*[x * spec.get("strength", 0.05) for x in c[:3]], 1)
    loc = P + d * spec.get("dist", 5.0); loc[2] = spec.get("z", 1.1)
    ob.location = loc.tolist(); ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector((-d).tolist()))
    ob.scale = (spec.get("radius", 3.0),) * 3
    print(f"backdrop: pool at {np.round(loc, 2).tolist()}, radius {spec.get('radius', 3.0)} m, strength {spec.get('strength', 0.05)}")


def sub_times(times, frames, step):
    """Sim times at sub-frame steps (frame - 0.5 .. last + 0.5), for motion-blur keys between frames."""
    sub = np.arange(frames[0] - 0.5, frames[-1] + 0.5 + 1e-9, step)
    ts = np.interp(sub, frames, times)
    lo, hi = sub < frames[0], sub > frames[-1]
    ts[lo] = times[0] + (sub[lo] - frames[0]) * (times[1] - times[0]); ts[hi] = times[-1] + (sub[hi] - frames[-1]) * (times[-1] - times[-2])
    return sub, ts


# ------------------------------------------------------------------------------------------------ main
def main():
    A = args()
    shots = json.load(open(A.shots)); S = next(s for s in shots["shots"] if s["name"] == A.shot)
    fps = shots["fps"]; res = S.get("res", shots["res"])
    cache = {}
    def recs(path):
        if path not in cache: cache[path] = dict(np.load(path))
        return cache[path]
    rec = recs(S["rec"])
    times = frame_times(S["time"], fps, A.ramp); N = len(times); frames = np.arange(1, N + 1, dtype=float)
    xp, xq = interp_poses(rec, times)
    print(f"shot {S['name']}: {N} frames at {fps} fps, sim {times[0]:.3f}..{times[-1]:.3f} s ({S['rec']})")

    sc = bpy.context.scene
    sc.render.fps = fps; sc.frame_start = 1; sc.frame_end = N; sc.frame_current = 1
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.resolution_percentage = 100

    robot = [o for c in ("R1", "Opponent") for o in bpy.data.collections[c].objects if "mj_body" in o]
    sub, tsub = sub_times(times, frames, SUBSTEP); sxp, sxq = interp_poses(rec, tsub)
    for ob in robot:
        b = int(ob["mj_body"]); l = np.array(ob["mj_local"], float)
        pos, quat = mj.geom_world(sxp[:, b], sxq[:, b], l[:3], l[3:])
        ob.rotation_mode = "QUATERNION"
        key(ob, "location", pos, sub); key(ob, "rotation_quaternion", continuous(quat), sub)
    # shutter: 0.5 at real time, 0.25 in the 5x slow motion (crisp), eased with the speed ramps
    speed = np.gradient(times) * fps; shutter = 0.25 + 0.25 * np.clip((speed - 0.2) / 0.8, 0, 1)
    key(sc, "render.motion_blur_shutter", shutter, frames); sc["shutter_keyed"] = True
    print(f"pose keys: {len(sub)} sub-frame samples ({SUBSTEP} frame step); shutter {shutter.min():.2f}..{shutter.max():.2f}")

    spot = rec["spot"].astype(float)
    ring = bpy.data.objects["Ring"]; ring.location = (spot[0], spot[1], 0.0); hide(ring, not S.get("ring", False))
    bpy.data.objects["Mat"].location = (spot[0], spot[1], 0.0)
    if S.get("trail"): bake_trail(rec, times, frames, times[0], times[-1])
    else: hide(bpy.data.objects["Trail"])
    yaw, Ccam = bake_camera(S, shots, times, frames, xp, rec, res, robot, recs)
    fx = bake_fx(rec, times, frames, [o for o in robot if not str(o["mj_body_name"]).startswith("opp_")], Ccam, A.fx_back) if S.get("fx") else []
    bake_backdrop(S.get("backdrop"), xp, rec, Ccam)
    rig = bpy.data.objects["LightRig"]; P = xp[:, list(rec["body_names"]).index("torso_link")]
    rig.location = (float(P[:, 0].mean()), float(P[:, 1].mean()), 0.0); rig.rotation_euler = (0, 0, math.radians(yaw))
    # mat depth fade: from just behind the robots to black a couple of metres further (no horizon line at grazing angles)
    dc = float(np.linalg.norm(Ccam - np.array([P[:, 0].mean(), P[:, 1].mean(), 0.0]), axis=1).mean())
    fn, fl = S.get("mat_fade", [0.3, 2.6]); nodes = bpy.data.materials["Mat_Rubber"].node_tree.nodes
    nodes["fade_near"].outputs[0].default_value = dc + fn; nodes["fade_far"].outputs[0].default_value = dc + fn + fl
    print(f"mat depth fade: {dc + fn:.2f} -> {dc + fn + fl:.2f} m from the camera")
    for name, g in S.get("lights", {}).items():   # per-shot light gains, e.g. {"Rim_L": 1.6} = +0.7 stop
        bpy.data.lights[name].energy *= g; print(f"light {name}: x{g} -> {bpy.data.lights[name].energy:.0f} W")

    # contact check: the lowest point of each robot at the first frame (z = 0 is the MuJoCo floor = the mat's top)
    smp = geom_samples(robot)
    for nm, ours in (("R1", True), ("opponent", False)):
        z0 = frame_points(rec, times[:1], smp, ours)[0, :, 2].min(); print(f"contact: {nm} lowest point at frame 1: z = {z0 * 1000:+.1f} mm")
    sc["frame_times"] = [float(x) for x in times]
    sc["shot"] = json.dumps(S); sc["shot_name"] = S["name"]; sc["fx"] = json.dumps(fx)
    sc["ramp_s"] = A.ramp
    out = os.path.abspath(A.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    fj = os.path.join(os.path.dirname(out), f"{S['name']}_frames.json")
    with open(fj, "w") as fh: json.dump([round(float(x), 6) for x in times], fh)
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True, compress=True)
    print("saved", A.out, "and", os.path.relpath(fj))


if __name__ == "__main__":
    main()
