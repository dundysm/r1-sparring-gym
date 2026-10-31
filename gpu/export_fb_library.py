"""Library L1 for the full-body M2 v2 opponent (M2V2_PLAN_v3 2.2 and build order M8; design_learned-tracker 9.1), exported on CPU from
the train clips only. The one motion source is gpu/motions/fb/fb_motion.npz's q (M1: FightReplay(distance=0).q of kfab_200/201/203,
in place, facing -x, seamless loops); strikes are the eval's rule (r1gym.strikes.detect) on each base. Every row is a whole loopable
clip. Priors fixed in design_learned-tracker 9.1; generator np.random.default_rng([SEED, family, clip, mirror, draw]).
  mirror   L/R mirror of each clip (3 rows): swap left_* / right_*, negate the roll and yaw joints (waist included) and root y,
           quaternion (w, -x, y, -z); clipped to the model's ranges, which are mirror-symmetric within RANGE_TOL (asserted).
  warp     2 draws per clip and mirror state (12 rows): each hand strike whose window [onset - 0.2, end + 0.1] s lies inside the
           loop and clear of an accepted window plays at speed x U(0.7, 1.3), smoothstep ramps over [onset - 0.2, onset] and
           [end, end + 0.1]; the speed is then adjusted so the window lasts a whole number of frames (frames outside every
           window are the base's, bit for bit).
  reangle  2 draws per clip and mirror state (12 rows): each hook / straight / uppercut hand strike (downward: no prior): the palm
           path of the 0.10 s before the apex is rotated about the apex in elevation (about h x z, h the approach's horizontal
           direction over those 0.10 s, as strikes.py measures it): hooks U(-40, 40) deg, straights U(-20, 20), uppercuts to an
           elevation U(30, 70); ramped in by smoothstep over the 0.10 s before. The striking arm's 5 joints are re-solved per frame
           (export_warps.solve, +-0.8 rad, 95% of range); the apex frame is the base's.
  combo    1 per clip and mirror state (6 rows): up to 3 hand strikes of the next train clip (200 <- 201 <- 203 <- 200, either
           mirror state) spliced into the host's guard right after a host hand strike: the host fades out over the fade, the
           segment [onset - 0.1 s - fade, end + 0.1 s + fade] plays, and the host resumes where it paused (fading back in). The
           fade (both ends) is the shortest on the frame grid of the prior range [0.15, 0.35] s that passes the speed rule
           (plan A6.1; the two U(0.15, 0.35) draws are still made, unused, so the generator stream is M8's); none passes: rejected.
One draw per strike, no redraw (the prior is not reshaped by the rejections). Rejected: export_warps' joint speed rule (|dq| >
max(the clip set's p99, the window's own fastest) + 0.05 rad/s) in every family; for a combo the allowance is max(p99, the host
window's own fastest, the source segment's own fastest) + 0.05, each measured on its own frames (the join between them is never
a speed, A6.1); re-angle also when the palm misses its path by
> 3 cm, a foot moves > 1 cm, the CoM leaves the support polygon by > 2 cm more than the base, a glove (wrist + 0.08 m) is > 0.75 m
from its shoulder, or the palm goes further forward than the base apex anywhere in the window (no added reach). A rejected strike
stays as in the base. p99 (vmax) is per clip set (A6.3): the 3 train clips for L1, kfab_200/201 for LOCO.
Mirror conjugation (A6.2): every mirrored-base row (mirror = 1: the mirror family and the mirrored half of warp / re-angle /
combo) is driven by the mirror-conjugated tracker (gpu/fb_mirror.py), T0 or T1. T1 therefore learns only the unmirrored forms:
the T1 motion holds the originals, the original-base rows and mirror(row) of the mirrored-base warp / re-angle / combo rows (the
mirror family is the originals there); a row's shown reference is what the deployed (conjugated) driver must be fed, mirror_ref
of T1's for a mirrored-base row.
LOCO (--loco, A6.3): the same library over kfab_200/201 only (generator seeds unchanged, vmax of that set, no combo whose source
is kfab_203), written as fb_l1_loco* and fb_l1_track_loco*.
Outputs (gpu/motions/fb):
  fb_l1.npz              names, family, host, mirror, draw [R]; frames, row0; q [F, 31] float64 at distance 0; src_clip, src_frame,
                         src_mirror [F] (dominant source of every frame); conj [R] (driven conjugated, = mirror); ref_joint_pos / ref_joint_vel [F, 24], ref_anchor_pos [F, 3],
                         ref_anchor_quat [F, 4] float32: what the driver of a tracker trained on fb_l1_track.csv is fed at row frame
                         f (Tracker.frame on that motion, csv_to_npz's float32 grid as export_fb_motion; mirror_ref of it on a
                         conjugated row); strike_* (detect on each row); clean [R, 31, 300]
                         (scenario.clean_starts, the eval's call, as M1) at distances 0.35-0.65.
  fb_l1_track.csv        the T1 motion (r1_track_spar_cfg): the 3 originals then the L1 rows but the mirror family (conjugated
                         rows un-mirrored), each twice, cross-faded 0.4 s
                         (export_motion's layout: root xyz, quat xyzw, 24 joints, 50 fps); fb_l1_track_index.json its rows.
  fb_l1_manifest.json    per row: sources (train clip -> frame span), transforms, sha256 of q; per family rejection stats; sha256
                         of the outputs and every source; git hash; priors; vmax; disclosures.
Run in the repo: python gpu/export_fb_library.py [--out gpu/motions/fb] [--no-clean] [--workers 2] [--loco]"""
import sys, json, hashlib, argparse
from pathlib import Path
GPU = Path(__file__).resolve().parent; ROOT = GPU.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(GPU))
import numpy as np, mujoco
from r1gym import strikes as S
from r1gym.env import TRAIN_CLIPS
from r1gym.model import CTRL_JOINTS
from r1gym.opponent import _sub
from export_motion import blend
import export_fb_motion as EM, export_warps as EW, fb_mirror as FM

CLIPS = ("kfab_200", "kfab_201", "kfab_203"); MOTION = GPU / "motions" / "fb" / "fb_motion.npz"; OUT = GPU / "motions" / "fb"
FAMILIES = ("mirror", "warp", "reangle", "combo"); DRAWS = dict(mirror=1, warp=2, reangle=2, combo=1)
SEED, FPS, BLEND_S, RANGE_TOL = 0, 50, 0.4, 1e-4
PRIOR = dict(warp_speed=(0.7, 1.3), warp_pre_s=0.2, warp_post_s=0.1, reangle_deg=dict(hook=(-40.0, 40.0), straight=(-20.0, 20.0)),
             uppercut_elev_deg=(30.0, 70.0), reangle_s=0.10, reangle_ramp_s=0.10, combo_fade_s=(0.15, 0.35), combo_pad_s=0.1, combo_max=3,
             combo_gap_s=1.0, combo_source={"kfab_200": "kfab_201", "kfab_201": "kfab_203", "kfab_203": "kfab_200"})
REJECT = dict(path_m=0.03, foot_m=0.01, com_m=0.02, speed_tol=0.05, glove_m=0.75, reach_m=0.0, max_dq=0.8, ik_iters=40, ik_prior=1e-3)
GLOVE = (0.08, 0.0, 0.0)   # glove keypoint in the wrist_roll_link frame (plan 2.3)
SOURCES = ["gpu/motions/fb/fb_motion.npz", "r1gym/assets/r1/r1.xml", "r1gym/strikes.py", "r1gym/opponent.py", "r1gym/aim.py", "r1gym/scenario.py",
           "r1gym/model.py", "r1gym/env.py", "configs/r1_assumptions.yaml", "gpu/export_motion.py", "gpu/export_warps.py", "gpu/export_fb_motion.py",
           "gpu/export_fb_library.py", "gpu/fb_mirror.py"]
LOCO_CLIPS = ("kfab_200", "kfab_201")
FADES = tuple(range(int(np.ceil(PRIOR["combo_fade_s"][0] * FPS)), int(np.floor(PRIOR["combo_fade_s"][1] * FPS)) + 1))   # 8..17 frames
f32 = np.float32

def smooth(x): x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)

# ---- mirror
def mirror_map():
    """(perm, sign) over CTRL_JOINTS: joint i of the mirror = sign[i] x joint perm[i] of the source."""
    n = list(CTRL_JOINTS); sw = lambda s: s.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
    return np.array([n.index(sw(j)) for j in n]), np.array([-1.0 if ("_roll" in j or "_yaw" in j) else 1.0 for j in n])

def ranges():
    r = _sub("r1"); assert [r.joint(i).name for i in range(1, r.njnt)] == list(CTRL_JOINTS); return r.jnt_range[1:, 0].copy(), r.jnt_range[1:, 1].copy()

def range_symmetry():
    """Per joint |mirrored range - range| (max of both ends): the mirror of joint perm[i]'s range against joint i's."""
    lo, hi = ranges(); p, s = mirror_map(); mlo, mhi = np.where(s > 0, lo[p], -hi[p]), np.where(s > 0, hi[p], -lo[p])
    e = np.maximum(np.abs(mlo - lo), np.abs(mhi - hi)); return {j: float(x) for j, x in zip(CTRL_JOINTS, e)}

def mirror(q, clip=True):
    """The L/R mirror of qpos rows [T, 31] (reflection y -> -y); clip: into the model's ranges."""
    p, s = mirror_map(); o = q.copy(); o[:, 1] = -q[:, 1]; o[:, 3:7] = q[:, 3:7] * [1.0, -1.0, 1.0, -1.0]; o[:, 7:] = s * q[:, 7 + p]
    if clip: lo, hi = ranges(); o[:, 7:] = np.clip(o[:, 7:], lo, hi)
    return o

# ---- shared pieces
def base_motion(path=MOTION):
    """{clip: q [T, 31]} from fb_motion.npz (train clips only, asserted)."""
    with np.load(path, allow_pickle=False) as z: clips, fr, r0, q = z["clips"].tolist(), z["frames"], z["row0"], z["q"]
    assert tuple(clips) == CLIPS and set(clips) == set(TRAIN_CLIPS), clips
    return {c: q[int(r0[k]):int(r0[k]) + int(fr[k])].copy() for k, c in enumerate(clips)}

def bases(Q, clips=CLIPS):
    """{(clip, mirror): q} over `clips` and the joint-speed p99 of each base set of those clips (export_warps' vmax, per clip set)."""
    B = {(c, m): (mirror(Q[c]) if m else Q[c].copy()) for c in clips for m in (0, 1)}
    vmax = {m: np.percentile(np.abs(np.diff(np.concatenate([B[(c, m)] for c in clips])[:, 7:], axis=0)) * FPS, 99, axis=0) for m in (0, 1)}
    return B, vmax

def strikes(q, name=""): return S.detect(q, FPS, clip=name, split="train")

def lerp_q(q, u):
    """FightReplay.qpos_at between frames floor(u) and floor(u) + 1 (hemisphere matched, renormalised)."""
    i = int(np.floor(u)); a = u - i; q0, q1 = q[i], q[min(i + 1, len(q) - 1)].copy()
    if np.dot(q0[3:7], q1[3:7]) < 0: q1[3:7] *= -1
    o = (1 - a) * q0 + a * q1; o[3:7] /= np.linalg.norm(o[3:7]); return o

def fastest(x): return np.abs(np.diff(x[:, 7:], axis=0)).max(0) * FPS

def speed_over(new, old, vmax, tol=REJECT["speed_tol"]):
    """export_warps' rule: joints of `new` faster than max(vmax, the own fastest of `old`) + tol (names). old: a frame span, or a
    list of spans each measured on its own frames (no speed across their joins)."""
    own = np.max([fastest(o) for o in (old if isinstance(old, list) else [old])], 0)
    return [CTRL_JOINTS[j] for j in np.where(fastest(new) > np.maximum(vmax, own) + tol)[0]]

# ---- warp
def warp_map(fa, on, en, fb, sp, sub=400):
    """Output frames of a window [fa, fb] played at rate 1 + (sp - 1) g(u) (g: smoothstep up over [fa, on], 1, down over [en, fb]):
    (sp', u_k) with sp' the rate whose window lasts exactly L = round(duration(sp)) frames, u_k the source frame of output k < L."""
    u = np.linspace(fa, fb, (fb - fa) * sub + 1)
    g = np.where(u < on, smooth((u - fa) / max(on - fa, 1e-9)), np.where(u <= en, 1.0, smooth((fb - u) / max(fb - en, 1e-9))))
    def t(s): r = 1.0 / (1.0 + (s - 1.0) * g); return np.r_[0.0, np.cumsum((r[1:] + r[:-1]) / 2 * np.diff(u))]
    L = int(round(t(sp)[-1])); lo, hi = 0.3, 3.0
    for _ in range(200):   # the duration falls monotonically with the rate
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if t(mid)[-1] > L else (lo, mid)
    s = (lo + hi) / 2; tt = t(s); return s, L, np.interp(np.arange(L), tt, u)

def warp_row(q, src, sts, rng, vmax):
    """One warp draw over a base: (q', src', mods)."""
    pre, post = int(round(PRIOR["warp_pre_s"] * FPS)), int(round(PRIOR["warp_post_s"] * FPS)); T = len(q); acc, mods = [], []
    for i in rng.permutation(len(sts)):
        s = sts[i]; fa, fb = s.onset - pre, s.end + post; sp = float(rng.uniform(*PRIOR["warp_speed"]))
        rec = dict(limb=s.limb, type=s.type, onset=s.onset, apex=s.apex, end=s.end, speed=round(sp, 6))
        if fa < 1 or fb > T - 2: mods.append(dict(rec, status="skip", reason="seam")); continue
        if any(not (fb < a or fa > b) for a, b, *_ in acc): mods.append(dict(rec, status="skip", reason="overlap")); continue
        s2, L, uk = warp_map(fa, s.onset, s.end, fb, sp); seg = np.array([lerp_q(q, x) for x in uk])
        over = speed_over(np.concatenate([q[fa - 1:fa], seg, q[fb:fb + 1]]), q[fa - 1:fb + 1], vmax)
        if over: mods.append(dict(rec, status="reject", reason="speed", joints=over)); continue
        acc.append((fa, fb, seg, uk)); mods.append(dict(rec, status="accept", speed_frames=round(s2, 6), frames_in=fb - fa, frames_out=L))
    out, so, cur = [], [], 0
    for fa, fb, seg, uk in sorted(acc, key=lambda x: x[0]):
        out += [q[cur:fa], seg]; so += [src[cur:fa], np.interp(uk, np.arange(T), src)]; cur = fb
    return np.concatenate(out + [q[cur:]]), np.concatenate(so + [src[cur:]]), mods

# ---- re-angle
class Kin(EW.Kin):
    def __init__(self):
        super().__init__(); m = self.m; self.palm_id = self.palm
        self.wrist = {k: m.body(f"{s}_wrist_roll_link").id for k, s in (("l_hand", "left"), ("r_hand", "right"))}
        self.shoulder = {k: m.body(f"{s}_shoulder_pitch_link").id for k, s in (("l_hand", "left"), ("r_hand", "right"))}
    def glove_shoulder(self, q, limb):
        d = self.fk(q); w = self.wrist[limb]; return float(np.linalg.norm(d.xpos[w] + d.xmat[w].reshape(3, 3) @ GLOVE - d.xpos[self.shoulder[limb]]))
    def palm_at(self, q, limb): return self.fk(q).site_xpos[self.palm_id[limb]].copy()

def rot(ax, a):
    ax = ax / np.linalg.norm(ax); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

def reangle_strike(k, q, s, draw, vmax):
    """Re-angle strike s of q in place on a copy: (q', info) or (None, info with the reason). draw: U(0, 1) for the prior."""
    n, nr = int(round(PRIOR["reangle_s"] * FPS)), int(round(PRIOR["reangle_ramp_s"] * FPS)); ap, limb = s.apex, s.limb; f0 = ap - n - nr
    info = dict(limb=limb, type=s.type, onset=s.onset, apex=ap, end=s.end)
    if f0 < 1 or ap > len(q) - 2: return None, dict(info, reason="seam")
    P = np.array([k.palm_at(q[f], limb) for f in range(f0, ap + 1)]); Pa = P[-1]; u = Pa - P[max(s.onset, ap - n) - f0]; hx = np.linalg.norm(u[:2])
    if hx < 1e-6: return None, dict(info, reason="vertical")
    e0 = float(np.degrees(np.arctan2(u[2], hx))); h = np.r_[u[:2] / hx, 0.0]; ax = np.cross(h, [0.0, 0.0, 1.0])
    if s.type == "uppercut": e1 = PRIOR["uppercut_elev_deg"][0] + draw * (PRIOR["uppercut_elev_deg"][1] - PRIOR["uppercut_elev_deg"][0]); dl = e1 - e0
    else: a, b = PRIOR["reangle_deg"][s.type]; dl = a + draw * (b - a); e1 = e0 + dl
    info.update(elev0_deg=round(e0, 3), delta_deg=round(dl, 3), elev1_deg=round(e1, 3))
    arm = EW.ARM[limb]; var = [6 + j for j in arm]; out = q.copy(); prev = np.zeros(5); res = foot = com = glove = 0.0; reach = -np.inf
    floor = min(k.fk(q[ap]).geom_xpos[g][2] - k.m.geom_size[g][0] for gs in k.sole.values() for g in gs); fwd_ap = -Pa[0]
    for f in range(f0 + 1, ap):
        w = float(smooth((f - f0) / nr)); tgt = Pa + rot(ax, np.radians(w * dl)) @ (P[f - f0] - Pa); qa = q[f][7 + np.array(arm)]
        st = q[f].copy(); st[7 + np.array(arm)] += prev
        bnd = (np.maximum(k.lo[arm], qa - REJECT["max_dq"]), np.minimum(k.hi[arm], qa + REJECT["max_dq"]))
        qf = EW.solve(k, q[f], [("site", k.palm_id[limb], tgt, 1.0)], var, [REJECT["ik_prior"]] * 5, bnd, REJECT["ik_iters"], start=st)
        prev = qf[7 + np.array(arm)] - qa; out[f] = qf; p = k.palm_at(qf, limb); res = max(res, float(np.linalg.norm(p - tgt))); reach = max(reach, -p[0])
        foot = max(foot, max(float(np.linalg.norm(a[0] - b[0])) for a, b in zip(k.feet_pose(qf), k.feet_pose(q[f]))))
        com = max(com, k.support_out(qf, q[f], floor) - k.support_out(q[f], q[f], floor)); glove = max(glove, k.glove_shoulder(qf, limb))
    info.update(path_m=round(res, 5), foot_m=round(foot, 6), com_m=round(com, 4), glove_m=round(glove, 4), reach_excess_m=float(reach - fwd_ap), apex_fwd_m=float(fwd_ap))
    over = speed_over(out[f0 - 1:ap + 2], q[f0 - 1:ap + 2], vmax); info["speed_joints"] = over
    why = [r for r, bad in (("path", res > REJECT["path_m"]), ("foot", foot > REJECT["foot_m"]), ("com", com > REJECT["com_m"]), ("speed", bool(over)),
                            ("glove", glove > REJECT["glove_m"]), ("reach", reach - fwd_ap > REJECT["reach_m"])) if bad]
    return (None, dict(info, reason=",".join(why))) if why else (out, info)

def reangle_row(k, q, src, sts, rng, vmax):
    out, mods, acc = q.copy(), [], []; n = int(round((PRIOR["reangle_s"] + PRIOR["reangle_ramp_s"]) * FPS))
    for i in rng.permutation(len(sts)):
        s = sts[i]; u = float(rng.random()); w = (s.apex - n, s.apex)
        if s.type not in ("hook", "straight", "uppercut"): mods.append(dict(limb=s.limb, type=s.type, onset=s.onset, apex=s.apex, status="skip", reason="type")); continue
        if any(l == s.limb and not (w[1] < a or w[0] > b) for l, a, b in acc): mods.append(dict(limb=s.limb, type=s.type, onset=s.onset, apex=s.apex, status="skip", reason="overlap")); continue
        q2, info = reangle_strike(k, out, s, u, vmax)
        if q2 is None: mods.append(dict(info, status="skip" if info["reason"] in ("seam", "vertical") else "reject")); continue
        out = q2; acc.append((s.limb, *w)); mods.append(dict(info, status="accept"))
    return out, src.copy(), mods

# ---- combo
def blend_aux(a, b, n):
    """blend()'s frame bookkeeping for a per-frame label: the dominant side (weight < 0.5: a)."""
    w = (np.arange(n) / n).reshape(-1, *[1] * (np.ndim(a) - 1)); return np.concatenate([a[:-n], np.where(w < 0.5, a[-n:], b[:n]), b[n:]])

def splice(q, aux, e, f, sq, sa, t):
    """Source strike t of (sq, sa) spliced after host frame e with fade f at both ends: (q', aux', the span of q' from host frame e
    to the host's resume frame r + f, the host window q[e:r + f + 1], the source segment)."""
    pad = int(round(PRIOR["combo_pad_s"] * FPS)); a, b, r = t.onset - pad - f, t.end + pad + f, e + 1 + f   # host fades out over [e + 1, r), resumes at r
    seg, sg = sq[a:b + 1], sa[a:b + 1]; new = blend(blend(q[:r], seg, f), q[r:], f); na = blend_aux(blend_aux(aux[:r], sg, f), aux[r:], f)
    return new, na, new[e:e + len(seg) + 2], q[e:r + f + 1], seg

def combo_row(host, hsts, srcs, rng, vmax):
    """host: (q, aux [T, 3] = src clip, src frame, src mirror), hsts: its hand strikes; srcs: {mirror: (q, aux, strikes)} of the
    source clip. Up to combo_max splices in the host's guard, each with the shortest fade in FADES that passes the speed rule
    (A6.1); (q', aux', mods)."""
    q, aux = host; T = len(q); pad = int(round(PRIOR["combo_pad_s"] * FPS)); gap = int(round(PRIOR["combo_gap_s"] * FPS)); plan, mods = [], []
    for i in rng.permutation(len(hsts)):
        s = hsts[i]; drawn = [round(float(rng.uniform(*PRIOR["combo_fade_s"])), 6) for _ in range(2)]; sm = int(rng.integers(2))   # M8's stream; the drawn fades are unused (A6.1)
        sq, sa, ss = srcs[sm]; j = int(rng.integers(len(ss))); t = ss[j]; e = s.end
        rec = dict(host_limb=s.limb, host_end=e, fade_drawn_s=drawn, src_mirror=sm, src_limb=t.limb, src_type=t.type, src_onset=t.onset, src_apex=t.apex, src_end=t.end)
        seam = [f for f in FADES if t.onset - pad - f >= 0 and t.end + pad + f < len(sq) and e + 1 + 2 * f <= T - 1]
        fit = [f for f in seam if not any(e < h.onset <= e + 1 + 2 * f or h.onset <= e < h.end for h in hsts if h is not s)]
        if len(plan) >= PRIOR["combo_max"]: mods.append(dict(rec, status="skip", reason="max")); continue
        if not seam: mods.append(dict(rec, status="skip", reason="seam")); continue
        if not fit: mods.append(dict(rec, status="skip", reason="guard")); continue
        if any(abs(e - p[0]) < gap for p in plan): mods.append(dict(rec, status="skip", reason="gap")); continue
        plan.append((e, fit, sq, sa, t, rec))
    for e, fit, sq, sa, t, rec in sorted(plan, key=lambda x: -x[0]):   # last first: earlier host indices stay valid
        tried = {}
        for f in fit:
            new, na, span, hw, seg = splice(q, aux, e, f, sq, sa, t); tried[f] = speed_over(span, [hw, seg], vmax)
            if not tried[f]: break
        if tried[f]: mods.append(dict(rec, status="reject", reason="speed", fades=fit, joints=sorted({j for o in tried.values() for j in o}))); continue
        q, aux = new, na; mods.append(dict(rec, status="accept", fade=f, fade_s=f / FPS, fades_failed=[x for x in tried if tried[x]], inserted_frames=len(seg) - 2 * f))
    return q, aux, mods

# ---- the library
def library(Q=None, clips=CLIPS):
    """[(name, family, host, mirror, draw, q, aux [F, 3], mods)], the per-base strike tables and vmax, over `clips` (LOCO: LOCO_CLIPS;
    generator seeds by the clip's index in CLIPS, so a LOCO draw is the L1 draw; no combo row whose source is outside `clips`)."""
    Q = Q or base_motion(); B, vmax = bases(Q, clips); k = Kin(); rows = []
    aux0 = {(c, m): np.stack([np.full(len(B[(c, m)]), CLIPS.index(c), float), np.arange(len(B[(c, m)]), dtype=float), np.full(len(B[(c, m)]), float(m))], 1) for c, m in B}
    ST = {key: [s for s in strikes(B[key], f"{key[0]}{'_m' if key[1] else ''}") if s.hand] for key in B}
    for fi, fam in enumerate(FAMILIES):
        for ci, c in ((CLIPS.index(c), c) for c in clips):
            if fam == "combo" and PRIOR["combo_source"][c] not in clips: continue
            for m in ((1,) if fam == "mirror" else (0, 1)):
                for dr in range(DRAWS[fam]):
                    rng = np.random.default_rng([SEED, fi, ci, m, dr]); q, aux = B[(c, m)], aux0[(c, m)]; name = f"l1_{fam}_{c}_{'m' if m else 'o'}{dr}"
                    if fam == "mirror": rows.append((name, fam, c, m, dr, q.copy(), aux.copy(), [])); continue
                    if fam == "warp":
                        q2, s2, mods = warp_row(q, aux[:, 1], ST[(c, m)], rng, vmax[m]); aux2 = np.stack([np.full(len(q2), ci, float), s2, np.full(len(q2), float(m))], 1)
                    elif fam == "reangle": q2, _, mods = reangle_row(k, q, aux[:, 1], ST[(c, m)], rng, vmax[m]); aux2 = aux.copy()
                    else:
                        sc = PRIOR["combo_source"][c]; srcs = {sm: (B[(sc, sm)], aux0[(sc, sm)], ST[(sc, sm)]) for sm in (0, 1)}
                        q2, aux2, mods = combo_row((q, aux), ST[(c, m)], srcs, rng, np.maximum(vmax[0], vmax[1]))
                    rows.append((name, fam, c, m, dr, q2, aux2, mods))
    return rows, ST, vmax

def reject_stats(rows):
    """Per family: strikes seen, accepted, rejected by reason, skipped by reason; re-angle and warp parameter summaries."""
    out = {}
    for name, fam, c, m, dr, q, aux, mods in rows:
        o = out.setdefault(fam, dict(rows=0, strikes=0, accept=0, reject={}, skip={}, frames=0)); o["rows"] += 1; o["frames"] += len(q)
        for x in mods:
            o["strikes"] += 1
            if x["status"] == "accept": o["accept"] += 1
            else:
                for r in x["reason"].split(","): o[x["status"]][r] = o[x["status"]].get(r, 0) + 1
    for fam, o in out.items():
        tried = o["accept"] + sum(x["status"] == "reject" for r in rows if r[1] == fam for x in r[7])
        o["tried"] = tried; o["reject_share"] = round(1 - o["accept"] / tried, 4) if tried else None; o["seconds"] = round(o["frames"] / FPS, 2)
    C = [x for r in rows if r[1] == "combo" for x in r[7] if x["status"] in ("accept", "reject")]
    if C: out["combo"]["fades_accepted_frames"] = sorted(x["fade"] for x in C if x["status"] == "accept"); out["combo"]["splices_surviving"] = sum(x["status"] == "accept" for x in C)
    W = [x for r in rows if r[1] == "warp" for x in r[7]]
    for st in ("accept", "reject"):
        sp = [x["speed"] for x in W if x["status"] == st]
        if sp: out["warp"][f"speed_{st}"] = dict(n=len(sp), mean=round(float(np.mean(sp)), 4), faster=int(sum(v > 1 for v in sp)), slower=int(sum(v < 1 for v in sp)))
    out["warp"]["speed_drawn_mean"] = round(float(np.mean([x["speed"] for x in W if x["status"] in ("accept", "reject")])), 4)
    R = [x for r in rows if r[1] == "reangle" for x in r[7] if "delta_deg" in x]
    for st in ("accept", "reject"):
        d = [x["delta_deg"] for x in R if x["status"] == st]
        if d: out["reangle"][f"delta_{st}"] = dict(n=len(d), mean=round(float(np.mean(d)), 3), mean_abs=round(float(np.mean(np.abs(d))), 3), up=int(sum(v > 0 for v in d)), down=int(sum(v < 0 for v in d)))
    return out

# ---- the T1 motion and what a tracker trained on it is shown
def t1_rows(rows):
    """The L1 rows as T1 learns them (A6.2): [(name, q, group, conj)]; the mirror family left out (conjugated, it is the originals),
    a mirrored-base row un-mirrored (the conjugated driver shows T1 mirror(row))."""
    return [(r[0], mirror(r[5]) if r[3] else r[5], r[1], bool(r[3])) for r in rows if r[1] != "mirror"]

def track_motion(Q, rows, clips=CLIPS):
    """export_motion's layout over the originals of `clips` then t1_rows(rows): (seq [N, 31], index {name: start, frames, group,
    conj}, blend frames)."""
    n = int(round(BLEND_S * FPS)); seq, index = None, {}
    for name, q, grp, cj in [(c, Q[c], "original", False) for c in clips] + t1_rows(rows):
        two = np.concatenate([q, q]); start = 0 if seq is None else len(seq) - n; seq = two if seq is None else blend(seq, two, n)
        index[name] = dict(start_frame=int(start), frames=int(len(q)), group=grp, conj=cj)
    return seq, index, n

def fed(seq, index, B, rows):
    """{row name: the reference its driver is fed}: shown() at the row's T1 entry; a conjugated row: mirror_ref of the entry it is
    conjugated to (its un-mirrored form, or the host original for the mirror family)."""
    at = {r[0]: (r[2] if r[1] == "mirror" else r[0]) for r in rows}; ref = shown(seq, index, B, sorted(set(at.values())))
    return {r[0]: (tuple(FM.mirror_ref(*ref[at[r[0]]])) if r[3] else ref[at[r[0]]]) for r in rows}

def track_frame(start, T, B, f):
    """Tracker.frame on that motion: the second copy for f <= B (the first copy's first B rows are cross-faded)."""
    f %= T; return start + (T + f if f <= B else f)

def shown(seq, index, B, names):
    """{name: (joint_pos, joint_vel, anchor_pos, anchor_quat) float32} at every frame of the row: the CSV (%.6f -> float32) through
    csv_to_npz's float32 interpolation grid at Tracker.frame (export_fb_motion.reference on the whole T1 motion)."""
    rows = np.char.mod("%.6f", seq).astype(np.float64).astype(f32); N = len(seq); r = _sub("r1"); d = mujoco.MjData(r); tor = r.body("torso_link").id; out = {}
    for nm in names:
        s0, T = index[nm]["start_frame"], index[nm]["frames"]
        def at(g):
            i0, w = EM.grid(g, N); i1 = min(int(i0) + 1, N - 1); assert s0 <= i0 and i1 < s0 + 2 * T - B, (nm, g, i0)   # its own two copies
            return (rows[i0] * (f32(1) - w) + rows[i1] * w).astype(f32), EM.slerp(rows[i0, 3:7], rows[i1, 3:7], w)
        jp, jv, ap, aq = [], [], [], []
        for f in range(T):
            g = track_frame(s0, T, B, f); x, rq = at(g); jp.append(x[7:]); jv.append(((at(g + 1)[0][7:] - at(g - 1)[0][7:]) * f32(1 / EM.DT) / f32(2)).astype(f32))
            d.qpos[:3] = x[:3]; d.qpos[3:7] = rq; d.qpos[7:] = x[7:]; mujoco.mj_kinematics(r, d); ap.append(d.xpos[tor].copy()); aq.append(d.xquat[tor].copy())
        out[nm] = (np.array(jp, f32), np.array(jv, f32), np.array(ap).astype(f32), np.array(aq).astype(f32))
    return out

# ---- replays of a row in the eval's world (clean starts here; the tracker-alone runs in scripts/fb_m8_check.py)
def replay(m, name, q0, distance, prefix="opp_"):
    """A FightReplay of in-place row q0 at `distance`, built as FightReplay.__init__ builds an r1 one; no tracker (placement only)."""
    from r1gym import opponent as O
    r = O.FightReplay.__new__(O.FightReplay); q = q0.copy(); q[:, 0] += distance
    r.fps, r.speed, r.clip, r.raw, r.distance, r.root, r.kind, r.model = float(FPS), 1.0, name, None, distance, "inplace", "r1", m
    r.q, r.T = q, len(q); j = m.joint(prefix + "floating_base_joint"); r.qadr, r.vadr, r.nq = m.jnt_qposadr[j.id], m.jnt_dofadr[j.id], q.shape[1]
    r.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(prefix)}
    sub = _sub("r1"); r.act = np.array([m.actuator(prefix + sub.joint(i).name.replace("_joint", "")).id for i in range(1, sub.njnt)])
    r.pel = m.body(prefix + "pelvis").id; r.mass = float(m.body_subtreemass[r.pel]); r.torso = m.body(prefix + "torso_link").id
    r.tracker, r.puppet = None, False; return r

def clean_masks(named_q, distances=EM.DISTANCES):
    """scenario.clean_starts (the eval's call and config, as export_fb_motion) on rows {name: q0}: [len, 31, 300] bool."""
    import r1gym.scenario as SC
    from r1gym.model import load_cfg
    real = SC.FightReplay; SC.FightReplay = lambda m, clip, distance, cfg: replay(m, clip, named_q[clip], distance)
    try: good = SC.clean_starts({**load_cfg(), "opponent_aim": "none"}, list(named_q), list(distances), SC.REF_POSES)
    finally: SC.FightReplay = real
    return np.array([[good[(c, d)] for d in distances] for c in named_q])

def _clean_job(a): return clean_masks(*a)

def names_for(loco):
    """Output file names: (npz, track csv, track index, manifest)."""
    x = "_loco" if loco else ""; return f"fb_l1{x}.npz", f"fb_l1_track{x}.csv", f"fb_l1_track{x}_index.json", f"fb_l1{x}_manifest.json"

def disclosures(st, loco):
    w = st["warp"]; a, r = w.get("speed_accept", {}), w.get("speed_reject", {})
    return [f"Warp speed skew (A6.4): the warp prior U(0.7, 1.3) is symmetric (drawn mean {w['speed_drawn_mean']}), but the accepted warps skew slow: "
            f"mean speed {a.get('mean')} ({a.get('faster')} faster, {a.get('slower')} slower of {a.get('n')}); of {r.get('n', 0)} rejected, {r.get('faster', 0)} are speed-ups "
            f"(mean {r.get('mean')}). This is a structural consequence of export_warps' speed rule (|dq| <= max(p99, the window's own fastest) + 0.05 rad/s): a strike sped "
            "up by s passes only where the clip set's p99 is above s x its own fastest joint speed, so the accepted set is slow-biased although the prior is not.",
            "Mirror conjugation (A6.2): rows with mirror = 1 are performed by the mirror-conjugated tracker (gpu/fb_mirror.py); the T1 motion holds them un-mirrored "
            "and leaves out the mirror family.",
            "Combos (A6.1): the stated speed rule (allowance max(p99, the host window's own fastest, the source segment's own fastest) + 0.05 rad/s, the join never "
            f"counted) with the shortest passing fade in the prior range [0.15, 0.35] s; {st.get('combo', {}).get('splices_surviving', 0)} of {st.get('combo', {}).get('tried', 0)} "
            "tried splices survive.",
            f"vmax (p99 joint speed) is computed per clip set (A6.3): {'kfab_200/201 (LOCO)' if loco else 'kfab_200/201/203'}."]

def export(out=OUT, clean=True, workers=2, loco=False):
    assert set(CLIPS) == set(TRAIN_CLIPS) == {"kfab_200", "kfab_201", "kfab_203"}, "L1 is built from the train clips only"
    clips = LOCO_CLIPS if loco else CLIPS; f_npz, f_csv, f_ix, f_man = names_for(loco)
    out = Path(out); out.mkdir(parents=True, exist_ok=True); Q = base_motion(); rows, ST, vmax = library(Q, clips)
    for r in rows: assert {CLIPS[int(i)] for i in np.unique(r[6][:, 0])} <= set(clips), r[0]
    seq, index, B = track_motion(Q, rows, clips); names = [r[0] for r in rows]; ref = fed(seq, index, B, rows)
    L = [(r, strikes(r[5], r[0])) for r in rows]; sk = [(i, s) for i, (r, ss) in enumerate(L) for s in ss]
    A = dict(names=np.array(names), family=np.array([r[1] for r in rows]), host=np.array([r[2] for r in rows]), mirror=np.array([r[3] for r in rows], np.int8),
             draw=np.array([r[4] for r in rows], np.int8), frames=np.array([len(r[5]) for r in rows], np.int32), fps=np.float64(FPS),
             row0=np.cumsum([0] + [len(r[5]) for r in rows[:-1]]).astype(np.int32), q=np.concatenate([r[5] for r in rows]),
             src_clip=np.concatenate([r[6][:, 0] for r in rows]).astype(np.int8), src_frame=np.concatenate([r[6][:, 1] for r in rows]),
             src_mirror=np.concatenate([r[6][:, 2] for r in rows]).astype(np.int8), clips=np.array(CLIPS), conj=np.array([r[3] for r in rows], np.int8),
             **{k: np.concatenate([ref[n][i] for n in names]) for i, k in enumerate(("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"))},
             strike_row=np.array([i for i, _ in sk], np.int32), strike_limb=np.array([s.limb for _, s in sk]), strike_type=np.array([s.type for _, s in sk]),
             strike_band=np.array([s.band for _, s in sk]), distances=np.array(EM.DISTANCES))
    for kk in EM.STRIKE_NUM: A["strike_" + kk] = np.array([getattr(s, kk) for _, s in sk], np.int32 if kk in ("onset", "apex", "end") else np.float64)
    if clean:
        jobs = [({r[0]: r[5]},) for r in rows]
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(workers) as pool: cm = pool.map(_clean_job, jobs, chunksize=1)
        else: cm = [_clean_job(j) for j in jobs]
        A["clean"] = np.concatenate(cm)
    EM.save_npz(out / f_npz, **A)
    csv = np.concatenate([seq[:, :3], seq[:, [4, 5, 6, 3]], seq[:, 7:]], 1); np.savetxt(out / f_csv, csv, delimiter=",", fmt="%.6f")
    json.dump(dict(fps=FPS, joints=list(CTRL_JOINTS), frames=int(len(seq)), blend_frames=B, loco_drop="kfab_203" if loco else None, rows=index), open(out / f_ix, "w", encoding="utf-8"), indent=1)
    rowinfo = []
    for (name, fam, c, m, dr, q, aux, mods), (_, ss) in zip(rows, L):
        src = {CLIPS[int(i)]: [int(np.floor(aux[aux[:, 0] == i, 1].min())), int(np.ceil(aux[aux[:, 0] == i, 1].max()))] for i in np.unique(aux[:, 0])}
        rowinfo.append(dict(name=name, family=fam, host=c, mirror=m, conj=bool(m), draw=dr, frames=len(q), sources=src, q_sha256=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
                            strikes=len(ss), hand_strikes=int(sum(s.hand for s in ss)), mods=mods))
    outs = {f: EM.sha256(out / f) for f in (f_npz, f_csv, f_ix)}; st = reject_stats(rows)
    man = dict(clips=list(clips), loco=bool(loco), outputs=outs, sources={s: EM.sha256(ROOT / s) for s in sorted(set(SOURCES))}, git=EM.git_hash(), versions=EM.versions(),
               seed=SEED, families=list(FAMILIES), draws=DRAWS, prior=PRIOR, combo_fades=list(FADES), reject=REJECT, range_tol=RANGE_TOL, blend_s=BLEND_S, clean_starts=bool(clean),
               vmax={str(m): [round(float(x), 9) for x in v] for m, v in vmax.items()}, disclosures=disclosures(st, loco),
               rows=rowinfo, reject_stats=st, track_motion=dict(frames=int(len(seq)), rows=len(index), originals=list(clips), conjugated=[n for n, x in index.items() if x["conj"]]))
    json.dump(man, open(out / f_man, "w", encoding="utf-8"), indent=1); return A, man

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--no-clean", action="store_true"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--loco", action="store_true"); a = ap.parse_args(); assert a.workers <= 2
    A, man = export(a.out, not a.no_clean, a.workers, a.loco); st = man["reject_stats"]
    print(f"L1{' LOCO' if a.loco else ''}: {len(A['names'])} rows, {len(A['q'])} frames ({len(A['q']) / FPS:.1f} s); " + "; ".join(f"{f} {st[f]['rows']} rows {st[f]['seconds']} s accept {st[f]['accept']}/{st[f]['tried']}" for f in FAMILIES if f in st)
          + f"; sha256 {man['outputs'][names_for(a.loco)[0]][:16]}")
