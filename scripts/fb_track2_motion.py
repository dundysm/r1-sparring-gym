"""Track 2 motion (M2V2_PLAN_v3 6.2 item 3, T2-3): a KungFuAthlete org_smooth_mj clip -> the motion the eval's opponent would
perform, with the repo's own pipeline, and the per-clip eligibility screen (>= 4 s kept, >= 2 head-band hand strikes, stay-up
>= 90% for the frozen T0 tracker alone in the CPU world).
What the files are (checked on train-pool clips; load() asserts it on every clip): fps [1] int64 = 50; joint_pos [T, 36] = the
  G1 29-DoF qpos (root xyz, unit quaternion wxyz, the 29 hinges in r1gym's G1 model order; joint_names asserted); joint_vel
  [T, 35]; body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w [T, 53, *] of the G1 (world first). r1gym's G1 FK of
  joint_pos reproduces body_pos_w / body_quat_w (fk_err). The repo's kfab_*.npz hold the same qpos as 'qpos' (non-unit
  quaternions there), so the only conversion is joint_pos -> qpos; the G1 -> R1 retarget is the repo's retarget_g1_to_r1.
process()   FightReplay.__init__'s chain on a raw R1 qpos: clean_clip, tracker cuts, style cuts (scripts/style_cuts.py's rule on
            that stage), ground, _canon (in place, facing -x), _seamless 0.3 s. Also returns each frame's index into the cleaned
            clip (an extra column carried through the joins), so a tracker fall maps to the cleaned frame a cut is written in.
reference() what the tracker is shown at clip frame f: joint pos, joint vel (central difference over the looping clip, as a clip
            sits twice in the tracker's motion), torso_link position and quaternion by FK at distance 0; float32.
alone()     scripts/eval_tracker.run verbatim (R1 at home, opponent 4 m away facing it, 10 s, no aim, fall: pelvis up-z < 0.5 or
            z < 0.25) with the opponent a FightReplay of this motion driven by fb_opponent's TrackerDriver (model_13749.pt, float32
            as deployed) on a one-clip Motion: the M2 G0-3b set-up. Seed-free: eval_tracker.run fixes env seed 0, and that RNG
            only feeds the defender's observation noise, which its zero actions never read (tests/test_fb_track2.py).
screen()    the pre-declared procedure (RULE): process; kept < 4 s or < 2 head-band strikes -> stop (ineligible); else 24 starts;
            while stay-up < 0.9 and cut rounds < 3: cut the fall cluster with most falls ([first - 0.8 s, last + 0.1 s] in
            cleaned frames, falls <= 1 s apart are one cluster), re-process, re-check kept and strikes, re-run the 24 starts.
            Eligible on the last round: kept >= 4 s, stay-up >= 0.9 and >= 2 head-band strikes, scored twice: hand strikes
            (eligible_hand; recon_data 5.3 and the val2 unit) and any limb (eligible_any; the plan's 6.2 wording).
Pure functions of the clip; nothing is written. The sealed reduction lives in fb_track2_eligibility.py."""
import os, sys, traceback
from pathlib import Path
from types import SimpleNamespace
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_k, "1")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import numpy as np, mujoco, torch
from r1gym import opponent as O, strikes as S

DATA = ROOT.parent / "kfab_data"
FPS = 50
KEYS = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names"}
BLEND_S = 0.3                                     # FightReplay blend_s (the loop seam)
IDLE_SPEED, IDLE_S, POSE_DZ, POSE_S = 0.6, 0.4, 0.25, 0.3   # scripts/style_cuts.py (asserted equal in tests)
RULE = dict(kept_min_s=4.0, head_min=2, stay_up_min=0.9, starts=24, seconds=10.0, alone_distance_m=4.0, tracker="model_13749.pt float32",
            cut_pre_s=0.8, cut_post_s=0.1, cluster_gap_s=1.0, max_cut_rounds=3, head_band=S.RULE["bands_m"]["head"])

def clip_name(i): return f"kfab_{int(i)}"

def load(i, data=DATA, fk=False):
    """(G1 qpos [T, 36] float64, fps) of clip id i, format asserted; fk=True also returns the FK error vs body_pos_w / body_quat_w."""
    g = O._sub("g1"); names = [g.joint(j).name for j in range(1, g.njnt)]
    with np.load(Path(data) / "npz" / f"{int(i)}.npz", allow_pickle=False) as z:
        assert set(z.files) == KEYS, f"clip {i}: keys {sorted(z.files)}"
        A = {k: z[k] for k in z.files}
    assert A["fps"].shape == (1,) and int(A["fps"][0]) == FPS, f"clip {i}: fps"
    assert A["joint_names"].tolist() == names, f"clip {i}: joint order"
    q = A["joint_pos"].astype(np.float64); T = len(q)
    assert q.shape == (T, g.nq) and T > 2 and np.isfinite(q).all(), f"clip {i}: joint_pos"
    assert A["body_pos_w"].shape == (T, len(A["body_names"]), 3) and A["body_quat_w"].shape == (T, len(A["body_names"]), 4)
    if not fk: return q, FPS
    d = mujoco.MjData(g); bn = A["body_names"].tolist(); ix = [(g.body(b).id, j) for j, b in enumerate(bn) if b != "world" and mujoco.mj_name2id(g, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
    ep = eq = 0.0
    for t in range(T):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(g, d)
        for b, j in ix:
            ep = max(ep, np.abs(d.xpos[b] - A["body_pos_w"][t, j]).max()); eq = max(eq, min(np.abs(d.xquat[b] - A["body_quat_w"][t, j]).max(), np.abs(d.xquat[b] + A["body_quat_w"][t, j]).max()))
    return q, FPS, dict(bodies=len(ix), pos=float(ep), quat=float(eq))

def retarget(q): return O.retarget_g1_to_r1(q)

def style_ranges(q, fps=FPS):
    """scripts/style_cuts.ranges on a given q (cleaned, tracker-cut, before grounding): [(a, b, kind)]."""
    qi = q.copy(); qi[:, :2] -= O._lowpass(q[:, :2], int(round(fps)))
    r = O._sub("r1"); d = mujoco.MjData(r); ee = [r.body(n).id for n in ("left_wrist_roll_link", "right_wrist_roll_link", "left_ankle_roll_link", "right_ankle_roll_link")]
    tor = r.body("torso_link").id; P = np.zeros((len(q), 4, 3)); T = np.zeros((len(q), 3))
    for t in range(len(q)):
        d.qpos[:] = qi[t]; mujoco.mj_kinematics(r, d); P[t] = d.xpos[ee]; T[t] = d.xpos[tor]
    speed = np.linalg.norm(np.gradient(P, 1 / fps, axis=0), axis=2).max(1)
    idle = np.convolve(speed < IDLE_SPEED, np.ones(5) / 5, "same") > 0.99
    pose = ((P[:, :2, 2] - T[:, None, 2]) > POSE_DZ).all(1)
    return sorted([(a, b, "idle") for a, b in O._runs(idle) if b - a >= IDLE_S * fps] + [(a, b, "posing") for a, b in O._runs(pose) if b - a >= POSE_S * fps])

class NoMotion(Exception): pass

def _kept(f, *a):
    """f(*a) of clean_clip / apply_cuts; NoMotion when no piece is long enough to keep (their _join then indexes an empty list)."""
    try: return f(*a)
    except IndexError as e:
        if traceback.extract_tb(e.__traceback__)[-1].name == "_join": raise NoMotion(f.__name__) from None
        raise

def process(qr, fps=FPS, tcuts=(), style=None, distance=0.0):
    """FightReplay(...).q for a raw R1 qpos qr [T, 31] with tracker cuts `tcuts` (cleaned-clip frames) and style cuts `style` (None:
    computed by the style_cuts rule; else a list of ranges, as style_cuts.json). Returns (q [F, 31], src [F] float: cleaned-clip
    frame of each row, info). Nothing kept by the cleaning or a cut (no piece >= its minimum): F = 0, kept 0 s."""
    try:
        qc, rep = _kept(O.clean_clip, qr, fps)
        ext = _kept(O.apply_cuts, np.hstack([qc, np.arange(len(qc), dtype=float)[:, None]]), fps, [list(c) for c in tcuts])
        sr = [[a, b] for a, b, _ in style_ranges(ext[:, :-1], fps)] if style is None else [list(c) for c in style]
        ext = _kept(O.apply_cuts, ext, fps, sr, O.STYLE_MIN_KEEP_S)
    except NoMotion as e:
        return np.zeros((0, qr.shape[1])), np.zeros(0), dict(clean=None, frames_clean=0, style_cuts=[], tracker_cuts=[list(map(int, c)) for c in tcuts], frames=0, kept_s=0.0, empty_at=str(e))
    ext = np.hstack([O.ground(ext[:, :-1], fps), ext[:, -1:]])
    ext = O.FightReplay._canon(SimpleNamespace(root="inplace", distance=distance, fps=float(fps)), ext)
    ext = O._seamless(ext, int(round(BLEND_S * fps)))
    return ext[:, :-1], ext[:, -1], dict(clean=rep, frames_clean=len(qc), style_cuts=sr, tracker_cuts=[list(map(int, c)) for c in tcuts], frames=len(ext), kept_s=len(ext) / fps)

def reference(q0, fps=FPS):
    """(joint_pos, joint_vel, anchor_pos, anchor_quat) float32 per frame of an in-place looping clip q0 (distance 0)."""
    r = O._sub("r1"); d = mujoco.MjData(r); tor = r.body("torso_link").id; ap, aq = np.zeros((len(q0), 3)), np.zeros((len(q0), 4))
    for t in range(len(q0)): d.qpos[:] = q0[t]; mujoco.mj_kinematics(r, d); ap[t] = d.xpos[tor]; aq[t] = d.xquat[tor]
    jv = (np.roll(q0[:, 7:], -1, 0) - np.roll(q0[:, 7:], 1, 0)) * fps / 2
    return tuple(x.astype(np.float32) for x in (q0[:, 7:], jv, ap, aq))

def motion(clip, q0, fps=FPS):
    """fb_opponent.Motion of one clip (index 0): q0 for placement (a bout adds its distance), the reference above."""
    import fb_opponent as FO
    M = FO.Motion.__new__(FO.Motion); ref = reference(q0, fps); t = lambda x, dt=None: torch.as_tensor(x, dtype=dt)
    M.clips, M.frames, M.row0, M.q = (clip,), t([len(q0)], torch.long), t([0], torch.long), t(q0, torch.float64)
    M.ref_q, M.ref_dq, M.ref_apos, M.ref_aquat = (t(x) for x in ref); M.clean = M.distances = None; M.strikes = {}; M.device = "cpu"
    return M

def replay(m, clip, q0, distance, fps=FPS, prefix="opp_"):
    """A FightReplay (r1, policy opponent) of in-place clip q0 placed at `distance`, built as FightReplay.__init__ builds one."""
    r = O.FightReplay.__new__(O.FightReplay); q = q0.copy(); q[:, 0] += distance
    r.fps, r.speed, r.clip, r.raw, r.distance, r.root, r.kind, r.model = float(fps), 1.0, clip, None, distance, "inplace", "r1", m
    r.q, r.T = q, len(q); j = m.joint(prefix + "floating_base_joint"); r.qadr, r.vadr, r.nq = m.jnt_qposadr[j.id], m.jnt_dofadr[j.id], q.shape[1]
    r.bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(prefix)}
    sub = O._sub("r1"); hinges = [sub.joint(i).name for i in range(1, sub.njnt)]
    r.act = np.array([m.actuator(prefix + h.replace("_joint", "")).id for h in hinges])
    r.pel = m.body(prefix + "pelvis").id; r.mass = float(m.body_subtreemass[r.pel]); r.torso = m.body(prefix + "torso_link").id
    r.tracker, r.puppet = None, False; return r

_W = {}
def _world():
    """The eval's tracker-alone world (cached per process): eval config, policy opponent, no aim."""
    if "m" not in _W:
        from r1gym.model import build_spec, load_cfg
        cfg = load_cfg(); assert cfg.get("opponent_aim", "none") == "none" and cfg["opponent_mode"] == "policy" and cfg.get("opponent_model", "r1") == "r1"
        _W.update(cfg=cfg, m=build_spec(opponent=True, cfg=cfg).compile())
    return _W["m"], _W["cfg"]

def alone(clip, q0, t0, seconds=RULE["seconds"], fps=FPS, dtype=torch.float32, full=False):
    """Fall time (s) or None: eval_tracker.run with this clip's FightReplay and the T0 driver (full: its (fell, joint err, z err))."""
    import eval_tracker as ET, fb_opponent as FO
    torch.set_num_threads(1); m, cfg = _world(); M = motion(clip, q0, fps)
    def make(m_, clip_, distance, cfg):
        assert clip_ == clip and m_ is m
        r = replay(m_, clip, q0, distance, fps); tr = FO.MjTracker.__new__(FO.MjTracker); tr.replay, tr.k, tr.drv = r, 0, FO.TrackerDriver(1, dtype, motion=M)
        return tr.install().replay
    ET.FightReplay = make
    try: res = ET.run(m, cfg, clip, t0, seconds)
    finally: ET.FightReplay = O.FightReplay   # a pool worker runs other jobs next
    return res if full else res[0]

def head_hands(q0, fps=FPS, clip=""):
    L = S.detect(q0, fps, clip=clip, split="track2")
    return L, sum(s.hand and s.band == "head" for s in L)

def clusters(frames, gap):
    """Sorted frames grouped where consecutive ones are <= gap apart: [[...], ...]."""
    out = []
    for f in sorted(frames):
        if out and f - out[-1][-1] <= gap: out[-1].append(f)
        else: out.append([f])
    return out

def merge(cuts):
    out = []
    for a, b in sorted(cuts):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def stay_up(clip, q0, fps=FPS, run=alone, pool=None):
    """Fall times of the RULE starts (t0 = s x T / starts), clip seconds T of q0."""
    T = len(q0) / fps; jobs = [(clip, q0, s * T / RULE["starts"]) for s in range(RULE["starts"])]
    return list(pool.starmap(run, jobs)) if pool is not None else [run(*j) for j in jobs]

def screen(i, data=DATA, qr=None, run=alone, pool=None):
    """The full record of clip id i under RULE (see the module doc). qr: a precomputed retarget (train pool only). Both readings of
    the strike bar are scored on the same rounds: 'hand' (head-band hand strikes, the val2 unit) and 'any' (head-band strikes, any
    limb); the starts run when kept and the looser 'any' bar hold."""
    clip = clip_name(i); fps = FPS; k = RULE["head_min"]
    if qr is None: q, fps = load(i, data); qr = retarget(q)
    rec = dict(clip=clip, rounds=[]); cuts = []
    for rnd_i in range(RULE["max_cut_rounds"] + 1):
        q0, src, info = process(qr, fps, cuts); L, nh = head_hands(q0, fps, clip) if len(q0) else ([], 0)
        rnd = dict(cuts=[list(c) for c in cuts], kept_s=round(info["kept_s"], 3), frames=info["frames"], head_hands=int(nh), hands=int(sum(s.hand for s in L)),
                   head_any=int(sum(s.band == "head" for s in L)), strikes=len(L)); rec["rounds"].append(rnd)
        rec.update(info=dict(frames_clean=info["frames_clean"], clean=info["clean"], style_cuts=info["style_cuts"]), q0=q0, strikes=L)
        if info["kept_s"] < RULE["kept_min_s"] or rnd["head_any"] < k: break
        fell = stay_up(clip, q0, fps, run, pool); up = sum(f is None for f in fell) / len(fell); rnd.update(fell=fell, stay_up=up)
        if up >= RULE["stay_up_min"] or rnd_i == RULE["max_cut_rounds"]: break
        T = len(q0); t0s = [s * T / fps / RULE["starts"] for s in range(RULE["starts"])]
        ff = [int(round(src[int(round((t0 + f) * fps)) % T])) for t0, f in zip(t0s, fell) if f is not None]
        c = max(clusters(ff, RULE["cluster_gap_s"] * fps), key=lambda c: (len(c), -c[0]))
        cuts = merge(cuts + [[max(0, c[0] - int(round(RULE["cut_pre_s"] * fps))), c[-1] + int(round(RULE["cut_post_s"] * fps)) + 1]])
    last = rec["rounds"][-1]; up = last.get("stay_up")
    rec.update(kept_ok=last["kept_s"] >= RULE["kept_min_s"], hand_ok=last["head_hands"] >= k, any_ok=last["head_any"] >= k, stay_up=up,
               stay_up_ok=up is not None and up >= RULE["stay_up_min"], head_hands=last["head_hands"], head_any=last["head_any"])
    rec["eligible_hand"] = bool(rec["kept_ok"] and rec["hand_ok"] and rec["stay_up_ok"]); rec["eligible_any"] = bool(rec["kept_ok"] and rec["any_ok"] and rec["stay_up_ok"])
    return rec
