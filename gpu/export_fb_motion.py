"""Train-only opponent motion for the full-body M2 v2 task (M2V2_PLAN_v3 M1; design_learned-tracker 3.2), exported on CPU.
Only TRAIN_CLIPS (kfab_200/201/203) enter the outputs. The clean masks come from the eval's own clean_starts call, whose
FightReplay (opponent_mode policy) loads the tracker ONNX (every clip's reference) for Tracker.check on the train clip; nothing
of it is used (a bout's reset places the opponent kinematically). No other file holding a held-out clip's motion is read.

fb_motion.npz   clips [3] str, frames [3], row0 [3] (first row of each clip below), fps 50.0;
  q [F, 31] float64        FightReplay(m, clip, distance=0).q (strikes.motion): root and joints of the in-place clip, facing -x.
                           A bout at distance d places the opponent at q + (d, 0) in xy (FightReplay._canon adds it last).
  ref_joint_pos, ref_joint_vel [F, 24], ref_anchor_pos [F, 3], ref_anchor_quat [F, 4] (wxyz) float32: what the tracker is
                           shown at clip frame f (Tracker.reference at Tracker.frame(clip, f)), rebuilt from the clip through
                           export_motion.py's CSV (6 decimals) and csv_to_npz's float32 pipeline on the GPU: time t = f32(i) *
                           f32(0.02), phase t * f32(1 / duration) * f32(n - 1), lerp, slerp, gradient (x[i+1] - x[i-1]) * 50 / 2,
                           kinematics. i is the frame's row in the tracker's motion (motion_index.json: the train rows and the
                           length only), so the float32 rounding is the one the tracker trained on. A neighbouring row
                           cross-faded with a train clip is rebuilt from it; one cross-faded with any other clip is taken
                           unblended (it would enter only through an interpolation weight; check_tref measures the effect).
  clean [3, 31, 300] bool  scenario.clean_starts with the eval's config and REF_POSES (eval_opponent.py) at distances 0.35-0.65 m
                           in 0.01 m steps (distances [31]) on the 0.02 s start grid of the first 6 s.
  strike_*  [38]           strikes.json train rows (37 hand strikes + 1 kick): clip index, limb, type, band, frames, rise, speeds,
                           apex and approach (heading frame), torso height and the unwarped reaches.
fb_motion_manifest.json    clip set, sha256 of the output and of every source file, git hash, rules.
Run in the repo: python gpu/export_fb_motion.py [--out gpu/motions/fb]"""
import sys, io, json, hashlib, zipfile, argparse, subprocess
from pathlib import Path
GPU = Path(__file__).resolve().parent; ROOT = GPU.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(GPU))
import numpy as np, mujoco
from r1gym import strikes as S
from r1gym.env import TRAIN_CLIPS
from r1gym.model import load_cfg
from r1gym.opponent import _sub
from r1gym.scenario import clean_starts, REF_POSES, START_STEP, START_SPAN
from export_motion import blend

CLIPS = ("kfab_200", "kfab_201", "kfab_203")
INDEX = ROOT / "results" / "opponent_tracker" / "motion_index.json"   # the tracker's motion layout
DISTANCES = tuple(round(0.35 + 0.01 * i, 2) for i in range(31))
FPS, DT = 50, 0.02
LIMBS = ("l_hand", "r_hand", "l_foot", "r_foot")
STRIKE_NUM = ("onset", "apex", "end", "fps", "rise_s", "peak_fwd_mps", "peak_speed_mps", "apex_m", "approach", "torso_z_m", "reach_head_m", "reach_torso_m")
SOURCES = [f"r1gym/assets/motions/{n}" for c in CLIPS for n in (f"{c}.npz", f"r1_{c}.npy")] + \
          ["r1gym/assets/motions/tracker_cuts.json", "r1gym/assets/motions/style_cuts.json", "r1gym/assets/motions/strikes.json", "r1gym/assets/r1/r1.xml",
           "configs/r1_assumptions.yaml", "results/opponent_tracker/motion_index.json", "r1gym/opponent.py", "r1gym/scenario.py", "r1gym/strikes.py",
           "r1gym/model.py", "r1gym/env.py", "gpu/export_motion.py", "gpu/export_fb_motion.py"]
f32 = np.float32

def save_npz(path, **arrays):
    """np.savez with fixed zip timestamps: the bytes, and so the manifest's sha256, depend only on the arrays."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for k, v in arrays.items():
            b = io.BytesIO(); np.lib.format.write_array(b, np.asanyarray(v), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0)), b.getvalue())

TEXT = (".py", ".json", ".yaml", ".yml", ".xml", ".md", ".sh", ".txt", ".csv")
def sha256(p):
    """sha256 of a file; text LF-normalised (eval_opponent.text_sha16), so a CRLF checkout (autocrlf) and the pod's LF one agree."""
    b = Path(p).read_bytes(); return hashlib.sha256(b.replace(b"\r\n", b"\n") if Path(p).suffix.lower() in TEXT else b).hexdigest()

def versions(): return dict(mujoco=mujoco.__version__, numpy=np.__version__, python=sys.version.split()[0])   # hulls are MuJoCo's qhull

def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        return h + ("+dirty" if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip() else "")
    except Exception: return "unknown"

def layout():
    """{clip: (first motion row, frames)} for the train clips, the motion's length and its cross-fade length."""
    ix = json.load(open(INDEX, encoding="utf-8")); assert ix["fps"] == FPS
    return {c: (int(ix["clips"][c]["start_frame"]), int(ix["clips"][c]["frames"])) for c in CLIPS}, int(ix["frames"]), int(ix["blend_frames"])

def onnx_row(lay, B, c, f):
    """Tracker.frame: the motion row shown at clip frame f (second copy for f <= B), asserted inside the clip's own rows."""
    s0, T = lay[c]; f %= T; g = s0 + (T + f if f <= B else f); assert s0 <= g < s0 + 2 * T, (c, f, g); return g

def csv_rows(c, Q, lay, B):
    """The CSV rows of clip c as csv_to_npz reads them (6 decimals -> float32), local i = motion row start + i, 0 <= i < 2T."""
    s0, T = lay[c]; two = np.concatenate([Q[c], Q[c]])
    prev = [p for p in lay if lay[p][0] + 2 * lay[p][1] - B == s0]   # a train clip right before it in the motion
    if prev: qp = Q[prev[0]]; two = blend(np.concatenate([qp, qp]), two, B)[2 * len(qp) - B:]
    assert len(two) == 2 * T
    return np.char.mod("%.6f", two).astype(np.float64).astype(f32)

def grid(g, n):
    """csv_to_npz's interpolation (CUDA float32) at output frame g of an n-row motion: source row i0 and weight w."""
    t = (np.asarray(g).astype(f32) * f32(DT)).astype(f32); p = ((t * f32(1.0 / ((n - 1) * DT))).astype(f32) * f32(n - 1)).astype(f32)
    i0 = np.floor(p).astype(np.int64); return i0, (p - i0).astype(f32)

def slerp(a, b, tau):
    """mjlab quat_slerp (wxyz) on float32 inputs."""
    a, b, tau = a.astype(np.float64), b.astype(np.float64), float(tau)
    if tau == 0.0: return a
    if tau == 1.0: return b
    d = float(np.dot(a.astype(f32), b.astype(f32))); eps = float(np.finfo(f32).eps) * 4
    if abs(abs(d) - 1) < eps: return a
    if d < 0: d, b = -d, -b
    ang = np.arccos(np.clip(d, -1, 1))
    if abs(ang) < eps: return a
    return (a * np.sin((1 - tau) * ang) + b * np.sin(tau * ang)) / np.sin(ang)

def reference(c, Q, lay, n, B):
    """(joint_pos, joint_vel, anchor_pos, anchor_quat) float32 per clip frame: Tracker.reference's arrays at Tracker.frame."""
    s0, T = lay[c]; rows = csv_rows(c, Q, lay, B); r = _sub("r1"); d = mujoco.MjData(r); tor = r.body("torso_link").id
    def at(g):   # lerp / slerp of the rows around output frame g
        i0, w = grid(g, n); i1 = min(int(i0) + 1, n - 1); a, b = int(i0) - s0, i1 - s0
        assert 0 <= a and b < 2 * T - B, (c, g, a, b)   # never a row cross-faded into the next clip
        x = (rows[a] * (f32(1) - w) + rows[b] * w).astype(f32); return x, slerp(rows[a, 3:7], rows[b, 3:7], w)
    jp, jv, ap, aq = [], [], [], []
    for f in range(T):
        g = onnx_row(lay, B, c, f); x, rq = at(g)
        jp.append(x[7:]); jv.append(((at(g + 1)[0][7:] - at(g - 1)[0][7:]) * f32(1 / DT) / f32(2)).astype(f32))
        d.qpos[:3] = x[:3]; d.qpos[3:7] = rq; d.qpos[7:] = x[7:]; mujoco.mj_kinematics(r, d)
        ap.append(d.xpos[tor].copy()); aq.append(d.xquat[tor].copy())
    return np.array(jp, f32), np.array(jv, f32), np.array(ap).astype(f32), np.array(aq).astype(f32)

def strike_table():
    L = S.load(split="train")
    assert {s.clip for s in L} == set(CLIPS) and len(L) == 38 and sum(s.hand for s in L) == 37, "strikes.json train rows changed"
    out = dict(strike_clip=np.array([CLIPS.index(s.clip) for s in L], np.int32), strike_limb=np.array([s.limb for s in L]),
               strike_type=np.array([s.type for s in L]), strike_band=np.array([s.band for s in L]))
    for k in STRIKE_NUM: out["strike_" + k] = np.array([getattr(s, k) for s in L], np.int32 if k in ("onset", "apex", "end") else np.float64)
    return out

def export(out=GPU / "motions" / "fb", clean=True):
    assert set(CLIPS) == set(TRAIN_CLIPS) == {"kfab_200", "kfab_201", "kfab_203"}, "the full-body task trains on the train clips only"
    out = Path(out); out.mkdir(parents=True, exist_ok=True); lay, n, B = layout()
    Q = {c: S.motion(c)[0].copy() for c in CLIPS}
    for c in CLIPS: assert S.motion(c)[1] == FPS and len(Q[c]) == lay[c][1] == S.clip_frames(c), c
    ref = [reference(c, Q, lay, n, B) for c in CLIPS]
    A = dict(clips=np.array(CLIPS), frames=np.array([len(Q[c]) for c in CLIPS], np.int32), row0=np.cumsum([0] + [len(Q[c]) for c in CLIPS[:-1]]).astype(np.int32),
             fps=np.float64(FPS), q=np.concatenate([Q[c] for c in CLIPS]),
             **{k: np.concatenate([r[i] for r in ref]) for i, k in enumerate(("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"))},
             distances=np.array(DISTANCES), start_step=np.float64(START_STEP), start_span=np.float64(START_SPAN), **strike_table())
    if clean:   # the eval's own call (eval_opponent.py), so a start is clean exactly when the eval would draw it
        good = clean_starts({**load_cfg(), "opponent_aim": "none"}, list(CLIPS), list(DISTANCES), REF_POSES)
        A["clean"] = np.array([[good[(c, d)] for d in DISTANCES] for c in CLIPS])
    save_npz(out / "fb_motion.npz", **A)
    rules = dict(clips="TRAIN_CLIPS only", q="FightReplay(distance=0).q (strikes.motion); a bout at distance d: q + (d, 0) in xy",
                 ref="export_motion CSV (%.6f) -> csv_to_npz float32 on CUDA at each frame's tracker motion row (Tracker.frame); rows cross-faded "
                     "with a non-train clip taken unblended", clean=f"clean_starts(eval cfg, opponent_aim none, REF_POSES {list(REF_POSES)}), yaw 0",
                 strikes="strikes.json train rows", sha256="text sources LF-normalised, binaries raw")
    man = dict(clips=list(CLIPS), outputs={"fb_motion.npz": sha256(out / "fb_motion.npz")}, sources={s: sha256(ROOT / s) for s in sorted(set(SOURCES))},
               git=git_hash(), versions=versions(), rules=rules, frames={c: len(Q[c]) for c in CLIPS}, distances=list(DISTANCES), clean_starts=bool(clean))
    json.dump(man, open(out / "fb_motion_manifest.json", "w", encoding="utf-8"), indent=1); return A, man

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(GPU / "motions" / "fb")); a = ap.parse_args()
    A, man = export(a.out)
    print(f"fb_motion: {len(A['q'])} frames {dict(zip(CLIPS, A['frames'].tolist()))}; strikes {len(A['strike_onset'])} "
          f"({int(np.isin(A['strike_limb'], LIMBS[:2]).sum())} hand); clean starts per clip {A['clean'].sum((1, 2)).tolist()} of {A['clean'][0].size}; "
          f"sha256 {man['outputs']['fb_motion.npz'][:16]}")
