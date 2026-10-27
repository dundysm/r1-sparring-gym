"""Export the opponent's fight clips as one motion for mjlab's tracking task (gpu/r1_track_task.py).

Each clip is taken exactly as the opponent replays it here (FightReplay: retargeted to the R1, drift and heading removed so it
stays in place facing -x, seamless loop), at distance 0. Every clip appears twice in a row, so the tracker also trains on its
own loop seam, and consecutive clips are cross-faded over 0.4 s. Writes a CSV in unitree_rl_mjlab's csv_to_npz layout
(root xyz, root quat xyzw, 24 joints in CTRL_JOINTS order, 50 fps) and a JSON with where each clip starts."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r1gym.model import build_spec, CTRL_JOINTS
from r1gym.opponent import FightReplay, list_clips, _sub

def blend(a, b, n):
    """Cross-fade the last n frames of a into the first n frames of b."""
    w = (np.arange(n) / n)[:, None]; tail = a[-n:].copy()
    tail[:, 3:7] *= np.sign(np.sum(tail[:, 3:7] * b[:n, 3:7], axis=1, keepdims=True) + 1e-12)
    mid = (1 - w) * tail + w * b[:n]; mid[:, 3:7] /= np.linalg.norm(mid[:, 3:7], axis=1, keepdims=True)
    return np.concatenate([a[:-n], mid, b[n:]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="gpu/motions"); ap.add_argument("--blend_s", type=float, default=0.4); a = ap.parse_args()
    m = build_spec(opponent=True).compile(); r = _sub("r1"); assert [r.joint(i).name for i in range(1, r.njnt)] == CTRL_JOINTS
    seq, index, n = None, {}, int(round(a.blend_s * 50))
    for clip in list_clips():
        q = FightReplay(m, clip, distance=0.0).q
        two = np.concatenate([q, q])                     # the clip is seamless, so its loop seam is a normal step
        start = 0 if seq is None else len(seq) - n       # where the second clip's first frame lands after the blend
        seq = two if seq is None else blend(seq, two, n)
        index[clip] = {"start_frame": int(start), "frames": int(len(q))}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv = np.concatenate([seq[:, :3], seq[:, [4, 5, 6, 3]], seq[:, 7:]], 1)   # wxyz -> xyzw
    np.savetxt(out / "r1_kungfu.csv", csv, delimiter=",", fmt="%.6f")
    json.dump({"fps": 50, "joints": CTRL_JOINTS, "clips": index, "frames": int(len(seq)), "blend_frames": n}, open(out / "r1_kungfu_index.json", "w"), indent=1)
    print(len(seq), "frames,", round(len(seq) / 50, 1), "s;", index)
