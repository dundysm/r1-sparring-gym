"""Track 2 pipeline checks (T2-3) on CPU, on the repo's train clips (kfab_200/201/203) and the Track 2 train pool only: the
generic path of scripts/fb_track2_motion.py must reproduce the eval's own motion, strikes and tracker-alone falls before it
screens a sealed clip.
  C1 retarget   retarget_g1_to_r1 of the raw train clip = the cached r1_<clip>.npy the eval replays (bit for bit).
  C2 process    process(cached retarget, tracker_cuts.json, style_cuts.json) = FightReplay(distance=0).q (strikes.motion), bit for
                bit; the style rule recomputed on that stage = style_cuts.json; process with the computed style cuts = the same q.
  C3 strikes    strikes.detect on it = strikes.json's rows (onset, apex, end, limb, type, band) and head-band hand counts.
  C4 reference  reference() vs the tracker's own (fb_motion.npz = Tracker.reference at Tracker.frame, M1): max abs, reported.
  C5 alone      24 starts x 3 train clips, scripts/eval_tracker.run: the eval (FightReplay + the eval's Tracker) vs the generic path
                (process + reference + FightReplay of it + T0 float32): fall within 0.1 s or both up (the G0-3b rule).
  C6 format     every train-pool clip: keys, fps 50, joint order, finite; r1gym's G1 FK of joint_pos = body_pos_w / body_quat_w
                (<= 1e-9).
Run in the repo: python -I scripts/fb_track2_check.py [--out t2_check.json] [--workers 2]"""
import sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import fb_track2_motion as T2
import fb_track2_split as SP
from r1gym import opponent as O, strikes as S
from r1gym.env import TRAIN_CLIPS

TOL = dict(alone_s=0.1, fk=1e-9)
MOT = ROOT / "r1gym" / "assets" / "motions"
SFIELDS = ("onset", "apex", "end", "limb", "type", "band")

def _raw(c):
    with np.load(MOT / f"{c}.npz", allow_pickle=False) as z: return z["qpos"].astype(float), float(z["fps"])

def _cached(c): return np.load(MOT / f"r1_{c}.npy", allow_pickle=False)

def c1(c): q, fps = _raw(c); a = T2.retarget(q); b = _cached(c); return dict(clip=c, frames=len(a), bit_equal=bool(np.array_equal(a, b)), max_abs=float(np.abs(a - b).max()))

def c234():
    cuts, sty = O._cuts(), O._cuts("style_cuts"); out = {}
    with np.load(ROOT / "gpu" / "motions" / "fb" / "fb_motion.npz", allow_pickle=False) as z: F = {k: z[k] for k in z.files}
    for k, c in enumerate(F["clips"].tolist()):
        qr = _cached(c); q0, _, info = T2.process(qr, T2.FPS, cuts.get(c, []), sty.get(c, [])); ref = S.motion(c)[0]
        q1, _, info1 = T2.process(qr, T2.FPS, cuts.get(c, [])); L, nh = T2.head_hands(q0, T2.FPS, c); J = S.load(clip=c)
        r0, n = int(F["row0"][k]), int(F["frames"][k])
        out[c] = dict(C2=dict(q_bit_equal=bool(np.array_equal(q0, ref)), style_rule_equal=info1["style_cuts"] == sty.get(c, []), q_computed_style_bit_equal=bool(np.array_equal(q1, ref))),
                      C3=dict(strikes=len(L), library=len(J), rows_equal=[tuple(getattr(s, f) for f in SFIELDS) for s in L] == [tuple(getattr(s, f) for f in SFIELDS) for s in J],
                              head_hands=int(nh), library_head_hands=int(sum(s.hand and s.band == "head" for s in J))),
                      C4={k2: float(np.abs(a - F[k2][r0:r0 + n]).max()) for k2, a in zip(("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"), T2.reference(q0))})
        out[c]["ok"] = all(out[c]["C2"].values()) and out[c]["C3"]["rows_equal"] and out[c]["C3"]["head_hands"] == out[c]["C3"]["library_head_hands"]
    return out

def _alone(args):
    c, s, mode = args
    import torch, eval_tracker as ET
    torch.set_num_threads(1); m, cfg = T2._world(); t = time.time()
    if mode == "eval":   # the eval's own: FightReplay (loads the eval's Tracker through the config) inside eval_tracker.run
        T = O.FightReplay(m, c, distance=4.0, cfg=cfg).T / 50.0; fell, _, _ = ET.run(m, cfg, c, s * T / T2.RULE["starts"], T2.RULE["seconds"])
    else:
        cuts, sty = O._cuts(), O._cuts("style_cuts"); q0, _, _ = T2.process(_cached(c), T2.FPS, cuts.get(c, []), sty.get(c, []))
        fell = T2.alone(c, q0, s * len(q0) / T2.FPS / T2.RULE["starts"])
    return dict(clip=c, start=s, mode=mode, fell=fell, wall_s=round(time.time() - t, 2))

def c5(workers=2):
    import multiprocessing as mp
    jobs = [(c, s, m) for c in TRAIN_CLIPS for s in range(T2.RULE["starts"]) for m in ("eval", "generic")]
    with mp.get_context("spawn").Pool(workers) as pool: out = pool.map(_alone, jobs, chunksize=1)
    by = {(r["clip"], r["start"], r["mode"]): r["fell"] for r in out}; bad, both_up, both_fell, dts = [], 0, 0, []
    for c in TRAIN_CLIPS:
        for s in range(T2.RULE["starts"]):
            a, b = by[(c, s, "eval")], by[(c, s, "generic")]
            if a is None and b is None: both_up += 1
            elif a is not None and b is not None and abs(a - b) <= TOL["alone_s"] + 1e-9: both_fell += 1; dts.append(abs(a - b))
            else: bad.append(dict(clip=c, start=s, eval=a, generic=b))
    return dict(runs=len(by) // 2, both_up=both_up, both_fell_within=both_fell, max_dt_s=max(dts) if dts else None, mismatches=bad,
                eval_falls={c: [by[(c, s, "eval")] for s in range(T2.RULE["starts"])] for c in TRAIN_CLIPS},
                wall_s_mean={m: round(float(np.mean([r["wall_s"] for r in out if r["mode"] == m])), 2) for m in ("eval", "generic")}, ok=not bad)

def _fmt(i): _, _, e = T2.load(i, fk=True); return dict(clip=T2.clip_name(i), **e)

def c6(workers=2):
    import multiprocessing as mp
    S_ = SP.load(); ids = [int(c.split("_")[1]) for c in S_["splits"]["train"]]
    with mp.get_context("spawn").Pool(workers) as pool: out = pool.map(_fmt, ids, chunksize=1)
    return dict(clips=len(out), bodies=sorted({r["bodies"] for r in out}), pos_max=max(r["pos"] for r in out), quat_max=max(r["quat"] for r in out),
                ok=max(max(r["pos"], r["quat"]) for r in out) <= TOL["fk"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    assert a.workers <= 2; t0 = time.time(); res = dict(tol=TOL, rule=T2.RULE)
    res["C1"] = [c1(c) for c in TRAIN_CLIPS]; print("C1", json.dumps(res["C1"]), flush=True)
    res["C2-C4"] = c234(); print("C2-C4", json.dumps(res["C2-C4"]), flush=True)
    res["C6"] = c6(a.workers); print("C6", json.dumps(res["C6"]), flush=True)
    res["C5"] = c5(a.workers); print("C5", json.dumps({k: v for k, v in res["C5"].items() if k != "eval_falls"}), flush=True)
    res["ok"] = dict(C1=all(r["bit_equal"] for r in res["C1"]), C2_C4=all(v["ok"] for v in res["C2-C4"].values()), C5=res["C5"]["ok"], C6=res["C6"]["ok"])
    res["wall_s"] = round(time.time() - t0, 1); print("ok", json.dumps(res["ok"]), "wall", res["wall_s"], flush=True)
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    sys.exit(0 if all(res["ok"].values()) else 1)

if __name__ == "__main__": main()
