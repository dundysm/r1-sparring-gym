"""M1 checks of the full-body M2 v2 exports (M2V2_PLAN_v3 3 M1), on CPU against the eval's own code:
  (1) clips == {kfab_200, kfab_201, kfab_203}; the manifests' sha256 of the outputs and of the sources match the files;
  (2) T-ref: every train frame's reference equals the tracker's at Tracker.frame(clip, f) within 1e-4, every index inside the
      clip's own rows; and q at a bout distance d is q + (d, 0) (FightReplay at d);
  (3) the clean-start masks equal scenario.clean_starts (the eval's call) exactly;
  (4) the strike table equals strikes.json's train rows (37 hand + 1 kick);
  (5) Hausdorff of every exported hull to the eval's full hull, all 64 meshes: <= 3 mm on the fine links, <= 1 cm elsewhere.
The tracker ONNX is read here as the reference to check against; the exports read none of it (export_fb_motion's clean_starts
call loads it only through the eval's FightReplay, unused).
Run in the repo: python scripts/fb_m1_check.py [--out m1_check.json] [--quick]   (--quick: clean starts at 6 distances only)"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu"))
import numpy as np
from r1gym import strikes as S
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay, Tracker
from r1gym.scenario import clean_starts, REF_POSES
import export_fb_motion as EM, export_fb_hulls as EH

OUT = ROOT / "gpu" / "motions" / "fb"; TOL_REF = 1e-4; EVAL_D = (0.4, 0.45, 0.5, 0.55)

def load(out=OUT): return dict(np.load(Path(out) / "fb_motion.npz")), json.load(open(Path(out) / "fb_motion_manifest.json", encoding="utf-8"))

def check_manifest(out=OUT):
    A, man = load(out); H = json.load(open(Path(out) / "fb_hulls_manifest.json", encoding="utf-8"))
    stale = [s for M in (man, H) for s, h in M["sources"].items() if EM.sha256(ROOT / s) != h]
    outs = {k: EM.sha256(Path(out) / k) == h for M in (man, H) for k, h in M["outputs"].items()}
    return dict(clips=A["clips"].tolist(), clips_ok=set(A["clips"].tolist()) == {"kfab_200", "kfab_201", "kfab_203"} == set(man["clips"]),
                outputs_sha_ok=all(outs.values()) and len(outs) == 2, stale_sources=stale)

def check_tref(out=OUT):
    A, _ = load(out); tr = Tracker(str(ROOT / load_cfg()["opponent_policy"])); res = {}
    m = build_spec(opponent=True, cfg={**load_cfg(), "opponent_mode": "kinematic"}).compile()
    for k, c in enumerate(A["clips"].tolist()):
        r0, T = int(A["row0"][k]), int(A["frames"][k]); s0 = tr.index[c]["start_frame"]; assert tr.index[c]["frames"] == T
        g = np.array([tr.frame(c, f) for f in range(T)]); assert ((g >= s0) & (g < s0 + 2 * T)).all(), c   # the clip's own rows only
        sl = slice(r0, r0 + T); e = lambda x, y: float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
        res[c] = dict(frames=T, rows=[int(g.min()), int(g.max())], joint_pos=e(A["ref_joint_pos"][sl], tr.ref_q[g]), joint_vel=e(A["ref_joint_vel"][sl], tr.ref_dq[g]),
                      anchor_pos=e(A["ref_anchor_pos"][sl], tr.ref_anchor_pos[g]), anchor_quat=e(A["ref_anchor_quat"][sl], tr.ref_anchor[g]),
                      exact_joint_pos=float(np.mean(A["ref_joint_pos"][sl] == tr.ref_q[g])), exact_joint_vel=float(np.mean(A["ref_joint_vel"][sl] == tr.ref_dq[g])),
                      q_vs_ref_joint_pos=e(A["q"][sl, 7:], tr.ref_q[g]))
        q0 = A["q"][sl]; res[c]["q_at_distance"] = max(e(FightReplay(m, c, distance=d, cfg={**load_cfg(), "opponent_mode": "kinematic"}).q, q0 + np.r_[d, 0.0, [0.0] * 29]) for d in EVAL_D)
    keys = ("joint_pos", "joint_vel", "anchor_pos", "anchor_quat")
    return dict(per_clip=res, max={k: max(r[k] for r in res.values()) for k in keys}, ok=all(r[k] <= TOL_REF for r in res.values() for k in keys))

def check_clean(out=OUT, distances=None):
    A, _ = load(out); D = A["distances"].tolist(); want = D if distances is None else [d for d in D if any(abs(d - x) < 1e-9 for x in distances)]
    good = clean_starts({**load_cfg(), "opponent_aim": "none"}, A["clips"].tolist(), want, REF_POSES)
    diff = {f"{c} {d}": int((A["clean"][k, D.index(d)] != good[(c, d)]).sum()) for k, c in enumerate(A["clips"].tolist()) for d in want}
    return dict(compared=len(diff), cells=len(diff) * A["clean"].shape[2], mismatches=sum(diff.values()), ok=sum(diff.values()) == 0,
                clean_share={c: round(float(A["clean"][k].mean()), 4) for k, c in enumerate(A["clips"].tolist())})

def check_strikes(out=OUT):
    A, _ = load(out); L = S.load(split="train"); clips = A["clips"].tolist(); bad = []
    for i, s in enumerate(L):
        if clips[A["strike_clip"][i]] != s.clip or A["strike_limb"][i] != s.limb or A["strike_type"][i] != s.type or A["strike_band"][i] != s.band: bad.append(i)
        for k in EM.STRIKE_NUM:
            if not np.array_equal(np.asarray(A["strike_" + k][i], np.float64), np.asarray(getattr(s, k), np.float64)): bad.append(i)
    hands = int(np.isin(A["strike_limb"], ["l_hand", "r_hand"]).sum())
    return dict(rows=len(A["strike_onset"]), library_train_rows=len(L), hand=hands, kick=len(L) - hands, mismatched_rows=sorted(set(bad)),
                ok=not bad and len(A["strike_onset"]) == len(L) == 38 and hands == 37)

def check_hulls(out=OUT):
    z = np.load(Path(out) / "fb_hulls.npz"); res = EH.gate(EH.eval_model(), z["meshes"].tolist(), z["points"], z["n"])
    fine = {k: v for k, v in res.items() if k.replace("opp_", "") in EH.FINE}; coarse = {k: v for k, v in res.items() if k not in fine}
    worst = lambda R: max((round(v["hausdorff_m"] * 1e3, 3), k) for k, v in R.items())
    return dict(meshes=len(res), fine=len(fine), coarse=len(coarse), worst_fine_mm=worst(fine), worst_coarse_mm=worst(coarse),
                n=dict(zip(z["meshes"].tolist(), z["n"].tolist())), ok=len(res) == 64 and all(v["hausdorff_m"] <= v["tol_m"] for v in res.values()),
                per_mesh_mm={k: round(v["hausdorff_m"] * 1e3, 3) for k, v in sorted(res.items())})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    R = dict(manifest=check_manifest(), tref=check_tref(), clean=check_clean(distances=(0.35,) + EVAL_D + (0.65,) if a.quick else None),
             strikes=check_strikes(), hulls=check_hulls())
    R["ok"] = all(R[k]["ok"] for k in ("tref", "clean", "strikes", "hulls")) and R["manifest"]["clips_ok"] and R["manifest"]["outputs_sha_ok"] and not R["manifest"]["stale_sources"]
    if a.out: json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("per_clip", "per_mesh_mm", "n")} if isinstance(v, dict) else v) for k, v in R.items()}, indent=1))
    sys.exit(0 if R["ok"] else 1)
