"""Glove keypoints from the R1's head camera (sparring plan, piece C) -> results/perception/gloves_check.json.

The opponent wears green gloves in the render twin only (render_scene.set_gloves: geom_rgba of its hand meshes; physics
untouched) and the colour-key detector reports each glove (detect_colorkey.py gloves=True). This measures it against the
truth on rendered frames:
- bouts: the footwork guard (guard_fw) against the tracked opponent aimed by the chosen operator
  (results/opponent_tracker/aim.json), no scripted hits, SECONDS each, at DISTANCES (0.40-0.65 m), starts drawn from the
  clean ones as eval_opponent draws them; a state every EVERY control steps (25 Hz, near the camera's 30) while the R1 is up;
- strike frames: inside an aimed hand-strike intent's window [t_on, t_end] (r1gym/aim.py; gated intents excluded);
- truth per glove: the fist keypoint (render_scene KEYPOINTS l_fist / r_fist: the hand mesh's centre, 8 cm along the
  wrist_roll_link) and the wrist joint (the link origin), through the head camera; in_fov: inside the 160x120 image;
  visible px: the glove's own pixels in a segmentation render (render size, area-downsampled to network pixels);
- matching: each frame's 0-2 detections go to the 0-2 true fists by least total angle, so errors are geometric; the
  detector's own left / right label (image side) is scored separately (side_correct), and so is GloveTracker's label run
  over each bout's frames in order (side_correct_tracked); fists_crossed_in_image: of the frames with both fists in FOV,
  the share where the opponent's left fist is left of its right one in the R1's view (no image-side rule labels those);
- errors: also split by edge (the blob touches the image border, so part of the glove may be cut off);
- detection rate: a glove in FOV (and, separately, with >= GLOVE_MIN_PX visible px) that some detection was matched to;
- false detections: a detection matched to a glove with no visible pixel; any glove detection on the same frames rendered
  with gloves off (the nominal orange opponent) or with the opponent parked 5 m behind the R1 (outside the room);
- body: the existing opponent bearing (orange mask) with gloves on vs off on the same frames (gloves take the hands out
  of the orange mask).
The truth proprioception (exact encoders and gravity) feeds mount_fk, so xyz errors exclude sensor noise.
--split train fits GLOVE_AREA (median of solid angle x true range^2 over matched train detections), the one constant the
detector takes from data; --split val measures with the constant in the code. Development (thresholds, the fragment merge)
and GloveTracker's gate / gap used train frames only. Output keeps every val glove-frame row (raw) so the summary recomputes."""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, mujoco
from r1gym import ROOT
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.model import build_spec, R1Robot
from r1gym.opponent import FightReplay
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start
from r1gym.perception import run_meta
from r1gym.perception.camera import HeadCameraSpec, area_down, rz
from r1gym.perception.render_scene import RenderTwin, KEYPOINTS, GLOVE_RGBA
from r1gym.perception.detect_colorkey import ColorKeyDetector, GloveTracker

POLICY = "guard_fw=results/policy_mjlab_guard_footwork/policy.onnx"
DISTANCES = [0.40, 0.475, 0.55, 0.65]; SECONDS = 8.0; EVERY = 2; SEED = 50_000
SPLITS = dict(train=dict(clips=TRAIN_CLIPS, starts=1), val=dict(clips=VAL_CLIPS, starts=3))
OUT = ROOT.parent / "results" / "perception" / "gloves_check.json"
SIDES = ("left", "right"); KI = {s: list(KEYPOINTS).index(s[0] + "_fist") for s in SIDES}

def bouts(split):
    """Physics only: (episode, [(t, qpos)], aimed hand-strike windows)."""
    c = load_controller(POLICY); op = json.load(open(ROOT.parent / "results" / "opponent_tracker" / "aim.json"))["operator"]
    cfg = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": op}; m = build_spec(opponent=True, cfg=cfg).compile()
    S = SPLITS[split]; good = clean_starts(cfg, S["clips"], DISTANCES); out = []; k = 0
    for clip in S["clips"]:
        for dist in DISTANCES:
            for rep in range(S["starts"]):
                rng = np.random.default_rng(SEED + k); k += 1; t0 = pick_start(good[(clip, dist)], float(rng.uniform(0, 6)))
                e = R1HitEnv(seed=SEED + k, cfg=cfg, model=m, episode_s=SECONDS + 1, max_impulse=0, train=False)
                e.opp = FightReplay(m, clip, distance=dist, cfg=cfg); e.opp_t0 = t0; o = e.reset(hits=[], delay=1); Q = []
                for i in range(int(round(SECONDS / e.dt))):
                    o, _, done, info = e.step(c.act(o[None])[0])
                    if done: break
                    if i % EVERY == EVERY - 1: Q.append((round(float(e.d.time), 4), e.d.qpos.copy()))
                win = [(it.t_on, it.t_end) for it in e.opp_state["aim"].intents if it.limb in ("l_hand", "r_hand") and not it.gated]
                out.append((dict(clip=clip, distance=dist, rep=rep, t0=t0), Q, win))
    return out, cfg, m, op

def ang(a, b): return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

def head_dir(b, e): return np.array([np.cos(e) * np.cos(b), np.cos(e) * np.sin(b), np.sin(e)])

def truth(tw, qpos, qadr, glove_ids, Rm):
    """Sync the twin to a state; per glove: in_fov, visible px, head-frame direction to the fist and the wrist joint, lens range,
    heading-frame position. Also the true proprioception (q, g) the detector gets."""
    tw.sync(qpos); m, d = tw.m, tw.d; pel = m.body("pelvis").id; Rp = d.xmat[pel].reshape(3, 3); H = rz(-np.arctan2(Rp[1, 0], Rp[0, 0]))
    kp = tw.keypoints("opp_"); seg = tw.render("head", "seg"); geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1); T = {}
    for s in SIDES:
        P = kp[KI[s]]; W = d.xpos[m.body(f"opp_{s}_wrist_roll_link").id]; (c, w) = tw.to_camera(np.stack([P, W]))
        _, _, inside = tw.project(P[None], "head"); t, tw_ = Rm @ c, Rm @ w
        T[s] = dict(in_fov=bool(inside[0]), vis_px=round(float(area_down(np.isin(geom, glove_ids[s]).astype(float), tw.spec.scale).sum()), 2),
                    dir=t / np.linalg.norm(t), wdir=tw_ / np.linalg.norm(tw_), range=float(np.linalg.norm(c)), xyz=H @ (P - d.xpos[pel]))
    return T, qpos[qadr], Rp.T @ np.array([0, 0, -1.0])

def checks(tw, det, qpos, q, g, oq, est):
    """Frame-level: the same state with gloves off (any glove detection is false; the body bearing it gives vs gloved) and with
    the opponent parked 5 m behind the R1, outside the room (any detection is false). Leaves the twin synced and gloved."""
    tw.set_gloves(None); plain = det(tw.frame("head"), q, g); Rp = tw.d.xmat[tw.m.body("pelvis").id].reshape(3, 3); yaw = np.arctan2(Rp[1, 0], Rp[0, 0])
    qq = qpos.copy(); qq[oq:oq + 2] = qpos[:2] - 5.0 * np.array([np.cos(yaw), np.sin(yaw)]); tw.sync(qq); tw.set_gloves(GLOVE_RGBA)
    empty = det(tw.frame("head"), q, g); tw.sync(qpos)
    return dict(n_det_plain=sum(v is not None for v in plain["gloves"].values()), n_det_empty=sum(v is not None for v in empty["gloves"].values()),
                presence=est["presence"], presence_plain=plain["presence"],
                body_shift_deg=round(float(np.degrees(abs(np.angle(np.exp(1j * (est["bearing"] - plain["bearing"])))))), 3) if est["presence"] and plain["presence"] else None)

def match(G, T, Gt):
    """Per-glove rows: G (find_gloves) matched to the true fists by least total angle; side_correct: G's image-side label,
    side_correct_tracked: Gt's (GloveTracker) label of the same detection."""
    D = [(k, v) for k, v in G.items() if v is not None]; dirs = [head_dir(v["bearing"], v["elevation"]) for _, v in D]
    tl = {(v["u"], v["v"]): k for k, v in Gt.items() if v is not None}
    perms = [((0, "left"), (1, "right")), ((0, "right"), (1, "left"))] if len(D) == 2 else [((0, "left"),), ((0, "right"),)] if D else [()]
    best = min(perms, key=lambda pm: sum(ang(dirs[i], T[s]["dir"]) for i, s in pm)); mt = {s: i for i, s in best}; rows = []
    for s in SIDES:
        tr = T[s]; r = dict(side=s, in_fov=tr["in_fov"], vis_px=tr["vis_px"], range_true=round(tr["range"], 4), n_det=len(D),
                            bearing_true_deg=round(float(np.degrees(np.arctan2(tr["dir"][1], tr["dir"][0]))), 3), elevation_true_deg=round(float(np.degrees(np.arcsin(tr["dir"][2]))), 3))
        if s in mt:
            lab, v = D[mt[s]]; dv = dirs[mt[s]]
            r.update(detected=True, side_correct=lab == s, side_correct_tracked=tl[(v["u"], v["v"])] == s, n_px=v["n_px"], edge=v["edge"], u=round(v["u"], 3), v=round(v["v"], 3),
                     err_deg=round(ang(dv, tr["dir"]), 3), wrist_err_deg=round(ang(dv, tr["wdir"]), 3),
                     bearing_err_deg=round(float(np.degrees(np.angle(np.exp(1j * (v["bearing"] - np.arctan2(tr["dir"][1], tr["dir"][0])))))), 3),
                     elevation_err_deg=round(float(np.degrees(v["elevation"] - np.arcsin(tr["dir"][2]))), 3), solid_angle=v["solid_angle"],
                     range_err=None if not np.isfinite(v["range"]) else round(v["range"] - tr["range"], 4),
                     xyz_err=None if not np.isfinite(v["range"]) else round(float(np.linalg.norm(np.array(v["xyz"]) - tr["xyz"])), 4))
        else: r.update(detected=False)
        rows.append(r)
    return rows

def rows(raw): return [dict(zip(raw["columns"], r)) for r in raw["rows"]]

def pct(x, p): x = [v for v in x if v is not None]; return round(float(np.percentile(np.abs(x) if p != "bias" else x, 50 if p == "bias" else p)), 3) if x else None

def errors(det):
    """Error percentiles (abs; bias: signed median) over matched detections."""
    rel = [r["range_err"] / r["range_true"] for r in det if r["range_err"] is not None]; E = lambda k: [r[k] for r in det]
    return dict(n=len(det), angle_err_deg=dict(median=pct(E("err_deg"), 50), p90=pct(E("err_deg"), 90)),
                bearing_err_deg=dict(median=pct(E("bearing_err_deg"), 50), p90=pct(E("bearing_err_deg"), 90)),
                elevation_err_deg=dict(median=pct(E("elevation_err_deg"), 50), p90=pct(E("elevation_err_deg"), 90), bias=pct(E("elevation_err_deg"), "bias")),
                vs_wrist_joint_angle_err_deg=dict(median=pct(E("wrist_err_deg"), 50), p90=pct(E("wrist_err_deg"), 90)),
                range_err_m=dict(median=pct(E("range_err"), 50), p90=pct(E("range_err"), 90), bias=pct(E("range_err"), "bias")),
                range_err_rel=dict(median=pct(rel, 50), p90=pct(rel, 90)), xyz_err_m=dict(median=pct(E("xyz_err"), 50), p90=pct(E("xyz_err"), 90)))

def summarize(R, F):
    """R: glove rows; F: frame rows. Rates, side labels and errors as described in the module docstring; errors also split by
    whether the blob touches the image border (edge: the glove may be cut off)."""
    fov = [r for r in R if r["in_fov"]]; vis = [r for r in fov if r["vis_px"] >= ColorKeyDetector.GLOVE_MIN_PX]; det = [r for r in R if r["detected"]]
    both = [(a, b) for a, b in zip(R[::2], R[1::2]) if a["in_fov"] and b["in_fov"]]
    return dict(frames=len(F), glove_frames=len(R), in_fov=len(fov), visible=len(vis),
                detection_rate_in_fov=round(np.mean([r["detected"] for r in fov]), 4) if fov else None,
                detection_rate_visible=round(np.mean([r["detected"] for r in vis]), 4) if vis else None,
                detections=len(det), side_correct=round(np.mean([r["side_correct"] for r in det]), 4) if det else None,
                side_correct_tracked=round(np.mean([r["side_correct_tracked"] for r in det]), 4) if det else None,
                fists_crossed_in_image=round(np.mean([a["bearing_true_deg"] > b["bearing_true_deg"] for a, b in both]), 4) if both else None,
                errors=errors(det), errors_not_edge=errors([r for r in det if not r["edge"]]), errors_edge=errors([r for r in det if r["edge"]]),
                false_detections=dict(matched_to_invisible_glove=sum(r["vis_px"] == 0 for r in det),
                                      gloves_off_frames_with_any=sum(f["n_det_plain"] > 0 for f in F),
                                      no_opponent_frames_with_any=sum(f["n_det_empty"] > 0 for f in F), frames=len(F)),
                body_bearing_shift_deg=dict(median=pct([f["body_shift_deg"] for f in F], 50), p90=pct([f["body_shift_deg"] for f in F], 90),
                                            presence_changed=sum(f["presence"] != f["presence_plain"] for f in F)))

def dumps(out):
    """indent=1, but each raw row on one line (the val block's raw rows would otherwise take a line per value)."""
    rr = {k: v["raw"]["rows"] for k, v in out.items() if isinstance(v, dict) and isinstance(v.get("raw"), dict)}
    for k in rr: out[k]["raw"]["rows"] = f"@ROWS:{k}@"
    t = json.dumps(out, indent=1)
    for k, R in rr.items(): t = t.replace(f'"@ROWS:{k}@"', "[\n" + ",\n".join(json.dumps(r) for r in R) + "\n]"); out[k]["raw"]["rows"] = R
    return t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=list(SPLITS), required=True); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    t_start = time.time(); B, cfg, m, op = bouts(a.split); spec = HeadCameraSpec(gym_cfg=cfg); tw = RenderTwin(cfg, physics=m, spec=spec)
    det = ColorKeyDetector(spec, cfg, gloves=True); oq = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]; ids = tw.glove_geoms(); Rm = spec.mount_torso[1]; qadr = R1Robot(m, cfg).qadr
    R, F = [], []
    for ep, Q, win in B:
        trk = GloveTracker()
        for t, qp in Q:
            T, q, g = truth(tw, qp, qadr, ids, Rm); tw.set_gloves(GLOVE_RGBA); t0 = time.perf_counter(); est = det(tw.frame("head"), q, g); ms = (time.perf_counter() - t0) * 1e3
            rows = match(est["gloves"], T, trk(est["gloves"], t)); fr = checks(tw, det, qp, q, g, oq, est); strike = any(a_ <= t <= b_ for a_, b_ in win)
            for r in rows: r.update(clip=ep["clip"], distance=ep["distance"], rep=ep["rep"], t=t, strike=strike)
            fr.update(strike=strike, distance=ep["distance"], ms=ms); R += rows; F.append(fr)
    fit = None
    if a.split == "train":
        x = [r["solid_angle"] * r["range_true"] ** 2 for r in R if r["detected"] and r["vis_px"] >= ColorKeyDetector.GLOVE_MIN_PX]
        fit = dict(glove_area_m2=round(float(np.median(x)), 6), n=len(x), p10_p90=[round(float(v), 6) for v in np.percentile(x, [10, 90])],
                   rule="median over matched train detections (>= GLOVE_MIN_PX visible px) of solid angle x true lens-to-fist range^2")
    S = dict(all=summarize(R, F), strike=summarize([r for r in R if r["strike"]], [f for f in F if f["strike"]]),
             by_distance={f"{dd:g}": summarize([r for r in R if r["distance"] == dd], [f for f in F if f["distance"] == dd]) for dd in DISTANCES})
    path = Path(a.out); old = json.loads(path.read_text()) if path.exists() else {}
    block = dict(clips=SPLITS[a.split]["clips"], distances=DISTANCES, starts=SPLITS[a.split]["starts"], seconds=SECONDS, every_steps=EVERY, seed=SEED,
                 episodes=[ep for ep, _, _ in B], strike_windows=sum(len(w) for _, _, w in B), summary=S,
                 glove_area_m2_used=ColorKeyDetector.GLOVE_AREA, detect_ms_median=round(float(np.median([f["ms"] for f in F])), 2), wall_s=round(time.time() - t_start, 1))
    if fit: block["fit"] = fit
    if a.split == "val":   # columnar (rows() rebuilds the dicts); a glove not detected has None in the detection columns
        cols = list(dict.fromkeys(k for r in R for k in r)); block["raw"] = dict(columns=cols, rows=[[round(r[k], 6) if isinstance(r.get(k), float) else r.get(k) for k in cols] for r in R])
    out = dict(about=__doc__, script="scripts/glove_check.py", policy=POLICY, opponent_aim=op, glove_rgba=list(GLOVE_RGBA),
               detector=dict(glove_hue_deg=list(ColorKeyDetector.GLOVE_HUE), sat_min=ColorKeyDetector.SAT, val_min=ColorKeyDetector.VAL,
                             glove_min_px=ColorKeyDetector.GLOVE_MIN_PX, glove_merge_px=ColorKeyDetector.GLOVE_MERGE_PX, glove_area_m2=ColorKeyDetector.GLOVE_AREA,
                             tracker=dict(gate_deg=GloveTracker().gate, max_gap_s=GloveTracker().gap)),
               **{k: v for k, v in old.items() if k in SPLITS and k != a.split}, **{a.split: block}, **run_meta(gl=tw.gl()))
    tw.close(); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(dumps(out))
    print(json.dumps(dict(fit=fit, all=S["all"], strike=S["strike"]), indent=1, default=float))

if __name__ == "__main__":
    main()
