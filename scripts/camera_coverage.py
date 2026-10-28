"""Head-pitch coverage study (phase 3, M0.2) -> results/perception/coverage.json, which sets head_pitch_deg in configs/r1_camera.yaml.

How much of the opponent the R1's head camera sees, by head pitch about the assumed neck pivot: the design's candidates
0 / -10 / -20 / -30 deg, plus -40 / -50 deg as a bracket (measured, never chosen):
- bouts: both shipped policies against the tracked opponent on the train + val clips (never test), as the headline eval
  draws them (scripts/eval_opponent.py episodes, 1 per clip and distance): standard tier (0.50-0.70 m, 10 s, two pushes)
  and hard tier (0.40-0.55 m, 20 s, six strikes and pushes, randomised physics); a state every 0.24 s while the R1 is up;
- 11 opponent keypoints (r1gym/perception/render_scene.py KEYPOINTS), the opponent's head pitched like the R1's;
- in_fov: inside the network image (virtual pinhole, 160x120, fovy 100 deg). in_fisheye: inside the physical lens image
  (146 x 124 deg f-theta), FOV only, i.e. what the rectified crop gives up;
- unoccluded: in_fov and not behind the R1's own body (depth render of the R1 alone, guard included, its head shell
  hidden as on the robot). visible: also not behind another of the opponent's own links (segmentation render of the
  opponent alone: the keypoint's pixel shows one of its own links; a pixel off the silhouette counts as visible);
- own_body_px: the fraction of the image the R1's own body covers.
Rendered at the network size with the LOD render twin. Also reported (not in the rule): the opponent's bearing from the
R1's heading (both shipped gym.json files leave the turn command off, and hard-tier hits spin the R1), and the same
fractions over the states with the opponent within +-45 deg of the heading, where the pitch, not the yaw, decides.

Rule, fixed before the first run: score(pitch) = the mean over the four policy x tier cells (train + val pooled) of the
mean `unoccluded` fraction over the 11 keypoints; head_pitch_deg = the smallest |pitch| among the candidates whose score is
within 0.01 of the best candidate's. The score has no cost for pitching further (own-body pixels, the head nearing the
image's top edge), so it can rise to the end of the range: the bracket, added after review and outside the rule, shows
whether the choice is an interior optimum or the bound of the candidate range (choice.at_bound). The R1's head-pitch range
is unpublished; the design allows 0 to -30."""
import sys, json, time, copy, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import eval_opponent as eo
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS, _DR_NOMINAL
from r1gym.model import build_spec, load_cfg
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts
from r1gym.perception import run_meta
from r1gym.perception.render_scene import RenderTwin, KEYPOINTS, G_R1, G_OPP, G_OPPHEAD

POLICIES = ["arms_down=results/policy_mjlab/policy.onnx", "guard=results/policy_mjlab_guard/policy.onnx"]
PITCHES = [0.0, -10.0, -20.0, -30.0, -40.0, -50.0]; CANDIDATES = PITCHES[:4]; EVERY = 12; MARGIN = 0.01; NET = (160, 120)
KP = list(KEYPOINTS); METRICS = ("in_fov", "in_fisheye", "unoccluded", "visible")

def bouts(c, cfg, m, eps, T, dr, threads=4):
    """Physics only (threads, no GL): qpos snapshots of each episode while the R1 is up. With DR each thread steps its own
    model copy, as eval_opponent.run does."""
    ms = [copy.deepcopy(m) for _ in range(threads)] if dr else [m] * threads
    def go(g):
        out, reps = [], {}
        for i in range(g, len(eps), threads):
            ep = eps[i]; key = (ep["clip"], ep["distance"])
            if key not in reps: reps[key] = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg)
            e = R1HitEnv(seed=ep.get("env_seed", ep["seed"]), cfg=cfg, model=ms[g], episode_s=T, max_impulse=0, train=False, dr=dr)
            e.opp = reps[key]; e.opp_t0 = ep["t0"]; rng = np.random.default_rng(ep.get("hit_seed", ep["seed"])); Q = []
            o = e.reset(hits=[make_hit(k, l, imp, t, y, rng) for k, l, imp, t, y in ep["hits"]], delay=ep["delay"])
            for t in range(int(round(T / e.dt))):
                o, r, dn, info = e.step(c.act(o[None])[0])
                if info["fallen"]: break
                if (t + 1) % EVERY == 0: Q.append(e.d.qpos.copy())
            out.append((i, Q))
        return out
    with ThreadPoolExecutor(threads) as pool: res = sorted(x for r in pool.map(go, range(threads)) for x in r)
    for x in ms if dr else []: _DR_NOMINAL.pop(id(x), None)
    return [Q for _, Q in res]

def measure(tw, q, kg):
    """One state: (4 metrics, pitches, keypoints) booleans, own-body pixel fraction per pitch, keypoint elevations (pitch 0).
    Leaves the twin at the pitch it came in with."""
    p0 = tw.pitch; tw.sync(q); out = np.zeros((len(METRICS), len(PITCHES), len(KP)), bool); own = []
    for j, p in enumerate(PITCHES):
        tw.set_head_pitch(p); P = tw.keypoints("opp_"); uv, z, fov = tw.project(P, "head")
        if j == 0: Pc = tw.to_camera(P, "head"); elev = np.degrees(np.arctan2(-Pc[:, 1], np.hypot(Pc[:, 0], Pc[:, 2])))
        dep = tw.render("head", "depth", size=NET, groups=(G_R1,)); seg = tw.render("head", "seg", size=NET, groups=(G_OPP, G_OPPHEAD))
        own.append(float((dep < 20).mean())); out[0, j], out[1, j] = fov, tw.in_fisheye(P)   # background depth is the far plane (40 m)
        for k, name in enumerate(KP):
            if not fov[k]: continue
            u, v = int(uv[k, 0]), int(uv[k, 1]); out[2, j, k] = dep[v, u] >= z[k] - 0.01   # 1 cm: depth-buffer slack
            gid = seg[v, u, 0]; out[3, j, k] = out[2, j, k] and (gid < 0 or gid in kg[name])   # only the opponent is drawn
    tw.set_head_pitch(p0); return out, own, elev

def frac(x): return {k: round(float(v), 3) for k, v in zip(KP, x.mean(0))} | {"mean": round(float(x.mean()), 3)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/perception/coverage.json"); a = ap.parse_args()
    t0 = time.time(); clips = TRAIN_CLIPS + VAL_CLIPS; base = {**load_cfg(), "opponent_mode": "policy"}
    good = clean_starts(base, clips, sorted({d for t in eo.TIERS.values() for d in t["distances"]})); print(f"clean starts {time.time() - t0:.0f}s", flush=True)
    rows = []   # (policy, tier, clip, distance, qpos)
    for spec in POLICIES:
        lab = spec.split("=")[0]; c = load_controller(spec); cfg = {**controller_cfg(c), "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=cfg).compile()
        for tier, T in eo.TIERS.items():
            eps = eo.episodes(clips, 1, tier, good); Qs = bouts(c, cfg, m, eps, T["seconds"], T["dr"])
            rows += [(lab, tier, ep["clip"], ep["distance"], q) for ep, Q in zip(eps, Qs) for q in Q]
            print(lab, tier, len(eps), "bouts", sum(map(len, Qs)), "states", f"{time.time() - t0:.0f}s", flush=True)
    tw = RenderTwin(base, physics=m); kg = tw.keypoint_geoms("opp_")
    B, OWN, EL = [], [], []
    for i, (lab, tier, clip, dist, q) in enumerate(rows):
        b, own, el = measure(tw, q, kg); B.append(b); OWN.append(own); EL.append(el)
        if i % 500 == 0: print(i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    B, OWN, EL = np.array(B), np.array(OWN), np.array(EL)   # (N, metric, pitch, kp), (N, pitch), (N, kp)
    pol, tie, spl = (np.array([r[i] for r in rows]) for i in (0, 1, 2)); spl = np.where(np.isin(spl, VAL_CLIPS), "val", "train")
    oq = tw.m.jnt_qposadr[tw.m.joint("opp_floating_base_joint").id]; Q = np.array([r[4] for r in rows]); v = Q[:, oq:oq + 2] - Q[:, :2]; rng_m = np.linalg.norm(v, axis=1)
    w, x, y, z = Q[:, 3:7].T; brg = np.degrees(np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))))   # as env._face
    cells = [(p, t) for p in ("arms_down", "guard") for t in eo.TIERS]; front = np.abs(brg) <= 45
    res = dict(about=__doc__, **run_meta(gl=tw.gl()), lod=tw.lod, pitches=PITCHES, candidates=CANDIDATES, keypoints={k: dict(body=b, offset=o) for k, (b, o) in KEYPOINTS.items()},
               samples=dict(total=len(rows), by_cell={f"{p}/{t}": int(((pol == p) & (tie == t)).sum()) for p, t in cells}, by_split={s: int((spl == s).sum()) for s in ("train", "val")}),
               opponent_range_m_p5_p50_p95={t: [round(float(x), 2) for x in np.percentile(rng_m[tie == t], [5, 50, 95])] for t in eo.TIERS},
               opponent_bearing={f"{a}/{t}": dict(abs_deg_p50_p90=[round(float(x), 1) for x in np.percentile(np.abs(brg[(pol == a) & (tie == t)]), [50, 90])],
                   beyond_57p8deg=round(float(np.mean(np.abs(brg[(pol == a) & (tie == t)]) > 57.8)), 3), beyond_73deg=round(float(np.mean(np.abs(brg[(pol == a) & (tie == t)]) > 73)), 3),
                   within_45deg=round(float(np.mean(front[(pol == a) & (tie == t)])), 3)) for a, t in cells},
               keypoint_elevation_deg_p5_p50_p95={k: [round(float(x), 1) for x in np.percentile(EL[:, j], [5, 50, 95])] for j, k in enumerate(KP)},
               by_pitch={}, choice={})
    for j, p in enumerate(PITCHES):
        r = res["by_pitch"][f"{p:g}"] = {mt: frac(B[:, i, j]) for i, mt in enumerate(METRICS)}
        r["unoccluded_by_cell"] = {f"{a}/{t}": frac(B[(pol == a) & (tie == t), 2, j]) for a, t in cells}
        r["unoccluded_mean_by_split"] = {s: round(float(B[spl == s, 2, j].mean()), 3) for s in ("train", "val")}
        r["own_body_px_by_policy"] = {a: round(float(OWN[pol == a, j].mean()), 3) for a in ("arms_down", "guard")}
        r["unoccluded_within_45deg"] = frac(B[front, 2, j]); r["visible_within_45deg"] = frac(B[front, 3, j])
        r["score_within_45deg"] = round(float(np.mean([B[(pol == a) & (tie == t) & front, 2, j].mean() for a, t in cells])), 4)
        r["score"] = round(float(np.mean([B[(pol == a) & (tie == t), 2, j].mean() for a, t in cells])), 4)
    sc = {p: res["by_pitch"][f"{p:g}"]["score"] for p in PITCHES}; own = lambda p: res["by_pitch"][f"{p:g}"]["own_body_px_by_policy"]
    rule = lambda ps: min((p for p in ps if sc[p] >= max(sc[x] for x in ps) - MARGIN), key=abs)
    pick, ext = rule(CANDIDATES), rule(PITCHES); u = res["by_pitch"][f"{pick:g}"]["unoccluded"]; br = ", ".join(f"{p:g}: {sc[p]:.3f}" for p in PITCHES if p not in CANDIDATES)
    res["choice"] = dict(rule=f"smallest |pitch| among the candidates {CANDIDATES} with score within {MARGIN} of the best candidate", scores={f"{p:g}": sc[p] for p in CANDIDATES},
        bracket_scores={f"{p:g}": sc[p] for p in PITCHES if p not in CANDIDATES}, head_pitch_deg=pick, rule_over_all_measured=ext, at_bound=bool(ext != pick),
        _why=f"results/perception/coverage.json (scripts/camera_coverage.py): mean unoccluded keypoint fraction (score) by pitch "
             + ", ".join(f"{p:g}: {sc[p]:.3f}" for p in CANDIDATES) + f"; {pick:g} is the smallest candidate within {MARGIN} of the best"
             + (f", and the same rule over the bracket beyond the design's range ({br}) picks it too, so it is an interior optimum of 0 to {min(PITCHES):g}" if ext == pick else
                f", but it is the bound of the design's range (0 to {min(CANDIDATES):g}), not an optimum: over the bracket ({br}) the same rule picks {ext:g}")
             + f". At {pick:g}: pelvis {u['pelvis']:.3f}, head {u['head']:.3f}, fists {u['l_fist']:.3f}/{u['r_fist']:.3f}, feet {u['l_foot']:.3f}/{u['r_foot']:.3f} unoccluded "
             f"(train + val, both policies, both tiers); the R1's own body covers {own(pick)['guard']:.2f} of the image in guard (0 deg: {own(0.0)['guard']:.2f}).")
    tw.close(); res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({p: [res["by_pitch"][p]["unoccluded"], res["by_pitch"][p]["unoccluded_within_45deg"]] for p in res["by_pitch"]}, indent=1)); print(json.dumps(res["opponent_bearing"])); print(json.dumps(res["choice"], indent=1), f"{res['wall_s']} s")
