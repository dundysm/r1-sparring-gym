"""Turn-toward-the-opponent A/B for the stepping policies -> results/face_opponent_ab.json (the evidence behind gym.json).

1. Turn response: yaw-rate command held from standing for 6 s, mean measured rate after 2 s, 3 seeds.
2. Bouts with the shipped opponent on train + val clips (never test): 2 distances x 6 clean starts per clip, 12 s, the demo's
   scripted pushes, obs delay 1 step. Each policy runs with the command off, on, and on with a deadband (turn once the
   bearing error passes 20 deg, stop below 6 deg). Facing error = |bearing to the opponent - pelvis heading|, median per
   bout, then median over the bouts it survived; drift = pelvis distance from its start at the end; ring-out = got more
   than 1 m from its start at any time (rates over survived bouts).
--hold: the same bouts for the hold-your-spot outer loop (off / hold / hold_tight) -> results/hold_ab.json.
3. --val: survival (and up-and-within-1 m, drift) on the headline eval's validation episodes (scripts/eval_opponent.py, standard and hard tier) for each
   setting, merged into the same file. Settings are chosen on these, never on test.
4. --offaxis (phase 3, M1): the R1 turning toward what its own head camera sees, from off-axis starts; see M1_DOC below."""
import sys, json, argparse, time, hashlib, datetime, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS, TEST_CLIPS
from r1gym.hits import HitMeter
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, pick_start, SCRIPTED, offaxis_episodes, OFFAXIS_BEARINGS_DEG, OFFAXIS_STARTS, OFFAXIS_SEED
from r1gym.stats import cluster_paired_diff
from r1gym.perception.percept import attach, wrap

SETTINGS = {"off": dict(face_opponent=False), "on": dict(face_opponent=True), "deadband": dict(face_opponent=True, face_deadband=[0.35, 0.1])}
# --hold: the hold-your-spot outer loop (r1gym/env.py _hold) instead, turn command off throughout
HOLD = {"off": dict(hold_position=False), "hold": dict(hold_position=True, hold_deadband=[0.25, 0.08]), "hold_tight": dict(hold_position=True, hold_deadband=[0.12, 0.05])}

def turn_response(c, cfg):
    out = {}; m = build_spec(cfg=cfg).compile()   # no opponent in the scene
    for wz in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        r = []
        for seed in range(3):
            e = R1HitEnv(seed=seed, cfg=cfg, model=m, episode_s=8, max_impulse=0, train=False); o = e.reset(hits=[], delay=1); e.command = np.array([0, 0, wz]); ys = []
            for _ in range(300):
                o, *_ = e.step(c.act(o[None])[0]); R = e.d.xmat[e.r.pel].reshape(3, 3); ys.append(np.arctan2(R[1, 0], R[0, 0]))
            r.append(np.mean(np.diff(np.unwrap(ys[100:]))) / e.dt)
        out[f"{wz:.1f}"] = round(float(np.mean(r)), 2)
    return out

def bouts(c, cfg, m, eps, seconds=12.0, settings=None):
    def run(ep):
        cl, d, t0 = ep; e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=seconds + 1, max_impulse=0, train=False)
        e.opp = FightReplay(m, cl, distance=d, cfg=cfg); e.opp_t0 = t0; rng = np.random.default_rng(3)
        o = e.reset(hits=[make_hit(k, l, i, t, y, rng) for k, l, i, t, y in SCRIPTED], delay=1); q = e.r.free_q; start = e.d.qpos[q:q + 2].copy(); f = []; maxr = 0.0
        for _ in range(int(seconds / e.dt)):
            o, r, dn, info = e.step(c.act(o[None])[0])
            if info["fallen"]: return dict(fell=True)
            v = e.d.xpos[min(e.opp.bodies)][:2] - e.d.xpos[e.r.pel][:2]; R = e.d.xmat[e.r.pel].reshape(3, 3)
            f.append(abs(np.degrees(np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.arctan2(R[1, 0], R[0, 0])))))))
            maxr = max(maxr, float(np.linalg.norm(e.d.qpos[q:q + 2] - start)))
        return dict(fell=False, face=float(np.median(f)), face90=float(np.percentile(f, 90)), drift=float(np.linalg.norm(e.d.qpos[q:q + 2] - start)), out=maxr > 1.0)
    rows = list(ThreadPoolExecutor(8).map(run, eps)); ok = [r for r in rows if not r["fell"]]
    return dict(bouts=len(rows), falls=sum(r["fell"] for r in rows), facing_err_median_deg=round(float(np.median([r["face"] for r in ok])), 1),
                facing_err_p90_median_deg=round(float(np.median([r["face90"] for r in ok])), 1), drift_mean_m=round(float(np.mean([r["drift"] for r in ok])), 2),
                ring_out=round(float(np.mean([r["out"] for r in ok])), 3))

def val_survival(c, base, m, settings=SETTINGS):
    import eval_opponent as eo
    from r1gym.model import load_cfg
    out = {}
    for tier in ("standard", "hard"):
        T = eo.TIERS[tier]; good = clean_starts({**load_cfg(), "opponent_mode": "policy"}, VAL_CLIPS, T["distances"]); eps = eo.episodes(VAL_CLIPS, 16, tier, good)
        out[tier] = {}
        for k, v in settings.items():
            rows = eo.run(c, eps, m, {**base, "face_deadband": None, **v}, T["seconds"], dr=T["dr"])
            out[tier][k] = dict(survival=round(float(np.mean([r["fell_s"] is None for r in rows])), 3),
                                in_ring=round(float(np.mean([r["fell_s"] is None and not r["ring_out"] for r in rows])), 3),
                                drift_m=round(float(np.mean([r["drift_m"] for r in rows if r["fell_s"] is None] or [0])), 2))
    return out

# ---- --offaxis (phase 3, M1): the R1 turns toward what its own head camera sees -> results/perception/m1_offaxis_{val,test}.json
M1_DOC = """--offaxis (phase 3, M1): the R1 turns toward what its own head camera sees, with no learning yet.
Off-axis bouts (r1gym/scenario.py offaxis_episodes): the R1 starts turned so the opponent sits at +-30 / +-50 deg; the
steered opponent keeps to the bout's world axis, so it stays off-axis unless the R1 turns. Guard policy by default (it
turns both ways; --policy picks another, e.g. a newer footwork policy), hold as its gym.json ships it, distances 0.55 /
0.65 m, 12 s, the demo's pushes, obs delay 1 step; clean starts per clip and distance: 6 on val, 12 on test (below).
Settings, each on the same episodes:
  blind             no turn command (gym.json as shipped)
  truth             the true bearing (env._face as before)
  truth_lag         exact values on the head camera's clock, latency and drops (r1gym/perception/percept.py)
  truth_lag_nocomp  the same without ego compensation (the gyro yaw since capture): what compensation buys
  camera            the head camera: render twin -> colour-key detector (r1gym/perception/hub.py, detect_colorkey.py)
  shuffled          another bout's camera estimates at the same times (a seeded derangement): a control
Also an arms-down row (it can only turn right: offsets -30 / -50 deg; --arms) and an empty-ring control (camera; the R1
starts facing away from the opponent, parked 5 m behind it beyond the room's wall).
Per bout: facing error = |true bearing| (median and p90 over the last 6 s), time to face within 15 deg (never: 12 s),
falls, drift, ring-out (> 1 m from the start), the fraction of ticks with a turn command, landed head/torso hits
(r1gym/hits.py), and the percept's error against the truth at capture and at use, latency, drops and track loss.
--split smoke (one train clip, kfab_200): the code paths at any size, for checking a change; never val or test clips.
--split val (kfab_204): tunes only percept_filter_s x face_deadband (M1_GRID), then runs everything at the choice.
--prereg: writes results/perception/preregistration_m1.json from the val choice, before the test run (never overwritten).
--split test (kfab_202 / kfab_205): once, against the preregistration; refuses unless the preregistration is tracked in git
and unmodified (committed before the test run), and if a file it hashed (every r1gym/**/*.py, configs/*.yaml, this script,
the policies and the opponent tracker) changed, the code set grew, or the result exists. Val rule, fixed before the val run: objective = camera's mean over bouts of the per-bout median facing error over
the last 6 s (a fall counts 180 deg); the lowest wins, but within 0.5 deg of it a wider deadband (none < 15/5 deg < 20/6
deg) wins, then a shorter filter. Paired differences resample whole (clip, start) clusters (the 8 bouts of a start draw:
its 4 bearings share a clip time, and each distance takes the clean start at the same draw, on test within 0.54 s of the
other; r1gym/stats.py cluster_paired_diff, 4000 resamples, seed 0); a pair where either bout fell is left out.
Test size, set from the val run before any test bout ran: with 6 (clip, start) clusters val's camera - truth_lag CI was
+-2.9 deg wide (results/perception/m1_offaxis_val.json), about the whole +-3 deg margin of criterion (a); the design's 6
test starts (12 clusters) would leave it near +-2, so test draws 12 starts per clip and distance (24 clusters, 192 bouts;
expected +-1.5 deg; preregistration_m1.json records the arithmetic).
Disclosure (this run's test result): before the preregistration was written, the --prereg and --split test code paths were
smoke-run once on the test clips (3 s, 1 start, against a scratch preregistration, output discarded); the run printed its
criteria and summary. Nothing was chosen from it (the val choice and the test size come from the val file), but the test
clips were not unseen at preregistration; --split smoke now exists so this cannot recur. The empty-ring control (e) only
shows no false detections on the R1's own body and an empty room: the opponent is never seen there (presence 0, no_track
1, reported next to it), so it says nothing about losing a track once the opponent was seen."""
M1 = dict(seconds=12.0, distances=[0.55, 0.65], last_s=6.0, face_deg=15.0, ring_m=5.0, fell_deg=180.0, tie_deg=0.5, threads=4, test_starts=12)
M1_GRID = dict(percept_filter_s=[0.0, 0.1], face_deadband=[None, [0.26, 0.08], [0.35, 0.1]])   # deadbands [start, stop] rad: 15/5 and 20/6 deg
M1_LADDER = ("blind", "truth", "truth_lag", "truth_lag_nocomp", "camera", "shuffled"); M1_ARMS = ("blind", "truth", "truth_lag", "camera", "shuffled")
M1_CRITERIA = {"a": "median facing error over the last 6 s, paired camera - truth_lag: 95% CI within [-3, +3] deg",
               "b": "camera and truth_lag both beat blind on it: blind - camera and blind - truth_lag have 95% CIs above 0",
               "c": "time to face within 15 deg, paired camera - truth_lag: 95% CI upper bound <= 0.5 s",
               "d": "shuffled is no better than blind: blind - shuffled (facing error) has a 95% CI lower bound <= 0",
               "e": "empty ring (camera): a turn command on <= 2% of control ticks, pooled over bouts"}
ROOT = Path(__file__).resolve().parents[1]
def m1_files():   # everything a bout runs on: the whole package and configs (a hand-picked list missed policy.py, perturb.py, sensors.py)
    return sorted({p.relative_to(ROOT).as_posix() for g in ("r1gym/**/*.py", "configs/*.yaml") for p in ROOT.glob(g)} | {"scripts/face_ab.py", "results/opponent_tracker/policy.onnx"})
M1_PRE, M1_OUT = "results/perception/preregistration_m1.json", "results/perception/m1_offaxis_{}.json"
M1_SPLITS = dict(smoke=TRAIN_CLIPS[:1], val=VAL_CLIPS, test=TEST_CLIPS)
M1_SMOKE_DISCLOSURE = dict(when="before the preregistration was written (01:52Z)", clips=TEST_CLIPS, seconds=3.0, starts=1, preregistration="a scratch file, discarded",
                           seen="the printed criteria and summary of those bouts", used_for="nothing: the val choice and the test size come from the val file")   # this run only

def m1_tracked(p):
    """p is tracked in git and matches HEAD: the preregistration provably predates the test run in the history."""
    import subprocess; g = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True).returncode
    return g("ls-files", "--error-unmatch", p) == 0 and g("diff", "--quiet", "HEAD", "--", p) == 0

def sha(p): return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]   # LF-normalised: the same on every checkout

def m1_cfg(base, setting, tune):
    return {**base, **tune, "face_opponent": setting != "blind", "perception": {"blind": "truth", "truth_lag_nocomp": "truth_lag"}.get(setting, setting)}

def m1_bouts(c, base, m, eps, setting, tune, spec, hub=None, donors=None, threads=M1["threads"], T=None):
    """One setting on the episodes. Every env steps in lockstep (threads: physics and the policy at batch 1, so a bout never
    depends on which others still run); after each tick the hub renders the frames captured in it, in the main thread."""
    T = T or M1["seconds"]; cfg = m1_cfg(base, setting, tune); reps, X = {}, []
    for i, ep in enumerate(eps):
        k = (ep["clip"], ep["distance"])
        if k not in reps: reps[k] = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg)
        e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=T + 1, max_impulse=0, train=False); e.opp, e.opp_t0 = reps[k], ep["t0"]
        s = attach(e, cfg["perception"], spec=spec, seed=ep["pseed"], comp=setting != "truth_lag_nocomp", donor=donors[i] if donors else None) if cfg["perception"] != "truth" else None
        hm = HitMeter(m, cfg.get("hit_metric")); e.substep_hooks.append(hm); rng = np.random.default_rng(3)
        o = e.reset(hits=[make_hit(kd, l, imp, t, y, rng) for kd, l, imp, t, y in SCRIPTED], delay=1, init_yaw=ep["init_yaw"]); q = e.r.free_q
        X.append(dict(e=e, s=s, hm=hm, o=o, fell=None, start=e.d.qpos[q:q + 2].copy(), maxr=0.0, err=[float(e.true_bearing()[0])], wz=[float(e.command[2])]))
    def tick(g):
        for x in g:
            if x["fell"] is not None: continue
            e = x["e"]; x["o"], _, _, info = e.step(c.act(x["o"][None])[0]); q = e.r.free_q
            if info["fallen"]: x["fell"] = round(e.t * e.dt, 3); continue
            x["err"].append(float(e.true_bearing()[0])); x["wz"].append(float(e.command[2])); x["maxr"] = max(x["maxr"], float(np.linalg.norm(e.d.qpos[q:q + 2] - x["start"])))
    groups = [X[g::threads] for g in range(threads)]
    with ThreadPoolExecutor(threads) as pool:
        for _ in range(int(round(T / X[0]["e"].dt))):
            list(pool.map(tick, groups))
            if hub: hub.process([x["s"] for x in X if x["fell"] is None])
    return [m1_row(ep, x, T) for ep, x in zip(eps, X)], [x["s"].camera_log() if setting == "camera" else None for x in X]

def m1_row(ep, x, T):
    e, fell = x["e"], x["fell"]; err = np.degrees(np.abs(x["err"])); t = np.arange(len(err)) * e.dt; late = err[t > T - M1["last_s"] + 1e-9]; hit = np.flatnonzero(err <= M1["face_deg"]); q = e.r.free_q
    row = dict(clip=ep["clip"], distance=ep["distance"], start=ep["start"], bearing_deg=ep["bearing_deg"], fell_s=fell,
               face_last6_med_deg=None if fell else round(float(np.median(late)), 3), face_last6_p90_deg=None if fell else round(float(np.percentile(late, 90)), 3),
               ttf15_s=round(float(t[hit[0]]), 3) if len(hit) else None, drift_m=round(float(np.linalg.norm(e.d.qpos[q:q + 2] - x["start"])), 3), max_disp_m=round(x["maxr"], 3),
               ring_out=bool(x["maxr"] > 1.0), ticks=len(x["wz"]), turn_on=round(float(np.mean(np.abs(x["wz"]) > 0)), 4),
               hits={k: v for k, v in x["hm"].summary(fell or T)["n"].items() if k in ("head_landed", "head_clean", "torso_landed", "torso_clean")})
    if x["s"] is not None:   # at use: the bearing env._face acted on vs the true bearing at that tick
        u = np.degrees([abs(wrap(b - a)) for (_, b, _), a in zip(x["s"].used, x["err"]) if b is not None])
        row["percept"] = dict(x["s"].stats(), use_err_deg_median=round(float(np.median(u)), 3) if len(u) else None, use_err_deg_p90=round(float(np.percentile(u, 90)), 3) if len(u) else None)
    return row

def m1_summary(rows, T=None):
    T = T or M1["seconds"]; ok = [r for r in rows if r["fell_s"] is None]; f = np.array([r["face_last6_med_deg"] for r in ok]); ttf = [r["ttf15_s"] if r["ttf15_s"] is not None else T for r in ok]
    out = dict(bouts=len(rows), falls=len(rows) - len(ok), facing_err_last6_deg=dict(mean=round(float(f.mean()), 2), median=round(float(np.median(f)), 2), p90_median=round(float(np.median([r["face_last6_p90_deg"] for r in ok])), 2)) if ok else None,
               ttf15_s_mean=round(float(np.mean(ttf)), 3) if ok else None, faced_within_15deg=round(float(np.mean([r["ttf15_s"] is not None for r in ok])), 3) if ok else None,
               drift_m_mean=round(float(np.mean([r["drift_m"] for r in ok])), 3) if ok else None, ring_out=round(float(np.mean([r["ring_out"] for r in ok])), 3) if ok else None,
               turn_on=round(float(np.sum([r["turn_on"] * r["ticks"] for r in rows]) / np.sum([r["ticks"] for r in rows])), 4),
               landed_per_bout={k: round(float(np.mean([r["hits"][k] for r in rows])), 3) for k in rows[0]["hits"]})
    if "percept" in rows[0]:
        P = [r["percept"] for r in rows]; out["percept_bout_median"] = {k: (round(float(np.median([p[k] for p in P if p[k] is not None])), 4) if any(p[k] is not None for p in P) else None) for k in P[0]}
    return out

def m1_pair(A, B, key="face_last6_med_deg", T=None):
    """A - B over the bouts both survived, whole (clip, start) clusters resampled; ttf15: never faced = T."""
    T = T or M1["seconds"]; ix = [i for i, (a, b) in enumerate(zip(A, B)) if a["fell_s"] is None and b["fell_s"] is None]; v = lambda r: T if r[key] is None else r[key]
    d, ci = cluster_paired_diff([v(A[i]) for i in ix], [v(B[i]) for i in ix], [(A[i]["clip"], A[i]["start"]) for i in ix]); return dict(diff=d, ci95=list(ci), n=len(ix))

def m1_criteria(R, ring):
    P = lambda a, b, k="face_last6_med_deg": m1_pair(R[a], R[b], k); a_, b1, b2, c_, d_ = P("camera", "truth_lag"), P("blind", "camera"), P("blind", "truth_lag"), P("camera", "truth_lag", "ttf15_s"), P("blind", "shuffled")
    on = float(np.sum([r["turn_on"] * r["ticks"] for r in ring]) / np.sum([r["ticks"] for r in ring]))
    pm = lambda k: round(float(np.median([r["percept"][k] for r in ring if r["percept"][k] is not None])), 4)
    out = dict(a=dict(rule=M1_CRITERIA["a"], **a_, passed=bool(a_["ci95"][0] >= -3 and a_["ci95"][1] <= 3)), b=dict(rule=M1_CRITERIA["b"], camera=b1, truth_lag=b2, passed=bool(b1["ci95"][0] > 0 and b2["ci95"][0] > 0)),
               c=dict(rule=M1_CRITERIA["c"], **c_, passed=bool(c_["ci95"][1] <= 0.5)), d=dict(rule=M1_CRITERIA["d"], **d_, passed=bool(d_["ci95"][0] <= 0)),
               e=dict(rule=M1_CRITERIA["e"], turn_on=round(on, 4), bouts=len(ring), passed=bool(on <= 0.02), presence=pm("presence"), no_track=pm("no_track"), scope=M1_E_SCOPE))
    out["all_passed"] = all(v["passed"] for v in out.values()); return out

M1_E_SCOPE = ("the opponent is never in view (presence 0, no_track 1 when nothing is seen): (e) shows no false detections on the R1's own body and the empty "
              "room, not how the percept behaves after losing an opponent it had seen")

def m1_pairs(R, keys=("face_last6_med_deg", "ttf15_s", "drift_m", "max_disp_m")):
    """Every reported pair: the ladder's steps (timing, error, compensation) and each setting against blind."""
    pairs = [("truth_lag", "truth"), ("camera", "truth_lag"), ("camera", "truth"), ("truth_lag_nocomp", "truth_lag"), ("shuffled", "camera")] + [(s, "blind") for s in R if s != "blind"]
    return {f"{a} - {b}": {k: m1_pair(R[a], R[b], k) for k in keys} for a, b in pairs if a in R and b in R}

def derange(n, seed=0):
    """Donor bout for each bout, never itself: a seeded permutation, each bout taking the next one's stream."""
    o = np.random.default_rng(seed).permutation(n); d = np.empty(n, int); d[o] = np.roll(o, -1); return d

def m1_ladder(c, base, m, eps, settings, tune, spec, hub, have=None, logs=None):
    """Run the settings in order (camera before shuffled: its streams are the donors); have/logs: rows and camera streams already run."""
    R, L, cost = dict(have or {}), dict(logs or {}), {}
    for s in settings:
        if s in R: continue
        t0 = time.time(); f0 = hub.frames; don = [L["camera"][j] for j in derange(len(eps))] if s == "shuffled" else None
        R[s], lg = m1_bouts(c, base, m, eps, s, tune, spec, hub=hub if s == "camera" else None, donors=don)
        if s == "camera": L["camera"] = lg
        cost[s] = dict(wall_s=round(time.time() - t0, 1), frames=hub.frames - f0); print(s, json.dumps(m1_summary(R[s])), cost[s], flush=True)
    return R, L, cost

def m1_annotate():
    """--annotate: review follow-ups written into the stored test result without re-running a bout: (e)'s presence and
    no_track from the file's own empty-ring rows, and the pre-test smoke run disclosure. The preregistration and the val file
    stay byte-identical (the test file pins the one's sha, the preregistration the other's), so the disclosure lives here."""
    for split in ("test",):
        f = Path(M1_OUT.format(split)); r = json.load(open(f)); ring = r["raw"]["empty_ring"]
        for k in ("presence", "no_track"): r["criteria"]["e"][k] = round(float(np.median([x["percept"][k] for x in ring if x["percept"][k] is not None])), 4)
        r["criteria"]["e"]["scope"] = M1_E_SCOPE; r["about"] = M1_DOC
        r["pre_test_smoke_run"] = dict(M1_SMOKE_DISCLOSURE, added="after the test run, by --annotate (review); no number in this file changed")
        json.dump(r, open(f, "w"), indent=1); print("annotated", f, r["criteria"]["e"])

def m1_main(a):
    from r1gym.perception import run_meta
    from r1gym.perception.camera import HeadCameraSpec
    from r1gym.perception.render_scene import RenderTwin
    from r1gym.perception.detect_colorkey import ColorKeyDetector
    from r1gym.perception.hub import PerceptHub
    out = Path(a.out or M1_OUT.format(a.split)); pol = (a.policy or ["guard=results/policy_mjlab_guard/policy.onnx"])[0]; t_start = time.time()
    hashed = m1_files() + [p for s in (pol, a.arms) for p in (s.split("=", 1)[1], Path(s.split("=", 1)[1]).with_name("gym.json").as_posix())]
    if a.prereg:   # after val, before test: fixes the criteria, the statistics, the val choice and the code
        pre = Path(M1_PRE); v = json.load(open(M1_OUT.format("val")))
        if pre.exists(): sys.exit(f"{pre} exists: the test run is preregistered once")
        if v["split"] != "val" or v["seconds"] != M1["seconds"] or v["starts"] != OFFAXIS_STARTS: sys.exit("the val file is a smoke run")
        ci = v["criteria"]["a"]["ci95"]; hw = round((ci[1] - ci[0]) / 2, 3); nv = len({(r["clip"], r["start"]) for r in v["raw"]["main"]["camera"]}); nt = len(TEST_CLIPS) * M1["test_starts"]
        doc = dict(about="M1 preregistration (design: the R1 turns toward what its own camera sees). Written after the val run and before the test run; the "
                         "test run loads it, refuses to run if any file hashed below changed or its result exists, and evaluates the criteria below once.",
                   written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **run_meta(), criteria=M1_CRITERIA,
                   statistics="per bout: facing error = |true bearing of the opponent's pelvis from the R1's pelvis heading| (env.true_bearing) at every 20 ms tick; its median over "
                              "t > 6 s; time to face = the first tick with facing error <= 15 deg (never: 12 s). Paired differences: r1gym/stats.py cluster_paired_diff, clusters "
                              "(clip, start draw), 4000 resamples, seed 0, over the bouts both settings survived; empty ring: ticks with |wz command| > 0 / all ticks.",
                   choice=v["choice"], policy=v["policy"], arms_policy=v["arms_policy"], val_file_sha=sha(M1_OUT.format("val")), val_criteria={k: x["passed"] for k, x in v["criteria"].items() if k != "all_passed"},
                   power=dict(val_clusters=nv, val_ci_half_width_deg=hw, test_clusters=nt, expected_test_ci_half_width_deg=round(hw * np.sqrt(nv / nt), 2),
                              note=f"criterion (a)'s camera - truth_lag CI half-width on val, scaled by sqrt(clusters): why test draws {M1['test_starts']} starts per clip and distance, not the design's 6"),
                   test=dict(clips=TEST_CLIPS, distances=M1["distances"], bearings_deg=list(OFFAXIS_BEARINGS_DEG), starts=M1["test_starts"], seconds=M1["seconds"], seed=OFFAXIS_SEED,
                             ladder=list(M1_LADDER), arms=dict(settings=list(M1_ARMS), bearings_deg=[-30.0, -50.0]), ring=dict(distance_m=M1["ring_m"], bearing_deg=180.0)),
                   code_sha256={f: sha(f) for f in hashed})
        pre.parent.mkdir(parents=True, exist_ok=True); json.dump(doc, open(pre, "w"), indent=1); print("wrote", pre); return
    clips = M1_SPLITS[a.split]
    if a.split == "test":
        if not m1_tracked(M1_PRE): sys.exit(f"{M1_PRE} must be committed (tracked and unmodified) before the test run")
        pre = json.load(open(M1_PRE)); ph = {f.replace("\\", "/"): h for f, h in pre["code_sha256"].items()}   # the stored M1 preregistration keys the gym.json files with Windows separators
        bad = [f for f, h in ph.items() if sha(f) != h] + [f"{f} (not hashed)" for f in hashed if f not in ph]
        if a.starts is not None or a.seconds != pre["test"]["seconds"]: sys.exit("test runs at the preregistered size")
        a.starts = pre["test"]["starts"]
        if bad: sys.exit(f"changed since the preregistration: {bad}")
        if out.exists(): sys.exit(f"{out} exists: test runs once")
        if pol != pre["policy"] or a.arms != pre["arms_policy"]: sys.exit("not the preregistered policies")
    a.starts = a.starts or OFFAXIS_STARTS; M1["seconds"] = a.seconds; c = load_controller(pol); base = {**controller_cfg(c), "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=base).compile()
    spec = HeadCameraSpec(gym_cfg=base); tw = RenderTwin(base, physics=m, spec=spec); hub = PerceptHub(tw, ColorKeyDetector(spec, base))
    eps = offaxis_episodes(base, clips, M1["distances"], starts=a.starts); print(len(eps), "bouts", f"{time.time() - t_start:.0f}s", flush=True)
    res = dict(about=M1_DOC, **run_meta(gl=tw.gl()), split=a.split, clips=clips, policy=pol, arms_policy=a.arms, seconds=a.seconds, starts=a.starts, head_pitch_deg=spec.pitch_deg,
               detector={k: getattr(ColorKeyDetector, k) for k in ("HUE", "SAT", "VAL", "MIN_PX", "NEAR_PCT", "RANGE_SD")}, criteria_rules=M1_CRITERIA,
               episodes=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ep.items()} for ep in eps])
    have, logs = {}, {}
    if a.split == "val":   # tune percept_filter_s x face_deadband on the camera (truth_lag alongside, for reference)
        res["tuning"] = []; runs = {}
        for fs, db in itertools.product(M1_GRID["percept_filter_s"], M1_GRID["face_deadband"]):
            tune = dict(percept_filter_s=fs, face_deadband=db); rc, lc = m1_bouts(c, base, m, eps, "camera", tune, spec, hub=hub); rt, _ = m1_bouts(c, base, m, eps, "truth_lag", tune, spec)
            obj = float(np.mean([M1["fell_deg"] if r["fell_s"] is not None else r["face_last6_med_deg"] for r in rc])); runs[json.dumps(tune)] = (rc, lc)
            res["tuning"].append(dict(tune=tune, objective_deg=round(obj, 3), camera=m1_summary(rc), truth_lag=m1_summary(rt))); print(tune, round(obj, 3), json.dumps(m1_summary(rc)), f"{time.time() - t_start:.0f}s", flush=True)
        lo = min(x["objective_deg"] for x in res["tuning"]); width = lambda t: (0.0 if t["face_deadband"] is None else t["face_deadband"][0], -t["percept_filter_s"])   # wider deadband, then shorter filter
        tune = max((x for x in res["tuning"] if x["objective_deg"] <= lo + M1["tie_deg"]), key=lambda x: width(x["tune"]))["tune"]
        res["choice"] = dict(rule="lowest camera objective; within 0.5 deg of it, the widest deadband, then the shortest filter", **tune); have["camera"], logs["camera"] = runs[json.dumps(tune)]
    elif a.split == "smoke":   # no tuning: the val choice if there is one (a code check, never evidence)
        v = Path(M1_OUT.format("val")); tune = {k: json.load(open(v))["choice"][k] for k in ("percept_filter_s", "face_deadband")} if v.exists() else dict(percept_filter_s=0.0, face_deadband=None); res["choice"] = tune
    else: tune = {k: pre["choice"][k] for k in ("percept_filter_s", "face_deadband")}; res["choice"] = pre["choice"]; res["preregistration_sha"] = sha(M1_PRE)
    R, L, cost = m1_ladder(c, base, m, eps, M1_LADDER, tune, spec, hub, have=have, logs=logs)
    ring_eps = offaxis_episodes(base, clips, [M1["ring_m"]], bearings_deg=(180.0,), starts=a.starts); ring, _ = m1_bouts(c, base, m, ring_eps, "camera", tune, spec, hub=hub)
    c2 = load_controller(a.arms); base2 = {**controller_cfg(c2), "opponent_mode": "policy"}; arms_eps = offaxis_episodes(base2, clips, M1["distances"], bearings_deg=(-30.0, -50.0), starts=a.starts)
    RA, _, cost_a = m1_ladder(c2, base2, build_spec(opponent=True, cfg=base2).compile(), arms_eps, M1_ARMS, tune, spec, hub)
    res.update(summary={s: m1_summary(r) for s, r in R.items()}, paired=m1_pairs(R), criteria=m1_criteria(R, ring), empty_ring=dict(summary=m1_summary(ring), bouts=len(ring)),
               arms_down=dict(summary={s: m1_summary(r) for s, r in RA.items()}, paired=m1_pairs(RA), cost=cost_a),
               cost=dict(settings=cost, hub=hub.cost(), wall_s=round(time.time() - t_start, 1)), raw=dict(main=R, empty_ring=ring, arms_down=RA))
    tw.close(); out.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(dict(criteria=res["criteria"], summary=res["summary"]), indent=1), f"{res['cost']['wall_s']} s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="+", default=None, help="label=path; --offaxis: one (default the guard)")
    ap.add_argument("--out", default=None); ap.add_argument("--val", action="store_true"); ap.add_argument("--hold", action="store_true")
    ap.add_argument("--offaxis", action="store_true"); ap.add_argument("--split", choices=list(M1_SPLITS), default="val"); ap.add_argument("--prereg", action="store_true"); ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--arms", default="arms_down=results/policy_mjlab/policy.onnx", help="--offaxis: the right-turn-only row")
    ap.add_argument("--seconds", type=float, default=M1["seconds"]); ap.add_argument("--starts", type=int, default=None, help="--offaxis val / smoke: clean starts per clip and distance (default 6)"); a = ap.parse_args()
    if a.annotate: m1_annotate(); sys.exit(0)
    if a.offaxis: m1_main(a); sys.exit(0)
    a.policy = a.policy or ["arms_down=results/policy_mjlab/policy.onnx", "guard=results/policy_mjlab_guard/policy.onnx"]
    S = HOLD if a.hold else SETTINGS; a.out = a.out or ("results/hold_ab.json" if a.hold else "results/face_opponent_ab.json")
    if a.val:   # merge validation survival into the existing file
        sys.path.insert(0, str(Path(__file__).resolve().parent)); res = json.load(open(a.out)); res["about"] = __doc__; res.setdefault("val_survival", {})
        for spec in a.policy:
            lab, path = spec.split("=", 1); c = load_controller(f"{lab}={path}"); base = {**controller_cfg(c), "opponent_mode": "policy"}
            res["val_survival"][lab] = val_survival(c, base, build_spec(opponent=True, cfg=base).compile(), S); print(lab, res["val_survival"][lab], flush=True)
        json.dump(res, open(a.out, "w"), indent=1); sys.exit(0)
    res = {"about": __doc__, "turn_response_cmd_to_measured_rad_s": {}, "bouts": {}}
    for spec in a.policy:
        lab, path = spec.split("=", 1); c = load_controller(f"{lab}={path}"); base = {**controller_cfg(c), "opponent_mode": "policy"}
        m = build_spec(opponent=True, cfg=base).compile()
        if not a.hold: res["turn_response_cmd_to_measured_rad_s"][lab] = turn_response(c, base); print(lab, res["turn_response_cmd_to_measured_rad_s"][lab], flush=True)
        clips, dists = TRAIN_CLIPS + VAL_CLIPS, [0.55, 0.65]; good = clean_starts(base, clips, dists)
        eps = [(cl, d, pick_start(good[(cl, d)], u)) for cl in clips for d in dists for u in np.linspace(0, 5.9, 6)]
        res["bouts"][lab] = {k: bouts(c, {**base, **v}, m, eps) for k, v in S.items()}
        print(lab, json.dumps(res["bouts"][lab]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
