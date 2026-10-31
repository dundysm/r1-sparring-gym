"""Headline eval: survive a physical R1 opponent (r1gym/opponent.py; default: the tracking-policy opponent on its own motors)
performing held-out fight clips.

Tiers (--tier):
  standard  10 s; opponent 0.50-0.70 m away; two scripted side/back pushes the opponent can't deliver (4-6 N*s).
  hard      20 s; opponent 0.40-0.55 m away; six scripted strikes and pushes from any direction at any hit location
            (8-14 N*s); the R1's mass, COM, foot friction, motor strength, armature and command latency randomised per
            episode (the training ranges, r1gym/env.py). Set on the validation clip once the stepping policies
            saturated the standard tier (guard 1.000, arms-down 0.994 on test).
  hard_r2   the hard tier on fresh episodes for sparring M2 run 2: seeds 21000+ on val, 22000+ on test (per split); its test
            split is refused until scripts/m2_verdict.py --run 2 select wrote go = true (results/sparring/m2_run2_selection.json).
            --run 3 (sparring M2 run 3, prereg_m2_run3.json): the same tier and seeds, the guard reads m2_run3_selection.json;
            a hard_r2 test runs once across runs (refused if a raw_test_hard_r2 file exists anywhere under results/eval_opponent
            outside --out).
Each episode samples the clip start time, the obs delay (0-2 steps) and the hits. The start time is drawn uniformly from
the clean ones: those where no part of the opponent begins inside the R1 in either reference pose (REF_POSES), which
skips 7-85% of starts in the stored (shell-contact) runs, by clip and distance (meta.start_times_skipped). Every controller sees exactly the same
episodes, so differences are paired, and the episodes don't depend on which controllers are in the run. --split test
(default) uses TEST_CLIPS, --split val uses VAL_CLIPS: choose candidates on val, report on test.
--hits 1 (default) also scores the opponent's landed hits per bout (r1gym/hits.py HitMeter, read-only: survival is the same
with it off) into each summary's 'landed' and 'sensitivity' blocks; definitions preregistered in docs/AIMED_OPPONENT.md. With
the tracked opponent its strike scheduler then runs as 'shadow' (r1gym/aim.py: it logs strike intents and changes nothing, bit
for bit), so window outcomes can be scored: the library's strike windows shifted by the tracker's lag (blocked share), and the
intent windows an aimed run is paired on. These v1 tiers keep their episodes (meta.benchmark_version 'v1').
--aim MODE (or 'chosen': results/opponent_tracker/aim.json) runs the same tier's episodes with the aimed opponent and pairs
every controller with the unaimed run of the tier (its raw file must exist): differences in survival, hits and window
outcomes, with (clip, start) cluster CIs, in each summary's 'vs_unaimed'. Output: summary_{split}{tier tag}_aim.json.
meta records the git commit, a config hash, the preregistration's hash, the shell contact solref and the opponent's caps.
Added keys (sparring; earlier keys and values unchanged): per bout the opponent's first fall flag (opp_fall: time, cause 'tilt' /
'height' / both, and the robot-robot contacts of the 0.5 s before it by body pair and zone, impulse and peak: ContactTail) and
both-up exposure: the control steps the R1 was live (not yet fallen at the step's start) AND the opponent upright at the step's
end (the same OPP_TILT / OPP_Z test, per step, not sticky: an opponent flagged and back up counts again), as (t_a, t_b] intervals
(both_up_iv) and their total (both_up_s); with --hits, head / torso clean strikes whose onset falls inside them per minute of
them (landed.*_pm_both_up, with landed.minutes); summary.opponent_falls pools the fall logs. opp_fell (stored) stays sticky. --rev_hits 1 also scores the R1's own strikes on the opponent
(HitMeter reverse=True) into 'rev_meter' per bout and 'landed_rev' per summary.
--keypoints truth|camera (sparring M2, docs/SPARRING_M1M2.md section 4): controllers with a widened 111-d input get the 28-d
opponent keypoint block (r1gym/keypoints.py KeypointSource, one per bout, delayed by the bout's obs delay) appended after their
83 inputs; a blind twin (observation_names ending opp_keypoints_blind) gets 28 zeros whatever the flag says. 83-d controllers
and every row are unchanged; meta gains 'keypoints' only when the flag is set. A sighted widened policy without it is an error."""
import sys, json, time, argparse, itertools, copy, hashlib, subprocess, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from r1gym.env import R1HitEnv, VAL_CLIPS, TEST_CLIPS, _DR_NOMINAL
from r1gym.model import build_spec, load_cfg, HOME, CTRL_JOINTS, SHELL_BIT
from r1gym.hits import HitMeter, landed_block, sensitivity_block, r1_zone, opp_limb
from r1gym.opponent import FightReplay
from r1gym.perturb import make_hit, LOCATIONS
from r1gym.policy import PDStand, load_controller, controller_cfg
from r1gym.keypoints import KeypointSource, KP_DIM, KP_NAMES
from r1gym.stats import wilson, paired_diff, cluster_paired_diff
from r1gym.scenario import clean_starts, pick_start, REF_POSES, START_STEP, START_SPAN

TIERS = {"standard": dict(distances=[0.5, 0.55, 0.6, 0.65, 0.7], seconds=10.0, hits=2, impulse=(4, 6), dr=False, seed=10_000),
         "hard": dict(distances=[0.4, 0.45, 0.5, 0.55], seconds=20.0, hits=6, impulse=(8, 14), dr=True, seed=20_000)}
# Sparring M2 run 2 (docs/SPARRING_M1M2.md section 9, results/sparring/prereg_m2_run2.json): the hard tier on fresh episodes. The
# clips are the same (val kfab_204, test kfab_202 / 205: no other held-out clip exists, kfab_199 is not performable), the seeds
# are new and differ between the splits (val 21000+, test 22000+), so start times, obs delays, DR and hit draws are new. Run 1's
# val episodes (seed 20000) were spent on its diagnosis; run 1's test (seed 20000) was run once for run 1's record. The test
# split of this tier runs only after run 2's val screen said go (r2_test_allowed).
TIERS["hard_r2"] = dict(TIERS["hard"], seed=21_000, seed_by_split=dict(val=21_000, test=22_000))
R2_SELECTION = Path(__file__).resolve().parents[1] / "results" / "sparring" / "m2_run2_selection.json"
R2_PREREG = Path(__file__).resolve().parents[1] / "results" / "sparring" / "prereg_m2_run2.json"
# run 3 (prereg_m2_run3.json) takes over hard_r2's unused episodes (run 2 ended in training, no eval ran): --run 3 points the test
# guard at its selection and prereg; the test split of hard_r2 runs once across runs (r2_test_spent)
R_FILES = {2: (R2_SELECTION, R2_PREREG), 3: (R2_SELECTION.with_name("m2_run3_selection.json"), R2_PREREG.with_name("prereg_m2_run3.json"))}
def r2_test_spent(out, root=Path(__file__).resolve().parents[1] / "results" / "eval_opponent"):
    """hard_r2 test raw files anywhere under root (searched recursively) outside `out` itself (a test split spent by another --out)."""
    return sorted(str(p) for p in Path(root).rglob("raw_test_hard_r2*.json") if p.parent.resolve() != Path(out).resolve())
def text_sha16(path):
    """sha256[:16] of a text file, LF-normalised (the same on every checkout); scripts/m2_verdict.py writes the same hash."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
def r2_test_allowed(selection=None, prereg=None, run=2):
    """(ok, why): run `run`'s test split may run only once m2_verdict.py --run <run> select wrote go = true against the current
    prereg (its LF-normalised hash). Defaults: R2_SELECTION, R2_PREREG, run 2 (the run number only names it in the messages)."""
    selection, prereg = Path(selection or R2_SELECTION), Path(prereg or R2_PREREG)
    if not Path(selection).exists(): return False, f"{selection} missing: run scripts/m2_verdict.py --run {run} select on val first"
    try: S = json.loads(Path(selection).read_bytes())
    except ValueError as e: return False, f"{selection} unreadable: {e}"
    if S.get("go") is not True: return False, f"{selection}: go is {S.get('go')!r} (run {run} stopped at the val screen: no test run)"
    if not Path(prereg).exists(): return False, f"{prereg} missing"
    if S.get("prereg_sha256_16") != text_sha16(prereg): return False, f"{selection} was made against prereg hash {S.get('prereg_sha256_16')}, {prereg} is now {text_sha16(prereg)}"
    return True, "go"
RING_M = 1.0       # the R1's pelvis got more than 1 m from where it started (pushed or stepped): out of the sparring ring
OPP_TILT, OPP_Z = 0.5, 0.25   # opponent fell: its pelvis up-axis z below 0.5 (tilt) or its pelvis below 0.25 m (height)
TAIL_S = 0.5       # robot-robot contacts kept before the opponent's fall
AIM_JSON = Path(__file__).resolve().parents[1] / "results" / "opponent_tracker" / "aim.json"   # M1's operator and the tracker's lag
def episodes(clips, reps, tier, good, hits_on=True, seconds=None, split=None):
    T = TIERS[tier]; out = []; S = seconds or T["seconds"]; T = dict(T, seed=T.get("seed_by_split", {}).get(split, T["seed"]))   # hard_r2: per split
    for k, (clip, dist, r) in enumerate(itertools.product(clips, T["distances"], range(reps))):
        rng = np.random.default_rng(T["seed"] + k); hits = []; ep = {}
        if hits_on and tier == "standard":
            for t in sorted(rng.uniform(1.0, 9.0, 2)):
                yaw = rng.choice([np.pi / 2, -np.pi / 2, np.pi]) + rng.uniform(-0.3, 0.3)   # side or from behind
                hits.append(("push", str(rng.choice(["l_shoulder", "r_shoulder", "pelvis", "chest"])), float(rng.uniform(4, 6)), float(t), float(yaw)))
        elif hits_on:   # strikes and pushes, any direction (yaw = where the R1 is pushed toward), any location incl. head and knees
            for t in sorted(rng.uniform(1.0, S - 1.0, T["hits"])):
                hits.append((str(rng.choice(["strike", "push"])), str(rng.choice(list(LOCATIONS))), float(rng.uniform(*T["impulse"])), float(t), float(rng.uniform(-np.pi, np.pi))))
        if T["dr"]:   # the env (DR, sensor noise) and the hit durations draw from their own streams, independent of the hits
            ep = dict(env_seed=int(np.random.SeedSequence([T["seed"] + k, 1]).generate_state(1)[0]), hit_seed=int(np.random.SeedSequence([T["seed"] + k, 2]).generate_state(1)[0]))
        t0 = pick_start(good[(clip, dist)], float(rng.uniform(0, 6)))
        out.append(dict(clip=clip, distance=dist, rep=r, t0=t0, delay=int(rng.integers(0, 3)), hits=hits, seed=T["seed"] + k, **ep))
    return out

class ContactTail:
    """Read-only substep hook: robot-robot contacts of the last TAIL_S, normal force per (R1 body, its group, opponent body, its
    zone) per substep, labelled per geom with HitMeter(reverse=True)'s maps (the opponent's head shells sit on opp_torso_link: by
    body alone a hand on the head reads 'torso'); geoms outside those maps fall back to opp_limb / r1_zone of the body. snap()
    sums them (impulse = force * timestep, peak, last touch before now): whether the R1 had touched the opponent, and with what,
    just before it fell. Once done is set (run() sets it at the opponent's first fall flag, after snap) it logs nothing more."""
    def __init__(self, m, keep_s=TAIL_S):
        self.dt, self.keep = float(m.opt.timestep), keep_s; self.f6 = np.zeros(6); nb = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self.opp = np.array([n.startswith("opp_") for n in nb]); self.r1 = np.array([m.geom_bodyid[g] != 0 and not nb[g].startswith("opp_") for g in range(m.ngeom)])
        hm = HitMeter(m, reverse=True)   # reversed: limb_of holds the R1's geoms, zone_of the opponent's
        self.lab = {int(g): (nb[g], hm.limb_of.get(g, opp_limb(nb[g])[1])) if self.r1[g] else (nb[g][4:], hm.zone_of.get(g, r1_zone(nb[g][4:]))) for g in np.flatnonzero(self.r1 | self.opp)}
        self.reset()
    def reset(self): self.log = []; self.done = False
    def substep(self, m, d):
        if self.done or d.ncon == 0: return
        g1, g2 = d.contact.geom1, d.contact.geom2; ix = np.flatnonzero((self.r1[g1] & self.opp[g2]) | (self.r1[g2] & self.opp[g1]))
        if not len(ix): return
        acc = {}
        for i in ix:
            a, b = int(g1[i]), int(g2[i]); r, o = (a, b) if self.r1[a] else (b, a); mujoco.mj_contactForce(m, d, int(i), self.f6); k = (*self.lab[r], *self.lab[o])
            acc[k] = acc.get(k, 0.0) + abs(float(self.f6[0]))
        t = float(d.time); self.log.append((t, acc))
        while self.log and self.log[0][0] <= t - self.keep - 1e-9: self.log.pop(0)
    def snap(self, now):
        P = {}
        for t, acc in self.log:
            if t <= now - self.keep - 1e-9: continue
            for k, f in acc.items(): p = P.setdefault(k, [0.0, 0.0, t]); p[0] += f * self.dt; p[1] = max(p[1], f); p[2] = t
        pairs = sorted(({"r1": r, "opp": o, "r1_limb": rl, "opp_zone": oz, "impulse_Ns": round(i, 4), "peak_N": round(pk, 1), "last_s_before": round(now - t, 3)}
                        for (r, rl, o, oz), (i, pk, t) in P.items()), key=lambda x: -x["impulse_Ns"])
        return dict(contact=bool(pairs), impulse_Ns=round(sum(x["impulse_Ns"] for x in pairs), 4), n_pairs=len(pairs), pairs=pairs[:8],
                    r1_strike=any(x["r1_limb"] in ("hand", "forearm") and x["opp_zone"] in ("head", "torso") for x in pairs))

def meter_row(e, ep, secs, sp, lag, both_up=None):
    """The HitMeter's per-bout summary, scored on the library's strike windows (each occurrence in the bout, shifted by the
    tracker's lag) and, when the opponent runs an Aim (shadow or aimed), on its intent windows; plus the first contact time.
    both_up: the (t_a, t_b] intervals both robots were up (adds the both-up exposure keys, r1gym/hits.py)."""
    hm, op = e.substep_hooks[0], e.opp; rate = op.fps * op.speed; T = op.T; k0 = int(np.floor(ep["t0"] * rate / T)) - 1
    lib = [((k * T + s.onset) / rate - ep["t0"] + lag, (k * T + s.end) / rate - ep["t0"] + lag, s.limb) for s in op.strikes for k in range(k0, int(np.ceil((secs + ep["t0"]) * rate / T)) + 1)]
    out = hm.summary(secs, sp, [(round(a, 4), round(b, 4), l) for a, b, l in lib if a >= 0 and b <= secs], both_up=both_up); ev = hm.events()
    out["first_contact_s"] = round(min(x["t0"] for x in ev), 3) if ev else None
    aim = e.opp_state.get("aim")
    if aim is not None:
        tail = aim.p["window_tail_s"]; its = [it for it in aim.intents if it.t_on >= 0 and it.t_apex + tail <= secs]; o = hm.windows([it.window(tail) for it in its])
        out.update(intent_windows={k: o.count(k) for k in ("head", "torso", "blocked", "miss")}, intents_gated=sum(bool(it.gated) for it in its),
                   intent_closest_cm=[round(100 * it.closest_m, 1) for it in its])
    return out

def run(ctrl, eps, m, cfg, T=10.0, threads=8, dr=False, meter=False, lag=0.0, rev=False, falls=False, keypoints=None):
    # meter: main() turns it on (--hits); other callers skip its ~15%. falls: the opponent-fall log and both-up exposure
    # (main() turns it on; off, the rows are exactly the earlier ones). rev: the R1's strikes on the opponent (--rev_hits)
    # keypoints: the opponent keypoint source ('truth' | 'camera') for a widened controller (kp_mode 'obs'); it reads the
    # env's MjData only, on its own twin, so the bout is the same bout; kp_mode 'zeros' appends 28 zeros; None: nothing new runs
    # DR envs write their sampled parameters into the model before each step, so with DR each thread steps its own copy
    ms = [copy.deepcopy(m) for _ in range(threads)] if dr else [m] * threads
    reps = {}; envs = []; kpm = getattr(ctrl, "kp_mode", None)
    if kpm == "obs" and keypoints is None: raise ValueError(f"{ctrl.name} sees opponent keypoints (observation_names ...,opp_keypoints): pass --keypoints")
    for i, ep in enumerate(eps):
        key = (ep["clip"], ep["distance"])
        if key not in reps: reps[key] = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg)
        e = R1HitEnv(seed=ep.get("env_seed", ep["seed"]), cfg=cfg, model=ms[i % threads], episode_s=T, max_impulse=0, train=False, dr=dr)
        e.opp = reps[key]; e.opp_t0 = ep["t0"]; rng = np.random.default_rng(ep.get("hit_seed", ep["seed"]))
        if meter: e.substep_hooks.append(HitMeter(e.m, cfg.get("hit_metric")))   # read-only: logs contacts after every mj_step
        tail = ContactTail(e.m) if falls else None   # after the HitMeter: meter_row reads hooks[0]
        if tail is not None: e.substep_hooks.append(tail)
        rv = HitMeter(e.m, cfg.get("hit_metric"), reverse=True) if rev else None
        if rv is not None: e.substep_hooks.append(rv)
        hits = [make_hit(k, l, imp, t, y, rng) for k, l, imp, t, y in ep["hits"]]
        o = e.reset(hits=hits, delay=ep["delay"]); q = e.r.free_q
        envs.append(dict(e=e, o=o, fell=None, at=None, start=e.d.qpos[q:q + 2].copy(), maxr=0.0, opp_fell=False, sp=[], tail=tail, rv=rv, ofall=None, bu=[]))
        if kpm == "obs":   # after e.reset, the same delay k as the 83 proprioceptive dims, zero noise
            envs[-1]["src"] = KeypointSource(e.m, keypoints, delay=ep["delay"]); envs[-1]["kp"] = envs[-1]["src"].reset(e.d)
        elif kpm == "zeros": envs[-1]["kp"] = np.zeros(KP_DIM, np.float32)
    obs = (lambda x: x["o"]) if kpm is None else (lambda x: np.concatenate([x["o"], x["kp"]]))   # widened: 83 + 28 = 111
    groups = [list(range(g, len(envs), threads)) for g in range(threads)]; pool = ThreadPoolExecutor(threads)
    for _ in range(int(T / envs[0]["e"].dt)):
        live = [i for i in range(len(envs)) if envs[i]["fell"] is None]
        if not live: break
        A = dict(zip(live, ctrl.act(np.stack([obs(envs[i]) for i in live]))))
        def st(g):   # one thread per model copy: its envs step one after another
            for i in g:
                if i not in A: continue
                x = envs[i]; e = x["e"]; x["o"], r, d, info = e.step(A[i]); q = e.r.free_q; xy = e.d.qpos[q:q + 2]
                if "src" in x: x["kp"] = x["src"].observe(e.d)   # right after the step, as the env's own obs
                x["maxr"] = max(x["maxr"], float(np.linalg.norm(xy - x["start"])))
                if meter: x["sp"].append(float(np.linalg.norm(e.d.qpos[e.opp.qadr:e.opp.qadr + 2] - xy)))   # actual pelvis spacing
                b = min(e.opp.bodies)   # opponent pelvis; deep stances reach z ~0.37, so judge by tilt
                tl, ht = bool(e.d.xmat[b][8] < OPP_TILT), bool(e.d.xpos[b][2] < OPP_Z)
                if tl or ht:   # an opponent on the floor is no threat
                    x["opp_fell"] = True
                    if falls and x["ofall"] is None:   # the first time: when, why, and the robot-robot contacts just before
                        x["ofall"] = dict(t_s=round(e.t * e.dt, 3), cause="tilt+height" if tl and ht else "tilt" if tl else "height", up_z=round(float(e.d.xmat[b][8]), 3),
                                          pelvis_z=round(float(e.d.xpos[b][2]), 3), r1_contact_0p5s=x["tail"].snap(float(e.d.time))); x["tail"].done = True
                elif falls:   # both up over this step (the R1 live at its start, the opponent upright at its end): extend or open an interval
                    iv = x["bu"]
                    if iv and iv[-1][1] == e.t - 1: iv[-1][1] = e.t
                    else: iv.append([e.t, e.t])
                if info["fallen"]: x["fell"] = e.t * e.dt; x["at"] = xy.copy()   # where it was when it fell
        list(pool.map(st, groups))
    pool.shutdown()
    if dr:   # the copies die here; drop their cached nominals so a later model can't pick up a stale entry by a reused id()
        for x in ms: _DR_NOMINAL.pop(id(x), None)
    # drift: how far the R1's pelvis is from its start (xy) at the end (or where it fell), and the ring-out rate. The tracked
    # opponent is steered after the R1; the puppet stays where it was placed, so there stepping back also leaves its reach.
    rows = []
    for ep, x in zip(eps, envs):
        q = x["e"].r.free_q; p = (x["at"] if x["at"] is not None else x["e"].d.qpos[q:q + 2]) - x["start"]
        secs = x["fell"] or T; dt = x["e"].dt; bu = [((k0 - 1) * dt, k1 * dt) for k0, k1 in x["bu"]]   # both up: (t_a, t_b] in s, control steps k0..k1
        rows.append(dict(clip=ep["clip"], distance=ep["distance"], fell_s=x["fell"], opp_fell=x["opp_fell"], drift_m=round(float(np.linalg.norm(p)), 3),
                         back_m=round(float(-p[0]), 3), max_disp_m=round(x["maxr"], 3), ring_out=x["maxr"] > RING_M,
                         **({"hit_meter": meter_row(x["e"], ep, secs, x["sp"], lag, bu if falls else None)} if meter else {}),
                         **(dict(opp_fall=x["ofall"], both_up_s=round(sum(k1 - k0 + 1 for k0, k1 in x["bu"]) * dt, 3), both_up_iv=[[round(a, 3), round(c, 3)] for a, c in bu]) if falls else {}),
                         **({"rev_meter": x["rv"].summary(secs, both_up=bu if falls else None)} if rev else {})))
    return rows

def summarise(rows, T, clusters=None, shell_solref=None):
    s = [x["fell_s"] is None for x in rows]; n = len(s); k = int(sum(s))
    ring = [x["fell_s"] is None and not x["ring_out"] for x in rows]
    ttf = [x["fell_s"] if x["fell_s"] is not None else T for x in rows]
    by_d = {d: round(float(np.mean([x["fell_s"] is None for x in rows if x["distance"] == d])), 3) for d in sorted({x["distance"] for x in rows})}
    surv = [x for x in rows if x["fell_s"] is None]
    return dict(episodes=n, survival=round(k / n, 3), survival_ci95=wilson(k, n), mean_time_to_fall_s=round(float(np.mean(ttf)), 2), survival_by_distance=by_d,
                survival_in_ring=round(float(np.mean(ring)), 3), survival_in_ring_ci95=wilson(int(sum(ring)), n),
                survivors_ring_out=round(float(np.mean([x["ring_out"] for x in surv])), 3) if surv else None,
                survivors_mean_drift_m=round(float(np.mean([x["drift_m"] for x in surv])), 3) if surv else None,
                survivors_mean_back_m=round(float(np.mean([x["back_m"] for x in surv])), 3) if surv else None,
                survivors_back_over_0p3m=round(float(np.mean([x["back_m"] > 0.3 for x in surv])), 3) if surv else None,
                opponent_fell=round(float(np.mean([x["opp_fell"] for x in rows])), 3),
                **({"landed": landed_block([x["hit_meter"] for x in rows], clusters, shell_solref), "sensitivity": sensitivity_block([x["hit_meter"] for x in rows], clusters),
                    **aim_block([x["hit_meter"] for x in rows])} if rows and "hit_meter" in rows[0] else {}),
                **({"opponent_falls": fall_block(rows, T)} if rows and "opp_fall" in rows[0] else {}),
                **({"landed_rev": landed_block([x["rev_meter"] for x in rows], clusters, shell_solref)} if rows and "rev_meter" in rows[0] else {}))

BOTH_UP_DEF = "control steps the R1 was live (not fallen at the step's start) and the opponent upright at the step's end (up-axis z >= 0.5 and pelvis >= 0.25 m), per step"

def fall_block(rows, T):
    """The opponent's first fall flags (rows with opp_fall): count, time quantiles, cause, how many came within 0.5 s of a
    robot-robot contact or of an R1 hand / forearm on its head or torso, the median contact impulse over those with contact,
    how many were upright again later in the bout (got_up: a both-up interval after the flag); minutes the R1 was up and
    minutes both robots were up (BOTH_UP_DEF)."""
    F = [x["opp_fall"] for x in rows if x["opp_fall"]]; t = [f["t_s"] for f in F]; c = [f["r1_contact_0p5s"] for f in F]; ci = [x["impulse_Ns"] for x in c if x["contact"]]
    return dict(n=len(F), of=len(rows), t_s=dict(zip(("p10", "median", "p90"), np.round(np.percentile(t, [10, 50, 90]), 2).tolist())) if t else None,
                cause={k: sum(f["cause"] == k for f in F) for k in ("tilt", "height", "tilt+height")}, after_contact_0p5s=sum(x["contact"] for x in c),
                after_r1_strike_0p5s=sum(x["r1_strike"] for x in c), contact_impulse_Ns_median=round(float(np.median(ci)), 3) if ci else None,
                got_up=sum(any(a >= x["opp_fall"]["t_s"] - 1e-9 for a, _ in x["both_up_iv"]) for x in rows if x["opp_fall"]),
                r1_up_min=round(sum(x["fell_s"] if x["fell_s"] is not None else T for x in rows) / 60, 3), both_up_min=round(sum(x["both_up_s"] for x in rows) / 60, 3), both_up_def=BOTH_UP_DEF)

def aim_block(hm):
    """Intent windows (shadow or aimed opponent): outcome totals, head share, gated share, closest palm-to-target; and the share
    of bouts with robot-robot contact in the first 0.5 s (aiming must not start bouts in contact)."""
    out = dict(early_contact_share=round(float(np.mean([h["first_contact_s"] is not None and h["first_contact_s"] < 0.5 for h in hm])), 3))
    if "intent_windows" in hm[0]:
        W = {k: sum(h["intent_windows"][k] for h in hm) for k in hm[0]["intent_windows"]}; n = sum(W.values()); cm = [c for h in hm for c in h["intent_closest_cm"]]
        out["intent_windows"] = dict(W, n=n, head_share=round(W["head"] / n, 3) if n else None, gated_share=round(sum(h["intents_gated"] for h in hm) / n, 3) if n else None,
                                     closest_cm_median=round(float(np.median(cm)), 1) if cm else None)
    return out

def vs_unaimed(a, b, clusters):
    """Aimed minus unaimed on identical episodes: survival, opponent falls, the primary (head clean strikes/min) and every hit rule,
    exposure-weighted, and the head share of intent windows; (clip, start) cluster bootstrap."""
    D = lambda x, y, wa=None, wb=None: dict(zip(("diff", "ci95"), cluster_paired_diff(x, y, clusters, wa=wa, wb=wb)))
    out = dict(survival=D([float(r["fell_s"] is None) for r in a], [float(r["fell_s"] is None) for r in b]), opponent_fell=D([float(r["opp_fell"]) for r in a], [float(r["opp_fell"]) for r in b]))
    if "hit_meter" in a[0] and "hit_meter" in b[0]:
        H, G = [r["hit_meter"] for r in a], [r["hit_meter"] for r in b]; wa, wb = [h["secs"] / 60 for h in H], [h["secs"] / 60 for h in G]
        out.update({k: D([h[k] for h in H], [h[k] for h in G], wa, wb) for k in ("head_clean_pm", "head_landed_pm", "head_impulse_pm", "torso_clean_pm", "torso_landed_pm")})
        out["rules"] = {r: {z: D([h["rules"][r][i] for h in H], [h["rules"][r][i] for h in G], wa, wb) for i, z in enumerate(("head", "torso"))} for r in H[0]["rules"]}
        if "intent_windows" in H[0] and "intent_windows" in G[0]:
            share = lambda h: h["intent_windows"]["head"] / max(1, sum(h["intent_windows"].values())); n = lambda h: max(1, sum(h["intent_windows"].values()))
            out["intent_window_head_share"] = D([share(h) for h in H], [share(h) for h in G], [n(h) for h in H], [n(h) for h in G])
    return out

def shell_solref(m):   # robot-vs-robot shell contact time constant: peak forces scale with it (impulse doesn't)
    sh = [g for g in range(m.ngeom) if m.geom_conaffinity[g] & SHELL_BIT]; return m.geom_solref[sh[0]].round(4).tolist() if sh else None

def provenance(m, cfg, ccfg=None):
    """What a result was computed with: commit (and whether the working tree had uncommitted changes, untracked files included), hashes of the base config,
    of each controller's config and of the preregistration (docs/AIMED_OPPONENT.md), the MuJoCo version, the robot-vs-robot
    shell contact solref (peak forces depend on it) and the opponent's torque caps."""
    import mujoco
    root = Path(__file__).resolve().parents[1]
    def git(*a):   # None when git fails (no repo, a lock held by another process): unknown, not 'clean'
        r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True); return r.stdout.strip() if r.returncode == 0 else None
    h = lambda b: hashlib.sha256(b).hexdigest()[:16]; hc = lambda c: h(json.dumps(c, sort_keys=True, default=str).encode()); pre = root / "docs" / "AIMED_OPPONENT.md"
    oa = [m.actuator("opp_" + j.replace("_joint", "")).id for j in CTRL_JOINTS]
    caps = "mjlab" if cfg.get("opponent_mode") == "policy" else cfg.get("opponent_cap_policy")   # the tracker always runs mjlab caps (r1gym/model.py)
    st = git("status", "--porcelain")   # untracked files count: a new, uncommitted module is not in git_hash either
    mh = git("rev-parse", "-q", "--verify", "MERGE_HEAD")   # an uncommitted merge in progress: the other parent
    return dict(git_hash=git("rev-parse", "HEAD"), git_dirty=None if st is None else bool(st), **({"git_merge_head": mh} if mh else {}), mujoco=mujoco.__version__,
                config_hash=hc(cfg), controller_config_hashes={n: hc(c) for n, c in (ccfg or {}).items()}, prereg_sha=h(pre.read_bytes().replace(b"\r\n", b"\n")) if pre.exists() else None,   # LF-normalised: same on every checkout
                prereg_rules_sha=h(pre.read_bytes().replace(b"\r\n", b"\n").split(b"\n## Results")[0]) if pre.exists() else None,   # without the Results section (added after the first M1 runs)
                shell_solref=shell_solref(m),
                opponent_caps=dict(policy=caps, Nm={j.replace("_joint", ""): float(m.actuator_forcerange[a, 1]) for j, a in zip(CTRL_JOINTS, oa)}),
                hit_metric=cfg.get("hit_metric"))

def pairs(raw, summ):
    """Paired differences on the same episodes between every two controllers (and vs PD): survival, and up and in the ring."""
    up = {n: [x["fell_s"] is None for x in r] for n, r in raw.items()}
    ring = {n: [x["fell_s"] is None and not x["ring_out"] for x in r] for n, r in raw.items()}
    for name in raw:
        summ[name]["survival_minus"] = {o: dict(zip(["diff", "ci95"], paired_diff(up[name], up[o]))) for o in raw if o != name}
        summ[name]["survival_in_ring_minus"] = {o: dict(zip(["diff", "ci95"], paired_diff(ring[name], ring[o]))) for o in raw if o != name}
        if name != "pd_stand": summ[name]["survival_minus_pd"], summ[name]["survival_minus_pd_ci95"] = paired_diff(up[name], up["pd_stand"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", nargs="*", default=["v1=results/policy_v1/policy.pt", "v2=results/policy/policy.pt", "v3=results/policy_v3_candidate/policy.pt"],
                    help="label=path (.pt residual policy, .onnx unitree_rl_mjlab policy)")
    ap.add_argument("--split", choices=["test", "val"], default="test"); ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--tier", choices=list(TIERS), default="standard")
    ap.add_argument("--opponent", choices=["tracked", "puppet"], default="tracked",
                    help="tracked: the R1 on its own motors (default, realistic); puppet: assisted replay (stress test, up to 250 N horizontal assist)")
    ap.add_argument("--no_pushes", action="store_true"); ap.add_argument("--seconds", type=float, default=None); ap.add_argument("--out", default="results/eval_opponent")
    ap.add_argument("--hits", type=int, default=1, help="1: score landed hits per bout (r1gym/hits.py; read-only, survival unchanged); 0: off")
    ap.add_argument("--aim", default=None, help="aimed opponent (r1gym/aim.py mode, or 'chosen': results/opponent_tracker/aim.json), paired with this tier's unaimed raw file")
    ap.add_argument("--threads", type=int, default=8, help="threads stepping the bouts (results don't depend on it)")
    ap.add_argument("--rev_hits", type=int, default=0, help="1: also score the R1's strikes on the opponent (HitMeter reverse=True, read-only)")
    ap.add_argument("--from_raw", action="store_true", help="recompute summary_*.json from the stored raw_*.json (no simulation)")
    ap.add_argument("--run", type=int, choices=sorted(R_FILES), default=2, help="hard_r2 test guard: 2 prereg_m2_run2.json, 3 prereg_m2_run3.json")
    ap.add_argument("--keypoints", choices=["truth", "camera"], default=None, help="opponent keypoints for widened (sparring M2) controllers: truth "
                    "(exact state, mask = in the head camera's view) or camera (M3, not implemented); default off: no controller may need them")
    a = ap.parse_args()
    if a.tier == "hard_r2" and a.split == "test":   # run 2's / 3's test: only after its val screen said go (its prereg), once across runs
        ok, why = r2_test_allowed(*R_FILES[a.run], run=a.run)
        if not ok: sys.exit(f"--tier hard_r2 --split test refused: {why}")
        if r2_test_spent(a.out): sys.exit(f"--tier hard_r2 --split test refused: already run into {r2_test_spent(a.out)}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tier = TIERS[a.tier]; T = a.seconds or tier["seconds"]
    cfg = load_cfg(); models = {}; tag = ("" if a.tier == "standard" else f"_{a.tier}") + ("" if a.opponent == "tracked" else "_puppet") + ("_nohits" if a.no_pushes else "")
    if a.from_raw:
        st = json.load(open(out / f"raw_{a.split}{tag}.json")); raw = st["raw"]; cl = [(e["clip"], e["t0"]) for e in st["episodes"]]
        summ = {n: summarise(r, st["meta"]["seconds"], cl, st["meta"].get("shell_solref")) for n, r in raw.items()}; pairs(raw, summ)
        json.dump(dict(meta=st["meta"], summary=summ), open(out / f"summary_{a.split}{tag}.json", "w"), indent=1); return
    if a.opponent == "puppet": cfg["opponent_mode"] = "puppet"
    aimj = json.load(open(AIM_JSON)) if AIM_JSON.exists() else {}; lag = float(aimj.get("lag_s", 0.0)) if a.opponent == "tracked" else 0.0; aim = aimj["operator"] if a.aim == "chosen" else a.aim
    if aim and a.opponent != "tracked": sys.exit("--aim needs the tracked opponent")
    cfg["opponent_aim"] = aim or ("shadow" if a.hits and a.opponent == "tracked" else "none"); base_tag = tag; tag += "_aim" if aim else ""
    if a.tier == "hard_r2" and a.split == "test" and (out / f"raw_test{tag}.json").exists():   # run once (prereg_m2_run{a.run}.json)
        sys.exit(f"--tier hard_r2 --split test refused: {out / f'raw_test{tag}.json'} exists (run {a.run}'s test runs once; --from_raw re-summarises)")
    clips = TEST_CLIPS if a.split == "test" else VAL_CLIPS
    ctrls = [PDStand()]
    for lp in a.policy:
        lab, path = lp.split("=", 1)
        if Path(path).exists(): ctrls.append(load_controller(f"RL {lab}={path}"))
    need = [c.name for c in ctrls if getattr(c, "kp_mode", None) == "obs"]
    if need and not a.keypoints: sys.exit(f"{need} see opponent keypoints: pass --keypoints truth")
    if a.keypoints == "camera": sys.exit("--keypoints camera: the gloves-detector source comes with M3 (r1gym/keypoints.py raises NotImplementedError); use --keypoints truth")   # before any bout runs
    ccfg = {c.name: {**controller_cfg(c), "opponent_mode": cfg["opponent_mode"], "opponent_aim": cfg["opponent_aim"]} for c in ctrls}   # own contract per controller
    poses = dict(REF_POSES)
    for n, cc in ccfg.items():   # a pose outside REF_POSES changes every controller's episodes: allowed, but loudly
        h = np.array(cc.get("home", HOME), float)
        if not any(np.allclose(h, p, atol=1e-3) for p in poses.values()):
            print(f"WARNING: {n}'s default pose is not in REF_POSES; adding it changes the episodes, so this run is not paired with stored results", flush=True)
            poses[n] = h
    good = clean_starts({**cfg, "opponent_aim": "none"}, clips, tier["distances"], poses); eps = episodes(clips, a.reps, a.tier, good, not a.no_pushes, T, split=a.split)   # judged on the unwarped clip
    raw, summ = {}, {}; clusters = [(e["clip"], e["t0"]) for e in eps]   # bootstrap clusters: (clip, start)
    for c in ctrls:
        cc = ccfg[c.name]; key = str(getattr(c, "cfg_path", "default"))
        if key not in models: models[key] = build_spec(opponent=True, cfg=cc).compile()
        t0 = time.time(); raw[c.name] = run(c, eps, models[key], cc, T, threads=a.threads, dr=tier["dr"], meter=bool(a.hits), lag=lag, rev=bool(a.rev_hits), falls=True, keypoints=a.keypoints); summ[c.name] = summarise(raw[c.name], T, clusters, shell_solref(models[key]))
        print(c.name, json.dumps(summ[c.name]), f"({time.time() - t0:.0f} s)", flush=True)
    pairs(raw, summ)
    if aim:   # pair with the unaimed run of this tier: same episodes, every controller
        ref = out / f"raw_{a.split}{base_tag}.json"
        if not ref.exists(): sys.exit(f"--aim pairs with {ref}: run this tier unaimed first")
        R = json.load(open(ref)); assert R["episodes"] == json.loads(json.dumps(eps)), f"{ref} holds other episodes"
        for name in raw:
            if name in R["raw"]: summ[name]["vs_unaimed"] = vs_unaimed(raw[name], R["raw"][name], clusters)
    skipped = {f"{c}@{d}": round(1 - float(np.mean(g)), 3) for (c, d), g in good.items()}
    meta = dict(tier=a.tier, tier_cfg=tier, opponent=a.opponent, split=a.split, clips=clips, reps=a.reps, pushes=not a.no_pushes, seconds=T,
                opponent_mode=cfg.get("opponent_mode", "puppet"), ring_m=RING_M, start_poses=list(poses), start_times_skipped=skipped,
                hit_meter=bool(a.hits), benchmark_version="v1" + (f"+aim:{aim}" if aim else ""), opponent_aim=cfg["opponent_aim"], strike_lag_s=lag,
                **({"keypoints": dict(source=a.keypoints, kp_dim=KP_DIM, names=list(KP_NAMES), contract="docs/SPARRING_M1M2.md 1.2-1.4, 4",
                                      kp_mode={c.name: c.kp_mode for c in ctrls if getattr(c, "kp_mode", None)})} if a.keypoints else {}),
                **({"aim": dict(operator=aim, from_aim_json=a.aim == "chosen", aim_json=aimj or None, paired_with=f"raw_{a.split}{base_tag}.json", paired_with_git=R["meta"].get("git_hash"))} if aim else {}),
                **provenance(next(iter(models.values())), cfg, ccfg))
    json.dump(dict(meta=meta, summary=summ), open(out / f"summary_{a.split}{tag}.json", "w"), indent=1)
    json.dump(dict(meta=meta, episodes=eps, raw=raw), open(out / f"raw_{a.split}{tag}.json", "w"))
    print(json.dumps(summ, indent=1))

if __name__ == "__main__": main()
