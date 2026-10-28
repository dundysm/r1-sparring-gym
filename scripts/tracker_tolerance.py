"""How much reference change the shipped tracker (results/opponent_tracker/policy.onnx) tolerates, unretrained, with the
opponent alone (the R1 standing 4 m away, as in scripts/eval_tracker.py) -> results/opponent_tracker/tolerance.json.
Train + val clips only (test is never looked at here).

lag        reference-to-tracked apex delay: per library hand strike (unwarped), the time of the palm's most forward point (toward
           the R1) within [-0.1, +0.3] s of the reference apex, minus the reference apex. Median on train is the tracker's lag,
           frozen in results/opponent_tracker/aim.json and used to shift library windows (docs/AIMED_OPPONENT.md).
arm warps  per hand strike, the striking arm's 5 reference joints are changed so the reference palm's apex moves 10-25 cm up,
           sideways toward the centreline, or forward (in the reference torso frame; r1gym/aim.py IK without the 0.8 rad cap),
           blended in and out with ArmWarp's weight. Closest approach of the palm, in its own torso frame, to what the warped
           reference asks for (the palm of the warped arm at the apex), share within 5 cm, and opponent falls (paired with 0 cm).
lunges     the steering spot is pulled L m toward the R1 (speed cap v, lag 0.1 s) from 0.15 s before onset to the strike's end:
           the palm's extra reach toward the R1, paired with no lunge, and falls.
retiming   the clip clock runs at rate s inside every hand strike (+-0.1 s, 0.1 s ramps; reference velocities scaled), 1 outside,
           uncompensated (the loop gets longer): falls in 48 runs of 10 s (12 starts per train / val clip), against s = 1.
select     r1gym/aim.py 'select' (splices to head-level strikes) on the same 48 runs: falls. It is a candidate for the M1 operator
           only if it falls in at most one run more than the unspliced runs (docs/AIMED_OPPONENT.md, amendment 1).
B's design probe measured (rg/p4_warp.py, p5_lunge.py): within 5 cm 72.7 / 65.2 / 42.4% at 0 / 10 / 20 cm up; a 20 cm lunge at
1 m/s added 1.8 cm of reach."""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from concurrent.futures import ThreadPoolExecutor
from r1gym.model import build_spec, load_cfg
from r1gym.env import R1HitEnv, TRAIN_CLIPS, VAL_CLIPS
from r1gym.opponent import FightReplay
from r1gym.aim import Aim, ARM, StrikeIntent, merged
from r1gym import strikes as S

AWAY, PRE, POST = 4.0, 1.0, 1.0   # opponent distance; run from 1 s before onset to 1 s after the apex
MAGS, DIRS = [0.10, 0.15, 0.20, 0.25], {"up": (0, 0, 1), "in": (0, -1, 0), "fwd": (1, 0, 0)}   # 'in': toward the centreline (sign per arm)
LUNGES = {"l10_v03": (0.10, 0.3), "l10_v06": (0.10, 0.6), "l20_v06": (0.20, 0.6), "l20_v10": (0.20, 1.0), "l30_v10": (0.30, 1.0)}
RATES = [0.5, 0.7]

def fell(d, op): return d.xmat[op.pel][8] < 0.5 or d.xpos[op.pel][2] < 0.25

class Op:
    """A tolerance probe in the aim slot of Tracker.act: an open-loop arm change, a lunge, or a retimed clock."""
    def __init__(self, s, rate, dq=None, lunge=None, retime=None, t0=0.0):
        self.s, self.rate, self.dq, self.lunge, self.retime = s, rate, dq, lunge, retime; self.a = Aim.__new__(Aim); self.a.p = merged(None); self.clip_t = None
        if s is not None: self.it = StrikeIntent(s.limb, "head", s.clip, s.onset, PRE, PRE + s.rise_s, PRE + (s.end - s.onset) / rate, np.zeros(3), 0.0)
    def current(self, now): return None
    def clock(self, t, replay):
        if self.retime is None: return int(round(t * self.rate))
        if self.clip_t is None: self.clip_t = t
        else: self.clip_t += 0.02 * self.k
        fc = (self.clip_t * self.rate) % replay.T; self.k = 1 - (1 - self.retime[0]) * self.retime[1](fc)   # rate for the next step
        return int(round(self.clip_t * self.rate))
    def reference(self, tr, m, d, replay, fi, state, t, dt):
        rq, rdq, apos, aori = tr.reference(replay, tr.frame(replay.clip, fi), state); now = d.time + dt
        if self.retime is not None: return rq, rdq * self.k, apos, aori
        if self.dq is not None:
            w = self.a.weight(self.it, now); dw = (self.a.weight(self.it, now + 1e-4) - self.a.weight(self.it, now - 1e-4)) / 2e-4
            ix = ARM[self.s.limb]; rq, rdq = rq.copy(), rdq.copy(); rq[ix] += w * self.dq; rdq[ix] += dw * self.dq
        if self.lunge is not None:
            L, v = self.lunge; on = self.it.t_on - 0.15 <= now <= self.it.t_end
            state["dist"] = AWAY - (L if on else 0.0); state["steer_max"] = v; state["steer_tau"] = 0.1 if on else tr.STEER_TAU
        return rq, rdq, apos, aori

def bumps(strikes, T, rate, pad=0.1, ramp=0.1):
    """Clip frame -> 1 inside a hand strike window (+-pad), 0 outside, smoothstep ramps of `ramp` s at the edges (circular)."""
    x = np.zeros(T); f = np.arange(T)
    for s in strikes:
        for k in (-1, 0, 1):
            a, b = s.onset - pad * rate + k * T, s.end + pad * rate + k * T; r = ramp * rate
            u = np.clip(np.minimum((f - a + r) / r, (b + r - f) / r), 0, 1); x = np.maximum(x, u * u * (3 - 2 * u))
    return lambda fc: float(np.interp(fc, f, x))

def strike_run(m, cfg, s, op_kw, palm_target=None):
    rep = FightReplay(m, s.clip, distance=AWAY, cfg=cfg); rate = rep.fps * rep.speed; op = Op(s, rate, **op_kw)
    e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=PRE + s.rise_s + POST + 1, max_impulse=0, train=False); e.opp = rep; e.opp_t0 = s.t_on - PRE
    e.reset(hits=[], delay=0); e.opp_state["aim"] = op; side = "left" if s.limb == "l_hand" else "right"; palm = m.site(f"opp_{side}_palm").id
    t_apex = PRE + s.rise_s; best, fwd, fall = 9.9, [], False
    for k in range(int((PRE + s.rise_s + POST) / e.dt)):
        e.step(np.zeros(24)); d = e.d; now = d.time
        if fell(d, rep): fall = True; break
        if t_apex - 0.1 <= now <= t_apex + 0.3: fwd.append((now, -float(d.site_xpos[palm][0])))   # toward the R1 (it stands at the origin)
        if palm_target is not None and PRE - 0.2 <= now <= t_apex + 0.3:
            Rt = d.xmat[rep.torso].reshape(3, 3); best = min(best, float(np.linalg.norm(Rt.T @ (d.site_xpos[palm] - d.xpos[rep.torso]) - palm_target)))
    tf = max(fwd, key=lambda x: x[1]) if fwd else (None, None)
    return dict(fell=fall, lag_s=None if tf[0] is None else round(tf[0] - t_apex, 3), reach_m=None if tf[1] is None else round(tf[1] + AWAY, 4), closest_m=round(best, 4))

def warp_dq(a, tr, rep, s, vec):
    """Joint change on the striking arm moving the reference palm at the apex by vec (reference torso frame); the palm it asks for."""
    rq = tr.ref_q[tr.frame(rep.clip, s.apex)].copy(); p0 = a._palm_b(rq, s.limb)[0]; qa, res = a.ik(rq, s.limb, p0 + np.asarray(vec, float))
    q = rq.copy(); q[ARM[s.limb]] = qa; return qa - rq[ARM[s.limb]], a._palm_b(q, s.limb)[0], res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default="results/opponent_tracker/tolerance.json"); a = ap.parse_args()
    import eval_opponent as eo
    cfg = {**load_cfg(), "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=cfg).compile(); t0 = time.time(); pool = ThreadPoolExecutor(a.threads)
    lib = [s for s in S.load() if s.hand and s.split in ("train", "val")]
    ik = Aim(m, FightReplay(m, lib[0].clip, distance=AWAY, cfg=cfg), "armwarp", {"arm": {"max_dq": 3.2}}); tr = FightReplay(m, lib[0].clip, distance=AWAY, cfg=cfg).tracker
    jobs = []
    for s in lib:
        rep = FightReplay(m, s.clip, distance=AWAY, cfg=cfg); sgn = 1.0 if s.limb == "l_hand" else -1.0   # +y is the opponent's left: 'in' is -y for its left hand
        _, p0, _ = warp_dq(ik, tr, rep, s, (0, 0, 0)); jobs.append((s, "0", 0.0, dict(), p0, 0.0))
        for dn, v in DIRS.items():
            for mag in MAGS:
                dq, p, res = warp_dq(ik, tr, rep, s, mag * np.array([v[0], sgn * v[1], v[2]])); jobs.append((s, dn, mag, dict(dq=dq), p, res))
        for ln, lv in LUNGES.items(): jobs.append((s, ln, 0.0, dict(lunge=lv), p0, 0.0))
    rows = list(pool.map(lambda j: dict(clip=j[0].clip, split=j[0].split, onset=j[0].onset, limb=j[0].limb, cond=j[1], mag=j[2], ik_res_m=round(j[5], 4),
                                        **strike_run(m, cfg, j[0], j[3], j[4])), jobs)); print(f"strike runs: {len(rows)} ({time.time() - t0:.0f} s)", flush=True)
    key = lambda r: (r["clip"], r["onset"], r["limb"]); base = {key(r): r for r in rows if r["cond"] == "0"}
    tr_lag = [r["lag_s"] for r in rows if r["cond"] == "0" and r["split"] == "train" and not r["fell"]]; va_lag = [r["lag_s"] for r in rows if r["cond"] == "0" and r["split"] == "val" and not r["fell"]]
    q = lambda x, p: round(float(np.percentile(x, p)), 3)
    lag = dict(train_median_s=float(np.median(tr_lag)), train_p10_p90_s=[q(tr_lag, 10), q(tr_lag, 90)], val_median_s=float(np.median(va_lag)) if va_lag else None, n_train=len(tr_lag), n_val=len(va_lag))
    def warp(sel):
        ok = [r for r in sel if not r["fell"]]
        return dict(n=len(sel), fell=sum(r["fell"] for r in sel), fell_unwarped_same=sum(base[key(r)]["fell"] for r in sel), within_5cm=round(float(np.mean([r["closest_m"] <= 0.05 for r in ok])), 3) if ok else None,
                    closest_cm_median=round(100 * float(np.median([r["closest_m"] for r in ok])), 1) if ok else None, ik_res_cm_median=round(100 * float(np.median([r["ik_res_m"] for r in sel])), 1))
    curves = {"0": warp([r for r in rows if r["cond"] == "0"])}
    for dn in DIRS:
        for mag in MAGS: curves[f"{dn}{int(round(mag * 100))}"] = warp([r for r in rows if r["cond"] == dn and r["mag"] == mag])
    for mag in MAGS: curves[f"any{int(round(mag * 100))}"] = warp([r for r in rows if r["cond"] in DIRS and r["mag"] == mag])
    lunges = {}
    for ln, (L, v) in LUNGES.items():
        sel = [r for r in rows if r["cond"] == ln]; ok = [(r, base[key(r)]) for r in sel if not r["fell"] and not base[key(r)]["fell"]]; g = [r["reach_m"] - b["reach_m"] for r, b in ok]
        lunges[ln] = dict(lunge_m=L, vmax_mps=v, n=len(sel), fell=sum(r["fell"] for r in sel), fell_unlunged_same=sum(base[key(r)]["fell"] for r in sel),
                          extra_reach_cm_median=round(100 * float(np.median(g)), 1), extra_reach_cm_p10_p90=[round(100 * q(g, 10), 1), round(100 * q(g, 90), 1)])
    # 48 runs of 10 s: 12 starts per train / val clip, as eval_tracker (opponent alone)
    clips = TRAIN_CLIPS + VAL_CLIPS; starts = [(c, i * (FightReplay(m, c, distance=AWAY, cfg=cfg).T / 50.0) / 12) for c in clips for i in range(12)]
    def long_run(c, t0_, mode, retime=None):
        cc = {**cfg, "opponent_aim": mode}; rep = FightReplay(m, c, distance=AWAY, cfg=cc); e = R1HitEnv(seed=0, cfg=cc, model=m, episode_s=11, max_impulse=0, train=False)
        e.opp = rep; e.opp_t0 = t0_; e.reset(hits=[], delay=0)
        if retime is not None: e.opp_state["aim"] = Op(None, rep.fps * rep.speed, retime=(retime, bumps([s for s in rep.strikes if s.hand], rep.T, rep.fps)))
        for k in range(500):
            e.step(np.zeros(24))
            if fell(e.d, rep): return dict(clip=c, t0=round(t0_, 3), fell_s=round((k + 1) * e.dt, 2), splices=getattr(e.opp_state.get("aim"), "splices", None))
        return dict(clip=c, t0=round(t0_, 3), fell_s=None, splices=getattr(e.opp_state.get("aim"), "splices", None))
    runs = {}
    for name, mode, rt in [("unchanged", "none", None)] + [(f"retime_{r}", "none", r) for r in RATES] + [("select", "select", None)]:
        runs[name] = list(pool.map(lambda x: long_run(*x, mode, rt), starts)); print(name, sum(r["fell_s"] is not None for r in runs[name]), "falls", f"({time.time() - t0:.0f} s)", flush=True)
    nf = lambda n: sum(r["fell_s"] is not None for r in runs[n]); f0 = nf("unchanged")
    long = {n: dict(runs=len(v), fell=nf(n), extra_vs_unchanged=nf(n) - f0, fell_unchanged_same=sum(r["fell_s"] is not None for r, b in zip(v, runs["unchanged"]) if b["fell_s"] is not None)) for n, v in runs.items()}
    long["select"].update(splices_per_run_median=float(np.median([r["splices"] for r in runs["select"]])), passes=nf("select") <= f0 + 1)
    out = dict(about=__doc__, meta=dict(eo.provenance(m, cfg), threads=a.threads, wall_s=round(time.time() - t0, 1)), strikes=len(lib), lag=lag, arm_warps=curves, lunges=lunges, long_runs=long,
               design_B_probe=dict(within_5cm={"0": 0.727, "up10": 0.652, "up20": 0.424}, lunge_20cm_1mps_extra_reach_cm=1.8), rows=rows, long_run_rows=runs)
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "long_run_rows", "about", "meta")}, indent=1))

if __name__ == "__main__": main()
