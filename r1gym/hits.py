"""Landed-hit metric (read-only): which part of the opponent hit which part of the R1, how hard and how fast.

A HitMeter rides in R1HitEnv.substep_hooks: after every mujoco.mj_step it sums the normal force of each robot-vs-robot
contact per (side, opponent limb group, R1 zone) and, when a pair first touches, the closing speed along the contact
normal. It never writes MjData. After mj_step, d.contact, efc_force, cvel and xipos all come from that step's forward pass
(the state the contact was detected in), so the closing speed is the speed before this step's contact force acted, and
force * timestep is the impulse the step applied.
Definitions are preregistered in docs/AIMED_OPPONENT.md, numbers in configs/r1_assumptions.yaml (hit_metric):
  contact episode  per (opponent limb group, R1 zone): substeps in contact at most merge_gap_s apart merged (a contact-free
                   gap of up to merge_gap_s - 5 ms); duration = first to last substep, both included
  landed           a strike limb (hand / forearm / foot / shin) on the head or torso, impulse >= min_impulse_Ns and
                   peak >= min_peak_N
  clean strike     landed, closing speed at onset >= v_close_min and duration <= max_strike_s; a press is any other landed
  window outcome   head | torso | blocked (the limb touched the R1's arms before its head or torso) | miss
  rate             count per minute until the R1 fell; over bouts, total count / total minutes (exposure-weighted)
RULES re-score the same contacts with the design and review probes' other definitions (docs/AIMED_OPPONENT.md,
Sensitivity): the sign of an aim effect depended on which one was used, so every summary carries all of them."""
from __future__ import annotations
import numpy as np, mujoco

ZONES = ("head", "torso", "pelvis", "arm", "leg")
LIMBS = ("hand", "forearm", "upper_arm", "foot", "shin", "thigh", "trunk")
STRIKE_LIMBS = ("hand", "forearm", "foot", "shin")
DEFAULTS = dict(merge_gap_s=0.1, min_impulse_Ns=0.5, min_peak_N=5.0, v_close_min=1.0, max_strike_s=0.15)
HEAD_MESHES = ("head_pitch_link", "head_yaw_link")   # fixed on torso_link: the real R1's 2-DoF neck is rigid in this MJCF

def r1_zone(body, name=""):
    """name: the mesh for a shell, the geom's own name otherwise (robot_contact: capsule has one head capsule on torso_link)."""
    if name in HEAD_MESHES or name == "head_collision": return "head"
    if body == "torso_link": return "torso"           # waist_yaw shell (torso_collision with robot_contact: capsule)
    if body in ("pelvis", "waist_roll_link"): return "pelvis"
    if any(k in body for k in ("shoulder", "elbow", "wrist")): return "arm"
    if any(k in body for k in ("hip", "knee", "ankle")): return "leg"
    return None

def opp_limb(body):
    """Opponent body name without the opp_ prefix -> (side, limb group)."""
    side = "l" if body.startswith("left") else "r" if body.startswith("right") else ""
    for k, g in (("wrist", "hand"), ("elbow", "forearm"), ("shoulder", "upper_arm"), ("ankle", "foot"), ("knee", "shin"), ("hip", "thigh")):
        if k in body: return side, g
    return side, "trunk"                              # pelvis, waist, torso and head: a bump or a clinch, not a strike

# name: (merge gap s, force floor N, which episodes count, zone renames). clean/landed are the preregistered rules (their
# gap comes from the config); A_peak20 is design A's actual counting, B_strike design B's strike rule, impulse_only both
# designs' stated impulse rule, each as the review probe counted it (B_strike's 0.15 s is last minus first substep, so 31
# substeps, where clean's duration includes both ends: 30). Unlike the probe, rates end only when the R1 falls (the probe
# also stopped at the opponent's fall) and closing speed is taken at the strongest point, about the body's COM.
_B = STRIKE_LIMBS + ("thigh",)
RULES = {"clean": (None, 0.0, lambda e, p: _landed(e, p) and _clean(e, p), {}),
         "landed": (None, 0.0, lambda e, p: _landed(e, p), {}),
         "A_peak20": (0.02, 5.0, lambda e, p: e["limb"] in STRIKE_LIMBS and e["peak_N"] >= 20.0, {"pelvis": "torso"}),
         "B_strike": (0.1, 0.0, lambda e, p: e["limb"] in _B and (e["v_close"] or 0.0) >= 1.0 and e["t1"] - e["t0"] <= 0.15 + 1e-9 and e["peak_N"] >= 5.0, {}),
         "impulse_only": (0.1, 0.0, lambda e, p: e["limb"] in _B and e["impulse_Ns"] >= 0.5, {})}

def _struck(e, p):   # a strike limb's contact hard enough to count, on any zone
    return bool(e["limb"] in STRIKE_LIMBS and e["impulse_Ns"] >= p["min_impulse_Ns"] and e["peak_N"] >= p["min_peak_N"])
def _landed(e, p): return e["zone"] in ("head", "torso") and _struck(e, p)
def _clean(e, p): return bool((e["v_close"] or 0.0) >= p["v_close_min"] and e["dur_s"] <= p["max_strike_s"] + 1e-9)

class HitMeter:
    def __init__(self, m, params=None):
        self.dt = float(m.opt.timestep); self.p = {**DEFAULTS, **(params or {})}; self.v6 = np.zeros(6); self.f6 = np.zeros(6)
        self.zone_of, self.limb_of, self.keys = {}, {}, {}; self.is_r1, self.is_opp = np.zeros(m.ngeom, bool), np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]; bn = m.body(b).name
            if b == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
            if bn.startswith("opp_"): self.keys[g] = opp_limb(bn[4:]); self.limb_of[g] = self.keys[g][1]; self.is_opp[g] = True
            else:
                z = r1_zone(bn, m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else m.geom(g).name)
                if z: self.zone_of[g] = z; self.is_r1[g] = True
        self.reset()

    def reset(self):
        self.log = []; self.prev = set(); self._ev = {}

    def substep(self, m, d):
        """Hook: call right after mujoco.mj_step. Logs (t, {(side, limb, zone): normal force N}, {key: closing m/s at first touch})."""
        if d.ncon == 0: self.prev = set(); return
        g1, g2 = d.contact.geom1, d.contact.geom2
        hit = np.flatnonzero((self.is_r1[g1] & self.is_opp[g2]) | (self.is_r1[g2] & self.is_opp[g1]))
        if not len(hit): self.prev = set(); return
        acc, vc, best = {}, {}, {}
        for i in hit:
            a, b = int(g1[i]), int(g2[i]); o, r = (a, b) if a in self.keys else (b, a)
            k = (*self.keys[o], self.zone_of[r]); mujoco.mj_contactForce(m, d, int(i), self.f6); f = abs(float(self.f6[0]))
            acc[k] = acc.get(k, 0.0) + f
            if k not in self.prev and f >= best.get(k, -1.0):   # first touch: closing speed at the strongest contact point
                best[k] = f; vc[k] = self._vclose(m, d, int(i), a, b)
        self.prev = set(acc); self.log.append((round(d.time, 4), acc, vc)); self._ev = {}

    def _vclose(self, m, d, i, a, b):
        p, n = d.contact.pos[i], d.contact.frame[i, :3]; v = []   # normal points from geom1 to geom2: (v1 - v2) . n > 0 closes
        for g in (a, b):
            bb = m.geom_bodyid[g]; mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bb, self.v6, 0)
            v.append(self.v6[3:] + np.cross(self.v6[:3], p - d.xipos[bb]))
        return round(float((v[0] - v[1]) @ n), 3)

    def events(self, gap=None, f_on=0.0, by_side=False):
        """Contact episodes, per (limb, zone) or per (side, limb, zone): substeps with force >= f_on at most gap apart merged
        (so a contact-free gap of up to gap - dt)."""
        gap = self.p["merge_gap_s"] if gap is None else gap; ck = (gap, f_on, by_side)
        if ck in self._ev: return self._ev[ck]
        open_, out = {}, []
        for t, acc, vc in self.log:
            agg = {}
            for k, f in acc.items():
                a = agg.setdefault(k if by_side else k[1:], [0.0, set(), None]); a[0] += f; a[1].add(k[0])
                if k in vc: a[2] = vc[k] if a[2] is None else max(a[2], vc[k])
            for k, (f, sides, v) in agg.items():
                if f < f_on: continue
                e = open_.get(k)
                if e is None or t - e["t1"] > gap + 1e-9:
                    e = open_[k] = dict(limb=k[-2], zone=k[-1], side="", t0=t, t1=t, peak_N=0.0, impulse_Ns=0.0, v_close=None); out.append(e)
                e["t1"] = t; e["peak_N"] = max(e["peak_N"], f); e["impulse_Ns"] += f * self.dt; e["side"] = "".join(sorted(set(e["side"]) | sides))
                if e["v_close"] is None and v is not None: e["v_close"] = v
        for e in out:
            e["dur_s"] = round(e["t1"] - e["t0"] + self.dt, 4); e["peak_N"] = round(float(e["peak_N"]), 1); e["impulse_Ns"] = round(float(e["impulse_Ns"]), 4)
        self._ev[ck] = out; return out

    def landed(self, e): return _landed(e, self.p)
    def clean(self, e): return _landed(e, self.p) and _clean(e, self.p)

    def count(self, rule):
        """{zone: episodes counted} under one of RULES."""
        gap, f_on, ok, ren = RULES[rule]; c = {}
        for e in self.events(gap, f_on):
            if ok(e, self.p): z = ren.get(e["zone"], e["zone"]); c[z] = c.get(z, 0) + 1
        return c

    def windows(self, wins):
        """Outcome per strike window (t0, t1, limb), limb as 'l_hand' / 'r_foot' / 'hand': head | torso | blocked | miss.
        Blocked: inside the window the limb touched the R1's arms before it landed on the head or torso."""
        out = []; ev = self.events(by_side=True)
        for t0, t1, limb in wins:
            side, g = limb.split("_", 1) if limb[:2] in ("l_", "r_") else ("", limb)
            mine = sorted((e for e in ev if e["limb"] == g and (not side or side in e["side"]) and t0 <= e["t0"] <= t1), key=lambda e: e["t0"])
            hit = next((e for e in mine if _landed(e, self.p)), None); arm = next((e for e in mine if e["zone"] == "arm"), None)
            out.append("blocked" if arm and (hit is None or arm["t0"] < hit["t0"]) else hit["zone"] if hit else "miss")
        return out

    def summary(self, secs, spacing=None, wins=None):
        """Per-bout fields for scripts/eval_opponent.py: rates per minute of bout (until the R1 fell), per-hit lists, every rule."""
        mins = max(secs, 1e-9) / 60; ev = self.events(); pm = lambda x: round(x / mins, 3)
        landed = [e for e in ev if self.landed(e)]; clean = [e for e in landed if self.clean(e)]
        n = dict(head_landed=sum(e["zone"] == "head" for e in landed), head_clean=sum(e["zone"] == "head" for e in clean),
                 torso_landed=sum(e["zone"] == "torso" for e in landed), torso_clean=sum(e["zone"] == "torso" for e in clean),
                 arm_clean=sum(e["zone"] == "arm" and _struck(e, self.p) and _clean(e, self.p) for e in ev), episodes=len(ev))
        out = dict(secs=round(secs, 3), n=n, **{f"{k}_pm": pm(n[k]) for k in ("head_landed", "head_clean", "torso_landed", "torso_clean", "arm_clean")},
                   head_impulse_pm=pm(sum(e["impulse_Ns"] for e in ev if e["zone"] == "head" and e["limb"] in STRIKE_LIMBS)),
                   head_hit_imp_Ns=[e["impulse_Ns"] for e in landed if e["zone"] == "head"], hit_peak_N=[e["peak_N"] for e in landed],
                   rules={r: [pm(c.get("head", 0)), pm(c.get("torso", 0))] for r in RULES for c in [self.count(r)]})
        if spacing is not None and len(spacing): out["spacing_m"] = [round(float(x), 3) for x in np.percentile(spacing, [10, 50, 90])]
        if wins is not None:
            o = self.windows(wins); out["windows"] = {k: o.count(k) for k in ("head", "torso", "blocked", "miss")}
        return out

def _q(x, ps):
    return {f"p{p}" if p != 50 else "median": round(float(np.percentile(x, p)), 3) for p in ps} if len(x) else None

def _rate(hits, clusters, x):
    """Total count / total minutes over bouts (per-bout rates weighted by their minutes until the R1 fell) with a cluster-
    bootstrap 95% CI, and the plain mean of the per-bout rates (secondary: a bout the R1 fell early in weighs as much as a
    full one, so one hit in its 1.5 s reads as 40/min)."""
    from .stats import rate_ci
    r, ci = rate_ci(x, clusters, w=[h["secs"] / 60 for h in hits]); return dict(rate=r, ci95=ci, bout_mean=round(float(np.mean(x)), 3) if len(x) else None)

def landed_block(hits, clusters, shell_solref=None):
    """Over bouts ('landed' in eval_opponent summaries): exposure-weighted rates with a cluster-bootstrap 95% CI (_rate),
    per-hit impulse and peak force pooled over bouts, the per-bout pelvis spacing quantiles' medians."""
    ci = lambda k: _rate(hits, clusters, [h[k] for h in hits])
    imp = [x for h in hits for x in h["head_hit_imp_Ns"]]; pk = [x for h in hits for x in h["hit_peak_N"]]
    N = lambda k: sum(h["n"][k] for h in hits); up = N("arm_clean") + N("head_clean") + N("torso_clean")
    sp = np.array([h["spacing_m"] for h in hits if "spacing_m" in h])
    out = dict(primary="head_clean_pm", rates="rate: total count / total minutes until the R1 fell; bout_mean: mean of per-bout rates",
               **{k: ci(k) for k in ("head_clean_pm", "head_landed_pm", "torso_clean_pm", "torso_landed_pm", "head_impulse_pm", "arm_clean_pm")},
               counts={k: N(k) for k in ("head_clean", "head_landed", "torso_clean", "torso_landed", "arm_clean")},
               impulse_per_head_hit_Ns=dict(n=len(imp), **(_q(imp, [50, 90]) or {})),
               peak_force_N=dict(n=len(pk), **(_q(pk, [50, 95]) or {}), max=round(float(max(pk)), 1) if pk else None, shell_solref=shell_solref),
               arm_share_of_clean=round(N("arm_clean") / up, 3) if up else None,   # window-free stand-in for 'blocked share' until M1.2's strike windows
               spacing_m=dict(zip(("p10", "median", "p90"), np.round(np.median(sp, 0), 3).tolist())) if len(sp) else None)
    w = [h["windows"] for h in hits if "windows" in h]
    if w:
        W = {k: sum(x[k] for x in w) for k in w[0]}; tot = sum(W.values())
        out["windows"] = dict(W, blocked_share=round(W["blocked"] / tot, 3) if tot else None)
    return out

def sensitivity_block(hits, clusters):
    """The same bouts scored under every rule: head and torso per minute, estimated like the primary (_rate)."""
    return {r: {z: _rate(hits, clusters, [h["rules"][r][i] for h in hits]) for i, z in enumerate(("head", "torso"))} for r in RULES}
