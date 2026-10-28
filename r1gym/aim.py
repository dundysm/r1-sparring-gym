"""Aimed opponent, M1: operators on the shipped tracker's reference so that its library strikes go for the R1's head. The
tracker is unchanged; which operator is used is chosen by the rule preregistered in docs/AIMED_OPPONENT.md
(scripts/aim_check.py -> results/opponent_tracker/aim.json).

One Aim per env, in opp_state['aim'] (FightReplay and the Tracker are shared across envs and threads); Tracker.act calls it
for the clip clock and the reference it shows the policy.
  StrikeScheduler  a StrikeIntent per library hand strike once the clip clock is within lead_s (0.8 s) of its onset: limb,
                   zone, the target point in the R1's torso frame (head: 3 cm behind the face; torso: 3 cm behind the chest
                   front), t_on / t_apex / t_end in sim time. The target follows the R1's true pose (privileged, as the steering
                   already is) until t_apex - commit_s (0.15 s), then stays put in the world, so a defender that moves after
                   that slips it. Upright gate: no aimed intent while the opponent's torso tilts more than 15 deg, or within
                   0.3 s after its hand, forearm, foot or shin touched the R1's head or torso (the intent is still logged).
  StandOff         from lead_s before onset to the strike's end, steer to the bout distance at which the strike's apex goes
                   depth_m (3 cm) into the target's surface (the library's unwarped reach), clamped to [min_m, the bout
                   distance]; the tracker's steering keeps its 0.3 m/s cap and 0.5 s lag (the drift it trained with).
  ArmWarp          closed loop: every control step, iters (12) damped least-squares steps on the striking arm's 5 joints put
                   the reference palm on the target as seen from the opponent's current torso, pulled toward the clip's own
                   arm angles; the change is clipped to +-max_dq (0.8 rad) and to 95% of each joint's range, blended in by a
                   smoothstep from t_on - 0.15 s to the apex, held 0.05 s and blended out by t_end + 0.1 s (zero outside the
                   window, C1 in time), and its rate is added to the reference joint velocity. 'vz' keeps the reference palm's
                   own forward distance and aims only height and sideways: the arm can't add reach.
  StrikeSelector   'select': when a hand strike ends and the next isn't a head-level one within 0.6 s, splice the clip clock
                   to fade_s (0.35 s) before the onset of the bout clip's best-matching head-level strike (joint-space pose
                   distance; the clip's highest strike if none is head-level), cross-fading the reference over fade_s. Only
                   within the bout's clip, so the splits stay clean.
Modes (cfg opponent_aim): none (no Aim), shadow (intents logged, reference untouched: the unaimed side of a paired comparison,
bit-identical to none), standoff, armwarp, armwarp_vz, standoff+armwarp_vz, select."""
from __future__ import annotations
import numpy as np, mujoco
from dataclasses import dataclass, field
from .model import CTRL_JOINTS
from .hits import opp_limb, r1_zone, STRIKE_LIMBS

AIM = dict(zone="head", targets=dict(head=[0.057, 0.0, 0.36], torso=[0.04, 0.0, 0.15]), lead_s=0.8, commit_s=0.15, window_tail_s=0.15,
           arm=dict(lead_s=0.15, hold_s=0.05, tail_s=0.1, min_end_s=0.1, max_dq=0.8, range_frac=0.95, iters=12, prior=1e-3, damp=1e-3),
           standoff=dict(min_m=0.35, depth_m=0.03), upright_gate_deg=15.0, gate_after_touch_s=0.3, select=dict(fade_s=0.35, min_z_m=1.02, next_within_s=0.6))
OPERATORS = {"shadow": (), "standoff": ("standoff",), "armwarp": ("arm",), "armwarp_vz": ("arm", "vz"), "standoff+armwarp_vz": ("standoff", "arm", "vz"), "select": ("select",)}
ARM = {k: [CTRL_JOINTS.index(f"{side}_{j}_joint") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")] for k, side in (("l_hand", "left"), ("r_hand", "right"))}
PALM = {"l_hand": "left_palm", "r_hand": "right_palm"}

def smoothstep(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

def merged(cfg):
    """AIM with the config's aim block deep-merged over it (configs/r1_assumptions.yaml: aim)."""
    return {k: ({**v, **(cfg or {}).get(k, {})} if isinstance(v, dict) else (cfg or {}).get(k, v)) for k, v in AIM.items()}

@dataclass
class StrikeIntent:
    limb: str; zone: str; clip: str; onset: int; t_on: float; t_apex: float; t_end: float   # times: sim seconds
    target_r1: np.ndarray        # target point in the R1's torso frame
    reach_m: float               # the strike's unwarped reach to the zone (library)
    gated: str = ""              # 'tilt' / 'touch' when the upright gate held it back ('' = aimed)
    target_w: np.ndarray = None  # world target: live until committed
    committed: bool = False
    ik_res_m: float = None       # full-target IK residual at the apex (what the arm alone can't make up)
    closest_m: float = 9.9       # closest striking palm to target over the window
    arm_err: list = field(default_factory=list)   # per control step: RMS of the striking arm's joints minus the reference shown
    def window(self, tail): return (round(self.t_on, 4), round(self.t_apex + tail, 4), self.limb)
    def as_dict(self):
        return dict(limb=self.limb, zone=self.zone, onset=self.onset, t_on=round(self.t_on, 3), t_apex=round(self.t_apex, 3), t_end=round(self.t_end, 3), gated=self.gated,
                    reach_m=self.reach_m, ik_res_m=None if self.ik_res_m is None else round(self.ik_res_m, 4), closest_m=round(self.closest_m, 4),
                    arm_rmse=round(float(np.sqrt(np.mean(np.square(self.arm_err)))), 4) if self.arm_err else None,
                    target_w=None if self.target_w is None else [round(float(x), 4) for x in self.target_w])

_GATE = {}
def _gate_geoms(m):
    """Opponent strike-limb geoms and R1 head / torso geoms, as the HitMeter maps them (cached per model)."""
    if id(m) not in _GATE:
        o, r = np.zeros(m.ngeom, bool), np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            b = m.body(m.geom_bodyid[g]).name
            if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
            if b.startswith("opp_"): o[g] = opp_limb(b[4:])[1] in STRIKE_LIMBS
            else: r[g] = r1_zone(b, m.mesh(m.geom_dataid[g]).name if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else m.geom(g).name) in ("head", "torso")
        _GATE[id(m)] = (o, r)
    return _GATE[id(m)]

class Aim:
    def __init__(self, m, replay, mode, cfg=None, r1_torso=None):
        from .opponent import _sub
        assert mode in OPERATORS, f"opponent_aim {mode!r}: one of none, {', '.join(OPERATORS)}"
        self.mode, self.ops, self.p = mode, OPERATORS[mode], merged(cfg); self.T = replay.T; self.clip = replay.clip
        self.lib = [s for s in replay.strikes if s.hand]; self.r1_torso = m.body("torso_link").id if r1_torso is None else r1_torso
        self.palm = {k: m.site("opp_" + v).id for k, v in PALM.items()}; self.qarm = {k: replay.qadr + 7 + np.array(ix) for k, ix in ARM.items()}
        self.sub = _sub("r1"); self.sd = mujoco.MjData(self.sub); self.sd.qpos[3] = 1.0; self.J = np.zeros((3, self.sub.nv)); self.tor = self.sub.body("torso_link").id
        self.ssite = {k: self.sub.site(v).id for k, v in PALM.items()}
        lo, hi = self.sub.jnt_range[1:, 0], self.sub.jnt_range[1:, 1]; mid, half = (lo + hi) / 2, (hi - lo) / 2; fr = self.p["arm"]["range_frac"]
        self.lo, self.hi = mid - fr * half, mid + fr * half; self.g_opp, self.g_r1 = _gate_geoms(m)
        zs = [s.apex_m[2] for s in self.lib]; mz = self.p["select"]["min_z_m"]
        self.high = [s for s in self.lib if s.apex_m[2] >= mz] or ([max(self.lib, key=lambda s: s.apex_m[2])] if self.lib else [])
        self.intents, self.made, self.prev, self.touch_t, self.offset, self.fade, self.splices, self.trace = [], set(), np.zeros(24), -9.0, 0, None, 0, []

    # ---- clock and reference (called by Tracker.act)
    def clock(self, t, replay):
        return int(round(t * replay.fps * replay.speed)) + self.offset

    def reference(self, tr, m, d, replay, fi, state, t, dt):
        rate = replay.fps * replay.speed; now = d.time + dt   # the reference shown is the pose due at the end of this control step
        if "select" in self.ops: fi = self._select(tr, replay, fi, now, rate, t)
        self._schedule(d, replay, fi / rate, now, rate)
        rq, rdq, apos, aori = tr.reference(replay, tr.frame(replay.clip, fi), state)
        if self.fade is not None: rq, rdq, apos, aori = self._blend(tr, replay, (rq, rdq, apos, aori), state, t, now, rate)
        if "standoff" in self.ops: state["dist"] = self._standoff(replay, now)
        if "arm" in self.ops:
            delta = self._arm(d, replay, rq, now); rq = rq + delta; rdq = rdq + (delta - self.prev) / dt; self.prev = delta
        self._log(d, replay, rq, now); self.trace.append((now, float(np.abs(self.prev).max()), state.get("dist")))   # per step: |warp| max, stand-off
        return rq, rdq, apos, aori

    # ---- scheduler and gate
    def _touching(self, d):
        if d.ncon == 0: return False
        g1, g2 = d.contact.geom1, d.contact.geom2
        return bool(np.any((self.g_opp[g1] & self.g_r1[g2]) | (self.g_opp[g2] & self.g_r1[g1])))

    def _schedule(self, d, replay, tc, now, rate):
        p = self.p; T = self.T
        if self._touching(d): self.touch_t = now
        tilt = np.degrees(np.arccos(np.clip(d.xmat[replay.torso][8], -1.0, 1.0))); occ = int(np.floor(tc * rate / T))
        for k in (occ, occ + 1):
            for s in self.lib:
                key = k * T + s.onset; ton = key / rate
                if key in self.made or not tc <= ton <= tc + p["lead_s"]: continue
                self.made.add(key); sim = lambda f: now + (k * T + f) / rate - tc
                gated = "tilt" if tilt > p["upright_gate_deg"] else "touch" if now - self.touch_t < p["gate_after_touch_s"] else ""
                z = p["zone"] if p["zone"] in ("head", "torso") else ("head" if s.band in ("head", "high") else "torso")   # 'band': the zone it is level with
                self.intents.append(StrikeIntent(s.limb, z, self.clip, s.onset, sim(s.onset), sim(s.apex), sim(s.end), np.array(p["targets"][z], float),
                                                 s.reach_head_m if z == "head" else s.reach_torso_m, gated))
        R = d.xmat[self.r1_torso].reshape(3, 3); x = d.xpos[self.r1_torso]
        for it in self.intents:   # privileged target: the R1's true torso pose, until committed
            if it.committed or now > it.t_end + 1.0: continue
            it.target_w = x + R @ it.target_r1
            if now >= it.t_apex - p["commit_s"]: it.committed = True

    def active(self, now, pad=0.0):
        return [it for it in self.intents if it.t_on - self.p["lead_s"] - pad <= now <= it.t_end + self.p["arm"]["tail_s"] + self.p["window_tail_s"] + pad]

    def current(self, now):
        """The intent whose window [t_on, t_apex + 0.15 s] holds `now` (the latest onset), for env info['strike_intent']."""
        c = [it for it in self.intents if it.t_on <= now <= it.t_apex + self.p["window_tail_s"]]; return max(c, key=lambda it: it.t_on).as_dict() if c else None

    def windows(self):
        """(t0, t1, limb) per intent, for HitMeter.windows: the preregistered aimed window [t_on, t_apex + 0.15 s]."""
        return [it.window(self.p["window_tail_s"]) for it in self.intents]

    # ---- operators
    def _standoff(self, replay, now):
        home, p = replay.distance, self.p["standoff"]; want = home
        for it in self.intents:
            if not it.gated and it.t_on - self.p["lead_s"] <= now <= it.t_end: want = min(want, float(np.clip(it.reach_m - p["depth_m"], p["min_m"], home)))
        return want

    def weight(self, it, now):
        a = self.p["arm"]; t0, tp = it.t_on - a["lead_s"], it.t_apex; th = tp + a["hold_s"]; te = max(it.t_end, tp + a["min_end_s"]) + a["tail_s"]
        if now <= t0 or now >= te: return 0.0
        return smoothstep((now - t0) / max(tp - t0, 1e-6)) if now < tp else 1.0 if now <= th else smoothstep((te - now) / max(te - th, 1e-6))

    def _palm_b(self, q, limb):
        """Palm in the torso frame for reference joint angles q (forward kinematics on a private MjData)."""
        sd = self.sd; sd.qpos[7:] = q; mujoco.mj_kinematics(self.sub, sd); Rt = sd.xmat[self.tor].reshape(3, 3)
        return Rt.T @ (sd.site_xpos[self.ssite[limb]] - sd.xpos[self.tor]), Rt

    def ik(self, rq, limb, p_t):
        """Arm joints (5) putting the palm at p_t (torso frame), from the reference arm with a pull toward it; clipped to
        +-max_dq around it and to 95% of each joint's range. Returns (angles, residual m)."""
        a = self.p["arm"]; ix = ARM[limb]; q = np.array(rq, float); sd = self.sd; site = self.ssite[limb]; n = len(ix)
        lo = np.minimum(np.maximum(self.lo[ix], q[ix] - a["max_dq"]), q[ix]); hi = np.maximum(np.minimum(self.hi[ix], q[ix] + a["max_dq"]), q[ix])
        for _ in range(a["iters"]):
            pb, Rt = self._palm_b(q, limb); e = p_t - pb; mujoco.mj_comPos(self.sub, sd); mujoco.mj_jacSite(self.sub, sd, self.J, None, site)
            A = Rt.T @ self.J[:, [6 + i for i in ix]]
            q[ix] = np.clip(q[ix] + np.linalg.solve(A.T @ A + (a["damp"] + a["prior"]) * np.eye(n), A.T @ e + a["prior"] * (rq[ix] - q[ix])), lo, hi)
        return q[ix], float(np.linalg.norm(p_t - self._palm_b(q, limb)[0]))

    def _arm(self, d, replay, rq, now):
        delta = np.zeros(24); best = {}
        for it in self.active(now):
            w = self.weight(it, now)
            if not it.gated and w > 0 and w > best.get(it.limb, (0.0, None))[0]: best[it.limb] = (w, it)
        Rt, xt = d.xmat[replay.torso].reshape(3, 3), d.xpos[replay.torso]
        for limb, (w, it) in best.items():
            p_t = Rt.T @ (it.target_w - xt)
            if "vz" in self.ops: p_t[0] = self._palm_b(rq, limb)[0][0]   # the reference palm's own forward distance: aim height and sideways only
            qa, _ = self.ik(rq, limb, p_t); delta[ARM[limb]] = w * (qa - rq[ARM[limb]])
        return delta

    # ---- logging (every mode): closest palm to target, arm tracking error, full-target IK residual at the apex
    def _log(self, d, replay, rq, now):
        tail = self.p["window_tail_s"]; Rt, xt = d.xmat[replay.torso].reshape(3, 3), d.xpos[replay.torso]
        for it in self.intents:
            if not it.t_on <= now <= it.t_apex + tail or it.target_w is None: continue
            it.closest_m = min(it.closest_m, float(np.linalg.norm(d.site_xpos[self.palm[it.limb]] - it.target_w)))
            it.arm_err.append(float(np.sqrt(np.mean((d.qpos[self.qarm[it.limb]] - rq[ARM[it.limb]]) ** 2))))
            if it.ik_res_m is None and now >= it.t_apex: it.ik_res_m = self.ik(rq, it.limb, Rt.T @ (it.target_w - xt))[1]

    # ---- strike selector
    def _select(self, tr, replay, fi, now, rate, t):
        sp = self.p["select"]; T = self.T
        if self.fade is not None and now >= self.fade[0] + sp["fade_s"]: self.fade = None
        if self.fade is not None or not self.high: return fi
        fc = fi % T
        if not any((s.end + 1) % T == fc for s in self.lib): return fi                   # only right after a hand strike ends
        gap, nxt = min(((s.onset - fc) % T, i) for i, s in enumerate(self.lib)); nxt = self.lib[nxt]
        if nxt in self.high and gap / rate <= sp["next_within_s"]: return fi           # a head-level strike is coming anyway
        lead = int(round(sp["fade_s"] * rate)); cur = tr.ref_q[tr.frame(replay.clip, fi)]
        s = min(self.high, key=lambda s: (float(np.linalg.norm(tr.ref_q[tr.frame(replay.clip, s.onset - lead)] - cur)), s.onset))
        new = (fi // T) * T + s.onset - lead; self.fade = (now, self.offset); self.offset += new - fi; self.splices += 1
        self.intents = [it for it in self.intents if it.t_on <= now]; self.made = set()   # upcoming strikes come from the new timeline
        return new

    def _blend(self, tr, replay, ref, state, t, now, rate):
        """Cross-fade from the timeline before the splice (old offset) to the current one; velocities include the fade's rate."""
        t0, off0 = self.fade; w = smoothstep((now - t0) / self.p["select"]["fade_s"]); fade_s = self.p["select"]["fade_s"]
        x = (now - t0) / fade_s; dw = 6 * x * (1 - x) / fade_s if 0 < x < 1 else 0.0
        q0, dq0, p0, a0 = tr.reference(replay, tr.frame(replay.clip, int(round(t * replay.fps * replay.speed)) + off0), state); q1, dq1, p1, a1 = ref
        a1 = a1 if np.dot(a0, a1) >= 0 else -a1; aq = (1 - w) * a0 + w * a1
        return (1 - w) * q0 + w * q1, (1 - w) * dq0 + w * dq1 + dw * (q1 - q0), (1 - w) * p0 + w * p1, aq / np.linalg.norm(aq)
