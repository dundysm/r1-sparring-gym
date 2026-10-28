"""Defence oracle: can the R1's arms (or a slip) stop the aimed opponent's head strikes at all? A feasibility ceiling for the
sparring plan (head clean strikes received per minute, r1gym/hits.py), not a controller: it reads the opponent's true state.

Built on the shipped footwork guard policy (results/policy_mjlab_guard_footwork, its gym.json as shipped). Every control step
the policy acts as usual; OracleEnv (an R1HitEnv) then edits its action before the env applies it.
v1 (ops 'arm', 'slip'; preregistered, NO-GO: results/sparring/oracle_val.json 'v1'), reactive:
  threat   an opponent hand is a threat while one of its StrikeIntents (r1gym/aim.py, privileged: env.opp_state['aim'],
           gated ones included) is within [t_on - pre_s, t_apex + tail_s], or while its palm is within near_m of the R1's
           head centre and closing on it faster than v_close (true palm positions). Of two threats, the nearer palm. Each
           step's threat is logged by its source (the intent's zone, or 'near').
  arm      the near arm (the side of the threatening palm in the R1's torso frame, latched while that hand stays a threat):
           shoulder pitch / roll / yaw and elbow from damped least squares (iters steps from the guard pose, pulled toward it
           by `prior`, as r1gym/aim.py's ArmWarp) that put the forearm (the wrist link's shell centre) on the line from the head
           centre to the incoming palm, block_frac of the way out, clamped to block_m; each joint within +-max_dq of the guard
           and range_frac of its range. Re-solved every step (it chases the palm). Wrist roll and the other arm: the policy's.
  slip     waist roll target +-slip_roll rad away from the threat's side (+ when the palm is on the R1's left: the head moves
           right) and a knee dip added to both legs (knee +dip, hip and ankle pitch -dip/2 each, feet stay level), as offsets on
           the policy's targets slewed at slip_rate rad/s in and out.
  delay    a decision is made from the state at step k and applied at step k + delay_steps (2: 40 ms), on top of the hard
           tier's DR command latency (0 or 20 ms per bout, which delays the policy's own actions too).
v2 (op 'arm2', arm_block_v2; designed after v1's review), anticipating: the R1's arms are soft position servos (kp 20-40,
  r1gym/model.py) and v1's 40 ms reactive chase never got the forearm to its block point (review: median 0.11 m off it, joints
  0.27 rad off their targets). Per head-aimed StrikeIntent (zone in `zones`; gated ones too with gated=True, the strike is still
  thrown) one guard placement is planned when its plan time reaches t_on - lead_s: the point block_d out from the intent's
  target (r1gym/aim.py: 3 cm behind the face, R1 torso frame) toward the striking side's opponent shoulder (the predicted apex
  line), fixed in the R1's torso frame; the near arm (the side that shoulder is on) gets its shoulder / elbow angles q* from the
  same IK with the wrist link's origin (mid forearm-and-hand) as the point, within +-max_dq of the guard. q* is commanded from
  t_on - lead_s, held to t_apex + tail_s and blended back to the policy's targets over release_s; per arm, the intent with the
  larger weight (then the earlier apex). over > 0 commands q* + over * (q* - the policy's target) to beat the soft servo (in the
  joint range). Decisions go through the same delay_steps queue, planned for the step they are applied at.
Targets go to the policy's action space as a = (q - default pose) / action scale (configs/r1_mjlab.yaml contract: no action
filter, scale 0.25 * effort / kp; the default pose is the ONNX's guard). The policy is shown its own last action (what it
emitted), not the edited one, so its proprioceptive history stays its own. Mode 'off' (ops ()) runs every decision and logs it
but applies nothing: bit-identical to the plain env (tests/test_defence_oracle.py), so it doubles as the reproduction check.
Every mode logs the PHYSICAL placement (read-only, from the true state at the start of each control step): the distance from
the nearer R1 forearm (segment elbow link origin -> wrist shell centre) to the incoming segment (striking palm -> R1 head
centre) at the last step before each head clean strike's onset and at each head intent's apex, the IK point's distance to the
planned guard point and the guard joints' error at the apex (v2), and the arm servo error (measured angle minus the target
in d.ctrl) on steps whose target the oracle edited and on the others.

Stages (--stage):
  tune  TRAIN_CLIPS only, the same tier settings: the unmodified policy and GRID's arm_block_v2 configs on the same episodes;
        pick the largest train reduction in head clean/min (paired diff / policy rate) with survival >= 0.90 ->
        results/sparring/oracle_train_tuning.json.
  val   VAL only, scripts/eval_opponent.py's hard tier without scripted hits, aimed opponent (--aim chosen), 16 reps (the 64
        bouts of v1), run by its own run() / summarise() / vs_unaimed() (the env class swapped for OracleEnv for the duration
        of each run): the unmodified policy, arm_block_v2 with the config chosen on train (read from the tuning file), and v1's
        arm_block re-run with the review fixes as a placement diagnostic (not a gate). The unmodified variant must reproduce the
        stored results/eval_opponent/aim/summary_val_hard_nohits_aim.json 'RL guard_fw' head clean/min. Go (the v1 rule):
        arm_block_v2 cuts val head clean/min by >= 40% with the paired (clip, start) cluster-bootstrap CI upper bound < 0.
        Output: results/sparring/oracle_val.json, v1's preregistered result kept unchanged under 'v1'."""
import sys, json, time, argparse, hashlib
from collections import deque
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from r1gym.env import R1HitEnv, VAL_CLIPS, TRAIN_CLIPS
from r1gym.model import build_spec, load_cfg, CTRL_JOINTS
from r1gym.policy import load_controller, controller_cfg
from r1gym.scenario import clean_starts, REF_POSES
from r1gym.hits import HitMeter
import eval_opponent as EO

ROOT = Path(__file__).resolve().parents[1]
POLICY = "RL guard_fw=results/policy_mjlab_guard_footwork/policy.onnx"
STORED = ROOT / "results" / "eval_opponent" / "aim" / "summary_val_hard_nohits_aim.json"; STORED_RAW = STORED.with_name("raw_val_hard_nohits_aim.json")
TUNING = ROOT / "results" / "sparring" / "oracle_train_tuning.json"
ORACLE = dict(delay_steps=2, pre_s=0.10, tail_s=0.15, near_m=0.6, v_close=1.0, block_frac=0.5, block_m=(0.20, 0.30), iters=12, damp=1e-3, prior=3e-3,
              max_dq=1.0, range_frac=0.95, slip_roll=0.3, dip=0.15, slip_rate=4.0)
ORACLE_V2 = dict(lead_s=0.25, over=0.0, block_d=0.15, tail_s=0.15, release_s=0.10, zones=("head",), gated=True, max_dq=1.2)
GRID = {f"lead{round(l * 100)}_over{round(o * 100)}": dict(lead_s=l, over=o) for l in (0.15, 0.25, 0.4) for o in (0.0, 0.25, 0.5)}   # 9: lead x over-command
GRID.update({"lead25_over0_d10": dict(lead_s=0.25, over=0.0, block_d=0.10), "lead25_over0_d22": dict(lead_s=0.25, over=0.0, block_d=0.22),
             "lead25_over0_ungated": dict(lead_s=0.25, over=0.0, gated=False)})   # +3: guard distance, aimed (ungated) intents only
# over 1.0 left out after a 12-bout train smoke (lead 0.25): q* + 1.0 * (q* - guard) overshot, guard joints 0.28 rad past q* at the apex (0.05 at over 0)
PICK = dict(rule="largest train reduction in head clean/min (paired diff / policy rate) among configs with train survival >= 0.90", min_survival=0.90)
VARIANTS = {"policy": (), "arm_block": ("arm",), "slip": ("slip",), "arm_block+slip": ("arm", "slip"), "arm_block_v2": ("arm2",)}
J = {j: i for i, j in enumerate(CTRL_JOINTS)}
ARM_IX = {s: [J[f"{s}_{k}_joint"] for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")] for s in ("left", "right")}
WAIST_ROLL = J["waist_roll_joint"]; KNEE = [J["left_knee_joint"], J["right_knee_joint"]]
HIP_ANKLE = [J[f"{s}_{k}_joint"] for s in ("left", "right") for k in ("hip_pitch", "ankle_pitch")]
HANDS = {"l_hand": "opp_left_palm", "r_hand": "opp_right_palm"}
SHOULDERS = {"l_hand": "opp_left_shoulder_pitch_link", "r_hand": "opp_right_shoulder_pitch_link"}
# per-step record (OracleEnv._record): time | torso pos, rot | head centre | opponent palms l, r | per R1 arm (left, right): elbow
# origin, wrist shell centre, wrist origin | arm angles (left 4, right 4) | their targets in d.ctrl | target edited by the oracle (left, right)
REC = dict(t=0, tx=slice(1, 4), tR=slice(4, 13), head=slice(13, 16), l_hand=slice(16, 19), r_hand=slice(19, 22),
           left=(slice(22, 25), slice(25, 28), slice(28, 31)), right=(slice(31, 34), slice(34, 37), slice(37, 40)),
           q=slice(40, 48), ctrl=slice(48, 56), edited=slice(56, 58))
NEAR_M = 0.07   # forearm capsule (r 0.03-0.035) plus a hand's: centre lines within this could touch

def _mesh_geom(m, body, mesh):
    return next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body).id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name == mesh)

def seg_dist(p1, q1, p2, q2):
    """Closest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2; a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12: return float(np.linalg.norm(r))
    if a <= 1e-12: s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = d1 @ r
        if e <= 1e-12: s, t = float(np.clip(-c / a, 0, 1)), 0.0
        else:
            b = d1 @ d2; den = a * e - b * b; s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > 1e-12 else 0.0; t = (b * s + f) / e
            if t < 0: s, t = float(np.clip(-c / a, 0, 1)), 0.0
            elif t > 1: s, t = float(np.clip((b - c) / a, 0, 1)), 1.0
    return float(np.linalg.norm(p1 + d1 * s - p2 - d2 * t))

class OracleEnv(R1HitEnv):
    """R1HitEnv whose step() edits the controller's action with the oracle (cfg 'oracle': {'ops': [...], v1 params over ORACLE,
    'v2': params over ORACLE_V2}); ops () = 'off'. Every instance is appended to `made` (episode order in eval_opponent.run) for
    the per-bout oracle log."""
    made = []
    def __init__(self, *a, **k):
        super().__init__(*a, **k); OracleEnv.made.append(self); m = self.m
        oc = self.cfg.get("oracle", {}); self.ops = tuple(oc.get("ops", ())); self.p = {**ORACLE, **{k: v for k, v in oc.items() if k not in ("ops", "v2")}}
        self.p2 = {**ORACLE_V2, **oc.get("v2", {})}
        self.kd = mujoco.MjData(m); self.jac = np.zeros((3, m.nv)); jr = m.jnt_range[[m.joint(j).id for j in CTRL_JOINTS]]
        mid, half = jr.mean(1), (jr[:, 1] - jr[:, 0]) / 2; self.lo, self.hi = mid - self.p["range_frac"] * half, mid + self.p["range_frac"] * half
        hg = _mesh_geom(m, "torso_link", "head_yaw_link"); self.head_off = m.geom_pos[hg].copy()   # head shell centre, torso frame
        self.fore = {s: (m.body(f"{s}_wrist_roll_link").id, m.geom_pos[_mesh_geom(m, f"{s}_wrist_roll_link", f"{s}_wrist_roll_link")].copy()) for s in ARM_IX}
        self.elbow = {s: m.body(f"{s}_elbow_link").id for s in ARM_IX}; self.palm = {h: m.site(s).id for h, s in HANDS.items()}
        self.shoulder = {h: m.body(b).id for h, b in SHOULDERS.items()}
        self.vadr = {s: self.r.vadr[ix] for s, ix in ARM_IX.items()}; self.qarm = self.r.qadr[ARM_IX["left"] + ARM_IX["right"]]
        self.aarm = self.r.act[ARM_IX["left"] + ARM_IX["right"]]; self.pol_a = None

    def reset(self, *a, **k):
        self.pol_a = None; self.pol_prev = np.zeros(24, np.float32); self.queue = deque([None] * self.p["delay_steps"]); self.prev = {}; self.latch = None; self.slip = np.zeros(2)   # slip offsets: waist roll, knee dip
        self.plans = {}; self.rec = []; self.ed = deque([np.zeros(2, bool)] * 2, maxlen=2)   # v2 plans by intent; per-step record; arm-edited flags of the last 2 actions
        self.log = dict(steps=0, threat=0, applied=0, edited=0, ik_res=[], on=[]); return super().reset(*a, **k)

    def _raw_obs(self):   # the policy sees its own last action (one step late under DR's command latency, as the env shows it)
        if self.pol_a is None: return super()._raw_obs()
        la = self.last_a; self.last_a = self.pol_a
        try: return super()._raw_obs()
        finally: self.last_a = la

    def head(self, kd):
        t = self.r.torso; return kd.xpos[t] + kd.xmat[t].reshape(3, 3) @ self.head_off

    def forearm(self, kd, side, pt="shell"):
        """The arm's IK point: the wrist link's shell centre (v1), or its origin (pt 'origin', v2: mid forearm-and-hand)."""
        b, off = self.fore[side]; return kd.xpos[b] + (kd.xmat[b].reshape(3, 3) @ off if pt == "shell" else 0.0)

    def _record(self):
        """Read-only: the true state at the start of this control step (kd), for the physical placement in bout_log. The arm
        targets in d.ctrl are the last step's; their edited flags are those of the action the env applied (one more step back
        under DR command latency)."""
        kd = self.kd; kd.qpos[:] = self.d.qpos; mujoco.mj_kinematics(self.m, kd); t = self.r.torso
        self.rec.append(np.concatenate([[self.d.time], kd.xpos[t], kd.xmat[t], self.head(kd), *(kd.site_xpos[self.palm[h]] for h in HANDS),
                                        *(np.r_[kd.xpos[self.elbow[s]], self.forearm(kd, s), kd.xpos[self.fore[s][0]]] for s in ARM_IX),
                                        self.d.qpos[self.qarm], self.d.ctrl[self.aarm], self.ed[-1 - self.act_delay]]))

    def decide(self):
        """v1 oracle decision from the current true state (None: no threat)."""
        kd, p = self.kd, self.p; kd.qpos[:] = self.d.qpos; mujoco.mj_kinematics(self.m, kd); now = self.d.time; H = self.head(kd)
        aim = self.opp_state.get("aim") if self.opp is not None else None; th = []
        for h, s in self.palm.items():
            W = kd.site_xpos[s].copy(); dist = float(np.linalg.norm(W - H)); vc = (self.prev[h] - dist) / self.dt if h in self.prev else 0.0; self.prev[h] = dist
            its = [it for it in aim.intents if it.limb == h and it.t_on - p["pre_s"] <= now <= it.t_apex + p["tail_s"]] if aim is not None else []
            if its or (dist < p["near_m"] and vc > p["v_close"]): th.append((dist, h, W, its[0].zone if its else "near"))
        if not th: self.latch = None; return None
        dist, h, W, src = min(th, key=lambda x: x[0]); y = float((kd.xmat[self.r.torso].reshape(3, 3).T @ (W - H))[1])
        side = self.latch[1] if self.latch and self.latch[0] == h else ("left" if y > 0 else "right"); self.latch = (h, side)
        dec = dict(hand=h, side=side, away=1.0 if y > 0 else -1.0, src=src)
        if "arm" in self.ops:
            u = (W - H) / max(dist, 1e-9); P = H + float(np.clip(p["block_frac"] * dist, *p["block_m"])) * u
            dec["q"], dec["res"] = self.ik(side, P)
        return dec

    def ik(self, side, P, pt="shell", max_dq=None):
        """Shoulder pitch / roll / yaw and elbow putting the arm's IK point (forearm()) on P (world), from the guard pose with a
        pull toward it; within +-max_dq of it and range_frac of each joint's range. kd must hold the current state. Returns (angles, residual m)."""
        kd, p, ix = self.kd, self.p, ARM_IX[side]; qa = self.r.qadr[ix]; g = self.home[ix]; b = self.fore[side][0]; mdq = p["max_dq"] if max_dq is None else max_dq
        lo, hi = np.maximum(self.lo[ix], g - mdq), np.minimum(self.hi[ix], g + mdq); q = np.clip(g, lo, hi)
        for _ in range(p["iters"]):
            kd.qpos[qa] = q; mujoco.mj_kinematics(self.m, kd); mujoco.mj_comPos(self.m, kd); x = self.forearm(kd, side, pt)
            mujoco.mj_jac(self.m, kd, self.jac, None, x, b); A = self.jac[:, self.vadr[side]]
            q = np.clip(q + np.linalg.solve(A.T @ A + (p["damp"] + p["prior"]) * np.eye(len(ix)), A.T @ (P - x) + p["prior"] * (g - q)), lo, hi)
        kd.qpos[qa] = q; mujoco.mj_kinematics(self.m, kd); return q, float(np.linalg.norm(P - self.forearm(kd, side, pt)))

    @staticmethod
    def key(it): return (it.limb, it.onset, round(it.t_on, 4))

    def plan(self, it):
        """v2 guard placement for one intent, from the current true state: the point block_d out from its target toward the
        striking side's opponent shoulder (R1 torso frame), the arm on that shoulder's side, and that arm's angles (IK on the
        wrist link's origin). Planned once; cached in self.plans."""
        kd, p2 = self.kd, self.p2; kd.qpos[:] = self.d.qpos; mujoco.mj_kinematics(self.m, kd); t = self.r.torso; R, x = kd.xmat[t].reshape(3, 3), kd.xpos[t].copy()
        s_t = R.T @ (kd.xpos[self.shoulder[it.limb]] - x); tgt = np.asarray(it.target_r1, float); u = (s_t - tgt) / max(float(np.linalg.norm(s_t - tgt)), 1e-9)
        P_t = tgt + p2["block_d"] * u; side = "left" if s_t[1] > 0 else "right"; q, res = self.ik(side, x + R @ P_t, pt="origin", max_dq=p2["max_dq"])
        pl = self.plans[self.key(it)] = dict(side=side, q=q, P_t=P_t, res=res, t_apex=it.t_apex, limb=it.limb, made_s=float(self.d.time)); return pl

    def plan_v2(self):
        """v2 decision for the step it will be applied at (now + delay_steps * dt): per R1 arm the head intent it guards, its
        weight (1 from t_on - lead_s to t_apex + tail_s, then linear to 0 over release_s) and q* (None: nothing to guard)."""
        p2 = self.p2; aim = self.opp_state.get("aim") if self.opp is not None else None
        if aim is None: return None
        tp = self.d.time + self.p["delay_steps"] * self.dt; arms = {}
        for it in aim.intents:
            if it.zone not in p2["zones"] or (it.gated and not p2["gated"]): continue
            th = it.t_apex + p2["tail_s"]; te = th + p2["release_s"]
            if not it.t_on - p2["lead_s"] <= tp <= te: continue
            pl = self.plans.get(self.key(it)) or self.plan(it); w = 1.0 if tp <= th else max(0.0, (te - tp) / p2["release_s"]); c = arms.get(pl["side"])
            if c is None or (w, -it.t_apex) > (c[0], -c[1]["t_apex"]): arms[pl["side"]] = (w, pl)
        return {s: dict(w=w, q=pl["q"]) for s, (w, pl) in arms.items()} or None

    def edit(self, a, dec):
        """The policy's action with a decision (None: none due) applied; targets -> action space: (q - home) / scale."""
        a = a.copy(); h, s, p = self.home, self.scale, self.p; w = WAIST_ROLL
        if "arm2" in self.ops and dec is not None:   # v2: per arm, q* (over-commanded) blended over the policy's target
            for side, g in dec.items():
                ix = ARM_IX[side]; qp = h[ix] + s[ix] * a[ix]; qc = np.clip(g["q"] + self.p2["over"] * (g["q"] - qp), self.lo[ix], self.hi[ix])
                a[ix] = (g["w"] * qc + (1 - g["w"]) * qp - h[ix]) / s[ix]
            return a.astype(np.float32)
        if "arm" in self.ops and dec is not None: ix = ARM_IX[dec["side"]]; a[ix] = (dec["q"] - h[ix]) / s[ix]
        if "slip" in self.ops:   # offsets on the policy's targets, slewed toward the goal (0 without a decision)
            goal = np.zeros(2) if dec is None else np.array([h[w] + p["slip_roll"] * dec["away"] - (h[w] + s[w] * a[w]), p["dip"]])
            st = p["slip_rate"] * self.dt; self.slip += np.clip(goal - self.slip, -st, st)
            if self.slip.any():
                for ix, dq in (([w], self.slip[0]), (KNEE, self.slip[1]), (HIP_ANKLE, -self.slip[1] / 2)):
                    q = np.clip(h[ix] + s[ix] * a[ix] + dq, self.lo[ix], self.hi[ix]); a[ix] = (q - h[ix]) / s[ix]
        return a.astype(np.float32)

    def step(self, a):
        a = np.clip(a, -self.clip, self.clip).astype(np.float32); self.pol_a, self.pol_prev = (self.pol_prev, a) if self.act_delay else (a, self.pol_prev)
        if self.dr: self._dr_apply()   # this env's DR parameters in the (thread-shared) model before decide()'s kinematics and Jacobians; step() writes the same again
        self._record(); dec = self.plan_v2() if "arm2" in self.ops else self.decide(); self.queue.append(dec); D = self.queue.popleft()
        L = self.log; L["steps"] += 1; L["threat"] += dec is not None
        if dec is not None and "res" in dec: L["ik_res"].append(dec["res"])
        if D is not None and self.ops: L["applied"] += 1
        a0 = a
        if self.ops and (D is not None or self.slip.any()): a = self.edit(a, D)
        ed = np.array([not np.array_equal(a[ARM_IX[s]], a0[ARM_IX[s]]) for s in ARM_IX]); self.ed.append(ed); L["edited"] += bool((a != a0).any())
        # per step: threat seen, decision applied, action edited (any joint; slip ramp-outs count, 'applied' doesn't), the threat's source (v1)
        L["on"].append((dec is not None, D is not None and bool(self.ops), bool((a != a0).any()), dec.get("src") if dec is not None and "hand" in dec else None))
        return super().step(a)

def _q(x, near=True):
    return dict(n=len(x), median=round(float(np.median(x)), 4), p90=round(float(np.percentile(x, 90)), 4), **({"within_0p07m": round(float(np.mean(np.array(x) < NEAR_M)), 3)} if near else {})) if len(x) else dict(n=0)

def placement(e, hc):
    """Physical placement from e.rec (state at each control step's start): per head clean strike by a hand / forearm (hc), the
    nearer R1 forearm's distance to the segment striking palm -> head centre at the last step before its onset; the same at
    each head intent's apex (in the bout, before the R1 fell); v2: per planned intent, the IK point's distance to its planned
    guard point and the guard joints' RMS error at the apex; per step and arm, the RMS servo error (angle - target in d.ctrl)
    when the oracle edited that arm's target, and when it didn't."""
    R = np.array(e.rec); ts = R[:, REC["t"]]; at = lambda t: R[max(0, int(np.searchsorted(ts, t + 1e-9, "right")) - 1)]
    fore = lambda r: [(r[REC[s][0]], r[REC[s][1]]) for s in ("left", "right")]
    dline = lambda r, h: min(seg_dist(a, b, r[REC[h]], r[REC["head"]]) for a, b in fore(r))
    hit = []
    for x in hc:
        if x["limb"] not in ("hand", "forearm"): continue
        r = at(x["t0"]); hs = [f"{c}_hand" for c in x["side"] if c in "lr"] or list(HANDS)
        hit.append(dline(r, min(hs, key=lambda h: np.linalg.norm(r[REC[h]] - r[REC["head"]]))))
    aim = e.opp_state.get("aim"); apex = []; pd, ge = [], []
    for it in (aim.intents if aim is not None else []):
        if it.zone != "head" or it.t_on < 0 or it.t_apex > ts[-1]: continue
        r = at(it.t_apex); apex.append(dline(r, it.limb)); pl = e.plans.get(e.key(it))
        if pl is not None:
            Rt = r[REC["tR"]].reshape(3, 3); ix = slice(0, 4) if pl["side"] == "left" else slice(4, 8)
            pd.append(float(np.linalg.norm(r[REC[pl["side"]][2]] - (r[REC["tx"]] + Rt @ pl["P_t"])))); ge.append(float(np.sqrt(np.mean((r[REC["q"]][ix] - pl["q"]) ** 2))))
    err = R[:, REC["q"]] - R[:, REC["ctrl"]]; se = np.sqrt(np.stack([np.mean(err[:, :4] ** 2, 1), np.mean(err[:, 4:] ** 2, 1)], 1)); edf = R[:, REC["edited"]] > 0.5
    return dict(hit=hit, apex=apex, v2_point=pd, v2_guard_err=ge, v2_ik_res=[pl["res"] for pl in e.plans.values()], servo_edited=se[edf].tolist(), servo_free=se[~edf].tolist())

def bout_log(e):
    """Per bout: shares of control steps with a threat seen / a decision applied / the action edited, IK residuals, how many of
    the bout's head clean strikes began during a step with a threat seen (by its source: the intent's zone or 'near') / a
    decision applied / the arm target edited (step = the control step holding the strike's onset), head intents by gate, and
    the physical placement (placement(); its lists are pooled by block(), kept out of the per-bout rows)."""
    L = e.log; r = L["ik_res"]; hm = next(h for h in e.substep_hooks if isinstance(h, HitMeter) and not getattr(h, "reverse", False))
    hc = [x for x in hm.events() if x["zone"] == "head" and hm.landed(x) and hm.clean(x)]; on = L["on"]
    at = lambda x, j: on[min(int(x["t0"] / e.dt), len(on) - 1)][j] if on else None
    aim = e.opp_state.get("aim"); its = [it for it in (aim.intents if aim is not None else []) if it.t_on >= 0 and it.t_apex <= e.t * e.dt]
    src = {k: sum(at(x, 3) == k for x in hc) for k in ("head", "torso", "near")}
    return dict(threat_share=round(L["threat"] / max(L["steps"], 1), 4), applied_share=round(L["applied"] / max(L["steps"], 1), 4), edited_share=round(L["edited"] / max(L["steps"], 1), 4),
                threat_share_by_src={k: round(sum(o[3] == k for o in on) / max(len(on), 1), 4) for k in ("head", "torso", "near")},
                ik_res_median_m=round(float(np.median(r)), 4) if r else None, ik_res_p90_m=round(float(np.percentile(r, 90)), 4) if r else None,
                head_clean_in_threat=sum(bool(at(x, 0)) for x in hc), head_clean_in_threat_by_src=src, head_clean_in_applied=sum(bool(at(x, 1)) for x in hc),
                head_clean_in_edited=sum(bool(at(x, 2)) for x in hc), head_clean_kicks=sum(x["limb"] in ("foot", "shin") for x in hc),
                intents=dict(head=sum(it.zone == "head" for it in its), torso=sum(it.zone == "torso" for it in its), gated=sum(bool(it.gated) for it in its)), place=placement(e, hc))

def run_variant(ctrl, eps, m, cc, ops, T, threads, lag, params=None):
    """eval_opponent.run with the env class swapped for OracleEnv; returns (rows, per-bout oracle logs, seconds)."""
    OracleEnv.made = []; t0 = time.time(); cv = {**cc, "oracle": {"ops": list(ops), **(params or {})}}
    with mock.patch.object(EO, "R1HitEnv", OracleEnv): rows = EO.run(ctrl, eps, m, cv, T, threads=threads, dr=True, meter=True, lag=lag)
    assert len(OracleEnv.made) == len(eps); return rows, [bout_log(e) for e in OracleEnv.made], time.time() - t0

def block(rows, logs, T, clusters, solref):
    s = EO.summarise(rows, T, clusters, solref); L = s["landed"]; pool = lambda k: [v for x in logs for v in x["place"][k]]; med = lambda k: [x[k] for x in logs if x[k] is not None]
    hc = sum(r["hit_meter"]["n"]["head_clean"] for r in rows)
    return dict(head_clean_pm=L["head_clean_pm"], torso_clean_pm=L["torso_clean_pm"], head_landed_pm=L["head_landed_pm"], head_impulse_pm=L["head_impulse_pm"],
                survival=s["survival"], survival_ci95=s["survival_ci95"], r1_fell=round(1 - s["survival"], 3), opponent_fell=s["opponent_fell"],
                intent_windows=s.get("intent_windows"), windows=L.get("windows"),
                oracle=dict(threat_share=round(float(np.mean([x["threat_share"] for x in logs])), 4), applied_share=round(float(np.mean([x["applied_share"] for x in logs])), 4),
                            edited_share=round(float(np.mean([x["edited_share"] for x in logs])), 4),
                            threat_share_by_src={k: round(float(np.mean([x["threat_share_by_src"][k] for x in logs])), 4) for k in ("head", "torso", "near")},
                            ik_res_median_m=round(float(np.median(med("ik_res_median_m"))), 4) if med("ik_res_median_m") else None,
                            head_clean_total=hc, head_clean_in_threat=sum(x["head_clean_in_threat"] for x in logs),
                            head_clean_in_threat_by_src={k: sum(x["head_clean_in_threat_by_src"][k] for x in logs) for k in ("head", "torso", "near")},
                            head_clean_in_applied=sum(x["head_clean_in_applied"] for x in logs), head_clean_in_edited=sum(x["head_clean_in_edited"] for x in logs),
                            head_clean_kicks=sum(x["head_clean_kicks"] for x in logs), intents={k: sum(x["intents"][k] for x in logs) for k in ("head", "torso", "gated")}),
                placement=dict(at_head_clean_onset_m=_q(pool("hit")), at_intent_apex_m=_q(pool("apex")), v2_point_to_guard_m_at_apex=_q(pool("v2_point")),
                               v2_guard_joint_err_rad_at_apex=_q(pool("v2_guard_err"), False), v2_plan_ik_res_m=_q(pool("v2_ik_res"), False),
                               servo_err_rad_edited_steps=_q(pool("servo_edited"), False), servo_err_rad_other_steps=_q(pool("servo_free"), False),
                               defs="forearm = segment R1 elbow link origin -> wrist shell centre (nearer arm); incoming = striking palm -> R1 head centre; "
                                    "state at the last control step start before the onset / at the apex; within_0p07m: share with centre lines closer than 0.07 m "
                                    "(capsule radii); joint errors RMS over shoulder pitch / roll / yaw and elbow; servo error = angle - target in d.ctrl"))

def compare(out, raw, clusters):
    for v in out:
        if v == "policy" or "policy" not in raw: continue
        vs = EO.vs_unaimed(raw[v], raw["policy"], clusters); vs.pop("rules", None); out[v]["vs_policy"] = vs
        base = out["policy"]["head_clean_pm"]["rate"]; d = vs["head_clean_pm"]
        out[v]["head_clean_change_frac"] = round(d["diff"] / base, 3) if base else None

def setup(clips, reps):
    tier = EO.TIERS["hard"]; T = tier["seconds"]; aimj = json.load(open(EO.AIM_JSON)); lag, aim = float(aimj["lag_s"]), aimj["operator"]   # --aim chosen
    cfg = load_cfg(); cfg["opponent_aim"] = aim; ctrl = load_controller(POLICY)
    cc = {**controller_cfg(ctrl), "opponent_mode": cfg["opponent_mode"], "opponent_aim": aim}
    assert any(np.allclose(np.array(cc["home"], float), p, atol=1e-3) for p in REF_POSES.values())   # its pose is a reference pose: episodes unchanged
    good = clean_starts({**cfg, "opponent_aim": "none"}, clips, tier["distances"], dict(REF_POSES)); eps = EO.episodes(clips, reps, "hard", good, False, T)
    m = build_spec(opponent=True, cfg=cc).compile(); return dict(T=T, lag=lag, aim=aim, cfg=cfg, ctrl=ctrl, cc=cc, eps=eps, clusters=[(e["clip"], e["t0"]) for e in eps], m=m, solref=EO.shell_solref(m))

def rows_of(eps, raw, logs):
    return [dict(clip=e["clip"], distance=e["distance"], t0=e["t0"], fell_s=r["fell_s"], opp_fell=r["opp_fell"], secs=r["hit_meter"]["secs"], head_clean=r["hit_meter"]["n"]["head_clean"],
                 torso_clean=r["hit_meter"]["n"]["torso_clean"], **{k: v for k, v in l.items() if k != "place"},
                 place_median_m={k: round(float(np.median(l["place"][k])), 4) if l["place"][k] else None for k in ("hit", "apex", "v2_point")}) for e, r, l in zip(eps, raw, logs)]

def tune(a):
    """TRAIN_CLIPS only: the policy and every GRID config on the same episodes; pick by PICK."""
    S = setup(TRAIN_CLIPS, a.reps); t_all = time.time(); out, raw, secs = {}, {}, {}
    grid = {k: v for k, v in GRID.items() if not a.configs or k in a.configs}
    for v, ops, prm in [("policy", (), None)] + [(k, ("arm2",), {"v2": g}) for k, g in grid.items()]:
        raw[v], logs, secs[v] = run_variant(S["ctrl"], S["eps"], S["m"], S["cc"], ops, S["T"], a.threads, S["lag"], prm); out[v] = block(raw[v], logs, S["T"], S["clusters"], S["solref"])
        if prm: out[v]["v2"] = {**ORACLE_V2, **prm["v2"]}
        print(v, json.dumps({k: out[v][k] for k in ("head_clean_pm", "survival")}), json.dumps(out[v]["placement"]["at_intent_apex_m"]), f"({secs[v]:.0f} s)", flush=True)
    compare(out, raw, S["clusters"])
    ok = {k: x for k, x in out.items() if k != "policy" and x["survival"] >= PICK["min_survival"] and x["head_clean_change_frac"] is not None}
    best = min(ok, key=lambda k: (ok[k]["head_clean_change_frac"], list(grid).index(k))) if ok else None
    chosen = dict(name=best, v2=out[best]["v2"], train_change_frac=out[best]["head_clean_change_frac"], train_diff=out[best]["vs_policy"]["head_clean_pm"],
                  train_survival=out[best]["survival"]) if best else None
    meta = dict(script="scripts/defence_oracle.py --stage tune", policy=POLICY, split="train", clips=TRAIN_CLIPS, tier="hard", pushes=False, reps=a.reps, seconds=S["T"], episodes=len(S["eps"]),
                opponent_aim=S["aim"], strike_lag_s=S["lag"], episodes_from="scripts/eval_opponent.py episodes() on TRAIN_CLIPS (hard tier, no scripted hits)",
                pre_grid_smoke="before the grid: one 12-bout train run (reps 1) of lead 0.25 with over 0 and over 1.0, a code smoke; it set the over-command values (see GRID)",
                oracle_v1=ORACLE, oracle_v2_defaults=ORACLE_V2, grid=grid, pick=PICK, threads=a.threads, run_s={k: round(v) for k, v in secs.items()}, total_s=round(time.time() - t_all),
                why="v2 (arm_block_v2) was designed after v1's preregistered gate failed (results/sparring/oracle_val.json 'v1': -6%, NO-GO) and its review (forearm a median 0.11 m "
                    "from its block point, arm joints 0.27 rad off target); tuned here on train clips only, then run once on val",
                stats="vs_policy: config minus unmodified policy on identical episodes, exposure-weighted, (clip, start) cluster bootstrap (eval_opponent.vs_unaimed)",
                **EO.provenance(S["m"], S["cfg"], {v: {**S["cc"], "oracle": {"ops": [] if v == "policy" else ["arm2"], **({"v2": out[v]["v2"]} if v != "policy" else {})}} for v in out}))
    p = Path(a.out or TUNING); p.parent.mkdir(parents=True, exist_ok=True); json.dump(dict(meta=meta, chosen=chosen, configs=out), open(p, "w"), indent=1)
    print(json.dumps(dict(chosen=chosen, change={k: (x.get("head_clean_change_frac"), x["survival"]) for k, x in out.items()}), indent=1))

def val(a):
    """VAL (or --split train for a smoke): policy, arm_block_v2 (the tuning file's choice), v1's arm_block re-run; the go/no-go on v2."""
    clips = VAL_CLIPS if a.split == "val" else TRAIN_CLIPS; S = setup(clips, a.reps); T, eps, clusters = S["T"], S["eps"], S["clusters"]; t_all = time.time()
    tf = Path(a.tuning); tun = json.load(open(tf)); assert tun["chosen"], "no train config met the survival floor"; v2 = tun["chosen"]["v2"]
    st = json.load(open(STORED_RAW)) if a.split == "val" and a.reps == 16 and STORED_RAW.exists() else None
    if st: assert st["episodes"] == json.loads(json.dumps(eps)), "episodes differ from the stored aimed val run"
    prm = {"arm_block_v2": {"v2": v2}}; raw, logs, out, secs = {}, {}, {}, {}
    for v in a.variants:
        raw[v], logs[v], secs[v] = run_variant(S["ctrl"], eps, S["m"], S["cc"], VARIANTS[v], T, a.threads, S["lag"], prm.get(v)); out[v] = block(raw[v], logs[v], T, clusters, S["solref"])
        print(v, json.dumps({k: out[v][k] for k in ("head_clean_pm", "torso_clean_pm", "survival", "opponent_fell")}), json.dumps(out[v]["placement"]["at_intent_apex_m"]), f"({secs[v]:.0f} s)", flush=True)
    compare(out, raw, clusters)
    repro = None
    if "policy" in raw and st is not None:
        stored = json.load(open(STORED))["summary"]["RL guard_fw"]; R = st["raw"]["RL guard_fw"]
        sub = lambda x, y: all(k in x and sub(x[k], y[k]) for k in y) if isinstance(y, dict) else x == y   # every field the stored row has
        same = [sub(json.loads(json.dumps(x)), y) for x, y in zip(raw["policy"], R)]
        repro = dict(stored_head_clean_pm=stored["landed"]["head_clean_pm"]["rate"], ours=out["policy"]["head_clean_pm"]["rate"], stored_survival=stored["survival"],
                     bouts_identical=f"{sum(same)}/{len(same)} (every field the stored raw rows have)", stored_git=json.load(open(STORED))["meta"].get("git_hash"))
        repro["pass"] = repro["stored_head_clean_pm"] == repro["ours"] and all(same)
    gate = None
    if "arm_block_v2" in out and "vs_policy" in out["arm_block_v2"]:
        g = out["arm_block_v2"]; ci = g["vs_policy"]["head_clean_pm"]["ci95"]
        gate = dict(rule="arm_block_v2 cuts val head clean/min by >= 40% and the paired cluster-bootstrap CI upper bound < 0 (v1's preregistered rule, applied to v2)",
                    preregistered=False, change_frac=g["head_clean_change_frac"], diff=g["vs_policy"]["head_clean_pm"]["diff"], ci95=ci,
                    go=bool(g["head_clean_change_frac"] is not None and g["head_clean_change_frac"] <= -0.40 and ci[1] < 0))
    meta = dict(script="scripts/defence_oracle.py --stage val", policy=POLICY, split=a.split, clips=clips, tier="hard", pushes=False, reps=a.reps, seconds=T, episodes=len(eps),
                opponent_aim=S["aim"], strike_lag_s=S["lag"], episodes_from="scripts/eval_opponent.py episodes() (asserted equal to " + STORED_RAW.name + ")" if st else "scripts/eval_opponent.py episodes()",
                history=dict(v1="v1's gate (arm_block cuts val head clean/min >= 40%, CI upper < 0) was preregistered and failed: -6%, CI [-1.801, 0.947], NO-GO (the 'v1' key, unchanged)",
                             v1_tuning_on_val="v1's parameters were adjusted after a 4-bout val smoke (its bout 0 is evaluated val bout 0: same seed, distance and start) and traces of "
                                              "evaluated val bouts 0, 17 and 33, before its full run (review)",
                             v2="arm_block_v2 was designed after seeing v1's diagnosis (review: physical forearm a median 0.111 m from its block point, arm joints 0.27 rad off their "
                                "targets; soft servos and a 40 ms reactive chase), tuned on TRAIN_CLIPS only (oracle_train_tuning.json) and run once on val with the config chosen there",
                             arm_block_rerun="variants.arm_block is v1's oracle re-run with the review fixes (DR parameters written before decide(); threat sources and physical placement "
                                             "logged), a diagnostic, not a gate; v1's preregistered numbers are the 'v1' key"),
                tuning=dict(file=str(tf.relative_to(ROOT)) if tf.is_relative_to(ROOT) else str(tf), sha256=hashlib.sha256(tf.read_bytes()).hexdigest()[:16], chosen=tun["chosen"]),
                oracle_v1=ORACLE, oracle_v2={**ORACLE_V2, **v2}, variants={k: list(VARIANTS[k]) for k in a.variants}, threads=a.threads, run_s={k: round(v) for k, v in secs.items()},
                total_s=round(time.time() - t_all), stats="vs_policy: variant minus unmodified policy on identical episodes, exposure-weighted, (clip, start) cluster bootstrap (eval_opponent.vs_unaimed)",
                **EO.provenance(S["m"], S["cfg"], {v: {**S["cc"], "oracle": {"ops": list(VARIANTS[v]), **prm.get(v, {})}} for v in a.variants}))
    p = Path(a.out); prev = json.load(open(p)) if p.exists() else None
    v1 = prev.get("v1", prev if prev and "go_no_go" in prev and "history" not in prev.get("meta", {}) else None) if prev else None   # the v1 file itself (round 1), or its 'v1' key
    res = dict(v1=v1, meta=meta, reproduction=repro, go_no_go=gate, variants=out, bouts={v: rows_of(eps, raw[v], logs[v]) for v in raw})
    p.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(p, "w"), indent=1)
    print(json.dumps(dict(reproduction=repro, go_no_go=gate), indent=1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["tune", "val"], default="val"); ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--out", default=None); ap.add_argument("--tuning", default=str(TUNING))
    ap.add_argument("--split", choices=["val", "train"], default="val", help="val stage: train only for a smoke of the stage")
    ap.add_argument("--configs", nargs="*", default=None, help="tune stage: a subset of GRID (smoke)")
    ap.add_argument("--variants", nargs="*", default=["policy", "arm_block_v2", "arm_block"])
    a = ap.parse_args()
    if a.stage == "tune": a.reps = a.reps or 4; tune(a)
    else: a.reps = a.reps or 16; a.out = a.out or "results/sparring/oracle_val.json"; val(a)

if __name__ == "__main__": main()
