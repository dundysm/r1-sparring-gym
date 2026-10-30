"""Full-body M2 v2 reward terms (M2V2_PLAN_v3 2.4 with the v3 fixes m1-m5 and B1 fix 5; build order M5). Pure torch.

Convention (mjlab): reward = weight x value x step_dt (0.02 s); 'he' = one clean head hit = 0.8 (spar_head_clean -40 x 0.02).
WEIGHTS per stage; values from FullBodyHitAccount (opponent -> defender) and the scene:
  spar_head_clean / _impulse, spar_torso_impulse   clean head episodes closed; head / torso N s of the strike limbs (run 2's terms)
  spar_block      HIT_R2 windows paid (m2: run 2's clause, plus any same-side hand or forearm landing in the window cancels it)
  spar_clearance  A only (m3: dropped in B, where it pays the lean-back); clearance_value
  fb_knockdown    1 on the rising edge of opp_down_ever iff the defender's work W on the opponent (torso, pelvis and leg shells, not arm
                  shells) over the last WORK_WINDOW_S >= W0 (and > 0) and no opponent strike limb's landed episode touched the
                  defender's head or torso within LAND_GATE_S (m1): -120 = -2.4 = -3 he per attributed knockdown
  fb_down_neutral rate / 60 while opp_down_now (non-sticky, m4): rate = the clean-head EMA per both-up minute of the env's population
                  row (DownNeutral); weight -he, so a downed second costs the clean hits an upright opponent would have landed
  fb_down_approach  max(0, v_pelvis . u) while opp_down_now, u = the unit xy vector from the defender's pelvis to the downed opponent's
  fb_down_contact   the pelvis + leg shells' impulse on the opponent / step_dt while opp_down_now (-0.1 per N s)
  fb_leg_tangle   the same impulse / step_dt always (-0.05 per N s)
  fb_lean_away    B: max(0, lean - 5 deg)^2, lean = the torso up axis' tilt away from the opponent's pelvis bearing (deg, signed)
  res_l2          B: |residual|^2, weight res_l2_weight(B iteration): -0.5 annealed to -0.1 over B 0-2000
  spar_ring, spar_alive and the base Unitree / RS terms: unchanged (gpu/sparring_task.py), listed for the weights only.
Push mask (m5): every hit-derived term (spar_head_clean, spar_head_impulse, spar_torso_impulse, spar_block, fb_knockdown,
fb_down_approach, fb_down_contact, fb_leg_tangle) is 0 while t - t_push <= PUSH_MASK_S; fb_down_neutral is not a hit term."""
import math, torch

HE, STEP_DT = 0.8, 0.02
WORK_WINDOW_S, LAND_GATE_S, PUSH_MASK_S, LEAN_FREE_DEG = 1.0, 0.3, 0.5, 5.0
W0 = 0.29906465943470145   # J, frozen by scripts/fb_m5_check.py G0-12: the 10th percentile of W over G's EO knockdowns with contact on
# 60300-60363 (4 knockdowns, 2 with contact: W 0 and 2.9906 J). n is tiny (M5_NOTES): the main session decides before M11's hash
ROWS = ("t0_nominal", "t0_shadow", "t1", "t1_dr")   # population rows (2.2), the down-neutral EMA's keys (m4)
WEIGHTS = {"A": dict(spar_head_clean=-40.0, spar_head_impulse=-20.0, spar_torso_impulse=-5.0, spar_block=40.0, spar_clearance=0.5, spar_ring=2.5, spar_alive=4.0,
                     fb_knockdown=-120.0, fb_down_neutral=-HE, fb_down_approach=-2.0, fb_down_contact=-0.1, fb_leg_tangle=-0.05)}
WEIGHTS["B"] = {**{k: v for k, v in WEIGHTS["A"].items() if k != "spar_clearance"}, "fb_lean_away": -0.01, "res_l2": -0.5}
HIT_TERMS = ("spar_head_clean", "spar_head_impulse", "spar_torso_impulse", "spar_block", "fb_knockdown", "fb_down_approach", "fb_down_contact", "fb_leg_tangle")
RES_L2 = (-0.5, -0.1, 2000)

def res_l2_weight(it_b):
    a, b, n = RES_L2; return a + (b - a) * min(max(float(it_b), 0.0) / n, 1.0)

def push_mask(t, t_push):
    """1 where the hit terms count, 0 within PUSH_MASK_S after the env's last push (t_push -inf: never pushed)."""
    return (t - t_push > PUSH_MASK_S).to(t.dtype)

def clearance_value(d, win):
    """Run 2's spar_clearance: sum over aimed head intents in [t_on - 0.15 s, t_apex + 0.15 s] (win (N, K)) of clip((d - 0.10) / 0.15,
    0, 1), d (N, K) the striking palm to the defender's head point."""
    return (((d - 0.10) / 0.15).clamp(0.0, 1.0) * win).sum(-1)

def lean_away_deg(torso_up, def_xy, opp_xy):
    """The torso up axis' tilt away from the opponent (deg): atan2(-up . u, up_z), u the unit xy bearing to the opponent's pelvis."""
    u = torch.nn.functional.normalize(opp_xy - def_xy, dim=-1); h = (torso_up[..., :2] * u).sum(-1)
    return torch.rad2deg(torch.atan2(-h, torso_up[..., 2]))

class KnockdownAttribution:
    """m1: the defender's work on the opponent per control step (FullBodyHitAccount.step['work']) in a ring of the last
    WORK_WINDOW_S; charge(edge, t, land_t) = edge & W >= w0 & W > 0 & no landed opponent strike within LAND_GATE_S before t."""
    def __init__(self, n, w0, dt=STEP_DT, dtype=torch.float32, device="cpu"):
        self.w0, self.L = float(w0), int(round(WORK_WINDOW_S / dt)); self.ring = torch.zeros(n, self.L, dtype=dtype, device=device); self.i = 0
    def push(self, work): self.ring[:, self.i] = work; self.i = (self.i + 1) % self.L
    def W(self): return self.ring.sum(1)
    def reset(self, ids): self.ring[ids] = 0
    def charge(self, edge, t, land_t):
        W = self.W(); return edge & (W >= self.w0) & (W > 0) & ~(t - land_t <= LAND_GATE_S + 1e-9)

class DownNeutral:
    """m4: per population row an exposure-weighted EMA of clean head hits per both-up minute: decayed sums C <- lam C + hits,
    T <- lam T + minutes (lam = 0.5^(1/half_life) per update; only rows with exposure in the update decay), rate = C / T, started at
    prior_pm over prior_min minutes."""
    def __init__(self, rows=len(ROWS), prior_pm=5.0, prior_min=10.0, half_life=500, dtype=torch.float64, device="cpu"):
        self.lam = 0.5 ** (1.0 / half_life); self.C = torch.full((rows,), prior_pm * prior_min, dtype=dtype, device=device); self.T = torch.full((rows,), prior_min, dtype=dtype, device=device)
    def update(self, hits, up_s, row):
        R = len(self.C); oh = torch.nn.functional.one_hot(row.long(), R).to(self.C.dtype)
        m = oh.T @ (up_s.to(self.C.dtype) / 60.0); lam = torch.where(m > 0, torch.full_like(m, self.lam), torch.ones_like(m))
        self.C = lam * self.C + oh.T @ hits.to(self.C.dtype); self.T = lam * self.T + m
    def rate_pm(self): return self.C / self.T.clamp(min=1e-9)

class FbRewards:
    """The fb term values for N envs, one control step at a time (step); total(values, stage, it_b) applies WEIGHTS x step_dt."""
    def __init__(self, n, w0=None, dt=STEP_DT, dtype=torch.float32, device="cpu", down=None):
        w0 = W0 if w0 is None else w0; assert w0 is not None and w0 > 0, "W0 is frozen by the M5 calibration"
        self.n, self.dt, self.dtype = n, dt, dtype; self.attr = KnockdownAttribution(n, w0, dt, dtype, device); self.down = down or DownNeutral(device=device)
        self.events = torch.zeros(n, dtype=torch.long, device=device)   # charged knockdowns this episode
    def reset(self, ids): self.attr.reset(ids); self.events[ids] = 0

    def step(self, acc, edge, down_now, t, row, t_push=None, v_pel=None, def_xy=None, opp_xy=None, torso_up=None, residual=None, clear=None):
        """acc: FullBodyHitAccount (process() already run for this step; its land_t); edge / down_now: fb_events.OppDown.update's edge and
        .now; t (N,) episode time; row (N,) population row; t_push (N,) the last push (None: never); v_pel (N, 3) the defender pelvis'
        linear velocity, def_xy / opp_xy (N, 2) the pelvises; torso_up (N, 3) the defender torso's z axis; residual (N, 14); clear (N,)
        precomputed clearance. -> {term: value (N,)}; the EMA is updated with this step's clean head hits while both were up."""
        s = acc.step; one = torch.ones_like(t); pm = one if t_push is None else push_mask(t, t_push); dn = down_now.to(self.dtype)
        self.attr.push(s["work"]); kd = self.attr.charge(edge, t, acc.land_t) & (pm > 0); self.events += kd.long()
        up = (~down_now).to(self.dtype); self.down.update(s["head_clean"] * up, up * self.dt, row)
        v = dict(spar_head_clean=s["head_clean"] * pm, spar_head_impulse=s["head_imp"] * pm, spar_torso_impulse=s["torso_imp"] * pm, spar_block=s["block"] * pm,
                 fb_knockdown=kd.to(self.dtype), fb_down_neutral=dn * self.down.rate_pm().to(self.dtype)[row.long()] / 60.0,
                 fb_down_contact=dn * s["tangle"] / self.dt * pm, fb_leg_tangle=s["tangle"] / self.dt * pm)
        if v_pel is not None:
            u = torch.nn.functional.normalize(opp_xy - def_xy, dim=-1); v["fb_down_approach"] = dn * (v_pel[..., :2] * u).sum(-1).clamp(min=0) * pm
        if torso_up is not None: v["fb_lean_away"] = (lean_away_deg(torso_up, def_xy, opp_xy) - LEAN_FREE_DEG).clamp(min=0) ** 2
        if residual is not None: v["res_l2"] = (residual ** 2).sum(-1)
        if clear is not None: v["spar_clearance"] = clear
        return v

    def total(self, v, stage, it_b=0):
        """-> (sum, {term: reward}) over the stage's weighted terms present in v (res_l2 annealed)."""
        W = dict(WEIGHTS[stage])
        if "res_l2" in W: W["res_l2"] = res_l2_weight(it_b)
        r = {k: W[k] * x * self.dt for k, x in v.items() if k in W}; return sum(r.values()), r
