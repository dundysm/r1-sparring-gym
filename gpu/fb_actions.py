"""Full-body M2 v2 defender action term and the start's composition (M2V2_PLAN_v3 2.3 m11, 2.5; build order M4). Pure torch.

HybridDelayedAction  the action term: clip at action_clip and cast to float32 (ENV:94), applied = where(act_delay, a_prev, a),
                     a_prev <- a (ENV:96: one control step, 20 ms, of motor-command latency, per episode), joint targets
                     home + scale x applied (mjlab contract: no EMA), `applied` = the `actions` obs slot (ENV:125). act_delay is
                     read from the latency buffer at reset (fb_events.Latency, drawn before every manager reset). It reads nothing
                     else of the env: no observation (m11); a reset env's first delayed step applies 0 (ENV:74). Its own delay
                     replaces mjlab's DelayedActuatorCfg, whose DelayBuffer resets a reset env's lag to 0 and clamps it to the
                     buffer length in the first steps.
m3s_input / compose  the start H(D)/M3s (RUN4_PLAN_v3 2.1, P1's p1_eval.py M3sHybrid): G (legs and waist, action 0-13) reads
                     obs[:, :83] with its arm last-action slots replaced by the action the arm joint positions imply,
                     clip(obs[:, 25:35] / scale[14:24], +-clip); the arm net (14-23) reads its own width (111: the whole row; 83:
                     obs[:, :83]; blind: 28 zeros). Lives in the actor, which consumes the same delayed, noisy 111-d obs as at eval."""
import torch

ACT_DIM, LEGS, ARMS = 24, slice(0, 14), slice(14, 24)
ARM_Q, ARM_LAST = slice(25, 35), slice(73, 83)        # obs83: arm joint positions (q - home), arm last-action slots

class HybridDelayedAction:
    """N envs. latency: fb_events.Latency; home, scale [24] (float64 on CPU); clip: action_clip (r1_mjlab.yaml 1e9)."""
    def __init__(self, n, latency, home, scale, clip=1e9, dtype=torch.float64, device="cpu"):
        self.lat, self.clip, self.n = latency, float(clip), n
        self.home, self.scale = torch.as_tensor(home, dtype=dtype, device=device), torch.as_tensor(scale, dtype=dtype, device=device)
        self.a_prev = torch.zeros(n, ACT_DIM, dtype=torch.float32, device=device); self.applied = torch.zeros_like(self.a_prev)
        self.delay = torch.zeros(n, dtype=torch.bool, device=device); self.switches = torch.zeros((), dtype=torch.long, device=device)

    def reset(self, ids):
        self.a_prev[ids] = 0; self.applied[ids] = 0; self.delay[ids] = self.lat.act_delay[ids]

    def process(self, a):
        """Policy actions [N, 24] -> joint position targets [N, 24]; sets `applied` (the actions slot of the next obs)."""
        a = torch.clamp(torch.as_tensor(a), -self.clip, self.clip).to(torch.float32)
        self.switches += (self.lat.act_delay != self.delay).sum()   # act_delay constant per episode (M10: zero switches)
        self.applied = torch.where(self.delay[:, None], self.a_prev, a); self.a_prev = a
        return self.home + self.scale * self.applied.to(self.home.dtype)

def m3s_input(obs, scale, clip):
    """G's input: obs[:, :83] with obs[:, 73:83] := clip(obs[:, 25:35] / scale[14:24], +-clip), in obs' dtype."""
    g = obs[:, :83].clone(); s = torch.as_tensor(scale, dtype=obs.dtype, device=obs.device)
    g[:, ARM_LAST] = torch.clamp(obs[:, ARM_Q] / s[ARMS], -clip, clip); return g

def arm_input(obs, width):
    """The arm net's view of a 111-d row: 111 the row, 83 its proprio, 'zeros' the proprio + 28 zeros (blind twin)."""
    if width == 111: return obs
    if width == 83: return obs[:, :83]
    return torch.cat([obs[:, :83], torch.zeros(len(obs), 28, dtype=obs.dtype, device=obs.device)], 1)

def compose(obs, legs, arms, scale, clip, arm_width=111):
    """H(arms)/M3s on [N, 111] rows: legs(m3s_input) on 0-13, arms(arm_input) on 14-23. legs, arms: callables rows -> [N, 24]."""
    a = legs(m3s_input(obs, scale, clip)).clone(); b = arms(arm_input(obs, arm_width)); a[:, ARMS] = b[:, ARMS]; return a
