"""Full-body M2 v2 reset events, latency draws and the eval's fall rules (M2V2_PLAN_v3 2.2-2.3, build order M4). Pure torch.

Latency        env-level buffers (env.fb_latency): k [N] obs lag in control steps, act_delay [N] one-step motor delay, per episode.
fb_draw_latency  the reset-mode event listed FIRST (B3): k ~ U{0, 1, 2}, act_delay ~ Bernoulli(0.5) (the eval: EO:103 draws the
               bout's obs delay, ENV:199 act_delay = rng < 0.5). mjlab's _reset_idx runs reset events before every manager reset
               (observation, action, command), so no reset reads a stale draw; nothing redraws inside an episode.
B3_ORDER       the call order of a reset in mjlab 1.2.0 (_reset_idx, then the step body's write_data_to_sim, the single
               sim.forward(), command_manager.compute, observation compute), replayed by the CPU check (M4 G0-6b) and asserted by
               ResetLog wherever the env wrapper logs it (P1).
fell_eval      the defender's termination, R1.fallen (MOD:174-176, configs/r1_assumptions.yaml fall): pelvis z < 0.45 or up-z <
               cos 60 deg. mjlab's termination manager reads derived quantities one physics substep stale (after mj_step, before
               the step's single forward), exactly what R1.fallen reads in the eval after mj_step.
OppDown        the opponent's state (EO:86, OPP_TILT 0.5 / OPP_Z 0.25): `now` non-sticky, recomputed every step (the eval's both-up
               logic, eval_opponent.py:192-202); `ever` sticky (metrics, the knockdown event's rising edge); `age` control steps
               since it last went down. An opponent fall never terminates."""
import numpy as np, torch

K_LAGS, P_ACT_DELAY = (0, 1, 2), 0.5
FALL_Z, FALL_TILT_DEG = 0.45, 60.0                       # configs/r1_assumptions.yaml fall (asserted in tests/test_fb_events.py)
UP_MIN = float(np.cos(np.deg2rad(FALL_TILT_DEG)))         # the eval's own float (R1.fallen: up < np.cos(np.deg2rad(max_tilt_deg)))
OPP_TILT, OPP_Z = 0.5, 0.25                              # scripts/eval_opponent.py OPP_TILT, OPP_Z
B3_ORDER = ("event:fb_draw_latency", "event:reset_defender", "obs.reset", "action.reset", "command.reset", "forward", "command.compute", "obs.compute")

class Latency:
    def __init__(self, n, device="cpu"):
        self.k = torch.zeros(n, dtype=torch.long, device=device); self.act_delay = torch.zeros(n, dtype=torch.bool, device=device)
        self.draws = torch.zeros(n, dtype=torch.long, device=device)   # per env: episodes drawn (P1: shares of k and act_delay)

def draw_latency(lat, ids, generator=None):
    ids = torch.as_tensor(ids, dtype=torch.long, device=lat.k.device); n = len(ids)
    lat.k[ids] = torch.randint(0, len(K_LAGS), (n,), device=lat.k.device, generator=generator)
    lat.act_delay[ids] = torch.rand(n, device=lat.k.device, generator=generator) < P_ACT_DELAY; lat.draws[ids] += 1
    return lat.k[ids], lat.act_delay[ids]

def fb_draw_latency(env, env_ids, generator=None):
    """mjlab EventTermCfg(func=fb_draw_latency, mode='reset'), first in the event list; env.fb_latency is a Latency."""
    if env_ids is None: env_ids = torch.arange(env.num_envs, device=env.fb_latency.k.device)
    return draw_latency(env.fb_latency, env_ids, generator)

class ResetLog:
    """Records the calls of one reset and checks them against B3_ORDER."""
    def __init__(self): self.calls = []
    def __call__(self, name): self.calls.append(name)
    def ok(self): return tuple(self.calls) == B3_ORDER

# ---- fall rules (z, up: pelvis height and the pelvis up axis' world z, i.e. xmat[2, 2], from the same derived quantities)
def up_z(quat):
    """xmat[2, 2] of a body from its xquat [..., 4] (wxyz), mju_quat2Mat's term order: w w - x x - y y + z z."""
    w, x, y, z = quat.unbind(-1); return w * w - x * x - y * y + z * z

def fell_eval(z, up): return (z < FALL_Z) | (up < UP_MIN)

def opp_down_now(z, up): return (up < OPP_TILT) | (z < OPP_Z)

class OppDown:
    def __init__(self, n, device="cpu"):
        self.now = torch.zeros(n, dtype=torch.bool, device=device); self.ever = torch.zeros(n, dtype=torch.bool, device=device)
        self.age = torch.zeros(n, dtype=torch.long, device=device)
    def reset(self, ids): self.now[ids] = False; self.ever[ids] = False; self.age[ids] = 0
    def update(self, z, up):
        """Once per control step after the physics; returns the knockdown edge (first time down in the episode)."""
        now = opp_down_now(z, up); edge = now & ~self.ever
        self.age = torch.where(now & ~self.now, torch.zeros_like(self.age), self.age + 1); self.now = now; self.ever |= now; return edge
