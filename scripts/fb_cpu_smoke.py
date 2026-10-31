"""M10 tiny CPU training smoke (M2V2_PLAN_v3 3 M10, m19; A5.1, A7, A9.2, A10, A11): the defender trained by a minimal PPO in K
replica envs on this CPU, through the same pieces the pod runs.
Env       SmokeEnv: fb_replica.Batch (A7: K bouts in lockstep, rowwise) at the R-EQ configuration (fb_m6_run.REQ_CONF: hulls 'mix' with the
          A10 names, solver 10/20, torch32 modules, the account scoring with the 'site' closing speed) with the training reset (reset
          'train': B3 order, k / act_delay drawn first by the episode's generator, DR, a bout heading, t0 on the clean grid), auto-reset per
          env as mjlab's step does it: physics, then OppDown / the account / FbRewards (the termination phase, one-substep-stale reads),
          terminations (fell_eval, ring_out 1 m, nan, time_out 20 s bootstrapped), rewards, the resets (the account closed first), the
          eval's outer loops, then the observation. Population: T0 nominal rows only (train clips, eval distances 0.40-0.55, aimed, no
          pushes): A11.2 makes every stage's iterations 1-100 T0-only, which covers the plumbing (and the canary's first 100).
          Episodes: seed s of the role's block -> clip TRAIN[(s - s0) % 3], distance HARD[((s - s0) // 3) % 4] (fb_record's map), checked
          by fb_seeds before it is simulated.
Rewards   fb_rewards.FbRewards (WEIGHTS of stage A for A0 / A, B for B; W0, W_EPS, START_S frozen there) + spar_ring + spar_alive, the
          task's definitions (gpu/fullbody_task.py: clearance on the aimed head intents, ring on the start spot, alive = not terminated).
          The base Unitree / RS terms (2.4 'unchanged') live in mjlab only (feet sensors, posture tables): not in this smoke.
Policy    the M9 runner factory fullbody_task.FbActorModel (HybridActor: A0 nothing trains, A arms + std (ArmOnlyGaussian), B to_stage_b +
          residual, Gaussian over 24; blind flag in every state dict, A9.2); the critic an rsl_rl 5.0.1 MLPModel equivalent (hybrid_actor.Net:
          EmpiricalNormalization -> 512-256-128 ELU -> 1, fresh) on the clean 111 + the 10 intent features + the PRIV block + ring_home (the
          task's critic without run 3's mjlab-only base inputs). Stages restart as on the pod (FbRunner.load: optimizer kept A0 -> A, fresh
          A -> B via load_into_b; every restart a new env reset).
PPO       rsl_rl 5.0.1 PPO.act / process_env_step / compute_returns / update and RolloutStorage.mini_batch_generator, operation for
          operation (no tensordict): unitree_r1_ppo_runner_cfg's algorithm (clip 0.2, clipped value loss, value coef 1, 5 epochs x 4 mini-
          batches, adaptive lr at desired_kl 0.01, gamma 0.99, lam 0.95, max grad norm 1, Adam) with fullbody_task.RL's lr 2e-4 and entropy
          0.003, 24 steps per env, time-outs bootstrapped.
Checks    plumbing (blocking; role 'smoke', 69000-69099; 8 envs x 24 steps, 20 A0 + 30 A + 30 B iterations, then the scripted episodes):
          no NaN; legs bit-frozen in A0 / A (G's state bit-equal, no leg grad, sampled leg dims = G's mean); residual exactly 0 at B start
          (B's action = A's on the same rows); value loss falls >= 30% over A0 (mean of iterations 18-20 vs 1-3); every reward term fires with
          the right sign and a per-second magnitude within 2x of the 2.4 table (reward / its driving quantity measured independently in numpy
          from the MuJoCo state, the sensors and the eval's HitMeter, vs TABLE), fb_knockdown / fb_down_* / the W_eps and start-window rule on
          scripted episodes; k and act_delay constant per episode (zero switches); knockdown, tangle, lean and W metrics populated;
          checkpoints carry the blind flag (and a cross-twin load is refused); head hits non-zero in the env's own account.
          canary (report only, m19; role 'canary', 69100-69999): 16 envs, 400 A iterations from the plumbing's A0 checkpoint.
Usage: python -I scripts/fb_cpu_smoke.py plumbing --out DIR | canary --out DIR [--a0 DIR/ckpt_A0.pt]"""
import sys, os, json, time, math, copy, argparse
from pathlib import Path
from itertools import chain
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, torch, mujoco
import fb_replica as RP, fb_rewards as RW, fb_events as EV, fb_obs as OB, fb_hits as HT, fb_opponent as FO, fb_dr as DR
import fullbody_task as FT, hybrid_actor as HA

TRAIN, HARD = ("kfab_200", "kfab_201", "kfab_203"), (0.4, 0.45, 0.5, 0.55)
SMOKE_CONF = dict(hulls="mix", solver=(10, 20), backend="torch32", reset="train", scorer="account", vmode="site")   # fb_m6_run.REQ_CONF, training reset
EP_S, DT, SPE = 20.0, 0.02, 24
PPO_CFG = dict(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3,
               schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)   # src/tasks/velocity/config/r1/rl_cfg.py (unitree_rl_mjlab @1425b15)
PLUMB = dict(n=8, A0=20, A=30, B=30, role="smoke", block=(69000, 69099), train=(69000, 69079), scripted=(69080, 69099))
CANARY = dict(n=16, A=400, role="canary", block=(69100, 69999))
VLOSS_DROP, MAG_X = 0.30, 2.0
# 2.4's per-unit rate of each term (reward per unit of its driving quantity, measured independently): (rate, unit, stages weighted)
TABLE = dict(spar_head_clean=(-0.8, "HitMeter clean head hit", "AB"), spar_head_impulse=(-0.4, "N s strike limbs on head shells", "AB"),
             spar_torso_impulse=(-0.1, "N s strike limbs on torso shells", "AB"), spar_block=(0.8, "paid HIT_R2 window", "AB"),
             spar_clearance=(0.5, "clearance x s", "A"), spar_ring=(2.5, "ring x s", "AB"), spar_alive=(4.0, "live s", "AB"),
             fb_knockdown=(-2.4, "charged knockdown (rule recomputed)", "AB"), fb_down_neutral=(-5.0 * RW.HE / 60, "downed s (2.4: 4-6 clean head hits per min, 5)", "AB"),
             fb_down_approach=(-2.0, "m toward the downed opponent", "AB"), fb_down_contact=(-0.1, "N s pelvis + leg shells while down", "AB"),
             fb_leg_tangle=(-0.05, "N s pelvis + leg shells", "AB"), fb_lean_away=(-0.01, "deg^2 s beyond 5 deg", "B"), res_l2=(-0.5, "|r|^2 s", "B"))
TERMS = tuple(TABLE)

# ---- episodes and seeds
def episode(seed, block0):
    c = TRAIN[(seed - block0) % 3]; return dict(seed=int(seed), clip=c, distance=HARD[((seed - block0) // 3) % 4])

class Seeds:
    """The role's seeds in order, each checked by fb_seeds before its episode is simulated; refuses when the range is spent."""
    def __init__(self, role, lo, hi, block0):
        from fb_seeds import check
        self.check, self.role, self.it, self.hi, self.b0, self.used = check, role, iter(range(lo, hi + 1)), hi, block0, []
    def __call__(self):
        s = next(self.it, None); assert s is not None, f"{self.role}: seeds spent (last {self.hi})"
        ep = episode(s, self.b0); self.check([s], self.role, {ep["clip"]}); self.used.append(s); return ep

# ---- the env
class SmokeEnv:
    """n replica envs (fb_replica.Batch slots, module docstring), stepped by a policy's actions; auto-reset per env (auto_reset False:
    a finished env is frozen out: no reset, its later steps count nowhere). scripts: per env None or a Script (scripted episodes)."""
    def __init__(self, n, seeds, conf=SMOKE_CONF, seconds=EP_S, ctrl="H", auto_reset=True, scripts=None):
        c = RP.conf(**conf); ct, cfg, lag, m, names = RP.world(ctrl, c)
        self.B = B = RP.Batch(ct, cfg, c, m, n, native=names, seconds=seconds, lag=lag, rowwise=True); B.n = n; B._modules(n)
        self.n, self.m, self.S, self.seeds, self.T, self.auto = n, m, B.S, seeds, int(round(seconds / DT)), auto_reset
        self.rw = RW.FbRewards(n, dtype=torch.float32); self.stage, self.it_b = "A", 0
        self.pal = [m.site(f"opp_{s}_palm").id for s in ("left", "right")]; self.head_p = np.array(FT.DEF_HEAD_T)
        sh = B.S[0].sens.s; z = np.asarray(sh.bzone_idx); self.bt = np.isin(z, [HT.ZONES.index(x) for x in HT.TANGLE_ZONES]); self.bw = np.isin(z, [HT.ZONES.index(x) for x in HT.WORK_ZONES])
        zi = np.asarray(sh.zone_idx); self.hz, self.tz = zi == HT.HEAD, zi == HT.TORSO
        self.dr_p = [None] * n; self.ep = [None] * n; self.lat0 = [None] * n; self.script = list(scripts or [None] * n); self.residual = None
        self.Wr = np.zeros((n, int(round(RW.WORK_WINDOW_S / DT)))); self.wi = 0; self.down = np.zeros(n, bool); self.ever = np.zeros(n, bool)
        self.over = np.zeros(n, bool); self.inj = [None] * n
        self.q = self._zq(); self.log = dict(episodes=[], kd=[], switch_ep=0); self.v_last = {}
        for A in B.accts.values(): A.reset(list(range(n))); A.log.clear()
        for i in range(n): self._reset(i)
        self.obs = self._obs()

    @staticmethod
    def _zq(): return {k: 0.0 for k in ("alive_s", "ring", "clear", "head_imp", "torso_imp", "tangle", "down_s", "approach", "down_contact", "lean", "res", "work", "kd_expected", "kd_edges", "hm_head_clean", "paid", "steps")}

    def _reset(self, i):
        """Close env i's account episodes (FbOppCommand.reset), then the replica's training reset (B3 order) on a fresh seed."""
        B, S = self.B, self.S
        if self.ep[i] is not None: self._harvest(i)
        B.acct.close_envs([i]); self._paid()
        ep = dict(self.seeds(), **({} if self.script[i] is None else {"script": self.script[i].name})); cap = {}
        f = DR.sample
        def samp(*a, **k): p = f(*a, **k); cap["p"] = p; return p
        DR.sample = samp
        try: B._reset(i, S[i], ep)
        finally: DR.sample = f
        self.dr_p[i] = {k: v[0].float() for k, v in cap["p"].items()}; self.ep[i] = ep; self.rw.reset([i])
        self.lat0[i] = (int(B.lat.k[i]), bool(B.lat.act_delay[i])); self.Wr[i] = 0; self.down[i] = self.ever[i] = False; self.inj[i] = None

    def _harvest(self, i):
        """The finished episode's eval-HitMeter counts (independent of the account) and its row."""
        s = self.S[i]; secs = s.t * DT; hm = s.hm.summary(max(secs, DT)); self.q["hm_head_clean"] += hm["n"]["head_clean"]
        self.log["episodes"].append(dict(seed=self.ep[i]["seed"], clip=self.ep[i]["clip"], steps=s.t, k=self.lat0[i][0], act_delay=self.lat0[i][1], hm=hm["n"]))

    def _paid(self):
        A = self.B.acct; self.q["paid"] += sum(x["paid"] for x in A.log if x["kind"] == "window"); A.log.clear()

    def _obs(self):
        o = self.B._obs(list(range(self.n))); return torch.as_tensor(np.stack([o[i] for i in range(self.n)]), dtype=torch.float32)

    def critic_obs(self):
        """[n, 152]: the clean 111 (undelayed, noiseless), the intent features, the PRIV block, ring_home (fullbody_task's functions)."""
        B, S, n = self.B, self.S, self.n; f, v, o = S[0].r.free_q, S[0].r.free_v, S[0].oi
        q = np.stack([S[i].d.qpos for i in range(n)]); qv = np.stack([S[i].d.qvel for i in range(n)]); T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        yaw = lambda w, x, y, z: np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        dy = yaw(*q[:, f + 3:f + 7].T); oy = yaw(*q[:, o.qadr + 3:o.qadr + 7].T)
        now = torch.tensor([S[i].t * DT + DT for i in range(n)], dtype=torch.float64)
        it = FT.intent_features(B.aim, torch.zeros(n, B.aim.valid.shape[1], dtype=torch.bool), now)
        dr = {k: torch.stack([self.dr_p[i][k] for i in range(n)]) for k in self.dr_p[0]}
        pr = FT.priv_block(T(q[:, f:f + 2]), T(dy), T(q[:, o.qadr:o.qadr + 3]), T(oy), T(qv[:, o.vadr:o.vadr + 3]), B.od.now, B.od.age.float() * DT, B.lat.act_delay, B.lat.k, dr,
                           B.acct.step["tangle"].float(), self.rw.attr.W().float())
        rh = FT.ring_home(T(np.stack([S[i].spot for i in range(n)])), T(q[:, f:f + 2]), T(dy))
        return torch.cat([B.obs.clean.float(), it, pr, rh], 1)

    def step(self, a):
        """a [n, 24] -> (obs, reward [n], dones [n], time_outs [n], info); rewards and dones of frozen-out envs are 0."""
        B, S, n, q = self.B, self.S, self.n, self.q; live = list(range(n))
        for i in live:
            if self.script[i] is not None and not self.over[i]: self.script[i].pre(self, i)
        tg = B.act.process(torch.as_tensor(a, dtype=torch.float32)).double().numpy()
        for i in live: s = S[i]; B._dr(s); s.d.ctrl[s.r.act] = tg[i]; mujoco.mj_kinematics(s.m, s.d)
        B._opp(live)
        for i in live:
            s = S[i]; d = s.d; B._dr(s); s.sens.begin()
            for _ in range(s.nsub):
                d.xfrc_applied[:] = 0; mujoco.mj_step(s.m, d)
                for h in s.hooks: h.substep(s.m, d)
            d.xfrc_applied[:] = 0; s.t += 1
        P = lambda x: torch.from_numpy(np.stack([x(S[i].d) for i in live]).copy()); pel, op = S[0].r.pel, S[0].oi.pel
        fallen = EV.fell_eval(P(lambda d: d.xpos[pel][2]), EV.up_z(P(lambda d: d.xquat[pel])))
        B._score(live)
        edge = B.od.update(P(lambda d: d.xpos[op][2]), EV.up_z(P(lambda d: d.xquat[op])))
        t = torch.tensor([S[i].t * DT for i in live], dtype=torch.float32); tr = S[0].r.torso
        up = P(lambda d: d.xmat[tr].reshape(3, 3)[:, 2]).float(); dxy = P(lambda d: d.xpos[pel][:2]).float(); oxy = P(lambda d: d.xpos[op][:2]).float()
        vp = P(lambda d: d.qvel[S[0].r.free_v:S[0].r.free_v + 3]).float(); res = self.residual
        for i in live:
            if self.script[i] is not None and not self.over[i] and bool(edge[i]): self.script[i].on_edge(self, i)   # W injection (A5.1 (a) scripted)
        v = self.rw.step(B.acct, edge, B.od.now, t, torch.zeros(n, dtype=torch.long), v_pel=vp, def_xy=dxy, opp_xy=oxy, torso_up=up, residual=res, clear=self._clearance(t))
        if res is None: v["res_l2"] = torch.zeros(n)
        f = S[0].r.free_q; rr = np.array([np.linalg.norm(S[i].d.qpos[f:f + 2] - S[i].spot) for i in live])
        v["spar_ring"] = torch.as_tensor(1.0 - np.clip((rr - FT.RING["free"]) / (FT.RING["out"] - FT.RING["free"]), 0.0, 1.0), dtype=torch.float32)
        nan = torch.tensor([not (np.isfinite(S[i].d.qpos).all() and np.isfinite(S[i].d.qvel).all()) for i in live])
        term = fallen | torch.as_tensor(rr > FT.RING["out"]) | nan; tout = torch.tensor([S[i].t >= self.T for i in live]) & ~term
        v["spar_alive"] = (~term).float(); self.v_last = v; on = torch.as_tensor(~self.over)
        tot, parts = self.rw.total({k: x for k, x in v.items() if k not in ("spar_ring", "spar_alive")}, self.stage, self.it_b)
        for k, w in (("spar_ring", 2.5), ("spar_alive", 4.0)): parts[k] = w * v[k] * DT; tot = tot + parts[k]
        parts = {k: x * on for k, x in parts.items()}; tot = tot * on
        self._quant(v, edge, t, term, dxy, oxy, vp, up)
        done = (term | tout) & on; nan = nan & on
        info = dict(parts=parts, values=v, edge=edge & on, nan=int(nan.sum()), fallen=fallen & on, ring_out=torch.as_tensor(rr > FT.RING["out"]) & on, lean=RW.lean_away_deg(up, dxy, oxy) * on, on=on)
        for i in torch.nonzero(done).flatten().tolist():
            if self.auto: self._reset(i)
            else: self._harvest(i); self.over[i] = True
        for i in live:
            if not bool(done[i]): S[i]._face(); S[i]._hold()
            if self.script[i] is not None and not self.over[i]: self.script[i].post(self, i)
        self.obs = self._obs(); self._switch_check(); return self.obs, tot.float(), done, tout & on, info

    def _switch_check(self):
        """k and act_delay of every env equal its episode's draw (B3: drawn once at reset), on top of the modules' own switch counters."""
        B = self.B
        for i in range(self.n):
            if (int(B.lat.k[i]), bool(B.lat.act_delay[i])) != self.lat0[i] or int(B.obs.k_ep[i]) != self.lat0[i][0] or bool(B.act.delay[i]) != self.lat0[i][1]: self.log["switch_ep"] += 1

    def _clearance(self, t):
        """fullbody_task FbOppCommand._clearance on the replica's state (palm sites, torso head point; the aim's intent slots)."""
        A, S, n = self.B.aim, self.S, self.n; tr = S[0].r.torso; now = t.double()
        hp = torch.as_tensor(np.stack([S[i].d.xpos[tr] + S[i].d.xmat[tr].reshape(3, 3) @ self.head_p for i in range(n)]))
        palm = torch.as_tensor(np.stack([S[i].d.site_xpos[self.pal] for i in range(n)])); pl = palm.gather(1, A.limb[..., None].expand(-1, -1, 3).clamp(max=1))
        d = torch.linalg.norm(pl - hp[:, None], dim=-1); win = A.valid & (A.gated == 0) & A.aimed[:, None] & (now[:, None] >= A.t_on - 0.15) & (now[:, None] <= A.t_apex + 0.15)
        return RW.clearance_value(d, win.double()).float()

    def _quant(self, v, edge, t, term, dxy, oxy, vp, up):
        """The driving quantities of TABLE, measured independently in numpy (MuJoCo state, the sensors' substep forces, the aim's slots)."""
        S, q, n, A = self.S, self.q, self.n, self.B.aim; op = S[0].oi.pel; tr = S[0].r.torso; f = S[0].r.free_q; j = self.wi
        for i in range(n):
            if self.over[i]: continue
            d, s = S[i].d, S[i].sens; alive = not bool(term[i]); q["steps"] += 1; q["alive_s"] += DT * alive
            r = np.linalg.norm(d.qpos[f:f + 2] - S[i].spot); q["ring"] += DT * (1.0 - min(max((r - 0.25) / 0.75, 0.0), 1.0))
            nrm = np.linalg.norm(s.F, axis=-1); q["head_imp"] += nrm[:, self.hz].sum() * HT.DT; q["torso_imp"] += nrm[:, self.tz].sum() * HT.DT
            fb = np.linalg.norm(s.Fb, axis=-1); tg = fb[self.bt].sum() * HT.DT; w = np.clip((s.Fb * s.vb).sum(-1), 0, None)[self.bw].sum() * HT.DT
            q["tangle"] += tg; q["work"] += w; self.Wr[i, j] = w
            dn = bool(d.xmat[op][8] < EV.OPP_TILT or d.xpos[op][2] < EV.OPP_Z); new = dn and not self.ever[i]; self.down[i] = dn; self.ever[i] |= dn
            if dn:
                u = d.xpos[op][:2] - d.xpos[S[i].r.pel][:2]; u = u / max(np.linalg.norm(u), 1e-12)
                q["down_s"] += DT; q["down_contact"] += tg; q["approach"] += DT * max(0.0, float(d.qvel[S[i].r.free_v:S[i].r.free_v + 2] @ u))
            if new:
                W = self.Wr[i].sum() if self.inj[i] is None else self.inj[i]; tt = S[i].t * DT; land = float(self.B.acct.land_t[i])
                ok = W >= RW.W0 and W > 0 and W > RW.W_EPS and tt > RW.START_S + 1e-9 and not (tt - land <= RW.LAND_GATE_S + 1e-9); q["kd_edges"] += 1; q["kd_expected"] += ok
                self.log["kd"].append(dict(seed=self.ep[i]["seed"], t=round(tt, 3), W=float(W), W_np=float(self.Wr[i].sum()), W_rw=float(self.rw.attr.W()[i]), injected=self.inj[i] is not None, land_t=land,
                                           expected=bool(ok), charged=bool(v["fb_knockdown"][i] > 0), script=self.ep[i].get("script")))
            uz = d.xmat[tr].reshape(3, 3)[:, 2]; b = d.xpos[op][:2] - d.xpos[S[i].r.pel][:2]; b = b / max(np.linalg.norm(b), 1e-12)
            lean = np.degrees(np.arctan2(-(uz[:2] @ b), uz[2])); q["lean"] += DT * max(0.0, lean - RW.LEAN_FREE_DEG) ** 2
            hp = d.xpos[tr] + d.xmat[tr].reshape(3, 3) @ self.head_p; now = S[i].t * DT
            for k in range(A.valid.shape[1]):
                if bool(A.valid[i, k]) and int(A.gated[i, k]) == 0 and bool(A.aimed[i]) and float(A.t_on[i, k]) - 0.15 <= now <= float(A.t_apex[i, k]) + 0.15:
                    dd = np.linalg.norm(d.site_xpos[self.pal[min(int(A.limb[i, k]), 1)]] - hp); q["clear"] += DT * min(max((dd - 0.10) / 0.15, 0.0), 1.0)
            if self.residual is not None: q["res"] += DT * float((self.residual[i].double() ** 2).sum())
        self.wi = (self.wi + 1) % self.Wr.shape[1]

    def finish(self):
        """Harvest the live episodes (HitMeter counts, paid windows) at the end of a run."""
        for i in range(self.n): self._harvest(i)
        self.B.acct.close_envs(list(range(self.n))); self._paid()

class Script:
    """A scripted episode: at t_kick the defender's root gets vd m/s toward the opponent, the opponent vo m/s away from it and a tip of
    w rad/s away (its root angular velocity, local frame), so the opponent falls with (vd > 0) or without (vd 0) the defender's work; while
    the opponent is down the defender is commanded toward it at walk m/s (fb_down_approach / fb_down_contact); inject: the W (J) the
    knockdown edge sees (the attribution ring set to it at the edge: A5.1 (a)'s floor on a scripted value)."""
    def __init__(self, name, t_kick, vd, vo, w, walk=0.5, inject=None):
        self.name, self.k, self.vd, self.vo, self.w, self.walk, self.inject = name, int(round(t_kick / DT)), vd, vo, w, walk, inject
    def pre(self, env, i):
        s = env.S[i]; d, o = s.d, s.oi
        if s.t != self.k: return
        u = d.xpos[o.pel][:2] - d.xpos[s.r.pel][:2]; u = u / np.linalg.norm(u)
        d.qvel[s.r.free_v:s.r.free_v + 2] += self.vd * u; d.qvel[o.vadr:o.vadr + 2] += self.vo * u
        d.qvel[o.vadr + 3:o.vadr + 6] += d.xmat[o.pel].reshape(3, 3).T @ (np.cross([0.0, 0.0, 1.0], [u[0], u[1], 0.0]) * self.w)
    def post(self, env, i):
        if self.walk and env.down[i]: env.S[i].command[0] = self.walk; env.S[i].command[1] = 0.0
    def on_edge(self, env, i):
        if self.inject is None: return
        env.rw.attr.ring[i] = 0; env.B.acct.step["work"][i] = self.inject; env.inj[i] = float(torch.tensor(self.inject, dtype=torch.float32))

# scripted rounds (plumbing; 8 envs each, one episode per env, SCRIPT_S): pushed in after the start window (charged, unless a strike landed
# within 0.3 s: m1), pushed in at once (the edge inside the start window, A5.1 (b)), tipped away untouched (W = 0), and untouched falls with W
# injected at 0.5 / 0.9 x and 2 x W_EPS (A5.1 (a)); 3 s each. Kicks fixed before the plumbing run on probes of seeds 69000-69031 (A actor)
_P, _E, _N = (2.5, 0.5, 10.0), (2.0, 1.5, 10.0), (0.0, 2.0, 10.0)
SCRIPTS = ([Script("push_in_a", 1.0, *_P), Script("push_in_b", 1.0, *_E), Script("push_in_c", 1.5, *_P), Script("early_a", 0.02, *_P), Script("early_b", 0.04, *_E),
            Script("early_c", 0.02, 2.0, 1.0, 8.0), Script("no_contact_a", 1.0, *_N), Script("no_contact_b", 1.5, *_N)],
           [Script("inject_lo_a", 1.0, *_N, inject=0.5 * RW.W_EPS), Script("inject_lo_b", 1.5, *_N, inject=0.5 * RW.W_EPS), Script("inject_lo_c", 1.0, 0.0, 2.5, 10.0, inject=0.9 * RW.W_EPS),
            Script("inject_hi_a", 1.0, *_N, inject=2.0 * RW.W_EPS), Script("inject_hi_b", 1.5, *_N, inject=2.0 * RW.W_EPS), Script("inject_hi_c", 1.0, 0.0, 2.5, 10.0, inject=2.0 * RW.W_EPS),
            Script("push_in_d", 0.8, *_P), Script("early_d", 0.02, *_P)])
SCRIPT_S = 3.0

# ---- PPO (rsl_rl 5.0.1, module docstring)
class Storage:
    def __init__(self, T, n, na, nc):
        z = lambda *s: torch.zeros(T, n, *s); self.T, self.n, self.i = T, n, 0
        self.oa, self.oc, self.act, self.rew, self.done, self.val, self.logp, self.mu, self.sd = z(OB.OBS_DIM), z(nc), z(na), z(1), z(1), z(1), z(1), z(na), z(na)
        self.ret, self.adv = z(1), z(1)
    def add(self, oa, oc, a, r, d, v, lp, mu, sd):
        i = self.i
        for x, y in ((self.oa, oa), (self.oc, oc), (self.act, a), (self.rew, r.view(-1, 1)), (self.done, d.view(-1, 1).float()), (self.val, v), (self.logp, lp.view(-1, 1)), (self.mu, mu), (self.sd, sd)): x[i] = y
        self.i += 1
    def batches(self, nmb, epochs):
        bs = self.n * self.T; mbs = bs // nmb; idx = torch.randperm(nmb * mbs); F = lambda x: x.flatten(0, 1)
        oa, oc, a, v, ret, lp, adv, mu, sd = (F(x) for x in (self.oa, self.oc, self.act, self.val, self.ret, self.logp, self.adv, self.mu, self.sd))
        for _ in range(epochs):
            for i in range(nmb):
                b = idx[i * mbs:(i + 1) * mbs]; yield oa[b], oc[b], a[b], v[b], adv[b], ret[b], lp[b], (mu[b], sd[b])

class PPO:
    """rsl_rl 5.0.1 PPO on (actor, critic): act, process_env_step, compute_returns, update (module docstring)."""
    def __init__(self, actor, critic, n, nc, lr, entropy_coef, cfg=PPO_CFG):
        self.actor, self.critic, self.c = actor, critic, dict(cfg, learning_rate=lr, entropy_coef=entropy_coef); self.lr = lr
        self.opt = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=lr); self.st = Storage(SPE, n, 24, nc)
    def obs(self, oa, oc): return {"actor": oa, "critic": oc}
    def act(self, oa, oc):
        a = self.actor(self.obs(oa, oc), stochastic_output=True).detach(); v = self.critic(oc).detach(); lp = self.actor.get_output_log_prob(a).detach()
        mu, sd = (p.detach() for p in self.actor.output_distribution_params); self.tr = (oa, oc, a, v, lp, mu, sd); return a
    def process(self, oa2, oc2, r, done, tout):
        self.actor.update_normalization(self.obs(oa2, oc2)); self.critic.obs_normalizer.update(oc2)
        oa, oc, a, v, lp, mu, sd = self.tr; r = r.clone() + self.c["gamma"] * torch.squeeze(v * tout.unsqueeze(1).float(), 1)
        self.st.add(oa, oc, a, r, done, v, lp, mu, sd)
    def returns(self, oc_last):
        st, g, lam = self.st, self.c["gamma"], self.c["lam"]; last = self.critic(oc_last).detach(); adv = 0
        for k in reversed(range(st.T)):
            nv = last if k == st.T - 1 else st.val[k + 1]; nt = 1.0 - st.done[k].float(); delta = st.rew[k] + nt * g * nv - st.val[k]
            adv = delta + nt * g * lam * adv; st.ret[k] = adv + st.val[k]
        st.adv = st.ret - st.val; st.adv = (st.adv - st.adv.mean()) / (st.adv.std() + 1e-8)
    def update(self):
        c = self.c; mv = ms = me = 0.0; kls = []
        for oa, oc, a, v0, adv, ret, lp0, old in self.st.batches(c["num_mini_batches"], c["num_learning_epochs"]):
            self.actor(self.obs(oa, oc), stochastic_output=True); lp = self.actor.get_output_log_prob(a); val = self.critic(oc)
            params = tuple(p for p in self.actor.output_distribution_params); ent = self.actor.output_entropy
            if c["desired_kl"] is not None and c["schedule"] == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(old, params).mean(); kls.append(float(kl))
                    if kl > c["desired_kl"] * 2.0: self.lr = max(1e-5, self.lr / 1.5)
                    elif c["desired_kl"] / 2.0 > kl > 0.0: self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups: g["lr"] = self.lr
            ratio = torch.exp(lp - torch.squeeze(lp0)); s1 = -torch.squeeze(adv) * ratio; s2 = -torch.squeeze(adv) * torch.clamp(ratio, 1 - c["clip_param"], 1 + c["clip_param"])
            sl = torch.max(s1, s2).mean()
            if c["use_clipped_value_loss"]:
                vc = v0 + (val - v0).clamp(-c["clip_param"], c["clip_param"]); vl = torch.max((val - ret).pow(2), (vc - ret).pow(2)).mean()
            else: vl = (ret - val).pow(2).mean()
            loss = sl + c["value_loss_coef"] * vl - c["entropy_coef"] * ent.mean()
            self.opt.zero_grad(); loss.backward(); self.grads()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c["max_grad_norm"]); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c["max_grad_norm"]); self.opt.step()
            mv += vl.item(); ms += sl.item(); me += ent.mean().item()
        k = c["num_learning_epochs"] * c["num_mini_batches"]; self.st.i = 0
        return dict(value=mv / k, surrogate=ms / k, entropy=me / k, kl=float(np.mean(kls)) if kls else None, lr=self.lr)
    def grads(self):
        """Legs never get a gradient (G frozen; asserted every update)."""
        assert all(p.grad is None for p in self.actor.legs.parameters()), "a leg parameter got a gradient"
    def save(self, path, it, infos):
        torch.save(dict(actor_state_dict=self.actor.state_dict(), critic_state_dict=self.critic.state_dict(), optimizer_state_dict=self.opt.state_dict(), iter=it, infos=infos), path); return path

def make_actor(stage, blind=False, seed=0):
    """The M9 runner factory (rsl_rl's model constructor signature)."""
    return FT.FbActorModel(None, {"actor": ["actor"], "critic": ["critic"]}, "actor", 24, fb_stage=stage, fb_blind=blind, fb_root=str(ROOT), fb_seed=seed)

def restart(stage, ckpt, n, nc, blind=False, seed=0):
    """A stage restart as the pod does it: the factory's actor for `stage`, a fresh critic, then FbRunner.load's rule (actor, critic,
    the optimizer only when the stage keeps the parameter set)."""
    a = make_actor(stage, blind, seed); cr = HA.Net(nc, 1); ppo = PPO(a, cr, n, nc, FT.RL["B" if stage == "B" else "A"]["learning_rate"], FT.RL["B" if stage == "B" else "A"]["entropy_coef"])
    if ckpt is not None:
        d = torch.load(ckpt, map_location="cpu", weights_only=False); same = "optimizer_state_dict" in d and any(k.startswith("residual.") for k in d["actor_state_dict"]) == (stage == "B")
        a.load_state_dict(d["actor_state_dict"], strict=True); cr.load_state_dict(d["critic_state_dict"], strict=True)
        if same: ppo.opt.load_state_dict(d["optimizer_state_dict"])
    return ppo

def g_ref():
    """G's actor state as loaded from its checkpoint (the bit-frozen reference)."""
    return {k: v.clone() for k, v in HA.Net.from_actor_sd(HA.load_actor_sd(ROOT / HA.G_PT)).state_dict().items()}

def legs_frozen(a, ref):
    return all(torch.equal(v, ref[k]) for k, v in a.legs.state_dict().items()) and not any(p.requires_grad for p in a.legs.parameters())

# ---- one stage
def run_stage(env, ppo, stage, iters, it0=0, log=print, extra=None):
    """iters PPO iterations of `stage` on env -> per-iteration rows. extra(row) adds stage checks."""
    a = ppo.actor; env.stage = "B" if stage == "B" else "A"; rows = []; ref = g_ref() if stage != "B" else None
    torch.manual_seed(1000 + it0)
    for it in range(iters):
        t0 = time.time(); ppo.actor.train(); ppo.critic.train(); R = {k: 0.0 for k in TERMS}; rsum = 0.0; ndone = 0; legdev = 0.0; nan = 0
        met = dict(kd_edges=0, kd_charged=0, tangle=0.0, lean_deg_s=0.0, W=0.0, down_s=0.0, head_clean=0.0, head_landed=0.0)
        env.it_b = it if stage == "B" else 0
        with torch.no_grad():   # rsl_rl rolls out under inference_mode; no_grad is the same arithmetic and leaves the env's tensors usable outside
            for k in range(SPE):
                oa, oc = env.obs, env.critic_obs(); act = ppo.act(oa, oc)
                if stage != "B": legdev = max(legdev, float((act[:, :14] - a._mu[:, :14]).abs().max()))
                env.residual = a.r.detach().clone() if a.stage == "B" else None   # the residual of the forward that produced the action (fullbody_task res_l2)
                o2, r, done, tout, info = env.step(act.numpy()); oc2 = env.critic_obs(); ppo.process(o2, oc2, r, done, tout)
                for kk, x in info["parts"].items(): R[kk] += float(x.sum())
                rsum += float(r.sum()); ndone += int(done.sum()); nan += info["nan"] + int((~torch.isfinite(r)).sum()) + int((~torch.isfinite(o2)).sum())
                v = info["values"]; met["kd_edges"] += int(info["edge"].sum()); met["kd_charged"] += int(v["fb_knockdown"].sum()); met["tangle"] += float(env.B.acct.step["tangle"].sum())
                met["W"] += float(env.B.acct.step["work"].sum()); met["down_s"] += DT * float(env.B.od.now.sum())
                met["lean_deg_s"] += DT * float(info["lean"].clamp(min=0).sum()); met["head_clean"] += float(env.B.acct.step["head_clean"].sum()); met["head_landed"] += float(env.B.acct.step["head_landed"].sum())
            ppo.returns(env.critic_obs())
        L = ppo.update(); nan += sum(int(not math.isfinite(x)) for x in (L["value"], L["surrogate"], L["entropy"]))
        nan += sum(int(not torch.isfinite(p).all()) for p in chain(a.parameters(), ppo.critic.parameters()))
        row = dict(stage=stage, it=it0 + it + 1, reward=rsum / (env.n * SPE), dones=ndone, terms=R, loss=L, leg_dev=legdev, nan=nan, wall_s=round(time.time() - t0, 2),
                   arm_std=float(a.std()[14:].mean()), leg_std=float(a.std()[:14].mean()), metrics=met)
        if stage != "B": row["legs_frozen"] = legs_frozen(a, ref)
        if extra: extra(row)
        rows.append(row); log(json.dumps(dict(stage=stage, it=row["it"], r=round(row["reward"], 4), v=round(L["value"], 4), kl=L["kl"], lr=L["lr"], dones=ndone, nan=nan, wall=row["wall_s"])), flush=True)
    return rows

def scripted_round(seeds, actor, scripts, stage="B"):
    """One episode per scripted env (no auto-reset), SCRIPT_S long, the actor's deterministic action -> (env, parts sums, nan count)."""
    env = SmokeEnv(len(scripts), seeds, auto_reset=False, scripts=scripts); env.stage = stage; R = {k: 0.0 for k in TERMS}; nan = 0
    with torch.no_grad():
        for _ in range(int(round(SCRIPT_S / DT))):
            a = actor({"actor": env.obs, "critic": env.critic_obs()}); env.residual = actor.r.detach().clone() if actor.stage == "B" else None
            o, r, done, tout, info = env.step(a.numpy())
            for k, x in info["parts"].items(): R[k] += float(x.sum())
            nan += info["nan"] + int((~torch.isfinite(r)).sum())
            if env.over.all(): break
    env.finish(); return env, R, nan

QKEY = dict(spar_head_clean="hm_head_clean", spar_head_impulse="head_imp", spar_torso_impulse="torso_imp", spar_block="paid", spar_clearance="clear", spar_ring="ring",
            spar_alive="alive_s", fb_knockdown="kd_expected", fb_down_neutral="down_s", fb_down_approach="approach", fb_down_contact="down_contact", fb_leg_tangle="tangle",
            fb_lean_away="lean", res_l2="res")

def magnitudes(groups):
    """Per TABLE term: its reward R over the env-steps where its stage weights it, its driving quantity Q (QKEY, measured independently) over
    the same env-steps, and R / Q against the 2.4 rate: fires (R != 0 with the rate's sign) and within MAG_X. groups: [(stage letter, {term:
    R}, q)]."""
    out = {}
    for k, (rate, unit, st) in TABLE.items():
        R = sum(r.get(k, 0.0) for s, r, q in groups if s in st); Q = sum(q[QKEY[k]] for s, r, q in groups if s in st); ratio = R / Q if Q > 0 else None
        fires = R != 0.0 and math.copysign(1, R) == math.copysign(1, rate); within = ratio is not None and 1 / MAG_X <= ratio / rate <= MAG_X
        out[k] = dict(R=R, Q=float(Q), unit=unit, rate=rate, ratio=ratio, x=None if ratio is None else ratio / rate, fires=bool(fires), within_2x=bool(within), ok=bool(fires and within))
    return out

def blind_checks(cks, tmp):
    """A9.2: every checkpoint states blind = False; a blind twin's checkpoint states True; each twin refuses the other's checkpoint, and
    export_hybrid_fb.build refuses the other --blind."""
    import export_hybrid_fb as EX
    sd = lambda c: torch.load(c, map_location="cpu", weights_only=False)["actor_state_dict"]
    out = dict(flags={Path(c).name: HA.stored_blind(sd(c)) for c in cks})
    tw = make_actor("A", blind=True); bp = Path(tmp) / "ckpt_blind_twin.pt"; torch.save(dict(actor_state_dict=tw.state_dict(), iter=0), bp); out["blind_twin"] = HA.stored_blind(sd(bp))
    out["ckpt_blind"] = {Path(c).name: EX.ckpt_blind(c) for c in list(cks) + [bp]}
    def refused(f):
        try: f(); return False
        except AssertionError: return True
    out["refused"] = dict(sighted_into_blind=refused(lambda: make_actor("A", blind=True).load_state_dict(sd(cks[1]))), blind_into_sighted=refused(lambda: make_actor("A").load_state_dict(sd(bp))),
                          build_blind_on_sighted=refused(lambda: EX.build(blind=True, ckpt=cks[1])), build_sighted_on_blind=refused(lambda: EX.build(blind=False, ckpt=bp)))
    out["ok"] = bool(all(v is False for v in out["flags"].values()) and out["blind_twin"] is True and all(out["refused"].values()) and out["ckpt_blind"][bp.name] is True
                     and not any(v for k, v in out["ckpt_blind"].items() if k != bp.name))
    return out

def kd_class(e):
    """A knockdown edge's class under the final rule (A5.1): charged | start_window | w_floor | land_gate | other."""
    if e["charged"]: return "charged"
    if e["t"] <= RW.START_S + 1e-9 and e["W"] > RW.W_EPS: return "start_window"
    if e["W"] <= RW.W_EPS: return "w_floor"
    if e["t"] - e["land_t"] <= RW.LAND_GATE_S + 1e-9: return "land_gate"
    return "other"

def plumbing(out, log=print):
    """The blocking plumbing smoke (module docstring) -> result dict with checks; writes out/plumbing.json and the checkpoints."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(PLUMB["block"][0]); np.random.seed(PLUMB["block"][0]); t00 = time.time()
    n, b0 = PLUMB["n"], PLUMB["block"][0]; seeds = Seeds(PLUMB["role"], *PLUMB["train"], b0); rows, groups, sw, eps = [], [], 0, []
    def tally(env):
        nonlocal sw
        sw += int(env.B.obs.k_switches) + int(env.B.act.switches) + env.log["switch_ep"]; eps.extend(env.log["episodes"])
    def stage(name, ckpt, iters, it0):
        env = SmokeEnv(n, seeds); nc = env.critic_obs().shape[1]; ppo = restart(name, ckpt, n, nc); r = run_stage(env, ppo, name, iters, it0, log); env.finish(); tally(env)
        groups.append(("B" if name == "B" else "A", {k: sum(x["terms"][k] for x in r) for k in TERMS}, env.q)); rows.extend(r)
        return ppo, ppo.save(out / f"ckpt_{name}.pt", it0 + iters, dict(stage=name, blind=False)), nc
    _, ck0, nc = stage("A0", None, PLUMB["A0"], 0)
    _, ckA, _ = stage("A", ck0, PLUMB["A"], PLUMB["A0"])
    # B start: B (the restart through load_into_b) acts as A on fresh rows, its residual exactly zero
    env = SmokeEnv(n, seeds); oa, oc = env.obs, env.critic_obs(); aA = restart("A", ckA, n, nc).actor.eval(); aB = restart("B", ckA, n, nc).actor.eval()
    with torch.no_grad(): mA, mB = aA({"actor": oa, "critic": oc}), aB({"actor": oa, "critic": oc}); r0 = aB.r.clone()
    bstart = dict(residual_max_abs=float(r0.abs().max()), residual_zero=bool((r0 == 0).all()), action_equal_A=bool(torch.equal(mA, mB)), leg_std=float(aB.std()[:14].mean()),
                  residual_last_layer_zero=bool((aB.residual[-1].weight == 0).all() and (aB.residual[-1].bias == 0).all())); tally(env); del env
    pB, ckB, _ = stage("B", ckA, PLUMB["B"], PLUMB["A0"] + PLUMB["A"]); bstart["res_l2_iter1"] = rows[PLUMB["A0"] + PLUMB["A"]]["terms"]["res_l2"]
    # scripted knockdowns (stage-B weights, the trained B actor's deterministic action)
    ss = Seeds(PLUMB["role"], *PLUMB["scripted"], b0); kd, nan_s = [], 0; aB = pB.actor.eval()
    for sc in SCRIPTS:
        env, R, nn_ = scripted_round(ss, aB, sc); nan_s += nn_; kd += env.log["kd"]; groups.append(("B", R, env.q)); tally(env)
    mag = magnitudes(groups); A0 = [r["loss"]["value"] for r in rows if r["stage"] == "A0"]
    vl = dict(first3=float(np.mean(A0[:3])), last3=float(np.mean(A0[-3:]))); vl["drop"] = 1 - vl["last3"] / vl["first3"]; vl["ok"] = bool(vl["drop"] >= VLOSS_DROP)
    for e in kd: e["class"] = kd_class(e)
    kdc = {c: sum(e["class"] == c for e in kd) for c in ("charged", "start_window", "w_floor", "land_gate", "other")}
    kd_r = sum(g[1]["fb_knockdown"] for g in groups); n_ch = sum(e["charged"] for e in kd) + sum(r["metrics"]["kd_charged"] for r in rows)
    knock = dict(by_class=kdc, rule_equal=all(e["charged"] == e["expected"] for e in kd), injected_lo_blocked=sum(e["injected"] and e["W"] <= RW.W_EPS and not e["charged"] for e in kd),
                 injected_hi_charged=sum(e["injected"] and e["W"] > RW.W_EPS and e["charged"] for e in kd), charges=n_ch, reward_per_charge=kd_r / n_ch if n_ch else None)
    knock["ok"] = bool(knock["rule_equal"] and kdc["charged"] >= 1 and kdc["start_window"] >= 1 and kdc["w_floor"] >= 1 and kdc["other"] == 0 and knock["injected_lo_blocked"] >= 1
                       and knock["injected_hi_charged"] >= 1 and knock["reward_per_charge"] is not None and abs(knock["reward_per_charge"] - TABLE["fb_knockdown"][0]) < 1e-6)
    M = [r["metrics"] for r in rows]; tot = {k: sum(m[k] for m in M) for k in M[0]}; sq = [g[2] for g in groups[3:]]
    tot.update(kd_edges=tot["kd_edges"] + len(kd), kd_charged=tot["kd_charged"] + sum(e["charged"] for e in kd), W=tot["W"] + sum(q["work"] for q in sq), tangle=tot["tangle"] + sum(q["tangle"] for q in sq),
               lean_deg_s=tot["lean_deg_s"] + sum(q["lean"] for q in sq), down_s=tot["down_s"] + sum(q["down_s"] for q in sq))
    metrics_ok = bool(all(set(m) >= {"kd_edges", "kd_charged", "tangle", "lean_deg_s", "W", "down_s"} and all(math.isfinite(float(v)) for v in m.values()) for m in M)
                      and all(tot[k] > 0 for k in ("kd_edges", "kd_charged", "tangle", "lean_deg_s", "W", "down_s")))
    legs = dict(frozen_every_iter=all(r["legs_frozen"] for r in rows if r["stage"] in ("A0", "A")), leg_dev_max=max(r["leg_dev"] for r in rows if r["stage"] in ("A0", "A")))
    legs["ok"] = bool(legs["frozen_every_iter"] and legs["leg_dev_max"] == 0.0)
    blind = blind_checks([ck0, ckA, ckB], out)
    head = dict(acct_head_clean=tot["head_clean"], acct_head_landed=tot["head_landed"], hm_head_clean=sum(g[2]["hm_head_clean"] for g in groups)); head["ok"] = bool(head["acct_head_clean"] > 0 and head["acct_head_landed"] > 0)
    nan = sum(r["nan"] for r in rows) + nan_s
    checks = dict(no_nan=nan == 0, legs_frozen_A0_A=legs["ok"], residual_zero_at_B=bool(bstart["residual_zero"] and bstart["action_equal_A"] and bstart["residual_last_layer_zero"] and bstart["res_l2_iter1"] == 0.0),
                  value_loss_drop_A0=vl["ok"], reward_terms=all(v["ok"] for v in mag.values()), knockdown_rule=knock["ok"], zero_switches=sw == 0, metrics_populated=metrics_ok,
                  blind_flag=blind["ok"], head_hits_in_account=head["ok"])
    sh = lambda key, vals: {str(v): round(sum(e[key] == v for e in eps) / max(len(eps), 1), 3) for v in vals}
    res = dict(plan="M2V2_PLAN_v3 3 M10 plumbing (blocking)", conf=SMOKE_CONF, ppo=dict(PPO_CFG, **FT.RL["A"]), n_envs=n, steps_per_env=SPE, iters=dict(A0=PLUMB["A0"], A=PLUMB["A"], B=PLUMB["B"]),
               seeds=dict(train=[seeds.used[0], seeds.used[-1], len(seeds.used)], scripted=[ss.used[0], ss.used[-1], len(ss.used)]), checks=checks, ok=all(checks.values()),
               value_loss=vl, A0_value_loss=A0, legs=legs, b_start=bstart, magnitudes=mag, knockdowns=knock, kd_edges=kd, metrics_total=tot, switches=sw, episodes=len(eps),
               shares=dict(k=sh("k", (0, 1, 2)), act_delay=sh("act_delay", (False, True))), blind=blind, head=head, nan=nan, rows=rows, wall_s=round(time.time() - t00, 1))
    (out / "plumbing.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8"); return res

def canary(out, a0, iters=CANARY["A"], log=print):
    """The learning canary (report only, m19): 16 envs, `iters` stage-A iterations from the plumbing's A0 checkpoint; v2's two
    indicators reported (return over the last 50 iterations vs the first 50; the account's clean head hits per live minute not rising)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(CANARY["block"][0]); np.random.seed(CANARY["block"][0]); t00 = time.time()
    seeds = Seeds(CANARY["role"], *CANARY["block"], CANARY["block"][0]); env = SmokeEnv(CANARY["n"], seeds); nc = env.critic_obs().shape[1]; ppo = restart("A", a0, CANARY["n"], nc)
    rows = []
    with open(out / "canary_rows.jsonl", "w", encoding="utf-8") as fh:
        keep = lambda row: (fh.write(json.dumps(row, default=float) + "\n"), fh.flush())
        for j in range(0, iters, 50):
            r = run_stage(env, ppo, "A", min(50, iters - j), j, log, extra=keep); rows += r; ppo.save(out / "ckpt_canary_A.pt", j + len(r), dict(stage="A", canary=True))
    env.finish()
    live_min = lambda R: sum(x["terms"]["spar_alive"] for x in R) / 4.0 / 60.0; hc = lambda R: sum(x["metrics"]["head_clean"] for x in R) / max(live_min(R), 1e-9)
    a, b = rows[:50], rows[-50:]
    rep = dict(plan="M2V2_PLAN_v3 3 M10 canary (report only, m19)", n_envs=CANARY["n"], iters=len(rows), seeds=[seeds.used[0], seeds.used[-1], len(seeds.used)], from_ckpt=str(a0),
               return_first50=float(np.mean([x["reward"] for x in a])), return_last50=float(np.mean([x["reward"] for x in b])),
               head_clean_per_live_min_first50=hc(a), head_clean_per_live_min_last50=hc(b), dones_first50=sum(x["dones"] for x in a), dones_last50=sum(x["dones"] for x in b),
               arm_std=[rows[0]["arm_std"], rows[-1]["arm_std"]], value_loss=[rows[0]["loss"]["value"], rows[-1]["loss"]["value"]], kl_mean=float(np.mean([x["loss"]["kl"] for x in rows])),
               lr_last=rows[-1]["loss"]["lr"], nan=sum(x["nan"] for x in rows), legs_frozen=all(x["legs_frozen"] for x in rows), episodes=len(env.log["episodes"]), wall_s=round(time.time() - t00, 1))
    rep["return_up"] = rep["return_last50"] >= rep["return_first50"]; rep["head_clean_not_rising"] = rep["head_clean_per_live_min_last50"] <= rep["head_clean_per_live_min_first50"]
    (out / "canary.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8"); return rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plumbing", "canary"]); ap.add_argument("--out", required=True); ap.add_argument("--a0", default=None)
    ap.add_argument("--iters", type=int, default=CANARY["A"]); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.cmd == "plumbing":
        r = plumbing(a.out); print(json.dumps(dict(ok=r["ok"], checks=r["checks"], value_loss=r["value_loss"], wall_s=r["wall_s"])), flush=True); sys.exit(0 if r["ok"] else 1)
    r = canary(a.out, a.a0 or str(Path(a.out) / "ckpt_A0.pt"), a.iters); print(json.dumps(r, default=float), flush=True)
