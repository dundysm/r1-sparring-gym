"""Hit-robust balance env for the Unitree R1 (CPU MuJoCo; same physics/obs/action as the GPU recipe in docs/GPU.md).

obs (78): base ang vel (3) | projected gravity (3) | joint pos - home (24) | joint vel * 0.05 (24) | last action (24)
          -> delayed by 0-2 control steps (0-40 ms) and noised, per episode.
act (24): residual joint targets, target = HOME + 0.25 * a   (same convention as unitree_rl_mjlab)."""
from __future__ import annotations
import numpy as np, mujoco
from concurrent.futures import ThreadPoolExecutor
from .model import build_spec, R1Robot, load_cfg, HOME, action_scale
from .perturb import HitApplier, random_schedule
from .sensors import DelayedNoisyObs

OBS_DIM, ACT_DIM = 78, 24      # residual contract; the mjlab contract (configs/r1_mjlab.yaml) has 83-d obs
MJLAB_OBS_DIM = 83
# Clip splits. Train on TRAIN_CLIPS; pick checkpoints / candidates on VAL_CLIPS only; report on TEST_CLIPS
# (kfab_202 is also the demo clip). Before this split, the v2 candidate was chosen partly on kfab_202 demo variants.
TRAIN_CLIPS = ["kfab_200", "kfab_201", "kfab_203"]   # kfab_199 dropped: the opponent's tracker can't perform it (tracker_cuts.json)
VAL_CLIPS = ["kfab_204"]
TEST_CLIPS = ["kfab_202", "kfab_205"]

_DR_NOMINAL = {}

class R1HitEnv:
    def __init__(self, seed=0, cfg=None, episode_s=10.0, max_impulse=20.0, model=None, train=True, opp_prob=0.0, opp_clips=TRAIN_CLIPS, dr=False, reward="v2", gap=(1.0, 2.5)):
        self.cfg = cfg or load_cfg(); self.m = model or build_spec(cfg=self.cfg, opponent=opp_prob > 0).compile()
        self.dr = dr; self.reward = reward; self.gap = gap; self.act_delay = 0; self.a_prev = np.zeros(24, np.float32)
        self.opp_prob = opp_prob; self.opp_clips = opp_clips; self._opp_cache = {}; self.d = mujoco.MjData(self.m)
        self.r = R1Robot(self.m, self.cfg); self.hits_ap = HitApplier(self.m); self.rng = np.random.default_rng(seed)
        if dr: self._dr_init()
        self.nsub = int(round(self.cfg["model"]["control_dt"] / self.cfg["model"]["timestep"])); self.dt = self.cfg["model"]["control_dt"]
        self.T = int(episode_s / self.dt); self.max_impulse = max_impulse; self.train = train
        n = self.cfg["sensors"]["noise"]; self.contract = self.cfg.get("contract", "residual")
        self.home = np.array(self.cfg.get("home", HOME), float)   # default pose: HOME, or the policy's own (e.g. a guard)
        self.scale = action_scale(self.cfg); self.clip = float(self.cfg.get("action_clip", 4.0)); self.alpha = self.cfg.get("action_filter", 0.3)
        if self.contract == "mjlab":   # uniform half-widths, as unitree_rl_mjlab trains; command and phase are noise-free
            self.noise = np.concatenate([np.full(3, n["ang_vel"]), np.full(3, n["gravity"]), np.zeros(5), np.full(24, n["joint_pos"]), np.full(24, n["joint_vel"]), np.zeros(24)])
        else:
            self.noise = np.concatenate([np.full(3, n["ang_vel"]), np.full(3, n["gravity"]), np.full(24, n["joint_pos"]), np.full(24, n["joint_vel"] * 0.05), np.zeros(24)])
        self.hits = []; self.schedule_fn = None; self.opp = None; self.contact_log = None   # set by demo/eval
        self.command = np.zeros(3)   # mjlab contract only: velocity command (vx, vy, wz); zero = stand / spar in place

    def _raw_obs(self):
        # orientation from qpos, not d.xmat: after mj_step, xmat is from before the last integration (one 5 ms substep
        # stale), while a real IMU (and r1gym/deploy.py) reports the current orientation
        d, r = self.d, self.r; R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[r.free_q + 3:r.free_q + 7]); R = R.reshape(3, 3)
        if self.contract == "mjlab":   # base_ang_vel, projected_gravity, command (vx, vy, wz), gait phase, q - q0, dq, last action
            c = self.command; ph = 2 * np.pi * ((self.t * self.dt) % 0.6) / 0.6   # mjlab phase(period=0.6), zero when standing
            phase = np.zeros(2) if np.linalg.norm(c) < 0.1 else np.array([np.sin(ph), np.cos(ph)])
            return np.concatenate([d.qvel[r.free_v + 3:r.free_v + 6], R.T @ np.array([0, 0, -1.0]), c, phase,
                                   d.qpos[r.qadr] - self.home, d.qvel[r.vadr], self.last_a]).astype(np.float32)
        return np.concatenate([d.qvel[r.free_v + 3:r.free_v + 6], R.T @ np.array([0, 0, -1.0]),
                               d.qpos[r.qadr] - self.home, 0.05 * d.qvel[r.vadr], self.last_a]).astype(np.float32)

    def reset(self, hits=None, delay=None):
        self.r.reset(self.m, self.d)
        if self.train:   # small initial pose noise
            self.d.qpos[self.r.qadr] += self.rng.normal(0, 0.02, 24); mujoco.mj_forward(self.m, self.d)
        self.t = 0; self.last_a = np.zeros(24, np.float32); self.filt = np.zeros(24)
        lo, hi = self.cfg["sensors"]["delay_steps"]; k = self.rng.integers(lo, hi + 1) if delay is None else delay
        self.sensor = DelayedNoisyObs(k, self.noise, self.rng, uniform=self.contract == "mjlab")
        self.hits = hits if hits is not None else random_schedule(self.rng, self.T * self.dt, self.max_impulse, gap=self.gap)
        if self.dr: self._dr_sample()
        self.a_prev = np.zeros(24, np.float32)
        if self.opp_prob > 0:   # training: a fight-clip opponent in range with prob opp_prob, else parked 5 m away
            from .opponent import FightReplay
            close = self.rng.random() < self.opp_prob; clip = self.opp_clips[self.rng.integers(len(self.opp_clips))]
            dist = round(float(self.rng.uniform(0.5, 0.7)), 2) if close else 5.0   # in-place opponent: strikes reach at 0.5-0.7 m
            key = (clip, dist)
            if key not in self._opp_cache: self._opp_cache[key] = FightReplay(self.m, clip, distance=dist, cfg=self.cfg)
            self.opp = self._opp_cache[key]; self.opp_t0 = float(self.rng.uniform(0, 5))
        if self.opp is not None:
            self.opp.place(self.m, self.d, getattr(self, "opp_t0", 0.0)); mujoco.mj_forward(self.m, self.d)
            self.opp_state = {"last": np.zeros(24, np.float32)}   # tracker opponent: its last action
        o = self._raw_obs(); self.sensor.reset(o); return self.sensor(o)

    def step(self, a):
        a = np.clip(a, -self.clip, self.clip).astype(np.float32); d = self.d
        if self.dr: self._dr_apply()
        if self.act_delay: a, self.a_prev = self.a_prev, a          # DR: one control step (20 ms) of motor-command latency
        self.filt = self.alpha * self.filt + (1 - self.alpha) * a   # EMA low-pass (mjlab contract: alpha 0, no filter)
        self.r.pd(d, self.home + self.scale * self.filt)
        if self.opp is not None and self.opp.tracker is not None:   # tracking-policy opponent acts at the control rate
            # as in mjlab training: kinematics current (sim.forward before obs), and the reference shown is the frame to reach
            # by the end of this control step (MotionCommand increments time_steps before the obs is computed)
            mujoco.mj_kinematics(self.m, d)
            self.opp.tracker.act(self.m, d, self.opp, self.opp_state, d.time + getattr(self, "opp_t0", 0.0) + self.dt)
        for _ in range(self.nsub):
            self.hits_ap.apply(d, self.hits, d.time)   # clears xfrc_applied, then writes the scripted hits
            if self.opp is not None: self.opp.apply(self.m, d, d.time + getattr(self, "opp_t0", 0.0), self.m.opt.timestep)
            mujoco.mj_step(self.m, d)
        if self.contact_log is not None:
            from .contacts import per_body_force
            f = per_body_force(self.m, d, self.r.r1_bodies)
            if f: self.contact_log.append({"t": round(d.time, 3), **{k: round(v, 1) for k, v in f.items()}})
        d.xfrc_applied[:] = 0; self.t += 1
        fallen = self.r.fallen(d)
        R = d.xmat[self.r.pel].reshape(3, 3); up = R[2, 2]; z = d.xpos[self.r.pel][2]
        dq = d.qpos[self.r.qadr] - self.home
        if self.reward == "v3":   # stay up / recover: no penalty for leaving the stance, mild pull back to home legs
            w = d.qvel[self.r.free_v + 3:self.r.free_v + 6]
            rew = (1.0 + 1.0 * up - 2.0 * max(0.0, 0.64 - z) - 0.003 * np.sum(dq[:12] ** 2) - 0.01 * np.sum(dq[12:] ** 2)
                   - 0.005 * np.sum((a - self.last_a) ** 2) - 0.0002 * np.sum(np.square(d.qvel[self.r.vadr])) - 0.02 * np.sum(w[:2] ** 2))
        else: rew = (1.0 + 0.5 * up - 2.0 * abs(z - 0.72) - 0.02 * np.sum(dq ** 2) - 0.01 * np.sum((a - self.last_a) ** 2)
               - 0.001 * np.sum(np.square(d.qvel[self.r.vadr])) * 0.05 - 0.05 * np.sum(np.square(d.qvel[self.r.free_v + 3:self.r.free_v + 5])))
        if fallen: rew -= 50.0
        self.last_a = a
        done = fallen or self.t >= self.T
        return self.sensor(self._raw_obs()), float(rew), bool(done), {"fallen": fallen, "timeout": self.t >= self.T and not fallen}

    # ---- domain randomisation (training only): mass, COM, foot friction, motor strength, armature, command latency
    def _dr_init(self):
        m = self.m   # nominal values cached per model object (later envs see a model already DR-modified by earlier ones)
        self._nom = _DR_NOMINAL.setdefault(id(m), dict(mass=m.body_mass.copy(), ipos=m.body_ipos.copy(), fric=m.geom_friction.copy(), gain=m.actuator_gainprm.copy(),
                         bias=m.actuator_biasprm.copy(), frc=m.actuator_forcerange.copy(), arm=m.dof_armature.copy()))
        self._r1b = np.array(sorted(self.r.r1_bodies))   # DR touches the protagonist only, never the opponent
        self._feet = np.array([g for g in range(m.ngeom) if "foot" in (m.geom(g).name or "") and m.geom_bodyid[g] in self.r.r1_bodies])
        self._r1dof = np.flatnonzero(np.isin(m.dof_bodyid, list(self.r.r1_bodies)))
    # Envs in one process share one MjModel (a private copy per env costs ~200 MB): each env keeps its own sampled
    # parameters and writes them into the shared model right before it steps. Safe because envs step sequentially
    # within a worker process; do NOT step DR envs that share a model from multiple threads.
    def _dr_apply(self):
        m, p = self.m, self._drp
        m.body_mass[:] = p["mass"]; m.body_ipos[:] = p["ipos"]; m.geom_friction[:] = p["fric"]; m.actuator_gainprm[:] = p["gain"]
        m.actuator_biasprm[:] = p["bias"]; m.actuator_forcerange[:] = p["frc"]; m.dof_armature[:] = p["arm"]
    def _dr_sample(self):
        self._dr_sample_into_model(); self._drp = {k: getattr(self.m, a).copy() for k, a in [("mass", "body_mass"), ("ipos", "body_ipos"),
            ("fric", "geom_friction"), ("gain", "actuator_gainprm"), ("bias", "actuator_biasprm"), ("frc", "actuator_forcerange"), ("arm", "dof_armature")]}
    def _dr_sample_into_model(self):
        m, n, g = self.m, self._nom, self.rng
        m.body_mass[:] = n["mass"]; m.body_mass[self._r1b] *= g.uniform(0.85, 1.15, len(self._r1b))
        m.body_mass[self.r.torso] += g.uniform(-0.5, 2.0)                       # payload / battery / padding
        m.body_ipos[:] = n["ipos"]; m.body_ipos[[self.r.pel, self.r.torso]] += g.uniform(-0.025, 0.025, (2, 3))
        m.geom_friction[:] = n["fric"]
        if len(self._feet): m.geom_friction[self._feet, 0] = g.uniform(0.5, 1.25)
        k = g.uniform(0.8, 1.2, 24); kd = g.uniform(0.8, 1.2, 24); a = self.r.act
        m.actuator_gainprm[:] = n["gain"]; m.actuator_biasprm[:] = n["bias"]; m.actuator_forcerange[:] = n["frc"]
        m.actuator_gainprm[a, 0] *= k; m.actuator_biasprm[a, 1] *= k; m.actuator_biasprm[a, 2] *= kd
        m.actuator_forcerange[a] *= g.uniform(0.8, 1.05, 24)[:, None]
        m.dof_armature[:] = n["arm"]; m.dof_armature[self._r1dof] *= g.uniform(0.5, 1.5, len(self._r1dof))
        self.act_delay = int(g.random() < 0.5)

class BatchEnv:
    """N independent envs stepped by a thread pool (MuJoCo releases the GIL inside mj_step)."""
    def __init__(self, n, seed=0, threads=8, **kw):
        if kw.get("dr") and threads > 1:   # DR envs write their parameters into the shared model before stepping
            raise ValueError("BatchEnv with dr=True needs threads=1; use ProcBatchEnv (--workers) for parallel DR training")
        first = R1HitEnv(seed=seed, **kw)                       # one compiled model shared by all envs
        self.envs = [first] + [R1HitEnv(seed=seed + i, model=first.m, **kw) for i in range(1, n)]; self.n = n
        self.pool = ThreadPoolExecutor(threads); self.chunks = np.array_split(np.arange(n), threads)
    def set_max_impulse(self, v):
        for e in self.envs: e.max_impulse = v
    def reset(self): return np.stack([e.reset() for e in self.envs])
    def step(self, A):
        out = [None] * self.n
        def run(ix):
            for i in ix:
                o, r, dn, info = self.envs[i].step(A[i])
                if dn: o = self.envs[i].reset()
                out[i] = (o, r, dn, info)
        list(self.pool.map(run, self.chunks))
        O, R, D, I = zip(*out); return np.stack(O), np.array(R, np.float32), np.array(D), I

def _worker(conn, n, seed, kw):
    import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
    first = R1HitEnv(seed=seed, **kw); envs = [first] + [R1HitEnv(seed=seed + i, model=first.m, **kw) for i in range(1, n)]
    while True:
        try: cmd, x = conn.recv()
        except EOFError: break
        if cmd == "reset": conn.send(np.stack([e.reset() for e in envs]))
        elif cmd == "imp":
            for e in envs: e.max_impulse = x
        elif cmd == "step":
            out = []
            for e, a in zip(envs, x):
                o, r, dn, info = e.step(a)
                if dn: o = e.reset()
                out.append((o, r, dn, info))
            conn.send(out)
        else: break

class ProcBatchEnv:
    """Same API as BatchEnv, one OS process per worker (scales with cores; use on many-vCPU machines)."""
    def __init__(self, n, seed=0, workers=8, **kw):
        import multiprocessing as mp
        ctx = mp.get_context("spawn"); self.n = n; self.split = np.array_split(np.arange(n), workers); self.conns = []; self.ps = []
        for w, ix in enumerate(self.split):
            a, b = ctx.Pipe(); p = ctx.Process(target=_worker, args=(b, len(ix), seed + int(ix[0]), kw), daemon=True); p.start()
            self.conns.append(a); self.ps.append(p)
    def set_max_impulse(self, v):
        for c in self.conns: c.send(("imp", v))
    def reset(self):
        for c in self.conns: c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])
    def step(self, A):
        for c, ix in zip(self.conns, self.split): c.send(("step", A[ix]))
        out = [x for c in self.conns for x in c.recv()]
        O, R, D, I = zip(*out); return np.stack(O), np.array(R, np.float32), np.array(D), I
    def close(self):
        for c in self.conns: c.send(("close", None))
