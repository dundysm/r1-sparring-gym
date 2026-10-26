"""Hit-robust balance env for the Unitree R1 (CPU MuJoCo; same physics/obs/action as the GPU recipe in docs/GPU.md).

obs (78): base ang vel (3) | projected gravity (3) | joint pos - home (24) | joint vel * 0.05 (24) | last action (24)
          -> delayed by 0-2 control steps (0-40 ms) and noised, per episode.
act (24): residual joint targets, target = HOME + 0.25 * a   (same convention as unitree_rl_mjlab)."""
from __future__ import annotations
import numpy as np, mujoco
from concurrent.futures import ThreadPoolExecutor
from .model import build_spec, R1Robot, load_cfg, HOME
from .perturb import HitApplier, random_schedule
from .sensors import DelayedNoisyObs

OBS_DIM, ACT_DIM = 78, 24

class R1HitEnv:
    def __init__(self, seed=0, cfg=None, episode_s=10.0, max_impulse=20.0, model=None, train=True):
        self.cfg = cfg or load_cfg(); self.m = model or build_spec(cfg=self.cfg).compile(); self.d = mujoco.MjData(self.m)
        self.r = R1Robot(self.m, self.cfg); self.hits_ap = HitApplier(self.m); self.rng = np.random.default_rng(seed)
        self.nsub = int(round(self.cfg["model"]["control_dt"] / self.cfg["model"]["timestep"])); self.dt = self.cfg["model"]["control_dt"]
        self.T = int(episode_s / self.dt); self.max_impulse = max_impulse; self.train = train
        n = self.cfg["sensors"]["noise"]
        self.noise = np.concatenate([np.full(3, n["ang_vel"]), np.full(3, n["gravity"]), np.full(24, n["joint_pos"]), np.full(24, n["joint_vel"] * 0.05), np.zeros(24)])
        self.hits = []; self.schedule_fn = None; self.opp = None; self.contact_log = None   # set by demo/eval

    def _raw_obs(self):
        d, r = self.d, self.r; R = d.xmat[r.pel].reshape(3, 3)
        return np.concatenate([d.qvel[r.free_v + 3:r.free_v + 6], R.T @ np.array([0, 0, -1.0]),
                               d.qpos[r.qadr] - HOME, 0.05 * d.qvel[r.vadr], self.last_a]).astype(np.float32)

    def reset(self, hits=None, delay=None):
        self.r.reset(self.m, self.d)
        if self.train:   # small initial pose noise
            self.d.qpos[self.r.qadr] += self.rng.normal(0, 0.02, 24); mujoco.mj_forward(self.m, self.d)
        self.t = 0; self.last_a = np.zeros(24, np.float32); self.filt = np.zeros(24)
        lo, hi = self.cfg["sensors"]["delay_steps"]; k = self.rng.integers(lo, hi + 1) if delay is None else delay
        self.sensor = DelayedNoisyObs(k, self.noise if self.train or delay is None else self.noise, self.rng)
        self.hits = hits if hits is not None else random_schedule(self.rng, self.T * self.dt, self.max_impulse)
        o = self._raw_obs(); self.sensor.reset(o); return self.sensor(o)

    def step(self, a):
        a = np.clip(a, -4, 4).astype(np.float32); d = self.d
        self.filt = self.cfg.get("action_filter", 0.3) * self.filt + (1 - self.cfg.get("action_filter", 0.3)) * a   # EMA low-pass
        self.r.pd(d, HOME + self.cfg["action_scale"] * self.filt)
        for _ in range(self.nsub):
            if self.opp is not None: self.opp.apply(self.m, d, d.time, self.m.opt.timestep)
            self.hits_ap.apply(d, self.hits, d.time); mujoco.mj_step(self.m, d)
        if self.contact_log is not None:
            from .contacts import per_body_force
            f = per_body_force(self.m, d, self.r.r1_bodies)
            if f: self.contact_log.append({"t": round(d.time, 3), **{k: round(v, 1) for k, v in f.items()}})
        d.xfrc_applied[:] = 0; self.t += 1
        fallen = self.r.fallen(d)
        R = d.xmat[self.r.pel].reshape(3, 3); up = R[2, 2]; z = d.xpos[self.r.pel][2]
        dq = d.qpos[self.r.qadr] - HOME
        rew = (1.0 + 0.5 * up - 2.0 * abs(z - 0.72) - 0.02 * np.sum(dq ** 2) - 0.01 * np.sum((a - self.last_a) ** 2)
               - 0.001 * np.sum(np.square(d.qvel[self.r.vadr])) * 0.05 - 0.05 * np.sum(np.square(d.qvel[self.r.free_v + 3:self.r.free_v + 5])))
        if fallen: rew -= 50.0
        self.last_a = a
        done = fallen or self.t >= self.T
        return self.sensor(self._raw_obs()), float(rew), bool(done), {"fallen": fallen, "timeout": self.t >= self.T and not fallen}

class BatchEnv:
    """N independent envs stepped by a thread pool (MuJoCo releases the GIL inside mj_step)."""
    def __init__(self, n, seed=0, threads=8, **kw):
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
