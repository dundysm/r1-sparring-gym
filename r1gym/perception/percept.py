"""Percept v1 and the perception sources that feed env._face (phase 3, M1: the R1 turns toward what its own camera sees).

A PerceptStream rides in R1HitEnv.substep_hooks (read-only, like hits.HitMeter). After every physics substep it integrates
the gyro's yaw rate and polls the head camera's CaptureClock (camera.py); when an exposure falls due it takes a snapshot:
the full qpos (only the render twin reads it: it is the world the camera looks at), the sensed proprioception (joint
encoders and IMU gravity with the env's own sensor-noise model, drawn from the stream's own RNG so the physics stays
paired across sources) and the integrated gyro yaw. Sources (cfg perception):
- truth: no stream; env._face reads the true state exactly as before;
- truth_lag: the exact bearing and range at capture, on the camera's clock, latency and drops (the timing cost alone);
- camera: the snapshot rendered on the twin and run through a detector by a PerceptHub (hub.py, main thread only);
- shuffled: another episode's camera estimates at the same times since its start (a control: no information about this bout).
Transport is a LatencyQueue: delivered at t_capture + latency, in order, some frames dropped (never estimated).
Use: env._face calls bearing(): the latest delivered estimate, low-passed (percept_filter_s, in a gyro-stabilised frame)
and ego-compensated to now by the gyro yaw since capture, or None once nothing has been seen for percept_lost_s (the R1
then stands still). The estimate is already in the pelvis heading frame at capture (mount_fk applies the sensed waist), so
the pelvis gyro alone compensates it.
Firewall: no estimate is made from MjData. The hub sees a snapshot's qpos only through the renderer, and the detector sees
pixels and sensed proprioception; tests/test_perception_percept.py wraps MjData to check it. One exception, by design: the
yaw-rate integration projects the (noisy) gyro onto the true world vertical, i.e. it assumes an ideal roll/pitch estimate
(an IMU's attitude filter gets within about a degree; the error that leaves is that tilt times the small roll/pitch rates).
Projecting onto the sensed gravity instead would feed the policy-obs noise model (uniform +-0.05 per axis, a raw
accelerometer, not a filter) into the yaw every substep; the stored M1 results were run with the ideal projection."""
from __future__ import annotations
import bisect
from typing import NamedTuple
import numpy as np, mujoco
from .camera import HeadCameraSpec, CaptureClock, LatencyQueue

SCHEMA, SOURCES = 1, ("truth", "truth_lag", "camera", "shuffled")

class Percept(NamedTuple):
    """Percept v1: the opponent in the R1's pelvis heading frame (env._face's frame: x = the pelvis heading on the ground, no
    world yaw). bearing [rad] and range [m, horizontal, pelvis to pelvis] to the opponent's pelvis with 1-sigma; rel_heading =
    the opponent's yaw - the R1's - pi, wrapped (0: squarely facing each other; nan: not estimated). Times: sim seconds."""
    t_capture: float; t_avail: float; presence: bool; bearing: float; bearing_sd: float; range: float; range_sd: float; rel_heading: float; source: str

class Snapshot(NamedTuple):
    t: float; yaw: float; qpos: np.ndarray; q: np.ndarray; g: np.ndarray; slot: list   # slot = [gyro yaw at capture, estimate]: the hub fills it

def wrap(a):
    """Angle into [-pi, pi]; one already inside comes back unchanged (bit-identical), so zero compensation changes nothing."""
    return a if -np.pi <= a <= np.pi else float(np.angle(np.exp(1j * a)))

def _yaw(q): w, x, y, z = q; return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def _rot(q): R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)

NONE = dict(presence=False, bearing=np.nan, bearing_sd=np.nan, range=np.nan, range_sd=np.nan, rel_heading=np.nan)

class PerceptStream:
    def __init__(self, env, source, spec: HeadCameraSpec | None = None, seed=0, filter_s=None, lost_s=None, comp=True, donor=None):
        """env: the R1HitEnv it rides in (its R1Robot indices, opponent and cfg). seed: the perception RNG of this episode, the
        same for every source so clocks, latencies, drops and sensor noise are paired. comp=False: no ego compensation (an
        ablation). donor (shuffled): another episode's camera estimates, [(t_capture, estimate)] (camera_log())."""
        assert source in SOURCES[1:], source
        self.env, self.source, self.comp, self.donor, self.seed = env, source, comp, donor, seed
        self.spec = spec or HeadCameraSpec(gym_cfg=env.cfg); c = env.cfg; n = c["sensors"]["noise"]
        self.filter_s = float(c.get("percept_filter_s", 0.0) if filter_s is None else filter_s); self.lost_s = float(c.get("percept_lost_s", 1.0) if lost_s is None else lost_s)
        self.noise = (n["joint_pos"], n["gravity"], n["ang_vel"])   # uniform half-widths, as the policy's own obs (mjlab); pessimistic for an IMU
        self.dt_sub = c["model"]["timestep"]; self.phase = None; self.hub = None   # hub set: render inline (one env, main thread)
        if donor is not None: self.donor_t = [t for t, _ in donor]
        r = np.random.default_rng(0); self.clock, self.queue = CaptureClock(self.spec, r), LatencyQueue(self.spec, r)

    def reset(self):
        """Hook reset (env.reset, after the opponent is placed): a new episode on this stream's seed."""
        ss = np.random.SeedSequence([self.seed, 4]).spawn(3); self.rng = np.random.default_rng(ss[2])   # clock, transport, sensor noise: separate streams
        self.clock.reset(np.random.default_rng(ss[0])); self.queue.reset(np.random.default_rng(ss[1]))
        if self.phase is not None: self.clock.t0 = self.phase   # tests: frames exactly on the control ticks
        self.yaw = 0.0; self.now = self.env.d.time; self.pending, self.captures, self.delivered, self.used = [], [], [], []
        self.track = None; self.t_seen = -np.inf; self._poll(self.env.d)

    def substep(self, m, d):
        """Hook: after every mj_step. Gyro (body rates) dotted with the world vertical in the pelvis frame = yaw rate
        (the true vertical: an ideal attitude estimate, the one exception to the firewall; see the module docstring)."""
        r = self.env.r; w = d.qvel[r.free_v + 3:r.free_v + 6] + self.rng.uniform(-1, 1, 3) * self.noise[2]
        self.yaw += float(_rot(d.qpos[r.free_q + 3:r.free_q + 7])[2] @ w) * self.dt_sub; self.now = d.time; self._poll(d)

    def _poll(self, d):
        tc = self.clock.poll(d.time)
        if tc is None: return
        e, r = self.env, self.env.r; R = _rot(d.qpos[r.free_q + 3:r.free_q + 7])
        q = d.qpos[r.qadr] + self.rng.uniform(-1, 1, 24) * self.noise[0]; g = R.T @ np.array([0, 0, -1.0]) + self.rng.uniform(-1, 1, 3) * self.noise[1]
        b, v = e.true_bearing(); oq = d.qpos[e.opp.qadr + 3:e.opp.qadr + 7]
        truth = dict(presence=True, bearing=b, bearing_sd=0.0, range=float(np.hypot(v[0], v[1])), range_sd=0.0,
                     rel_heading=wrap(float(_yaw(oq) - _yaw(d.qpos[r.free_q + 3:r.free_q + 7]) - np.pi)))
        slot = [self.yaw, None]; ta = self.queue.push(tc, slot); self.captures.append((tc, ta, truth, slot))   # the transport decides first:
        if ta is None: return                                                                                   # a dropped frame is never estimated
        if self.source == "truth_lag": slot[1] = truth
        elif self.source == "shuffled": i = bisect.bisect_right(self.donor_t, tc + 1e-9) - 1; slot[1] = self.donor[i][1] if i >= 0 else NONE
        else: self.pending.append(Snapshot(tc, self.yaw, d.qpos.copy(), q, g, slot))

    def bearing(self):
        """For env._face: the opponent's bearing now [rad], or None (no estimate yet, or nothing seen for lost_s)."""
        if self.hub is not None: self.hub.process([self])
        for tc, ta, (yaw_c, est) in self.queue.pop(self.now + 1e-9):   # 1e-9: d.time accumulates rounding
            self.delivered.append((self.now, Percept(tc, ta, bool(est["presence"]), *(float(est[k]) for k in ("bearing", "bearing_sd", "range", "range_sd", "rel_heading")), self.source)))
            if not est["presence"]: continue
            if self.track is None or self.filter_s <= 0: self.track = [est["bearing"], yaw_c]
            else:   # low-pass in a gyro-stabilised frame: the old estimate is first carried to this frame's heading
                b0 = self.track[0] - (yaw_c - self.track[1]) if self.comp else self.track[0]
                self.track = [wrap(b0 + (1 - np.exp(-(tc - self.t_seen) / self.filter_s)) * wrap(est["bearing"] - b0)), yaw_c]
            self.t_seen = tc
        if self.track is None or self.now - self.t_seen > self.lost_s: self.used.append((self.now, None, self.now - self.t_seen)); return None
        b = wrap(self.track[0] - (self.yaw - self.track[1]) if self.comp else self.track[0]); self.used.append((self.now, b, self.now - self.t_seen)); return b

    def camera_log(self):
        """[(t_capture, estimate)] of the frames that were estimated (not dropped): a donor for another episode's shuffled source."""
        return [(tc, slot[1]) for tc, ta, _, slot in self.captures if ta is not None]

    def stats(self, t_from=0.2):
        """Per-bout percept quality against the truth at capture (frames delivered by the end), and at use."""
        cap = [(tc, ta, tr, s[1]) for tc, ta, tr, s in self.captures]; est = [(tr, x) for tc, ta, tr, x in cap if ta is not None and x is not None]
        seen = [(tr, x) for tr, x in est if x["presence"]]; eb = np.degrees(np.abs([wrap(x["bearing"] - tr["bearing"]) for tr, x in seen])) if seen else np.zeros(0)
        er = np.array([x["range"] - tr["range"] for tr, x in seen]) if seen else np.zeros(0); rr = np.array([tr["range"] for tr, _ in seen]) if seen else np.zeros(0)
        sd = np.degrees([x["bearing_sd"] for _, x in seen]) if seen else np.zeros(0); lat = [p.t_avail - p.t_capture for _, p in self.delivered]
        use = [(t, b, age) for t, b, age in self.used if t >= t_from]; q = lambda x, p: round(float(np.nanpercentile(x, p)), 3) if len(x) and np.isfinite(x).any() else None
        return dict(captures=len(cap), dropped=round(float(np.mean([ta is None for _, ta, _, _ in cap])), 4) if cap else None, estimated=len(est),
                    presence=round(len(seen) / len(est), 4) if est else None, bearing_err_deg_median=q(eb, 50), bearing_err_deg_p90=q(eb, 90),
                    range_err_m_median=q(np.abs(er), 50), range_err_rel_median=q(np.abs(er) / np.maximum(rr, 1e-6), 50), range_bias_m=q(er, 50),
                    sigma1_coverage=round(float(np.mean(eb <= sd)), 3) if len(sd) and np.isfinite(sd).all() and sd.any() else None,
                    latency_ms_median=q(np.array(lat) * 1e3, 50), latency_ms_max=round(float(max(lat)) * 1e3, 1) if lat else None,
                    no_track=round(float(np.mean([b is None for _, b, _ in use])), 4) if use else None, stale_0p5s=round(float(np.mean([age > 0.5 for _, _, age in use])), 4) if use else None)

def attach(env, source, hub=None, **kw):
    """Give env a PerceptStream for its cfg's perception source; hub: render inline (one env, in the main thread). A runner
    stepping many envs in threads leaves hub None and calls PerceptHub.process(streams) between control steps."""
    assert env.perception == source, f"env cfg perception is {env.perception!r}, not {source!r}"
    s = PerceptStream(env, source, **kw); s.hub = hub; env.percepts = s; env.substep_hooks.append(s); return s
