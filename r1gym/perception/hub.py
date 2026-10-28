"""PerceptHub: renders pending camera snapshots and runs the detector, in the main thread only (phase 3, M1).

A GL context belongs to the thread that made it, so env workers never render. Each camera PerceptStream queues a snapshot
when an exposure falls due; the hub draws it on the render twin (sync(qpos): kinematics only, physics untouched) after the
control step that captured it and fills in the estimate. That is never late: a frame captured during control step k is due
at t_capture + latency >= t_capture + control_dt, after the end of step k (camera.py asserts latency_min >= control_dt), so
env._face at the end of step k can't need it yet. Streams are processed in the order given and each one's frames in
capture order; rendering is deterministic on one machine (render_bench.json), so a run is too. Single-env use (demo,
tests) sets stream.hub and the stream calls process([self]) itself: the same frames, the same estimates."""
from __future__ import annotations
import time
from collections import deque

class PerceptHub:
    def __init__(self, twin, detector, keep=0):
        """twin: RenderTwin with the R1's head camera; detector(img, q, g) -> estimate dict; keep: last images kept per stream (demo)."""
        self.twin, self.det, self.keep = twin, detector, keep; self.frames = 0; self.render_s = self.detect_s = 0.0; self.images = {}

    def process(self, streams):
        """Render and estimate every pending snapshot of these streams; returns the number of frames."""
        n = 0
        for s in streams:
            for sn in s.pending:
                t0 = time.perf_counter(); self.twin.sync(sn.qpos); img = self.twin.frame("head"); t1 = time.perf_counter()
                sn.slot[1] = self.det(img, sn.q, sn.g); self.render_s += t1 - t0; self.detect_s += time.perf_counter() - t1; n += 1
                if self.keep: self.images.setdefault(id(s), deque(maxlen=self.keep)).append((sn.t, img, sn))
            s.pending = []
        self.frames += n; return n

    def cost(self):
        return dict(frames=self.frames, render_ms=round(1e3 * self.render_s / max(self.frames, 1), 2), detect_ms=round(1e3 * self.detect_s / max(self.frames, 1), 2))
