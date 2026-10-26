"""Observation latency + noise (idea from mjlab#1230: train with the delay the real robot has)."""
from collections import deque
import numpy as np

class DelayedNoisyObs:
    def __init__(self, delay_steps: int, noise_std: np.ndarray, rng):
        self.k = int(delay_steps); self.std = noise_std; self.rng = rng; self.buf = deque(maxlen=self.k + 1)
    def reset(self, obs):
        self.buf.clear(); [self.buf.append(obs) for _ in range(self.k + 1)]
    def __call__(self, obs):
        self.buf.append(obs); o = self.buf[0]
        return o + self.rng.normal(0, 1, o.shape) * self.std
