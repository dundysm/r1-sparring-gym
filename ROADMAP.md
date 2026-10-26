# Roadmap

Phases 1–2 (this repo, today): R1 sim on the official Unitree MJCF, an opponent replayed from KungFuAthleteBot motion, procedural strikes/pushes/grabs, sensor delay+noise, per-body contact logging, a hit-robust balance policy, eval vs a PD-stand baseline, ONNX export.

## Phase 3: a sparring controller (strike / block / clinch) at ≥50 Hz on Jetson Orin
- **Observation:** the R1 head stereo camera plus proprioception. No mocap and no privileged opponent state at deploy time.
- **Teacher:** a privileged PPO policy that sees the opponent's body poses and velocities and the contact forces from `contact_log`. Reward terms: land strikes on allowed target zones, block incoming strikes (forearm contact before torso contact), clinch (keep both hands on the opponent's shoulders/torso while staying upright), and the phase-2 balance terms.
- **Student:** DAgger / teacher-student distillation onto a small CNN (or a frozen pretrained encoder) over stereo frames rendered in sim with domain randomization (lighting, textures, opponent appearance, camera extrinsics ±2 cm / ±2°, 20–60 ms latency).
- **Opponents:** curriculum from scripted replays (KungFuAthleteBot clips) → self-play against frozen past policies → mixed.
- **Deploy budget:** ONNX → TensorRT FP16 on Orin. Target <10 ms end-to-end per 20 ms tick. Measure on hardware; the phase-2 MLP is ~8 µs on CPU, so the vision encoder is the whole budget.
- **Dexterous hands (R1 EDU):** start with hands as rigid fists/pads for strike and block; add hand closure only for clinch grips.

## Phase 4: safety and sim-to-real on the R1 EDU
- **Safety caps in the action layer:** per-joint torque/velocity limits below the vendor limits, end-effector speed caps when within X m of a human, a contact-force cap on strike links, and a watchdog (on a stale obs or missed tick, go to damping mode).
- **E-stop:** a hardware wireless e-stop plus a software stop. Both drop to a damped crouch, never a limp fall.
- **Padded mode:** mandatory foam on fists and forearms, reduced strike-speed caps, and a "touch, not hit" reward variant. Human sparring only in padded mode, and only after robot-vs-dummy and robot-vs-robot.
- **Regulatory note:** human-vs-robot combat events are not a free-for-all. California's athletic commission stopped a planned human-vs-robot fighting league, so any public human sparring needs a legal/commission check first.
- **Sim-to-real steps:**
  1. System-ID the real R1 EDU: joint friction, armature, motor torque–speed curves, PD gains, IMU latency. Replace every value marked ASSUMPTION in `configs/r1_assumptions.yaml`.
  2. Re-train with the measured values and wider randomization.
  3. Gantry/harness tests with a padded push stick and an instrumented impulse (force-sensor pendulum), and compare the real survival curve to the sim one.
  4. Free-standing push tests, then a dummy opponent, then phase-3 skills.
