# Deploying the balance policy

`policy.onnx` contains only the MLP and the observation normaliser (input `obs` [N, 78] float32, output `action` [N, 24]).
Everything else the robot-side loop must reproduce is in `deploy_contract.json`, written next to the ONNX by
`scripts/export_onnx.py` from `r1gym/deploy.py`. `r1gym.deploy.DeployLoop` is a reference implementation:
`tests/test_gym.py::test_deploy_loop_matches_env` checks that it reproduces the training env's joint targets exactly,
step by step, from raw sensor readings.

## Loop at 50 Hz
1. Read the IMU (gyro in the pelvis frame, orientation quaternion) and the 24 joint positions and velocities, in
   `joint_order` (Unitree deploy order).
2. Build the observation in `obs_layout` order:
   - gyro
   - `R^T (0, 0, -1)`
   - `q - home`
   - `0.05 * dq`
   - the previous raw action
3. Run the ONNX graph to get `a`. Clip it to ±4.
4. Filter: `f = 0.3 f + 0.7 a`, with `f` starting at 0.
5. Set the targets: `q_target = home + 0.25 f`.
6. Store the clipped `a` as the next "previous action". It is the policy's own output, not the filtered one.
7. Joint PD: `tau = kp (q_target - q) - kd dq`, clipped to the torque cap. kp/kd and caps are per joint in the contract.

## Known gaps before hardware
- **Gains.** The policy is a residual on `stand_pd` (kp 300 on legs and ankles). That is 3–7.5× Unitree's deploy gains and
  has not been tried on a real R1. Stiff gains plus motor latency can chatter. Start in a harness, at reduced gains, and
  re-train if they have to change.
- **Latency.** Training randomised 0–40 ms observation delay. Measure the real loop and keep it inside that range.
- **Exploration-limited policy.** See `docs/MODEL_AUDIT.md`: this policy family can't step. The stepping base
  (below) uses Unitree's own contract.
- **Safety layer.** E-stop, torque/velocity caps below the vendor limits and a stale-observation watchdog are not
  part of the policy. See ROADMAP phase 4.

# Stepping base (unitree_rl_mjlab contract)
The stepping base policy (`gpu/r1_spar_task.py`, trained in mjlab) is exported by mjlab with its contract in the ONNX
metadata: `joint_names`, `joint_stiffness`, `joint_damping`, `default_joint_pos`, `action_scale`,
`observation_names`. `r1gym.policy.MjlabPolicy` asserts that these match `configs/r1_mjlab.yaml` before running.

- **Observation (83):**
  - base angular velocity
  - projected gravity
  - velocity command (vx, vy, wz); zero means stand
  - gait phase: sin/cos of t mod 0.6 s, zero when |command| < 0.1
  - `q - default_joint_pos`
  - `dq` (unscaled)
  - last action
- **Action:** `q_target = default_joint_pos + action_scale * a`, with no clip and no filter. `action_scale = 0.25 * effort_limit / stiffness` per joint:

  | joints | action_scale |
  |---|---|
  | legs, waist | 0.150 |
  | ankles | 0.312 |
  | shoulder pitch/roll | 0.375 |
  | shoulder yaw, elbow, wrist | 0.412 |
- **PD:** Unitree's deploy gains (kp 100 legs/waist, 40 ankles/shoulders, 20 arms), with torque caps 60/50/60/60/33 N·m.

**Check before deploying with Unitree's C++ stack.** `unitree_rl_mjlab`'s `deploy/robots/r1/config/policy/velocity/v0/params/deploy.yaml`
(@1425b15) lists `scale: 0.25` for every joint. The policy that the same repo trains and exports uses the per-joint
scales above (from the ONNX metadata). Deployed with 0.25, the legs and waist would get 1.67× the trained motion and the arms
and ankles 0.6–0.8×. Use the metadata values.
