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
  (`gpu/`) uses Unitree's own contract (kp 100/40, per-joint action scales), which this doc will cover once it ships.
- **Safety layer.** E-stop, torque/velocity caps below the vendor limits and a stale-observation watchdog are not
  part of the policy. See ROADMAP phase 4.
