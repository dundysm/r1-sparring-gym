# R1 model audit

Everything below was found by actually compiling and stepping the official Unitree files in MuJoCo 3.x.
Re-run with `python -m r1gym.check_model` (prints joint ranges, caps, mass and the stand tests).

## Which MJCF
| source | file | result |
|---|---|---|
| `unitreerobotics/unitree_mujoco` @1eb6642 (BSD-3) | `unitree_robots/r1/R1_C++.xml` | **not usable for balance as-is**: `waist_pitch`, `left/right_wrist_pitch`, `left/right_wrist_yaw` are hinge joints with `ctrlrange="0 0"` and no joint limits, so they flop freely; the robot can't hold a stand. No head joints. |
| `unitreerobotics/unitree_rl_mjlab` @1425b15 (Apache-2.0) | `src/assets/robots/unitree_r1/xmls/r1.xml` | **used**. Those five joints are fixed; 24 hinge DoF; 28.93 kg. This is the file Unitree trains its own R1 policies on. |

## Upstream bug: dangling contact excludes
`r1.xml` @1425b15 has `<exclude>` entries naming `left_wrist_pitch_link` / `right_wrist_pitch_link`, bodies that don't exist in that
file. Plain MuJoCo refuses to compile it. mjlab doesn't hit this because it edits the spec first. We drop dangling excludes in
`r1gym/model.py`. (Worth a one-line upstream PR.)

## Torque caps: the two official sources disagree
| joint group | unitree_mujoco `ctrlrange` | unitree_rl_mjlab `effort_limit` | **used (min)** |
|---|---|---|---|
| hip pitch/roll/yaw | 88 | 60 | 60 |
| knee | 139 | 60 | 60 |
| ankle pitch/roll | 50 | 50 | 50 |
| waist roll / yaw | 50 / 88 | 60 / 60 | 50 / 60 |
| shoulder pitch/roll | 25 | 60 | 25 |
| shoulder yaw, elbow, wrist roll | 25 | 33 | 25 |

Default is the element-wise minimum (`torque_cap_policy: min` in `configs/r1_assumptions.yaml`). It's conservative: a policy that
survives here should survive the real robot. Switch to `mujoco` or `mjlab` for the other source.
**Ask Unitree / measure on the EDU unit before trusting either.**

## The stock PD gains can't hold a passive stand
Unitree's deploy gains (ankle kp 40) give ankle stiffness below m·g·h ≈ 190 N·m/rad. With PD alone, the R1 tips over in about 1.1 s
(`check_model` output: `gains=pd: FALLS at 1.09s`). Unitree's walking policy keeps it up actively. For a fair hit study, both
the baseline and the RL policy use `stand_pd` (leg/ankle kp 300, kd 10). The RL policy is a residual on top of it, so the
only difference between the two is the learned feedback.

## Numerical assumptions we had to make (all in the YAML)
* **Integrator `implicitfast`.** With Euler at dt = 5 ms, PD damping above about 4 N·m·s/rad on the light distal links is
  numerically unstable.
* **Armature 0.03 (Unitree uses 0.01 everywhere).** At 0.01, the stiff stand falls in 16/16 trials from 0.1·0.25 rad white
  leg-target noise; at 0.03 it falls in 3/16. That's a placeholder until rotor inertia × gear² is known.
* **Foot friction μ = 1.0 (mjlab uses 0.6).** At 0.6, the stand slips: 6/10 falls from 0.02 rad initial joint noise.
* **Action EMA filter 0.3.** Keeps exploration noise from shaking the R1's small feet (about 16 cm × 5 cm per foot).
* **Fall definition:** pelvis z < 0.45 m or tilt > 60°.

`scripts/sweep_stand.py` reproduces the gain sweep.

## The stiff stand caps exploration
With no hits at all, white Gaussian residual actions knock the `stand_pd` R1 over (`python scripts/noise_tolerance.py`, 16 × 5 s per level):

| exploration std (action units) | ≈ joint-target noise | falls |
|---|---|---|
| 0.03 | 0.007 rad | 0/16 |
| 0.06 | 0.015 rad | 0/16 |
| 0.10 | 0.025 rad | 3/16 |
| 0.15 | 0.037 rad | 16/16 |
| 0.25 | 0.062 rad | 16/16 |

So PPO on a residual over this stand can only explore about ±0.015 rad per joint. That is far too little to discover stepping or
large recovery motions, which is why every run (v1, v2, both v3 runs) stalls at 8–9 N·s. Locomotion policies that do learn
push recovery (e.g. Unitree's own R1 velocity task) act on softer gains and are trained with noise an order of magnitude larger.

## Collision shapes are thinner than the visible robot
`r1.xml` collides through capsules and spheres that sit inside the visual meshes: the pelvis mesh is ~16.5 cm across
against a 10 cm sphere, the elbow ~9 cm against a 6 cm capsule, the knee ~10 cm against 7 cm, and 9 of the 25 links
(shoulder pitch/roll housings, hip and ankle pitch, waist roll) have no collision shape at all. For walking that is
fine, but with two robots the visible shells passed up to ~8 cm into each other before the capsules touched ("clipping").
With an opponent the gym now lets each link's visual mesh (MuJoCo collides its convex hull) collide with the other robot's
only (`robot_contact: shell`, `r1gym/model.py`), with stiffer, frictionless contact (condim 1, as mjlab's non-foot geoms), and keeps the capsules for the ground and the
opponent's self-collision, which is what the policies trained with. `scripts/clip_check.py` measures the overlap
(`results/clip_check.json`): the deepest overlap per bout drops from a median ~5.5 cm to under 1 cm.
