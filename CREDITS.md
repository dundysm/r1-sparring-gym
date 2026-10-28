# Credits & licenses
- **Unitree R1 MJCF / meshes:** [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), Apache-2.0. Also see [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), BSD-3-Clause.
- **Opponent motions:** KungFuAthleteBot. Code is MIT, motion data is Apache-2.0 (per its repo). Replayed kinematically on a Unitree G1 model.
- **[MuJoCo](https://github.com/google-deepmind/mujoco)**, Apache-2.0.
- **[mjlab](https://github.com/mujocolab/mjlab)** (MuJoCo Warp), Apache-2.0: the batched GPU path in docs/GPU.md.
- PyTorch, ONNX Runtime.
- **Motion tracking:** unitree_rl_mjlab's tracking task, a re-implementation of [BeyondMimic](https://beyondmimic.github.io/) (HybridRobotics/whole_body_tracking), used to train the opponent (`gpu/r1_track_task.py`).
- **[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)** (Apache-2.0): the R1 velocity task the stepping base is built on.
- **[fast-simplification](https://github.com/pyvista/fast-simplification)** (MIT; wraps Sven Forstmann's [Fast-Quadric-Mesh-Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification), MIT): quadric decimation of the R1's visual meshes for the render twin (`scripts/build_render_lod.py`). The results, `r1gym/assets/r1/render_lod/`, are modified Unitree meshes under Apache-2.0; `render_lod/SOURCE` records the source commit and the hash of each original. Build-time only, not a runtime dependency.
