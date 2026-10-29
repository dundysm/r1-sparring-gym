# Sparring M1-M2: a GPU attacker, and a defender that sees the fists

Spec for two implementers working in parallel (GPU side, CPU side). Section 1 is the shared contract: every name, frame,
order and dimension both sides depend on. Nothing in it changes without editing this file first. The M2 pass rule
(section 5) is preregistered here and in `results/sparring/prereg_m2.json`, before any M1/M2 code or pod.

Sources read for this spec: mjlab 1.2.0 (wheel source), unitree_rl_mjlab @1425b15, mjlab dev (`5af32e3`), this repo at
`583a0bc`, and `results/policy_mjlab_guard_footwork_r/` (checkpoint, ONNX, `params/`). What could not be checked without a
GPU is listed in section 7. The architect's CPU probes (plain MuJoCo 3.17, not committed) are quoted where they set a number.

**Milestones.** M1: an attacker in the mjlab stepping-base scene that hits the frozen warm start about as hard and as
often as the CPU aimed opponent does. M2: `R1-Spar-Block` (sees the opponent's keypoints) and `R1-Spar-Block-Blind`
(the same inputs zeroed), warm-started from `model_7496.pt`, scored on CPU against the real tracker opponent.

---

## 1. Shared contract (pinned)

### 1.1 Frames and units
- **Defender torso frame (T).** Origin and axes of the defender's `torso_link` body (x forward, y left, z up; it follows
  waist roll and yaw). All observed keypoints are expressed in T. Units: m, m/s, rad.
- **Heading frame (H)** of a strike motion: forward (toward the defender), the opponent's left, up; origin on the floor
  at the opponent's spot (the in-place clip's centre). `r1gym/strikes.py` uses it (`H = P * [-1, -1, 1]` of the in-place
  motion, which faces -x).
- **Head camera.** The nominal `configs/r1_camera.yaml` net pinhole (`HeadCameraSpec().mount_torso`, `K()`), as numbers:
  lens at `c = (0.1310, 0.0, 0.2473)` in T; camera axes in T as columns (OpenCV: x right, y down, z optical):
  `x_c = (0, -1, 0)`, `y_c = (-0.7660, 0, -0.6428)`, `z_c = (0.6428, 0, -0.7660)` (head pitch -50 deg);
  `fx = fy = 50.346 px`, `cx = 80, cy = 60` (160 x 120, hfov 115.6 deg, vfov 100 deg).
  A point `p` (in T) is **in view** iff, with `q = R_cᵀ (p - c)`: `q_z > 0.05` and `|q_x / q_z| <= 80 / 50.346` and
  `|q_y / q_z| <= 60 / 50.346`. FOV only: self-occlusion by the defender's own arms is not modelled (section 7).

### 1.2 Opponent keypoints (definitions identical on both sides)
| # | name | CPU (real tracker opponent, bodies prefixed `opp_`) | GPU (attacker) |
|---|---|---|---|
| 0 | `l_glove` | `opp_left_wrist_roll_link` frame + (0.08, 0, 0) | origin of attacker body `opp_left_wrist_roll_link` |
| 1 | `r_glove` | `opp_right_wrist_roll_link` frame + (0.08, 0, 0) | origin of attacker body `opp_right_wrist_roll_link` |
| 2 | `l_elbow` | `opp_left_elbow_link` origin | left attacker body origin - 0.195 x its body x axis |
| 3 | `r_elbow` | `opp_right_elbow_link` origin | right attacker body origin - 0.195 x its body x axis |
| 4 | `head` | `opp_torso_link` frame + (0.01, 0, 0.357) | virtual: schedule head path placed by the live frame (2.4) |
| - | opponent yaw | heading (x axis on the ground) of `opp_torso_link` | facing direction `f` + recorded torso yaw (2.4) |

The CPU offsets are `r1gym/perception/render_scene.py` `KEYPOINTS` (`l_fist`, `l_elbow`, `head`, without the camera's
head-pitch term: the opponent's head is rigid). 0.195 m is the R1's elbow-origin-to-fist distance, constant under wrist
roll; the export's elbow-to-fist distance is the same by construction. Left/right are the **opponent's own** sides (as
the gloves detector labels them).

### 1.3 The keypoint block (`KP_DIM = 28`, float32)
| index | content |
|---|---|
| 0:3, 3:6, 6:9, 9:12, 12:15 | positions in T of keypoints 0..4 |
| 15:18, 18:21 | `l_glove`, `r_glove` velocity in T: `(p_T(t) - p_T(t - 0.02)) / 0.02` of the **true** T positions, one control step apart (0 at the first step of an episode) |
| 21:23 | `(sin, cos)` of `wrap(yaw_opp - yaw_def_torso)`; facing each other is about (0, -1) |
| 23:28 | masks for keypoints 0..4: 1 = present, 0 = absent |

**Zero-plus-mask.** A keypoint with mask 0 has all its entries zeroed (position; for gloves also velocity). The yaw pair
is present when any of the five masks is 1 (section 8 item 14; was: the `head` mask). Mask = in view (1.1) AND not dropped (GPU training only, 3.2). Truth on CPU: mask = in view.
Order of operations: true positions -> velocities -> (GPU training: noise and dropout) -> mask applied -> delay.

### 1.4 Observation layouts and ONNX metadata
| policy | actor input | `observation_names` metadata |
|---|---|---|
| warm start / all shipped | 83 | `base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions` |
| `R1-Spar-Block-*` sighted | 83 + 28 = 111 | the above + `,opp_keypoints` |
| `R1-Spar-Block-Blind*` | 111 | the above + `,opp_keypoints_blind` (CPU feeds 28 zeros) |

The keypoint block is **appended after the 83** (indices 83:111). Critic (GPU only): 98 (unchanged) + 28
(`opp_keypoints_truth`: no noise, dropout or delay; mask = in view) + 10 (`strike_phase`, 2.6) = 136.
`command_names` metadata becomes `twist,attack` (nothing on CPU reads it). Joint names, order, default pose (the guard),
stiffness, damping and action scale are unchanged from the warm start; `MjlabPolicy`'s existing asserts still apply.

### 1.5 Files both sides read
- `gpu/motions/attacks_train.npz` (+ `attacks_train.json`): section 2.3.
- `gpu/motions/attacks_shells.npz` (+ zone map inside): section 2.1.
- `results/policy_mjlab_spar_{a,b,blind}/` after M2: `policy.onnx`, `model_*.pt`, `params/`, and `gym.json` **copied
  unchanged from `results/policy_mjlab_guard_footwork_r/gym.json`** (face + hold outer loops, as the warm start).

---

## 2. M1: the attacker in the stepping-base scene

### 2.1 Scene (mjlab 1.2.0 APIs, as read)
Scene composition in mjlab: `Scene.__init__` builds terrain, then each `SceneCfg.entities` entry (`EntityCfg.build()`,
attached with prefix `"<name>/"`), then sensors, then `SceneCfg.spec_fn(spec)` (runs last, before compile; the documented
place for cross-entity edits). Entity editors (`CollisionCfg`, `disable_other_geoms=True`) run inside `Entity.__init__`,
after the entity's own `spec_fn`.

**Why not free fists on soft welds.** Probed on CPU (scratchpad `m1m2/probe_attacker.py`): a 1.5 kg free capsule welded to
a mocap target (solref 0.01-0.04) gave 18-60 N·s per head hit against 2.37 measured on CPU, because a weld has no force
limit and keeps pressing after impact; softening it (tc 0.04) left the fist 0.23-0.29 m behind its target. Also, MjSpec's
`add_equality()` default `data` is not an identity weld (it compiled to relpos (0, 1, 1)); `data[0:11]` must be set.
**Chosen: an actuated chain with force-capped position servos**, the path mjlab supports best (built-in position actuators,
joint targets per env), with no per-step mocap writes and no equality constraints.

**Entity `attacker`** (`SceneCfg.entities = {"robot": ..., "attacker": ...}`, attacker second so the robot's qpos, ctrl
and actuator ids, and `get_base_metadata`, are unchanged). Its `spec_fn` returns, per side `s in (l, r)`
(`nm = left/right`), under one root body `att_root` (no joint; mjlab auto-wraps a fixed-base entity in a mocap body that
stays at the world origin, so slide values are world coordinates):
`att_s_x` (slide x) -> `att_s_y` (slide y) -> `att_s_z` (slide z) -> `att_s_yaw` (hinge z) -> body
`opp_{nm}_wrist_roll_link` (hinge y, joint `att_s_pitch`).
- Intermediate bodies: mass 0.01 kg, inertia 1e-5 (explicit). All attacker bodies `gravcomp = 1`.
- Last body: capsule geom `att_s_capsule`, `fromto (-0.195, 0, 0, 0.03, 0, 0)`, radius 0.045, mass 1.5 kg (calibrated,
  2.5); body origin = the glove keypoint; distal tip 0.075 m beyond it. `contype 4, conaffinity 0, condim 1`,
  `solref (0.01, 1)`, `solimp (0.95, 0.99, 0.001, 0.5, 2.0)` (`r1gym/model.py` SHELL_*).
  The name `opp_*_wrist_roll_link` lets r1gym's `HitMeter` label the capsule `hand` in the CPU replica.
- Actuators: `EntityArticulationInfoCfg(actuators=(BuiltinPositionActuatorCfg(target_names_expr=("att_._[xyz]",),
  stiffness=1500, damping=95.5, effort_limit=150, armature=0), BuiltinPositionActuatorCfg(target_names_expr=
  ("att_._(yaw|pitch)",), stiffness=60, damping=3, effort_limit=30, armature=0.01)))`. Damping 95.5 = critical on 1.52 kg.
- `EntityCfg.build()` returns `AttackerEntity(Entity)` which overrides `write_data_to_sim()` (called every physics
  substep by `Scene.write_data_to_sim`) to set `joint_pos_target = lerp(q_prev, q_next, min(1, (k+1)/4))`, plus
  `(95.5/1500) * (q_next - q_prev)/0.02` on the three slides (damping feed-forward), then `super().write_data_to_sim()`;
  `update(dt)` increments `k`. The attack command sets `q_prev <- q_next, q_next <- new, k <- 0` once per env step.

**Contact filtering (bitmasks, no explicit pairs).** Fist capsules: contype 4 / conaffinity 0. Defender collision proxies
(below): contype 0 / conaffinity 4. Everything else keeps its bits (robot capsules 1/1, terrain plane 1/1). So a fist
touches only the proxies: not the floor, not the other fist, not the robot's capsules; proxies touch nothing else, so the
robot's self-collision sensor, `self_collisions` reward and foot contacts are unchanged.

**Defender collision proxies (CPU shells, GPU-sized).** The CPU eval collides robot-vs-robot on the visible shells (convex
hulls of the visual meshes, `robot_contact: shell`). Their hulls have 532-3811 vertices (head_yaw 3811). MuJoCo's
`maxhullvert` is not usable: at 64 it shrank hull extents by up to 9 cm on the head and 12-25 cm on the torso/waist (probed).
Instead `gpu/export_attacks.py` exports, per link, the 64 farthest-point samples of the full hull's vertices in the **body
frame** (`geom_pos + geom_mat @ v` from the compiled r1gym model): extents within 0.4 cm (1.4 cm in y on `waist_yaw`),
probed. 16 links, with their CPU zones:

| zone | links (proxy geom name = `<link>_shell`) |
|---|---|
| head | `head_pitch_link`, `head_yaw_link` (both on `torso_link`) |
| torso | `waist_yaw_link` (on `torso_link`) |
| pelvis | `pelvis_link`, `imu_in_pelvis_link` (on `pelvis`), `waist_roll_link` |
| block (forearm/hand) | `left_elbow_link`, `right_elbow_link`, `left_wrist_roll_link`, `right_wrist_roll_link` |
| upper_arm | `{left,right}_shoulder_{pitch,roll,yaw}_link` |

Added in the **robot's** `spec_fn` (a wrapper around `get_spec`): one mesh asset per link from `uservert` and a geom
`<link>_shell` on that body, `density 0`, group 3. `CollisionCfg(disable_other_geoms=True)` then zeroes their bits (they
don't match `.*_collision`), so `SceneCfg.spec_fn` sets them on `robot/<link>_shell`: contype 0, conaffinity 4, condim 1,
SHELL solref/solimp. The names exist when the contact sensors resolve them (sensors run before `SceneCfg.spec_fn`).
Fallback if mesh collision misses the speed gate (2.5): `shell_proxy="capsule"`, one PCA capsule per link enclosing 95% of
its 64 points (also in the export).

**Contact sensors** (2, one per fist): `ContactSensorCfg(name="hit_l", primary=ContactMatch(mode="geom", entity="robot",
pattern=<the 16 *_shell names, zone order of the table>), secondary=ContactMatch(mode="body", entity="attacker",
pattern="opp_left_wrist_roll_link"), fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4)`; same
for `hit_r`. `force_history` is `[B, 16, 4, 3]` (index 0 = latest substep). Per substep zone force =
sum over the zone's primaries of `|F|` (condim 1: normal force only, as `HitMeter` sums `|f_normal|`).

**Sim settings.** Inherit the footwork task's (`njmax 300`, timestep 5 ms, decimation 4, implicitfast). Fist contacts add
1 row per contact; no new equality or limit rows.

### 2.2 Placement and reset
- Defender root pose at reset is read from **qpos** (`env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]`):
  derived quantities (`root_link_pose_w`, `xpos`) are stale until the `sim.forward()` that follows the reset block.
- Attacker reset (in the attack command's `_resample_command`): both hands idle at guard in the live frame, written with
  `attacker.write_joint_state_to_sim(q_idle, zeros, env_ids=env_ids)`, `q_prev = q_next = q_idle`.
- If the mocap base is ever moved: `write_mocap_pose_to_sim(pose, env_ids=<tensor>)`. With `env_ids=None` it raises in
  mjlab 1.2.0 (`mocap_pos[:, id] = pose.unsqueeze(1)`, (N,1,3) into (N,3); checked with torch). Not needed here.

### 2.3 Strike library export (`gpu/export_attacks.py`, CPU, train split only)
For every **train** hand strike in `r1gym/strikes.py` (`S.load(split="train")`, `hand` only: 37 strikes, 17 left / 20
right, clips kfab_200/201/203; the script asserts every segment's clip is in `TRAIN_CLIPS`): the in-place motion
(`S.motion(clip)`, 50 fps, wraps modulo the clip length), frames `onset - 12` to `end + 12` (0.24 s each side), forward
kinematics on `_sub("r1")`: the 5 keypoints of 1.2 (opponent side) in H, and the torso yaw in H
(`wrap(yaw_world - pi)`, 0 = facing the defender). Output `attacks_train.npz`:
- `kp` float32 `[F, 5, 3]` (segments concatenated), `yaw` float32 `[F]`, `fps` = 50.0;
- `seg` int32 `[S, 6]`: start row in `kp`, length n, onset, apex, end (segment-local frames), hand (0 left, 1 right);
- `guard` float32 `[2, 5, 3]`: per hand, the median over that hand's segments of their first and last frames.
`attacks_train.json`: per segment clip, onset frame, limb, band, type, rise_s, peak_speed; the rule; the git hash.
`attacks_shells.npz`: per link (2.1) `[64, 3]` body-frame points, a capsule fallback `[link] -> (p0, p1, r)`, the zone map.

### 2.4 Attack schedule (torch, batched; `AttackSchedule` in `gpu/sparring_task.py`, no mjlab import)
Per episode: stand-off `D ~ U(0.35, 0.65)` m (section 8 item 20; was 0.35-0.55) (defender pelvis to opponent spot, the CPU "distance"); rate
`lam ~ U(0.5, 2.0)` strikes/s; first strike after `U(0.5, 1.5)` s.

**Live frame.** Bout axis angle `alpha` (world): at reset `yaw_def + beta + U(-50, 50) deg`; every `U(2, 5)` s a new target
`yaw_def + beta + U(-50, 50) deg` (per-episode `beta ~ U(-30, 0)` deg and virtual-head lift `U(0, 0.12)` m: section 8 item 20; was `U(-30, 30)`), followed at <= 0.5 rad/s. Spot target = defender pelvis xy + `D (cos alpha, sin alpha)`; the
spot `S` follows it with a 0.5 s first-order lag capped at 0.3 m/s (the CPU tracker's steering). Facing
`f = unit(pelvis_xy - S)`, left `l = z x f`. `World(k) = S + k_x f + k_y l + k_z z`.

**Strikes.** At `t >= t_next`: pick a hand uniformly among idle hands (none idle: wait), a segment of that hand uniformly,
speed `sigma ~ U(0.7, 1.3)`, zone head with probability 0.8 else torso; `t_next = t + max(0.25, Exp(mean 1/lam))`. The
segment's frame `(S0, f0)` is frozen at its start. Segment clock `tau = (t - t0) * 50 * sigma` frames, `n` frames long.
- Guard blend: `K'(tau) = K(tau) + (1-u)(G - K(0)) + u(G - K(n-1))`, `u = smoothstep(tau/(n-1))`, G = that hand's guard.
- Aim weight `w(tau)` (CPU ArmWarp timing, frames at 50 fps): smoothstep up from `on - 7.5` to `apex`, 1 until
  `apex + 2.5`, smoothstep down to `end + 5`; 0 outside.
- Aim point (zone surface in the defender's T, live): head `(0.087, 0, 0.36)`, torso `(0.070, 0, 0.15)`; the glove
  target is `surface_w - (0.075 - delta) u_hat`, `u_hat` = horizontal unit from `S0` to the surface point, `delta = 0.005`
  m (tip 5 mm into the shell; calibrated, 2.5). `Delta = aim_w - World0(K'(apex))[glove]`, recomputed every step while
  `tau < onset` and **latched at onset** (world-fixed), so slipping after onset makes the strike miss. `|Delta| <= 0.5` m
  (scaled down if larger).
- Targets: glove `World0(K'(tau))[glove] + w Delta`, elbow `World0(K'(tau))[elbow] + w Delta` (the forearm translates).
- After `tau >= n-1` the hand blends to its idle target `World(G)` over 0.2 s (smoothstep), then idles.
- Joint targets: slides = glove target; `u = glove - elbow`; yaw = `atan2(u_y, u_x)` unwrapped to the current joint value;
  pitch = `-atan2(u_z, |u_xy|)`.
- Virtual head and yaw: the most recently started active segment's `World0(K'(tau))[head]` and `atan2(f0) + yaw(tau)`
  (guard-blended like K); with no active segment `World(G_head)` (mean of both hands' guard heads) and `atan2(f)`.

### 2.5 Hit accounting on GPU (`HitAccount`, torch) and calibration
Mirrors `r1gym/hits.py` DEFAULTS: per (fist, zone) episodes from the per-substep zone forces of 2.1, merged over gaps
<= 0.1 s; impulse = sum `F * 0.005`; peak; duration = last - first + 5 ms; landed = head/torso, impulse >= 0.5 N·s, peak >= 5
N; clean = landed, closing speed >= 1.0 m/s, duration <= 0.15 s. Closing speed (approximation, no per-substep velocities):
at the control step before the episode's first substep, `(v_glove - v_torso_link) . n_hat`, `n_hat` = unit from the glove
tip (glove + 0.075 x body x) to the zone centre (head `(0.03, 0, 0.36)`, torso `(0, 0, 0.15)` in T). An episode is
scored when it closes (<= 0.1 s after its last contact). `update(env)` is idempotent per `env.common_step_counter`, so
the first reward term that calls it does the work (rewards run before commands in `ManagerBasedRlEnv.step`).
Per strike window `[t_on, t_apex + 0.15 s]` it records the striking fist's first contacted zone (`blocked` = arms first,
as `HitMeter.windows`). Metrics (attack command `metrics`, logged at reset): head/torso clean per minute, strikes per
minute, blocked share, mean impulse per head landed; a 4096-entry ring buffer of head-landed impulses for medians.

**Calibration (CPU, before the pod): `python gpu/sparring_task.py calibrate`.** Builds r1gym's model (`build_spec`, mjlab
contract, guard home), adds the 16 proxies and the attacker chain with the same pure-MjSpec functions the GPU task uses,
the same two contact sensors (MuJoCo `mjSENS_CONTACT`, intprm as mjlab writes them), runs `AttackSchedule` (torch CPU)
against the frozen warm-start ONNX in `R1HitEnv` (no FightReplay: zero command), and scores with **both** r1gym's
`HitMeter` and `HitAccount`. 64 bouts x 20 s, seeds 0..63. `HitMeter` zones heads by mesh name, so in the replica the
proxy meshes take the original mesh names on the visual geoms (originals renamed `<name>_full`); `--geometry shell` keeps
the full hulls (the eval's geometry) for comparison and is reported, not gated. Grid, in this order, first that passes wins:
`(mass, F_max, delta) = (1.5, 150, 0.005), (1.5, 150, -0.02), (1.5, 80, 0.005), (1.0, 150, 0.005), (1.5, 80, -0.02)`.
Pass: HitMeter median impulse per head hit in [1.58, 3.56] N·s (x1.5 of 2.37, `policy_mjlab_guard_footwork_r/
summary_val_hard.json`); head clean/min >= 7.7; glove speed p98 >= 0.85 x reference p98; HitAccount within +-15% of
HitMeter on head clean/min and +-10% on the median impulse. Writes `results/sparring/m1_calibration.json`.
Architect's probe (16 bouts x 10 s, rate 1/s, all head-aimed, D = 0.45, sigma = 1): `(1.5 kg, 150 N, tip +5 mm)`: median
3.20 N·s, head clean 17.7/min, speed 92% of reference, survival 0.94; at 40 N: 2.81 N·s but speed 65%.

### 2.6 Task `R1-Spar-Attack` and the M1 gate
`R1-Spar-Attack` = `R1-Spar-Footwork-Guard` (`_footwork(_guard(_cfg(2)))`) + the attacker + proxies + sensors + `attack`
command + `nan` termination (`mjlab.envs.mdp.terminations.nan_detection`) + metrics; actor/critic, rewards unchanged.
`strike_phase` (critic, M2), per hand l then r: active (0/1), segment progress `u`, time to apex in s clipped to
[-0.5, 0.5], zone head (0/1), zone torso (0/1).

**M1 gate** (all on the pod, written to `results/sparring/m1_gate.json`; M2 training starts only if it passes):
1. `python sparring_task.py m1check` on `R1-Spar-Attack`, 4096 envs, 1000 steps, frozen warm-start actor
   (`model_7496.pt`, deterministic mean, normaliser applied), `push_robot` and `shove` removed (the CPU 7.72 is "no
   scripted hits"), corruption and DR on: **head clean strikes/min >= 7.7** (pooled count / pooled minutes up) and
   **median impulse per head landed hit in [1.58, 3.56] N·s**.
2. 200 iterations of `R1-Spar-Block-A` from the widened checkpoint: **mean iteration time <= 2.0 s** (iterations 20-200)
   and **0 `nan` terminations**.
If 2 fails on speed: `shell_proxy="capsule"`, then `num_envs 3072`, recorded in the gate file.

---

## 3. M2: `R1-Spar-Block`, `-B`, `-Blind`

### 3.1 Warm start and widening (`python sparring_task.py widen <src.pt> <dst.pt>`)
Verified on `model_7496.pt` (rsl_rl 5.0.1 layout): actor `obs_normalizer._mean/_var/_std (1, 83)`, `count` 7.37e8,
`distribution.std_param (24,)`, `mlp.0.weight (512, 83)`, hidden 512-256-128 ELU; critic `(1, 98)`, `mlp.0.weight
(512, 98)`; one Adam param group (actor params 0-8, critic 9-16), `exp_avg (512, 83)` and `(512, 98)`; `iter` 7496;
`agent.yaml`: `obs_groups actor=(actor,), critic=(critic,)`, lr adaptive (now 2.56e-4). The actor's 83 =
`base_ang_vel 3, projected_gravity 3, command 3, phase 2, joint_pos 24, joint_vel 24, actions 24`; the critic's 98 =
those + `base_lin_vel 3, foot_height 2, foot_air_time 2, foot_contact 2, foot_contact_forces 6` (Unitree builds the
critic as `{**actor_terms, ...}`, so the actor's 0-40 ms delay also applies to those critic terms; leave it).
`widen` inserts **actor 28 columns at 83**, **critic 38 columns at 98**: `mlp.0.weight` zeros; Adam `exp_avg`,
`exp_avg_sq` zeros for those columns (matched by shape `(512, 83)` / `(512, 98)`, asserted unique); normaliser for the
new dims (the count is 7.4e8, so these stay put): positions mean 0 std 0.5, velocities mean 0 std 2.0, yaw pair mean
(0, -1) std 0.5, masks mean 0.5 std 0.5, strike phase mean 0 std 1 (var = std²). The widened policy acts exactly like the
warm start until the new weights move. Output in `logs/rsl_rl/r1_spar/widen_spar/model_7496.pt`.

### 3.2 Task definition (`gpu/sparring_task.py`)
`R1-Spar-Block-{A,B}` = `R1-Spar-Attack` plus:
- **Actor** term `opp_keypoints` (func returns the 28-d block; `noise=None`: noise is applied inside the func so the mask
  stays exact; `delay_min_lag 0, delay_max_lag 2` as the other actor terms; play config: no delay, no corruption).
  Training corruption (param `corrupt=True`): per step U(+-0.02) m on positions plus a per-episode bias U(+-0.02) m per
  keypoint axis; U(+-0.25) m/s on glove velocities; U(+-0.1) rad on the relative yaw before sin/cos; each keypoint
  dropped with p = 0.10 per step (gloves found in 88% of in-view frames on val, `results/perception/gloves_check.json`).
- **Critic** terms `opp_keypoints_truth` (28, clean) and `strike_phase` (10), appended after the 98.
- **Command** `twist` -> `OuterLoopTwistCommand(UniformVelocityCommand)`: per episode 50% of envs take the CPU outer loops
  (`r1gym/env.py` `_face`, `_hold`, with the warm start's `gym.json`) every step, the rest keep the footwork command
  (30% standing): `wz = clip(2.0 * bearing_err, +-0.8)` with the bearing of the live spot `S` from the pelvis heading;
  hold the episode's start pelvis xy with hysteresis 0.25 / 0.08 m, `vx = clip(1.5 e_fwd, -0.5, 0.8)`,
  `vy = clip(1.5 e_left, +-0.5)`. The start spot is read from qpos at reset. `twist` before `attack` in `commands`.
- **Kept**: footwork ranges, 1.5x tracking weights, `push_robot`, `shove`, DR, terminations (+ `nan`).
- **Posture** (`pose`, both weightings): non-overlapping keys (Unitree resolves each joint to one pattern):
  `std_standing = {hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll, waist_yaw: 0.05; waist_roll: 0.15;
  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist: 0.35}`; in `std_walking` and `std_running` the arm keys
  become 0.35 and `waist_roll` 0.15, the rest unchanged.
- **Rewards** (mjlab multiplies every term by `step_dt` = 0.02; events are counts per step):

| term | value per step | A | B |
|---|---|---|---|
| `spar_head_impulse` | head N·s this step, both fists | -20 | -40 |
| `spar_head_clean` | clean head episodes closed this step | -40 | -80 |
| `spar_torso_impulse` | torso N·s this step | -5 | -10 |
| `spar_block` | strike windows whose first contact is on the `block` zone | +20 | +40 |
| `spar_clearance` | per active head-aimed window `[t_on - 0.15, t_apex + 0.15]`: `clip((d - 0.10)/0.15, 0, 1)`, `d` = glove tip to the defender head point (T `(0.01, 0, 0.357)`), summed over hands | +0.5 | +1.0 |

  A clean 2.4 N·s head hit costs 1.76 (A) / 3.52 (B), against 4.0 for a fall (`is_terminated` -200 x 0.02).
- **`R1-Spar-Block-Blind`** = `-A` with the actor term renamed `opp_keypoints_blind` returning zeros (28); critic
  identical to `-A`. If B is selected (5), `R1-Spar-Block-Blind-B` is trained before any test run (preregistered).

### 3.3 Runs (RunPod RTX 4090, unitree_rl_mjlab @1425b15, mjlab 1.2.0, mujoco 3.5.0, warp-lang <1.13)
```bash
python sparring_task.py widen model_7496.pt logs/rsl_rl/r1_spar/widen_spar/model_7496.pt
python sparring_task.py m1check --checkpoint model_7496.pt          # gate 1
python sparring_task.py R1-Spar-Block-A --env.scene.num-envs 4096 --agent.max-iterations 200 --agent.logger tensorboard \
  --agent.resume True --agent.load-run widen_spar --agent.load-checkpoint model_7496.pt --agent.run-name smoke   # gate 2
for t in A B Blind; do python sparring_task.py R1-Spar-Block-$t --env.scene.num-envs 4096 --agent.max-iterations 1500 \
  --agent.logger tensorboard --agent.resume True --agent.load-run widen_spar --agent.load-checkpoint model_7496.pt \
  --agent.run-name spar_$t; done
```
1500 iterations each (ends at `model_8995.pt`; the final checkpoint is the candidate, no checkpoint picking). Budget:
<= 2.0 s/iter -> <= 50 min per run, about 3 h of pod with the gate (~$2.7 secure). Download every artefact before
deleting the pod (the guard checkpoint was lost that way once).

---

## 4. CPU eval contract

- **Where the keypoints come from.** `r1gym/keypoints.py` `KeypointSource(m, mode, delay)`: on its **own** `MjData`
  twin, copies the env's `qpos` and runs `mj_kinematics` (exact current state; the env's `xpos` lag one substep after
  `mj_step`), reads 1.2's CPU column (defender `torso_link`, opponent `opp_*`), builds the 28-d block (1.3) with
  mask = in view, no noise, no dropout. It never writes the env's MjData.
- **Timing and delay.** `eval_opponent.run` calls `source.reset(d)` right after `e.reset` and `source.observe(d)` right
  after each `e.step`, and delays it with a `DelayedNoisyObs`-identical line of the episode's `ep["delay"]` (0-2 control
  steps; the same k the 83 proprioceptive dims get), zero noise. Velocities are one-step differences of true T positions
  (0 at reset). The block is appended: `obs = concat(o, kp)`.
- **Sources** (`--keypoints`): `truth` (above); `camera` (later, M3): gloves from the PerceptStream gloves detector
  (`xyz` in the pelvis heading frame -> T through the sensed waist encoders, `mount_fk`), velocities by differencing,
  elbows, head and yaw absent (mask 0). The interface is fixed now, the camera body raises `NotImplementedError`.
- **Blind**: a policy whose metadata ends in `opp_keypoints_blind` always gets 28 zeros, whatever `--keypoints` says.
- **Policies**: `r1gym/policy.py` `MjlabPolicy` accepts the two widened `observation_names` (1.4), exposes
  `kp_mode in (None, "obs", "zeros")` and `obs_dim`; `act` expects 111-d rows for widened policies. A widened policy
  without `--keypoints` (sighted) is an error.
- **Defaults are bit-identical**: without `--keypoints` and without widened controllers nothing new runs; `meta` gains a
  `keypoints` key only when the flag is set.

---

## 5. Preregistered M2 pass rule (also `results/sparring/prereg_m2.json`)

**Selection (val only).** `scripts/eval_opponent.py --split val --tier hard --no_pushes --reps 32 --hits 1 --keypoints
truth`, unaimed then `--aim chosen` (128 bouts each), controllers `guard_fw_r` (warm start), A, B, blind. Among A and B,
those with aimed val survival >= 0.95 and up-within-1 m drop <= 5 pp vs the warm start; of those (or of both, if none
qualifies) the lower aimed head clean/min; a difference under 0.1/min picks A. If B wins, train Blind-B, then test.

**Test (once).** Same command with `--split test --reps 16` (2 clips x 4 distances x 16 = 128 bouts), unaimed then
aimed, controllers `guard_fw_r`, the selected candidate, its blind twin, in one invocation (identical episodes). From
`raw_test_hard_nohits_aim.json`, exposure-weighted, cluster bootstrap over `(clip, t0)` (4000 resamples, seed 0, clusters
resampled jointly for paired controllers), ratio of rates `R = (sum n_c / sum min_c) / (sum n_ref / sum min_ref)`:
1. head clean/min received: `R_warm <= 0.60` and the 95% CI upper bound of `R_warm` `< 0.80`;
2. vs the blind twin: CI upper bound of `R_blind < 0.90`;
3. torso clean/min: `R_warm(torso) <= 1.5` (point estimate);
4. survival (aimed run) `>= 0.95`;
5. up-within-1 m (`survival_in_ring`): warm minus candidate `<= 0.05`.
All five pass -> M2 passes. Reported regardless: every `HitMeter` rule (sensitivity), intent-window blocked share,
impulse per head hit, unaimed rows. `scripts/m2_verdict.py` computes it and writes `results/sparring/m2_verdict.json`.

---

## 6. Task lists (disjoint files; section 1 is the interface)

**GPU implementer**
1. `gpu/export_attacks.py` (new, runs on CPU in the repo): 2.3 and the proxies of 2.1 -> `gpu/motions/attacks_train.npz`,
   `attacks_train.json`, `attacks_shells.npz`. Asserts train split, 37 segments, 16 links, 64 points each.
2. `gpu/sparring_task.py` (new; mjlab and unitree imports only inside functions, like `r1_spar_task.py`, so CPU tests can
   import it): pure-MjSpec `attacker_spec()`, `add_shell_proxies(spec, prefix)`, `add_contact_sensors_cpu(spec)`;
   torch `AttackSchedule`, `HitAccount`, `keypoint_block(...)` (1.3, no mjlab); mjlab glue (`AttackerEntity`,
   `AttackCommand`, `OuterLoopTwistCommand`, obs/reward/termination funcs); tasks `R1-Spar-Attack`, `R1-Spar-Block-A`,
   `-B`, `-Blind`, `-Blind-B` (experiment `r1_spar`); subcommands `widen`, `calibrate` (CPU), `m1check`, and training
   via `train.main()`.
3. `tests/test_sparring_task.py` (new, CPU): schedule latching (a target moved after onset does not move the strike),
   guard blend continuity, `|Delta|` cap, widen shapes and zero-output change on the warm start (actor outputs equal to
   1e-6 on random obs with arbitrary new dims), attacker + proxies compile in plain MuJoCo with r1gym's R1 and a fist
   never touches the floor, export schema.
4. Pod: section 3.3; results to `results/sparring/m1_calibration.json` (CPU), `m1_gate.json`,
   `results/policy_mjlab_spar_{a,b,blind}/` (+ `gym.json` copy).

**CPU implementer**
1. `r1gym/keypoints.py` (new): constants of 1.1-1.3 (names, offsets, layout slices, camera numbers), `in_view`,
   `to_torso`, `assemble(P_T, P_T_prev, rel_yaw, mask)` (numpy), `KeypointSource` (4), the camera-source stub.
2. `r1gym/policy.py`: widened metadata in `MjlabPolicy` (4); existing asserts unchanged for the 83-d policies.
3. `scripts/eval_opponent.py`: `--keypoints {truth,camera}` (default off) and the per-env source + delay line in
   `run()` for controllers with `kp_mode`; `meta["keypoints"]` only when set.
4. `scripts/m2_verdict.py` (new): selection (val) and verdict (test) of 5, ratio bootstrap, from the raw files.
5. Tests (new): `tests/test_keypoints.py` (layout, in-view on known points, zero-plus-mask, delay = DelayedNoisyObs,
   env MjData untouched); `tests/test_policy_widened.py` (synthetic widened ONNX: a Slice to the first 83 inputs in
   front of the warm-start graph, metadata set; loader accepts it); `tests/test_eval_keypoints.py` (flag off: rows equal
   to before on 2 short bouts; that synthetic widened policy with `--keypoints truth` reproduces the warm start's rows
   exactly); `tests/test_spar_contract.py` (skipped until `gpu/sparring_task.py` exists: torch `keypoint_block` equals
   `assemble` on random inputs to 1e-6; export files match 1.5).

---

## 7. API findings and what is not verified

Found by reading or CPU probes:
- `write_mocap_pose_to_sim(pose)` with `env_ids=None` raises (shape (N,1,3) into (N,3)); pass a tensor. Avoided here.
- MjSpec `add_equality()` default `data` is not an identity relpose (compiled relpos (0, 1, 1) in MuJoCo 3.17); set
  `data[0:11]`. Welds are avoided (unbounded force, 18-60 N·s per hit in the probe).
- `maxhullvert` destroys shell geometry (head -9 cm, waist up to -25 cm at 64); farthest-point proxies keep <= 1.4 cm.
- Contact sensors are added before `SceneCfg.spec_fn`; geoms they match must be named at entity level. Entity
  `CollisionCfg(disable_other_geoms=True)` zeroes any extra robot geom's bits, so proxy bits are set in `SceneCfg.spec_fn`.
- Derived quantities are stale during the reset block: read qpos there.
- The obs normaliser count is 7.4e8, so new dims keep the mean/std `widen` writes.
- Rewards run before commands each step: `HitAccount.update` is idempotent per step and called from the reward terms.
- The laptop has MuJoCo 3.17; the pod pins 3.5.0. The CPU calibration and eval run on 3.17.

Not verified (needs the GPU; first thing the pod checks, in `m1check`/smoke order):
- MuJoCo Warp 3.5.0.x: capsule-vs-mesh (64-point hull) collision speed and correctness at 4096 worlds; `gravcomp`
  (fallback: `ctrl_z += m g / kp` and drop gravcomp); `mjSENS_CONTACT` with a geom primary and a body secondary,
  `netforce`, 32 primaries in total; 10 extra DoF and two extra contact sensors within 2.0 s/iter.
- That `AttackerEntity.write_data_to_sim` sees `k` advance per substep under CUDA graph capture (it is plain torch
  before each `sim.step()`, as actions are, so it should).
- `HitAccount`'s closing speed agrees with `HitMeter` (checked on CPU by `calibrate`, not on GPU).
- Ignored by design: self-occlusion in the FOV mask (the defender's guard covers 31% of the image at -50 deg,
  `configs/r1_camera.yaml`); the opponent's body (only fists, elbows and a virtual head exist on GPU; no clinches, no
  kicks); the GPU attacker is not the CPU tracker, which is why M2 is judged on CPU only.

---

## 8. Changes during build (GPU side, 2026-10-28)

What the build changed against sections 1-7, and why. Checked on CPU only: the scratchpad stub probe
(`m1m2/probe_sparring.py`, 86 checks after the fix rounds, against stubbed mjlab objects and plain MuJoCo 3.17), the tests
(`tests/test_sparring_task.py`, `tests/test_keypoints.py`, `tests/test_m2_verdict.py`) and the calibration below.
Nothing here has run on a GPU.

1. **Attacker fist mass 1.0 kg, not 1.5 (2.1, 2.5).** (The build's schedule; on the final schedule the grid picks another
   row and 1.0 kg is kept on purpose: items 11 and 21.) The 2.5 grid, run as written (`python gpu/sparring_task.py
   calibrate`, 64 bouts x 20 s, seeds 0-63):

   | (mass, F_max, delta) | HitMeter median N·s per head hit | head clean / min | glove speed p98 / reference | HitAccount clean / min, median | pass |
   |---|---|---|---|---|---|
   | (1.5, 150, +5 mm) | 3.78 | 13.4 | 0.98 | 13.7, 3.78 | no (impulse) |
   | (1.5, 150, -2 cm) | 3.77 | 8.5 | 0.98 | 8.7, 3.77 | no (impulse) |
   | (1.5, 80, +5 mm) | 3.61 | 12.3 | 0.88 | 12.5, 3.63 | no (impulse) |
   | **(1.0, 150, +5 mm)** | **3.21** | **13.2** [11.0, 15.4] | **1.05** | **13.1, 3.24** | **yes** |

   So `ATTACK["mass"] = 1.0`, and the slide damping is critical on 1.02 kg, 78.2 (95.5 was for 1.52 kg). The full hulls
   (`geometry shell`, reported only) give 13.4 / min and 3.31 N·s on the same bouts, so the proxies are faithful. The
   architect's probe got 3.20 N·s at 1.5 kg with every strike at sigma 1 and D 0.45; the schedule's sigma up to 1.3 hits
   harder. Not gated, but worth knowing: the defender stays up in 75% of these 20 s bouts, and this attacker lands 1.7x
   the CPU aimed opponent's 7.72 head clean strikes / min (the gate is a floor only). The calibration file is now
   `results/sparring/m1_calibration.json` (item 21 has the final run).
2. **Closing speed (2.5).** As specified (the glove-minus-torso velocity at the control step before contact, along glove tip ->
   zone centre), it overstated HitMeter's by 0.5 m/s on average: the servo brakes the fist hard (up to F_max / m) as it
   reaches its target 5 mm inside the shell, and the contact normal is not tip -> centre. On 16 CPU bouts it called 33%
   more head strikes clean than HitMeter (clean calls agreed 77%). Built instead (`fist_substep_vel`, `closing_speeds`): the
   free fist's velocity at the start of each substep, predicted from the step-start slide state (qpos / qvel, which are
   exact, not one substep stale) under the target law, force cap and implicit damping (within 0.24 m/s of MuJoCo at up to
   8 m/s), minus the torso velocity, along that substep's contact normal: the direction of the zone's net contact force
   (condim 1 is frictionless; the sensor's force points against the push, `FORCE_SIGN = -1`, measured). Clean calls now
   agree 97% (66 landed episodes, correlation 0.94), and HitAccount is within 3% of HitMeter on head clean / min in every
   grid row.
3. **Episode duration.** HitAccount builds episodes from substeps with force > 0. HitMeter uses substeps in contact, which
   often include one zero-force substep as the fist leaves, so HitAccount durations are usually 5 ms shorter (relevant
   only to the 0.15 s clean rule). mjlab keeps no per-substep `found`, so this stays; the agreement checks cover it.
4. **Sensor primary order (2.1).** mjlab resolves a `pattern` tuple with `find_geoms(preserve_order=False)`, i.e. in the
   robot's geom order (pelvis_link first), not the table's zone order. `AttackCommand` maps zones from the sensor's own
   slots (`zone_index`).
5. **Hit accounting in `R1-Spar-Attack`.** That task has no sparring rewards, so `HitAccount` updates from the attack
   command's `_update_metrics`, which mjlab calls after the reset block: when an episode ends by termination, its last
   step's contacts are lost (the reset zeroes the sensor history). The Block tasks update from the first reward term,
   before resets, as specified.
6. **m1check's command.** 2.6 does not say. m1check puts every env on the CPU outer loops (`OuterLoopTwistCommand`, share
   1.0), as in the CPU eval that gave 7.72; the footwork's random walking would carry the defender off the spot, which
   follows at <= 0.3 m/s. The CPU calibration replica has a zero command, as the architect's probe (no FightReplay, so
   `_face` / `_hold` do nothing).
7. **Gate 2 and fallbacks.** A new subcommand, `gate --m1 m1_gate.json --smoke-log <log>`, takes the mean "Iteration
   time" over iterations 20-200 and requires every logged `Episode_Termination/nan` to be 0; it merges the result into the
   gate file. The proxy kind is the env var `SPAR_SHELL_PROXY=mesh|capsule`, because tasks are registered before tyro
   parses flags. The pod script reruns m1check when it falls back to capsules (the contact geometry changed), then tries
   3072 envs. `SPAR_GRAVCOMP=0` is the gravcomp fallback of section 7 (z targets + m g / kp).
8. **Guard (2.3).** The per-coordinate median guard has elbow-to-glove 0.11 m (left) and 0.15 m (right), not 0.195. Only its
   direction is used (the idle fist's yaw and pitch), and the GPU elbow keypoint stays glove - 0.195 x axis. Left as
   specified.
9. **Proxies (2.1).** Farthest-point samples of the hull MuJoCo itself compiles (`mesh_graph`; 42-3811 vertices); every
   link's extents are within 0.5 cm of its hull's (head_yaw 0.49 cm). The capsule fallback holds >= 61 of 64 points.
10. **strike_phase "progress u"** is the guard blend's u = smoothstep(tau / (n - 1)); the five entries are 0 when the hand
    is not striking.

**Fix round after the build reviews (2026-10-28, GPU side; CPU-checked only).** Items 11-20 supersede 2 (the sign) and 5.

11. **Attacker strength (superseded by item 28).** ~~Kept on purpose at (1.0 kg, 150 N, +5 mm).~~ The attacker stays at (1.0 kg, 150 N, +5 mm), the grid's first passing row on the
    build's schedule. Training is deliberately harsher than the CPU eval opponent: 13.2 clean head strikes / min against
    the aimed tracker's 7.72, and the warm start stays up in 75% of the 20 s calibration bouts (13.8 / min and 0.73 on the
    final schedule, item 21). The gate only sets a floor, and this was chosen, not overlooked. On the final schedule the
    grid itself no longer picks this row (item 21), so the pod refuses to train with it unless the deviation is declared.
12. **Strike windows (2.5) follow `HitMeter.windows`.** Only contact episodes that open inside the window count, so a fist
    already resting on a forearm when the window opens is never its first contact (the review's repro `m1m2rev/blk.py`
    now gives block 0, head clean 1). Pelvis contacts never count. A window is blocked iff an arm episode (block or
    upper_arm, merged as HitMeter's one `arm` zone) opens before the first landed head or torso episode that opened in it,
    or opens and nothing lands. Landing is monotone, so the window is decided once it is past and no unlanded head or torso
    episode that opened before its arm episode is still open; a new strike of that fist, a reset or the flush decides it on
    what is known. `spar_block` pays a blocked window whose arm episode was on the `block` zone, at that decision (so up to
    about 0.1 s after `apex + 0.15 s`), never a strike that landed first. On 48 random contact sequences the blocked
    windows equal HitMeter's exactly (183 of 837); in calibration rows the shares agree to 0.006 (0.639 / 0.639, 0.616 /
    0.622).
13. **Closing speed is sign-free.** `closing_speeds` returns `|v . n|`: at a contact's first substep fist and target are
    closing, so the absolute value is the closing speed whichever way MuJoCo Warp's sensor force points. `FORCE_SIGN`
    now only feeds a diagnostic: m1check writes `head_landed_vneg_share` (head-landed episodes whose signed closing speed
    is < 0), which is 0-2% on CPU. A large share on the pod means the Warp sign is flipped, which no longer matters.
14. **The relative-yaw pair is present when any of the five keypoints is** (1.3 said: the head mask), on both sides
    (`r1gym/keypoints.py pack`, `pack_t`). The opponent's head alone is in view on only 36-41% of CPU eval steps; a camera
    that sees any part of the opponent can judge where it faces. With every mask 0 the pair stays +0.0. This supersedes
    1.3's sentence "The yaw pair uses the `head` mask" (section 1 is left as written; this item is the change record).
    `tests/test_keypoints.py` asserts the new rule.
15. **Hit update timing.** Every spar task gets a metrics term `spar_hits` (mjlab 1.2.0 runs metrics after the rewards and
    before the reset block), so `R1-Spar-Attack` and m1check count an ending episode's last step too; at a reset
    `HitAccount.close_envs` scores the episodes and windows still open (they were dropped before). The closing speed's
    step-start state (slide qpos / qvel, torso velocity) is now saved in `_update_command`, after the single forward of
    the step, so it is current for every env, reset ones included. An interval `push_robot` applied after the commands
    (step order: commands, then events) is still missed for that one step; that is a small and rare effect, left as is.
    **Metrics include the episode-end events; rewards do not.** `close_envs` runs in the command's reset, after
    `reward_manager.compute` of that step, and the new episode's `HitAccount.reset` zeroes the per-step values before the
    next one. So a clean head episode still open (within 0.1 s of its last contact) when the episode terminates or times
    out never pays `spar_head_clean`, and a window decided at the reset never pays `spar_block`; both are counted in the
    logged metrics and in m1check. The impulse terms are paid per substep and a fall costs 4.0, so nothing here is worth
    exploiting; mjlab's step order (rewards before resets) leaves no place to pay them.
16. **Head timing.** In mjlab 1.2.0's `ManagerBasedRlEnv.step` (physics, counters, terminations, rewards, metrics, resets,
    `sim.forward()`, commands, events, observations), the gloves seen by the observations are where the physics step
    drove them: to the targets of the previous command update. `_update_command` now stores the schedule's head and yaw
    before advancing it, so head and gloves share a timestamp (they were one control step apart).
17. **Glove velocities on GPU: `KeypointHistory`.** It is a pure-torch class holding each env's previous T-frame block. The
    velocity is the current T-frame block minus the previous one, as the CPU computes it, and at reset the previous block
    is set to the current one, so the velocity is 0. Driven by 3 recorded CPU tracker bouts with resets in between, it
    reproduces `KeypointSource`'s blocks to 6e-6.
18. **The attacker's root stays at the world origin.** Slide values are world coordinates. `AttackCommand` asserts that the
    attacker's `init_state` is at the origin with identity rotation, and that no event is `reset_scene_to_default` or takes
    the attacker as a parameter. m1check also checks `root_link_pos_w` on the device after its reset.
19. **Gravity-compensation fallback runs on its own.** If m1check writes no JSON, `pod_m1m2.sh` reruns it once with
    `SPAR_GRAVCOMP=0` (no body gravcomp, z targets + m g / kp, 6.7 mm). That setting then stays exported for the smoke run
    and the trainings, and is recorded in the gate file (`gravcomp`). The progress file says `fallback gravcomp off`.
    `ALL_DONE` and `FAILED` are unchanged.
20. **Keypoint visibility: training now covers the CPU eval** (supersedes 2.4's `D ~ U(0.35, 0.55)` and `U(-30, 30) deg`;
    section 2 is left as written). Measured on the eval's own bouts: `eval_opponent` episodes,
    hard tier, val + test clips, 2 reps, unaimed (shadow) and aimed (`standoff+armwarp_vz`), DR on, warm start, 48 bouts x
    1000 steps, through a recording `KeypointSource` (scratchpad `m1m2/fix/measure_cpu.py`). The CPU opponent sits
    further away and more to the side than the spec's schedule placed it. Pelvis spacing is median 0.55 m (p10 0.42, p90
    0.74-0.81). The bearing from the defender's heading is median -15 deg (p10 -69, p90 +37): the opponent tends to be on
    the defender's right. The head is higher in T (median z 0.23 vs 0.17 m). The schedule now draws, per episode:
    - D ~ U(0.35, 0.65);
    - a bearing offset beta ~ U(-30, 0) deg, added to every bout-axis draw, with alpha_deg 50;
    - the virtual head's height offset `head_dz ~ U(0, 0.12)` m. The frame offset `base_*` (the whole opponent: gloves,
      elbows, head) exists but is 0.

    In-view rates (GPU: perfect fist tracking, the warm start's standing torso pose, 512 envs x 1500 steps, 2 seeds,
    spread <= 0.016; `m1m2/fix/vis_table.py`):

    | keypoint | CPU eval bouts | GPU before (D 0.35-0.55, alpha 30) | GPU after |
    |---|---|---|---|
    | l_glove | 0.429 | 0.698 | 0.489 |
    | r_glove | 0.571 | 0.393 | 0.609 |
    | l_elbow | 0.537 | 0.782 | 0.536 |
    | r_elbow | 0.689 | 0.791 | 0.822 |
    | head | 0.411 | 0.873 | 0.463 |

    Every keypoint is now within 0.15 of the CPU rate (largest gap: r_elbow, +0.13). Raising the whole base by 0-12 cm
    matched as well, but lifted the calibration row's median impulse per head hit from 3.50 to 4.07 N·s (64 bouts x
    20 s; the base height alone: 3.68), outside the gate. The head is virtual, so `head_dz` changes no physics. On the
    final schedule the row (1.0 kg, 150 N, +5 mm) gives:
    - HitMeter: 13.8 head clean / min, median 3.65 N·s;
    - HitAccount: 14.2 / min, 3.52 N·s;
    - survival 0.73.

    The same physics with different random streams gave 3.21 N·s, and the spec's own schedule gave 3.21 and 3.50 on two
    stream sets. The median moves about ±0.3 N·s between seed sets at 64 bouts, which is as wide as the margin to the 3.56
    limit; item 21 has the rerun on this schedule. Not checked: the rates under real servo tracking and with the defender
    moving (footwork, outer loops); the probe holds the defender still and facing.

**Second fix round (2026-10-28, after the second reviews; CPU-checked only).**

21. **Calibration on the final schedule: the grid does not choose the attacker in use.** `python gpu/sparring_task.py
    calibrate` (64 bouts x 20 s, seeds 0-63, after items 22-23), written to `results/sparring/m1_calibration.json`:

    | (mass, F_max, delta) | HitMeter median N·s per head hit | head clean / min [95% CI] | speed p98 / ref | HitAccount clean / min, median | survival | pass |
    |---|---|---|---|---|---|---|
    | (1.5, 150, +5 mm) | 4.42 | 14.6 [12.9, 16.4] | 0.99 | 15.0, 4.39 | 0.69 | no (impulse) |
    | (1.5, 150, -2 cm) | 4.04 | 10.7 [8.9, 12.5] | 0.99 | 10.8, 3.89 | 0.70 | no (impulse) |
    | (1.5, 80, +5 mm) | 3.59 | 14.3 [12.8, 15.9] | 0.89 | 14.6, 3.51 | 0.84 | no (impulse) |
    | **(1.0, 150, +5 mm)** = `ATTACK` | **3.65** | 13.8 [12.1, 15.6] | 1.05 | 14.2, 3.52 | 0.73 | **no (impulse)** |
    | (1.5, 80, -2 cm) | 3.52 | 11.0 [9.2, 13.0] | 0.90 | 11.2, 3.42 | 0.88 | yes: `chosen` |
    | `ATTACK` again, seeds 64-127 (`attack_recheck`) | 3.95 | 14.4 [12.6, 16.4] | 1.03 | 15.2, 3.78 | 0.77 | no (impulse) |

    HitMeter's numbers are the same as before items 22-23 (they only change HitAccount); the HitAccount columns moved by
    at most 0.17 / min and 0.07 N·s, and every row still agrees within the 15% / 10% checks. `constants_match` is false:
    the 1.0 kg row fails the impulse limit on both seed sets (3.65 and 3.95 > 3.56), and the chosen row clears it by only
    0.04. `ATTACK` is unchanged (item 11 is a decision, not a calibration result). So the preregistered "CPU calibration
    passes first" is enforced on the pod (item 24): with this file the pod stops at `FAILED calibration_mismatch` unless
    `SPAR_CALIB_DEVIATION="<reason>"` is exported, which runs it and records the deviation.
22. **Equal onsets are hits, as in HitMeter.** `HitAccount._pending` kept a window open only for head / torso episodes that
    opened strictly before the arm episode; one opening on the same substep was ignored, so a window could be decided
    'blocked' (and `spar_block` paid) before that head episode landed, although HitMeter (`arm.t0 < hit.t0`) calls it a hit.
    Now `t0 <= w_arm`. Review repro `rev2_hits/r1_simul.py`: blocked 0, spar_block 0, head clean 1 (was 1, 1, 1);
    `tests/test_sparring_task.py::test_hitaccount_equal_onset_is_a_hit`.
23. **Float32 episode clock.** Substep times are float32 (`episode_length_buf.float() * step_dt`), off by about 1e-6 s at
    20 s, and the merge / duration tests had a 1e-9 slack, so an exact 0.1 s gap split an episode and an exact 0.15 s
    contact failed the clean rule about half the time late in an episode. Both now use `T_TOL = 1e-4` (well above float32
    rounding, well below a 5 ms substep). Window edges are not on the substep grid and HitMeter compares them rounded to 4
    decimals, so window membership and "window is past" use `W_TOL = 5e-5`, which is exactly that rounding. On the
    review's adversarial fuzz (`rev2_hits/r3_fuzz.py`: arm and head bursts 0-2 substeps apart near window ends, 64 envs x
    8 s), HitAccount now equals HitMeter on windows, blocked, head landed / clean and torso landed in 64 of 64 envs on each
    of 16 seeds (before: 42 and 34 envs differed on seeds 3 and 11).
    `test_hitaccount_float32_clock_late_in_an_episode` checks a contact after 12 s.
24. **The pod checks the calibration before training** (`calcheck`, `calibration_check`). Right after setup, `pod_m1m2.sh`
    runs `python sparring_task.py calcheck /root/m1_calibration.json`. It passes only when the file's `chosen` row is
    `ATTACK` and `constants_match` is true, or when `SPAR_CALIB_DEVIATION` holds a reason. Otherwise the pod writes
    `FAILED calibration_mismatch` and packs. m1check records the same check in `m1_gate.json` (`calibration`, with the
    file's sha256, the `ATTACK` row and its recheck; part of `pass`), and a declared deviation goes into
    `m1_gate.json["deviations"]`. The calibration file is now an upload, checked at the start and packed.
25. **Gate 2's 3072-env fallback is a deviation.** The preregistration says <= 2.0 s per iteration at 4096 envs. A speed
    pass at 3072 is now listed in `m1_gate.json["deviations"]`, and the pod's line becomes `M1_GATE_PASSED_DEVIATION ...`
    (also used for a declared calibration deviation), to be reported as such in the M2 write-up. The gate also logs the
    20th iteration's time and the mean `Collection time`, so a miss from Python launch overhead (env stepping) can be told
    from one in learning. `pack` now includes `/root/progress.txt`. Launch overhead: the per-step constant tensors
    (camera centre and axes, aim points, the defender head point, the slide masses) are built once per device instead of
    by a host-to-device copy every step. The S = 4 substep loop and the host syncs are unchanged; whether the overhead
    fits in 2.0 s / iteration is not known until the smoke run.
26. **`spar_block` follows the eval's window on purpose.** It pays any blocked window, including a forearm touch on the
    retracting glove after the strike missed (`rev2_hits/r5_tap_after_miss.py`; HitMeter also calls that window
    'blocked'). At most one payout per window, nothing for idle gloves, nothing for contact from before the window. A
    reward for blocks before the apex only would make the training signal differ from the eval's blocked share. Left
    as is, for the spec owner to change.
27. **The verdict checks the whole hard tier.** `scripts/m2_verdict.py` `protocol()` now also requires 20 s bouts, the hard
    tier's `tier_cfg` (distances 0.4-0.55, DR on, seed 20000, pinned in the script), and the tracker opponent
    (`opponent` 'tracked', `opponent_mode` 'policy'). The review's `rev2_contract/adv_protocol.py` run (10 s bouts, DR
    off, a puppet opponent) now exits instead of returning a verdict.

**Attacker row, final (2026-10-28, lead decision).**

28. **`ATTACK` follows the grid: (1.5 kg, 80 N, -2 cm).** Item 11's choice to keep 1.0 kg assumed it passed the calibration;
    on the final schedule it fails the impulse limit on both seed sets (item 21), and the preregistration requires the CPU
    calibration to pass before training. So `ATTACK` is set to the row the 2.5 rule picks, and no deviation is declared.
    It is the softer attacker: head clean 11.0 / min (still 1.4x the aimed CPU tracker's 7.72) and warm-start survival 0.88
    (0.73 for 1.0 kg), so training is closer to the eval opponent. Its margin to 3.56 N*s is thin (3.52): the rerun of
    `calibrate` with this `ATTACK` writes its second seed set as `attack_recheck` (reported, never gating), and the GPU gate
    1 re-measures the median on the pod.

**Gate 1 on the GPU (2026-10-29, pod 1; written before pod 2 and before any M2 training).**

29. **The CPU-calibrated attacker fails gate 1 on the GPU, so a GPU retune grid is fixed here.** Pod 1 (`m1check`, 4096
    envs x 1000 steps, mesh proxies, gravcomp on, `ATTACK` = (1.5 kg, 80 N, -2 cm)): head clean 11.81 / min (passes
    >= 7.7), median impulse per head landed hit **3.945 N·s (fails [1.58, 3.56])**, mean 4.90, torso clean 9.21 / min,
    blocked share 0.624, signed-closing-speed < 0 share 0.008, 0.31 falls / min. The CPU replica gave 3.52 (3.23 on the
    second seed set) for the same row, so MuJoCo Warp's contacts land about 12% harder than CPU MuJoCo's. 2.6 has no
    gate-1 fallback; this adds one, before any M2 training and without looking at any M2 outcome:
    - `pod_m1m2.sh` runs gate 1 with `ATTACK`; if it fails, it tries these rows in this order, `SPAR_ATTACK="mass,fmax,delta"`:
      (1.5, 70, -2 cm), (1.5, 60, -2 cm), (1.0, 80, -2 cm), (1.0, 60, -2 cm), (1.5, 50, -2 cm). Order: CPU calibration put
      the force cap and the mass as the strong levers (4.42 -> 3.59 N·s for 150 -> 80 N; 4.42 -> 3.65 for 1.5 -> 1.0 kg) and
      the aim depth as weak (3.59 -> 3.52 for +5 mm -> -2 cm), so the grid steps the cap down first, then the mass.
    - The first row that passes gate 1 (both criteria, unchanged) is the attacker for gate 2 and all three trainings; its
      values stay exported. If none passes: `FAILED m1_gate_grid_exhausted`, nothing is trained.
    - A retuned row is a deviation from "CPU calibration passes first": the pod sets `SPAR_CALIB_DEVIATION`, so
      `m1_gate.json` lists it and the run ends `M1_GATE_PASSED_DEVIATION`. Every row tried keeps its own
      `m1_gate_<proxy>_<row>.json` and log in the download. The M2 write-up reports the row and this deviation.
    - The pass rule of the preregistration is unchanged: the CPU eval opponent is the same, only the training attacker
      changes.

---

## 9. Run 2 (2026-10-29; preregistered in `results/sparring/prereg_m2_run2.json` before any run-2 code ran on GPU)

Run 1 failed (`results/sparring/M2_RUN1.md`): both sighted defenders learned to back out of reach. Run 2 keeps run 1's pass
rule and its substance, trains new task IDs (`R1-Spar-Block2-A`, `R1-Spar-Block2-Blind`), and is scored on re-seeded episodes of the same clips.
The run-1 tasks, `SCHED`, `HIT`, `WEIGHTS` A / B and every default are unchanged bit for bit, so run 1 stays reproducible
(`tests/test_sparring_r2.py` drives `AttackSchedule` and `HitAccount` against a frozen copy of the run-1 file,
`tests/fixtures/sparring_task_576727a.py`).

### 9.1 Diagnosis (run 1's val runs, training logs and code)

**Val, hard tier, 128 bouts per controller.** Columns are survival, in-ring (up and within 1 m of the start) and head
clean strikes per minute.

| | guard_fw_r | A | B |
|---|---|---|---|
| unaimed | 0.977 / 0.953 / 5.48 (blocked share 0.264) | 0.984 / 0.00 / 0.62 (0.06); drift 5.3 m, back-off 4.0 m | 0.945 / 0.00 / 0.71 (0.05) |
| aimed | 0.969 / 0.930 / about 7.3 | 0.797 / 0.0 / 0.72 | 0.664 / 0.0 / 1.19 |

The test summary, committed in 576727a and seen before this section was written, gave A 0.750 / 0.0.

**Confirmed in the code (`gpu/sparring_task.py` at 576727a):**
- The spot's chase is capped at 0.3 m/s (`SCHED.spot_vmax`). The CPU tracker uses the same cap (`STEER_MAX`), so an
  escape learned on GPU transfers to the eval.
- The aim correction latches at the strike's onset (`latched |= act & (tau >= on)`), while the CPU aimed opponent commits
  at apex - 0.15 s (`r1gym/aim.py commit_s`).
- `_block` used `outer=0.5`: half the envs kept the footwork walking commands (mean 0.47 m/s, faster than the chase).
- No term refers to the ring.
- `clearance()` pays `clip((d - 0.10) / 0.15)` whatever the reach, so a strike that could never reach pays in full.

**Confirmed in the logs** (`train_A.log`, A's last iteration, 8995 of 8996):
- Mean reward was -17.23 over 844-step episodes, about -1.09 per live second. `action_rate_l2` alone was -1.38.
- Tracking rewards were tiny (0.18 / 0.23) and `error_vel_xy` was 2.1, so the hold command alone did not anchor the policy.
- `is_terminated` is -200 x 0.02 = -4 (the Unitree config).
- `stand_still` (-0.112) cost more than `spar_block` paid (+0.040). It acts on every joint while |command| <= 0.1, and the
  default pose is the guard (`_guard` sets `init_state`), so it taxed every block.

### 9.2 What run 2 changes (`_block2`; overrides reach only the run-2 tasks)

1. **Every env on the outer loops** (`RING["outer"]` = 1.0): face the spot, hold the start.
2. **Ring anchor.** The in-ring bonus `spar_ring` pays weight 2.5 x (1 - clip((r - 0.25) / 0.75, 0, 1)) per second,
   where r is the pelvis xy distance from the episode's start (`OuterLoopTwistCommand.spot`). It is 1 within the hold
   loop's 0.25 m and 0 at 1 m, where the `ring_out` termination fires (training config only, not play). `ring_out` is
   not a time-out, so the -4 applies as for a fall.
   - It is a bonus and not a penalty because, with the net rate negative, a penalty near the edge (about -2.6 per second
     for a -2 x clip(r - 0.25) design) makes continuing worth about -5.2 against -4 for leaving, so leaving would pay.
   - The critic gets `ring_home`: the start minus the pelvis in the heading frame, 2 inputs appended last, for 138 critic
     inputs. The actor stays at 111.
3. **Aim commit** (`SCHED_R2["commit_s"]` = 0.15). Delta follows the defender until max(onset, apex - 0.15 s), the CPU
   aimed opponent's commit, then is world-fixed. A slip that starts within 0.15 s of the apex still makes a strike miss.
   The spot cap stays 0.3 m/s. A faster chase or a velocity lead would be a train / eval mismatch that gate 1 does not
   check, and with the defender held in the ring it changed nothing in design 2's kinematic check (within 0.012).
4. **Clearance only in reach** (`SCHED_R2["reach_gate"]`, `SCHED["reach_tol"]` = 0.05 m). At the latch the schedule
   records whether the pelvis was at most 0.05 m beyond the stand-off D from the live spot (`reach_ok`). `spar_clearance`
   pays only those strikes; before the latch it uses the live excess (`reach_mask`).
5. **Block pay for intercepts only** (`HIT_R2`). `spar_block` (weight 40, twice run 1's A) pays an arm-first window only if
   the arm contact opened by apex + 0.05 s and no head or torso hit landed in that window after it: a graze that does not
   stop the strike pays nothing (the exploit review found that otherwise touching every glove paid about +37 per minute even
   while more head hits landed). HitAccount's `blocked` and `block_first` per episode and in total keep run 1's definition
   (arm first, landed or not), so the eval's blocked share is unchanged; the "blocks" claim reads a stopped share instead.
6. **`stand_still` on legs and waist only** (`STAND_STILL_JOINTS`: 12 leg + 2 waist joints). The arms keep the pose
   term's std of 0.35.
7. **Metrics, logged and never gated:** `spar_ring_r` (episode mean of r), `ring_max_m` (the twist command's per-episode
   maximum r), `spar_arm_idle_touch` (an idle fist touching an arm proxy: pinning).
8. **Tooling:**
   - `widen2` widens the critic to 138 (the ring_home columns have mean 0, std 0.5 and zero weights).
   - `m1check --r2 1` runs gate 1 with `SCHED_R2`.
   - `gate --last 200 --max-iter-s 2.2 --prereg-envs 3072` reads gate 2 from A2's own log.

Dropped from run 1: B (doubling the weights destabilised run 1; falls rose about 4x). A second sighted seed (A2s, seed
43) takes its place. The attacker row (1.0 kg, 60 N, -2 cm), capsule proxies and 3072 envs carry over from run 1's gates
(section 8 item 29, `results/sparring/m1_gate/`). They are declared again as a deviation from "CPU calibration passes
first" through `SPAR_CALIB_DEVIATION`.

### 9.3 Gates, training, selection

- **Gate 1:** `m1check --r2 1`, capsule proxies, 3072 envs, the frozen warm start. It needs head clean >= 7.7 / min and a
  median impulse per landed head hit in [1.58, 3.56] N·s. Rows are tried in order (1.0, 60, -2 cm), (1.0, 50, -2 cm),
  (1.0, 40, -2 cm), and the first that passes is used. If none passes, nothing is trained. The margin is thin: run 1's
  capsule gate gave 3.44 N·s at 4096 envs with the onset latch.
- **Gate 2:** A2's own log, iterations 20-200, <= 2.2 s per iteration, and no NaN termination. If it fails, the run stops
  after A2 with no selection. There is no separate smoke run for the gate; a 3-iteration load smoke of both tasks only
  checks that they build and log the ring terms.
- **Training:** A2 (seed 42, run 1 A's), then blind2 (seed 42), then A2s (seed 43). Each runs 1500 iterations from
  `widen2(model_7496.pt)` and the final checkpoint is the candidate. A2s is skipped only if blind2 ends more than 2.4 h after A2 started (budget).
- **Eval:** `scripts/eval_opponent.py --tier hard_r2 --no_pushes --hits 1 --keypoints truth`, unaimed then `--aim chosen`.
  - Val: `--reps 32`, seeds 21000-21127.
  - Test: `--reps 16`, seeds 22000-22127. Run once, with guard_fw_r, the selected candidate and blind2 in one invocation.
  - Output goes to `results/eval_opponent/m2r2`.
- **Screen** (`scripts/m2_verdict.py --run 2 select`, val, aimed):
  - A candidate qualifies with survival >= 0.95, a ring drop vs guard_fw_r <= 0.05 and head-clean R_warm (point) <= 0.80.
  - Of the qualifiers the lower head clean/min wins; a difference under 0.1/min picks A2.
  - If none qualifies, `go` is false and the test is not run.
  - The output is `results/sparring/m2_run2_selection.json`, with `go` and the prereg's LF-normalised hash.
- **Verdict** (`--run 2 verdict`): run 1's five-part rule, unchanged, against blind2. blind2 is seed-matched for A2 only;
  for A2s this is reported. The verdict refuses a selection with `go` false or one made against another prereg. The output
  is `results/sparring/m2_run2_verdict.json`.
- **"Blocks" claim (not part of the pass):** the head-intent blocked share must rise by >= 0.10 vs guard_fw_r (paired
  cluster CI > 0), and head slip (closest approach) must stay within 3 cm of guard_fw_r's. Otherwise the write-up says
  "avoids".

### 9.4 Keeping the test set honest

- **Re-seeded:** run-2 episodes replay the same clips with new seeds (new start times, delays, DR and hit draws). Seeds, env seeds and hit seeds are
  disjoint from run 1's (20000-20127) and between the splits; `tests/test_m2_run2.py` checks this. The v1 tiers' episodes
  are unchanged.
- **Reused:** the clips. No clean alternative exists: kfab_199 cannot be performed by the tracker, and the train clips
  feed the GPU attacker. 66% of the new test start times (87% on val) fall within 0.1 s of a run-1 start.
- **Disclosed:**
  - run 1's test summary was committed and seen (survival 0.750, ring 0.0) before run 2 was designed;
  - run 2's design came from run 1's val runs and training logs;
  - run 1's val episodes (seed 20000) are spent.
- **Controls:**
  - The prereg is committed and hashed before the pod launches.
  - `eval_opponent.py` refuses `--tier hard_r2 --split test` unless `m2_run2_selection.json` says `go` against the
    current prereg hash.
  - The test runs once, and no threshold changes.

### 9.5 Exploits still open, and risks

- **Most likely: a late lean or slip.** GPU strikes aim 2 cm short, while the CPU opponent aims 3 cm deep and commits at
  the same moment. A policy that learns a lean will probably fail rule 1 honestly rather than game it.
  - The exploit screen (reported; the run-1 diagnosis's per-step time series, not committed, with pelvis z, torso pitch
    and hand-distance columns added) checks head
    displacement away from the opponent between t_on and t_apex, intent_closest_cm against the blocked share, ring
    creep, crouching, turning away and pinning.
- **Critic offset:** the warm-start critic is off by about +5 because of the bonus. `fell_over` is watched over the
  first 50 iterations; this is reported, not gated.
- **Other risks:**
  - Pushes may cause some ring-outs; they are logged.
  - Blocks learned against two floating fists may not transfer to the CPU humanoid.
