# M9 independent verification (M2V2_PLAN_v3 section 3 row M9, 2.1-2.5, section 4 P0/P1, WATCH6; A9, A10), 2026-10-30

**Verdict: ok = true, after fixes.** Every frozen M9 check (as amended by A9/A10) re-ran and passed, both on the builder's code and
after my fixes. Reading the code against the eval, I found two semantic mismatches in the task's hit and aim inputs (section 2, F1-F2)
and five plumbing bugs that would have broken the pod run (F3-F7). I fixed all of them, added one hardening fix (F8), and re-ran
everything. No threshold or margin was changed. Nothing was committed. CPU only; every process I started used at most 2 workers.
Open items for the main session are in section 4. None of them is an unfixed semantic mismatch.

## 1. Re-runs (frozen checks)

| check | result (mine) |
|---|---|
| `fb_m9_check.py --out build\verify_m9\m9_check.json` | exit 0, 31.8 s: model / inputs / watch / pod all true (the builder's code gave the same before my edits, 32.6 s) |
| (1) model | mjlab_x = the mjlab-1.2.0 wheel on all 187 .py/.xml files. The real `Scene(scene_cfg)` was compiled with warp, mujoco_warp, mediapy, scipy and tyro stubbed (prettytable stubbed automatically). Six runs (mix / all256 / full at 10/20; mix at 25/30, 50/50, 100/50): **0 mismatches** each. Each run covers 127 geoms (64 mesh, hull vertex sets), 50 bodies (order), 50 joints and 48 actuators. opt is 0.005 / implicitfast / pyramidal / impratio 1 / **ccd_iterations 35** / iterations as set. Floor 1/3 on mjlab's compiled model; without spec_fn it is 1/1 and the diff has 257 mismatches (not vacuous). Zone maps (A10.3): 0 mismatches on both entities in every run, with 2 head and 4 block shells. The pre-A10-name negative control differs on 32/32 shells per entity. The DR write touches defender rows only; OppDR touches opponent rows only. 17 sensors, primaries = shells in geom order |
| (1+) found_check (new, F1) | 512 found addresses (2 x 8 sensors x 32 shells): names, type, dim, intprm [found, netforce, 1], primary geom and secondary (body / XBODY subtree) all as expected. On 24 random overlapping two-robot states (mj_forward on mjlab's compiled model), sensordata at every address = the MuJoCo contact count for that shell and secondary: 0 mismatches, 271 contacts matched, 116 slots non-zero |
| (2) inputs | all 8 motion/hull files = their M1/M8 manifest sha256; `manifest_ok` for L1 and LOCO; no unlisted motion input; banned scan clean over 50 inputs |
| (3) WATCH6 | 23/23 stop logs stop for their own rule by the deadline. Healthy A (17 checks) and B (41) never stop. Run 3's spar3_A3s / A3 / blind3 (1500 blocks, 16 checks each): 0 E0 trips, 0 stops. The blind twin is never stopped (1 reason reported) |
| (4) pod | `bash -n` clean; required asserts present; the FB_FAKE_GPU=1 FB_DRY=1 dry run ends `DRY OK`; PIN train 0-7, bg 8-15, disjoint (re-run after my pod edits) |
| A10 blocking test | tests/test_fb_zonemap.py 9 passed (inside the suite and in a targeted run: test_fb_task + zonemap + watch, 26 passed) |
| A10 REQ_CONF smoke | re-run `build\verify_m9\smoke_found.py` (60200-60231, role hit_scoring, 2 workers). It reproduces the builder's rows exactly: G mix head clean/landed 36/71 (3.38 / 6.66 per up-min), H mix 34/69, G full 34/75. Head hits under mix are non-zero and at full-hull rates. (My H-full row is void: the found=None monkeypatch of an earlier job leaked into that worker. The builder's H full, 29/61, stands.) |
| full suite (frozen command) | builder's state: 434 passed, 3 skipped, exit 0 (970 s, `suite1.log`). After my fixes: see section 5 |
| seeds | `fb_seeds --audit --scan`: ok (20 roles disjoint, ledger consistent, 50 inputs clean). Ledger row "M9 verify" appended (60200-60231, hit_scoring, re-run in role) |

## 2. Findings and fixes

### Semantic mismatches (task vs the eval / the replica R-EQ judges)

**F1. The account's touch ignored `found` (fixed).**
- **What was wrong.** FbOppCommand.update called `acct.process(F, ...)` without `found`, so `zone_forces` took touch = |F| > 0.
  - The replica (G0-11, R-EQ) and the eval's HitMeter count every matched contact, including zero-force ones: the HitMeter logs f = 0 contacts and merges with f_on = 0.
  - M6a measured that 11% of found slots carry zero net force.
- **Effect, measured on the same 32 REQ_CONF bouts with the physics unchanged.** With found=None, head clean is +1 for both G and H, torso clean +4/+4 and torso landed +5/+4. These are 3-4% shifts in the rewarded counts, and the GPU path never had them validated.
- **Fix.** mjlab keeps no found history, so the opponent's per-substep hook records the hit and reverse sensors' found from `sim.data.sensordata` into rings. The sensordata addresses come from `fullbody_task.sensor_adr`.
  - The hook runs in Scene.update after mj_step and before the sensors' history update, so it records the same substep's data.
  - update() passes `found=` to both accounts. A rolled ring keeps the substep order oldest first, as the force history is flipped.
- **Check.** found_check is added to fb_m9_check (1).

**F2. The aim's touch gate and the face bearing read post-forward data where the eval reads the last mj_step (fixed).**
- **What the eval does.**
  - R1HitEnv.step runs `_face` after the substeps with no forward. `true_bearing` uses `d.xpos[opp pelvis]` of the last mj_step.
  - Tracker.act runs after `mj_kinematics`, which leaves `d.contact` from the last mj_step (r1gym.aim._gate_geoms).
  - Only after a reset (R1HitEnv.reset's mj_forward) are both fresh.
  - fb_replica's `_touching` / `_face` do the same.
- **What the task did.**
  - It read the opponent pelvis after mjlab's single forward: fresh, one substep later.
  - Its touch gate read `sensor.data.found`, which is the termination-phase cache (stale), or fresh whenever *any* env reset that step, because scene.reset invalidates every sensor's cache. With 4096 envs that is almost every step.
- **Fix.** fb_update (the termination phase, the eval's staleness) stores the touch from the found ring's last substep and the opponent pelvis xy. `_update_command` and FbTwistCommand use those values, and the fresh post-forward values only for envs placed this step (`placed`).

### Plumbing bugs that would have broken the pod (fixed)

| id | bug | fix |
|---|---|---|
| F3 | `pod_fb.sh`: the venv is created and activated only inside the install's `bash -c`, so every later `python` (versions, assert_sim, the M9 diff on the pod, start checkpoints, training, watch, and any later `bash pod_fb.sh P2` etc.) ran the system interpreter | `act()` sources `$W/venv/bin/activate` at script start and right after the install |
| F4 | P0 copied fullbody_task.py, fb_*.py and hybrid_actor.py into `$U/scripts` and ran that copy. There, ROOT = `$U`, so r1gym, `gpu/motions/fb`, the donors, export_hybrid_fb and export_fb_library are all missing. Training was also launched with cwd `$R`, so `import train` picks the repo's own `scripts/train.py` and `logs/` lands under `$R`, and `--agent.load-run start_sighted` (`$RUNS = $U/logs/...`) is not found | Run `$R/gpu/fullbody_task.py` with cwd `$U` (`cdu` before the P0 timing runs and inside `train()`, which always runs in a background subshell); only the T1 tracker files are copied. P0 now fails if no timing run completes (before, every failure was only logged) |
| F5 | `fullbody_task.py` run as `__main__` while rsl_rl imports `class_name 'fullbody_task:FbActorModel'` gives a second module. The actor registers into that module's `_ACTOR`, and the env's terms (`__main__`) never see it: res_l2 = 0 always, and the actor metrics are never logged. WATCH6 E0 arm_std / leg_dev / residual and E3 would be silently n/a | `sys.modules.setdefault("fullbody_task", sys.modules["__main__"])` before `register()` |
| F6 | `start_checkpoint` calls `torch.save` into `$RUNS/start_<twin>/`, which does not exist yet, so P0 fails `start_sighted` | `Path(out).parent.mkdir(parents=True, exist_ok=True)` |
| F7 | P2 trains T1 on `motions/fb_l1_track.npz` with its index, resuming `model_13749.pt` with no load-run. None of those exist on the pod: the CSV is uploaded but never converted (r1_track_spar_cfg's own `convert` step) | P0 stages T0 as `logs/rsl_rl/r1_tracking_spar/t0/model_13749.pt`, converts both CSVs and copies their index files. P2 passes `--agent.load-run t0` |

**F8 (hardening).** FbActorModel's `load_state_dict` now refuses a checkpoint whose stored `blind_flag` differs from the run's (`load_same_blind`). Without this, a cross-twin load would silently relabel the run, because the loaded buffer overwrites it. A test was added.

Files I touched: `gpu/fullbody_task.py`, `gpu/pod_fb.sh`, `scripts/fb_m9_check.py` (found_check) and `tests/test_fb_task.py` (+1 test). All are LF.

## 3. Line-by-line review, no other mismatch found

- **Step order vs mjlab 1.2.0's `step` / `_reset_idx` / `reset`.**
  - The tracker acts in `_update_command` after the single forward: the defender target written, then kinematics, then the tracker. Clock = t_ep + t0 + dt and aim `now` = t_ep + dt, as in fb_replica's `_opp`.
  - fb_update is the first termination term, so OppDown, fell_eval and the account all see the one-substep-stale state, as the replica does.
  - Account t_end = (step + 1) x dt.
  - Reset events run in order: latency first (min_step_count 0), defender, population, DR (which needs fb_pop, drawn just before).
  - Placement happens only from command reset (resampling 1e9; `command_counter` is 1 per episode). The first-ever compute(dt = 0) is triggered by `undriven`.
  - The actor group is computed before the critic (asserted). `get_observations` returns the cache, so EvalActorObs advances once per step.
- **Frames.**
  - Root write: the angular qvel is rotated to world before mjlab's write_root_velocity rotates it back.
  - `root_link_ang_vel_b` = free-joint qvel[3:6].
  - Closing-speed points (site / body COM) match fb_replica's `_vel` (body COM + w x r).
  - Sensor force history index 0 is the newest, so it is flipped to oldest first.
- **Contact sensors.** mjlab's `_add_contact_sensor_to_spec` builds exactly the replica's native mjSENS_CONTACT (GEOM primary; BODY or XBODY secondary; [bit, 3, 1]), and found_check now asserts this on the compiled model. The aim gate's geom sets (r1gym.aim._gate_geoms) equal the sensors' coverage: the only cross-robot pairs that collide are shell vs shell, and R1's strike-limb bodies are wrist_roll, elbow, the ankle_pitch subtree and knee.
- **DR.** `fb_dr.model_fields` are declared, so mjlab expands them per world. fb_dr.write restores every field to nominal for the envs before the defender's draws and the t1_dr rows' OppDR, so a previous episode's opponent DR never persists.
- **Population and WATCH6.**
  - Shares, t0_only, T1 = reangle only (A9.3), conj mask and SwapAim on jitter rows only (A9.4) are as tested.
  - WATCH6 counters: sums per reset batch, so window ratios are ratios of totals. The E0 T0 share expects 0.50 = 0.40 + 0.10.
- **Blind critic.** Run 3's blind twin's critic also had `opp_keypoints_truth` (sparring_task._block), so FbActorObs.clean keeping the true block for the critic matches run 3.
- **A10.** fb_replica.build_model and entity_spec now give each hull mesh its original mesh name and rename the original to `fborig_<mesh>`. MjSpec does not rewrite `geom.meshname` on a rename, and no non-shell geom references a shell mesh. The zone-map identity holds for full, mix and all256 on both entities (test + M9 diff).

## 4. Open items for the main session (not decided here)

1. **mujoco version on the GPU pod.** P0 installs `mujoco==3.17.0` over unitree_rl_mjlab's `mujoco-warp==3.5.0` pin; run 3's pod ran mujoco 3.5.0. Whether mujoco_warp 3.5.0 works with mujoco 3.17.0 is unverified. P0 fails loudly if it does not (install, assert_sim, the M9 diff with FB_POD=1). The options are:
   - (a) keep 3.17.0 if P0 passes;
   - (b) mujoco 3.5.0 with the M9 diff rerun on the pod, as the plan already foresees ("records mujoco / mujoco_warp versions vs local 3.17.0").
2. **"Iterations 1-100 of every run".** The task applies T0-only per process: A0, then A's first 100 and B's first 100 iterations, as each restart's WATCH6 E2/E5 baseline. If "run" means the A0 stage only, A's and B's first 100 iterations train on T0 rows only, about 100 + 100 of 5600 iterations. The plan's wording allows both readings; this is a choice, not a code defect.
3. **The mjlab glue cannot be stepped on this machine** (mujoco_warp is not installed). FbOppCommand / FbTwistCommand / FbActorObs, including my F1/F2 edits, are verified by review and by found_check's address semantics only. Their first execution is P0's 20-iteration smokes and P1. P1's B3 asserts and P1-Hit (GPU account vs CPU HitMeter on dumped contacts) are where F1/F2 get exercised.
4. **Builder notes that stand.**
   - `build\m7t9\ck_*_B` lack `blind_flag` and are refused by `build(--ckpt)`.
   - `pod_fb.sh md5` needs M11's prereg json.
   - fb_p1.py, fb_t1c.py and fb_loco_eval.py are later milestones.
   - The push interval event writes the root velocity after the command compute, so the T1 rows' obs sees it one step late (training only).
5. **Critic warm start.** The 10 intent features occupy run 3's `strike_phase` columns, with different semantics. That is critic only, and A0 trains the critic for 100 iterations.
6. **The A10.4 rerun on 100000-112287** and the R2 hulls-mix step are the main session's. The 32-bout smoke is not a gate.

## 5. Suite and housekeeping

- Full suite with the frozen command after my fixes: **435 passed, 3 skipped, exit 0** (905 s; +1 = the new blind-lineage test) (`build\verify_m9\suite2.log`).
- MUJOCO_LOG.TXT in the repo (from the scene compile's empty-prefix warning) deleted; no `Users*` folder.
- Outputs: `build\verify_m9\` holds m9_check.json/log, m9_model_found.json, smoke_found.py/json/log, suite1.log (builder's state), suite2.log, and edit_pod.py / edit_task.py (the scripted edits).
