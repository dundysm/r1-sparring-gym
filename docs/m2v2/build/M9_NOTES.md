# M9 build notes: mjlab task, pod script, WATCH6, A9 wiring, and the A10 hull-name fix (2026-10-30)

M9 is built and its frozen CPU checks (1)-(4) pass. The A10 fix is in, its blocking zone-map test passes, and the zone-map identity is
part of the M9 model diff. The REQ_CONF smoke shows head hits are back at full-hull rates. Full suite: 434 passed, 3 skipped, exit 0.
Nothing was committed. CPU only; every process used at most 2 workers. No threshold or margin was changed.

I continued from a stopped builder's partial work (it had passed (1)-(4) at 18:17, before A10). I reviewed it, kept it, and made the
changes listed below.

## 1. Files

| file | state | what |
|---|---|---|
| `gpu/fullbody_task.py` | new (builder) + edits | the mjlab 1.2.0 task (2.1-2.5): scene with `robot/` + `opponent/`, CollisionCfg, actuators, `spec_fn` (`entity_spec`, `scene_edit`: shells' bits, OPP solref/solimp, floor conaffinity 3), fb contact sensors, FbOppCommand (FbMotion + PolicyMux T0/T1 + fb_mirror conjugation + SwapAim p 0.2, OppDown, account, FbRewards; placement only from command reset, resampling (1e9, 1e9)), FbTwistCommand (eval outer loops), FbAction (HybridDelayedAction), FbActorObs (EvalActorObs, one compute per step; critic reads `.clean`), reset events in order `fb_draw_latency` (first, min_step_count 0), `fb_reset_defender`, `fb_draw_population`, `fb_dr` (DRIndex with explicit `is_defender`/`prefix`, then the t1_dr rows' OppDR), `fb_update` the first termination term (OppDown.update on the stale step quantities, account, rewards), critic PRIV block, WATCH6 counters, FbActorModel factory (A9.2 blind flag), `start_checkpoint`, FbRunner, task registration. **My edits:** A10 hull naming in `entity_spec`; an assert that the `actor` group precedes `critic` (M4_VERIFY 4.1) |
| `gpu/fb_population.py` | new (builder) | FbMotion (originals + L1 / LOCO rows, conj, families), Population (40/10/35/15, t0_only for iterations 1-100), PolicyMux, `driver()` (T0/T1 mux conjugated per env by fb_mirror), SwapAim (design 4.2 p 0.2 mirrored-strike swap on jitter rows only; inert on T0 rows), OppDR |
| `gpu/r1_track_spar_t1.py` | new (builder) | A9.3: T1 trains on the reangle family only (sampling bins outside it set to 0) |
| `gpu/pod_fb.sh` | new (builder) + edit | P0-P7, taskset plan (train 0..N/2-1, bg N/2..N-1), uploads list, `md5sum -c`, versions, `assert_sim` (sim options, njmax/nconmax, tf32 off), M9 model diff rerun with `FB_POD=1`, start checkpoints, WATCH6 loop, FB_DRY / FB_FAKE_GPU. **My edit:** a local `md5 <file>` command that writes the uploads' md5 list P0 checks (fails while `prereg_m2_fb.json` does not exist yet, M11), and no `mkdir $FB_ROOT` for that command |
| `scripts/fb_watch.py` | new (builder) | `watch --rule 6` (E0-E8 as the plan's WATCH6 table) and `replays` (M9 (3)) |
| `scripts/fb_m9_check.py` | new (builder) + edits | the M9 checks. **My edits:** `zone_maps()` (A10.3) added to every model-diff run, a negative control with the pre-A10 names, `_mesh` no longer strips `fbhull_` (a hull-name regression now shows as a geom-key mismatch), `--logs` defaults to run 3's logs (`scratchpad\m1m2\dl5`, as tests/test_fb_watch.py) |
| `scripts/fb_replica.py` | edit (builder: A9.4 motion 'l1' wiring; me: A10) | **A10:** `build_model` renames each replaced shell mesh to `fborig_<mesh>` and adds the hull under the original name. Builder: `opp_modules` / `opp_clip` (fb_mirror mask from the npz `conj` at reset, the npz reference fed, SwapAim) behind conf key `motion='l1'`; default `motion='fb'` is M6's modules unchanged |
| `gpu/fb_dr.py` | edit (builder) | DRIndex finds actuators named `<prefix><joint>` (mjlab) as well as the eval's `<prefix><joint minus _joint>` |
| `gpu/hybrid_actor.py` | edit (builder) | A9.2: `blind_flag` buffer in every state dict; `stored_blind(sd)` |
| `scripts/export_hybrid_fb.py` | edit (M7T9 + builder) | `build()` asserts the stored blind flag equals `--blind`; `ckpt_blind(ckpt)` for selection/test |
| `tests/test_fb_zonemap.py` | **new (me)** | A10.3 blocking test, 9 tests (below) |
| `tests/test_fb_task.py` | new (builder) + edit | 13 tests; spec-edit test updated for A10 (hull meshes carry the original names, `fborig_<mesh>` present) |
| `tests/test_fb_replica.py` | edit (me) | reduced-hull test reads the hull mesh by its original name |
| `tests/test_fb_watch.py` | new (builder) | 4 tests |

Line endings: every touched file is LF.

## 2. A10 fix (first, as required)

- **Cause, confirmed:** the old `fbhull_<mesh>` name makes `r1_zone("torso_link", "fbhull_head_pitch_link")` and
  `fb_hits.zone_of` return "torso", so head shells scored as torso (test `test_negative_control_the_old_hull_names_lose_the_head`).
- **Fix:** in both `fb_replica.build_model` and `fullbody_task.entity_spec`, the original mesh becomes `fborig_<mesh>`, the hull mesh takes
  `<mesh>`, and the shell geom points at it. MjSpec does not rewrite `geom.meshname` on a mesh rename (probed), and no non-shell geom
  references a shell mesh (64 mesh geoms, 64 shells, none shared), so the rename touches nothing else.
- **tests/test_fb_zonemap.py (blocking), 9 passed:**
  - for hulls mix and all256 x reverse False/True: the HitMeter's `zone_of` / `limb_of` / `keys`, MjContactSensors' primaries, `zone_idx`,
    `block_idx`, block-shell set, body subset (`bprim`, `bzone_idx`) and limb map equal the full-hull model's, keyed geom for geom by
    (body, ordinal); 32 shells, 2 head shells, 4 block shells (both elbows and wrist_roll links);
  - every geom's mesh name equals the full model's, and every original is present as `fborig_<mesh>`;
  - the task's `entity_spec` for full / mix / all256: shells keep the original mesh names in r1.xml order, `<mesh>_shell` geom names,
    and the same zones on the compiled spec (2 head shells, no None);
  - the negative control above.

## 3. Checks (frozen M9 row + A10.3)

Command, in the repo: `python -I scripts/fb_m9_check.py --out build\m9\m9_check.json` -> exit 0, 32.8 s:
`{"model": true, "inputs": true, "watch": true, "pod": true, "ok": true}`. Log `build\m9\m9_check.log`. (The builder's 18:17 json there was
overwritten; it predates A10.)

| check | result |
|---|---|
| (1) model | mjlab 1.2.0's real `Scene(scene_cfg)` compiled on CPU with warp, mujoco_warp, mediapy, scipy, tyro stubbed (plus `prettytable`, stubbed automatically); mjlab_x = mjlab-1.2.0 wheel on all 187 .py/.xml files. Diffed by name against `fb_replica`'s model for hulls mix / all256 / full at 10/20, and mix at 25/30, 50/50, 100/50: **0 mismatches in all 6 runs** over 127 geoms (64 mesh, hull vertex sets as row sets), 50 bodies (order per entity), 50 joints, 48 actuators, opt (timestep 0.005, implicitfast, pyramidal, impratio 1, **ccd_iterations 35**, iterations/ls as set). Inert, reported: 48 ctrlrange (ctrllimited 0 in both) and the plane's size[2]. **Floor contype 1 / conaffinity 3** on mjlab's compiled model; without `scene_edit` 1/1 (C1 reproduced), and the diff then has 257 mismatches (not vacuous). **Zone maps (A10.3): 0 mismatches**, both entities, every run: task `shell_tables` on mjlab = replica account sensors = full-hull replica (mesh name, zone, block, body subset, body zone), 32 shells, 2 head, 4 block each. **Negative control:** a replica with the pre-A10 names differs on all 32 shells per entity. DR on mjlab's model: a defender write changes defender rows only (25 bodies, 30 dofs, 24 actuators, 14 feet = the whole `robot/` entity), OppDR opponent rows only. Sensors: 17 fb sensors, primaries = the 32 shells in geom order, fb_body = 20 BODY_ZONES shells |
| (2) inputs | the 8 motion/hull files equal their M1/M8 manifests' sha256; no unlisted motion input; `manifest_ok` for L1 and LOCO; banned-name scan clean over 50 task inputs |
| (3) watch | 23 synthetic stop logs, each stops for its own rule at its deadline (E0 x10, E2, E3 x2, E4, E5, E6 x5, E7 x2, E8); healthy A (17 checks) and B (41 checks) never stop; run 3's spar3_A3s / A3 / blind3 logs (1500 blocks each, 16 checks): 0 E0 trips, 0 stops (fb counters absent there: reported n/a); blind twin never stopped (1 reason reported) |
| (4) pod | `bash -n` ok; asserts present (assert_sim, precision_ok, `md5sum -c`, versions incl. mujoco-warp, fb_m9_check rerun, `taskset -c`, fb_watch, FB_FAKE_GPU, --blind, start checkpoints); dry run with FB_FAKE_GPU=1 FB_DRY=1 FB_CPUS=16 prints every stage and `DRY OK`; PIN train 0-7, bg 8-15 (disjoint) |

**Env cfg probe (report, `build\m9a\probe_envcfg.py`):** `env_cfg` for A sighted and B blind builds on mjlab's source with the extra
third-party modules stubbed (trimesh, viser, wandb, rsl_rl, tensordict). Observation groups `[actor, critic]`; reset events `fb_draw_latency`,
`fb_reset_defender`, `fb_draw_population`, `fb_dr` (min_step_count 0); terminations `fb_update` first; B has `res_l2` and `fb_lean_away`
and drops `spar_clearance` (m3).

## 4. A10 replica smoke (REQ_CONF)

`build\m9a\smoke_req.py` -> `smoke_req.json`: 32 dev bouts 60200-60231 (role `hit_scoring`, an M6 block), 20 s, REQ_CONF (hulls mix,
solver 10/20, torch32, eo reset, account, vmode site) vs the same with hulls full, Batch K 32 rowwise, 2 workers, 4 jobs, about 82 s each.

| ctrl | hulls | up min | head_clean (acct / HM) | per up-min | head_landed | per up-min | torso_clean | kd | falls |
|---|---|---|---|---|---|---|---|---|---|
| G | mix | 10.67 | 36 / 37 | 3.38 | 71 | 6.66 | 96 | 6 | 0 |
| G | full | 10.44 | 34 / 33 | 3.26 | 75 | 7.18 | 123 | 5 | 1 |
| H | mix | 10.43 | 34 / 36 | 3.26 | 69 | 6.62 | 112 | 3 | 1 |
| H | full | 10.49 | 29 / 30 | 2.77 | 61 | 5.82 | 106 | 7 | 1 |

Head hits are > 0 under mix, at full-hull rates (before the fix every REQ_CONF row had 0). N = 32 is a smoke, not a gate; the rerun on
100000-112287 (A10.4) is the main session's.

## 5. Suite and housekeeping

- Full suite, frozen command: **434 passed, 3 skipped, exit 0, 900 s** (`build\m9\suite.log`). After the suite I added the actor-first
  assert in `env_cfg` (not exercised by any test); tests/test_fb_task.py + tests/test_fb_zonemap.py were re-run after it: 21 passed. The
  env_cfg probe ran with it.
- `fb_seeds.py --audit SEED_LEDGER.md --scan`: ok (20 roles disjoint, ledger consistent, 50 task inputs clean). Ledger row "M9" appended
  (60200-60231, role hit_scoring, re-run in role).
- MUJOCO_LOG.TXT (from the mjlab scene compile's empty-prefix warning) deleted; no `Users*` folder.

## 6. Notes for the main session (not M9 defects)

1. P0 on the pod reruns `fb_m9_check.py --only model,inputs` with FB_POD=1, which now includes the zone-map identity and the negative control.
2. `pod_fb.sh md5 <file>` must run after M11 writes `results/sparring/prereg_m2_fb.json`; until then it stops with "an upload is missing".
3. pod_fb.sh calls `scripts/fb_p1.py`, `fb_t1c.py`, `fb_loco_eval.py`, which are not written yet (later milestones; header says so).
4. The M7T9 stage-B checkpoints in `build\m7t9\ck_*_B` were made before `blind_flag`; t9b with `--ckpt` on them would now be refused by the
   A9.2 assert. Real M10 candidates carry the flag.
5. The mjlab env itself (managers, Warp) cannot be stepped on this CPU (mujoco_warp not installed); the step-order hazards are checked by
   construction and by P1's asserts on the pod.
