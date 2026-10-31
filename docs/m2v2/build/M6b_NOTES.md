# M6b build notes: Amendment A7 (batched replica), A5.1, A6.5, A6.6, 2026-10-30

**Status.** Every M6b item is done. No threshold or margin was changed, and nothing that belongs to the main session was decided.

| item | rule | result |
|---|---|---|
| A7.1 speed | profile first, then batch K bouts in lockstep; pick K by measurement; target <= 2x EO | **K = 64, row-wise variant: 1.17x / 1.16x / 1.03x EO** per 20-s bout per worker (G / H / A3). EO was timed in the same worker, before and after the replica, under the same load. Before A7 the replica ran at 16-18x EO. |
| A7.2 batch invariance | rows at K vs N = 1 on 64 bouts per controller; report every bitwise difference and its source | **Deployed configuration ('rowwise', K = 64): 64/64 rows and 64/64 per-step traces bit identical to N = 1 for G, H and A3.** The plain batched modules ('batch') differ on 64/64 bouts from step 1. There are two sources, both named by same-state probes (section 2). |
| fb_m6_run uses the batched configuration | | `BATCH, ROWWISE = 64, True` for the torch backends. The batch size goes into the chunk meta and the manifest. The numpy (eval) conf still runs at N = 1. A dry `req` run works end to end. |
| A5.1 / A7.3 | W_eps = 10 x max \|W_f32 - W_f64\| over the 144 w0_r4pre bouts; no charge for an edge in the first 0.5 s; scripted G0-12 cases; fb_m5_check re-run in full | **W_EPS = 1.1609e-4 J, frozen.** Scripted `w_eps_floor` and `start_window` cases pass. `fb_m5_check` exits 0: G0-11 true, G0-12 true. **Charged knockdowns under the final rule: 41 of 144 on w0_r4pre** (63 under M5's rule; the start window removes 22 and W_eps removes 0). Per-block counts are in section 3. |
| A6.5 | R1's 1e-9 gates on the diagnostic configuration; rdq gated at 1e-9 x 50 | `fb_m6_check.py all` exits 0. Identity is 16/16 + 16/16 bit identical. On the diagnostic configuration qpos max is **2.0e-15**, 0/32 bouts over, and same-state items are all within gate (rdq **3.1e-9 <= 5e-8**). G0-11 passes: 0/64 mismatches for G and for H, clean agreement 0.9675 / 0.9744. |
| A6.6 | deployed closing-speed input 'site' | Unchanged (`DEPLOYED_VMODE = "site"`). G0-11 re-run passes with it. A4.3 probe 88.301%. |
| full suite | passes | **407 passed, 3 skipped, exit 0** (1217 s) |

Rules kept:
- I edited fb_* workstream files and added one new file, `scripts/fb_m6b_check.py`. No run 1-3 file was touched. `gpu/fb_opponent.py` and the M8 files were not edited; the `M gpu/fb_opponent.py` entry in git status is the M8 lane's change.
- No commit, push, stash or branch switch.
- CPU only. Each of my processes used at most 2 workers. I started no R-EQ or other long M6 stage.
- Seeds came only from roles `identity` (60100-60115), `hit_scoring` (60200-60263) and `w0_r4pre` (saved recordings, no simulation), all checked through `fb_seeds`. Three ledger rows were appended ("M6b"), and `fb_seeds --audit --scan` reports ok: 20 roles, ledger consistent, 45 task inputs clean.
- No MUJOCO_LOG.TXT and no `Users*` folder was left in the repo.
- LF files stayed LF. `gpu/fb_rewards.py` and `tests/test_fb_rewards.py` already had CRLF in the working copy and keep it.

## Files
| file | change |
|---|---|
| `scripts/fb_replica.py` | **`Batch`** (A7.1): K bouts in lockstep, each a `Bout` slot (new `slot=` argument) with its own MjData, rng, R1HitEnv outer loops, sensors, HitMeter, ContactTail and bookkeeping.<ul><li>Once per step on the batch: TrackerDriver, AimOp, EvalActorObs, HybridDelayedAction, OppDown, fell_eval and the account.</li><li>The controller acts on the live bouts' stacked obs, as EO.run does.</li><li>DR fields are written into the shared model before each bout's own MuJoCo calls (EO's `_dr_apply`), and one kinematics twin serves the keypoints.</li><li>A finished bout stays frozen; its account is closed and summarised at its end.</li><li>`rowwise` runs the two batch-variant ops one row at a time: the tracker MLP, and `rowwise_face` (TrackerDriver.face with its float32 atan2 per row).</li><li>Torch inference mode; per-phase wall times in `tm`; a `probe(i, slot, action)` hook.</li></ul>`run(..., batch=, rowwise=)` caches one Batch per process.<br>Refactor: `run_bout` = `_Book` (EO.run's bookkeeping) + `_finish` (row and info), shared by both paths, with arithmetic unchanged; identity still passes 32/32.<br>**`Sensors._read`**: MjContactSensors.read's loop, computing the strongest-contact velocities only when vmode 'com' / 'point' needs them. Its `_vel` writes np.cross's products out. Both are tested bit-equal. |
| `scripts/fb_m6b_check.py` (new) | `speed` (EO vs the replica at K, interleaved, per controller) and `invariance` (A7.2 report: N = 1 vs 'batch' and 'rowwise'; per-step traces; first differing quantity; same-state module probes). |
| `scripts/fb_m6_run.py` | `BATCH, ROWWISE = 64, True`; `batch_of(conf)`; batch and rowwise in `cmeta`, so a resume never mixes them; `manifest(out, "batch", ...)`; `work` passes them to `fb_replica.run`. |
| `gpu/fb_rewards.py` | `W_EPS = 0.00011609440647930569` (frozen) and `START_S = 0.5`. `KnockdownAttribution(eps=)` charges only when W > eps and t > START_S (+1e-9), on top of M5's clauses. `FbRewards(eps=)`. |
| `scripts/fb_m5_check.py` | `charge_rule` (the final rule; `eps=0, start=False` gives M5's rule), `w_eps()` (the A5.1 measurement), `replay(eps=)`. `recorded_checks` checks the final rule and reports `charged_m5_rule`, `not_charged_start_window` and `not_charged_w_eps`. `early_kd` adds `charged_final_rule`. Scripted cases 11 `w_eps_floor` and 12 `start_window`; case 10's expectation uses W_EPS. `main` also asserts W_EPS frozen = measured. |
| `scripts/fb_m6_check.py` | A6.5: f64 gates the diagnostic configuration (actor onnx + ref eval) with a same-state shadow there. `RDQ_AMP = 50`. The obs dims carrying rdq (24:48) are split out as `obs132_rdq`. `mlp_vs_graph64` always uses the deployed TrackerMLP f64 and is report only. The deployed configuration is reported, not gated. |
| tests | `test_fb_replica.py`: `_read` = `read` and `_vel` = np.cross; Batch rowwise rows = N = 1 (1 s, K 3); `rowwise_face` = the driver's face. `test_fb_m6_run.py`: batched configuration and meta. `test_fb_rewards.py`: W_EPS and the start window, `w_eps_floor` and `start_window` in the scripted set, `test_w0_frozen` updated (2 x W_EPS of work). |

Scratch (`build\m6b\`): `prof1..3.log` (cProfile N = 1 / K 16 / K 64), `t_speed_10s.log` (first, overloaded timing), `speed\m6b_speed.json`, `inv_run1\` (invariance before the facing fix), `inv\m6b_invariance.json` (final), `probe_src*.py/.json` (op-level probes), `weps_pre.json`, `m5_check.json/.log`, `m6check\m6_*.json`, `dry\` (req --dry), `suite.log`, and pre-edit backups (`*.pre.py`, copies).

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=1, PY = scratchpad venv python)
```
$PY -I build\m6b\weps_pre.py <m5w>\rec\fb_m5rec_w0_r4pre.npz build\m6b\weps_pre.json      # W_eps before the freeze (142 s)
$PY -I scripts/fb_m5_check.py --out <m6b>\m5_check.json --rec <verify_m5>\rec\fb_m5rec_60200-60363.npz --rec-w0 <m5w>\rec\fb_m5rec_w0_r4pre.npz --p1 <P1>\p1_eval.py --workers 2 --reuse   # exit 0
$PY -I scripts/fb_m6b_check.py speed --out <m6b>\speed --ks 64 --workers 2                  # 1218 s
$PY -I scripts/fb_m6b_check.py invariance --out <m6b>\inv --k 64 --workers 2                # 5990 s
$PY -I scripts/fb_m6_check.py all --out <m6b>\m6check --workers 2                          # exit 0
$PY -I scripts/fb_m6_run.py req --dry --workers 1 --seconds 2 --out <m6b>\dry
$PY -I scripts/fb_seeds.py --audit <fullbody>\SEED_LEDGER.md --scan
$PY -m pytest -q -p no:cacheprovider --basetemp="<scratchpad>\pytest-tmp" tests             # 407 passed, 3 skipped
```

## 1. Speed (A7.1)
**Profile at N = 1 (REQ_CONF, cProfile):**
- The replica was op-overhead bound. AimOp's 12-step IK (FK in about 150 small torch ops per call) and the account's per-substep logic dominated.
- mj_step was about 3% of the time.
- Both of these costs are paid once per step regardless of the batch, so batching amortises them.

**Measures taken, in order:**
1. Lockstep batch with per-bout MjData; `torch.set_num_threads(1)`.
2. `Sensors._read` computes the 'com' / 'point' velocities only when they are used. This saved about 40% of the sensor time at 'site'.
3. Body velocity only for touched body shells (`flatnonzero` instead of a per-shell `any`).
4. Torch inference mode.

Offline scoring of logged sensors was not needed.

**Choosing K** (first attempt on an overloaded machine, 10-s bouts):

| K | s per 20-s bout |
|---|---|
| 32 | 13.7 |
| 64 | 11.1 |
| 128 | 10.7 (twice the memory) |

EO went from 3.6 to 9.7 s during that run. The machine was swapping, so this run gives only the shape of the curve. **K = 64** it is: 192-bout shards are exactly 3 batches, and memory is about 1.3 GB commit per worker.

**Final measurement** (`speed\m6b_speed.json`, after R0-EO finished; 20-s bouts, episodes 60200-60263; 2 workers, one controller each; EO = eval_opponent.run as R0-EO runs it, 32-bout calls, timed before and after):

| ctrl | EO s/20-s bout (runs) | rowwise K64 | x EO | batch K64 | x EO |
|---|---|---|---|---|---|
| G | 2.276 (2.18, 2.37) | 2.653 | **1.17** | 2.546 | 1.12 |
| H | 2.490 (2.42, 2.56) | 2.877 | **1.16** | 2.620 | 1.05 |
| A3 | 2.539 (2.81, 2.27) | 2.628 | **1.03** | 2.452 | 0.97 |

- Per bout (not normalised to 20 s): rowwise 2.65 / 2.82 / 2.39 s vs EO 2.23 / 2.43 / 2.33 s.
- G's phases (rowwise, 64 bouts): physics + hooks 89 s, AimOp/driver 43 s, account 23 s, obs 6 s, controller 4 s.
- **Target <= 2x EO: met. A7.4 (rent CPU if > 3x) does not apply.**

## 2. Batch invariance (A7.2) - `inv\m6b_invariance.json`
Setup:
- 64 bouts per controller: 60100-60115 (identity) and 60200-60247 (hit_scoring).
- REQ_CONF, 20 s; N = 1 (`run_bout`) vs K = 64.
- Traces: the action, d.ctrl and qpos at every step.
- Probes: every batched call of the tracker MLP, TrackerDriver.face and the controller, and sampled calls of AimOp.ik, zone_forces, closing_speeds, proprio83 and kp_state, each recomputed one row at a time on the same inputs.

| ctrl | variant | rows equal | traces equal | first differing quantity | probes with differences | bouts whose outcome changed (fell / kd / head clean) |
|---|---|---|---|---|---|---|
| G | batch | 0/64 | 0/64 | opponent targets, step 1, all 64 | **mlp: 64000/64000 rows, max 9.5e-6**; **face: 26 rows in 26 calls, max 1.5e-8** | 1 (kd 1) |
| G | **rowwise** | **64/64** | **64/64** | none | none | 0 |
| H | batch | 0/64 | 0/64 | opponent targets, step 1, all 64 | mlp 64000/64000 (9.5e-6); face 32 rows (3.0e-8) | 8 (fell 1, kd 7) |
| H | **rowwise** | **64/64** | **64/64** | none | none | 0 |
| A3 | batch | 0/64 | 0/64 | opponent targets, step 1, all 64 | mlp 64000/64000 (1.3e-5); face 29 rows (3.0e-8) | 12 (fell 10, kd 4) |
| A3 | **rowwise** | **64/64** | **64/64** | none | none | 0 |

**Sources:**
1. **The tracker MLP in float32.** At batch K, torch's sgemm accumulates differently from batch 1 on every call and every row (up to 1.3e-5 in the targets). The opponent's targets therefore differ from step 1, and chaos amplifies that: 13-19% of 'batch' bouts change fell, kd or head-clean outcome.
2. **TrackerDriver.face's float32 atan2.** It is vectorised (SLEEF) at batch 64 and scalar at batch 1, which gives 1-ulp differences on about 0.04% of rows.
   - Found in the first invariance run, `inv_run1` (MLP row-wise only): 53/53/51 traces equal; the residual 11/11/13 bouts diverged later (steps 14-994) at the opponent targets by 6e-8 to 5e-7.
   - Op-level probes (`probe_src.json`: einsum, matmul [460k calls], vector_norm, normalize, cross, solve, all 0 differences; `probe_src2.json`: sin, cos, arccos, rad2deg, 0; **atan2: 767/2000 calls differ, all at fb_opponent.py:74 `wrap` in `face`**).

Every other probed module is bit-identical at K = 64: the controllers (G's ONNX, P1's hybrid, A3), AimOp.ik with its solve and FK, zone_forces, closing_speeds, proprio83, kp_state, and every elementwise op on the path. The rowwise row of the table is the proof.

**Consequence:** the deployed batched configuration (rowwise, K = 64) is bitwise the N = 1 replica. So R-EQ judges exactly the replica that M6a verified, independent of K. The GPU run stays batched (P1-R4g / P1-Hit judge that), as A7.2 intends. Row-wise costs +0.1 to +0.25 s per 20-s bout (section 1). The plain 'batch' variant stays available (`rowwise=False`) and is reported above.

## 3. A5.1 W_eps and the start window (A7.3)
**W_eps** (`fb_m5_check.w_eps` on `build\m5w\rec\fb_m5rec_w0_r4pre.npz`, measured before the freeze, `weps_pre.json`, re-measured inside the check):
- Over 141,556 live steps of the 144 bouts, max |W_f32 - W_f64| = 1.16094e-5 J. This occurs at 67285, step 901, where W = 91.08 J; the max relative error is 1.8e-5 and the max W is 110.6 J.
- **W_eps = 1.1609440647930569e-4 J**, frozen in `fb_rewards.W_EPS`, and the check asserts frozen = measured (`frozen_ok` true).
- No knockdown edge has 0 < W <= W_eps (66 edges have W > 0). Edges and edge times are identical in f32 and f64 (edge W max diff 1.05e-6).

**Start window:** `START_S = 0.5`. An edge at t <= 0.5 s (t = the step's end, as the replay and the GPU pass it) is never charged.
- Boundary reading: the step ending at 0.50 s lies inside the first 0.5 s.
- No recorded edge falls exactly at 0.50 s (`edges_at_start_s` 0), so the reading changes no count.

**Scripted G0-12 (all 14 cases pass):**
- `w_eps_floor`: positive work of 0.5x and 0.99x W_eps gives 0; 2x gives -2.4 (edge at 0.62 s).
- `start_window`: attributed knockdowns at 0.02 / 0.12 / 0.48 / 0.50 s give 0; at 0.52 / 0.82 s, -2.4.
- `frozen_w0`: tiny_work (6.2e-4 J > W_eps) gives -2.4.

**Charged counts under the final rule** (`m5_check.json`, `g0_12_recorded`; kd_ok, neutral_ok, approach/contact and tangle all true):

| block / ctrl | knockdown edges | charged, final rule | M5 rule | removed by start window | removed by W_eps |
|---|---|---|---|---|---|
| w0_r4pre / G (144 bouts) | 144 | **41** | 63 | 22 | 0 |
| hit_scoring / G | 7 | 1 | 3 | 2 | 0 |
| hit_scoring / H | 9 | 2 | 4 | 2 | 0 |
| w0_calib / G | 4 | 1 | 1 | 0 | 0 |
| w0_calib / H | 3 | 1 | 1 | 0 | 0 |

- A4.2 report (`early_kd_report`): 24 of 144 edges are early (< 0.5 s), 23 of them with contact.
- These knockdowns still count in every eval metric (fb_r0eo `kd`, R-EQ kd gates, E5), as A5.1 requires; nothing in fb_eq / fb_r0eo changed.
- W0 = 0.0 stays (98 contact knockdowns, n_ok, edges = EO).
- G0-11 on EO states passes.
- The check took 3420 s wall; the machine was swapping, see section 6.

## 4. A6.5 R1 re-check (`m6check\m6_identity.json`, `m6_f64.json`, `m6_g011.json`; exit 0)
**Identity:** G 16/16, H 16/16 bit identical to eval_opponent.run. This also proves the `run_bout` refactor changed nothing.

**f64, diagnostic configuration (gated):**
- qpos max 1.998e-15, 0/32 bouts over 1e-9.
- Same-state over 32,032 steps: dr_fields, act_delay, aim_dist, proprio83, delayed83_noise, keypoints, action_targets, applied, fell, opp_down and targets_vs_onnx32 are all 0.
  - aim_rq 6.2e-11
  - obs132_other 6.2e-11
  - obs132_anchor 8.9e-16
  - yaw_spot_axis 5.6e-17
- **rdq: aim 3.10e-9, obs dims 3.10e-9 <= gate 5e-8 (1e-9 x 50). Reported.** On the deployed configuration rdq is 1.095e-9.
- `over` is empty, so `ok` is true.

**f64, deployed configuration (reported, as M6a):** qpos max 8.52e-7 (30/32 over), obs anchor 2.69e-7, targets vs ONNX32 2.11e-6, deployed MLP vs graph64 7.1e-15. These are gated by M2 G0-1 / G0-2 and M3, and distributionally by R-EQ.

**G0-11:** passes with 'site'.
- 128/128 rows equal to the M5 recording's EO rows.
- A4.3 probe unchanged: 88.301% of all contacts, 99.874% of contacts carrying force; found / force mismatches 0 / 0.

## 5. Wall-time estimates on 4 workers
Estimates use the measured rowwise K = 64 speed: per bout 2.65 / 2.82 / 2.39 s for G / H / A3, 7.86 s per episode for all three. EO-speed confs use the measured EO per bout, about 2.4 s. Memory is about 1.3 GB commit per batched worker.

| stage | bouts | estimate |
|---|---|---|
| **R-EQ** (A5.3 looks, 3 controllers) | look 1: 4224 / look 2: 8256 / look 3: 12288 per ctrl | **2.3 h / 4.5 h / 6.7 h** if it stops at look 1 / 2 / 3. **Expected about 3.3 h** (mean N about 6040 per ctrl, fb_eq dry run under shards_ceil). The plan budgeted 6.9 h expected / 12.8 h max. |
| **T1-c T0 arm** (G, shards 0-31) | shared with R-EQ's G shards | 0 h if R-EQ reaches look 2; **0.35 h** (shards 22-31, 1920 G bouts) if R-EQ passes at look 1 |
| **T-opp** | no simulation | minutes |
| **R2** (768 x 3 per step) | (a) torch32, (d) torch64 + training reset: batched (each about 0.4 h); (b) hulls and (c) solver 10/20: numpy N = 1 at about EO speed (each about 0.4 h); **(e) account scoring on the numpy backend runs at N = 1** with the per-step account (estimate about 11 s per bout from the N = 1 profile) | **about 3.5 h** in total, about 1.8 h of it step (e) |
| R6 | 2304 numpy N = 1 | about 0.4 h |
| **Total** | | **about 7.5 h expected, about 11 h max** (vs 18 h / 24 h in the plan, and about 55 h / 100 h before A7) |

R2 (e) is the one remaining N = 1 torch stage. Batching it would need a numpy-backend Batch, which A7 does not ask for. It is diagnostic only, so I left it.

## 6. Notes for the main session
1. **Machine load, 11:05-12:40.** The R0-EO workers slowed from about 3 min to about 20 min per shard. Their private memory was 3.1 GB each against a 0.87 GB working set, free RAM was 2-4 GB, and other non-workstream processes were busy. Every long-running python process ran at 15-25% of a core while fresh processes ran at full speed.
   - My 10-s timing run with K = 128 (11:01-11:32, about 2.6 GB commit) overlapped the onset. I stopped my invariance run at 11:50 to free memory; it was restarted at 11:52 and the R0-EO rate did not recover until later.
   - R0-EO finished at 13:14 with 64/64/64 shards (I did not touch it or its files).
   - For R-EQ on 4 workers, keep K = 64 (about 1.3 GB per worker). Do not use 128.
2. **Deployed batched configuration = rowwise K = 64.** It is bitwise N = 1 on 192/192 bouts. If the main session prefers the plain batched modules (closer to how the GPU computes), set `ROWWISE = False`; that variant's invariance is reported above.
3. **A5.1 (b) boundary:** "in the first 0.5 s" is implemented as t <= 0.5 s at the step's end (+1e-9). No recorded edge sits on the boundary.
4. **W_eps on Warp:** A5.1 asks for a P1-Hit re-measurement with the larger floor used. `fb_rewards.KnockdownAttribution(eps=)` takes it.
