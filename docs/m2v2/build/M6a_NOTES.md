# M6a build notes (M2V2_PLAN_v3 3 M6 part 1, 3.1, 3.2; A4.3), 2026-10-30

**Status (no threshold or margin changed; nothing decided that belongs to the main session):**

| check | frozen rule | result |
|---|---|---|
| R1 eval identity, 60100-60115 | replica (numpy, full hulls, Newton 100/50, EO episodes) bit identical to `EO.run` for G and H(A3s)/M3s | **PASS** 16/16 G, 16/16 H (whole row: fell_s, every HitMeter count, windows, opp_fall, both_up) |
| f64 qpos before first shell contact | <= 1e-9 per step | **FAIL as deployed (impossible as specified, proof below)**: max 8.5e-7, 30/32 bouts over at step 1. With the eval's own tracker session and reference arrays in the torch driver: **max 2.0e-15, 0/32 over** |
| f64 same-state on the replica's own states | <= 1e-9 | **all defender-side modules exact (0.0)**; aim rq 2.2e-11; **three opponent items over**: tracker obs anchor dims 2.7e-7, rdq / obs rdq dims 1.095e-9, targets vs the eval's float32 ONNX 2.1e-6 (MLP f64 vs the ONNX graph in f64 on the same obs: 7.1e-15) |
| A4.3 contact-sensor pos | pos within 1 cm of MuJoCo's contact position on >= 99% of contacts | **literal reading: 88.3% -> fails -> deployed input 'site'**; on contacts carrying force: 99.874% (would pass -> 'pos'). Reading to confirm (below) |
| G0-11 on the replica, 60200-60263, deployed input | counts per (limb, zone) exact, clean agreement >= 0.95, median impulse within 10% | **PASS** with 'site': G 0/64 mismatch bouts, clean 0.9675, rel 4e-6; H 0/64, 0.9744, 9e-6. Also passes with 'pos' (0.9952 / 0.9987), float32 identical |
| fb_eq formula = 3.1 power table | gate_power_v3's model, its RNG | R-EQ (fixed N, sequential correct / W1-W6 / q0), P3 LOCO, P1-R4g: **every entry equal**. T1-c: **every pass count equal** (per-trial decisions equal on 36,000/36,000 trials); 6 entries printed 0.001 apart in the first two runs only because gate_power_v3's count is a numpy integer and round() of x.xxx5 differs by type (section 5). **Run 3 (operand types matched): the whole gate_power_v3_all.json table equal, 0 diffs** (eq_reproduce3.json) |
| fb_eq dry run (synthetic replica rows, 12288 R-EQ keys, 20 trials per case) | correct passes, W1-W6 fail | every rule: correct 19/20 pass; **W1, W2, W2b, W3, W3b, W4, W5, W6: 0/20 pass** |
| fb_m6_run dry run, 8 bouts | every stage runs end to end, resumable | **PASS**: eo, req (2 looks), t1c (reuses req's G chunk), topp, r2 a-e, r6; a resumed stage finds its chunks done (todo 0) |
| full suite | passes | see the last section |

Rules kept: new files plus edits to workstream fb_* files only (`gpu/fb_opponent.py`: one opt-in flag). No run 1-3 file touched. No commit / push / stash / branch switch. CPU only; my processes used at most 2 workers each (R0-EO's 4 untouched, still running). Seeds: 60100-60115 (role identity, first use) and 60200-60263 (role hit_scoring, re-run in role) only, through `fb_seeds`; ledger rows appended ("M6a"); `fb_seeds.py --audit ... --scan` ok (20 roles, ledger consistent, 30 task inputs clean). No val / test / kfab_202-205 content, no Track 2 motion. No MUJOCO_LOG.TXT or `Users*` folder left.

## Files

| file | what |
|---|---|
| `scripts/fb_replica.py` (new) | The CPU replica. `conf()` switches: hulls full / mix (fb_hulls.npz) / all256; solver (iterations, ls) with 3.2's FALLBACK list; reset eo / train (T0-nominal training reset, torch generators per episode seed: latency first, bout heading U(-pi, pi), DR, t0 uniform on the clean grid, obs noise); scorer hitmeter / account (FullBodyHitAccount primary, HitMeter rides along); backend numpy / torch64 / torch32; self_collision (R6, bit 8); vmode pos / site / com / point; diagnostics actor onnx and ref eval. `build_model` (hull meshes replaced on both robots' shells, native contact sensors found / force / pos as mjlab builds them). `Bout`: its own reset and step loop in the eval's order, borrowing R1HitEnv's `_raw_obs`, `_face`, `_hold`, DR draws; torch path = TrackerDriver + AimOp (8-slot ring, logged by sequence), EvalActorObs + kp_state + proprio83 with the eval's rng noise, HybridDelayedAction, fb_dr, fb_events. `Sensors` (MjContactSensors per substep, closing-speed inputs per vmode, block touches, A4.3 probe, sparse rows). `run_bout` (EO.run's bookkeeping and row: meter_row, ContactTail, both-up), `rep_row` (fb_r0eo.bout_row format + acct / hm), `run`, `episodes(role, seeds)` (fb_seeds-checked before any episode is built). |
| `scripts/fb_eq.py` (new) | 3.1 statistics: `req_gates` (gate_power_v3's arithmetic, operation for operation, with est / se / CI / margin), GATED / REPORT, `sequential` (looks 4096 / 8192 / 12288 on `look_sets`), `classify` (3.2 noise-sized / conclusive), `report` (0.R report-only items), `t1c_gate` / `t1c_set`, `r4g_gates`, `loco_bars`, `reproduce` (gate_power_v3's own __main__ with every decision replaced), `synth` / `dry_run`. **`LOOK_RULE = None` and `T1C_RULE = None`: refused until the main session freezes them.** |
| `scripts/fb_m6_check.py` (new) | identity, f64 (Trace, Contact, Shadow = the eval's code on the torch replica's states), g011 (+ A4.3 probe; logs sparse sensor rows in the replica, then `fb_m5_check.replay` / `compare` unchanged). |
| `scripts/fb_m6_run.py` (new) | The resumable driver for the later CPU stages: req, r2, t1c, topp, r6, all, status, --dry. Chunk files in fb_r0eo's shard format (atomic, matched meta), manifest refuses a mismatching resume, STOP file. |
| `gpu/fb_opponent.py` (edited) | `AimOp(skip_idle=False)`: when True, no IK on a step where no limb has a live aimed intent (d5 is 0 either way; one host sync). Only the replica sets it. Test: bit-identical qpos with and without it. |
| `tests/test_fb_replica.py` (new, 7 tests) | hulls = fb_hulls' hulls on both robots and nothing else changes; solver; self-collision bits; native sensors = MjContactSensors, pos = force-weighted contact centroid (0 without force); eval mode = EO.run (60100, 1 s); diag torch64 bit-equal before contact and deployed torch64 > 1e-9 at step 1; training reset deterministic; account rows; guards; skip_idle. |
| `tests/test_fb_eq.py` (new, 6 tests) | TOST arithmetic, margins, classify, look / T1-c sets of every rule, pairing refusal, decisions = gate_power_v3's on its cells, sequential on synthetic rows (correct passes at look 1, W1 and W5 fail). |
| `tests/test_fb_m6_run.py` (new, 3 tests) | balanced chunks (= fb_r0eo shards), confs, chunk files and manifest. |

Scratch outputs (`...\fullbody\build\m6a\`): `check\m6_identity.json`, `m6_f64.json`, `m6_g011.json`, `g011_bouts.npz` (the replica's logged bouts, reusable via `g011(BA=...)`), `m6_check.log`, `m6_g011.log`; `eq_reproduce.json` / `.log`, `eq_reproduce2.json`; `eq_dry.json`; `gp_self.py` / `.log`, `rng_dbg.py` / `.log` (T1-c control); `dry\` (fb_m6_run dry run: result_*.json, manifest, chunks); `m6a_suite.log`; build probes `probe_pos.py`, `smoke*.py`, `diag_f32.py`; backup `fb_opponent.pre.py`.

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, OMP_NUM_THREADS=1, PY = scratchpad venv python)
```
$PY -I scripts/fb_m6_check.py all  --out <m6a>\check --workers 2     # identity (94 s) + f64 (1026 s); g011 was re-run alone:
$PY -I scripts/fb_m6_check.py g011 --out <m6a>\check --workers 2     # 922 s, exit 0
$PY -I scripts/fb_eq.py reproduce --out <m6a>\eq_reproduce.json      # 748 s
$PY -I scripts/fb_eq.py dry --trials 20 --out <m6a>\eq_dry.json      # 426 s
$PY -I scripts/fb_m6_run.py all --dry --workers 1 --seconds 6 --out <m6a>\dry
$PY -I scripts/fb_seeds.py --audit <fullbody>\SEED_LEDGER.md --scan
$PY -m pytest -q -p no:cacheprovider --basetemp="<scratchpad>\pytest-tmp" tests
```
Launch after R0-EO (main session, once LOOK_RULE / T1C_RULE are frozen in fb_eq.py): `$PY -I scripts/fb_m6_run.py all --out <dir> --workers 4` (or one stage at a time; `status --out <dir>`; `<dir>\STOP` stops after the running chunks).

## 1. R1 eval identity (PASS)
- 32 bouts (G, H x 60100-60115): `json.dumps(row, sort_keys=True)` of the replica equals `eval_opponent.run`'s on every bout. Falls 0, opponent knockdowns 1 per controller, head clean 19 / 19, torso landed 191 / 168.
- The replica's own loop (reset order, DR, the action and obs arithmetic, the opponent step after `mj_kinematics`, 4 substeps, bookkeeping) is therefore the eval. The same holds on 128 more bouts: in G0-11 the eval-mode replica's rows equal the M5 recording's EO rows on **128/128** (60200-60263, G and H; lag 0 as recorded, rev_meter excluded), also proving the native sensors change nothing.

## 2. Torch float64 modules (impossible as specified; proof and options)
**Measured (32 bouts, `m6_f64.json`):**
- First defender-opponent shell contact comes early: steps 1-59 (median 6). Bouts start at 0.40-0.55 m.
- Deployed torch64 (TrackerMLP f64, fb_motion's reference): qpos differs by 4-9e-8 after **step 1** in 30/32 bouts (the other 2 touch at step 1); max 8.5e-7 before contact. The defender's targets at step 1 are identical (0.0); the **opponent's** differ by up to 5.6e-7.
- Same torch64 modules with the eval's tracker session (actor onnx) and the eval Tracker's own reference arrays (ref eval): **qpos max 2.0e-15, 0/32 bouts over** before contact. Every other module (AimOp, TrackerDriver's placement, clock, facing, steering and obs assembly, EvalActorObs, keypoints, HybridDelayedAction, fb_dr, fb_events) therefore reproduces the eval's dynamics.

**Same-state on the deployed torch64 replica's own states (31,939 steps; shadow = the eval's code on each state):**

| item | max diff | |
|---|---|---|
| DR fields, act_delay | 0, 0 | |
| defender proprio83, delayed 83 + the eval's noise, keypoint block, action targets, applied action | 0 (bit-equal) | |
| fall rule, opponent down (EO:86) | 0 | |
| facing yaw / spot / axis | 1.1e-16 | |
| aim rq / dist | 2.2e-11 / 0 | |
| aim rdq, and the tracker obs' non-anchor dims (via rdq) | **1.095e-9** | over 1e-9 by 9.5% |
| tracker obs anchor dims (pos 3 + R6) | **2.7e-7** | |
| TrackerMLP f64 vs the ONNX graph in f64, same obs | 7.1e-15 | |
| opponent targets vs the eval's float32 ONNX | **2.1e-6** | |

**Why 1e-9 cannot be met, with the eval's own code:**
1. The eval's tracker is a **float32 ONNX** session (`r1gym/opponent.py` Tracker.act casts the obs to float32 and runs onnxruntime). Its output differs from the same graph in float64 by ~1e-6 (M2 G0-1: 7.6e-6 vs torch f32). Same state, step 1: eval opponent targets vs the float64 driver differ by 5.6e-7 rad; the plant integrates that into qpos 4-9e-8 after one step. No float64 implementation can match a float32 graph to 1e-9; gated in M2 G0-1 (<= 1e-5 f32, <= 1e-9 f64 vs the graph, which holds: 7.1e-15).
2. fb_motion's anchor reference differs from the eval Tracker's ONNX initializers by up to 1.2e-7 (quaternion) by construction: the task must not read the tracker ONNX (M0's banned input), so M1 exported it from the clips (M1 gate 1e-4; M2 G0-2 (b) 2.7e-7 <= 1e-6).
3. rdq is the warp's finite difference / 0.02 s (aim.py), so the AimOp-vs-Aim difference in q (2.2e-11: the 12-step bounded IK amplifies floating-point order differences, M3 measured ~3e4x) appears x50 in rdq: 1.095e-9. M3 G0-4 reported rdq as information only (6.3e-9 on its recording).

**Options (main session; not decided):**
- (a) Gate f64 qpos on the diagnostic configuration (eval tracker session + eval reference in the torch driver): passes at 2e-15; the two excluded pieces stay gated by M2 G0-1 / G0-2.
- (b) Gate the deployed configuration at the measured floor (for example qpos <= 1e-6 before contact), keeping same-state at 1e-9 for every item except the three opponent items above, which keep their M2 / M3 gates.
- (c) Same-state rdq at 1e-9 relative to |rdq| or with the x50 finite-difference factor (5e-8 absolute).

## 3. A4.3 contact-sensor pos, and the deployed closing speed
- Native mjSENS_CONTACT (reduce netforce, num 1; fields found, force, pos; 8 opponent strike limbs x 32 defender shells, as mjlab builds them) in the replica on 60200-60263 (G, H): found and force equal MjContactSensors on every slot (0 / 0 mismatches).
- **Semantics (probe, test): under netforce, pos = the normal-force-weighted centroid of the pair's contacts; with zero net force it is (0, 0, 0).**
- 453,714 strike-limb-to-shell contacts: within 1 cm of their slot's pos **400,635 = 88.30%**. 164,860 contacts (36%) carry zero force; on the 288,854 force-carrying contacts the share is **99.874%** (max error 11.8 cm on multi-contact slots). Slot-level: strongest contact within 1 cm 89.07%; 15,338 slots have pos 0.
- **Literal reading ("on >= 99% of contacts") -> fails -> deployed input = 'site'** (palm site for hands, foot site for feet, the limb body COM for forearm / shin), set as `fb_m6_run.DEPLOYED_VMODE`. The other reading (contacts whose force defines the point; a zero-force slot's closing speed is 0 in the account whatever v is, because F_hat = 0) passes and would deploy 'pos'. **Decision for the main session.** G0-11 passes either way; 'pos' tracks the HitMeter's own point exactly (same numbers as 'point').

## 4. G0-11 on the replica (PASS)
Replica in eval mode (identity proven), 60200-60263, G and H, sparse sensor rows replayed through fb_m5_check's replay / compare unchanged (the input in its velocity slot):

| input | G: landed HM/ours, mismatch bouts, clean agree, med. head impulse rel | H |
|---|---|---|
| **site (deployed, gated)** | 831/831, **0/64, 0.9675**, 4e-6 | 782/782, **0/64, 0.9744**, 9e-6 |
| pos | 831/831, 0/64, 0.9952, 4e-6 | 782/782, 0/64, 0.9987, 9e-6 |
| com (M5's gated input) | 0/64, 0.9519 | 0/64, 0.9578 |
| point (HitMeter's) | 0/64, 0.9952 | 0/64, 0.9987 |

- **float32 accounts = float64 on every count and call** for all four inputs once times are compared on the HitMeter's 4-dp grid. With fb_m5_check.compare's 1e-6 t0 match, float32 times (t up to 20 s, ulp ~1e-6) go unmatched: a 2-bout smoke showed "0.82 clean agreement" that was entirely this matching artefact (diag_f32.py: 0 differing calls). Relevant to P1-Hit on the GPU: match float32 episodes on the 5 ms substep grid.
- Windows (report): outcomes 1847/1853 and 1838/1843, pay 1852/1853 and 1842/1843, paid 270 = 270 and 258 = 258: as in M5.

## 5. fb_eq.py and the 3.1 power table
- `reproduce()` runs gate_power_v3.py's own simulation (its RNG seed 20261030, its resampling of the R4-pre main rows, the same call order as its __main__) with every gate decision taken from fb_eq. First run (748 s): **R-EQ fixed N (5 N), sequential correct, W1-W6 (8 cases), q0, P3 LOCO (12 cases), P1-R4g (3 cases), calibration: equal to gate_power_v3_all.json in every entry.** T1-c: 6 of 18 entries differ by 0.001 (one trial in 2000).
- Control (`gp_self`): gate_power_v3's own functions in the same order reproduce its json's T1-c table exactly. `rng_dbg`: the RNG state at the start of T1-c is identical with gate_power_v3's and fb_eq's req decisions, and **fb_eq.t1c_gate equals gate_power_v3's inline T1-c rule on all 36,000 T1-c trials** of that stream. The formula is therefore the same. The 6 entries are the counts 505, 87, 1575, 103, 1839, 1945 of 2000, all ending in 5 at the 4th decimal: gate_power_v3 accumulates `np.bool_ and bool` (its count becomes numpy.int64, and numpy's round of 0.2525 gives 0.252) while fb_eq added Python bools (Python's round gives 0.253). Checked directly: `round(505/2000, 3) = 0.253`, `round(np.int64(505)/2000, 3) = 0.252`, and likewise for all six. A second run reproduced the same 6 (deterministic). `reproduce()` now accumulates with gate_power_v3's operand types (decision unchanged): **run 3 reproduces every entry of gate_power_v3_all.json, 0 diffs (676 s, `eq_reproduce3.json`).**
- Per-trial decisions also equal gate_power_v3.req_gates on 200 + 24 random cells (smoke, test).
- `dry_run` (`eq_dry.json`): synthetic EO rows = R4-pre main rows laid on the 12,288 R-EQ episode keys by (clip, distance), replica rows by gate_power_v3's q-model (q 0.1) and its wrong-replica effects, through rows -> paired() -> look_sets -> req_gates -> sequential:

| rule | correct: passed / mean N | W1, W2, W2b, W3, W3b, W4, W5, W6 |
|---|---|---|
| shards_floor (4032/8064/12288) | 19/20, 5453 | 0/20 each |
| shards_ceil (4224/8256/12288) | 19/20, 6038 | 0/20 each |
| reps (4092/8184/12288) | 19/20, 5730 | 0/20 each |
| seed_prefix (literal 70000+n) | 19/20, 10240 | 0/20 each |

  Fail classes are reported per gate (for example W1: surv conclusive, gap_A3_G noise-sized).

## 6. fb_m6_run.py and its dry run
- Stages as specified: **req** (REQ_CONF = 3.2's first fallback entry: hulls mix, solver 10/20, torch32 modules, EO reset, account scoring with the deployed input; shard by shard in the order LOOK_RULE needs, a look evaluated only once all three controllers' shards and R0-EO's are present, stop at the first passing look), **r2** a-e on 67000-67767 vs R4-pre's rows (paired fb_eq statistics; (d) unpaired, P1-R4g's form; (e) also same-state counts), **t1c** (G, REQ_CONF, T1C_RULE's 6144; with 'shards' it is the req stage's G shards 0-31, simulated only where req stopped earlier), **topp** (no simulation, from req and R0-EO rows), **r6** (self-collision on; eval solver vs chosen, = R2 (c) when the chosen solver is 10/20).
- Dry run (8 bouts 60100-60107, 6 s, 1 worker, about 16 min): every stage completed; resumed stages reused finished chunks (todo 0); req evaluated looks (4, 8) and stopped failing (8 bouts cannot pass gap_sign / R5); r2 (e) same-state: account = HitMeter on all 24 bouts' landed / clean counts.
- **Measured cost (loaded machine, 3 other jobs):** eval-mode numpy ~EO speed (1 s per 6-s bout); torch32 modules ~7.5 s per 6-s bout; REQ_CONF (torch32 + account) ~12.5 s per 6-s bout, i.e. about 40 s per 20-s bout per worker. That is 3-4x the plan's 2x-EO budget: R-EQ at the mean 6,600 bouts x 3 controllers is ~55 h on 4 workers (max ~100 h), T1-c's T0 arm adds up to ~17 h for G. The torch modules run at N = 1 and are op-overhead bound (AimOp's IK and the account's per-substep episode logic).

## 7. Decisions for the main session (none taken here)
1. **LOOK_RULE** (fb_eq.LOOK_RULES): the seeds are clip-major, so "the first 4096" is kfab_200 only (R0E note 1). shards_floor / shards_ceil / reps are balanced (power in the dry run as above); seed_prefix is the literal, unbalanced reading (needs all 64 shards at look 1). Freeze one in fb_eq.py before any look; the driver refuses while None.
2. **T1C_RULE**: 'shards' (shards 0-31, all 12 strata, 6144) or 'seed_range' (70000-76143 literally: kfab_200 + kfab_201 at 0.40/0.45 m only).
3. **A4.3 reading** (section 3): literal -> 'site' (set); force-carrying contacts -> 'pos'.
4. **f64 gate** options (section 2).
5. **Replica speed** (section 6): accept the wall time, or authorise batching bouts (N > 1 per process) through the torch modules (needs a batch-invariance proof; the plan says N = 1), or offline batched scoring of logged sensors (the account is row-independent).
6. **R6 and T-opp seeds / definitions** chosen here (flagged): R6 runs on 67000-67767 (role r2_diag, an M6 block) as single-switch diagnostics vs step 0; T-opp = opponent first falls with no robot-robot contact in the 0.5 s before (ContactTail), per bout and per up-minute, from the req rows and R0-EO rows (no new seeds). The training reset of R2 (d) is the T0-nominal row (eval distances, clean-grid t0, random heading, torch draws); T1 rows do not exist yet.

## Findings beyond the gates
- The contact sensor's netforce pos is (0, 0, 0) on zero-net-force slots: 11% of found slots. Any GPU code using pos must mask by |F| > 0.
- 'pos' and the HitMeter's 'point' give identical G0-11 numbers; 'site' loses ~3 points of clean agreement; 'com' is thinnest (0.9519).
- R2 (b) hulls mix moved several statistics in the 6-s dry run (8 bouts; no inference possible) - worth watching in the real R2.

## Full suite
**366 passed, 2 skipped (the env-gated tests), 1470 s, exit 0** (`build\m6a\m6a_suite.log`; run with fb_eq's T1-c accumulation before its last one-line change, after which tests/test_fb_eq.py re-ran: 6 passed). No MUJOCO_LOG.TXT or `Users*` folder left in the repo.
