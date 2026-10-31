# M10 build notes: CPU training smoke and the pod evaluation scripts (2026-10-30)

**The blocking plumbing smoke passes: all 10 checks are true, exit 0.** The learning canary was run and is report only (m19); it shows no learning in 400 iterations. The pod evaluation scripts are written and wired into pod_fb.sh: `fb_p1.py`, `fb_t1c.py` and `fb_loco_eval.py`. Each was dry-run on this CPU. The full suite passes: 452 passed, 3 skipped. Nothing was committed. Everything ran on CPU, and no process I started used more than 2 workers. No threshold or margin was changed.

## 1. Files

| file | state | what |
|---|---|---|
| `scripts/fb_cpu_smoke.py` | new | **SmokeEnv**: `fb_replica.Batch` (A7, rowwise) at REQ_CONF (hulls mix with the A10 names, solver 10/20, torch32, account scoring, vmode site), using the training reset. It auto-resets in mjlab's step order: physics, then OppDown / account / FbRewards on stale reads, then terminations (fell_eval, ring_out 1 m, nan, time-out bootstrapped), rewards, resets (account closed first), outer loops, obs. Population: T0-nominal rows only (A11.2). **PPO**: rsl_rl 5.0.1's act / process_env_step / compute_returns / update and mini_batch_generator, operation for operation, without tensordict. It uses unitree_r1_ppo_runner_cfg's algorithm with fullbody_task.RL's lr 2e-4 and entropy 0.003, 24 steps per env. **Policy**: the M9 runner factory `fullbody_task.FbActorModel` (A0 / A / B), restarted between stages as FbRunner.load does it (optimizer kept A0 -> A, fresh A -> B through load_into_b). **Critic**: an rsl_rl MLPModel equivalent (EmpiricalNormalization -> 512-256-128 -> 1, fresh) on clean 111 + intents 10 + PRIV 29 + ring_home 2. **Scripted knockdown rounds**, the **magnitude check**, the **canary**, and the CLI `plumbing | canary` |
| `scripts/fb_p1.py` | new | P1 items (1) t0_alone, (2) restricted, (3) r4g, (4) hit, (5) b3, (6) runner: the GPU glue plus pure comparators; `--dry` runs the CPU stand-ins (section 4) |
| `scripts/fb_t1c.py` | new | P2 T1-c. Replica G at REQ_CONF (K 64 rowwise) vs T0 and T1 on fb_eq.t1c_set (shards 0-31), gated by fb_eq.t1c_gate. Both arms run on this machine (A8.2); `--t0-arm` is accepted only with an equal platform fingerprint. T1 is found by run name `*_t1`. Resumable chunks |
| `scripts/fb_loco_eval.py` | new | P3. Pilots sighted + blind (B final; a sighted B stopped by WATCH6 falls back to A final), exported with the blind flag read from the checkpoint (A9.2). The start is the exported start ONNX (A9.1); G is its shipped ONNX. All run on this platform (A8.2) through an unchanged EO.run on 88000-89151 / 89500-90651. Bars via fb_eq.loco_bars, a decision with A5.2's "stop and report", g bootstrap CI, R(pilot vs blind), R(pilot vs G) |
| `tests/test_fb_smoke.py` | new | 17 tests, including **minimal PPO = rsl_rl 5.0.1 PPO.compute_returns / update bit for bit** (same rollout, rsl_rl's own source, a storage shim yielding the same mini-batches: returns, advantages, losses, adaptive lr and every parameter equal) |
| `gpu/pod_fb.sh` | edit (7 lines) | The header now names the eval scripts. P1 passes `--r0eo ${FB_R0EO:-$W/r0eo_pod} --workers N-T` with FB_ROOT / FB_UNITREE. P2's T1 runs get `--agent.run-name t1` / `t1_loco` (fb_t1c finds T1 by name). fb_t1c / fb_loco_eval get FB_ROOT and `--workers N-T` |

Line endings: every file listed is LF. `bash -n` is clean, and the FB_DRY / FB_FAKE_GPU dry run ends `DRY OK`. `fb_m9_check --only pod` gives ok, and its logs show the three eval commands on the bg cores.

## 2. Plumbing smoke (blocking): commands and results

Command (repo): `python -I scripts/fb_cpu_smoke.py plumbing --out build\m10\plumbing`. Result: exit 0, 203 s. Log `build\m10\plumbing.log`, result `build\m10\plumbing\plumbing.json`, checkpoints `ckpt_A0/A/B.pt` and `ckpt_blind_twin.pt`.
Run shape: 8 envs x 24 steps; 20 A0 + 30 A + 30 B iterations (each stage a fresh env reset, as a pod restart). Seeds 69000-69039 (40 of the 69000-69079 training range) and 69080-69095 (scripted). 56 episodes.

| check | result |
|---|---|
| no NaN | 0 NaN / inf in obs, rewards, losses, parameters, sim state |
| legs bit-frozen A0 / A | G's state dict bit-equal to model_7496.pt after every iteration; leg grads None at every update (asserted); sampled leg dims = G's mean (max dev 0.0) |
| residual exactly 0 at B start | B restarted from ckpt_A through load_into_b: residual max abs 0.0, last layer zero, B's action = A's on fresh rows (torch.equal), leg std 0.15, res_l2 over B iteration 1 = 0.0 |
| value loss falls >= 30% over A0 | mean of A0 iterations 1-3: 0.866; mean of 18-20: 0.343; drop **60.4%** |
| every term fires, right sign, within 2x | table below (all 14 ok) |
| knockdown rule (scripted) | 15 edges. Charged 8 (7 scripted + 1 natural), start window 3, W floor 3 (injected 0.5 / 0.9 x W_EPS), land gate 1, other 0. Charged == the independently recomputed rule on every edge. Injected 2 x W_EPS charged 3/3. Reward per charge **-2.4** (float32 -2.39999986) |
| act_delay / k constant per episode | 0 switches (EvalActorObs.k_switches + HybridDelayedAction.switches + per-step comparison with the episode's draw). Shares over 56 episodes: k 0.29 / 0.21 / 0.50, act_delay 0.46 |
| knockdown, tangle, lean, W metrics | populated every iteration. Totals: kd edges 15, charged 8, tangle 1244 N s, lean 1832 deg s, W 409 J, downed 21.0 s |
| checkpoints carry the blind flag | A0 / A / B store False and the blind twin True; `export_hybrid_fb.ckpt_blind` agrees. A cross-twin load_state_dict is refused both ways, and `export_hybrid_fb.build` is refused with the other `--blind` both ways |
| head hits in the env's own account | account clean head 26, landed head 44 (eval HitMeter clean head 31) |

**Per-term magnitude (2.4 table).** R is the term's reward over the env-steps whose stage weights it (A: A0 + A; B: B + the scripted rounds). Q is its driving quantity, measured independently in numpy: from the MuJoCo state, the sensors' substep forces, the eval's own HitMeter for clean hits, the account log for paid windows, and the knockdown rule recomputed from a numpy work ring. x = (R / Q) / rate, with pass at 0.5 <= x <= 2.

| term | rate (2.4) | R | Q | x |
|---|---|---|---|---|
| spar_head_clean | -0.8 / HitMeter clean head hit | -22.4 | 31 | 0.903 |
| spar_head_impulse | -0.4 / N s on head shells | -56.93 | 142.3 | 1.000 |
| spar_torso_impulse | -0.1 / N s on torso shells | -83.51 | 835.1 | 1.000 |
| spar_block | +0.8 / paid window | 54.4 | 73 | 0.932 |
| spar_clearance (A) | +0.5 / clearance s | 39.09 | 78.17 | 1.000 |
| spar_ring | +2.5 / ring s | 702.1 | 280.9 | 1.000 |
| spar_alive | +4 / live s | 1354 | 338.5 | 1.000 |
| fb_knockdown | -2.4 / charged knockdown | -19.2 | 8 | 1.000 |
| fb_down_neutral | -(5/min) x 0.8 / 60 / downed s | -1.376 | 20.96 | 0.984 |
| fb_down_approach | -2.0 / m toward the downed opponent | -8.554 | 4.277 | 1.000 |
| fb_down_contact | -0.1 / N s while down | -19.87 | 198.7 | 1.000 |
| fb_leg_tangle | -0.05 / N s | -62.22 | 1244 | 1.000 |
| fb_lean_away (B) | -0.01 / deg^2 s | -13.06 | 1306 | 1.000 |
| res_l2 (B) | -0.5 annealed / unit^2 s | -0.574 | 1.155 | 0.995 |

For most terms x = 1.000 because the numpy quantity reproduces the definition: it verifies the weight x step_dt x unit wiring end to end. Head clean (0.90) and block (0.93) compare the account against the eval's HitMeter / account log, so they carry real definitional slack.

**Scripted rounds** (`SCRIPTS`; fixtures chosen from kick probes on the A actor, seeds 69000-69031 / 69080-69099, before this run). Each round is 8 envs, one 3-s episode each, using the trained B actor's mean. The kicks are:
- the defender shoved toward the opponent while the opponent is tipped away, at 1.0 / 1.5 / 0.8 s (charged) and at 0.02 / 0.04 s (start window);
- the opponent tipped away untouched;
- the same untouched falls with W injected at the edge: 0.5, 0.5 and 0.9 x W_EPS (blocked) and 2 x W_EPS (charged).

While the opponent is down, the defender is commanded toward it at 0.5 m/s (approach / contact).

## 3. Learning canary (report only, m19)

`python -I scripts/fb_cpu_smoke.py canary --out build\m10\canary --a0 build\m10\plumbing\ckpt_A0.pt` -> `canary.json`, 1170 s. 16 envs, 400 A iterations, seeds 69100-69320 (221 episodes).

| indicator | first 50 | last 50 |
|---|---|---|
| return per env-step | 0.1180 | 0.1158 |
| account clean head hits per live min | 2.188 | 2.190 |
| dones | 17 | 31 (time-outs start after step 1000) |

- Arm std went 0.703 -> 0.701. Value loss went 2.66 -> 0.029. Legs stayed frozen, with 0 NaN.
- **The adaptive lr sat at its 1e-5 floor from iteration ~25 on**, because KL stayed at ~0.016 (desired 0.01, so the halving threshold 0.02 is hit often).
- Reading: no learning signal at 384 samples per iteration, exactly as m19 predicts. I probed the actor: KL is 0.0 exactly before any step. The KL comes from Adam's lr-sized per-parameter steps at the donors' arm std, not from a plumbing defect.
- Per m19 this triggers no work and no gate. It is worth knowing for P0/P4: with desired_kl 0.01 at 4096 envs the lr may also sit low early. That is WATCH6 E4's business, not M10's.

## 4. Pod evaluation scripts: dry runs (CPU)

| script | command | result |
|---|---|---|
| `fb_t1c.py` | `--dry --workers 2 --seeds 69000-69007 --out build\m10\dry_t1c.json` | Both arms (T1 = T0's checkpoint through the T1 swap path) produced bit-identical rows (`arms_identical` true), kd 3 / 3, ratio 1.0, gate pass; 106 s |
| `fb_loco_eval.py` | `--dry --workers 2 --seeds 69000-69011 --ckdir build\m10\plumbing --out build\m10\dry_loco.json` | 3 exports: pilot = M10 ckpt_B, blind = the blind twin (flags read from the checkpoints), start. 4 controllers x {held-out 4, seen 8} through EO.run; bars, decision, g bootstrap and ratios all computed; 173 s. The decision is meaningless at n = 12 |
| `fb_p1.py` | `--dry --ckdir build\m10\plumbing --out build\m10\dry_p1.json` | (1) eval vs torch32 driver, 2 starts x 3 clips: all both-up / equal (max dt 0.0). (2) eval vs driver pelvis paths: max 5.0e-8 m. (3) r4g pipeline: 8 replica bouts per controller vs 2 laptop R0-EO shards; plumbing only, not a verdict (gates fail, as expected at n = 8). (4) the recorder's fields replayed through the HitMeter shim **equal the live HitMeter on 4/4 bouts**; account vs replay counts exact on 4/4; clean agreement 0.926 over 54 landed episodes. That is the account's known same-state agreement on a 4-bout sample (M5/M6 G0-11: 0.9675 on 64 bouts), not a replay error. mjlab's real Scene (compiled with stubs) maps to the replica 126/126 hit geoms and 50/50 bodies. (5) B3 asserts on the smoke env: 4 first obs, 0 bad, 0 switches (few resets in 50 steps; the GPU asserts run on every reset of P1-R4g). (6) commands printed; M10 checkpoints: blind False, residual only in B, legs = G |

## 5. Decisions taken inside the frozen spec (for review), and open items

1. **Smoke reset.** REQ_CONF uses reset 'eo'. The smoke uses REQ_CONF with reset 'train' (the training reset is what training runs). Population is T0 nominal only, which A11.2 makes exact for every stage's iterations 1-100 (the whole plumbing and the canary's first 100).
2. **Base Unitree / RS reward terms** (2.4 "unchanged") exist only in mjlab (feet sensors, posture tables). They are not in the replica, so not in the smoke. The 2x check covers the 14 terms with a 2.4 magnitude. The critic likewise lacks run 3's mjlab-only base inputs and is fresh, not warm-started. P1 (6)'s real-runner smoke exercises the real ones.
3. **"Within 2x of 2.4"** is read as reward per unit of its driving quantity, measured independently, vs the table's rate. fb_down_neutral's rate takes the table's midpoint, 5 clean hits per min.
4. **Scripted kick fixtures** were chosen from probes so that every knockdown class occurs; they are fixtures, not thresholds. W_eps cases between 0 and W_EPS cannot be produced physically, so they are injected into the attribution ring at the edge (A5.1 (a) on a scripted value); the numpy rule uses the injected W.
5. **A8.2 on the pod scripts:**
   - T1-c computes both arms on the GPU pod's CPU. M6's cpu3g T0 arm has no platform record and would be refused.
   - LOCO re-runs the start and G on the pod beside the pilots instead of the plan's "precomputed locally".
   - P1-R4g is GPU vs EO and unpaired, so no CPU platform pairs with it. It is judged against A8.2's pod R0-EO, with the laptop R0-EO optional beside it.
   - **The main session must upload the pod R0-EO** (`podpull\r0eo_pod.tgz`, unpacked) to `$W/r0eo_pod`, or set FB_R0EO. P1 stops with that message otherwise.
6. **GPU glue is unexercised here** (no mujoco_warp). This covers gpu_env, gpu_alone, gpu_r4g (B3 hooks), gpu_hit's Warp field access (`nacon`, `contact.{geom, worldid, pos, frame, efc_address}`, `efc.force`, `data.cvel / xipos / subtree_com`), and runner_smoke. They follow mjlab 1.2.0's source as read. The Warp field names are a best reading of mujoco_warp 3.5 and fail loudly if wrong (P1's budgeted fix cycle). Their CPU halves and comparators are dry-run and tested.
7. The account-vs-HitMeter clean agreement is the same quantity P1-Hit gates at 0.95. It was 0.926 on the 4-bout dry sample and 0.9675 at G0-11. P1-Hit's 32 worlds x 1000 steps is the real measurement.

## 6. Suite, seeds, housekeeping

- tests/test_fb_smoke.py: 17 passed (38 s).
- Full suite (frozen command): **452 passed, 3 skipped, exit 0, 946 s** (`build\m10\suite.log`).
- Seeds: rows appended to SEED_LEDGER.md (69000-69079 smoke, 69080-69099 scripted, 69100-69999 canary). `fb_seeds.py --audit SEED_LEDGER.md --scan`: ok (20 roles disjoint, ledger consistent, 54 task inputs clean; the new scripts are scanned and are free of banned names).
- MUJOCO_LOG.TXT (from the stubbed mjlab scene compiles in fb_p1 --dry and the suite) was deleted. There is no `Users*` folder.
- Scratch probes are in `build\m10\` (probe_env.py, probe_kick.py, probe_early.py, tiny.py, tail.py).
