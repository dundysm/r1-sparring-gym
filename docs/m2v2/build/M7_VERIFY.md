# M7 independent verification (M2V2_PLAN_v3 section 3 row M7, 2.5; items (1)-(3) and (4a); (4b) is the Join step), 2026-10-30

**Verdict: ok = true.** Every frozen M7 check in scope passes on my re-run, and an independent float64 reference written from the raw
checkpoints agrees with the export bit for bit. No threshold was touched. One fix (CPU hygiene, results unchanged). One decision stays
with the main session: the float64-internal ONNX (builder's issue 1; I confirm the evidence, see 2.1). Two hazards for M9/M10/selection
(2.3, 2.4) are not M7 defects.

## 1. Re-runs (cwd repo, PYTHONIOENCODING=utf-8, venv python -I, B = scratchpad\m1m2\fullbody\build)
| what | result |
|---|---|
| `scripts/fb_m7_check.py --m4rec B\rec\fb_m4rec_60000-60015.npz --p1 ...\P1\p1_eval.py --dir B\verify_m7\exports --out B\verify_m7\m7_check_v.json` | exit 0, M7-1..M7-4 all ok. **Equal to the build's `m7\m7_check.json` on every key** except wall time (2338 s, the machine was loaded by the other lanes) and the m4rec path spelling. |
| `export_hybrid_fb.py export` (sighted, blind) into `B\verify_m7\start_*` | deterministic: policy.onnx sha256 450556ac... (sighted) and fef39160... (blind) = the build's. |
| `export_hybrid_fb.py t9a --blind` on my blind export | max_abs 0.0, 0 / 368,736 bits, exit 0 (`verify_m7\t9a_start_blind_v.json`). |
| `B\verify_m7\probe_v1.py` (independent, see 2.1) | numpy float64 from the checkpoints = the build's start ONNX through MjlabPolicy: 0.0, **0 bits**, sighted and blind. |
| full suite, frozen command | **407 passed, 3 skipped, exit 0, 1160 s** (`verify_m7\suite_v.log`; 407 = the build's 406 + a test another lane added). Its MUJOCO_LOG.TXT deleted; no Users* folder. |
| `fb_seeds --audit SEED_LEDGER.md --scan` | ok: 20 roles disjoint, ledger consistent, 45 task inputs clean (also after my ledger row). |

Per item (my numbers = the build's):
- **(1)** a: HybridActor with P1's own ONNX sessions plugged in = P1's `M3sHybrid.act` (0 bits) = the recorded eval action (0 bits) on
  all 15,364 H rows; blind twin = P1's H(blind3)/M3s (0). b: torch nets = `fb_actions.compose` (0, f64/f32 rows, batched and batch 1) and
  = rsl_rl 5.0.1's own EmpiricalNormalization + MLP (0). c: leg-mean gradient 0 tensors; G has no grad; std_param.grad[0:14] = 0; arm
  net's leg rows 0; sample legs = G mean; log-prob ignores legs; Adam step leaves G and leg actions bit-identical. PASS.
- **(2)** B at init = A: 0 mismatches in all 4 cases x (f64, f32, batch 1, ref), residual exactly 0, arm std bit-equal, leg std 0.15,
  B-init ONNX = A ONNX. PASS.
- **(3)** anchor gradient: 6/6 residual tensors non-zero, no other parameter has a gradient tensor; at r = 0 exactly 0. PASS.
- **(4a)** ONNX through r1gym's MjlabPolicy = HybridActor.mean(ref=True): 0.0 / 0 bits in 4 configs (start, trained-like B; sighted,
  blind; B residual up to 0.129); contract kp_mode obs/zeros, obs_dim 111, batch 1, controller_cfg = G's = P1's H. PASS.

## 2. Line-by-line review against the eval and the training stack
Read: `gpu/hybrid_actor.py`, `scripts/export_hybrid_fb.py`, `scripts/fb_m7_check.py`, `tests/test_fb_hybrid.py` against
`r1gym/policy.py` (MjlabPolicy, load_controller, controller_cfg), `r1gym/env.py` (scale, clip), P1 `p1_eval.py` M3sHybrid and
`falls/resim.py` Hybrid (`_o`, cfg from base), `gpu/fb_actions.py`, rsl_rl 5.0.1 (`models/mlp_model.py`, `modules/distribution.py`,
`modules/normalization.py`, `algorithms/ppo.py`), `scripts/fb_replica.py` / `fb_r0eo.py` / `fb_m6_check.py` (for t9b), the plan 2.5 and
A1-A8.

Matches (no defect): M3s = obs[:, :83] with 73:83 := clip(obs[25:35] / scale[14:24], +-clip) in float64, then float32 (P1 g_in +
MjlabPolicy cast); clip 1e9 = R1HitEnv's `action_clip`; blind twin = 83 + 28 zeros (resim `_o`, MjlabPolicy kp_mode zeros), and the
residual reads the same view; ONNX metadata = the arm donor's (A3s/blind3 differ from G only in command_names/observation_names;
default_joint_pos and action_scale equal), gym.json = G's byte copy, so controller_cfg = P1's H cfg (Hybrid takes base's cfg);
Normalizer forward/update = rsl_rl's (same formula, `training` gate; G kept in eval by the `train()` override); the PPO call sites
(ppo.py 144-147, 157, 184, 256-271, 374, 420/427, 435/458) all resolve on HybridActor; ArmOnlyGaussian slices dims 14-23 for
sample/log-prob/entropy/KL as rsl_rl's Gaussian sums them; stage B std = std_param ('scalar', as the donors' `distribution.std_param`)
with legs set to 0.15 and clamped <= 0.35 (no lower clamp, same as rsl_rl's scalar std); residual 111-256-128-14 ELU, last layer
zeroed.

### 2.1 Decision for the main session: float64-internal ONNX (builder issue 1) - evidence confirmed
- The plan's T9 "ONNX = torch <= 1e-6" cannot be met by a float32 graph. The eval's own shipped exports vs their checkpoints on the
  15,364 rows: ORT vs torch float32 G 1.67e-6, A3s 2.38e-6, blind3 2.86e-6 (13-14% of elements bit-equal); vs torch float64 2.7e-6 /
  4.3e-6 / 3.7e-6. So no float32 graph meets 1e-6 against either torch arithmetic (|a| up to 12.7, ulp 9.5e-7; GEMM order).
- The float64-internal graph meets it exactly, and my independent numpy float64 implementation (written from the .pt files, not from
  hybrid_actor.py) equals the ONNX bit for bit. Input/output stay float32 [1, 111] -> [1, 24]; the eval's loader asserts all pass.
- Cost: the deployed policy differs from the float32 training forward by <= 2.9e-6 per action (the donors' own ORT gap size). I see
  no semantic objection; option (i) is inside the frozen text (2.5 names no dtype). The main session should still confirm (i) vs
  (ii)/(iii) as the builder listed them; I did not decide.

### 2.2 Disclosure correction (not a defect): why the start ONNX != P1's recorded actions (2.86e-6)
The build notes attribute it to GEMM order. There is a second, exact cause: MjlabPolicy casts the 111-d row to float32 **before** the
graph computes M3s, while P1's M3sHybrid computes M3s from the **float64** noisy row and only then casts. G's input slots 73:83 differ
by 1 ulp in **40,464 of 153,640** elements (probe_v1: `m3s_f64row_vs_f32row_g32_bits`); HybridActor's float32 output on float64 rows
vs their float32 casts differs in 122,873 elements. The ONNX's arithmetic (M3s on the float32 row) is the training arithmetic (mjlab
obs are float32), so it is the right one; this is M4_VERIFY 4.5's known 1-ulp M3s rounding. It means "start export vs P1's H" is
paired by episode, not bit-identical, for two reasons, as the builder already concluded.

### 2.3 Hazard for M10 / selection (not fixed): blindness is not in the checkpoint
`export_hybrid_fb.build(blind, ckpt)` takes blindness only from `--blind`. A blind checkpoint exported without it loads strictly
(identical shapes) and is written with the A3s metadata (kp_mode `obs`), so the eval would feed real keypoints to a blind-trained net;
the reverse error feeds zeros to a sighted net. The weights cannot tell them apart (blind3's normalizer has non-zero keypoint stats:
std 0.47-1.87, mean up to 0.87). Recommend the M10 runner factory store the flag (e.g. a `blind` buffer or the run's task name in the
checkpoint) and `build` assert it. No M7 check is affected.

### 2.4 Notes for M9/M10 (facts)
- `HybridActor(obs_groups=("actor",))` is a default; the runner factory must pass the task's actor group names (rsl_rl's
  `obs_groups["actor"]`), or `_cat` raises on the TensorDict.
- Stage-B resume from a checkpoint needs `to_stage_b()` before `load_state_dict` (build() does this by detecting `residual.*` keys; the
  runner factory must do the same).
- (4b) `t9b` (not run, needs 60400-60415): reviewed, consistent with `fb_m6_check._identity`: same `REC.episodes` map and role guard
  (`t9` = 60400-60415, TRAIN clips), EO.run with the arguments of `fb_r0eo.run_cell` (threads 1, DR, meter, falls, keypoints truth,
  aim.json lag), replica `RP.conf()` = EVAL (numpy, full hulls, Newton 100/50), controller injected through `R0._CELLS` (which
  `RP.world` reads via `R0.cell`), `TorchCtrl` casts to float32 as MjlabPolicy does, rows compared by JSON identity. Untested end to
  end; `ok` covers the bout identity only (same-state is reported).

## 3. Fix made
- `scripts/fb_m7_check.py`: the two bare `ort.InferenceSession(...)` calls (c2 B-vs-A sessions, c4 donor sessions) opened onnxruntime's
  default thread pool (all cores) beside the R0-EO job and the M6 lane. Now `ort1()` opens them with `intra_op_num_threads = 1`, as the
  eval's MjlabPolicy does. Every output key is unchanged (re-run above).

## 4. Hygiene
- No other file edited by me; nothing committed/stashed/switched. The working tree also holds other lanes' edits (fb_opponent,
  fb_rewards, fb_replica, ...) and an untracked `scripts/fb_m6b_check.py`; the suite passed with them.
- No simulation; 60400-60415 untouched; sealed seeds and test clips never opened. Ledger row "M7 verify" appended (read-only use of the
  60000-60015 M4 recording).
- MUJOCO_LOG.TXT from the suite deleted; no `Users*` folder in the repo.
- Outputs: `B\verify_m7\` (m7_check_v.json/.log, exports\, start_sighted\, start_blind\, t9a_start_blind_v.json, probe_v1.py/.json,
  suite_v.log).
