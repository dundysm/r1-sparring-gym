# M7 build notes (M2V2_PLAN_v3 section 3, M7 items (1)-(3) and (4a); 2.5), 2026-10-30

**Status: M7 (1)-(3) and (4a) PASS.** (4b), the 16 dev bouts 60400-60415, is prepared and was not run: it is the Join step and needs
scripts/fb_replica.py.
- `scripts/fb_m7_check.py` exits 0 in 229 s. Every ok is true: M7-1, M7-2, M7-3, M7-4.
- No frozen threshold was changed.
- There is one design choice that the main session should confirm: the ONNX computes in float64 internally. See "Decision for the main
  session" below; it is the reason (4a) passes.

**Rules kept:**
- Only new files were added: `gpu/hybrid_actor.py`, `scripts/export_hybrid_fb.py`, `scripts/fb_m7_check.py` and `tests/test_fb_hybrid.py`.
  No existing file was edited.
- Nothing was committed, stashed or branch-switched. The branch is feat/sparring-m2v2.
- CPU only: one process with 2 torch threads, plus at most one more short process. $0 spent.
- No simulation. The checks read the existing M4 recording.

## Files
| file | what |
|---|---|
| `gpu/hybrid_actor.py` | <ul><li>**`HybridActor`** (pure torch) with the rsl_rl 5.0.1 actor-model interface: `forward(obs, stochastic_output)`, `get_output_log_prob`, `output_entropy`/`mean`/`std`/`distribution_params`, `get_kl_divergence`, `update_normalization`, `reset`, and the hidden-state no-ops.</li><li>**Legs:** G `model_7496.pt`, frozen (requires_grad False; its normalizer never updates). It reads f32(M3s computed in float64 with the env's float64 scale). This is P1's arithmetic for a float64 row and for a float32 row alike.</li><li>**Arms:** A3s or blind3 `model_8995.pt` on f32(fb_actions.arm_input).</li><li>**Stage A:** ArmOnlyGaussian. `std_param` stays 24-d; sample, log-prob, entropy and KL use dims 14-23 only, and the leg dims of a sample are G's mean.</li><li>**`to_stage_b(seed)`:** adds the residual MLP 111-256-128-14 (ELU) on the arm net's view. Its last layer is exactly 0 and it has its own normalizer, initialised as a copy of the arm net's. The Gaussian then covers 24 dims: leg std 0.15, clamped at <= 0.35; the arm std continues.</li><li>**`res_l2()`** = \|r\|^2 on the residual mean.</li><li>The nets (`Net`) use rsl_rl's state-dict keys and load the donors' `actor_state_dict` strictly.</li><li>`mean(obs, ref=True)` evaluates the same function with every net in float64 on the float32 weights and rounds the action to float32.</li><li>**`export_onnx` / `onnx_model`:** builds a hand-written ONNX graph: input `obs` [1, 111] float32, output `actions` [1, 24] float32. It holds M3s, G, the arm net and the residual. The compute dtype is float64 by default, or float32. Metadata is the arm donor's, plus `fb_hybrid` (stage, blind, dtype, scale, clip, source sha256).</li></ul> |
| `scripts/export_hybrid_fb.py` | <ul><li>`export`: the start, or an rsl_rl checkpoint of a HybridActor (residual keys mean stage B), written to `policy.onnx`, `gym.json` (G's, byte copy, which the eval's MjlabPolicy reads) and `export_manifest.json` (sha256 of every source and output).</li><li>`t9a`: ONNX = torch on recorded rows.</li><li>`t9b` (Join, prepared): EO.run unchanged with the ONNX on 60400-60415, against fb_replica in eval mode driven by the torch composition. It also runs a same-state check of the composition on every obs the ONNX received. Seeds go through fb_seeds role `t9`; at most 2 workers.</li></ul> |
| `scripts/fb_m7_check.py` | The M7 gates (1), (2), (3), (4a) on the recorded eval obs, written to JSON. |
| `tests/test_fb_hybrid.py` | 15 tests, about 7 s: the env's scale and strict loading; the nets = rsl_rl bit for bit; stage A = compose and = P1's arithmetic with the ONNX plugged in (sighted, blind); no leg-dim gradient; normalization updates the arms only; B at init = A (sighted, blind) and the std clamp; the anchor gradient reaches only the residual; ONNX = reference through MjlabPolicy (A/B x sighted/blind), metadata and gym.json; checkpoint round trip; t9b's seed guard. |

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, S = ...\scratchpad, B = $S\m1m2\fullbody\build)
```
$PY -I scripts/fb_m7_check.py --out $B\m7\m7_check.json --m4rec $B\rec\fb_m4rec_60000-60015.npz --p1 $S\m1m2\gap\run4pre\P1\p1_eval.py --dir $B\m7\exports   # exit 0, 229 s (m7_check.log)
$PY -I scripts/export_hybrid_fb.py export --out $B\m7\start_sighted          # the start H(A3s)/M3s ONNX (sha256 450556ac...)
$PY -I scripts/export_hybrid_fb.py export --blind --out $B\m7\start_blind
$PY -I scripts/export_hybrid_fb.py t9a --dir $B\m7\start_sighted --m4rec $B\rec\fb_m4rec_60000-60015.npz --out $B\m7\t9a_start_sighted.json   # exit 0: 0.0, 0 / 368,736 bits
$PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests/test_fb_hybrid.py     # 15 passed, 6.4 s
$PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests                       # full suite: 406 passed, 3 skipped (env-gated), 1284 s, exit 0 (m7_suite.log); its MUJOCO_LOG.TXT deleted
$PY -I scripts/fb_seeds.py --audit $S\m1m2\fullbody\SEED_LEDGER.md --scan                    # ok: 20 roles disjoint; ledger consistent; 45 task inputs clean
```
Join step (not run here): `$PY -I scripts/export_hybrid_fb.py t9b --dir <export dir> --out t9b.json --workers 2`.

## Rows
- 15,364 rows, which is every P1 H(A3s)/M3s control step of the M4 recording (16 bouts, seeds 60000-60015, role `record`).
- Each row is the 111-d obs exactly as the controller received it: the float32 83 plus the eval's float64 noise, then the delivered
  keypoint block. The recording also holds P1's action for each row.
- Read only. No new bout was simulated.

## Results (`build\m7\m7_check.json`)
| item | rule | result |
|---|---|---|
| **(1) a** | Stage A = H(A3s)/M3s bit for bit (P1's p1_eval.py hybrid) | **PASS.** HybridActor with P1's own ONNX sessions plugged in (P1's `c.base`, `c.arms`), batch 1 as recorded, 15,364 rows:<ul><li>vs P1's `M3sHybrid.act`: **0 bit mismatches**;</li><li>vs the **recorded eval action**: **0**;</li><li>blind twin vs P1's H(blind3)/M3s: **0**.</li></ul> |
| **(1) b** | the same, with the deployed torch nets (M4 G0-10's composition) | **PASS.** HybridActor.mean (torch float32) vs `fb_actions.compose` with the same nets: **0 bit mismatches**.<ul><li>This holds on the float64 rows and on their float32 casts, batched (15,364) and at batch 1 (2,195), sighted and blind.</li><li>The torch nets equal **rsl_rl 5.0.1's own** EmpiricalNormalization + MLP (training-stack source, run2b/src/rsl_x): **0** (legs and arms, sighted and blind).</li></ul>Info:<ul><li>torch float32 vs the donors' ONNX: legs 1.9e-6, arms 2.4e-6 (GEMM order).</li><li>vs P1's action: 1.9e-6.</li><li>Torch batch 1 vs batched: 2.4e-6 (torch float32 is not batch-invariant).</li></ul> |
| **(1) c** | leg dims have no gradient | **PASS** on 4,096 rows, with a stage-A loss = arm log-prob x advantage + entropy + an MSE on **all 24** mean dims:<ul><li>d(leg dims of the mean)/d(trained parameters): 0 tensors non-zero (arms: 8 of 8 non-zero).</li><li>G: no parameter requires a gradient or gets one.</li><li>`std_param.grad[0:14]` = 0 (arms 10/10 non-zero).</li><li>The arm net's leg-output rows: 0 non-zero gradient entries (arm rows 1,280).</li><li>Sample leg dims = G's mean: 0 mismatches. The log-prob ignores the leg dims.</li><li>After an Adam step on `actor.parameters()`, G's state is bit-identical and so are the leg actions (0 mismatches); the arm actions change.</li></ul> |
| **(2)** | Stage B at init = stage A output bit for bit | **PASS: 0 mismatches everywhere.**<ul><li>Covered: sighted and blind; from the start and from an emulated stage-A final (every trained tensor + 0.01 N(0,1), seeded); to_stage_b(seed 42).</li><li>The mean matches on float64 and float32 rows, batched and at batch 1, and on the float64 reference path.</li><li>The residual output is exactly 0 on every row, and the arm std is bit-equal to A's. The leg std is 0.15 (float32).</li><li>**The exported B-init ONNX = the A ONNX bit for bit** on all 15,364 rows.</li></ul> |
| **(3)** | the anchor-loss gradient reaches only the residual | **PASS.** Loss = 0.5 x mean res_l2 on 4,096 rows, with arms perturbed:<ul><li>Non-zero residual (last layer 0.01 N(0,1), mean \|r\|^2 0.0105): all **6/6** residual tensors have non-zero gradients. Every other parameter has none: no gradient tensor at all on arms, std or G.</li><li>At init (r = 0): the anchor gradient is exactly 0 everywhere.</li></ul> |
| **(4a)** | T9: ONNX = torch on 1e4 recorded rows, <= 1e-6 | **PASS: max abs 0.0, 0 bit mismatches of 368,736 elements**, in each of 4 configurations: start sighted, start blind, trained-like stage B sighted, and trained-like stage B blind (arms perturbed, residual non-zero up to \|r\| 0.13).<ul><li>The ONNX runs through r1gym's `MjlabPolicy` (the eval's loader: float32 cast, batch 1); it is compared with `HybridActor.mean(ref=True)` on the same float32 rows.</li><li>**Loader contract:** kp_mode `obs`/`zeros`, obs_dim 111, batch 1, and controller_cfg = G's = P1's H cfg (home, gym.json face/hold) in all 4.</li></ul>Info:<ul><li>The deployed float32 torch path vs the ONNX: 2.9e-6.</li><li>The start ONNX vs P1's recorded actions: 2.9e-6.</li></ul> |

## Decision for the main session: the ONNX computes in float64 (the evidence)
- (4a)'s "ONNX = torch <= 1e-6" cannot be met by any **float32** ONNX against **float32** torch. The proof uses the eval's own shipped
  exports on the same 15,364 rows:
  - onnxruntime vs the checkpoints' rsl_rl nets in torch float32 gives G 1.67e-6, A3s 2.38e-6 and blind3 2.86e-6. Only 13-14% of
    elements are bit-equal. \|a\| reaches 8.9-12.7, where a float32 ulp is 9.5e-7.
  - Our own float32-graph export gives 2.9e-6 vs the reference and 2.1e-6 vs float32 torch.
  - The cause is GEMM summation order (MLAS vs torch's CPU kernel). M2 G0-1 met the same thing on the tracker (7.6e-6).
- **What I built (default `--dtype float64`):** the graph casts the float32 obs to float64 and evaluates M3s, both nets and the
  residual in float64, on the exact float32 weights. It then rounds the action to float32.
  - The torch reference is the same function: `mean(ref=True)`. It matches the graph bit for bit (0.0 above).
  - The input and output are float32, so the eval's loader and contract are unchanged.
  - With this export, (4b)'s "bit identical to the replica's composition" is achievable with the **torch** composition in the replica.
    With a float32 graph it would only hold if the replica ran the ONNX session itself.
- **What it costs:** the deployed policy differs from the float32 training forward by up to 2.9e-6 per action. That is the same size
  as the run-3 donors' own ORT-vs-torch gap (1.7-2.9e-6), and well below the eval's obs noise.
  - It is also not bit-identical to P1's H(A3s)/M3s evaluation of the start: 2.9e-6 vs the recorded actions.
  - The start's EO numbers (R0-EO H, LOCO "start", selection's G/H) come from P1's float32 ORT pair. A paired comparison of an export
    against them is therefore paired by episode, not bit-identical at init. This was already true of any single-file export: a float32
    graph is not bit-identical to P1's two-session pair either.
- **Options:**
  - **(i)** keep float64 (built; (4a) passes as frozen);
  - **(ii)** export float32 and state that T9 (4a) is impossible as frozen with any float32 graph. That would need an amendment (e.g.
    a float64 reference comparison, as M2 G0-1 f64);
  - **(iii)** also export the start through `export_hybrid_fb.py` and use that file as the "start" in LOCO / selection, so start and
    candidates share one arithmetic. That is report-level, not a gate.
- I did not decide between them. (4a) is reported as passing under (i), which is the default.

## Notes for the Join step, M9 and M10 (facts)
1. **(4b) `t9b`.**
   - **What it runs:** the EO side is eval_opponent.run unchanged with the exported ONNX through `load_controller`, as R0-EO runs a
     cell (threads 1, DR, meter, falls, keypoints truth, aim.json lag). The replica side is fb_replica in eval mode (`RP.conf()`), driven
     by `TorchCtrl`, which runs HybridActor.mean(ref=True) row by row.
   - **How the controller is plugged in:** `TorchCtrl` is injected as fb_r0eo's cell `FB_T9` in the worker, with the same cfg and lag.
   - **What it compares:** EO rows by JSON identity, as fb_m6_check identity does. It also compares, same-state, the composition on
     every obs the ONNX received (logged by a read-only wrapper).
   - Not run here.
2. **Batch invariance:** torch float32 nets at batch 1 differ from batched by up to 2.4e-6. The single ONNX has a fixed batch of 1
   (as the donors): MjlabPolicy runs it row by row, so its action never depends on how many envs are live.
3. **Constructing the actor for M9/M10's rsl_rl runner:**
   - HybridActor has the rsl_rl 5.0.1 model methods, but not MLPModel's constructor signature `(obs, obs_groups, obs_set, output_dim,
     **cfg)`. The runner needs a thin factory class (`class_name`) that calls `HybridActor.from_checkpoints(...)` and, for B,
     `to_stage_b`.
   - The PPO optimizer may take `actor.parameters()`: G's parameters never get a gradient, and an Adam step leaves them bit-identical
     (tested).
   - `trainable()` gives the trained set.
4. **WATCH E0 asserts this module supports:**
   - leg_dev = 0 in A0/A (the leg dims are G's mean exactly; sample = mean);
   - the residual is exactly 0 at B start;
   - the arm std equals the donor's at load (`std_param` is the donor's `distribution.std_param`).
5. **The residual's input normalizer** is a copy of the arm net's (count 7.4e8 from run 3), so it barely moves. This is a choice
   inside the frozen spec ("MLP(111 -> 256 -> 128 -> 14)", with normalization unspecified). The residual reads the blind twin's 28
   zeros, as the arm net does.
6. **res_l2** is per env on the residual's **mean** (plan 2.4: "the residual's mean is penalised"). Weight -0.5 annealed to -0.1 over
   B 0-2000 (M10 / the task).
7. **Hygiene:** `hybrid_actor.py` reads only the G and arm donor files (and their ONNX for metadata).
   - The tracker ONNX, kfab_202/204/205/206 and r1_kungfu.csv are never touched.
   - `fb_seeds --scan` is clean (45 task inputs).

## Ledger
- One row was appended ("M7"). M7 reads the existing recording 60000-60015 (role `record`) and simulates no new bout.
- 60400-60415 (role `t9`) stay untouched.
- The t9b seed-guard test asks for 22000, 21000, 60399, 60416 and 60000, and each is refused before any episode is built.

## Outputs
- `build\m7\m7_check.json` and `m7_check.log`: the check.
- `build\m7\exports\`: the 4 T9 exports in float64 and float32, plus the c2 A/B exports.
- `build\m7\start_sighted\` and `start_blind\`: the start ONNX with gym.json and manifest.
- `build\m7\t9a_start_sighted.json`.
- `build\m7\m7_suite.log`.
- Dev probes: `build\m7\probe_ort.py` and `smoke.py`.
