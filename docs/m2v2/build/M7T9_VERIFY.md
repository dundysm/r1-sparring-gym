# M7T9 verify (M2V2_PLAN_v3 3 row M7 item (4b), plus re-run of (1)-(4a); A9.1), 2026-10-30

**Verdict: ok = true.** Every frozen check passes when re-run independently. One fail-safe robustness bug in t9b was fixed (below).
No threshold or margin was changed. CPU only, at most 2 worker processes per process, $0.

Verify outputs: `build\m7t9v\` (re-made checkpoints and exports, t9b_{ss,sb,bs,bb}.json/.log, m7_check.json/.log, suite.log,
stale.py, summ.py, run.sh).

## 1. Re-runs
| check | result |
|---|---|
| mk_ckpt (builder's script, fresh dir) | ck_{sighted,blind}_B model_1.pt sha256 = the builder's; reload = original; residual max 0.49 / 0.49 (non-trivial stage B) |
| 4 exports re-made | policy.onnx sha256 = builder's for all 4 (450556ac, fef39160, 5a0d4ed1, 212b375a); start sighted/blind = `build\m7\` |
| t9b start sighted | ok, 16/16 identical, same-state 16,000 rows 0.0 / 0 bits, ctrl rows 16,000 = 16,000, 0 falls, f32 control 0/16 |
| t9b start blind | ok, 16/16, 15,703 rows 0 bits, 15,703 = 15,703, 1 fall, control 0/16 |
| t9b trained-like B sighted | ok, 16/16, 16,000 rows 0 bits, equal counts, control 0/16 |
| t9b trained-like B blind | ok, 16/16, 15,891 rows 0 bits, equal counts, 1 fall, control 0/16 |
| t9b JSONs vs builder's | equal on every key except wall_s (all 4) |
| fb_m7_check.py (1)-(4a) | exit 0, M7-1..M7-4 true; JSON = `build\m7t9\` and `build\m7\` on every key except wall_s and the m4rec path spelling |
| tests/test_fb_hybrid.py | 16 passed (8.4 s) |
| full suite, frozen command | **408 passed, 3 skipped, exit 0** (790 s), after the fix |
| fb_seeds --audit --scan | ok: 20 roles disjoint, ledger consistent, 45 task inputs clean (after appending row M7T9-V) |

## 2. Code read against the eval
- `_t9b_job` builds cfg exactly as `fb_r0eo.cell` (controller_cfg + opponent_mode policy + AIM; aim.json operator asserted, lag_s) and
  calls `EO.run(..., threads=1, dr=True, meter=True, lag, rev=False, falls=True, keypoints="truth")`, R0-EO's call verbatim. EO.run,
  r1gym and the replica are unmodified (`git status`: only the two builder files + this fix).
- EO.run steps all of a worker's bouts in lockstep and calls `ctrl.act` on the stacked live rows; MjlabPolicy (batch1) casts to float32
  and runs row by row. TorchCtrl casts to float32 and runs `mean(r[None], ref=True)` row by row: same input rounding, same batch 1.
  MjlabPolicy does no post-processing after the session, so the raw float32 action is the comparable quantity.
- The replica (RP.conf() = EVAL: full hulls, solver 100/50, reset eo, hitmeter, numpy backend) builds its own model via `world()`,
  so EO.run's DR writes into its deep-copied model cannot leak into the replica.
- `equal` compares the whole EO-format row (json sort_keys), the same identity as fb_m6_check R1. The ctrl_rows equality closes the
  hole of a replica silently driven by a different controller object (see 3).
- Blind guards: ONNX kp_mode must be 'zeros' iff --blind, and export_manifest.json's blind must equal --blind (the manifest check is
  skipped if the file is absent; all M7 exports write it).
- Seeds: `t9b_episodes` checks role 't9' before and after building episodes; train clips only; no val/test seeds, kfab_202/204/205
  or Track 2 data touched.
- The --control replica (float32 torch, <= 2.9e-6 from the ONNX) diverges on 64/64 bouts, so 0-bit identity at 20 s is a powerful test.

## 3. Bug fixed: stale controller from fb_replica.world's per-process cache
`RP.world(name, c)` caches (ctrl, cfg, lag, model) per (name, conf) in `RP._W`; `_t9b_job` re-registers `R0._CELLS["FB_T9"]` each call
but the cached world keeps the first call's TorchCtrl. A second t9b job in the same process (workers=1 called twice in one process,
e.g. a later M10 selection loop or the pytest process; or Pool(2) handing both chunks to one worker) would drive the replica with the
**previous** actor. Shown in `build\m7t9v\stale.py`: after call 1, `RP.world("FB_T9", conf)` returns the old object although
_CELLS was replaced. Before the fix this fails safe (the new TorchCtrl serves 0 rows, so the builder's ctrl_rows check makes ok false),
but it would mis-report a correct export as failing. Fix (scripts/export_hybrid_fb.py, one line before the cells are registered):
```
for k in [k for k in RP._W if k[0] in ("FB_T9", "FB_T9_F32")]: del RP._W[k]   # RP.world caches per name: a 2nd job in this process brings its own actor
```
After the fix: start sighted then trained-like B sighted in one process (60400, 1 s) both ok, 50 = 50 controller rows, 0 bits.
All 4 20 s t9b runs above were made with the fixed code and equal the builder's JSONs.

## 4. Notes (non-blocking, agree with the builder's)
1. t9b is batch 1 in the replica; Batch (K > 1) identity is M6b's rowwise contract, not covered here.
2. Blindness still comes from --blind (now cross-checked against manifest and kp_mode); A9.2's checkpoint flag is M9/M10's.
3. The stage-B checkpoints are fb_m7_check.c4's perturbed construction; real M10 candidates must re-run t9b before selection.
4. Identity is versus `mean(ref=True)` (A9.1 float64 reference). Training runs float32 torch, which is 2.9e-6 away and does NOT give
   bout identity (control 0/64); that gap is the disclosed A9.1 one.
5. MUJOCO_LOG.TXT from the suite (mjSpec empty-prefix warnings) deleted; no `Users*` folder. Ledger row "M7T9-V" appended.
