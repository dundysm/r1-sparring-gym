# M7T9 build notes (M2V2_PLAN_v3 3 row M7, item (4b), plus a re-run of (1)-(4a); A9.1 float64 ONNX), 2026-10-30

**Status: M7 (4b) PASS. (1)-(4a) re-run: PASS.**
- (4b): in all 4 exports, the 16 dev bouts 60400-60415 run through an **unchanged** `eval_opponent.run` with the exported hybrid ONNX
  are **bit identical** (the whole EO row, as JSON) to `scripts/fb_replica.py` in eval mode (`RP.conf()`) driven by the torch
  composition (`HybridActor.mean(ref=True)`, row by row).
- No frozen threshold or margin was changed. CPU only, at most 2 worker processes, $0.

## Files (repo C:\Users\dundy\Downloads\projects\r1-sparring-gym-spar, feat/sparring-m2v2; not committed)
| file | change |
|---|---|
| `scripts/export_hybrid_fb.py` | <ul><li>**t9b hardened.** `TorchCtrl` counts the rows it serves (`ref` param). The job asserts that the ONNX loader's kp_mode matches `--blind` and that the export's `export_manifest.json` blind flag matches `--blind`.</li><li>**`ok` also requires** the controller rows served on each side to be equal (ONNX in EO = torch in the replica).</li><li>**New `--control`** (report only): the replica is also driven by the deployed **float32** torch forward (`mean(ref=False)`, within 2.9e-6 of the ONNX). This shows that the identity check has power.</li><li>The docstring was updated.</li></ul> |
| `tests/test_fb_hybrid.py` | <ul><li>New `test_t9b_one_second_bout_is_identical`: 60400 for 1 s (role `t9`). It asserts ok, identity, 50 = 50 = 50 controller rows and 0 same-state bits.</li><li>It also checks that a `--blind` contradicting the export's manifest is refused.</li><li>The file now has 16 tests (8 s).</li></ul> |

No other repo file was touched. Untouched: run 1-3 files, fb_opponent, fb_mirror, the library files and fb_replica.

Scratch files (`build\m7t9\`):
- `mk_ckpt.py`: writes the trained-like stage-B checkpoints. It uses fb_m7_check.c4's construction: `perturbed(start, 7).to_stage_b(42)`, then the residual's last layer is set to 1e-2 N(0,1) with seed 11. Each is saved as rsl_rl `{'actor_state_dict'}` in `ck_{sighted,blind}_B\model_1.pt`. Reload = original, bit for bit.
- `smoke.py`, `patch.py`, `run_t9b.sh`, `summ.py`, `cmp.py`.

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, PY = scratchpad venv python, D = build\m7t9)
```
$PY -I D\mk_ckpt.py D
$PY -I scripts/export_hybrid_fb.py export --out D\start_sighted                       # sha256 450556ac... = build\m7\start_sighted (deterministic)
$PY -I scripts/export_hybrid_fb.py export --blind --out D\start_blind                 # fef39160... = build\m7\start_blind
$PY -I scripts/export_hybrid_fb.py export --ckpt D\ck_sighted_B\model_1.pt --out D\trained_B_sighted          # 5a0d4ed1...
$PY -I scripts/export_hybrid_fb.py export --blind --ckpt D\ck_blind_B\model_1.pt --out D\trained_B_blind      # 212b375a...
$PY -I scripts/export_hybrid_fb.py t9b --dir D\<export> [--blind] [--ckpt ...] --out D\t9b_<export>.json --workers 2 --control   # x4, each exit 0
$PY -I scripts/fb_m7_check.py --out D\m7_check.json --m4rec build\rec\fb_m4rec_60000-60015.npz --p1 ...\P1\p1_eval.py --dir D\exports   # exit 0, 154 s
$PY -m pytest -q -p no:cacheprovider --basetemp="...\pytest-tmp" tests/test_fb_hybrid.py     # 16 passed, 7.9 s
$PY -m pytest -q -p no:cacheprovider --basetemp="...\pytest-tmp" tests                       # full suite: see below (D\suite.log)
$PY -I scripts/fb_seeds.py --audit SEED_LEDGER.md --scan                                      # ok: 20 roles disjoint; ledger consistent; 45 task inputs clean
```

## (4b) results: `t9b_<export>.json`, 16 bouts each, 20 s, 2 workers
| export | identical rows | same-state (every obs the ONNX received) | controller rows ONNX = torch | falls | head clean | torso landed | control: f32 torch identical | wall |
|---|---|---|---|---|---|---|---|---|
| start sighted (the A9.1 start) | **16 / 16** | 16,000 rows, max abs 0.0, **0 bits** | 16,000 = 16,000 | 0 | 13 | 168 | 0 / 16 | 75 s |
| start blind | **16 / 16** | 15,703 rows, 0.0, **0 bits** | 15,703 = 15,703 | 1 | 7 | 179 | 0 / 16 (1 fall time differs) | 71 s |
| trained-like B sighted (ckpt) | **16 / 16** | 16,000 rows, 0.0, **0 bits** | 16,000 = 16,000 | 0 | 14 | 193 | 0 / 16 | 79 s |
| trained-like B blind (ckpt) | **16 / 16** | 15,891 rows, 0.0, **0 bits** | 15,891 = 15,891 | 1 | 13 | 191 | 0 / 16 (1 fall time differs) | 76 s |

**What "identical" covers.** Each row is EO.run's whole row, compared as `json.dumps(sort_keys)`: fell_s, opp_fell, drift, ring,
every HitMeter count / n_both_up / window, opp_fall and both_up_iv. This is the same identity fb_m6_check R1 uses.

**The check has power.** The control replica is driven by the float32 torch forward, which is within 2.9e-6 of the ONNX. It is
identical on **0 of 64** bouts. So a per-action difference of a few ulp is enough to break bout identity, and 16/16 is not trivial.
(At 1-3 s the control still matched, so the 20 s length matters.)

**What else the 4 configs cover:**
- the blind (`kp_mode zeros`) path through the replica, which M6 R1 had not exercised (it ran G and H only);
- falls inside the window: 1 per blind config, at the same step on both sides;
- a non-zero residual (stage B), loaded through `build(..., ckpt)`.

## (1)-(4a) re-run: `build\m7t9\m7_check.json`
- `fb_m7_check.py` exits 0 and every ok is true: M7-1, M7-2, M7-3, M7-4.
- The JSON equals the build's `build\m7\m7_check.json` on **every key** except wall_s and the m4rec path.
- Numbers as before:
  - (1) stage A vs P1's M3sHybrid, vs the recorded eval action, and blind twin: **0 bits** on 15,364 rows. Torch nets = compose = rsl_rl 5.0.1: 0. The leg dims get no gradient.
  - (2) B init = A: 0 mismatches everywhere, and the B-init ONNX = the A ONNX.
  - (3) anchor gradient: 6/6 residual tensors, nothing else.
  - (4a) ONNX = torch ref: **max abs 0.0, 0 / 368,736 bits** in each of 4 configs. Contract ok (kp_mode obs/zeros, obs_dim 111, batch 1, cfg = G's = P1's H).
  - Info: the donors' shipped ONNX vs their checkpoints in torch float32 differ by 1.67e-6 (G), 2.38e-6 (A3s) and 2.86e-6 (blind3). This is A9.1's reason.

## Full suite
Frozen command: **408 passed, 3 skipped (env-gated), exit 0, 752 s** (`build\m7t9\suite.log`; 408 = 407 + the new t9b test).
The suite's MUJOCO_LOG.TXT (mjSpec empty-prefix warnings) was deleted; no `Users*` folder in the repo.

## Seeds
- 60400-60415 (role `t9`) were used for the first time, through `fb_seeds.check` (t9b_episodes; train clips only).
- 60000-60015 were read only, from the M4 recording.
- One ledger row was appended ("M7T9"). The `fb_seeds --audit --scan` result is ok.
- No hard_r2 test, val seeds, kfab_202/204/205 or Track 2 data was touched.

## Notes for M9/M10 / selection (facts)
- `t9b` takes blindness from `--blind`. It now refuses a disagreement with the export's manifest and with the ONNX's kp_mode, so a
  blind export cannot be checked against a sighted torch actor.
- The checkpoint's own stored flag (A9.2) is still M9/M10's job, in build().
- t9b runs the replica at batch 1. Its identity does not cover `Batch` (K > 1); that is M6b's rowwise contract.
