# M6b independent verification (A7, A5.1, A6.5, A6.6), 2026-10-30

**Verdict: ok = true.** Every frozen check (as amended) re-ran and matched the build. I found no semantic mismatch, so I fixed nothing and made no code edits. Two items are for the main session (section 4).

Outputs are in `build\m6b_verify\`. The machine was nearly idle (no R0-EO), and I ran at most 4 python workers at once.

## 1. Re-runs

| check | build | verify re-run | match |
|---|---|---|---|
| A7.1 speed (`fb_m6b_check.py speed --ks 64`, 2 workers) | rowwise K64 1.17 / 1.16 / 1.03x EO; plain 1.12 / 1.05 / 0.97x | rowwise **1.10 / 1.04 / 1.05x** EO (G / H / A3; EO 2.05 / 2.28 / 2.13 s per 20-s bout); plain 1.05 / 0.98 / 1.01x | yes, target <= 2x met |
| A7.2 invariance, A3 (`invariance --ctrls A3`, K 64, 64 bouts) | rowwise 64/64 rows and traces; batch 0/64, mlp 64000/64000 rows (1.34e-5), face 29 rows | identical numbers; the 'batch' diff list (seed, N = 1 vs batched outcomes) is identical to the build's | yes |
| Extra (mine): `v_chunks.py`, G and H, REQ_CONF 20 s, 16 bouts | not tested | 4 cases, all bit-equal to N = 1 on 16/16 rows: (1) rowwise K = 4 across several chunks and two run() calls, so module rows are reused and the last chunk is partial; (2) rowwise K = 16; (3) R2 step d (torch64, training reset) at K = 3 on 8/8 rows. Plain K = 16 gives 0/16, as the build reports | new evidence |
| A5.1 (`fb_m5_check.py --reuse`, 2 workers) | G011 true, G012 true, W_eps 1.1609440647930569e-4, final-rule counts | **identical JSON on every key except wall time** | yes |
| A6.5 (`fb_m6_check.py all`, 2 workers) | identity 16/16 + 16/16; diagnostic qpos 2.0e-15, 0/32 over; rdq 3.10e-9 <= 5e-8; G0-11 128/128, A4.3 88.301% | identical (diagnostic `ok` true, `over` []; g011 ok, 'site', 128/128) | yes |
| full suite | 407 passed, 3 skipped | **407 passed, 3 skipped** (962 s) | yes |
| `fb_seeds --audit --scan` | ok | ok (after my 3 ledger rows) | yes |

Bouts with falls were covered: N = 1 A3 has 10 falls among the 64 invariance bouts, and its rowwise rows are still 64/64 equal. So the frozen-finished-bout path is exercised.

The charged counts under the final rule match the build: w0_r4pre/G 41 of 144 (M5's rule 63; the start window removes 22 and W_eps removes 0); hit_scoring G 1 (3), H 2 (4); w0_calib G 1, H 1. Other matching values:
- 24 early edges.
- 0 edges at exactly 0.50 s.
- W_eps `frozen_ok` true.
- No edge with 0 < W <= W_eps.
- f32 and f64 edges are identical.

## 2. Code read against the source

**Batch vs EO.run, Bout.step and run_bout:**
- The step order is the same: ctrl on the live bouts' stacked obs (as EO.run:184), then HybridDelayedAction, d.ctrl, mj_kinematics, the opponent (clock, AimOp, driver), the substeps with hooks, fell_eval, the account, _face/_hold, obs, OppDown, then the book.

**Shared-model DR:**
- `_dr(B)` rewrites all seven DR FIELDS before each bout's mj_kinematics and mj_step. The hooks (HitMeter, ContactTail, Sensors) run inside that window.
- Outside the window, the code uses only data fields: _touching, fell, _face/_hold, proprio, and the kinematics twin's xpos/xquat. None of these reads mass or ipos.
- At reset, DR.write precedes the bout's own mj_forward, the same order as Bout.reset.

**Training-reset rng order** (Batch._reset vs Bout.reset):
- The generator draws in the same order: draw_latency, then heading, then DR.sample, then t0, then obs noise. The explicit noise equals EvalActorObs.compute's default draw.
- Confirmed bit-equal by the v_chunks R2-d case.

**rowwise_face:** it is TrackerDriver.face (fb_opponent.py:152-155) operation for operation. Why only `wrap` differs at batch K:
- Its sin/cos outputs are contiguous, so torch's vectorised atan2 kernel runs at n >= Vec width.
- At n = 1 (tail), the scalar kernel runs.
- `want`'s atan2 and kp_state's heading take strided inputs, so they run the scalar loop.

This agrees with the build's op probes and with the 0 differences for kp_state.

**fb_m6_run:** batch_of = (64, True) only for torch backends. batch and rowwise are in cmeta (checked by chunk_ok) and in the manifest. The numpy (eval) and R2 (b), (c), (e) confs stay at N = 1.

**A5.1 in fb_rewards:**
- `W > eps` and `t > START_S + 1e-9`. In float32, 0.5 + 1e-9 rounds to 0.5, so the step ending at 0.50 s is not charged in either dtype.
- `t` is the step's end, both in the replay (post time) and in mjlab's reward call, which comes after episode_length_buf += 1.
- eps defaults to the frozen W_EPS and refuses 0.
- W_eps is measured over all 141,556 live steps (the "over the 144 bouts" wording) with the account and ring in each dtype on the same recorded rows. The eps argument does not enter W.

**A6.5 in fb_m6_check:**
- On the diagnostic configuration (actor onnx, ref eval), every same-state key is gated at 1e-9, except aim_rdq and obs132_rdq at 1e-9 x 50.
- mlp_vs_graph64 is not gated: it is the deployed TrackerMLP, which M2 G0-1 covers.
- The deployed configuration is reported, not gated, as A6.5 says.

**A6.6:** DEPLOYED_VMODE = 'site' (palm/foot site velocity), and G0-11 passes with it.

## 3. Hygiene
- The only diffs are in workstream fb_* files, plus the new `scripts/fb_m6b_check.py`. No run 1-3 file changed.
- `gpu/fb_opponent.py` M is skip_idle (M6a/M8 lane), not edited here.
- Nothing was committed or stashed.
- Line endings: LF files stay LF. fb_rewards.py and test_fb_rewards.py were CRLF in the working copy and stay CRLF; git normalises them.
- Seeds were used only in roles identity, hit_scoring and w0_r4pre (recorded, no simulation). I added 3 ledger rows ("M6b verify"), and the audit is ok.
- Sealed data was not touched: no 21000-23255, no kfab_202/204/205, no Track 2.
- The suite created a MUJOCO_LOG.TXT in the repo (empty-prefix attach warning, 16:23). I deleted it. No `Users*` folder exists.

## 4. For the main session (not failures)
1. **ROWWISE = True is a reading of A7.1, not its letter.** A7.1 says the tracker runs "once per step on the batch of K". Rowwise runs the tracker MLP and face's atan2 one row at a time; everything else is batched. That makes R-EQ judge exactly the M6a-verified N = 1 numerics.
   - The plain batch (closer to the GPU's batched float32, and what A7.2's rationale describes) is one flag away: `ROWWISE = False`. That variant changes 1 / 8 / 12 of 64 bout outcomes (G / H / A3) through chaos, and it is equally fast.
   - Either variant is consistent with A7.2 ("reported, not required to be bitwise"). The main session should confirm which one R-EQ judges before the first look.
2. **R2 step d (torch64) at K = 64:** float64 transcendentals at vector width, which my K = 3 test does not reach, could still differ from N = 1. Step d is unpaired and report-only, so this is acceptable under A7.2. The R-EQ conf (torch32) is the one proven bitwise.
3. **A8 (pod) supersedes the laptop 4-worker wall-time estimates** in M6b_NOTES section 5. Two consequences:
   - R-EQ must use the pod's r0eo_pod reference. That wiring is outside M6b's scope and is not checked here.
   - Memory is about 1.3 GB per K = 64 worker, about 42 GB on 32 workers, which fits in 128 GB.
4. The R2 (e) account scorer runs at N = 1 on the numpy backend (about 1.8 h on the laptop, less on the pod), as the build reports.
