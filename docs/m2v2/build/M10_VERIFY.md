# M10 independent verification (2026-10-30)

**Verdict: ok = true after four fixes.** Every frozen M10 check (plan 3 row M10, as amended by A5.1, A7, A9.2, A10 and A11) was re-run here and passes. The blocking plumbing smoke is bit-identical to the builder's run, and so is the report-only canary.

Line-by-line review found four real defects in the pod evaluation code and the pod script. All four are fixed and re-run:
- **F1:** the P3 survival bar had its sign flipped.
- **F2:** P1-Hit's GPU dump took the wrong normal force on pyramidal contacts.
- **F3:** P1-Hit's onset match used a 1e-6 s tolerance, finer than the float32 account's time step.
- **F4:** P3's bars did not stop P4-P6.

Two integration gaps remain open. They predate M10 and are outside its frozen check, but they need a decision before the pod (section 4):
- T1 is never wired into training.
- The GPU glue cannot run here.

Nothing was committed. No threshold or margin was changed. Every process used at most 2 workers.

Outputs: `build\verify_m10\`. That folder holds plumbing\, canary\, the dry_*.json files, the logs, suite.log, suite_final.log and the probes.

## 1. Frozen checks, re-run

| check | command / evidence | result |
|---|---|---|
| Plumbing smoke (blocking) | `python -I scripts/fb_cpu_smoke.py plumbing --out build\verify_m10\plumbing`, exit 0, 128 s | **All 10 checks true.** Everything except wall time is **bit-identical** to `build\m10\plumbing\plumbing.json`: every row's reward and losses, the magnitudes, knockdowns, metrics and checkpoints' checks. |
| no NaN | plumbing.json `nan` | 0 |
| legs bit-frozen A0 / A | `legs` | Frozen every iteration (state equal to model_7496.pt). No leg gradient (asserted in every update). Leg deviation 0.0. |
| residual exactly 0 at B start | `b_start` | Residual max abs 0.0, last layer zero, B action == A action (torch.equal), leg std 0.15, res_l2 over B iteration 1 = 0.0 |
| value loss falls >= 30% over A0 | `value_loss` | 0.866 -> 0.343, a **60.4%** drop |
| every term fires, right sign, within 2x of 2.4 | `magnitudes` | All 14 ok. x = 0.903 (head_clean), 0.932 (block), 0.984 (down_neutral), 0.995 (res_l2), 1.000 for the rest. fb_down_approach and fb_down_contact fire on the scripted knockdowns (Q 4.28 m, 198.7 N s). |
| knockdown rule (W_eps, start window) | `knockdowns` | 15 edges: charged 8 (7 scripted + 1 natural), start_window 3, w_floor 3, land_gate 1, other 0. Charged equals the recomputed rule on every edge. Injected 0.5 / 0.9 x W_EPS blocked 3/3, 2 x W_EPS charged 3/3. -2.4 per charge. |
| act_delay / k constant per episode | `switches` | 0 over 56 episodes |
| metrics populated | `metrics_total` | kd edges 15 / charged 8, tangle 1244 N s, lean 1832 deg s, W 409 J, downed 21 s |
| blind flag (A9.2) | `blind` | A0 / A / B False, twin True. Cross-twin loads and export builds are refused both ways. |
| head hits in the env's own account | `head` | Account clean head 26, landed 44 (eval HitMeter 31) |
| Canary (report only, m19) | `fb_cpu_smoke.py canary --a0 <verify A0>`, 16 envs x 400 A iterations, 69100-69320 | `canary.json` **identical** to the builder's (every field but wall / path). Return 0.1180 -> 0.1158, head clean per live min 2.188 -> 2.190, value loss 2.66 -> 0.029, 0 NaN, legs frozen. No learning, which triggers nothing (m19). |
| Minimal PPO = rsl_rl 5.0.1 | tests/test_fb_smoke.py::test_returns_and_update_are_rsl_rl_ppo (rsl_rl's own PPO.compute_returns / update on the same rollout) | Passes (not skipped). Returns, advantages, losses, adaptive lr and every parameter are bit-equal. |
| Seeds | `fb_seeds.py --audit SEED_LEDGER.md --scan` | ok: 20 roles disjoint, ledger consistent, 54 task inputs clean. Verify rows appended. |
| pod_fb.sh | `bash -n`; `FB_DRY=1 FB_FAKE_GPU=1 bash gpu/pod_fb.sh plan`; `fb_m9_check.py --only pod` | Clean, DRY OK, ok (after F4) |
| Dry runs (after the fixes) | `fb_p1.py --dry`, `fb_t1c.py --dry`, `fb_loco_eval.py --dry`, 2 workers each | p1: **ok true**. (1) and (2) pass; (4) replay = live 4/4, counts exact 4/4, **clean agreement 0.962** (was 0.926, see F3), geom map 126/126, bodies 50/50; (5) and (6) pass. (3) is plumbing only. t1c: arms identical, ratio 1.0. loco: runs end to end; the bars are recomputed with F1. |
| Full suite (frozen command) | after F1 (+ test) and F2: **453 passed, 3 skipped, exit 0** (`suite.log`). Final run after F3 / F4 (+ test): `suite_final.log` | See section 5 |

## 2. Fixes (clear bugs; each re-run)

**F1. P3 LOCO survival bar had its sign flipped** (`scripts/fb_eq.py` loco_bars; consumed by M10's fb_loco_eval). Column 0 of `fb_eq.arrays` is *fell*, so `d = st[:,0] - pl[:,0]` already equals survival(pilot) - survival(start). The code tested `-d.mean() >= -0.015 and -d.mean() - z sd >= -0.04`, which is start - pilot.

Effect before the fix:
- A pilot that falls 10 points more **passed** survival and reported `surv_diff` +0.10. With good head-clean bars this gave `proceed`.
- A pilot that fell 4 points less **failed** the bar.

The plan text is unambiguous: "survival (pooled 2304): pilot - start point >= -0.015 and 90% lower bound >= -0.04". The fix uses `d.mean()`, and a new test `test_loco_survival_bar_direction` covers both directions.

critique_checks\gate_power_v3.py's sim_loco carries the same flip. I did not edit it (it is outside the repo). Reproducing its LOCO table with the old and the fixed bar on the same RNG stream (`build\verify_m10\probe_loco_repro.py`) changes 4 cells by 0.001 and nothing else, so 3.1's LOCO power claims are unchanged. The M11 prereg hash must cover the fixed fb_eq.py.

**F2. P1-Hit's GPU dump took one efc row as the contact normal force** (`scripts/fb_p1.py` gpu_hit). The model's cone is pyramidal (`opt.cone` 0). Both robots' foot geoms are condim 3 at priority 1, so any foot contact with the zoned shells, such as a kick or tangled feet, is condim 3. There mj_contactForce's normal is the **sum of the 2 (dim - 1) pyramid rows**, not `efc_force[efc_address]`.

On one replica bout (`probe_pyramid.py`, 62825 condim-3 contacts):
- the summed rows equal mj_contactForce exactly (max diff 0);
- the first row alone differs by up to 707 N.

Without the fix, the replayed HitMeter would mis-call foot landings, and P1-Hit would fail on correct physics. The fix sums the contact's rows (it handles both a 2-D per-row `efc_address` and a 1-D one) and asserts that the rows exist.

**F3. P1-Hit onset matching was finer than float32 time** (`hit_compare`). The GPU account is float32 (`t_end.float()`), so a late onset's t0 is up to 1 ulp off the HitMeter's grid time: 1.9e-6 s at 16-20 s. The 1e-6 match then splits one event into two "disagreements". In the builder's dry run, 52 of 53 landed events matched within 1e-6 and one sat at 1.14e-6 (`probe_hitmatch.py`), so the agreement read 50/54 = 0.926.

Onsets lie on the 5 ms substep grid. The fix matches within `ONSET_TOL = 0.0025` s (half a substep), which is unambiguous and is not a threshold: the 0.95 and 90% bars are unchanged. The dry P1-Hit now reads **0.962 over 53**, and the dry P1 is ok. New test: `test_p1_hit_compare_matches_float32_onsets`.

Note: fb_m5_check.compare (G0-11) uses the same 1e-6 match, so its 0.9675 is, if anything, slightly understated. I did not change it.

**F4. P3's LOCO bars did not gate P4-P6** (`gpu/pod_fb.sh` P3, the line M10 edited). fb_loco_eval exits 3 unless `proceed`, but P3 only logged the failure, so `pod_fb.sh all` went on to the main pairs. Plan section 4 says: "Proceed iff all four hold ... stop and report ... stop and diagnose".

P3 now calls `fail loco` (pack and exit 1) after logging the decision. A crash also stops it, because the bars cannot be judged. `bash -n` is clean, the dry run ends DRY OK, and `fb_m9_check --only pod` is ok.

**Hardening of the unexecuted GPU glue** (same file, gpu_hit). Substep times came from a global hook counter `j % 4`. That counter misaligns if any `scene.update` runs outside the decimation loop; the task itself guards against this with its `rot` roll. A misaligned counter would shift every dumped time by multiples of 5 ms, and every match would then fail.

The counter is now reset at each `env.step`. It dumps only during the step's first DECIM updates, and it asserts that at least DECIM hooks ran. This follows mjlab's step order: `sim.step(); scene.update()` per substep, with `episode_length_buf += 1` after the loop.

## 3. Review notes (no mismatch found)

- **SmokeEnv vs fb_replica.Batch._chunk.** The action, kinematics, opponent, physics and hooks run in the same order. The smoke calls OppDown before `_face` / `_hold`, which is harmless because those only change commands. After the resets the outer loops run, then one obs per step. A reset env's first obs is the B3 fresh row (dry B3 asserts pass).
- **Time-outs and alive.** Time-outs are bootstrapped exactly as in rsl_rl's process_env_step. alive = not terminated, where terminated = fell_eval, ring_out at 1 m or NaN.
- **Stage restarts.** The A0 -> A restart keeps the optimizer (its stale lr is overwritten at the first minibatch, as in rsl_rl). The A -> B restart is fresh through load_into_b.
- **Independent quantities.** The magnitude check's numpy quantities are independent of the reward code: MuJoCo state, sensor substep forces, the eval HitMeter and a numpy work ring. The 2x reading (reward per unit of the driving quantity) is reasonable.
- **k shares.** 0.29 / 0.21 / 0.50 over 56 episodes. The draw is `randint(3)` on the per-episode generator, so this is sampling noise and not a gate here. P1 (5) checks the shares at the pod's N.
- **fb_t1c.**
  - The T1 swap happens before Batch's rowwise wrap.
  - The conf carries the tracker sha, and chunk metas are asserted.
  - The A8.2 platform refusal is tested.
  - The episodes are fb_eq.t1c_set (A5.3 shards 0-31, role eq).
- **fb_loco_eval.**
  - The episodes are 1152 + 1152 in roles loco_heldout / loco_seen.
  - The blind flag is read from the checkpoint.
  - EO.run is unchanged and wrapped the same way as R0-EO's.
  - A stopped sighted B falls back to A's final checkpoint.
  - The start is the exported ONNX (A9.1).
- **fb_p1 r4g.**
  - Rows come from each world's first episode.
  - secs = up seconds, which matches fb_r0eo's `secs`.
  - kd is the OppDown edge.
  - The T0-only window is re-armed per batch.
  - The gates are fb_eq.r4g_gates (two-sided, so they do not depend on the d sign).

## 4. Open issues for the main session (not fixed: decisions or GPU-only)

1. **T1 is never used in training (pre-M10, pod_fb.sh / fullbody_task).**
   - The training task reads T1 only from `FB_T1_PT`, and pod_fb.sh never sets it. So P3 and P4-P6 train with 100% T0 whatever T1-c says. Plan 0 / 2.2 calls for T0 50% / T1 50% if T1-a..e pass.
   - T1-a, T1-b, T1-d and T1-e have no script. Only T1-c (fb_t1c) exists.
   - The P3 LOCO pilot should use T1-LOCO, which is also not wired.
   - Needed before P2 or P3: a T1-a..e gate driver, and `FB_T1_PT` set from it (T1 for the main pairs, T1-LOCO for P3).
2. **GPU glue is unexecuted** (builder issue 3). gpu_env, gpu_alone, gpu_r4g with the B3 hooks, gpu_hit (Warp fields `nacon`, `contact.{geom, worldid, pos, frame, efc_address, dim}`, `efc.force`, `data.cvel / xipos / subtree_com`) and runner_smoke run first on the pod, inside P1's fix cycle. F2 and the counter fix are reasoned from MuJoCo / mjlab semantics and from a CPU check of the force formula.
3. **P1-Hit margin.** Clean agreement is 0.962 on the 53-event dry sample and 0.9675 at G0-11, against the 0.95 bar. 32 GPU worlds is a small sample, so a noise-sized fail is possible; 3.2's reserve (a second GPU world set) covers it.
4. **Pod R0-EO** (builder issue 1). Upload `podpull\r0eo_pod.tgz` unpacked to `$W/r0eo_pod`, or set FB_R0EO. P1-R4g refuses to run without it.
5. **Pod CPU cost.** LOCO now runs the start and G on the pod beside the pilots (A8.2), which doubles the P3 eval wait from about 0.3 h to an estimated 0.6-1 h. T1-c recomputes its T0 arm on the pod. Both follow A8.2's same-platform rule.
6. **Canary.** The adaptive lr is pinned at its 1e-5 floor from iteration 1: per-minibatch KL exceeds 0.02 after Adam's first steps at the donors' arm std. This is report only, but keep it in mind for WATCH6 E4 at P4.
7. **critique_checks\gate_power_v3.py** keeps the F1 sign flip. Its LOCO table is unaffected (to within 0.001), but fix it or footnote it before the M11 hash.

## 5. Files

Changed in this verification (all LF):
- `scripts/fb_eq.py` (F1, 2 lines)
- `scripts/fb_p1.py` (F2, F3, the counter fix)
- `gpu/pod_fb.sh` (F4, P3 line)
- `tests/test_fb_smoke.py` (2 new tests)

The suite before F3 / F4: 453 passed, 3 skipped, exit 0 (`build\verify_m10\suite.log`). The final suite after every edit is in `build\verify_m10\suite_final.log` (see the result line appended below). MUJOCO_LOG.TXT was removed from the repo, and no `Users*` folder exists in the repo root.

**Final suite (frozen command, after every edit but the gpu_hit counter): 454 passed, 3 skipped, exit 0** (`suite_final.log`; the skips are the three
FB_* env-gated tests). tests/test_fb_smoke.py re-run after the counter edit: 19 passed.
