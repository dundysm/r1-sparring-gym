# M4 build notes (M2V2_PLAN_v3 section 3, M4: G0-5, G0-6, G0-6b, G0-7, G0-8, G0-9, G0-10), 2026-10-30

**Status: M4 PASS.**
- `scripts/fb_m4_check.py` exits 0 in 97 s, with the recording taking 40 s of that.
- Every `ok` is true: G05, G06, G06b, G07, G08, G09, G010.
- All four G0-6b negative controls fail, as they must.

**Rules kept:**
- Every check ran as frozen. No threshold was changed, and no check needed an amendment.
- Only new, untracked files were added. The tracked `git diff` is empty.
- Nothing was committed, stashed or branch-switched. The branch is feat/sparring-m2v2.
- CPU only, with at most 4 worker processes. $0 spent.
- No `Users*` folder exists in the repo. The suite's MUJOCO_LOG.TXT (the known empty-prefix warning) was deleted.
- Full suite: **300 passed, 2 skipped** (the two env-gated tests; both pass when their env vars are set).

**Seeds:** block 60000-60015 in role `record` (the M4 recording, re-run in role). I appended one ledger row ("M4").

## Files (all new)
| file | what |
|---|---|
| `gpu/fb_obs.py` | **Keypoints** (KeypointSource('truth') in batched torch, on the opponent's real links):<ul><li>`kp_state(xpos, xquat)` gives the T-frame points, the relative yaw and the FOV mask from the poses of 6 bodies (`body_names()`).</li><li>`fma` is an exact FMA built from Dekker's product and TwoSum. `rowmat` is numpy's small-matmul order on this machine, an FMA chain, in float64; float32 uses a plain matmul.</li><li>`np_wrap` is numpy's `%`. `pack` and `KeypointHistory` match keypoints.py.</li></ul>**`proprio83`** is R1HitEnv._raw_obs under the mjlab contract.<br>**`EvalActorObs`**:<ul><li>a ring of 3 clean 111-d rows and one lag k per env and episode, shared by all 111 dims;</li><li>`reset(ids)` only flags the lazy pre-fill and drops the keypoint history;</li><li>`compute()` pre-fills flagged envs with their fresh row, then returns ring[k] + noise. Uniform noise is added after the delay, and the keypoint half-widths are 0.</li></ul>It also has `k_switches`, `clean` (for the critic) and `blind`. Constants are duplicated from r1gym/keypoints.py and asserted equal in tests. mjlab's own obs delay stays off. |
| `gpu/fb_actions.py` | **`HybridDelayedAction`:**<ul><li>clips and casts to float32;</li><li>applied = where(act_delay, a_prev, a);</li><li>targets = home + scale x applied, with no EMA;</li><li>`applied` fills the actions slot;</li><li>act_delay is read from the latency buffer at reset, and `switches` counts any mid-episode change.</li></ul>It reads no env observation. It replaces mjlab's DelayedActuatorCfg and so avoids the DelayBuffer lag-reset and clamp trap.<br>**`m3s_input` / `arm_input` / `compose`:** H(arms)/M3s, as in P1's M3sHybrid. |
| `gpu/fb_events.py` | **Latency:**<ul><li>`Latency` holds the env-level k, act_delay and draws.</li><li>`draw_latency` / `fb_draw_latency` is the reset event listed first: k ~ U{0,1,2} and act_delay ~ Bernoulli(0.5).</li><li>`B3_ORDER` and `ResetLog` record and check the reset order.</li></ul>**Fall rules:**<ul><li>`up_z(quat)` is mju_quat2Mat's [2,2] term order.</li><li>`fell_eval`: z < 0.45 or up < cos 60 deg. `UP_MIN` is the eval's own float.</li><li>`opp_down_now`: up < 0.5 or z < 0.25.</li><li>`OppDown` has `now` (non-sticky), `ever` (sticky) and `age`, plus the knockdown edge.</li></ul> |
| `gpu/fb_dr.py` | `RANGES` are ENV:187-199. `FIELDS` lists the 7 model fields the eval writes.<br>`DRIndex` builds the eval's index sets.<br>`sample` (torch) and `sample_np` (the eval's draw order, act_delay included) draw the parameters.<br>`write(fields, nominal, ix, p, ids)` resets each field to nominal, then applies the eval's operations. It writes only FIELDS and only for envs `ids`, with no set_const.<br>`model_fields(m)` exposes an MjModel's fields as N = 1 torch views. |
| `scripts/fb_m4_check.py` | **Recording:** 16 record bouts x {G, P1's H(A3s)/M3s} through eval_opponent.run, with read-only hooks:<ul><li>an R1HitEnv subclass;</li><li>a DelayedNoisyObs subclass that has the same arithmetic and rng draws and keeps the exact noise term;</li><li>a KeypointSource subclass;</li><li>the controller's act.</li></ul>Each bout is also run without the hooks, and the rows are asserted identical. The controller's input is asserted to equal the delivered obs, and G's states equal M2's recording bit for bit.<br>**Gates:** G0-5 to G0-10 and the G0-6b negative controls. `--p1` points at P1's p1_eval.py. |
| `tests/test_fb_obs.py` | 9 test functions (11 cases): the contract vs keypoints.py; fma exact (Fraction) and rowmat = numpy matmul; np_wrap; kp_state = true_state bit for bit on 400 random states with glove-elbow; float32 keypoints; the delayed block with resets vs KeypointSource for k = 0, 1, 2; EvalActorObs vs DelayedNoisyObs with random resets and draws; reset only flags, with noise after the delay and blind rows; proprio83 = _raw_obs on 200 random states. |
| `tests/test_fb_actions.py` | 4 tests: the delay semantics and targets as R1HitEnv.step; act_delay read at reset only, with clip and cast; reads no obs; compose = P1's logic. A fifth runs against P1's own class when FB_P1_EVAL is set (skipped otherwise). |
| `tests/test_fb_events.py` | 6 tests: the latency shares and that only the reset envs are drawn; the B3 order; constants and boundaries; up_z = xmat[8]; the rules = R1.fallen and EO:86 on 2000 random states; the non-sticky flag. |
| `tests/test_fb_dr.py` | 5 tests: the eval's index sets; the writer = _dr_sample_into_model on 25 seeds; torch ranges; only defender FIELDS change, invweight0 stays nominal and mj_setConst changes it; per-world envs. |
| `tests/test_fb_m4.py` | 4 tests on one recorded G bout (60001, 2 s, role `record`): the recorder refuses other blocks; the pipeline equals the eval; B3 on 8 resets, with the eager and no-forward controls failing; the fall rules. |

Recording (scratch, 80 MB): `...\fullbody\build\rec\fb_m4rec_60000-60015.npz`.<br>
Check output: `...\fullbody\build\m4_check.json`, `m4_check.log`.<br>
Suite: `...\fullbody\build\m4_suite.log`.<br>
Build smokes: `...\fullbody\build\m4\dev_rec.npz` and `probe_*.py`.

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, S = ...\scratchpad, B = $S\m1m2\fullbody\build)
```
$PY -I scripts/fb_m4_check.py --out $B\m4_check.json --m4rec $B\rec\fb_m4rec_60000-60015.npz --rec $B\rec\fb_rec_60000-60015.npz --p1 $S\m1m2\gap\run4pre\P1\p1_eval.py --workers 4   # exit 0, 97 s
FB_P1_EVAL=$S\m1m2\gap\run4pre\P1\p1_eval.py $PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests/test_fb_obs.py tests/test_fb_actions.py tests/test_fb_events.py tests/test_fb_dr.py tests/test_fb_m4.py   # 31 passed, 7 s
$PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests          # full suite: 300 passed, 2 skipped (FB_SEED_LEDGER, FB_P1_EVAL unset), 474 s (m4_suite.log)
FB_SEED_LEDGER=... FB_P1_EVAL=... $PY -m pytest ... tests/test_fb_actions.py tests/test_fb_m4.py tests/test_fb_hygiene.py   # 62 passed (both skipped tests run and pass)
$PY -I scripts/fb_seeds.py --audit $S\m1m2\fullbody\SEED_LEDGER.md --scan        # ok: 19 roles disjoint; ledger consistent; 17 task inputs clean
```

## Recording
- 32 bouts (16 per controller), 30,808 obs steps and 30,776 post-step states.
- Falls: G 60014 at 8.24 s and H 60014 at 7.28 s.
- Opponent knockdowns: G 60004 and 60005, and H 60013.
- k is drawn per bout in {0, 1, 2}. act_delay is on in 9 of 16 bouts (18 of 32).
- The hooked rows are identical to the plain rows on every bout.
- **G's states are bit-equal to M2's recording** (`same_as_m2_recording` true).

## Results (thresholds as frozen; `build\m4_check.json`)
| gate | rule | result |
|---|---|---|
| **G0-5** | keypoint block vs KeypointSource('truth', delay=k), k in {0,1,2}, 2000 recorded states, pre-fill included: positions, masks, yaw bit-equal; velocities <= 1e-6; glove-elbow 0.194862 m | **PASS.**<ul><li>2000 states: 8 segments x 250 with resets at 0 and 125, including the knockdown and fall bouts. That is 6000 rows over the 3 ks.</li><li>Mismatches: pos 0, mask 0, yaw 0. Velocity max diff **0.0**.</li><li>On the eval's own delivered block, over every H step (15,380): 0 / 0 / 0 and 0.0.</li><li>Glove-elbow is 0.1948622324 m, both arms, every state (min = max to 1.5e-15), so it is within 2.4e-7 of 0.194862.</li></ul> |
| **G0-6** | obs lag constant per episode and shared by all 111 dims; output = DelayedNoisyObs with noise 0 | **PASS.**<ul><li>32 bouts batched, 30,808 steps.</li><li>proprio83 vs `_raw_obs`: 0 mismatches.</li><li>With **the eval's own noise replayed**, the output equals the delivered 83 on every step (0 mismatches) and the delivered keypoint block on every H step (0).</li><li>With noise 0, the output vs r1gym's `DelayedNoisyObs(k, zeros)` on the same clean rows: 0 mismatches. out(t) = clean(max(0, t-k)) in all 111 dims: 0 mismatches. ks {0, 1, 2}, k_switches 0.</li><li>Own noise: inside the half-widths, exactly 0 on command, phase, actions and the keypoint dims, and reaching 0.999 of the half-width.</li></ul> |
| **G0-6b** | B3 reset order in the replica, 64 resets incl. after a fall and after a knockdown: ring rows = fresh row; keypoints = placed opponent's; k and act_delay = the event's draw; no previous-episode rows | **PASS, 64/64 on every item.** The resets are 2 after a fall, 3 at a knockdown's first step, 3 at 10 steps later and 56 random; ks {0, 1, 2}; act_delay share 0.469.<ul><li>Every ring row equals the fresh row: `R1HitEnv._raw_obs` plus `KeypointSource.reset` on the placed, forwarded state.</li><li>Output = fresh row; keypoint block = the eval's on the placed opponent; k_ep = act.delay = the draw.</li><li>No previous-episode row; glove velocities 0; a_prev = applied = 0.</li><li>Opponent targets are finite and non-zero; the call order is B3_ORDER.</li></ul>**Negative controls, all fail:**<ul><li>eager pre-fill: ring_fresh 0/64;</li><li>latency drawn after the manager resets: draw_ok 27/64;</li><li>no pre-fill: ring_fresh 0/64, no_prev_rows 0/64;</li><li>obs before the forward: kp_placed 0/64.</li></ul> |
| **G0-7** | applied(t) = a(t-1) when delayed, first delayed step = 0, actions slot = applied; the action term reads no env obs | **PASS** on 30,808 steps:<ul><li>applied vs the eval's `last_a`: 0 mismatches;</li><li>targets vs the eval's `d.ctrl`: 0, bit for bit;</li><li>the eval's next-obs actions slot vs applied: 0;</li><li>the first delayed step is 0 in 18/18 delayed bouts; act_switches 0.</li></ul>Reads no obs: a stub latency buffer raises on any attribute but act_delay, and none of the 30 AST identifiers contains "obs". |
| **G0-8** | fall rule and opp_down_now vs R1.fallen / EO:86 on 1e4 states | **PASS** on 40,776 states.<ul><li>**Recorded post-step states, as the eval read them (30,776):** fell 0 mismatches (2 falls), also 0 via xmat; opp_down 0 (1,357 down states); up_z vs xmat[8] bit-equal.</li><li>**Rebuilt from the non-sticky flag:** EO's both-up intervals 0 mismatches over 32 bouts; first-fall time 0; opp_fell 0; 3 knockdown edges; 1 up-again.</li><li>**10,000 random states:** fell 0 mismatches (6,524 fallen), opp_down 0 (4,933 down).</li><li>The constants equal the cfg and EO.</li></ul> |
| **G0-9** | DR ranges = ENV:187-199; opponent untouched; invweight0, subtreemass, constraint regularisation nominal after a write (no set_const), as in the eval | **PASS.**<ul><li>**Writer vs the eval's sampler:** with the eval's draws (the 16 bouts' env seeds plus numpy 0-199), every field equals `R1HitEnv._dr_sample_into_model`'s (0 mismatches), act_delay too (0), and act_delay matches the recorded bouts' (0). The index sets equal the eval's.</li><li>**Ranges:** 1e5 torch draws stay inside every range and reach within 1% of both ends.</li><li>**What changes:** a full MjModel diff shows only the 7 FIELDS change, with 0 rows off the defender. With per-world fields, untouched envs stay nominal and opponent entries stay nominal.</li><li>**No set_const:** dof_invweight0, body_invweight0, body_subtreemass, dof_M0 and actuator_acc0 equal nominal (ours and the eval's).</li><li>**Constraint regularisation:** at a contact state (60004 step 761, 14 contacts, 69 efc rows), efc_R, efc_D, efc_KBIP and efc_diagA equal the eval's DR model's bit for bit.</li><li>**Negative control:** mj_setConst changes all 5 constants and efc_R.</li></ul> |
| **G0-10** | hybrid composition and M3s = P1's p1_eval.py hybrid on recorded obs (exact) | **PASS** on 15,364 recorded H rows (batch 1, as recorded):<ul><li>`compose` with the same ONNX actors vs P1's `M3sHybrid.act`: 0 mismatches;</li><li>`m3s_input` vs `g_in`: 0 mismatches.</li></ul>Scale and clip are the env's (clip 1e9). |

## Notes for the main session (facts; nothing decided here)
1. **numpy's small matmuls on this machine are FMA chains.**
   - `to_torso` and `in_view` compute `(P - x) @ R` in r1gym/keypoints.py.
   - A plain float64 torch matmul differs from them by 1 ulp in 28% of elements.
   - `rowmat` emulates the FMA exactly (Dekker plus TwoSum, verified against Fraction), so the float64 keypoints are bit-equal, and so is the float32 block.
   - If the pod's or another machine's numpy rounds differently, `test_fma_exact_and_rowmat_is_numpys_matmul` fails there. This only matters for CPU parity; the GPU path is float32.
2. **Staleness matches by construction.**
   - mjlab's termination and reward managers read derived quantities one substep stale (`manager_based_rl_env.py` step: they run before the single `sim.forward()`).
   - That is exactly what R1.fallen and EO's opponent test read after `mj_step`. G0-8's recorded states are those stale values.
   - Keypoints and obs read fresh kinematics after the forward, as KeypointSource's twin does.
3. **G0-6b "replica" is a replay in the eval's own two-R1 MjModel,** because M6's `fb_replica.py` does not exist yet.
   - The previous episodes are the recorded bouts. Placement uses TrackerDriver.reset with clean starts at the hard distances.
   - The outer-loop command is 0 at the reset obs in both our row and the reference, because the outer loops are not M4.
   - M6 and the P1 asserts should reuse `fb_events.B3_ORDER` and `ResetLog`.
4. **The first-ever reset in mjlab:** `env.reset()` computes the obs after `_reset_idx` and the forward, with placement in the command reset. So EvalActorObs pre-fills from the placed opponent even before the wrapper's `command_manager.compute(dt=0)`.
5. **For M9 (mjlab adapter):**
   - **Observation reads:** base_ang_vel must be the free joint's qvel[3:6] (body frame), and gravity must come from the root qpos quaternion, as `_raw_obs` does.
   - **DR fields:** the 7 FIELDS must be expanded per world.
   - **Delays off:** mjlab's `delay_max_lag` must be 0 on the actor terms, and no DelayedActuatorCfg is used.
6. **float32 GPU rounding (info).**
   - The eval adds float64 noise to the float32 83 and the ONNX then rounds it. A float32 EvalActorObs rounds once, so the two can differ by 1 float32 ulp.
   - The M3s rewiring on float32 rows differs from P1's float64 path by up to 3.4e-7.
7. **G0-10 composition and the torch actors.** G0-10 checks the composition with the shipped ONNX actors. The torch `.pt` actors vs ONNX is M7's T9.
8. **Ledger row:** "M4", block 60000-60015, role `record`. The build smokes (60001/60004 at 2 s; 60001/60004/60005/60014 at 20 s) are in the same row. G0-6b's placement and latency draws, G0-8's random states and G0-9's replays use generator seeds, not episode seeds, and no new bout.
