# M6a independent verification (M2V2_PLAN_v3 3 M6 part 1, 3.1, 3.2; A4.3, A5.3), 2026-10-30

**Verdict: ok = false.** One frozen check cannot pass as specified: the torch float64 qpos and same-state checks, both <= 1e-9. I proved this independently with the eval's own code (section 2). It is not a code defect, and the decision belongs to the main session. Every other M6a check passes on my re-run, bit-equal to the builder's outputs. I found one plan mismatch and fixed it: A5.3 had frozen LOOK_RULE / T1C_RULE, but the code still said None (section 4). No threshold or margin was changed.

## 1. Re-runs (my output: `build\verify_m6a\`)
| check | result |
|---|---|
| R1 identity, `fb_m6_check.py all` (2 workers), 60100-60115 | **PASS**: G 16/16, H 16/16, whole row bit identical to `eval_opponent.run` (126 s) |
| f64 | Per-bout JSON **equal** to the builder's `m6_f64.json`: deployed torch64 qpos max 8.52e-7, 30/32 bouts over; onnx+evalref max 2.0e-15, 0/32 over. Same-state: defender-side items 0.0, aim rq 2.19e-11, rdq 1.095e-9, obs anchor 2.69e-7, targets vs ONNX32 2.11e-6, MLP vs graph64 7.1e-15 |
| G0-11 on the replica, 60200-60263, deployed input 'site' | **PASS**, equal to the builder's on every per-input entry and the A4.3 probe. 0/64 count-mismatch bouts for G and for H. Clean agreement G 0.9675, H 0.9744. Median head impulse rel 4.4e-6 / 8.9e-6. Rows equal to the M5 recording's EO rows on 128/128. Clean totals (HitMeter vs account) G 323/318, H 276/278, so no systematic bias for the hc gates |
| A4.3 probe | 453,714 contacts; within 1 cm 88.301%, so 'site' under the literal reading. On force-carrying contacts 99.874%. found / force mismatch 0 / 0 |
| `fb_eq.py reproduce` | **equal: true, 0 diffs** against `gate_power_v3_all.json` (806 s) |
| `fb_eq.py dry --trials 20` | Every rule: correct passes 19/20; W1, W2, W2b, W3, W3b, W4, W5 and W6 pass 0/20 each. Mean N equals the builder's (shards_ceil 6038) |
| `fb_m6_run.py all --dry` (1 worker, 6 s), then `req --dry` again | Every stage ran end to end; the resume reran 0 of 3 tasks. REQ_CONF cost about 100 s per 8 x 6-s bouts per worker, which confirms the builder's speed finding |
| `fb_seeds --audit --scan` | ok: 20 roles disjoint, ledger consistent, 36 task inputs clean |
| full suite | see section 6 |

## 2. f64 at 1e-9: impossible as specified (independent proof)
Probe `verify_m6a\probe_f64.py` (`probe_f64.json`; 60100-60107, G, 0.3 s; qpos vs the numpy eval before the first shell contact):

| configuration | max qpos diff | bouts > 1e-9 |
|---|---|---|
| **the eval's own code, tracker graph evaluated in float64** (Tracker.s swapped, nothing else changed) | 6.4e-7 (step 1: 2.8e-8 to 1.0e-7) | **8/8** |
| torch64 deployed (MLP, fb_motion reference) | 6.4e-7 | 8/8 |
| torch64, eval ONNX session, fb_motion reference (only (b) differs) | 4.1e-7 | 8/8 |
| torch64, MLP, eval reference (only (a) differs) | 6.4e-7 | 8/8 |
| torch64, eval ONNX session + eval reference | 7.8e-16 | 0/8 |

- (a) Precision alone breaks 1e-9. The eval's float32 tracker graph, compared with the same graph in float64 and everything else the eval's own code, already moves qpos by up to 1e-7 at step 1. No float64 implementation can meet 1e-9 against a float32 ONNX.
- (b) fb_motion's anchors alone also break it. They are exported from the clips by design (the tracker ONNX is a banned task input).
- (c) rdq: the eval computes `rdq = rdq + (delta - prev) / dt` (r1gym/aim.py:106, dt 0.02). The aim-q floor of 2.19e-11 therefore becomes 50 x 2.19e-11 = 1.095e-9 in rdq.
- Everything else is exact. With (a) and (b) removed the trajectories agree to 1e-15, and every defender-side module is bit-equal on the same states.
- The builder's options (a) gate the diagnostic config, (b) gate the deployed config at a floor, and (c) gate rdq relative to its own scale are all valid. **Not decided here.**

## 3. Line-by-line review (against EO.run, R1HitEnv, Tracker.act, Aim.reference, fb_r0eo, gate_power_v3, plan 3 / 3.1 / 3.2 / A4 / A5)
- **Replica numpy path vs R1HitEnv.reset / step and EO.run.** The order matches: rng, robot reset, DelayedNoisyObs, DR draws, opponent placement, forward, Aim, hooks reset, face/hold, obs.
  - Step order: clip, act_delay swap, ctrl, `mj_kinematics`, tracker.act, 4 substeps with xfrc cleared, fall test, face/hold, obs.
  - `FightReplay.apply` is a no-op for a tracked opponent, so omitting it is correct.
  - Per-step bookkeeping (opponent fall, ContactTail snapshot, both-up and down runs, ring, spacing) and the row equal EO.run's. The identity check confirms this.
- **Torch path.** OppDown is updated on the post-step (one-substep-stale) quantities and asserted equal to EO:86 every step (M4 hazard 3). `last_a` = applied action. The eval-reset draws follow the eval's rng order. Same-state proves DR, obs, noise, keypoints, action, fall and opp_down bit-equal.
- **AimOp `skip_idle`** (the only edit to a committed file). When no limb has a live intent, `d5 = where(has, ..., 0)` is 0 whatever the IK returns. The change is output-preserving; the test showed bit-equal qpos.
- **Switches.**
  - Hulls: meshes are swapped on both robots' shells only, and masses and inertias are unchanged (tested).
  - Self-collision bits (8, 9): defender capsules collide with each other and the floor. Contacts with shells and opponent capsules are unchanged (worked through the bitmasks).
  - Solver: iterations only.
  - Training reset = T0-nominal row (eval distances, clean-grid t0, random heading). It needs a torch backend, so R2 (d) is torch64 + train reset, reported unpaired (P1-R4g form).
- **fb_eq vs gate_power_v3.**
  - `req_gates`, `tost`, `log_ratio_u`, `rate_u`, `pfk_u`, R5, `t1c_gate`, `r4g_gates` and `loco_bars` match operation for operation. reproduce() also shows this.
  - `paired` asserts (seed, clip, distance, t0, k, act_delay).
  - kd = opponent down at any time (`fb_r0eo.bout_row`), so the first-0.5-s knockdowns are counted, as A5.1 requires for the R-EQ kd gates.
  - `classify` follows 3.2 (one-sided: conclusive iff the upper bound is <= 0).
- **fb_m6_run.**
  - Looks: each look needs its shards for all three controllers, and the R0-EO rows of the same shards are loaded seed-sorted (`R0.load`).
  - t1c shares the req G shards under 'shards'. Manifest and chunk meta refuse a mismatching resume. Seeds are checked through `fb_seeds` before any episode.
  - A 3.2 rerun stage on the reserve block (100000-112287) is not in the driver; it is not required by M6a.
- **Seed and data hygiene.** Only 60100-60115 (identity) and 60200-60263 (hit_scoring) were simulated, plus 70000-82287 / 67000-67767 as episode lists without simulation. No val / test / kfab_202-205 / Track 2 content.

## 4. Fix applied
- `scripts/fb_eq.py`: `LOOK_RULE = "shards_ceil"`, `T1C_RULE = "shards"`, with docstrings.
  - Plan **A5.3** (in the frozen plan, decided before any replica or gate data) sets the R-EQ looks at the first 22 / 43 / 64 shards (4224 / 8256 / 12288) and T1-c = shards 0-31.
  - Checked: `look_sets` gives 4224 / 8256 / 12288 on shards 0-21 / 0-42 / 0-63, and `t1c_set` gives 6144 on shards 0-31, exactly as A5.3 says.
  - The builder had left both None as "to be frozen by the main session"; A5.3 already froze them.
- `tests/test_fb_eq.py::test_look_and_t1c_sets` asserts the frozen values and still checks that None refuses (monkeypatched).
- `scripts/fb_m6_run.py`: the comment on DEPLOYED_VMODE gave the probe's share as 89.9%; corrected to the measured 88.3%.
- LF line endings were kept.

## 5. Open items for the main session (none decided here)
1. **f64 gate**: choose among the builder's options (section 2). This is the only reason ok = false.
2. **A4.3 reading**: literal (all contacts, 88.3%) gives 'site', which is deployed now. The force-carrying reading (99.874%) gives 'pos'. G0-11 passes either way. Any GPU use of pos must mask slots with |F| = 0, where pos is (0, 0, 0).
3. **Speed**: REQ_CONF runs about 40 s per 20-s bout per worker. R-EQ is about 55 h expected and about 100 h worst case on 4 workers, against the plan's 6.9 / 12.8 h. Options: accept, batch (needs a batch-invariance proof; the plan says N = 1), or score logged sensors offline.
4. **Builder's flagged choices**, which I agree are consistent with the plan:
   - REQ_CONF = 3.2's first fallback entry;
   - R2 (d) = T0-nominal row;
   - R6 on 67000-67767;
   - T-opp = first falls with no robot-robot contact in the previous 0.5 s (matches E6's use as a no-contact self-fall baseline);
   - G0-11 with lag 0.

## 6. Housekeeping
- Two "M6a verify" ledger rows were appended.
- No MUJOCO_LOG.TXT and no `Users*` folder in the repo.
- Processes used at most 2 workers each; R0-EO was not touched.
- No commit, push, stash or branch switch.
- The suite ran with basetemp `scratchpad\pytest-tmp-vm6a`, because another session's pytest was using `pytest-tmp` at the time.
- **Suite, first run** (`verify_m6a\suite.log`): 1 failed, 378 passed, 2 skipped.
  - The failure was mine: my first edit of fb_eq.py put the new comment in front of `T1C_LO, T1C_POINT` on the same line, which commented them out (NameError in test_decisions_equal_gate_power_v3).
  - Fixed. `test_fb_eq.py` and `test_fb_m6_run.py` then passed 9/9.
  - A full `fb_eq.py reproduce` on the fixed file is again **equal, 0 diffs** (`eq_reproduce_postfix.json`). The earlier reproduce had loaded the module before any edit.
- **Suite, second run** (`verify_m6a\suite2.log`): **1 failed, 378 passed, 2 skipped**.
  - Every M6a test and every other test passes.
  - The one failure is `tests/test_fb_library.py::test_manifest_sources_are_train_only_and_hashes_match` (M8's workstream), a hash mismatch. It passed in the first run.
  - `gpu/export_fb_library.py` was modified at 10:12 by another session while this suite was running; the L1 data files date from 08:41. This is not M6a code, and I did not touch it. The M8 owner should re-export or re-hash.
- One MUJOCO_LOG.TXT written by each suite run was deleted.
