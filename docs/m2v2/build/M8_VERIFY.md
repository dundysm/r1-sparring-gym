# M8 independent verification: library L1 and the T1 config (CPU part), 2026-10-30

Verified against: M2V2_PLAN_v3 section 3 row M8, 2.2 (L1, T1), 6.4 (60500-60599), A1-A5; design_learned-tracker 9.1-9.3 (the
priors and rejection rules); gpu/export_warps.py (the speed rule L1 says it mirrors); r1gym/strikes.py, r1gym/opponent.py
(Tracker.frame, retarget), scripts/eval_tracker.py, gpu/export_fb_motion.py, gpu/r1_track_task.py, and mjlab 1.2.0
`tasks/tracking/mdp/commands.py` `_adaptive_sampling` (local source, commit 5af32e3). Nothing in the repo was modified.

**Verdict: ok = false.** Every frozen M8 check passes and reproduces exactly. One semantic defect remains unfixed on purpose:
the combo family's joint-speed rule does not do what the code says it does (section 3, V1). 7 of the 8 accepted combo splices
only pass because the allowance includes a fake jump. Fixing it to the stated rule leaves 1 of 13 combos, so the combo family
almost disappears. That is a design choice for the main session, so I report the options and do not pick one.

## 1. Re-runs (2 workers, CPU only)

| item | result |
|---|---|
| Regeneration `export_fb_library.py --workers 2` into `verify_m8\regen` (23.9 min, clean masks included) | **bit-equal**: `fb_l1.npz` 413f4802..., `fb_l1_track.csv` and `_index.json` equal byte for byte, every npz array equal, manifest equal on every key (git too) |
| `fb_m8_check.py --workers 2` (6.6 min, exit 0) -> `verify_m8\m8_check_v.json` | **equal to the build's `m8\m8_check.json` on every key except wall_s** |
| `tests/test_fb_library.py` + `tests/test_fb_hygiene.py` | 68 passed, 1 skipped (26 s) |
| `fb_seeds.py --audit SEED_LEDGER.md --scan` | ok: 20 roles disjoint, ledger consistent, 36 task inputs clean |
| Repo | no MUJOCO_LOG.TXT, no stray pytest folder. I did not re-run the full suite because I changed no code; the build's 379 passed / 2 skipped stands |

Hash note: the manifest's output hashes (5c21a07b... csv, 789c6420... index) are `EM.sha256`, which normalises line endings to LF
(the eval_opponent.text_sha16 convention). On disk the files have CRLF endings (np.savetxt and json.dump in text mode on
Windows), so their raw sha256 values are 5659533f... and 9b384d41.... That is consistent by design and not a defect.

## 2. Frozen checks

| check | verdict | my evidence beyond the builder's |
|---|---|---|
| (1) manifest, sources in TRAIN_CLIPS, sha256 | **PASS** | reproduced. Independently, every frame's dominant src_clip is a train clip. The LOCO motion keeps 20 rows plus originals 200/201, and no frame comes from 203 |
| (2) rejection stats; re-angled apex forward distance = original | **PASS** | reproduced (48/48 apex frames bit-equal; window max excess -0.31 mm). Independent elevation probe on the 48 accepted re-angles: measured approach elevation (strikes.py's u over the last 0.1 s) differs from the drawn target by at most 2.48 deg (median 0.26). The sign of the change is right 48/48, and the base elevation is reproduced exactly (0.0). Over all 112 tried re-angles: foot 0.0, CoM <= 5.5 mm, glove-shoulder <= 0.438 m, so only the path, speed and reach rules ever bind |
| (3) T0-alone performability, 24 starts per family, 60500-60595 | **PASS (reported)** | reproduced exactly: mirror 1/24, warp 12/24, reangle 10/24, combo 11/24, all below 0.9. The runner equals the eval Tracker on 72 original starts (67/67, 0 mismatches). The conjugated-T0 diagnostic is reproduced. I agree with its reading: the mirror is correct and T0 is not mirror-equivariant |
| (4) mirrored joint ranges symmetric (asserted) | **PASS** | 9.0e-5 rad worst. This is the MJCF's print precision (`-0.22689 2.4784` vs `-2.47849 0.2268`). FK of the reflection is exact (0.0). The quaternion (w, -x, y, -z) equals M R M (test). vmax of the mirrored set is exactly the permuted original vmax |

Code read line by line, with no mismatch found:
- **Mirror map.** It matches design 9.1.1.
- **Warp time map.** The duration ratio matches 1/speed within 0.048, from the whole-frame adjustment. Frames outside the windows are the base's.
- **Warp and re-angle speed rule.** It is export_warps line 158-159 exactly: max(p99, own window fastest) + 0.05.
- **Re-angle geometry.** The rotation axis h x z raises the elevation for delta > 0. The apex is untouched.
- **Shown reference.** `track_frame` is Tracker.frame (`<= blend`). Shown ref_joint_pos vs the row's q: max 9.8e-4 rad. A one-frame shift gives 0.37 rad, so the indexing is right. The 9.8e-4 is csv_to_npz's float32 time grid on a 29,888-frame motion, against 6.1e-5 on fb_motion. That is what T1 itself is trained on, so I report it only.
- **Clean masks.** They use the eval's `clean_starts` call. Identity on the originals is 27,900/27,900 cells.
- **Seams.** The row loop seams equal the bases' seams (0.126 rad max in both).
- **`r1_track_spar_cfg`.**
  - `bin_count`, bin-to-time mapping, kernel padding and conv, and the metrics all equal mjlab 1.2.0's `_adaptive_sampling` (commands.py 246-288).
  - The group renormalisation gives the originals exactly 0.5 (tested).
  - The hand impulse equals the Aim task's.
  - Steer2's arguments are equal.
  - The play cfg asserts sampling_mode "start".

## 3. Findings

**V1 (semantic defect, unfixed, needs a decision): the combo speed allowance includes a fake jump.**
- **What the rule should be.** `export_fb_library.py` docstring line 19-20: "Rejected: export_warps' joint speed rule (|dq| > max(the base set's p99, the window's own fastest) + 0.05 rad/s) in every family."
- **What the code does.** In `combo_row` (line 208-209), the "own fastest" is taken over `old = concatenate([q[e:r+no+1], seg])`. That array joins two unrelated poses, the host frame r+no and the source segment's first frame. The diff across that join is not motion, but it enters the allowance.
- **Size of the fake jump.** 25-128 rad/s on the joints involved (`build\verify_m8\probe3.py`).
- **Measured effect.** With the allowance built from what the rule names (max(vmax, the host window's own fastest, the source segment's own fastest)):
  - 7 of the 8 accepted splices exceed it on at least one joint, at the cross-fades.
  - Examples: l1_combo_kfab_203_m0 right_shoulder_pitch reaches 18.6 rad/s against a fair allowance of 12.55; kfab_201_m1 left_shoulder_pitch 16.4; kfab_200_m0 left_hip_yaw 10.7 against 6.5.
  - The fastest joint speed in all of L1 (18.6 rad/s) is a combo fade. The fastest in any base clip or base mirror is 17.06, and the retarget caps it at 12.5 rad/s per joint.
  - Warp, re-angle and mirror rows never exceed their host's own per-joint maximum (`probe2.py`: excess 0.0).
- **Why I did not fix it.** Design 9.1.4 lists no rejection rule for combos. The speed rule on combos was the builder's own choice, and the fade range 0.15-0.35 s is a frozen prior. Fade velocities from that range break a fair speed rule on most splices. So the correct fix is a design choice, not a code repair.
- **Options for the main session** (each is about 25 min of regeneration plus 7 min of checks; the combo performability then changes):
  - (A) Implement the stated rule, allowance from the host window and the segment separately. 1 of 13 splices survives (kfab_203_o0, host end 245). The 6 combo rows become almost the base clips, and "2-3 strike sequences" (design 9.1.4) mostly do not occur.
  - (B) Drop the speed rule for combos, as design 9.1.4 states it. All 13 tried splices are accepted, with fade joint speeds up to about 20 rad/s. T0 already keeps 11/12 original-base combo starts up.
  - (C) Keep the current code and disclose it. I advise against this: the stated rule and the code disagree, and the manifest's reject stats ("speed 5") overstate how binding the rule is.

**V2 (disclosure, LOCO hygiene, minor): the warp threshold depends on kfab_203.**
- The per-joint p99 (`vmax`) that decides which 200/201 warp and re-angle draws are accepted is computed over all three clips, kfab_203 included.
- Removing kfab_203 changes it by up to 30% on some joint.
- T1-LOCO's 200/201 rows are therefore filtered by a statistic of the held-out clip. It is a weak, scalar dependence, and T0 itself was trained on kfab_203.
- Disclose it, or compute vmax per LOCO set if P3's "removed from every training tensor" is meant strictly.

**V3 (agree with the builder's issues 1-2, 4-7).**
- **Mirror.** The finding that T0 fails mirrored rows is real; the mirror itself is correct.
- **Warp slow bias.** It is real and is the structural consequence of export_warps' rule, which is faithfully copied: a sped-up strike passes only where p99 exceeds speed x its own fastest. It needs a disclosure.
- **Builder choices.** I find no contradiction with the plan in their listed choices, apart from V1.
- **`SparMotionCommand`.** It still needs P2's smoke. Its `ix["frames"] == time_step_total` assert fails loudly, not silently, if csv_to_npz's frame count differs.
- **AimOp mirrored-swap knob.** It is still open (M9).

## 4. Files

- `build\verify_m8\run_all.sh`: regeneration plus the check re-run.
- `regen\` and `regen.log`; `check.log` and `m8_check_v.json`.
- `cmp_regen.py`.
- `probe1.py` / `probe1.json`: elevation, warp timing, shown-ref indexing, LOCO, vmax, seams.
- `probe2.py`: per-family speed excess.
- `probe3.py`: the combo allowance (V1).
- Ledger: one row appended (60500-60595 re-run in role `l1_perf`, seed-free).
