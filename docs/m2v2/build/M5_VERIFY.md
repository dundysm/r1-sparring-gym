# M5 independent verification (M2V2_PLAN_v3 section 3, M5: G0-11, G0-12), 2026-10-30

**Verdict: ok = false.** The reason is one item that needs a main-session decision; it is not a code failure.

**The open item: W0.**
- The frozen G0-12 clause says "W0 calibrated on the R4-pre G knockdowns re-recorded with contact logs (60300-60363)". It cannot be met as written.
- 60300-60363 are 64 fresh episodes, not the R4-pre bouts. The R4-pre G knockdowns are 56 in 67000-67767 and 88 in 68000-68767, and no M5 role in `fb_seeds` may run those seeds.
- On the ledger's block, the eval produced 4 G knockdowns. Two had contact, and their W values are {0, 2.99} J.
- So the frozen W0 = 0.299 J is an interpolated 10th percentile of two values. I confirmed the builder's facts and options (section 4). I decide nothing.

**What else I found:**
- **One semantic mismatch, now fixed.** `spar_block` paid for a window whose first arm touch was on an **upper-arm (shoulder) shell**.
  - Run 2's clause, which the plan quotes "verbatim" (`gpu/sparring_task.py` `_finish` :593), pays only `bz`. That is a window whose arm episode opened on the **block zone**: the elbow or wrist_roll shells.
  - On the recorded bouts this cut G's paid windows from 435 to 270 and H's from 419 to 258. So 38% of the pay was for shoulder-first touches.
  - The builder's python reference had the same omission, so the two agreed with each other.
  - I fixed the account, the reference and the tests (section 2).
- **Every other frozen check passes**, re-run by me on a fresh, bit-identical recording with the fix in place. No threshold was changed.
- **G0-11 is not yet due.** The plan orders it after M6 R1, on replica bouts. The EO-state preview passes (section 1).

**Rules kept:**
- Only new, untracked files exist. The tracked `git diff` is empty. Branch feat/sparring-m2v2. Nothing was committed, stashed or branch-switched.
- CPU only, at most 4 workers, $0.
- One ledger row was appended ("M5 verify"). `fb_seeds --audit --scan`: ok (19 roles; ledger consistent; 20 task inputs clean).
- No MUJOCO_LOG.TXT and no `Users*` folder is in the repo.

## 1. Re-runs
| run | result |
|---|---|
| `fb_m5_check.py --reuse` on the builder's recording, with the builder's code (`verify_m5\m5_check_reuse.json` / `.log`) | Exit 0. A recursive diff against `m5_check.json` shows no difference (wall times excluded). |
| A **fresh recording** plus every check, with the fixed code (`verify_m5\rec\fb_m5rec_60200-60363.npz`, `verify_m5\m5_check_fresh.json` / `.log`, 4 workers) | Exit 0: `{"G011_on_EO_states": true, "G012": true}`.<ul><li>All 256 hooked bouts equal their plain twins (asserted).</li><li>**All 1280 arrays are bit-identical to the builder's recording**, and so are the rows, events, reverse events and episodes in the meta.</li><li>The intents differ only in `pay`, which is now block-first. The builder's rule is kept as `pay_any_arm` and equals the builder's `pay` on all 7468 intents.</li></ul> |
| Full suite (prescribed command) -> `verify_m5\suite_v.log` | **326 passed, 2 skipped** (the env-gated tests), 704 s, exit 0, with the fix in place |

**G0-11 (EO states, block 60200-60263; the replica run belongs to M6 R1):**

| | landed HM / ours | (limb, zone) mismatch bouts | clean agreement (gate >= 0.95) | median head impulse, rel. diff (gate 10%) | windows: outcome / pay agree | paid ours = ref (builder's any-arm rule) |
|---|---|---|---|---|---|---|
| G, com | 831 / 831 | 0 / 64 | 0.9519 | 4.4e-6 | 1847 / 1852 of 1853 | 270 = 270 (435) |
| H, com | 782 / 782 | 0 / 64 | 0.9578 | 8.9e-6 | 1838 / 1842 of 1843 | 258 = 258 (419) |
| G, point | 831 / 831 | 0 / 64 | 0.9952 | 4.4e-6 | same | same |
| H, point | 782 / 782 | 0 / 64 | 0.9987 | 8.9e-6 | same | same |

- Reverse direction (R1 -> opponent): G 412 = 412 and H 346 = 346, with 0 mismatch bouts.
- **The COM gated pass is thin: 0.9519 against 0.95.** I agree with the builder that M9 should feed contact-point (or palm / foot site) velocities. The replica run at M6 must use the input that will be deployed.

**G0-12:**
- **Scripted events:** all 11 pass. The HIT_R2 case now also checks that a shoulder-first touch gives `blocked` and pays 0.
- **Recorded bouts (all 4 sets):**
  - kd_ok, neutral_ok, approach/contact 0 while up, and tangle against numpy (max 3.6e-15) all pass.
  - Edges / charged: 7/2, 9/3, 4/1 and 3/1.
- **W0:** recomputed 0.29906465943470145 J; `frozen_ok` true.

## 2. The fix (new files only)
- **`gpu/fb_hits.py`:**
  - `BLOCK_LINKS = ("elbow", "wrist")` (run 2's SHELLS 'block').
  - `FullBodyHitAccount(..., block=None)`: a (P,) mask of block primaries. None means every arm primary, and it is asserted to be a subset of the arm zone.
  - Per window, `w_armb` records whether a block shell of the window's (side, limb) touched at the substep its arm episode opened. This matches run 2's `w_armz` rule, where the block zone wins a tie.
  - `pay` requires `w_armb`. The window **outcome** is unchanged (HitMeter's `blocked`, any arm shell).
  - `MjContactSensors.block_idx` was added, and a duplicated `__init__` line was removed.
- **`scripts/fb_m5_check.py`:**
  - `replay` and `Script` pass `block`.
  - `pay_ref(..., arm_block)` and `block_touches()` derive the reference from the logged sensor rows: the substep time is the post-step time minus (3 - k mod 4) x 5 ms, exactly as the replay clocks it.
  - Intents store `pay` (block-first) and `pay_any_arm`, and `compare` reports both.
  - Scripted case 7 now strikes a forearm shell and adds an `upper_arm` case.
- **Tests:**
  - `test_fb_hits.py`: the synthetic logs now carry a block shell and an upper-arm shell. Arm touches land on either or both, and the test asserts that some upper-arm-first windows go unpaid. A further assertion checks that the block primaries equal `ST.SHELLS["block"]`.
  - 23/23 pass in test_fb_hits + test_fb_rewards; test_fb_m5 runs in the suite.
- Backups of the builder's versions: `verify_m5\fb_hits.pre.py`, `fb_m5_check.pre.py`, `test_fb_hits.pre.py`.

## 3. Line-by-line review against the eval (no other mismatch)
- **Sensor sign:** probe `verify_m5\probe_sign.py`.
  - A box on a plane, and a stacked box with the primary as geom2.
  - MuJoCo 3.17's contact-sensor netforce is the force **on the secondary**, and the `MjContactSensors` formula gives the same thing.
  - So the hit sensors read the target's push on the limb (FORCE_SIGN -1 gives a positive closing speed), and the body sensor reads the defender's push on the opponent, which is what W's `max(0, F_on_opp . v_def_link)` needs.
- **Collision map of the eval model:** only shells collide between robots (opponent shells 4/0, defender shells 0/4; capsules 2/2 and 0/1 never meet).
  - So |netforce| per shell is the normal force (condim 1). The only difference from HitMeter's sum of |f_n| per contact is several normals on one shell, which explains the 4e-6 median-impulse difference.
  - The foot group is ankle_pitch plus 3 ankle_roll shells, so the ankle_pitch subtree is the right secondary.
- **HitMeter rules:**
  - The gap merge is `> gap + tol`.
  - Rounding comes before the thresholds.
  - The duration includes both ends.
  - The closing speed is taken at the episode's first substep; the merged key's vc is the max over the sides touching, all of which are new at an opening.
  - Zero-force touches count as touch.
  - `windows`: arm = first arm episode of any strength; blocked iff arm.t0 < hit.t0 strictly.
  - Window bounds are `Intent.window` (t_on, t_apex + 0.15 s, rounded to 4 dp), on the same intent filter as `meter_row` (EO:149). Intents are hand strikes only (aim.py:7), so slot limb 0 is right.
- **Run 2 HIT_R2:**
  - by apex + 0.05 s;
  - nothing landed in the window (`~isfinite(w_hit)` is contained in m2's `w_any`);
  - **block-zone first (fixed)**.
  - `_pending`: ours waits for any unlanded open head/torso episode, not only when an arm touched. That is closer to HitMeter's full-episode `landed`, and it cannot change pay.
- **Attribution and terms:**
  - The work ring holds 50 steps and includes the current step.
  - Edge = rising edge of `opp_down_ever` on the post-step pelvis (EO:86).
  - `land_t` = the last touch of a struck head/torso merged episode; the 0.3 s gate holds.
  - The push mask is `t - t_push > 0.5`.
  - Weights and units match the 2.4 table: -2.4 per knockdown; -0.8 x rate/60 per s; -2 per (m/s x s); -0.1 and -0.05 per N s; lean -0.01 x (deg - 5)^2; res_l2 -0.5 -> -0.1.
- **Push mask scope:** the builder masks approach and contact but not down-neutral. This agrees with the plan's vocabulary: R4-pre NOTES:61-62 names "post-knockdown term" as `fb_down_approach` / `fb_down_contact` (B1 fix 5), which is what 2.3's "post-knockdown terms are masked" refers to. It is not a mismatch.

**Notes for M9 (not M5 defects):**
- `clearance_value(d, win)` has no reach gate. Run 2's `spar_clearance` ANDs `reach_mask` (SCHED_R2 reach_gate). The 2.4 table does not mention it, so the caller must decide and pass `win & reach` if it is to be kept.
- W clamps per shell's netforce, not per contact. This is the mjlab sensor's resolution, and W0 is calibrated with the same definition.

## 4. W0: facts checked and options (not decided)
- `fb_seeds.ROLES`:
  - `w0_calib` = 60300-60363 only;
  - 67000-67767 belongs to `r2_diag` (M6) only;
  - 68000-68767 is in no role.
- R4-pre NOTES: G knockdowns are 56 (main block) and 88 (close block).
- `edges_all.json` (23 edges, both controllers, both blocks):
  - W0 = 0.299 J charges 7 (60201 G/H, 60230 G/H, 60204 H, 60302 G, 60341 H).
  - 2.99 J would charge 3.
  - 4 of the 7 charged edges come at t = 0.20-0.34 s, in the first 0.34 s of the bout (start-of-bout tangles at 0.40-0.55 m). These are eval-real knockdowns, but whether the defender should be charged for them is worth a look before M11.
- **Options:**
  - (a) Add a role so the about 144 R4-pre G knockdown bouts can be re-recorded with contact logs (amends 6.4 and fb_seeds).
  - (b) Enlarge `w0_calib` with fresh seeds (about 800 bouts for about 60 knockdowns).
  - (c) Keep 0.299 J.
  - (d) Use the work-only reading, 2.99 J.

  W0 enters the M11 hash, so the decision is needed before then.

## 5. Suite
- 326 passed and 2 skipped in 703.5 s, exit 0 (`verify_m5\suite_v.log`).
- No MUJOCO_LOG.TXT or `Users*` folder was left in the repo afterwards.
