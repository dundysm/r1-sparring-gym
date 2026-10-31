# T2 independent verification (M2V2_PLAN_v3 6.2 item 3: T2-2 split, T2-3 sealed eligibility), 2026-10-30

**Verdict: ok = false.** As with M5, the reason is a frozen clause that the main session or the user must decide. It is not a
code failure.

**The open item.** The frozen T2-3 requirement says "val2 needs >= 30 head-band hand strikes". It cannot be met with this
partition, and section 4 shows it cannot be met with **any** partition of the 80 downloaded clips under the builder's reading
of the kept bar.

**The code.** Every frozen and builder check reproduces exactly when I re-run it:
- the split rebuilds byte-identical;
- C1-C6 are equal to the builder's output (wall times excepted);
- the train-pool, val2 and test2 eligibility outputs are equal to the committed files (wall times excepted);
- the full suite passes (section 6).

**Semantic review.** A line-by-line read against the eval (r1gym/opponent.py, strikes.py, scripts/eval_tracker.py,
style_cuts.py, gpu/fb_opponent.py) found **no semantic mismatch**. An independent probe that runs the eval's **own**
`FightReplay.__init__` on Track 2 train-pool data is bit-equal to the generic path (section 2).

**Changes.** Nothing was fixed and no repo file was edited.

## 1. Re-runs (outputs in scratchpad `t2v\`)

| run | result |
|---|---|
| `fb_track2_split.py --out t2v\split.json` | Byte-identical to `results/sparring/track2_split.json` (sha256 67a97d4a...aee3). split_sha256 6a30c053...68ed2. All 80 npz files re-hashed against the manifest: ok. |
| `fb_track2_check.py --workers 2` (`t2v\t2_check.json`, `check.log`) | Exit 0. Recursively equal to `build\t2_check.json` with wall times excluded.<ul><li>C1: bit-equal, 344 / 768 / 581 frames.</li><li>C2 and C3: equal; 8 / 20 / 10 strikes.</li><li>C4: the same maxima.</li><li>C5: 67 both up, 5 both fell, max dt 0.00 s, 0 mismatches.</li><li>C6: pos 1.8e-15, quat 6.7e-16.</li></ul> |
| `fb_track2_eligibility.py --split train`, **without** the retarget cache (every retarget recomputed) | Equal to the committed train JSON except `wall_s`: 1 eligible clip (kfab_1021, 4 head-band hand strikes); first round 397.5 s and 45 strikes. |
| `--split val2` and `--split test2` (sealed, unchanged script, output to scratchpad) | Equal to the committed files except `wall_s`.<ul><li>val2: 0/40 eligible; first round 100.1 s and 15 strikes; `val2_ok` false.</li><li>test2: 0/7 eligible; 53.0 s and 6 strikes.</li><li>The logs hold progress counts and the aggregate line only.</li></ul> |
| `tests/test_fb_track2.py` | 12 passed. |
| `fb_seeds.py --audit SEED_LEDGER.md --scan` | ok: 19 roles disjoint; ledger consistent; 25 task inputs clean. That includes the 4 new `scripts/fb_track2_*.py`, which are task inputs under `TASK_GLOBS` and hold no banned name. |
| Full suite | See section 6. |

## 2. Independent probes

1. **The eval's own chain on Track 2 data** (`t2v\probe_eval_path.py`, train pool only).
   - Set-up: I pointed `r1gym.opponent.ASSETS` at a temp folder holding the clip as `motions/kfab_<id>.npz` (`qpos` =
     joint_pos), `tracker_cuts.json` (the builder's per-round cuts) and `style_cuts.json` (from `scripts/style_cuts.ranges`
     itself). I then built `FightReplay(m, clip, distance=0)`. That path covers `_retarget`, clean_clip, the tracker cuts,
     the style cuts, ground, `_canon` and `_seamless`.
   - Result: **bit-equal** to `fb_track2_motion.process` on kfab_1021 (rounds 0-2), 230 (rounds 0-2, including the 0.54 s
     round), 1707, 1316 and 1081 (rounds 0-1).
   - The style ranges are equal. `strikes.detect` gives the same rows, and the head-band hand counts and kept seconds equal
     the train JSON.
   - This covers what C1/C2 cannot: unit-quaternion input, multi-round tracker cuts in cleaned frames, and very short clips.
2. **Quaternion order.**
   - The dataset card says the org_smoothed_mj files use **xyzw**. The builder reads joint_pos[3:7] as MuJoCo **wxyz**.
   - FK agreement alone does not settle this, because body_pos_w was produced by the same FK.
   - The physics does settle it. Across all 33 train-pool clips, reading wxyz gives a pelvis up-z median of 0.94-1.00, with
     feet on the floor (median ankle z 0.007-0.031 m). Reading xyzw gives medians from -0.87 to 0.89, with minimums near -1:
     upside-down robots.
   - So **wxyz is correct**, and the card's "xyzw" does not describe joint_pos.
3. **The kept reading.**
   - The builder's kept seconds are the replayed loop length `FightReplay.T`, after the 0.3 s seam blend.
   - On the train pool, only kfab_1707 is affected (3.74 s; it would be 4.04 s before the seam; 2 head-band hand strikes).
   - I ran its 24 T0 starts: stay-up is 0.0. So under T0 the reading changes no train-pool boolean. It matters only for
     the upper bound in section 4.
4. **Bounds** (`t2v\probe_cap.py`, committed outputs only): see section 4.

## 3. Line-by-line review (no mismatch found)

- **Split (T2-2).**
  - It reads titles only, groups by the title minus `-Scene-NNN.mp4`, and stratifies by a style keyword in the title.
  - The quota method is largest remainder: 22 videos -> val2 2/1/1 and test2 2/1/1 (changquan/nanquan/taichi). I checked
    this by hand.
  - The seed is committed. The npz files are hashed, never parsed. IDs 196-206 are asserted absent.
  - `load()` re-checks `split_sha256`.
  - Only 80 of the 141 fist-titled IDs were downloaded. The other 61 are 196-206 plus IDs that are not in the 848-sample
    release (card: 1,726 sub-clips -> 848 after filtering; recon 1 predicted this).
  - Recon's stratification advice ("each held-out split ... ideally a Nanquan and a Changquan video") is met.
- **Motion (T2-3).** `process` follows `FightReplay.__init__`'s order and arguments exactly:
  - fps 50 vs 50.0 gives the same integer windows;
  - the style cuts use `STYLE_MIN_KEEP_S`;
  - `opponent_style_cuts` is on in the eval config;
  - the extra src column only rides through operations that act on columns :7 or blend all columns.
- **Reference and tracker.** `reference()` mirrors the tracker's export (central-difference velocity over the loop and the
  torso anchor at distance 0, in float32; C4).
  - `motion()` and `replay()` build a one-clip Motion and FightReplay the way `__init__` does.
  - The MjTracker is made with `__new__` and `k = 0`, which bypasses the `TRAIN.index` assertion. That is correct for a
    one-clip Motion.
  - `alone()` swaps only `eval_tracker.FightReplay` and restores it in `finally`.
- **Fall rule and seeds.** The fall rule, 10 s, 4 m and the 24 starts are eval_tracker.run's and G0-3b's.
  - Seed-freeness is tested: env seed 4321 gives identical outputs.
- **Sealed reduction.**
  - Workers return booleans and five counts per clip. The parent keeps per-clip booleans and sums the counts.
  - An exception returns its type and file:function only (tested with a message that carries numbers).
  - The cache is asserted off for sealed splits. Pools are spawned with at most 2 workers.
  - The scratchpad `t2\cache` holds only the 33 train ids.
  - The run-1 IndexError fix is narrow: an IndexError counts as nothing kept only when the innermost frame is `_join`. That is
    the eval's own failure on an empty clip.

**Minor notes (no fix; none changes a result):**
- **src is blended.** The src column is linearly blended across `_join` cross-fades and the first 15 frames after
  `_seamless`. A fall inside a 0.3 s blend therefore maps to an intermediate cleaned frame. This is pre-declared and
  train-pool only: no sealed clip reached the starts (`measured` = 0).
- **Bookkeeping ids in the val2 block.** The train-pool ids 24900-24932 lie inside the plan's val2 block (6.4: 24000-24999).
  The worlds are seed-free, so no random stream is consumed. The track2 role covers 24000-25999 jointly, so fb_seeds cannot
  tell val2 from train. A future seeded val2 eval should simply avoid 24900-24999.
- **Possible future leak via MUJOCO_LOG.TXT.** If a later sealed run (for example with the T2 tracker) simulates sealed
  clips, MuJoCo instability warnings go to MUJOCO_LOG.TXT with a simulation time, which is a per-clip number. This did not
  happen here because no sealed clip was simulated. A `--policy` extension should redirect or suppress it.
- **fetch.log's FAIL lines** are an artefact of the fetch script. The Python re-hash of all 80 files passes.

## 4. The open item, sharpened (options; nothing decided)

- **val2 under this partition.** val2 holds 15 head-band hand strikes over all 40 clips before any bar, against the 30 it
  needs. No val2 or test2 clip passes the kept and strike bars together (`measured` = 0 for both).
- **Any partition, any tracker.** A perfect tracker would have stay-up 1.0 and need no cut, so the per-clip bars would
  decide on round 0. Under the builder's kept reading:
  - the train pool's clips with >= 4 s kept and >= 2 head-band hand strikes hold **29** head-band hand strikes (kfab_230 2,
    231 3, 918 2, 1021 5, 1022 2, 1081 2, 1307 6, 1315 3, 1317 2, 1423 2);
  - val2 and test2 add 0;
  - so the **upper bound over all 80 clips is 29 < 30**.
- **Under frozen T0** the whole pool yields **4**.
- **Correction to the builder's option (c).** Re-partitioning the 47 held-out clips cannot reach 30: they hold 21 before
  any bar and 0 that pass the bars. A re-partition that moves train-pool videos into val2 also stays at <= 29 with a
  perfect tracker. So (c) is not a route under the frozen bars. That leaves:
  - (a) no val2 from Track 2; S1/S2 stay on kfab_204; Track 2 adds only a train pool;
  - (b) a new held-out pool. This needs a download approval and a new title-only split. Note that the forms material runs
    at about 0.11 head-band hand strikes per kept second, against recon's assumed 0.2;
  - (d) a change to a frozen bar, band or the 30. This is the user's decision.
- **The kept reading as a boundary case.** If "kept" were read before the 0.3 s seam, kfab_1707 (4.04 s, 2 strikes) would
  pass the first two bars and the bound would be 31. That is still not reachable with T0, which gives kfab_1707 stay-up 0.0.
  The reading is not frozen anywhere. The builder's choice is the eval's replay length; I flag it and do not change it.
- **Routine overlap** (report only, a decision for T1'). test2's 国一长拳第四段 and 42式太极拳第四段 are sections of
  routines whose complete versions are in the train pool. guoyi_nanquan likewise spans all three splits. If T1' trains on
  the overlapping train-pool titles, test2 is partly seen choreography. The builder's quarantine option (20 clips) addresses
  this.

## 5. Hygiene

- **Files.** Only the files the builder listed are new. `scripts/fb_seeds.py` is the only tracked file edited: TRACK2_OUT
  is 196-206, and the CRLF working copy is consistent with `core.autocrlf=true` (index LF, `git ls-files --eol` i/lf w/crlf).
  The new files are LF.
- **Other lanes.** The untracked `scripts/fb_r0eo.py` and `tests/test_fb_r0eo.py` belong to the R0E lane and were not
  touched.
- **Git.** No commit, stash or branch switch.
- **Data handling.**
  - The data was loaded with `allow_pickle=False` under `python -I`, and the scripts live in the repo.
  - I read val2/test2 only through the sealed script, whose outputs hold booleans and aggregates only.
  - No hard_r2 val/test seed or kfab_202/204/205 content was used. CPU only, at most 2 workers per process.
- **Ledger.** 3 rows were appended to SEED_LEDGER.md ("T2 verify": 24900-24932, 24950-24989, 25950-25956, seed-free
  re-runs). The audit passes.
- **Leftovers.** No MUJOCO_LOG.TXT and no `Users*` folder is in the repo.

## 6. Full suite

- **Result:** 345 passed, 2 skipped in 1135 s (`t2v\suite.log`).
- **Leftover.** The run left a MUJOCO_LOG.TXT at 06:08:39 ("Attaching a spec with an empty prefix" warnings). I deleted it.
- **A concurrent lane edited files during the suite.** Between 05:55 and 05:59 another lane (A4.1 W0) modified
  `scripts/fb_seeds.py`, adding the role `w0_r4pre`, LISTS and CTRL_ONLY. It also modified `scripts/fb_m5_check.py`,
  `tests/test_fb_hygiene.py` and `tests/test_fb_m5.py`. Those edits are not mine and not T2's.
- **Track 2 under the current fb_seeds.** The concurrent edit keeps T2's `TRACK2_OUT` and the `track2` role. After it, I
  re-ran:
  - `tests/test_fb_track2.py` and `tests/test_fb_hygiene.py`: 67 passed, 1 skipped;
  - `fb_seeds --audit --scan`: ok (20 roles disjoint; ledger consistent; 25 task inputs clean).
- **What the main session should do.** Re-run the full suite once the W0 lane finishes. The suite above straddled its edits.
