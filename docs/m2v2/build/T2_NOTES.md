# T2 build notes: Track 2 split (T2-2) and sealed eligibility (T2-3)

Lane T2, 2026-10-30. Repo `r1-sparring-gym-spar`, branch feat/sparring-m2v2, nothing committed. CPU only, at most 2 worker
processes. No GPU, no install. No hard_r2 val/test seed, no kfab_202/204/205 content, no screen-val seed. val2/test2 motion was
touched only by `scripts/fb_track2_eligibility.py` in sealed mode.

## Headline (aggregate only)

| split | clips | eligible (hand reading) | eligible (any-limb reading) | head-band hand strikes, eligible clips | all clips, before any tracker cut |
|---|---|---|---|---|---|
| val2 | 40 (4 videos) | **0** | **0** | 0 (needs >= 30): **val2_ok false** | 100.1 s kept, **15** head-band hand strikes, 9 clips >= 4 s |
| test2 | 7 (4 videos) | **0** | **0** | 0 | 53.0 s kept, 6 head-band hand strikes, 5 clips >= 4 s |
| train pool (open) | 33 (14 videos) | 1 (kfab_1021) | 1 | 4 | 397.5 s kept, 45 head-band hand strikes, 26 clips >= 4 s |

- val2 failures per bar (last round): kept < 4 s 32/40, < 2 head-band hand strikes 39/40, stay-up not reached on the last round
  40/40. No val2 or test2 clip passed both the kept and the strike bar, so the T0 stay-up bar never decided a sealed clip.
- **The val2 requirement (>= 30 head-band hand strikes) cannot be met with this partition under the strike rule.** All 40 val2
  clips together hold 15 head-band hand strikes before any per-clip bar or tracker cut; val2 + test2 together hold 21. No
  tracker, cut rule or reading of the per-clip bars reaches 30. Options are listed at the end. Nothing was decided.

## Files (all new except fb_seeds.py)

| file | what |
|---|---|
| `scripts/fb_track2_split.py` | T2-2. Titles only: groups the 80 downloaded fist-form clips by source video title (scene suffix removed), splits per style taken from the title (taichi 太极 / nanquan 南拳 / changquan), largest remainder, `np.random.default_rng(20261030).permutation` of sorted titles, val2 then test2 then train. It hashes the npz files against `manifest_expected.tsv` but never parses them. It asserts that IDs 196-206 are absent. `load()` re-checks `split_sha256`. |
| `results/sparring/track2_split.json` | The split, its inputs' sha256, the rule, excluded IDs 196-206 with their titles, per video (title, style, routine, split, ids), counts, `split_sha256`, `routine_overlap` (report only). |
| `scripts/fb_track2_motion.py` | T2-3 pipeline: `load` (format guard), `retarget` (= r1gym `retarget_g1_to_r1`), `process` (FightReplay's chain plus a cleaned-frame index column), `style_ranges`, `reference`, `motion`/`replay` (one-clip fb_opponent Motion and FightReplay), `alone` (eval_tracker.run with the T0 driver), `screen` (the pre-declared rule). |
| `scripts/fb_track2_eligibility.py` | `--split train` writes per-clip records. `--split val2/test2` is sealed: workers return booleans and counts only; the output holds per-clip `eligible_hand` / `eligible_any` booleans and aggregates. The retarget is never cached. Exceptions report the type and code path only. |
| `scripts/fb_track2_check.py` | C1-C6 pipeline checks on kfab_200/201/203 and the train pool (below). |
| `tests/test_fb_track2.py` | 12 tests, about 7 s. |
| `results/sparring/track2_eligibility_{train,val2,test2}.json` | Outputs: train is detailed; val2/test2 hold booleans and aggregates only. |
| `scripts/fb_seeds.py` (edited) | The `track2` role existed (24000-25999). Its clip rule now excludes IDs **196-206** (`TRACK2_OUT`; it was 199-206), matching TRACK2_APPROVED. CRLF is kept. |

sha256: track2_split.json 67a97d4a...aee3; eligibility train 693e4419...1986, val2 c4c181a6...c316, test2 6d0ee0ca...1834.

## T2-2 split

- **Inputs** (`C:\Users\dundy\Downloads\projects\kfab_data`), sha256:
  - action_titles.csv e80bfa41...27d6
  - fist_ids.txt bdfa045d...720a
  - manifest_expected.tsv 83314f0f...4f98
- **npz integrity:** all 80 npz files match the manifest's size and LFS sha256. `fetch.log`'s "FAIL" lines are a fetch-script
  artefact; the bytes are correct.
- **Rule and seed:** seed 20261030 was committed in the script before the split was first computed, and it was run once.
- **Exclusions:** IDs 196-206 (duan-4 196-198, duan-6 199-206) were excluded before the download and are absent (asserted). Their
  titles are recorded in the split file.
- **Result:** split_sha256 `6a30c05307d2187bfb56cfc338795d5a714b77f975fd4a1303f8d407c8768ed2`; 22 videos.

  | split | videos | clips | styles (changquan / nanquan / taichi) |
  |---|---|---|---|
  | train | 14 | 33 | 8 / 2 / 4 |
  | val2 | 4 | 40 | 2 / 1 / 1 |
  | test2 | 4 | 7 | 2 / 1 / 1 |

  - val2 holds both 少年规定拳 videos (21 + 12 scenes), the complete 国一南拳 video and 42式太极拳 section 3.
  - test2 holds 段位制长拳一段, 国一长拳 section 4, 国一南拳 section 1 and 42式太极拳 section 4.
- **Deviation from a plain shuffle (disclosed):** the split is stratified by a style taken from the title, so that neither
  held-out split can be all tai chi. The unit is still the source video title, decided from titles only.
- **60/20/20 is by video, not by clip.** By clip the split is 33 / 40 / 7, because scene counts per video range from 1 to 21.
- **Routine overlap (report only, `routine_overlap`):** some sections and the complete version of one routine are separate
  videos, so the same choreography lands on both sides of the split:
  - guoyi_changquan: test2 and train;
  - guoyi_nanquan: val2, test2 and train;
  - taichi42: val2, test2 and train.

  The plan's "by source video" was followed literally. Grouping by routine was infeasible for a 3-way split (it gives 2 units
  for taichi and for nanquan).

## T2-3: what the npz are and what retargeting is needed (train-pool clips only)

- **Contents:** `fps` [1] int64 = 50 and `joint_pos` [T, 36], which is the G1 29-DoF qpos: root xyz, a unit quaternion (wxyz) and
  the 29 hinges in r1gym's G1 model order (`joint_names` equal `_sub('g1')`'s with the `_joint` suffix). The other arrays are
  `joint_vel` [T, 35] and `body_pos_w` / `body_quat_w` / `body_lin_vel_w` / `body_ang_vel_w` for the G1's 53 bodies (world
  first).
- **FK check (C6):** r1gym's G1 forward kinematics of joint_pos reproduces body_pos_w / body_quat_w on all 30 shared bodies of all
  33 train-pool clips, to 1.8e-15 m and 6.7e-16.
- **Comparison with the repo's kfab_*.npz:** those hold the same qpos as `qpos`, but with non-unit quaternions there.
- **Retargeting:** no new retarget is needed. The only conversion is joint_pos -> qpos, then the repo's existing
  `retarget_g1_to_r1` (G1 -> R1 keypoint IK), and then FightReplay's chain:
  - clean_clip;
  - tracker cuts;
  - style cuts by scripts/style_cuts.py's rule;
  - ground;
  - _canon;
  - _seamless.

  The same functions run inside the sealed script for val2/test2 (said here as instructed).
- **Reference shown to T0 for a new clip:** joint pos, central-difference joint vel over the looping clip, and torso_link pose by
  FK at distance 0, in float32. The tracker's own reference for the train clips comes through the CSV (6 decimals) and its
  float32 time grid. These differ by <= 6.2e-5 rad, 1.5e-3 rad/s, 6.7e-6 m and 1.5e-5 (C4). C5 shows the difference does not
  change behaviour.

## Pipeline checks (`scripts/fb_track2_check.py`, `build\..\..\t2\t2_check.json` in the scratchpad `t2` folder)

| check | result |
|---|---|
| C1 retarget of raw kfab_200/201/203 = the cached r1_kfab_*.npy | **bit-equal**, 344 / 768 / 581 frames |
| C2 process(cached retarget, tracker_cuts.json, style_cuts.json) = FightReplay(distance=0).q | **bit-equal** on all 3. The style rule recomputed equals style_cuts.json. Process with the computed style cuts is also bit-equal. |
| C3 strikes.detect = strikes.json rows (onset, apex, end, limb, type, band) | **equal**: 8 / 20 / 10 strikes; head-band hand 0 / 4 / 1 |
| C4 reference vs the tracker's (fb_motion.npz) | max: joint pos 6.1e-5, joint vel 1.5e-3, anchor pos 6.7e-6, anchor quat 1.5e-5 (reported) |
| C5 tracker alone, 24 starts x 3 train clips: the eval (FightReplay + the eval's Tracker) vs the generic path (process + reference + replay + T0 float32) | **PASS**: 67 both up, 5 both fell, max dt **0.00 s**, 0 mismatches. Mean wall per 10 s run: eval 0.62 s, generic 1.16 s. |
| C6 format + FK on all 33 train-pool clips | **PASS**: pos 1.8e-15, quat 6.7e-16 |

Commands (python = scratchpad venv; cwd = scratchpad `t2`):
```
python -I -X utf8 scripts/fb_track2_split.py                                   # writes results/sparring/track2_split.json
python -I -X utf8 scripts/fb_track2_check.py --out t2_check.json --workers 2   # C1-C6 ok, 87 s
python -I -X utf8 scripts/fb_track2_eligibility.py --split train --workers 2 --cache cache   # 216 s (retargets cached in scratchpad t2\cache)
python -I -X utf8 scripts/fb_track2_eligibility.py --split val2 --workers 2    # sealed, 48 s
python -I -X utf8 scripts/fb_track2_eligibility.py --split test2 --workers 2   # sealed, 37 s
python -I scripts/fb_seeds.py --audit ...\SEED_LEDGER.md --scan               # ok: 19 roles disjoint; ledger consistent; 25 task inputs clean
python -m pytest -q -p no:cacheprovider --basetemp="...\pytest-tmp" tests     # see the end
```

## The pre-declared screen (fb_track2_motion.RULE; fixed before any train-pool or sealed run, unchanged since)

- **Kept:** the final clip length, after cleaning, tracker cuts, style cuts and the 0.3 s seam, must be >= 4 s.
- **Strikes:** at least 2 head-band strikes by `strikes.detect` on the in-place clip. Head band is 1.02-1.23 m.
- **Stay-up:** >= 0.9 over 24 starts (t0 = s x T / 24), 10 s each.
  - The world is `scripts/eval_tracker.run` verbatim: R1 at home, opponent 4 m away facing it, no aim. A fall is pelvis up-z
    < 0.5 or z < 0.25.
  - The opponent is T0 = `model_13749.pt` through fb_opponent's TrackerDriver, float32.
- **Procedure:**
  1. Process the clip. If kept or strikes fail, stop: the clip is ineligible.
  2. Run the 24 starts.
  3. While stay-up < 0.9 and cut rounds < 3:
     - map the falls to cleaned-clip frames;
     - cut the cluster with the most falls (falls <= 1 s apart are one cluster) over [first - 0.8 s, last + 0.1 s];
     - re-process, re-check kept and strikes, and re-run the 24 starts.

  The 0.8 s pre-margin follows tracker_cuts.json ("each cut starts ~0.8 s before the cluster").
- **Both readings of the strike bar are scored:**
  - `eligible_hand`: head-band **hand** strikes, as in recon_data 5.3; this is also val2's unit.
  - `eligible_any`: any limb, as in the plan's 6.2 wording.

  The starts run when the looser "any" bar holds. Using the stricter reading alone would have tightened a frozen bar, so both are
  reported. They agree everywhere here.
- **Seeds:** tracker-alone worlds are seed-free. eval_tracker.run fixes env seed 0, and that RNG only feeds the defender's
  observation noise, which its zero actions never read. A test re-runs a start with env seed 4321 and gets identical outputs.
  - Bookkeeping ids in the `track2` role, checked by fb_seeds and appended to SEED_LEDGER.md: train 24900-24932, val2
    24950-24989, test2 25950-25956.
  - 24000-24949 and 25000-25949 are left free for a future val2/test2 eval.

## Train-pool findings (open inspection)

- **T0 stay-up on new forms is poor.** 10 clips reached the starts.
  - Round-0 stay-up was 0.0 on kfab_230, 231, 918 and 1081; 0.04-0.21 on kfab_1022, 1307, 1315, 1317 and 1423; and 0.71 on
    kfab_1021, which reached 0.917 after two cuts and is the one eligible clip.
  - Falls are spread over 0.3-10 s (medians 2-5 s), not at placement. C5 rules out a pipeline fault, so this is the tracker
    never having seen these motions.
- **The strike bar is the main filter.** 26 of 33 train-pool clips have < 2 head-band hand strikes. The material is wushu forms:
  0.11 head-band hand strikes per kept second, against 5 in 25 s on the kfab train clips.

## Issues met and fixed (before the final runs)

1. **fb_track2_check C5 assertion.** My first check asserted that the world's RNG is never drawn. It is drawn, for the
   defender's obs noise only. I replaced the assertion with the seed-independence test.
2. **numpy ints in the train JSON.** These would have crashed the write. The first train run was stopped and restarted after the
   fix. Its partial output was never written; the cached retargets were kept.
3. **Sealed val2 run 1: 6 of 40 clips raised IndexError.** Only the type was recorded.
   - Cause: a clip where cleaning or a cut keeps no piece of at least the minimum length. `opponent._join` then indexes an empty
     list.
   - Fix: `process` turns exactly that case (innermost frame `_join`) into kept = 0 s, so the clip is ineligible by kept. A test
     uses a train clip turned upside down, and one cut to nothing.
   - Sealed errors now also record the code path (file:function, no values).
4. **Sealed re-runs.** val2 was re-run after the fix (0 errors; kept failures 26 -> 32 = the 6 former errors), and once more to
   add the `all_clips_first_round` aggregates. test2 was re-run with the final code: identical booleans, 0 errors. The rule never
   changed between runs. The superseded outputs were overwritten; the logs are in the scratchpad `t2` folder.

## Open items for the main session (options, not decided)

1. **val2 cannot reach 30 head-band hand strikes** (15 in the whole split; 21 in val2 + test2). Options:
   - (a) Track 2 gives no usable val2. S1/S2 stay on kfab_204 as frozen. Track 2 contributes only its train pool to T1'.
   - (b) Download more KungFuAthlete material for a new held-out pool, for example daily-training ground clips. This needs a new
     user approval and a new title-only partition.
   - (c) Any re-partition of these 47 held-out clips is now outcome-informed (aggregates have been seen), so it would need the
     user's explicit OK and disclosure.
   - (d) Change the strike bar, the band or the 30. Each would loosen a frozen rule; user decision only.
2. **Per-clip bars on scene clips.** Most val2 scenes are shorter than 4 s after cleaning and style cuts (only 9 of 40 are
   >= 4 s). A video-level variant, pooling a video's scenes, would change the frozen rule. Even pooled, val2 has only 15
   head-band hand strikes.
3. **Stay-up uses frozen T0 (as instructed).** T0 fails most unseen forms (train pool: 1 of 10 screened clips stays up). The plan's
   broad T2 tracker (eval apparatus) would be the one to measure stay-up for any future val2. The sealed script would need a
   `--policy` option for that (not added).
4. **Routine overlap** across splits (guoyi_changquan, guoyi_nanquan, taichi42). An option is to quarantine the overlapping
   train-pool titles from T1' training (20 of 33 clips). This is report only.
5. CREDITS must cite the paper with "MIT per paper / Apache-2.0 per HF card" (TRACK2_APPROVED). Not edited here: it is not an
   fb_* file.

## Final verification (final code)

- `fb_track2_check.py` re-run after all edits: C1-C6 ok (C5: 67 both up, 5 both fell, max dt 0.00 s; 119 s). Output copied to
  `build\t2_check.json`.
- Full suite: **345 passed, 2 skipped in 982 s** (the 2 skips: the FB_SEED_LEDGER audit test and one pre-existing skip).
  `tests/test_fb_track2.py`: 12 passed, about 7 s.
- `fb_seeds.py --audit SEED_LEDGER.md --scan`: ok, 19 roles disjoint, ledger consistent, 25 task inputs clean.
- The suite run left a `MUJOCO_LOG.TXT` in the repo ("Attaching a spec with an empty prefix" warnings, 05:28); it was deleted.
  No pytest-tmp folder in the repo root.
