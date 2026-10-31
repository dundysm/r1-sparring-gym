# M5W independent verification: Amendment A4.1 (W0 on the R4-pre G knockdowns) and A4.2 (early knockdowns), 2026-10-30

**Verdict: ok = true.** Every frozen check, as amended by A4, passes when re-run here. The W0 statistic was reproduced by my own numpy code from the raw contact rows. A fresh re-recording is bit-equal to the builder's. No semantic mismatch was found, so no code was changed. W0 = 0.0 J is the frozen definition applied exactly; it is not a defect. Its consequences are listed under "Open items for the main session".

Rules kept:
- CPU only, at most 2 workers, $0.
- No repo file was edited.
- Nothing was committed, stashed or branch-switched.
- R0-EO was not touched: still running at 07:10, 6 processes, 20 of 64 G shards written.
- The suite's MUJOCO_LOG.TXT was deleted from the repo. No `Users*` folder was created.
- No val or test seed or clip was simulated. The R0-EO shards were only read through the builder's report script outputs.

All my outputs are in `build\verify_m5w\`.

## 1. Frozen checks re-run

| check | how | result |
|---|---|---|
| fb_m5_check (G0-11 + G0-12, A4.1 role) | `scripts/fb_m5_check.py --rec verify_m5\rec\fb_m5rec_60200-60363.npz --rec-w0 m5w\rec\fb_m5rec_w0_r4pre.npz --workers 2 --reuse` -> `verify_m5w\m5w_check_v.json/.log` | exit 0, `{"G011_on_EO_states": true, "G012": true}`, 397 s. A recursive diff against the builder's `m5w_check.json` finds **no difference except wall_s**. |
| A4.1 list | `main` re-derives the list from G_main.pkl (9b833434...) and G_close.pkl (4d1041ce...) and asserts it before anything loads. My own code (`indep_w0.py`) does the same. | 144 seeds (56 main + 88 close), sha256 7b00794f...ba523 = `W0_R4PRE_SHA256`. `opp_fell is not None` and `knockdown` give the same set. The list json (05:54) predates the recording file (06:09). |
| episodes = R4-pre's | `rerecord.py`: all 144 recorded episode dicts vs `r4pre_eval.episodes` (R4-pre's own generator, imported from `r4pre\`, not `fb_r0eo`) | 144/144 identical (JSON). |
| re-recorded bouts = R4-pre bouts | `r4pre_match` in the check | fell_s, opp_fell t, HitMeter counts, max_disp, ContactTail, episode keys equal on 144/144. |
| determinism of the recording | `rerecord.py`: a fresh `fb_m5_check.record` (role w0_r4pre, 20 s, 2 workers; hooked = plain asserted inside) of 67602, 68522, 67173, 68666, 67265, 67667. These cover: early charged, early contact with W = 0, late contact with W = 0, no contact, max W, and late charged. | Every array (post, hit, body, rhit, rbody) and every meta field (episode, row, events, intents, rev_events) is **bit-equal** to the saved recording. |
| W0 statistic (independent) | `indep_w0.py`: its own edge rule (the first step with opponent pelvis up_z < 0.5 or z < 0.25, EO:86), its own 50-step window and W = sum of max(0, F . v) x 0.005 over the work-zone shell rows, its own contact flag. No `replay`, `OppDown` or `FbRewards`. | 144 edges, at EO's and R4-pre's opp_fall t on every bout. 98 have contact. 66 have W > 0, none of them without contact. 32 contact knockdowns have W = 0. **P10 = 0.0 under all 7 numpy percentile methods.** Max \|W difference\| vs the builder: 1.8e-15. Contact flags are equal. Quantiles match the notes: 0 / 0 / 0 / 0 / 0.271 / 2.292 / 4.402 / 15.81. Work-only P10 = 0.10408. |
| W = 0 is not a rounding artefact | the max per-row power over all 32 zero-W contact knockdowns | -0.0113 W, strictly negative. These contacts only ever do negative work: the opponent drives into the R1, or the R1 backs away. |
| n >= 30 | | 98 >= 30. |
| frozen | `fb_rewards.W0 == 0.0`; `frozen_ok` | true. |
| G0-12 recorded, with the frozen W0 | edges / charged | 60200 G 7/3, H 9/4; 60300 G 4/1, H 3/1; w0_r4pre G 144/63. kd_ok, neutral_ok, approach/contact zero while up and tangle (3.6e-15) all true. |
| charged 63 (independent) | W > 0 (66) minus the land gate. I checked the gated edges against the HitMeter's landed head/torso episodes. | 3 gated: 68072, 68200, 68737. Each has a landed torso episode whose end lies within 0.3 s; 68737's episode runs 10.335-10.605 s against a knockdown at 10.88 s. Total 63, of them 22 at t < 0.5 s. |
| G0-12 scripted | 12 cases incl. `frozen_w0` | all ok. alone / no-work / backing give 0 with W = 0; 6.2e-4 J of positive work gives -2.4. |
| G0-11 (EO states) | com and point | Unchanged: com G/H 0/64 mismatch bouts, clean agreement 0.9519 / 0.9578, landed 831 = 831 and 782 = 782. Point mode is ok. |
| hygiene | `fb_seeds.py --audit SEED_LEDGER.md --scan` | ok: 20 roles disjoint, ledger consistent, 25 task inputs clean. |
| full suite | the given pytest command | **350 passed, 2 skipped, 1037 s, exit 0** (`suite_v.log`). |

## 2. Line-by-line review (eval source vs the new code)
- **fb_seeds.**
  - `check(..., ctrl=)` refuses a seed outside the list, a list whose sha256 has changed, and any ctrl other than G, including None. The val, test, screen and Track 2 refusals still come first.
  - `ledger_problems` exempts an overlap only for an identical block that both roles declare in REUSE.
  - The w0_r4pre clip rule is train only.
  - The Track 2 lane's TRACK2_OUT edits are intact, and its tests pass.
- **Episode source.** `work()` uses `fb_r0eo.episodes` with R4PRE_BLOCKS. These are main 67000 reps 64 at 0.40-0.55 m, and close 68000 reps 128 at 0.40/0.45 m, which matches `r4pre_eval.py` and the pkl meta. I checked equality directly against `r4pre_eval.episodes`, as shown above.
- **Bout.** The bout is the same `EO.run(..., threads=1, dr=True, meter=True, falls=True, rev=True, keypoints=None)` as M5. The R4-pre equality on 144/144 shows that p1_eval's G bout and EO.run's G bout coincide on these episodes.
- **Contact and W definitions.** They are unchanged from M5:
  - body sensor: the R1's torso, pelvis and leg shells (20 shells; arm shells excluded);
  - Fb = force on the opponent; vb = the R1 link's COM velocity (`mj_objectVelocity`, mjOBJ_BODY);
  - window = the 50 control steps ending at the edge step, the same for W and for the contact flag.
- **calibrate_w0.**
  - `edges_equal_eo` requires one edge per bout iff EO's opp_fall, at EO's t.
  - The gate uses `w0 is not None` (the old `if w0` would have treated 0.0 as missing; that bug is fixed).
  - `frozen_ok` tolerance: 1e-4 x max(1, w0).
- **FbRewards.** The assert changed from `> 0` to `>= 0`. The `W > 0` clause of `charge` was already in M5 (HEAD e9fdd66, line 60) and is unchanged.
- **Tests.** The new tests check: the list, sha256 and ctrl refusals; the recorder refusing 67003, 60300 and H; the 67602 re-record with its edge at R4-pre's 0.12 s; and W0 == 0 (W = 0 never charges, 1e-6 J does). In `scripted`, `frozen_w0` is ok.
- **Line endings.** The HEAD blobs and working files are both CRLF in all 6 files, the same as the builder's backups. The diffs are content only.
- **Run 1-3 files.** `git diff HEAD -- r1gym scripts/eval_opponent.py gpu/sparring_task.py` is empty.

## 3. A4.2 (report only): spot-checked
- From the pkls with my own code, knockdowns before 0.5 s:
  - G: 24 of 144, at 0.12-0.28 s;
  - H: 23 of 144;
  - A3: 24 of 86.
  - Across controllers: the union is 25 episodes and 22 are early for all three.
- From the recording, at the 24 G early bouts:
  - **0 have contact at the first substep.**
  - First contact comes at 0.095-0.205 s.
  - The opponent's pelvis up_z at step 1 is 0.76-0.81 for kfab_203 (t0 1.54-1.66) and 0.90 for kfab_200 (t0 2.62). Only 2 of the other 120 bouts are below 0.85.
- Small wording slip in M5W_NOTES: the knockdowns occur *at* 0.12-0.28 s, not "0.12-0.28 s later" than first contact. At t0 = 1.60 the knockdown, at 0.12 s, almost coincides with first contact (0.095-0.125 s). 68666 falls at 0.12 s with no contact at all. So at these starts the opponent is already falling.
- The R0-EO partial numbers (13 shards) came from the builder's read-only script. I checked that the script only reads shard rows. I did not re-run it, because the shard set has grown since.

## 4. Open items for the main session (facts; nothing decided here)
1. **W0 = 0.0 makes the threshold vacuous.** The charge is now "any positive leg, pelvis or torso work in the last 1 s, and no landed strike within 0.3 s". The smallest positive W on these bouts is 4.8e-4 J. In float32 on the GPU, any spurious positive power, for example from a sliding contact, is therefore charged a full -2.4.
   - The plan's literal "W >= W0" with W0 = 0 would charge every knockdown. That contradicts G0-12's "tracker-alone fall gives 0". The `W > 0` clause, present since M5, is the only reading consistent with both.
   - The builder's options, each an amendment before M11:
     - (a) keep 0;
     - (b) work-only P10 = 0.104 J;
     - (c) late-only P10, which is also 0;
     - (d) mask the first 0.5 s.
   - Charged counts by threshold, verified: 0 -> 63 and 0.104 -> 56.
2. **Early knockdowns.** 22 of the 63 charged G knockdowns (35%) are start-window artefacts at kfab_203 t0 1.54-1.66 and kfab_200 t0 2.62. They are eval-real: the same episodes are early for G, H and A3. They are largely outside the policy's control.

## Files (build\verify_m5w\)
- `m5w_check_v.json` / `.log`: the full check re-run.
- `indep_w0.py` / `indep_w0.json`: the independent list, W0 and A4.2 counts.
- `rerecord.py` / `rerecord.json` / `.log`: the fresh 6-bout re-record (bit-equal) and the 144/144 episodes vs `r4pre_eval`.
- `suite_v.log`: the full suite.
- SEED_LEDGER.md: one row "M5W verify" appended (6 listed w0_r4pre seeds, G, re-run in role).
