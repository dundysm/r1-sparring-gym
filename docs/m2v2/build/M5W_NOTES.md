# M5W build notes: Amendment A4.1 (W0 on the R4-pre G knockdowns) and A4.2 (early knockdowns), 2026-10-30

**Status:** `scripts/fb_m5_check.py` exits 0: `{"G011_on_EO_states": true, "G012": true}`. The full suite result is in the last section.

**W0 = 0.0 J, frozen** in `gpu/fb_rewards.py` and checked by `fb_m5_check.py` (`frozen_ok` true).
- This is the frozen definition, computed and not tuned: the 10th percentile of W over the R4-pre G knockdowns that had contact.
- Of the 98 knockdowns with contact (n >= 30 holds), **32 (33%) have W = 0 exactly**, so P10 = P25 = 0.
- **Consequence for the main session:** the threshold is now vacuous. The operative rule is the `W > 0` clause that the charge already had (M5, `KnockdownAttribution.charge`): any positive work by the leg, pelvis or torso shells in the last 1 s, and no landed strike in the last 0.3 s.
- On the 144 bouts, that charges **63** knockdowns. **22 of the 63 are start-of-bout knockdowns at t < 0.3 s** (A4.2 below). The options are listed at the end; nothing was decided here.

Rules kept:
- New or untracked-workstream files only. No run 1-3 file was touched.
- Nothing was committed, stashed or branch-switched.
- CPU only, at most 2 workers, $0 spent. R0-EO (4 workers) was left running and untouched.
- No threshold or margin was changed.

## Files changed (all fb_* of this workstream)
| file | change |
|---|---|
| `scripts/fb_seeds.py` | **New role `w0_r4pre`:**<ul><li>blocks 67000-67767 and 68000-68767, train clips;</li><li>`LISTS`: only `W0_R4PRE`, the 144 listed seeds. `W0_R4PRE_SHA256` = 7b00794f254cf39d012500ed54e6d490cc501159b32c69bd36c4cb48424ba523, the sha256 of `json.dumps(sorted list)`, asserted on every check;</li><li>`CTRL_ONLY`: G only, via a new keyword `check(..., ctrl=)`.</li></ul>**`REUSE`** is now a list per role: r2_diag gets 67000-67767, w0_r4pre gets both R4-pre blocks. `ledger_problems` allows two roles to overlap only on an identical block that both declare as a re-run.<br>**Track 2 lane:** its earlier edits to this file (TRACK2_OUT) were left as they were. |
| `scripts/fb_m5_check.py` | **New constants:** `W0_ROLE`, `W0_MIN_N` = 30, `EARLY_S` = 0.5, `R4PRE_BLOCKS`, `R4PRE`.<br>**New functions:**<ul><li>`r4pre_episodes`: R4-pre's own episodes, built with `fb_r0eo.episodes` (R0E check (2) proved it equals `r4pre_eval.episodes`);</li><li>`r4pre_list`: re-derives the list from `G_{main,close}.pkl` and returns the files' sha256;</li><li>`merge`: combines recordings;</li><li>`r4pre_match`: compares re-recorded rows with R4-pre rows;</li><li>`early_kd`: the A4.2 report.</li></ul>**Changed functions:**<ul><li>`work` and `record` pass `ctrl` to `fb_seeds.check`, and use R4-pre episodes for the role;</li><li>`calibrate_w0(role='w0_r4pre')` adds `n_ok` (>= 30), `edges_equal_eo` (one edge per bout, at EO's opp_fall time) and quantiles;</li><li>`recorded_checks` also covers w0_r4pre;</li><li>`scripted()` gains case 10, `frozen_w0`: with the frozen W0, a tracker-alone fall gives 0, contact with zero or negative work gives 0, and 6e-4 J of positive work gives -2.4.</li></ul>**`main`:**<ul><li>new arguments `--rec-w0` and `--r4pre`;</li><li>asserts that the derived list equals the frozen list and its sha256 before anything runs;</li><li>if fewer than 30 contact knockdowns: no W0 and G0-12 fails (stop);</li><li>reports the 60300-60363 W0 under `w0_60300_report` only.</li></ul> |
| `gpu/fb_rewards.py` | `W0 = 0.0`, with a comment. The `FbRewards` assert is now `w0 >= 0`; the charge still requires W > 0. |
| `tests/test_fb_hygiene.py` | `test_w0_r4pre_role_takes_only_the_listed_g_bouts`:<ul><li>the list is 144 unique seeds inside the R4-pre blocks, and its sha256 matches;</li><li>refusals: an unlisted seed, ctrl H / A3 / None, other blocks, val clips, a tampered list (sha256);</li><li>a non-declared overlap of 67000-67767 is flagged.</li></ul> |
| `tests/test_fb_m5.py` | The recorder refuses `w0_r4pre` for an unlisted seed, for seed 60300, and for ctrl H.<br>`test_w0_r4pre_bout_is_the_r4pre_knockdown`: re-records 67602 for 1 s (G). It checks that:<ul><li>the episode is R4-pre's;</li><li>there is exactly one edge, equal to EO's;</li><li>the edge time equals R4-pre's opp_fell time (0.12 s).</li></ul> |
| `tests/test_fb_rewards.py` | `test_w0_frozen`: W0 == 0.0, and a ring with W = 0 never charges while 1e-6 J does. `frozen_w0` was added to the scripted parametrisation. |

Backups of the pre-edit versions are in `build\m5w\` (`*.pre.py`, `fb_seeds.HEAD.py`, and copies of the three tests).

## Order (A4.1 "before any re-recording")
1. **06:00:** `build\m5w\derive_list.py` read `r4pre\runs\G_main.pkl` (sha256 9b833434d2362fda...) and `G_close.pkl` (4d1041cee42a74f0...).
   - It wrote `build\m5w\w0_r4pre_list.json`: 144 seeds (56 main + 88 close) with sha256 7b00794f...ba523 and each seed's R4-pre kd time.
2. The list and its sha256 were frozen into `fb_seeds.py`, and the ledger row was appended (role `w0_r4pre`).
3. Re-recording (06:01-06:09, 2 workers): 144 G bouts. Each ran hooked and plain, and every row was identical (asserted).
4. The pre-freeze check (`m5w_check_pre.json`) computed W0 = 0.0.
5. W0 was frozen, and the check was re-run fully on the saved recordings (`m5w_check.json`).

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, S = scratchpad, B = $S\m1m2\fullbody\build)
```
$PY -I $B\m5w\derive_list.py $S\m1m2\fullbody\r4pre $B\m5w\w0_r4pre_list.json
$PY -I scripts/fb_m5_check.py --out $B\m5w\m5w_check_pre.json --rec $B\verify_m5\rec\fb_m5rec_60200-60363.npz --rec-w0 $B\m5w\rec\fb_m5rec_w0_r4pre.npz --p1 $S\m1m2\gap\run4pre\P1\p1_eval.py --workers 2 --reuse   # records w0_r4pre (absent), reuses 60200-60363
$PY -I scripts/fb_m5_check.py --out $B\m5w\m5w_check.json  ... (same) --reuse      # after the freeze: exit 0, 370 s
$PY -I $B\m5w\early_r4pre.py $S\m1m2\fullbody\r4pre $B\m5w\early_r4pre.json
$PY -I $B\m5w\early_r0eo.py $S\m1m2\fullbody\r0eo\out $B\m5w\early_r0eo_partial.json     # read-only, shards present at 06:50
$PY -I scripts/fb_seeds.py --audit $S\m1m2\fullbody\SEED_LEDGER.md --scan                 # ok: 20 roles; ledger consistent; 25 task inputs clean
$PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests                       # -> m5w\m5w_suite.log
```

**Recordings:**
- 60200-60363 is the M5-verify recording (`verify_m5\rec\fb_m5rec_60200-60363.npz`). It is bit-identical to the M5 build recording, was made with the block-first fix, and was reused rather than re-simulated.
- The new recording is `m5w\rec\fb_m5rec_w0_r4pre.npz` (144 bouts, 27.9 MB).

## Results (`build\m5w\m5w_check.json`)

### A4.1 re-recording
- **List:** re-derived inside the check from the pkls; it equals the frozen list and its sha256.
- **Re-recorded rows vs R4-pre's G rows** (`r4pre_match`, 144 bouts): fell_s, opp_fell time, HitMeter counts, max_disp_m, the ContactTail r1_contact block and the episode keys are **equal on 144/144**. The bouts are R4-pre's bouts.
- **Edges:** 144 bouts, **144 knockdown edges, exactly one per bout, each at EO's opp_fall t_s** (`edges_equal_eo` true).

### W0 calibration
| | value |
|---|---|
| knockdowns / with contact (M5 definition) / with W > 0 | 144 / **98** / 66 (every W > 0 knockdown has contact) |
| n >= 30 | yes (98) |
| W over contact knockdowns: P0 / P5 / **P10** / P25 / P50 / P75 / P90 / max (J) | 0 / 0 / **0** / 0 / 0.271 / 2.292 / 4.402 / 15.81 |
| share of contact knockdowns with W = 0 | 0.327 (32) |
| **W0 frozen** | **0.0 J** |
| report: work-only P10 (over W > 0) | 0.1041 J |
| report: 60300-60363 (old, n = 2) | 0.29906 J |

**Charged G knockdowns on the 144 bouts, by W0 (report only; the strike gate removes 3 that have W > 0):**

| W0 (J) | 0 (frozen) | 0.104 | 0.299 | 1.0 | 2.99 |
|---|---|---|---|---|---|
| charged | 63 | 56 | 46 | 35 | 17 |
| of them t < 0.5 s | 22 | 22 | 22 | 19 | 7 |

### G0-12 with the frozen W0
- **Scripted:** 12/12 ok, including `frozen_w0`.
- **Recorded bouts:**

  | set | bouts | edges | charged | kd_ok | down-neutral only while down | approach / contact 0 while up | tangle vs numpy |
  |---|---|---|---|---|---|---|---|
  | 60200 G | 64 | 7 | 3 | yes | yes | yes | 3.6e-15 |
  | 60200 H | 64 | 9 | 4 | yes | yes | yes | 3.6e-15 |
  | 60300 G | 64 | 4 | 1 | yes | yes | yes | 1.8e-15 |
  | 60300 H | 64 | 3 | 1 | yes | yes | yes | 1.8e-15 |
  | w0_r4pre G | 144 | 144 | 63 | yes | yes | yes | 3.6e-15 |

  - 30 of the w0_r4pre bouts got up again.
  - The charged counts for 60200 G and H went from 2 and 3 to 3 and 4 because W0 went from 0.299 to 0.
- **G0-11** (EO states, unchanged): com G/H have 0/64 mismatch bouts, clean agreement 0.9519 / 0.9578, median impulse rel 4.4e-6 / 8.9e-6, paid 270 = 270 and 258 = 258. Point mode: 0.9952 / 0.9987.

## A4.2: early knockdowns (t < 0.5 s), report only

| source | bouts | knockdowns | t < 0.5 s | of them with R1 contact (ContactTail 0.5 s) |
|---|---|---|---|---|
| R4-pre G (p1_eval = EO bout), 67000-67767 + 68000-68767 | 1536 | 144 | **24 (16.7%)**, all at 0.12-0.28 s | 23 |
| R4-pre H | 1536 | 144 | 23 | 22 |
| R4-pre A3 | 1536 | 86 | 24 (10 then fell) | 24 |
| **Eval's own bouts: R0-EO G** (EO.run unchanged; 13 of 64 shards, read-only) | 2496 | 210 | **40 (19%)** | 30 |
| R0-EO H / A3 (same 13 shards) | 2496 | 225 / 161 | 42 / 41 | 32 / 34 |
| M5 recordings 60200-60363, G and H | 256 | 23 | 4 (60201, 60230, both controllers) | 4 |

**Does the eval's reset produce them? Yes. They are a property of the start, not of the defender's behaviour.**
- **The same episodes, whatever the controller.**
  - R4-pre: 22 of the 25 early-knockdown episodes are early for all three controllers (G, H and A3). G's 24 are early for H on 23 and for A3 on 23.
  - R0-EO: 37 of 45 are early for all three.
- **Narrow start-time windows (the eval's `pick_start` over the clean-start grid).**
  - The windows are kfab_203 at t0 = 1.52-1.66 s (and 0.84 and 2.52 s), and kfab_200 at t0 = 2.60-2.64 s.
  - Inside kfab_203 1.54-1.66 s and kfab_200 2.62 s: R4-pre G has **24 early knockdowns in 40 bouts and 0 in the other 1496**. R0-EO G has 29 in 61 and 11 in 2435.
- **Mechanism (re-recorded contact logs, 24 G bouts):**
  - Contact never exists at the reset instant: 0 of 24 touch at the first substep, so the clean-start mask holds.
  - First contact comes at 0.095-0.205 s, and the knockdown 0.12-0.28 s later.
  - In all 21 kfab_203 cases, the opponent pelvis up-axis z at the first step is already 0.76-0.81 (36-40 deg of lean). The 3 kfab_200 cases are at 0.90. Below 0.85 occurs in only 2 of the other 120 bouts.
  - The contact pairs are mostly the **R1 thigh against the opponent's head** (R4-pre ContactTail: thigh 47 and shin 9 pairs; opponent zones head 32, arm 15, leg 8).
  - So at these t0 the clip is in a deep forward lean or duck. At 0.40-0.55 m, the opponent's head drives into the R1's thighs and the opponent tilts over (cause tilt in every case).
- **Under the frozen W0**, 22 of these 24 are charged. Only 68666 (no contact) and 68522 (contact, W = 0) are not. They are **35% of G's 63 charged knockdowns**, and they occur almost regardless of the policy.

## Notes for the main session (facts; nothing decided here)
1. **W0 = 0 is degenerate.**
   - The A4.1 definition, applied as frozen, gives 0. The `W > 0` clause of the charge rule (in the code since M5 and verified) is what remains. The plan's text "W >= W0" without that clause would charge every knockdown, including tracker-alone falls (which G0-12 forbids).
   - The value is frozen as computed. Options if the main session wants a non-vacuous threshold (each needs an amendment before M11):
     - (a) keep 0 (any positive work);
     - (b) work-only P10 = 0.104 J;
     - (c) P10 over contact knockdowns with t >= 0.5 s: also 0 (31 of 75 have W = 0);
     - (d) exclude early knockdowns from the charge (see 2).
2. **Early knockdowns are a start-distribution artifact of the train-clip episodes.**
   - They are eval-real: the eval's own bouts show them at the same rate, so R-EQ and R0-EO include them consistently.
   - But under W0 = 0 they put an uncontrollable -2.4 into roughly 1.5% of G episodes (24/1536). Possible responses: a knockdown mask in the first 0.5 s (like the push mask), or leaving it as is.
   - The val clip's start windows were not examined: no val data was opened.
3. **R0E builder issue 2** (early knockdowns flagged at 0.12-0.30 s in shards 0-4) is the same phenomenon, confirmed here with contact logs.

## Ledger
- One row was appended ("M5W"): 67000-67767 and 68000-68767, 144 listed seeds, role `w0_r4pre`, G only.
- It also covers the 67602 1-s re-runs in tests/test_fb_m5.py (the test and the full suite).
- `fb_seeds.py --audit --scan`: ok.

## Suite
- **350 passed, 2 skipped** in 1121 s (exit 0; `build\m5w\m5w_suite.log`). The 2 skipped tests are the env-gated ones.
- The run was slower than M5's 702 s because R0-EO holds 4 cores.
- The suite left a MUJOCO_LOG.TXT in the repo; it was deleted. No `Users*` folder was created.
- R0-EO is still running (44 of 189 tasks done at 07:05).
