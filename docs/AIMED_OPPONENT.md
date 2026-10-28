# Aimed opponent: preregistration

Fixed before the first aimed validation or test eval (milestone M0). The hit metric, the primary metric, the splits and
the decision rules below don't change after that. A change goes into **Amendments** at the bottom, dated and with its
reason, and results computed under the old text keep the old hash.

**Why fix them first.** In the review probe (36 paired 10 s bouts on train clips, 0.40-0.55 m, arms-down R1), design A's
arm aim changed head hits per minute by +4.7 [0.5, 9.0], +3.2 [-0.5, 7.0] or -0.3 [-2.3, 2.0] depending only on how a hit
was counted. Choosing the definition after seeing the effect would decide the sign.

**Provenance.** `scripts/eval_opponent.py` (`provenance()`) writes into every result's `meta`: `git_hash` and `git_dirty`,
`config_hash` (the base config) and `controller_config_hashes`, `prereg_sha` (sha256 of this file with LF line endings,
first 16 hex), `mujoco`, `shell_solref` (peak forces depend on it), `opponent_caps` (cap policy and N·m per joint) and
`hit_metric`. Later scripts (aim_check, tolerance, red-team) record the same.

## Definitions
Implemented in `r1gym/hits.py` (`HitMeter`); the numbers are `hit_metric` in `configs/r1_assumptions.yaml`.

**How contacts are read.** The meter is an entry in `R1HitEnv.substep_hooks`, called after every `mujoco.mj_step`. It only
reads MjData: a bout is bit-identical with it on or off (`tests/test_hits.py`). After `mj_step`, the contacts, constraint
forces and body velocities all come from that step's forward pass, so force × 5 ms is the impulse the step applied, and the
closing speed is the speed before this step's contact force acted. Only robot-vs-robot contacts count (the visible shells,
`robot_contact: shell`; with `capsule` the zones are the same bodies, the head being the `head_collision` capsule).

**R1 zones** (by shell mesh):

| zone | shells |
|---|---|
| head | `head_pitch_link` and `head_yaw_link` meshes, fixed on `torso_link` |
| torso | the `waist_yaw_link` shell on `torso_link` |
| pelvis | `pelvis_link`, `imu_in_pelvis_link`, `waist_roll_link` |
| arm | shoulder, elbow and wrist links |
| leg | hip, knee and ankle links |

**Opponent limb groups:** hand (wrist_roll), forearm (elbow), upper_arm (shoulder links), foot (ankle links), shin (knee),
thigh (hip links), trunk (pelvis, waist and torso, its head included). **Strike limbs:** hand, forearm, foot, shin.

**Contact episode.** Per (opponent limb group, R1 zone), left and right pooled: the substeps in which the pair has contact
force, where two such substeps at most `merge_gap_s` = 0.1 s apart belong to one episode (a contact-free gap of up to
`merge_gap_s` - 5 ms = 95 ms), so a bounce is one hit. Per episode:
- onset time;
- closing speed along the contact normal at first touch, at the pair's strongest contact point;
- duration: first to last substep, both included (a single substep lasts 5 ms);
- peak normal force: the pair's contact points summed within a substep, maximum over the episode;
- impulse: the sum of force × timestep.

**Landed:** a strike limb on the head or torso, impulse >= `min_impulse_Ns` = 0.5 N·s and peak >= `min_peak_N` = 5 N.

**Clean strike:** landed, closing speed >= `v_close_min` = 1.0 m/s at onset, and duration <= `max_strike_s` = 0.15 s.
**Press:** a landed episode that isn't clean.

**Strike window.** A library strike's [onset, end] (M1.2's detector on the reference the tracker follows), shifted by the
tracker's measured lag (median reference-to-tracked apex delay on train clips, measured once in M1.4 and frozen in
`results/opponent_tracker/aim.json`); or, for an aimed strike, its StrikeIntent's [t_on, t_apex + 0.15 s]. Each window
belongs to one limb (side included).

**Window outcome.** Over the limb's episodes that start inside the window:
- **blocked** if it touched the R1's arms before its first landed episode on the head or torso, or touched the arms and
  landed nowhere;
- otherwise **head** or **torso**, by its first landed episode;
- otherwise **miss**.

**Blocked share:** blocked windows / all windows. Until M1.2 supplies windows, eval summaries report `arm_share_of_clean`
(clean-rule contacts on the R1's arms / those on arms, head and torso) and don't call it blocking.

**Actual spacing:** the pelvis-to-pelvis horizontal distance at every control step while the R1 is up; per bout p10,
median, p90, next to the nominal bout distance.

**Rates:** per bout, the count divided by the minutes until the R1 fell or the bout ended (the opponent falling doesn't end
it). Over bouts, a rate is **exposure-weighted**: total count / total minutes, i.e. the per-bout rates weighted by their
minutes. The plain mean of per-bout rates (`bout_mean`) is reported as secondary. Why: a bout the R1 falls in after 1.5 s
turns one contact into 40/min, so a plain mean (and its bootstrap) follows small shifts in fall time rather than aiming.

## Metrics
- **Primary: head clean strikes per minute**, exposure-weighted over bouts (`rate`), with a cluster bootstrap 95% CI. The
  effect of aiming is aimed minus unaimed on identical episodes: each side's total / total minutes, the same clusters
  resampled for both (`cluster_paired_diff` with each bout's minutes as weights).
- **Co-primary, aim accuracy:** per aimed window, the share that is head (head-landed share), torso, blocked or miss, and
  the closest palm-to-target distance in cm.
- **Secondary:**
  - the plain mean of per-bout rates (`bout_mean`) for every rate;
  - head and torso landed per minute (presses included);
  - head impulse per minute: every strike-limb episode on the head;
  - impulse per head hit (median, p90);
  - peak force per landed hit (median, p95, max), always with the shell solref;
  - actual spacing;
  - opponent falls, alone (eval_tracker, 48 runs) and in bouts (`opponent_fell`);
  - defender survival (Wilson CI), up and within 1 m, drift.
- **Sensitivity, in every summary:** the same contact log scored four more ways, each estimated like the primary. The
  review probe's unaimed head rates (train clips, 0.40-0.55 m, hold-position on) are given for scale. A merge gap is the
  time between consecutive contact substeps, so 20 ms merges contact-free gaps of up to 15 ms.

  | rule | merge gap | force floor | counts | review probe, head/min |
  |---|---|---|---|---|
  | `clean` (primary) | 0.1 s | 0 | as above | — |
  | `landed` | 0.1 s | 0 | as above | — |
  | `A_peak20` (design A's actual counting) | 20 ms | 5 N | strike limb, peak >= 20 N; pelvis counts as torso | 13.7 |
  | `B_strike` (design B's strike rule) | 0.1 s | 0 | strike limb or thigh, closing >= 1 m/s, duration <= 0.15 s, peak >= 5 N | 8.2 |
  | `impulse_only` | 0.1 s | 0 | strike limb or thigh, impulse >= 0.5 N·s | 11.5 |

  The rules count episodes as the probe did; `B_strike`'s duration is last minus first substep (31 substeps fit in
  0.15 s), where the primary's includes both ends (30). The probe's rates differ in three ways: they are plain means of
  per-bout rates, its bouts also ended when the opponent fell (1 of its 36 unaimed bouts, after 0.2 s), and its closing
  speed came from the first contact point found, about the body origin (here: the strongest point, about the COM).

- **Regression:** the v1 tiers' survival and trajectories are bit-identical with the meter on (max |Δqpos| = 0).

## Splits
- **Train** kfab_200, kfab_201, kfab_203 (25.0 s after cuts); **val** kfab_204 (3.9 s); **test** kfab_202, kfab_205
  (4.5 s). kfab_199 is dropped: the tracker can't perform it.
- kfab_204 was used by both designs' probes and by the review probe. Validation is therefore not clean, and **test is the
  only clean confirmation**.
- The tracker trained on all six clips' unwarped motion (`gpu/export_motion.py`). Aiming is measured on unseen strikes
  only if M2's warps come from train-clip strikes alone.
- Every choice is made on train + val and written down (with a `_why`) before test is run, once.
- Test is thin: about 10-16 strikes in 4.5 s, 1-2 of them at head height. Claims about test stay narrow.

## Statistics
- Bootstrap: 4000 resamples, seed 0, percentile 95% intervals. Whole **(clip, start)** clusters are resampled
  (`r1gym/stats.py`: `cluster_paired_diff`, `rate_ci`). Rates are weighted by minutes, so every resample's rate is again
  total count / total minutes.
  - In `eval_opponent.py` each bout draws its own start, so clusters are mostly single bouts.
  - In aim_check each (clip, start) runs at several distances and under every operator; those bouts share a cluster.
- Survival rates: Wilson intervals.
- "Above 0" means the lower bound of the 95% CI is above 0.

## Decision rules
**M1 operator (M1.4).** aim_check on train + val, before any test run:
1. Candidates: standoff, armwarp, armwarp_vz, standoff + armwarp_vz, and select if its splice tolerance passes.
2. An operator is rejected if opponent falls in bouts exceed the unaimed rate + 2 pp, pooled over the grid.
3. Of the rest, pick the one with the largest paired Δ primary (aimed minus none, arms-down target) whose CI is above 0.
4. If the top two differ by less than 0.5 strikes/min, the higher head-landed share per window wins, then the simpler one,
   in this order: standoff, armwarp_vz, armwarp, standoff + armwarp_vz, select.
5. If no operator's CI is above 0, M1 ships StandOff as "distance management", says so, and M2 goes ahead.
6. The chosen operator's Δ is reported under every sensitivity rule. If any rule flips its sign, the headline says so in
   the same sentence.
7. The choice and its `_why` go into `results/opponent_tracker/aim.json` before any test run.

**M1.5 hard `--aim` tier.** Δ survival and Δ primary per controller, with and without scripted pushes, on val and then
once on test, reported whatever their sign.

**M2 gates.** Selected on train + val only:
- alone: >= 45/48 in eval_tracker (shipped 46/48), with unwarped joint RMSE within shipped + 0.02 rad;
- parity_mjlab on warped references: shared falls within 0.1 s;
- tolerance on val/test strikes never warped in training: 20 cm warps within 5 cm in >= 65% (shipped 42.4%), and a 20 cm
  lunge adds >= 12 cm of median reach (shipped 1.8 cm);
- aim_check against the arms-down and guard targets: primary and head-landed share per window both beat M1's chosen
  operator with paired CIs above 0, and opponent falls in bouts are at most shipped + 2 pp.

The goal-conditioned fallback run is made only if run 1 misses the tolerance gates. Then hard `--aim` on val, then on test
once, reported whatever the sign.

**B2 gate (two-robot duel, GPU).** Both must hold:
- B1's red-team parameters beat the hand-made M2 scheduler on val by >= 1.0 head clean strikes/min **and** >= 20%
  relative, with a paired CI above 0. Opponent falls must stay <= base + 2 pp, with no rise in rule violations
  (presses > 0.3 s, trunk ramming, opponent tilt).
- A MuJoCo Warp check shows that robot-robot contact (capsules or hulls) works.

## What the numbers can't say
- **Peak force is bounded by the timestep.** Shell solref 0.01 is 2 × the 5 ms timestep, the stiffest MuJoCo allows; peak
  force halves at 0.02 while impulse holds. Claims are about impulse and closing speed. A peak is always quoted with its
  solref.
- **Momentum check** (`tests/test_hits.py`). A free 0.5 kg fist shell thrown at the face at 0.5-3 m/s:
  - the measured impulse is within 1% of the momentum it lost (the test allows 10%);
  - the closing speed is within 1.5% of the throw;
  - touches under 5 N, or under 0.5 N·s, are not landed hits.
- **Opponent arms.** They run the mjlab caps (60/33 N·m), where unitree_mujoco lists 25 N·m, so strikes may be optimistic.
  A "min"-caps sensitivity run is reported (M1.6).
- **Privileged aim.** The aim reads the R1's true torso pose, as steering already does.
- **Rigid head.** The head is rigid on `torso_link`. The real R1 has a 2-DoF neck holding the binocular camera, so head
  contact on hardware means touch level and a foam head cover.

## Amendments
**1. 2026-10-27, before any aimed run.** Written while implementing M1.2-M1.4, before the tracker tolerance runs and before
anything aimed ran on any split. The text above left these open:
- **Strike library** (`r1gym/strikes.py`): onset is the forward speed rising through 1.5 m/s; the apex is the most forward
  point within 0.4 s and must be >= 0.25 m ahead of the spot; the end is the first frame from the apex with forward speed
  <= 0. In this motion that is the apex or the frame after it, so a library window is [onset, apex] shifted by the lag. The
  library has 55 strikes (51 hands). Test has 9 of them, one level with the head (the 10-16 and 1-2 above came from the
  design probes' rules).
- **Unaimed side of an aim comparison:** the scheduler runs without aiming (`opponent_aim: shadow`, bit-identical to
  `none`), so both sides are scored on the same intent windows, [t_on, t_apex + 0.15 s].
- **Zone:** every hand strike is aimed at the head (`aim.zone: head`). Feet are not aimed.
- **Upright gate:** "after a landed hit" is read as any contact between a strike limb and the R1's head or torso shells,
  checked at the control rate. The meter's episodes close only after the merge gap, too late for the gate.
- **ArmWarp's pull toward the clip's arm angles:** 0.001. Design A's 0.02 left 1.3 cm (up to 2.7 cm) between the palm and
  reachable targets, and M1.3 requires <= 1 cm.
- **Splice tolerance:** 'select' is a candidate only if, with the opponent alone over 48 runs of 10 s on train and val clips,
  it falls in at most one run more than without splices.
- **aim_check grid:** per train or val clip, 3 start times that are clean at every distance in both reference poses; obs
  delay 1 step; no scripted hits; nominal physics. The mannequin is the R1 welded at its pelvis and held at home by the
  residual config's PD stand.
- **Decision rules:**
  - Rule 2 pools opponent falls over the whole grid at shell solref 0.01: all three targets and every distance.
  - Rule 3's CI is the cluster bootstrap of the exposure-weighted paired difference, aimed minus shadow, on the arms-down
    target.
  - Rule 4's "top two" are the two largest differences among the candidates whose CI is above 0.

## Results
M1.2-M1.5, added after the runs. Everything between the markers is generated from the stored JSON by
`python scripts/aim_tables.py --write`, which first checks that the text above this section is unchanged: it hashes to the
`prereg_sha` the first M1 results carry (the whole file at the time, which had no Results section). Later results record
both the whole file's hash (`prereg_sha`) and this one (`prereg_rules_sha`).

<!-- results:begin -->
### Provenance and the test-once record

- The 18 result files of the first M1 runs (`ASRUN` in `scripts/aim_tables.py`; the hard tier now archived in `results/eval_opponent/aim_pre14/`) carry `prereg_sha` 562834645ce51777; this file without this Results section hashes to 562834645ce51777 (the same). The definitions, rules and amendment 1 above are the text they were computed under.
- They were run from the uncommitted working tree on adcd6cb (`git_dirty` true): 72ad022, which added this section, is the code that produced them.
- Order, from the files' write times and the run logs: amendment 1; the strike library; tolerance; aim check and `aim.json`; the hard tier on val (unaimed and aimed, with and without scripted hits); then the hard tier on test, each variant once; then the standard and puppet tiers, val before test. No test result was generated twice, and nothing was chosen after a test run. The standard-tier val `--aim` run was repeated once (val, after the test runs) because its first run recorded no commit (git failed during it): the bouts and summary came out identical, only `meta` changed.

### M1.2 strike library

- 55 strikes, 51 with a hand and 4 kicks. Train 38 (5 level with the head) in 25.0 s, val 8 (1) in 3.9 s, test 9 (1) in 4.5 s: test can say little about head-level strikes.
- Audit against 106 labelled candidates (9 unsure, left out): precision 0.78 [0.58, 0.90], recall 0.79 [0.59, 0.91]; hands 0.76 / 0.81; val 1.00 / 0.78. One labeller, the agent that wrote the detector (`strike_labels.json`: `labeller`), no second rater.
- The design probes counted 70 (A) and 107 (B) events; the rule variants on the same labels are in `strikes.json` `variants` (B-like trigger: 115 events, precision 0.52).

### M1.4 tolerance (opponent alone, train + val)

| reference change | palm within 5 cm of the apex | design B's probe |
|---|---|---|
| none | 69.8% of 43 | 72.7% |
| apex 10 cm up | 67.4% of 43 | 65.2% |
| apex 20 cm up | 53.5% of 43 | 42.4% |

A 20 cm lunge at 1 m/s adds 3.5 cm of reach (median; B's probe 1.8 cm). Falls in 48 runs of 10 s: unchanged 2, retimed 0.7x 2, 0.5x 16, spliced ('select') 15. Lag 20 ms (train median, frozen in `aim.json`).

### M1.4 decision rules (aim check, train + val; `results/opponent_aim/aim_check.json` → `results/opponent_tracker/aim.json`)

1. **Candidates:** StandOff, ArmWarp, ArmWarp height and sideways only, StandOff + ArmWarp (height, sideways). The selector is out: spliced runs fell 13 more times than unchanged ones (amendment 1 allows 1).
2. **Opponent falls,** pooled over the grid at solref 0.01 (unaimed 0.5%, limit +2 pp): StandOff 0.0%; ArmWarp 0.5%; ArmWarp, height and sideways only 0.0%; StandOff + ArmWarp (height, sideways) 0.9%. None rejected.
3. **Paired Δ head clean strikes/min, arms-down target** (aimed minus shadow, cluster bootstrap): StandOff +1.00 [0.00, 2.08]; ArmWarp +2.08 [0.33, 3.75]; ArmWarp, height and sideways only +2.58 [0.75, 4.50]; StandOff + ArmWarp (height, sideways) +3.17 [1.08, 5.17].
4. **Tie-break:** the top two with CIs above 0 differ by 0.584/min (not under 0.5), so the largest wins: **StandOff + ArmWarp (height, sideways)**.
5. **Fallback:** not triggered.
6. **Every hit rule:** the chosen Δ is clean +3.17, landed +6.00, A_peak20 +7.92, B_strike +3.08, impulse_only +6.00/min; the sign holds under every rule.
7. **Written before any test run:** `aim.json` (`decided_on`: "train + val (results/opponent_aim/aim_check.json), before any aimed test run").

The other targets, same operator: the welded mannequin +1.8 [-1.0, 4.4] (train clips +3.7 [1.2, 6.0], val clip -4.0 [-9.0, -1.0]), the guard +0.9 [-0.2, 2.0]. Neither CI is above 0 on its own; the rule is judged on the arms-down target only.

### M1.5 hard `--aim` tier, as run against PR #13's stepping bases (archived: `results/eval_opponent/aim_pre14/`)

Aimed minus unaimed, same episodes. These were the shipped policies when the runs were made; PR #14 replaced them.

| split | scripted hits | controller | Δ survival [95% CI] | Δ head clean strikes/min [95% CI] | Δ opponent falls [95% CI] | hit rules with the other sign |
|---|---|---|---|---|---|---|
| val | on | PD stand | +0.000 [0.000, 0.000] | -2.87 [-7.51, 1.84] | +0.031 [0.000, 0.081] | A_peak20 |
| val | on | RL v1 (residual) | +0.000 [0.000, 0.000] | -5.32 [-8.85, -1.91] | +0.031 [0.000, 0.081] | A_peak20 |
| val | on | RL v2 (residual) | +0.000 [0.000, 0.000] | -4.99 [-8.49, -1.20] | +0.016 [0.000, 0.051] | A_peak20 |
| val | on | RL v3 cand. (residual) | +0.000 [0.000, 0.000] | -3.35 [-6.83, 0.29] | +0.031 [0.000, 0.081] | – |
| val | on | Stepping base, stage 1 | +0.062 [-0.016, 0.149] | -1.39 [-3.39, 0.62] | -0.016 [-0.065, 0.034] | landed, A_peak20, impulse_only |
| val | on | Stepping base, arms down (PR #13) | -0.016 [-0.081, 0.052] | +1.44 [-0.08, 2.92] | +0.062 [-0.017, 0.149] | – |
| val | on | Stepping base + guard (PR #13) | -0.062 [-0.138, 0.000] | +2.31 [0.99, 3.66] | +0.109 [0.000, 0.215] | – |
| val | off | PD stand | -0.156 [-0.274, -0.058] | -9.23 [-12.45, -6.02] | +0.000 [0.000, 0.000] | – |
| val | off | RL v1 (residual) | -0.078 [-0.167, 0.000] | -1.80 [-5.44, 1.33] | +0.016 [0.000, 0.051] | A_peak20 |
| val | off | RL v2 (residual) | -0.109 [-0.197, -0.030] | -6.05 [-8.89, -3.25] | +0.016 [0.000, 0.051] | landed, A_peak20, impulse_only |
| val | off | RL v3 cand. (residual) | -0.062 [-0.143, 0.000] | -4.21 [-7.79, -0.83] | +0.000 [0.000, 0.000] | A_peak20 |
| val | off | Stepping base, stage 1 | -0.031 [-0.106, 0.045] | -5.20 [-7.80, -2.56] | +0.000 [0.000, 0.000] | A_peak20 |
| val | off | Stepping base, arms down (PR #13) | +0.000 [0.000, 0.000] | +1.79 [0.00, 3.48] | +0.047 [0.000, 0.104] | – |
| val | off | Stepping base + guard (PR #13) | +0.000 [-0.048, 0.045] | +2.60 [1.08, 4.20] | +0.016 [-0.051, 0.085] | – |
| test | on | PD stand | +0.000 [0.000, 0.000] | +5.49 [3.24, 7.88] | +0.000 [0.000, 0.000] | – |
| test | on | RL v1 (residual) | +0.000 [0.000, 0.000] | +5.73 [3.78, 7.75] | +0.008 [0.000, 0.024] | – |
| test | on | RL v2 (residual) | +0.000 [0.000, 0.000] | +3.51 [1.57, 5.68] | +0.016 [0.000, 0.040] | – |
| test | on | RL v3 cand. (residual) | +0.000 [0.000, 0.000] | +3.25 [1.14, 5.42] | +0.023 [0.000, 0.054] | – |
| test | on | Stepping base, stage 1 | -0.016 [-0.072, 0.047] | +5.23 [3.92, 6.65] | +0.000 [0.000, 0.000] | – |
| test | on | Stepping base, arms down (PR #13) | -0.023 [-0.084, 0.033] | +2.45 [1.43, 3.49] | +0.141 [0.083, 0.205] | – |
| test | on | Stepping base + guard (PR #13) | -0.062 [-0.116, -0.015] | +1.15 [0.51, 1.79] | +0.133 [0.077, 0.194] | – |
| test | off | PD stand | -0.258 [-0.331, -0.189] | +8.11 [5.59, 10.65] | +0.000 [0.000, 0.000] | – |
| test | off | RL v1 (residual) | -0.156 [-0.229, -0.083] | +4.99 [3.51, 6.51] | +0.000 [0.000, 0.000] | – |
| test | off | RL v2 (residual) | -0.180 [-0.244, -0.118] | +5.36 [3.59, 7.29] | +0.008 [0.000, 0.024] | – |
| test | off | RL v3 cand. (residual) | -0.141 [-0.205, -0.085] | +5.05 [2.89, 7.26] | +0.008 [0.000, 0.024] | – |
| test | off | Stepping base, stage 1 | -0.086 [-0.156, -0.024] | +4.46 [2.93, 5.94] | +0.023 [0.000, 0.054] | – |
| test | off | Stepping base, arms down (PR #13) | +0.000 [-0.024, 0.023] | +4.68 [3.58, 5.86] | +0.094 [0.047, 0.146] | – |
| test | off | Stepping base + guard (PR #13) | +0.000 [-0.023, 0.024] | +1.69 [1.10, 2.30] | +0.109 [0.060, 0.161] | – |

- **Stepping bases on test:** arms down +2.45 [1.43, 3.49], guard +1.15 [0.51, 1.79] head clean strikes/min; on val guard +2.31 [0.99, 3.66], arms down +1.44 [-0.08, 2.92]. Guard survival on test -0.062 [-0.116, -0.015].
- **Opponent falls** on test: 10.9% → 25.0% (arms down), 10.9% → 24.2% (guard). M1.5 has no fall rule; rule 2 (aim check) passed, so this is reported, not a rejection.
- **PD, the residual policies and stage 1**, which fall in most hard bouts, show the aim's effect with opposite signs on val and test; their bouts are short, so their rates rest on few minutes. They are reported, not interpreted.

### Addendum: re-measured against the PR #14 defenders after merging main; operator unchanged (`results/eval_opponent/aim/`)

Merging main (PR #14's footwork policies, PR #15, PR #16) made the footwork policies the shipped defenders. The hard tier was run again, unaimed (scheduler in shadow) and aimed (standoff+armwarp_vz, from `aim.json` as fixed on train + val; nothing selected or tuned here), with and without scripted hits, against all 9 controllers of main's headline: val first, then test once. Code: 72ad022 with main 6b42d13 merged in, uncommitted (the merge commit pins it); these 8 files ran under the rules hash 562834645ce51777 (the text above). The footwork policies run with their `gym.json` as shipped (hold your spot; the guard also turns to face).

- **Same bouts where nothing changed:** PD, v1-v3, stage 1 and PR #13's bases reproduce the archived runs bout for bout in every file, hits included (yes), and every controller's unaimed bouts match main's stored headline files in every field those have (yes). The re-measurement differs from the as-run one only in its defenders.

| split | scripted hits | controller | Δ survival [95% CI] | Δ head clean strikes/min [95% CI] | Δ opponent falls [95% CI] | hit rules with the other sign |
|---|---|---|---|---|---|---|
| val | on | **Footwork, arms down (shipped)** | -0.078 [-0.161, -0.015] | +1.19 [-0.40, 2.76] | +0.016 [-0.101, 0.117] | – |
| val | on | **Footwork + guard (shipped)** | -0.016 [-0.065, 0.034] | +1.35 [0.18, 2.49] | +0.094 [0.000, 0.197] | – |
| val | off | **Footwork, arms down (shipped)** | -0.016 [-0.067, 0.034] | +1.54 [-0.29, 3.40] | +0.031 [0.000, 0.078] | – |
| val | off | **Footwork + guard (shipped)** | +0.016 [0.000, 0.051] | +2.67 [1.55, 3.74] | +0.047 [0.000, 0.102] | – |
| test | on | **Footwork, arms down (shipped)** | +0.016 [-0.038, 0.071] | +3.28 [2.18, 4.44] | +0.133 [0.061, 0.211] | – |
| test | on | **Footwork + guard (shipped)** | -0.023 [-0.060, 0.008] | +2.15 [1.17, 3.18] | +0.133 [0.074, 0.197] | – |
| test | off | **Footwork, arms down (shipped)** | -0.008 [-0.024, 0.000] | +4.85 [3.71, 6.06] | +0.070 [0.031, 0.116] | – |
| test | off | **Footwork + guard (shipped)** | -0.023 [-0.063, 0.000] | +2.62 [1.68, 3.56] | +0.086 [0.029, 0.150] | – |

- **Head clean strikes/min, footwork arms down:** test +3.28 [2.18, 4.44] (without scripted hits +4.85 [3.71, 6.06]), val +1.19 [-0.40, 2.76]; unaimed → aimed on test 5.41 → 8.68.
- **Footwork guard:** test +2.15 [1.17, 3.18] (without scripted hits +2.62 [1.68, 3.56]), val +1.35 [0.18, 2.49]; 2.09 → 4.24.
- **Survival** on test: footwork arms down 0.906 → 0.922 (+0.016 [-0.038, 0.071]), footwork guard 0.992 → 0.969 (-0.023 [-0.060, 0.008]); up and within 1 m 0.727 → 0.680 and 0.812 → 0.719.
- **Opponent falls** on test: 12.5% → 25.8% (footwork arms down, +0.133 [0.061, 0.211]), 11.7% → 25.0% (footwork guard, +0.133 [0.074, 0.197]).
- Every controller's rows, the other hit rules and the window outcomes are in `results/eval_opponent/aim/summary_*_aim.json` (`vs_unaimed`).

### What this doesn't show

- Most aimed windows can't reach the head from where the opponent stands: on the aim check's arms-down target 796 of 1056 windows' targets are 8 cm or more beyond the arm, and 48% of windows are not aimed (lean or touch gate). The effect is per minute of bout, not per strike.
- The operator was chosen against PR #13's arms-down base (aim check, train + val) and not re-chosen for the footwork policies.
- Test holds 9 library strikes, 1 of them level with the head; a test result is one replication on two short clips.
<!-- results:end -->
