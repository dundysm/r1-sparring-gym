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
None.
