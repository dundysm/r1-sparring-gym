# Run 3 transfer gap: synthesis (2026-10-29)

Sources: the five gap analyses under `scratchpad\m1m2\gap\` (falls, headclean, opponent, replay, sim; each folder's NOTES.md),
`run4design\RUN4_PLAN.md` and `CRITIQUE.md`, and docs/SPARRING_M1M2.md 9.7-9.8 at HEAD b81fa16. Writing this synthesis involved
no new simulation. The repo is untouched.

**Data provenance, for the run-4 prereg.**
- All five analyses read run 3's hard_r2 val raw files (`results/eval_opponent/m2r3/raw_val_hard_r2_nohits{,_aim}.json`).
- Every new simulation used dev seeds:
  - 50000-50255 on the val clip kfab_204;
  - 51000+ on the train clips kfab_200/201/203;
  - 50000-50063 on the CPU replica of the training attacker.
- Nothing touched the hard_r2 test split (seeds 22000+, clips kfab_202/205), and no GPU was used.
- **The val clip itself has now been analysed strike by strike** (`opponent/per_strike.json`). That makes any run-4 val screen
  on kfab_204 a design-informed screen. Section 4 covers this.

## 0. Answer in five lines

1. **Survival (0.883 against 0.969) comes from the opponent's body and from motor latency, and training contains neither.**
   - The run-3 defenders tangle legs with the full-body opponent and knock it down, then fall themselves.
   - A whole-bout 20 ms motor delay, which training never applies, doubles the damage.
   - With both removed on dev, A3's falls go 12/64 -> 0/64.
2. **Clean head hits (R_warm 1.14 against the 0.80 needed) come from one held-out strike plus a keypoint channel that means
   something different at eval.**
   - Against the training attacker, A3 really does take 0.58-0.66x the warm start's clean head hits, on CPU as well as GPU.
   - Against the full-body eval opponent replaying the *training* clips, A3 takes 0.46x.
   - Only the held-out val clip, kfab_204, breaks it, and one rising left hook on that clip does most of the breaking.
3. **Physics, solver, hit rules, mesh shells, keypoint noise and the fall definition are ruled out**, each with a measured
   effect of 0 to about 1 pp.
4. **More training on the run-3 setup would not pass the screen or rules 1, 2 or 4. The training setup has to change.**
   - Run 3 already wins on the distribution it trains on.
   - It fails on things the training distribution does not contain.
5. **Run 4 (RUN4_PLAN_v2.md) is worth GPU money only if two $0 CPU pre-checks pass.**
   - If it launches, P(val screen) is about 0.2.
   - From today: P(val screen) is about 0.05 and P(M2 pass) about 0.01-0.02.
   - Stopping now and writing up run 3 is a fully defensible option.

## 1. Causes of the gap, ranked by strength of evidence

Strength reflects:
- how many independent analyses agree;
- whether the evidence is paired or interventional rather than correlational;
- the sample size.

### 1A. Survival (screen needs >= 0.95; val aimed: guard_fw_r 0.969, A3 0.883, A3s 0.875, blind3 0.930)

| rank | cause | key numbers | evidence type | agreeing analyses | strength |
|---|---|---|---|---|---|
| 1 | **Leg entanglement with a full-body opponent, mostly through knocking it down.** Training's attacker is two world-anchored 1 kg fists; nothing touches the R1's legs, and no opponent can fall (ST:62-68, :197-228, :255-261). | **Val aimed:** 12 of A3's 15 falls come after the opponent fell; guard 0 of 4. The opponent fell in 19 A3 bouts against 5 guard bouts. Without the post-knockdown falls, survival is A3 0.977 / A3s 0.969 / blind3 0.977 against guard 0.969. **Dev (256 bouts, paired):** knockdowns 13.3% (A3) vs 5.1% (guard), McNemar p 1e-4. 91-95% of knockdowns follow more than 10 N*s of leg-on-leg contact. **Interventional (64 bouts):** making the R1's legs non-colliding (`legs0`) takes A3's falls 12 -> 5 and the opponent's falls 12 -> 5. **Contact on legs:** 0 N*s/min in training against 363 on eval (538 for A3). | val + paired dev + intervention | falls, sim, opponent, replay (4 of 4 that looked) | **very strong** |
| 2 | **Untrained motor-command latency.** Eval DR applies one control step (20 ms) of delay for the whole bout with p 0.5 (r1gym/env.py:96, :199). Training has no actuator delay (R1C:39-90). The two stack into a constant 60 ms in 17% of bouts. | **Dev:** with the motor delay forced to 0 (`act0`), A3's falls go 12 -> 7. The removed falls are exactly the delayed bouts (6 -> 1); the undelayed bouts replay bit for bit (6 -> 6). Both knobs together give 12 -> 0 (blind3 3 -> 0). **Val:** run-3 falls are 60/384 in delayed bouts against 18/384 without delay (3.3x). All 4 of guard_fw_r's val falls are in latency-3 bouts. | intervention, deterministic replay, val split by DR draw | sim only (no other analysis switched it) | **strong, one analyst** |
| 3 | **What run 3 learned in its legs and waist (the reason it trips the opponent 2.6-3.2x as often)** | **Pelvis lean back from the opponent:** A3 9.8°, A3s 7.9°, blind3 6.4°, guard 3.0°; this follows the survival order. A3's stance is 0.32 m wide against guard's 0.27 m. Leg contact per minute is up 48-82%. **Action-split hybrids (dev, 256 bouts):** guard legs and waist with A3 arms survive **0.953** [0.920, 0.973]; A3 legs and waist with guard arms survive **0.852**. | paired hybrids; lean is correlational | falls, opponent (lean of -10° on both opponents) | **strong** for "legs and waist, not arms"; **medium** for "the lean is why" |
| 4 | **A3 reacts to a downed opponent's keypoints** | Blanking keypoints after the knockdown (physics identical up to it): P(fall given knockdown) 0.647 -> 0.235, survival 0.863 -> 0.918. With real keypoints A3 moves toward the downed body at +0.17 against -0.10 m/s, in 88% of bouts. | paired intervention | falls | **medium-strong** (34 knockdowns) |
| 5 | Fall-definition mismatch (training: tilt > 70°; eval: z < 0.45 or tilt > 60°) | Under training's rule: A3 0.875, under the eval's 0.863; guard 0.965 against 0.961. 34 of 35 eval-flagged falls reach the floor. | measured | falls, sim, opponent, replay | **ruled out as a cause** (about 1 pp) |
| - | DR other than the delay, pushes, solver, crouch, arm extension | `dr0` gives 7 falls, the same as `act0`. Pelvis height is 0.717 against 0.724 m. Arm contact before falls is about 0. | measured | sim, falls | ruled out |

How the causes fit together:
- On dev, rank 1 accounts for about 6-7 of A3's 12 falls and rank 2 for about 6, with an overlap of 1. Together they explain all
  of them.
- Ranks 3 and 4 are why rank 1 hits run 3 and not the warm start.
- Against the CPU training-attacker replica there is no fall gap:
  - falls analysis: A3 6.6%, guard 4.3%;
  - replay: 0.984 for both;
  - sim: 0/32 for both;
  - opponent replica: guard 0.81, A3 0.875, blind3 0.92.

  Replica survival levels differ between analysts (see 3, conflict 2). None shows A3 worse than guard by more than about 2 pp.

### 1B. Clean head hits (screen needs R_warm <= 0.80; val aimed: guard_fw_r 6.71, A3 7.65 (R 1.14), A3s 5.80 (0.86), blind3 6.05 per min)

| rank | cause | key numbers | evidence type | agreeing analyses | strength |
|---|---|---|---|---|---|
| 1 | **A held-out strike class: kfab_204's slow, rising left hook `l_hand@13`.** No training strike resembles it: 8 of the 9 training left hooks come downward. | **Clean head hits per min:** guard 2.67, A3 5.65, blind3 4.30. On `l_hand@41`, A3 *lowers* it from 2.47 to 1.43. Those two strikes are 5.1 of guard's 5.3 clean hits per min on that clip. **Without l@13, A3's R is about 0.56.** **The same eval opponent on the training clips:** A3's R is **0.46** [0.31, 0.63], blind3's 0.67 [0.53, 0.85]. **On the val clip:** 1.34 [1.03, 1.80]. 94% of A3's dev clean head hits are hooks, and 99% come from the left hand. | per-strike counts (211-241 windows each); intervals disjoint across clip sets | opponent, headclean | **strong** that it is strike-specific; **medium** that the reason is "absent from the training library" (inferred from approach vectors, not tested) |
| 2 | **The learned tight guard turns presses into through-the-arm clean hits. Both the training and eval `blocked` rules count these as blocks** (`hits.py:141-143`, ST:593). | **Head presses:** 14.10 -> 7.25 per min (-6.86 [-8.07, -5.51]), with clean hits unchanged or higher. **Dev:** blocked-then-head-clean 2.72 -> 6.11 per min; the chance a hook lands clean rises from 0.113 to 0.201; unblocked clean hits fall from 1.36 to 0.17. 96% of A3's clean head hits come after the same arm touched the R1's arm (median 0.11 s before). **Guard geometry:** palm to face 0.211 -> 0.135 m, palm gap 0.286 -> 0.174 m. | val counts + dev per-hit logs | headclean, replay (inferred from counts), opponent (l@13: arm first 0.90, then the hook lands) | **strong** (dev); val cannot check it, because the raw files hold no per-hit data |
| 3 | **The keypoint channel misfires at eval.** In training, the head keypoint is virtual (library head plus head_dz drawn from U(0, 0.12)) and the elbow sits on the fist axis (ST:93, :443, :506, :885-893). At eval they are the real head and the real elbow link (KP:18-21). | **Dev, 64 bouts:** A3 with zeroed keypoints takes **5.60** clean head hits per min against **7.93** with the true block (+2.32 [0.97, 3.71]). Training-style corruption gives 7.25, so noise is not the issue. Keypoint marginals match: glove-to-head p10 0.24 against 0.20 m. 94-97% of eval glove states are more than 5 cm from any training sample (self-check 26-33%). | intervention | sim (kpzero), opponent (input support), falls (the downed-opponent reaction) | **medium** (one 64-bout block; which property drives it is not isolated) |
| 4 | **The R_warm denominator moves with the opponent.** The warm start takes 0.41x as many clean head hits from the eval opponent as from the training attacker (5.6 against 13.5 per min). The eval's strikes mostly stop short: glove-to-head at apex is 0.36 m against 0.21 m, and the eval's aim cannot add reach (aim.py:182). Training's Delta warp bends every strike onto the face (ST:483-494). | Only 2 of kfab_204's 6 hand strikes ever reach. **The val screen's R is in effect a two-strike sample.** **kfab_204 R estimates across seed blocks:** 0.99 (headclean, 256 dev bouts), 1.14 (val, 128), 1.34 (opponent, 48), 1.42 (sim, 64). | measured | sim, opponent | **strong** as a fact; it amplifies cause 1 rather than being a separate cause |
| 5 | Follow-through: a 1 kg, 60 N servo fist (POD:236-264) against a robot arm with a 29 kg body behind it | Hits landing after an arm-first block: 0.35 on the replica against 0.19-0.25 on eval, on average. For l@13 it is decisive (0.42 for A3). | measured, mixed | opponent, headclean (hypothesis) | **weak to medium** |
| - | Physics engine, solver, control stack, mesh shells against capsules | **CPU replica of the training attacker, A3 over guard:** 0.657 [0.542, 0.796] on capsules, 0.657 [0.513, 0.816] on mesh shells (replay); 0.64 (sim); 0.58 (opponent). | measured | replay, sim, opponent | ruled out |
| - | Hit-scoring rule, latency, DR, leg contact, falls | Under the training rule, A3 over guard is still 1.76. A3 stays at 7.5-8.5 per min under every latency, DR and legs knob. In bouts where both survive, A3 7.61 against guard 6.79. | measured | sim, headclean | ruled out |

How the causes fit together:
- The tight guard (cause 2) is a real, transferable skill.
  - On the training clips it roughly halves clean head hits even against the full-body R1 (R 0.46).
  - It halves torso landed hits on val (18.9 -> 9.4 per min).
- On the one held-out hook it backfires: the hook meets the forearm and carries on into the head (cause 1).
- A3's keypoint reading makes that worse (cause 3). blind3 has the same guard and is hurt less (4.30 against 5.65).
- Because only two strikes on the val clip reach at all (cause 4), that one hook decides the screen.

## 2. Did run 3 learn anything that would pass an eval rule with more training?

**Plainly: no, not for the rules it failed. The training setup itself has to change.**

| eval rule (prereg_m2_run3 `rule` / screen) | run 3 now (val aimed) | would more run-3-style training fix it? | why |
|---|---|---|---|
| screen and rule 4: survival >= 0.95 | A3 0.883, A3s 0.875 | **No** | Training has no opponent legs, no falling opponent and no motor delay. Nothing it pays for teaches the lower body to avoid tangling. The lean-back posture that drives the knockdowns costs nothing against fists that stop at 60 N. |
| screen: head clean R_warm <= 0.80 | A3 1.14, A3s 0.86 | **No** | A3 already scores 0.58-0.66 against the training attacker. That distribution is already won; the failure is on a strike type it never sees. |
| rule 1: head R <= 0.60, CI upper < 0.80 | fails | **No** | Same reason. On the training clips it would pass (0.46), but val and test are held-out clips. |
| rule 2: head vs blind, CI upper < 0.90 | A3 is worse than blind3 on val (7.65 against 6.05) | **No** | At eval the keypoints hurt A3: zeroed keypoints give 5.60 against 7.93. Against the training attacker they help (A3 over blind3 is about 0.73-0.81), so more training deepens a mapping that misfires at eval. |
| rule 3: torso R <= 1.5 | 0.50 (passes) | already passes | Torso landed is halved. |
| rule 5: ring drop <= 0.05 | 0.016 (passes) | already passes | |
| blocks claim: head-intent stopped share +0.10 | **cannot be computed** (CRITIQUE B2) | unknown, probably not | Blocked-then-head-clean more than doubled, so much of the rise in `blocked_share` is not stopping. |

**What run 3 did learn that is worth keeping:** A3's *arms*.
- Their tight guard removes presses and almost all unblocked clean hits.
- On held-out-in-time but familiar strikes it beats the warm start against the full-body opponent (R 0.46 on the training clips).
- Grafted onto the warm start's legs and waist, it survives 0.953 on dev, the only configuration anyone measured near 0.95.

**What run 3 learned that is harmful at eval:**
- its legs and waist (the lean back, the wider stance, stepping toward a downed body);
- its keypoint mapping.

## 3. Conflicts between the analyses, and how they are resolved

1. **Lean direction.**
   - headclean calls the guard "forward-leaning".
   - falls (signed pelvis lean relative to the opponent) and opponent (torso pitched back 11°, head 3.7 cm behind the pelvis)
     measured lean *back*.
   - headclean measured tilt magnitude from up_z only (5.1° -> 11.5°), which agrees in size and order.
   - **Resolution:** the lean is backward; headclean's direction word is wrong.
2. **Survival against the training-attacker replica.**
   - The four replicas give guard 0.81-1.00, depending on DR, keypoint path and bout handling. The opponent analysis's replica
     falls about 4x more than GPU gate 1.
   - All agree on the sign: A3 is not clearly worse than guard there (-2 to +6 pp).
   - **Resolution:** the conclusion "no fall gap against the training attacker" stands. The absolute levels are not
     comparable across replicas.
3. **Size of R on kfab_204 (0.99-1.42 across seed blocks).**
   - The guard's own rate moves too (5.57 against 7.81 per min on different dev blocks).
   - **Resolution:** the between-block spread is larger than each block's interval implies. That is consistent with a
     two-strike clip whose outcome depends on start-time and DR draws.
   - Every block is far above 0.80, so the conclusion does not change. A run-4 screen pass on this clip, however, needs a
     margin and not a point estimate.
4. **Keypoints versus posture as the l@13 mechanism.**
   - opponent: blind3 shows the same pattern, so it is posture.
   - sim: zeroing A3's keypoints helps, so it is the keypoints.
   - **Resolution:** both hold and they stack. The tight guard alone raises l@13 (blind3 2.67 -> 4.30). A3's keypoint reading
     raises it further (5.65), and kpzero brings A3 back to about blind or guard level overall (5.60).
5. **Motor latency.**
   - Only sim tested it. falls' "no leg-contact falls at all gives 0.973" does not contradict it: the 5 falls that remain
     under `legs0` are all latency-3 bouts.
   - **Resolution:** treat latency as strong but single-sourced, and re-check it in pre-check P1 (RUN4_PLAN_v2 section 3).

## 4. What this means for the run-4 prereg

- **What has been looked at.** The val raw files have been analysed by five separate analyses. kfab_204 has been decomposed
  strike by strike on dev seeds, and its decisive strike (l_hand@13, a rising slow left hook) is known.
- **Consequence for a kfab_204 screen.** A run-4 training change that adds rising or otherwise re-angled hooks is *val-informed*.
  A run-4 val screen on kfab_204 (fresh seeds 23000+) will therefore be optimistic, and has to be disclosed as such.
- **What stays clean:** the test, clips kfab_202 and kfab_205 at seeds 22000-22127. No file from it was opened and no episode
  run, so it remains the only clean measurement.
- **Seed blocks already used:**
  - 50000-50255 (kfab_204);
  - 51000+ (train clips);
  - 50000-50063 (replica).

  New pre-checks should use 52000+.

## 5. Recommendation

- **First**, run the $0 CPU pre-checks in RUN4_PLAN_v2.md section 3. They answer the two open questions directly:
  - P1: do the warm start's legs plus A3's arms clear 0.95 on a fresh block?
  - P2: can a modified training attacker reproduce A3's eval head failure on CPU?
- **If either fails, stop spending GPU on M2.** Write up run 3 and this gap analysis as the result: a scripted-fist attacker
  teaches a guard that transfers to familiar strikes and fails on a held-out hook, and a legless attacker teaches a lower body
  that trips a real opponent.
- **If both pass**, run 4 as specified, about $1.85, with a hard stop at about $0.50 if the early-stop rules fire. Even then,
  P(val screen) is about 0.2.

Files: this synthesis
`C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\gap\SYNTHESIS.md`;
the plan
`C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\gap\RUN4_PLAN_v2.md`.
