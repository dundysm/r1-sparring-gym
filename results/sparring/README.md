# Sparring M0: scorecard, fall log, blocking oracle, gloves

The sparring plan (defender first) changes the headline from survival, which is near 1.0 for every stepping policy, to
**clean head strikes received per minute** (`r1gym/hits.py`).

- **`scorecard_baseline.json`** (`scripts/scorecard.py --write`): every stored hard-tier summary, as the scorecard reads it.
  - `tests/test_scorecard.py` checks it against the stored files, for example guard_fw against the aimed opponent on test: 4.24
    head clean strikes/min, survival 0.969.
  - For aimed tiers it adds `intent_blocked_share`, the share of the aimed opponent's strikes that hit the arms first: 0.256 for
    guard_fw on test.
- **`scripts/eval_opponent.py`** now logs, in new keys only (existing rows unchanged, checked bout by bout on 128 bouts):
  - each opponent fall: when, why, and the robot-robot contacts in the 0.5 s before;
  - exposure while both robots are up, counted per control step;
  - optionally (`--rev_hits 1`), the R1's own strikes on the opponent.
- **`oracle_val.json`, `oracle_train_tuning.json`** (`scripts/defence_oracle.py`): can the R1's arms block head strikes at all?
  The test is a privileged scripted arm on top of the shipped footwork guard, run against the aimed opponent on the hard tier
  with no scripted hits, on val only.

| oracle | head clean strikes/min received | Δ vs the policy [95% CI] | torso clean/min | survival |
|---|---|---|---|---|
| policy as shipped | 7.72 | | 7.96 | 0.969 |
| v1: react 40 ms after the strike starts (preregistered go/no-go: -40%) | 7.26 | -0.46 [-1.80, 0.95], -6%: **no go** | 10.32 | 0.844 |
| v2: raise the near forearm 0.25 s before the strike, head-aimed strikes only | 4.95 | -2.77 [-4.20, -1.29], **-36%** | 9.21 | 0.953 |

- **Why v1 failed:** its forearm never reached the block point (median 11 cm away), because the arm servos are soft and it moved
  late.
- **What v2 shows:** v2 was designed after seeing v1's diagnosis, tuned on train clips only, and run once on val. It misses the
  40% bar, which was set for v1, but the effect is clear. The arms can block when they move early.
- **v2 caveats:**
  - Its guard is up on 86% of control steps, because aimed strikes come almost back to back. The privilege mostly supplies the
    side and the line. A non-privileged "guard always up on the near side" control was not run.
  - At the head hits that still land, its forearm is a median 9 cm off the incoming line (8% within 7 cm). The remaining misses
    are a placement-prediction problem, which is what a learned defender is for.
  - The chosen setting is the best of 12 on 48 train bouts (-62% on train, -36% on val: a winner's curse). Its block distance,
    0.10 m, is at the edge of the range tried.
- **Gloves** (`results/perception/gloves_check.json`, `docs/HEAD_CAMERA.md`): the head camera finds the opponent's gloves on
  val as follows:
  - found in 88% of frames where they are in the field of view (99% when visible);
  - direction error of 0.9° median, 7.8° at the 90th percentile;
  - left/right label correct 92% of the time;
  - no false detections.

**Next (M1-M2):** a defender that sees the opponent's fists, trained on the GPU against strikes aimed at its head, with a blind
twin as the control. Its warm start is the retrained guard checkpoint (`results/policy_mjlab_guard_footwork_r/`), which missed
its own 3 pp rule (verdict.json), so every comparison uses its own baseline.
