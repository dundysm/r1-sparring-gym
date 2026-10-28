# Aimed hard tier, as run against PR #13's stepping bases (archived)

These are the M1.5 runs exactly as they were made and committed in 72ad022 (`feat/aimed-opponent`), moved here unchanged
(byte-identical to `git show 72ad022:results/eval_opponent/<file>`) when the branch was merged with main after PR #14.

- **Defenders:** `results/policy_mjlab/` (arms down) and `results/policy_mjlab_guard/` (guard), with the `gym.json` of
  d1990f9 (hold-your-spot on, the work-in-progress deadbands of that commit, turning off), plus PD, v1-v3 and stage 1.
  Not the footwork policies PR #14 shipped.
- **Code:** the working tree committed as 72ad022 (`meta.git_dirty` is true: the runs predate that commit, which pins them).
- **What is here:** the unaimed v1 tiers re-run with the hit metric and the strike scheduler in shadow (`raw_*`, `summary_*`),
  the aimed runs paired with them (`*_aim.json`; operator from `results/opponent_tracker/aim.json`), hard and standard,
  val and test, with and without scripted hits (hard), and the puppet tier.
- **Not the headline:** main's `results/eval_opponent/*.json` are the v1 headline. The aimed hard tier against the shipped
  footwork policies is in `results/eval_opponent/aim/`. `docs/AIMED_OPPONENT.md` (Results) cites both.
