# Aimed hard tier against the shipped footwork policies

The M1.5 re-measurement after merging main (PR #14-#16) into `feat/aimed-opponent`: the hard tier, val then test once, with
and without scripted hits, against main's headline controllers (PD, v1-v3, stage 1, PR #13's bases, and the shipped footwork
policies `results/policy_mjlab_footwork/`, `results/policy_mjlab_guard_footwork/` with their `gym.json` as shipped).

- `raw_*_hard[_nohits].json`, `summary_*`: unaimed, the strike scheduler in shadow (bit-identical to the shipped opponent) and
  the hit metric on. Their bouts match main's `results/eval_opponent/raw_*_hard*.json` in every field those have.
- `*_aim.json`: the same episodes with the aimed opponent (`--aim chosen`: the operator in `results/opponent_tracker/aim.json`,
  chosen on train + val before any of this ran), paired in `vs_unaimed`.
- Made by `scripts/eval_opponent.py --tier hard --split {val,test} [--no_pushes] [--aim chosen] --out results/eval_opponent/aim`.
  `docs/AIMED_OPPONENT.md` (Results, addendum) and the README's aim block are generated from these files by `scripts/aim_tables.py`.
- The first run, against PR #13's bases, is archived in `../aim_pre14/`.
