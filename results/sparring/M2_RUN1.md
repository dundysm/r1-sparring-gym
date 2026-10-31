# Sparring M2 run 1: the sighted defender learned to back away (not shipped)

The defender sees the opponent's keypoints and is meant to learn to block. It was trained in
`R1-Spar-Block-A` / `-B` (`gpu/sparring_task.py`), with a blind twin as the control. All three were warm-started
from `guard_fw_r` (`results/policy_mjlab_guard_footwork_r/model_7496.pt`, widened from 83 to 111 inputs).

- **Pass rule:** [`prereg_m2.json`](prereg_m2.json), committed before any pod (ad6ba44).
- **Selection** (val only): [`m2_selection.json`](m2_selection.json).
- **Verdict** (test, run once): [`m2_verdict.json`](m2_verdict.json).
- **Eval files:** `results/eval_opponent/m2/`.

**Verdict: fails 2 of 5 rule parts (survival, ring), so it doesn't ship.**

## Result (test, hard tier, aimed opponent, 128 bouts per controller, same episodes)

| controller | survival | up and within 1 m | head clean / min | torso clean / min | blocked share | back-off of survivors |
|---|---|---|---|---|---|---|
| guard_fw_r (warm start) | 0.969 | 0.883 | 4.61 | 10.28 | 0.236 | 0.17 m |
| **A** (selected) | **0.750** | **0.000** | 0.105 | 0.63 | 0.052 | 4.27 m |
| blind twin | 0.992 | 0.008 | 0.422 | 4.46 | 0.112 | 0.58 m |

| rule part | needed | A | |
|---|---|---|---|
| head clean vs warm start | R <= 0.60, CI hi < 0.80 | R 0.023 [0, 0.059] | pass |
| head clean vs blind twin | CI hi < 0.90 | R 0.249 [0, 0.552] | pass |
| torso clean vs warm start | R <= 1.5 | 0.061 | pass |
| survival | >= 0.95 | 0.750 | **fail** |
| ring (warm start minus candidate) | <= 0.05 | 0.883 | **fail** |

The head numbers are not blocking. A steps straight back from about 1 s into the bout, at about the opponent's
0.3 m/s chase cap, and spends the bout out of reach. Its blocked share falls from 0.24 to 0.05. On val, B did the same
(unaimed back-off 4.2 m). The blind twin also leaves the 1 m ring, but it drifts rather than retreats (0.6 m back).

Unaimed test, for reference: guard_fw_r 1.00 / 0.961 / 1.97 head clean per minute; A 0.938 / 0.0 / 0.098; blind 1.00 / 0.008 / 0.375.

## Why it backed away (`docs/SPARRING_M1M2.md` section 9, run 2)

- **A slow retreat already beats the GPU attacker.** The aim is latched at strike onset and set 2 cm short, so backing
  away at 0.2 m/s makes about 85% of head strikes fall short.
- **Nothing anchored the defender.** Half the envs kept the footwork walking commands (mean 0.47 m/s, faster than the
  0.3 m/s chase), and no reward or termination referred to the ring.
- **Staying up didn't pay.** Net reward per live second was negative throughout training (A -1.1 to -1.6 per second),
  so ending episodes early cost little. B's mean episode fell from 14 s to 5.5 s.

## Gates and deviations (all recorded in `m1_gate/`)

- **Gate 1, attacker strength.** On MuJoCo Warp the CPU-calibrated attacker (1.5 kg, 80 N, −2 cm) gave a median
  3.945 N·s per head hit, and 4.04 on the rerun, above the 3.56 limit. The predeclared GPU retune grid
  (section 8 item 29, committed in 86cfcbe before any training) picked (1.0 kg, 60 N, −2 cm): 3.27 N·s, head clean
  12.2 per minute.
- **Gate 2, speed.** 2.11 s per iteration at 4096 envs on mesh proxies, and 2.115 on capsule proxies. It passed at
  3072 envs with capsule proxies, 1.945 s per iteration, through 2.6's fallbacks. The cost is launch overhead, not
  physics: about 2,300 kernels and 25 host syncs per env step. A speed-up patch is in progress.
- **Training:** 1500 iterations each, about 1.95 s per iteration, no NaN.
- **Cost:** two RTX 4090 pods, about 5 min and 3 h, about $2.83 in total. Both pods were deleted after an md5-checked
  download.

## Files

- `results/policy_mjlab_spar_{a,b,blind}/`: `policy.onnx`, `model_8995.pt`, `params/`, `gym.json` (the warm start's,
  unchanged) and `train_*.log.gz`.
