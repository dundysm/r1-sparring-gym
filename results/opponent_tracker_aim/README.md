# M2 run 1: the aim-capable tracker (not shipped)

This is the opponent tracker fine-tuned on whole-body aimed references: `R1-Track-KungFu-Aim` in `gpu/r1_track_task.py`, trained on the motion from `gpu/export_warps.py`.

- **Preregistration:** the gates, the single candidate and the evaluation order are fixed in [`prereg.json`](prereg.json), committed before the pod was created.
- **Evaluation:** [`scripts/aim_m2.py`](../../scripts/aim_m2.py) evaluates the gates on train + val and writes [`gates.json`](gates.json).
- **Verdict:** **3 of the 4 gates fail, so it doesn't ship.** As the prereg says, nothing ran on test.

| gate (train + val) | needed | M2 run 1 | shipped tracker |
|---|---|---|---|
| alone (`eval_tracker.json`, 48 runs) | stays up >= 45/48, joint RMSE <= shipped + 0.02 rad | 45/48, 0.258 rad: **pass** | 46/48, 0.248 rad |
| parity (`parity_mjlab_side.json` vs `eval_tracker.json`) | every shared fall within 0.1 s | 2 shared falls, 0.12 s apart at most: **fail** | within 0.1 s |
| tolerance, val strikes (`tolerance_val.json`; shipped: `tolerance_val_shipped.json`) | 20 cm warps within 5 cm >= 65%; 20 cm lunge at 1 m/s adds >= 12 cm reach | 33% of 18; −0.1 cm: **fail** | 39% of 18; 2.6 cm |
| aim check vs the shipped tracker with M1's operator (`gates.json`, same 72 episodes per target) | Δ head clean strikes/min and Δ head-landed share per window, both CIs > 0 on both targets; opponent falls <= shipped + 2 pp | arms down −0.17 [−2.25, 2.25] and +0.042 [0.011, 0.079]; guard +0.42 [−0.25, 1.08] and +0.016 [0.002, 0.029]; falls equal: **fail** | |

- **Operator:** M2's operator was chosen by M1's rule 3, on M2 alone: StandOff + ArmWarp (height, sideways), +1.83 [0.17, 4.00]/min against its unaimed self. StandOff + full 3-D ArmWarp was not eligible: +1.0 [−1.5, 3.25].
- **Reproducibility:** the shipped tracker's re-runs reproduce the stored `results/opponent_aim/aim_check_raw.json` bout for bout.

## What went wrong, as far as the numbers show

- **It doesn't follow lunges.** A 20 cm lunge adds about 0 cm of reach (shipped 2.6 cm). Two causes:
  - Training did move the reference in lunges, but the anchor-position reward has a 0.5 m std, so trailing a 20 cm lunge costs only about 15% of that one term.
  - The world-frame apex reward fires only at strike apexes, and the lunges came at random times, so they rarely coincided.

  Lunges should have been timed to the strikes and scored at the apex.
- **It overshoots unwarped strikes.** On val strikes its palm reaches further than the reference (for example 0.54 m against the shipped 0.44 m), and fewer unwarped strikes end within 5 cm (3/6 against 6/6).
  - The apex reward pulled the wrist toward warped apexes. It plateaued after about 250 iterations (`train_log.json`).
  - Joint error rose slowly over the run (1.58 → 1.71 rad, summed).
- **Aim effect:** it lands a slightly larger share of aimed windows on the head (+4.2 points on arms down, +1.6 on the guard; CIs above 0), but no more head clean strikes per minute.

## Run

- **Hardware:** 1× RTX 4090, RunPod secure, 2026-10-28.
- **Time:** 1500 iterations from `results/opponent_tracker/model_13749.pt` (ending at `model_15248.pt`), about 28 min. The pod was up 33 min, about $0.43, and was deleted after an md5-checked download.
- **Kept:** `policy.onnx` (the motion ONNX the repo's Tracker loads), `model_15248.pt`, `params/` and `motion_index.json` (the aim motion's index; its clips sit at the same frames as the shipped motion).
