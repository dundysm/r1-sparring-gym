# R4-pre: the run-3 failure signature on the TRAIN clips, in the real eval bout

Run 2026-10-29 for M2V2_PLAN_v3 (CRITIQUE_M2V2 B1 fix 1). CPU only, 4 worker processes. Repo
`r1-sparring-gym-spar` (branch feat/sparring-m2v2 @ b81fa16) read only; no repo file added or changed; no MUJOCO_LOG.TXT left.

**Result.** On the train clips the run-3 signature is **"A3 falls, mostly under motor delay and after a knockdown"**, not
"A3 knocks the opponent down more". A3 knocks the opponent down *less* than G (ratio 0.77, CI [0.55, 1.04]; at 0.40/0.45 m
0.49). R4's contrasts are frozen accordingly (bottom of this file).

## Method
- Harness `r4pre_eval.py`: P1's `p1_eval.run()` (= `sim/dev_eval.run()` line for line: `R1HitEnv` + eval opponent, aim
  `standoff+armwarp_vz`, hard-tier distances, 20 s, DR on, act_delay and obs delay as drawn, no pushes, `KeypointSource` truth,
  AltMeter = HitMeter + closing speed, ContactTail, knockdown = EO:86 flag no later than the defender's eval fall).
  P1 validated this path bit for bit against `dev_base.json` and `EO.run` (P1 V1, V2, VH).
- Controllers: G = `guard_fw_r` plain; H = `guard_fw_r+a3s` M3s (P1's `M3sHybrid`); A3 = the full run-3 A3 ONNX.
- Episodes: `EO.episodes(train clips, reps, tier = hard with seed = block start)`, clean starts with G's cfg (cached in
  `good_kfab_200_kfab_201_kfab_203.npz`). Paired: every controller has the identical episode list (asserted).
  - main: 67000-67767, kfab_200/201/203 x 0.40/0.45/0.50/0.55 m x 64 reps = 768 bouts per controller;
  - close: 68000-68767, the same clips x 0.40/0.45 m x 128 reps = 768 bouts per controller.
- Parallelism: episodes split i % 4 over 4 spawn processes (threads 1 each), merged in order. Check: 67000-67007 run this way
  and through P1's own in-process 4-thread path are **bit identical** (fell_s, every HitMeter count, head events, tilt, z).
- Seed asserts in the harness: never 21000-21999 or 22000-22511; only the two blocks above; clips asserted train only.
- `analyze_r4pre.py` -> `R4PRE_RESULT.json`: paired bootstrap over episodes, 10,000 resamples, `default_rng(20261029)`.
- Speed (m14): 445-490 s per 768 bouts on 4 workers = **0.58-0.64 s wall per bout** (2.3-2.6 s per bout-worker).

## Results, main block (768 paired bouts, eval distance mix)

| | G | H(A3s)/M3s | full A3 |
|---|---|---|---|
| falls / survival | 15 / 0.980 | 17 / 0.978 | **80 / 0.896** |
| knockdowns (share) | 56 (0.073) | 60 (0.078) | 43 (0.056) |
| falls after a knockdown / P(fall given kd) | 4 / 0.071 | 5 / 0.083 | **25 / 0.581** |
| P(fall given no kd) | 0.015 | 0.017 | **0.076** |
| falls with act_delay 1 (n 368) / act_delay 0 (n 400) | 10 / 5 | 14 / 3 | **71 / 9** |
| head clean per up-minute | 4.20 | 3.53 | 2.71 |

Paired contrasts (95% CI):
- A3 - G fall rate **+0.085 [0.063, 0.108]**; of it, the no-knockdown route +0.057 [0.039, 0.077].
- A3 / G knockdowns **0.77 [0.55, 1.04]** (the old R4 clause ">= 1.5x" is false here, as C3 found).
- P(fall given kd), A3 - G **+0.51 [0.34, 0.67]**.
- H - G fall rate +0.003 [-0.010, +0.016]; H / G knockdowns 1.07 [0.80, 1.46]; P(fall given kd) H - G +0.01 [-0.08, +0.10].
- R(H vs G) head clean **0.841 [0.772, 0.917]** (P1(b): 0.834 [0.712, 0.976]); R(A3 vs G) 0.646 [0.591, 0.706].

By clip (main): A3 - G gap kfab_200 +0.121, kfab_201 +0.035, kfab_203 +0.098. R(H vs G) kfab_200 0.66, kfab_201 0.95, kfab_203 0.83.
kfab_203 carries most head hits (G 8.1/min vs 1.4 and 3.1).

## Close distances (0.40/0.45 m)

| | G | H | A3 |
|---|---|---|---|
| close block 768: falls / kd / falls after kd | 21 / 88 / 3 | 21 / 84 / 6 | 88 / 43 / 28 |
| near = main 0.40/0.45 + close (1152): falls / kd / P(fall given kd) | 30 / 124 / 0.032 | 32 / 124 / 0.065 | 127 / 70 / 0.60 |

- Close range does **not** strengthen a knockdown contrast: A3 / G knockdowns 0.49 [0.34, 0.66] in the close block.
- The fall gap is the same (+0.087 [0.064, 0.112]), and P(fall given kd) is +0.62 [0.47, 0.76].
- So R4 stays on the eval distance mix. Close range is a report-only diagnostic (B1 fix 3 asked to harden R4 there only if
  the train-clip contrast were weak; it is not weak).

## What this changes in the plan
1. The train-clip failure is **latency + post-knockdown survival**. 71 of A3's 80 falls are in act_delay bouts (19.3% vs 2.3%),
   and 25 of 43 knockdowns end in a fall. Training under the eval's motor delay (one per-episode flag) is the first lever; a priced
   post-knockdown term is the second (`fb_down_approach`, `fb_down_contact`, B1 fix 5).
2. The knockdown penalty stays (G knocks the opponent down in 7.3% of bouts already), but it is no longer the headline lever,
   and section 1's first row is rewritten.
3. E5's baseline is the run's own iterations 1-100 (m6); for scale, G's train-clip knockdown rate is about 0.22 per up-minute.

## R4 contrasts, frozen (used in M6 R-EQ and P1-R4g, CPU EO reference vs replica / GPU)
- **gap_A3_G:** (A3 - G fall rate) of the replica minus that of EO, 90% CI inside [-0.03, +0.03]. EO reference here +0.085.
- **gap_sign:** the replica's A3 - G fall rate, 90% one-sided lower bound > 0.
- **pfk_contrast:** (P(fall given kd) A3 - G) of the replica minus EO's, 90% CI inside [-0.20, +0.20]. EO here +0.51.
- **R5_latency:** the replica's A3 fall rate with act_delay 1 minus with act_delay 0, 90% one-sided lower bound > 0.
  EO here 0.193 vs 0.023.
- Report only: A3/G knockdown ratio (EO 0.77); H - G survival (paired CI); G's latency interaction (EO 10/368 vs 5/400).
- Deleted: "A3 knockdowns >= 1.5 x G", "A3 survival <= G - 0.03" as a point rule, and "H within 0.015 of G".

## Files
- `r4pre_eval.py`, `run_r4pre.sh`, `run.log`, `runs/{G,H,A3}_{main,close}.pkl`, `analyze_r4pre.py`, `R4PRE_RESULT.json`
  (all in `C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody\r4pre\`).
