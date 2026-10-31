# M2 v2 status (paused 2026-10-30)

M2 v2 trains the R1 defender against the eval's own full-body opponent: a second R1 driven by the frozen tracker, with the
eval's aim, keypoints, latency, randomisation and fall rule. Runs 1-3 trained against two floating fists, and the defender did
not transfer to the full-body test. The frozen plan is `M2V2_PLAN_v3.md`, with amendments A1-A11 appended. Each milestone has a
build report and an independent verifier report in `build/`.

## Done, each built and independently verified

| milestone | what | result |
|---|---|---|
| M0-M1 | seed ledger, hygiene guards, train-clip motion and hull exports | pass |
| M2 | torch tracker, facing and placement against the eval | pass |
| M3 (A1-A3) | aim and arm warp; training jitter adds no forward reach (target-space correction) | pass |
| M4 | defender observations, obs lag, motor delay, fall rule, DR writer | bit-equal to the eval |
| M5 (A4, A5) | full-body hit account and rewards; W0 from R4-pre knockdowns; W_eps floor and 0.5 s start window | pass |
| M6 R1 | CPU replica, bit-identical to EO.run; batched 64-lockstep at 1.04-1.17x EO | pass |
| M7 | hybrid actor and ONNX export; T9 bit-identical through an unchanged EO.run | pass |
| M8 | L1 punch library, LOCO library, mirror-conjugated tracker | pass |
| M9 | mjlab full-body task, pod script, WATCH6; model equals the replica by name | pass |
| M10 | CPU training smoke (plumbing gate) and the pod evaluation scripts | pass; learning canary flat (report only) |
| Track 2 | sealed split of 80 KungFuAthlete clips | built; left out of this prereg (A5: val2 cannot reach 30 head strikes) |

## R-EQ (replica vs eval) on a RunPod CPU pod

- The R0-EO reference was run twice: on the laptop in 8.2 h, and on the pod in 0.5 h. The two platforms are not bitwise equal,
  but their summary rates agree well inside the margins (A8).
- The first R-EQ failed conclusively on head clean: every replica row had 0 head hits.
  - Cause: the reduced hull meshes were renamed `fbhull_*`, and the hit meter finds the head by mesh name. Fixed in 8153dbb,
    with a blocking zone-map test (A10).
  - All other R2 diagnostic steps kept the eval's head-hit rates.

## Next

1. Rerun R-EQ once on the reserve block 100000-112287, with EO and the replica both on the pod (A10.4). This costs about $2 of
   cloud CPU.
2. M11: hash the preregistration.
3. GPU top-up (about $65), then P0-P7 per plan section 4.

**Open risk:** the M10 learning canary did not learn in 400 small-batch iterations, because the adaptive lr was pinned at its
floor. WATCH6 E4 watches for the same thing on GPU.
