# Handoff: r1-sparring-gym M2 v2 (as of 2026-10-30)

This is for the next agent. Read it fully before touching anything. Then read `docs/m2v2/STATUS.md`, and from
`docs/m2v2/M2V2_PLAN_v3.md` read section 3, section 4, section 6 and every amendment A1-A11 at the end of the file.

## 1. The goal in one paragraph

Train the **defender**, an R1 humanoid in MuJoCo/mjlab, to stop clean head hits from a kung-fu opponent without falling. The
target is the preregistered bar on the hard_r2 test:
- clean head hits vs the warm start: R <= 0.60, and 90% CI upper < 0.80;
- vs the blind twin: CI upper < 0.90;
- torso R <= 1.5;
- survival >= 0.95;
- ring drop <= 0.05.

The opponent is **not trained**. It is a second R1 driven by a frozen motion tracker (R1-Track-KungFu-Steer2), as the eval uses.
Runs 1-3 trained the defender against two floating fists and failed to transfer to the full-body eval. M2 v2 rebuilds the eval's
exact opponent, aim, latency, randomisation, fall rule and hit scoring inside the GPU training world. It proves on CPU that this
world matches the eval before any GPU money is spent.

## 2. The user: Dundy Pasupuleti (GitHub dundysm)

- Wants the best possible result and is happy to top up GPU money. Report spend, but never cut quality for budget.
- Said "dont ask me, just do it" for routine decisions. Ask only for money, publishing, or genuinely new scope.
- Often on mobile. Give **short, plain status updates**; avoid jargon walls. They ask "status" a lot.
- The user tops up RunPod themselves. Ask before any spend the balance cannot cover. Always show the hourly price before
  renting.

## 3. Hard rules (non-negotiable)

- **Commits:**
  - signed (SSH signing is already configured in the repo);
  - author **Dundy Pasupuleti <dundysm@gmail.com> only**;
  - **no Co-Authored-By / Claude trailer**, and no "Generated with" line;
  - commit messages are short, factual and lowercase-ish, like the existing ones.
- **Branch and merging:**
  - Work on `feat/sparring-m2v2`, which is pushed to github.com/dundysm/r1-sparring-gym.
  - Never merge, never open a PR, never push to main unless the user asks.
- **Never touch the hard_r2 test** (seeds 22000-22511) or the test clips kfab_202/205. They are run once only, at the very end,
  under the test-once guard.
- **Other protected data:**
  - Never simulate val seeds 21000-21999 or 23000-23255.
  - kfab_204 is frozen (used once, in the S1/S2 screen after training).
  - Track 2 val2/test2 (24000-25999) stays sealed.
- **Seeds:** use only the blocks the plan's ledger (section 6.4 plus amendments) assigns, checked via `scripts/fb_seeds.py`.
  Append every used block to `docs/m2v2/SEED_LEDGER.md` (the working copy lives in the work folder, see section 5).
- **Thresholds:** never loosen a frozen threshold or margin.
  - If a check proves impossible as specified, prove it with the eval's own code.
  - Then write a dated **Amendment A12, A13, ...** at the end of the plan **before** re-running. Decide on mechanism, never on
    outcomes. This is how A1-A11 were done; follow the pattern.
- **Run 1-3 files stay untouched:** `r1gym/*`, `scripts/eval_opponent.py`, `gpu/sparring_task.py`. New code goes in
  `gpu/fb_*.py`, `scripts/fb_*.py` and `tests/test_fb_*.py`.
- **Verification:**
  - Every milestone gets an **independent verifier** that re-runs the checks and reads the code against the eval source. Only
    commit work the verifier passes.
  - The pattern used so far is a builder agent followed by a verifier agent, each writing NOTES/VERIFY md files.
- **Local machine (8 cores, 32 GB):**
  - at most 4 worker processes;
  - Python with `-I` and `PYTHONIOENCODING=utf-8`;
  - delete any `MUJOCO_LOG.TXT` left in the repo;
  - pytest: `python -m pytest -q -p no:cacheprovider --basetemp="<quoted path>" tests`. An unquoted basetemp created a stray
    `Users...` folder in the repo once; delete it if it appears.

## 4. Where everything is

- **Repo:** `C:\Users\dundy\Downloads\projects\r1-sparring-gym-spar`, a git worktree on branch `feat/sparring-m2v2`, head
  `cae9b97`. It is pushed.
- **Docs:** `docs/m2v2/`
  - `M2V2_PLAN_v3.md`: the frozen plan plus amendments A1-A11;
  - `STATUS.md`;
  - `SEED_LEDGER.md`;
  - `GAP_DIAGNOSIS.md`: why runs 1-3 failed;
  - `R4PRE_NOTES.md`;
  - `TRACK2_APPROVED.md`;
  - `build/*_NOTES.md` and `*_VERIFY.md`: one pair per milestone.
- **Permanent work folder:** `C:\Users\dundy\Downloads\projects\r1-sparring-m2v2-work` (6.6 GB, a copy of the old temp
  scratchpad).
  - `m1m2/fullbody/`: the plan's working copy, SEED_LEDGER working copy, `build/` (all check outputs and recordings),
    `r0eo/out` (the laptop R0-EO reference, 12288 x 3), `r4pre/`, `critique_checks/` and `podpull/`.
  - `podpull/` holds `r0eo_pod.tgz` (the pod-platform R0-EO, sha c840d719...) and `m6out.tgz` (the failed first R-EQ, plus R2,
    T1-c, T-opp and R6 outputs).
  - `m1m2/gap/`: the run-4 diagnostics, including `run4pre/P1/p1_eval.py`, which fb_r0eo imports.
  - `pod/`: the cloud helper scripts: `podrun.sh`, `sendparts.sh`, `port.py` and the `t*.sh` examples.
  - `venv_freeze.txt`: the exact Python package versions.
- **Hardcoded paths:** several scripts point at the OLD temp scratchpad, `C:\Users\dundy\AppData\Local\Temp\claude\
  C--Users-dundy-Downloads\96df40a6-...\scratchpad\m1m2`. These include fb_r0eo.SCR, fb_m6_run.SCR, fb_eq.SCR,
  fb_m5_check and fb_m6_check, plus the gap scripts.
  - If that temp folder is gone, either recreate it by copying the work folder's `m1m2` back to that exact path (simplest), or
    switch those constants to an env var (for example `FB_SCR`) in a small commit.
  - `pod/port.py` shows the mapping.
- **Python venv:** `C:\...\scratchpad\venv` (Python 3.12) holds mujoco 3.17.0, numpy 2.5.4, torch 2.14.1+cpu, onnx 1.23.2,
  onnxruntime 1.31.0 and r1gym as editable. pip is not installed in it.
  - If it is gone, recreate it from `venv_freeze.txt`, then `pip install -e` the repo.
- **Memory notes** for this project are in `C:\Users\dundy\.claude\projects\C--Users-dundy-Downloads\memory\
  r1-sparring-m2-runs.md`.

## 5. What exists (code map, all committed)

| file | role |
|---|---|
| `scripts/fb_seeds.py` | seed roles, block guards, ledger audit |
| `gpu/export_fb_motion.py`, `export_fb_hulls.py` | train-clip motion (kfab_200/201/203) and reduced hulls |
| `gpu/fb_opponent.py` | torch tracker driver, ArmKinematics, AimOp (standoff + armwarp_vz, training jitter with the A3 reach correction) |
| `gpu/fb_mirror.py` | mirror-conjugated tracker for mirrored rows (A6.2) |
| `gpu/fb_obs.py`, `fb_actions.py`, `fb_events.py`, `fb_dr.py` | defender obs with one shared per-episode lag and lazy pre-fill, hybrid delayed action, latency draw, fall rule, DR writer |
| `gpu/fb_hits.py`, `fb_rewards.py` | full-body hit account (zones by mesh name) and rewards; W0 = 0, W_EPS = 1.16e-4 J, 0.5 s start window (A4/A5) |
| `gpu/hybrid_actor.py`, `scripts/export_hybrid_fb.py` | stage A (G legs + trained arms) and stage B (leg residual); ONNX computed in float64 (A9.1); blind flag in checkpoints |
| `gpu/export_fb_library.py`, `gpu/r1_track_spar_cfg.py`, `gpu/r1_track_spar_t1.py` | L1 punch library, LOCO library, T1 tracker fine-tune config (reangle family only, A9.3) |
| `gpu/fullbody_task.py`, `gpu/fb_population.py` | mjlab 1.2.0 task; compiled model equals the replica's by name, including zone maps |
| `gpu/pod_fb.sh` | GPU pod stages P0-P7 (setup, parity, T1, LOCO, main pairs, pack) |
| `scripts/fb_watch.py` | WATCH6 early stops |
| `scripts/fb_replica.py` | CPU replica (two-R1 world, N = 1 or 64-lockstep batch, bit-identical to N = 1) |
| `scripts/fb_r0eo.py` | R0-EO reference runner (sharded, resumable); `FB_MAX_WORKERS` lifts the 4-worker cap |
| `scripts/fb_eq.py`, `fb_m6_run.py` | equivalence statistics (paired TOST, looks at 22/43/64 shards) and the M6 stage driver (`req`, `r2`, `t1c`, `topp`, `r6`, `all`) |
| `scripts/fb_cpu_smoke.py`, `fb_p1.py`, `fb_t1c.py`, `fb_loco_eval.py` | M10 smoke and the pod evaluation scripts |
| `scripts/fb_m1..m9_check.py`, `fb_m6b_check.py` | the milestone checks |

## 6. Status

**Done and verified (M0-M10).** See STATUS.md. Notable facts:
- The replica in eval mode equals EO.run bit for bit.
- The batched replica runs at about 1.1x EO speed.
- The exported ONNX is bit-identical through an unchanged EO.run.
- The plumbing smoke gate passes.

**The first R-EQ failed (A10).**
- It ran on RunPod CPU, 12288 bouts x 3 controllers, and failed **conclusively** on head clean: replica head hits were 0.
- Cause: the reduced hull meshes had been renamed `fbhull_*`, and the hit meter finds the head by mesh name.
- Fixed in `8153dbb`, with a blocking zone-map test. A 32-bout smoke shows head-hit rates are back to full-hull levels.
- Every other R2 diagnostic step kept the eval's head-hit rates. Survival matched exactly.

**Platform (A8).**
- The RunPod CPU pod (AMD EPYC) is not bitwise equal to the laptop (Intel). So R-EQ runs **EO and the replica on the same
  pod**.
- The pod R0-EO summary rates match the laptop's well within the margins.

**Learning canary (M10, report only): no learning** in 400 small-batch iterations, because the adaptive lr was pinned at its
1e-5 floor (KL about 0.016). This is a real risk for GPU. WATCH6 E4 will catch a stall early. Consider it at P0/P4 without
changing frozen rules; any change needs an amendment.

**Money.** RunPod balance about $0.90; GPU spend so far $0. CPU pod spend about $3.3 in total. No pods are running.

## 7. Next steps, in order

### Step 1: R-EQ rerun (A10.4)

Needs a top-up of about $5; ask the user. Expected cost about $2. It runs on the reserve block 100000-112287, EO and replica both
on one pod.

**1. Rent a CPU pod** through the RunPod REST API. runpodctl cannot set vCPU.
- `POST https://rest.runpod.io/v1/pods` with:
  - computeType CPU, cpuFlavorIds ["cpu3g", "cpu5g"], vcpuCount 32 (the max);
  - imageName `runpod/base:1.0.2-ubuntu2404`, containerDiskInGb 40, ports ["22/tcp"], cloudType SECURE.
- That costs $1.28/h. Live prices: `GET https://api.runpod.io/v2/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=32`.
- API key: in `~/.runpod/config.toml`, **single-quoted**. Strip the quotes. Never print it or put it in a URL.

**2. Access the pod.**
- Direct TCP to the mapped SSH port **times out** from this PC.
- Use the RunPod SSH proxy: `ssh -tt -i ~/.ssh/id_ed25519_occlubench_runpod <podHostId>@ssh.runpod.io`. It is PTY-only.
  - Get podHostId from GraphQL `pod(input:{podId}) { machine { podHostId } }`.
  - The runpodctl key `~/.runpod/ssh/runpodctl-ssh-key` is not readable from the agent sandbox.
- `pod/podrun.sh <script> [timeout]` runs a local bash script remotely by base64 over stdin. Update its `H=` line with the new
  podHostId.

**3. Install on the pod.**
- The venv: see `t2.sh`. Use torch 2.14.1 from the CPU index, mujoco 3.17.0 and numpy 2.5.4, as in `venv_freeze.txt`.

**4. Move files to the pod.**
- `runpodctl send/receive` (croc) corrupts large files silently, even when the size matches.
- Bundle the repo's tracked files plus the m1m2 subset; see the bundle commands in the transcript pattern and `sendparts.sh`.
- Split into 25 MB parts. `pod/sendparts.sh` sends each part and **verifies sha256**, with retries. `ONLY="b.03 b.05"` resends
  specific parts.
- Unpack on the pod, then run `pod/port.py /root/repo /root/m1m2` to rewrite the Windows paths.
- Then `pip install -e /root/repo`.

**5. Pull results back** with `runpodctl send` from the pod and `receive` locally. Verify sha256 and retry; it can take
several tries.

**6. Delete the pod** as soon as it is idle (`runpodctl pod delete <id>`). Idle time burns $1.28/h, and RunPod kills the pod at a
zero balance.

**7. Implement the rerun first.**
- `fb_m6_run.py req` currently takes R0-EO's episodes on 70000-82287. The rerun needs EO **and** the replica on the reserve
  block 100000-112287 (3.2 step 2).
- Add a role/block option to fb_r0eo and fb_m6_run (the fb_seeds role for 100000-112287 may need adding; check
  `fb_seeds.py`).
- Run R0-EO on the pod for that block: about 30 min at 30 workers.
- Then `req` against it, and R2 for the hulls-mix step only. Use `FB_MAX_WORKERS=32`, `OMP_NUM_THREADS=1`, `--workers 30`.
- Have an independent verifier review the code change before spending.

**8. If it passes,** continue. **If it fails a second time,** follow 3.2 step 3:
- noise-sized: write a report and ask the user;
- conclusive: stop, find the cause, no GPU.

### Step 2: M11, the preregistration (plan section 3, row M11)

- Write `results/sparring/prereg_m2_fb.json`, with `scripts/m2_verdict.py --run fb` roles.
- The hash covers:
  - the plan with all amendments, D2 and the L1 priors;
  - the WATCH6 thresholds;
  - the R-EQ/R4g/T1-c/LOCO statistics and margins;
  - W0 / W_EPS;
  - the seed ledger;
  - the Track 2 decision (excluded, A5.2);
  - the test (512 bouts, 22000-22511, test-once guard).
- Dry-run the verdict on synthetic rows. Builder plus verifier, then commit.

### Step 3: GPU (plan section 4)

- Ask the user to top up. The plan says about $65 pessimistic for Track 1.
- RTX 4090 with at least 16 vCPU, about $0.89/h. Run `gpu/pod_fb.sh` stages P0-P7.
- Upload the pod R0-EO (`podpull/r0eo_pod.tgz`) for P1-R4g. Set FB_R0EO or use `$W/r0eo_pod`.
- **A11.1:** on the GPU pod keep run 3's mujoco 3.5.0 / mujoco-warp stack. P0 re-runs the M9 model diff, including zone maps.
- The mjlab GPU glue has **never executed** (no mujoco_warp locally). Expect a fix cycle at P0/P1; it is budgeted.
- Stages:
  - P1: parity;
  - P2: T1 for the reangle family plus T1-c;
  - P3: LOCO pilot, whose bars decide proceed or stop;
  - P4-P6: three seed pairs, sighted and blind, with WATCH6;
  - P7: pack and terminate.
- Report spend at $45 and at $80.
- Then selection (section 5, local CPU) and the screen S1-S5 (section 6.2; kfab_204 used once). Run the hard_r2 test **once**,
  only on go.

## 8. Decisions already made (do not re-litigate)

- A1-A3: aim gates. Training jitter keeps forward reach at nominal through a target-space secant correction, with a bisection
  safety net.
- A4/A5:
  - W0 comes from the R4-pre knockdown bouts and is 0.
  - W_EPS floor; no knockdown charge in the first 0.5 s.
  - Track 2 is out of this prereg.
  - R-EQ looks are at 22/43/64 shards; T1-c uses shards 0-31.
- A6:
  - Combo speed rule as stated.
  - Mirrored rows use the mirror-conjugated tracker.
  - LOCO vmax per clip set.
  - The R1 1e-9 gates apply to the configuration with the eval tracker plugged in.
  - Closing speed comes from the palm/foot site.
- A7: batched replica; batch invariance is reported (rowwise = bitwise N = 1).
- A8: heavy CPU on the cloud; same-platform comparisons.
- A9: ONNX in float64; blind flag; T1 on reangle only; the p 0.2 mirrored swap.
- A10: the hull-name bug, the fix and the reserve rerun.
- A11:
  - mujoco 3.5.0 on the GPU pod.
  - "Iterations 1-100" is read per training process.
  - The remaining order is: R-EQ rerun, then M11, then GPU.

## 9. Known risks

- The learning canary is flat (section 6). It could mean the GPU run stalls; WATCH6 E4 stops it early.
- The GPU glue is untested until P0/P1.
- The pass odds are honestly well under 50% (plan section 8). The point of M2 v2 is that a failure is real, not a setup bug.
- Docs in `docs/m2v2` contain local Windows paths. That is harmless, but don't be surprised.
