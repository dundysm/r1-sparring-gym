# M5 build notes (M2V2_PLAN_v3 section 3, M5: G0-11, G0-12), 2026-10-30

**Status:**
- **G0-12: PASS.** Every scripted event passes, and every recorded-bout check passes on 256 EO bouts. W0 is frozen.
  - **Open item 1:** W0 rests on 2 knockdowns (see below). The main session must decide before M11 hashes it.
- **G0-11: the harness is built.** It passes on the EO's own states for block 60200-60263, G and H.
  - Landed counts are exact on all 128 bouts.
  - Clean agreement is 0.952 / 0.958 with COM velocities, and 0.995 / 0.999 with contact-point velocities.
  - The median impulse differs by 1e-5 (relative).
  - The plan orders G0-11 after M6 R1, so its gating run on replica bouts remains for M6 (section "What remains for M6").
- `scripts/fb_m5_check.py` exits 0 (`{"G011_on_EO_states": true, "G012": true}`).
  - Recording: 486 s, 256 bouts, each run twice.
  - Checks with `--reuse` on the saved recording: 167 s.

**Rules kept:**
- Only new, untracked files were added. The tracked `git diff` is empty.
- Nothing was committed, stashed or branch-switched.
- CPU only, at most 4 workers, $0 spent.
- No thresholds were changed.
- No `Users*` folder or MUJOCO_LOG.TXT was left in the repo.
- `fb_seeds.py --audit --scan`: ok (19 roles; ledger consistent; 20 task inputs clean).
- **Full suite: 326 passed, 2 skipped** in 702 s (`m5_suite.log`). The 2 skipped are the env-gated tests. MUJOCO_LOG.TXT from the suite was deleted.

## Files (all new)
| file | what |
|---|---|
| `gpu/fb_hits.py` | **Constants:**<ul><li>ZONES, LIMBS, SENSORS (side x strike limb), HIT_P, FORCE_SIGN, T_TOL / W_TOL, WIN_TAIL_S, BLOCK_BY_APEX_S;</li><li>`zone_of` and `limb_of` (copies of r1_zone / opp_limb, asserted equal);</li><li>LIMB_SECONDARY: the foot sensor's secondary is the **ankle_pitch subtree**. The HitMeter maps both ankle bodies to "foot", so D2 7.1's ankle_roll subtree would miss the ankle_pitch shell.</li></ul>**`zone_forces`:** sum of \|F\| over each zone's primaries, touch from `found`, and the zone's strongest primary.<br>**`closing_speeds`:** FORCE_SIGN x (v_s - v_t) . F_hat at the strongest primary; signed or \|.\|.<br>**`Episodes`:** HitMeter.events (gap merge, impulse, peak, closing speed at the first substep, HitMeter's rounding before thresholds).<br>**`FullBodyHitAccount`:**<ul><li>merged-side (limb, zone) episodes for counts and rewards;</li><li>per-side episodes for windows;</li><li>AimOp-slot windows with HitMeter.windows outcomes and the HIT_R2 + m2 pay;</li><li>`land_t` (the last touch of a landed strike on head or torso);</li><li>body sensor: tangle (pelvis + leg, per-substep \|F\| dt) and work W (torso + pelvis + leg, max(0, F . v) dt);</li><li>`close_envs` and `reset`; an optional log.</li></ul>**`MjContactSensors`:** the CPU sensor. It **equals MuJoCo 3.17's native mjSENS_CONTACT exactly** when built as mjlab builds it (netforce = the force on the secondary, world frame). |
| `gpu/fb_rewards.py` | **WEIGHTS A / B:** the plan's 2.4 table. Clearance is A only. B adds fb_lean_away -0.01 and res_l2 (-0.5 to -0.1 over B 0-2000, `res_l2_weight`).<br>**Constants:** HIT_TERMS, `push_mask`, `clearance_value`, `lean_away_deg`.<br>**`KnockdownAttribution`:** a 1 s work ring. charge = edge & W >= W0 & W > 0 & no landed strike within 0.3 s.<br>**`DownNeutral`:** a per-row exposure-weighted EMA of head clean per both-up minute. Only rows with exposure decay.<br>**`FbRewards`:** term values under mjlab's convention (reward = w x value x 0.02).<br>**W0 = 0.29906465943470145 J** (frozen). |
| `scripts/fb_m5_check.py` | **Recording:** EO with read-only hooks; a substep sensor logger both ways; post-step state; the HitMeter's events and intent windows; a python HIT_R2 + m2 `pay_ref`.<br>**Replay:** the batched account and FbRewards.<br>**Checks:** `g0_11` / `compare`, `calibrate_w0`, `recorded_checks`, `scripted()`. |
| `tests/test_fb_hits.py` | 7 tests:<ul><li>constants vs r1gym, aim and run 2;</li><li>zone and limb maps vs HitMeter both ways;</li><li>**MjContactSensors = MuJoCo's contact sensor** on random overlapping two-R1 states (8 limb sensors + body sensor);</li><li>zone forces and closing speeds;</li><li>**the account = HitMeter.events / windows and pay = pay_ref on random synthetic contact logs** (24 envs, more than 500 episodes, more than 100 windows), in float64 and float32;</li><li>close_envs.</li></ul> |
| `tests/test_fb_rewards.py` | 16 tests: the 11 scripted events (each a test); the weights table; W0 frozen; the attribution ring; the EMA rows; the lean sign and the push-mask boundary. |
| `tests/test_fb_m5.py` | 3 tests on G 60200 (8 s, role hit_scoring): the recorder refuses other blocks; account = HitMeter (point: counts, clean, windows, pay; com: counts); the recorded terms. |

**Outputs:**
- Recording (scratch): `...\fullbody\build\rec\fb_m5rec_60200-60363.npz`
- Results: `build\m5_check.json` and `m5_check.log`. Pre-freeze run: `m5_check_pre.json` / `.log`. Suite: `m5_suite.log`.
- Probes: `build\m5\probe_sensor.py` (the first sensor-sign probe, on M4 recorded states) and `build\m5\edges_all.json` (all 23 knockdown edges).

## Commands (cwd = repo, PYTHONIOENCODING=utf-8, S = ...\scratchpad, B = $S\m1m2\fullbody\build)
```
$PY -I scripts/fb_m5_check.py --out $B\m5_check.json --rec $B\rec\fb_m5rec_60200-60363.npz --p1 $S\m1m2\gap\run4pre\P1\p1_eval.py --workers 4           # record + checks
$PY -I scripts/fb_m5_check.py ... --reuse                                                    # checks on the saved recording, exit 0, 167 s
$PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests/test_fb_hits.py tests/test_fb_rewards.py tests/test_fb_m5.py
$PY -m pytest -q -p no:cacheprovider --basetemp="$S\pytest-tmp" tests                          # full suite -> m5_suite.log
$PY -I scripts/fb_seeds.py --audit $S\m1m2\fullbody\SEED_LEDGER.md --scan
```

## Recording
- **Episodes:**
  - fb_record's mapping: seed s -> clip TRAIN[(s-60000) % 3], distance HARD[((s-60000)//3) % 4];
  - EO.episodes draws;
  - aim standoff+armwarp_vz, DR, delays as drawn, no pushes, 20 s.
- **Controllers:** G and P1's H(A3s)/M3s (keypoints truth).
- **Blocks:** 60200-60263 (role hit_scoring) and 60300-60363 (role w0_calib), checked through `fb_seeds.check`.
- **EO flags:** `meter`, `falls` and `rev_hits` are on.
- **Hooks:** the R1HitEnv subclass appends its logger **after** the eval's hooks, so hooks[0] is still the HitMeter.
- **Every bout also ran without hooks, and the rows were identical (asserted).**
- 256 bouts. Memory stays at 0.56 GB per worker after the env/logger cycle is broken explicitly. The first launch grew to 2.8 GB per worker and was killed, then restarted.

## Results

### G0-11 (same-state: FullBodyHitAccount on the logged sensors vs the eval's HitMeter on the same contacts), block 60200-60263

| | landed HM / ours | bouts with a count mismatch per (limb, zone) | clean agreement (gate >= 0.95) | clean HM / ours | median landed head impulse HM / ours (gate within 10%) |
|---|---|---|---|---|---|
| G, **com** (gated) | 831 / 831 | **0 / 64** | **0.9519** | 323 / 299 | 2.9910 / 2.99099 (rel 4e-6) |
| H, **com** (gated) | 782 / 782 | **0 / 64** | **0.9578** | 276 / 257 | 2.52655 / 2.52657 (rel 9e-6) |
| G, point (report) | 831 / 831 | 0 / 64 | 0.9952 | 323 / 319 | same |
| H, point (report) | 782 / 782 | 0 / 64 | 0.9987 | 276 / 275 | same |

- **Windows (report):**
  - Intent window outcomes vs HitMeter.windows: 1847 / 1853 (G) and 1838 / 1843 (H).
  - Pay vs the python HIT_R2 + m2 reference on the HitMeter's own per-side episodes: 1852 / 1853 and 1841 / 1843.
  - Paid totals are equal: 435 = 435 and 419 = 419.
  - The few differences come from ring-slot visibility. In the replay an intent appears at t_on - 0.8 s, while HitMeter scores every window independently.
- **Reverse direction (R1 -> opponent, report):** landed counts equal HitMeter(reverse=True) on every bout: G 412 = 412, H 346 = 346.
- **Closing-speed input matters:**
  - Body-COM velocities (the cheap GPU input) under-call clean hits by about 7% (299 vs 323), because they miss the limb's rotation. That leaves the COM agreement only 0.002-0.008 above the gate.
  - Velocities at the strongest contact's point (HitMeter's own) give 0.995-0.999.
  - **Recommendation for M9:** feed point velocities (cvel + w x r at the sensor's contact position, or at the palm or foot site).

### G0-12

**Scripted events** (`scripted()`, all ok):

| event | result |
|---|---|
| attributed knockdown | exactly one -2.4 reward, at the edge. Up again and down again gives no second event (`ever` is sticky). |
| tracker-alone fall (no contact) | 0 |
| knockdown 0.12 s after a landed strike | 0 |
| control: the same strike 0.36 s before | -2.4 |
| blocked strike (hand on the arm zone), then a stumble with only arm-shell contact (W = 0 even with 300 N at 0.5 m/s on the arm shell) | 0 |
| down-neutral pattern 0x10 / 1x20 / 0x10 / 1x5 / 0x5 | charged on exactly the 25 down steps, at -0.0666 per s (prior 5/min x 0.8 / 60) |
| fb_down_approach and fb_down_contact | 0 while both are up; when down, exactly -2 x 0.3 m/s x 0.02 and -0.1 x 1.0 N s |
| HIT_R2: the glove touches the arm before the apex | pays 1; with an l forearm then landing on the head, 0; with an r forearm (other side), 1; with an l hand landing, 0 |
| tangle with alternating +-40 N on a leg shell plus 10 N on the pelvis | 1.0 N s, as per-substep \|F\| summed. The torso is excluded. Reward -0.05 per N s. |
| push mask | every hit term is 0 for 0.5 s after a push: head clean, impulse, tangle, approach, contact and knockdown are all non-zero after it. A knockdown with W = 12.4 J inside the mask gives 0, while down-neutral still accrues. |
| stages | clearance only in A; res_l2 anneal; -2.4 = 3 he |

**W0 calibration** (G, 60300-60363):
- 4 knockdown edges:

  | seed | t (s) | W (J) | contact |
  |---|---|---|---|
  | 60333 | 1.02 | 0 | no |
  | 60302 | 11.44 | 2.991 | yes (EO ContactTail: yes) |
  | 60326 | 12.44 | 0 | no |
  | 60331 | 18.34 | 0 | yes (a pelvis/leg touch with no positive work; ContactTail 0.5 s: no) |

- **W0 = P10 over the knockdowns with contact = P10{0, 2.9906} = 0.29906 J**, frozen in `fb_rewards.W0`. `frozen_ok` is true.
- The work-only variant (W > 0) would be 2.9906 J (n = 1).

**Recorded bouts** (all 256, W0 frozen):

| block / ctrl | edges | charged | down steps | fb_down_neutral only while down | approach / contact 0 while up | tangle vs numpy | tangle N s/min |
|---|---|---|---|---|---|---|---|
| 60200 G | 7 | 2 | 4187 | yes | yes | 3.6e-15 | 178.0 |
| 60200 H | 9 | 3 | 4139 | yes | yes | 3.6e-15 | 199.3 |
| 60300 G | 4 | 1 | 1842 | yes | yes | 1.8e-15 | 149.1 |
| 60300 H | 3 | 1 | 1530 | yes | yes | 1.8e-15 | 185.2 |

- Every charged event is one step of -2.4 at a rising edge. Its W >= W0, and no landed strike came within 0.3 s.
  - 60229 G has W 1.26 J but a strike landed 0.01 s before, so it is not charged.
- 2 bouts got up again. Down-neutral returns to 0 for them.
- Lean-away mean: G 8.1 / 7.5 deg, H 11.9 / 11.9 deg.

## Notes for the main session (facts; nothing decided here)

1. **W0 is statistically empty as specified.**
   - G0-12 says "W0 calibrated on the R4-pre G knockdowns re-recorded with contact logs (60300-60363)". The ledger gives M5 only the 64 fresh episodes 60300-60363.
   - At G's knockdown rate (0.073, R4-pre), that gives 4 knockdowns, 2 with contact and 1 with W > 0, so the frozen W0 = 0.299 J comes from n = 2. The eval's own bout produced these.
   - The R4-pre G knockdowns themselves (56 in 67000-67767, 88 in 68000-68767) cannot be re-recorded under any M5 role: fb_seeds gives 67000-67767 to `r2_diag` and nothing to 68000-68767.
   - **Sensitivity on all 23 recorded edges** (G and H, both blocks; `build\m5\edges_all.json`):
     - W0 = 0.299 J charges 7: W 0.57-5.40 J.
     - W0 = 2.99 J would charge 3.
     - The 16 uncharged are: 13 with W = 0, two with W 0.02-0.05 J, and one (60229 G, W 1.26 J) with a landed strike 0.01 s earlier.
     - Early knockdowns at t = 0.20-0.34 s (60201, 60230, both controllers) carry W 2.2-5.4 J. These are start-of-bout tangles.
   - **Options (not decided):**
     - (a) Add a role for re-recording the R4-pre G knockdown bouts (about 144 edges) with contact logs, an amendment to 6.4 and fb_seeds.
     - (b) Enlarge `w0_calib` with fresh seeds (about 800 bouts for about 60 G knockdowns).
     - (c) Keep 0.299 J.
     - (d) Use the work-only reading (2.99 J).
   - The check reads "with contact" literally: a defender torso, pelvis or leg shell touched the opponent in the last 1 s. The charge rule also needs W > 0, so a no-work knockdown is never charged whatever W0 is.
2. **G0-11 ordering.** The plan puts G0-11 after M6 R1, so this run is the same scorer comparison on **EO states**.
   - Block 60200-60263 is reserved for G0-11 and was used in its own role.
   - The R1 replica in eval mode is promised to be bit-identical to EO.run, so its states would be these.
   - **What remains for M6:**
     - (i) Run `fb_m5_check.replay`/`compare` on 60200-60263 driven by `fb_replica.py`, in the final configuration (hulls of 2.1, the chosen solver, torch modules), with the same gates.
     - (ii) The replica must log the same sensor tuple, `MjContactSensors.read`, after every mj_step.
     - (iii) The ledger gets a "re-run in role" row.
3. **Closing-speed input (for M9 and P1-Hit).**
   - COM velocities pass G0-11 by only 0.002 (G). Contact-point velocities pass by about 0.045.
   - mjlab's ContactSensorCfg lists a `pos` field. Whether it is meaningful under reduce "netforce" is unchecked. If it is not, the palm or foot site serves as the point. Then v = v_com + w x (p - xipos) of both bodies is cheap on the GPU.
   - This is a choice for M9. The account takes any v_rel.
4. **A zero-force first touch has no direction.** A netforce sensor cannot give it a closing speed, so the account calls it 0, while the HitMeter uses the contact normal.
   - This is part of the COM/point gap above. The synthetic test excludes it by construction (stated in its docstring).
5. **Netforce sign.**
   - On CPU MuJoCo 3.17, the contact sensor's netforce is the force on the **secondary**: probe and test, exact.
   - The hit sensors (primary = defender shell) therefore read -push, and FORCE_SIGN = -1 matches run 2's measurement. The body sensor reads the defender's push on the opponent directly, which is what W uses.
   - Warp's sign must be re-checked at P1 (unchanged from run 2's note).
6. **Not covered by M5 (glue for M9):**
   - wiring sensors and history order (mjlab's force_history index 0 is the latest; the account wants oldest first);
   - the per-substep velocity ring;
   - push times and population rows into FbRewards;
   - spar_ring / spar_alive (existing ST terms);
   - clearance distances from AimOp's head intents.
7. **GPU memory.** The hit sensors as (N, 8, 32, 4, 3) float32 take about 50 MB at 4096 envs.

## Ledger
Two rows were appended ("M5"):
- 60200-60263, role hit_scoring, which also covers the 2-bout smoke, the killed first run and the test's 60200 re-run;
- 60300-60363, role w0_calib, which also covers the 3-bout timing probe.
