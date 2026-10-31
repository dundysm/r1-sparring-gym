# M3 + Amendment A1 independent verification (M2V2_PLAN_v3 G0-4 as amended), 2026-10-29

**Verdict: ok = false. A plan decision is needed; there is no code bug.**

- **The build reproduces exactly.** `fb_m3_check.py` exits 0, all 7 groups are ok, and my JSON equals the builder's
  `m3a_check.json` in every field except wall time.
- **No semantic mismatch.** I read the A1 code line by line against r1gym/aim.py and found no mismatch, so I fixed nothing.
- **Why ok is false:** the A1 jitter gate passes only at the one generator seed the check uses (seed 0, which was fixed
  before A1). It is not a property of the jitter design:
  - Over jitter seeds 0-19, on the same recorded states, the full jitter gate passes at **9 of 20 seeds**: (ii) fails at
    7 and (iii) fails at 9.
  - Pooled over the 20 seeds, (iii) fails: jitter puts the palm **+14.0 mm** beyond nominal's maximum at the apexes, against a
    +5 mm allowance.
  - Training draws millions of intents, so the tail that (iii) bounds will be reached there.
- **Everything else passes,** including A1's float32 gate.

## What I re-ran
- **The full check** on the existing recording (60000-60015, role `record`, loaded and not re-simulated), with 4 workers:
  - Exit 0, in 297 s.
  - Output in `build\verify_m3a\m3a_check_v.json` and `m3a_check_v.log`. It is equal to `m3a_check.json` in all fields
    except wall time.
- **Unchanged gates vs `m3_check.json`:** params, FK, ring, the f32 floor and the recording fields are identical. AimOp f64 is
  identical apart from the new `q_gate` label (q 1.26e-10). Every AimOp f32 key that existed before A1 is identical too.
- **Full suite** (prescribed command, FB_SEED_LEDGER set): **268 passed**, exit 0, 337 s (`build\verify_m3a\suite_v.log`).
  test_fb_aim.py has 16 tests.
- **Housekeeping:**
  - `fb_seeds.py --audit --scan`: ok.
  - `git diff` on tracked files is empty, nothing was committed, and the branch is feat/sparring-m2v2.
  - My suite run wrote a MUJOCO_LOG.TXT to the repo root, and I deleted it. No `Users*` folder exists.
  - CPU only, at most 4 processes, $0.
- **Seeds:**
  - No new block. 60001 was re-run (4 s and 1 s) by the suite in role `record`, and I appended a row to SEED_LEDGER.md.
  - No val or test seed, no test clip and no kfab_204 was used.
  - The probes use the recorded states only. Seeds 0-19 there are AimOp's jitter generator seeds, not episode seeds.

## A1 gates, reproduced
| gate | f64 | f32 |
|---|---|---|
| f32 (i): vs the float64 port on the same f32-rounded states, every row <= 1e-4 | | PASS 1.19e-7 (output rounding: the f32 op computes in f64 on the same inputs) |
| f32 (ii): vs the eval <= 1e-4 on >= 99.9% of rows | | PASS 99.974% of all 15,412 rows. On the 5,466 warped rows only, 99.927%: also passes |
| f32 (iii): closest palm <= 1 mm (plus dist <= 1e-6 and intents identical) | PASS | PASS 8.0e-8 m (dist 1.3e-8) |
| jitter (i): x target = reference palm x = nominal, bit-exact | PASS: 6,627 warped (row, limb), same set in both runs | PASS |
| jitter (ii): 90% t CI of mean dx at the 218 aimed apexes inside +-5 mm | PASS [-4.35, +0.81] mm | PASS |
| jitter (iii): max forward excess at apex, jitter <= nominal + 5 mm | PASS 123.72 vs 122.16 mm | PASS |
| jitter (iv): mixed mask, T0 bit-equal and T1 differs | PASS (7,412 T0 rows) | PASS |

## Independent probes (`build\verify_m3a\probe_v.py` -> `probe_v.json`; `probe_seeds.py` -> `probe_seeds_*.json`, `seeds_summary.txt`)
1. **Gate (ii) and (iii), recomputed with the eval's own FK.**
   - Method: MuJoCo `_palm_b`-style FK on `_sub("r1")` instead of ArmKinematics. The apex rows are re-derived from the eval's
     intents (first row with now >= t_apex; at most 0.02 s late, so the same 218 apexes). The t quantile comes from a fine
     trapezoid (t_0.95,217 = 1.651906, against the builder's 1.651906).
   - Result at seed 0: the builder's numbers exactly (-1.77 mm, [-4.35, +0.81], 123.72 vs 122.16 mm).
2. **Mixed mask, a different mask and seed** (first 8 bouts jittered, seed 3): T0 rq, rdq and dist are bit-equal to the
   no-jitter run, and T1 rows differ.
3. **Seed robustness of the jitter gate (f64), jitter generator seeds 0-19:**

   | | count / value |
   |---|---|
   | (ii) passes | 13 / 20. Every fail is on the low side, CI lower bound < -5 mm: seeds 1, 2, 4, 5, 8, 9, 15 |
   | (iii) passes | 11 / 20. Fails at seeds 1, 2, 5, 9, 11, 13, 15, 17, 18; excess up to +14.0 mm over nominal |
   | both pass (the gate) | **9 / 20** (0, 3, 6, 7, 10, 12, 14, 16, 19) |
   | per-seed mean dx | -1.80 mm (sd 1.32, range -5.06 to +0.46) |
   | pooled mean dx, 4,360 apexes | -1.80 mm, well inside +-5 mm |
   | same-apex max (jitter - nominal) | up to +137 mm |

   **Reading:**
   - The systematic quantity (ii) is fine in expectation: about -1.8 mm, so jitter adds no reach on average.
   - A single 218-apex draw has a CI half-width of 2.5-4 mm, so the lower bound often crosses -5 mm.
   - (iii) is a maximum, so it grows with the number of draws. Seed 0 passes it partly because of the small sample.

## Line-by-line read (A1 changes; AimOp/fb_opponent.py is unchanged since M3_VERIFY, file time 20:37)
- **`replay(round_to=)`:** inputs are rounded x.to(f32).to(f64), including the base reference rq0/rdq0, exactly as the f32
  run's io cast does. Clock and time stay f64, and touch is bool. So (i) is the stated same-input comparison.
- **`q_gate_a1`, `compare(port=)`:**
  - (ii) is the fraction of rows with a per-row max |q - eval| <= 1e-4, over all rows.
  - The dist, intents and closest gates are unchanged.
  - The f64 path is unchanged.
  - TOL values are pinned in a test, and nothing is loosened.
- **`jitter_test`:**
  - (i) checks p_t x against a recomputed fk(rq5) on `has`, for jitter and nominal, plus equality of the `has` sets and of
    rq5. This matches aim.py:182, where vz uses `_palm_b(rq)` x of the unwarped reference.
  - (ii) and (iii) use the warped reference palm (FK of the output rq) at the eval's apex log step (the first step with
    now >= t_apex, matching `_log`), in the opponent torso frame.
  - (iv) uses one op with a per-env mask. RNG draws are made for all envs every step, so a T0 row cannot see the mask.
- **(iii) "over the clip" has two readings, and both pass at seed 0:** at the apexes (gated: +1.57 mm), or every warped step
  (info: jitter 195 mm vs nominal 279 mm).
- **`t_quantile`, `mean_ci`, `cluster_ci`:** correct. The cluster CI is a ratio bootstrap, reported only.
- **`superseded_by_A1`:** both replaced gates still fail (f32 q 1.94e-3 on 4 rows; jitter v0 0.143 m), and `eval_explains`
  is true. They are not in ok.
- **Tests:**
  - (ii) and (iii) arithmetic is checked on the 4 s bout, and the gate runs in the script, as the builder notes.
  - The f32 test asserts eval_rows_ok on about 200 rows. This means zero rows over, which is stricter than the gate, and it
    passes.

## For the main session (options, not decided)
- **A. Keep A1 as written, with seed 0 as the gate's fixed seed.**
  - Seed 0 was in the check before A1 (`fb_m3_check.pre_A1.py` docstring), so it was not chosen after seeing results.
  - Then M3 passes.
  - Disclose that the gate passes at 9/20 seeds and that (iii) pooled is +14 mm.
- **B. Gate on the jitter distribution rather than one draw** (many seeds pooled).
  - (ii) pooled passes (-1.8 mm, tight CI).
  - (iii) as a maximum fails (+14 mm), and keeps failing as draws grow.
  - So (iii) would need restating (for example a high quantile, or a per-apex mean). That is a change to a frozen gate.
- **C. Change JITTER instead** (M3/M8 design), then re-gate:
  - The z range (-0.05, +0.03) is asymmetric, a likely source of the -1.8 mm bias.
  - Narrower yz would shrink the tail that (iii) sees.
- **Also open (builder's notes, confirmed):**
  - The bout-cluster bootstrap of (ii) at seed 0 is [-5.10, +1.39] mm. My t interval on the 16 bout means is [-4.45, +2.15].
  - The eval's own warp reaches **27.9 cm** beyond the clip over every warped step (12.2 cm at the apexes). This belongs in
    the write-up next to A1 finding 3.
  - Stand-off depth jitter moves the opponent -2.9 to +3.0 cm per intent (M3_VERIFY).

## Files
- `build\verify_m3a\`:
  - `m3a_check_v.json`, `m3a_check_v.log`, `suite_v.log`
  - `probe_v.py`, `probe_v.out`, `probe_v.json`
  - `probe_seeds.py`, `probe_seeds_*.json`, `seeds_*.out`, `seeds_summary.txt`
  - `cmp.py`
- Ledger: one row appended (M3 A1 verify).
