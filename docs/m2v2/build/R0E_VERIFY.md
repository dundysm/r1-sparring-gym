# R0E verify: M6 R0-EO reference runner (scripts/fb_r0eo.py)

2026-10-30, independent verifier, branch feat/sparring-m2v2 (on e9fdd66). CPU only, at most 2 worker processes, the live 4-worker
run left untouched. No repo file changed by this verification (fb_r0eo.py / test_fb_r0eo.py read only, still LF). Probes and logs:
`build\verify_r0e\` (probe_static.py, probe_direct.py + .json/.log, probe_poses.py + poses.log, suite.log).

**Verdict: ok.** Every frozen R0E check passes when re-run; the code reads line for line as eval_opponent.main's hard-tier
`--aim chosen --no_pushes --hits 1 --keypoints truth` cell (with R4-pre's controllers and episode construction); no semantic
mismatch found. The live run is healthy and its shards so far are correct.

## 1. Re-run checks

| check | result |
|---|---|
| Episodes, independently | `EO.episodes(train clips, 1024, hard tier with seed 70000, hits off, 20 s)` with **R4-pre's own `good_*.npz`** (not fb_motion.npz) == `fb_r0eo.episodes()` on all 12288 dicts; sha256 `08051fdf...36cf4a`; seeds exactly 70000-82287; seed = 70000 + (clip x 4 + distance) x 1024 + rep on every episode; hits empty; every t0 on a clean grid point of its stratum; k counts 4148/4046/4094 |
| EO main's clean starts | G, H and A3 homes are all the `guard` pose of REF_POSES, so eval_opponent.main would add no pose: its clean_starts (base cfg) == fb_motion.npz (M1 (3)) == R4-pre's cache (builder check). Eval's construction and R4-pre's coincide |
| Cell config | cfg = controller_cfg + opponent_mode policy (= load_cfg's) + opponent_aim standoff+armwarp_vz (= aim.json operator); lag 0.02 = aim.json lag_s; hit_metric from the controller cfg, as main's `ccfg`. EO.run args = main's (dr=tier dr True, meter, falls True, rev False, keypoints truth) |
| H scale | `action_scale(cfg)`, `cfg.get("action_clip", 4.0)` == a fresh `R1HitEnv(cfg).scale / .clip` exactly (R1HitEnv computes the same expressions on the same cfg; P1 sets them from the env) |
| Rows vs a direct EO.run, **new episodes and new batch composition** | 14 episodes from live-run shards 1-2 (chosen for multi-edge knockdowns, falls after kd, falls without kd, plus spread fillers: 70016 ... 81293), controllers built **independently** (load_controller / P1 make_ctrl + set_scale from env), one plain EO.run per controller, threads 1, fresh spawn process: **14/14 bit-identical `eo` rows for G, H and A3** |
| Rows vs R4-pre's own harness | p1_eval.run (the R4-pre path) on the same 14 episodes: fell_s, knockdown flag, knockdown time, all 6 HitMeter counts and act_delay **equal for all 42 bouts** -> R0-EO is the R4-pre "step 0" bout, not just EO |
| Stored shards (G 0-4, H 0-4, A3 0-4 at the time of the probe; 11 shards in probe_static, more since) | meta == shard_meta, row seeds == shard episodes; **gzip bytes == re-serialisation** (byte-deterministic); every derived field re-derived from the verbatim EO row: identity/draw keys, fell_s/survived/secs/up_min, kd == opp_fell, kd_t == kd_edges[0] == opp_fall.t_s, kd_n, n / n_both_up / both_up_s, fall_after_kd, kd_contact; down_iv and EO's both_up_iv tile (0, secs] exactly with no gap/overlap; opp_up_end; 0 row errors. Paired on 576 common seeds with identical (t0, k, env_seed, act_delay) across controllers |
| fall_after_kd vs R4-pre | analyze_r4pre.py uses `f * k` (bout-level fall and knockdown) -> `fall_after_kd = kd and fell` is R4-pre's exact P(fall given kd) numerator |
| up-minutes | secs = fell_s or 20 = R4-pre's `secs` and HitMeter's exposure; head clean per up-minute = sum head_clean / sum(secs)/60 |
| tests/test_fb_r0eo.py | 7 passed (6.2 s) |
| fb_seeds | `--audit SEED_LEDGER.md --scan`: ok, 19 roles disjoint, ledger consistent, 25 task inputs clean (after my row) |
| Full suite | see section 4 |

Not re-run (proved by the builder, consistent with everything above, re-running would cost ~13 min of 2 workers per controller):
shard 0 as 32-bout CHUNK calls byte-identical to the 192-bout files; act_delay vs fresh-reset draw on shard 0. My 14-bout
identity on CHUNK-produced shards 1-2, with a different batch composition, covers the same property (rows depend on the episode only).

## 2. Code read against the eval (no mismatch)
- Probe: `EO.R1HitEnv` is a module global resolved inside `EO.run` at call time, so the subclass is used; it only reads
  `act_delay` after reset and `d.xmat/xpos` after step with EO:193's exact test (`min(opp.bodies)`, OPP_TILT, OPP_Z, `e.t`);
  restored in `finally`. The bout_row partition assert ties it to EO's own both_up_iv on every bout of the full run.
- Edge times `a * dt` = EO's `round(e.t * e.dt, 3)`; down_iv `((a-1) dt, b dt]` = EO's both_up_iv convention (EO:213).
- `n = round(fell/dt)` / `int(T/dt)` = the number of steps EO ran (its loop stops after the falling step).
- Seed guard (role `eq`, train clips) is called on every EO.run chunk; nothing opens val/test seeds or clips; the check reads
  R4-pre's run files and builds R4-pre's episodes without simulating.
- Shards: one task = (ctrl, shard), shard j = reps 16j..16j+15 of all 12 strata (balanced); atomic temp+fsync+replace;
  resume validates meta + seeds; manifest refuses another configuration (fmt, block, clips, distances, seconds, aim, lag,
  cfg sha, episodes sha, fb_motion sha, P1 sha).

## 3. Live run (not waited for, not touched)
- PID 24024 (main) under launcher 5228, 4 pool workers (20264, 4392, 15088, 8472), all alive at 05:32; it survived the
  builder's exit. Commit 2.2-2.8 GB per worker (~11 GB total), free virtual ~18 GB.
- 12 of 189 tasks done at 0.42 h (438-536 s per task) -> about **6.4-6.8 h**, done around **11:30-12:00** (the builder's
  11:05-11:40 is slightly optimistic with other lanes on the CPU). Resume command as in R0E_NOTES.md.

## 4. Full suite
**345 passed, 2 skipped in 1137 s** (`build\verify_r0e\suite.log`; slower than the builder's 954 s with the R0-EO run and
another lane on the CPU). The repo `MUJOCO_LOG.TXT` (rewritten 05:50:45 during the suite) was deleted; no "Users..." folder;
git shows no tracked change from this lane. At 05:51 the live run had 20/189 tasks done, ETA 6.1 h, run.err empty.

## 5. Issues and notes (no blocker; decisions belong to fb_eq.py / the main session)
1. **Group-sequential looks and T1-c's set** (builder issues 3, 4): confirmed. Seeds are clip-major, so 70000-74095 is all
   kfab_200 and "70000-76143" = kfab_200 + kfab_201 at 0.40/0.45 m (seed 76143: k 6143 -> clip 1, distance index 1). Looks
   and T1-c must be defined on whole shards (or exact per-stratum reps) and frozen before any look. Not decided here.
2. **Early knockdowns (A4.2 data, report only):** in shards 0-4, 21 of 81 G knockdowns (H 22/96, A3 21/67) are flagged at
   0.12-0.30 s, all by tilt, mostly with contact (18/21), concentrated on kfab_203 0.40-0.50 m and kfab_200 0.55 m, the same
   episodes across controllers. They are EO's own definition (R0-EO mirrors it); A4.2's report can read them from `kd_t`.
3. **Job object:** the run is in a job object; it survived the builder subagent's exit, but whether it survives the whole
   session's end is unknown. Shards are atomic; use `status` and the resume command.
4. Minor robustness (no effect on rows, not fixed to avoid touching the file under a live run): `load()` checks only
   fmt/ctrl/shard of each meta (not seeds/seconds; `shard_ok` does), and each resume rewrites manifest `provenance`
   (git hash/dirty) with the latest start's, keeping only the `runs` list. fb_eq.py may call `shard_ok` before `load`.
5. Repo `MUJOCO_LOG.TXT`: one existed at 05:28:33 from another lane's concurrent pytest; see section 4 for cleanup.
6. Suite basetemp: run with `pytest-tmp-r0ev` instead of `pytest-tmp`, because another lane's pytest was running on
   `pytest-tmp` at the same time (pytest wipes its basetemp at start).

## Seed ledger
Appended one row: 14 seeds inside 70000-82287 re-run in role `eq` (EO.run and p1_eval.run, G/H/A3) plus the test re-runs.
