# R0E: M6 R0-EO reference runner (scripts/fb_r0eo.py)

2026-10-30, lane R0E, branch feat/sparring-m2v2 (on e9fdd66). CPU only. New untracked files only: `scripts/fb_r0eo.py`,
`tests/test_fb_r0eo.py`. No tracked file changed, nothing committed. Seeds: 70000-82287 only (role `eq`), ledger row appended.

## What it does
- **Bout:** `eval_opponent.run` unchanged, called as eval_opponent.main calls it for a hard_r2 cell with `--aim`. That is
  `dr=True, meter=True, falls=True, rev=False, keypoints='truth', threads=1`, with lag = aim.json `lag_s` (0.02) and 20 s.
  - The cfg is `controller_cfg + opponent_mode policy + opponent_aim standoff+armwarp_vz`.
  - The controllers are those of R4-pre:
    - G = guard_fw_r ONNX;
    - H = P1's `p1_eval.make_ctrl('guard_fw_r+a3s', 'm3s')`, with the scale and clip set from `action_scale(cfg)` and
      `action_clip`, as fb_m4_check does;
    - A3 = the run-3 A3 ONNX.
- **Episodes:** R4-pre's construction, `EO.episodes(train clips, 1024 reps, tier hard with seed 70000, hits off, 20 s)`.
  - Seeds run 70000 + k, with k = (clip x 4 + distance) x 1024 + rep.
  - Clean starts come from `fb_motion.npz`, which equal R4-pre's `good_*.npz`.
  - Every controller runs the identical list.
- **Parallelism:** one task = one (controller, shard), run on a spawn pool, shard-major.
  - Shard j holds reps 16j..16j+15 of all 12 (clip, distance) strata: 192 bouts, so every shard is balanced.
    Any prefix of whole shards is a stratified subset, which suits the 4096 / 8192 / 12288 looks.
  - A shard runs as six EO.run calls of 32 bouts each (`CHUNK`, for memory; see the issues section).
- **Read-only probe:** a subclass of R1HitEnv, installed in EO's namespace for one EO.run call and then restored.
  - It reads `act_delay` after reset.
  - After every step it applies the EO:86 opponent-down test, which gives non-sticky down runs.
  - On every bout it is asserted equal to EO's own `opp_fell`, to `opp_fall.t_s` (the first edge), and to the complement of
    `both_up_iv` (the steps 1..n partition exactly).
- **Per-bout row (what fb_eq.py needs):**
  - identity and draws: `seed, clip, distance, rep, t0, k` (obs lag), `act_delay, env_seed, hit_seed`;
  - falls: `fell_s, survived, secs, up_min`;
  - knockdowns: `kd` (= EO opp_fell = R4-pre's knockdown), `kd_t, kd_edges` (every rising edge, s), `down_iv`
    ((t_a, t_b] s), `kd_n, opp_up_end, fall_after_kd` (P(fall|kd) inputs), `kd_contact` (an R1 contact in the 0.5 s before
    the first flag: the T-opp "no defender contact" filter);
  - hits: `n` (every HitMeter count), `n_both_up, both_up_s`;
  - `eo`: the EO.run row verbatim, including the full hit_meter (rules, windows, intent windows, impulses), opp_fall with
    ContactTail pairs, both_up_iv, drift and ring.
- **Output:** `r0eo\out\<G|H|A3>\shard_NNN.json.gz`.
  - Files are gzip with mtime 0 and sort_keys, so they are byte-deterministic. Each is written to a temp file, fsynced and
    renamed.
  - `manifest.json` holds the config, the episodes sha256, cfg hashes, the P1 sha, provenance and a runs log. A resume with
    another configuration is refused.
  - `run.pid` holds the PID of the main python.
  - Resume skips valid shards. A shard is valid when its meta (fmt, ctrl, spec, shard, seconds, seeds) and its row seeds match.
  - A `STOP` file stops the run after the running tasks.
- **For fb_eq.py:** `fb_r0eo.load(out)` returns `{ctrl: rows in seed order}`. It asserts that every shard is present and that
  the controllers are paired.

## Check (`fb_r0eo.py check`, report `build\r0e_check.json`, log `r0eo\check.log`): pass

| item | result |
|---|---|
| block | 12288 episodes, seeds exactly 70000..82287, fb_seeds role `eq` ok; 64 shards partition it, each 192 bouts, 16 per stratum; k counts 0/1/2 = 4148/4046/4094; episodes sha256 `08051fdf...36cf4a` |
| R4-pre construction | `r4pre_eval.episodes(70000, 1024, 'hard')` (its own code, seed assert widened to this block only, no simulation) == ours, all 12288 dicts; clean masks == R4-pre's cache (12/12); ours with (67000, 64, hard) and (68000, 128, 0.40/0.45) == the keys (seed, clip, distance, t0, delay) of all six R4-pre run files (G/H/A3 x main/close; no simulation) |
| rows vs a direct EO.run, 70000-70015 | shard 0 (one 192-bout EO.run call) vs a plain EO.run (no probe, fresh spawn process, threads 2) of those 16 episodes: **16/16 bit identical (json of every field) for G, H and A3**; act_delay is the same across controllers |
| act_delay | the probe's value == a fresh R1HitEnv reset draw (no step) on all 192 shard-0 episodes (90 delayed) |
| CHUNK split | shard 0 re-run with 32-bout calls: all three files **byte-identical** to the 192-bout files (also a determinism re-run) |
| shard 0 summary (192 bouts each) | G: falls 5, kd 10, head clean 250 in 63.2 up-min; H: 7, 20 (23 edges), 229; A3: 15, 6, 163; fall after kd G 3, H 2, A3 4 |

- The probe-vs-EO consistency asserts ran on every bout: shard 0, the CHUNK re-run, and every shard of the full run.
- Tests: `tests/test_fb_r0eo.py`, 7 tests, about 5 s.
  - the block and shard layout;
  - the seed guard refuses 69999, 82288, 21000, 22000, 67000 and kfab_204;
  - edge derivation, and refusal on a mismatch;
  - atomic write and resume validity;
  - a 2 s bout through the runner equals a plain EO.run, for G and for A3;
  - rows do not depend on how the calls are split (seeds 70000 and 71024).
- Full suite (with the other lanes' untracked files present): **345 passed, 2 skipped** in 954 s (`build\r0e_suite.log`). The suite left a MUJOCO_LOG.TXT in the repo, which I deleted. git: no tracked diff from this lane. `scripts/fb_seeds.py` shows as modified, but that edit belongs to the Track 2 lane, not this one.

## Full run (launched detached)
- Command, run from `r0eo\` as the working directory so any MUJOCO_LOG.TXT lands there:
  `Start-Process <venv python> -I <repo>\scripts\fb_r0eo.py run --out <r0eo>\out --workers 4`, with -WindowStyle Hidden and
  stdout/stderr going to `r0eo\run.log` / `run.err`.
- Launched 2026-10-30 05:05:31. Launcher PID 5228 (venv python.exe); main python PID **24024** (`out\run.pid`); 4 pool workers.
- Todo: 189 of 192 tasks. Shard 0 came from the check and was resumed, which shows that resume works.
- First shards landed at 05:12-05:13: G 1 and 2, A3 1, H 1, at 438-488 s per task. The next round took 452-509 s, with the suite and other lanes running.
- **Expected wall: about 6.0-6.5 h** (189 / 4 x 460-500 s), so it should finish around 11:05-11:40.
- Commit memory: about 2.0-2.6 GB per worker, about 10.4 GB in total.
- **Resume command:** re-run the same Start-Process line, or:
  `cd C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody\r0eo; $env:PYTHONIOENCODING='utf-8'; & C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\venv\Scripts\python.exe -I C:\Users\dundy\Downloads\projects\r1-sparring-gym-spar\scripts\fb_r0eo.py run --out .\out --workers 4`
- Progress: `python -I scripts\fb_r0eo.py status --out <r0eo>\out`.

## Issues / hazards for the main session
1. **Memory: the first launch was stopped.**
   - Launched at 04:49 with one 192-bout EO.run call per task. It committed 7.6 GB (G) to 11.2 GB (H and A3, which have a
     keypoint twin) per worker, about 32 GB for the run.
   - System commit headroom fell to 1.3 GB of 74.5 GB, with other lanes running.
   - I killed it after about 2 minutes. No shard had been written and none is partial (writes are atomic).
   - Fix: CHUNK = 32 bouts per EO.run call. It is proven byte-identical on shard 0 and costs about 5% in speed.
   - R4-pre ran 192 per worker. M5's "killed first run (memory)" was likely the same cause.
2. **Job object.** Start-Process's children are in a Windows job object (IsProcessInJob = true for the launcher, main and
   workers). If the harness closes that job with kill-on-close when this session ends, the run stops. Nothing is lost: shards
   are atomic. Check `status` and use the resume command above.
3. **Group-sequential looks.**
   - EO's episode order is clip-major, so the first 4096 seeds are all kfab_200.
   - fb_eq.py must define each look as a stratified subset: whole shards. 4096 is not a multiple of 192 (21.33 shards).
     Use shards 0-20 (4032), 0-41 (8064) and 0-63 (12288), or take 4096 as 341/342 reps per stratum.
   - Section 3.1's power was simulated on resampled bouts, so either choice is equally valid. Pick one in fb_eq.py and freeze
     it before looking.
4. **T1-c's "70000-76143".** By seed, that range is kfab_200 (70000-74095) plus half of kfab_201 at 0.40/0.45 m. It is not a
   balanced set. If T1-c means "the first 6144 of the block", use shards 0-31 (6144 bouts, balanced) instead of the seed
   range. That is a decision for the main session, recorded here, not made.
5. **Knockdown definitions.**
   - `kd` = EO's sticky opp_fell (bout-level, as in R4-pre).
   - `kd_n` counts every rising edge, so a bout can have more than one: H's shard 0 has 23 edges in 20 kd bouts.
   - The P(fall|kd) statistic of R4-pre uses the bout-level `kd` and `fall_after_kd`.
6. rev_hits is off (the eval default). The R1-on-opponent HitMeter is not stored. It is not a 3.1 gate input.

## Files
- `C:\Users\dundy\Downloads\projects\r1-sparring-gym-spar\scripts\fb_r0eo.py`, `...\tests\test_fb_r0eo.py`
- In `C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\fullbody\`:
  - `r0eo\out\` (shards, manifest.json, run.pid) and `r0eo\run.log`, `run.err`, `check.log`;
  - `r0eo\run_aborted1.log` (the first launch);
  - `r0eo\chk_chunk\` and `chk_chunk.log` (the CHUNK identity re-run);
  - `build\r0e_check.json`, `build\r0e_suite.log`;
  - `SEED_LEDGER.md` (row 70000-82287).
