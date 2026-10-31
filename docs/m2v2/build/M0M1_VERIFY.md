# M0 + M1 independent verification (M2V2_PLAN_v3 section 3), 2026-10-29

**Verdict: PASS.** Every frozen M0/M1 check passes when re-run, with no threshold changed. I found 1 real semantic bug and fixed it
in the new files: the manifest source hashes depended on the checkout. I also made 2 small hardening fixes and 2 docstring
corrections. After the fixes, both npz outputs are byte-identical to the builder's (sha256 b66f0f7e..., bbe073fc...).

Setup: repo `r1-sparring-gym-spar`, feat/sparring-m2v2 @ b81fa16. Only untracked new files; `git diff` is empty; nothing was
committed. CPU only, at most 3 processes at once. No episode seed was simulated, so nothing was added to SEED_LEDGER.md. The
MUJOCO_LOG.TXT that the pytest runs left in the repo was deleted. My probes and logs are in `scratchpad\m1m2\fullbody\verify01\`.

## Re-run of the frozen checks

| check | result (mine) |
|---|---|
| M0 (1) grep | `fb_seeds.py --audit SEED_LEDGER.md --scan`: "19 roles disjoint; ledger consistent; 8 task inputs clean", exit 0 |
| M0 (2) B7 guard | 53 hygiene tests pass. Extra probes also refused: a dev role on 20000, `eq` on 67000, `s3` on kfab_204, `loco_seen` on kfab_203, `track2` on kfab_199, a mixed block, `test` without a guard, and a clip given as a bare string. `r2_diag` 67000 and an int64 array are accepted |
| M0 (3) disjoint | `ledger_problems() == []`. 6.4 matches RUN4_PLAN_v3 3.7 and the run4pre notes (58100-58123 and 58128-58143 are covered by 58100-58143). S3 56000-56191 was never used |
| M1 (1) | clips {kfab_200, 201, 203}; outputs' sha256 OK; 0 stale sources |
| M1 (2) T-ref | joint_pos 0 and joint_vel 0 (100% bit-equal, 1248 x 24); anchor pos 5.96e-8 m; anchor quat 1.19e-7. Rows 21-270, 501-1222, 2151-2426, each inside [start, start+2T). q at d = q0 + (d, 0) to 1.1e-16 |
| M1 (3) clean | 93 cells x 300 = 27,900, 0 mismatches. The masks come from a fresh eval `clean_starts` call in a separate process |
| M1 (4) strikes | 38 rows = 37 hand + 1 kick, every field equal |
| M1 (5) hulls | 64 meshes pass: worst fine waist_yaw 2.971 mm, worst coarse pelvis 9.366 mm |
| Reproducibility | Both exporters re-run into a scratch dir (and later into the repo) give byte-identical npz: b66f0f7e7d5aba7f..., bbe073fc5c5b312e... |
| Tests | fb files: 64 passed. Full suite with the prescribed command after my fixes: **236 passed, 1 skipped** (FB_SEED_LEDGER unset), 230 s. With FB_SEED_LEDGER set, before my fixes: 236 passed |

### Independent cross-checks (my own code, not the builder's)
- **Hausdorff via MuJoCo's own collider** (`verify01\v_haus.py`):
  - For every mesh vertex of all 64 eval meshes, I called `mj_geomDistance` (GJK) to a mesh geom built from the exported points. Body frame uses my own quaternion-to-matrix code.
  - Max over vertices: waist_yaw 2.9707 mm and pelvis 9.3655 mm, all 64 pass. This agrees with the builder's Ericson point-triangle numbers to 0.0005 mm (manifest rounding).
- **Body frame** (`verify01\v_misc.py`):
  - In a randomly posed two-R1 eval model, body xpos + xmat @ point lands on the geom's world-frame mesh vertices to 2.3e-16 m.
  - `bodies[k]` (and "opp_" + it) is the geom's body, for both robots.
  - All 64 mesh geoms carry the shell bit.
- **Eval-source read-through:**
  - `Tracker.frame` uses `<= blend`. The exporter's `onnx_row` matches it.
  - Every aim.py reference access goes through `tr.frame(clip, fi)`, so a per-clip-frame table covers everything the aim port will read.
  - `export_motion.blend` row layout: prev start + 2T - n == start. `csv_rows` reproduces it, and the `b < 2T - B` assert keeps reads off rows faded into the next clip.
  - Max row g + 1 = s0 + T + B + 1 < s0 + 2T - B, since T >= 250.
  - CSV quaternions are stored xyzw; rounding per value is order-free, so keeping wxyz is exact.
  - Joint order is CTRL_JOINTS: Tracker asserts the ONNX `joint_names`, and export_motion asserts the sub-model's joints.
  - The eval's clean-start call is `clean_starts({**cfg, "opponent_aim": "none"}, clips, tier distances, poses)` with `cfg = load_cfg()` (opponent_mode policy for the tracked opponent). The exporter makes the same call. All 4 hard_r2 distances lie on the 0.01 grid as identical floats.
  - `R1HitEnv.reset` places the opponent kinematically and runs no tracker step, so the masks do not depend on the ONNX.
  - `strikes.motion` = FightReplay(kinematic, distance 0, the eval's style cuts) = the q the eval replays. Mode does not change q.

## Fixes made (new files only)
1. **Bug: the manifest source hashes depended on the checkout.**
   - Files: `gpu/export_fb_motion.py` `sha256`, also used by `export_fb_hulls.py` and `fb_m1_check.py`.
   - Problem: the repo has `core.autocrlf=true`, so tracked text files are CRLF in this Windows checkout and LF in the index (`git ls-files --eol`: r1.xml, opponent.py, strikes.json, motion_index.json, r1_assumptions.yaml, ...). The builder hashed raw bytes.
   - Effect: on the pod's LF clone, every text source would read as stale. That breaks M1 (1) re-runs and the M9 (2) manifest check.
   - Fix: text sources (.py/.json/.yaml/.xml/...) are now hashed LF-normalised, the same convention as `eval_opponent.text_sha16`. Binaries are hashed raw.
   - Verified: the new hashes equal `git show HEAD:<file> | sha256sum` (for example opponent.py c63e1348..., r1.xml 19c2291c..., motion_index.json df8e898d...).
   - Both manifests were regenerated. The npz bytes did not change. New test: `test_source_hashes_are_checkout_independent`.
2. **Provenance.**
   - Both manifests now record `versions` (mujoco 3.17.0, numpy, python). The hull gate is MuJoCo's qhull, and P0/M9 compare the pod's mujoco against it.
   - The hull manifest now also hashes `gpu/export_fb_motion.py`, whose `save_npz` writes the hull npz bytes.
3. **Seed guard hardening.**
   - Problem: `check()` truncated non-integer seeds (`int(60000.9)` -> 60000) and accepted bools.
   - Fix: it now refuses any seed that is not integral. Two cases were added to `test_role_blocks_and_clip_sets`.
4. **Docstrings.**
   - `export_fb_motion.py` said "no file holding another clip's motion is read", and `fb_m1_check.py` said "no export reads [the ONNX]". Both were inaccurate.
   - The eval's `clean_starts` builds FightReplay in policy mode, and that loads the tracker ONNX (all clips) for `Tracker.check`. Its data never reaches an output, which is now stated.
   - The builder had already disclosed this in its notes.

## Remaining notes (not blockers; nothing semantic left unfixed)
- **waist_yaw passes by 0.029 mm** (2.971 / 3 mm), confirmed by GJK. A different MuJoCo qhull on the pod could flip it. The P0 fallback (all 256 / full hull) applies; the threshold stays as it is.
- **The reference depends on held-out metadata.** `fb_motion` reads `motion_index.json`: the train rows, the total length 3234 and the blend of 20. The float32 grid depends on the total length, which includes the held-out clips' frame counts. This is necessary: a plain export fails (2) at 1.5e-3 rad/s. It is metadata only, it is disclosed, and the open-intercept test shows no held-out motion file is opened.
- **kfab_203's first 20 rows are taken unblended** (their predecessor is held out). The ONNX comparison shows this has exactly 0 effect on all 276 frames.
- **Parity condition for M2 onward.** The clean masks equal the eval's only while every evaluated controller's default pose is in REF_POSES; otherwise the eval adds that pose and changes the masks. The fb defender's home must be HOME or the guard pose. M2 or M9 should assert this.
- **TestOnce is minimal**, as the builder disclosed. A granted guard can be reused within one process; the run-once rule is enforced by the spent-file check. M11 wires it to `m2_verdict --run fb`.
- **strikes.json holds held-out strike rows.** The exporter reads it and filters to train, as M1 (4) prescribes; the outputs carry no held-out name (scan clean).

## Files touched by this verification
- `scripts/fb_seeds.py`: integral-seed check.
- `tests/test_fb_hygiene.py`: 2 refusal cases.
- `gpu/export_fb_motion.py`: `sha256` LF-normalisation, `versions()`, docstring.
- `gpu/export_fb_hulls.py`: SOURCES += export_fb_motion.py, versions, rules note.
- `scripts/fb_m1_check.py`: docstring.
- `tests/test_fb_export.py`: +1 test.
- `gpu/motions/fb/*_manifest.json`: regenerated. Motion manifest sha256 dd6add16..., hull manifest 344487d2.... The npz files are unchanged.
- Fresh `scripts/fb_m1_check.py` output after the fixes: `verify01\m1_check_after.json` (ok = true).
