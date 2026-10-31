# M8b independent verification: A6.1-A6.4 on L1 / LOCO and the mirror conjugation, 2026-10-30

Verified against: M2V2_PLAN_v3 section 3 row M8, 6.4 (60500-60599), Amendment A6.1-A6.4 (A6.5 for the tolerance origin);
M8_VERIFY.md (V1, V2); r1gym/opponent.py Tracker.act (lines 280-312, the eval's obs), gpu/fb_opponent.py TrackerDriver,
gpu/export_motion.blend, the MJCF (r1gym/assets/r1/r1.xml via opponent._sub). Nothing in the repo was modified.

**Verdict: ok = true.** Every frozen M8 check (as amended by A6) passes and reproduces exactly; both libraries regenerate bit for
bit; my own code (not the builder's helpers) re-derives the combo rule outcome and proves the obs conjugation exact. No unfixed
semantic mismatch. The open items below are decisions/disclosures, not defects.

## 1. Re-runs (CPU, at most 2 workers)

| item | result |
|---|---|
| `export_fb_library.py --workers 2 --out verify_m8b\regen` (30.1 min, loaded machine) | **bit-equal**: every npz array, CSV, index (LF-normalised sha), manifest equal on every key (git and output hashes included). npz EM.sha256 cb0bc0aa... |
| `... --loco --out verify_m8b\regen_loco` (13.3 min) | **bit-equal** likewise; 9f939255... |
| `fb_m8_check.py --workers 2 --rec build\rec\fb_rec_60000-60015.npz` (6.4 min, exit 0) -> `verify_m8b\m8b_check_v.json` | **equal to the build's `m8b\m8b_check.json` on every key except wall_s** |
| tests/test_fb_mirror.py (FB_REC set) + test_fb_library.py + test_fb_hygiene.py | 75 passed, 1 skipped (env-gated, not test_fb_mirror) |
| full suite | see section 5 |
| `fb_seeds.py --audit SEED_LEDGER.md --scan` | ok: 20 roles disjoint, ledger consistent, 42 task inputs clean (one more than the build: another lane's file) |
| repo | no MUJOCO_LOG.TXT, no stray pytest folder; all M8b files LF |

## 2. Frozen checks (as amended)

| check | verdict | numbers (reproduced) |
|---|---|---|
| (1) manifest L1 | PASS | 33 rows, sources {200, 201, 203}; hashes; index = 30 non-mirror rows + originals, conj = mirror (15); CSV = rebuild; vmax = p99 of the set; combos 4/13 at fades 8, 9, 10, 17 |
| (1) manifest LOCO | PASS | 20 rows, row and frame sources {200, 201} only; vmax = LOCO's own p99; combos 1/4 (fade 9) |
| (2) reach L1 / LOCO | PASS | apex frames bit-equal 48/48, 43/43; window max excess -0.31 mm |
| (3) performability (A6.2 basis), 60500-60595 | reported | mirror 23/24, warp 22/24, reangle **20/24 (0.833)**, combo 22/24; families_needing_T1 = [reangle]; conjugated T0 on mirrored base = T0 on original base per family (11/12, 10/12, 11/12); mirror vs originals 24/24 same; runner = eval Tracker 67/67, 0 mismatches |
| (4) mirror | PASS | ranges 9.0e-5 <= 1e-4; FK 0.0; involution 0.0; strike detection equal |
| (5) conjugation on 15,412 recorded rows | PASS | FK of mirrored state 0.0; f64 obs 2.0e-15, action 7.1e-15; f32 obs 8.6e-7, action 3.8e-6 |
| fed reference L1 / LOCO; clean identity | PASS | 33/33, 20/20 bit-equal; 27,900 cells, 0 mismatches |

## 3. Independent evidence (my code, `build\verify_m8b\`)

- **`probe_combo.py` (A6.1 with my own blend, splice and speed rule).** For every tried splice of L1 (13) and LOCO (4): fit
  (seam, guard) per fade 8..17, the new motion built from scratch (asserting it equals the host outside the span), allowance
  max(vmax, fastest(host frames e..e+1+2f), fastest(source segment)) + 0.05 measured apart. Result: exactly the builder's accepts
  at exactly the shortest passing fade (200_o0@41 f9, 201_o0@471 f8, 201_m0@306 f17, 203_o0@245 f10; LOCO 200_o0@41 f9); every
  rejected splice fails at every fitting fade; each row's q equals its accepted splices applied to the base, bit for bit. vmax per
  clip set recomputed = manifest (L1 and LOCO). The combo uses max(vmax[0], vmax[1]) (M8's carry-over); with the host's own
  vmax[m] instead the accepted set and fades are identical, so the reading does not matter.
  The fade grid 8..17 frames is exactly the frame grid of [0.15, 0.35] s (and what round(U(0.15, 0.35) x 50) could draw in M8).
- **`probe_conj.py` (M_o and the joint map from first principles).** Joint permutation and signs derived from the MJCF joint axes
  at qpos0 (sign = -(a_i . M a_perm(i)), M = diag(1, -1, 1)) equal fb_mirror's JP/JS. The 132-d obs rebuilt in numpy exactly as
  r1gym Tracker.act builds it (mju quats, torso xmat, recorded post-step yaw/spot) equals the recorded obs64 on all 15,412 rows
  (0.0). In the mirrored world (own qpos/qvel mirror, MuJoCo FK, yaw -> -yaw, spot/apos/aori/rq/rdq/last mirrored) that obs
  equals FM.mirror_obs(obs) **bit for bit on 15,412/15,412 rows (0.0)**. The builder's non-bit-exact rows are therefore only the
  driver's face() path (want = atan2 - pi sits 2 pi apart between worlds), which enters through cos/sin only; inside 1e-9/1e-5.
  Gyro is a pseudovector (-x, y, -z), free-joint linear qvel world-frame (x, -y, z), angular local-frame: all as fb_mirror has it.
- **`probe_fed.py` (the conjugated row is fed the reference of its own, mirrored, world).** npz ref_joint_pos vs the row's own q
  joints: conj rows <= 6.7e-4 rad, others <= 9.8e-4 (M8's csv_to_npz grid scale); ref anchor vs the row's FK torso <= 8.3e-5 m.
  A fed reference left un-mirrored would miss by tenths of a radian, so the A6.2 contract (npz ref = mirror_ref of T1's own
  shown reference) holds in the real world, and M_o of the driver's obs is what T1 was trained on.

Code read line by line (export_fb_library combo_row/splice/speed_over/bases/library/t1_rows/track_motion/fed, fb_mirror,
r1_track_spar_cfg.build_motion, fb_m8_check (1)/(3)/(5)/shown): no mismatch. Checked specifically: splice span new[e:e+L+2]
covers both unmodified boundary frames (host e and host r+f at index e+1+L); splices applied last-first and the 1 s gap keeps
each host window untouched by a later splice; `index_path` of the converted LOCO npz = fb_l1_track_loco_index.json; build_motion
passes the mirror flag in t1_rows' slot r[3]; MirrorPolicy's `last` stays real-world (the driver stores the conjugated action),
DEFAULT and SCALE are mirror-symmetric (tested and covered by (5)'s joint targets).

## 4. Findings (none blocks ok)

1. **Reangle below T1-b's bar (0.833), as frozen.** Reproduced; all 4 falls are kfab_200 t0 2.5 s at 0.52 s (o0, o1, m0, m1),
   T0's own fall on the unmodified original at that start; falls_after_modified 0. Whether T1-b's "after cuts" excludes such
   starts is the main session's call. Frozen reading: reangle needs T1 (T1 is trained anyway, A6.2).
2. **Two no-op combo rows (disclosure gap).** l1_combo_kfab_200_m0 and kfab_203_m0 have no surviving splice, so they equal the
   mirror rows of 200 and 203 in L1 and the originals in the T1 motion. Consequences: 8 of the combo family's 24 performability
   starts run on these unmodified rows (combo 22/24 is partly a mirror measurement), and in T1 they add duplicate original weight to
   the L1 group. The manifest's combo disclosure gives "4 of 13 survive" but does not name them. Recommend: name them in the
   disclosure, or drop them / report combo performability on spliced rows only. Not decided here.
3. **CONJ_TOL is the builder's** (1e-9 f64 from A6.5, 1e-5 f32 from G0-1); measured margins are 5 and 1 orders, and my probe shows
   the obs map itself is exact (0.0). Not a frozen threshold; disclose as chosen.
4. **Integration notes for M6/M9.** fb_mirror is not wired into fb_replica.py or the mjlab task (contract in M8b_NOTES 1). On GPU,
   MirrorPolicy.__call__ does bool(m.all()) / bool(m.any()), two host syncs per step; prefer the torch.where path unconditionally in
   the task. MirrorPolicy.set() needs a mask (conjugate(drv) with mask None cannot be set per env). The p 0.2 mirrored-strike swap
   (design 4.2) is still absent from AimOp (A6.2's "mirrored strikes" only bites once it exists).

## 5. Full suite

`pytest -q -p no:cacheprovider --basetemp=<scratchpad>\pytest-tmp tests` (no FB_REC): **390 passed, 3 skipped** in 1028 s, as the build (log `verify_m8b\suite.log`). It left a MUJOCO_LOG.TXT (13:16, the "empty prefix" warning, as in M8/M8b); deleted. No stray pytest folder.

## 6. Files (`build\verify_m8b\`)

run_all.sh (regen L1, LOCO, check), regen.log, regen_loco.log, regen\, regen_loco\, check.log, m8b_check_v.json, cmp_regen.py,
probe_combo.py, probe_conj.py, probe_fed.py, suite.log. Ledger: one row appended (60500-60595 re-run in role l1_perf, seed-free).
