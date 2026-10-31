"""M8 checks of library L1 and the T1 config, CPU part (M2V2_PLAN_v3 3 M8 with A6; gpu/export_fb_library.py, gpu/r1_track_spar_cfg.py,
gpu/fb_mirror.py):
  (1) manifest: every row's sources (and every frame's src_clip) in TRAIN_CLIPS (LOCO: kfab_200/201); sha256 of the outputs, of every
      source and of every row's q match; no banned name in the outputs (fb_seeds.scan); the T1 motion index = the npz rows but the
      mirror family, conj flags = mirror flags; the T1 CSV = track_motion rebuilt from the npz rows (conjugated rows un-mirrored);
      vmax = the p99 of the manifest's clip set (A6.3); combos (A6.1): every accepted splice passes the stated rule with its fade,
      which is the shortest passing one in the prior's frame grid. Run on L1 and on the LOCO library.
  (2) rejection stats per family (reported); re-angle: the apex frame of every accepted re-angle is the base's bit for bit, so the
      palm's apex forward distance equals the original's exactly; no palm frame of the window goes further forward (max excess <= 0);
      the re-detected strike's apex forward distance vs the base's (reported). Warp: forward excess of the resampled palm (reported).
  (3) T0 alone (scripts/eval_tracker.run: R1 at home, opponent 4 m away, 10 s, no aim; fb_opponent's TrackerDriver on
      model_13749.pt, float32 as deployed, fed each row's fb_l1 reference), 24 starts per family, ids 60500-60595 (role l1_perf;
      seed-free world, env seed 0 as eval_tracker fixes it); A6.2: a mirrored-base row is driven by T0 conjugated by the mirror
      (fb_mirror.conjugate), an original-base row by T0. Stay-up share per family and the falls within 1 s after a modified frame;
      reported. A family T0 keeps up in >= 90% of starts (T1-b's bar) does not need T1. The runner itself: the originals (72
      starts, seed-free, as M2 G0-3b) through the same path vs the eval's own Tracker (fb_m2_check._alone 'eval'). Reported: the
      mirror rows start for start against T0 on the originals (start k of 8 = original start 3k of 24); diagnostic (same ids
      re-run): the mirrored-base rows driven by plain T0 (M8's main result).
  (4) mirror: the mirrored R1 joint ranges equal the ranges within RANGE_TOL (asserted); FK of mirror(q) = the reflected FK of q
      (bodies and sites, every frame of the 3 clips); the eval's strike detector on a mirrored clip = the mirrored strikes.
  (5) conjugation exact (A6.2) on recorded rows: every control step of the M2 recording (fb_record, 16 eval bouts, opponent on
      its tracker, aim standoff+armwarp_vz): the whole MjData state (both R1s), the reference, the target and the driver's facing /
      steering / last action mirrored, the mirrored world's kinematics recomputed by MuJoCo; then the driver's obs in the mirrored
      world = M_o(obs) and the conjugated tracker's action there = M_a(T0's action), in float64 (<= CONJ_TOL f64) and float32
      (<= CONJ_TOL f32), with the bit-exact share reported; the obs of the unmirrored world vs the recorded obs64 (reported).
  extra: the shown reference of the originals in the T1 motion vs fb_motion's (T0's), and every row's fed reference rebuilt from
      the CSV = the npz's (bit for bit); clean masks: export_fb_library's replay through scenario.clean_starts on the originals =
      fb_motion's masks exactly.
Run in the repo: python scripts/fb_m8_check.py --rec <fb_rec_60000-60015.npz> [--out m8_check.json] [--workers 2] [--skip-perf] [--skip-clean]"""
import os, sys, json, time, argparse
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_k, "1")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import numpy as np, torch, mujoco
import export_fb_library as L, export_fb_motion as EM, fb_seeds, fb_mirror as FM
from r1gym.env import TRAIN_CLIPS
from r1gym.opponent import _sub

OUT = L.OUT; FPS = L.FPS; STARTS, SECONDS, PERF_BAR, ATTR_S = 24, 10.0, 0.9, 1.0
CONJ_TOL = dict(f64=1e-9, f32=1e-5)   # the diagnostic 1e-9 (A6.5) and G0-1's float32 action tolerance
IDS = {f: list(range(60500 + 24 * i, 60524 + 24 * i)) for i, f in enumerate(L.FAMILIES)}   # bookkeeping ids, role l1_perf

def load(out=OUT, loco=False):
    f_npz, _, f_ix, f_man = L.names_for(loco)
    with np.load(Path(out) / f_npz, allow_pickle=False) as z: A = {k: z[k] for k in z.files}
    return A, json.load(open(Path(out) / f_man, encoding="utf-8")), json.load(open(Path(out) / f_ix, encoding="utf-8"))

def lib_rows(A):
    """The npz rows as export_fb_library's row tuples (name, family, host, mirror, draw, q)."""
    return [(n, f, h, int(m), int(d), A["q"][row(A, i)]) for i, (n, f, h, m, d) in enumerate(zip(A["names"].tolist(), A["family"].tolist(), A["host"].tolist(), A["mirror"], A["draw"]))]

def row(A, i): r0, T = int(A["row0"][i]), int(A["frames"][i]); return slice(r0, r0 + T)

# ---- (1)
def check_combos(A, man, vmax):
    """A6.1 recomputed from the bases: an accepted splice passes the stated rule at its fade, which is in the prior's frame grid,
    and every shorter fade fails it or does not fit (seam, guard); a rejected splice fails it at every fade that fits; the row's q
    is its accepted splices applied (last first) to its base."""
    Q = L.base_motion(); B, _ = L.bases(Q, tuple(man["clips"])); bad, n = [], 0; pad = int(round(L.PRIOR["combo_pad_s"] * FPS))
    for i, r in enumerate(man["rows"]):
        if r["family"] != "combo": continue
        q = B[(r["host"], r["mirror"])].copy(); sc = L.PRIOR["combo_source"][r["host"]]; hs = [s for s in L.strikes(q) if s.hand]
        for x in sorted([x for x in r["mods"] if x["status"] in ("accept", "reject")], key=lambda x: -x["host_end"]):
            sq = B[(sc, x["src_mirror"])]; e = x["host_end"]; s0 = [s for s in hs if s.end == e][0]
            t = [s for s in L.strikes(sq) if s.hand and (s.onset, s.apex, s.end, s.limb) == (x["src_onset"], x["src_apex"], x["src_end"], x["src_limb"])][0]
            fits = lambda f: t.onset - pad - f >= 0 and t.end + pad + f < len(sq) and e + 1 + 2 * f <= len(q) - 1 and not any(e < h.onset <= e + 1 + 2 * f or h.onset <= e < h.end for h in hs if h is not s0)
            def rule(f): new, _, span, hw, seg = L.splice(q, np.zeros((len(q), 3)), e, f, sq, np.zeros((len(sq), 3)), t); return new, L.speed_over(span, [hw, seg], vmax)
            if x["status"] == "accept":
                n += 1; f = x["fade"]; new, over = rule(f); ok = f in L.FADES and not over and all(not fits(g) or rule(g)[1] for g in L.FADES if g < f); q = new
            else: ok = bool(x["fades"]) and all(rule(g)[1] for g in L.FADES if fits(g)) and x["fades"] == [g for g in L.FADES if fits(g)]
            if not ok: bad.append((r["name"], e, x["status"]))
        if not np.array_equal(q, A["q"][row(A, i)]): bad.append((r["name"], "q"))
    return dict(accepted=n, tried=sum(x["status"] in ("accept", "reject") for r in man["rows"] if r["family"] == "combo" for x in r["mods"]),
                fades=sorted(x["fade"] for r in man["rows"] if r["family"] == "combo" for x in r["mods"] if x["status"] == "accept"), bad=bad, ok=not bad)

def check_manifest(out=OUT, loco=False):
    A, man, ix = load(out, loco); tr = set(L.LOCO_CLIPS if loco else TRAIN_CLIPS); clips = A["clips"].tolist(); f_npz, f_csv, f_ix, f_man = L.names_for(loco)
    srcs = sorted({c for r in man["rows"] for c in r["sources"]}); frame_clips = sorted({clips[int(i)] for i in np.unique(A["src_clip"])})
    stale = [s for s, h in man["sources"].items() if EM.sha256(ROOT / s) != h]; outs = {k: EM.sha256(Path(out) / k) == h for k, h in man["outputs"].items()}
    rowsha = [r["name"] for i, r in enumerate(man["rows"]) if L.hashlib.sha256(np.ascontiguousarray(A["q"][row(A, i)]).tobytes()).hexdigest() != r["q_sha256"]]
    names, fam = A["names"].tolist(), A["family"].tolist(); t1 = [n for n, f in zip(names, fam) if f != "mirror"]
    idx_ok = [n for n in ix["rows"] if ix["rows"][n]["group"] != "original"] == t1 and [n for n in ix["rows"] if ix["rows"][n]["group"] == "original"] == list(man["clips"]) and \
        all(ix["rows"][n]["frames"] == int(A["frames"][i]) and ix["rows"][n]["conj"] == bool(A["mirror"][i]) for i, n in enumerate(names) if n in ix["rows"])
    conj_ok = np.array_equal(A["conj"], A["mirror"]) and all(r["conj"] == bool(r["mirror"]) for r in man["rows"])
    seq, _, _ = L.track_motion(L.base_motion(), lib_rows(A), tuple(man["clips"])); csv = np.loadtxt(Path(out) / f_csv, delimiter=",")
    motion_ok = bool(np.array_equal(np.char.mod("%.6f", np.concatenate([seq[:, :3], seq[:, [4, 5, 6, 3]], seq[:, 7:]], 1)).astype(float), csv))
    _, vmax = L.bases(L.base_motion(), tuple(man["clips"])); vmax_ok = all(np.allclose(man["vmax"][str(m)], vmax[m], rtol=0, atol=1e-8) for m in (0, 1))
    combos = check_combos(A, man, np.maximum(vmax[0], vmax[1]))
    banned = fb_seeds.scan([Path(out) / f for f in (f_npz, f_csv, f_ix, f_man)] + [ROOT / "gpu" / "export_fb_library.py", ROOT / "gpu" / "r1_track_spar_cfg.py", ROOT / "gpu" / "fb_mirror.py", Path(__file__)])
    res = dict(rows=len(names), clips=man["clips"], row_sources=srcs, frame_sources=frame_clips, outputs_sha=outs, stale_sources=stale, row_sha_mismatch=rowsha, index_ok=bool(idx_ok),
               conj_ok=bool(conj_ok), t1_motion_rows=len(ix["rows"]), t1_motion_conjugated=sum(x["conj"] for x in ix["rows"].values()), t1_csv_equals_rebuild=motion_ok, vmax_per_clip_set=vmax_ok,
               combos=combos, banned=banned, per_family={f: int(sum(x == f for x in fam)) for f in L.FAMILIES}, seconds=round(len(A["q"]) / FPS, 2),
               disclosures=man["disclosures"])
    res["ok"] = set(man["clips"]) == tr and set(srcs) <= tr and set(frame_clips) <= tr and all(outs.values()) and len(outs) == 3 and not stale and not rowsha and idx_ok \
        and conj_ok and motion_ok and vmax_ok and combos["ok"] and not banned and bool(man["disclosures"])
    return res

# ---- (2)
def check_reach(out=OUT, loco=False):
    A, man, _ = load(out, loco); Q = L.base_motion(); B, _ = L.bases(Q, tuple(man["clips"])); k = L.Kin(); names = A["names"].tolist(); n = int(round((L.PRIOR["reangle_s"] + L.PRIOR["reangle_ramp_s"]) * FPS))
    apex_bit, excess, redet, warp_ex = [], [], [], []
    for i, r in enumerate(man["rows"]):
        q = A["q"][row(A, i)]; base = B[(r["host"], r["mirror"])]
        if r["family"] == "reangle":
            det = [s for s in L.strikes(q) if s.hand]; bdet = {(s.limb, s.apex): s for s in L.strikes(base) if s.hand}
            for x in r["mods"]:
                if x["status"] != "accept": continue
                ap, limb = x["apex"], x["limb"]; apex_bit.append(bool(np.array_equal(q[ap], base[ap])))
                fa = -k.palm_at(base[ap], limb)[0]
                excess.append(max(-k.palm_at(q[f], limb)[0] for f in range(ap - n + 1, ap)) - fa)
                m = [s for s in det if s.limb == limb and abs(s.apex - ap) <= 2]; b = bdet[(limb, ap)]
                redet.append(dict(row=r["name"], limb=limb, apex=ap, type0=b.type, found=bool(m), type1=m[0].type if m else None,
                                  apex_shift=(m[0].apex - ap) if m else None, fwd_diff_m=round(m[0].apex_m[0] - b.apex_m[0], 4) if m else None))
        if r["family"] == "warp":
            src = A["src_frame"][row(A, i)]
            for x in r["mods"]:
                if x["status"] != "accept": continue
                f = np.where((src >= x["onset"] - 10) & (src <= x["end"] + 5))[0]; fb = max(-k.palm_at(base[j], x["limb"])[0] for j in range(x["onset"] - 10, x["end"] + 6))
                warp_ex.append(max(-k.palm_at(q[j], x["limb"])[0] for j in f) - fb)   # vs the base's own furthest over the window
    found = [x for x in redet if x["found"]]
    res = dict(stats=man["reject_stats"], reangle_accepted=len(apex_bit), apex_frame_bit_equal=int(sum(apex_bit)), window_max_excess_m=float(max(excess)) if excess else None,
               redetected=len(found), redetected_same_apex_fwd=int(sum(x["fwd_diff_m"] == 0 for x in found)), redetect_fwd_diff_max_m=max((abs(x["fwd_diff_m"]) for x in found), default=None),
               redetect_type_changes={f"{x['type0']}->{x['type1']}": sum(1 for y in found if (y["type0"], y["type1"]) == (x["type0"], x["type1"])) for x in found if x["type0"] != x["type1"]},
               not_redetected=[x for x in redet if not x["found"]], warp_window_max_excess_m=float(max(warp_ex)) if warp_ex else None, warp_windows=len(warp_ex))
    res["ok"] = bool(apex_bit) and all(apex_bit) and res["window_max_excess_m"] <= L.REJECT["reach_m"]; return res

# ---- (4)
def check_mirror():
    sym = L.range_symmetry(); worst = max(sym, key=sym.get); r = _sub("r1"); d0, d1 = mujoco.MjData(r), mujoco.MjData(r)
    sw = lambda s: s.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
    bp = np.array([r.body(sw(r.body(b).name)).id for b in range(r.nbody)]); sp = np.array([r.site(sw(r.site(s).name)).id for s in range(r.nsite)])
    Q = L.base_motion(); eb = es = eq = 0.0; R = np.diag([1.0, -1.0, 1.0])
    for c in L.CLIPS:
        qm = L.mirror(Q[c], clip=False); eq = max(eq, float(np.abs(L.mirror(qm, clip=False) - Q[c]).max()))
        for t in range(len(qm)):
            d0.qpos[:] = Q[c][t]; d1.qpos[:] = qm[t]; mujoco.mj_kinematics(r, d0); mujoco.mj_kinematics(r, d1)
            eb = max(eb, float(np.abs(d1.xpos[bp] - d0.xpos @ R).max())); es = max(es, float(np.abs(d1.site_xpos[sp] - d0.site_xpos @ R).max()))
    clipped = {c: float(np.abs(L.mirror(Q[c]) - L.mirror(Q[c], clip=False)).max()) for c in L.CLIPS}
    det = {}
    for c in L.CLIPS:
        a = L.strikes(Q[c]); b = L.strikes(L.mirror(Q[c])); lm = {"l_hand": "r_hand", "r_hand": "l_hand", "l_foot": "r_foot", "r_foot": "l_foot"}
        key = lambda s: (s.onset, s.apex, s.end)
        pa = sorted([(key(s), lm[s.limb], s.type, s.band, (s.apex_m[0], -s.apex_m[1], s.apex_m[2])) for s in a]); pb = sorted([(key(s), s.limb, s.type, s.band, s.apex_m) for s in b])
        same = len(pa) == len(pb) and all(x[:4] == y[:4] and np.allclose(x[4], y[4], atol=1.5e-3) for x, y in zip(pa, pb))
        det[c] = dict(strikes=len(a), mirrored=len(b), equal=bool(same))
    lib = [s for s in L.S.load(split="train")]; same_lib = all(sorted((s.onset, s.apex, s.end, s.limb) for s in L.strikes(Q[c])) == sorted((s.onset, s.apex, s.end, s.limb) for s in lib if s.clip == c) for c in L.CLIPS)
    res = dict(range_max_rad=sym[worst], range_worst_joint=worst, range_tol=L.RANGE_TOL, range_per_joint={k: v for k, v in sym.items() if v > 0}, fk_body_m=eb, fk_site_m=es,
               mirror_involution=eq, clip_change_rad=clipped, detect=det, detect_on_originals_equals_strikes_json=bool(same_lib))
    res["ok"] = sym[worst] <= L.RANGE_TOL and eb <= 1e-9 and es <= 1e-9 and eq == 0.0 and all(v["equal"] for v in det.values()) and same_lib
    return res

# ---- (3)
def lib_motion(name, q0, ref):
    """fb_opponent.Motion of one row (index 0): q0 for placement, `ref` (joint pos, joint vel, anchor pos, anchor quat) shown."""
    import fb_opponent as FO
    M = FO.Motion.__new__(FO.Motion); t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), dtype=dt)
    M.clips, M.frames, M.row0, M.q = (name,), t([len(q0)], torch.long), t([0], torch.long), t(q0, torch.float64)
    M.ref_q, M.ref_dq, M.ref_apos, M.ref_aquat = (t(x) for x in ref); M.clean = M.distances = None; M.strikes = {}; M.device = "cpu"; return M

_W = {}
def _alone(args):
    """(name, t0, src) -> fall time or None: eval_tracker.run with the row's replay and the T0 driver (float32). src: 'l1' a library
    row driven by T0, 'l1conj' a library row driven by T0 conjugated by the mirror (fb_mirror.conjugate), 'orig' a train clip
    (fb_motion's reference)."""
    name, t0, src = args; torch.set_num_threads(1)
    import eval_tracker as ET, fb_opponent as FO
    from r1gym import opponent as O
    from r1gym.model import build_spec, load_cfg
    if "m" not in _W:
        cfg = load_cfg(); assert cfg.get("opponent_aim", "none") == "none" and cfg["opponent_mode"] == "policy"; _W.update(cfg=cfg, m=build_spec(opponent=True, cfg=cfg).compile())
    m, cfg = _W["m"], _W["cfg"]
    if src in ("l1", "l1conj"):
        A, _, _ = load(); i = A["names"].tolist().index(name); sl = row(A, i); q0 = A["q"][sl]; ref = tuple(A[k][sl] for k in ("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"))
    else:
        with np.load(L.MOTION, allow_pickle=False) as z: k = z["clips"].tolist().index(name); sl = slice(int(z["row0"][k]), int(z["row0"][k]) + int(z["frames"][k])); q0 = z["q"][sl]; ref = tuple(z[x][sl] for x in ("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat"))
    M = lib_motion(name, q0, ref)
    def make(m_, clip_, distance, cfg):
        assert clip_ == name and m_ is m
        r = L.replay(m_, name, q0, distance); tr = FO.MjTracker.__new__(FO.MjTracker); tr.replay, tr.k, tr.drv = r, 0, FO.TrackerDriver(1, torch.float32, motion=M)
        if src == "l1conj": FM.conjugate(tr.drv)
        return tr.install().replay
    ET.FightReplay = make
    try: fell, j, z = ET.run(m, cfg, name, t0, SECONDS)
    finally: ET.FightReplay = O.FightReplay
    return fell

def _eval(args):
    import fb_m2_check as M2
    return M2._alone(args)["fell"]

def modified(A, man, i):
    """Row frames that differ from the host base (a mirror row: every frame)."""
    r = man["rows"][i]; q = A["q"][row(A, i)]
    if r["family"] == "mirror": return np.ones(len(q), bool)
    Q = L.base_motion(); base = L.mirror(Q[r["host"]]) if r["mirror"] else Q[r["host"]]
    sf, sc, sm = A["src_frame"][row(A, i)], A["src_clip"][row(A, i)], A["src_mirror"][row(A, i)]; ok = (sf == np.round(sf)) & (sc == L.CLIPS.index(r["host"])) & (sm == r["mirror"])
    j = np.clip(np.round(sf).astype(int), 0, len(base) - 1); return ~(ok & np.all(q == base[j], axis=1))

def same_fall(a, b): return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 0.1 + 1e-9)

def performability(workers=2, out=OUT):
    A, man, _ = load(out); names = A["names"].tolist(); fam = A["family"].tolist(); jobs, meta = [], []
    for f in L.FAMILIES:
        ids = fb_seeds.check(IDS[f], "l1_perf", L.CLIPS); R = [n for n, x in zip(names, fam) if x == f]; per = STARTS // len(R); assert per * len(R) == STARTS, (f, len(R))
        for s, sid in enumerate(ids):
            n = R[s % len(R)]; T = int(A["frames"][names.index(n)]) / FPS; t0 = (s // len(R)) * T / per
            jobs.append((n, t0, "l1conj" if int(A["conj"][names.index(n)]) else "l1")); meta.append((f, sid, n, t0))   # A6.2: conjugated on mirrored-base rows
    ojobs = [(c, s * L.base_motion()[c].shape[0] / FPS / STARTS, "orig") for c in L.CLIPS for s in range(STARTS)]
    ejobs = [(c, s, STARTS, "eval") for c in L.CLIPS for s in range(STARTS)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        pj = [(n, t0, "l1") for n, t0, src in jobs if src == "l1conj"]   # diagnostic: the mirrored-base rows driven by plain T0, same starts
        fl = pool.map(_alone, jobs + ojobs + pj, chunksize=1); ev = pool.map(_eval, ejobs, chunksize=1)
    fl, fo, fp = fl[:len(jobs)], fl[len(jobs):len(jobs) + len(ojobs)], fl[len(jobs) + len(ojobs):]; res = {}; mods = {n: modified(A, man, i) for i, n in enumerate(names)}
    pres = {(n, t0): x for (n, t0, _), x in zip(pj, fp)}; orig = {(c, round(t0, 9)): x for (c, t0, _), x in zip(ojobs, fo)}
    for f in L.FAMILIES:
        rr = [(sid, n, t0, x) for (ff, sid, n, t0), x in zip(meta, fl) if ff == f]; up = sum(x is None for *_, x in rr); att = 0
        for sid, n, t0, x in rr:
            if x is None: continue
            T = len(mods[n]); fr = int(round((t0 + x) * FPS)); att += bool(mods[n][[(fr - j) % T for j in range(int(ATTR_S * FPS) + 1)]].any())
        res[f] = dict(starts=len(rr), ids=[rr[0][0], rr[-1][0]], stays_up=up, share=round(up / len(rr), 4), falls_after_modified=att, driver="T0 on original-base rows, T0 conjugated on mirrored-base rows",
                      falls=[dict(id=sid, row=n, t0=round(t0, 3), fell=x, conj=(n, t0) in pres) for sid, n, t0, x in rr if x is not None], needs_T1=up / len(rr) < PERF_BAR)
        mm = [(n, t0, x, pres[(n, t0)]) for sid, n, t0, x in rr if (n, t0) in pres]; ro = [(n, t0, x) for sid, n, t0, x in rr if (n, t0) not in pres]
        res[f]["by_base"] = dict(original_base_T0=dict(starts=len(ro), stays_up=sum(x is None for *_, x in ro)), mirrored_base_T0_conjugated=dict(starts=len(mm), stays_up=sum(x is None for _, _, x, _ in mm)),
                                 mirrored_base_T0_plain_diagnostic=dict(starts=len(mm), stays_up=sum(y is None for *_, y in mm)))
    mr = [(n, t0, x) for (ff, sid, n, t0), x in zip(meta, fl) if ff == "mirror"]
    pair = [(n, t0, x, orig.get((man["rows"][names.index(n)]["host"], round(t0, 9)), "missing")) for n, t0, x in mr]
    res["mirror_vs_originals"] = dict(pairs=len(pair), same=sum(o != "missing" and same_fall(x, o) for *_, x, o in pair), differ=[dict(row=n, t0=round(t0, 3), conj=x, original=o) for n, t0, x, o in pair if o == "missing" or not same_fall(x, o)])
    bad = [dict(clip=c, start=s % STARTS, eval=e, runner=o) for s, (c, (o, e)) in enumerate(zip([j[0] for j in ojobs], zip(fo, ev))) if not same_fall(o, e)]
    res["originals"] = dict(starts=len(fo), runner_stays_up=sum(x is None for x in fo), eval_stays_up=sum(x is None for x in ev), mismatches=bad, ok=not bad)
    res["bar"] = PERF_BAR; res["families_needing_T1"] = [f for f in L.FAMILIES if res[f]["needs_T1"]]; res["ok"] = res["originals"]["ok"]; return res

# ---- (5)
def check_conjugation(rec, dtypes=(torch.float64, torch.float32)):
    """Obs and action conjugation on the recorded eval rows (module docstring (5))."""
    import fb_opponent as FO
    from r1gym.model import build_spec, load_cfg
    with np.load(rec, allow_pickle=False) as z: R = {k: z[k] for k in z.files}
    eps = json.loads(str(R["meta"]))["episodes"]; N = len(R["t"]); m = build_spec(opponent=True, cfg=load_cfg()).compile(); d = mujoco.MjData(m)
    fq = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == 0]; fv = [int(m.jnt_dofadr[j]) for j in range(m.njnt) if m.jnt_type[j] == 0]
    assert m.nq == 31 * len(fq) and m.nv == 30 * len(fv) and len(fq) == 2
    qa, va = int(m.jnt_qposadr[m.joint("opp_floating_base_joint").id]), int(m.jnt_dofadr[m.joint("opp_floating_base_joint").id])
    W = dict(qpos=R["qpos"], qvel=R["qvel"]); M = dict(qpos=R["qpos"].copy(), qvel=R["qvel"].copy())
    for a, v in zip(fq, fv): M["qpos"][:, a:a + 31] = FM.mirror_qpos(R["qpos"][:, a:a + 31]); M["qvel"][:, v:v + 30] = FM.mirror_qvel(R["qvel"][:, v:v + 30])
    tor, pel, dpel = m.body("opp_torso_link").id, m.body("opp_pelvis").id, m.body("pelvis").id
    for X in (W, M):
        X.update(tp=np.zeros((N, 3)), tq=np.zeros((N, 4)), pp=np.zeros((N, 3)), dp=np.zeros((N, 3)))
        for i in range(N):
            d.qpos[:] = X["qpos"][i]; mujoco.mj_kinematics(m, d); X["tp"][i], X["tq"][i], X["pp"][i], X["dp"][i] = d.xpos[tor], d.xquat[tor], d.xpos[pel], d.xpos[dpel]
    e = lambda a, b: float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
    V, Q4 = np.array([1.0, -1.0, 1.0]), np.array([1.0, -1.0, 1.0, -1.0])
    res = dict(rows=N, bouts=len(eps), fk_unmirrored_vs_recorded=max(e(W["tp"], R["opp_torso_xpos"]), e(W["tq"], R["opp_torso_xquat"]), e(W["pp"], R["opp_pel_xpos"])),
               fk_mirror=max(e(M["tp"], W["tp"] * V), e(M["tq"], W["tq"] * Q4), e(M["pp"], W["pp"] * V), e(M["dp"], W["dp"] * V)), target_is_defender_pelvis=e(R["target"], W["dp"][:, :2]))
    dist0 = np.array([eps[b]["distance"] for b in R["bout"]]); spot = np.where(np.isnan(R["spot_pre"]), np.stack([dist0, 0 * dist0], 1), R["spot_pre"])
    yset, aset = np.isfinite(R["yaw_pre"][:, 0]), np.isfinite(R["axis_pre"]).all(1)
    S = dict(W=dict(target=R["target"], pel=W["pp"][:, :2], tp=W["tp"], tq=W["tq"], w=R["qvel"][:, va + 3:va + 6], q=R["qpos"][:, qa + 7:qa + 31], dq=R["qvel"][:, va + 6:va + 30],
                    rq=R["obs64"][:, :24], rdq=R["obs64"][:, 24:48], apos=R["apos"], aori=R["aori"], yaw=np.nan_to_num(R["yaw_pre"][:, 0]), spot=spot, axis=np.nan_to_num(R["axis_pre"]), last=R["last_pre"]))
    s0 = S["W"]; jm = FM.mirror_joints
    S["M"] = dict(target=s0["target"] * V[:2], pel=M["pp"][:, :2], tp=M["tp"], tq=M["tq"], w=M["qvel"][:, va + 3:va + 6], q=M["qpos"][:, qa + 7:qa + 31], dq=M["qvel"][:, va + 6:va + 30],
                  rq=jm(s0["rq"]), rdq=jm(s0["rdq"]), apos=s0["apos"] * V, aori=s0["aori"] * Q4, yaw=-s0["yaw"], spot=s0["spot"] * V[:2], axis=s0["axis"] * V[:2], last=jm(s0["last"]))
    res["world_mirror"] = dict(gyro=e(S["M"]["w"], R["qvel"][:, va + 3:va + 6] * -V), joints=max(e(S["M"]["q"], jm(s0["q"])), e(S["M"]["dq"], jm(s0["dq"]))), target_vs_mirrored_defender=e(S["M"]["target"], M["dp"][:, :2]))
    for dt in dtypes:
        out = {}
        for k in ("W", "M"):
            x = {a: torch.as_tensor(np.ascontiguousarray(b), dtype=torch.float64).to(dt) for a, b in S[k].items()}; drv = FO.TrackerDriver(N, dt)
            if k == "M": FM.conjugate(drv)
            drv.yaw, drv.spot, drv.axis, drv.last = x["yaw"], x["spot"], x["axis"], x["last"]
            drv.yaw_set[:] = torch.as_tensor(yset); drv.axis_set[:] = torch.as_tensor(aset); drv.dist = torch.as_tensor(R["dist"][:, 0]).to(dt)
            drv.face(x["target"], x["pel"]); drv.steer(x["target"], drv.dist)
            o = drv.observation(x["rq"], x["rdq"], x["apos"], x["aori"], x["tp"], x["tq"], x["w"], x["q"], x["dq"]); a = drv.mlp(o)
            out[k] = dict(obs=o, act=a, tgt=drv.default + drv.scale * a, yaw=drv.yaw, spot=drv.spot, axis=drv.axis)
        w, mm = out["W"], out["M"]; ex = lambda a, b: float((a.double() - b.double()).abs().max())
        r = dict(obs=ex(mm["obs"], FM.mirror_obs(w["obs"])), action=ex(mm["act"], FM.mirror_act(w["act"])), joint_targets=ex(mm["tgt"], FM.mirror_joints(w["tgt"])),
                 obs_bit_exact_rows=int((mm["obs"] == FM.mirror_obs(w["obs"])).all(1).sum()), action_bit_exact_rows=int((mm["act"] == FM.mirror_act(w["act"])).all(1).sum()),
                 yaw=float(FO.wrap(mm["yaw"].double() + w["yaw"].double()).abs().max()), spot=ex(mm["spot"], w["spot"] * torch.tensor([1.0, -1.0], dtype=dt)), axis=ex(mm["axis"], w["axis"] * torch.tensor([1.0, -1.0], dtype=dt)),
                 unmirrored_obs_vs_recorded_obs64=e(w["obs"].double().numpy(), R["obs64"]), unmirrored_action_vs_recorded=e(w["act"].double().numpy(), R["action"]))
        tol = CONJ_TOL["f64" if dt == torch.float64 else "f32"]; r["tol"] = tol; r["ok"] = max(r["obs"], r["action"], r["joint_targets"]) <= tol
        res[str(dt).split(".")[-1]] = r
    res["ok"] = all(res[str(dt).split(".")[-1]]["ok"] for dt in dtypes) and res["fk_mirror"] <= CONJ_TOL["f64"]; return res

# ---- extra
def check_shown(out=OUT, loco=False):
    """The originals' reference in the T1 motion (rebuilt from the CSV) vs fb_motion's (the tracker's own, M1); every row's fed
    reference (export_fb_library.fed on the CSV) vs the npz's."""
    A, man, ix = load(out, loco); Q = L.base_motion(); seq = np.loadtxt(Path(out) / L.names_for(loco)[1], delimiter=","); seq = np.concatenate([seq[:, :3], seq[:, [6, 3, 4, 5]], seq[:, 7:]], 1)
    fed = L.fed(seq, ix["rows"], ix["blend_frames"], lib_rows(A)); keys = ("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat")
    fed_bad = [n for i, n in enumerate(A["names"].tolist()) if not all(np.array_equal(fed[n][j], A[k][row(A, i)]) for j, k in enumerate(keys))]
    ref = L.shown(seq, ix["rows"], ix["blend_frames"], man["clips"]); z = np.load(L.MOTION); e = {}
    for c in man["clips"]:
        k = z["clips"].tolist().index(c)
        sl = slice(int(z["row0"][k]), int(z["row0"][k]) + int(z["frames"][k]))
        for j, key in enumerate(("ref_joint_pos", "ref_joint_vel", "ref_anchor_pos", "ref_anchor_quat")):
            a, b = ref[c][j].astype(np.float64), z[key][sl].astype(np.float64); d = np.abs(a - b).max(1)
            if key == "ref_anchor_quat": d = np.minimum(d, np.abs(a + b).max(1))   # q and -q: the same rotation (the tracker reads its matrix)
            e[key] = max(e.get(key, 0.0), float(d.max()))
    csv_q = max(float(np.abs(seq[ix["rows"][c]["start_frame"] + ix["blend_frames"]:ix["rows"][c]["start_frame"] + len(Q[c])] - Q[c][ix["blend_frames"]:]).max()) for c in man["clips"])
    return dict(max=e, csv_rounding_m=csv_q, note="report only: two float32 csv_to_npz time grids of different motion lengths (T2 C4 measured the same scale)",
                fed_rows=len(fed), fed_mismatch=fed_bad, ok=not fed_bad)

def _clean_one(c):
    return L.clean_masks({c: L.base_motion()[c]})[0]

def check_clean(workers=2):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool: got = pool.map(_clean_one, list(L.CLIPS), chunksize=1)
    z = np.load(L.MOTION); mis = int(sum(int((g != z["clean"][k]).sum()) for k, g in enumerate(got))); return dict(compared=int(z["clean"].size), mismatches=mis, ok=mis == 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--skip-clean", action="store_true"); ap.add_argument("--rec", required=True, help="the M2 recording fb_rec_60000-60015.npz (scripts/fb_record.py)")
    a = ap.parse_args(); assert a.workers <= 2; t = time.time(); res = {}
    for k, f in (("(1) manifest", check_manifest), ("(1) manifest LOCO", lambda: check_manifest(loco=True)), ("(2) rejection and reach", check_reach),
                 ("(2) rejection and reach LOCO", lambda: check_reach(loco=True)), ("(4) mirror", check_mirror), ("(5) conjugation", lambda: check_conjugation(a.rec)), ("shown reference", check_shown),
                 ("shown reference LOCO", lambda: check_shown(loco=True))):
        res[k] = f(); print(k, json.dumps({x: y for x, y in res[k].items() if x not in ("stats", "not_redetected")}, default=str)[:3000], flush=True)
    if not a.skip_clean: res["clean identity"] = check_clean(a.workers); print("clean identity", json.dumps(res["clean identity"]), flush=True)
    if not a.skip_perf: res["(3) performability"] = performability(a.workers); print("(3)", json.dumps({f: {x: y for x, y in v.items() if x != "falls"} if isinstance(v, dict) else v for f, v in res["(3) performability"].items()}), flush=True)
    res["ok"] = {k: v["ok"] for k, v in res.items() if isinstance(v, dict) and "ok" in v}; res["wall_s"] = round(time.time() - t, 1); print(json.dumps(res["ok"]))
    if a.out: json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    sys.exit(0 if all(res["ok"].values()) else 1)

if __name__ == "__main__": main()
