"""Library L1 and the T1 config (M2V2_PLAN_v3 M8 with A6; gpu/export_fb_library.py, gpu/r1_track_spar_cfg.py, scripts/fb_m8_check.py):
the mirror (ranges, FK, strikes), the manifests (L1 and LOCO), no added reach on re-angled strikes, the warp time map, the combo speed
rule (A6.1) and bookkeeping, vmax per clip set (A6.3), regeneration without opening a held-out file, the T1 motion (conjugated rows
un-mirrored, A6.2) and its group-mixed sampling, the performability runner (plain and conjugated) and its ids. The 24-start runs per
family, the clean masks and the conjugation proof on the recorded rows run in scripts/fb_m8_check.py."""
import sys, json, builtins, re
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import export_fb_library as L, r1_track_spar_cfg as C, fb_m8_check as K, fb_seeds

def test_mirror_ranges_fk_and_strikes():
    r = K.check_mirror(); assert r["ok"], {k: v for k, v in r.items() if k != "range_per_joint"}
    assert r["range_max_rad"] <= L.RANGE_TOL and r["fk_body_m"] <= 1e-9 and r["mirror_involution"] == 0.0
    p, s = L.mirror_map(); assert sorted(p) == list(range(24)) and (p[p] == np.arange(24)).all() and s[12] == -1 and s[13] == -1   # waist roll, yaw

def test_mirror_quaternion_is_the_reflection():
    q = L.base_motion()["kfab_200"][:50]; m = L.mirror(q, clip=False)
    import mujoco
    for a, b in zip(q, m):
        Ra, Rb = np.zeros(9), np.zeros(9); mujoco.mju_quat2Mat(Ra, a[3:7]); mujoco.mju_quat2Mat(Rb, b[3:7]); M = np.diag([1.0, -1.0, 1.0])
        assert np.allclose(Rb.reshape(3, 3), M @ Ra.reshape(3, 3) @ M, atol=1e-12)

def test_manifest_sources_are_train_only_and_hashes_match():
    r = K.check_manifest(); assert r["ok"], r
    assert r["per_family"] == dict(mirror=3, warp=12, reangle=12, combo=6) and 240 <= r["seconds"] <= 300   # about 4-5 min (design 9.1)
    assert r["t1_motion_rows"] == 3 + 30 and r["t1_motion_conjugated"] == 15 and r["combos"]["ok"] and r["combos"]["accepted"] >= 1
    r = K.check_manifest(loco=True); assert r["ok"], r
    assert r["clips"] == ["kfab_200", "kfab_201"] and r["row_sources"] == r["frame_sources"] == ["kfab_200", "kfab_201"] and r["per_family"] == dict(mirror=2, warp=8, reangle=8, combo=2)

def test_reangle_adds_no_reach():
    for loco in (False, True):
        r = K.check_reach(loco=loco); assert r["ok"] and r["apex_frame_bit_equal"] == r["reangle_accepted"] > 0 and r["window_max_excess_m"] <= 0.0, {k: v for k, v in r.items() if k != "stats"}

def test_warp_map_whole_frames_and_identity():
    s, n, u = L.warp_map(10, 20, 30, 35, 1.0); assert n == 25 and abs(s - 1.0) < 1e-6 and np.allclose(u, np.arange(10, 35), atol=1e-6)
    for sp in (0.7, 0.85, 1.15, 1.3):
        s, n, u = L.warp_map(10, 20, 30, 35, sp); assert u[0] == 10 and np.all(np.diff(u) > 0) and u[-1] < 35 and abs(s - sp) < 0.05
        assert (n > 25) == (sp < 1)   # slower: more frames

def test_warp_rows_keep_the_base_outside_their_windows():
    A, man, _ = K.load(); Q = L.base_motion()
    for i, r in enumerate(man["rows"]):
        if r["family"] != "warp": continue
        q = A["q"][K.row(A, i)]; sf = A["src_frame"][K.row(A, i)]; base = L.mirror(Q[r["host"]]) if r["mirror"] else Q[r["host"]]
        acc = [x for x in r["mods"] if x["status"] == "accept"]; assert len(q) - len(base) == sum(x["frames_out"] - x["frames_in"] for x in acc)
        out = np.ones(len(sf), bool)
        for x in acc: out &= ~((sf >= x["onset"] - 10) & (sf < x["end"] + 5))   # [onset - 0.2 s, end + 0.1 s): the resampled window
        assert np.array_equal(q[out], base[sf[out].astype(int)]) and np.all(sf[out] == np.round(sf[out])) and np.all(np.diff(sf) > 0)

def test_combo_rows_splice_the_paired_train_clip():
    A, man, _ = K.load()
    for r in man["rows"]:
        if r["family"] != "combo": continue
        other = set(r["sources"]) - {r["host"]}; acc = [x for x in r["mods"] if x["status"] == "accept"]
        assert other == ({L.PRIOR["combo_source"][r["host"]]} if acc else set()) and len(acc) <= L.PRIOR["combo_max"]
    a = np.arange(10.0)[:, None] * [1, 1, 1]; b = 100 + np.arange(10.0)[:, None] * [1, 1, 1]; o = L.blend_aux(a, b, 4)
    assert len(o) == 16 and (o[6:8] == a[6:8]).all() and (o[8:10] == b[2:4]).all()
    assert L.FADES == tuple(range(8, 18)) and all(0.15 <= f / L.FPS <= 0.35 for f in L.FADES)

def test_combo_speed_rule_never_counts_the_join():
    """A6.1: the allowance is max(vmax, each span's own fastest) + tol with the spans measured apart; the jump between them is no
    allowance (M8 V1: it was)."""
    h = np.zeros((6, 31)); h[:, 7] = np.arange(6) * 0.01; g = np.zeros((6, 31)); g[:, 7] = 2.0 + np.arange(6) * 0.02   # 0.5 and 1.0 rad/s; a 2 rad jump
    new = np.zeros((4, 31)); new[:, 7] = np.arange(4) * 0.05; vmax = np.zeros(24)                                     # 2.5 rad/s
    assert L.speed_over(new, [h, g], vmax) == [L.CTRL_JOINTS[0]] and L.speed_over(new, np.concatenate([h, g]), vmax) == []
    new[:, 7] = np.arange(4) * 0.02; assert L.speed_over(new, [h, g], vmax) == []

def test_library_regenerates_without_a_held_out_file(tmp_path, monkeypatch):
    opened, real = [], builtins.open; L.S._lib.cache_clear()
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real(f, *a, **k))[1])
    out = [L.export(tmp_path, clean=False, workers=1, loco=loco)[0] for loco in (False, True)]; monkeypatch.undo()
    bad = [p for p in opened if re.search(r"kfab_\d+\.npz|r1_kungfu|\.onnx|strikes\.json", p)]; assert not bad, bad
    for loco, A in zip((False, True), out):
        f_npz, f_csv, f_ix, _ = L.names_for(loco); old = dict(np.load(L.OUT / f_npz)); assert set(old) - set(A) == {"clean"} and all(np.array_equal(A[k], old[k]) for k in A)
        for f in (f_csv, f_ix): assert L.EM.sha256(tmp_path / f) == L.EM.sha256(L.OUT / f)

def test_t1_motion_rebuilds_and_loco_drops_kfab_203():
    for loco in (False, True):
        f_npz, f_csv, f_ix, f_man = L.names_for(loco); seq, ix = C.build_motion(loco); ex = json.load(open(L.OUT / f_ix, encoding="utf-8")); assert ix == ex
        csv = np.loadtxt(L.OUT / f_csv, delimiter=","); assert np.array_equal(np.char.mod("%.6f", np.concatenate([seq[:, :3], seq[:, [4, 5, 6, 3]], seq[:, 7:]], 1)).astype(float), csv)
        A, man, _ = K.load(loco=loco); src = {r["name"]: set(r["sources"]) for r in man["rows"]}; names = A["names"].tolist()
        assert [n for n, r in ix["rows"].items() if r["group"] == "original"] == man["clips"] and all(not (loco and C.LOCO_DROP in s) for s in src.values())
        for n, r in ix["rows"].items():   # each T1 entry is the row (or its un-mirrored form), twice
            if r["group"] == "original": continue
            i = names.index(n); q = A["q"][K.row(A, i)]; assert r["group"] != "mirror" and r["conj"] == bool(A["mirror"][i])
            q1 = L.mirror(q) if r["conj"] else q; s0 = r["start_frame"] + ix["blend_frames"]; assert np.array_equal(seq[s0:r["start_frame"] + len(q)], q1[ix["blend_frames"]:])
        assert {n for n in ix["rows"] if ix["rows"][n]["group"] != "original"} == {n for n, f in zip(names, A["family"].tolist()) if f != "mirror"}
    assert man["vmax"] != json.load(open(L.OUT / "fb_l1_manifest.json", encoding="utf-8"))["vmax"]   # LOCO's own p99 (A6.3)

def test_sampling_keeps_half_the_weight_on_the_originals():
    ix = json.load(open(L.OUT / "fb_l1_track_index.json", encoding="utf-8")); g = torch.as_tensor(C.bin_groups(ix, ix["frames"]))
    assert len(g) == int(ix["frames"] // 50) + 1 and g[0] == 0 and g[-1] == 1 and 0.05 < (g == 0).float().mean() < 0.2
    k = torch.tensor([1.0]); rng = torch.Generator().manual_seed(0)
    for failed in (torch.zeros(len(g)), torch.rand(len(g), generator=rng) * (g == 1), torch.rand(len(g), generator=rng)):
        p = C.group_probs(failed, 0.1, k, g); assert abs(float(p.sum()) - 1) < 1e-6 and abs(float(p[g == 0].sum()) - C.ORIGINAL_SHARE) < 1e-6
        base = failed + 0.1 / len(g); assert torch.allclose(p[g == 1] / p[g == 1].sum(), base[g == 1] / base[g == 1].sum(), atol=1e-6)
    k3 = torch.tensor([0.8 ** i for i in range(3)]); k3 = k3 / k3.sum(); p = C.group_probs(torch.rand(len(g), generator=rng), 0.1, k3, g); assert abs(float(p[g == 0].sum()) - 0.5) < 1e-6

def test_t1_config_contract():
    src = (ROOT / "gpu" / "r1_track_task.py").read_text(encoding="utf-8")
    assert '"force_range": (-25.0, 25.0), "torque_range": (0.0, 0.0), "duration_s": (0.05, 0.12), "cooldown_s": (1.0, 3.0)' in src and C.HAND_IMPULSE == dict(
        force_range=(-25.0, 25.0), torque_range=(0.0, 0.0), duration_s=(0.05, 0.12), cooldown_s=(1.0, 3.0))
    assert "steer=True, strict=True)" in src and C.STEER2 == dict(relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True) and C.ITERATIONS == {C.TASK: 4000, C.TASK_LOCO: 3000}
    assert not fb_seeds.scan([ROOT / "gpu" / "r1_track_spar_cfg.py", ROOT / "gpu" / "export_fb_library.py", ROOT / "scripts" / "fb_m8_check.py", ROOT / "tests" / "test_fb_library.py"])
    assert Path(ROOT / "gpu" / "r1_track_spar_cfg.py") in fb_seeds.task_inputs() and Path(ROOT / "gpu" / "export_fb_library.py") in fb_seeds.task_inputs()

def test_performability_ids_are_the_l1_perf_block():
    ids = [i for f in L.FAMILIES for i in K.IDS[f]]; assert ids == list(range(60500, 60596)) and fb_seeds.check(ids, "l1_perf", L.CLIPS) == ids
    with pytest.raises(fb_seeds.SeedRefused): fb_seeds.check([60600], "l1_perf", L.CLIPS)

def test_conjugated_runner_on_a_mirror_row(monkeypatch):
    """A6.2: the mirror row of kfab_201 driven by T0 conjugated stays up from start 0 for 3 s, as T0 does on kfab_201."""
    monkeypatch.setattr(K, "SECONDS", 3.0); assert K._alone(("l1_mirror_kfab_201_m0", 0.0, "l1conj")) is None

def test_runner_equals_the_evals_tracker_on_an_original(monkeypatch):
    """The runner path (row replay + T0 driver on fb_motion's reference) against the eval's own Tracker, kfab_201 start 0, 3 s."""
    import fb_m2_check as M2
    monkeypatch.setattr(K, "SECONDS", 3.0); a = K._alone(("kfab_201", 0.0, "orig")); b = M2._alone(("kfab_201", 0, 24, "eval", 3.0))["fell"]
    assert a is None and b is None
