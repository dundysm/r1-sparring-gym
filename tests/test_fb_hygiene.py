"""Full-body M2 v2 held-out hygiene (M2V2_PLAN_v3 M0; scripts/fb_seeds.py): no banned name in any fullbody task input, the
B7 seed guard per role (val/test/screen/Track 2 blocks, ledger blocks, clip sets, the test-once guard, A4.1's listed G-only role
w0_r4pre), and a disjoint ledger (the declared re-runs of used blocks excepted).
FB_SEED_LEDGER=<path to SEED_LEDGER.md> also audits the workstream's ledger file (skipped without it)."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import fb_seeds as F
from r1gym.env import TRAIN_CLIPS, VAL_CLIPS, TEST_CLIPS

def clips_of(role):
    kind, S = F.ROLES[role][1]
    return sorted(S) if kind == "sub" else ["cmu_15_13"]

def granted(tmp_path):
    pre = tmp_path / "prereg.json"; pre.write_bytes(b'{"rules": 1}\n')
    sel = tmp_path / "sel.json"; sel.write_text(json.dumps(dict(go=True, prereg_sha256_16=hashlib.sha256(pre.read_bytes()).hexdigest()[:16])), encoding="utf-8")
    return F.TestOnce(sel, prereg=pre, eval_root=tmp_path / "eval")

def test_task_inputs_carry_no_banned_name():
    paths = F.task_inputs(); assert any(p.name == "fb_seeds.py" for p in paths)
    assert F.scan(paths) == []

@pytest.mark.parametrize("token", ["kfab_202", "kfab_204", "kfab_205", "kfab_206", "gpu/motions/r1_kungfu.csv", "r1_kungfu_aim.csv",
                                   "results/opponent_tracker/policy.onnx", "results\\opponent_tracker\\policy.onnx"])
def test_scan_catches_a_planted_name_in_every_format(tmp_path, token):
    (tmp_path / "a.py").write_text(f"x = {token!r}\n", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(dict(src=token)), encoding="utf-16")
    np.savez(tmp_path / "c.npz", clips=np.array(["kfab_200", token])); np.savez(tmp_path / "d.npz", **{token.replace("/", "_").replace("\\", "_"): np.zeros(1)})
    hits = {Path(p).name for p, _, _ in F.scan(sorted(tmp_path.iterdir()))}
    assert hits >= {"a.py", "b.json", "c.npz"} and ("d.npz" in hits or "/" in token or "\\" in token)

def test_scan_passes_train_names_and_the_defender_onnx(tmp_path):
    (tmp_path / "a.py").write_text("C = ['kfab_200', 'kfab_201', 'kfab_203', 'kfab_2030']\nP = 'logs/rsl_rl/r1_fb/policy.onnx'\n", encoding="utf-8")
    np.savez(tmp_path / "b.npz", clips=np.array(TRAIN_CLIPS)); assert F.scan(sorted(tmp_path.iterdir())) == []

@pytest.mark.parametrize("role", sorted(F.ROLES))
def test_every_role_refuses_hard_r2_val_and_test(role, tmp_path):
    g = granted(tmp_path) if role == "test" else None
    for s in (21000, 21500, 21999):
        with pytest.raises(F.SeedRefused, match="21000-21999"): F.check([s], role, clips_of(role), g)
    if role != "test":
        for s in (22000, 22255, 22511):
            with pytest.raises(F.SeedRefused, match="22000-22511"): F.check([s], role, clips_of(role))

@pytest.mark.parametrize("role", sorted(F.ROLES))
def test_screen_and_track2_blocks_only_for_their_roles(role):
    for s, own in ((23000, "screen_val"), (23255, "screen_val"), (24000, "track2"), (25999, "track2")):
        if role != own:
            with pytest.raises(F.SeedRefused): F.check([s], role, clips_of(role))

def test_role_blocks_and_clip_sets(tmp_path):
    assert F.check(range(60000, 60016), "record", TRAIN_CLIPS) == list(range(60000, 60016))
    assert F.check([88000, 89151], "loco_heldout", ["kfab_203"]) and F.check([89500], "loco_seen", ["kfab_200", "kfab_201"])
    assert F.check(range(23000, 23256), "screen_val", VAL_CLIPS)[-1] == 23255 and F.check([24000], "track2", ["cmu_15_13"])
    bad = [([60016], "record", TRAIN_CLIPS), ([60000], "record", TRAIN_CLIPS + VAL_CLIPS), ([60000], "record", []), ([60000], "record", TEST_CLIPS[:1]),
           ([88000], "loco_heldout", ["kfab_200"]), ([89500], "loco_seen", ["kfab_203"]), ([23000], "screen_val", TRAIN_CLIPS[:1]),
           ([24000], "track2", ["kfab_200"]), ([24000], "track2", []), ([70000], "nope", TRAIN_CLIPS), ([], "record", TRAIN_CLIPS),
           ([67000], "eq", TRAIN_CLIPS), ([56192], "s3", TRAIN_CLIPS), ([60000.5], "record", TRAIN_CLIPS), ([True], "record", TRAIN_CLIPS)]
    for seeds, role, clips in bad:
        with pytest.raises(F.SeedRefused): F.check(seeds, role, clips)
    with pytest.raises(F.SeedRefused): F.check([22000], "test", TEST_CLIPS, F.TestOnce.__new__(F.TestOnce))   # built without its checks

def test_test_role_only_through_a_granted_test_once(tmp_path):
    with pytest.raises(F.SeedRefused, match="TestOnce"): F.check([22000], "test", TEST_CLIPS)
    with pytest.raises(F.SeedRefused, match="no prereg"): F.TestOnce(ROOT / "results" / "sparring" / "m2_fb_selection.json")   # M11 not built: refused
    g = granted(tmp_path); assert g.granted and F.check(range(22000, 22512), "test", TEST_CLIPS, g) == list(range(22000, 22512))
    for clips in (TRAIN_CLIPS[:1], VAL_CLIPS, TEST_CLIPS + VAL_CLIPS):
        with pytest.raises(F.SeedRefused): F.check([22000], "test", clips, g)
    with pytest.raises(F.SeedRefused): F.check([22512], "test", TEST_CLIPS, g)
    pre, sel = tmp_path / "prereg.json", tmp_path / "sel.json"
    pre.write_bytes(b'{"rules": 2}\n')
    with pytest.raises(F.SeedRefused, match="hash"): F.TestOnce(sel, prereg=pre, eval_root=tmp_path / "eval")
    sel.write_text(json.dumps(dict(go=False)), encoding="utf-8")
    with pytest.raises(F.SeedRefused, match="go is False"): F.TestOnce(sel, prereg=pre, eval_root=tmp_path / "eval")
    g = granted(tmp_path); spent = tmp_path / "eval" / "run3"; spent.mkdir(parents=True); (spent / "raw_test_hard_r2_aim.json").write_text("{}")
    with pytest.raises(F.SeedRefused, match="already ran"): granted(tmp_path)

def test_ledger_blocks_are_disjoint_and_the_check_sees_an_overlap(monkeypatch):
    assert F.ledger_problems() == []
    assert {b for bl, _ in F.ROLES.values() for b in bl} >= {(60000, 60015), (63000, 63383), (64000, 65535), (70000, 82287), (100000, 112287), (22000, 22511)}
    monkeypatch.setitem(F.ROLES, "late", ([(60010, 60020)], ("sub", F.TRAIN))); assert any("late" in p for p in F.ledger_problems())
    monkeypatch.setitem(F.ROLES, "late", ([(68000, 68010)], ("sub", F.TRAIN))); assert any("used" in p for p in F.ledger_problems())
    monkeypatch.setitem(F.ROLES, "late", ([(57100, 57110)], ("sub", F.TRAIN))); assert len(F.ledger_problems()) == 2   # retired and legacy zone

def test_w0_r4pre_role_takes_only_the_listed_g_bouts(monkeypatch):
    L = F.W0_R4PRE; assert len(L) == len(set(L)) == 144 and F.list_sha(L) == F.W0_R4PRE_SHA256 and set(L) <= set(range(67000, 67768)) | set(range(68000, 68768))
    assert F.check(L, "w0_r4pre", TRAIN_CLIPS, ctrl="G") == sorted(L) and F.ROLES["w0_r4pre"][0] == F.REUSE["w0_r4pre"]
    for seeds, ctrl, why in (([67000], "G", "list"), ([67002], "H", "G only"), ([67002], "A3", "G only"), ([67002], None, "G only"), ([60300], "G", "block"), ([70000], "G", "block")):
        with pytest.raises(F.SeedRefused, match=why): F.check(seeds, "w0_r4pre", TRAIN_CLIPS, ctrl=ctrl)
    with pytest.raises(F.SeedRefused): F.check([67002], "w0_r4pre", VAL_CLIPS, ctrl="G")
    monkeypatch.setitem(F.LISTS, "w0_r4pre", L + (67000,))
    with pytest.raises(F.SeedRefused, match="sha256"): F.check([67000], "w0_r4pre", TRAIN_CLIPS, ctrl="G")
    monkeypatch.setitem(F.ROLES, "late", ([(67000, 67767)], ("sub", F.TRAIN))); assert any("late" in p for p in F.ledger_problems())   # not a declared re-run

def test_ledger_audit_flags_an_unknown_block(tmp_path):
    p = tmp_path / "L.md"; p.write_text("| date | block |\n|---|---|\n| 2026-10-29 | 67000-67767 (768) | x |\n| 2026-10-30 | 60000-60015 (16) | M2 |\n", encoding="utf-8")
    assert F.audit(p) == []
    p.write_text(p.read_text() + "| 2026-10-30 | 61000-61015 (16) | stray |\n", encoding="utf-8"); assert len(F.audit(p)) == 1

@pytest.mark.skipif(not os.environ.get("FB_SEED_LEDGER"), reason="FB_SEED_LEDGER not set (the workstream ledger lives outside the repo)")
def test_workstream_seed_ledger_is_consistent():
    p = Path(os.environ["FB_SEED_LEDGER"]); rows = [l for l in p.read_text(encoding="utf-8").splitlines() if F.ROW.match(l)]
    assert len(rows) >= 4 and F.audit(p) == []
