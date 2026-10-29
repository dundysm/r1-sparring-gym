"""Sparring M2 run 2's evaluation plumbing (results/sparring/prereg_m2_run2.json, docs/SPARRING_M1M2.md section 9):
scripts/eval_opponent.py's hard_r2 tier (fresh per-split seeds, disjoint from run 1's; the v1 tiers' episodes unchanged) and
its test-split guard; scripts/m2_verdict.py --run 2 (tier and seed checks, the val screen with its tie rule and go flag, the
verdict refusing go = false or another prereg, the blind2 twin) on synthetic raw files, with run 1 the default throughout."""
import sys, json, math, importlib.util, shutil
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import m2_verdict as mv

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
tv = _load("test_m2_verdict_helpers", Path(__file__).resolve().parent / "test_m2_verdict.py")   # bout(), episodes() of the run-1 tests

@pytest.fixture(scope="module")
def eo(): return _load("eval_opponent_r2", ROOT / "scripts" / "eval_opponent.py")

def good_all(eo, clips):
    n = int(round(eo.START_SPAN / eo.START_STEP)); return {(c, d): np.ones(n, bool) for c in clips for d in eo.TIERS["hard"]["distances"]}

# ---- 8. eval seeds
def test_v1_episodes_unchanged(eo):
    """hard / standard: the split argument changes nothing, and the hard tier's episodes are run 1's stored ones (every key the
    clean-start grid does not decide: seeds, delays, hits, env and hit seeds)."""
    from r1gym.env import VAL_CLIPS, TEST_CLIPS
    n = int(round(eo.START_SPAN / eo.START_STEP))
    for tier in ("hard", "standard"):
        for clips, reps in ((VAL_CLIPS, 32), (TEST_CLIPS, 16)):
            g = {(c, d): np.ones(n, bool) for c in clips for d in eo.TIERS[tier]["distances"]}
            a = eo.episodes(clips, reps, tier, g, False, None); assert a == eo.episodes(clips, reps, tier, g, False, None, split="val") == eo.episodes(clips, reps, tier, g, False, None, split="test")
    for split, clips, reps in (("val", VAL_CLIPS, 32), ("test", TEST_CLIPS, 16)):
        f = ROOT / "results" / "eval_opponent" / "m2" / f"raw_{split}_hard_nohits.json"
        if not f.exists(): continue
        stored = json.loads(f.read_bytes())["episodes"]; now = eo.episodes(clips, reps, "hard", good_all(eo, clips), False, 20.0, split=split)
        drop = lambda e: {k: v for k, v in e.items() if k != "t0"}
        assert [drop(e) for e in now] == [drop(json.loads(json.dumps(e))) for e in stored] and stored[0]["seed"] == 20000

def test_hard_r2_seeds(eo):
    """hard_r2: val seeds 21000-21127, test 22000-22127; seeds, env seeds and hit seeds disjoint from run 1's (20000-20127) and
    from each other; same clips, distances, bout length and DR as the hard tier."""
    from r1gym.env import VAL_CLIPS, TEST_CLIPS
    v = eo.episodes(VAL_CLIPS, 32, "hard_r2", good_all(eo, VAL_CLIPS), False, 20.0, split="val")
    t = eo.episodes(TEST_CLIPS, 16, "hard_r2", good_all(eo, TEST_CLIPS), False, 20.0, split="test")
    r1 = eo.episodes(VAL_CLIPS, 32, "hard", good_all(eo, VAL_CLIPS), False, 20.0, split="val") + eo.episodes(TEST_CLIPS, 16, "hard", good_all(eo, TEST_CLIPS), False, 20.0, split="test")
    assert [e["seed"] for e in v] == list(range(21000, 21128)) and [e["seed"] for e in t] == list(range(22000, 22128)) and len(v) == len(t) == 128
    for k in ("seed", "env_seed", "hit_seed"):
        sv, st, s1 = ({e[k] for e in x} for x in (v, t, r1)); assert not (sv & st) and not (sv & s1) and not (st & s1)
    H, R = eo.TIERS["hard"], eo.TIERS["hard_r2"]; assert all(R[x] == H[x] for x in ("distances", "seconds", "hits", "impulse", "dr"))
    assert {e["clip"] for e in v} == set(VAL_CLIPS) and {e["clip"] for e in t} == set(TEST_CLIPS)

def test_test_guard(eo, tmp_path, monkeypatch):
    """The hard_r2 test split runs only after a go selection made against the current prereg (LF-normalised hash)."""
    pre = tmp_path / "prereg.json"; pre.write_bytes(b'{"a": 1}\r\n'); sel = tmp_path / "sel.json"
    assert eo.r2_test_allowed(sel, pre)[0] is False
    sel.write_text(json.dumps(dict(go=False, prereg_sha256_16=eo.text_sha16(pre)))); ok, why = eo.r2_test_allowed(sel, pre); assert not ok and "go is False" in why
    sel.write_text(json.dumps(dict(go=True, prereg_sha256_16=eo.text_sha16(pre)))); assert eo.r2_test_allowed(sel, pre) == (True, "go")
    assert eo.text_sha16(pre) == mv.text_sha16(pre) and eo.text_sha16(pre) == __import__("hashlib").sha256(b'{"a": 1}\n').hexdigest()[:16]
    pre.write_bytes(b'{"a": 2}\n'); ok, why = eo.r2_test_allowed(sel, pre); assert not ok and "prereg hash" in why
    monkeypatch.setattr(eo, "R2_SELECTION", tmp_path / "absent.json")
    monkeypatch.setattr(sys, "argv", ["eval_opponent.py", "--tier", "hard_r2", "--split", "test", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="refused"): eo.main()
    assert not (tmp_path / "o").exists()   # refused before anything ran

# ---- 9. m2_verdict --run 2
@pytest.fixture
def run2(tmp_path):
    """m2_verdict pointed at run 2 with a copy of the prereg (or a stand-in) in tmp_path; run 1 restored afterwards."""
    mv.use_run(2); pre = tmp_path / "prereg_m2_run2.json"; real = ROOT / "results" / "sparring" / "prereg_m2_run2.json"
    if real.exists(): shutil.copy(real, pre)
    else: pre.write_text(json.dumps(dict(selection=dict(rule="screen"))))
    mv.PREREG = pre
    try: yield pre
    finally: mv.use_run(1)

def r2_episodes(split, seed0=None):
    E = tv.episodes(split); s0 = seed0 if seed0 is not None else dict(val=21000, test=22000)[split]
    for k, e in enumerate(E): e["seed"] = s0 + k
    return E

def write_r2(out, split, aimed, ctrls, tier="hard_r2", seed0=None):
    """raw_{split}_hard_r2_nohits[_aim].json as eval_opponent writes it for run 2: ctrls {label: (kp_mode, f(k, ep) -> bout kwargs)}."""
    E = r2_episodes(split, seed0); kp = {n: m for n, (m, _) in ctrls.items() if m}
    meta = dict(tier=tier, tier_cfg=dict(distances=[0.4, 0.45, 0.5, 0.55], seconds=20.0, hits=6, impulse=[8, 14], dr=True, seed=21000, seed_by_split=dict(val=21000, test=22000)),
                opponent="tracked", opponent_mode="policy", split=split, reps=mv.REPS[split], pushes=False, seconds=20.0, hit_meter=True,
                opponent_aim="head_op" if aimed else "shadow", keypoints=dict(source="truth", kp_mode=kp), **({"aim": dict(operator="head_op", from_aim_json=True)} if aimed else {}))
    raw = {n: [tv.bout(e, **f(k, e)) for k, e in enumerate(E)] for n, (_, f) in ctrls.items()}
    f = Path(out) / f"raw_{split}_hard_r2_nohits{'_aim' if aimed else ''}.json"; json.dump(dict(meta=meta, episodes=E, raw=raw), open(f, "w")); return f

REF = tv.REF   # 7.5 head clean / min
def val_r2(out, a2, a2s=None, blind=REF, **kw):
    c = {"pd_stand": (None, tv.const()), "RL guard_fw_r": (None, REF), "RL A2": ("obs", a2), "RL blind2": ("zeros", blind)}
    if a2s is not None: c["RL A2s"] = ("obs", a2s)
    for aimed in (False, True): write_r2(out, "val", aimed, c, **kw)

def test_screen_pick_and_tie(tmp_path, run2):
    """Both qualify: A2s only when lower by >= 0.1/min; A2s alone if A2 fails the head screen; A2s may be absent."""
    lo = lambda k, e: dict(head=1 if k % 2 else 0, torso=1)   # 1.5/min: R_warm 0.2
    lo2 = lambda k, e: dict(head=1 if k % 4 == 0 else 0, torso=1)   # 0.75/min
    lo_tie = lambda k, e: dict(head=1 if (k % 2 or k == 0) else 0, torso=1)   # 1 more hit in 128 bouts: +0.023/min
    val_r2(tmp_path, lo, lo2); s = mv.select(tmp_path, tmp_path / "s.json")
    assert s["go"] and s["selected"] == "A2s" and s["pool"] == ["A2", "A2s"] and s["twin"] == "blind2" and s["twin_seed_matched"] is False
    assert math.isclose(s["head_clean_pm_A2_minus_A2s"], 0.75) and s["candidates"]["A2"]["R_warm_head"] == pytest.approx(1.5 / 7.5)
    J = json.load(open(tmp_path / "s.json")); assert J["go"] is True and J["prereg_sha256_16"] == mv.text_sha16(run2) and J["meta"]["prereg_sha256_16"] == mv.text_sha16(run2)
    val_r2(tmp_path, lo, lo_tie); s = mv.select(tmp_path, tmp_path / "s.json"); assert s["selected"] == "A2" and s["tie_break_to_A2"] and s["twin_seed_matched"] is True
    hi = lambda k, e: dict(head=2 + k % 2, torso=1)   # = guard_fw_r: R_warm 1.0 > 0.8
    val_r2(tmp_path, hi, lo); s = mv.select(tmp_path, tmp_path / "s.json"); assert s["selected"] == "A2s" and s["pool"] == ["A2s"] and not s["candidates"]["A2"]["qualifies"]
    for f in tmp_path.glob("raw_*"): f.unlink()
    val_r2(tmp_path, lo); s = mv.select(tmp_path, tmp_path / "s.json"); assert s["selected"] == "A2" and s["absent"] == ["A2s"] and s["head_clean_pm_A2_minus_A2s"] is None

def test_screen_survival_and_ring(tmp_path, run2):
    """Survival < 0.95 or a ring drop > 0.05 disqualifies whatever the head rate; none qualifying: go = false, nothing selected."""
    fall = lambda k, e: dict(head=0, torso=0, fell=10.0 if k % 10 == 0 else None)   # survival 0.898
    ring = lambda k, e: dict(head=0, torso=0, ring_out=k % 10 == 0)                 # survival 1.0, in ring 0.898
    val_r2(tmp_path, fall, ring); s = mv.select(tmp_path, tmp_path / "s.json")
    assert s["go"] is False and s["selected"] is None and s["pool"] == [] and s["twin"] is None
    assert not s["candidates"]["A2"]["qualifies"] and s["candidates"]["A2s"]["survival"] == 1.0 and s["candidates"]["A2s"]["ring_drop_vs_ref"] > 0.05

def test_protocol_refuses_wrong_tier_or_seed(tmp_path, run2):
    lo = lambda k, e: dict(head=0, torso=1)
    val_r2(tmp_path, lo, tier="hard")
    with pytest.raises(SystemExit, match="not 'val hard_r2 nohits"): mv.select(tmp_path, tmp_path / "s.json")
    val_r2(tmp_path, lo, seed0=20000)
    with pytest.raises(SystemExit, match="episode seeds"): mv.select(tmp_path, tmp_path / "s.json")
    val_r2(tmp_path, lo, seed0=22000)   # the test split's seeds on val
    with pytest.raises(SystemExit, match="episode seeds"): mv.select(tmp_path, tmp_path / "s.json")
    for f in tmp_path.glob("raw_*"): f.unlink()
    with pytest.raises(SystemExit, match="raw_val_hard_r2_nohits.json missing"): mv.select(tmp_path, tmp_path / "s.json")

def test_verdict_go_twin_and_prereg(tmp_path, run2):
    """go = false stops the verdict; a selection made against another prereg too; A2s is judged against blind2 (not
    seed-matched, recorded); the rule is run 1's."""
    sel = tmp_path / "sel.json"; h = mv.text_sha16(run2)
    sel.write_text(json.dumps(dict(go=False, selected=None, prereg_sha256_16=h)))
    with pytest.raises(SystemExit, match="go is False"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    sel.write_text(json.dumps(dict(go=True, selected="A2s", selected_label="RL A2s", prereg_sha256_16="0" * 16)))
    with pytest.raises(SystemExit, match="prereg hash"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    for pick in ("A2s", "A2"):
        sel.write_text(json.dumps(dict(go=True, selected=pick, selected_label=f"RL {pick}", prereg_sha256_16=h)))
        for aimed in (False, True):
            write_r2(tmp_path, "test", aimed, {"pd_stand": (None, tv.const()), "RL guard_fw_r": (None, REF), f"RL {pick}": ("obs", tv.GOOD), "RL blind2": ("zeros", REF)})
        v = mv.verdict(tmp_path, sel, tmp_path / "v.json")
        assert v["passed"] and v["roles"]["twin"] == "RL blind2" and v["roles"]["cand"] == f"RL {pick}" and v["twin_seed_matched"] is (pick == "A2") and v["run"] == 2
        assert math.isclose(v["ratios"]["R_warm_head"]["R"], 96 / 320)
        J = json.load(open(tmp_path / "v.json")); assert any(k.endswith("raw_test_hard_r2_nohits_aim.json") for k in J["meta"]["files_sha256_16"])
    sel.write_text(json.dumps(dict(go=True, selected="A", selected_label="RL A", prereg_sha256_16=h)))
    with pytest.raises(SystemExit, match="not one of"): mv.verdict(tmp_path, sel, tmp_path / "v.json")

def test_run1_is_the_default():
    """use_run(1) (the CLI default) keeps run 1's files, tier and candidates."""
    mv.use_run(1); assert mv.RUN_N == 1 and mv.PREREG.name == "prereg_m2.json" and mv.SELECTION.name == "m2_selection.json" and mv.RUN["tier"] == "hard" and mv.RUN["cands"] == ("A", "B")
    assert mv.TWIN["A"] == "blind" and mv.TWIN["A2"] == mv.TWIN["A2s"] == "blind2"
