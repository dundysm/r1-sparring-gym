"""Strike library (r1gym/strikes.py, r1gym/assets/motions/strikes.json): regenerates byte-identical, matches the tracker's
motion frame for frame, follows its rule, keeps the splits apart, and carries its audit."""
import json, subprocess, sys, pathlib
import numpy as np, pytest
from r1gym import strikes as S
from r1gym.env import TRAIN_CLIPS, VAL_CLIPS, TEST_CLIPS
from r1gym.opponent import list_clips

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = json.load(open(S.PATH, encoding="utf-8"))

def test_library_regenerates_byte_identical():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "strike_library.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr

def test_frames_match_the_tracker_motion_and_windows_lie_inside():
    ix = json.load(open(ROOT / "results" / "opponent_tracker" / "motion_index.json"))["clips"]
    assert {c: LIB["per_clip"][c]["frames"] for c in list_clips()} == {c: ix[c]["frames"] for c in list_clips()} == dict(kfab_200=250, kfab_201=722, kfab_202=123, kfab_203=276, kfab_204=194, kfab_205=102)
    for s in S.load():
        T = LIB["per_clip"][s.clip]["frames"]; assert 0 <= s.onset <= s.apex <= s.end < T, s
    for c in list_clips():
        assert all(0 <= t0 <= t1 < LIB["per_clip"][c]["frames"] / 50.0 for t0, t1, _ in S.windows_for(c))

def test_strikes_follow_the_rule():
    """Every stored strike re-derives from the motion: onset is a rising edge through v_on, the apex is the most forward point
    within 0.4 s and >= 0.25 m ahead of the spot, the end is the first non-positive forward speed, kicks are >= 0.30 m high."""
    for c in list_clips():
        q, fps = S.motion(c); P, Tq = S.effector_paths(q); H = P * [-1.0, -1.0, 1.0]; V = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) * fps / 2
        for s in S.load(clip=c):
            j = list(S.EFFECTORS).index(s.limb); v = V[:, j, 0]
            assert v[s.onset] > 1.5 >= v[s.onset - 1] and s.apex == s.onset + int(np.argmax(H[s.onset:s.onset + 21, j, 0]))
            assert H[s.apex, j, 0] >= 0.25 and np.isclose(s.apex_m[0], H[s.apex, j, 0], atol=1e-3) and all(v[s.apex:s.end] > 0) and v[s.end] <= 0
            assert s.hand or s.apex_m[2] >= 0.30
            assert s.band == ("head" if 1.02 <= s.apex_m[2] <= 1.23 else "torso" if 0.79 <= s.apex_m[2] < 1.02 else "high" if s.apex_m[2] > 1.23 else "low")
            assert np.isclose(s.reach_head_m, s.apex_m[0] + S.r1_surfaces()["head"]["front_m"], atol=2e-3)

def test_splits_are_disjoint_and_match_the_env():
    sp = {c: LIB["per_clip"][c]["split"] for c in LIB["per_clip"]}
    assert {c for c, x in sp.items() if x == "train"} == set(TRAIN_CLIPS) and {c for c, x in sp.items() if x == "val"} == set(VAL_CLIPS) and {c for c, x in sp.items() if x == "test"} == set(TEST_CLIPS)
    assert all(s.split == sp[s.clip] for s in S.load()) and sum(len(S.load(split=x)) for x in ("train", "val", "test")) == len(S.load())

def test_audit_is_stored_with_its_labels():
    a = LIB["audit"]; lab = json.load(open(S.ASSETS / "motions" / "strike_labels.json", encoding="utf-8"))
    assert a["labelled"] == len(lab["labels"]) >= 30 and {"protocol", "labeller", "about"} <= set(lab)
    for k in ("all", "hands", "val"):
        assert 0 < a[k]["precision"] <= 1 and 0 < a[k]["recall"] <= 1 and a[k]["precision_ci95"][0] <= a[k]["precision"] <= a[k]["precision_ci95"][1]
    assert set(LIB["designs"]) >= {"A", "B", "why"} and "rule" in LIB["variants"] and "test_note" in LIB["summary"]
    assert {(l["clip"], l["limb"], l["frame"]) for l in lab["labels"]} == {(c, limb, f) for c, _, limb, f, _ in S.audit_candidates(list_clips())}

def test_replay_exposes_its_strikes():
    from r1gym.model import build_spec, load_cfg
    cfg = {**load_cfg(), "opponent_mode": "kinematic"}; m = build_spec(opponent=True, cfg=cfg).compile()
    from r1gym.opponent import FightReplay
    r = FightReplay(m, "kfab_204", distance=0.5, cfg=cfg); assert r.strikes == S.load(clip="kfab_204") and len(r.strikes) > 0
    r2 = FightReplay(m, "kfab_201", distance=0.5, cfg={**cfg, "opponent_style_cuts": False})   # another motion: refused
    with pytest.raises(AssertionError, match="strike library"): r2.strikes
