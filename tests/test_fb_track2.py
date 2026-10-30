"""Track 2 (M2V2_PLAN_v3 6.2 item 3; scripts/fb_track2_split.py, fb_track2_motion.py, fb_track2_eligibility.py, fb_track2_check.py):
the title-only split (rebuilt from the data folder when present, its sha256, video and style rules, excluded IDs), the npz
format guard, the motion pipeline = FightReplay's on a train clip, the cut loop and its bound, the tracker-alone world's seed
independence and a known fall, the sealed reduction (booleans and aggregates only, exception type only), the track2 seed role.
The full checks (3 train clips, 72 tracker-alone starts, 33 train-pool clips) run in scripts/fb_track2_check.py."""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "gpu"))
import fb_track2_split as SP, fb_track2_motion as T2, fb_track2_eligibility as EL, fb_seeds as F
import style_cuts as SC
from r1gym import opponent as O, strikes as S

MOT = ROOT / "r1gym" / "assets" / "motions"
HAVE_DATA = (SP.DATA / "action_titles.csv").exists() and (SP.DATA / "npz").is_dir()

def cached(c): return np.load(MOT / f"r1_{c}.npy", allow_pickle=False)

def test_committed_split_and_its_hash():
    S_ = SP.load(); ids = {k: [int(c.split("_")[1]) for c in v] for k, v in S_["splits"].items()}
    assert S_["seed"] == SP.SEED and S_["excluded_ids"] == list(range(196, 207)) and not set(sum(ids.values(), [])) & set(range(196, 207))
    assert sum(map(len, ids.values())) == 80 and len(set(sum(ids.values(), []))) == 80
    assert {k: v["videos"] for k, v in S_["counts"].items()} == dict(train=14, val2=4, test2=4)
    for v in S_["videos"]: assert set(f"kfab_{i}" for i in v["ids"]) <= set(S_["splits"][v["split"]])   # a video never straddles splits
    for k in ("val2", "test2"): assert S_["counts"][k]["styles"] == dict(changquan=2, nanquan=1, taichi=1)
    bad = dict(S_); bad["splits"] = dict(S_["splits"], val2=S_["splits"]["val2"][1:]); p = ROOT / "results" / "sparring" / "track2_split.json"
    tmp = Path(str(p) + ".tmp_test"); tmp.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    try:
        with pytest.raises(AssertionError): SP.load(tmp)
    finally: tmp.unlink()

@pytest.mark.skipif(not HAVE_DATA, reason="Track 2 data folder absent")
def test_split_rebuilds_bit_for_bit_from_titles():
    S_ = SP.build(); C = SP.load()
    assert S_ == C and S_["inputs"] == C["inputs"] and S_["npz_checked"]["files"] == 80

def test_partition_rules_on_synthetic_titles():
    t = {i: f"{k}-Scene-{i:03d}.mp4" for i, k in enumerate(["长拳A"] * 3 + ["长拳B", "长拳C", "长拳D", "长拳E", "长拳F", "长拳G"] + ["南拳A", "南拳B"] + ["太极拳A", "太极拳B", "太极拳C"], 1)}
    v1, s1 = SP.partition(t, sorted(t), 1); v2, s2 = SP.partition(t, sorted(t), 1); v3, s3 = SP.partition(t, sorted(t), 2)
    assert s1 == s2 and v1["长拳A"] == [1, 2, 3] and len(s1) == 12
    for s in (s1, s3):
        cnt = lambda k: sum(x == k for x in s.values()); assert (cnt("val2"), cnt("test2"), cnt("train")) == (2, 2, 8)
        assert all(sum(x == k and SP.style(v) == st for v, x in s.items()) <= {"changquan": 2, "nanquan": 1, "taichi": 1}[st] for k in ("val2", "test2") for st in ("changquan", "nanquan", "taichi"))
    assert SP.video("段位制长拳三段单练套路-Scene-002.mp4") == "段位制长拳三段单练套路" and SP.style("42式太极拳-第三段") == "taichi" and SP.style("国一南拳") == "nanquan"
    assert SP.routine("第一套国际武术竞赛套路-长拳-第三段（动作名称）") == SP.routine("第一套国际武术竞赛套路-长拳（完整版）国一长拳") == "guoyi_changquan"

def test_split_refuses_an_excluded_id(tmp_path):
    (tmp_path / "action_titles.csv").write_text("Data Number,Source Video Title\n200.mp4,x-Scene-001.mp4\n230.mp4,y-Scene-001.mp4\n", encoding="utf-8")
    (tmp_path / "fist_ids.txt").write_text("200\n230\n", encoding="ascii")
    (tmp_path / "manifest_expected.tsv").write_text("".join(f"{i}\tk/datasets/org_smooth_mj/{i}.npz\t1\t{'0' * 64}\n" for i in (200, 230)), encoding="ascii")
    with pytest.raises(AssertionError, match="excluded"): SP.read_inputs(tmp_path)

def _npz(d, i, **over):
    g = O._sub("g1"); T = 6; q = np.zeros((T, 36)); q[:, 2] = 0.79; q[:, 3] = 1.0
    A = dict(fps=np.array([50]), joint_pos=q, joint_vel=np.zeros((T, 35)), body_pos_w=np.zeros((T, 2, 3)), body_quat_w=np.zeros((T, 2, 4)),
             body_lin_vel_w=np.zeros((T, 2, 3)), body_ang_vel_w=np.zeros((T, 2, 3)), joint_names=np.array([g.joint(j).name for j in range(1, g.njnt)]),
             body_names=np.array(["world", "pelvis"])); A.update(over); A = {k: v for k, v in A.items() if v is not None}
    (d / "npz").mkdir(exist_ok=True); np.savez(d / "npz" / f"{i}.npz", **A)

def test_load_guards_the_format(tmp_path):
    _npz(tmp_path, 1); q, fps = T2.load(1, tmp_path); assert q.shape == (6, 36) and fps == 50
    names = np.array([O._sub("g1").joint(j).name for j in range(1, O._sub("g1").njnt)])
    for i, over in enumerate([dict(fps=np.array([30])), dict(joint_names=names[::-1].copy()), dict(joint_vel=None), dict(extra=np.zeros(1)),
                              dict(joint_pos=np.full((6, 36), np.nan))], 2):
        _npz(tmp_path, i, **over)
        with pytest.raises(AssertionError): T2.load(i, tmp_path)
    _npz(tmp_path, 9, body_names=np.array([object(), object()], dtype=object))
    with pytest.raises(ValueError): T2.load(9, tmp_path)   # allow_pickle=False

def test_style_rule_constants_are_style_cuts_py():
    assert (T2.IDLE_SPEED, T2.IDLE_S, T2.POSE_DZ, T2.POSE_S) == (SC.IDLE_SPEED, SC.IDLE_S, SC.POSE_DZ, SC.POSE_S)

def test_process_is_fightreplay_on_a_train_clip():
    c = "kfab_201"; qr = cached(c); q0, src, info = T2.process(qr, 50, O._cuts().get(c, []))
    assert np.array_equal(q0, S.motion(c)[0]) and info["style_cuts"] == O._cuts("style_cuts")[c] and len(src) == len(q0)
    assert np.all(np.diff(src[30:]) >= 0) and 0 <= src.min() and src.max() <= info["frames_clean"] - 1   # cleaned-frame map, monotone past the seam
    L, nh = T2.head_hands(q0, 50, c); assert nh == 4 and len(L) == len(S.load(clip=c))
    ref = T2.reference(q0); assert all(x.dtype == np.float32 and len(x) == len(q0) for x in ref)

def test_cut_loop_and_its_bound():
    qr = cached("kfab_201"); calls = []
    def run(clip, q0, t0): calls.append(t0); return 0.1   # every start falls 0.1 s in: stay-up 0 every round
    rec = T2.screen(0, qr=qr, run=run); n = T2.RULE["max_cut_rounds"] + 1
    assert len(rec["rounds"]) <= n and len(calls) == T2.RULE["starts"] * sum("stay_up" in r for r in rec["rounds"])
    assert not rec["eligible_hand"] and not rec["eligible_any"] and all(r["cuts"] for r in rec["rounds"][1:])
    up = T2.screen(0, qr=qr, run=lambda clip, q0, t0: None)
    assert len(up["rounds"]) == 1 and up["stay_up"] == 1.0 and up["eligible_hand"] and up["eligible_any"] and up["head_hands"] == 4
    assert T2.clusters([5, 1, 50, 3, 200], 45) == [[1, 3, 5, 50], [200]] and T2.merge([[10, 20], [0, 5], [15, 30]]) == [[0, 5], [10, 30]]

def test_a_clip_with_nothing_kept_is_kept_0_s_not_an_error():
    qr = cached("kfab_200").copy(); qr[:, 3:7] = [0.0, 1.0, 0.0, 0.0]   # upside down throughout
    q0, src, info = T2.process(qr, 50); assert q0.shape == (0, 31) and info["kept_s"] == 0.0 and info["empty_at"] == "clean_clip"
    rec = T2.screen(0, qr=qr, run=lambda *a: pytest.fail("no start on an empty clip")); assert rec["rounds"][0]["kept_s"] == 0.0 and not rec["kept_ok"] and not rec["eligible_any"]
    q0, _, info = T2.process(cached("kfab_200"), 50, [[0, 10 ** 6]]); assert len(q0) == 0 and info["empty_at"] == "apply_cuts"
    with pytest.raises(IndexError): T2._kept(lambda: [][0])

def test_alone_is_seed_free_and_falls_where_the_eval_does(monkeypatch):
    c = "kfab_200"; q0, _, _ = T2.process(cached(c), 50, O._cuts().get(c, []), O._cuts("style_cuts").get(c, []))
    t0 = 12 * len(q0) / 50 / 24; a = T2.alone(c, q0, t0, 1.0, full=True)
    assert a[0] == 0.52   # fb_track2_check C5 / fb_m2_check G0-3b: the eval's tracker falls at start 12 after 0.52 s
    import eval_tracker as ET
    env = ET.R1HitEnv; monkeypatch.setattr(ET, "R1HitEnv", lambda **kw: env(**{**kw, "seed": 4321}))
    b = T2.alone(c, q0, t0, 1.0, full=True); assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

def test_sealed_reduction_carries_booleans_and_counts_only(monkeypatch):
    rec = dict(eligible_hand=True, eligible_any=True, kept_ok=True, hand_ok=True, any_ok=True, stay_up_ok=True, stay_up=0.96, head_hands=3, head_any=4,
               rounds=[dict(fell=[0.3], kept_s=5.5, head_hands=3, head_any=4)], strikes=["x"], q0=np.zeros(3), info={})
    cnt = {"head_hands", "head_any", "kept0_s", "head_hands0", "head_any0"}
    r = EL.sealed(rec); assert set(r) == set(EL.BOOLS + EL.REASONS) | {"measured"} | cnt and all(type(v) is bool for k, v in r.items() if k not in cnt)
    def boom(*a, **k): raise ValueError("frame 123 apex 0.42")
    monkeypatch.setattr(T2, "screen", boom)
    e = EL._job((1316, "val2", str(SP.DATA), None)); assert set(e) == {"clip", "ok", "error", "where"} and e["error"] == "ValueError" and e["where"][-1] == "test_fb_track2.py:boom"
    assert "123" not in json.dumps(e) and "0.42" not in json.dumps(e)
    with pytest.raises(ValueError): EL._job((1316, "train", str(SP.DATA), None))
    rows = [dict(clip="kfab_1", ok=True, **r), dict(clip="kfab_2", ok=True, **dict(r, eligible_hand=False, hand_ok=False, head_hands=1)), dict(clip="kfab_3", ok=False, error="X")]
    A = EL.aggregate("val2", rows)
    assert A["clips"] == 3 and A["errors"] == 1 and A["eligible_hand"] == dict(eligible=1, head_band_hand_strikes=3, head_band_strikes=4)
    assert A["all_clips_first_round"] == dict(kept_s=11.0, head_band_hand_strikes=6, head_band_strikes=8, clips_kept_ok=2)
    assert A["eligible_any"]["head_band_hand_strikes"] == 4 and A["fails"]["hand_ok"] == 1 and A["val2_ok"] == dict(eligible_hand=False, eligible_any=False)

def test_track2_seed_role():
    for i in range(196, 207):
        with pytest.raises(F.SeedRefused): F.check([24000], "track2", [f"kfab_{i}"])
    assert F.check([24950], "track2", ["kfab_1316", "kfab_535"]) == [24950]
    for split, n in (("train", 33), ("val2", 40), ("test2", 7)):
        s = EL.seeds_for(split, EL.clips_of(split)); assert len(s) == n and all(24000 <= x <= 25999 for x in s)
    with pytest.raises(F.SeedRefused): F.check([23999], "track2", ["kfab_1316"])
    with pytest.raises(F.SeedRefused): F.check([24000], "eq", ["kfab_200"])
