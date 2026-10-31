"""scripts/m2_verdict.py on synthetic eval_opponent raw files: the preregistered selection (results/sparring/prereg_m2.json) and
the five rule parts of the verdict, the cluster-bootstrap ratio with zero-reference resamples as inf, robust label matching,
protocol checks, and that select never reads a test file."""
import sys, json, math
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import m2_verdict as mv

DIST = [0.4, 0.45, 0.5, 0.55]

def episodes(split):
    """The preregistered layout: val 1 clip x 4 distances x 32, test 2 clips x 4 x 16; one start per bout (128 clusters)."""
    clips, reps = (["v0"], 32) if split == "val" else (["t0", "t1"], 16)
    return [dict(clip=c, distance=d, rep=r, t0=round(0.5 + 0.01 * k, 3), delay=k % 3, hits=[], seed=k) for k, (c, d, r) in
            enumerate((c, d, r) for c in clips for d in DIST for r in range(reps))]

def bout(ep, head=0, torso=0, fell=None, ring_out=False, secs=20.0):
    """One eval_opponent row with a hit_meter block (only the keys the scripts read, with the per-minute rates HitMeter stores)."""
    s = fell if fell is not None else secs; pm = lambda x: round(x / (s / 60), 3)
    return dict(clip=ep["clip"], distance=ep["distance"], fell_s=fell, opp_fell=False, drift_m=0.1, back_m=0.0, max_disp_m=1.2 if ring_out else 0.2, ring_out=ring_out,
                hit_meter=dict(secs=s, n=dict(head_landed=head, head_clean=head, torso_landed=torso, torso_clean=torso, arm_clean=0, episodes=head + torso),
                               head_clean_pm=pm(head), torso_clean_pm=pm(torso), rules=dict(clean=[pm(head), pm(torso)], landed=[pm(head), pm(torso)]),
                               head_hit_imp_Ns=[2.0] * head, hit_peak_N=[100.0] * (head + torso), windows=dict(head=head, torso=torso, blocked=1, miss=1),
                               intent_windows=dict(head=head, torso=torso, blocked=2, miss=0), intents_gated=0, intent_closest_cm=[]))

def write_run(out, split, aimed, ctrls, reps=None):
    """raw_{split}_hard_nohits[_aim].json: ctrls {label: (kp_mode, f(k, ep) -> bout kwargs)}."""
    E = episodes(split); kp = {n: m for n, (m, _) in ctrls.items() if m}
    meta = dict(tier="hard", tier_cfg=dict(distances=[0.4, 0.45, 0.5, 0.55], seconds=20.0, hits=6, impulse=[8, 14], dr=True, seed=20000), opponent="tracked",
                opponent_mode="policy", split=split, reps=reps or mv.REPS[split], pushes=False, seconds=20.0, hit_meter=True,
                opponent_aim="head_op" if aimed else "shadow", keypoints=dict(source="truth", kp_mode=kp), **({"aim": dict(operator="head_op", from_aim_json=True)} if aimed else {}))
    raw = {n: [bout(e, **f(k, e)) for k, e in enumerate(E)] for n, (_, f) in ctrls.items()}
    f = Path(out) / f"raw_{split}_hard_nohits{'_aim' if aimed else ''}.json"; json.dump(dict(meta=meta, episodes=E, raw=raw), open(f, "w")); return f

const = lambda **kw: (lambda k, e: kw)
REF = lambda k, e: dict(head=2 + k % 2, torso=1)   # 2.5 per 20 s bout: 7.5/min

def val_run(out, a, b, ref=REF):
    for aimed in (False, True): write_run(out, "val", aimed, {"pd_stand": (None, const()), "RL guard_fw_r": (None, ref), "RL A": ("obs", a), "RL RL B": ("obs", b), "RL blind": ("zeros", ref)})

def make_test_run(out, cand, twin=REF, ref=REF, pick="A"):
    tw = "RL blind" if pick == "A" else "RL Blind-B"
    for aimed in (False, True): write_run(out, "test", aimed, {"pd_stand": (None, const()), "RL guard_fw_r": (None, ref), f"RL RL {pick}": ("obs", cand), tw: ("zeros", twin)})
    return write_sel(out, pick)

def write_sel(out, pick):
    f = Path(out) / "sel.json"; json.dump(dict(selected=pick, selected_label=f"RL {pick}"), open(f, "w")); return f

GOOD = lambda k, e: dict(head=1 if k % 4 else 0, torso=1)   # 0.75 per bout: R_warm 0.3, and R_blind 0.3 against the default twin (hits like guard_fw_r)

def run_verdict(tmp_path, cand, **kw):
    sel = make_test_run(tmp_path, cand, **kw); return mv.verdict(tmp_path, sel, tmp_path / "v.json")

def oks(v): return {k: p["ok"] for k, p in v["rule"].items()}

def test_pass(tmp_path):
    v = run_verdict(tmp_path, GOOD); R = v["ratios"]["R_warm_head"]
    assert v["passed"] and all(oks(v).values()) and v["roles"]["cand"] == "RL RL A" and v["roles"]["twin"] == "RL blind"
    assert math.isclose(R["R"], 96 / 320) and R["ci95"][0] < R["R"] < R["ci95"][1] < 0.8 and R["resamples_inf"] == 0
    assert math.isclose(R["rate"], 96 / (128 * 20 / 60)) and math.isclose(R["ref_rate"], 320 / (128 * 20 / 60))
    assert math.isclose(v["ratios"]["R_warm_torso"]["R"], 1.0) and v["rule"]["survival"]["survival"] == 1.0
    J = json.load(open(tmp_path / "v.json")); assert J["passed"] is True and J["meta"]["n_boot"] == 4000 and J["meta"]["seed"] == 0
    assert set(J["meta"]["files_sha256_16"]) >= {"results/sparring/prereg_m2.json"} and any(k.endswith("raw_test_hard_nohits_aim.json") for k in J["meta"]["files_sha256_16"])
    rr = J["reported_regardless"]; a = rr["aimed_rows"]["RL RL A"]
    assert a["windows"]["blocked_share"] == pytest.approx(128 / 480) and a["intent_windows"]["blocked_share"] == pytest.approx(256 / 480) and a["impulse_per_head_hit_Ns"] == dict(n=96, median=2.0, p90=2.0)
    assert set(rr["unaimed_rows"]) == {"pd_stand", "RL guard_fw_r", "RL RL A", "RL blind"} and rr["scorecard"] == dict(aimed=None, unaimed=None)
    assert a["rules"]["clean"]["head"] == pytest.approx(a["head_clean_pm"], abs=1e-3)

def test_bootstrap_matches_by_hand(tmp_path):
    """The ratio's resamples are stats.cluster_paired_diff's draw (default_rng(0), 4000 x clusters), ratio of resampled sums."""
    E = episodes("test"); cl = [(e["clip"], e["t0"]) for e in E]; rng = np.random.default_rng(5)
    num = mv.counts([bout(e, head=int(h)) for e, h in zip(E, rng.integers(0, 3, 128))]); den = mv.counts([bout(e, head=int(h)) for e, h in zip(E, rng.integers(1, 5, 128))])
    i = np.random.default_rng(0).integers(0, 128, (4000, 128)); bs = num["head"][i].sum(1) / den["head"][i].sum(1)   # equal minutes per bout
    R = mv.ratio(num, den, "head", cl); assert R["ci95"] == pytest.approx([np.percentile(bs, 2.5), np.percentile(bs, 97.5)], rel=1e-12)
    assert R["R"] == pytest.approx(num["head"].sum() / den["head"].sum())

@pytest.mark.parametrize("part", ["head_vs_warm_point", "head_vs_warm_ci", "head_vs_blind", "torso_vs_warm", "survival", "ring"])
def test_each_part_fails_alone(tmp_path, part):
    kw = {}
    if part == "head_vs_warm_point": cand = lambda k, e: dict(head=7, torso=1); kw = dict(ref=const(head=10, torso=1), twin=const(head=20, torso=1))   # R 0.70 exactly
    if part == "head_vs_warm_ci": cand = lambda k, e: dict(head=32 if k in (5, 77) else 0, torso=1); kw = dict(ref=const(head=1, torso=1), twin=const(head=3, torso=1))   # R 0.5, all in 2 bouts
    if part == "head_vs_blind": cand = GOOD; kw = dict(twin=GOOD)                                    # R_blind 1
    if part == "torso_vs_warm": cand = lambda k, e: dict(head=GOOD(k, e)["head"], torso=2)            # R_torso 2
    if part == "survival":   # the R1 falls in the same 8 bouts under both: survival 0.9375, up within 1 m unchanged vs guard_fw_r
        fall = lambda f: (lambda k, e: dict(f(k, e), **({"fell": 10.0} if k % 16 == 0 else {})))
        cand = fall(GOOD); kw = dict(ref=fall(REF), twin=fall(REF))
    if part == "ring": cand = lambda k, e: dict(GOOD(k, e), ring_out=k < 7)                           # 121 / 128: drop 0.0547
    v = run_verdict(tmp_path, cand, **kw); o = oks(v); want = part.rsplit("_point", 1)[0].rsplit("_ci", 1)[0]
    assert not v["passed"] and [k for k, x in o.items() if not x] == [want], o
    if part == "head_vs_warm_point": assert v["ratios"]["R_warm_head"]["R"] == pytest.approx(0.7) and v["ratios"]["R_warm_head"]["ci95"] == pytest.approx([0.7, 0.7])
    if part == "head_vs_warm_ci": assert v["ratios"]["R_warm_head"]["R"] == pytest.approx(0.5) and v["ratios"]["R_warm_head"]["ci95"][1] >= 0.8
    if part == "survival": assert v["rule"]["survival"]["survival"] == 120 / 128 and v["rule"]["ring"]["ref_minus_candidate"] == 0

def test_zero_reference_resamples_are_inf(tmp_path):
    """The reference's head hits all in one bout: ~37% of resamples miss it (R = inf), so the CI's upper end is inf and part 1
    fails although the point estimate passes; written as "inf". A reference with no head hit at all: R = inf."""
    v = run_verdict(tmp_path, const(head=0, torso=1), ref=lambda k, e: dict(head=40 if k == 3 else 0, torso=1), twin=const(head=1, torso=1))
    R = v["ratios"]["R_warm_head"]; n = R["resamples_inf"]
    assert R["R"] == 0.0 and abs(n / 4000 - (1 - 1 / 128) ** 128) < 0.03 and R["ci95"][1] == math.inf and not v["rule"]["head_vs_warm"]["ok"]
    J = json.load(open(tmp_path / "v.json")); assert J["ratios"]["R_warm_head"]["ci95"][1] == "inf" and J["rule"]["head_vs_warm"]["ci95_hi"] == "inf"
    v = run_verdict(tmp_path, const(head=1, torso=1), ref=const(head=0, torso=1), twin=const(head=2, torso=1))
    assert v["ratios"]["R_warm_head"]["R"] == math.inf and v["ratios"]["R_warm_head"]["resamples_inf"] == 4000 and not v["rule"]["head_vs_warm"]["ok"]

def test_percentile_with_inf():
    x = np.r_[np.arange(10.0), [np.inf] * 3]
    assert mv.pct(x, 50) == np.percentile(x, 50) and mv.pct(x, 97.5) == math.inf and mv.pct(np.arange(7.0), 97.5) == np.percentile(np.arange(7.0), 97.5)
    assert mv.pct([1.0, np.inf], 0) == 1.0 and mv.pct([np.inf, np.inf], 50) == math.inf

def rate_const(total):   # 128 bouts of 20 s: total head hits spread over the bouts
    return lambda k, e: dict(head=total // 128 + (k < total % 128), torso=1)

def test_select_tie_break_and_rule(tmp_path):
    """Head/min over 128 x 20 s: A 213 hits (4.992/min), B 211 (4.945): under 0.1/min apart -> A although B is lower;
    B 206 (4.828): 0.164 lower -> B, Blind-B to train; B lower but survival < 0.95 -> A; neither qualifies -> both, lower wins."""
    sel = lambda: mv.select(tmp_path, tmp_path / "sel.json")
    val_run(tmp_path, rate_const(213), rate_const(211)); s = sel()
    assert s["selected"] == "A" and s["tie_break_to_A"] and s["pool"] == ["A", "B"] and s["twin"] == "blind" and not s["train_blind_b"]
    assert s["head_clean_pm_A_minus_B"] == pytest.approx(2 / (128 * 20 / 60)) and s["selected_label"] == "RL A"
    val_run(tmp_path, rate_const(213), rate_const(206)); s = sel()
    assert s["selected"] == "B" and not s["tie_break_to_A"] and s["twin"] == "blind_b" and s["train_blind_b"] and s["selected_label"] == "RL RL B"
    fall = lambda f, n: (lambda k, e: dict(f(k, e), **({"fell": 10.0} if k < n else {})))
    val_run(tmp_path, rate_const(300), fall(rate_const(100), 7)); s = sel()   # B: survival 121/128 = 0.945
    assert s["selected"] == "A" and s["pool"] == ["A"] and not s["candidates"]["B"]["qualifies"] and s["candidates"]["B"]["survival"] == 121 / 128
    val_run(tmp_path, fall(rate_const(300), 7), lambda k, e: dict(rate_const(100)(k, e), ring_out=k < 7)); s = sel()   # A falls, B rings out (drop 7/128)
    assert s["pool"] == ["A", "B"] and s["selected"] == "B" and not any(c["qualifies"] for c in s["candidates"].values())
    val_run(tmp_path, rate_const(213), fall(rate_const(150), 6)); s = sel()   # 122/128 = 0.953 and a 6/128 = 0.047 ring drop still qualify
    assert s["candidates"]["B"]["qualifies"] and s["selected"] == "B"
    J = json.load(open(tmp_path / "sel.json")); assert J["selected"] == "B" and J["meta"]["mode"] == "select" and "aimed_rows" in J and "unaimed_rows" in J

def test_select_never_reads_a_test_file(tmp_path, monkeypatch):
    """Only the two val raw files and the preregistration are read, even with test files (and a selection) lying beside them."""
    val_run(tmp_path, rate_const(213), rate_const(206)); make_test_run(tmp_path, GOOD)
    (tmp_path / "summary_test_hard_nohits_aim.json").write_text("not json"); (tmp_path / "summary_val_hard_nohits_aim.json").write_text("not json")
    seen = []; real = mv.read; monkeypatch.setattr(mv, "read", lambda p: (seen.append(Path(p).name), real(p))[1])
    s = mv.select(tmp_path, tmp_path / "sel_out.json"); assert s["selected"] == "B"
    assert sorted(set(seen)) == ["prereg_m2.json", "raw_val_hard_nohits.json", "raw_val_hard_nohits_aim.json"] and not any("test" in n for n in seen)
    assert all("test" not in Path(k).name for k in s["meta"]["files_sha256_16"])

def test_verdict_uses_blind_b_for_b(tmp_path):
    v = run_verdict(tmp_path, GOOD, pick="B"); assert v["selected"] == "B" and v["roles"] == dict(ref="RL guard_fw_r", cand="RL RL B", twin="RL Blind-B") and v["passed"]

def test_scorecard_rows_when_summaries_exist(tmp_path):
    sel = make_test_run(tmp_path, GOOD); s = dict(episodes=128, survival=1.0, survival_ci95=[0.97, 1.0], survival_in_ring=1.0, survival_in_ring_ci95=[0.97, 1.0], opponent_fell=0.0,
                                         landed=dict(head_clean_pm=dict(rate=2.25, ci95=[1.9, 2.6]), torso_clean_pm=dict(rate=3.0, ci95=[2.9, 3.1])), sensitivity={"clean": {}})
    json.dump(dict(meta=json.load(open(tmp_path / "raw_test_hard_nohits_aim.json"))["meta"], summary={"RL RL A": s}), open(tmp_path / "summary_test_hard_nohits_aim.json", "w"))
    v = mv.verdict(tmp_path, sel, tmp_path / "v.json"); sc = v["reported_regardless"]["scorecard"]
    assert sc["unaimed"] is None and sc["aimed"]["RL RL A"]["row"]["head_clean_pm"] == 2.25 and sc["aimed"]["RL RL A"]["sensitivity"] == {"clean": {}}

def test_protocol_and_labels(tmp_path):
    """Not the preregistered runs, or ambiguous labels: the script exits before computing anything."""
    assert mv.norm("RL RL Blind-B") == "blind_b" and mv.norm("rl guard_fw_r") == "guard_fw_r" and mv.norm("A") == "a"
    make_test_run(tmp_path, GOOD); sel = write_sel(tmp_path, "A")
    for aimed in (False, True): write_run(tmp_path, "test", aimed, {"RL guard_fw_r": (None, REF), "RL A": ("obs", GOOD), "RL blind": ("zeros", REF)}, reps=8)
    with pytest.raises(SystemExit, match="reps 8"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    for aimed in (False, True): write_run(tmp_path, "test", aimed, {"RL guard_fw_r": (None, REF), "RL A": ("zeros", GOOD), "RL blind": ("obs", REF)})   # twin and candidate swapped
    with pytest.raises(SystemExit, match="kp_mode"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    for aimed in (False, True): write_run(tmp_path, "test", aimed, {"RL guard_fw_r": (None, REF), "RL A": ("obs", GOOD), "RL RL A": ("obs", GOOD), "RL blind": ("zeros", REF)})
    with pytest.raises(SystemExit, match="2 controllers"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    assert mv.verdict(tmp_path, sel, tmp_path / "v.json", labels={"A": "RL RL A"})["roles"]["cand"] == "RL RL A"
    write_run(tmp_path, "test", False, {"RL guard_fw_r": (None, REF), "RL A": ("obs", GOOD), "RL blind": ("zeros", REF)})
    E = json.load(open(tmp_path / "raw_test_hard_nohits.json")); E["episodes"][0]["t0"] = 9.9; json.dump(E, open(tmp_path / "raw_test_hard_nohits.json", "w"))
    write_run(tmp_path, "test", True, {"RL guard_fw_r": (None, REF), "RL A": ("obs", GOOD), "RL blind": ("zeros", REF)})
    with pytest.raises(SystemExit, match="episodes differ"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    for key, val, msg in (("seconds", 10.0, "10.0 s bouts"), ("tier_cfg", dict(distances=[0.4, 0.45, 0.5, 0.55], seconds=20.0, dr=False, seed=20000), "tier_cfg"),
                          ("opponent_mode", "puppet", "not the tracker")):   # the review's adv_protocol.py: each departure alone is caught
        for aimed in (False, True): write_run(tmp_path, "test", aimed, {"RL guard_fw_r": (None, REF), "RL A": ("obs", GOOD), "RL blind": ("zeros", REF)})
        f = tmp_path / "raw_test_hard_nohits_aim.json"; d = json.load(open(f)); d["meta"][key] = val; json.dump(d, open(f, "w"))
        with pytest.raises(SystemExit, match=msg): mv.verdict(tmp_path, sel, tmp_path / "v.json")
    (tmp_path / "raw_test_hard_nohits.json").unlink()
    with pytest.raises(SystemExit, match="missing"): mv.verdict(tmp_path, sel, tmp_path / "v.json")
