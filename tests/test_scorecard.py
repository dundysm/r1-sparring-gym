"""scripts/scorecard.py: the sparring scorecard reproduces the stored eval summaries exactly, and the stored baseline is what the
script makes from them."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import scorecard as sc

def test_headline_numbers():
    """The baselines quoted in the sparring plan, as stored: guard_fw against the aimed opponent on the hard test tier 4.241 clean
    head strikes received / min at survival 0.969; without scripted hits 4.818 (test) and 7.72 (val); the aimed opponent fell
    in 33 of step_fw's 128 hard test bouts."""
    t, _ = sc.build(); g = lambda tier, c="RL guard_fw": t[tier]["controllers"][c]
    assert g("test hard aimed")["head_clean_pm"] == 4.241 and g("test hard aimed")["survival"] == 0.969 and g("test hard aimed")["head_clean_pm_ci95"] == [3.244, 5.314]
    assert g("test hard nohits aimed")["head_clean_pm"] == 4.818 and g("val hard nohits aimed")["head_clean_pm"] == 7.72
    assert g("test hard aimed", "RL step_fw")["opponent_fell_n"] == 33 and g("test hard aimed", "RL step_fw")["episodes"] == 128
    assert g("test standard")["head_clean_pm"] is None and g("test standard")["survival"] == 1.0
    assert g("test hard aimed")["intent_blocked_share"] == 0.256 and g("test hard aimed")["blocked_share"] == 0.176 and g("test hard aimed")["intent_n"] == 4405   # 1126 / 4405 aimed strikes blocked

def test_every_value_is_the_stored_one():
    """Every row field equals its summary's value (no rounding), opponent_fell_n agrees with the stored share, and each hard-tier
    unaimed summary's survival matches main's results/eval_opponent/summary_*_hard*.json for every controller both have. Aimed
    tiers carry the intent windows' count, head share and blocked / n; the other tiers (shadow intents included) don't."""
    t, sha = sc.build(); assert len(t) == 12 and all(k in sha for k in (v["source"] for v in t.values()))
    for lab, v in t.items():
        S = json.load(open(ROOT / v["source"]))["summary"]; assert set(v["controllers"]) == set(S)
        for n, r in v["controllers"].items():
            s = S[n]; L = s.get("landed") or {}
            assert (r["survival"], r["survival_ci95"], r["up_within_1m"], r["up_within_1m_ci95"], r["opponent_fell"]) == (s["survival"], s["survival_ci95"], s["survival_in_ring"], s["survival_in_ring_ci95"], s["opponent_fell"])
            assert r["head_clean_pm"] == (L["head_clean_pm"]["rate"] if L else None) and r["torso_clean_pm"] == (L["torso_clean_pm"]["rate"] if L else None)
            assert r["blocked_share"] == ((L.get("windows") or {}).get("blocked_share")) and round(r["opponent_fell_n"] / r["episodes"], 3) == r["opponent_fell"]
            W = s.get("intent_windows")
            if lab.endswith(" aimed"): assert (r["intent_n"], r["intent_head_share"], r["intent_blocked_share"]) == (W["n"], W["head_share"], round(W["blocked"] / W["n"], 3))
            else: assert not any(k.startswith("intent_") for k in r)
        main = ROOT / "results" / "eval_opponent" / Path(v["source"]).name
        if v["source"].startswith("results/eval_opponent/aim/") and "aimed" not in lab and main.exists():
            M = json.load(open(main))["summary"]; assert all(M[n]["survival"] == v["controllers"][n]["survival"] for n in M)

def test_stored_baseline_is_current():
    """results/sparring/scorecard_baseline.json is the script's output on the files it hashed (they haven't changed since), by
    this version of the script, with the tree's git state recorded."""
    B = json.load(open(ROOT / sc.OUT)); t, sha = sc.build()
    assert B["tiers"] == json.loads(json.dumps(t)) and B["meta"]["sources_sha256_16"] == sha and B["meta"]["script"] == "scripts/scorecard.py"
    assert B["meta"]["script_sha256_16"] == sc.script_sha() and "git_dirty" in B["meta"] and "git_hash" in B["meta"]
