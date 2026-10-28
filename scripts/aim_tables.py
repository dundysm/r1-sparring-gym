"""README tables for the aimed opponent (M1), generated from the stored results only: the strike library and its audit
(r1gym/assets/motions/strikes.json), the tracker's tolerance (results/opponent_tracker/tolerance.json), the paired aim check
and the operator it chose (results/opponent_aim/aim_check.json, results/opponent_tracker/aim.json), and the hard tier with and
without the aim (results/eval_opponent/summary_{val,test}_hard[_nohits][_aim].json). --write replaces the blocks between
'<!-- tag:begin -->' and '<!-- tag:end -->': aim, headline and spacing in README.md, results in docs/AIMED_OPPONENT.md (after
checking that the preregistered text above its Results section still has the prereg_sha every M1 result carries); without
it the blocks are printed."""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from r1gym.stats import cluster_paired_diff
from r1gym.env import TRAIN_CLIPS, VAL_CLIPS
CTRL = [("pd_stand", "PD stand"), ("RL v1", "RL v1 (residual)"), ("RL v2", "RL v2 (residual)"), ("RL v3", "RL v3 cand. (residual)"), ("RL step_s1", "Stepping base, stage 1"),
        ("RL step", "**Stepping base, arms down**"), ("RL guard", "**Stepping base + guard**")]
OPS = [("standoff", "StandOff"), ("armwarp", "ArmWarp"), ("armwarp_vz", "ArmWarp, height and sideways only"), ("standoff+armwarp_vz", "StandOff + ArmWarp (height, sideways)"), ("select", "StrikeSelector")]

def ci(x, f="{:+.1f}"): return f"{f.format(x['diff'])} [{f.format(x['ci95'][0]).lstrip('+')}, {f.format(x['ci95'][1]).lstrip('+')}]"
def j(p): p = ROOT / p; return json.load(open(p, encoding="utf-8")) if p.exists() else None

def block():
    L = []; lib = j("r1gym/assets/motions/strikes.json"); tol = j("results/opponent_tracker/tolerance.json"); ac = j("results/opponent_aim/aim_check.json"); aim = j("results/opponent_tracker/aim.json")
    au, s = lib["audit"], lib["summary"]
    L += [f"**Strike library** (`r1gym/strikes.py`, `scripts/strike_library.py`): {s['strikes']} strikes ({s['hands']} with a hand) in the motion the tracker performs; "
          f"{s['splits']['train']['head_band']} train, {s['splits']['val']['head_band']} val and {s['splits']['test']['head_band']} test strikes are level with the head. "
          f"Against {au['labelled']} hand-labelled candidates ({au['unsure']} unsure; `strike_labels.json` says how they were labelled): precision {au['all']['precision']:.2f} "
          f"[{au['all']['precision_ci95'][0]:.2f}, {au['all']['precision_ci95'][1]:.2f}], recall {au['all']['recall']:.2f} [{au['all']['recall_ci95'][0]:.2f}, {au['all']['recall_ci95'][1]:.2f}]. "
          f"Sheet: [`media/strikes_sheet.png`](media/strikes_sheet.png).", ""]
    w = tol["arm_warps"]; lu = tol["lunges"]["l20_v10"]; lr = tol["long_runs"]
    L += [f"**What the shipped tracker tolerates** (`scripts/tracker_tolerance.py`, opponent alone, {tol['strikes']} train + val hand strikes): its palm comes within 5 cm of the "
          f"reference apex in {w['0']['within_5cm']:.0%} of strikes unwarped, {w['up10']['within_5cm']:.0%} / {w['up20']['within_5cm']:.0%} when the apex is moved 10 / 20 cm up, "
          f"{w['any20']['within_5cm']:.0%} for 20 cm in any of three directions; no warp up to 25 cm made it fall. A 20 cm lunge at 1 m/s adds "
          f"{lu['extra_reach_cm_median']:.1f} cm of reach (median). Slowing strikes to 0.5x: {lr['retime_0.5']['fell']}/48 falls (unchanged: {lr['unchanged']['fell']}/48). "
          f"Splicing to head-level strikes: {lr['select']['fell']}/48 falls, so the selector {'passes' if lr['select']['passes'] else 'fails'} its preregistered tolerance"
          f"{'' if lr['select']['passes'] else ' and is out'}. Its lag: {1000 * tol['lag']['train_median_s']:.0f} ms (median, train).", ""]
    if ac:
        sa = ac["summary"]["arms_down"]; base = sa["by_mode"]["shadow"]; d = ac["decision"]
        L += [f"**Paired aim check** (`scripts/aim_check.py`, train + val, {ac['meta']['grid']['bouts_per_cell']} bouts of 10 s per operator and target at 0.40-0.65 m; "
              f"against the arms-down base; the unaimed opponent lands {base['landed']['head_clean_pm']['rate']:.1f} head clean strikes/min):", "",
              "| operator | Δ head clean strikes/min [95% CI] | Δ head-landed share per strike window | opponent falls, pooled (unaimed {:.1%}) | sign under the other hit rules |".format(
                  d["candidates"]["standoff"]["opp_fell_unaimed"]), "|---|---|---|---|---|"]
        for op, lab in OPS:
            c = d["candidates"].get(op, {})
            if not c.get("considered", False): L.append(f"| {lab} | not run: failed its splice tolerance | | | |"); continue
            p = sa["paired"][op]; r = p["rules"]; sg = " ".join(f"{k} {r[k]['head']['diff']:+.1f}" for k in r if k != "clean")
            L.append(f"| {lab}{' **(chosen)**' if op == d['operator'] else ''} | {ci(p['head_clean_pm'])} | {ci(p['window_head_share'], '{:+.3f}')} | {c['opp_fell_pooled']:.1%}{' (rejected)' if c['rejected_falls'] else ''} | {sg} |")
        cs = d["candidates"]; ok = sorted([c for c in cs if cs[c].get("considered") and cs[c]["ci_above_0"] and not cs[c]["rejected_falls"]], key=lambda c: -cs[c]["delta"]); nm = dict(OPS)
        nxt = f" The next best, {nm[ok[1]]}, is {cs[ok[0]]['delta'] - cs[ok[1]]['delta']:.1f}/min behind (the tie margin is 0.5)." if len(ok) > 1 else ""
        flip = "Its sign is the same under every hit rule." if not d["sign_flips_under"] else f"Its sign flips under {', '.join(d['sign_flips_under'])}."
        L += ["", f"Chosen by the preregistered rule (`results/opponent_tracker/aim.json`, before any aimed test run): **{nm[aim['operator']]}**, the largest paired difference "
                  f"whose CI is above 0.{nxt} Opponent falls: {cs[aim['operator']]['opp_fell_pooled']:.1%} of bouts against {cs[aim['operator']]['opp_fell_unaimed']:.1%} unaimed "
                  f"(the rule allows +2 pp). {flip}" + ("" if cs["standoff"]["ci_above_0"] else f" StandOff alone misses the bar: its CI starts at {cs['standoff']['ci95'][0]:.1f}."), ""]
        op = d["operator"]; raw = j("results/opponent_aim/aim_check_raw.json")["raw"]
        ncl = {k: len({(r["clip"], r["t0"]) for r in raw["arms_down"]["shadow"] if r["clip"] in v}) for k, v in (("train", TRAIN_CLIPS), ("val", VAL_CLIPS))}
        L += [f"The chosen operator on every target (same episodes; peak forces at the shell solref named):", "",
              f"| target | head clean strikes/min, unaimed → aimed | Δ [95% CI] | Δ on train clips ({ncl['train']} start times) / on the val clip ({ncl['val']}) | Δ head-landed/min | Δ head impulse, N·s/min | strike windows ending on the head, unaimed → aimed | windows not aimed (lean / touch gate) | windows whose target is >= 8 cm beyond the arm's reach | head-hit peak force, median / p95 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t, lab in (("mannequin", "mannequin (welded R1: the aim alone)"), ("arms_down", "arms-down base"), ("guard", "guard"), ("arms_down_solref0.02", "arms-down, shell solref 0.02")):
            S_, P_ = ac["summary"][t]["by_mode"], ac["summary"][t]["paired"][op]; I = [i for r in raw[t][op] for i in r["intents"]]
            far = sum(i["ik_res_cm"] is not None and i["ik_res_cm"] >= 8 for i in I) / len(I); gt = sum(i["gated"] == "tilt" for i in I) / len(I); gc = sum(i["gated"] == "touch" for i in I) / len(I)
            pk = S_[op]["landed"]["peak_force_N"]; sub = {}
            for nm_, cl_ in (("train", TRAIN_CLIPS), ("val", VAL_CLIPS)):   # the same estimate on each split's clips alone
                A_, B_ = [r for r in raw[t][op] if r["clip"] in cl_], [r for r in raw[t]["shadow"] if r["clip"] in cl_]; f_ = lambda r: r["hit_meter"]["head_clean_pm"]; w_ = lambda r: r["hit_meter"]["secs"] / 60
                sub[nm_] = ci(dict(zip(("diff", "ci95"), cluster_paired_diff([f_(r) for r in A_], [f_(r) for r in B_], [(r["clip"], r["t0"]) for r in A_], wa=[w_(r) for r in A_], wb=[w_(r) for r in B_]))))
            L.append(f"| {lab} | {S_['shadow']['landed']['head_clean_pm']['rate']:.1f} → {S_[op]['landed']['head_clean_pm']['rate']:.1f} | {ci(P_['head_clean_pm'])} | {sub['train']} / {sub['val']} | {ci(P_['head_landed_pm'])} | "
                     f"{ci(P_['head_impulse_pm'])} | {S_['shadow']['windows']['head_share']:.1%} → {S_[op]['windows']['head_share']:.1%} | {gt:.0%} / {gc:.0%} | {far:.0%} | "
                     f"{pk['median']:.0f} / {pk['p95']:.0f} N at {ac['summary'][t]['shell_solref'][0]} |")
        fr = j("results/opponent_aim/aim_frames.json")
        if fr: L += ["", f"![unaimed vs aimed](media/aim_frames.png)", "", f"*Left unaimed, right aimed ({fr['operator']}): the same {fr['bout']['clip']} bout at {fr['bout']['distance']} m, "
                         f"at the apex of each aimed strike that ended on the head (`scripts/aim_frames.py`; the bout where the aim added the most head windows, so the best case, not a typical one).*"]
        bk = ac["summary"]["arms_down"]["by_mode"][op]["by_ik_residual"]
        L += ["", f"Reach is the limit, as the design predicted: on the arms-down base, strikes whose target the arm can reach from where the opponent stands "
                  f"(IK residual < 3 cm, {bk['<3cm']['n']} windows) end on the head {bk['<3cm']['head_share']:.0%} of the time, those 3-8 cm short {bk['3-8cm']['head_share']:.0%}, "
                  f"and the {bk['>=8cm']['n']} that are 8 cm or more short {bk['>=8cm']['head_share']:.0%}. The actual pelvis spacing stays near "
                  f"{ac['summary']['arms_down']['by_mode'][op]['landed']['spacing_m']['median']:.2f} m (median) at nominal 0.40-0.65 m. Whole-body aiming (M2, GPU) is what can change that.", ""]
    rows = []
    for split in ("val", "test"):
        for push in ("", "_nohits"):
            u, a = j(f"results/eval_opponent/summary_{split}_hard{push}.json"), j(f"results/eval_opponent/summary_{split}_hard{push}_aim.json")
            if not (u and a): continue
            for k, lab in CTRL[4:]:   # the stepping bases; PD and v1-v3 fall in every hard bout either way (all of them are in the JSON)
                if k not in a["summary"] or "vs_unaimed" not in a["summary"][k]: continue
                U, A, V = u["summary"][k], a["summary"][k], a["summary"][k]["vs_unaimed"]
                rows.append(f"| {split}{', no scripted hits' if push else ''} | {lab} | {U['survival']:.3f} → {A['survival']:.3f} | {ci(V['survival'], '{:+.3f}')} | "
                            f"{U['landed']['head_clean_pm']['rate']:.1f} → {A['landed']['head_clean_pm']['rate']:.1f} | {ci(V['head_clean_pm'])} | "
                            f"{ci(V['opponent_fell'], '{:+.3f}')} | {U['landed']['spacing_m']['median'] if U['landed'].get('spacing_m') else '–'} m |")
    if rows:
        L += [f"**Hard tier with the aimed opponent** (`--aim chosen`: {aim['operator']}; the same episodes as the unaimed tier, paired; nominal distance 0.40-0.55 m):", "",
              "| split | controller | survival, unaimed → aimed | Δ survival [95% CI] | head clean strikes/min, unaimed → aimed | Δ [95% CI] | Δ opponent falls [95% CI] | actual spacing (median, unaimed) |",
              "|---|---|---|---|---|---|---|---|", *rows, ""]
        T, U = j("results/eval_opponent/summary_test_hard_aim.json"), j("results/eval_opponent/summary_test_hard.json")
        if T:
            V = {k: T["summary"][k]["vs_unaimed"] for k in ("RL step", "RL guard")}; up = [k for k in V if V[k]["opponent_fell"]["ci95"][0] > 0.02]
            name = lambda k: "arms down" if k == "RL step" else "guard"; of = lambda D, k: D["summary"][k]["opponent_fell"]
            N, M = j("results/eval_opponent/summary_test_hard_nohits_aim.json"), j("results/eval_opponent/summary_test_hard_nohits.json")
            nh = f" ({', '.join(f'{of(M, k):.1%} → {of(N, k):.1%}' for k in up)} without the scripted hits)" if N and M else ""
            L += [f"On test (run once, after the choice): head clean strikes/min {ci(V['RL step']['head_clean_pm'])} against arms down and {ci(V['RL guard']['head_clean_pm'])} "
                  f"against the guard; survival {ci(V['RL step']['survival'], '{:+.3f}')} and {ci(V['RL guard']['survival'], '{:+.3f}')}. PD and the residual policies fall in every hard bout with scripted hits, aimed or not (their rows, "
                  f"with and without the hits, are in the JSON and in `docs/AIMED_OPPONENT.md`)."
                  + (f" **The aimed opponent falls more in the hard tier:** in {', '.join(f'{of(U, k):.1%} → {of(T, k):.1%} ({name(k)})' for k in up)} of test bouts{nh}. "
                     f"The preregistered fall rule is judged on the aim-check grid (10 s, nominal physics, 0.40-0.65 m), where it stayed within +2 pp; in the hard tier "
                     f"(20 s, randomised R1, 0.40-0.55 m) it doesn't, with or without the scripted hits, so M1's operator is not fit to be the default opponent." if up else ""), ""]
    return "\n".join(L)

PREREG = ROOT / "docs" / "AIMED_OPPONENT.md"
M1_RESULTS = ["results/opponent_tracker/tolerance.json", "results/opponent_aim/aim_check.json", "results/opponent_tracker/aim.json", "results/opponent_aim/aim_frames.json"] + \
             [f"results/eval_opponent/summary_{s}{t}{a}.json" for s in ("val", "test") for t in ("", "_hard", "_hard_nohits") for a in ("", "_aim")] + \
             [f"results/eval_opponent/summary_{s}_puppet.json" for s in ("val", "test")]

def prereg_sha():
    """sha256 (first 16 hex) of docs/AIMED_OPPONENT.md, LF line endings, without its '## Results' section: the text the results
    were computed under (scripts/eval_opponent.py provenance() hashes the whole file, which had no Results section then)."""
    s = PREREG.read_bytes().replace(b"\r\n", b"\n"); i = s.find(b"\n## Results")
    import hashlib; return hashlib.sha256(s if i < 0 else s[:i]).hexdigest()[:16]

def provenance_rows():
    out = []
    for f in M1_RESULTS:
        d = j(f)
        if d is None: continue
        m = d.get("meta", {}); out.append((f, m.get("prereg_sha"), (m.get("git_hash") or "")[:7] or None, m.get("git_dirty")))
    return out

def doc_block():
    """The results section of docs/AIMED_OPPONENT.md (M1.2-M1.5): every number from the stored JSON."""
    lib = j("r1gym/assets/motions/strikes.json"); tol = j("results/opponent_tracker/tolerance.json"); ac = j("results/opponent_aim/aim_check.json"); aim = j("results/opponent_tracker/aim.json")
    lab = j("r1gym/assets/motions/strike_labels.json"); au, s = lib["audit"], lib["summary"]; d = ac["decision"]; cs = d["candidates"]; op = d["operator"]; nm = dict(OPS)
    pr = provenance_rows(); sha = prereg_sha(); bad = [f for f, p, _, _ in pr if p != sha]; nogit = [f for f, _, g, _ in pr if g is None]
    L = ["### Provenance and the test-once record", "",
         f"- The {len(pr)} M1 result files (`M1_RESULTS` in `scripts/aim_tables.py`) carry `prereg_sha` {', '.join(sorted({p for _, p, _, _ in pr}))}; this file without this Results section hashes to "
         f"{sha} ({'the same' if not bad else 'DIFFERENT for ' + ', '.join(bad)}). The definitions, rules and amendment 1 above are the text they were computed under.",
         f"- They were run from the uncommitted working tree on {sorted({g for _, _, g, _ in pr if g})[0]} (`git_dirty` true"
         + (f"; {', '.join(nogit)} records no commit" if nogit else "") + "): the commit that adds this section is the code that produced them.",
         "- Order, from the files' write times and the run logs: amendment 1; the strike library; tolerance; aim check and `aim.json`; the hard tier on val "
         "(unaimed and aimed, with and without scripted hits); then the hard tier on test, each variant once; then the standard and puppet tiers, val before test. "
         "No test result was generated twice, and nothing was chosen after a test run. The standard-tier val `--aim` run was repeated once (val, after the "
         "test runs) because its first run recorded no commit (git failed during it): the bouts and summary came out identical, only `meta` changed.", "",
         "### M1.2 strike library", "",
         f"- {s['strikes']} strikes, {s['hands']} with a hand and {s['feet']} kicks. Train {s['splits']['train']['strikes']} ({s['splits']['train']['head_band']} level with the head) in "
         f"{s['splits']['train']['seconds']:.1f} s, val {s['splits']['val']['strikes']} ({s['splits']['val']['head_band']}) in {s['splits']['val']['seconds']:.1f} s, test "
         f"{s['splits']['test']['strikes']} ({s['splits']['test']['head_band']}) in {s['splits']['test']['seconds']:.1f} s: test can say little about head-level strikes.",
         f"- Audit against {au['labelled']} labelled candidates ({au['unsure']} unsure, left out): precision {au['all']['precision']:.2f} [{au['all']['precision_ci95'][0]:.2f}, {au['all']['precision_ci95'][1]:.2f}], "
         f"recall {au['all']['recall']:.2f} [{au['all']['recall_ci95'][0]:.2f}, {au['all']['recall_ci95'][1]:.2f}]; hands {au['hands']['precision']:.2f} / {au['hands']['recall']:.2f}; "
         f"val {au['val']['precision']:.2f} / {au['val']['recall']:.2f}. One labeller, the agent that wrote the detector (`strike_labels.json`: `labeller`), no second rater.",
         f"- The design probes counted {lib['designs']['A']['events']} (A) and {lib['designs']['B']['events']} (B) events; the rule variants on the same labels are in `strikes.json` `variants` "
         f"(B-like trigger: {lib['variants']['B_like: total speed >= 1.0, no kick height']['events']} events, precision {lib['variants']['B_like: total speed >= 1.0, no kick height']['all_precision']:.2f}).", "",
         "### M1.4 tolerance (opponent alone, train + val)", "",
         f"| reference change | palm within 5 cm of the apex | design B's probe |", "|---|---|---|"]
    w = tol["arm_warps"]; bp = tol["design_B_probe"]
    for k, lab_ in (("0", "none"), ("up10", "apex 10 cm up"), ("up20", "apex 20 cm up")):
        L.append(f"| {lab_} | {w[k]['within_5cm']:.1%} of {w[k]['n']} | {100 * bp['within_5cm'][k]:.1f}% |")
    lr = tol["long_runs"]
    L += ["", f"A 20 cm lunge at 1 m/s adds {tol['lunges']['l20_v10']['extra_reach_cm_median']:.1f} cm of reach (median; B's probe {bp['lunge_20cm_1mps_extra_reach_cm']} cm). "
              f"Falls in {lr['unchanged']['runs']} runs of 10 s: unchanged {lr['unchanged']['fell']}, retimed 0.7x {lr['retime_0.7']['fell']}, 0.5x {lr['retime_0.5']['fell']}, "
              f"spliced ('select') {lr['select']['fell']}. Lag {1000 * tol['lag']['train_median_s']:.0f} ms (train median, frozen in `aim.json`).", "",
          "### M1.4 decision rules (aim check, train + val; `results/opponent_aim/aim_check.json` → `results/opponent_tracker/aim.json`)", ""]
    sel = lr["select"]; base = cs["standoff"]["opp_fell_unaimed"]
    L.append(f"1. **Candidates:** StandOff, ArmWarp, ArmWarp height and sideways only, StandOff + ArmWarp (height, sideways). The selector is out: spliced runs fell "
             f"{sel['extra_vs_unchanged']} more times than unchanged ones (amendment 1 allows 1).")
    L.append(f"2. **Opponent falls,** pooled over the grid at solref 0.01 (unaimed {base:.1%}, limit +2 pp): " + "; ".join(
        f"{nm[c]} {cs[c]['opp_fell_pooled']:.1%}{' (rejected)' if cs[c]['rejected_falls'] else ''}" for c in cs if cs[c].get("considered")) + ". None rejected." * (not any(cs[c].get("rejected_falls") for c in cs if cs[c].get("considered"))))
    P = ac["summary"]["arms_down"]["paired"]
    L.append(f"3. **Paired Δ head clean strikes/min, arms-down target** (aimed minus shadow, cluster bootstrap): " + "; ".join(
        f"{nm[c]} {ci(P[c]['head_clean_pm'], '{:+.2f}')}" for c in cs if cs[c].get("considered")) + ".")
    ok = sorted([c for c in cs if cs[c].get("considered") and cs[c]["ci_above_0"] and not cs[c]["rejected_falls"]], key=lambda c: -cs[c]["delta"])
    gap = cs[ok[0]]["delta"] - cs[ok[1]]["delta"] if len(ok) > 1 else None
    L.append(f"4. **Tie-break:** the top two with CIs above 0 differ by {gap:.3f}/min ({'under' if gap < 0.5 else 'not under'} 0.5), so " +
             ("the tie-break decides." if gap < 0.5 else f"the largest wins: **{nm[op]}**.") if gap is not None else "4. **Tie-break:** one candidate only.")
    L.append(f"5. **Fallback:** {'not triggered' if ok else 'triggered: StandOff ships as distance management'}.")
    L.append(f"6. **Every hit rule:** the chosen Δ is " + ", ".join(f"{r} {v:+.2f}" for r, v in d["rule_head_deltas"].items()) + "/min; "
             + ("the sign holds under every rule." if not d["sign_flips_under"] else f"the sign flips under {', '.join(d['sign_flips_under'])}."))
    L.append(f"7. **Written before any test run:** `aim.json` (`decided_on`: \"{aim['decided_on']}\").")
    mp = ac["summary"]["mannequin"]["paired"][op]["head_clean_pm"]; gp = ac["summary"]["guard"]["paired"][op]["head_clean_pm"]; raw = j("results/opponent_aim/aim_check_raw.json")["raw"]
    def split_ci(t, cl):
        A_, B_ = [r for r in raw[t][op] if r["clip"] in cl], [r for r in raw[t]["shadow"] if r["clip"] in cl]; f_ = lambda r: r["hit_meter"]["head_clean_pm"]; w_ = lambda r: r["hit_meter"]["secs"] / 60
        return ci(dict(zip(("diff", "ci95"), cluster_paired_diff([f_(r) for r in A_], [f_(r) for r in B_], [(r["clip"], r["t0"]) for r in A_], wa=[w_(r) for r in A_], wb=[w_(r) for r in B_]))))
    L += ["", f"The other targets, same operator: the welded mannequin {ci(mp)} (train clips {split_ci('mannequin', TRAIN_CLIPS)}, val clip {split_ci('mannequin', VAL_CLIPS)}), "
              f"the guard {ci(gp)}. Neither CI is above 0 on its own; the rule is judged on the arms-down target only.", ""]
    L += ["### M1.5 hard `--aim` tier: Δ survival and Δ primary per controller (aimed minus unaimed, same episodes)", "",
          "| split | scripted hits | controller | Δ survival [95% CI] | Δ head clean strikes/min [95% CI] | Δ opponent falls [95% CI] | hit rules with the other sign |",
          "|---|---|---|---|---|---|---|"]
    for split in ("val", "test"):
        for push, pl in (("", "on"), ("_nohits", "off")):
            a = j(f"results/eval_opponent/summary_{split}_hard{push}_aim.json")
            if not a: continue
            for k, lb in CTRL:
                if k not in a["summary"]: continue
                V = a["summary"][k]["vs_unaimed"]; sg = V["head_clean_pm"]["diff"] > 0
                fl = [r for r, x in V["rules"].items() if r != "clean" and (x["head"]["diff"] > 0) != sg]
                L.append(f"| {split} | {pl} | {lb} | {ci(V['survival'], '{:+.3f}')} | {ci(V['head_clean_pm'], '{:+.2f}')} | {ci(V['opponent_fell'], '{:+.3f}')} | {', '.join(fl) or '–'} |")
    T, U = j("results/eval_opponent/summary_test_hard_aim.json")["summary"], j("results/eval_opponent/summary_test_hard.json")["summary"]
    Vv = j("results/eval_opponent/summary_val_hard_aim.json")["summary"]
    L += ["", f"- **Stepping bases on test:** the aim adds head clean strikes against both (arms down {ci(T['RL step']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}, guard "
              f"{ci(T['RL guard']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}/min), as on val for the guard ({ci(Vv['RL guard']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}); on val the "
              f"arms-down CI touches 0 ({ci(Vv['RL step']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}). The guard's survival drops on test ({ci(T['RL guard']['vs_unaimed']['survival'], '{:+.3f}')}).",
          f"- **The opponent falls more** against the stepping bases on test: {U['RL step']['opponent_fell']:.1%} → {T['RL step']['opponent_fell']:.1%} (arms down), "
          f"{U['RL guard']['opponent_fell']:.1%} → {T['RL guard']['opponent_fell']:.1%} (guard). M1.5 has no fall rule; rule 2 (aim check) passed, so this is reported, not a rejection.",
          f"- **PD, the residual policies and stage 1**, which fall in most hard bouts, show the aim's effect with opposite signs on val and test; their bouts are short, "
          f"so their rates rest on few minutes. They are reported, not interpreted.", "",
          "### What this doesn't show", "",
          f"- The defenders are the stepping bases of this branch (`results/policy_mjlab*/`, with the hold-your-spot settings of d1990f9), not later footwork policies.",
          f"- Most aimed windows can't reach the head from where the opponent stands: on the arms-down target {ac['summary']['arms_down']['by_mode'][op]['by_ik_residual']['>=8cm']['n']} of "
          f"{ac['summary']['arms_down']['by_mode'][op]['windows']['n']} windows' targets are 8 cm or more beyond the arm, and "
          f"{ac['summary']['arms_down']['by_mode'][op]['windows']['gated_share']:.0%} of windows are not aimed (lean or touch gate). The effect is per minute of bout, not per strike.",
          f"- Test holds {s['splits']['test']['strikes']} library strikes, {s['splits']['test']['head_band']} of them level with the head; the test result is one replication on two short clips."]
    return "\n".join(L)

def headline():
    """The headline table (the frozen v1 tiers: every cell from results/eval_opponent/summary_*.json) with the hit columns, and
    the actual pelvis spacing next to the hard tier's nominal distance."""
    S = {f: (j(f"results/eval_opponent/summary_{f}.json") or {}).get("summary", {}) for f in ("test_hard", "val_hard", "test", "val", "test_puppet", "val_puppet")}
    f3 = lambda x: f"{x:.3f}"; c2 = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"; sv = lambda f, k: f3(S[f][k]["survival"]) if k in S[f] else "–"
    L = ["| | **hard: test survival** [95% CI] | hard: val | hard: test, up and within 1 m of start | hard: test, head clean strikes taken/min [95% CI] | hard: test, blocked share of strike windows | standard: test [95% CI] | standard: val | puppet: test | puppet: val |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in CTRL:
        h = S["test_hard"][k]; ld = h["landed"]; lab = lab[:-2] + " (shipped)**" if lab.endswith("**") else lab
        L.append(f"| {lab} | **{f3(h['survival'])}** {c2(h['survival_ci95'])} | {sv('val_hard', k)} | {f3(h['survival_in_ring'])} | {ld['head_clean_pm']['rate']:.1f} {c2(ld['head_clean_pm']['ci95'])} | "
                 f"{ld['windows']['blocked_share']:.2f} | {f3(S['test'][k]['survival'])} {c2(S['test'][k]['survival_ci95'])} | {sv('val', k)} | {sv('test_puppet', k)} | {sv('val_puppet', k)} |")
    sp, sg = S["test_hard"]["RL step"]["landed"]["spacing_m"], S["test_hard"]["RL guard"]["landed"]["spacing_m"]
    return "\n".join(L), f"actual pelvis spacing on test: median {sp['median']:.2f} m (p10-p90 {sp['p10']:.2f}-{sp['p90']:.2f}) against arms down, {sg['median']:.2f} m against the guard"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args(); b = block(); hl, sp = headline(); db = doc_block()
    if not a.write: sys.stdout.reconfigure(encoding="utf-8"); print(b, hl, sp, db, sep="\n\n"); sys.exit(0)
    def put(p, blocks):   # each generated block sits between <!-- tag:begin --> and <!-- tag:end -->
        s = p.read_text(encoding="utf-8")
        for tag, txt in blocks:
            i, k = s.index(f"<!-- {tag}:begin -->"), s.index(f"<!-- {tag}:end -->"); s = s[:i] + f"<!-- {tag}:begin -->" + txt + s[k:]
        p.write_bytes(s.replace("\r\n", "\n").encode("utf-8"))
    put(ROOT / "README.md", (("aim", "\n" + b + "\n"), ("headline", "\n" + hl + "\n"), ("spacing", sp))); print("README blocks written")
    sha = prereg_sha(); put(PREREG, (("results", "\n" + db + "\n"),))
    assert prereg_sha() == sha and all(p == sha for _, p, _, _ in provenance_rows()), "docs/AIMED_OPPONENT.md above '## Results' changed since the runs"
    print("docs/AIMED_OPPONENT.md results written; preregistration text unchanged:", sha)
