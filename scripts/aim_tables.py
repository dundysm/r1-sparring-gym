"""README tables for the aimed opponent (M1), generated from the stored results only: the strike library and its audit
(r1gym/assets/motions/strikes.json), the tracker's tolerance (results/opponent_tracker/tolerance.json), the paired aim check
and the operator it chose (results/opponent_aim/aim_check.json, results/opponent_tracker/aim.json), and the hard tier with and
without the aim: against the shipped footwork policies (results/eval_opponent/aim/, after merging main) and, archived, as first
run against PR #13's bases (results/eval_opponent/aim_pre14/). --write replaces the blocks between '<!-- tag:begin -->' and
'<!-- tag:end -->': aim in README.md, results in docs/AIMED_OPPONENT.md (after checking that the preregistered text above its
Results section still hashes to what every M1 result was computed under); without it the blocks are printed."""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from r1gym.stats import cluster_paired_diff
from r1gym.env import TRAIN_CLIPS, VAL_CLIPS
CTRL = [("pd_stand", "PD stand"), ("RL v1", "RL v1 (residual)"), ("RL v2", "RL v2 (residual)"), ("RL v3", "RL v3 cand. (residual)"), ("RL step_s1", "Stepping base, stage 1"),
        ("RL step", "Stepping base, arms down (PR #13)"), ("RL guard", "Stepping base + guard (PR #13)"),
        ("RL step_fw", "**Footwork, arms down (shipped)**"), ("RL guard_fw", "**Footwork + guard (shipped)**")]
PRE, NEW = "results/eval_opponent/aim_pre14", "results/eval_opponent/aim"   # as first run (PR #13's bases), and against main's shipped policies
SHIP = ("RL step_fw", "RL guard_fw"); NAME = {"RL step": "arms down (PR #13)", "RL guard": "guard (PR #13)", "RL step_fw": "footwork arms down", "RL guard_fw": "footwork guard"}
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
              f"against PR #13's arms-down base; the unaimed opponent lands {base['landed']['head_clean_pm']['rate']:.1f} head clean strikes/min):", "",
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
        for t, lab in (("mannequin", "mannequin (welded R1: the aim alone)"), ("arms_down", "arms-down base (PR #13)"), ("guard", "guard (PR #13)"), ("arms_down_solref0.02", "arms-down (PR #13), shell solref 0.02")):
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
    for split in ("val", "test"):   # README: stage 1, PR #13's bases and the shipped footwork policies
        for push in ("", "_nohits"):
            u, a = j(f"{NEW}/summary_{split}_hard{push}.json"), j(f"{NEW}/summary_{split}_hard{push}_aim.json")
            if not (u and a): continue
            for k, lab in CTRL[4:]:   # the stepping bases; PD and v1-v3 fall in every hard bout either way (all of them are in the JSON)
                if k not in a["summary"] or "vs_unaimed" not in a["summary"][k]: continue
                U, A, V = u["summary"][k], a["summary"][k], a["summary"][k]["vs_unaimed"]
                rows.append(f"| {split}{', no scripted hits' if push else ''} | {lab} | {U['survival']:.3f} → {A['survival']:.3f} | {ci(V['survival'], '{:+.3f}')} | "
                            f"{U['landed']['head_clean_pm']['rate']:.1f} → {A['landed']['head_clean_pm']['rate']:.1f} | {ci(V['head_clean_pm'])} | "
                            f"{ci(V['opponent_fell'], '{:+.3f}')} | {U['landed']['spacing_m']['median'] if U['landed'].get('spacing_m') else '–'} m |")
    if rows:
        L += [f"**Hard tier with the aimed opponent** (`--aim chosen`: {aim['operator']}, chosen before the footwork policies existed and not re-chosen; `{NEW}/`: the same "
              f"episodes as the unaimed tier, the scheduler in shadow there, paired; nominal distance 0.40-0.55 m):", "",
              "| split | controller | survival, unaimed → aimed | Δ survival [95% CI] | head clean strikes/min, unaimed → aimed | Δ [95% CI] | Δ opponent falls [95% CI] | actual spacing (median, unaimed) |",
              "|---|---|---|---|---|---|---|---|", *rows, ""]
        T, U = j(f"{NEW}/summary_test_hard_aim.json"), j(f"{NEW}/summary_test_hard.json")
        if T:
            V = {k: T["summary"][k]["vs_unaimed"] for k in SHIP}; up = [k for k in V if V[k]["opponent_fell"]["ci95"][0] > 0.02]
            name = lambda k: NAME[k]; of = lambda D, k: D["summary"][k]["opponent_fell"]
            N, M = j(f"{NEW}/summary_test_hard_nohits_aim.json"), j(f"{NEW}/summary_test_hard_nohits.json")
            nh = f" ({', '.join(f'{of(M, k):.1%} → {of(N, k):.1%}' for k in up)} without the scripted hits)" if N and M else ""
            L += [f"On test (run once, after val): head clean strikes/min {ci(V['RL step_fw']['head_clean_pm'])} against the footwork arms-down policy and {ci(V['RL guard_fw']['head_clean_pm'])} "
                  f"against the footwork guard; survival {ci(V['RL step_fw']['survival'], '{:+.3f}')} and {ci(V['RL guard_fw']['survival'], '{:+.3f}')}. PD, the residual policies and PR #13's bases "
                  f"are in the JSON and in `docs/AIMED_OPPONENT.md`, as is the first run against PR #13's bases (`{PRE}/`)."
                  + (f" **The aimed opponent falls more in the hard tier:** in {', '.join(f'{of(U, k):.1%} → {of(T, k):.1%} ({name(k)})' for k in up)} of test bouts{nh}. "
                     f"The preregistered fall rule is judged on the aim-check grid (10 s, nominal physics, 0.40-0.65 m), where it stayed within +2 pp; in the hard tier "
                     f"(20 s, randomised R1, 0.40-0.55 m) it doesn't, with or without the scripted hits, so M1's operator is not fit to be the default opponent." if up else ""), ""]
    return "\n".join(L)

PREREG = ROOT / "docs" / "AIMED_OPPONENT.md"
ASRUN = ["results/opponent_tracker/tolerance.json", "results/opponent_aim/aim_check.json", "results/opponent_tracker/aim.json", "results/opponent_aim/aim_frames.json"] + \
        [f"{PRE}/summary_{s}{t}{a}.json" for s in ("val", "test") for t in ("", "_hard", "_hard_nohits") for a in ("", "_aim")] + [f"{PRE}/summary_{s}_puppet.json" for s in ("val", "test")]
REMEASURED = [f"{NEW}/summary_{s}_hard{t}{a}.json" for s in ("val", "test") for t in ("", "_nohits") for a in ("", "_aim")]
M1_RESULTS = ASRUN + REMEASURED

def prereg_sha():
    """sha256 (first 16 hex) of docs/AIMED_OPPONENT.md, LF line endings, without its '## Results' section: the text the results
    were computed under. The first runs hashed the whole file (prereg_sha), which had no Results section then; later runs also
    record this hash (prereg_rules_sha, scripts/eval_opponent.py provenance())."""
    s = PREREG.read_bytes().replace(b"\r\n", b"\n"); i = s.find(b"\n## Results")
    import hashlib; return hashlib.sha256(s if i < 0 else s[:i]).hexdigest()[:16]

def provenance_rows():
    """(file, hash of the rules it ran under, commit, merge head, dirty) per M1 result."""
    out = []
    for f in M1_RESULTS:
        d = j(f)
        if d is None: continue
        m = d.get("meta", {}); out.append((f, m.get("prereg_rules_sha", m.get("prereg_sha")), (m.get("git_hash") or "")[:7] or None, (m.get("git_merge_head") or "")[:7] or None, m.get("git_dirty")))
    return out

M15_HEAD = ["| split | scripted hits | controller | Δ survival [95% CI] | Δ head clean strikes/min [95% CI] | Δ opponent falls [95% CI] | hit rules with the other sign |",
            "|---|---|---|---|---|---|---|"]

def m15_rows(D, ctrl):
    """M1.5 table rows (aimed minus unaimed, same episodes) from the summaries in directory D."""
    L = []
    for split in ("val", "test"):
        for push, pl in (("", "on"), ("_nohits", "off")):
            a = j(f"{D}/summary_{split}_hard{push}_aim.json")
            if not a: continue
            for k, lb in ctrl:
                if k not in a["summary"]: continue
                V = a["summary"][k]["vs_unaimed"]; sg = V["head_clean_pm"]["diff"] > 0
                fl = [r for r, x in V["rules"].items() if r != "clean" and (x["head"]["diff"] > 0) != sg]
                L.append(f"| {split} | {pl} | {lb} | {ci(V['survival'], '{:+.3f}')} | {ci(V['head_clean_pm'], '{:+.2f}')} | {ci(V['opponent_fell'], '{:+.3f}')} | {', '.join(fl) or '–'} |")
    return L

def same_bouts(k):
    """Per file: are controller k's bouts (every field, hits included) identical in PRE and NEW? (the re-measurement reproduces them)"""
    out = {}
    for split in ("val", "test"):
        for t in ("_hard", "_hard_nohits", "_hard_aim", "_hard_nohits_aim"):
            a, b = j(f"{PRE}/raw_{split}{t}.json"), j(f"{NEW}/raw_{split}{t}.json")
            if a and b and k in a["raw"] and k in b["raw"]: out[f"{split}{t}"] = a["episodes"] == b["episodes"] and a["raw"][k] == b["raw"][k]
    return out

def main_identical(k):
    """Per file: does the re-measured unaimed hard tier match main's stored headline raw file (no hit columns there) in every field it has?"""
    out = {}
    for split in ("val", "test"):
        for t in ("_hard", "_hard_nohits"):
            a, b = j(f"results/eval_opponent/raw_{split}{t}.json"), j(f"{NEW}/raw_{split}{t}.json")
            if a and b and k in a["raw"] and k in b["raw"]:
                out[f"{split}{t}"] = a["episodes"] == b["episodes"] and all(all(x[f] == y[f] for f in x) for x, y in zip(a["raw"][k], b["raw"][k]))
    return out

def doc_block():
    """The results section of docs/AIMED_OPPONENT.md (M1.2-M1.5): every number from the stored JSON."""
    lib = j("r1gym/assets/motions/strikes.json"); tol = j("results/opponent_tracker/tolerance.json"); ac = j("results/opponent_aim/aim_check.json"); aim = j("results/opponent_tracker/aim.json")
    lab = j("r1gym/assets/motions/strike_labels.json"); au, s = lib["audit"], lib["summary"]; d = ac["decision"]; cs = d["candidates"]; op = d["operator"]; nm = dict(OPS)
    pr = [r for r in provenance_rows() if r[0] in ASRUN]; sha = prereg_sha(); bad = [f for f, p, _, _, _ in pr if p != sha]; nogit = [f for f, _, g, _, _ in pr if g is None]
    L = ["### Provenance and the test-once record", "",
         f"- The {len(pr)} result files of the first M1 runs (`ASRUN` in `scripts/aim_tables.py`; the hard tier now archived in `{PRE}/`) carry `prereg_sha` {', '.join(sorted({p for _, p, _, _, _ in pr}))}; this file without this Results section hashes to "
         f"{sha} ({'the same' if not bad else 'DIFFERENT for ' + ', '.join(bad)}). The definitions, rules and amendment 1 above are the text they were computed under.",
         f"- They were run from the uncommitted working tree on {sorted({g for _, _, g, _, _ in pr if g})[0]} (`git_dirty` true"
         + (f"; {', '.join(nogit)} records no commit" if nogit else "") + "): 72ad022, which added this section, is the code that produced them.",
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
    L += [f"### M1.5 hard `--aim` tier, as run against PR #13's stepping bases (archived: `{PRE}/`)", "",
          "Aimed minus unaimed, same episodes. These were the shipped policies when the runs were made; PR #14 replaced them.", "", *M15_HEAD, *m15_rows(PRE, CTRL)]
    T, U = j(f"{PRE}/summary_test_hard_aim.json")["summary"], j(f"{PRE}/summary_test_hard.json")["summary"]; Vv = j(f"{PRE}/summary_val_hard_aim.json")["summary"]
    L += ["", f"- **Stepping bases on test:** arms down {ci(T['RL step']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}, guard "
              f"{ci(T['RL guard']['vs_unaimed']['head_clean_pm'], '{:+.2f}')} head clean strikes/min; on val guard {ci(Vv['RL guard']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}, "
              f"arms down {ci(Vv['RL step']['vs_unaimed']['head_clean_pm'], '{:+.2f}')}. Guard survival on test {ci(T['RL guard']['vs_unaimed']['survival'], '{:+.3f}')}.",
          f"- **Opponent falls** on test: {U['RL step']['opponent_fell']:.1%} → {T['RL step']['opponent_fell']:.1%} (arms down), "
          f"{U['RL guard']['opponent_fell']:.1%} → {T['RL guard']['opponent_fell']:.1%} (guard). M1.5 has no fall rule; rule 2 (aim check) passed, so this is reported, not a rejection.",
          f"- **PD, the residual policies and stage 1**, which fall in most hard bouts, show the aim's effect with opposite signs on val and test; their bouts are short, "
          f"so their rates rest on few minutes. They are reported, not interpreted.", ""]
    N = {f: j(f"{NEW}/summary_{f}.json") for f in ("val_hard", "val_hard_aim", "val_hard_nohits", "val_hard_nohits_aim", "test_hard", "test_hard_aim", "test_hard_nohits", "test_hard_nohits_aim")}
    if all(N.values()):
        nm_ = N["test_hard_aim"]["meta"]; SH = {k: N["test_hard_aim"]["summary"][k]["vs_unaimed"] for k in SHIP}; SV = {k: N["val_hard_aim"]["summary"][k]["vs_unaimed"] for k in SHIP}
        SN = {k: N["test_hard_nohits_aim"]["summary"][k]["vs_unaimed"] for k in SHIP}; UT, AT = N["test_hard"]["summary"], N["test_hard_aim"]["summary"]
        rep = {k: same_bouts(k) for k in ("pd_stand", "RL v1", "RL v2", "RL v3", "RL step_s1", "RL step", "RL guard")}; rep_all = all(v and all(v.values()) for v in rep.values())
        mi = {k: main_identical(k) for k, _ in CTRL}; mi_all = all(v and all(v.values()) for v in mi.values())
        nr = [r for r in provenance_rows() if r[0] in REMEASURED]
        L += [f"### Addendum: re-measured against the PR #14 defenders after merging main; operator unchanged (`{NEW}/`)", "",
              f"Merging main (PR #14's footwork policies, PR #15, PR #16) made the footwork policies the shipped defenders. The hard tier was run again, "
              f"unaimed (scheduler in shadow) and aimed ({nm_['aim']['operator']}, from `aim.json` as fixed on train + val; nothing selected or tuned here), with and without "
              f"scripted hits, against all {len(UT)} controllers of main's headline: val first, then test once. Code: {nm_['git_hash'][:7]} with main {(nm_.get('git_merge_head') or '')[:7]} "
              f"merged in, uncommitted (the merge commit pins it); these {len(nr)} files ran under the rules hash "
              f"{', '.join(sorted({p for _, p, _, _, _ in nr}))} ({'the text above' if all(p == prereg_sha() for _, p, _, _, _ in nr) else 'DIFFERENT from the text above'}). "
              f"The footwork policies run with their `gym.json` as shipped (hold your spot; the guard also turns to face).", "",
              f"- **Same bouts where nothing changed:** PD, v1-v3, stage 1 and PR #13's bases reproduce the archived runs bout for bout in every file, hits included "
              f"({'yes' if rep_all else 'NO: ' + json.dumps(rep)}), and every controller's unaimed bouts match main's stored headline files in every field those have "
              f"({'yes' if mi_all else 'NO: ' + json.dumps(mi)}). The re-measurement differs from the as-run one only in its defenders.", "", *M15_HEAD,
              *m15_rows(NEW, [c for c in CTRL if c[0] in SHIP]), "",
              f"- **Head clean strikes/min, footwork arms down:** test {ci(SH['RL step_fw']['head_clean_pm'], '{:+.2f}')} (without scripted hits {ci(SN['RL step_fw']['head_clean_pm'], '{:+.2f}')}), "
              f"val {ci(SV['RL step_fw']['head_clean_pm'], '{:+.2f}')}; unaimed → aimed on test {UT['RL step_fw']['landed']['head_clean_pm']['rate']:.2f} → {AT['RL step_fw']['landed']['head_clean_pm']['rate']:.2f}.",
              f"- **Footwork guard:** test {ci(SH['RL guard_fw']['head_clean_pm'], '{:+.2f}')} (without scripted hits {ci(SN['RL guard_fw']['head_clean_pm'], '{:+.2f}')}), "
              f"val {ci(SV['RL guard_fw']['head_clean_pm'], '{:+.2f}')}; {UT['RL guard_fw']['landed']['head_clean_pm']['rate']:.2f} → {AT['RL guard_fw']['landed']['head_clean_pm']['rate']:.2f}.",
              f"- **Survival** on test: footwork arms down {UT['RL step_fw']['survival']:.3f} → {AT['RL step_fw']['survival']:.3f} ({ci(SH['RL step_fw']['survival'], '{:+.3f}')}), "
              f"footwork guard {UT['RL guard_fw']['survival']:.3f} → {AT['RL guard_fw']['survival']:.3f} ({ci(SH['RL guard_fw']['survival'], '{:+.3f}')}); up and within 1 m "
              f"{UT['RL step_fw']['survival_in_ring']:.3f} → {AT['RL step_fw']['survival_in_ring']:.3f} and {UT['RL guard_fw']['survival_in_ring']:.3f} → {AT['RL guard_fw']['survival_in_ring']:.3f}.",
              f"- **Opponent falls** on test: {UT['RL step_fw']['opponent_fell']:.1%} → {AT['RL step_fw']['opponent_fell']:.1%} (footwork arms down, {ci(SH['RL step_fw']['opponent_fell'], '{:+.3f}')}), "
              f"{UT['RL guard_fw']['opponent_fell']:.1%} → {AT['RL guard_fw']['opponent_fell']:.1%} (footwork guard, {ci(SH['RL guard_fw']['opponent_fell'], '{:+.3f}')}).",
              f"- Every controller's rows, the other hit rules and the window outcomes are in `{NEW}/summary_*_aim.json` (`vs_unaimed`).", ""]
    L += ["### What this doesn't show", "",
          f"- Most aimed windows can't reach the head from where the opponent stands: on the aim check's arms-down target {ac['summary']['arms_down']['by_mode'][op]['by_ik_residual']['>=8cm']['n']} of "
          f"{ac['summary']['arms_down']['by_mode'][op]['windows']['n']} windows' targets are 8 cm or more beyond the arm, and "
          f"{ac['summary']['arms_down']['by_mode'][op]['windows']['gated_share']:.0%} of windows are not aimed (lean or touch gate). The effect is per minute of bout, not per strike.",
          "- The operator was chosen against PR #13's arms-down base (aim check, train + val) and not re-chosen for the footwork policies.",
          f"- Test holds {s['splits']['test']['strikes']} library strikes, {s['splits']['test']['head_band']} of them level with the head; a test result is one replication on two short clips."]
    return "\n".join(L)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args(); b = block(); db = doc_block()
    if not a.write: sys.stdout.reconfigure(encoding="utf-8"); print(b, db, sep="\n\n"); sys.exit(0)
    def put(p, blocks):   # each generated block sits between <!-- tag:begin --> and <!-- tag:end -->
        s = p.read_text(encoding="utf-8")
        for tag, txt in blocks:
            i, k = s.index(f"<!-- {tag}:begin -->"), s.index(f"<!-- {tag}:end -->"); s = s[:i] + f"<!-- {tag}:begin -->" + txt + s[k:]
        p.write_bytes(s.replace("\r\n", "\n").encode("utf-8"))
    put(ROOT / "README.md", (("aim", "\n" + b + "\n"),)); print("README block written")
    sha = prereg_sha(); put(PREREG, (("results", "\n" + db + "\n"),))
    assert prereg_sha() == sha and all(p == sha for _, p, _, _, _ in provenance_rows()), "docs/AIMED_OPPONENT.md above '## Results' changed since the runs"
    print("docs/AIMED_OPPONENT.md results written; preregistration text unchanged:", sha)
