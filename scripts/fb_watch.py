"""WATCH6: the full-body M2 v2 pod's early stops (M2V2_PLAN_v3 section 4 'WATCH6 early stops'; m6, m15, m16, B1 fix 5), on an rsl_rl
training log of gpu/fullbody_task.py. Checks at iterations 50, 100, 200, then every 100 (counted from the log's first complete block,
as the run-3 pod counted), on 20-iteration windows. A stopped sighted run is exported `_stopped` (a stopped stage B falls back to the
stage-A final, the pod's job); the blind twin is never stopped (its reasons are reported).
Counters: Metrics/opp/<COUNTERS> are sums since the previous reset batch (fullbody_task.FbOppCommand.reset), so a window's ratio of the
logged means is the ratio of totals; Episode_Metrics/<ACTOR> are the actor's std, leg deviation and |residual| (micro units).
| rule | stop the sighted run if |
| E0 load | first iteration != start_it; arm std != the donor's +- 0.01 (first block); leg_dev > 1e-5 in A0 / A; residual != 0 at B start
|          | (first block); act_delay share outside 0.50 +- 0.05; a k share outside 1/3 +- 0.05; any within-episode lag switch; T0 share
|          | outside 0.50 +- 0.03 (1.00 while the window ends at or before iteration 100); command_counter != 1 at an episode end; the
|          | clip-manifest check < 1 |
| E2 hazard | from 200: eval-rule falls + ring_out per live s of no-push envs > 3 x the run's own iterations 1-100 (every run, B's restart
|           | included, draws T0 only there: fullbody_task.fb_draw_population; --base-log overrides) |
| E3 std | arm std > 0.80; leg std > 0.35 (B) |
| E4 progress | from A-500 / B-1000: clean head hits per live minute > 0.97 x the run's own iterations 101-120 |
| E5 knockdowns | knockdowns per live minute of no-push envs > max(1.5 x base, base + 0.15), base = iterations 1-100 (from the first check
|               | where the base is complete, or --base-log's) |
| E6 opponent / stance | opponent no-contact self-falls per live s > 2 x the T-opp baseline (--topp, M6); in B: lean_away mean > 6 deg;
|                      | tangle N s per live minute > 1.3 x the A final (--a-final-log's last window); fb_down_approach or fb_down_contact
|                      | per downed second > 1.5 x the A final |
| E7 eval probe | --probe JSON (EO.run on 63000-63383): survival < 0.94; R vs G > 0.95 at A-1500, > 0.85 from B-2500 |
| E8 keypoint use | --probe at B-2500: R with zeroed keypoints < 0.90 x R with true keypoints (reported at B-1000) |
A rule whose input is missing (no baseline, no counter in the log) is reported 'n/a'; a required input missing for an evaluated stage
(--topp; --a-final-log in B) is a config error (exit 2), never a silent pass.
CLI: python scripts/fb_watch.py watch --rule 6 --log train.log --at 200 --stage A [--blind] [--start-it N] [--donor-arm-std S]
     [--base-log A.log] [--a-final-log A.log] [--topp R] [--probe probe.json]   (exit 3 = stop, 2 = config error, 0 = continue)
     python scripts/fb_watch.py replays [--logs DIR]   (M9 (3))"""
import re, sys, json, math, argparse
from pathlib import Path

COUNTERS = ("w_live_s", "w_live_s_np", "w_term_np", "w_kd_np", "w_head_clean", "w_eps", "w_t0", "w_ad", "w_k0", "w_k1", "w_k2", "w_switch",
            "w_cmd_bad", "w_opp_nc_fall", "w_lean_deg_s", "w_tangle", "w_down_s", "w_down_app", "w_down_con", "w_manifest_ok")   # = fullbody_task.WATCH_KEYS
ACTOR = ("fb_arm_std", "fb_leg_std", "fb_leg_dev_u", "fb_res_abs_u")                                                          # = fullbody_task.ACTOR_KEYS
CPFX, APFX = "Metrics/opp/", "Episode_Metrics/"
W6 = dict(window=20, checks=(50, 100) + tuple(range(200, 6001, 100)), arm_std_tol=0.01, leg_dev=1e-5, ad=(0.5, 0.05), k=(1 / 3, 0.05), t0=(0.5, 0.03), t0_only_until=100,
          hazard_x=3.0, hazard_from=200, arm_std_max=0.80, leg_std_max=0.35, prog_ratio=0.97, prog_from=dict(A=500, B=1000), prog_ref=(100, 120), kd_x=1.5, kd_add=0.15,
          base=(0, 100), topp_x=2.0, lean_max=6.0, tangle_x=1.3, down_x=1.5, probe_surv=0.94, probe_R=dict(A1500=0.95, B2500=0.85), kp_use=0.90)
STAGES = ("A0", "A", "B")
NAN = float("nan")

def parse(path):
    """[(iteration, {key: float})] of the complete blocks (with 'Iteration time:'), ANSI stripped."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(errors="replace")); parts = re.split(r"Learning iteration\s+(\d+)/\d+", txt); out = []
    for it, b in zip(parts[1::2], parts[2::2]):
        if "Iteration time:" not in b: continue
        kv = dict((k.strip(), float(v)) for k, v in re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_/ \-]*?):\s+(-?[0-9.]+(?:[eE][-+]?\d+)?)s?\s*$", b, re.M)); out.append((int(it), kv))
    return out

def tot(blocks, k): return sum(d.get(CPFX + k, 0.0) for _, d in blocks)
def has(blocks, k): return any(CPFX + k in d for _, d in blocks)
def ratio(blocks, a, b, scale=1.0):
    if not (has(blocks, a) and has(blocks, b)): return NAN
    d = tot(blocks, b); return tot(blocks, a) / d * scale if d > 0 else NAN
def amax(blocks, k): v = [d[APFX + k] for _, d in blocks if APFX + k in d]; return max(v) if v else NAN
def alast(blocks, k): v = [d[APFX + k] for _, d in blocks if APFX + k in d]; return v[-1] if v else NAN

def watch6(log, at, stage="A", blind=False, start_it=None, donor_arm_std=None, base_log=None, a_final_log=None, topp=None, probe=None, cfg=W6, blocks=None):
    """WATCH6 at check iteration `at` (module docstring) -> dict(ready, stop, reasons, values, na, error)."""
    assert stage in STAGES; B = blocks if blocks is not None else parse(log); w = cfg["window"]
    out = dict(log=str(log), at=at, stage=stage, blind=blind, iterations=len(B), window=w, ready=len(B) >= at >= w, stop=False, reasons=[], na=[], error=[], values={})
    if not out["ready"]: return out
    win, first, v = B[at - w:at], B[:1], out["values"]; R = []
    def rule(name, bad, txt):
        if isinstance(bad, float) and math.isnan(bad): out["na"].append(name)
        elif bad: R.append(f"{name} {txt}")
    base = parse(base_log)[cfg["base"][0]:cfg["base"][1]] if base_log else (B[cfg["base"][0]:cfg["base"][1]] if len(B) >= cfg["base"][1] else None)
    if stage == "B" and not a_final_log: out["error"].append("stage B needs --a-final-log (E6's A final)")
    if topp is None: out["error"].append("--topp (E6's T-opp baseline from M6) missing")
    # E0 load
    if start_it is not None: v["first_iteration"] = B[0][0]; rule("E0 start", B[0][0] != start_it, f"first iteration {B[0][0]} != {start_it}")
    a0 = alast(first, "fb_arm_std"); v["arm_std_first"] = a0
    if donor_arm_std is not None: rule("E0 arm_std", NAN if math.isnan(a0) else abs(a0 - donor_arm_std) > cfg["arm_std_tol"], f"{a0:.4f} != donor {donor_arm_std:.4f} +- {cfg['arm_std_tol']}")
    if stage in ("A0", "A"): ld = amax(win, "fb_leg_dev_u") * 1e-6; v["leg_dev"] = ld; rule("E0 leg_dev", NAN if math.isnan(ld) else ld > cfg["leg_dev"], f"{ld:.3g} > {cfg['leg_dev']}")
    if stage == "B": r0 = alast(first, "fb_res_abs_u"); v["residual_at_start_u"] = r0; rule("E0 residual", NAN if math.isnan(r0) else r0 != 0.0, f"|r| {r0} micro != 0 at B start")
    ad = ratio(win, "w_ad", "w_eps"); v["act_delay_share"] = ad; rule("E0 act_delay", NAN if math.isnan(ad) else abs(ad - cfg["ad"][0]) > cfg["ad"][1], f"share {ad:.4f} outside {cfg['ad'][0]} +- {cfg['ad'][1]}")
    for i in range(3):
        ks = ratio(win, f"w_k{i}", "w_eps"); v[f"k{i}_share"] = ks; rule(f"E0 k{i}", NAN if math.isnan(ks) else abs(ks - cfg["k"][0]) > cfg["k"][1], f"share {ks:.4f} outside 1/3 +- {cfg['k'][1]}")
    sw = tot(win, "w_switch") if has(win, "w_switch") else NAN; v["lag_switches"] = sw; rule("E0 lag_switch", NAN if math.isnan(sw) else sw > 0, f"{sw:g} within-episode switches")
    t0 = ratio(win, "w_t0", "w_eps"); v["t0_share"] = t0; only = at <= cfg["t0_only_until"]
    rule("E0 t0_share", NAN if math.isnan(t0) else (t0 != 1.0 if only else abs(t0 - cfg["t0"][0]) > cfg["t0"][1]), f"{t0:.4f} " + ("!= 1.0 before iteration 100" if only else f"outside {cfg['t0'][0]} +- {cfg['t0'][1]}"))
    cb = tot(win, "w_cmd_bad") if has(win, "w_cmd_bad") else NAN; v["command_counter_bad"] = cb; rule("E0 command_counter", NAN if math.isnan(cb) else cb > 0, f"{cb:g} episodes with command_counter != 1")
    mo = [d[CPFX + "w_manifest_ok"] for _, d in win if CPFX + "w_manifest_ok" in d]; v["manifest_ok_min"] = min(mo) if mo else NAN
    rule("E0 manifest", NAN if not mo else min(mo) < 1.0, "clip-manifest hash mismatch")
    # E2 hazard (no-push envs)
    hz = ratio(win, "w_term_np", "w_live_s_np"); hb = ratio(base, "w_term_np", "w_live_s_np") if base else NAN; v.update(hazard=hz, hazard_base=hb)
    if at >= cfg["hazard_from"]: rule("E2 hazard", NAN if math.isnan(hz) or math.isnan(hb) else hz > cfg["hazard_x"] * hb, f"{hz:.5f}/s > {cfg['hazard_x']} x base {hb:.5f}/s")
    # E3 std
    asd, lsd = alast(win, "fb_arm_std"), alast(win, "fb_leg_std"); v.update(arm_std=asd, leg_std=lsd)
    rule("E3 arm_std", NAN if math.isnan(asd) else asd > cfg["arm_std_max"], f"{asd:.4f} > {cfg['arm_std_max']}")
    if stage == "B": rule("E3 leg_std", NAN if math.isnan(lsd) else lsd > cfg["leg_std_max"], f"{lsd:.4f} > {cfg['leg_std_max']}")
    # E4 progress
    pf = cfg["prog_from"].get(stage)
    if pf is not None and at >= pf:
        p1 = ratio(win, "w_head_clean", "w_live_s", 60.0); p0 = ratio(B[cfg["prog_ref"][0]:cfg["prog_ref"][1]], "w_head_clean", "w_live_s", 60.0); v.update(head_clean_pm=p1, head_clean_pm_ref=p0)
        rule("E4 progress", NAN if math.isnan(p1) or math.isnan(p0) else p1 > cfg["prog_ratio"] * p0, f"{p1:.4f}/live min > {cfg['prog_ratio']} x its 101-120 rate {p0:.4f}")
    # E5 knockdowns (no-push envs)
    kd = ratio(win, "w_kd_np", "w_live_s_np", 60.0); kb = ratio(base, "w_kd_np", "w_live_s_np", 60.0) if base else NAN; v.update(kd_pm=kd, kd_pm_base=kb)
    if base: rule("E5 knockdowns", NAN if math.isnan(kd) or math.isnan(kb) else kd > max(cfg["kd_x"] * kb, kb + cfg["kd_add"]), f"{kd:.4f}/live min > max({cfg['kd_x']} x {kb:.4f}, {kb:.4f} + {cfg['kd_add']})")
    # E6 opponent health and stance
    of = ratio(win, "w_opp_nc_fall", "w_live_s"); v["opp_nc_fall_per_s"] = of
    if topp is not None: rule("E6 opponent", NAN if math.isnan(of) else of > cfg["topp_x"] * topp, f"no-contact self-falls {of:.5f}/s > {cfg['topp_x']} x T-opp {topp:.5f}/s")
    if stage == "B":
        lean = ratio(win, "w_lean_deg_s", "w_live_s"); v["lean_away_deg"] = lean; rule("E6 lean", NAN if math.isnan(lean) else lean > cfg["lean_max"], f"{lean:.3f} deg > {cfg['lean_max']}")
        if a_final_log:
            AF = parse(a_final_log)[-w:]; ta, tf = ratio(win, "w_tangle", "w_live_s", 60.0), ratio(AF, "w_tangle", "w_live_s", 60.0); v.update(tangle_pm=ta, tangle_pm_A_final=tf)
            rule("E6 tangle", NAN if math.isnan(ta) or math.isnan(tf) else ta > cfg["tangle_x"] * tf, f"{ta:.3f} N s/live min > {cfg['tangle_x']} x A final {tf:.3f}")
            for k, nm in (("w_down_app", "fb_down_approach"), ("w_down_con", "fb_down_contact")):
                x, xf = ratio(win, k, "w_down_s"), ratio(AF, k, "w_down_s"); v[nm + "_per_down_s"] = x; v[nm + "_per_down_s_A_final"] = xf
                rule(f"E6 {nm}", NAN if math.isnan(x) or math.isnan(xf) else x > cfg["down_x"] * xf, f"{x:.4f} per downed s > {cfg['down_x']} x A final {xf:.4f}")
    # E7 / E8 eval probe (pod CPU, background EO.run)
    if probe:
        P = json.loads(Path(probe).read_text()) if not isinstance(probe, dict) else probe; tag = f"{P['stage']}{P['iteration']}"; v["probe"] = P
        rule("E7 survival", P["survival"] < cfg["probe_surv"], f"{P['survival']:.4f} < {cfg['probe_surv']} at {tag}")
        lim = cfg["probe_R"]["A1500"] if tag == "A1500" else cfg["probe_R"]["B2500"] if P["stage"] == "B" and P["iteration"] >= 2500 else None
        if lim is not None: rule("E7 R_vs_G", P["R_vs_G"] > lim, f"R vs G {P['R_vs_G']:.4f} > {lim} at {tag}")
        if P["stage"] == "B" and P.get("R_zeroed_kp") is not None:
            kp = P["R_zeroed_kp"] < cfg["kp_use"] * P["R_true_kp"]; v["E8_zeroed_beats_true"] = bool(kp)
            if P["iteration"] == 2500: rule("E8 keypoint_use", kp, f"zeroed keypoints R {P['R_zeroed_kp']:.4f} < {cfg['kp_use']} x true {P['R_true_kp']:.4f}")
    out["reasons"] = R; out["stop"] = bool(R) and not blind and not out["error"]; return out

# ---- synthetic logs (M9 (3))
def block(it, total, d):
    """One rsl_rl log block (rsl_rl 5.0.1 Logger.log's layout, pad 40)."""
    pad = 40; L = ["#" * 80, f" Learning iteration {it}/{total} ".center(80), "", f"{'Mean action std:':>{pad}} {d.get('std', 0.7):.2f}", f"{'Mean episode length:':>{pad}} 600.00"]
    L += [f"{(k + ':'):>{pad}} {x:.4f}" for k, x in d.items() if "/" in k]; L += ["-" * 80, f"{'Iteration time:':>{pad}} 3.50s", ""]; return "\n".join(L)

def healthy(i, stage="A", eps=200.0):
    """A healthy fullbody iteration i (0-based) of a stage: exact shares, steady rates (per reset batch: 25 batches per iteration)."""
    n = 25.0; live = 4096 * 24 * 0.02 / n; t0 = 1.0 if i < 100 else 0.5   # iterations 1-100 of every run: T0 only
    c = dict(w_live_s=live, w_live_s_np=live * 0.6, w_term_np=live * 0.6 * 0.004, w_kd_np=live * 0.6 * 0.004, w_head_clean=live * 0.06 * (1.0 if i < 120 else 0.9), w_eps=eps / n,
             w_t0=eps / n * t0, w_ad=eps / n * 0.5, w_k0=eps / n / 3, w_k1=eps / n / 3, w_k2=eps / n / 3, w_switch=0.0, w_cmd_bad=0.0, w_opp_nc_fall=live * 0.0005,
             w_lean_deg_s=live * 4.0, w_tangle=live * 0.3, w_down_s=live * 0.02, w_down_app=live * 0.02 * 0.1, w_down_con=live * 0.02 * 0.2, w_manifest_ok=1.0)
    a = dict(fb_arm_std=0.64, fb_leg_std=0.15 if stage == "B" else 0.0, fb_leg_dev_u=0.0, fb_res_abs_u=0.0 if (stage != "B" or i == 0) else 1200.0)
    return {**{CPFX + k: x for k, x in c.items()}, **{APFX + k: x for k, x in a.items()}}

def synth(path, n, stage="A", start=0, mutate=None):
    """Write an n-iteration synthetic log; mutate(i, d) edits iteration i's values in place."""
    out = []
    for i in range(n):
        d = healthy(i, stage)
        if mutate: mutate(i, d)
        out.append(block(start + i, start + n, d))
    Path(path).write_text("\n".join(out), encoding="utf-8"); return path

def _set(k, f, frm=0):
    def m(i, d):
        if i >= frm: d[k] = f(d.get(k, 0.0))
    return m

STOPS = {   # rule -> (stage, mutate, iteration it must stop at or before)
    "E0 start": ("A", None, 50), "E0 arm_std": ("A", _set(APFX + "fb_arm_std", lambda x: x + 0.02), 50), "E0 leg_dev": ("A", _set(APFX + "fb_leg_dev_u", lambda x: 20.0, 30), 50),
    "E0 residual": ("B", _set(APFX + "fb_res_abs_u", lambda x: 3.0), 50), "E0 act_delay": ("A", _set(CPFX + "w_ad", lambda x: x * 1.12, 130), 200),
    "E0 k0": ("A", lambda i, d: d.update({CPFX + "w_k0": d[CPFX + "w_k0"] * 1.2, CPFX + "w_k1": d[CPFX + "w_k1"] * 0.9, CPFX + "w_k2": d[CPFX + "w_k2"] * 0.9}) if i >= 130 else None, 200),
    "E0 lag_switch": ("A", _set(CPFX + "w_switch", lambda x: 1.0, 170), 200), "E0 t0_share": ("A", _set(CPFX + "w_t0", lambda x: x * 0.9, 130), 200),
    "E0 command_counter": ("A", _set(CPFX + "w_cmd_bad", lambda x: 1.0, 260), 300), "E0 manifest": ("A", _set(CPFX + "w_manifest_ok", lambda x: 0.0, 10), 50),
    "E2 hazard": ("A", _set(CPFX + "w_term_np", lambda x: x * 3.5, 150), 200), "E3 arm_std": ("A", _set(APFX + "fb_arm_std", lambda x: 0.82, 380), 400),
    "E3 leg_std": ("B", _set(APFX + "fb_leg_std", lambda x: 0.36, 380), 400), "E4 progress": ("A", _set(CPFX + "w_head_clean", lambda x: x / 0.9 * 0.98, 120), 500),
    "E5 knockdowns": ("A", _set(CPFX + "w_kd_np", lambda x: x * 2.5, 150), 200), "E6 opponent": ("A", _set(CPFX + "w_opp_nc_fall", lambda x: x * 2.5, 150), 200),
    "E6 lean": ("B", _set(CPFX + "w_lean_deg_s", lambda x: x * 1.6, 150), 200), "E6 tangle": ("B", _set(CPFX + "w_tangle", lambda x: x * 1.4, 150), 200),
    "E6 fb_down_approach": ("B", _set(CPFX + "w_down_app", lambda x: x * 1.6, 150), 200), "E6 fb_down_contact": ("B", _set(CPFX + "w_down_con", lambda x: x * 1.6, 150), 200)}
PROBES = {"E7 survival": dict(stage="A", iteration=1500, survival=0.93, R_vs_G=0.90), "E7 R_vs_G": dict(stage="B", iteration=2500, survival=0.97, R_vs_G=0.86),
          "E8 keypoint_use": dict(stage="B", iteration=2500, survival=0.97, R_vs_G=0.80, R_zeroed_kp=0.70, R_true_kp=0.80)}
TOPP = 0.0005 * 1.0   # the synthetic T-opp baseline (healthy w_opp_nc_fall per live s)

def run_checks(path, stage, n, blind=False, start_it=0, base=None, afinal=None, probe=None, donor=0.64, upto=None):
    """watch6 at every check up to `upto` (default n) -> [(at, out)]."""
    B = parse(path); upto = upto or n; return [(at, watch6(path, at, stage, blind, start_it, donor, base, afinal, TOPP, probe, blocks=B)) for at in W6["checks"] if at <= min(upto, len(B))]

def replays(logs=None, tmp=None):
    """M9 (3): every synthetic stop log stops for its own rule (first stopping check <= its deadline, and that rule among the reasons);
    healthy A (1600 iterations) and B (4000) never stop; a blind twin with a stopping rule is never stopped (reasons reported); the run-3
    logs (sighted A3s / A3, blind3) trip no E0 rule at any check (the keys they lack are n/a)."""
    import tempfile
    tmp = Path(tmp or tempfile.mkdtemp(prefix="fbw6_")); res = dict(stops={}, healthy={}, run3={}, blind={})
    hA, hB = synth(tmp / "hA.log", 1600, "A"), synth(tmp / "hB.log", 4000, "B", start=1600)
    for nm, p, st, n, base in (("A", hA, "A", 1600, None), ("B", hB, "B", 4000, hA)):
        cs = run_checks(p, st, n, start_it=0 if st == "A" else 1600, base=base, afinal=hA if st == "B" else None)
        res["healthy"][nm] = dict(checks=len(cs), stops=[(at, o["reasons"]) for at, o in cs if o["stop"]], errors=sorted({e for _, o in cs for e in o["error"]}), na=sorted({x for _, o in cs for x in o["na"]}))
    for rule, (st, mut, dl) in STOPS.items():
        n = dl + 40; p = synth(tmp / f"s_{rule.replace(' ', '_')}.log", n, st, start=0 if st == "A" else 1600, mutate=mut)
        cs = run_checks(p, st, n, start_it=(1 if rule == "E0 start" else 0) if st == "A" else 1600, base=hA if st == "B" else None, afinal=hA if st == "B" else None)
        hit = [at for at, o in cs if o["stop"]]; own = [at for at, o in cs if any(r.startswith(rule) for r in o["reasons"])]
        res["stops"][rule] = dict(first_stop=hit[0] if hit else None, first_own=own[0] if own else None, deadline=dl, reasons=next((o["reasons"] for at, o in cs if o["stop"]), []),
                                  ok=bool(hit and own and own[0] == hit[0] <= dl))
    for rule, P in PROBES.items():
        st = P["stage"]; p = hB if st == "B" else hA; o = watch6(p, 1500 if st == "A" else 2500, st, False, 0 if st == "A" else 1600, 0.64, hA if st == "B" else None, hA if st == "B" else None, TOPP, P)
        res["stops"][rule] = dict(first_stop=o["at"] if o["stop"] else None, reasons=o["reasons"], ok=o["stop"] and any(r.startswith(rule) for r in o["reasons"]))
    st, mut, dl = STOPS["E3 arm_std"]; pb = synth(tmp / "blind_stop.log", dl + 40, st, mutate=mut); cs = run_checks(pb, st, dl + 40, blind=True, start_it=0)
    res["blind"] = dict(stops=sum(o["stop"] for _, o in cs), reported=sum(bool(o["reasons"]) for _, o in cs), ok=not any(o["stop"] for _, o in cs) and any(o["reasons"] for _, o in cs))
    D = Path(logs) if logs else None
    for nm, bl in (("spar3_A3s.log", False), ("spar3_A3.log", False), ("spar3_blind3.log", True)):
        f = D / nm if D else None
        if not f or not f.exists(): res["run3"][nm] = dict(missing=True); continue
        B = parse(f); cs = [(at, watch6(f, at, "A", bl, B[0][0], None, None, None, TOPP, None, blocks=B)) for at in W6["checks"] if at <= len(B)]
        e0 = [(at, [r for r in o["reasons"] if r.startswith("E0")]) for at, o in cs]; e0 = [x for x in e0 if x[1]]
        res["run3"][nm] = dict(blocks=len(B), first_iteration=B[0][0], checks=len(cs), e0_trips=e0, stops=sum(o["stop"] for _, o in cs), other_reasons=sorted({r.split(" ")[0] + " " + r.split(" ")[1] for _, o in cs for r in o["reasons"] if not r.startswith("E0")}),
                                na=sorted({x for _, o in cs for x in o["na"]}), ok=not e0)
    res["ok"] = (all(v["ok"] for v in res["stops"].values()) and all(not v["stops"] and not v["errors"] for v in res["healthy"].values()) and res["blind"]["ok"]
                 and all(v.get("ok", False) for v in res["run3"].values()) and len(res["run3"]) == 3); return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["watch", "replays"]); ap.add_argument("--rule", type=int, default=6); ap.add_argument("--log"); ap.add_argument("--at", type=int)
    ap.add_argument("--stage", default="A", choices=STAGES); ap.add_argument("--blind", action="store_true"); ap.add_argument("--start-it", type=int); ap.add_argument("--donor-arm-std", type=float)
    ap.add_argument("--base-log"); ap.add_argument("--a-final-log"); ap.add_argument("--topp", type=float); ap.add_argument("--probe"); ap.add_argument("--logs"); ap.add_argument("--out"); a = ap.parse_args()
    if a.cmd == "replays": r = replays(a.logs); code = 0 if r["ok"] else 1
    else:
        assert a.rule == 6, "this file is WATCH6 (gpu/sparring_task.py keeps run 2b / run 3's rules)"
        r = watch6(a.log, a.at, a.stage, a.blind, a.start_it, a.donor_arm_std, a.base_log, a.a_final_log, a.topp, a.probe); code = 2 if r["error"] else 3 if r["stop"] else 0
    s = json.dumps(r, indent=1, default=str); print(s[:6000])
    if a.out: Path(a.out).write_text(s, encoding="utf-8")
    sys.exit(code)

if __name__ == "__main__": main()
