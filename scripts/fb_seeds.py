"""Held-out guard for the full-body M2 v2 work (M2V2_PLAN_v3 3 M0, 6.4; CRITIQUE_M2V2 B7): seed roles, the seed ledger,
clip sets, and the banned-input scan of every fullbody task input.

check(seeds, role, clips, guard=None) is called by every new harness before it simulates anything:
  - every role refuses hard_r2 val 21000-21999; hard_r2 test 22000-22511 only for role 'test', and only with a granted TestOnce;
  - 23000-23255 (hard_r4 val, the val clip) only for 'screen_val'; 24000-25999 (Track 2 val2/test2) only for 'track2';
  - every seed lies in its role's ledger block (6.4), and the clip set is the role's: dev roles train clips only (LOCO roles
    their own subset), 'screen_val' the val clip only, 'test' the test clips only, 'track2' none of the duan-6 routine.
LEDGER blocks are disjoint from each other and from the blocks used before v3 (USED, RETIRED), except the two reuses 6.4
declares (REUSE). audit(path) reads SEED_LEDGER.md: every row is a USED block or lies inside its role's block.
scan(paths) finds banned names in task inputs: the val/test/quarantined clips, the all-clip tracker motion files, the
tracker ONNX (it bakes every clip's reference). task_inputs() lists the fullbody task's code and data files.
CLI: python scripts/fb_seeds.py [--audit SEED_LEDGER.md] [--scan]"""
import sys, re, json, hashlib, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from r1gym.env import TRAIN_CLIPS, VAL_CLIPS, TEST_CLIPS

TRAIN, VAL, TEST = frozenset(TRAIN_CLIPS), frozenset(VAL_CLIPS), frozenset(TEST_CLIPS)
DUAN6 = frozenset(f"kfab_{i}" for i in range(199, 207))   # Track 2 excludes the routine of the kfab clips (IDs 199-206)
VAL_R2, TEST_R2, SCREEN_VAL, TRACK2 = (21000, 21999), (22000, 22511), (23000, 23255), (24000, 25999)
# role: (blocks, clip rule): ("sub", S) clips a non-empty subset of S; ("not", S) clips disjoint from S. 6.4, one ledger
ROLES = {"record": ([(60000, 60015)], ("sub", TRAIN)),          # M2 recording
         "identity": ([(60100, 60115)], ("sub", TRAIN)),        # M6 R1
         "hit_scoring": ([(60200, 60263)], ("sub", TRAIN)),     # M5 G0-11
         "w0_calib": ([(60300, 60363)], ("sub", TRAIN)),        # M5 G0-12
         "t9": ([(60400, 60415)], ("sub", TRAIN)),              # M7 T9
         "l1_perf": ([(60500, 60599)], ("sub", TRAIN)),         # M8 L1 performability
         "e7_probe": ([(63000, 63383)], ("sub", TRAIN)),        # E7 fixed paired probe set
         "selection": ([(64000, 65535)], ("sub", TRAIN)),       # section 5
         "smoke": ([(69000, 69099)], ("sub", TRAIN)),           # M10 plumbing smoke
         "canary": ([(69100, 69999)], ("sub", TRAIN)),          # M10 learning canary
         "eq": ([(70000, 82287)], ("sub", TRAIN)),              # M6 R0-EO + R-EQ, P1-R4g's EO side; 70000-76143 is T1-c's set
         "loco_heldout": ([(88000, 89151)], ("sub", frozenset({"kfab_203"}))),
         "loco_seen": ([(89500, 90651)], ("sub", frozenset({"kfab_200", "kfab_201"}))),
         "reserve": ([(100000, 112287)], ("sub", TRAIN)),       # 3.2 rerun reserve
         "r2_diag": ([(67000, 67767)], ("sub", TRAIN)),         # M6 R2 diagnostics: step 0 is R4-pre's main block (REUSE)
         "s3": ([(56000, 56191)], ("sub", TRAIN)),              # screen S3
         "screen_val": ([SCREEN_VAL], ("sub", VAL)),            # screen S1/S2, once, after the pod
         "track2": ([TRACK2], ("not", DUAN6)),                  # only if Track 2 is approved
         "test": ([TEST_R2], ("sub", TEST))}                    # once, through TestOnce
DEV = frozenset(r for r in ROLES if r not in ("screen_val", "track2", "test"))
# blocks simulated before v3 (6.4, SEED_LEDGER.md, RUN4_PLAN_v3 3.7, P1/P2b notes); 51000-51999 covers "51000-51xxx"
USED = [(50000, 50255), (51000, 51999), (52000, 52255), (53000, 53191), (54000, 54255), (55000, 55255), (55500, 55691),
        (58100, 58143), (59000, 59255), (66000, 66047), (66100, 66291), (67000, 67767), (68000, 68767)]
RETIRED = [(57000, 58071)]                       # RUN4_PLAN_v3 E7 reservations, never used
LEGACY = (50000, 59255)                          # "50000-59255 (various)": no new block in it except s3 (reserved there in RUN4)
REUSE = {"r2_diag": (67000, 67767)}              # planned reuse of a used block (6.4)

class SeedRefused(Exception): pass

def _in(s, b): return b[0] <= s <= b[1]
def overlap(a, b): return a[0] <= b[1] and b[0] <= a[1]

class TestOnce:
    """The only way into role 'test' (6.3): the fb prereg exists, the selection says go = true and was made against the
    prereg's current LF-normalised hash (m2_verdict's text_sha16), and no hard_r2 test raw file exists under the eval output
    root (the test split runs once across runs, eval_opponent.r2_test_spent). M11 wires it to m2_verdict --run fb."""
    def __init__(self, selection, prereg=ROOT / "results" / "sparring" / "prereg_m2_fb.json", eval_root=ROOT / "results" / "eval_opponent"):
        self.granted = False; prereg, selection = Path(prereg), Path(selection)
        if not prereg.exists(): raise SeedRefused(f"test refused: no prereg {prereg}")
        if not selection.exists(): raise SeedRefused(f"test refused: no selection {selection}")
        S = json.loads(selection.read_text(encoding="utf-8")); h = hashlib.sha256(prereg.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
        if S.get("go") is not True: raise SeedRefused(f"test refused: {selection.name} go is {S.get('go')!r}")
        if S.get("prereg_sha256_16") != h: raise SeedRefused(f"test refused: selection made against prereg hash {S.get('prereg_sha256_16')}, prereg is now {h}")
        spent = sorted(str(p) for p in Path(eval_root).rglob("raw_test_hard_r2*.json")) if Path(eval_root).exists() else []
        if spent: raise SeedRefused(f"test refused: the hard_r2 test already ran into {spent}")
        self.granted = True

def check(seeds, role, clips, guard=None):
    """Sorted seeds if `role` may simulate them on `clips`; else SeedRefused naming the first reason."""
    if role not in ROLES: raise SeedRefused(f"unknown role {role!r}")
    raw = np.atleast_1d(np.asarray(seeds)); clips = frozenset(clips)
    if raw.size == 0: raise SeedRefused("no seeds")
    if raw.dtype.kind not in "iuf" or (raw.dtype.kind == "f" and not np.all(raw == np.floor(raw))): raise SeedRefused(f"{role}: seeds must be integers, got {raw[:4]}")
    seeds = [int(s) for s in raw]
    for s in seeds:
        if _in(s, VAL_R2): raise SeedRefused(f"{role}: seed {s} is hard_r2 val (21000-21999): never simulated")
        if _in(s, TEST_R2) and role != "test": raise SeedRefused(f"{role}: seed {s} is the hard_r2 test (22000-22511)")
        if _in(s, SCREEN_VAL) and role != "screen_val": raise SeedRefused(f"{role}: seed {s} is the screen val block (23000-23255)")
        if _in(s, TRACK2) and role != "track2": raise SeedRefused(f"{role}: seed {s} is Track 2's val2/test2 (24000-25999)")
        if not any(_in(s, b) for b in ROLES[role][0]): raise SeedRefused(f"{role}: seed {s} is outside its ledger block {ROLES[role][0]}")
    if role == "test" and not (isinstance(guard, TestOnce) and getattr(guard, "granted", False)): raise SeedRefused("test: only through a granted TestOnce")
    kind, S = ROLES[role][1]
    if kind == "sub" and not (clips and clips <= S): raise SeedRefused(f"{role}: clips {sorted(clips)} are not a non-empty subset of {sorted(S)}")
    if kind == "not" and (not clips or clips & S): raise SeedRefused(f"{role}: clips {sorted(clips)} touch {sorted(clips & S) or 'nothing (empty)'}")
    return sorted(seeds)

def ledger_problems():
    """Disjointness of the 6.4 ledger: role blocks pairwise, against USED/RETIRED (but the declared REUSE), and the legacy zone."""
    out, R = [], [(r, b) for r, (bl, _) in ROLES.items() for b in bl]
    for i, (r1, a) in enumerate(R):
        for r2, b in R[i + 1:]:
            if overlap(a, b): out.append(f"{r1} {a} overlaps {r2} {b}")
        for u in USED + RETIRED:
            if overlap(a, u) and REUSE.get(r1) != u: out.append(f"{r1} {a} overlaps used/retired {u}")
        if overlap(a, LEGACY) and r1 != "s3": out.append(f"{r1} {a} lies in the legacy zone {LEGACY}")
    return out

ROW = re.compile(r"^\|\s*\d{4}-\d\d-\d\d\s*\|\s*(\d+)-(\d+)")
def audit(path):
    """Problems with SEED_LEDGER.md: a row block that is neither a USED block nor inside one role block."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        m = ROW.match(ln)
        if not m: continue
        b = (int(m.group(1)), int(m.group(2)))
        if b in USED: continue
        own = [r for r, (bl, _) in ROLES.items() if any(x[0] <= b[0] and b[1] <= x[1] for x in bl)]
        if len(own) != 1: out.append(f"ledger row {b}: not a pre-v3 used block and inside {own or 'no'} role block")
        elif own[0] == "test" or _in(b[0], VAL_R2): out.append(f"ledger row {b}: a held-out block in the dev ledger")
    return out

# banned in any fullbody task input (M0 (1)); the patterns never match their own source text
BANNED = {"held-out clip": re.compile(r"kfab_20[2456](?!\d)"), "all-clip tracker motion": re.compile(r"r1_kungfu\w*\.(?:csv|npz)"),
          "tracker ONNX": re.compile(r"opponent_tracker\w*[/\\]+policy\.onnx")}
TASK_GLOBS = ("gpu/fb_*.py", "gpu/export_fb_*.py", "gpu/fullbody_task.py", "gpu/r1_track_spar_cfg.py", "gpu/hybrid_actor.py", "gpu/pod_fb.sh",
              "gpu/motions/fb/*", "scripts/fb_*.py", "scripts/export_hybrid_fb.py")

def task_inputs(root=ROOT):
    return sorted({p for g in TASK_GLOBS for p in Path(root).glob(g) if p.is_file()})

def _texts(p):
    """Every text view of a file: its bytes as ascii, UTF-16 and UTF-32 (numpy string arrays), and an npz's string arrays."""
    b = Path(p).read_bytes(); out = [b.decode("latin-1")] + [b[k:].decode(e, "ignore") for e in ("utf-16-le", "utf-32-le") for k in range(4 if "32" in e else 2)]
    if Path(p).suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            out += list(z.files) + [str(x) for k in z.files if z[k].dtype.kind in "US" for x in np.ravel(z[k])]
    return out

def scan(paths):
    """[(path, what, match)] for every banned name found."""
    hits = []
    for p in paths:
        for t in _texts(p):
            for what, rx in BANNED.items():
                hits += [(str(p), what, m.group(0)) for m in rx.finditer(t)]
    return sorted(set(hits))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--audit"); ap.add_argument("--scan", action="store_true"); a = ap.parse_args()
    bad = ledger_problems() + (audit(a.audit) if a.audit else []) + ([f"{p}: {w} {m}" for p, w, m in scan(task_inputs())] if a.scan else [])
    print("\n".join(bad) or f"ok: {len(ROLES)} roles disjoint" + (f"; {a.audit} consistent" if a.audit else "") + (f"; {len(task_inputs())} task inputs clean" if a.scan else ""))
    sys.exit(1 if bad else 0)
