"""Paired differences between controllers in the stored eval runs (results/eval_opponent/raw_*.json) and the same runs at an
earlier commit, for runs where both have exactly the same episodes (checked; otherwise the run is skipped and listed).
Used for "what changed since PR #13": this PR's footwork policies vs the policies shipped at 3fcbb3b, and the stage-2
bases with their new hold loop (gym.json) vs themselves at 3fcbb3b, without it.

python scripts/paired_vs_ref.py --ref 3fcbb3b --pair "RL step_fw=RL step" "RL guard_fw=RL guard" "RL step=RL step" "RL guard=RL guard" --out results/eval_opponent/vs_3fcbb3b.json"""
import sys, json, argparse, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1gym.stats import paired_diff

def up(rows): return [x["fell_s"] is None for x in rows]
def ring(rows): return [x["fell_s"] is None and not x["ring_out"] for x in rows]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--pair", nargs="+", required=True, help='"NEW=OLD": NEW in the current run, OLD in --ref')
    ap.add_argument("--dir", default="results/eval_opponent"); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = dict(about=__doc__, ref=a.ref, runs={}, skipped={})
    for f in sorted(Path(a.dir).glob("raw_*.json")):
        p = subprocess.run(["git", "show", f"{a.ref}:{f.as_posix()}"], capture_output=True, text=True, encoding="utf-8")
        if p.returncode: out["skipped"][f.name] = "not in ref"; continue
        old, new = json.loads(p.stdout), json.load(open(f))
        if old["episodes"] != new["episodes"]: out["skipped"][f.name] = "different episodes"; continue
        run = {}
        for pr in a.pair:
            n, o = pr.split("=", 1)
            if n not in new["raw"] or o not in old["raw"]: continue
            N, O = new["raw"][n], old["raw"][o]
            run[f"{n} - {o}@{a.ref}"] = dict(survival=[round(sum(up(O)) / len(O), 3), round(sum(up(N)) / len(N), 3)], survival_diff=paired_diff(up(N), up(O)),
                                             in_ring=[round(sum(ring(O)) / len(O), 3), round(sum(ring(N)) / len(N), 3)], in_ring_diff=paired_diff(ring(N), ring(O)))
        out["runs"][f.name] = run; print(f.name, json.dumps(run), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
