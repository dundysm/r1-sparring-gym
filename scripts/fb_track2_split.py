"""Track 2 partition (M2V2_PLAN_v3 6.2 item 3, T2-2; TRACK2_APPROVED.md): the downloaded KungFuAthlete fist-form clips split by
SOURCE VIDEO TITLE into a 60% train pool, 20% val2 and 20% test2, from action_titles.csv and fist_ids.txt only. No motion is
read: the npz files are hashed (bytes, never parsed) to check them against manifest_expected.tsv.
  video   the CSV title with its '-Scene-NNN.mp4' suffix removed; every clip of a video lands in one split.
  style   from the title only: 'taichi' (太极), 'nanquan' (南拳), else 'changquan'. Each style is split on its own (largest
          remainder, so the totals are round(0.2 n) / round(0.2 n) / rest), so neither held-out split can be all tai chi.
  order   per style, titles sorted, then np.random.default_rng(SEED).permutation; the first n_val go to val2, the next n_test
          to test2, the rest to the train pool.
  routine also from the title (sections and the complete version of one routine share it). Report only: a routine whose
          videos land in more than one split is listed in routine_overlap (the same choreography on both sides of the split).
IDs 196-206 (the duan-4 and duan-6 routines, which hold the kfab train/val/test scenes and the quarantined 206) were excluded
before the download; asserted absent here. Clip names are kfab_<id>, so fb_seeds' track2 guard sees an excluded clip.
Writes results/sparring/track2_split.json: inputs and their sha256, the rule, per video and per split, and split_sha256 (sha256
of the canonical JSON of {split: sorted ids}).
Run in the repo: python -I scripts/fb_track2_split.py [--data ...\\kfab_data] [--out results/sparring/track2_split.json]"""
import sys, re, csv, json, hashlib, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np

DATA = ROOT.parent / "kfab_data"
OUT = ROOT / "results" / "sparring" / "track2_split.json"
SEED = 20261030                                  # committed before the split was first computed
SHARES = dict(val2=0.2, test2=0.2)               # the rest is the train pool
EXCLUDED = tuple(range(196, 207))                # duan-4 (196-198) and duan-6 (199-206) routines
STYLES = (("taichi", "太极"), ("nanquan", "南拳"))  # first match; else changquan
DATASET = dict(hf="LuluCao/KungfuAthleteBot", commit="ff09f3795dc99bef73b21413b03fa8470917a22c", folder="org_smooth_mj",
               license="MIT per paper / Apache-2.0 per HF card")
SCENE = re.compile(r"-Scene-\d+\.mp4$")

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def video(title): return SCENE.sub("", title)

def style(v): return next((s for s, k in STYLES if k in v), "changquan")

def routine(v):
    """Routine key of a video title: sections and complete versions of one form share it (report only)."""
    v = re.sub(r"\s*#.*$", "", v)
    if "国际武术竞赛套路" in v: return "guoyi_" + ("nanquan" if "南拳" in v else "changquan")
    if "42式太极拳" in v: return "taichi42"
    if "少年规定拳" in v: return "shaonian"
    m = re.search(r"段位制长拳(.)段", v)
    return f"duanwei_{m.group(1)}" if m else v

def read_inputs(data):
    data = Path(data); rows = list(csv.reader(open(data / "action_titles.csv", encoding="utf-8-sig")))
    assert rows[0] == ["Data Number", "Source Video Title"], rows[0]
    title = {}
    for num, t in rows[1:]:
        i = int(num.split(".")[0]); assert num == f"{i}.mp4" and i not in title, num; title[i] = t
    ids = [int(x) for x in (data / "fist_ids.txt").read_text(encoding="ascii").split()]
    man = {int(r[0]): dict(path=r[1], size=int(r[2]), sha256=r[3]) for r in csv.reader(open(data / "manifest_expected.tsv", encoding="ascii"), delimiter="\t")}
    assert len(ids) == len(set(ids)) and set(ids) == set(man), "fist_ids.txt and the manifest disagree"
    assert not set(ids) & set(EXCLUDED), f"excluded IDs present: {sorted(set(ids) & set(EXCLUDED))}"
    assert all(SCENE.search(title[i]) for i in ids), "a fist clip title without a scene suffix"
    assert all(man[i]["path"].endswith(f"/{DATASET['folder']}/{i}.npz") for i in ids)
    return title, sorted(ids), man

def check_files(data, ids, man):
    """Downloaded bytes vs the manifest (size and LFS sha256); the files are hashed, never parsed."""
    bad = [i for i in ids if (p := Path(data) / "npz" / f"{i}.npz").stat().st_size != man[i]["size"] or sha256(p) != man[i]["sha256"]]
    assert not bad, f"npz files differ from manifest_expected.tsv: {bad}"; return len(ids)

def partition(title, ids, seed=SEED):
    vids = {}
    for i in ids: vids.setdefault(video(title[i]), []).append(i)
    by_style = {}
    for v in sorted(vids): by_style.setdefault(style(v), []).append(v)
    n = {s: len(v) for s, v in by_style.items()}; N = sum(n.values())
    want = {k: int(round(f * N)) for k, f in SHARES.items()}; alloc = {}
    for k in SHARES:   # largest remainder per held-out split, over styles, on what each style has left
        left = {s: n[s] - sum(alloc.get((s, j), 0) for j in SHARES) for s in n}
        quota = {s: SHARES[k] * n[s] for s in n}; base = {s: min(int(np.floor(quota[s])), left[s]) for s in n}
        rem = want[k] - sum(base.values())
        for s in sorted(n, key=lambda s: (-(quota[s] - np.floor(quota[s])), s)):
            if rem > 0 and base[s] < left[s]: base[s] += 1; rem -= 1
        assert rem == 0; alloc.update({(s, k): base[s] for s in n})
    rng = np.random.default_rng(seed); split = {}
    for s in sorted(by_style):
        order = [by_style[s][j] for j in rng.permutation(len(by_style[s]))]; nv, nt = alloc[(s, "val2")], alloc[(s, "test2")]
        for j, v in enumerate(order): split[v] = "val2" if j < nv else "test2" if j < nv + nt else "train"
    return vids, split

def build(data=DATA, seed=SEED):
    data = Path(data); title, ids, man = read_inputs(data); nfiles = check_files(data, ids, man); vids, split = partition(title, ids, seed)
    splits = {k: sorted(i for v, s in split.items() if s == k for i in vids[v]) for k in ("train", "val2", "test2")}
    canon = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    rout = {}
    for v in vids: rout.setdefault(routine(v), set()).add(split[v])
    out = dict(about=__doc__.split("\n")[0], dataset=DATASET, seed=seed, shares=dict(train=0.6, **SHARES),
               rule="group by video title (scene suffix removed); per style (title keyword) largest-remainder counts; titles sorted, "
                    "rng(seed).permutation; val2 first, then test2, rest train. Titles only; npz hashed, never parsed.",
               excluded_ids=list(EXCLUDED), excluded_titles={str(i): title[i] for i in EXCLUDED},
               excluded_note="IDs 196-206 (duan-4 196-198, duan-6 199-206: the kfab train/val/test routine and the quarantined 206) excluded "
                             "before the download (TRACK2_APPROVED.md); none of them is in fist_ids.txt (asserted)",
               inputs={k: sha256(data / k) for k in ("action_titles.csv", "fist_ids.txt", "manifest_expected.tsv")},
               npz_checked=dict(files=nfiles, against="manifest_expected.tsv size + LFS sha256", ok=True),
               clip_sha256={f"kfab_{i}": man[i]["sha256"] for i in ids},
               videos=[dict(title=v, style=style(v), routine=routine(v), split=split[v], ids=sorted(vids[v])) for v in sorted(vids, key=lambda v: (split[v], style(v), v))],
               counts={k: dict(videos=sum(s == k for s in split.values()), clips=len(splits[k]),
                               styles={st: sum(s == k and style(v) == st for v, s in split.items()) for st in ("changquan", "nanquan", "taichi")}) for k in splits},
               splits={k: [f"kfab_{i}" for i in v] for k, v in splits.items()}, split_sha256=hashlib.sha256(canon.encode()).hexdigest(),
               routine_overlap={r: sorted(s) for r, s in sorted(rout.items()) if len(s) > 1})
    return out

def load(path=OUT):
    """The committed split, its sha256 re-checked."""
    S = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = {k: sorted(int(c.split("_")[1]) for c in v) for k, v in S["splits"].items()}
    assert hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == S["split_sha256"], "track2 split edited"
    return S

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(DATA)); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    S = build(a.data); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(S, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(counts=S["counts"], split_sha256=S["split_sha256"], routine_overlap=S["routine_overlap"]), ensure_ascii=True))
