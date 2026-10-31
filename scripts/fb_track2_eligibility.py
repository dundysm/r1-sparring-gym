"""Track 2 eligibility (M2V2_PLAN_v3 6.2 item 3, T2-3), SEALED for val2/test2: every clip of a split of results/sparring/
track2_split.json goes through scripts/fb_track2_motion.py's screen (retarget, clean, tracker and style cuts, strikes, the frozen
T0 tracker alone in the CPU world from 24 starts). Eligible: >= 4 s kept, >= 2 head-band strikes, tracker stay-up >= 90%,
scored as hand strikes (eligible_hand) and as any limb (eligible_any). val2 also needs >= 30 head-band hand strikes over its
eligible clips.
  --split train   the train pool may be inspected freely: per-clip records (rounds, cuts, strikes, falls) to --out.
  --split val2 | test2   sealed. Each worker reduces a clip's record to its eligibility booleans and two strike counts before it
                  returns (a clip that fails returns only its exception's type and code path, file:function per frame), and the process keeps nothing else: no motion,
                  frame, strike row, fall time or per-clip number is printed or saved. The output holds per-clip booleans and
                  aggregate counts only; the retarget is never cached.
Seeds: tracker-alone worlds are seed-free (fb_track2_motion.alone); each clip still gets a bookkeeping id in the track2 block
(train 24900+k, val2 24950+k, test2 25950+k), checked by fb_seeds.check(..., 'track2', clips) and logged in SEED_LEDGER.md.
Run in the repo: python -I scripts/fb_track2_eligibility.py --split val2 [--out results/sparring/track2_eligibility_val2.json]
[--workers 2] [--data ...\\kfab_data] [--cache DIR (train only)]"""
import sys, json, time, argparse, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import fb_track2_motion as T2
import fb_track2_split as SP
import fb_seeds as F

SEALED = ("val2", "test2")
BASE = dict(train=24900, val2=24950, test2=25950)
VAL2_HEAD_MIN = 30
OUT = ROOT / "results" / "sparring"
BOOLS = ("eligible_hand", "eligible_any")
REASONS = ("kept_ok", "hand_ok", "any_ok", "stay_up_ok")

def clips_of(split, S=None):
    S = S or SP.load(); return [int(c.split("_")[1]) for c in S["splits"][split]]

def seeds_for(split, ids):
    s = list(range(BASE[split], BASE[split] + len(ids))); F.check(s, "track2", [T2.clip_name(i) for i in ids]); return s

def sealed(rec):
    """The only thing a sealed clip's worker returns: booleans, plus counts that are summed into aggregates and never kept per clip
    (last round: head-band strikes; first round, before any tracker cut: kept seconds and head-band strikes)."""
    r0 = rec["rounds"][0]
    return dict(**{k: bool(rec[k]) for k in BOOLS + REASONS}, measured=rec["stay_up"] is not None, head_hands=int(rec["head_hands"]), head_any=int(rec["head_any"]),
                kept0_s=float(r0["kept_s"]), head_hands0=int(r0["head_hands"]), head_any0=int(r0["head_any"]))

def _job(args):
    i, split, data, cache = args
    try:
        qr = None
        if split == "train" and cache:
            p = Path(cache) / f"r1_{int(i)}.npy"
            if p.exists(): qr = np.load(p, allow_pickle=False)
            else: q, _ = T2.load(i, data); qr = T2.retarget(q); p.parent.mkdir(parents=True, exist_ok=True); np.save(p, qr)
        rec = T2.screen(i, data, qr=qr)
        if split in SEALED: return dict(clip=T2.clip_name(i), ok=True, **sealed(rec))
        rec.pop("q0"); rec["strikes"] = [dict(limb=s.limb, type=s.type, band=s.band, onset=s.onset, apex=s.apex, apex_m=list(s.apex_m)) for s in rec["strikes"]]
        _, _, rec["fk_err"] = T2.load(i, data, fk=True); return dict(ok=True, **rec)
    except Exception as e:   # sealed: the type and the code path (file:function per frame), never the message (it could carry motion numbers)
        if split in SEALED: return dict(clip=T2.clip_name(i), ok=False, error=type(e).__name__, where=[f"{Path(f.filename).name}:{f.name}" for f in traceback.extract_tb(e.__traceback__)])
        raise

def aggregate(split, rows):
    ok = [r for r in rows if r["ok"]]; agg = dict(clips=len(rows), errors=len(rows) - len(ok), measured=sum(r["measured"] for r in ok))   # stay-up measured on the last round
    for b in BOOLS:
        el = [r for r in ok if r[b]]
        agg[b] = dict(eligible=len(el), head_band_hand_strikes=sum(r["head_hands"] for r in el), head_band_strikes=sum(r["head_any"] for r in el))
    agg["fails"] = {k: sum(not r[k] for r in ok) for k in REASONS}
    agg["all_clips_first_round"] = dict(kept_s=round(sum(r["kept0_s"] for r in ok), 1), head_band_hand_strikes=sum(r["head_hands0"] for r in ok),
                                        head_band_strikes=sum(r["head_any0"] for r in ok), clips_kept_ok=sum(r["kept0_s"] >= T2.RULE["kept_min_s"] for r in ok))
    if split == "val2":
        agg["val2_head_band_hand_strikes_min"] = VAL2_HEAD_MIN
        agg["val2_ok"] = {b: agg[b]["head_band_hand_strikes"] >= VAL2_HEAD_MIN for b in BOOLS}
    return agg

def run(split, data=T2.DATA, workers=2, cache=None):
    S = SP.load(); ids = clips_of(split, S); seeds = seeds_for(split, ids); assert workers <= 2
    if split in SEALED: assert cache is None, "sealed clips are never cached"
    import multiprocessing as mp
    jobs = [(i, split, str(data), cache) for i in sorted(ids, key=lambda i: -(Path(data) / "npz" / f"{i}.npz").stat().st_size)]   # longest first
    t = time.time(); rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            rows.append(r); print(f"{split}: {k + 1}/{len(jobs)} clips done, {time.time() - t:.0f} s", flush=True)
    rows.sort(key=lambda r: int(r["clip"].split("_")[1]))
    out = dict(about=__doc__.split("\n")[0], split=split, split_sha256=S["split_sha256"], rule=T2.RULE, val2_head_band_hand_strikes_min=VAL2_HEAD_MIN,
               bookkeeping_seeds=[seeds[0], seeds[-1]], seed_note="tracker-alone worlds are seed-free; ids checked by fb_seeds role track2",
               wall_s=round(time.time() - t, 1))
    if split in SEALED:
        out["sealed"] = True; out["per_clip"] = {r["clip"]: {b: r.get(b) if r["ok"] else None for b in BOOLS} for r in rows}
        out["aggregate"] = aggregate(split, rows); out["errors"] = sorted({(r["error"], " < ".join(r["where"][::-1])) for r in rows if not r["ok"]})
    else:
        out["sealed"] = False; out["per_clip"] = rows; out["aggregate"] = aggregate(split, [dict(r, **sealed(r)) for r in rows])
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=("train",) + SEALED); ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--data", default=str(T2.DATA)); ap.add_argument("--cache", default=None); a = ap.parse_args()
    out = run(a.split, a.data, a.workers, a.cache if a.split == "train" else None)
    p = Path(a.out or OUT / f"track2_eligibility_{a.split}.json"); p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=lambda x: x.item()) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(split=a.split, aggregate=out["aggregate"]), ensure_ascii=True)); print("wrote", p)
