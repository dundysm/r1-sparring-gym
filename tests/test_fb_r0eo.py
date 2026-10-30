"""Full-body M2 v2 M6 R0-EO runner (scripts/fb_r0eo.py; M2V2_PLAN_v3 3 M6): the R-EQ block 70000-82287 is R4-pre's episode construction
(seed 70000 + k over product(train clips, hard distances, 1024 reps)), cut into 64 balanced shards; the seed guard refuses anything else;
the knockdown edges derived from the read-only probe agree with EO's own opp_fall and both_up_iv (and a mismatch is refused); shards
are written atomically and a resume skips only valid ones; a 2 s bout (seed 70000, G and A3) through the runner equals a plain
eval_opponent.run of the same episode. The full check (shard 0 vs a direct EO.run on 70000-70015, R4-pre's own episodes()) is
`fb_r0eo.py check`."""
import sys, json, gzip
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import fb_r0eo as R
from fb_seeds import SeedRefused, check

@pytest.fixture(scope="module")
def eps(): return R.episodes()

def test_block_and_shards(eps):
    assert [e["seed"] for e in eps] == list(range(70000, 82288)) and len(eps) == 3 * 4 * 1024
    for e in eps[::397]:
        k = e["seed"] - 70000; assert (e["clip"], e["distance"], e["rep"]) == (R.TRAIN[k // 4096], R.HARD[(k // 1024) % 4], k % 1024)
    check([e["seed"] for e in eps], "eq", {e["clip"] for e in eps})
    sh = [R.shard_eps(eps, j) for j in range(R.NSHARD)]; assert sorted(e["seed"] for s in sh for e in s) == list(range(70000, 82288))
    for s in sh: assert len(s) == 192 and all(sum(e["clip"] == c and e["distance"] == d for e in s) == 16 for c in R.TRAIN for d in R.HARD)
    assert [e["seed"] for e in sh[0]][:16] == list(range(70000, 70016))

def test_guard_refuses_outside_block(eps):
    for s in (69999, 82288, 21000, 22000, 67000):
        with pytest.raises(SeedRefused): R.run_cell("G", [dict(eps[0], seed=s)], seconds=0.1)
    with pytest.raises(SeedRefused): R.run_cell("G", [dict(eps[0], clip="kfab_204")], seconds=0.1)

def _eo(fell, bu, ofall_t):
    return dict(fell_s=fell, opp_fell=ofall_t is not None, both_up_iv=bu, both_up_s=0.0, hit_meter=dict(n=dict(head_clean=1), n_both_up=dict(head_clean=0)),
                opp_fall=None if ofall_t is None else dict(t_s=ofall_t, r1_contact_0p5s=dict(contact=True)))

def test_bout_row_edges(eps):
    ep = eps[0]   # steps 1..10 live (fell at 0.2 s); opponent down at steps 3-4 and 7-10
    r = R.bout_row(ep, _eo(0.2, [[0.0, 0.04], [0.08, 0.12]], 0.06), dict(dt=0.02, act_delay=1, down=[[3, 4], [7, 10]]), 20.0)
    assert r["kd"] and r["kd_n"] == 2 and r["kd_edges"] == [0.06, 0.14] and r["down_iv"] == [[0.04, 0.08], [0.12, 0.2]] and r["kd_t"] == 0.06
    assert r["fall_after_kd"] and not r["opp_up_end"] and r["act_delay"] == 1 and r["k"] == ep["delay"] and r["secs"] == 0.2 and r["kd_contact"]
    r = R.bout_row(ep, _eo(None, [[0.0, 20.0]], None), dict(dt=0.02, act_delay=0, down=[]), 20.0)
    assert not r["kd"] and r["kd_t"] is None and r["kd_edges"] == [] and r["survived"] and r["opp_up_end"] and r["up_min"] == 20.0 / 60
    with pytest.raises(AssertionError): R.bout_row(ep, _eo(0.2, [[0.0, 0.04], [0.08, 0.12]], 0.06), dict(dt=0.02, act_delay=0, down=[[3, 4], [7, 9]]), 20.0)
    with pytest.raises(AssertionError): R.bout_row(ep, _eo(0.2, [[0.0, 0.04], [0.08, 0.12]], 0.08), dict(dt=0.02, act_delay=0, down=[[3, 4], [7, 10]]), 20.0)

def test_atomic_write_and_resume(eps, tmp_path):
    e0 = R.shard_eps(eps, 0); meta = R.shard_meta("G", 0, e0, 20.0); p = R.shard_path(tmp_path, "G", 0)
    assert not R.shard_ok(tmp_path, "G", 0, e0, 20.0)
    R.write_atomic(p, gzip.compress(json.dumps(dict(meta=meta, rows=[dict(seed=e["seed"]) for e in e0])).encode(), mtime=0))
    assert R.shard_ok(tmp_path, "G", 0, e0, 20.0) and sorted(x.name for x in p.parent.iterdir()) == ["shard_000.json.gz"]
    assert not R.shard_ok(tmp_path, "G", 0, e0, 8.0) and not R.shard_ok(tmp_path, "H", 0, e0, 20.0)
    R.write_atomic(p, gzip.compress(json.dumps(dict(meta=meta, rows=[dict(seed=e["seed"]) for e in e0[:-1]])).encode(), mtime=0)); assert not R.shard_ok(tmp_path, "G", 0, e0, 20.0)
    p.write_bytes(b"\x1f\x8b broken"); assert not R.shard_ok(tmp_path, "G", 0, e0, 20.0)

@pytest.mark.parametrize("name", ["G", "A3"])
def test_short_bout_equals_eval(eps, name):
    import eval_opponent as EO
    ref = R.direct(name, [eps[0]], seconds=2.0); row = R.run_cell(name, [eps[0]], seconds=2.0)[0]
    assert not getattr(EO.R1HitEnv, "_fb_probe", False), "probe left installed"
    assert json.dumps(row["eo"], sort_keys=True) == json.dumps(ref[0], sort_keys=True)
    assert row["seed"] == 70000 and row["act_delay"] in (0, 1) and row["secs"] == (row["fell_s"] or 2.0) and row["kd"] == bool(ref[0]["opp_fell"])

def test_rows_do_not_depend_on_the_call_split(eps):
    e = [eps[0], eps[1024]]   # two strata in one EO.run call vs one call each (the runner's CHUNK split)
    both = R.run_cell("G", e, seconds=2.0); one = R.run_cell("G", e[:1], seconds=2.0) + R.run_cell("G", e[1:], seconds=2.0)
    assert json.dumps(both, sort_keys=True) == json.dumps(one, sort_keys=True)
