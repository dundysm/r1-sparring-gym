"""Full-body M2 v2 M6 stage driver (scripts/fb_m6_run.py): the R-EQ and R2 blocks cut into balanced 192-bout chunks (the R-EQ ones are
fb_r0eo's shards); chunk files are atomic and a resume accepts only a matching chunk; the manifest refuses another conf / rule; every
configured conf is a valid replica conf, R2's steps switch one thing each from the eval, and the req conf is 3.2's first fallback entry
with the deployed closing-speed input. The 8-bout dry run of every stage is in the M6a notes."""
import sys, json, gzip
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"): sys.path.insert(0, str(p))
import fb_m6_run as D, fb_replica as RP, fb_r0eo as R0

def test_chunks_are_balanced_shards():
    eq = D.block_eps("eq"); J = D.shards(eq); assert len(J) == 64 and all(len(v) == 192 for v in J.values())
    eps = R0.episodes(); assert all(J[j] == [e["seed"] for e in R0.shard_eps(eps, j)] for j in (0, 17, 63))
    r2 = D.block_eps("r2_diag"); J2 = D.shards(r2); assert sorted(J2) == [0, 1, 2, 3] and sorted(s for v in J2.values() for s in v) == list(range(67000, 67768))
    for v in J2.values(): assert len({(r2[s]["clip"], r2[s]["distance"]) for s in v}) == 12 and len(v) == 192

def test_confs():
    c = RP.conf(**D.REQ_CONF); assert (c["hulls"], c["solver"], c["backend"], c["reset"], c["scorer"]) == (RP.FALLBACK["hulls"][0], RP.FALLBACK["solver"][0], "torch32", "eo", "account")
    assert D.DEPLOYED_VMODE in RP.VMODES and c["vmode"] == D.DEPLOYED_VMODE
    for k, s in D.R2_STEPS.items():
        cc = RP.conf(**s); e = RP.conf(); diff = {x for x in cc if cc[x] != e[x]}
        assert diff and len(diff - {"vmode"}) == (2 if k == "d_reset" else 1), (k, diff)
    assert D.conf_tag({}) == "eval" and D.conf_tag(D.REQ_CONF) == D.conf_tag(dict(D.REQ_CONF))

def test_chunk_files_and_manifest(tmp_path):
    rows = [dict(seed=1), dict(seed=2)]; p = D.cpath(tmp_path, "req", "t", "G", "shard_000")
    D.write_atomic(p, gzip.compress(json.dumps(dict(meta=D.cmeta("req", "t", "G", "shard_000", dict(solver=(10, 20)), [1, 2], 20.0), rows=rows)).encode(), mtime=0))
    assert D.chunk_ok(tmp_path, "req", "t", "G", "shard_000", dict(solver=(10, 20)), [1, 2], 20.0)
    assert not D.chunk_ok(tmp_path, "req", "t", "G", "shard_000", dict(solver=(25, 30)), [1, 2], 20.0) and not D.chunk_ok(tmp_path, "req", "t", "G", "shard_000", dict(solver=(10, 20)), [1, 3], 20.0)
    assert not list(tmp_path.rglob("*.tmp*"))
    D.manifest(tmp_path, "look_rule", "reps"); D.manifest(tmp_path, "look_rule", "reps")
    with pytest.raises(AssertionError): D.manifest(tmp_path, "look_rule", "shards_ceil")

def test_batched_configuration():
    """A7.1 / A7.2: torch-backend confs run BATCH = 64 bouts in lockstep, row-wise on the batch-variant ops; the eval conf one at a time;
    both are in each chunk's meta, so a resume never mixes them."""
    assert (D.BATCH, D.ROWWISE) == (64, True) and D.batch_of(D.REQ_CONF) == (64, True) and D.batch_of({}) == (1, False)
    assert D.batch_of(D.R2_STEPS["d_reset"]) == (64, True) and D.batch_of(D.R2_STEPS["b_hulls"]) == (1, False)
    m = D.cmeta("req", "t", "G", "shard_000", D.REQ_CONF, [1], 20.0); assert (m["batch"], m["rowwise"]) == (64, True) and 192 % D.BATCH == 0
