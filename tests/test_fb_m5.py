"""Full-body M2 v2 M5 checks on one recorded eval bout (M2V2_PLAN_v3 M5; scripts/fb_m5_check.py): G at seed 60200 (role 'hit_scoring',
8 s) through eval_opponent.run with the read-only hooks (rows identical, asserted in the recorder); the recorder refuses seeds outside
its blocks; replayed with the HitMeter's contact-point closing speeds, FullBodyHitAccount's landed counts, clean calls and intent
window outcomes equal the HitMeter's, its paid windows the python HIT_R2 + m2 reference; the tangle is the per-substep sum. The full
gates (2 x 64 bouts per controller, W0) run in scripts/fb_m5_check.py."""
import sys
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_m5_check as C
from fb_seeds import SeedRefused

@pytest.fixture(scope="module")
def rec():
    R = C.record(seeds={"hit_scoring": [60200]}, seconds=8.0, workers=1, ctrls=("G",)); return R, C.model()[0]

def test_recorder_refuses_other_blocks():
    for role, s in (("hit_scoring", 60264), ("w0_calib", 60299), ("hit_scoring", 21000), ("w0_calib", 22000), ("hit_scoring", 60300)):
        with pytest.raises(SeedRefused): C.record(seeds={role: [s]}, seconds=0.1, workers=1, ctrls=("G",))

def test_account_equals_hitmeter_on_a_recorded_bout(rec):
    R, m = rec; b = R["meta"]["bouts"][0]; assert b["episode"]["seed"] == 60200 and len(R["0/post"]) == 400 and len(R["0/hit"]) > 0
    x = C.compare(R, [0], C.replay(R, [0], m, "point"))
    assert x["count_mismatch_bouts"] == 0 and x["clean_agree"] == 1.0 and x["windows"]["outcome_agree"] == x["windows"]["intents"] > 0 and x["windows"]["pay_agree"] == x["windows"]["intents"], x
    y = C.compare(R, [0], C.replay(R, [0], m, "com")); assert y["count_mismatch_bouts"] == 0, y

def test_recorded_terms(rec):
    R, m = rec; rp = C.replay(R, [0], m, "com", w0=1.0); V, L = rp["values"], rp["live"]; dn = V["down_now"] > 0; up = L & ~dn
    assert np.all(V["fb_down_neutral"][up] == 0) and np.all(V["fb_down_approach"][up] == 0) and np.all(V["fb_down_contact"][up] == 0)
    b = R["0/body"]; tz = np.array([C.HT.ZONES[z] in C.HT.TANGLE_ZONES for z in rp["bzone_idx"]]); ref = np.zeros(L.shape[1])
    np.add.at(ref, (b[:, 0] // 4).astype(int), np.linalg.norm(b[:, 2:5], axis=1) * C.HT.DT * tz[b[:, 1].astype(int)])
    assert np.abs(ref - V["tangle_Ns"][0]).max() <= 1e-9
