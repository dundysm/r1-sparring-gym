"""Full-body M2 v2 M4 checks on one recorded eval bout (M2V2_PLAN_v3 M4; scripts/fb_m4_check.py): G at seed 60001 (role 'record',
2 s) through eval_opponent.run with the read-only hooks; the recorder refuses other blocks; on its states our proprio83, delayed
noisy obs (the eval's noise replayed), applied action and joint targets equal the eval's bit for bit and the noise-free output is
DelayedNoisyObs; B3 in the replica on 8 resets, with the eager-prefill and no-forward controls failing; the fall rules on its
post-step states. The full gates (32 bouts, 64 resets, 1e4 states) run in scripts/fb_m4_check.py."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_m4_check as C
from fb_seeds import SeedRefused

@pytest.fixture(scope="module")
def rec():
    R = C.record(seeds=(60001,), seconds=2.0, workers=1, ctrls=("G",)); return (R, *C.model_cfg())

def test_recorder_refuses_other_blocks():
    for s in ((60016,), (21000,), (22000,), (67000,)):
        with pytest.raises(SeedRefused): C.record(seeds=s, seconds=0.1, workers=1, ctrls=("G",))

def test_pipeline_equals_the_eval_on_a_recorded_bout(rec):
    R, m, cfg, r, ids6 = rec; b = R["meta"]["bouts"][0]; assert b["steps"] == 100 and b["episode"]["seed"] == 60001
    pe = C.pipeline(R, m, cfg, r, ids6, "eval"); p0 = C.pipeline(R, m, cfg, r, ids6, "zero")
    assert pe["steps"] == 101 and all(pe[k] == 0 for k in ("proprio83_vs_raw_obs_mismatch", "out_vs_eval_delivered83_mismatch", "applied_vs_eval_last_a_mismatch",
                                                            "targets_vs_eval_ctrl_mismatch", "eval_actions_slot_vs_applied_mismatch", "k_switches", "act_switches")), pe
    assert p0["out_vs_DelayedNoisyObs_mismatch"] == p0["out_vs_clean_t_minus_k_mismatch"] == 0

def test_b3_resets_in_the_replica(rec):
    R, m, cfg, r, ids6 = rec; x = C.g0_6b(R, m, cfg, r, ids6, n=8, need_kinds=False); assert x["ok"] and x["resets"] == 8, x
    for bug in ("eager", "no_forward"): y = C.g0_6b(R, m, cfg, r, ids6, bug=bug, n=8, need_kinds=False); assert not y["ok"] and y["ring_fresh"] == 0, (bug, y)

def test_fall_rules_on_the_recorded_states(rec):
    R, m, cfg, r, ids6 = rec; x = C.g0_8(R, m, cfg, r, n_rand=500)
    assert all(v == 0 for k, v in x["recorded"].items() if k.endswith("mismatch")) and x["random"]["fell_mismatch"] == x["random"]["opp_down_mismatch"] == 0 and all(x["constants"].values())
