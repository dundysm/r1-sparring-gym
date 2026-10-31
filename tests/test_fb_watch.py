"""WATCH6 (M2V2_PLAN_v3 section 4; scripts/fb_watch.py --rule 6, M9 (3)): every synthetic stop log stops at or before its deadline for
its own rule; healthy A (1600 it) and B (4000 it) logs never stop; the blind twin is never stopped; a missing required input is a config
error, not a pass; the parser keeps complete blocks only; the run-3 logs trip no load check (when present)."""
import sys, os, json, subprocess
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import fb_watch as WT

RUN3 = Path(os.environ.get("FB_RUN3_LOGS", r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\dl5"))

@pytest.fixture(scope="module")
def rep(tmp_path_factory): return WT.replays(RUN3 if (RUN3 / "spar3_A3s.log").exists() else None, tmp_path_factory.mktemp("w6"))

def test_every_stop_rule_fires_for_its_own_reason(rep):
    assert set(rep["stops"]) == set(WT.STOPS) | set(WT.PROBES)
    for rule, v in rep["stops"].items(): assert v["ok"], (rule, v)

def test_healthy_logs_never_stop_and_blind_is_never_stopped(rep):
    assert rep["healthy"]["A"]["checks"] == 17 and rep["healthy"]["B"]["checks"] == 41
    for v in rep["healthy"].values(): assert not v["stops"] and not v["errors"] and not v["na"], v
    assert rep["blind"]["ok"] and rep["blind"]["stops"] == 0 and rep["blind"]["reported"] > 0

@pytest.mark.skipif(not (RUN3 / "spar3_A3s.log").exists(), reason="run-3 training logs not present")
def test_run3_logs_trip_no_load_check(rep):
    assert len(rep["run3"]) == 3 and all(v["ok"] and v["blocks"] == 1500 and not v["e0_trips"] for v in rep["run3"].values()), rep["run3"]

def test_parser_and_config_errors(tmp_path):
    p = WT.synth(tmp_path / "h.log", 120, "A"); txt = p.read_text(); p.write_text("\x1b[1m" + txt + "\n Learning iteration 120/200 \n Mean action std: 0.70\n")   # ANSI + an incomplete block
    B = WT.parse(p); assert len(B) == 120 and B[0][0] == 0 and B[-1][0] == 119 and abs(B[5][1]["Metrics/opp/w_ad"] - B[5][1]["Metrics/opp/w_eps"] / 2) < 1e-3
    o = WT.watch6(p, 100, "A", start_it=0); assert o["error"] and not o["stop"]                       # --topp missing: a config error, never a silent pass
    o = WT.watch6(p, 100, "B", start_it=0, topp=WT.TOPP); assert any("a-final-log" in e for e in o["error"])
    assert not WT.watch6(p, 200, "A", topp=WT.TOPP)["ready"]
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "fb_watch.py"), "watch", "--rule", "6", "--log", str(p), "--at", "100", "--stage", "A"], capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "fb_watch.py"), "watch", "--rule", "6", "--log", str(p), "--at", "100", "--stage", "A", "--topp", str(WT.TOPP), "--start-it", "0"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["ready"]
