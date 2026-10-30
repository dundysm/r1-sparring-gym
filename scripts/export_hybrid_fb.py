"""Export of the full-body M2 v2 defender (M2V2_PLAN_v3 2.5 'Export', M7 T9; gpu/hybrid_actor.py): one ONNX 111 -> 24 (G + M3s + arm
net + residual) with the arm donor's metadata, and the T9 checks.
  export  the start (no --ckpt: H(A3s)/M3s, or H(blind3)/M3s with --blind) or a training checkpoint (rsl_rl 'actor_state_dict' of a
          HybridActor; residual keys -> stage B) -> <out>/policy.onnx, <out>/gym.json (G's, byte copy: the eval's MjlabPolicy reads it next
          to the ONNX), <out>/export_manifest.json (sha256 of every source and output). Compute dtype float64 by default (--dtype float32:
          the donors' arithmetic); scale / clip = R1HitEnv's under G's controller cfg (action_scale(cfg), action_clip), as P1 sets them.
  t9a     ONNX = torch on recorded rows (<= 1e-6): the exported file through r1gym's MjlabPolicy (the eval's loader: float32 cast, batch 1)
          vs HybridActor.mean(ref=True) on the same float32 rows; the deployed float32 torch path reported beside it.
  t9b     (the Join step; needs scripts/fb_replica.py) the 16 dev bouts 60400-60415 (fb_seeds role 't9', fb_record's seed -> episode map):
          eval_opponent.run unchanged with the exported ONNX (as R0-EO runs a cell: threads 1, DR, meter, falls, keypoints truth, aim.json
          lag) vs fb_replica in eval mode driven by the torch composition (HybridActor.mean(ref=True), row by row): every row bit identical
          (fell_s and every HitMeter count); plus, same-state, the torch composition on every obs the ONNX received in the EO bouts.
Run in the repo: python -I scripts/export_hybrid_fb.py export --out DIR [--blind] [--ckpt model_N.pt] [--dtype float64]
                 python -I scripts/export_hybrid_fb.py t9a --dir DIR [--blind] [--ckpt ...] --m4rec fb_m4rec_60000-60015.npz
                 python -I scripts/export_hybrid_fb.py t9b --dir DIR [--blind] [--ckpt ...] --out t9b.json [--workers 2]"""
import sys, json, shutil, hashlib, argparse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, torch
import hybrid_actor as HA

T9_SEEDS, TOL = tuple(range(60400, 60416)), 1e-6
AIM = "standoff+armwarp_vz"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def env_scale():
    """(scale [24] float64, clip, cfg): R1HitEnv.scale / .clip under G's controller cfg (P1's set_scale(action_scale(cfg), action_clip))."""
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import action_scale
    cfg = controller_cfg(load_controller(f"guard_fw_r={ROOT / HA.G_ONNX}")); return np.asarray(action_scale(cfg), np.float64), float(cfg.get("action_clip", 4.0)), cfg

def build(blind=False, ckpt=None):
    """The start (ckpt None) or a trained HybridActor from its rsl_rl checkpoint."""
    sc, clip, _ = env_scale(); a = HA.HybridActor.from_checkpoints(ROOT, sc, clip, blind)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
        if any(k.startswith("residual.") for k in sd): a.to_stage_b()
        a.load_state_dict(sd, strict=True)
    return a.eval()

def export(out, blind=False, ckpt=None, dtype="float64", actor=None):
    """policy.onnx + gym.json + export_manifest.json in out; returns the manifest."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); a = actor if actor is not None else build(blind, ckpt); kind = "blind" if a.blind else "sighted"
    src = dict(g_pt=HA.G_PT, arm_pt=HA.ARM_PT[kind], arm_onnx=HA.ARM_ONNX[kind], gym_json=str(Path(HA.G_ONNX).with_name("gym.json")).replace("\\", "/"))
    srcsha = {k: sha(ROOT / v) for k, v in src.items()}
    if ckpt: src["ckpt"] = str(ckpt); srcsha["ckpt"] = sha(ckpt)
    assert (ROOT / src["gym_json"]).read_bytes() == (ROOT / Path(src["arm_onnx"]).with_name("gym.json")).read_bytes(), "G's and the arm donor's gym.json differ"
    HA.export_onnx(a, out / "policy.onnx", ROOT / src["arm_onnx"], info=dict(sha256={k: v[:16] for k, v in srcsha.items()}), dtype=dtype)
    shutil.copyfile(ROOT / src["gym_json"], out / "gym.json")
    man = dict(stage=a.stage, blind=a.blind, dtype=dtype, sources=src, sha256=dict(srcsha, policy_onnx=sha(out / "policy.onnx"), gym_json_out=sha(out / "gym.json")))
    (out / "export_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8"); return man

def contract(onnx_path, p1=None):
    """The eval's loader on the export: MjlabPolicy's metadata asserts, kp_mode, batch 1; controller_cfg = G's (P1's H cfg is G's)."""
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller(f"fb_hybrid={onnx_path}"); g = load_controller(f"guard_fw_r={ROOT / HA.G_ONNX}")
    J = lambda x: json.dumps(x, sort_keys=True, default=str)
    res = dict(kp_mode=c.kp_mode, obs_dim=c.obs_dim, batch1=bool(c.batch1), cfg_equals_G=J(controller_cfg(c)) == J(controller_cfg(g)), home=c.cfg_overrides["home"])
    if p1:
        import fb_m4_check as M4
        h = M4.load_p1(p1).make_ctrl("guard_fw_r+a3s", "m3s"); res["cfg_equals_P1_H"] = J(controller_cfg(c)) == J(controller_cfg(h))
    return res

def bits(x): return np.ascontiguousarray(x, np.float32).view(np.int32)

def t9a(a, onnx_path, rows, tol=TOL):
    """rows [N, 111] (the eval's obs; the loader casts them to float32). ONNX via MjlabPolicy.act (batch 1) vs torch ref; info: f32 torch."""
    from r1gym.policy import load_controller
    c = load_controller(f"fb_hybrid={onnx_path}"); x32 = np.asarray(rows).astype(np.float32); y = c.act(np.asarray(rows))
    with torch.no_grad(): t = a.mean(torch.from_numpy(x32), ref=True).numpy(); f = a.mean(torch.from_numpy(x32)).numpy()
    d = np.abs(y.astype(np.float64) - t); res = dict(rows=len(rows), max_abs=float(d.max()), bit_mismatch=int((bits(y) != bits(t)).sum()), elems=int(y.size),
                                                  info_f32_torch_max_abs=float(np.abs(y.astype(np.float64) - f).max()), act_abs_max=float(np.abs(y).max()))
    res["ok"] = bool(len(rows) >= 10000 and res["max_abs"] <= tol); return res

# ---- t9b (Join): EO.run with the ONNX vs the replica driven by the torch composition
class TorchCtrl:
    """The torch composition as an EO / replica controller: same kp_mode and cfg as the exported ONNX's loader; row by row (batch 1)."""
    def __init__(self, a, onnx_ctrl):
        self.a, self.name, self.kp_mode = a, "fb_hybrid_torch", onnx_ctrl.kp_mode
        self.cfg_path, self.cfg_overrides = onnx_ctrl.cfg_path, onnx_ctrl.cfg_overrides
    def act(self, obs):
        x = np.asarray(obs).astype(np.float32)
        with torch.no_grad(): return np.concatenate([self.a.mean(torch.from_numpy(r[None]), ref=True).numpy() for r in x])

class LogCtrl:
    """Read-only wrapper: logs every obs batch the ONNX controller receives and its actions (same object semantics)."""
    def __init__(self, c): self.c, self.log = c, []
    def __getattr__(self, k): return getattr(self.c, k)
    def act(self, obs): y = self.c.act(obs); self.log.append((np.asarray(obs).copy(), y.copy())); return y

def t9b_episodes(seeds=T9_SEEDS, seconds=20.0):
    import fb_record as REC
    from fb_seeds import check
    seeds = [int(s) for s in seeds]; check(seeds, "t9", set(REC.TRAIN)); eps = REC.episodes(seeds, seconds); check(seeds, "t9", {e["clip"] for e in eps}); return eps

def _t9b_job(args):
    onnx_path, blind, ckpt, seeds, seconds = args; torch.set_num_threads(1)
    import eval_opponent as EO, fb_r0eo as R0, fb_replica as RP
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import build_spec
    eps = t9b_episodes(seeds, seconds); c = load_controller(f"fb_hybrid={onnx_path}"); a = build(blind, ckpt)
    cfg = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": AIM}; m = build_spec(opponent=True, cfg=cfg).compile()
    aim = json.load(open(ROOT / "results" / "opponent_tracker" / "aim.json")); assert aim["operator"] == AIM; lag = float(aim.get("lag_s", 0.0))
    lc = LogCtrl(c); eo = EO.run(lc, eps, m, cfg, seconds, threads=1, dr=True, meter=True, lag=lag, rev=False, falls=True, keypoints="truth")
    O = np.concatenate([o for o, _ in lc.log]); Y = np.concatenate([y for _, y in lc.log])
    with torch.no_grad(): T = np.concatenate([a.mean(torch.from_numpy(r[None].astype(np.float32)), ref=True).numpy() for r in O])
    R0._CELLS["FB_T9"] = (TorchCtrl(a, c), cfg, m, lag); rows = RP.run("FB_T9", eps, RP.conf(), seconds=seconds)
    J = lambda x: json.dumps(x, sort_keys=True, default=str)
    return dict(bouts=[dict(seed=e["seed"], equal=J(r["eo"]) == J(x), fell=[x["fell_s"], r["eo"]["fell_s"]], n=[x["hit_meter"]["n"], r["eo"]["hit_meter"]["n"]])
                       for e, r, x in zip(eps, rows, eo)], same_state=dict(rows=len(O), max_abs=float(np.abs(Y.astype(np.float64) - T).max()), bit_mismatch=int((bits(Y) != bits(T)).sum())))

def t9b(onnx_path, blind=False, ckpt=None, workers=2, seconds=20.0, seeds=T9_SEEDS):
    assert workers <= 2, "at most 2 worker processes"
    jobs = [(str(onnx_path), blind, ckpt, list(seeds[i::workers]), seconds) for i in range(workers)]
    if workers == 1: res = [_t9b_job(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool: res = pool.map(_t9b_job, jobs, chunksize=1)
    B = sorted(sum((r["bouts"] for r in res), []), key=lambda x: x["seed"]); ss = dict(rows=sum(r["same_state"]["rows"] for r in res), max_abs=max(r["same_state"]["max_abs"] for r in res),
                                                                                       bit_mismatch=sum(r["same_state"]["bit_mismatch"] for r in res))
    return dict(seeds=[min(seeds), max(seeds)], bouts=B, identical=sum(b["equal"] for b in B), same_state=ss, ok=bool(len(B) == len(seeds) and all(b["equal"] for b in B)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["export", "t9a", "t9b"]); ap.add_argument("--out"); ap.add_argument("--dir")
    ap.add_argument("--blind", action="store_true"); ap.add_argument("--ckpt"); ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--m4rec"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seconds", type=float, default=20.0); a = ap.parse_args()
    torch.set_num_threads(2); t0 = time.time()
    if a.cmd == "export": print(json.dumps(export(a.out, a.blind, a.ckpt, a.dtype), indent=1)); return
    onnx_path = Path(a.dir) / "policy.onnx"
    if a.cmd == "t9a":
        import fb_m7_check as M7
        r = t9a(build(a.blind, a.ckpt), onnx_path, M7.rows(a.m4rec)[0])
    else: r = t9b(onnx_path, a.blind, a.ckpt, a.workers, a.seconds)
    r["wall_s"] = round(time.time() - t0, 1); print(json.dumps(r, default=str)[:4000])
    if a.out: Path(a.out).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    sys.exit(0 if r["ok"] else 1)

if __name__ == "__main__": main()
