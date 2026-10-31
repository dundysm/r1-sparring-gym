"""M2 checks of the full-body opponent driver (M2V2_PLAN_v3 3 M2; gpu/fb_opponent.py) on CPU, against the eval's own code and
the states scripts/fb_record.py recorded from eval bouts (seeds 60000-60015):
  G0-1  TrackerMLP vs the tracker's ONNX on >= 10k recorded obs rows: float32 (as deployed) vs onnxruntime <= 1e-5; float64 vs
        the ONNX graph evaluated in float64 (its float32 initializers upcast) <= 1e-9. Weights and the folded divisor equal the
        checkpoint's bit for bit.
  G0-2  the 132-d obs in float64 <= 1e-6 vs the eval's (the float64 row before Tracker.act's float32 cast): (a) the builder on the
        recorded internals; (b) the driver's own path per bout (placement, clock, fb_motion reference, recurrent facing and
        steering), with the aim's warp, stand-off distance and the last action taken from the recording.
  G0-3  facing yaw, spot and axis, recurrent over the 16 recorded bouts from the recorded targets: <= 1e-6 (float64).
  G0-3b tracker alone (scripts/eval_tracker.run verbatim, 10 s at 4 m, no aim): 24 starts x 3 train clips, the eval's Tracker vs
        the driver (MjTracker) in float32 and float64: fall times within 0.1 s, or both stay up. Also placement vs FightReplay.place.
The tracker ONNX is read here only as the reference (via the eval config); the driver reads model_13749.pt and fb_motion.npz.
Run in the repo: python scripts/fb_m2_check.py --rec rec.npz [--out m2_check.json] [--workers 4] [--starts 24] [--skip-3b]"""
import sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, torch
import fb_opponent as FO

TOL = dict(g01_f32=1e-5, g01_f64=1e-9, g02=1e-6, g03=1e-6, g03b_s=0.1)
TRAIN = FO.TRAIN

def onnx_path():
    from r1gym.model import load_cfg
    return str(ROOT / load_cfg()["opponent_policy"])

def load_rec(path):
    with np.load(path, allow_pickle=False) as z: R = {k: z[k] for k in z.files}
    R["meta"] = json.loads(str(R["meta"])); return R

def onnx_f64(path):
    """The tracker graph's obs -> actions path (Sub, Div, Gemm, Elu) evaluated in float64 on its own initializers, upcast."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(path).graph; init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    nodes = [n for n in g.node if n.op_type in ("Sub", "Div", "Gemm", "Elu")]
    def run(x, dtype=np.float64):
        env = {"obs": np.asarray(x, dtype)}; env.update({k: v.astype(dtype) for k, v in init.items() if v.dtype.kind == "f"})
        for n in nodes:
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}; i = [env[k] for k in n.input]
            if n.op_type == "Sub": y = i[0] - i[1]
            elif n.op_type == "Div": y = i[0] / i[1]
            elif n.op_type == "Elu": y = np.where(i[0] > 0, i[0], a.get("alpha", 1.0) * (np.exp(i[0]) - 1))
            else:
                assert a.get("transB") == 1 and a.get("alpha", 1.0) == 1.0 and a.get("beta", 1.0) == 1.0 and not a.get("transA"), a
                y = i[0] @ i[1].T + i[2]
            env[n.output[0]] = y.astype(dtype)
        return env["actions"]
    return run, init

def g0_1(R, path=None, n_min=10_000):
    import onnxruntime as ort
    path = path or onnx_path(); run64, init = onnx_f64(path); pt = torch.load(FO.TRACKER_PT, map_location="cpu", weights_only=False)["actor_state_dict"]
    same = {k: bool(np.array_equal(init[f"policy.{k}"], pt[k].numpy())) for k in ("obs_normalizer._mean", *[f"mlp.{i}.{w}" for i in (0, 2, 4, 6) for w in ("weight", "bias")])}
    div = [k for k in init if k.startswith("onnx::Div")]; assert len(div) == 1
    same["divisor = std + 0.01 (float32)"] = bool(np.array_equal(init[div[0]], (pt["obs_normalizer._std"] + 0.01).numpy()))
    o32, o64 = R["obs32"], R["obs64"]; n = len(o32); assert n >= n_min, f"{n} recorded rows < {n_min}"
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    ref32 = np.concatenate([s.run(["actions"], {"obs": o32[i:i + 1], "time_step": R["g"][i:i + 1, None].astype(np.float32)})[0] for i in range(n)])
    assert np.array_equal(ref32.astype(np.float64), R["action"]), "onnxruntime here differs from the recorded eval actions"
    m32, m64 = FO.TrackerMLP(dtype=torch.float32), FO.TrackerMLP(dtype=torch.float64)
    t32 = m32(torch.from_numpy(o32)).numpy(); t64 = m64(torch.from_numpy(o64)).numpy(); r64 = run64(o64)
    e = lambda a, b: float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
    t32b = np.concatenate([m32(torch.from_numpy(o32[i:i + 1])).numpy() for i in range(0, n, max(1, n // 500))])   # batch 1 (the CPU bridge)
    res = dict(rows=n, weights_bit_equal=same, f32_torch_vs_onnxruntime=e(t32, ref32), f32_batch1_vs_onnxruntime=e(t32b, ref32[::max(1, n // 500)]),
               f64_torch_vs_onnx_graph_f64=e(t64, r64), info_f64_vs_onnxruntime_f32=e(t64, ref32), info_graph_f32_vs_onnxruntime=e(run64(o32, np.float32), ref32),
               action_absmax=float(np.abs(ref32).max()))
    res["ok"] = all(same.values()) and res["f32_torch_vs_onnxruntime"] <= TOL["g01_f32"] and res["f32_batch1_vs_onnxruntime"] <= TOL["g01_f32"] and res["f64_torch_vs_onnx_graph_f64"] <= TOL["g01_f64"]
    return res

def _bouts(R):
    eps = R["meta"]["episodes"]
    for i, ep in enumerate(eps):
        ix = np.flatnonzero(R["bout"] == i); yield ep, ix

def _opp(R):
    from r1gym.model import build_spec, load_cfg
    m = build_spec(opponent=True, cfg=load_cfg()).compile(); j = m.joint("opp_floating_base_joint").id
    return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])

def g0_2_3(R, dtype=torch.float64):
    """G0-2 (a) and (b), G0-3, the placement and clock on the recorded bouts, in `dtype` (float64 gates; float32 reported)."""
    qa, va = _opp(R); T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64); R = dict(R, opp_pel_xy=R["opp_pel_xpos"][:, :2])
    ea, eb, ey, es, ex, eq, ev, eref, eanc = [], [], [], [], [], [], [], [], []; fi_ok = True; nstep = 0
    for ep, ix in _bouts(R):
        k = TRAIN.index(ep["clip"]); drv = FO.TrackerDriver(1, dtype); qp, qv = drv.reset(None, k, ep["t0"], ep["distance"])
        eq.append(float((qp[0].numpy() - R["qpos"][ix[0], qa:qa + 31]).__abs__().max())); ev.append(float(np.abs(qv[0].numpy() - R["qvel"][ix[0], va:va + 30]).max()))
        for i in ix:
            c = lambda x: T(x[i])[None].to(dtype); w = lambda x: T(x)[None].to(dtype)
            q, dq, av = w(R["qpos"][i, qa + 7:qa + 31]), w(R["qvel"][i, va + 6:va + 30]), w(R["qvel"][i, va + 3:va + 6])
            o64 = R["obs64"][i]; rq, rdq = T(o64[:24])[None].to(dtype), T(o64[24:48])[None].to(dtype)
            # (a) builder only: the eval's own yaw / spot after this step, its reference and last action
            drv_a = FO.TrackerDriver.__new__(FO.TrackerDriver); drv_a.__dict__.update(drv.__dict__)
            drv_a.yaw, drv_a.spot, drv_a.last = c(R["yaw"]).reshape(1), c(R["spot"]), c(R["last_pre"])
            oa = drv_a.observation(rq, rdq, c(R["apos"]), c(R["aori"]), c(R["opp_torso_xpos"]), c(R["opp_torso_xquat"]), av, q, dq, c(R["opp_torso_xmat"]).reshape(1, 3, 3))
            ea.append(float(np.abs(oa[0].double().numpy() - o64).max()))
            # (b) the driver's path: clock, fb_motion reference + the aim's warp, own facing / steering, torso matrix from the quaternion
            fi = drv.clock(R["t"][i]); fi_ok &= int(fi) == int(R["fi"][i]) and int(drv.M.row(drv.clip, fi)) - int(drv.M.row0[k]) == int(R["frame"][i])
            brq, brdq, apos, aori = drv.reference(fi)
            eref.append(max(float(np.abs(brq[0].double().numpy() - R["rq0"][i]).max()), float(np.abs(brdq[0].double().numpy() - R["rdq0"][i]).max())))
            eanc.append(max(float(np.abs(apos[0].double().numpy() - R["apos"][i]).max()), float(np.abs(aori[0].double().numpy() - R["aori"][i]).max())))
            wq, wdq = T(o64[:24] - R["rq0"][i])[None].to(dtype), T(o64[24:48] - R["rdq0"][i])[None].to(dtype)
            dist = c(R["dist"]).reshape(1) if np.isfinite(R["dist"][i]).all() else drv.dist
            drv.face(c(R["target"]), c(R["opp_pel_xy"])); drv.steer(c(R["target"]), dist); drv.last = c(R["last_pre"])
            ob = drv.observation(brq + wq, brdq + wdq, apos, aori, c(R["opp_torso_xpos"]), c(R["opp_torso_xquat"]), av, q, dq)
            eb.append(float(np.abs(ob[0].double().numpy() - o64).max()))
            ey.append(float(np.abs(FO.wrap(drv.yaw.double() - T(R["yaw"][i])).numpy()).max())); es.append(float(np.abs(drv.spot[0].double().numpy() - R["spot"][i]).max()))
            ex.append(float(np.abs(drv.axis[0].double().numpy() - R["axis"][i]).max())); nstep += 1
    mx = lambda x: float(max(x))
    res = dict(dtype=str(dtype).split(".")[-1], bouts=len(R["meta"]["episodes"]), steps=nstep, clock_and_frame_equal=bool(fi_ok),
               g02a_builder=mx(ea), g02b_driver_path=mx(eb), g03_yaw=mx(ey), g03_spot=mx(es), g03_axis=mx(ex),
               placement_qpos=mx(eq), placement_qvel=mx(ev), reference_joint_vs_eval=mx(eref), reference_anchor_vs_eval=mx(eanc))
    res["g02_ok"] = fi_ok and res["g02a_builder"] <= TOL["g02"] and res["g02b_driver_path"] <= TOL["g02"]
    res["g03_ok"] = max(res["g03_yaw"], res["g03_spot"], res["g03_axis"]) <= TOL["g03"]
    return res

# ---- G0-3b: tracker alone, scripts/eval_tracker.run verbatim with the driver swapped in for the Tracker
def _alone(args):
    clip, s, starts, mode, *sec = args; seconds = sec[0] if sec else 10.0
    torch.set_num_threads(1)
    import eval_tracker as ET
    from r1gym.model import build_spec, load_cfg
    from r1gym.opponent import FightReplay
    cfg = load_cfg(); assert cfg.get("opponent_aim", "none") == "none" and cfg["opponent_mode"] == "policy"
    m = build_spec(opponent=True, cfg=cfg).compile(); T = FightReplay(m, clip, distance=4.0, cfg=cfg).T / 50.0; t0 = s * T / starts; place = {}
    assert ET.FightReplay is FightReplay, "eval_tracker left patched"
    if mode != "eval":
        dt = torch.float32 if mode == "torch32" else torch.float64
        def make(m_, clip_, distance, cfg):
            r = FightReplay(m_, clip_, distance=distance, cfg=cfg); mt = FO.MjTracker(r, dt)
            import mujoco
            d0 = mujoco.MjData(m_); d1 = mujoco.MjData(m_); r.place(m_, d0, t0); mt.place(m_, d1, t0)   # placement vs FightReplay.place
            place["qpos"] = float(np.abs(d0.qpos - d1.qpos).max()); place["qvel"] = float(np.abs(d0.qvel - d1.qvel).max())
            return mt.install().replay
        ET.FightReplay = make
    try: t = time.time(); fell, j, z = ET.run(m, cfg, clip, t0, seconds)
    finally: ET.FightReplay = FightReplay   # a pool worker runs eval and driver jobs one after another
    return dict(clip=clip, start=s, t0=round(t0, 4), mode=mode, seconds=seconds, fell=fell, joint_rmse=round(float(np.mean(j)), 4) if j else None, wall_s=round(time.time() - t, 2), **{f"place_{k}": v for k, v in place.items()})

def g0_3b(starts=24, workers=4):
    jobs = [(c, s, starts, mode) for c in TRAIN for s in range(starts) for mode in ("eval", "torch32", "torch64")]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool: out = pool.map(_alone, jobs, chunksize=1)
    by = {(r["clip"], r["start"], r["mode"]): r for r in out}; res = {}
    for mode in ("torch32", "torch64"):
        bad, dts, both_up, both_fell = [], [], 0, 0
        for c in TRAIN:
            for s in range(starts):
                a, b = by[(c, s, "eval")]["fell"], by[(c, s, mode)]["fell"]
                if a is None and b is None: both_up += 1
                elif a is not None and b is not None and abs(a - b) <= TOL["g03b_s"] + 1e-9: both_fell += 1; dts.append(abs(a - b))
                else: bad.append(dict(clip=c, start=s, eval=a, driver=b))
        res[mode] = dict(runs=starts * len(TRAIN), both_up=both_up, both_fell_within=both_fell, max_dt_s=max(dts) if dts else None, mismatches=bad,
                         place_qpos=max(r["place_qpos"] for r in out if r["mode"] == mode), place_qvel=max(r["place_qvel"] for r in out if r["mode"] == mode), ok=not bad)
    res["eval_falls"] = {c: [by[(c, s, "eval")]["fell"] for s in range(starts)] for c in TRAIN}
    res["wall_s_mean"] = {mode: round(float(np.mean([r["wall_s"] for r in out if r["mode"] == mode])), 2) for mode in ("eval", "torch32", "torch64")}
    res["ok"] = res["torch32"]["ok"] and res["torch64"]["ok"]; return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rec", required=True); ap.add_argument("--out", default=""); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--starts", type=int, default=24); ap.add_argument("--skip-3b", action="store_true"); a = ap.parse_args()
    assert a.workers <= 4; FO.pin_precision(); assert FO.precision_ok(); R = load_rec(a.rec); t0 = time.time(); res = dict(tol=TOL, rec=a.rec, rec_meta={k: R["meta"][k] for k in ("seeds", "aim", "seconds", "steps", "fb_motion_sha256", "versions")})
    res["G0-1"] = g0_1(R); print("G0-1", json.dumps(res["G0-1"]), flush=True)
    res["G0-2/3 f64"] = g0_2_3(R, torch.float64); print("G0-2/3 f64", json.dumps(res["G0-2/3 f64"]), flush=True)
    res["info f32"] = g0_2_3(R, torch.float32); print("info f32", json.dumps(res["info f32"]), flush=True)
    if not a.skip_3b: res["G0-3b"] = g0_3b(a.starts, a.workers); print("G0-3b", json.dumps({k: v for k, v in res["G0-3b"].items() if k != "eval_falls"}), flush=True)
    res["ok"] = dict(G0_1=res["G0-1"]["ok"], G0_2=res["G0-2/3 f64"]["g02_ok"], G0_3=res["G0-2/3 f64"]["g03_ok"], **({"G0_3b": res["G0-3b"]["ok"]} if "G0-3b" in res else {}))
    res["wall_s"] = round(time.time() - t0, 1); print(json.dumps(res["ok"]))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if all(res["ok"].values()) else 1)

if __name__ == "__main__": main()
