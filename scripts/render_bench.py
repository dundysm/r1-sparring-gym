"""Render benchmark for the head camera (phase 3, M0.2) -> results/perception/render_bench.json.

States: 60 snapshots (every 0.1 s) of a 6 s bout, the guard policy against the tracked opponent (demo scenario), so the
R1's own forearms are in view. Every timing is per frame and covers what the closed loop pays: sync(qpos) (kinematics),
update_scene, render and pixel readback, on the R1's head camera at the twin's head pitch.
1. Single process, 25 cells: rgb / depth / seg x full meshes (Unitree's) vs LOD (render_lod) x shadows off / on x 160x120
   / 320x240, plus the network frame (LOD 320x240 rgb area-downsampled to 160x120, "network_frame"). The cells are
   interleaved: 10 rounds, each one block per cell (LOD 40 frames, full 10) in a fresh random order, every block of a round
   from the same snapshots, so background load on this shared machine spreads over all cells instead of landing on one.
   Each block first renders 10 untimed frames: after another cell's GL context was current, the first frame takes 70-100 ms
   and the next few 15-35 ms (probed here), a switch the closed loop, one render size per process, never makes. Per cell: the median block, every block, and the background CPU load during its blocks
   (system busy minus this process, % of all cores, median over blocks; null without psutil, which is not a dependency).
   Gate, fixed before this run: LOD, shadows off, 320x240 rgb >= 120 fps in the MEDIAN block. The fastest block and part
   2's 1-process run are stored as secondary numbers only. (The first version gated on the fastest block, a statistic
   picked after its median had failed.)
2. Processes: 1 / 2 / 4 / 8 processes on this one GPU, each with its own twin and GL context, rendering the snapshots
   (320x240 rgb, LOD, no shadows) after a common start signal; aggregate frames per second.
3. Determinism: rgb / depth / seg of every snapshot rendered twice by one renderer, by a fresh twin (new GL context) in
   the same process, and (rgb) in each worker process; identical frames counted, else the largest pixel difference.
The machine is shared and timings move with background load: background_cpu_pct per cell, cpu_busy_pct before part 2."""
import sys, json, time, hashlib, argparse, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

KINDS, SIZES = ("rgb", "depth", "seg"), ((160, 120), (320, 240)); GATE_CELL = "lod 320x240 shadows_off rgb"; GATE_FPS = 120; WARM = 10
try: import psutil; _ME = psutil.Process()
except ImportError: psutil = None   # optional (not a dependency): loads are then recorded as null

def snapshots(n=60, every=5):
    from r1gym.model import build_spec
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.scenario import demo_env
    c = load_controller("guard=results/policy_mjlab_guard/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    m = build_spec(opponent=True, cfg=cfg).compile(); e, o = demo_env(m, cfg, t0=1.0, seconds=n * every * 0.02); Q = []
    for t in range(n * every):
        o, *_ = e.step(c.act(o[None])[0])
        if (t + 1) % every == 0: Q.append(e.d.qpos.copy())
    return np.array(Q), cfg, m

def load():
    """Background CPU load since the previous call [% of all cores]: system busy minus this process (None without psutil)."""
    if psutil is None: return None
    return max(0.0, psutil.cpu_percent(None) - _ME.cpu_percent(None) / psutil.cpu_count())

def busy():
    """System CPU busy over 1 s [%] (None without psutil)."""
    return None if psutil is None else psutil.cpu_percent(interval=1.0)

def block(tw, Q, kind, size, shadows, k, i0, warm=WARM):
    """(ms per frame over k frames from snapshot i0, background load meanwhile), after `warm` untimed frames: the cells
    take turns, so a block starts by switching kind, shadows or size (another GL context: see the docstring)."""
    f = (lambda: tw.frame()) if kind == "frame" else (lambda: tw.render("head", kind, size, shadows))
    for i in range(warm): tw.sync(Q[(i0 + i) % len(Q)]); f()
    load(); t0 = time.perf_counter()
    for i in range(k): tw.sync(Q[(i0 + i) % len(Q)]); f()
    return (time.perf_counter() - t0) / k * 1e3, load()

def worker(rank, qfile, rfile, pitch, k, conn):
    """One process: its own twin and GL context. Ready -> wait for "go" -> render k frames -> report. Pipes only: a
    multiprocessing Barrier/Queue hung here once (a semaphore raised WinError 5 under the sandbox)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from r1gym.perception.render_scene import RenderTwin
    from r1gym.model import load_cfg
    from r1gym import CONFIGS
    Q, ref = np.load(qfile), np.load(rfile); tw = RenderTwin({**load_cfg(CONFIGS / "r1_mjlab.yaml"), "opponent_mode": "policy"}, head_pitch_deg=pitch)
    diff = []
    for i, q in enumerate(Q): tw.sync(q); x = tw.render("head"); diff.append(int(np.abs(x.astype(int) - ref[i]).max()))   # also the warm-up
    conn.send("ready"); conn.recv(); t0 = time.time()
    for i in range(k): tw.sync(Q[i % len(Q)]); tw.render("head")
    conn.send(dict(rank=rank, t0=t0, t1=time.time(), frames=k, identical=sum(x == 0 for x in diff), max_diff=max(diff))); tw.close()

def procs(Q, ref, pitch, P, k):
    import multiprocessing as mp
    ctx = mp.get_context("spawn"); d = Path(tempfile.mkdtemp(prefix="r1bench_")); qf, rf = str(d / "q.npy"), str(d / "ref.npy"); np.save(qf, Q); np.save(rf, ref)
    pipes = [ctx.Pipe() for _ in range(P)]; ps = [ctx.Process(target=worker, args=(r, qf, rf, pitch, k, pipes[r][1]), daemon=True) for r in range(P)]
    for p in ps: p.start()
    try:
        for a, _ in pipes:
            if not a.poll(300): raise TimeoutError("render worker not ready in 300 s")
            a.recv()
        for a, _ in pipes: a.send("go")
        rs = []
        for a, _ in pipes:
            if not a.poll(300): raise TimeoutError("render worker did not finish in 300 s")
            rs.append(a.recv())
    finally:
        for p in ps:
            p.join(5)
            if p.is_alive(): p.kill()
    wall = max(r["t1"] for r in rs) - min(r["t0"] for r in rs)   # all loops, first start to last end (wall clock)
    return dict(processes=P, frames_total=P * k, frames_per_s=round(P * k / wall, 1), ms_per_frame_per_process=round(float(np.mean([(r["t1"] - r["t0"]) / r["frames"] * 1e3 for r in rs])), 2),
                rgb_identical_to_main_process=f"{sum(r['identical'] for r in rs)}/{P * len(Q)}", rgb_max_abs_diff=max(r["max_diff"] for r in rs))

def shots(tw, Q, kind):
    out = []
    for q in Q: tw.sync(q); out.append(tw.render("head", kind))
    return out

def determinism(cfg, Q, tw, pitch):
    """Returns the comparison and the rgb reference frames the worker processes compare against."""
    from r1gym.perception.render_scene import RenderTwin
    out = {}; tw2 = RenderTwin(cfg, head_pitch_deg=pitch); md = lambda x, y: float(max(np.abs(np.asarray(u, float) - np.asarray(v, float)).max() for u, v in zip(x, y)))
    for kind in KINDS:
        a, b, c = shots(tw, Q, kind), shots(tw, Q, kind), shots(tw2, Q, kind); ref = np.array(a) if kind == "rgb" else ref
        out[kind] = dict(same_renderer_identical=f"{sum(np.array_equal(u, v) for u, v in zip(a, b))}/{len(Q)}", same_renderer_max_abs_diff=md(a, b),
                         fresh_context_identical=f"{sum(np.array_equal(u, v) for u, v in zip(a, c))}/{len(Q)}", fresh_context_max_abs_diff=md(a, c),
                         sha1_first=hashlib.sha1(np.ascontiguousarray(a[0]).tobytes()).hexdigest())
    tw2.close(); return out, ref

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/perception/render_bench.json"); ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--frames", type=int, default=40); ap.add_argument("--full_frames", type=int, default=10)
    ap.add_argument("--proc_frames", type=int, default=240); a = ap.parse_args()
    from r1gym.perception import run_meta
    from r1gym.perception.render_scene import RenderTwin
    t0 = time.time(); Q, cfg, m = snapshots()
    twins = {"lod": RenderTwin(cfg, physics=m, head_pitch_deg=a.pitch) if a.pitch is not None else RenderTwin(cfg, physics=m)}
    pitch = twins["lod"].pitch; twins["full"] = RenderTwin(cfg, physics=m, lod=False, head_pitch_deg=pitch)
    res = dict(about=__doc__, **run_meta(gl=twins["lod"].gl()), head_pitch_deg=pitch, snapshots=len(Q), faces_total={k: int(t.m.nmeshface) for k, t in twins.items()},
               rounds=a.rounds, frames_per_block=dict(lod=a.frames, full=a.full_frames, untimed_first=WARM))
    cells = [(f"{n} {w}x{h} shadows_{'on' if sh else 'off'} {kind}", n, kind, (w, h), sh) for n in twins for (w, h) in SIZES for sh in (False, True) for kind in KINDS]
    cells.append(("network_frame", "lod", "frame", None, False)); blk, bg = {c[0]: [] for c in cells}, {c[0]: [] for c in cells}; load()
    for r in range(a.rounds):
        i0 = r * a.frames % len(Q)
        for j in np.random.default_rng(r).permutation(len(cells)):
            key, n, kind, size, sh = cells[j]; ms, b = block(twins[n], Q, kind, size, sh, a.frames if n == "lod" else a.full_frames, i0)
            blk[key].append(round(ms, 2)); bg[key].append(None if b is None else round(b, 1))
        print("round", r, blk[GATE_CELL][-1], "ms gate cell, background", bg[GATE_CELL][-1], f"{time.time() - t0:.0f}s", flush=True)
    med = lambda v: None if None in v else round(float(np.median(v)), 2)
    res.update(single_process_ms={k: med(v) for k, v in blk.items()}, background_cpu_pct={k: med(v) for k, v in bg.items()}, single_process_block_ms=blk, background_cpu_pct_blocks=bg)
    g = res["single_process_ms"][GATE_CELL]
    res["gate"] = dict(rule=f"LOD, shadows off, 320x240 rgb, one process: >= {GATE_FPS} fps in the median of {a.rounds} interleaved blocks of {a.frames} frames (fixed before the run)",
        ms_median_block=g, fps=round(1e3 / g, 1), passed=bool(1e3 / g >= GATE_FPS), background_cpu_pct=res["background_cpu_pct"][GATE_CELL],
        fps_fastest_block=round(1e3 / min(blk[GATE_CELL]), 1), full_mesh_fps=round(1e3 / res["single_process_ms"]["full 320x240 shadows_off rgb"], 1))
    twins["full"].close(); print("gate", res["gate"], flush=True)
    tw = twins["lod"]; res["determinism"], ref = determinism(cfg, Q, tw, pitch); print(res["determinism"], flush=True)
    res["cpu_busy_pct_before_processes"] = busy(); res["processes"] = [procs(Q, ref, pitch, P, a.proc_frames) for P in (1, 2, 4, 8)]
    res["gate"]["fps_one_process_run"] = res["processes"][0]["frames_per_s"]   # secondary: part 2, 1 process, proc_frames frames
    for r in res["processes"]: print(r, flush=True)
    tw.close(); res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print(f"{res['wall_s']} s")
