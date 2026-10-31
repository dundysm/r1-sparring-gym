"""Records CPU eval states for the full-body M2 v2 module checks (M2V2_PLAN_v3 M2 G0-1..G0-3; M3 G0-4 and M4 G0-5 read the
same file): the eval's own bout, eval_opponent.run (R1HitEnv, the tracker opponent on its motors, aim standoff+armwarp_vz =
results/opponent_tracker/aim.json, hard-tier distances, 20 s, DR on, obs and act delay as drawn, no pushes) for G on the train
clips at the record seeds 60000-60015 (fb_seeds role 'record'; seed s, k = s - 60000 -> clip TRAIN[k % 3], distance HARD[(k // 3) % 4]).
Episodes come from eval_opponent.episodes (one per seed, its own draws of t0, obs delay, env and hit seeds) with the clean
starts of fb_motion.npz (equal to the eval's clean_starts, M1 (3)).
Hooks, read-only: the shared Tracker's reference() (the clip reference before the aim) and act() (state before and after), and
the np.concatenate that builds the 132-d obs inside Tracker.act (the float64 row before its float32 cast). Each bout also runs
without hooks first; the two rows must be identical (asserted), so the recording is the eval bout.
Per control step of the opponent (arrays over all bouts, `bout` indexes meta.episodes): t (the clock time passed to act), time,
fi, g (motion row = time_step), frame (clip frame), qpos, qvel (the whole model, before act), ctrl (after), opponent torso
xpos/xquat/xmat and pelvis xpos, defender pelvis/torso xpos and torso xmat, target, state before (yaw, spot, axis: nan when
unset; last, dist) and after (yaw, spot, axis, dist, last = the action), base reference (rq0, rdq0, apos, aori), obs64, obs32.
Usage: python scripts/fb_record.py --out rec.npz [--seeds 60000-60015] [--workers 4] [--seconds 20]"""
import sys, json, time, argparse, hashlib, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

CTRL = "guard_fw_r=results/policy_mjlab_guard_footwork_r/policy.onnx"   # G, the reference defender
TRAIN = ("kfab_200", "kfab_201", "kfab_203"); HARD = (0.4, 0.45, 0.5, 0.55); AIM = "standoff+armwarp_vz"
MOTION = ROOT / "gpu" / "motions" / "fb" / "fb_motion.npz"

BASE = 60000   # the record block's first seed: seed s -> k = s - BASE, independent of how the seeds are split over workers
def assign(seeds): return [(TRAIN[(int(s) - BASE) % 3], HARD[((int(s) - BASE) // 3) % 4], int(s)) for s in seeds]

def episodes(seeds, seconds):
    """eval_opponent.episodes, one per seed, with fb_motion's clean masks (= the eval's clean_starts)."""
    import eval_opponent as EO
    with np.load(MOTION) as z: clips, D, clean = z["clips"].tolist(), z["distances"], z["clean"]
    out = []
    for c, d, s in assign(seeds):
        j = int(np.flatnonzero(np.isclose(D, d))[0]); EO.TIERS["fb_record"] = dict(EO.TIERS["hard"], seed=s, distances=[d])
        out += EO.episodes([c], 1, "fb_record", {(c, d): clean[clips.index(c), j]}, hits_on=False, seconds=seconds)
    assert [e["seed"] for e in out] == list(seeds) and {e["clip"] for e in out} <= set(TRAIN); return out

def setup():
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.model import build_spec
    from r1gym.scenario import REF_POSES
    from r1gym.model import HOME
    lab, p = CTRL.split("="); c = load_controller(f"{lab}={ROOT / p}"); aim = json.load(open(ROOT / "results" / "opponent_tracker" / "aim.json"))["operator"]
    assert aim == AIM, aim
    cfg = {**controller_cfg(c), "opponent_mode": "policy", "opponent_aim": aim}
    assert any(np.allclose(np.array(cfg.get("home", HOME), float), p_, atol=1e-3) for p_ in REF_POSES.values()), "G's home pose outside REF_POSES: masks differ"
    return c, cfg, build_spec(opponent=True, cfg=cfg).compile()

class _NpRec:
    """numpy for r1gym.opponent while recording: identical results; keeps the last 132-d concatenate (Tracker.act's obs)."""
    def __init__(self, np_): self._np = np_; self.last = None
    def __getattr__(self, k): return getattr(self._np, k)
    def concatenate(self, *a, **k):
        r = self._np.concatenate(*a, **k)
        if r.shape == (132,): self.last = r.copy()
        return r

class Recorder:
    def __init__(self, tracker, r1_pel):
        self.tr, self.pel, self.rows = tracker, r1_pel, []
    def install(self):
        import r1gym.opponent as OP
        self.OP, self.np0 = OP, OP.np; self.npr = _NpRec(OP.np); OP.np = self.npr
        tr = self.tr; self.act0, self.ref0 = type(tr).act.__get__(tr), type(tr).reference.__get__(tr)
        tr.reference = self.reference; tr.act = self.act; return self
    def remove(self):
        self.OP.np = self.np0; del self.tr.reference; del self.tr.act
    def reference(self, replay, f, state):
        out = self.ref0(replay, f, state); self.base = (int(f), *[np.array(x, np.float64) for x in out]); return out
    def act(self, m, d, replay, state, t, dt=0.02):
        g = lambda k, n: np.array(state[k], np.float64) if k in state else np.full(n, np.nan)
        ob, pb, r1t = replay.torso, replay.pel, state["r1_torso"]; aim = state.get("aim")
        x = dict(t=float(t), time=float(d.time), qpos=d.qpos.copy(), qvel=d.qvel.copy(), opp_torso_xpos=d.xpos[ob].copy(), opp_torso_xquat=d.xquat[ob].copy(),
                 opp_torso_xmat=d.xmat[ob].copy(), opp_pel_xpos=d.xpos[pb].copy(), r1_pel_xpos=d.xpos[self.pel].copy(), r1_torso_xpos=d.xpos[r1t].copy(),
                 r1_torso_xmat=d.xmat[r1t].copy(), target=g("target", 2), yaw_pre=g("yaw", 1).reshape(1), spot_pre=g("spot", 2), axis_pre=g("axis", 2),
                 last_pre=np.array(state["last"], np.float64), dist_pre=g("dist", 1).reshape(1), offset=0 if aim is None else int(aim.offset))
        self.npr.last = None; self.base = None; self.act0(m, d, replay, state, t, dt)
        fi = int(round(t * replay.fps * replay.speed)) + x["offset"]; gr, rq0, rdq0, apos, aori = self.base
        x.update(fi=fi, g=gr, frame=fi % replay.T, ctrl=d.ctrl[replay.act].copy(), yaw=g("yaw", 1).reshape(1), spot=g("spot", 2), axis=g("axis", 2), dist=g("dist", 1).reshape(1),
                 action=np.array(state["last"], np.float64), rq0=rq0, rdq0=rdq0, apos=apos, aori=aori, obs64=self.npr.last, obs32=self.npr.last.astype(np.float32))
        self.rows.append(x)

def work(args):
    seeds, seconds = args
    import eval_opponent as EO
    from r1gym.opponent import FightReplay
    from fb_seeds import check
    check(seeds, "record", TRAIN); c, cfg, m = setup(); eps = episodes(seeds, seconds); out = []
    for ep in eps:
        plain = EO.run(c, [ep], m, cfg, seconds, threads=1, dr=True, meter=False, falls=True)[0]
        tr = FightReplay(m, ep["clip"], distance=ep["distance"], cfg=cfg).tracker; rec = Recorder(tr, m.body("pelvis").id).install()
        try: row = EO.run(c, [ep], m, cfg, seconds, threads=1, dr=True, meter=False, falls=True)[0]
        finally: rec.remove()
        assert json.dumps(row, sort_keys=True, default=str) == json.dumps(plain, sort_keys=True, default=str), f"recording changed bout {ep['seed']}"
        assert all(r["obs64"] is not None for r in rec.rows), "an act without its 132-d obs"
        out.append((ep, row, rec.rows))
    return out

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="60000-60015")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seconds", type=float, default=20.0); a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-")); seeds = list(range(lo, hi + 1)); assert a.workers <= 4, "at most 4 worker processes"
    from fb_seeds import check
    check(seeds, "record", TRAIN); t0 = time.time()
    sl = [seeds[i::a.workers] for i in range(a.workers) if seeds[i::a.workers]]
    if len(sl) == 1: parts = [work((sl[0], a.seconds))]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(sl)) as pool: parts = pool.map(work, [(s, a.seconds) for s in sl], chunksize=1)
    bouts = sorted((b for p in parts for b in p), key=lambda b: b[0]["seed"])
    keys = list(bouts[0][2][0]); A = {k: np.stack([np.asarray(r[k]) for b in bouts for r in b[2]]) for k in keys}
    A["bout"] = np.concatenate([np.full(len(b[2]), i, np.int32) for i, b in enumerate(bouts)])
    import mujoco, torch
    meta = dict(seeds=seeds, role="record", ctrl=CTRL, aim=AIM, seconds=a.seconds, dr=True, pushes=False, episodes=[b[0] for b in bouts], rows=[b[1] for b in bouts],
                steps=[len(b[2]) for b in bouts], fb_motion_sha256=sha256(MOTION), versions=dict(mujoco=mujoco.__version__, numpy=np.__version__, torch=torch.__version__,
                python=platform.python_version()), wall_s=round(time.time() - t0, 1), read_only="each bout's row equals the same bout run without hooks")
    A["meta"] = np.array(json.dumps(meta, default=str)); Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out, **A)
    print(json.dumps(dict(out=a.out, bouts=len(bouts), steps=int(len(A["bout"])), falls=sum(r["fell_s"] is not None for r in meta["rows"]),
                          opp_fell=sum(bool(r["opp_fell"]) for r in meta["rows"]), wall_s=meta["wall_s"])), flush=True)

if __name__ == "__main__": main()
