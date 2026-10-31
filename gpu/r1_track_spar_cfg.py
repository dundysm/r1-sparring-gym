"""R1-Track-Spar: the T1 opponent tracker of the full-body M2 v2 work (M2V2_PLAN_v3 2.2 and section 4 P2; design_learned-tracker 9.2):
the shipped tracker T0 (model_13749.pt, R1-Track-KungFu-Steer2) fine-tuned on library L1 (gpu/export_fb_library.py) plus the 3 train
clips, with the 132-d actor contract unchanged, so T1 drops into r1gym's Tracker, fb_opponent's TrackerDriver and the CPU tools.
  task     Steer2's config (r1_track_task.env_cfg: relaxed hands 0.6 m, feet 0.4 m, steer, strict: anchor std 0.5 m weight 2, anchor_xy
           termination 0.6 m, drift <= 0.3 m/s), plus the Aim task's hand reaction impulses (both wrists, +-25 N for 0.05-0.12 s every
           1-3 s; training only). Not the Aim task's strike_apex reward (it overshot unwarped strikes) and not its lunges.
  motion   fb_l1_track.csv: the 3 originals then the 30 L1 rows of the warp / re-angle / combo families, each twice, cross-faded
           0.4 s (export_fb_library.track_motion); a mirrored-base row un-mirrored, since it is performed by the mirror-conjugated
           tracker (gpu/fb_mirror.py, plan A6.2), and the mirror family left out (conjugated, it is the originals).
           fb_l1_track_index.json next to the converted .npz gives each row's group and conj flag.
  sampling mjlab 1.2.0's adaptive sampling (1 s bins, failure counts + uniform ratio, kernel), then renormalised within the two groups
           and mixed ORIGINAL_SHARE : 1 - ORIGINAL_SHARE, so the originals keep half the sampling weight whatever L1's failures do
           (plan 2.2: half the weight on the originals, to protect them). A bin belongs to the group holding its centre.
  LOCO     R1-Track-Spar-LOCO (P2, 3000 iterations): the same task on the LOCO library (fb_l1_loco.npz: kfab_200/201 only, vmax of
           that clip set, plan A6.3) and its originals; `motion --loco` writes it.
  gates    T1-a..e (design_learned-tracker 9.3, T1-c as plan 3.1 / A5.3) run after training; not here.
On the GPU box, from the unitree_rl_mjlab checkout with r1_track_task.py and this file in scripts/ (as r1_track_task's README):
  python r1_track_spar_cfg.py motion [--loco] --out motions          # in the repo instead (needs gpu/motions/fb): writes the CSV + index
  python r1_track_spar_cfg.py convert fb_l1_track.csv motions/fb_l1_track.npz && cp fb_l1_track_index.json motions/
  python r1_track_spar_cfg.py R1-Track-Spar --env.scene.num-envs 4096 --agent.max-iterations 4000 --agent.logger tensorboard \
    --motion-file motions/fb_l1_track.npz --agent.resume True --agent.load-run <dir with model_13749.pt> --agent.load-checkpoint model_13749.pt"""
import sys, json
from pathlib import Path
import numpy as np

GPU = Path(__file__).resolve().parent
TASK, TASK_LOCO = "R1-Track-Spar", "R1-Track-Spar-LOCO"
MOTION, MOTION_LOCO = "fb_l1_track.csv", "fb_l1_track_loco.csv"   # export_fb_library.names_for
RESUME, ITERATIONS = "model_13749.pt", {TASK: 4000, TASK_LOCO: 3000}
ORIGINAL_SHARE, LOCO_DROP, STEP_DT = 0.5, "kfab_203", 0.02
STEER2 = dict(relaxed_hands=True, feet=0.4, hands=0.6, steer=True, strict=True)      # R1-Track-KungFu-Steer2's env_cfg arguments
HAND_IMPULSE = dict(force_range=(-25.0, 25.0), torque_range=(0.0, 0.0), duration_s=(0.05, 0.12), cooldown_s=(1.0, 3.0))   # the Aim task's

def index_path(motion_file): p = Path(motion_file); return p.with_name(p.stem + "_index.json")

def bin_groups(index, total, step_dt=STEP_DT):
    """Group per adaptive-sampling bin (0 originals, 1 L1) of a motion of `total` frames: MotionCommand's bin_count, and a bin's
    time steps (b + u) / bin_count x (total - 1); the group of the row holding the bin's centre (rows are contiguous, originals first)."""
    bc = int(total // (1 / step_dt)) + 1; c = (np.arange(bc) + 0.5) / bc * (total - 1)
    end = min(r["start_frame"] for r in index["rows"].values() if r["group"] != "original"); assert all(r["start_frame"] < end for r in index["rows"].values() if r["group"] == "original")
    return (c >= end).astype(np.int64)

def group_probs(failed, uniform_ratio, kernel, groups, share=ORIGINAL_SHARE):
    """mjlab 1.2.0 MotionCommand._adaptive_sampling's probabilities (failed + ratio / bins, replicate-padded causal-reversed kernel),
    renormalised within each group and mixed share : 1 - share. torch tensors on one device."""
    import torch
    p = failed + uniform_ratio / float(len(failed)); k = len(kernel)
    p = torch.nn.functional.conv1d(torch.nn.functional.pad(p[None, None], (0, k - 1), mode="replicate"), kernel.view(1, 1, -1)).view(-1)
    g = groups.to(p.device).bool(); o, l = torch.where(g, 0.0, p), torch.where(g, p, 0.0)
    return share * o / o.sum() + (1 - share) * l / l.sum()

def build_motion(loco=False, out=None):
    """The T1 motion from the repo's exports (fb_motion.npz originals, fb_l1.npz or, loco, fb_l1_loco.npz rows; t1_rows: conjugated
    rows un-mirrored, no mirror family). Returns (seq, index); out: also write the CSV and the index json (the export's names)."""
    sys.path.insert(0, str(GPU)); import export_fb_library as L
    f_npz, f_csv, f_ix, f_man = L.names_for(loco)
    with np.load(GPU / "motions" / "fb" / f_npz, allow_pickle=False) as z: A = {k: z[k] for k in ("names", "family", "host", "mirror", "frames", "row0", "q")}
    man = json.load(open(GPU / "motions" / "fb" / f_man, encoding="utf-8")); keep = tuple(man["clips"]); assert keep == (L.LOCO_CLIPS if loco else L.CLIPS)
    rows = [(n, f, h, int(m), None, A["q"][int(r0):int(r0) + int(T)]) for n, f, h, m, r0, T in zip(A["names"].tolist(), A["family"].tolist(), A["host"].tolist(), A["mirror"], A["row0"], A["frames"])]
    seq, index, B = L.track_motion(L.base_motion(), rows, keep)
    ix = dict(fps=L.FPS, joints=list(L.CTRL_JOINTS), frames=int(len(seq)), blend_frames=B, loco_drop=LOCO_DROP if loco else None, rows=index)
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / f_csv, np.concatenate([seq[:, :3], seq[:, [4, 5, 6, 3]], seq[:, 7:]], 1), delimiter=",", fmt="%.6f")
        json.dump(ix, open(out / f_ix, "w", encoding="utf-8"), indent=1)
    return seq, ix

_SPAR = {}
def _spar_cfg(mc):
    """A Steer2 command config -> the same config building a SparMotionCommand (group-mixed adaptive sampling, module docstring)."""
    import dataclasses, torch, r1_track_task as t
    from mjlab.utils.lab_api.math import sample_uniform
    if not _SPAR:
        class SparMotionCommand(t._STEER["cls"]):
            def __init__(self, cfg, env):
                super().__init__(cfg, env); ix = json.load(open(index_path(cfg.motion_file), encoding="utf-8"))
                assert ix["frames"] == self.motion.time_step_total, (ix["frames"], self.motion.time_step_total)
                self._group = torch.as_tensor(bin_groups(ix, self.motion.time_step_total, env.step_dt), device=self.device); assert len(self._group) == self.bin_count
            def _adaptive_sampling(self, env_ids):   # mjlab 1.2.0's, with group_probs in place of its probabilities
                failed = self._env.termination_manager.terminated[env_ids]
                if torch.any(failed):
                    b = torch.clamp((self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1)
                    self._current_bin_failed[:] = torch.bincount(b[env_ids][failed], minlength=self.bin_count)
                p = group_probs(self.bin_failed_count, self.cfg.adaptive_uniform_ratio, self.kernel, self._group, self.cfg.original_share)
                bins = torch.multinomial(p, len(env_ids), replacement=True)
                self.time_steps[env_ids] = ((bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)) / self.bin_count * (self.motion.time_step_total - 1)).long()
                H = -(p * (p + 1e-12).log()).sum(); pmax, imax = p.max(dim=0)
                self.metrics["sampling_entropy"][:] = H / np.log(self.bin_count) if self.bin_count > 1 else 1.0
                self.metrics["sampling_top1_prob"][:] = pmax; self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        @dataclasses.dataclass(kw_only=True)
        class SparMotionCommandCfg(t._STEER["cfg"]):
            original_share: float = ORIGINAL_SHARE
            def build(self, env): return SparMotionCommand(self, env)
        _SPAR["cfg"] = SparMotionCommandCfg
    return _SPAR["cfg"](**{f.name: getattr(mc, f.name) for f in dataclasses.fields(mc)})

def env_cfg(play=False, motion_file=""):
    import r1_track_task as t
    cfg = t.env_cfg(play, motion_file, **STEER2); mc = cfg.commands["motion"]; assert mc.sampling_mode == ("start" if play else "adaptive")
    cfg.commands["motion"] = _spar_cfg(mc)
    if not play:
        from mjlab.envs import mdp as envs_mdp
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        cfg.events["hand_impulse"] = EventTermCfg(func=envs_mdp.apply_body_impulse, mode="step", params={
            **HAND_IMPULSE, "asset_cfg": SceneEntityCfg("robot", body_names=("left_wrist_roll_link", "right_wrist_roll_link"))})
    return cfg

def register(motion_file=""):
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    rl = unitree_g1_tracking_ppo_runner_cfg(); rl.experiment_name = "r1_tracking_spar"; rl.save_interval = 250
    for task in (TASK, TASK_LOCO):
        register_mjlab_task(task_id=task, env_cfg=env_cfg(False, motion_file), play_env_cfg=env_cfg(True, motion_file), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)

if __name__ == "__main__":
    if sys.argv[1] == "motion":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--loco", action="store_true"); ap.add_argument("--out", default=str(GPU / "motions" / "fb")); a = ap.parse_args()
        seq, ix = build_motion(a.loco, a.out); print(len(seq), "frames,", len(ix["rows"]), "rows", "(LOCO)" if a.loco else ""); sys.exit(0)
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts"))
    import r1_track_task as t
    if sys.argv[1] == "convert": t.convert(sys.argv[2], sys.argv[3]); sys.exit(0)
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
