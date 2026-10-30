"""R1-Track-Spar-Reangle(-LOCO): the T1 opponent tracker trained on the reangle family only (M2V2_PLAN_v3 A9.3; P2), on top of
gpu/r1_track_spar_cfg.py (R1-Track-Spar, which is unchanged).
The T1 motion file stays the exported one (fb_l1_track.csv / _loco: originals, then the warp / reangle / combo rows, conjugated rows
un-mirrored), so every L1 row's fed reference (fb_l1.npz ref_*, export_fb_library.fed: csv_to_npz's float32 grid over the whole motion)
is still exactly what T1 is shown. Only the sampling changes: SparMotionCommand's group-mixed adaptive sampling with every bin whose
centre lies in a row of a family outside T1_FAMILIES set to probability 0 (ORIGINAL_SHARE kept on the originals, the rest on the
reangle rows). The warp, combo and mirror families stay with T0 (conjugated where mirrored), which performs them (M8b: >= 0.917).
On the GPU box, next to r1_track_spar_cfg.py: python r1_track_spar_t1.py R1-Track-Spar-Reangle [train.py flags] (gpu/pod_fb.sh P2)."""
import sys, json, types
from pathlib import Path
import numpy as np

GPU = Path(__file__).resolve().parent
if str(GPU) not in sys.path: sys.path.insert(0, str(GPU))
import r1_track_spar_cfg as C

T1_FAMILIES = ("reangle",)                     # = fb_population.T1_FAMILIES (asserted in tests/test_fb_task.py)
TASK, TASK_LOCO = "R1-Track-Spar-Reangle", "R1-Track-Spar-Reangle-LOCO"

def bin_allowed(index, total, step_dt=C.STEP_DT, families=T1_FAMILIES):
    """Per adaptive-sampling bin (C.bin_groups' bins and centres): True iff the row holding the bin's centre is an original or of
    `families`. A row spans its start frame to the next row's start (the last to the end)."""
    bc = int(total // (1 / step_dt)) + 1; c = (np.arange(bc) + 0.5) / bc * (total - 1)
    rows = sorted(index["rows"].values(), key=lambda r: r["start_frame"]); s = np.array([r["start_frame"] for r in rows])
    ok = np.array([r["group"] == "original" or r["group"] in families for r in rows]); return ok[np.searchsorted(s, c, side="right") - 1]

def masked_probs(failed, uniform_ratio, kernel, groups, allowed, share=C.ORIGINAL_SHARE):
    """C.group_probs with the bins not allowed at 0: the same padded causal kernel, renormalised within the originals and the allowed
    L1 bins, mixed share : 1 - share."""
    import torch
    p = failed + uniform_ratio / float(len(failed)); k = len(kernel)
    p = torch.nn.functional.conv1d(torch.nn.functional.pad(p[None, None], (0, k - 1), mode="replicate"), kernel.view(1, 1, -1)).view(-1)
    g, a = groups.to(p.device).bool(), allowed.to(p.device).bool(); o, l = torch.where(g, 0.0, p), torch.where(g & a, p, 0.0)
    return share * o / o.sum() + (1 - share) * l / l.sum()

def _sampling(self, env_ids):
    """SparMotionCommand._adaptive_sampling (r1_track_spar_cfg) with masked_probs."""
    import torch
    from mjlab.utils.lab_api.math import sample_uniform
    failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(failed):
        b = torch.clamp((self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1)
        self._current_bin_failed[:] = torch.bincount(b[env_ids][failed], minlength=self.bin_count)
    p = masked_probs(self.bin_failed_count, self.cfg.adaptive_uniform_ratio, self.kernel, self._group, self._allowed, self.cfg.original_share)
    bins = torch.multinomial(p, len(env_ids), replacement=True)
    self.time_steps[env_ids] = ((bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)) / self.bin_count * (self.motion.time_step_total - 1)).long()
    H = -(p * (p + 1e-12).log()).sum(); pmax, imax = p.max(dim=0)
    self.metrics["sampling_entropy"][:] = H / np.log(self.bin_count) if self.bin_count > 1 else 1.0
    self.metrics["sampling_top1_prob"][:] = pmax; self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

def env_cfg(play=False, motion_file=""):
    """R1-Track-Spar's cfg; its motion command, once built, samples only the allowed bins."""
    import dataclasses, torch
    cfg = C.env_cfg(play, motion_file); mc = cfg.commands["motion"]; base = type(mc)
    def build(self, env):
        cmd = base.build(self, env); ix = json.load(open(C.index_path(self.motion_file), encoding="utf-8"))
        cmd._allowed = torch.as_tensor(bin_allowed(ix, cmd.motion.time_step_total, env.step_dt), device=cmd.device); assert len(cmd._allowed) == cmd.bin_count
        if not play: cmd._adaptive_sampling = types.MethodType(_sampling, cmd)
        return cmd
    K = dataclasses.make_dataclass("ReangleMotionCommandCfg", [], bases=(base,), namespace={"build": build}, kw_only=True)   # a class: deepcopy-safe
    cfg.commands["motion"] = K(**{f.name: getattr(mc, f.name) for f in dataclasses.fields(mc)}); return cfg

def register(motion_file=""):
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    rl = unitree_g1_tracking_ppo_runner_cfg(); rl.experiment_name = "r1_tracking_spar"; rl.save_interval = 250
    for task in (TASK, TASK_LOCO):
        register_mjlab_task(task_id=task, env_cfg=env_cfg(False, motion_file), play_env_cfg=env_cfg(True, motion_file), rl_cfg=rl, runner_cls=MotionTrackingOnPolicyRunner)

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "scripts"))
    import mjlab.tasks, src.tasks  # noqa: F401  (populate the registry first)
    register()
    import train
    train.main()
