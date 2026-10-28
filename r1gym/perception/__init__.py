"""Phase 3 perception: the R1 head camera (camera.py: spec, mount kinematics, lens models, rectifier, clocks).

run_meta() is the one metadata block every perception result JSON carries, so perception_report.py (M9) and the README
checks read a single schema."""
import subprocess, platform, numpy as np, mujoco
from .. import ROOT

def run_meta(gl=None):
    """git hash plus the paths that differ from it (empty: the result reproduces from that commit alone), MuJoCo and numpy
    versions, platform, and the GL strings of the context that rendered (gl: {gl_vendor, gl_renderer, gl_version}; None:
    rendered nothing, gl_renderer null)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout
    return dict(git=git("rev-parse", "HEAD").strip(), git_dirty=sorted(l[3:] for l in git("status", "--porcelain", "-uall").splitlines()),
                mujoco=mujoco.__version__, numpy=np.__version__, platform=platform.platform(), cpu=platform.processor(), **(gl or dict(gl_renderer=None)))
