"""Unaimed vs aimed, side by side -> media/aim_frames.png (+ results/opponent_aim/aim_frames.json: which bout, times, GL).
The aim check's arms-down bout (train + val grid) where M1's operator (results/opponent_tracker/aim.json) added the most strike
windows ending on the head, re-run both ways; one row per aimed window that ended on the head, at its reference apex."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from PIL import Image, ImageDraw
from r1gym.model import build_spec
from r1gym.env import R1HitEnv
from r1gym.opponent import FightReplay
from r1gym.hits import HitMeter
from r1gym.policy import load_controller, controller_cfg
from strike_sheet import gl_info, font

def main():
    import eval_opponent as eo
    op = json.load(open("results/opponent_tracker/aim.json"))["operator"]; raw = json.load(open("results/opponent_aim/aim_check_raw.json"))["raw"]["arms_down"]
    gain = lambda a, s: sum(i["outcome"] == "head" for i in a["intents"]) - sum(i["outcome"] == "head" for i in s["intents"])
    b = max(zip(raw[op], raw["shadow"]), key=lambda x: gain(*x))[0]
    c = load_controller("arms_down=results/policy_mjlab/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=cfg).compile(); res = {}
    for mode in ("shadow", op):
        e = R1HitEnv(seed=0, cfg={**cfg, "opponent_aim": mode}, model=m, episode_s=11, max_impulse=0, train=False); e.opp = FightReplay(m, b["clip"], distance=b["distance"], cfg=cfg); e.opp_t0 = b["t0"]
        hm = HitMeter(m, cfg["hit_metric"]); e.substep_hooks.append(hm); o = e.reset(hits=[], delay=1); Q = []
        for _ in range(500): o, *_ = e.step(c.act(o[None])[0]); Q.append(e.d.qpos.copy())
        a = e.opp_state["aim"]; its = [it for it in a.intents if it.t_apex + 0.15 <= 10]; res[mode] = (Q, its, hm.windows([it.window(0.15) for it in its]))
    heads = [it for it, o in zip(res[op][1], res[op][2]) if o == "head"][:3]
    rn = mujoco.Renderer(m, 360, 480); d = mujoco.MjData(m); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 2.0, 100, -8; F = font(16); rows = []
    for it in heads:
        row = []
        for mode in ("shadow", op):
            Q, its, out = res[mode]; k = min(len(Q) - 1, int(round(it.t_apex / 0.02)) + 1); d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
            cam.lookat[:] = (d.xpos[m.body("pelvis").id] + d.xpos[m.body("opp_pelvis").id]) / 2 + [0, 0, 0.25]; rn.update_scene(d, cam); im = Image.fromarray(rn.render())
            o = next((x for i, x in zip(its, out) if i.limb == it.limb and abs(i.t_apex - it.t_apex) < 0.05), "?")
            ImageDraw.Draw(im).text((10, 8), f"{'unaimed' if mode == 'shadow' else 'aimed: ' + op}   t = {it.t_apex:.2f} s, {it.limb}: {o}", font=F, fill=(255, 255, 255)); row.append(np.asarray(im))
        rows.append(np.concatenate(row, 1))
    Path("media").mkdir(exist_ok=True); Image.fromarray(np.concatenate(rows, 0)).save("media/aim_frames.png")
    json.dump(dict(about=__doc__, bout=dict(clip=b["clip"], t0=b["t0"], distance=b["distance"]), operator=op, frames_at_s=[round(it.t_apex, 3) for it in heads],
                   meta=dict(eo.provenance(m, cfg), **gl_info(rn))), open("results/opponent_aim/aim_frames.json", "w"), indent=1)
    print("wrote media/aim_frames.png", b["clip"], b["t0"], b["distance"])

if __name__ == "__main__": main()
