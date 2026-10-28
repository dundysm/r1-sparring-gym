# Showcase video pipeline

A 34 s showcase cut of one held-out bout (hard tier, test split, episode 16), rendered from the exact simulated
motion. The creative brief is [`STORYBOARD.md`](STORYBOARD.md); the shot spec is [`shots.json`](shots.json).
It is a simulation (MuJoCo): the end card says so, and the scripted strikes are impulses applied by the eval.

```
record  ->  build_scene  ->  bake (per shot)  ->  render (per shot)  ->  compose  ->  encode
.npz        base.blend      NAME.blend +         frames/NAME/           frames/      media/showcase.mp4,
                            NAME_frames.json     f_0001.png ...         final/       loop .webp/.gif, poster
```

All commands run from the repo root, and every path is relative to it, so the same commands work on the Windows
preview box and the Linux render pod. `blender` is Blender 4.2 LTS (4.2.23 tested), headless (`-b`). Blender and Python
are not pinned to any machine path.

## 1. Pick and record the bout (MuJoCo, repo Python env)

```
python showcase/pick_bouts.py                       # -> showcase/bouts.json (the rule: guard survives + stays in, PD falls; hardest head strike)
python showcase/record.py --raw results/eval_opponent/raw_test_hard.json --episode 16 \
    --policy guard_fw=results/policy_mjlab_guard_footwork/policy.onnx --out showcase/rec/ep16_guard_fw.npz
python showcase/record.py --raw results/eval_opponent/raw_test_hard.json --episode 16 \
    --policy PD --after 2 --out showcase/rec/ep16_pd.npz
python showcase/record.py --scene showcase/rec/scene  # visual meshes + geom table, once
```

`record.py` replays the stored eval episode and refuses to write a recording that does not reproduce the stored outcome
(`meta_json.matches_stored`). The .npz schema is in its docstring: body poses per 5 ms substep, the start spot, contacts,
the scripted hits.

## 2. Build the base scene

```
blender -b -P showcase/blender/build_scene.py -- --scene showcase/rec/scene --out showcase/build/base.blend
    [--hdri showcase/blender/assets/studio_small_09_2k.hdr] [--hdri-strength 0.25]
```

- Both robots from `scene_geoms.json` + `meshes/*.obj` (64 visual meshes) with
  [`blender/mj2blender.py`](blender/mj2blender.py) (adapted from the grokbot `mujoco2blender` importer; header credits
  it). Collections `R1` and `Opponent`, objects named by geom, custom props `mj_body`, `mj_local`.
- Look: satin off-white R1 shell (light clearcoat, a touch of sheen and subsurface), small mechanical links in graphite;
  the opponent in graphite satin with burnt-orange (#E8692A) elbows, knees and waist. Near-black infinity cove (round,
  no horizon), a 6 x 6 m charcoal rubber mat (1 m tiles, procedural grain, bevelled edge, falls off towards its edges),
  front-top soft key, two cool rims behind, a low fill, an overhead pool, a faint cove wash, the Poly Haven
  `studio_small_09` HDRI (CC0, packed into the .blend) at 0.25 for reflections, AgX Medium High Contrast. z = 0 is the
  MuJoCo floor (the mat's top).
- `Ring` (1 m radius, ~2 cm soft emissive band, plus the start-spot marker), `Trail` (pelvis path ribbon; a
  geometry-nodes window keeps the last 3 sim seconds and fades its tail), `FX_Ripple` template (thin emissive orange
  ring with an echo ring, alpha from Object Info). Overlays are Transparent + Emission, so they work in both engines.
- The light rig is one empty (`LightRig`); bake.py turns it per shot so the key stays front-top and the rims behind.

## 3. Bake a shot

```
blender -b showcase/build/base.blend -P showcase/blender/bake.py -- --shots showcase/shots.json --shot hook        --out showcase/build/hook.blend
blender -b showcase/build/base.blend -P showcase/blender/bake.py -- --shots showcase/shots.json --shot split_pd    --out showcase/build/split_pd.blend
blender -b showcase/build/base.blend -P showcase/blender/bake.py -- --shots showcase/shots.json --shot split_guard --out showcase/build/split_guard.blend
blender -b showcase/build/base.blend -P showcase/blender/bake.py -- --shots showcase/shots.json --shot hold        --out showcase/build/hold.blend
```

- Screen frames at `shots.fps` from the shot's speed map (`--ramp 0.1` eases each speed change over 0.2 sim s inside
  the faster segment, so slow motion is fully slow at the impact). Each frame's sim time interpolates the two bracketing
  substeps (linear position, slerp rotation); every geom object's world transform is keyed per frame, LINEAR, so Cycles
  motion blur sees the real motion.
- Writes `showcase/build/NAME_frames.json` (sim seconds per frame, frame 1 first) and the same list as
  `scene["frame_times"]`; the compositor keys captions to sim time through it.
- Ring at the recorded spot (`ring`), trail (`trail`), impact ripples at each scripted hit inside the shot (`fx`: at the
  hit point = body pose + hit offset, facing the force, 0.05 -> 0.35 m, fading over 0.3 sim s).
- Cameras (all keyed, DOF on a `Focus` empty keyed to the smoothed R1 chest, f/4 unless `fstop` is given):
  - `hero`: low 3/4 from the opponent's side of the bout axis at the hit, `side_deg` counter-clockwise from above,
    `dist_m` eased in by `push_in`, smoothed torso tracking, aimed `look_mix` (0.3) of the way to the opponent. A medium
    shot by design (50 mm at 2.3 m frames ~0.9 m of height): legs leave the frame, the head and arms are kept in.
  - `fixed_side`: static, side-on to the R1 -> opponent axis at t = 0, centred between the start positions. Every shot
    sharing the camera (`same_as`) is checked: it slides along the axis, then backs off from `dist_m` until the whole
    group (the PD fall included) fits. split_pd and split_guard get the identical camera (both computed from the
    split_guard start).
  - `orbit`: around the start spot, `az0_deg` -> `az1_deg` eased, measured from the R1's mean heading over the shot
    (positive = to the R1's right), so the orbit shows its front; look-at between the spot and the R1, nudged so the
    whole R1 (and the opponent, if it can) stays in frame.
  - Lenses are full-frame equivalents on a 16:9 frame; the 960 x 1080 split halves are crops of that frame.
- Prints a contact check (lowest point of each robot at frame 1; should be within a few mm of z = 0).

## 4. Render

```
blender -b showcase/build/NAME.blend -P showcase/blender/render.py -- --engine CYCLES --out showcase/frames/NAME
    [--frames a:b[:step] | --frames 1,40,80 | --times 17.3,17.36]  [--scale 0.5]  [--samples N]
    [--device OPTIX|CUDA|HIP|ONEAPI|METAL|CPU]  [--no-mblur]
```

- Cycles (final): `--device OPTIX` (default) falls back through CUDA, HIP, oneAPI, Metal, then CPU; adaptive sampling
  (256 max, threshold 0.015), denoiser OptiX on an OptiX device else OpenImageDenoise, motion blur shutter 0.5,
  persistent data, light tree.
- EEVEE Next (previews): `--engine EEVEE`, screen-space ray tracing, jittered soft shadows, motion blur, 64 samples.
- Output `showcase/frames/NAME/f_0001.png ...` (frame number in the name, 8-bit RGB), render time per frame printed.

Final render on the RTX 4090 pod (`showcase/pod_setup.sh` installs Blender 4.2.23 for Linux):

```
for s in hook split_pd split_guard hold; do
  blender -b showcase/build/$s.blend -P showcase/blender/render.py -- --engine CYCLES --device OPTIX --out showcase/frames/$s
done
```

Local preview (what produced `showcase/preview/blender_*.png`):

```
blender -b showcase/build/hook.blend -P showcase/blender/render.py -- --engine EEVEE --out showcase/frames/_preview/hook \
    --times 16.9,17.3,17.36,17.5,18.2 --scale 0.5 --samples 32
blender -b showcase/build/hook.blend -P showcase/blender/render.py -- --engine CYCLES --device CPU \
    --out showcase/frames/_preview/hook_cycles --frames 49 --scale 0.5 --samples 32
```

On the Intel Arc preview box pass `--device CPU` for Cycles: the default falls back to oneAPI there, whose first run
compiles kernels for ~8 minutes.

Measured on the preview box (Intel laptop with an Arc 140V iGPU, shared with other jobs):

| | resolution | samples | per frame |
|---|---|---|---|
| EEVEE Next | 960 x 540 (and 480 x 540 split halves) | 32 | 3.4 - 5.0 s (first frame 9 - 12 s: shader compile) |
| EEVEE Next | 1920 x 1080 | 64 | 14 - 20 s |
| Cycles CPU | 960 x 540 | 32 + OIDN | 18.4 s |
| Cycles oneAPI (Arc) | 960 x 540 | 32 + OIDN | 477 s, almost all first-run kernel compile |

## 5. Compose and encode

```
python showcase/compose.py            # frames/<shot>/ + build/<shot>_frames.json -> showcase/frames/final/f_00001.png ...
python showcase/compose.py --loop     # README loop -> showcase/frames/loop/
python showcase/encode.py             # -> media/showcase.mp4, showcase_loop.webp/.gif, showcase_poster.png
```

`compose.py --facts` prints every on-screen number and the results file it came from. Type is Inter (variable, OFL) from
`showcase/fonts/`, the same on Windows and Linux.

## Files

| path | what |
|---|---|
| `showcase/blender/mj2blender.py` | MuJoCo -> Blender importer + numpy pose helpers |
| `showcase/blender/build_scene.py` | base scene: robots, look, set, lights, overlays, camera |
| `showcase/blender/bake.py` | one shot: timing, poses, camera, ring/trail/ripples, frame table |
| `showcase/blender/render.py` | Cycles / EEVEE frame renderer |
| `showcase/blender/assets/studio_small_09_2k.hdr` | Poly Haven HDRI (CC0), packed into base.blend at build time |
| `showcase/build/` | base.blend, NAME.blend, NAME_frames.json (generated) |
| `showcase/frames/` | rendered and composed frames (generated) |
| `showcase/preview/blender_*.png` | EEVEE preview stills at 960 x 540 (split: the PD and guard halves side by side) and one Cycles CPU still |
