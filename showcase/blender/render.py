"""Render a baked shot to PNG frames.

blender -b showcase/build/NAME.blend -P showcase/blender/render.py -- --engine EEVEE|CYCLES --out showcase/frames/NAME
    [--frames a:b[:step] | --frames 1,40,80]   frame range or list (1-based, as baked); default: all frames
    [--times 0,3,5.6]                         or: the frames nearest these sim times (scene["frame_times"])
    [--scale 0.5]                             resolution scale (0.5 of 1920x1080 = 960x540)
    [--samples N]                             Cycles max samples (adaptive) / EEVEE TAA samples
    [--device OPTIX|CUDA|HIP|ONEAPI|METAL|CPU] Cycles device (default OPTIX, falls back down that list, then CPU)
    [--no-mblur]

Writes OUT/f_0001.png ... (frame number in the name, 8-bit RGB PNG) and prints the render time of every frame.
Cycles: GPU with fallback, adaptive sampling, denoiser OptiX on an OptiX device else OpenImageDenoise, motion blur
(shutter keyed per frame by bake.py: 0.5 real time, 0.25 in slow motion), persistent data. EEVEE Next: ray tracing, jittered soft shadows, motion blur.
"""
import bpy, sys, os, time, argparse

ORDER = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"]); p.add_argument("--out", required=True)
    p.add_argument("--frames"); p.add_argument("--times"); p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--samples", type=int); p.add_argument("--device", default="OPTIX"); p.add_argument("--no-mblur", action="store_true")
    return p.parse_args(a)


def cycles_device(want):
    """Enable every device of the first available backend from `want` down ORDER; returns (backend, devices) or ('CPU', [])."""
    sc = bpy.context.scene
    if want.upper() == "CPU": sc.cycles.device = "CPU"; return "CPU", []
    prefs = bpy.context.preferences.addons["cycles"].preferences
    cands = [want.upper()] + [b for b in ORDER if b != want.upper()]
    for be in cands:
        try: prefs.compute_device_type = be
        except TypeError: continue
        prefs.refresh_devices() if hasattr(prefs, "refresh_devices") else prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == be]
        if not devs: continue
        for d in prefs.devices: d.use = d.type == be
        sc.cycles.device = "GPU"; return be, [d.name for d in devs]
    prefs.compute_device_type = "NONE"; sc.cycles.device = "CPU"; return "CPU", []


def setup_cycles(A):
    sc = bpy.context.scene; sc.render.engine = "CYCLES"; c = sc.cycles
    be, devs = cycles_device(A.device)
    c.samples = A.samples or 256; c.use_adaptive_sampling = True; c.adaptive_threshold = 0.015; c.adaptive_min_samples = 0
    c.use_denoising = True
    c.denoiser = "OPTIX" if be == "OPTIX" else "OPENIMAGEDENOISE"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    if hasattr(c, "denoising_use_gpu"): c.denoising_use_gpu = be != "CPU"
    c.max_bounces = 8; c.diffuse_bounces = 3; c.glossy_bounces = 4; c.transmission_bounces = 4; c.transparent_max_bounces = 16
    c.sample_clamp_indirect = 8.0; c.use_light_tree = True
    sc.render.use_persistent_data = True
    print(f"cycles: device {be} {devs}, samples {c.samples} adaptive, denoiser {c.denoiser}")


def setup_eevee(A):
    sc = bpy.context.scene; sc.render.engine = "BLENDER_EEVEE_NEXT"; e = sc.eevee
    e.taa_render_samples = A.samples or 64
    e.use_raytracing = True; e.ray_tracing_method = "SCREEN"; e.ray_tracing_options.resolution_scale = "1"
    e.ray_tracing_options.use_denoise = True
    e.use_shadows = True; e.shadow_ray_count = 2; e.shadow_step_count = 8; e.use_shadow_jitter_viewport = False
    e.fast_gi_method = "GLOBAL_ILLUMINATION"
    for l in bpy.data.lights:
        if hasattr(l, "use_shadow_jitter"): l.use_shadow_jitter = True
    if hasattr(e, "motion_blur_steps"): e.motion_blur_steps = 1
    print(f"eevee next: {e.taa_render_samples} samples, ray tracing on")


def pick_frames(A, sc):
    if A.times:
        tt = sc.get("frame_times"); assert tt is not None, "no frame_times in this .blend (bake it with bake.py)"
        tt = list(tt); return sorted({min(range(len(tt)), key=lambda k: abs(tt[k] - float(x))) + 1 for x in A.times.split(",")})
    if not A.frames: return list(range(sc.frame_start, sc.frame_end + 1))
    if "," in A.frames or ":" not in A.frames: return [int(x) for x in A.frames.split(",")]
    p = [int(x) for x in A.frames.split(":")]; return list(range(p[0], p[1] + 1, p[2] if len(p) > 2 else 1))


def main():
    A = args(); sc = bpy.context.scene
    (setup_cycles if A.engine == "CYCLES" else setup_eevee)(A)
    r = sc.render
    r.use_motion_blur = not A.no_mblur
    if not sc.get("shutter_keyed"): r.motion_blur_shutter = 0.5   # bake.py keys it per frame (0.5 real time, 0.25 slow motion)
    r.resolution_percentage = max(1, int(round(A.scale * 100)))
    r.image_settings.file_format = "PNG"; r.image_settings.color_mode = "RGB"; r.image_settings.color_depth = "8"
    r.use_file_extension = True
    out = os.path.abspath(A.out); os.makedirs(out, exist_ok=True)
    frames = pick_frames(A, sc); tt = list(sc.get("frame_times", []))
    print(f"rendering {len(frames)} frames of {sc.get('shot_name', '?')} at {r.resolution_x * r.resolution_percentage // 100}x"
          f"{r.resolution_y * r.resolution_percentage // 100} with {A.engine} -> {A.out}")
    t_all = time.time()
    for f in frames:
        sc.frame_set(f); r.filepath = os.path.join(out, f"f_{f:04d}.png")
        t0 = time.time(); bpy.ops.render.render(write_still=True); dt = time.time() - t0
        st = f" sim {tt[f - 1]:.3f} s" if tt and f - 1 < len(tt) else ""
        print(f"FRAME {f}{st}: {dt:.1f} s", flush=True)
    print(f"DONE {len(frames)} frames in {time.time() - t_all:.1f} s")


if __name__ == "__main__":
    main()
