# Showcase video: storyboard

One held-out bout, rendered from the exact simulated motion. Hard tier, test split, episode 16 of
`results/eval_opponent/raw_test_hard.json`, picked by rule (`showcase/pick_bouts.py` → `showcase/bouts.json`: the shipped
guard survives and stays in the ring, PD falls; ranked by the hardest strike to the head). `showcase/record.py` replays it
and checks the replay against the stored eval row, so what you see is the bout the eval scored.

Bout 16: clip kfab_202, opponent 0.45 m away, randomised physics, six scripted hits:

| t (s) | hit |
|---|---|
| 1.68 | push, right knee, 9.2 N·s |
| 2.98 | strike, left knee, 8.6 N·s |
| 8.92 | strike, chest, 9.5 N·s |
| 13.10 | push, head height, 12.3 N·s → the R1 is shoved 0.69 m off its spot (t = 14.7 s) |
| 15.42 | strike, pelvis, 9.1 N·s |
| 17.32 | strike, head height, 13.9 N·s |

PD (no stepping) falls at 5.5 s. The footwork guard survives all 20 s and ends 0.15 m from its spot.

## Cut (1920×1080, 30 fps, 34 s)

| # | screen | shot | content | on screen |
|---|---|---|---|---|
| 1 | 0–4.6 s | `hook` | sim 16.55–18.55 s; 5× slow motion through the 13.9 N·s head strike. Low hero angle, 50 mm, shallow depth of field, slow push-in. Impact ripple at the hit point. | "13.9 N·s to the head." → "It steps. It stays up." |
| 2 | 4.6–7 s | title over `split` frame 0 | | **R1 Sparring Gym** / "Teaching a Unitree R1 to take a hit, in simulation" |
| 3 | 7–14.5 s | `split_pd` + `split_guard` | the same bout from t = 0, side by side, same fixed camera, real time | left "Rigid stance (PD)", FELL 5.5 s marker; right "Footwork policy" ; each hit labelled when it lands ("strike · chest · 9.5 N·s") |
| 4 | 14.5–22 s | `hold` | sim 12.4–20 s, real time, slow orbit; the 1 m ring and its start spot on the mat, a fading trail of where the pelvis went | "Shoved 0.69 m off its spot…" (t≈14.7) → "…it walks back." (t≈18) |
| 5 | 22–30 s | cards over the last `hold` frame, blurred | three stats, each with its source file | survival on held-out hard bouts; up and within 1 m (vs PR #13); single-hit fall rate |
| 6 | 30–34 s | end card | | repo URL · "Simulation (MuJoCo). Scripted impulses stand in for strikes. The opponent is a second R1 on its own motors." |

README loop (8 s): the split (shot 3), with its labels.

## Rules
- Every number on screen is read from a results file at compose time (`showcase/compose.py`); nothing typed in by hand.
- Scripted pushes/strikes are impulses applied by the eval, not punches the opponent throws: the end card says so.
- The opponent's own contacts are real (MuJoCo), shown by the motion only.

## Look
Dark studio (near-black cyclorama, AgX), a charcoal sparring mat, a soft key from front-top, cool rim lights for
silhouettes, the Poly Haven `studio_small_09` HDRI (CC0) low for reflections. Our R1: satin off-white shell. Opponent:
graphite with burnt-orange accents. Cycles with OptiX for the final render (RTX 4090), EEVEE for local previews.
