"""Zone-map identity across hull sets (M2V2_PLAN_v3 A10.3, blocking): r1gym/hits.py r1_zone and fb_hits.zone_of key on mesh names, so a
reduced-hull model must carry the original mesh names. For every hulls option (full / mix / all256) of scripts/fb_replica.py's model and
both entities (the opponent striking, reverse=False; the defender striking, reverse=True): the HitMeter's zone_of / limb_of / keys, the
account's sensors (fb_hits.MjContactSensors: primaries, zone_idx, block_idx, body subset, limb map) and the block-shell set equal the
full-hull model's, geom for geom by body; the head shells are present (2 per robot); and gpu/fullbody_task.py's entity spec gives the
shells the same mesh names and zones under every hull set (shell_tables' zone rule on the spec). The mjlab scene's zone map against the
replica is scripts/fb_m9_check.py (1) 'zones'."""
import sys
from pathlib import Path
import numpy as np, pytest, mujoco
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import fb_replica as RP, fb_hits as HT, fullbody_task as FT
from r1gym.hits import HitMeter

HULLS = ("full", "mix", "all256")

@pytest.fixture(scope="module")
def models():
    ccfg = RP.world("G", RP.conf())[1]; return {h: RP.build_model(ccfg, RP.conf(hulls=h))[0] for h in HULLS}

def gkey(m, g):
    """A geom's identity independent of hulls: (body name, its ordinal among the body's geoms)."""
    b = int(m.geom_bodyid[g]); return m.body(b).name, int(g - m.body_geomadr[b])

def zone_map(m, reverse):
    """Every zone-relevant map of model m, keyed by gkey."""
    hm = HitMeter(m, reverse=reverse); s = HT.MjContactSensors(m, reverse=reverse); K = lambda d: {gkey(m, g): v for g, v in d.items()}
    return dict(zone_of=K(hm.zone_of), limb_of=K(hm.limb_of), keys=K(hm.keys), prim=[gkey(m, g) for g in s.prim], zone_idx=list(s.zone_idx), block=list(s.block_idx),
                block_set=sorted(gkey(m, g) for g, b in zip(s.prim, s.block_idx) if b), bprim=[gkey(m, g) for g in s.bprim], bzone_idx=list(s.bzone_idx),
                limb={gkey(m, g): int(s.limb[g]) for g in range(m.ngeom) if s.limb[g] >= 0})

@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("hulls", ("mix", "all256"))
def test_zone_maps_equal_full_hulls(models, hulls, reverse):
    ref, got = zone_map(models["full"], reverse), zone_map(models[hulls], reverse)
    for k in ref: assert got[k] == ref[k], (hulls, reverse, k)
    assert ref["zone_idx"].count(HT.ZONES.index("head")) == 2 and len(ref["prim"]) == 32 and len(ref["block_set"]) == 4   # shells only; elbows, wrists

def test_hull_meshes_take_the_original_names(models):
    full = models["full"]; names = lambda m: [m.mesh(m.geom_dataid[g]).name for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    for h in ("mix", "all256"):
        m = models[h]; assert names(m) == names(full)
        assert sorted(m.mesh(i).name for i in range(m.nmesh) if m.mesh(i).name.startswith("fborig_")) == sorted("fborig_" + n for n in names(full))

@pytest.mark.parametrize("hulls", HULLS)
def test_task_entity_spec_keeps_names_and_zones(hulls):
    base = lambda: mujoco.MjSpec.from_file(str(ROOT / "r1gym" / "assets" / "r1" / "r1.xml"))
    s = FT.entity_spec(base, hulls); sh = [g for g in s.geoms if g.name.endswith("_shell")]
    assert [g.meshname for g in sh] == list(FT._shell_meshes()) and all(g.name == FT.shell_name(g.meshname) for g in sh)
    zones = [HT.zone_of(g.parent.name, g.meshname) for g in sh]; assert zones.count("head") == 2 and None not in zones
    for e in list(s.excludes): s.delete(e)   # r1.xml alone: its excludes name bodies of the full scene
    m = s.compile(); G = [m.geom(g.name).id for g in sh]
    assert [m.mesh(m.geom_dataid[g]).name for g in G] == list(FT._shell_meshes())
    assert [HT.zone_of(m.body(m.geom_bodyid[g]).name, m.mesh(m.geom_dataid[g]).name) for g in G] == zones

def test_negative_control_the_old_hull_names_lose_the_head():
    """R-EQ's failure (A10.1): under 'fbhull_<mesh>' the head shells fall through r1_zone / zone_of to the torso."""
    from r1gym.hits import r1_zone
    assert [r1_zone("torso_link", m) for m in ("head_pitch_link", "fbhull_head_pitch_link")] == ["head", "torso"]
    assert [HT.zone_of("torso_link", m) for m in ("head_yaw_link", "fbhull_head_yaw_link")] == ["head", "torso"]
