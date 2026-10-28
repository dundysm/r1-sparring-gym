"""Build the showcase base scene: both robots, the dark studio, lights, overlays (ring, trail, impact FX template), camera.

blender -b -P showcase/blender/build_scene.py -- --scene showcase/rec/scene --out showcase/build/base.blend
    [--hdri showcase/blender/assets/studio_small_09_2k.hdr] [--hdri-strength 0.25]

Run from the repo root (all paths are relative to it, or pass your own). The HDRI is packed into the .blend, so the
.blend is self-contained and portable (Windows preview box -> Linux render pod). Ground z = 0 = the MuJoCo floor (the
mat's top surface). Collections:
  R1, Opponent    one object per visual geom, named by geom name; props mj_body, mj_local (pos+quat in its body frame)
  Set             cyclorama bowl (near-black, infinite curved backdrop) + sparring mat (6x6 m, top at z = 0, fades to black
                  radially and with view depth, so no edge or horizon line shows)
  Lights          LightRig empty (bake.py turns it so the key stays front-top and the rims behind, per camera)
  Overlays        Ring (1 m, soft emissive) + Ring_Spot, Trail (pelvis path ribbon, geometry-nodes time window)
  FX              impact ripples (bake.py copies FX_Ripple from Templates)
  Templates       FX_Ripple (shockwave ring), FX_Core (white-hot flash), BG_Pool (backdrop light pool); hidden, copied by bake.py
  Camera          Camera + Focus (DOF target)
The look follows showcase/STORYBOARD.md (Look).
"""
import bpy, bmesh, sys, os, math, argparse
from mathutils import Vector, Matrix, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mj2blender as mj

# ------------------------------------------------------------------------------------------------ palette (linear RGB)
SHELL = (0.80, 0.79, 0.765)          # satin off-white
GRAPHITE = (0.028, 0.029, 0.032)     # our R1's small mechanical links
OPP_BODY = (0.036, 0.037, 0.041)     # opponent graphite satin
ORANGE = (0.807, 0.141, 0.023)       # #E8692A in linear
MAT_A, MAT_B = (0.014, 0.0145, 0.016), (0.021, 0.022, 0.024)
CYC = (0.003, 0.003, 0.0033)
RING_COL = (0.70, 0.84, 1.00)
TRAIL_COL = (0.25, 0.62, 1.00)
FX_COL = (1.00, 0.30, 0.06)
FX_CORE = (1.00, 0.86, 0.70)         # white-hot, a touch warm
BG_COL = (0.55, 0.62, 0.75)          # cool-neutral backdrop pool

R1_DARK = ("imu_in_pelvis_link", "ankle_constraint", "ankle_pitch_link", "wrist_roll_link", "head_yaw_link", "waist_roll_link")
OPP_ACCENT = ("elbow_link", "knee_link", "waist_roll_link")


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="showcase/rec/scene"); p.add_argument("--out", default="showcase/build/base.blend")
    p.add_argument("--hdri", default="showcase/blender/assets/studio_small_09_2k.hdr"); p.add_argument("--hdri-strength", type=float, default=0.25)
    return p.parse_args(a)


# ------------------------------------------------------------------------------------------------ node helpers
class NT:
    """Tiny node-tree builder: n = NT(mat.node_tree); x = n.math('MULTIPLY', a, 2.0); n.link(x, node.inputs[0])."""
    def __init__(self, tree, clear=True):
        self.t = tree; self.N = tree.nodes; self.L = tree.links
        if clear: self.N.clear()
    def node(self, kind, **kw):
        nd = self.N.new(kind)
        for k, v in kw.items(): setattr(nd, k, v)
        return nd
    def _in(self, sock, v):
        if v is None: return
        if hasattr(v, "is_output") or isinstance(v, bpy.types.NodeSocket): self.L.new(v, sock)
        elif isinstance(v, bpy.types.Node): self.L.new(v.outputs[0], sock)
        else: sock.default_value = v
    def math(self, op, a, b=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp); self._in(nd.inputs[0], a); self._in(nd.inputs[1], b); return nd.outputs[0]
    def maprange(self, v, a, b, c, d, interp="LINEAR"):
        nd = self.node("ShaderNodeMapRange", interpolation_type=interp, clamp=True)
        self._in(nd.inputs["Value"], v)
        for s, x in zip(("From Min", "From Max", "To Min", "To Max"), (a, b, c, d)): self._in(nd.inputs[s], x)
        return nd.outputs[0]
    def link(self, a, b): self._in(b, a)


def principled(name, col, rough, coat=0.0, coat_rough=0.12, sheen=0.0, sss=0.0, metal=0.0, spec=0.5):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; i = b.inputs
    i["Base Color"].default_value = (*col, 1); i["Roughness"].default_value = rough; i["Metallic"].default_value = metal
    i["Coat Weight"].default_value = coat; i["Coat Roughness"].default_value = coat_rough
    i["Sheen Weight"].default_value = sheen; i["Sheen Roughness"].default_value = 0.45
    i["Specular IOR Level"].default_value = spec
    if sss > 0:
        i["Subsurface Weight"].default_value = sss; i["Subsurface Radius"].default_value = (1.0, 0.9, 0.8); i["Subsurface Scale"].default_value = 0.004
    return m


def glow_material(name, col, strength, profile):
    """Additive glow (Transparent + Emission), engine-agnostic. The intensity profile is evaluated from the object's own
    coordinates so it scales with the object: ('band', radius, sigma): soft ring; ('disc', radius, soft): soft dot;
    ('ribbon',): reads mesh attributes v (-1..1 across) and fade (from the Trail geometry nodes);
    ('ripple',): a thin bright ring at r = 1 and a faint echo ring inside it (40 % peak), times Object Info alpha (keyed by
    bake.py); ('flash',): a gaussian dot (sigma 0.35 of the unit radius) times Object Info alpha; ('glow', sigma): a gaussian
    dot of width sigma (unit-radius coordinates), constant: a soft light pool."""
    m = bpy.data.materials.new(name); m.use_nodes = True; n = NT(m.node_tree)
    out = n.node("ShaderNodeOutputMaterial"); add = n.node("ShaderNodeAddShader")
    tr = n.node("ShaderNodeBsdfTransparent"); em = n.node("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1)
    n.link(tr.outputs[0], add.inputs[0]); n.link(em.outputs[0], add.inputs[1]); n.link(add.outputs[0], out.inputs["Surface"])
    kind = profile[0]
    if kind in ("band", "disc", "ripple", "flash", "glow"):
        tc = n.node("ShaderNodeTexCoord"); sep = n.node("ShaderNodeSeparateXYZ"); n.link(tc.outputs["Object"], sep.inputs[0])
        r = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", sep.outputs[0], sep.outputs[0]), n.math("MULTIPLY", sep.outputs[1], sep.outputs[1])))
    if kind == "band":
        _, rad, sig = profile
        z = n.math("DIVIDE", n.math("SUBTRACT", r, rad), sig)
        inten = n.math("EXPONENT", n.math("MULTIPLY", n.math("MULTIPLY", z, z), -1.0))
    elif kind == "disc":
        _, rad, soft = profile
        inten = n.maprange(r, rad - soft, rad + soft, 1.0, 0.0, "SMOOTHSTEP")
    elif kind == "ripple":
        def gring(c, w):
            z = n.math("DIVIDE", n.math("SUBTRACT", r, c), w); return n.math("EXPONENT", n.math("MULTIPLY", n.math("MULTIPLY", z, z), -1.0))
        ring = n.math("ADD", gring(0.94, 0.03), n.math("MULTIPLY", gring(0.70, 0.022), 0.35))
        oi = n.node("ShaderNodeObjectInfo")
        inten = n.math("MULTIPLY", ring, oi.outputs["Alpha"])
    elif kind in ("flash", "glow"):
        sg = 0.35 if kind == "flash" else profile[1]
        z = n.math("DIVIDE", r, sg); inten = n.math("EXPONENT", n.math("MULTIPLY", n.math("MULTIPLY", z, z), -1.0))
        if kind == "flash":
            oi = n.node("ShaderNodeObjectInfo"); inten = n.math("MULTIPLY", inten, oi.outputs["Alpha"])
    elif kind == "ribbon":
        av = n.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="v")
        af = n.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="fade")
        across = n.math("POWER", n.math("SUBTRACT", 1.0, n.math("MULTIPLY", av.outputs["Fac"], av.outputs["Fac"]), clamp=True), 1.5)
        inten = n.math("MULTIPLY", across, n.math("POWER", af.outputs["Fac"], 1.6))
    n.link(n.math("MULTIPLY", inten, strength), em.inputs["Strength"])
    m.surface_render_method = "BLENDED"; m.use_backface_culling = False
    return m


def link(ob, coll):
    for c in ob.users_collection: c.objects.unlink(ob)
    coll.objects.link(ob); return ob


def new_obj(name, me, coll, mat=None):
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    if mat: me.materials.append(mat)
    return ob


def no_shadow(ob):
    ob.visible_shadow = False; ob.visible_diffuse = False; ob.visible_glossy = True


# ------------------------------------------------------------------------------------------------ materials
def materials():
    M = {}
    M["shell"] = principled("R1_Shell", SHELL, 0.36, coat=0.28, coat_rough=0.14, sheen=0.12, sss=0.05)
    M["graphite"] = principled("R1_Graphite", GRAPHITE, 0.40, coat=0.18, coat_rough=0.2, metal=0.15)
    M["opp"] = principled("Opp_Graphite", OPP_BODY, 0.34, coat=0.35, coat_rough=0.16, sheen=0.05)
    M["opp_accent"] = principled("Opp_Orange", ORANGE, 0.38, coat=0.3, coat_rough=0.16)
    M["opp_dark"] = principled("Opp_Black", (0.012, 0.012, 0.014), 0.45, coat=0.1, metal=0.2)
    return M


def mat_fn_factory(M):
    def f(g):
        mesh = g["mesh"]; opp = g["body_name"].startswith("opp_")
        if not opp: return M["graphite"] if any(k in mesh for k in R1_DARK) else M["shell"]
        if any(k in mesh for k in OPP_ACCENT): return M["opp_accent"]
        if any(k in mesh for k in ("ankle_constraint", "imu_in_pelvis", "ankle_pitch")): return M["opp_dark"]
        return M["opp"]
    return f


# ------------------------------------------------------------------------------------------------ set
def cyclorama(coll, floor_z=-0.04, r_floor=7.0, r_cove=3.0, height=12.0, seg=160):
    """Round infinity cove: flat floor -> quarter-circle cove -> vertical wall, all the way around (any camera azimuth)."""
    prof = [(0.0, floor_z)] + [(r, floor_z) for r in (0.5, 1.5, 3.0, 4.5, 6.0, r_floor)]
    for k in range(1, 13):
        a = math.radians(90 * k / 12); prof.append((r_floor + r_cove * math.sin(a), floor_z + r_cove * (1 - math.cos(a))))
    prof.append((r_floor + r_cove, height))
    bm = bmesh.new(); verts = [bm.verts.new((x, 0, z)) for x, z in prof]
    edges = [bm.edges.new((verts[i], verts[i + 1])) for i in range(len(verts) - 1)]
    bmesh.ops.spin(bm, geom=verts + edges, cent=(0, 0, 0), axis=(0, 0, 1), angle=math.radians(360), steps=seg, use_merge=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Cyclorama"); bm.to_mesh(me); bm.free()
    # normals must face the inside (up on the floor)
    if sum(p.normal.z for p in me.polygons if p.center.z < floor_z + 0.01) < 0: me.flip_normals()
    me.shade_smooth()
    m = principled("Cyc_Black", CYC, 1.0, spec=0.0)    # fully matte: no grazing sheen, so no horizon line where the mat ends
    ob = new_obj("Cyclorama", me, coll, m)
    ob.visible_glossy = False   # reflections see the HDRI, the camera sees the black cove
    return ob


def mat_material():
    """Charcoal sparring mat. 2 m panels in offset rows (running bond); the seams are a roughness + bump change only (no
    albedo darkening), so they read up close and vanish at a distance. A low-frequency roughness noise breaks up the
    sheen. Albedo, specular and sheen fade radially from the spot (albedo 1 inside r = 1.2 m, 0 by r = 2.8 m; specular and sheen
    0.9 -> 2.3 m, since the grazing sheen behind the robots is what draws a horizon line):
    the mat melts into the black cyc with no edge and no horizon step behind the robots."""
    m = bpy.data.materials.new("Mat_Rubber"); m.use_nodes = True; n = NT(m.node_tree)
    out = n.node("ShaderNodeOutputMaterial"); b = n.node("ShaderNodeBsdfPrincipled"); n.link(b.outputs[0], out.inputs[0])
    tc = n.node("ShaderNodeTexCoord"); sep = n.node("ShaderNodeSeparateXYZ"); n.link(tc.outputs["Object"], sep.inputs[0])
    P = 2.0                                                   # panel size [m]
    def seam(c):   # 1 on a panel seam (distance to the nearest multiple of P < ~3 mm)
        f = n.math("DIVIDE", c, P)
        d = n.math("MULTIPLY", n.math("SUBTRACT", 0.5, n.math("ABSOLUTE", n.math("SUBTRACT", n.math("FRACT", f), 0.5))), P)
        return n.maprange(d, 0.0012, 0.0035, 1.0, 0.0, "SMOOTHSTEP")
    row = n.math("FLOOR", n.math("DIVIDE", sep.outputs[1], P))
    par = n.math("SUBTRACT", row, n.math("MULTIPLY", n.math("FLOOR", n.math("DIVIDE", row, 2.0)), 2.0))   # 0 / 1 per row
    xs = n.math("ADD", sep.outputs[0], n.math("MULTIPLY", par, P / 2))                                   # offset rows
    s = n.math("MAXIMUM", seam(xs), seam(sep.outputs[1]))
    big = n.node("ShaderNodeTexNoise"); big.inputs["Scale"].default_value = 1.6; big.inputs["Detail"].default_value = 3.0
    n.link(tc.outputs["Object"], big.inputs["Vector"])
    macro = n.node("ShaderNodeTexNoise"); macro.inputs["Scale"].default_value = 0.55; macro.inputs["Detail"].default_value = 4.0
    macro.inputs["Roughness"].default_value = 0.6; n.link(tc.outputs["Object"], macro.inputs["Vector"])
    fine = n.node("ShaderNodeTexNoise"); fine.inputs["Scale"].default_value = 900.0; fine.inputs["Detail"].default_value = 2.0
    n.link(tc.outputs["Object"], fine.inputs["Vector"])
    def sk(socks, name): return [x for x in socks if x.name == name and x.enabled][0]   # Mix has one socket per data type
    mix = n.node("ShaderNodeMix", data_type="RGBA"); sk(mix.inputs, "A").default_value = (*MAT_A, 1); sk(mix.inputs, "B").default_value = (*MAT_B, 1)
    n.link(n.maprange(big.outputs["Fac"], 0.3, 0.7, 0.0, 1.0, "SMOOTHSTEP"), sk(mix.inputs, "Factor"))
    rr = n.math("SQRT", n.math("ADD", n.math("MULTIPLY", sep.outputs[0], sep.outputs[0]), n.math("MULTIPLY", sep.outputs[1], sep.outputs[1])))
    fall = n.maprange(rr, 1.2, 2.8, 1.0, 0.0, "SMOOTHSTEP")
    sfall = n.maprange(rr, 0.9, 2.3, 1.0, 0.0, "SMOOTHSTEP")   # the grazing sheen behind the robots goes first
    # depth fade: at grazing angles a radial fade is squeezed into a few pixel rows and still reads as a line, so the far
    # mat also fades with view depth, from fade_near to fade_far (bake.py sets them per shot from the camera-spot distance;
    # the defaults switch it off)
    cd = n.node("ShaderNodeCameraData")
    vn = n.node("ShaderNodeValue", name="fade_near", label="fade_near"); vn.outputs[0].default_value = 100.0
    vf = n.node("ShaderNodeValue", name="fade_far", label="fade_far"); vf.outputs[0].default_value = 101.0
    dfall = n.maprange(cd.outputs["View Z Depth"], vn.outputs[0], vf.outputs[0], 1.0, 0.0, "SMOOTHSTEP")
    fall = n.math("MULTIPLY", fall, dfall); sfall = n.math("MULTIPLY", sfall, dfall)
    edge = n.node("ShaderNodeMix", data_type="RGBA"); n.link(fall, sk(edge.inputs, "Factor"))   # fall 0 -> black, = the cyc floor
    sk(edge.inputs, "A").default_value = (0.0, 0.0, 0.0, 1); n.link(sk(mix.outputs, "Result"), sk(edge.inputs, "B"))
    n.link(sk(edge.outputs, "Result"), b.inputs["Base Color"])
    n.link(n.math("MULTIPLY", sfall, 0.2), b.inputs["Specular IOR Level"])
    # roughness: base 0.70, +-0.08 macro variation, +0.06 on the seams
    rough = n.math("ADD", n.math("ADD", 0.70, n.math("MULTIPLY", n.math("SUBTRACT", macro.outputs["Fac"], 0.5), 0.16)), n.math("MULTIPLY", s, 0.06))
    n.link(rough, b.inputs["Roughness"])
    n.link(n.math("MULTIPLY", sfall, 0.02), b.inputs["Sheen Weight"]); b.inputs["Sheen Tint"].default_value = (0.6, 0.7, 0.85, 1)
    bump = n.node("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25; bump.inputs["Distance"].default_value = 0.003
    n.link(n.math("SUBTRACT", n.math("MULTIPLY", fine.outputs["Fac"], 0.35), n.math("MULTIPLY", s, 0.4)), bump.inputs["Height"])
    n.link(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def sparring_mat(coll, size=6.0, thick=0.04):
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(size, size, thick), verts=bm.verts); bmesh.ops.translate(bm, vec=(0, 0, -thick / 2), verts=bm.verts)
    me = bpy.data.meshes.new("Mat"); bm.to_mesh(me); bm.free()
    ob = new_obj("Mat", me, coll, mat_material())
    bev = ob.modifiers.new("bevel", "BEVEL"); bev.width = 0.014; bev.segments = 4; bev.limit_method = "ANGLE"; bev.harden_normals = False
    me.shade_smooth()
    return ob


# ------------------------------------------------------------------------------------------------ overlays
def annulus(name, r0, r1, seg=256, rings=4, z=0.0):
    bm = bmesh.new(); grid = []
    for j in range(rings + 1):
        r = r0 + (r1 - r0) * j / rings
        grid.append([bm.verts.new((r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), z)) for i in range(seg)])
    for j in range(rings):
        for i in range(seg):
            bm.faces.new((grid[j][i], grid[j][(i + 1) % seg], grid[j + 1][(i + 1) % seg], grid[j + 1][i]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); return me


def disc(name, r, seg=128, rings=8):
    bm = bmesh.new(); c = bm.verts.new((0, 0, 0)); prev = None
    for j in range(1, rings + 1):
        rr = r * j / rings; ring = [bm.verts.new((rr * math.cos(2 * math.pi * i / seg), rr * math.sin(2 * math.pi * i / seg), 0)) for i in range(seg)]
        for i in range(seg):
            if prev is None: bm.faces.new((c, ring[i], ring[(i + 1) % seg]))
            else: bm.faces.new((prev[i], prev[(i + 1) % seg], ring[(i + 1) % seg], ring[i]))
        prev = ring
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); return me


def trail_nodes():
    """Geometry nodes: keep ribbon points with now - window <= t <= now and store fade = (t - (now - window)) / window."""
    ng = bpy.data.node_groups.new("TrailWindow", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_now = ng.interface.new_socket("Now", in_out="INPUT", socket_type="NodeSocketFloat")
    s_win = ng.interface.new_socket("Window", in_out="INPUT", socket_type="NodeSocketFloat"); s_win.default_value = 3.0
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    at = N.new("GeometryNodeInputNamedAttribute"); at.data_type = "FLOAT"; at.inputs["Name"].default_value = "t"
    a_out = [o for o in at.outputs if o.enabled and o.type == "VALUE"][0]
    def m(op, a, b):
        nd = N.new("ShaderNodeMath"); nd.operation = op
        for s, v in zip(nd.inputs, (a, b)):
            if isinstance(v, bpy.types.NodeSocket): L.new(v, s)
            else: s.default_value = v
        return nd.outputs[0]
    lo = m("SUBTRACT", gi.outputs["Now"], gi.outputs["Window"])
    def cmp(op, a, b):
        nd = N.new("FunctionNodeCompare"); nd.data_type = "FLOAT"; nd.operation = op
        L.new(a, nd.inputs[0]); L.new(b, nd.inputs[1]); return nd.outputs[0]
    orr = N.new("FunctionNodeBooleanMath"); orr.operation = "OR"
    L.new(cmp("GREATER_THAN", a_out, gi.outputs["Now"]), orr.inputs[0]); L.new(cmp("LESS_THAN", a_out, lo), orr.inputs[1])
    dl = N.new("GeometryNodeDeleteGeometry"); dl.domain = "POINT"
    L.new(gi.outputs["Geometry"], dl.inputs["Geometry"]); L.new(orr.outputs[0], dl.inputs["Selection"])
    st = N.new("GeometryNodeStoreNamedAttribute"); st.data_type = "FLOAT"; st.domain = "POINT"; st.inputs["Name"].default_value = "fade"
    fade = m("DIVIDE", m("SUBTRACT", a_out, lo), gi.outputs["Window"])
    val = [s for s in st.inputs if s.name == "Value" and s.enabled][0]
    L.new(dl.outputs["Geometry"], st.inputs["Geometry"]); L.new(fade, val)
    L.new(st.outputs["Geometry"], go.inputs["Geometry"])
    return ng


def overlays(coll, templates):
    ring = new_obj("Ring", annulus("Ring", 0.955, 1.045, rings=6, z=0.002), coll, glow_material("Ring_Glow", RING_COL, 1.15, ("band", 1.0, 0.0075)))
    spot = new_obj("Ring_Spot", annulus("Ring_Spot", 0.035, 0.085, seg=96, rings=4, z=0.002), coll, glow_material("Spot_Glow", RING_COL, 1.5, ("band", 0.06, 0.004)))
    dot = new_obj("Ring_Dot", disc("Ring_Dot", 0.03, seg=48, rings=4), coll, glow_material("Dot_Glow", RING_COL, 1.8, ("disc", 0.014, 0.006)))
    dot.location.z = 0.002
    for o in (spot, dot): o.parent = ring
    for o in (ring, spot, dot): no_shadow(o)
    # trail: bake.py fills the mesh (ribbon along the pelvis path, attributes t and v) and keys the Now input
    tm = bpy.data.meshes.new("Trail"); tr = new_obj("Trail", tm, coll, glow_material("Trail_Glow", TRAIL_COL, 4.5, ("ribbon",)))
    mod = tr.modifiers.new("Trail", "NODES"); mod.node_group = trail_nodes(); no_shadow(tr)
    # impact ripple template (unit radius disc in its XY plane; normal +Z is turned along the hit force)
    fx = new_obj("FX_Ripple", disc("FX_Ripple", 1.0, seg=160, rings=32), templates, glow_material("FX_Ripple_Glow", FX_COL, 3.0, ("ripple",)))
    fx.color = (1, 1, 1, 0); no_shadow(fx); fx.rotation_mode = "QUATERNION"
    # white-hot core flash at the hit point for 2-3 frames (unit radius; bake.py scales it and turns it to the camera)
    core = new_obj("FX_Core", disc("FX_Core", 1.0, seg=64, rings=12), templates, glow_material("FX_Core_Glow", FX_CORE, 40.0, ("flash",)))
    core.color = (1, 1, 1, 0); no_shadow(core); core.rotation_mode = "QUATERNION"
    # soft light pool on the cyc behind the subjects (camera-facing gaussian card, placed by bake.py for shots with
    # "backdrop"): silhouette separation without lighting the robots
    bg = new_obj("BG_Pool", disc("BG_Pool", 1.0, seg=96, rings=16), templates, glow_material("BG_Pool_Glow", BG_COL, 1.0, ("glow", 0.42)))
    no_shadow(bg); bg.visible_glossy = False; bg.rotation_mode = "QUATERNION"
    return ring, tr, fx


# ------------------------------------------------------------------------------------------------ lights / world / camera
def area(name, rig, loc, target, energy, col, shape, size, size_y=None, spread=None):
    l = bpy.data.lights.new(name, "AREA"); l.energy = energy; l.color = col; l.shape = shape; l.size = size
    if size_y: l.size_y = size_y
    if spread: l.spread = math.radians(spread)
    if hasattr(l, "use_shadow_jitter"): l.use_shadow_jitter = True
    o = bpy.data.objects.new(name, l); rig.users_collection[0].objects.link(o); o.parent = rig
    o.location = loc; o.rotation_mode = "QUATERNION"; o.rotation_quaternion = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y")
    return o


def lights(coll, k=1.0):
    """Rig-local frame: +X points from the subjects toward the camera (bake.py turns the rig), +Y is camera-right.
    Energies are balanced for AgX at exposure 0 in Cycles (EEVEE matches within ~1/3 stop); k scales them all."""
    rig = bpy.data.objects.new("LightRig", None); coll.objects.link(rig); rig.empty_display_size = 0.5
    area("Key", rig, (2.7, 1.7, 2.5), (0, 0, 1.0), 130 * k, (1.0, 0.96, 0.92), "DISK", 2.0)
    area("Fill", rig, (3.0, -2.4, 1.3), (0, 0, 0.8), 12 * k, (0.92, 0.95, 1.0), "RECTANGLE", 3.0, 2.0)
    area("Rim_L", rig, (-2.6, -1.9, 2.3), (0, 0, 1.0), 170 * k, (0.62, 0.78, 1.0), "RECTANGLE", 0.45, 2.2)
    area("Rim_R", rig, (-2.4, 2.1, 2.1), (0, 0, 0.95), 150 * k, (0.62, 0.78, 1.0), "RECTANGLE", 0.45, 2.2)
    # a soft pool of light on the mat around the robots, falling off to black towards the mat's edges
    l = bpy.data.lights.new("Pool", "SPOT"); l.energy = 220 * k; l.color = (1.0, 0.97, 0.93); l.spot_size = math.radians(48)
    l.spot_blend = 0.9; l.shadow_soft_size = 0.8
    if hasattr(l, "use_shadow_jitter"): l.use_shadow_jitter = True
    o = bpy.data.objects.new("Pool", l); coll.objects.link(o); o.parent = rig; o.location = (0.4, 0, 5.2)
    # background wash on the cove far behind the subjects: aimed away from the robots, gives the black depth a gradient
    area("BG_Wash", rig, (-5.0, 0.0, 0.6), (-10.0, 0.0, 5.5), 60 * k, (0.55, 0.65, 0.8), "RECTANGLE", 7.0, 2.0)
    return rig


def world(hdri, strength):
    w = bpy.data.worlds.new("World_Studio"); bpy.context.scene.world = w; w.use_nodes = True; n = NT(w.node_tree)
    out = n.node("ShaderNodeOutputWorld"); bg = n.node("ShaderNodeBackground"); bg.inputs["Strength"].default_value = strength
    n.link(bg.outputs[0], out.inputs[0])
    if hdri and os.path.exists(hdri):
        env = n.node("ShaderNodeTexEnvironment"); img = bpy.data.images.load(os.path.abspath(hdri)); img.pack()
        img.filepath = "//" + os.path.basename(hdri)   # packed; no absolute path is kept in the .blend
        env.image = img
        mp = n.node("ShaderNodeMapping"); tc = n.node("ShaderNodeTexCoord"); mp.inputs["Rotation"].default_value = (0, 0, math.radians(30))
        n.link(tc.outputs["Generated"], mp.inputs["Vector"]); n.link(mp.outputs[0], env.inputs["Vector"])
        # dome mask: only the upper sky of the HDRI (its softboxes) is kept, the horizon band is black. Grazing reflections
        # on the mat would otherwise pick up the HDRI's bright walls and turn the charcoal mat grey.
        sep = n.node("ShaderNodeSeparateXYZ"); n.link(tc.outputs["Generated"], sep.inputs[0])
        mask = n.maprange(sep.outputs[2], 0.28, 0.62, 0.0, 1.0, "SMOOTHSTEP")
        mul = n.node("ShaderNodeVectorMath", operation="SCALE"); n.link(env.outputs[0], mul.inputs[0]); n.link(mask, mul.inputs["Scale"])
        n.link(mul.outputs[0], bg.inputs["Color"])
    else:
        print("WARNING: HDRI not found (", hdri, "): using a flat dark world"); bg.inputs["Color"].default_value = (0.02, 0.02, 0.022, 1)
    return w


def camera(coll):
    cd = bpy.data.cameras.new("Camera"); cd.lens = 50; cd.sensor_width = 36; cd.sensor_fit = "AUTO"; cd.clip_start = 0.05; cd.clip_end = 80
    cam = bpy.data.objects.new("Camera", cd); coll.objects.link(cam); cam.rotation_mode = "QUATERNION"
    foc = bpy.data.objects.new("Focus", None); coll.objects.link(foc); foc.empty_display_size = 0.1
    cd.dof.use_dof = True; cd.dof.focus_object = foc; cd.dof.aperture_fstop = 4.0; cd.dof.aperture_blades = 7; cd.dof.aperture_rotation = math.radians(10)
    cam.location = (0.5, -3.3, 1.0); foc.location = (0, 0, 0.9)
    cam.rotation_quaternion = (foc.location - cam.location).to_track_quat("-Z", "Y")
    bpy.context.scene.camera = cam
    return cam


def render_defaults(sc):
    sc.render.fps = 30; sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    e = sc.eevee; e.taa_render_samples = 64; e.use_raytracing = True; e.use_shadows = True
    e.ray_tracing_method = "SCREEN"; e.ray_tracing_options.resolution_scale = "1"; e.use_shadow_jitter_viewport = False
    e.shadow_ray_count = 2; e.shadow_step_count = 8; e.fast_gi_method = "GLOBAL_ILLUMINATION"
    sc.cycles.samples = 256; sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.015
    sc.cycles.use_denoising = True; sc.cycles.max_bounces = 8; sc.cycles.glossy_bounces = 4; sc.cycles.diffuse_bounces = 3
    sc.cycles.transparent_max_bounces = 16; sc.cycles.sample_clamp_indirect = 8.0
    sc.view_settings.view_transform = "AgX"
    try: sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError: pass
    sc.view_settings.exposure = 0.0
    sc.render.film_transparent = False; sc.render.filepath = "//render/"
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGB"; sc.render.image_settings.color_depth = "8"


def main():
    A = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; root = sc.collection
    C = {k: bpy.data.collections.new(k) for k in ("R1", "Opponent", "Set", "Lights", "Overlays", "FX", "Templates", "Camera")}
    for c in C.values(): root.children.link(c)
    C["Templates"].hide_render = True; C["Templates"].hide_viewport = True
    M = materials()
    objs = mj.build(os.path.join(A.scene, "scene_geoms.json"), os.path.join(A.scene, "meshes"), C["R1"], mat_fn_factory(M))
    for ob in objs.values():
        if str(ob["mj_body_name"]).startswith("opp_"): link(ob, C["Opponent"])
    cyclorama(C["Set"]); sparring_mat(C["Set"])
    lights(C["Lights"]); world(A.hdri, A.hdri_strength)
    overlays(C["Overlays"], C["Templates"])
    camera(C["Camera"])
    render_defaults(sc)
    sc["showcase_base"] = True
    out = os.path.abspath(A.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True, compress=True)
    print("saved", A.out, "objects:", len(objs))


if __name__ == "__main__":
    main()
