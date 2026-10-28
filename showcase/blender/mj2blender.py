"""MuJoCo -> Blender importer (Blender 4.2), adapted for the r1-sparring-gym showcase.

Origin: the reusable importer from the grokbot `mujoco2blender` toolkit (blender_assets/_src/mj2blender.py), written for
the OccluBench and kitchen-G1 videos. Same schema and API; changes for this repo:
  - material slots are linked to the OBJECT, not the mesh, because both robots share one set of mesh datablocks
    (record.py exports one .obj per mesh, the opponent's geoms reuse the R1's meshes);
  - sharp edges are marked by angle and the mesh is shaded smooth (Blender 4.1+ has no auto-smooth), so the
    WEIGHTED_NORMAL modifier keeps hard edges crisp;
  - numpy helpers (`geom_world`, `quat_mul`, `quat_rot`, `slerp`) so bake.py can compute every geom's world pose for
    a whole shot at once;
  - no absolute paths: callers pass paths (relative to the repo root or as arguments).

Input: scene_geoms.json + meshes/*.obj (showcase/record.py --scene), and a recording npz with xpos/xquat (T, nbody, 3/4).
API:
  objs = build(geoms_json, mesh_dir, coll, mat_fn=None, skip=())   -> {geom_name: obj}; obj['mj_body'], obj['mj_local']
  pose(objs, xpos[t], xquat[t], offset=(0,0,0))                     -> sets matrix_world for every geom (one frame)
  cam_from_mj(cam_obj, pos, xmat9, fovy_deg, offset)                -> MuJoCo camera (-Z look, +Y up) = Blender convention
  geom_world(bpos, bquat, lpos, lquat)                              -> numpy world pos/quat of geoms from body poses
"""
import json, os, math
import numpy as np

try:  # importable outside Blender for the numpy helpers
    import bpy, bmesh
    from mathutils import Matrix, Quaternion, Vector
except ImportError:  # pragma: no cover
    bpy = None


# ---------------------------------------------------------------- numpy quaternion helpers (wxyz, MuJoCo convention)
def quat_mul(a, b):
    """Hamilton product, broadcasting over leading dims. a, b: (..., 4) wxyz."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_rot(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    w = q[..., :1]; u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def slerp(q0, q1, a):
    """Shortest-path slerp between unit quaternions q0, q1 (..., 4) with weights a (...)."""
    a = np.asarray(a, float)[..., None]
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1); d = np.abs(d)
    th = np.arccos(np.clip(d, -1.0, 1.0)); s = np.sin(th)
    small = s < 1e-6
    w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * th) / np.where(small, 1.0, s))
    w1 = np.where(small, a, np.sin(a * th) / np.where(small, 1.0, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def geom_world(bpos, bquat, lpos, lquat):
    """World pose of geoms given their bodies' poses. bpos (..., 3), bquat (..., 4), lpos (3,), lquat (4,)."""
    return bpos + quat_rot(bquat, np.broadcast_to(lpos, bpos.shape)), quat_mul(bquat, np.broadcast_to(lquat, bquat.shape))


# ---------------------------------------------------------------- Blender side
def _mat(rgba, name):
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = tuple(rgba[:3]) + (1,)
    b.inputs["Roughness"].default_value = 0.45
    return m


def _prim(t, size, name):
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    if t == "box":
        bmesh.ops.create_cube(bm, size=2.0); bmesh.ops.scale(bm, vec=Vector(size[:3]), verts=bm.verts)
    elif t == "plane":
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size[0] or 20)
    elif t in ("cylinder", "capsule"):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=size[0], radius2=size[0], depth=2 * size[1])
        if t == "capsule":
            for s in (1, -1):
                r = bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=size[0])
                bmesh.ops.translate(bm, vec=Vector((0, 0, s * size[1])), verts=r["verts"])
    elif t in ("sphere", "ellipsoid"):
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
        sc = size[:3] if t == "ellipsoid" else [size[0]] * 3
        bmesh.ops.scale(bm, vec=Vector(sc), verts=bm.verts)
    bm.to_mesh(me); bm.free(); return me


def _obj_mesh(path, name, sharp_deg=32.0):
    me = bpy.data.meshes.get(name)
    if me: return me
    V, F = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "): V.append(tuple(map(float, ln.split()[1:4])))
            elif ln.startswith("f "): F.append(tuple(int(x.split("/")[0]) - 1 for x in ln.split()[1:]))
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.validate()
    me.shade_smooth()
    if hasattr(me, "set_sharp_from_angle"): me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    return me


def build(geoms_json, mesh_dir, coll, mat_fn=None, skip=()):
    with open(geoms_json) as fh: G = json.load(fh)["geoms"]
    objs = {}
    for g in G:
        if g["name"] in skip: continue
        me = _obj_mesh(os.path.join(mesh_dir, g["mesh"] + ".obj"), g["mesh"]) if g["type"] == "mesh" else _prim(g["type"], g["size"], g["name"])
        ob = bpy.data.objects.new(g["name"], me); coll.objects.link(ob)
        if g["type"] == "mesh":
            mod = ob.modifiers.new("ws", "WEIGHTED_NORMAL"); mod.keep_sharp = True
        mat = (mat_fn(g) if mat_fn else None) or _mat(g["rgba"], "mj_" + (g["material"] or g["name"]))
        if len(me.materials) == 0: me.materials.append(None)   # one slot on the shared mesh ...
        ob.material_slots[0].link = "OBJECT"; ob.material_slots[0].material = mat   # ... filled per object
        ob["mj_body"] = g["body"]; ob["mj_local"] = [*g["pos"], *g["quat"]]
        ob["mj_body_name"] = g.get("body_name", ""); ob["mj_mesh"] = g.get("mesh") or ""
        ob.rotation_mode = "QUATERNION"
        objs[g["name"]] = ob
    return objs


def body_matrix(p, q, offset=(0, 0, 0)):
    return Matrix.Translation(Vector(p) + Vector(offset)) @ Quaternion(q).to_matrix().to_4x4()


def pose(objs, xpos, xquat, offset=(0, 0, 0)):
    for ob in objs.values():
        b = ob["mj_body"]; l = list(ob["mj_local"])
        ob.matrix_world = body_matrix(xpos[b], xquat[b], offset) @ Matrix.Translation(Vector(l[:3])) @ Quaternion(l[3:]).to_matrix().to_4x4()


def cam_from_mj(cam, pos, xmat9, fovy, offset=(0, 0, 0)):
    R = Matrix([xmat9[0:3], xmat9[3:6], xmat9[6:9]])
    cam.matrix_world = Matrix.Translation(Vector(pos) + Vector(offset)) @ R.to_4x4()
    cam.data.sensor_fit = "VERTICAL"; cam.data.angle_y = math.radians(fovy)
