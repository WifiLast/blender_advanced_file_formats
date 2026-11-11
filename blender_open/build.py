"""bpy helpers shared by the importers."""
import bpy
import numpy as np


def make_mesh(name, verts, tris):
    """numpy (N,3) float / (M,3) int -> bpy Mesh, built with foreach_set (fast)."""
    me = bpy.data.meshes.new(name)
    v = np.ascontiguousarray(verts, dtype=np.float32)
    t = np.ascontiguousarray(tris, dtype=np.int32)
    nt = len(t)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.reshape(-1))
    me.loops.add(nt * 3)
    me.loops.foreach_set("vertex_index", t.reshape(-1))
    me.polygons.add(nt)
    me.polygons.foreach_set("loop_start", np.arange(0, nt * 3, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32)) if hasattr(me.polygons[0] if nt else None, "loop_total") else None
    me.update(calc_edges=True)
    me.validate()
    return me


def srgb_to_linear(c):
    """Colours in NWD / Revit are 8-bit sRGB values; Blender colour sockets expect linear values."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_edge_mesh(name, verts, edges):
    """numpy (N,3) / (E,2) -> bpy Mesh with edges only (line geometry)."""
    me = bpy.data.meshes.new(name)
    v = np.ascontiguousarray(verts, dtype=np.float32)
    e = np.ascontiguousarray(edges, dtype=np.int32)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.reshape(-1))
    me.edges.add(len(e))
    me.edges.foreach_set("vertices", e.reshape(-1))
    me.update()
    return me


def get_material(cache, key, name, rgb, alpha=1.0):
    """rgb: sRGB 0..1 as stored in the file (converted to linear for Blender)."""
    mat = cache.get(key)
    if mat is not None:
        return mat
    rgb = tuple(srgb_to_linear(float(c)) for c in rgb)
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, alpha)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Roughness"].default_value = 0.8
    if alpha < 1.0 and hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    cache[key] = mat
    return mat


def new_collection(name, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col
