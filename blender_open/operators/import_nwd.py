import time
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector

from .. import build
from ..nwd.reader import read_nwd
from .. import log as cadlog
from ._threaded import ThreadedImportMixin

L = cadlog.get("build")


class IMPORT_SCENE_OT_nwd(Operator, ImportHelper, ThreadedImportMixin):
    bl_idname = "import_scene.nwd"
    bl_label = "Navisworks (.nwd)"
    bl_description = "Import triangle geometry from a Navisworks .nwd file"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".nwd"
    filter_glob: StringProperty(default="*.nwd", options={"HIDDEN"})
    units: EnumProperty(
        name="File Units",
        description="Length unit of the NWD coordinates (Navisworks stores the unit of the source model)",
        items=[
            ("AUTO", "Auto", "Millimetres if the model extends over more than 3000 units, otherwise metres"),
            ("M", "Metres", ""),
            ("CM", "Centimetres", ""),
            ("MM", "Millimetres", ""),
            ("FT", "Feet", ""),
        ],
        default="AUTO",
    )
    global_scale: FloatProperty(
        name="Extra Scale",
        description="Additional multiplier on top of the unit conversion",
        default=1.0, min=1e-6, soft_max=1000.0,
    )
    share_meshes: BoolProperty(
        name="Share Mesh Data",
        description="Instances of the same geometry share one mesh",
        default=True,
    )

    import_lines: BoolProperty(
        name="Lines",
        description="Also import line geometry (pipe / duct axes, grille louvres, ...) as edge-only objects",
        default=True,
    )
    max_objects: IntProperty(
        name="Max Objects",
        description="Import only the first N objects (0 = all). Large plant models hold over a million fragments and need "
                    "many GB of RAM and a long build time in Blender",
        default=0, min=0,
    )

    debug_log: BoolProperty(
        name="Debug Log",
        description="Verbose logging (system console) and a log file in the temp folder",
        default=False,
    )

    def invoke(self, context, event):
        if self.filepath and Path(self.filepath).is_file():   # file drop
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not Path(self.filepath).is_file():
            self.report({"ERROR"}, f"Input file not found: {self.filepath}")
            return {"CANCELLED"}
        self._path = self.filepath
        self._lines = self.import_lines
        return self._start(context)

    def _read(self):
        return read_nwd(self._path, self._say, self._lines)

    def _build(self, context, scene):
        if not scene.instances:
            skipped = ", ".join(f"{v}x {k}" for k, v in scene.skipped.items()) or "no geometry"
            self.report({"WARNING"}, f"No importable geometry found ({skipped})")
            return
        unit = self.units
        extent = scene.info.get("extent", 0.0)
        if unit == "AUTO":
            unit = "MM" if extent > 3000.0 else "M"
        s = {"M": 1.0, "CM": 0.01, "MM": 0.001, "FT": 0.3048}[unit] * self.global_scale
        L.info("units: %s (extent %.1f) -> scale %g", unit, extent, s)
        col = build.new_collection(Path(self._path).stem)
        meshes, mats = {}, {}
        n_matrix = n_shared = n_override = 0
        insts = scene.instances[:self.max_objects] if self.max_objects else scene.instances
        L.info("creating %d objects from %d meshes", len(insts), len(scene.meshes))
        t0 = time.perf_counter()
        for n_obj, inst in enumerate(insts):
            if n_obj and n_obj % 20000 == 0:
                L.info("  %d / %d objects (%.1f s)", n_obj, len(insts), time.perf_counter() - t0)
            v, t = scene.meshes[inst.mesh]
            me = meshes.get(inst.mesh) if self.share_meshes else None
            if me is not None:
                n_shared += 1
            if me is None:
                me = build.make_edge_mesh(inst.name, v, t) if t.shape[1] == 2 else build.make_mesh(inst.name, v, t)
                if self.share_meshes:
                    meshes[inst.mesh] = me
            ob = bpy.data.objects.new(inst.name, me)
            if inst.matrix is not None:
                n_matrix += 1
                lin = Matrix(inst.matrix).transposed()          # rows are the images of the basis vectors
                ob.matrix_basis = Matrix.Translation(Vector(inst.pos) * s) @ (lin * s).to_4x4()
            else:
                x, y, z, w = inst.quat
                ob.rotation_mode = "QUATERNION"
                ob.rotation_quaternion = Quaternion((w, x, y, z))
                ob.location = tuple(c * s for c in inst.pos)
                ob.scale = (inst.scale * s,) * 3
            r, g, b, a = (round(c, 3) for c in inst.rgba)
            mat = build.get_material(mats, (r, g, b, a), f"NWD_{r}_{g}_{b}", (r, g, b), a)
            if not me.materials:
                me.materials.append(mat)
            elif me.materials[0] is not mat:                  # shared mesh, different colour: override per object
                n_override += 1
                ob.data = me
                ob.active_material_index = 0
                ob.material_slots[0].link = "OBJECT"
                ob.material_slots[0].material = mat
            col.objects.link(ob)
        L.info("created %d objects, %d mesh datablocks, %d materials in %.1f s", len(insts), len(meshes), len(mats), time.perf_counter() - t0)
        L.info("build stats: %d matrix transforms, %d objects reusing a shared mesh, %d per-object material overrides", n_matrix, n_shared, n_override)
        if col.objects:
            import numpy as _np
            _p = _np.array([o.matrix_basis.translation[:] for o in list(col.objects)[:20000]])
            L.info("object origins (first %d, Blender units, local): min %s max %s, at (0,0,0): %d", len(_p), _p.min(0).round(2).tolist(),
                   _p.max(0).round(2).tolist(), int((abs(_p).sum(1) < 1e-9).sum()))
        msg = f"Imported {len(insts)} of {len(scene.instances)} objects ({len(scene.meshes)} unique meshes) from {Path(self._path).name}"
        if scene.skipped:
            msg += "; skipped " + ", ".join(f"{v}x {k}" for k, v in scene.skipped.items())
        self.report({"INFO"}, msg)
