import time
from pathlib import Path

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from .. import build
from ..rvt.reader import read_rvt
from .. import log as cadlog
from ._threaded import ThreadedImportMixin

L = cadlog.get("build")


class IMPORT_SCENE_OT_rvt(Operator, ImportHelper, ThreadedImportMixin):
    bl_idname = "import_scene.rvt"
    bl_label = "Revit (.rvt)"
    bl_description = "Import the 3D geometry stored in a Revit .rvt file (no Revit needed; large files take minutes)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".rvt"
    filter_glob: StringProperty(default="*.rvt", options={"HIDDEN"})
    category_filter: StringProperty(
        name="Category Filter",
        description="Regular expression matched against the Revit category (empty = everything)",
        default="",
    )
    global_scale: FloatProperty(
        name="Scale", description="Multiplier on the metre coordinates",
        default=1.0, min=1e-6, soft_max=1000.0,
    )
    import_params: BoolProperty(
        name="Revit Parameters",
        description="Store the element parameters as custom properties on the objects",
        default=True,
    )
    view_name: StringProperty(
        name="3D View (colours)",
        description="Revit 3D view whose filters colour the elements, as Navisworks does. Empty = guess the view "
                    "(logged), 'none' = Revit material colours only",
        default="",
    )
    import_lines: BoolProperty(
        name="Lines",
        description="Also import free lines (pipe / duct centre lines, grille louvres) as edge-only objects",
        default=False,
    )
    group_by_category: BoolProperty(name="Collection per Category", default=True)
    recentre: BoolProperty(
        name="Move to Origin",
        description="Revit models sit far from the origin (survey / internal coordinates). Shift everything so the "
                    "model's footprint centre is at X/Y = 0 and its lowest point at Z = 0. The shift is stored on the "
                    "root collection as 'revit_offset' (metres)",
        default=True,
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
        self._cat = self.category_filter
        self._params = self.import_params
        self._view = self.view_name
        self._lines = self.import_lines
        return self._start(context)

    def _read(self):
        return read_rvt(self._path, self._cat, self._params, self._say, self._view, self._lines)

    @staticmethod
    def _material(scene, mats, mid):
        if mid is not None and mid in scene.materials:
            m = scene.materials[mid]
            vo = m.get("view_override")
            nm = m["name"] + (" [" + ", ".join(vo["filters"]) + "]" if vo and vo.get("filters") else "")
            return build.get_material(mats, mid, nm, m["rgb"], max(0.15, 1.0 - float(m.get("transparency") or 0.0)))
        return build.get_material(mats, None, "Revit default", (0.8, 0.8, 0.82))

    def _build(self, context, scene):
        if not scene.elements:
            self.report({"WARNING"}, "No 3D geometry found in this Revit file")
            return
        s = self.global_scale
        shift = np.zeros(3)
        if self.recentre:
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for el in scene.elements:
                for arr in [p[0] for p in el.parts] + ([el.lines[0]] if el.lines is not None else []):
                    if len(arr):
                        lo = np.minimum(lo, arr.min(axis=0) * s)
                        hi = np.maximum(hi, arr.max(axis=0) * s)
            if np.all(np.isfinite(lo)):
                shift = np.array([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2]])
                L.info("model bounds min %s max %s -> shifting by %s", np.round(lo, 2), np.round(hi, 2), np.round(shift, 2))
        root = build.new_collection(Path(self._path).stem)
        root["revit_offset"] = [float(c) for c in shift]
        cols, mats = {}, {}
        n_multi = n_lines = 0
        L.info("creating %d objects", len(scene.elements))
        t0 = time.perf_counter()
        for n_obj, el in enumerate(scene.elements):
            if n_obj and n_obj % 2000 == 0:
                L.info("  %d / %d objects (%.1f s)", n_obj, len(scene.elements), time.perf_counter() - t0)
            parent = root
            if self.group_by_category:
                parent = cols.get(el.category)
                if parent is None:
                    parent = cols[el.category] = build.new_collection(el.category, root)
            if el.parts:
                offs = np.cumsum([0] + [len(p[0]) for p in el.parts])
                verts = np.vstack([p[0] for p in el.parts]) * s
                centre = (verts.min(axis=0) + verts.max(axis=0)) * 0.5        # object origin = bounding-box centre, not world origin
                verts = verts - centre
                tris = np.vstack([p[1] + o for p, o in zip(el.parts, offs)])
                me = build.make_mesh(el.name, verts, tris)
                slots = {}
                for p in el.parts:
                    mid = p[2]
                    if mid in slots:
                        continue
                    me.materials.append(self._material(scene, mats, mid))
                    slots[mid] = len(me.materials) - 1
                if len(slots) > 1:
                    n_multi += 1
                    idx = np.concatenate([np.full(len(p[1]), slots[p[2]], dtype=np.int32) for p in el.parts])
                    me.polygons.foreach_set("material_index", idx)
                ob = bpy.data.objects.new(el.name, me)
                ob.location = tuple(float(c) for c in centre - shift)
                for k, val in el.props.items():
                    try:
                        ob[k] = val
                    except Exception:
                        pass
                parent.objects.link(ob)
            if el.lines is not None:
                lv, ls, lmid = el.lines
                lv = lv * s
                lc = (lv.min(axis=0) + lv.max(axis=0)) * 0.5
                lme = build.make_edge_mesh(el.name + " (lines)", lv - lc, ls)
                lme.materials.append(self._material(scene, mats, lmid))
                lob = bpy.data.objects.new(el.name + " (lines)", lme)
                lob.location = tuple(float(c) for c in lc - shift)
                parent.objects.link(lob)
                n_lines += 1
        L.info("created %d objects, %d collections, %d materials in %.1f s", len(scene.elements), len(cols), len(mats), time.perf_counter() - t0)
        L.info("build stats: %d objects with several material slots, %d line objects; collections: %s", n_multi, n_lines, sorted(cols))
        msg = f"Imported {len(scene.elements)} Revit elements from {Path(self._path).name}"
        if scene.skipped:
            msg += f" ({sum(scene.skipped.values())} elements without 3D geometry skipped)"
        self.report({"INFO"}, msg)
