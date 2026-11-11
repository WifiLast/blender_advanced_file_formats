"""Revit .rvt -> element meshes (metres, Z up) using the vendored rvt_file_format reader. Pure Python + numpy, no bpy."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import time

import numpy as np

from .. import log as cadlog

from ..vendor.rvt_file_format import model as rmodel
from ..vendor.rvt_file_format.materials import Materials
from ..vendor.rvt_file_format.view_filters import ViewFilters

FT = 0.3048
L = cadlog.get("rvt")


@dataclass
class RvtElement:
    name: str
    category: str
    cls: str
    parts: list                       # [(vertices (N,3) metres, triangles (M,3), material id | None)]
    props: dict = field(default_factory=dict)
    lines: tuple | None = None        # (vertices (N,3) metres, segments (M,2), material id | None)


@dataclass
class RvtScene:
    elements: list = field(default_factory=list)
    materials: dict = field(default_factory=dict)     # material id -> {"name", "rgb", "transparency"}
    skipped: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def read_rvt(path: str, category_regex: str = "", with_params: bool = True, log=None, view: str = "",
             with_lines: bool = False) -> RvtScene:
    log = log or (lambda *a: None)
    log("reading Revit database (large files take minutes) ...")
    L.info("file %s, category filter %r, parameters %s", path, category_regex, with_params)
    t0 = time.perf_counter()
    m = rmodel.Model(path, log=lambda *a: (log(*a), L.debug("rvt: %s", " ".join(str(x) for x in a))))
    L.info("database read in %.1f s: %d elements, %d classes", time.perf_counter() - t0, len(m.elem_class), len(m.classes))
    m.with_lines = with_lines
    t0 = time.perf_counter()
    mats = Materials(m.doc)
    L.info("materials: %d (%.1f s)", len(mats.mats), time.perf_counter() - t0)
    vf = ViewFilters(m.doc, mats, view or None)
    if vf.view_name:
        L.info("view filters of 3D view %r (%d filters)%s", vf.view_name, len(vf.settings),
               "; guessed, candidates: " + ", ".join(map(repr, vf.auto_candidates)) if vf.auto_candidates else "")
        log("colours from 3D view " + repr(vf.view_name))
    else:
        L.info("no view filters: colours from Revit materials only")
    t0 = time.perf_counter()
    cats = {}
    rx = re.compile(category_regex, re.I) if category_regex else None
    scene = RvtScene()
    n = 0
    for eid, cls, cat, parts in m.element_meshes(rmodel.PHYSICAL, rx, 1):   # jobs=1: no worker processes inside Blender
        by_mat: dict = {}
        ov = vf.override(eid)
        lines = None
        for key, (v, tri) in parts.items():
            if key == rmodel.LINE_KEY:
                lines = (v, tri)
                continue
            mid = mats.resolve(key, eid)
            if ov is not None:
                mid = mats.with_view_override(mid, ov)
            by_mat.setdefault(mid, []).append({0: (v, tri)})
        out = []
        for mid, lst in by_mat.items():
            v, tri = rmodel._merge_dicts(lst)[0]
            out.append((np.asarray(v, dtype=np.float64) * FT, np.asarray(tri, dtype=np.int64), mid))
            if mid is not None and mid in mats.mats:
                scene.materials[mid] = mats.mats[mid]
        el = RvtElement(f"{m.doc.names.get(eid) or cls} [{eid}]", cat, cls, out)
        if lines is not None:
            lmid = mats.resolve((-1, -1), eid)
            if ov is not None:
                lmid = mats.with_view_override(lmid, ov)
            el.lines = (np.asarray(lines[0], dtype=np.float64) * FT, np.asarray(lines[1], dtype=np.int64), lmid)
            if lmid is not None and lmid in mats.mats:
                scene.materials[lmid] = mats.mats[lmid]
        if with_params:
            try:
                el.props = {"elementId": eid, "class": cls, "category": cat}
                for p in m.doc.parameters(eid):
                    val = p.get("value_si", p.get("value"))
                    if isinstance(val, (int, float, str)) and p.get("name"):
                        el.props[p["name"]] = val
            except Exception:
                pass
        scene.elements.append(el)
        cats[cat] = cats.get(cat, 0) + 1
        L.debug("element %s class=%s category=%s parts=%d tris=%d", el.name, cls, cat, len(out), sum(len(p[1]) for p in out))
        n += 1
        if n % 500 == 0:
            log(f"{n} elements ...")
    scene.skipped = dict(m.stats.skipped)
    scene.info = {"elements": n}
    log(f"{n} elements with geometry")
    L.info("meshing took %.1f s", time.perf_counter() - t0)
    if vf.view_name:
        L.info("view filters: %d elements coloured, %d transparent, %d hidden in the view (still imported); rules not evaluated: %s",
               vf.stats["coloured"], vf.stats["transparent"], vf.stats["hidden_in_view"], sorted(map(str, vf.stats["unknown_rules"])) or "none")
    L.info("elements per category: %s", dict(sorted(cats.items(), key=lambda kv: -kv[1])))
    L.info("triangles: %d, materials used: %d", sum(len(p[1]) for e in scene.elements for p in e.parts), len(scene.materials))
    L.info("skipped (reason: count): %s", scene.skipped or "nothing")
    if scene.elements:
        allv = np.vstack([p[0] for e in scene.elements for p in e.parts])
        L.info("model bounds (m): min %s max %s", allv.min(0).round(2).tolist(), allv.max(0).round(2).tolist())
    return scene
