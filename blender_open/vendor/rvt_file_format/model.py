"""Stage 5: assemble model geometry - elements, family instances, transforms.

- Elements of PHYSICAL classes are exported; their GElement B-rep is in model coordinates.
- GInstance nodes place family symbol geometry: world = R @ v + t with
  R = Trf.m_3x3 (applied column-wise, verified against stored instance bounding boxes) and
  t = Trf.m_or. Symbol geometry can itself contain GInstances (nested families).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry

PHYSICAL = {"SWall", "Floor", "Roof", "Ceiling", "FamilyInstance", "ImportInstance",
            "Stairs", "Railing", "CurtainSystem", "TopographySurface", "Toposolid",
            "DirectShape", "Duct", "Pipe", "FlexDuct", "FlexPipe", "CableTray", "Conduit",
            # MEP curves are stored under their "Rbs" class names in Revit 2023 files (found with tools/tally_classes.py:
            # 2381 pipes, 768 pipe insulations, 277 ducts, 5 flex ducts, 1 flex pipe, all with a stored GElement)
            "RbsPipeCurve", "RbsPipeInsulation", "RbsDuctCurve", "RbsFlexDuctCurve", "RbsFlexPipeCurve",
            # centre lines of pipes / ducts: points only, exported as line geometry (NWD --lines, on by default)
            "SegmentCenterLine", "PipeFittingCenterLine", "DuctFittingCenterLine", "FlexCenterLine"}
MAX_DEPTH = 8
parmesh_NO_GEOMETRY, parmesh_NO_MESH, parmesh_OK = 0, 1, 2      # statuses of parmesh.py (kept here: no import cycle)
# Subcategories that Revit hides by default and Navisworks therefore does not export. The light-source symbol of every
# lamp family (a 2 ft cone/cylinder below the fixture) lies on OST_LightingFixtureSource: 187 solids on Golden Nugget,
# all in lighting fixtures; compare_geometry.py showed 190+ lamps 3.4-4.3x too large against Navisworks' own nugget.nwd.
HIDDEN_CATEGORIES = {-2001121}
LINE_KEY = ("lines", 0)       # material key of an element's free lines: parts[LINE_KEY] = (vertices N x 3, segments M x 2)
# categories whose family instances are 2D (sheets, details, annotations): no 3D geometry
TWO_D_CATEGORY_TOKENS = ("Detail", "TitleBlock", "Annotation", "Tags", "Marks", "Callout", "Symbol")

_DATA = Path(__file__).with_name("data")


def _categories():
    try:
        return {int(k): v for k, v in json.loads((_DATA / "builtin_categories.json").read_text()).items()}
    except OSError:
        return {}


class Model:
    def __init__(self, path: str, log=None, doc=None):
        from .document import Document   # noqa: PLC0415
        self.log = log or (lambda *a: None)
        self.doc = doc or Document(path, log=self.log)
        self.elem_class = self.doc.elem_class
        self.category = self.doc.category
        self.gel = self.doc.gel
        self.classes = self.doc.classes
        self.stats = geometry.Stats()
        self._sym_cache: dict[int, tuple | None] = {}
        self.cat_names = _categories()
        self.with_lines = False          # also extract the free lines (Navisworks exports them as line geometry)
        self.hidden_styles = self._find_hidden_styles()
        self.secondary_options = self._find_secondary_options()

    def _find_secondary_options(self) -> frozenset:
        """Design options that are not the main option of their set. Navisworks exports the main options only
        (Golden Nugget: option 'Klimatisierung auf Luftbasis' is secondary, 'Dezentrale Klimatisierung' is the main
        option of its set - none of the 195+ elements of the secondary option is in Navisworks' own nugget.nwd)."""
        out = set()
        for eid, cls in self.doc.elem_class.items():
            if cls != "DesignOption":
                continue
            o = self.doc.objects.get(eid) or {}
            s = self.doc.objects.get(o.get("m_DesignOptionSetId")) or {}
            main = s.get("m_mainDesignOption")
            if isinstance(main, int) and main > 0 and main != eid:
                out.add(eid)
        return frozenset(out)

    def _find_hidden_styles(self) -> frozenset:
        """Graphics-style ids (GInfo.m_categoryId of a solid) of the subcategories Revit hides by default."""
        out = set()
        for eid, cls in self.doc.elem_class.items():
            if cls == "GStyleElem" and (self.doc.objects.get(eid) or {}).get("m_categoryId") in HIDDEN_CATEGORIES:
                out.add(eid)
        return frozenset(out)

    # ------------------------------------------------------------------

    def category_name(self, eid: int) -> str:
        c = self.category.get(eid)
        return self.cat_names.get(c, str(c)) if c is not None else "-"

    def _grep_solids(self, node, objs, depth):
        """Solids [(material key, v, t, is_mesh, from_symbol)] of a GRep subtree in its own coordinates, including
        nested instances (their solids transformed, from_symbol=True)."""
        out = [(k, v, t, mesh, False) for k, v, t, mesh in geometry.solid_parts(node, objs, self.stats)]
        if self.with_lines:
            fl = geometry.free_lines(node)
            if fl is not None:                      # flagged as a mesh part: the family-solid box test does not apply
                out.append((LINE_KEY, fl[0], fl[1], True, False))
        for gi in geometry.find_nodes(node, ("GInstance",), stop=("GInstance",)):
            if gi is node:
                continue
            sub = self._instance_solids(gi, objs, depth + 1)
            if sub:
                out.extend((k, v, t, mesh, True) for k, v, t, mesh, _s in sub)
        return out

    def _grep_mesh(self, node, objs, depth):
        """Mesh parts {material key: (v, t)} of a GRep subtree in its own coordinates,
        including nested instances (keys travel with the transformed parts)."""
        return _merge_solids(self._grep_solids(node, objs, depth))

    def element_mesh(self, gelem: dict, objs: dict):
        """{material key: (v, t)} of one element's GElement in model coordinates, without the family solids that
        Revit hides through family visibility parameters (see visible_solids)."""
        solids = self._grep_solids(gelem, objs, 0)
        if self.hidden_styles:
            shown = [s for s in solids if not (isinstance(s[0], tuple) and s[0][1] in self.hidden_styles)]
            if len(shown) < len(solids):
                why = "hidden subcategory (lighting fixture source)"
                self.stats.skipped[why] = self.stats.skipped.get(why, 0) + len(solids) - len(shown)
            solids = shown
        return _merge_solids(visible_solids(solids, gelem.get("m_tightbBox"), self.stats))

    def _symbol_solids(self, sym: int, depth: int):
        if sym in self._sym_cache:
            return self._sym_cache[sym]
        self._sym_cache[sym] = None                # guards against cycles
        g = self.gel.get(sym)
        solids = self._grep_solids(g[0], g[1], depth) if g else None
        self._sym_cache[sym] = solids
        return solids

    def _instance_solids(self, gi: dict, objs: dict, depth: int):
        if depth > MAX_DEPTH:
            self.stats.skip("instance depth")
            return None
        info = gi.get("m_instanceInfo") or {}
        trf = info.get("m_Trf")
        embedded = gi.get("m_oEmbeddedSymbolGRep")
        if isinstance(embedded, dict) and len(embedded) > 2:
            solids = self._grep_solids(embedded, objs, depth)
        else:
            solids = self._symbol_solids(info.get("m_symbolId"), depth)
        if not solids or not isinstance(trf, dict):
            if not solids:
                self.stats.skip("instance without symbol geometry")
            return None
        R = np.array(trf["m_3x3"], float)
        t = np.array(trf["m_or"], float)
        return [(k, v @ R.T + t, tris, mesh, fs) for k, v, tris, mesh, fs in solids]

    def element_meshes(self, classes=PHYSICAL, category_rx=None, jobs: int = 1):
        """Yield (eid, class, category, {material key: (vertices, triangles)}) in model
        coordinates (feet).
        category_rx: only elements whose category name matches are meshed (the rest are not even decoded).
        jobs > 1: the geometry stage runs in that many worker processes (parmesh.py); same output, same order."""
        # Family-definition content (nested parts of loaded families) is stored as elements too, but with no creation
        # phase (m_createdPhaseId == -1) and geometry in family-local coordinates: they sit at the world origin and
        # are not elements of the project (none of 230 such elements is in the Revit API dump of Golden Nugget;
        # no project element has -1). Their geometry already reaches the model through the owning instance.
        # Only applied when the file gives its physical elements creation phases at all: in files without them
        # (GN_Gebaeudetechnik_Neu.rvt: all 215 physical elements have -1, none has a phase) -1 says nothing and
        # the filter would drop every element.
        phased = any(isinstance(ph := (self.doc.objects.get(e) or {}).get("m_createdPhaseId"), int) and ph > 0
                     for e, c in self.elem_class.items() if c in classes and e in self.gel)
        if not phased:
            self.log("no physical element has a creation phase: family-internal filter (m_createdPhaseId == -1) off")
        work = []                                   # (eid, class, category, is 2D) of the elements to mesh, in file order
        for eid, cls in self.elem_class.items():
            if cls not in classes:
                continue
            rec = self.doc.objects.get(eid) or {}
            if phased and rec.get("m_createdPhaseId") == -1:
                self.stats.skip("family-internal element (no creation phase)")
                continue
            if rec.get("m_designOptionId") in self.secondary_options:
                self.stats.skip("element of a secondary design option")
                continue
            cat = self.category_name(eid)
            if category_rx is not None and not category_rx.search(cat):
                continue
            work.append((eid, cls, cat, any(tok in cat for tok in TWO_D_CATEGORY_TOKENS)))

        if jobs > 1 and len(work) >= 4 * jobs:
            from . import parmesh          # noqa: PLC0415
            results = parmesh.mesh_elements(self.doc, [w[0] for w in work], self.stats, jobs, self.hidden_styles,
                                            self.with_lines)
        else:
            results = self._mesh_sequential(w[0] for w in work)
        for (eid, cls, cat, two_d), (rid, status, m) in zip(work, results):
            assert rid == eid
            if status == parmesh_NO_GEOMETRY:
                self.stats.skip("2D/annotation element (no 3D geometry)" if two_d
                                else f"no stored geometry: {cls}")
            elif status == parmesh_NO_MESH:
                self.stats.skip("2D/annotation element (no 3D geometry)" if two_d
                                else f"no solid geometry (lines/symbols only): {cls} {cat}")
            else:
                yield eid, cls, cat, m

    def _mesh_sequential(self, eids):
        for eid in eids:
            g = self.gel.get(eid)
            if g is None:
                yield eid, parmesh_NO_GEOMETRY, None
                continue
            m = self.element_mesh(g[0], g[1])
            yield (eid, parmesh_OK, m) if m else (eid, parmesh_NO_MESH, None)


HIDDEN_TOL = 0.002 / 0.3048     # feet: a solid more than 2 mm outside the instance's tight box is hidden


def visible_solids(solids, tight_box, stats=None):
    """Drop the family solids Revit hides through family visibility parameters (e.g. the 2nd and 3rd socket of an
    outlet family whose type shows one): the symbol geometry stores every solid, but the instance's
    GElement.m_tightbBox is the box of what is visible. A B-rep solid from a family symbol that is not inside that box
    (+2 mm) is left out. Mesh parts (GPolyMesh) are kept - the box does not include them (lamps) - and so are the
    element's own solids. Verified on Golden Nugget against Revit's GLB export: 221 family instances fixed (all 209
    electrical fixtures, 91 lighting devices), none of 2,712 matching instances changed."""
    try:
        lo = np.asarray(tight_box[0], float) - HIDDEN_TOL
        hi = np.asarray(tight_box[1], float) + HIDDEN_TOL
        if lo.shape != (3,) or not (hi >= lo).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
            return solids
    except (TypeError, IndexError, ValueError):
        return solids
    keep = [s for s in solids if not s[4] or s[3] or len(s[1]) == 0 or _inside_or_rim(s[1], lo, hi)]
    if not keep:                                    # the box says nothing about this element: keep everything
        return solids
    if stats is not None and len(keep) < len(solids):
        why = "family solids hidden (outside the instance box)"
        stats.skipped[why] = stats.skipped.get(why, 0) + len(solids) - len(keep)
    return keep


RIM_TOL = 0.03 / 0.3048         # feet


def _inside_or_rim(v, lo, hi):
    """Solid inside the (widened) box, or a rim around it: spans the box completely in at least two axes and sticks
    out by at most 3 cm (ceiling lamp rings lie 2 cm outside Revit's tight box but are drawn; the hidden outlet sockets
    that also span two axes stick out 7 cm). Checked against the '3D Gebaeudetechnik' GLB export: 26 lamps fixed,
    none of the ~1,760 correctly dropped solids kept."""
    slo, shi = v.min(axis=0), v.max(axis=0)
    if (slo >= lo).all() and (shi <= hi).all():
        return True
    spans = int(((slo <= lo + HIDDEN_TOL) & (shi >= hi - HIDDEN_TOL)).sum())
    out = float(np.maximum(lo - slo, shi - hi).max()) - HIDDEN_TOL
    return spans >= 2 and out <= RIM_TOL


def _merge_solids(solids):
    groups: dict = {}
    for s in solids or []:
        groups.setdefault(s[0], []).append((s[1], s[2]))
    out = {}
    for k, parts in groups.items():
        vs, ts, base = [], [], 0
        for v, t in parts:
            vs.append(v)
            ts.append(np.asarray(t) + base)
            base += len(v)
        out[k] = (np.vstack(vs), np.vstack(ts))
    return out


def _merge_dicts(dicts):
    groups: dict = {}
    for d in dicts:
        for k, vt in (d or {}).items():
            groups.setdefault(k, []).append(vt)
    out = {}
    for k, parts in groups.items():
        vs, ts, base = [], [], 0
        for v, t in parts:
            vs.append(v)
            ts.append(t + base)
            base += len(v)
        out[k] = (np.vstack(vs), np.vstack(ts))
    return out


# ---------------------------------------------------------------- links --

def _strings(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            if not k.startswith("__"):
                _strings(v, out)
    elif isinstance(o, list):
        for v in o:
            _strings(v, out)
    return out


def links(doc) -> list[dict]:
    """Revit links of a document: [{instanceId, name, file, paths, trf (R 3x3, t)}]."""
    out = []
    for eid, cls in doc.elem_class.items():
        if cls != "RvtLinkInstance":
            continue
        o = doc.objects.get(eid) or {}
        info = o.get("m_pInstanceInfo") or {}
        trf = info.get("m_Trf") or {}
        sym = info.get("m_symbolId")
        so = doc.objects.get(sym) or {}
        strs = _strings(so.get("m_otherDocModelLocation"), []) + _strings(so.get("m_symbolInfo"), [])
        paths = [s for s in strs if s.lower().endswith(".rvt")]
        name = doc.names.get(sym) or (paths[0] if paths else None)
        out.append({"instanceId": eid, "symbolId": sym, "name": name,
                    "file": (paths[0].replace("\\", "/").rsplit("/", 1)[-1] if paths else name),
                    "paths": sorted(set(paths)), "instanceName": o.get("m_name"),
                    "R": trf.get("m_3x3"), "t": trf.get("m_or")})
    return out


def resolve_link(link: dict, search_dirs) -> str | None:
    import os   # noqa: PLC0415
    for p in link["paths"]:
        if os.path.exists(p):
            return p
    for d in search_dirs or []:
        for cand in [link.get("file"), link.get("name")]:
            if cand and os.path.exists(os.path.join(d, cand)):
                return os.path.join(d, cand)
    return None
