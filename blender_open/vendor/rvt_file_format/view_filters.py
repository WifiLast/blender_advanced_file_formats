"""View filters: colour / transparency a Revit 3D view gives elements through its filters (Navisworks colours this way).

Navisworks exports a Revit model from a 3D view and takes the surface colours from that view's filter overrides,
not only from the materials (Golden Nugget, nugget.nwd = Navisworks' own export of view '3D BIM Variante 02'):
ducts are coloured per system abbreviation (filter L_AB 'RA_R' -> orange, L_ABWC 'EXH_T' -> 0.79/0.41/0, ...),
accessories / fixtures by category-only filters (grey 0.75), pipe insulation gets 100 % surface transparency.

Storage (DBView3d):
  m_viewName; m_oFilterOverrides.m_oaSettings[] = FilterGraphicSettings {m_filterElemId, m_bEnabledFlag,
  m_bVisibilityFlag, m_pOverrideGraphicSettings: OverrideGraphicSettings {m_oProjFillPattern.m_fillPatternId,
  m_oProjFillColor.m_color (COLORREF, >= 0x1000000 = not overridden), m_bProjFillPatternVisible,
  m_oSurfacesTransparency.m_transparency (0..100), ...}}. List order = Revit's filter order: the first filter that
  overrides a property wins.
ParameterFilterElement: m_name, m_oCategoryRule.m_categories [category ids], m_oRuleCombination = None (category
  only) | LogicalAndFilterData / LogicalOrFilterData {m_inverted, m_filters[]} | ElementParameterFilterData
  {m_inverted, m_rules[] (all must pass)}; rules FilterStringRule {m_oValueProvider.m_parameter, m_oEvaluator
  (FilterStringEquals/BeginsWith/EndsWith/Contains/Greater/Less...), m_ruleString}, FilterElementIdRule /
  FilterIntegerRule / FilterDoubleRule {m_ruleValue, FilterNumeric*}, FilterInverseRule {m_oInnerRule}.
Surface colour: only a SOLID fill pattern shows as colour (FillPatternElem.m_pFillPattern.m_gridsArr empty);
  hatched fills keep the material colour (R_Daemmung: hatch + colour 0.5 -> nugget.nwd keeps the material colour).
Parameter values: the built-ins Revit computes (system name / abbreviation / type, family / type name) are derived
  here (Materials.systems for the element's MEP systems); everything else from the stored ParamValues of the
  element (m_elemOrSymbol 0) or its type (2). A rule whose parameter has no value does not match (also inverted).
"""
from __future__ import annotations

from .materials import colorref_rgb
from .params import collect_values

NO_OVERRIDE = 1 << 24
AUTO_VIEW_TOKENS = ("navisworks", "nwc", "nwd", "export", "koordination", "coordination", "bim")

_STR_EVAL = {
    "FilterStringEquals": lambda v, r: v == r,
    "FilterStringBeginsWith": lambda v, r: v.startswith(r),
    "FilterStringEndsWith": lambda v, r: v.endswith(r),
    "FilterStringContains": lambda v, r: r in v,
    "FilterStringGreater": lambda v, r: v > r,
    "FilterStringGreaterOrEqual": lambda v, r: v >= r,
    "FilterStringLess": lambda v, r: v < r,
    "FilterStringLessOrEqual": lambda v, r: v <= r,
}
_NUM_EVAL = {
    "FilterNumericEquals": lambda v, r: v == r,
    "FilterNumericGreater": lambda v, r: v > r,
    "FilterNumericGreaterOrEqual": lambda v, r: v >= r,
    "FilterNumericLess": lambda v, r: v < r,
    "FilterNumericLessOrEqual": lambda v, r: v <= r,
}


def view_names(doc) -> dict:
    """DBView3d id -> view name."""
    return {e: (doc.objects.get(e) or {}).get("m_viewName") or f"view {e}"
            for e, c in doc.elem_class.items() if c == "DBView3d"}


class ViewFilters:
    def __init__(self, doc, mats, view: str | None = None, log=None):
        """view: name of a 3D view; None = automatic (the 3D view with the most solid-fill colour filters among those
        whose name contains one of AUTO_VIEW_TOKENS, in that order, else among all); 'none' = no view filters."""
        self.doc, self.mats, self.log = doc, mats, log or (lambda *a: None)
        self.view_id, self.view_name, self.settings = None, None, []
        self.auto_candidates: list[str] = []          # set when the view was guessed: other views with colour filters
        self.stats = {"coloured": 0, "transparent": 0, "hidden_in_view": 0, "unknown_rules": set()}
        self._solid: dict[int, bool] = {}
        if view and view.lower() == "none":
            return
        names = view_names(doc)
        cand = {e: self._parse(e) for e in names}
        if view:
            hit = [e for e, n in names.items() if n == view] or [e for e, n in names.items() if n.lower() == view.lower()]
            if not hit:
                raise ValueError(f"3D view {view!r} not found; 3D views with filters: "
                                 + ", ".join(sorted(repr(names[e]) for e, s in cand.items() if s)))
            vid = hit[0]
        else:
            # The .rvt does not record which view Navisworks exported (InitialViewSettings is Revit's start view, in
            # Golden Nugget a drafting view). Heuristic: a view whose name says it is meant for coordination/export,
            # else the 3D view with the most colour filters. Golden Nugget: nugget.nwd = '3D BIM Variante 02'.
            coloured = {e: sum(1 for s in st if s["rgb"] is not None) for e, st in cand.items()}
            ranked = sorted((e for e in coloured if coloured[e]), key=lambda e: (-coloured[e], e))
            pick = []
            for tok in AUTO_VIEW_TOKENS:
                pick = [e for e in ranked if tok in names[e].lower()]
                if pick:
                    break
            pick = pick or ranked
            if not pick:
                return
            vid = pick[0]
            self.auto_candidates = [names[e] for e in ranked[:8]]
        self.view_id, self.view_name, self.settings = vid, names[vid], cand[vid]

    # ---- view -------------------------------------------------------------------------------------------------
    def _is_solid(self, pid) -> bool:
        if pid not in self._solid:
            fp = ((self.doc.objects.get(pid) or {}).get("m_pFillPattern")) if isinstance(pid, int) and pid > 0 else None
            self._solid[pid] = isinstance(fp, dict) and not (fp.get("m_gridsArr") or [])
        return self._solid[pid]

    def _parse(self, vid) -> list:
        o = self.doc.objects.get(vid) or {}
        out = []
        for s in ((o.get("m_oFilterOverrides") or {}).get("m_oaSettings") or []):
            f = self.doc.objects.get(s.get("m_filterElemId")) or {}
            if not s.get("m_bEnabledFlag", 1) or not f:
                continue
            g = s.get("m_pOverrideGraphicSettings") or {}
            col = (g.get("m_oProjFillColor") or {}).get("m_color")
            pat = (g.get("m_oProjFillPattern") or {}).get("m_fillPatternId")
            rgb = None
            if isinstance(col, int) and col < NO_OVERRIDE and g.get("m_bProjFillPatternVisible", 1) \
                    and self._is_solid(pat):
                rgb = tuple(float(x) for x in colorref_rgb(col))
            tr = (g.get("m_oSurfacesTransparency") or {})
            tr = tr.get("m_transparency") if isinstance(tr, dict) else tr
            tr = float(tr) / 100.0 if isinstance(tr, (int, float)) and tr > 0 else None
            out.append({"name": f.get("m_name") or str(s.get("m_filterElemId")),
                        "cats": set((f.get("m_oCategoryRule") or {}).get("m_categories") or []),
                        "rules": f.get("m_oRuleCombination"), "visible": bool(s.get("m_bVisibilityFlag", 1)),
                        "rgb": rgb, "transparency": tr})
        return out

    # ---- parameter values --------------------------------------------------------------------------------------
    def _value(self, eid, pid, symbol: bool):
        doc = self.doc
        name = doc.params.describe(pid).get("name") or ""
        systems = self.mats.systems(eid) if name.startswith("RBS_") else []
        if name == "RBS_SYSTEM_NAME_PARAM":
            v = ",".join((doc.objects.get(s) or {}).get("m_strName") or "" for s in systems)
            return v or None
        if name == "RBS_DUCT_PIPE_SYSTEM_ABBREVIATION_PARAM":
            v = ",".join((doc.objects.get((doc.objects.get(s) or {}).get("m_typeId")) or {}).get("m_abbreviation")
                         or "" for s in systems)
            return v or None
        if name in ("RBS_PIPING_SYSTEM_TYPE_PARAM", "RBS_DUCT_SYSTEM_TYPE_PARAM", "RBS_SYSTEM_TYPE_PARAM"):
            return self.mats.system_type(eid)
        t = doc.type_of(eid)
        if name == "ALL_MODEL_FAMILY_NAME":
            return ((doc.family_of(t) or {}).get("name")) if t else None
        if name in ("ALL_MODEL_TYPE_NAME", "SYMBOL_NAME_PARAM"):
            return doc.names.get(t) if t else None
        if name in ("ELEM_FAMILY_AND_TYPE_PARAM", "ELEM_FAMILY_PARAM"):
            f = ((doc.family_of(t) or {}).get("name")) if t else None
            return f"{f}: {doc.names.get(t)}" if f and name == "ELEM_FAMILY_AND_TYPE_PARAM" else f
        if name in ("ELEM_TYPE_PARAM", "SYMBOL_ID_PARAM"):
            return t
        for src in ((t,) if symbol else (eid, t)):
            if src is None:
                continue
            for _p, _c, p, v in collect_values(doc.objects.get(src) or {}):
                if p == pid and not isinstance(v, dict):
                    return v
        return None

    # ---- rules --------------------------------------------------------------------------------------------------
    def _rule(self, eid, r) -> bool:
        c = r.get("__class__")
        if c == "FilterInverseRule":
            inner = r.get("m_oInnerRule") or {}
            v = self._rule_value(eid, inner)
            return v is not None and not self._rule(eid, inner)
        v = self._rule_value(eid, r)
        if v is None:
            return False
        ev = (r.get("m_oEvaluator") or {}).get("__class__")
        if c == "FilterStringRule":
            f = _STR_EVAL.get(ev)
            if f is None:
                self.stats["unknown_rules"].add(ev)
                return False
            cs = r.get("m_caseSensitive", 0)
            a, b = str(v), str(r.get("m_ruleString") or "")
            return f(a, b) if cs else f(a.casefold(), b.casefold())
        if c in ("FilterElementIdRule", "FilterIntegerRule", "FilterDoubleRule"):
            f = _NUM_EVAL.get(ev)
            if f is None or not isinstance(v, (int, float)):
                self.stats["unknown_rules"].add(ev or c)
                return False
            rv = r.get("m_ruleValue")
            if c == "FilterDoubleRule":
                eps = float(r.get("m_epsilon") or 1e-9)
                if ev == "FilterNumericEquals":
                    return abs(v - rv) <= eps
            return f(v, rv)
        self.stats["unknown_rules"].add(c)
        return False

    def _rule_value(self, eid, r):
        vp = r.get("m_oValueProvider") or {}
        return self._value(eid, vp.get("m_parameter"), vp.get("m_elemOrSymbol") == 2)

    def _data(self, eid, d) -> bool:
        if not isinstance(d, dict):
            return True                                       # category-only filter
        c = d.get("__class__")
        if c == "ElementParameterFilterData":
            ok = all(self._rule(eid, r) for r in d.get("m_rules") or [])
        elif c == "LogicalAndFilterData":
            ok = all(self._data(eid, x) for x in d.get("m_filters") or [])
        elif c == "LogicalOrFilterData":
            ok = any(self._data(eid, x) for x in d.get("m_filters") or [])
        else:
            self.stats["unknown_rules"].add(c)
            return False
        return (not ok) if d.get("m_inverted") else ok

    # ---- per element --------------------------------------------------------------------------------------------
    def override(self, eid):
        """{"rgb", "transparency", "filters", "view"} for an element, or None when no filter of the view changes it.
        Per property the first matching filter (view list order) that overrides it wins, as in Revit."""
        if not self.settings:
            return None
        cat = self.doc.category.get(eid)
        rgb = tr = None
        names, hidden = [], False
        for s in self.settings:
            if cat not in s["cats"] or (s["rgb"] is None and s["transparency"] is None and s["visible"]):
                continue
            if not self._data(eid, s["rules"]):
                continue
            hidden = hidden or not s["visible"]
            used = False
            if rgb is None and s["rgb"] is not None:
                rgb, used = s["rgb"], True
            if tr is None and s["transparency"] is not None:
                tr, used = s["transparency"], True
            if used:
                names.append(s["name"])
        if hidden:
            self.stats["hidden_in_view"] += 1
        if rgb is None and tr is None:
            return None
        self.stats["coloured"] += rgb is not None
        self.stats["transparent"] += tr is not None
        return {"rgb": rgb, "transparency": tr, "filters": names, "view": self.view_name}
