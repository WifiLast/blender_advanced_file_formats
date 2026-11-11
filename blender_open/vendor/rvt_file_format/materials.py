"""Materials (RVT_FORMAT.md §14): MaterialElem -> name, colour, transparency, properties; per-face resolution.

Colour: Material.m_color is a Windows COLORREF 0x00BBGGRR (verified on materials whose names
state their RGB, e.g. "Glas - orange 255-128-0" = 0x0080FF). Transparency 0..1.

Resolution order for a face (first hit wins):
  1. Face.m_renderStyleId                    -> MaterialElem
  2. geometry node subcategory (GInfo.m_categoryId) -> GStyleElem/GStyle.m_materialElemId
  3. element type compound structure (CompoundStructureLayer.m_materialId, first set layer)
  4. element category (ElementHeader)       -> GStyle material
  5. MEP system type material override (RbsHvacSystemType / RbsPipingSystemType
     m_oSystemOverrides.m_oGMaterialOverrider.m_materialId) of the element's system (systems(): own connector
     m_idSystem / cell m_systemId, else inherited from connected elements; curves also m_idSystemType)
  6. none (default grey RGB 127, what Navisworks writes for faces without material)
On top of this, the view's filters (view_filters.py) may override colour / transparency per element:
with_view_override() derives a material entry ("view", base material, rgb, transparency).

Material properties (RVT_FORMAT.md §15), what Revit's material browser shows:
  - identity / graphics: Material fields (m_color, m_transparency, m_shininess, ...) + the MaterialElem's own
    ParamValues (description, manufacturer, cost, ...); appearance asset = AppearanceAssetElem.m_pAppearanceAsset.m_Name
  - physical / thermal: Material.m_assetMap.m_sharedAssetIdMap = [(MaterialAssetType GUID, PropertySetElement id)];
    GUID 2df6c9ed-... = structural asset (PropertySetElement.m_propertySetType 1), ec004816-... = thermal (type 2).
    The PropertySetElement's m_oParamSet holds ParamValue{Double,Int,AString} with built-in ids
    PHY_MATERIAL_PARAM_* / THERMAL_MATERIAL_PARAM_* / MATERIAL_ASSET_PARAM_* (names: the
    ParameterTypeId ForgeTypeIds exported by PersistenceDB.dll, autodesk.revit.parameter:phyMaterialParam...,
    = camelCase of the BuiltInParameter enum name).
  - The legacy Material.m_oStructuralPropertySet / m_oStructuralPropertySetNew and m_ownedAssets are NOT what Revit
    shows (old units, stale values) and are ignored.
Values are Revit internal units (feet, kg, s, K); UNIT below converts them to the units Revit displays. Factors
checked against the Revit API dump (data/*.groundtruth.jsonl.zip, raw vs display) where marked 'gt'.
"""
from __future__ import annotations

from .params import collect_values

DEFAULT_RGB = (127 / 255,) * 3       # Revit default grey: what Navisworks writes for faces without material
SYSTEM_HOPS = 4                      # connector steps a fitting / accessory looks for a system (systems())
MEP_CLASSES = {"FamilyInstance", "RbsDuctCurve", "RbsPipeCurve", "RbsFlexDuctCurve", "RbsFlexPipeCurve",
               "RbsCableTray", "RbsConduit"}

FT = 0.3048
ASSET_STRUCTURAL = "2df6c9ede4489e4aabb922c8664c012c"
ASSET_THERMAL = "ec0048162207264ba23504ffc53660d5"

# param id -> (factor applied to the internal value, display unit). Internal SI-with-feet units:
#   stress kg/(ft s2) -> MPa: /FT/1e6        unit weight kg/(ft2 s2) -> kN/m3: /FT2/1e3
_MPA = (1 / FT / 1e6, "MPa")
UNIT = {
    **{pid: _MPA for pid in (-1140414, -1140410, -1140409, -1140408, -1140407,                        # gt (wood)
                             -1140321, -1140319, -1140318, -1140316, -1140315, -1140314,              # gt (steel/concr.)
                             -1140300, -1140301, -1140302, -1140306, -1140307, -1140308,              # gt (moduli)
                             -1140401, -1140413, -1152301, -1152302, -1152305,
                             -1152315, -1152316, -1152317, -1152318)},
    -1140309: (1 / FT ** 2 / 1e3, "kN/m3"),          # unit weight                        gt
    -1152313: (1 / FT ** 3, "kg/m3"),                # density                            gt
    -1152312: (FT ** 2, "J/(kg K)"),                 # specific heat                      gt (0.71 J/(g C))
    -1152308: (FT, "W/(m K)"), -1152309: (FT, "W/(m K)"),
    -1152310: (FT, "W/(m K)"), -1152311: (FT, "W/(m K)"),   # thermal conductivity        gt
    -1152330: (FT ** 3, "Ohm m"),                    # electrical resistivity             gt
    **{pid: (1.0, "1/K") for pid in (-1140310, -1140311, -1140312, -1140415, -1152306, -1152307)},  # gt
    # from the dimensions only (no non-zero sample in the ground truth):
    -1152327: (1 / FT * 1e12, "ng/(Pa s m2)"),       # permeability, internal s/ft
    -1152325: (1 / FT, "Pa"),                        # vapour pressure
    -1152321: (1 / FT, "Pa s"), -1152323: (1 / FT, "Pa s"),  # gas / liquid viscosity
    -1152322: (FT, "1/Pa"),                          # compressibility
    -1152324: (FT ** 2, "J/kg"),                     # specific heat of vaporization
}
# integer parameters shown as text (German UI, as in the Revit API dump)
ENUMS = {
    -1150464: {0: "Nicht definiert", 1: "", 2: "Generisch", 3: "Metall", 4: "Beton", 5: "Holz", 6: "Flüssigkeit",
               7: "Gas", 8: "Kunststoff"},                                   # StructuralAssetClass (gt: 3, 1)
    -1140322: {0: "Isotrop", 1: "Orthotrop", 2: "Transversal isotrop"},      # StructuralBehavior (gt: 0)
    -1140400: {0: "Nicht definiert", 1: "Stahl", 2: "Beton", 3: "Holz", 4: "Sonstige", 5: "Betonfertigteil",
               6: "Generisch", 7: "Aluminium"},                              # StructuralMaterialType (gt: 4)
}
YES_NO = {-1152326, -1140323, -1152314, -1152319}     # -1152338 'Gemeinsame Nutzung' is shown as a number (gt)
# German labels for ids the Revit API dump did not list (own wording)
LABELS = {-1152325: "Dampfdruck", -1152324: "Spezifische Verdampfungswärme", -1152323: "Flüssigkeitsviskosität",
          -1152322: "Kompressibilität", -1152321: "Gasviskosität", -1152319: "Holzkonstruktion",
          -1152318: "Elastizitätsmodul (5 %-Fraktil)", -1152317: "Elastizitätsmodul (Mittelwert)",
          -1152316: "Zug lotrecht zu Körnung", -1152315: "Zug parallel zu Körnung",
          -1152314: "Thermisch behandelt", -1152307: "Wärmedehnungskoeffizient 2",
          -1152306: "Wärmedehnungskoeffizient 1", -1152305: "Schubmodul 12", -1152304: "Poissonsche Zahl 23",
          -1152303: "Poissonsche Zahl 12", -1152302: "Elastizitätsmodul (Young) 2",
          -1152301: "Elastizitätsmodul (Young) 1", -1140415: "Wärmedehnungskoeffizient",
          -1140413: "Schubmodul", -1140412: "Poissonsche Zahl", -1140401: "Elastizitätsmodul (Young)",
          -1140323: "Leichtbeton"}
# not material data: element bookkeeping Revit lists on every element
SKIP_IDS = {-1013201, -1013200, -1002001, -1002002, -1140363, -1140362, -1152385}


def colorref_rgb(c: int):
    return (c & 255) / 255.0, ((c >> 8) & 255) / 255.0, ((c >> 16) & 255) / 255.0


def _find(o, pred, out):
    if isinstance(o, dict):
        if pred(o):
            out.append(o)
        for k, v in o.items():
            if not k.startswith("__"):
                _find(v, pred, out)
    elif isinstance(o, list):
        for v in o:
            _find(v, pred, out)
    return out


def _ids_by_key(o, keys, out):
    """Integer values of the given field names anywhere in a decoded object, in order (no duplicates)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k.startswith("__"):
                continue
            if k in keys and isinstance(v, int) and v > 0 and v not in out:
                out.append(v)
            else:
                _ids_by_key(v, keys, out)
    elif isinstance(o, list):
        for v in o:
            _ids_by_key(v, keys, out)
    return out


def _num(v):
    return round(v, 9) if isinstance(v, float) else v


class Materials:
    def __init__(self, doc):
        self.doc = doc
        self.mats: dict[int, dict] = {}
        for eid, cls in doc.elem_class.items():            # class filter first: decode only these records
            if cls != "MaterialElem":
                continue
            o = doc.objects.get(eid) or {}
            m = o.get("m_pMaterial")
            if not isinstance(m, dict):
                continue
            r, g, b = colorref_rgb(int(m.get("m_color") or 0))
            t = float(m.get("m_transparency") or 0.0)
            self.mats[eid] = {"name": m.get("m_name") or f"material {eid}", "rgb": (r, g, b),
                              "transparency": t, "shininess": m.get("m_shininess"),
                              "smoothness": m.get("m_smoothness"),
                              "appearance_asset": m.get("m_appearanceAssetId")}
        # category / subcategory -> material via graphics styles
        self.cat_mat: dict[int, int] = {}
        for eid, cls in doc.elem_class.items():
            if cls != "GStyleElem":
                continue
            o = doc.objects.get(eid) or {}
            gs = o.get("m_pGStyle") or {}
            mid, cat = gs.get("m_materialElemId", -1), o.get("m_categoryId")
            if isinstance(cat, int) and mid in self.mats:
                self.cat_mat.setdefault(cat, mid)
        self._type_mat: dict[int, int | None] = {}
        self._props: dict[int, dict] = {}
        self._sys_types = {e for e, c in doc.elem_class.items() if c in ("RbsHvacSystemType", "RbsPipingSystemType")}
        self._systems: dict[int, list] = {}
        self._members, self._adj = None, None

    def type_material(self, type_id):
        if type_id is None:
            return None
        if type_id not in self._type_mat:
            o = self.doc.objects.get(type_id) or {}
            layers = _find(o, lambda d: d.get("__class__") == "CompoundStructureLayer", [])
            mid = next((l.get("m_materialId") for l in layers if l.get("m_materialId") in self.mats), None)
            self._type_mat[type_id] = mid
        return self._type_mat[type_id]

    def resolve(self, key, eid):
        """key = (face renderStyleId, node subcategory id) -> material id or None."""
        rs, node_cat = key
        if rs in self.mats:
            return rs
        if node_cat in self.cat_mat:
            return self.cat_mat[node_cat]
        mid = self.type_material(self.doc.type_of(eid))
        if mid is not None:
            return mid
        cat = self.doc.category.get(eid)
        mid = self.cat_mat.get(cat)
        if mid is not None:
            return mid
        return self.system_material(self.system_type(eid))

    # ---- MEP systems (resolution step 5, also used by view_filters) ---------------------------------------------
    def _mep_graph(self):
        """Built once: system membership (RbsHvacSystem / RbsPipingSystem .m_rgSections[].m_rgSection[].m_id) and the
        connector graph in both directions. Connections are often stored on one side only (Golden Nugget: duct
        2425192's connector refs fitting 2425193, the fitting has no m_pConnectorManager at all)."""
        if self._members is not None:
            return
        doc, cls = self.doc, self.doc.elem_class
        self._members, self._adj = {}, {}
        for e, c in cls.items():
            if c in ("RbsHvacSystem", "RbsPipingSystem"):
                o = doc.objects.get(e) or {}
                if o.get("m_typeId") not in self._sys_types:
                    continue
                for m in _ids_by_key(o.get("m_rgSections"), ("m_id",), []):
                    lst = self._members.setdefault(m, [])
                    if e not in lst:
                        lst.append(e)
            elif c in MEP_CLASSES:
                cm = (doc.objects.get(e) or {}).get("m_pConnectorManager")
                if not isinstance(cm, dict):
                    continue
                for n in _ids_by_key(cm, ("m_id",), []):
                    if n != e and n in cls and "System" not in cls[n]:
                        self._adj.setdefault(e, set()).add(n)
                        self._adj.setdefault(n, set()).add(e)

    def own_systems(self, eid) -> list:
        """RbsHvacSystem / RbsPipingSystem ids of an element itself: connector modifiers / cells (m_idSystem,
        m_systemId), then the systems that list it as a member."""
        o = self.doc.objects.get(eid) or {}
        out = [s for s in _ids_by_key(o, ("m_idSystem", "m_systemId"), [])
               if (self.doc.objects.get(s) or {}).get("m_typeId") in self._sys_types]
        self._mep_graph()
        for s in self._members.get(eid, []):
            if s not in out:
                out.append(s)
        return out

    def systems(self, eid) -> list:
        """Systems of an MEP element. Fittings and accessories often have none of their own: they inherit the
        systems of the elements they are connected to (connector graph, both directions, breadth first, up to
        SYSTEM_HOPS steps) - Revit shows the system of the connected network for them as well."""
        if eid in self._systems:
            return self._systems[eid]
        own = self.own_systems(eid)
        if not own:
            seen, front = {eid}, [eid]
            for _ in range(SYSTEM_HOPS):
                nxt = [n for e in front for n in sorted(self._adj.get(e, ())) if n not in seen]
                seen.update(nxt)
                found = []
                for n in nxt:
                    for sy in self.own_systems(n):
                        if sy not in found:
                            found.append(sy)
                if found:
                    own = found
                    break
                front = nxt
        self._systems[eid] = own
        return own

    def system_type(self, eid):
        """RbsHvacSystemType / RbsPipingSystemType id of an MEP element (its first system's type, else the curve's
        design system type m_pDesignPropManager.m_idSystemType), or None."""
        for sy in self.systems(eid):
            return (self.doc.objects.get(sy) or {}).get("m_typeId")
        o = self.doc.objects.get(eid) or {}
        if isinstance(o.get("m_pDesignPropManager"), dict):
            t = o["m_pDesignPropManager"].get("m_idSystemType")
            return t if t in self._sys_types else None
        return None

    def system_material(self, st):
        """Material override of a system type (m_oSystemOverrides.m_oGMaterialOverrider.m_materialId) or None.
        The system type's projection line colour is NOT a surface colour: Navisworks uses it for the centre lines
        only (nugget.nwd: pipes grey, their centre lines in the system colour); duct colours come from view filters."""
        if st is None:
            return None
        ov = ((self.doc.objects.get(st) or {}).get("m_oSystemOverrides") or {})
        mid = (ov.get("m_oGMaterialOverrider") or {}).get("m_materialId")
        return mid if mid in self.mats else None

    def with_view_override(self, mid, ov: dict):
        """Material key for material `mid` (None = default) shown with a view-filter override
        ov = {"rgb": (r, g, b) | None, "transparency": t | None, "filters": [names]} -> a derived entry in self.mats."""
        key = ("view", mid, ov.get("rgb"), ov.get("transparency"))
        if key not in self.mats:
            base = self.mats.get(mid) or {"name": "Standard", "rgb": DEFAULT_RGB, "transparency": 0.0}
            m = dict(base)
            if ov.get("rgb") is not None:
                m["rgb"] = tuple(ov["rgb"])
            if ov.get("transparency") is not None:
                m["transparency"] = ov["transparency"]
            m["view_override"] = {"base": mid, "filters": list(ov.get("filters") or []), "view": ov.get("view")}
            self.mats[key] = m
        return key

    # ---- properties -------------------------------------------------------------------------------------------
    def _label(self, pid):
        d = self.doc.params.describe(pid)
        if pid in LABELS:
            return LABELS[pid]
        if d.get("label"):
            return d["label"]
        n = d.get("name") or ""
        return None if n.startswith("builtin ") or n.startswith("param ") else n

    def _value(self, pid, v):
        """Internal value -> (display name suffix, value) or None (empty)."""
        if v is None or v == "":
            return None
        if pid in ENUMS and isinstance(v, int):
            s = ENUMS[pid].get(v, str(v))
            return ("", s) if s else None
        if pid in YES_NO and isinstance(v, int):
            return "", "Ja" if v else "Nein"
        if isinstance(v, str):
            v = v.strip()
            return ("", v) if v else None
        if pid in UNIT and isinstance(v, (int, float)):
            f, unit = UNIT[pid]
            return f" [{unit}]", _num(float(v) * f)
        return "", _num(v)

    def _param_props(self, obj, out: dict):
        for _path, cls, pid, v in collect_values(obj):
            if pid in SKIP_IDS or cls == "NamedParam":
                continue
            label = self._label(pid)
            if not label:
                continue
            if cls == "ParamValueElementId":
                if not isinstance(v, int) or v < 0:
                    continue
                v = self.doc.names.get(v) or str(v)
            r = self._value(pid, v)
            if r is None:
                continue
            out.setdefault(label + r[0], r[1])
        return out

    def asset_set(self, mid, guid):
        """PropertySetElement id of a material's structural / thermal asset, or None."""
        m = (self.doc.objects.get(mid) or {}).get("m_pMaterial") or {}
        for p in (m.get("m_assetMap") or {}).get("m_sharedAssetIdMap") or []:
            g = (((p.get("first") or {}).get("m_value") or {}).get("m_guid"))
            if g == guid and self.doc.elem_class.get(p.get("second")) == "PropertySetElement":
                return p["second"]
        return None

    def properties(self, mid) -> dict:
        """{tab: {property: value}} for one material, like Revit's material browser:
        'Material' (identity + graphics), 'Physikalisch' (structural asset), 'Thermisch' (thermal asset)."""
        if mid in self._props:
            return self._props[mid]
        vo = (self.mats.get(mid) or {}).get("view_override") if isinstance(mid, tuple) else None
        if vo is not None:                     # view-filter colour on top of a material (or the default)
            base = self.properties(vo["base"]) if vo["base"] is not None else {}
            out = {tab: dict(props) for tab, props in base.items()}
            ident = out.setdefault("Material", {"Name": "Standard"})
            m = self.mats[mid]
            r, g, b = (int(round(x * 255)) for x in m["rgb"])
            ident["Farbe in Ansicht"] = f"RGB {r} {g} {b}"
            ident["Transparenz in Ansicht"] = int(round(float(m.get("transparency") or 0.0) * 100))
            ident["Ansichtsfilter"] = ", ".join(vo["filters"])
            if vo.get("view"):
                ident["Ansicht"] = vo["view"]
            self._props[mid] = out
            return out
        o = self.doc.objects.get(mid) or {}
        m = o.get("m_pMaterial") or {}
        out: dict[str, dict] = {}
        ident = {"Name": m.get("m_name") or f"material {mid}", "Id": str(mid)}
        # 'Materialklasse' = the browser's Identity class; Revit's 'Klasse' parameter is the structural asset class
        for k, f in (("Materialklasse", "m_sMaterialClass"), ("Materialkategorie", "m_sCategory"),
                     ("Schlüsselwörter", "m_keywords")):
            if (m.get(f) or "").strip():
                ident[k] = m[f].strip()
        r, g, b = (int(round(x * 255)) for x in colorref_rgb(int(m.get("m_color") or 0)))
        ident["Farbe"] = f"RGB {r} {g} {b}"
        ident["Transparenz"] = int(round(float(m.get("m_transparency") or 0.0) * 100))
        if m.get("m_shininess") is not None:
            ident["Glanz"] = m["m_shininess"]
        if m.get("m_smoothness") is not None:
            ident["Glätte"] = int(round(float(m["m_smoothness"]) * 100))
        ident["Luminiszenz"] = "Ja" if m.get("m_glow") else "Nein"
        app = m.get("m_appearanceAssetId")
        if isinstance(app, int) and app > 0:
            an = ((self.doc.objects.get(app) or {}).get("m_pAppearanceAsset") or {}).get("m_Name")
            ident["Darstellung"] = an or str(app)
        for k, v in (("Schraffur Oberfläche", m.get("m_surfacePatternId")),
                     ("Schraffur Schnitt", m.get("m_cutPatternId"))):
            if isinstance(v, int) and v > 0:
                ident[k] = self.doc.names.get(v) or str(v)
        self._param_props({k: v for k, v in o.items() if k != "m_pMaterial"}, ident)
        out["Material"] = ident
        for tab, guid in (("Physikalisch", ASSET_STRUCTURAL), ("Thermisch", ASSET_THERMAL)):
            ps = self.asset_set(mid, guid)
            if ps is None:
                continue
            po = self.doc.objects.get(ps) or {}
            props = {"Id": str(ps)}
            self._param_props(po.get("m_oParamSet") or {}, props)
            props.setdefault("Name", props.get("Eigenschaftssatzname") or self.doc.names.get(ps) or str(ps))
            out[tab] = props
        self._props[mid] = out
        return out
