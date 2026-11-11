"""One decoding pass over a .rvt: every element record, header and GElement.

Shared by the GLB export (model.py) and the element/parameter export (elements.py).
Memory: keeps every decoded record (about 2.5-3 GB for a 40 MB project file).
"""
from __future__ import annotations

import re

from . import records, schema
from .container import RvtFile
from .params import ParamTable, element_parameters

TYPE_FIELD = re.compile(r"^m_\w*(AttributesId|SymbolId)$")
NAME_PATHS = (("m_symbolInfo", "m_name"), ("m_pMaterial", "m_name"), ("m_name",),
              ("m_pCategory", "m_name"), ("m_pParamDef", "m_caption"), ("m_pFont", "m_name"))


def element_name(o: dict):
    for path in NAME_PATHS:
        v = o
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, str) and v:
            return v
    return None


def type_id(o: dict):
    """Id of the element's type (wall type, family symbol, ...) or None."""
    for k, v in o.items():
        if isinstance(v, int) and v > 0 and TYPE_FIELD.match(k):
            return v
    return None


class LazyMap:
    """eid -> decoded value, decoded from the raw record bytes on first access (a Golden Nugget .rvt has 390k
    element records, of which an export needs a few thousand)."""

    def __init__(self, raw: dict, fn):
        self.raw, self.cache, self._fn = raw, {}, fn

    def __getitem__(self, k):
        try:
            return self.cache[k]
        except KeyError:
            pass
        v = self.cache[k] = self._fn(k, self.raw[k])        # KeyError for unknown ids, as a dict
        return v

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def __contains__(self, k):
        return k in self.raw

    def __len__(self):
        return len(self.raw)

    def __iter__(self):
        return iter(self.raw)

    def keys(self):
        return self.raw.keys()

    def items(self):                                        # decodes everything
        return ((k, self[k]) for k in self.raw)

    def values(self):
        return (self[k] for k in self.raw)


class _Names:
    """eid -> element name; only ids that have one (like the dict this replaces)."""

    def __init__(self, objects: LazyMap):
        self._o, self._c = objects, {}

    def _name(self, k):
        if k not in self._c:
            o = self._o.get(k)
            self._c[k] = element_name(o) if o is not None else None
        return self._c[k]

    def get(self, k, default=None):
        return self._name(k) or default

    def __getitem__(self, k):
        n = self._name(k)
        if not n:
            raise KeyError(k)
        return n

    def __contains__(self, k):
        return bool(self._name(k))

    def __iter__(self):
        return (k for k in self._o if self._name(k))

    def __bool__(self):
        # `if names:` must not fall back to __len__ (which decodes every record to count the named ones)
        return len(self._o.raw) > 0

    def __len__(self):
        return sum(1 for _ in self)                          # decodes every record: avoid

    def items(self):
        return ((k, self._name(k)) for k in self._o if self._name(k))


class Document:
    """Lazy: the scan keeps the raw record bytes (and decodes only the parameter definitions); records are decoded
    when first asked for. Use decode_all() when every record's errors/values are needed."""

    def __init__(self, path: str, log=None, keep_objects: bool = True):
        log = log or (lambda *a: None)
        self.path = path
        self.elem_class: dict[int, str] = {}
        self.errors: dict[int, str] = {}
        raw_obj: dict[int, bytes] = {}
        raw_gel: dict[int, bytes] = {}
        raw_hdr: dict[int, bytes] = {}
        with RvtFile(path) as r:
            self.classes = schema.parse(r.decoded("Formats/Latest"))
            self.params = ParamTable(self.classes)
            byname = {c.name: c.id for c in self.classes.values()}
            hcid, gid = byname["ElementHeader"], byname["GElement"]
            self._it = records.Interpreter(self.classes)
            self.objects = LazyMap(raw_obj, self._decode_obj)
            self.gel = LazyMap(raw_gel, self._decode_gel)                 # eid -> (GElement, objects table)
            self.category = LazyMap(raw_hdr, self._decode_hdr)            # eid -> category id or None
            self.names = _Names(self.objects)
            for p in [s for s in r.streams() if s.startswith("Partitions/")]:
                d = b"".join(m.data for m in r.members(p))
                log(f"  {p}: {len(d) / 1e6:.1f} MB")
                for _off, eid, pl, is_h in records.walk_stream(d, hcid):
                    if is_h:
                        raw_hdr[eid] = pl
                        self.category.cache.pop(eid, None)
                        continue
                    cid = int.from_bytes(pl[:2], "little")
                    if cid not in self.classes:
                        continue
                    if cid == gid:
                        raw_gel[eid] = pl
                        self.gel.cache.pop(eid, None)
                        continue
                    cls = self.classes[cid].name
                    if cls == "SerializedDummy":
                        continue
                    self.elem_class[eid] = cls
                    self.objects.cache.pop(eid, None)
                    self.names._c.pop(eid, None)
                    if keep_objects:
                        raw_obj[eid] = pl
                    if cls.startswith("ParamElem"):              # parameter definitions are needed up front
                        o = self._decode_obj(eid, pl)
                        if keep_objects:
                            self.objects.cache[eid] = o
                        self.params.add_param_elem(eid, cls, o)

    # -- lazy decoders ----------------------------------------------------------

    def _decode_obj(self, eid, pl):
        try:
            o = self._it.decode(pl)
        except records.DecodeError as e:
            o = getattr(e, "partial", {})
            self.errors[eid] = str(e)
        self.params.learn_builtin_types(o)
        return o

    def _decode_gel(self, eid, pl):
        try:
            return (self._it.decode(pl), self._it.objects)
        except records.DecodeError as e:
            self.errors[eid] = f"GElement: {e}"
            raise KeyError(eid) from None                         # as if the record were absent

    def _decode_hdr(self, eid, pl):
        try:
            h = self._it.decode(pl)
        except records.DecodeError as e:                          # header tail is custom-written
            h = getattr(e, "partial", {})
        c = h.get("m_categroryId")
        return c if isinstance(c, int) else None

    def decode_all(self):
        """Decode every record (fills `errors` completely, learns every built-in storage type)."""
        for k in list(self.objects.raw):
            self.objects.get(k)
        for k in list(self.gel.raw):
            self.gel.get(k)

    def parameters(self, eid: int) -> list[dict]:
        o = self.objects.get(eid)
        return element_parameters(o, self.params, self.names) if o else []

    def type_of(self, eid: int):
        """Element type id. Three storage forms (Golden Nugget sample, _work probe 2026-11-10):
        - m_*AttributesId / m_*SymbolId fields (walls, floors, some family instances),
        - m_idType (MEP curves: RbsPipeCurve -> RbsPipeType, RbsDuctCurve -> AbsDuctType, insulation, cable trays),
        - family instances without a type field: the symbol is only referenced from their own geometry, the
          top-level GInstance's m_instanceInfo.m_symbolId (-> FamilySymbol).
        Variant symbols (unnamed, e.g. for mirrored instances) carry m_masterId = the real type; it is followed
        (Golden Nugget: 757 variants, checked against the Revit API dump)."""
        t = self._raw_type_of(eid)
        for _ in range(4):
            m = (self.objects.get(t) or {}).get("m_masterId") if t else None
            if not (isinstance(m, int) and m > 0 and m in self.elem_class):
                break
            t = m
        return t

    def _raw_type_of(self, eid: int):
        o = self.objects.get(eid)
        if not o:
            return None
        t = type_id(o)
        if t:
            return t
        t = o.get("m_idType")
        if isinstance(t, int) and t > 0 and t in self.elem_class:
            return t
        g = self.gel.get(eid)
        if g:
            from .geometry import find_nodes        # noqa: PLC0415
            for gi in find_nodes(g[0], ("GInstance",)):
                sym = (gi.get("m_instanceInfo") or {}).get("m_symbolId")
                if self.elem_class.get(sym) == "FamilySymbol":
                    return sym
        return None

    def derived_parameters(self, eid: int) -> list[dict]:
        """Built-in parameters Revit fills from element fields, not from stored ParamValue objects (the most-missed
        ones in tools/compare_groundtruth.py). Rows like params.element_parameters: raw value = what the Revit API
        returns (element id), value_name = what Revit displays."""
        o = self.objects.get(eid) or {}
        rows = []

        def add(pid, value, shown):
            dsc = self.params.describe(pid)
            row = {"name": dsc["name"], "id": pid, "kind": "BuiltIn", "type": "elementId", "value": value,
                   "value_name": shown, "source": "derived"}
            if dsc.get("label"):
                row["label"] = dsc["label"]
            if dsc.get("group_label"):
                row["group"] = dsc["group_label"]
            rows.append(row)

        def ename(x, none_text):
            return none_text if not isinstance(x, int) or x < 0 else (self.names.get(x) or str(x))

        for field, pid in (("m_createdPhaseId", -1012100), ("m_demolishedPhaseId", -1012101)):
            if field in o:
                add(pid, o[field], ename(o[field], "Keine"))
        if "m_designOptionId" in o:
            add(-1013201, o["m_designOptionId"], ename(o["m_designOptionId"], "Hauptmodell"))
        cat = self.category.get(eid)
        if isinstance(cat, int):
            add(-1140362, cat, None)
        t = self.type_of(eid)
        if t:
            fam = self.family_of(t) or {}
            tname = self.names.get(t)
            add(-1002000, t, tname)
            add(-1002050, t, tname)
            add(-1002051, t, fam.get("name"))
            add(-1002052, t, f"{fam['name']}: {tname}" if fam.get("name") and tname else tname)
        st = (o.get("m_pDesignPropManager") or {}).get("m_idSystemType")
        if isinstance(st, int) and st > 0:
            add(-1140334, st, self.names.get(st))
        return rows

    def family_of(self, type_eid: int) -> dict | None:
        """Family of an element type -> {"id", "name", "system"}.

        Loadable families: FamilySymbol.m_familyId -> Family element, name = Family.m_name (Golden Nugget: all
        1,280 used symbols resolve to a named Family). NOT m_famId: that is the family document the element was
        loaded into (nested symbols point at their host family; project symbols have -1). Fallbacks: the .rfa file
        name in m_path, else "Family <id>". System families (walls, floors, pipes, ...): no Family element; the
        name comes from the type's class (BasicWallType -> "Basic Wall")."""
        o = self.objects.get(type_eid)
        if not o:
            return None
        fid = o.get("m_familyId")
        if not (isinstance(fid, int) and self.elem_class.get(fid) == "Family"):
            fid = o.get("m_famId")
        if self.elem_class.get(type_eid) == "FamilySymbol":
            if not (isinstance(fid, int) and self.elem_class.get(fid) == "Family"):
                return None
            fo = self.objects.get(fid) or {}
            name = fo.get("m_name") or ""
            path = (fo.get("m_path") or "").replace("\\", "/")
            if not name and path.lower().endswith(".rfa"):
                name = path.rsplit("/", 1)[-1][:-4]
            return {"id": fid, "name": name or f"Family {fid}", "system": False}
        return {"id": None, "name": system_family_name(self.elem_class.get(type_eid, "")), "system": True}


def system_family_name(type_class: str) -> str:
    """BasicWallType -> 'Basic Wall', FloorAttributes -> 'Floor', RbsPipeType -> 'Pipe'."""
    n = re.sub(r"(Attributes|Type)$", "", type_class) or type_class
    n = re.sub(r"^(Rbs|Abs)", "", n)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n) or "System Family"
