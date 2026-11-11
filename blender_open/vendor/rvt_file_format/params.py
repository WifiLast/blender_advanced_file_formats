"""Element parameters.

Stored values: objects of class ParamValue{Double,Int,AString,ElementId} anywhere in an element's
decoded record ({m_paramId, m_value}). Names:
  m_paramId < 0  -> BuiltInParameter enum name (data/builtin_parameters.json, from RevitAPI.dll)
  m_paramId > 0  -> ParamElem* element -> m_pParamDef.m_caption (+ m_typeId = data type/spec)
Values are in Revit internal units (feet, square feet, cubic feet, radians) unless converted.
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA = Path(__file__).with_name("data")
VALUE_CLASSES = {"ParamValueDouble": "double", "ParamValueInt": "int",
                 "ParamValueAString": "string", "ParamValueElementId": "elementId"}

FT = 0.3048
# spec id fragments -> (factor to SI, unit)
SPEC_UNITS = [("spec.aec:length", FT, "m"), ("spec.aec:area", FT * FT, "m2"),
              ("spec.aec:volume", FT ** 3, "m3"), ("spec.aec:angle", 57.29577951308232, "deg"),
              ("spec.aec.hvac:airFlow", FT ** 3 * 1000, "l/s"),
              ("spec.aec.hvac:ductSize", FT, "m"), ("spec.aec.piping:pipeSize", FT, "m")]


def ui_labels():
    """Revit UI labels + parameter groups (data/param_groups.json, from tools/revit_groundtruth.py on a German
    Revit 2023): {"builtin_label": {id: label}, "builtin_group": {id: groupId}, "group_labels": {groupId: label}}."""
    try:
        d = json.loads((_DATA / "param_groups.json").read_text(encoding="utf-8"))
    except OSError:
        return {}, {}, {}
    return ({int(k): v for k, v in d.get("builtin_label", {}).items()},
            {int(k): v for k, v in d.get("builtin_group", {}).items()}, d.get("group_labels", {}))


def builtin_names():
    try:
        return {int(k): v for k, v in json.loads((_DATA / "builtin_parameters.json").read_text()).items()}
    except OSError:
        return {}


def known_builtin_storage():
    """Storage type per built-in parameter id, learned once from full decodes (tools/make_builtin_storage.py): the
    reader decodes records lazily, so it cannot learn these from every record of the file any more."""
    try:
        return {int(k): v for k, v in json.loads((_DATA / "builtin_storage.json").read_text()).items()}
    except OSError:
        return {}


def _type_id(td):
    """ForgeTypeId -> its string (field layout varies; take the first string found)."""
    if isinstance(td, str):
        return td
    if isinstance(td, dict):
        for k, v in td.items():
            if not k.startswith("__"):
                s = _type_id(v)
                if s:
                    return s
    return None


# storage type by ParamDef class ancestry (first match walking up from the class itself)
STORAGE_BY_ANCESTOR = [
    ("ParamDefMaterialBrowse", "elementId"), ("ParamDefMaterial", "elementId"),
    ("ParamDefFillPattern", "elementId"),
    ("ParamDefComboIntMapResource", "int"), ("ParamDefComboString", "string"),
    ("ParamDefYesNo", "int"), ("ParamDefInt", "int"), ("ParamDefColor", "int"),
    ("ParamDefString", "string"),
    ("ParamDefLength", "double"), ("ParamDefArea", "double"), ("ParamDefVolume", "double"),
    ("ParamDefAngle", "double"), ("ParamDefDouble", "double"), ("ParamDefValue", "double"),
    ("ParamDefElementId", "elementId"), ("ParamDefCombo", "elementId"),
]
LENGTH_TOKENS = ("WIDTH", "HEIGHT", "LENGTH", "THICKNESS", "OFFSET", "ELEVATION", "DEPTH",
                 "RADIUS", "DIAMETER", "DIST", "SILL", "HEAD")
SPEC_BY_ANCESTOR = {"ParamDefLength": "autodesk.spec.aec:length", "ParamDefArea": "autodesk.spec.aec:area",
                    "ParamDefVolume": "autodesk.spec.aec:volume", "ParamDefAngle": "autodesk.spec.aec:angle"}


class ParamTable:
    def __init__(self, classes: dict | None = None):
        self.builtin = builtin_names()
        self.labels, self.groups, self.group_labels = ui_labels()
        self.defs: dict[int, dict] = {}     # paramElem id -> {name, kind, spec, storage, ...}
        self.builtin_storage: dict[int, str] = known_builtin_storage()
        self._byname = {c.name: c for c in (classes or {}).values()}
        self._cl = classes or {}

    def _ancestors(self, name):
        out, c = [], self._byname.get(name)
        while c is not None:
            out.append(c.name)
            c = self._cl.get(c.parent_id)
        return out

    def add_param_elem(self, eid: int, cls: str, obj: dict):
        pd = obj.get("m_pParamDef") or {}
        dc = pd.get("__class__")
        anc = self._ancestors(dc) if dc else []
        storage = next((s for a, s in STORAGE_BY_ANCESTOR if a in anc), None)
        spec = _type_id(pd.get("m_specTypeId")) or next((SPEC_BY_ANCESTOR[a] for a in anc
                                                         if a in SPEC_BY_ANCESTOR), None)
        self.defs[eid] = {"name": pd.get("m_caption") or f"param {eid}",
                          "kind": cls.replace("ParamElem", "") or "Param",
                          "spec": spec, "storage": storage,
                          "identity": _type_id(pd.get("m_typeId")),
                          "def_class": dc}

    def learn_builtin_types(self, obj):
        """Typed ParamValue* occurrences tell the storage type of built-in ids."""
        for _path, cls, pid, _val in collect_values(obj):
            if pid < 0 and cls in VALUE_CLASSES:
                self.builtin_storage.setdefault(pid, VALUE_CLASSES[cls])

    def describe(self, pid: int) -> dict:
        if pid < 0:
            name = self.builtin.get(pid, f"builtin {pid}")
            st = self.builtin_storage.get(pid)
            spec = None
            if st in (None, "double") and any(t in name for t in LENGTH_TOKENS) and "COUNT" not in name:
                spec = "length (inferred from name)"
            g = self.groups.get(pid)
            return {"name": name, "kind": "BuiltIn", "spec": spec, "storage": st, "label": self.labels.get(pid),
                    "group": g, "group_label": self.group_labels.get(g) if g else None}
        return self.defs.get(pid, {"name": f"param {pid}", "kind": "unknown", "spec": None})

    @staticmethod
    def to_si(value, spec):
        if not isinstance(value, float) or not spec:
            return None
        if spec.startswith("length (inferred"):
            return value * FT, "m"
        for frag, f, unit in SPEC_UNITS:
            if frag in spec:
                return value * f, unit
        return None


SLOT = {"string": "m_str", "elementId": "m_elemId", "int": "m_int", "double": "m_value"}


def _named_value(np_obj: dict, storage: str | None):
    """NamedParam has a slot per storage type; pick by the definition's storage type.
    -> (type, value, extra fields)"""
    extra = {"instance": bool(np_obj.get("m_instance")), "reporting": bool(np_obj.get("m_reporting"))}
    if np_obj.get("m_oExpression"):
        extra["formula"] = True
    if storage in SLOT:
        v = np_obj.get(SLOT[storage])
        if v in (None, "", -1, 0, 0.0):          # declared slot empty: another slot may hold it
            if np_obj.get("m_elemId") not in (None, -1):
                return "elementId", np_obj["m_elemId"], extra
            if np_obj.get("m_str"):
                return "string", np_obj["m_str"], extra
        return storage, v, extra
    for st, slot in (("string", "m_str"), ("elementId", "m_elemId"), ("double", "m_value"), ("int", "m_int")):
        v = np_obj.get(slot)
        if v not in (None, "", -1, 0, 0.0):
            extra["storage_guessed"] = True
            return st, v, extra
    # all slots default: report every slot
    vals = {k: np_obj.get(k) for k in ("m_value", "m_int", "m_str", "m_elemId")}
    vals = {k: v for k, v in vals.items() if v not in (None, 0, 0.0, "", -1)}
    extra["slots"] = vals
    return "unknown", np_obj.get("m_value"), extra


def collect_values(obj, out=None, path=""):
    """All ParamValue* objects in a decoded record -> list of (path, class, paramId, value)."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        c = obj.get("__class__")
        if c in VALUE_CLASSES and "m_paramId" in obj:
            out.append((path, c, obj["m_paramId"], obj.get("m_value")))
            return out
        if c == "NamedParam" and "m_paramId" in obj:      # family parameters (FamilyParams)
            out.append((path, c, obj["m_paramId"], obj))
            return out
        for k, v in obj.items():
            if not k.startswith("__"):
                collect_values(v, out, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for v in obj:
            collect_values(v, out, path)
    return out


def element_parameters(obj: dict, table: ParamTable, names: dict | None = None) -> list[dict]:
    """Parameters of one element as a list of dicts (name, id, value, unit, source)."""
    res = []
    for path, cls, pid, val in collect_values(obj):
        d = table.describe(pid)
        extra = {}
        if cls == "NamedParam":
            vtype, val, extra = _named_value(val, d.get("storage"))
        else:
            vtype = VALUE_CLASSES[cls]
        row = {"name": d["name"], "id": pid, "kind": d["kind"], "type": vtype,
               "value": val, "source": path.split(".")[0], **extra}
        if d.get("label"):
            row["label"] = d["label"]
        if d.get("group_label"):
            row["group"] = d["group_label"]
        if d.get("spec"):
            row["spec"] = d["spec"]
        si = table.to_si(val, d.get("spec"))
        if si:
            row["value_si"], row["unit"] = round(si[0], 9), si[1]
        if cls == "ParamValueElementId" and names and val in names:
            row["value_name"] = names[val]
        res.append(row)
    return res
