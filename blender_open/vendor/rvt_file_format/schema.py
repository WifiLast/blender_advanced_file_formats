"""Parser for the Revit class schema stream (Formats/Latest, decoded).

Grammar (reverse-engineered from Revit 2023 files, see RVT_FORMAT.md §10):

  schema   := record*                               (until end of data)
  record   := u16 lead, u16 nameLen, name[nameLen],
              u16 parentRef [record if parentRef & 0x8000],
              u32 version, u32 nFields, field[nFields], u32 trailer
  field    := u32 nameLen, name[nameLen], type
  type     := u32 code                              base = code & 0xFF,
                                                    ptr  = (code >> 8) & 0xF,
                                                    coll = (code >> 12) & 0xF
              [u32 count          if coll == 1]     (fixed-size array)
              [u16 classRef [record if & 0x8000]
                                  if base == 0x0E and ptr == 0]
              [u32 one, u8 0x20, type
                                  if base == 0x0D]  (nested element type)

Class ids are assigned in order of first appearance, starting at FIRST_ID; an
inline definition (ref | 0x8000) must carry exactly the next id.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field as dfield

FIRST_ID = 0x0C

# base type codes 1..12; names partly inferred from field names (see RVT_FORMAT.md)
BASE_NAMES = {
    0x01: "bool?", 0x02: "int8?", 0x03: "int16?", 0x04: "int32", 0x05: "uint32?",
    0x06: "float", 0x07: "double", 0x08: "string", 0x09: "base9?", 0x0A: "base10?",
    0x0B: "base11?", 0x0C: "base12?", 0x0D: "nested", 0x0E: "class",
}
COLL_NAMES = {0: "", 1: "array", 5: "coll5", 6: "coll6"}


@dataclass
class FieldType:
    code: int
    count: int | None = None        # fixed array length
    class_id: int | None = None     # referenced class (0 = unspecified)
    inner: "FieldType | None" = None

    @property
    def base(self): return self.code & 0xFF
    @property
    def ptr(self): return (self.code >> 8) & 0xF
    @property
    def coll(self): return (self.code >> 12) & 0xF

    def describe(self, classes) -> str:
        if self.base == 0x0E:
            t = classes[self.class_id].name if self.class_id in classes else (
                "?" if not self.class_id else f"class#{self.class_id:#x}")
        elif self.base == 0x0D and self.inner is not None:
            t = self.inner.describe(classes)
        else:
            t = BASE_NAMES.get(self.base, f"base{self.base:#x}")
        if self.ptr:
            t += f"*{self.ptr}"
        if self.coll:
            t = f"{COLL_NAMES.get(self.coll, f'coll{self.coll}')}<{t}>"
            if self.count is not None:
                t += f"[{self.count}]"
        return t


@dataclass
class Field:
    name: str
    type: FieldType


@dataclass
class ClassDef:
    id: int
    name: str
    parent_id: int
    version: int
    fields: list = dfield(default_factory=list)
    offset: int = 0
    lead: int = 0
    trailer: int = 0
    upgrades: list = dfield(default_factory=list)


class SchemaError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0
        self.classes: dict[int, ClassDef] = {}
        self.next_id = FIRST_ID

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v

    def u16(self):
        v = struct.unpack_from("<H", self.d, self.p)[0]; self.p += 2; return v

    def u32(self):
        v = struct.unpack_from("<I", self.d, self.p)[0]; self.p += 4; return v

    def text(self, n):
        s = self.d[self.p:self.p + n]
        if len(s) != n:
            raise SchemaError(f"truncated string at {self.p:#x}")
        self.p += n
        return s.decode("latin-1")

    def ref(self) -> int:
        r = self.u16()
        if r & 0x8000:
            cid = r & 0x7FFF
            self.record(expect_id=cid)
            return cid
        return r

    def record(self, expect_id: int | None = None) -> ClassDef:
        start = self.p
        cid = self.next_id
        if expect_id is not None and expect_id != cid:
            raise SchemaError(f"inline class id {expect_id:#x} != next id {cid:#x} at {start:#x}")
        self.next_id += 1
        lead = self.u16()
        n = self.u16()
        if not 0 < n < 512:
            raise SchemaError(f"bad class name length {n} at {start:#x}")
        name = self.text(n)
        c = ClassDef(cid, name, 0, 0, offset=start, lead=lead)
        self.classes[cid] = c          # register before parent/fields (recursive refs)
        c.parent_id = self.ref()
        c.version = self.u32()
        nf = self.u32()
        if nf > 4096:
            raise SchemaError(f"implausible field count {nf} in {name} at {start:#x}")
        for _ in range(nf):
            fl = self.u32()
            if not 0 < fl < 512:
                raise SchemaError(f"bad field name length {fl} in {name} at {self.p - 4:#x}")
            fname = self.text(fl)
            c.fields.append(Field(fname, self.ftype()))
        # trailer: u32 count + count x 16-byte GUIDs (upgrade ids, inferred)
        c.trailer = self.u32()
        if c.trailer > 100000 or self.p + 16 * c.trailer > len(self.d):
            raise SchemaError(f"implausible trailer count {c.trailer} in {name} at {self.p - 4:#x}")
        c.upgrades = [self.d[self.p + 16 * i: self.p + 16 * (i + 1)].hex() for i in range(c.trailer)]
        self.p += 16 * c.trailer
        return c

    def ftype(self) -> FieldType:
        t = FieldType(self.u32())
        if t.coll == 1:
            t.count = self.u32()
        if t.base == 0x0E and t.ptr == 0:
            t.class_id = self.ref()
        elif t.base == 0x0D:
            one = self.u32(); tag = self.u8()
            if (one, tag) != (1, 0x20):
                raise SchemaError(f"nested type marker {one},{tag:#x} at {self.p - 5:#x}")
            t.inner = self.ftype()
            # like ADDEntry::IsEmbeddedObject: follow the nested chain; if the
            # innermost type is an embedded class, the outer type repeats the ref
            leaf = t.inner
            while leaf.base == 0x0D and leaf.inner is not None:
                leaf = leaf.inner
            if t.ptr == 0 and leaf.base == 0x0E and leaf.ptr == 0:
                t.class_id = self.ref()
        return t


def parse(data: bytes) -> dict[int, ClassDef]:
    """Parse a decoded Formats/Latest stream -> {class_id: ClassDef}."""
    r = _Reader(data)
    while r.p < len(data):
        # trailing zero padding after the last record
        if not any(data[r.p:]):
            break
        r.record()
    return r.classes
