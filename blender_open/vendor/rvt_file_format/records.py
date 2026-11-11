"""Element records in Partitions/<n> (stage 3) - schema-driven interpreter.

Framing of a decoded partition member:
  record := u32 elementId, u32 hash, u32 len, payload[len], u32 len
  payload := u16 classId, fields of the class chain (base class first)

Value encoding (calibrated on FontElem, refined against "payload consumed exactly"):
  int32 / ElementId (Identifier.m_id)   4 bytes
  bool (base 0x01)                      1 byte
  double 8, float 4
  string (0x6008)                       u32 nChars + UTF-16LE
  pointer kind 1/2                      u32 0 = null | u32 0xFFFFFFFF, u16 classId, object
  pointer kind 3                        u32 reference (not inline)
  coll5 / coll6                         u32 count + items
  fixed array                           count items, no prefix
"""
from __future__ import annotations

import struct

from .schema import ClassDef, FieldType

PRESENT = 0xFFFFFFFF


class DecodeError(ValueError):
    pass


def _framed(m: bytes, off: int, hdr: int):
    """Payload end if a record with a `hdr`-byte header starts at off, else None."""
    if off + hdr + 4 > len(m):
        return None
    ln = struct.unpack_from("<I", m, off + hdr - 4)[0]
    end = off + hdr + ln
    if ln < 2 or end + 4 > len(m) or struct.unpack_from("<I", m, end)[0] != ln:
        return None
    return end


def walk_member(m: bytes):
    """Yield (offset, elementId, payload) for the records of one decoded member.

    Two framings:
      element record:   u32 id, u32 hash, u32 len, payload, u32 len
      element header:   u32 id, u32 len, payload (ElementHeader), u32 len
    """
    off = 0
    while off + 8 <= len(m):
        eid = struct.unpack_from("<I", m, off)[0]
        end = _framed(m, off, 12)
        hdr = 12
        if end is None:
            end = _framed(m, off, 8)
            hdr = 8
        if end is None:
            return
        yield off, eid, m[off + hdr:end]
        off = end + 4


END_MARKERS = (16, 12)      # ff ff ff ff + flag + zeros at the end of a run of records


def walk_stream(d: bytes, header_cid: int, stats: dict | None = None):
    """Yield (offset, elementId, payload, is_header) over the concatenated members of a
    partition (records cross member boundaries). header_cid = class id of ElementHeader;
    an 8-byte-header record is preferred when its payload starts with it, because the
    12-byte length check can match by accident."""
    hc = struct.pack("<H", header_cid)
    off, n = 0, len(d)

    def starts_record(pos):
        if pos == n:
            return True
        e8 = _framed(d, pos, 8)
        return (e8 is not None and d[pos + 8:pos + 10] == hc) or _framed(d, pos, 12) is not None

    while off + 8 <= n:
        e8 = _framed(d, off, 8)
        if e8 is not None and d[off + 8:off + 10] == hc:
            yield off, struct.unpack_from("<I", d, off)[0], d[off + 8:e8], True
            off = e8 + 4
            continue
        e12 = _framed(d, off, 12)
        if e12 is not None:
            yield off, struct.unpack_from("<I", d, off)[0], d[off + 12:e12], False
            off = e12 + 4
            continue
        if d[off:off + 4] == b"\xff\xff\xff\xff":
            for skip in END_MARKERS:
                if starts_record(off + skip):
                    off += skip
                    break
            else:
                break
            continue
        break
    if stats is not None:
        stats["stopped_at"] = off
        stats["length"] = n


def member_tiles(m: bytes) -> bool:
    last = 0
    for off, _eid, pl in walk_member(m):
        last = off + len(pl) + 4 + (12 if struct.unpack_from("<I", m, off + 8)[0] == len(pl) else 8)
    return last == len(m)


# primitive sizes by base type (inferred; refined by validation)
PRIM = {0x01: ("<B", 1), 0x02: ("<b", 1), 0x03: ("<h", 2), 0x04: ("<i", 4), 0x05: ("<I", 4),
        0x06: ("<f", 4), 0x07: ("<d", 8)}


class Interpreter:
    def __init__(self, classes: dict[int, ClassDef]):
        self.cl = classes
        self._chain = {}
        self.trace = None       # set to [] to record (start, end, field, value)

    def chain(self, cid: int) -> list[ClassDef]:
        if cid not in self._chain:
            seq, c = [], self.cl.get(cid)
            while c is not None:
                seq.insert(0, c)
                c = self.cl.get(c.parent_id)
            self._chain[cid] = seq
        return self._chain[cid]

    # -- public -------------------------------------------------------------

    def decode(self, payload: bytes) -> dict:
        """Decode one record payload; raises DecodeError unless it is consumed exactly."""
        self.d, self.p = payload, 0
        self.pending = []           # pointees are written after their owner (deferred)
        self.seen = set()           # object indices already introduced in this record
        self.objects = {}           # tag -> object (filled in place), for ptr-kind-3 refs
        obj = self.object()
        i = 0
        try:
            while i < len(self.pending):        # FIFO; pointees may queue more pointees
                slot, cid = self.pending[i]
                slot.update(self.object(cid))
                i += 1
        except DecodeError as e:
            # the root object's own fields were all read before the pointees: keep them (design option, phase, ...)
            # instead of the failing pointee's fragment, e.g. 208 Golden Nugget family instances with an unknown
            # ExtensibleStorage blob class
            e.partial = obj
            raise
        if self.p != len(payload):
            raise DecodeError(f"{obj.get('__class__')}: consumed {self.p} of {len(payload)}")
        return obj

    # -- reading ------------------------------------------------------------

    def need(self, n):
        if self.p + n > len(self.d):
            raise DecodeError(f"read past end at {self.p} (+{n})")

    def unpack(self, fmt, n):
        self.need(n)
        v = struct.unpack_from(fmt, self.d, self.p)[0]
        self.p += n
        return v

    def string(self) -> str:
        n = self.unpack("<I", 4)
        self.need(2 * n)
        s = self.d[self.p:self.p + 2 * n].decode("utf-16le", "replace")
        self.p += 2 * n
        return s

    def object(self, cid: int | None = None) -> dict:
        if cid is None:
            cid = self.unpack("<H", 2)
        if cid not in self.cl:
            raise DecodeError(f"unknown class id {cid:#x} at {self.p - 2}")
        out = {"__class__": self.cl[cid].name}
        for c in self.chain(cid):
            for f in c.fields:
                try:
                    start = self.p
                    out[f.name] = self.value(f.type)
                    if self.trace is not None:
                        v = out[f.name]
                        self.trace.append((start, self.p, f"{c.name}.{f.name}",
                                           v if not isinstance(v, (dict, list)) else type(v).__name__))
                except DecodeError as e:
                    err = DecodeError(f"{c.name}.{f.name}: {e}")
                    err.partial = out          # outermost object wins (set last)
                    raise err from None
        return out

    def value(self, t: FieldType):
        if t.coll == 6 and t.base == 0x08 and not t.ptr:   # 0x6008 = the string type itself
            return self.string()
        if t.coll in (5, 6):
            n = self.unpack("<I", 4)
            if n > 10_000_000:
                raise DecodeError(f"implausible count {n}")
            return [self.single(t) for _ in range(n)]
        if t.coll == 1:
            return [self.single(t) for _ in range(t.count)]
        if t.coll:
            raise DecodeError(f"unhandled collection kind {t.coll}")
        return self.single(t)

    def single(self, t: FieldType):
        if t.ptr in (1, 2, 4):
            tag = self.unpack("<I", 4)
            if tag == 0:
                return None
            if tag != PRESENT and tag in self.seen:     # back-reference to an earlier object
                return {"__backref__": tag}
            cid = self.unpack("<H", 2)
            if tag != PRESENT:
                self.seen.add(tag)
            if cid not in self.cl:
                raise DecodeError(f"unknown pointee class {cid:#x} at {self.p - 2}")
            slot = {"__class__": self.cl[cid].name, "__tag__": tag}
            if tag != PRESENT:
                self.objects[tag] = slot
            self.pending.append((slot, cid))
            return slot
        if t.ptr == 3:
            return {"__ref__": self.unpack("<I", 4)}
        if t.ptr:
            raise DecodeError(f"unhandled pointer kind {t.ptr}")
        b = t.base
        if b == 0x0D:
            return self.value(t.inner)
        if b == 0x0E:
            c = self.cl.get(t.class_id)
            if c is not None and c.name == "ElementId":
                return self.unpack("<i", 4)
            return self.object(t.class_id)
        if b == 0x08:
            return self.string()
        if b == 0x0A:                       # class reference (ClassDefinitionRef.m_ref): u16 class + u32
            cid = self.unpack("<H", 2)
            extra = self.unpack("<I", 4)
            return {"__classref__": self.cl[cid].name if cid in self.cl else cid, "extra": extra}
        if b == 0x09:                       # GUID (only used by GUIDvalue.m_guid)
            self.need(16)
            g = self.d[self.p:self.p + 16]
            self.p += 16
            return g.hex()
        if b in PRIM:
            fmt, n = PRIM[b]
            return self.unpack(fmt, n)
        raise DecodeError(f"unhandled base type {b:#x}")
