#!/usr/bin/env python3
"""nwd_fragments.py - decode / encode the NWD `LcOpNwdFragments` chunk (version 448 layout).

Independent implementation from (a) a real file (external_other/2024_12_19_IBN1 ...nwd) and
(b) layout facts learnt from the nwd-reader reference (no code copied; PolyForm Noncommercial).

Wire format (after the chunk's zlib envelope is removed):
  u32 count                              number of top-level shape objects
  count x object
object   = u32 id                        0 -> null reference
           id == 1 -> u32 refId          back reference to an earlier object
           else   -> u32 type, fields    new object with wire id `id` (>= 100)
types:  13 shape   f64 precision, u32 primitive_count, u32 bits, u32 flags, u32 path,
                   object transform, object appearance, [aux transform if flags&0x20000080==0x20000080],
                   object geometry_ref(type 93)
        14/15/16/17 run transform: 3 / 7 / 8 f64   (17: 13 f32 + 3 f64)
        54 material 14 f32 (ambient, diffuse, specular, emissive RGB, shininess, transparency)   55 appearance (flags + optional refs/u32)
        93 geometry reference: 6 f32 bounds, 3 f64 origin, f32 tolerance, u32 checksum, u32 geometry id (1-based)
All scalars are aligned to their size (u32/f32 -> 4, f64/u64 -> 8) relative to the chunk start.
"""
from __future__ import annotations

import struct
import sys

VERSION = 448
MATERIAL_FLOATS = 14


def material_floats(r: float, g: float, b: float, transparency: float = 0.0) -> list[float]:
    """Type-54 material record the way Navisworks writes it: ambient = 0.2 x diffuse, no specular, no emissive,
    shininess 0, transparency 0..1 unclamped. Verified on Navisworks-written files: nugget.nwd (its own Revit
    export, 28 materials, all 0.2 ambient / 0 specular / 0 shininess, transparency 1.0 on pipe insulation) and
    test1.nwd (0.2 ambient, shininess 0.2). Revit's m_shininess / m_smoothness are not used by Navisworks."""
    t = min(1.0, max(0.0, float(transparency)))
    return [r * 0.2, g * 0.2, b * 0.2, r, g, b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t]


class Cur:
    def __init__(self, d: bytes, pos: int = 0):
        self.d, self.p = d, pos

    def align(self, n):
        self.p = (self.p + n - 1) & ~(n - 1)

    def u32(self):
        self.align(4)
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def f32(self):
        self.align(4)
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def f64(self):
        self.align(8)
        v = struct.unpack_from("<d", self.d, self.p)[0]
        self.p += 8
        return v


class Writer:
    def __init__(self):
        self.b = bytearray()

    def align(self, n):
        self.b += bytes(-len(self.b) % n)

    def u32(self, v):
        self.align(4)
        self.b += struct.pack("<I", v)

    def f32(self, v):
        self.align(4)
        self.b += struct.pack("<f", v)

    def f64(self, v):
        self.align(8)
        self.b += struct.pack("<d", v)


# --------------------------------------------------------------------- model ---
# A node is a tuple:  ("null",)  |  ("ref", refId)  |  ("new", wireId, type, payload)
# payload depends on type (see decode_object). Floats are kept as python floats; the round
# trip test proves bit-exactness on the real file.
class FormatError(Exception):
    pass


def decode_object(c: Cur, depth: int = 0):
    if depth > 128:
        raise FormatError("recursion limit")
    oid = c.u32()
    if oid == 0:
        return ("null",)
    if oid == 1:
        return ("ref", c.u32())
    if oid < 100:
        raise FormatError(f"bad object id {oid} at {c.p}")
    t = c.u32()
    if t == 13:
        prec = c.f64()
        prim, bits, flags, path = c.u32(), c.u32(), c.u32(), c.u32()
        tr = decode_object(c, depth + 1)
        ap = decode_object(c, depth + 1)
        aux = None
        if (flags & 0x20000080) == 0x20000080:
            at = c.u32()
            n = 12 if at == 0 else 3 if at == 1 else 7
            if at > 2:
                raise FormatError("aux transform type")
            vals = [c.f64() for _ in range(n)]
            ori = c.u32() if at == 0 else None
            aux = (at, vals, ori)
        geo = decode_object(c, depth + 1)
        pl = {"prec": prec, "prim": prim, "bits": bits, "flags": flags, "path": path,
              "tr": tr, "ap": ap, "aux": aux, "geo": geo}
    elif 14 <= t <= 17:
        if t == 17:
            pl = [c.f32() for _ in range(13)] + [c.f64() for _ in range(3)]
        else:
            pl = [c.f64() for _ in range({14: 3, 15: 7, 16: 8}[t])]
    elif t == 54:
        pl = [c.f32() for _ in range(MATERIAL_FLOATS)]
    elif t == 55:
        flags = c.u32()
        mat = decode_object(c, depth + 1) if flags & 1 else None
        asset = decode_object(c, depth + 1) if flags & 32 else None
        ov = {}
        for bit in (64, 2, 4, 8):
            if flags & bit:
                ov[bit] = c.u32()
        pl = {"flags": flags, "mat": mat, "asset": asset, "ov": ov}
    elif t == 93:
        bounds = [c.f32() for _ in range(6)]
        origin = [c.f64() for _ in range(3)]
        tol = c.f32()
        chk = c.u32()
        gid = c.u32()
        pl = {"bounds": bounds, "origin": origin, "tol": tol, "chk": chk, "gid": gid}
    else:
        raise FormatError(f"unsupported object type {t} at {c.p}")
    return ("new", oid, t, pl)


def encode_object(w: Writer, node):
    if node[0] == "null":
        w.u32(0)
        return
    if node[0] == "ref":
        w.u32(1)
        w.u32(node[1])
        return
    _, oid, t, pl = node
    w.u32(oid)
    w.u32(t)
    if t == 13:
        w.f64(pl["prec"])
        for k in ("prim", "bits", "flags", "path"):
            w.u32(pl[k])
        encode_object(w, pl["tr"])
        encode_object(w, pl["ap"])
        if pl["aux"] is not None:
            at, vals, ori = pl["aux"]
            w.u32(at)
            for v in vals:
                w.f64(v)
            if ori is not None:
                w.u32(ori)
        encode_object(w, pl["geo"])
    elif 14 <= t <= 17:
        if t == 17:
            for v in pl[:13]:
                w.f32(v)
            for v in pl[13:]:
                w.f64(v)
        else:
            for v in pl:
                w.f64(v)
    elif t == 54:
        for v in pl:
            w.f32(v)
    elif t == 55:
        w.u32(pl["flags"])
        if pl["flags"] & 1:
            encode_object(w, pl["mat"])
        if pl["flags"] & 32:
            encode_object(w, pl["asset"])
        for bit in (64, 2, 4, 8):
            if pl["flags"] & bit:
                w.u32(pl["ov"][bit])
    elif t == 93:
        for v in pl["bounds"]:
            w.f32(v)
        for v in pl["origin"]:
            w.f64(v)
        w.f32(pl["tol"])
        w.u32(pl["chk"])
        w.u32(pl["gid"])
    else:
        raise FormatError(f"cannot encode type {t}")


def decode(data: bytes, limit: int | None = None):
    c = Cur(data)
    count = c.u32()
    n = count if limit is None else min(limit, count)
    return count, [decode_object(c) for _ in range(n)], c.p


def encode(count: int, objects: list) -> bytes:
    w = Writer()
    w.u32(count)
    for o in objects:
        encode_object(w, o)
    w.align(8)                       # chunk payloads are padded to a multiple of 8 (verified on real data)
    return bytes(w.b)


if __name__ == "__main__":
    import time
    path = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    raw = open(path, "rb").read()
    t0 = time.time()
    count, objs, end = decode(raw, lim)
    print(f"decoded {len(objs)}/{count} shapes, consumed {end}/{len(raw)} bytes in {time.time()-t0:.1f}s")
    out = encode(count, objs)
    ok = out[:end] == raw[:end]
    print("round trip over decoded span:", "BYTE-IDENTICAL" if ok else "DIFFERS")
    if lim is None:
        print("whole chunk identical:", out == raw)
