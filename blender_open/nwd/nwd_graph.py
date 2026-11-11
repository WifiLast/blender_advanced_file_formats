#!/usr/bin/env python3
"""nwd_graph.py - object-graph chunks of an NWD (version 448): LcOpNwdPartition, LcOpNwdLogicalHierarchy,
LcOaPartitionProps pages, LcOpOdyFileInfoChunk.

Independent implementation (layouts learnt from real files).

Wire grammar (all scalars aligned to their size relative to the chunk start; strings = u32 length (no NUL) + bytes,
0xFFFFFFFF = null string; the next scalar re-aligns):
  object := u32 0                       null
          | u32 1, u32 id               back reference to an earlier object
          | u32 id (>=100), u32 type, body(type)
  common := string name, object classNameObj, u32 flags                       (version >= 22)
  node   (types 22, 25, 32, 53):
      [type 32: u32 x]  common  u32 y  u32 nAttr  nAttr x object attribute
      [type != 22 and not root partition: u32 nChild  nChild x object child]
      [type 32: u32 z, u32 nRef, nRef x object, 3 x f64, u32 .., ... see decode_node]
  name object (52, 82): string, string
  property attribute (84, 86): common, u32 n, n x { object nameObj, value }
  GUID attribute (180): common, 16 raw bytes (aligned 4)
  value := u32 tag, payload   (tag 0 none; 1,6,7,10,11 f64; 2,3 i32; 4,9 string; 5,15,16 i64; 14 u32; 8 object; 12 3xf64; 13 2xf64)
"""
from __future__ import annotations

import struct
import sys

VERSION = 448


class GraphError(Exception):
    pass


class Cur:
    def __init__(self, d: bytes):
        self.d, self.p = d, 0

    def align(self, n):
        self.p = (self.p + n - 1) & ~(n - 1)

    def _chk(self, n):
        if self.p + n > len(self.d):
            raise GraphError(f"truncated at {self.p}")

    def u32(self):
        self.align(4); self._chk(4)
        v = struct.unpack_from("<I", self.d, self.p)[0]; self.p += 4; return v

    def i32(self):
        self.align(4); self._chk(4)
        v = struct.unpack_from("<i", self.d, self.p)[0]; self.p += 4; return v

    def f64(self):
        self.align(8); self._chk(8)
        v = struct.unpack_from("<d", self.d, self.p)[0]; self.p += 8; return v

    def i64(self):
        self.align(8); self._chk(8)
        v = struct.unpack_from("<q", self.d, self.p)[0]; self.p += 8; return v

    def string(self):
        n = self.u32()
        if n == 0xFFFFFFFF:
            return None
        self._chk(n)
        s = self.d[self.p:self.p + n]; self.p += n
        return s.rstrip(b"\0").decode("utf-8", "replace")

    def raw(self, n):
        self._chk(n); b = self.d[self.p:self.p + n]; self.p += n; return b


class Graph:
    """Decoded graph: objs[wireId] = dict(type=..., ...). `roots` = top level object ids in order."""

    def __init__(self):
        self.objs: dict[int, dict] = {}
        self.order: list[int] = []        # wire ids in definition order
        self.roots: list[int | None] = []
        self.type_counts: dict[int, int] = {}


def _value(c: Cur, g: Graph):
    tag = c.u32()
    if tag == 0:
        return (tag, None)
    if tag in (1, 6, 7, 10, 11):
        return (tag, c.f64())
    if tag in (2, 3):
        return (tag, c.i32())
    if tag in (4, 9):
        return (tag, c.string())
    if tag in (5, 15, 16):
        return (tag, c.i64())
    if tag == 14:
        return (tag, c.u32())
    if tag == 8:
        return (tag, _object(c, g))
    if tag == 12:
        return (tag, [c.f64() for _ in range(3)])
    if tag == 13:
        return (tag, [c.f64() for _ in range(2)])
    raise GraphError(f"unsupported value tag {tag} at {c.p}")


def _common(c: Cur, g: Graph, o: dict):
    o["name"] = c.string()
    o["cls"] = _object(c, g)
    o["flags"] = c.u32()


def _object(c: Cur, g: Graph, depth: int = 0, root_partition: bool = False):
    if depth > 400:
        raise GraphError("recursion")
    oid = c.u32()
    if oid == 0:
        return None
    if oid == 1:
        ref = c.u32()
        if ref not in g.objs:
            raise GraphError(f"back reference to unknown id {ref} at {c.p}")
        return ref
    if oid < 100:
        raise GraphError(f"bad object id {oid} at {c.p - 4}")
    if oid in g.objs:
        raise GraphError(f"duplicate id {oid}")
    t = c.u32()
    o = {"type": t, "id": oid}
    g.objs[oid] = o
    g.order.append(oid)
    g.type_counts[t] = g.type_counts.get(t, 0) + 1
    if t in (52, 82):
        o["s"] = [c.string(), c.string()]
    elif t in (22, 25, 32, 53):
        if t == 32:
            o["pre"] = c.u32()
        _common(c, g, o)
        o["y"] = c.u32()
        n = c.u32()
        o["attrs"] = [_object(c, g, depth + 1) for _ in range(n)]
        if t != 22 and not (root_partition and oid == 100 and t == 32):
            n = c.u32()
            o["children"] = [_object(c, g, depth + 1) for _ in range(n)]
        if t == 32:
            o["z"] = c.u32()
            n = c.u32()
            o["refs"] = [_object(c, g, depth + 1) for _ in range(n)]
            o["d3"] = [c.f64() for _ in range(3)]
            o["u4"] = c.u32()
            # version >= 3/5/53/80/45/57/109/230/421 fields (all present at 448)
            o["tail"] = []
            o["tail"].append(("s", c.string()))        # v>=3 string
            o["tail"].append(("u", c.u32()))           # v>=3
            o["tail"].append(("u", c.u32()))           # v>=5
            o["tail"].append(("u", c.u32()))           # v>=5
            o["tail"].append(("d", [c.f64() for _ in range(6)]))   # v>=53
            o["tail"].append(("u", c.u32()))           # v>=80
            o["tail"].append(("s", c.string()))        # v>=80
            o["tail"].append(("u", c.u32()))           # v>=45
            o["tail"].append(("s", c.string()))        # v>=57
            o["tail"].append(("d", [c.f64() for _ in range(3)]))   # v>=109
            o["tail"].append(("d", [c.f64() for _ in range(4)]))   # v>=230
            has = c.u32()                              # v>=421
            o["tail"].append(("u", has))
            if has:
                k = c.u32()
                o["tail"].append(("u", k))
                for _ in range(k):
                    o["tail"].append(("u", c.u32()))
    elif t in (84, 86):
        _common(c, g, o)
        n = c.u32()
        o["props"] = []
        for _ in range(n):
            nm = _object(c, g, depth + 1)
            o["props"].append((nm, _value(c, g)))
    elif t == 180:
        _common(c, g, o)
        c.align(4)
        o["guid"] = c.raw(16)
    elif t == 92:        # publish attribute
        _common(c, g, o)
        o["pub_flags"] = c.u32()
        o["pub_s"] = [c.string() for _ in range(4)]
        o["pub_t"] = [c.i64() for _ in range(2)]
        o["pub_s2"] = [c.string() for _ in range(4)]
    elif t == 31:
        _common(c, g, o)
        o["d"] = [c.f64() for _ in range(14)]
    elif t == 27:                       # light: common_light = u32, [ref if version>=252], flags ; then 12 doubles
        o["i"] = c.u32()
        o["ref"] = _object(c, g, depth + 1)
        o["flags"] = c.u32()
        o["d"] = [c.f64() for _ in range(12)]
    elif t == 185:                      # JSON asset (version >= 447: json + extra), file list
        o["json"] = c.string()
        o["extra"] = c.string()
        n = c.u32()
        o["files"] = []
        for _ in range(n):
            name = c.string()
            emb = c.u32()
            if emb:
                pre, suf = c.string(), c.string()
                size = c.i64()
                o["files"].append((name, 1, pre, suf, c.raw(size)))
            else:
                o["files"].append((name, 0, c.string()))
    elif t in (38, 68, 70):
        _common(c, g, o)
    elif t in (40, 41):
        _common(c, g, o)
        o["d"] = [c.f64() for _ in range(3 if t == 40 else 7)]
    elif t == 39:
        _common(c, g, o)
        o["d"] = [c.f64() for _ in range(12)]
        o["u"] = c.u32()
    elif t in (74, 76):
        _common(c, g, o)
        o["s"] = c.string()
    elif t in (64, 66, 67):
        _common(c, g, o)
        o["q"] = c.i64()
    elif t in (14, 15, 16, 17):
        if t == 17:
            o["v"] = [c.f64() for _ in range(3)]      # unused placeholder, real layout differs
        else:
            o["v"] = [c.f64() for _ in range({14: 3, 15: 7, 16: 8}[t])]
    else:
        raise GraphError(f"unsupported object type {t} at {c.p}")
    return oid


def decode_hierarchy(data: bytes):
    c = Cur(data)
    g = Graph()
    n = c.u32()
    pages = [c.u32() for _ in range(n)]
    root = c.u32()
    k = c.u32()
    g.roots = [_object(c, g) for _ in range(k)]
    tail = c.u32()
    return g, pages, root, tail, c.p


def decode_partition(data: bytes):
    c = Cur(data)
    g = Graph()
    g.roots = [_object(c, g, root_partition=True)]
    return g, c.p


def decode_props_page(data: bytes, n: int):
    c = Cur(data)
    g = Graph()
    g.roots = [_object(c, g) for _ in range(n)]
    return g, c.p


# ------------------------------------------------------------------ encoder ---
class W:
    def __init__(self):
        self.b = bytearray()

    def align(self, n):
        self.b += bytes(-len(self.b) % n)

    def u32(self, v):
        self.align(4); self.b += struct.pack("<I", v)

    def i32(self, v):
        self.align(4); self.b += struct.pack("<i", v)

    def f64(self, v):
        self.align(8); self.b += struct.pack("<d", v)

    def i64(self, v):
        self.align(8); self.b += struct.pack("<q", v)

    def string(self, s):
        if s is None:
            self.u32(0xFFFFFFFF)
            return
        bs = s.encode("utf-8")
        self.u32(len(bs)); self.b += bs

    def raw(self, b):
        self.b += b


def _e_value(w: W, g: Graph, done: set, v):
    tag, x = v
    w.u32(tag)
    if tag == 0:
        return
    if tag in (1, 6, 7, 10, 11):
        w.f64(x)
    elif tag in (2, 3):
        w.i32(x)
    elif tag in (4, 9):
        w.string(x)
    elif tag in (5, 15, 16):
        w.i64(x)
    elif tag == 14:
        w.u32(x)
    elif tag == 8:
        _e_obj(w, g, done, x)
    elif tag in (12, 13):
        for d in x:
            w.f64(d)
    else:
        raise GraphError(f"cannot encode value tag {tag}")


def _e_common(w, g, done, o):
    w.string(o["name"]); _e_obj(w, g, done, o["cls"]); w.u32(o["flags"])


def _e_obj(w: W, g: Graph, done: set, oid, root_partition=False):
    if oid is None:
        w.u32(0); return
    if oid in done:
        w.u32(1); w.u32(oid); return
    done.add(oid)
    o = g.objs[oid]; t = o["type"]
    w.u32(oid); w.u32(t)
    if t in (52, 82):
        w.string(o["s"][0]); w.string(o["s"][1])
    elif t in (22, 25, 32, 53):
        if t == 32:
            w.u32(o["pre"])
        _e_common(w, g, done, o)
        w.u32(o["y"]); w.u32(len(o["attrs"]))
        for a in o["attrs"]:
            _e_obj(w, g, done, a)
        if t != 22 and not (root_partition and oid == 100 and t == 32):
            w.u32(len(o["children"]))
            for ch in o["children"]:
                _e_obj(w, g, done, ch)
        if t == 32:
            w.u32(o["z"]); w.u32(len(o["refs"]))
            for r in o["refs"]:
                _e_obj(w, g, done, r)
            for d in o["d3"]:
                w.f64(d)
            w.u32(o["u4"])
            for kind, v in o["tail"]:
                if kind == "s":
                    w.string(v)
                elif kind == "u":
                    w.u32(v)
                else:
                    for d in v:
                        w.f64(d)
    elif t in (84, 86):
        _e_common(w, g, done, o)
        w.u32(len(o["props"]))
        for nm, val in o["props"]:
            _e_obj(w, g, done, nm); _e_value(w, g, done, val)
    elif t == 180:
        _e_common(w, g, done, o); w.align(4); w.raw(o["guid"])
    elif t == 92:
        _e_common(w, g, done, o); w.u32(o["pub_flags"])
        for x in o["pub_s"]:
            w.string(x)
        for x in o["pub_t"]:
            w.i64(x)
        for x in o["pub_s2"]:
            w.string(x)
    elif t == 31:
        _e_common(w, g, done, o)
        for d in o["d"]:
            w.f64(d)
    elif t in (38, 68, 70):
        _e_common(w, g, done, o)
    elif t in (40, 41):
        _e_common(w, g, done, o)
        for d in o["d"]:
            w.f64(d)
    elif t == 39:
        _e_common(w, g, done, o)
        for d in o["d"]:
            w.f64(d)
        w.u32(o["u"])
    elif t in (74, 76):
        _e_common(w, g, done, o); w.string(o["s"])
    elif t in (64, 66, 67):
        _e_common(w, g, done, o); w.i64(o["q"])
    elif t == 27:
        w.u32(o["i"]); _e_obj(w, g, done, o["ref"]); w.u32(o["flags"])
        for d in o["d"]:
            w.f64(d)
    elif t == 185:
        w.string(o["json"]); w.string(o["extra"]); w.u32(len(o["files"]))
        for f in o["files"]:
            w.string(f[0]); w.u32(f[1])
            if f[1]:
                w.string(f[2]); w.string(f[3]); w.i64(len(f[4])); w.raw(f[4])
            else:
                w.string(f[2])
    else:
        raise GraphError(f"cannot encode type {t}")


def encode_hierarchy(g: Graph, pages, root, tail=1) -> bytes:
    w = W(); done: set = set()
    w.u32(len(pages))
    for p in pages:
        w.u32(p)
    w.u32(root); w.u32(len(g.roots))
    for r in g.roots:
        _e_obj(w, g, done, r)
    w.u32(tail)
    w.align(8)
    return bytes(w.b)


def encode_props_page(g: Graph) -> bytes:
    """One LcOaPartitionProps page: g.roots = the page's property records (type 86 tabs), wire ids from 100 per page,
    back references only within the page. Padded to 8 like Navisworks' own pages (Example.nwd: byte-identical)."""
    w = W(); done: set = set()
    for r in g.roots:
        _e_obj(w, g, done, r)
    w.align(8)
    return bytes(w.b)


class PageBuilder:
    """Builds property pages: one type-86 tab per record, name objects (type 52) shared within a page.

    tab(display, internal, [(prop display name, value)]) appends one record; value = (tag, x) with tag 4 string,
    1 double (booleans as 0.0/1.0), 2 int32. new_page() starts the next page (ids restart at 100)."""

    def __init__(self, records_per_page: int = 10000):
        self.per_page = records_per_page
        self.pages: list[Graph] = []
        self._names: dict = {}
        self._next = 100
        self.new_page()

    def new_page(self):
        self.pages.append(Graph())
        self._names = {}
        self._next = 100

    def reserve(self, n: int):
        """Keep the next n records on one page (a node's placeholders never straddle a page boundary)."""
        if self.pages[-1].roots and len(self.pages[-1].roots) + n > self.per_page:
            self.new_page()

    def _new(self, t, **kw):
        g = self.pages[-1]
        oid = self._next
        self._next += 1
        o = {"type": t, "id": oid, **kw}
        g.objs[oid] = o
        g.order.append(oid)
        g.type_counts[t] = g.type_counts.get(t, 0) + 1
        return o

    def _name(self, display, internal):
        k = (display, internal)
        if k not in self._names:
            self._names[k] = self._new(52, s=[display, internal])["id"]
        return self._names[k]

    def tab(self, display: str, internal: str, props):
        if len(self.pages[-1].roots) >= self.per_page:
            self.new_page()
        rec = self._new(86, name=None, cls=None, flags=0, props=[])
        self.pages[-1].roots.append(rec["id"])
        rec["cls"] = self._name(display, internal)           # emitted after the record header, before its props
        for pname, val in props:
            rec["props"].append((self._name(pname, "LcDwfData"), val))

    def counts(self) -> list[int]:
        return [len(g.roots) for g in self.pages if g.roots]

    def encoded(self) -> list[bytes]:
        return [encode_props_page(g) for g in self.pages if g.roots]


def encode_partition(g: Graph) -> bytes:
    w = W()
    _e_obj(w, g, set(), g.roots[0], root_partition=True)
    w.align(8)
    return bytes(w.b)


if __name__ == "__main__":
    import json, os
    sys.setrecursionlimit(10000)
    BS = chr(92)
    T = os.path.join(os.environ["TEMP"], "real", "x")
    m = json.load(open(T + "/manifest.json"))
    ch = {c["name"].split(BS)[-1]: c for c in m["chunks"]}
    which = sys.argv[1] if len(sys.argv) > 1 else "partition"
    d = open(os.path.join(T, ch[{"partition": "LcOpNwdPartition", "hierarchy": "LcOpNwdLogicalHierarchy"}[which]]["decoded"]), "rb").read()
    try:
        if which == "partition":
            g, end = decode_partition(d)
            print(f"partition parsed {end}/{len(d)} bytes; types: {dict(sorted(g.type_counts.items()))}")
            print("partition re-encode identical:", encode_partition(g) == d)
        else:
            g, pages, root, tail, end = decode_hierarchy(d)
            print(f"hierarchy parsed {end}/{len(d)} bytes; pages={pages} root={root} tail={tail}; types: {dict(sorted(g.type_counts.items()))}")
            print("hierarchy re-encode identical:", encode_hierarchy(g, pages, root, tail) == d)
    except GraphError as e:
        print("ERROR:", e)
