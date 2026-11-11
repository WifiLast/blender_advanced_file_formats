"""Pure-Python (numpy only) reader: Navisworks .nwd -> placed triangle meshes.

Uses the codecs in this package (container, LcOpNwdFragments, LcOpNwdGeometry type-94 records, logical hierarchy).
Result: NwdScene with unique meshes (numpy vertices / triangles) and instances (transform, colour, name).
Limits: only the 'simple' type-94 record forms are decoded (others are counted in `skipped`); line/point
geometry (types 95/96) and quantised / UV forms are ignored.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import time

import numpy as np

from .. import log as cadlog

from . import nwd_fragments as fr
from . import nwd_geometry as geo
from . import nwd_graph as ng
from . import nwd_tool as nt

BS = chr(92)
# The type-93 'origin' field is NOT added to the vertices: tested against the GLB export of the same Revit project
# (nugget.nwd vs the Revit GLB): without it 7382 instance centres match GLB elements, with it only 1733.
APPLY_ORIGIN = False
L = cadlog.get("nwd")


@dataclass
class Instance:
    name: str
    mesh: int                       # index into NwdScene.meshes
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (0.0, 0.0, 0.0, 1.0)   # x, y, z, w
    scale: float = 1.0
    rgba: tuple = (0.8, 0.8, 0.8, 1.0)
    matrix: tuple | None = None     # 3 basis vectors (row-vector convention, scale included) for transform type 17


@dataclass
class NwdScene:
    meshes: list = field(default_factory=list)       # [(vertices (N,3) f64, triangles (M,3) i64)]
    instances: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def strips_to_mesh(rec: dict):
    if rec.get("line"):
        return lines_to_edges(rec)
    """decode94() dict -> (vertices (P,3), triangles (T,3)) with triangle strips expanded."""
    coords = np.asarray(rec["coords"], dtype=np.float64).reshape(-1, 3)
    cidx = np.asarray(rec["cidx"], dtype=np.int64)
    n = rec["n"]
    corner = np.asarray(rec["sidx"], dtype=np.int64) if rec["sidx"] else np.arange(n, dtype=np.int64)
    lengths = rec["lengths"]
    if lengths and all(k == 3 for k in lengths) and len(corner) == 3 * len(lengths):
        tri = corner.reshape(-1, 3)
    else:
        out, o = [], 0
        for k in lengths:
            for i in range(k - 2):
                a, b, c = corner[o + i], corner[o + i + 1], corner[o + i + 2]
                out.append((a, c, b) if i & 1 else (a, b, c))
            o += k
        tri = np.asarray(out, dtype=np.int64).reshape(-1, 3)
    tri = cidx[tri]
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return coords, tri[keep]


def lines_to_edges(rec: dict):
    """Line record (type 95): strips of k corners -> k-1 segments. -> (vertices (P,3), edges (E,2))"""
    coords = np.asarray(rec["coords"], dtype=np.float64).reshape(-1, 3)
    cidx = np.asarray(rec["cidx"], dtype=np.int64)
    segs, o = [], 0
    lengths = rec["lengths"]
    if lengths and all(k == 2 for k in lengths):
        e = cidx[: 2 * len(lengths)].reshape(-1, 2)
    else:
        for k in lengths:
            for i in range(k - 1):
                segs.append((cidx[o + i], cidx[o + i + 1]))
            o += k
        e = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    return coords, e[e[:, 0] != e[:, 1]]


class Unsupported(Exception):
    """A geometry record in a form this reader does not decode; .args[0] is the reason."""


def decode_record(rec: bytes, lines: bool = False):
    """Type-94 record -> decode94()-style dict. Like geo.decode94 but also accepts float attribute blocks (type 59,
    as written by our own NWD writer) and skips attributes it does not need (normals are recomputed by Blender)."""
    R = geo._R
    r = R(rec)
    head = (r.u32(), r.u32(), r.u32(), r.u32())
    is_line = lines and head == (100, 95, 101, 60)
    if head != (100, 94, 101, 60) and not is_line:
        raise Unsupported(f"header {head} (type {head[1]}: " + {94: "strips", 95: "lines", 96: "points"}.get(head[1], "unknown") + ")")
    n = r.i32()
    if n < 0:
        raise Unsupported(f"negative vertex count {n} (quantised / alternative coordinate form)")
    floats = r.u32()
    if floats % 3:
        raise Unsupported("coordinate float count not a multiple of 3")
    r.al(4)
    coords = np.frombuffer(rec, dtype="<f4", count=floats, offset=r.p).astype(np.float64)
    r.p += 4 * floats
    P = floats // 3
    cidx = geo.rle_decode(r, n) if P < n else list(range(n))
    for _slot in range(3):
        oid = r.u32()
        if not oid:
            continue
        typ = r.u32()
        if typ == 59:
            m = r.i32()
            fl = r.u32()
            r.p += 4 * fl
            if fl // 3 < abs(m):
                geo.rle_decode(r, abs(m))
        elif typ == 100:
            slots = r.i32()
            pal = r.u32()
            r.p += 4 * ((pal * 3 * 8 + 31) // 32)
            if pal < abs(slots):
                geo.rle_decode(r, abs(slots))
        else:
            raise Unsupported(f"attribute object type {typ}")
    groups = abs(r.i32())
    bits = r.u8()
    r.al(4)
    cnt = r.u32()
    nw = (bits * cnt + 31) // 32
    import struct
    words_ = struct.unpack_from(f"<{nw}I", rec, r.p)
    r.p += 4 * nw
    lengths = []
    for i in range(cnt):
        b = i * bits
        w, sh = b // 32, b % 32
        v = words_[w]
        if sh + bits <= 32:
            v >>= (32 - sh - bits)
        else:
            v = ((v << 32) | words_[w + 1]) >> (64 - sh - bits)
        lengths.append(v & ((1 << bits) - 1))
    if is_line:                                   # line strips end after the lengths (no explicit index count)
        return {"n": n, "coords": coords, "cidx": cidx, "groups": groups, "lengths": lengths, "sidx": None, "line": True}
    nidx = abs(r.i32())
    sidx = geo.rle_decode(r, nidx) if nidx else None
    return {"n": n, "coords": coords, "cidx": cidx, "groups": groups, "lengths": lengths, "sidx": sidx}


def _collect(node, table):
    """Register every 'new' object (recursively) so back references can be resolved."""
    if node[0] != "new":
        return
    table[node[1]] = node
    t, pl = node[2], node[3]
    if t == 13:
        for k in ("tr", "ap", "geo"):
            _collect(pl[k], table)
    elif t == 55:
        for k in ("mat", "asset"):
            if pl[k] is not None:
                _collect(pl[k], table)


def _resolve(node, table):
    return table.get(node[1]) if node[0] == "ref" else node


def _transform(node, table):
    n = _resolve(node, table)
    if n is None or n[0] != "new":
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1.0, None
    t, v = n[2], n[3]
    if t == 14:
        return tuple(v), (0.0, 0.0, 0.0, 1.0), 1.0, None
    if t == 15:
        return tuple(v[:3]), tuple(v[3:7]), 1.0, None
    if t == 16:
        return tuple(v[:3]), tuple(v[3:7]), float(v[7]), None
    if t == 17:
        # 13 f32 = three basis vectors stored as 4 floats each (x, y, z, 0) + 1.0, then the translation as 3 f64.
        # Row-vector convention (v' = v * rows + t); the scale is part of the rows (e.g. 304.8 = feet -> mm).
        return (v[13], v[14], v[15]), (0.0, 0.0, 0.0, 1.0), 1.0, (tuple(v[0:3]), tuple(v[4:7]), tuple(v[8:11]))
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1.0, None


def _colour(node, table):
    n = _resolve(node, table)
    if n is None or n[0] != "new" or n[2] != 55 or n[3]["mat"] is None:
        return (0.8, 0.8, 0.8, 1.0)
    m = _resolve(n[3]["mat"], table)
    if m is None or m[0] != "new" or len(m[3]) < 14:
        return (0.8, 0.8, 0.8, 1.0)
    f = m[3]
    return (f[3], f[4], f[5], max(0.05, 1.0 - f[13]))


def _path_names(entries, data):
    """path id -> leaf node name (pre-order, root = 0). Best effort, may be wrong for odd trees; {} when the hierarchy cannot be read."""
    try:
        e = next(x for x in entries if x["name"].split(BS)[-1] == "LcOpNwdLogicalHierarchy")
        raw = nt.decode_chunk(data[e["offset"]:e["offset"] + e["size"]],
                              {"flags": e["flags"], "prefix": e["prefix"], "index": e["index_bytes"]})
        g, _p, _r, _t, _e = ng.decode_hierarchy(raw)
        names, counter = {}, [-1]
        stack = [g.roots[0]]
        while stack:
            oid = stack.pop()
            o = g.objs[oid]
            counter[0] += 1
            # only leaves are trusted: a group that also carries meshes shifts the path numbering (see nwd_build._hierarchy)
            if not o.get("children"):
                names[counter[0]] = o.get("name") or ""
            stack.extend(reversed(o.get("children", []) or []))
        return names
    except Exception:
        return {}


def read_nwd(path: str, log=None, lines: bool = True) -> NwdScene:
    old = sys.getrecursionlimit()                 # deeply nested hierarchies need more; restore afterwards
    sys.setrecursionlimit(max(old, 20000))
    try:
        return _read_nwd(path, log, lines)
    finally:
        sys.setrecursionlimit(old)


def _read_nwd(path: str, log, lines: bool) -> NwdScene:
    log = log or (lambda *a: None)
    with open(path, "rb") as fh:
        data = fh.read()
    L.info("file %s: %.1f MB", path, len(data) / 1e6)
    info = nt.parse(data)
    L.info("container: NWD version %s, %d chunks, problems: %s", info["nwd_version"], len(info["entries"]), info["problems"] or "none")
    for _e in info["entries"]:
        L.debug("chunk %-28s flags=%d size=%d prefix=%d index=%d", _e["name"].split(BS)[-1], _e["flags"], _e["size"], _e["prefix"], _e["index_bytes"])
    ents = {e["name"].split(BS)[-1]: e for e in info["entries"]}
    scene = NwdScene(info={"version": info["nwd_version"], "chunks": len(info["entries"])})
    for need in ("LcOpNwdFragments", "LcOpNwdGeometry"):
        if need not in ents:
            raise RuntimeError(f"no {need} chunk: this NWD stores no geometry (or an unsupported variant)")

    def raw(name):
        e = ents[name]
        return data[e["offset"]:e["offset"] + e["size"]], e

    log("decoding fragments ...")
    t0 = time.perf_counter()
    blob, e = raw("LcOpNwdFragments")
    count, shapes, _end = fr.decode(nt.decode_chunk(blob, {"flags": e["flags"], "prefix": e["prefix"],
                                                           "index": e["index_bytes"]}))
    table = {}
    for s in shapes:
        _collect(s, table)
    L.info("fragments: %d shapes (header says %d), %d registered objects, %.1f s", len(shapes), count, len(table), time.perf_counter() - t0)
    _tc = {}
    for _o in table.values():
        _tc[_o[2]] = _tc.get(_o[2], 0) + 1
    L.info("object types in fragments: %s", dict(sorted(_tc.items())))
    t0 = time.perf_counter()

    log("decoding geometry records ...")
    blob, e = raw("LcOpNwdGeometry")
    damage = {}
    _pre, _blocks, records, _b, _r = geo.split_chunk(blob, e["prefix"], e["index_bytes"], damage)
    del _blocks
    bad_recs = damage.get("bad_records", set())
    if damage.get("bad_blocks"):
        scene.info["damaged_blocks"] = len(damage["bad_blocks"])
        log(f"WARNING: {len(damage['bad_blocks'])} damaged geometry blocks, {len(bad_recs)} records unreadable")
        _bb = damage["bad_blocks"]
        L.warning("damaged blocks: first %s ... last %s (contiguous: %s)", _bb[:10], _bb[-3:], _bb[-1] - _bb[0] + 1 == len(_bb))
    L.info("geometry: %d records, %d bytes, %.1f s", len(records), sum(len(r) for r in records), time.perf_counter() - t0)
    mesh_of = {}
    unknown_tr = 0
    why_count, why_example = {}, {}

    def mesh_for(gid, origin=(0.0, 0.0, 0.0)):
        if gid in mesh_of:
            return mesh_of[gid]
        idx = None
        if gid - 1 in bad_recs:
            scene.skipped["geometry in damaged file blocks"] = scene.skipped.get("geometry in damaged file blocks", 0) + 1
        elif 1 <= gid <= len(records):
            try:
                rec = decode_record(records[gid - 1], lines)
                if rec is not None:
                    v, t = strips_to_mesh(rec)
                    if APPLY_ORIGIN and any(origin):                 # geometry reference origin: vertices are relative to it
                        v = v + np.asarray(origin, dtype=np.float64)
                    if len(t):
                        scene.meshes.append((v, t))
                        if t.shape[1] == 2:
                            scene.info["line_meshes"] = scene.info.get("line_meshes", 0) + 1
                        idx = len(scene.meshes) - 1
                else:
                    scene.skipped["geometry record in unsupported form"] = scene.skipped.get("geometry record in unsupported form", 0) + 1
            except Unsupported as exc:
                why = str(exc.args[0])
                why_count[why] = why_count.get(why, 0) + 1
                why_example.setdefault(why, []).append((gid, len(records[gid - 1]), records[gid - 1][:40].hex()))
                scene.skipped["geometry record in unsupported form"] = scene.skipped.get("geometry record in unsupported form", 0) + 1
            except Exception as exc:
                why = f"decode error {type(exc).__name__}: {exc}"
                why_count[why] = why_count.get(why, 0) + 1
                why_example.setdefault(why, []).append((gid, len(records[gid - 1]), records[gid - 1][:40].hex()))
                scene.skipped["geometry record decode error"] = scene.skipped.get("geometry record decode error", 0) + 1
        mesh_of[gid] = idx
        return idx

    names = _path_names(info["entries"], data)
    used = {sh[3]["path"] for sh in shapes if sh[0] == "new" and sh[2] == 13}
    if not used <= set(names):          # numbering does not line up with the leaves: do not guess names
        names = {}
    for s in shapes:
        if s[0] != "new" or s[2] != 13:
            continue
        pl = s[3]
        g = _resolve(pl["geo"], table)
        if g is None or g[0] != "new":
            continue
        mi = mesh_for(g[3]["gid"], g[3]["origin"])
        if mi is None:
            continue
        pos, quat, sc, mat = _transform(pl["tr"], table)
        _tn = _resolve(pl["tr"], table)
        if _tn is None or _tn[0] != "new" or _tn[2] not in (14, 15, 16, 17):
            unknown_tr += 1
        nm = names.get(pl["path"]) or f"Item_{pl['path']}"
        scene.instances.append(Instance(nm, mi, pos, quat, sc, _colour(pl["ap"], table), mat))
    log(f"{len(scene.instances)} instances, {len(scene.meshes)} unique meshes")
    if why_count:
        L.info("records not decoded, by reason (%d distinct):", len(why_count))
        for _w, _c in sorted(why_count.items(), key=lambda kv: -kv[1]):
            L.info("  %6d x %s", _c, _w)
            for _gid, _sz, _hx in why_example[_w][:3]:
                L.debug("    example gid=%d size=%d head=%s", _gid, _sz, _hx)
    L.info("hierarchy names: %d usable leaf names for %d distinct shape paths (%s)", len(names), len(used), "used" if names else "falling back to Item_<path>")
    _col = {}
    for _i in scene.instances:
        _k = tuple(round(c, 2) for c in _i.rgba)
        _col[_k] = _col.get(_k, 0) + 1
    L.info("distinct colours: %d, top: %s", len(_col), sorted(_col.items(), key=lambda kv: -kv[1])[:5])
    _use = {}
    for _i in scene.instances:
        _use[_i.mesh] = _use.get(_i.mesh, 0) + 1
    L.info("mesh reuse: %d meshes used once, most reused mesh %d times", sum(1 for c in _use.values() if c == 1), max(_use.values()) if _use else 0)
    L.info("shapes without mesh: %d; with identity/unknown transform: %d", len(shapes) - len(scene.instances), unknown_tr)
    L.info("skipped: %s", scene.skipped or "nothing")
    if scene.instances:
        P = np.array([i.pos for i in scene.instances])
        L.info("instance positions: min %s max %s", P.min(0).round(2).tolist(), P.max(0).round(2).tolist())
        scene.info["extent"] = float(np.abs(P).max())
        L.info("instances with full matrix (type 17): %d", sum(1 for i in scene.instances if i.matrix is not None))
        scl = np.array([i.scale for i in scene.instances])
        L.info("instance scale: min %.4f max %.4f", scl.min(), scl.max())
        L.info("triangles in unique meshes: %d, vertices: %d", sum(len(t) for _v, t in scene.meshes), sum(len(v) for v, _t in scene.meshes))
        for i in scene.instances[:5]:
            L.debug("instance %s mesh=%d pos=%s quat=%s scale=%.4f rgba=%s", i.name, i.mesh, np.round(i.pos, 3).tolist(),
                    np.round(i.quat, 4).tolist(), i.scale, np.round(i.rgba, 3).tolist())
    return scene
