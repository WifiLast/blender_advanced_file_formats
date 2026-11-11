#!/usr/bin/env python3
"""nwd_geometry.py - `LcOpNwdGeometry` chunk: container codec + raw (uncompressed) geometry records.

Chunk layout (flags 1, stored form = what nwd_tool.py keeps in <n>_...LcOpNwdGeometry.bin):
  [prefix: 4 bytes][zlib block 0][zlib block 1]...[zlib index stream]
  prefix  = chunk 'prefix' = 4 (in the real file; value of the 4 bytes: see PREFIX)
  index   = zlib( u32 recordCount, u32 blockCount, blockCount x u32 compressedBlockSize,
                  recordCount x u32 recordSize )                        (all u32 aligned, padded to 8)
  The concatenation of all inflated blocks is the concatenation of the geometry records;
  record i has recordSize[i] bytes. Directory entry: prefix=4, index=len(index stream).

Record (raw / uncompressed mode, i.e. LcOpNwdGeometryCompress flags == 0), layout learnt from real files. All scalars aligned to their size relative to the record start:
  u32 100            root object id
  u32 94             type: 94 triangle strips, 95 lines (polyline strips), 96 points
  u32 101, u32 60    coordinate object id / type
  i32 count          >= 0 in raw mode, then count*3 f32 coordinates (no index array when palette == count)
  3 x { u32 objectId (0 = none) [, u32 type, attribute payload] }      attributes: normals/colours/uv
  strips (types 94/95), version >= 75: i32 groups, groups x u16 length, i32 n, n x u16 index
  tail: padded to a multiple of 8 with zero bytes
"""
from __future__ import annotations

import bisect
import logging
import struct
import sys
import zlib

from . import nwd_tool as nt


log = logging.getLogger("cad_importer.nwd_geometry")
BLOCK_SIZE = 65536      # inflated size of every block but the last (see slice_blocks)


def _inflate_lenient(buf: bytes):
    """zlib stream -> (data, bytes used, checksum ok). If only the Adler-32 check fails, the deflate data is decoded
    anyway and the 4 checksum bytes are skipped (checksum ok = False); any other damage raises."""
    try:
        return (*nt.inflate_stream(buf), True)
    except zlib.error:
        d = zlib.decompressobj(-15)
        out = d.decompress(buf[2:])
        if not d.eof:
            raise ValueError("truncated deflate stream")
        return out, 2 + (len(buf) - 2 - len(d.unused_data)) + 4, False


def split_chunk(raw: bytes, prefix_len: int, index_len: int, report: dict | None = None, strict: bool = False):
    """-> (prefix bytes, [inflated blocks], [record bytes], block_sizes, record_sizes)

    A damaged block (not inflatable, wrong size or boundary) no longer stops the split: it is logged (warning, one
    entry per block), replaced by zeros of the block size so that every later offset stays valid, and the records
    that overlap it are reported. report (optional dict) receives:
      bad_blocks  [block index]          bad_records  {record index}
      errors      {block index: message}  block_size  inflated block size used for the zero fill
      checksum_blocks [block index]      blocks whose only damage is the Adler-32 check (data kept)
    strict=True restores the old behaviour: the first damaged block raises ValueError."""
    prefix = raw[:prefix_len]
    idx, used = nt.inflate_stream(raw[len(raw) - index_len:])
    count, nb = struct.unpack_from("<II", idx, 0)
    bsz = list(struct.unpack_from(f"<{nb}I", idx, 8))
    rsz = list(struct.unpack_from(f"<{count}I", idx, 8 + 4 * nb))
    block_size = struct.unpack("<I", prefix)[0] if prefix_len == 4 and 0 < struct.unpack("<I", prefix)[0] <= 1 << 24 \
        else BLOCK_SIZE
    body = raw[prefix_len:len(raw) - index_len]
    blocks, bad_blocks, errors, checksum_blocks = None, [], {}, []
    if used != index_len:
        log.warning("geometry index stream: %d of %d bytes used", used, index_len)
        if strict:
            raise ValueError("geometry index stream length mismatch")
    if sum(bsz) == len(body):
        blocks, pos = [], 0
        for i, n in enumerate(bsz):
            try:
                d, u, ok = _inflate_lenient(body[pos:pos + n])
                if not ok:
                    if strict:
                        raise ValueError("Adler-32 checksum mismatch")
                    log.warning("geometry block %d of %d (chunk body offset %d, %d bytes): Adler-32 checksum "
                                "mismatch, deflate data decoded anyway", i, nb, pos, n)
                    checksum_blocks.append(i)
                if u != n:
                    raise ValueError(f"block boundary mismatch ({u} bytes used, {n} listed)")
                if len(d) != block_size and i != nb - 1:
                    raise ValueError(f"block size {len(d)}, expected {block_size}")
            except Exception as exc:          # noqa: BLE001  any damage: keep going
                if strict:
                    raise ValueError(f"geometry block {i} at offset {pos}/{len(body)} is damaged: {exc}") from exc
                log.warning("geometry block %d of %d (chunk body offset %d, %d bytes) is corrupt: %s - "
                            "replaced by %d zero bytes", i, nb, pos, n, exc, block_size)
                d = bytes(block_size)           # keeps all later offsets valid
                bad_blocks.append(i)
                errors[i] = str(exc)
            blocks.append(d)
            pos += n
    else:
        log.warning("geometry block table (%d bytes) does not cover the chunk body (%d bytes): reading the blocks "
                    "one after the other", sum(bsz), len(body))
        if strict:
            raise ValueError("block table does not cover the chunk")
    if blocks is None:                        # the index sizes cannot be trusted: walk the stream
        blocks, pos = [], 0
        while pos < len(body):
            try:
                d, u, ok = _inflate_lenient(body[pos:])
                if not ok:
                    log.warning("geometry block %d at chunk body offset %d: Adler-32 checksum mismatch, deflate data "
                                "decoded anyway", len(blocks), pos)
                    checksum_blocks.append(len(blocks))
            except Exception as exc:          # noqa: BLE001  no boundary to resynchronise on
                log.error("geometry block %d at chunk body offset %d/%d cannot be inflated and the block table is "
                          "unusable: %s", len(blocks), pos, len(body), exc)
                raise ValueError(f"geometry block {len(blocks)} at offset {pos}/{len(body)} cannot be inflated "
                                 f"(prefix {prefix_len}, index {index_len}, {nb} blocks): {exc}") from exc
            if not u:
                break
            blocks.append(d)
            pos += u
    blob = b"".join(blocks)
    if len(blob) < sum(rsz):
        raise ValueError(f"geometry blocks hold {len(blob)} bytes but the records need {sum(rsz)}")
    starts, s = [], 0
    for d in blocks:
        starts.append(s)
        s += len(d)
    bad_set = set(bad_blocks)
    recs, bad_recs, p = [], set(), 0
    for k, n in enumerate(rsz):
        recs.append(blob[p:p + n])
        if bad_set and n:
            first, last = bisect.bisect_right(starts, p) - 1, bisect.bisect_right(starts, p + n - 1) - 1
            if any(b in bad_set for b in range(first, last + 1)):
                bad_recs.add(k)
        p += n
    if bad_blocks:
        log.warning("%d of %d geometry blocks are corrupt (%s); %d of %d records overlap them and are unreadable",
                    len(bad_blocks), nb, ", ".join(map(str, bad_blocks[:20])) + (" ..." if len(bad_blocks) > 20 else ""),
                    len(bad_recs), count)
    if report is not None:
        report.update(bad_blocks=bad_blocks, bad_records=bad_recs, errors=errors, block_size=block_size,
                      checksum_blocks=checksum_blocks)
    return prefix, blocks, recs, bsz, rsz


def build_index(block_csizes: list[int], record_sizes: list[int]) -> bytes:
    b = struct.pack("<II", len(record_sizes), len(block_csizes))
    b += struct.pack(f"<{len(block_csizes)}I", *block_csizes)
    b += struct.pack(f"<{len(record_sizes)}I", *record_sizes)
    b += bytes(-len(b) % 8)
    return b


def build_chunk(prefix: bytes, blocks: list[bytes], record_sizes: list[int], level: int = 6):
    """-> (stored chunk bytes, prefix_len, index_len)"""
    comp = [zlib.compress(b, level) for b in blocks]
    idx = zlib.compress(build_index([len(c) for c in comp], record_sizes), level)
    return prefix + b"".join(comp) + idx, len(prefix), len(idx)


def slice_blocks(records: list[bytes], block_size: int = 65536) -> list[bytes]:
    """The REAL layout (verified on a 332 MB chunk): the concatenated record stream cut into exact block_size slices
    (last one shorter); records may span block boundaries. The directory prefix holds block_size."""
    blob = b"".join(records)
    return [blob[i:i + block_size] for i in range(0, len(blob), block_size)] or [b""]


def pack_records(records: list[bytes], block_size: int = 65536) -> list[bytes]:
    """Group whole records into blocks of ~block_size (a record never spans blocks in our writer)."""
    blocks, cur = [], bytearray()
    for r in records:
        if cur and len(cur) + len(r) > block_size:
            blocks.append(bytes(cur))
            cur = bytearray()
        cur += r
    if cur:
        blocks.append(bytes(cur))
    return blocks


# ------------------------------------------------------------- raw records ---
def _al(b: bytearray, n: int):
    b += bytes(-len(b) % n)


def triangle_strip_record(vertices: list[tuple], strips: list[list[int]], normals: list[tuple] | None = None) -> bytes:
    """Raw-mode triangle-strip geometry record (type 94). vertices: [(x,y,z)], strips: lists of vertex indices
    (a strip of k indices yields k-2 triangles; alternate winding as usual for strips)."""
    n = len(vertices)
    if n > 65535 or any(i >= n for s in strips for i in s):
        raise ValueError("raw records use u16 indices")
    b = bytearray()
    b += struct.pack("<IIII", 100, 94, 101, 60)
    b += struct.pack("<i", n)
    for v in vertices:
        b += struct.pack("<fff", *v)
    if normals:
        if len(normals) != n:
            raise ValueError("one normal per vertex")
        b += struct.pack("<I", 102) + struct.pack("<I", 59) + struct.pack("<i", n)
        for v in normals:
            b += struct.pack("<fff", *v)
        b += struct.pack("<II", 0, 0)
    else:
        b += struct.pack("<III", 0, 0, 0)
    _al(b, 4)
    b += struct.pack("<i", len(strips))
    b += b"".join(struct.pack("<H", len(s)) for s in strips)
    _al(b, 4)
    flat = [i for s in strips for i in s]
    b += struct.pack("<i", len(flat))
    b += b"".join(struct.pack("<H", i) for i in flat)
    _al(b, 8)
    return bytes(b)


def box_record(sx: float, sy: float, sz: float, normals: bool = False) -> bytes:
    """Axis aligned box centred at the origin as 6 quads (one strip of 4 vertices each, 24 vertices)."""
    x, y, z = sx / 2, sy / 2, sz / 2
    faces = [  # (normal, 4 corners in strip order)
        ((0, 0, 1), [(-x, -y, z), (x, -y, z), (-x, y, z), (x, y, z)]),
        ((0, 0, -1), [(-x, y, -z), (x, y, -z), (-x, -y, -z), (x, -y, -z)]),
        ((1, 0, 0), [(x, -y, -z), (x, y, -z), (x, -y, z), (x, y, z)]),
        ((-1, 0, 0), [(-x, y, -z), (-x, -y, -z), (-x, y, z), (-x, -y, z)]),
        ((0, 1, 0), [(x, y, -z), (-x, y, -z), (x, y, z), (-x, y, z)]),
        ((0, -1, 0), [(-x, -y, -z), (x, -y, -z), (-x, -y, z), (x, -y, z)]),
    ]
    verts, norms, strips = [], [], []
    for nrm, quad in faces:
        base = len(verts)
        verts += quad
        norms += [nrm] * 4
        strips.append([base, base + 1, base + 2, base + 3])
    return triangle_strip_record(verts, strips, norms if normals else None)


if __name__ == "__main__":
    # round trip of the real chunk: split -> rebuild with the original block boundaries -> identical bytes?
    import json, os
    BS = chr(92)
    T = os.path.join(os.environ["TEMP"], "real", "x")
    m = json.load(open(T + "/manifest.json"))
    c = next(c for c in m["chunks"] if c["name"].endswith(BS + "LcOpNwdGeometry"))
    raw = open(os.path.join(T, c["file"]), "rb").read()
    prefix, blocks, recs, bsz, rsz = split_chunk(raw, c["prefix"], c["index"])
    print(f"prefix={prefix.hex()} blocks={len(blocks)} records={len(recs)} total={sum(rsz)} B")
    out, pl, il = build_chunk(prefix, blocks, rsz)
    print("rebuilt identical:", out == raw, "prefix_len", pl, "index_len", il, "orig index_len", c["index"])


# ------------------------------------------------------------------------------------------------------------
# Compressed-mode (GeometryCompress flags 0x3f) type-94 records: the form real Navisworks files use. Grammar learnt
# from real records (verified by decode coverage, see decode94 / selftest_geometry).
#   u32 100, 94, 101, 60 | i32 n (vertex slots) | u32 floats, floats x f32 (palette; palette = floats/3 entries)
#   | [index RLE x n if palette < n]  | 3 x slot { u32 objectId (0 none) [, u32 type, payload] }
#   normals (type 100): i32 -n, u32 palette P, ceil(P*3*bits/32) u32 words, [index RLE x n if P < n]
#        index values -1..-6 (opcodes 0xD0..0xD5) = axis aligned normals (+X,-X,+Y,-Y,+Z,-Z order not yet verified)
#   strips: i32 groups (|v|), u8 bits, u32 count, packed lengths (MSB first in u32 words), i32 nIdx, [index RLE]
#   padded to 4 / tail zero.
# ------------------------------------------------------------------------------------------------------------
class _R:
    def __init__(self, d):
        self.d, self.p = d, 0

    def al(self, n):
        self.p = (self.p + n - 1) & ~(n - 1)

    def u32(self):
        self.al(4); v = struct.unpack_from("<I", self.d, self.p)[0]; self.p += 4; return v

    def i32(self):
        self.al(4); v = struct.unpack_from("<i", self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        self.al(4); v = struct.unpack_from("<f", self.d, self.p)[0]; self.p += 4; return v

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v


def rle_decode(r: _R, n: int):
    out, highest = [], -1
    while len(out) < n:
        t = r.u8()
        if t < 64:
            for _ in range(t + 1):
                highest += 1; out.append(highest)
        elif t < 128:
            out.append(out[-(t - 64)])
        elif t < 192:
            out += [out[-1]] * (t - 127)
        elif 0xD0 <= t < 0xE0:
            out.append(-(t - 0xD0 + 1))
        elif 0xE0 <= t < 0xF0:
            d = (t - 0xE0) * 256 + r.u8(); out.append(out[-d])
        elif t == 0xF0:
            v = r.u8(); v = (v << 8) | r.u8(); out.append(v)
        elif t == 0xF1:
            v = 0
            for _ in range(4):
                v = (v << 8) | r.u8()
            out.append(v)
        else:
            raise ValueError(f"index opcode {t:#x}")
        if len(out) > n:
            raise ValueError("index run overrun")
    r.al(4)
    return out


def decode94(rec: bytes, normal_bits: int = 8, types=(94,)):
    """-> dict for the simple compressed form, or None if the record uses another form (quantised coords, UV, ...).
    types: record types accepted (94 triangle strips; 95 = line strips has the same grammar)."""
    r = _R(rec)
    head = (r.u32(), r.u32(), r.u32(), r.u32())
    if head[0] != 100 or head[1] not in types or head[2:] != (101, 60):
        return None
    n = r.i32()
    if n < 0:
        return None
    floats = r.u32()
    if floats % 3:
        return None
    coords = [r.f32() for _ in range(floats)]
    P = floats // 3
    cidx = rle_decode(r, n) if P < n else list(range(n))
    attrs = []
    for slot in range(3):
        oid = r.u32()
        if not oid:
            attrs.append(None); continue
        typ = r.u32()
        if typ != 100:
            return None
        slots = r.i32()
        if slots != -n:
            return None
        pal = r.u32()
        words = (pal * 3 * normal_bits + 31) // 32
        packed = rec[r.p:r.p + 4 * words]; r.p += 4 * words
        aidx = rle_decode(r, n) if pal < n else list(range(n))
        attrs.append({"oid": oid, "type": typ, "pal": pal, "packed": packed, "idx": aidx})
    groups = abs(r.i32())
    bits = r.u8(); cnt = struct.unpack_from("<I", rec, (r.p + 3) & ~3)[0]; r.al(4); r.p += 4
    nw = (bits * cnt + 31) // 32
    words_ = struct.unpack_from(f"<{nw}I", rec, r.p); r.p += 4 * nw
    lengths = []
    for i in range(cnt):
        b = i * bits; w = b // 32; sh = b % 32
        v = words_[w]
        if sh + bits <= 32:
            v >>= (32 - sh - bits)
        else:
            v = ((v << 32) | words_[w + 1]) >> (64 - sh - bits)
        lengths.append(v & ((1 << bits) - 1))
    if head[1] == 95:                       # line strips end after the lengths (no explicit index count)
        sidx = None
    else:
        nidx = abs(r.i32())
        sidx = rle_decode(r, nidx) if nidx else None
    if len(rec) - r.p >= 8 or any(rec[r.p:]):
        return None
    return {"n": n, "coords": coords, "cidx": cidx, "attrs": attrs, "groups": groups, "lengths": lengths, "sidx": sidx}


def first_records(raw: bytes, prefix_len: int, index_len: int, max_blocks: int = 200, start_block: int = 0):
    """Memory-friendly: records contained in blocks [start_block, start_block+max_blocks) (records never span blocks
    in files written by our writer; in the real file a record may span blocks - such a trailing partial is skipped)."""
    idx, _ = nt.inflate_stream(raw[len(raw) - index_len:])
    count, nb = struct.unpack_from("<II", idx, 0)
    bsz = list(struct.unpack_from(f"<{nb}I", idx, 8))
    rsz = list(struct.unpack_from(f"<{count}I", idx, 8 + 4 * nb))
    pos = prefix_len + sum(bsz[:start_block])
    # records before start_block
    done = 0; ri = 0
    # find first record index starting at the byte offset of start_block (decoded bytes needed): inflate preceding blocks lazily
    skip_bytes = 0
    p = prefix_len
    for n in bsz[:start_block]:
        skip_bytes += len(nt.inflate_stream(raw[p:p + n])[0]); p += n
    acc = 0
    while ri < count and acc < skip_bytes:
        acc += rsz[ri]; ri += 1
    blob = bytearray(); p = pos
    for n in bsz[start_block:start_block + max_blocks]:
        blob += nt.inflate_stream(raw[p:p + n])[0]; p += n
    out, q = [], 0
    while ri < count and q + rsz[ri] <= len(blob):
        out.append(bytes(blob[q:q + rsz[ri]])); q += rsz[ri]; ri += 1
    return out


# ---------------------------------------------------------------- compressed-mode encoder (simple form) ---
def _rle_axis(codes):
    """Normal axis codes (-1..-6): the real writer emits one literal opcode 0xD0+(k-1) per vertex (verified byte-identical)."""
    out = bytearray(0xD0 + (-c - 1) for c in codes)
    out += bytes(-len(out) % 4)
    return bytes(out)


def _pack_lengths(lengths):
    bits = (max(lengths).bit_length() + 1) if lengths else 1      # real writer: one extra bit (signed field)
    nbits = bits * len(lengths); words = (nbits + 31) // 32
    acc = 0
    for v in lengths:
        acc = (acc << bits) | v
    acc <<= words * 32 - nbits
    ws = [(acc >> (32 * (words - 1 - w))) & 0xFFFFFFFF for w in range(words)]
    return bits, ws


def encode94(coords: list[tuple], axis_codes: list[int] | None, lengths: list[int], strip_idx: list[int] | None = None) -> bytes:
    """Triangle-strip record in the compressed-mode form (palette == vertex slots, axis-code normals, packed lengths).
    axis_codes: per vertex in -1..-6 (+X,-X,+Y,-Y,+Z,-Z) or None for no normals. strip_idx None = implicit sequential."""
    n = len(coords)
    b = bytearray(struct.pack("<IIII", 100, 94, 101, 60))
    b += struct.pack("<i", n) + struct.pack("<I", 3 * n)
    for v in coords:
        b += struct.pack("<fff", *v)
    b += struct.pack("<I", 0)                                   # slot 0: no UV
    if axis_codes:
        b += struct.pack("<III", 102, 100, 0) [:8] + struct.pack("<i", -n) + struct.pack("<I", 0)
        b += _rle_axis(axis_codes)
    else:
        b += struct.pack("<I", 0)
    b += struct.pack("<I", 0)                                   # slot 2: no colours
    b += struct.pack("<i", len(lengths))
    bits, ws = _pack_lengths(lengths)
    b += bytes([bits]); b += bytes(-len(b) % 4)
    b += struct.pack("<I", len(lengths)) + struct.pack(f"<{len(ws)}I", *ws)
    if strip_idx is None:
        b += struct.pack("<i", 0)
    else:
        raise NotImplementedError("explicit strip indices")
    b += bytes(-len(b) % 8)
    return bytes(b)


def box94(sx: float, sy: float, sz: float) -> bytes:
    """Box (24 vertices, 6 quad strips) in the real compressed-mode form."""
    x, y, z = sx / 2, sy / 2, sz / 2
    faces = [  # (axis code, 4 vertices in strip order, outward-facing winding)
        (-1, [(x, -y, -z), (x, y, -z), (x, -y, z), (x, y, z)]),
        (-2, [(-x, y, -z), (-x, -y, -z), (-x, y, z), (-x, -y, z)]),
        (-3, [(x, y, -z), (-x, y, -z), (x, y, z), (-x, y, z)]),
        (-4, [(-x, -y, -z), (x, -y, -z), (-x, -y, z), (x, -y, z)]),
        (-5, [(-x, -y, z), (x, -y, z), (-x, y, z), (x, y, z)]),
        (-6, [(-x, y, -z), (x, y, -z), (-x, -y, -z), (x, -y, -z)]),
    ]
    coords, codes = [], []
    for code, quad in faces:
        coords += quad; codes += [code] * 4
    return encode94(coords, codes, [4] * 6)


# ------------------------------------------------------------------ general triangle meshes (compressed-mode form) ---
def rle_encode(idx):
    """Inverse of rle_decode using sequential-new runs (0x00-0x3F), 16-bit (0xF0) and 32-bit (0xF1) literals."""
    out = bytearray(); highest = -1; i = 0; n = len(idx)
    while i < n:
        v = idx[i]
        if v == highest + 1:
            run = 1
            while run < 64 and i + run < n and idx[i + run] == highest + 1 + run:
                run += 1
            out.append(run - 1); highest += run; i += run
        elif 0 <= v < 65536:
            out += bytes((0xF0, v >> 8, v & 255)); i += 1
        else:
            out += b"\xF1" + struct.pack(">I", v); i += 1
    out += bytes(-len(out) % 4)
    return bytes(out)


MESH_WELD = True      # False: one palette entry per triangle corner (identity indices, no RLE stream)


def mesh94(positions, tris, normals=None):
    """Triangle mesh -> compressed-mode type-94 record (palette of float positions, RLE vertex indices, strips of length 3).
    positions: [(x,y,z)], tris: [(a,b,c)] indices into positions, normals: optional per-vertex [(nx,ny,nz)] (written as float
    normals, attribute type 59). Vertices are renumbered in first-use order so index runs compress."""
    remap, pos2, nrm2, cidx = {}, [], [], []
    for t in tris:
        for v in t:
            j = remap.get(v) if MESH_WELD else None
            if j is None:
                j = len(pos2)
                if MESH_WELD:
                    remap[v] = j
                pos2.append(positions[v])
                if normals is not None:
                    nrm2.append(normals[v])
            cidx.append(j)
    n = len(cidx); P = len(pos2)
    b = bytearray(struct.pack("<IIII", 100, 94, 101, 60))
    b += struct.pack("<i", n) + struct.pack("<I", 3 * P)
    b += struct.pack(f"<{3 * P}f", *[c for p in pos2 for c in p])
    if P < n:
        b += rle_encode(cidx)
    b += struct.pack("<I", 0)                                       # slot 0: no UV
    if normals is not None:
        b += struct.pack("<II", 102, 59) + struct.pack("<i", n) + struct.pack("<I", 3 * P)
        b += struct.pack(f"<{3 * P}f", *[c for p in nrm2 for c in p])
        if P < n:
            b += rle_encode(cidx)
    else:
        b += struct.pack("<I", 0)
    b += struct.pack("<I", 0)                                       # slot 2: no colours
    T = len(tris)
    b += struct.pack("<i", T)
    bits, ws = _pack_lengths([3] * T)
    b += bytes([bits]); b += bytes(-len(b) % 4)
    b += struct.pack("<I", T) + struct.pack(f"<{len(ws)}I", *ws)
    b += struct.pack("<i", 0)                                       # implicit sequential strip indices
    b += bytes(-len(b) % 8)
    return bytes(b)


def lines95_np(v, seg) -> bytes:
    """Line record (type 95) in the compressed-mode form Navisworks writes (verified on 600 records of nugget.nwd):
    header 100,95,101,60 | i32 n | u32 3n floats (one palette entry per segment end, identity indices) | three zero
    attribute slots | i32 groups | u8 bits (+pad) | u32 count + MSB-first packed strip lengths | no index count; padded
    to 8. Every segment is its own 2-vertex strip, as Navisworks writes them. v: (N,3) float, seg: (M,2) int."""
    import numpy as np
    seg = np.asarray(seg, dtype=np.int64).reshape(-1, 2)
    v = np.asarray(v, dtype=np.float32)
    M = len(seg)
    corners = v[seg.reshape(-1)]                                   # (2M, 3)
    n = 2 * M
    bits = 3                                                       # strip length 2 -> bit_length 2 + 1 (real writer)
    vals = np.full(M, 2, dtype=np.uint64)
    bitarr = ((vals[:, None] >> np.arange(bits - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)
    pad = (-len(bitarr)) % 32
    bitarr = np.concatenate([bitarr, np.zeros(pad, dtype=np.uint8)]).reshape(-1, 32)
    words = (bitarr.astype(np.uint64) << np.arange(31, -1, -1, dtype=np.uint64)).sum(axis=1).astype("<u4")
    b_ = bytearray(struct.pack("<IIII", 100, 95, 101, 60))
    b_ += struct.pack("<i", n) + struct.pack("<I", 3 * n)
    b_ += corners.astype("<f4").tobytes()
    b_ += struct.pack("<III", 0, 0, 0)                             # no UV / normals / colours
    b_ += struct.pack("<i", M)
    b_ += bytes([bits]); b_ += bytes(-len(b_) % 4)
    b_ += struct.pack("<I", M) + words.tobytes()
    b_ += bytes(-len(b_) % 8)
    return bytes(b_)


def mesh94_flat_np(v, tri) -> bytes:
    """Fast flat-shaded triangle mesh record with numpy: one palette entry per triangle corner (identity indices, no RLE),
    per-face float normals (attribute type 59), strips of length 3. v: (N,3) float, tri: (T,3) int."""
    import numpy as np
    tri = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
    v = np.asarray(v, dtype=np.float32)
    T = len(tri)
    corners = v[tri.reshape(-1)]                                   # (3T, 3)
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    fn = np.cross(b - a, c - a).astype(np.float64)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-20
    fn[ok] /= ln[ok][:, None]
    fn[~ok] = (0.0, 0.0, 1.0)
    normals = np.repeat(fn.astype(np.float32), 3, axis=0)
    n = 3 * T
    bits = 3                                                       # max strip length 3 -> bit_length 2 + 1 (real writer)
    vals = np.full(T, 3, dtype=np.uint64)
    bitarr = ((vals[:, None] >> np.arange(bits - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)
    pad = (-len(bitarr)) % 32
    bitarr = np.concatenate([bitarr, np.zeros(pad, dtype=np.uint8)]).reshape(-1, 32)
    words = (bitarr.astype(np.uint64) << np.arange(31, -1, -1, dtype=np.uint64)).sum(axis=1).astype("<u4")
    b_ = bytearray(struct.pack("<IIII", 100, 94, 101, 60))
    b_ += struct.pack("<i", n) + struct.pack("<I", 3 * n)
    b_ += corners.astype("<f4").tobytes()
    b_ += struct.pack("<I", 0)                                     # slot 0: no UV
    b_ += struct.pack("<II", 102, 59) + struct.pack("<i", n) + struct.pack("<I", 3 * n)
    b_ += normals.astype("<f4").tobytes()
    b_ += struct.pack("<I", 0)                                     # slot 2: no colours
    b_ += struct.pack("<i", T)
    b_ += bytes([bits]); b_ += bytes(-len(b_) % 4)
    b_ += struct.pack("<I", T) + words.tobytes()
    b_ += struct.pack("<i", 0)
    b_ += bytes(-len(b_) % 8)
    return bytes(b_)
