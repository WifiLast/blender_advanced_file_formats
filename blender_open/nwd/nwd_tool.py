#!/usr/bin/env python3
"""nwd_tool.py - read ("import") and write ("export") the Navisworks .nwd *container*.

Independent implementation. Facts come from (a) our own Ghidra decompiles of
lcutil.dll / lcodyplugin.dll / lcuzlib.dll (see NWD_FORMAT.md) and (b) cross-checking the
container layout against the third-party reader in external_other/nwd-reader
(PolyForm Noncommercial - no code was copied from it, it was used as a reference for the
file layout only).

Scope: container only - header, footer, zlib-compressed chunk directory, chunk payload
bytes. Optional decoding of chunk payload *envelopes* (zlib blocks, Blowfish segments).
It does NOT interpret scene graph / geometry / property data inside the chunks.

Layout (container revision "lichunk-008", Navisworks 2025 sample layout):
  0      52-byte header  "#LcUStream-V1.1 binary double release C lichunk-008\\n"
  52     chunk payloads, back to back (no gaps)
  at     directory stream: config line "...utf8 navisworks:<ver> lizlib\\n", padding, one zlib stream
         inflated: u32 0xFFFFFFFF, u32 count, per chunk
             string name, u32 flags, u64 offset, u64 size, u32 prefix, [u32 index if prefix], align 8
         (all values aligned to their size relative to the start of the inflated data)
  size-68  8 opaque trailer bytes (checksum; not validated by other readers; we only report it)
  size-60  u64 'at'
  size-52  copy of the 52-byte header
Chunk flags: 0 raw, 1 = [prefix bytes] + concatenated zlib streams, 3 = Blowfish segments + zlib.

Commands: info | list | verify | extract [--decode] | pack | selftest
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zlib
from pathlib import Path

HEADER_PREFIX = b"#LcUStream-V1.1 binary double release C lichunk-"        # 48 bytes
HEADER_LEN = 52
FOOTER_LEN = 52 + 8          # u64 pointer + header copy
TRAILER_LEN = 8
DIR_MARKER = 0xFFFFFFFF
# 32-byte keys found in the original DLLs
CHECKSUM_KEY = bytes.fromhex("5b515893711d22e3d04def350953a42550f01329064615cce3b758982927c4c1")  # lcutil.dll VA 0x18010f6e8
CHUNK_KEY = bytes.fromhex("20d9afe080f11752de9dd91243de1a98581e2db04d782fe469218e52d7051f07")  # lcodyplugin.dll file off 0x1dbc40 (LcOpNwdSerial)


# ------------------------------------------------------------------ blowfish ---
_PI: list[int] | None = None


def _pi_words(n: int = 18 + 4 * 256) -> list[int]:
    bits = n * 32 + 64

    def atan_inv(x: int) -> int:
        term = (1 << bits) // x
        total, k, sign, x2 = term, 1, -1, x * x
        while term:
            term //= x2
            k += 2
            total += sign * (term // k)
            sign = -sign
        return total

    frac = (16 * atan_inv(5) - 4 * atan_inv(239) - (3 << bits)) >> 64
    return [(frac >> (32 * (n - 1 - i))) & 0xFFFFFFFF for i in range(n)]


class Blowfish:
    """Blowfish on (L, R) 32-bit word pairs. NWD feeds it little-endian words."""

    def __init__(self, key: bytes):
        global _PI
        if _PI is None:
            _PI = _pi_words()
        self.P = list(_PI[:18])
        self.S = [list(_PI[18 + 256 * i:18 + 256 * (i + 1)]) for i in range(4)]
        j = 0
        for i in range(18):
            v = 0
            for _ in range(4):
                v = (v << 8) | key[j]
                j = (j + 1) % len(key)
            self.P[i] ^= v
        l = r = 0
        for i in range(0, 18, 2):
            l, r = self.encrypt(l, r)
            self.P[i], self.P[i + 1] = l, r
        for box in self.S:
            for i in range(0, 256, 2):
                l, r = self.encrypt(l, r)
                box[i], box[i + 1] = l, r

    def _f(self, x: int) -> int:
        s = self.S
        return ((((s[0][x >> 24] + s[1][(x >> 16) & 255]) & 0xFFFFFFFF) ^ s[2][(x >> 8) & 255])
                + s[3][x & 255]) & 0xFFFFFFFF

    def encrypt(self, l: int, r: int):
        p = self.P
        for i in range(16):
            l ^= p[i]
            r ^= self._f(l)
            l, r = r, l
        l, r = r, l
        return l ^ p[17], r ^ p[16]

    def decrypt(self, l: int, r: int):
        p = self.P
        for i in range(17, 1, -1):
            l ^= p[i]
            r ^= self._f(l)
            l, r = r, l
        l, r = r, l
        return l ^ p[0], r ^ p[1]


def adler32(data: bytes) -> int:
    return zlib.adler32(data) & 0xFFFFFFFF


# --------------------------------------------------------- chunk envelopes ---
def inflate_stream(buf: bytes, limit: int = (1 << 30)) -> tuple[bytes, int]:
    d = zlib.decompressobj()
    out = d.decompress(buf, limit)
    if not d.eof:
        raise ValueError("truncated zlib stream (or decoded size above limit)")
    return out, len(buf) - len(d.unused_data)


def inflate_blocks(buf: bytes) -> bytes:
    out, pos = [], 0
    while pos < len(buf):
        data, used = inflate_stream(buf[pos:])
        if not used:
            raise ValueError("empty zlib block")
        out.append(data)
        pos += used
    return b"".join(out)


def cipher_decode(buf: bytes) -> bytes:
    """flags==3: [enc(len, adler32)] [enc 8-byte blocks, padded] ... repeated."""
    if not buf or len(buf) % 8:
        raise ValueError("cipher chunk is not 8-byte aligned")
    bf = Blowfish(CHUNK_KEY)
    pos, out = 0, bytearray()

    def pair():
        nonlocal pos
        a, b = struct.unpack_from("<II", buf, pos)
        pos += 8
        return bf.decrypt(a, b)

    while pos < len(buf):
        length, ck = pair()
        padded = (length + 7) & ~7
        if length == 0 or padded > len(buf) - pos:
            raise ValueError("bad cipher segment length")
        seg = bytearray()
        for _ in range(padded // 8):
            seg += struct.pack("<II", *pair())
        seg = bytes(seg[:length])
        if adler32(seg) != ck:
            raise ValueError("cipher segment checksum mismatch")
        out += seg
    return bytes(out)


def cipher_encode(data: bytes, segment: int = 1 << 16, split_zlib_header: bool = False) -> bytes:
    """Inverse of cipher_decode. split_zlib_header=True reproduces the real writer for small payloads:
    the 2-byte zlib header is its own segment, the rest of the zlib stream the second (verified on the
    LcOpNwdGeometryCompress chunk of a real file)."""
    bf = Blowfish(CHUNK_KEY)
    out = bytearray()
    segs = []
    if split_zlib_header and len(data) > 2:
        segs = [data[:2]] + [data[i:i + segment] for i in range(2, len(data), segment)]
    else:
        segs = [data[i:i + segment] for i in range(0, len(data), segment)]
    for seg in segs:
        out += struct.pack("<II", *bf.encrypt(len(seg), adler32(seg)))
        seg = seg + bytes(-len(seg) % 8)
        for j in range(0, len(seg), 8):
            out += struct.pack("<II", *bf.encrypt(*struct.unpack_from("<II", seg, j)))
    return bytes(out)


def decode_chunk(raw: bytes, e: dict) -> bytes:
    f = e["flags"]
    if f == 0:
        return raw
    if f == 1:
        return inflate_blocks(raw[e["prefix"]:])
    if f == 3:
        if e["prefix"] or e["index"]:
            raise ValueError("cipher chunk with prefix/index not supported")
        comp = cipher_decode(raw)
        out, used = inflate_stream(comp)
        if used != len(comp):
            raise ValueError("cipher payload has trailing bytes after zlib stream")
        return out
    raise NotImplementedError(f"chunk flags {f}")


# --------------------------------------------------------------- directory ---
class Cur:
    def __init__(self, d: bytes):
        self.d, self.p = d, 0

    def align(self, n):
        self.p = (self.p + n - 1) & ~(n - 1)
        if self.p > len(self.d):
            raise ValueError("alignment outside directory")

    def _take(self, n):
        if n > len(self.d) - self.p:
            raise ValueError("truncated directory")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def u32(self):
        self.align(4)
        return struct.unpack("<I", self._take(4))[0]

    def u64(self):
        self.align(8)
        return struct.unpack("<Q", self._take(8))[0]

    def string(self):
        n = self.u32()
        if n == 0xFFFFFFFF:
            return ""
        return self._take(n).rstrip(b"\0").decode("latin-1")


def _pad(buf: bytearray, n: int):
    buf += bytes(-len(buf) % n)


def build_directory(entries: list[dict]) -> bytes:
    b = bytearray(struct.pack("<II", DIR_MARKER, len(entries)))
    for e in entries:
        nb = e["name"].encode("latin-1")
        _pad(b, 4)
        b += struct.pack("<I", len(nb)) + nb
        _pad(b, 4)
        b += struct.pack("<I", e["flags"])
        _pad(b, 8)
        b += struct.pack("<QQ", e["offset"], e["size"])
        b += struct.pack("<I", e["prefix"])
        if e["prefix"]:
            b += struct.pack("<I", e["index"])
        _pad(b, 8)
    return bytes(b)


def guess_trailer(entries: list[dict]) -> bytes:
    """Directory checksum, verified byte-for-byte on a real 2024 NWD (navisworks:448):
    Adler-32 over, per chunk, name (no NUL; one NUL if empty) + u32 flags + u64 offset + u64 size
    + u32 prefix + u32 index (the last two ALWAYS, even when 0), then Blowfish(CHECKSUM_KEY)
    applied to the word pair (w, w) and written as two little-endian u32."""
    a, b = 1, 0
    for e in entries:
        for part in (e["name"].encode("latin-1") or b"\0", struct.pack("<I", e["flags"]),
                     struct.pack("<QQ", e["offset"], e["size"]),
                     struct.pack("<II", e["prefix"], e.get("index", e.get("index_bytes", 0)))):
            for c in part:
                a = (a + c) % 65521
                b = (b + a) % 65521
    w = (b << 16) | a
    return struct.pack("<II", *Blowfish(CHECKSUM_KEY).encrypt(w, w))


# ------------------------------------------------------------------- parse ---
def parse(data: bytes) -> dict:
    if len(data) < HEADER_LEN + FOOTER_LEN + TRAILER_LEN or not data.startswith(b"#LcUStream-V1.1"):
        raise ValueError("not an LcUStream container")
    head = data[:HEADER_LEN]
    if b"lichunk-007" in head:
        raise NotImplementedError("legacy container revision 007 (different directory layout)")
    if b"lichunk-008" not in head:
        raise ValueError(f"unsupported container revision: {head!r}")
    foot = len(data) - HEADER_LEN
    if data[foot:] != head:
        raise ValueError("missing directory footer (header copy mismatch)")
    at = struct.unpack_from("<Q", data, foot - 8)[0]
    dir_end = foot - 8 - TRAILER_LEN
    if at == 0:
        raise NotImplementedError("directory pointer is 0 (lizlib writer variant) - not supported")
    if not (HEADER_LEN <= at < dir_end):
        raise ValueError(f"directory pointer {at} outside file")
    nl = data.find(b"\n", at, at + 256)
    if nl < 0:
        raise ValueError("directory config line not found")
    cfg = data[at:nl].decode("latin-1")
    m = re.search(r"navisworks:(\d+)", cfg)
    nwd_version = int(m.group(1)) if m else None
    z = next((i for i in range(nl + 1, min(nl + 33, dir_end - 1))
              if (data[i] & 15) == 8 and (data[i] * 256 + data[i + 1]) % 31 == 0), None)
    if z is None:
        raise ValueError("zlib stream of the directory not found")
    raw_dir, used = inflate_stream(data[z:dir_end])
    problems = []
    if z + used != dir_end:
        problems.append(f"directory zlib stream ends at {z + used}, expected {dir_end}")
    c = Cur(raw_dir)
    if c.u32() != DIR_MARKER:
        raise ValueError("directory marker missing")
    count = c.u32()
    if count > 1_000_000:
        raise ValueError("implausible chunk count")
    entries, prev = [], HEADER_LEN
    for i in range(count):
        name = c.string()
        flags = c.u32()
        off = c.u64()
        size = c.u64()
        prefix = c.u32()
        index = c.u32() if prefix else 0
        c.align(8)
        if off != prev:
            problems.append(f"chunk {i} offset {off} != previous end {prev}")
        prev = off + size
        entries.append({"n": i, "name": name, "flags": flags, "offset": off, "size": size,
                        "prefix": prefix, "index_bytes": index})
    if prev != at:
        problems.append(f"chunks end at {prev}, directory starts at {at}")
    trailer = data[dir_end:dir_end + TRAILER_LEN]
    return {"header": head.decode("latin-1"), "config": cfg, "nwd_version": nwd_version,
            "directory_offset": at, "entries": entries, "trailer": trailer,
            "trailer_matches_guess": trailer == guess_trailer([_norm(e) for e in entries]), "problems": problems}


def _norm(e):      # parse() entry -> build()/decode_chunk() entry ('index' = index byte count)
    return {"name": e["name"], "flags": e["flags"], "offset": e["offset"], "size": e["size"],
            "prefix": e["prefix"], "index": e["index_bytes"]}


def build(chunks: list[dict], nwd_version: int, trailer: bytes | None = None) -> bytes:
    """chunks: [{name, data(bytes, stored form), flags, prefix, index}]"""
    head = HEADER_PREFIX + b"008\n"
    out = bytearray(head)
    entries = []
    for c in chunks:
        entries.append({"name": c["name"], "flags": c.get("flags", 0), "offset": len(out),
                        "size": len(c["data"]), "prefix": c.get("prefix", 0),
                        "index": c.get("index", 0)})
        out += c["data"]
    at = len(out)
    out += f"#LcUStream-V1.1 binary double release utf8 navisworks:{nwd_version} lizlib\n".encode()
    out += bytes(-(len(out) - at) % 8)      # 8-byte alignment is relative to the stream start (verified on a real file)
    out += zlib.compress(build_directory(entries), 6)
    out += trailer if trailer is not None else guess_trailer(entries)
    out += struct.pack("<Q", at) + head
    return bytes(out)


# --------------------------------------------------------------------- CLI ---
def _safe(n):
    return re.sub(r"[^A-Za-z0-9._-]", "_", n) or "unnamed"


def cmd_info(a, brief=False):
    data = Path(a.file).read_bytes()
    info = parse(data)
    if not brief:
        print(f"header           : {info['header'].strip()}")
        print(f"directory @      : {info['directory_offset']}   config: {info['config']}")
        print(f"navisworks ver   : {info['nwd_version']}")
        print(f"trailer          : {info['trailer'].hex()} (directory checksum "
              f"{'OK' if info['trailer_matches_guess'] else 'MISMATCH'})")
        print(f"chunks           : {len(info['entries'])}")
    for e in info["entries"]:
        print(f"  [{e['n']:3d}] {e['name']:<34} off={e['offset']:<10} size={e['size']:<10} "
              f"flags={e['flags']} prefix={e['prefix']} index={e['index_bytes']}")
    for p in info["problems"]:
        print("WARNING:", p)


def cmd_verify(a):
    try:
        info = parse(Path(a.file).read_bytes())
    except Exception as ex:                      # noqa: BLE001
        print("FAIL:", ex)
        return 1
    print("OK" if not info["problems"] else "FAIL: " + "; ".join(info["problems"]))
    return 0 if not info["problems"] else 1


def cmd_extract(a):
    data = Path(a.file).read_bytes()
    info = parse(data)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    man = {"nwd_version": info["nwd_version"], "trailer": info["trailer"].hex(), "chunks": []}
    for e in info["entries"]:
        ne = _norm(e)
        raw = data[e["offset"]:e["offset"] + e["size"]]
        fn = f"{e['n']:04d}_{_safe(e['name'])}.bin"
        (out / fn).write_bytes(raw)
        rec = {"name": e["name"], "file": fn, "flags": e["flags"], "prefix": e["prefix"],
               "index": e["index_bytes"]}
        if a.decode and e["flags"] in (1, 3):
            try:
                dec = decode_chunk(raw, ne)
                dfn = fn[:-4] + ".dec"
                (out / dfn).write_bytes(dec)
                rec["decoded"] = dfn
            except Exception as ex:               # noqa: BLE001
                rec["decode_error"] = str(ex)
        man["chunks"].append(rec)
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    bad = sum(1 for c in man["chunks"] if "decode_error" in c)
    print(f"extracted {len(man['chunks'])} chunks to {out}" + (f" ({bad} decode errors)" if bad else ""))


def cmd_pack(a):
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    chunks = [{"name": c["name"], "data": (d / c["file"]).read_bytes(), "flags": c["flags"],
               "prefix": c.get("prefix", 0), "index": c.get("index", 0)} for c in man["chunks"]]
    trailer = bytes.fromhex(man["trailer"]) if (a.keep_trailer and man.get("trailer")) else None
    blob = build(chunks, man.get("nwd_version") or 448, trailer)
    Path(a.outfile).write_bytes(blob)
    info = parse(blob)
    print(f"wrote {a.outfile}: {len(blob)} bytes, {len(chunks)} chunks"
          + (f"  WARNINGS: {info['problems']}" if info["problems"] else ""))


def cmd_selftest(_a):
    bf = Blowfish(bytes(8))
    assert bf.encrypt(0, 0) == (0x4EF99745, 0x6198DD78)
    assert Blowfish(bytes.fromhex("FEDCBA9876543210")).encrypt(0x01234567, 0x89ABCDEF) == (0x0ACEAB0F, 0xC6A0A28D)
    payload = bytes(range(256)) * 20
    assert cipher_decode(cipher_encode(payload, 1000)) == payload
    z1 = zlib.compress(b"hello" * 100)
    z2 = zlib.compress(b"world" * 50)
    chunks = [{"name": "LcOpNwdStats", "data": b"\x01\0\0\0\x00\0\0\0" + z1 + z2, "flags": 1, "prefix": 8, "index": 4},
              {"name": "LcOpNwdSerial", "data": cipher_encode(zlib.compress(payload)), "flags": 3},
              {"name": "raw", "data": b"abc", "flags": 0}]
    blob = build(chunks, 448)
    info = parse(blob)
    assert not info["problems"], info["problems"]
    es = info["entries"]
    assert [e["name"] for e in es] == ["LcOpNwdStats", "LcOpNwdSerial", "raw"]
    assert decode_chunk(blob[es[0]["offset"]:es[0]["offset"] + es[0]["size"]], _norm(es[0])) == b"hello" * 100 + b"world" * 50
    assert decode_chunk(blob[es[1]["offset"]:es[1]["offset"] + es[1]["size"]], _norm(es[1])) == payload
    assert es[0]["index_bytes"] == 4 and es[1]["prefix"] == 0
    print("selftest OK (Blowfish KATs, cipher round trip, build/parse, zlib/cipher chunk decode)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("info", "list", "verify"):
        sub.add_parser(n).add_argument("file")
    p = sub.add_parser("extract")
    p.add_argument("file"); p.add_argument("outdir")
    p.add_argument("--decode", action="store_true", help="also write inflated/decrypted payloads (*.dec)")
    p = sub.add_parser("pack")
    p.add_argument("dir"); p.add_argument("outfile")
    p.add_argument("--keep-trailer", action="store_true", help="reuse the original 8 trailer bytes from the manifest")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"info": lambda: cmd_info(a), "list": lambda: cmd_info(a, True), "verify": lambda: cmd_verify(a),
            "extract": lambda: cmd_extract(a), "pack": lambda: cmd_pack(a), "selftest": lambda: cmd_selftest(a)}[a.cmd]() or 0


if __name__ == "__main__":
    sys.exit(main())
