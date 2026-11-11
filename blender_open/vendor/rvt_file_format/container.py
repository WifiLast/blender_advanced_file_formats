"""Read a Revit .rvt/.rfa container: OLE streams, block interleave, gzip members.

Read-only; needs `olefile`.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .. import olefile

DATA, GAP = 64896, 353          # payload bytes, interleaved foreign bytes (§5.1)
GZ = b"\x1f\x8b\x08"


def strip(raw: bytes) -> bytes:
    """Remove the 353 bytes that follow every 64896 data bytes."""
    out = bytearray()
    for p in range(0, len(raw), DATA + GAP):
        out += raw[p:p + DATA]
    return bytes(out)


@dataclass
class Member:
    offset: int        # in the stripped stream
    end: int
    gap: int           # marker bytes since the previous member
    data: bytes
    crc_ok: bool


def members(b: bytes):
    """Consecutive gzip members of a stripped stream."""
    # The deflate stream is fed through a growing window of a memoryview: slicing `b[g + 10:]` and reading
    # `unused_data` would copy the whole remaining stream for every member (quadratic on a 100 MB partition).
    mv = memoryview(b)
    n = len(b)
    off = 0
    window = 1 << 20
    while True:
        g = b.find(GZ, off)
        if g < 0:
            return
        start, w = g + 10, window
        while True:
            end_w = min(n, start + w)
            d = zlib.decompressobj(-15)
            try:
                out = d.decompress(mv[start:end_w])
            except zlib.error:
                d = None
                break
            if d.eof and (len(d.unused_data) >= 8 or end_w == n):
                break
            if end_w == n:
                break                       # truncated member
            w *= 2                          # member (or its trailer) does not fit the window yet
        if d is None or not d.eof or len(d.unused_data) < 8:
            off = g + 1
            continue
        crc, isz = struct.unpack_from("<II", d.unused_data, 0)
        end = end_w - len(d.unused_data) + 8
        yield Member(g, end, g - off, out, crc == zlib.crc32(out) and isz == len(out))
        off = end


class RvtFile:
    def __init__(self, path: str):
        self.path = path
        self.ole = olefile.OleFileIO(path)

    def close(self):
        self.ole.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def streams(self) -> list[str]:
        return ["/".join(e) for e in sorted(self.ole.listdir())]

    def raw(self, name: str) -> bytes:
        return self.ole.openstream(name).read()

    def stripped(self, name: str) -> bytes:
        return strip(self.raw(name))

    def members(self, name: str) -> list[Member]:
        return list(members(self.stripped(name)))

    def decoded(self, name: str) -> bytes:
        """Concatenated payload of all gzip members of a stream."""
        return b"".join(m.data for m in self.members(name))

    # --- uncompressed streams -------------------------------------------------

    def basic_file_info(self) -> dict:
        """Key/value text block of BasicFileInfo (§4)."""
        b = self.raw("BasicFileInfo")
        i = b.find("Worksharing".encode("utf-16le"))
        if i < 0:
            return {}
        info = {}
        # Lines are UTF-16LE with CRLF, but some values are written as narrow
        # ASCII + NUL (e.g. IsSingleUserCloudModel: "False\0"), which breaks
        # UTF-16 alignment. Split at the byte level and decode each value alone.
        for line in b[i:].split(b"\r\x00\n\x00"):
            sep = line.find(":\x00 \x00".encode("latin-1"))
            if sep < 0 or sep % 2:
                continue
            key = line[:sep].decode("utf-16le", "replace").strip()
            val = line[sep + 4:]
            if val.endswith(b"\r\n"):            # stream ends with a narrow CRLF
                val = val[:-2]
            even = val[:len(val) & ~1]          # ignore a dangling last byte
            if even and all(c == 0 for c in even[1::2]):
                v = even.decode("utf-16le", "replace")
            else:
                v = val.decode("latin-1")
            info[key] = v.strip().strip("\x00")
        return info

    def preview_png(self) -> bytes | None:
        if "RevitPreview4.0" not in self.streams():
            return None
        b = self.raw("RevitPreview4.0")
        i = b.find(b"\x89PNG")
        j = b.find(b"IEND", i)
        return b[i:j + 8] if i >= 0 and j > 0 else None

    def transmission_xml(self) -> str | None:
        if "TransmissionData" not in self.streams():
            return None
        b = self.raw("TransmissionData")
        return b[4:].decode("utf-16le", "replace")
