#!/usr/bin/env python3
"""Round-trip a VFX file through libvfx reader/writer.

Usage:
  python vfxstudio/roundtrip.py path/to/file.vfx

Writes:
  <input>.roundtrip.vfx

Then prints whether the output is byte-identical to the input.
"""

from __future__ import annotations

import hashlib
import os
import sys

# Allow running from repo root without installation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIBVFX_ROOT = os.path.join(REPO_ROOT, "vfxstudio", "libvfx")
if LIBVFX_ROOT not in sys.path:
    sys.path.insert(0, LIBVFX_ROOT)

from libvfx.reader import read_vfx  # type: ignore
from libvfx.writer import write_vfx  # type: ignore


def _sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python vfxstudio/roundtrip.py <file.vfx>")
        return 2

    inp = os.path.abspath(argv[1])
    if not os.path.exists(inp):
        print(f"File not found: {inp}")
        return 2

    outp = inp + ".roundtrip.vfx"

    v = read_vfx(inp)
    write_vfx(v, outp)

    a = _sha256(inp)
    b = _sha256(outp)
    print(f"IN : {inp}\n     sha256={a}")
    print(f"OUT: {outp}\n     sha256={b}")
    print("IDENTICAL" if a == b else "DIFF")
    return 0 if a == b else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
