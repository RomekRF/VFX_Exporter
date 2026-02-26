"""libvfx.writer

Chunk-preserving VSFX/VFX writer.

Right now the only supported write mode is "lossless":

  - We emit the raw header bytes exactly as read.
  - Then we emit every chunk exactly as read.

This gives us a safety net: we can prove our reader understands the file layout
well enough to walk the entire file, and we can build tests around round-trip
identity before implementing any authoring features.
"""

from __future__ import annotations

from .model import VfxFile


def write_vfx(vfx: VfxFile, out_path: str) -> None:
    """Write a VfxFile back to disk.

    For now this is a strict lossless writer.
    """

    with open(out_path, "wb") as f:
        f.write(vfx.header.raw)
        for ch in vfx.chunks:
            f.write(ch.raw)
