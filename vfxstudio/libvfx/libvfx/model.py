from dataclasses import dataclass
from typing import List, Optional


# -----------------------------
# High-level, stable DTOs used by summarize_vfx
# -----------------------------

@dataclass
class VfxMaterial:
    name: str
    texture: Optional[str] = None

@dataclass
class VfxMeshInfo:
    name: str
    morph: bool
    frames: int
    fps: int
    flags_hex: str

@dataclass
class VfxSummary:
    path: str
    file_size: int
    version_hex: str
    end_frame: int
    meshes: List[VfxMeshInfo]
    materials: List[VfxMaterial]
    dummy_count: int
    particle_count: int


# -----------------------------
# Low-level, chunk-preserving model (v1 of the "real" reader/writer)
#
# Goal:
#   - Read any VSFX/VFX file.
#   - Preserve every section chunk verbatim (including unknown ones).
#   - Allow a byte-identical rewrite when nothing is modified.
#
# We intentionally keep this model minimal until we finish migrating the
# per-section parsers from vfx2obj into libvfx.
# -----------------------------


@dataclass
class VfxHeader:
    """Parsed header fields (best-effort) + the raw header bytes."""

    version: int
    flags: Optional[int]
    end_frame: int

    # Raw header bytes (exact bytes from file start to first section).
    raw: bytes


@dataclass
class VfxChunk:
    """A single section chunk, preserved verbatim."""

    type_int: int
    type_tag: str  # 4-char ASCII tag (best-effort)
    length: int    # length field from file (sec_len)
    body: bytes    # raw body bytes (length-4)
    offset: int    # file offset where this chunk starts (at type)

    @property
    def raw(self) -> bytes:
        # Stored layout:
        #   int32 type
        #   int32 length
        #   <body bytes> (length - 4)
        import struct
        return struct.pack('<ii', self.type_int, self.length) + self.body


@dataclass
class VfxFile:
    header: VfxHeader
    chunks: List[VfxChunk]
