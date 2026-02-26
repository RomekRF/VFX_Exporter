"""libvfx.reader

Chunk-preserving VSFX/VFX reader.

This is the first "real" step away from the monolithic vfx2obj script:

- We parse the header (variable size depending on version).
- Then we iterate sections/chunks until EOF.
- Every chunk (known or unknown) is preserved verbatim.

Why this matters:
  Once we can round-trip VFX -> VfxFile -> VFX with *no changes*, we can safely
  start implementing authoring/writing features without breaking existing files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from .model import VfxChunk, VfxFile, VfxHeader


class VfxReadError(RuntimeError):
    pass


@dataclass
class _Bin:
    data: bytes
    ofs: int = 0

    def tell(self) -> int:
        return self.ofs

    def seek(self, ofs: int) -> None:
        self.ofs = ofs

    def read(self, n: int) -> bytes:
        b = self.data[self.ofs : self.ofs + n]
        if len(b) != n:
            raise VfxReadError(f"Unexpected EOF at {self.ofs}, need {n}")
        self.ofs += n
        return b

    def s32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]


def _parse_header(b: _Bin) -> VfxHeader:
    start = b.tell()
    magic = b.read(4)
    if magic != b"VSFX":
        raise VfxReadError("Not a VSFX/VFX file (missing 'VSFX' magic)")

    version = b.s32()
    flags: Optional[int] = None

    # This logic is intentionally identical to vfx2obj.parse_header so that
    # our offsets line up with existing, known-good behavior.
    if version >= 0x30008:
        flags = b.s32()

    end_frame = b.s32()

    # counts
    _ = b.s32()  # num_meshes
    _ = b.s32()  # num_lights
    _ = b.s32()  # num_dummies
    _ = b.s32()  # num_particle_systems
    _ = b.s32()  # num_spacewarps
    _ = b.s32()  # num_cameras

    if version >= 0x3000F:
        _ = b.s32()  # num_selsets

    if version >= 0x40000:
        _ = b.s32()  # num_materials
    if version >= 0x40002:
        _ = b.s32()  # num_mix_frames
    if version >= 0x40003:
        _ = b.s32()  # num_self_illumination_frames
    if version >= 0x40005:
        _ = b.s32()  # num_opacity_frames

    # faces + other totals
    _ = b.s32()  # num_faces
    _ = b.s32()  # num_mesh_material_indices
    _ = b.s32()  # num_vertex_normals
    _ = b.s32()  # num_adjacent_faces
    _ = b.s32()  # num_mesh_frames

    if version >= 0x3000D:
        _ = b.s32()  # num_uv_frames

    if version >= 0x30009:
        _ = b.s32()  # num_mesh_transform_frames
        _ = b.s32()  # num_mesh_transform_keyframe_lists
        _ = b.s32()  # num_mesh_translation_keys
        _ = b.s32()  # num_mesh_rotation_keys
        _ = b.s32()  # num_mesh_scale_keys

    _ = b.s32()  # num_light_frames
    _ = b.s32()  # num_dummy_frames
    _ = b.s32()  # num_part_sys_frames
    _ = b.s32()  # num_spacewarp_frames
    _ = b.s32()  # num_camera_frames

    if version >= 0x3000F:
        _ = b.s32()  # num_selset_objects

    end = b.tell()
    raw = b.data[start:end]
    return VfxHeader(version=version, flags=flags, end_frame=end_frame, raw=raw)


def _tag_from_int(i: int) -> str:
    # Sections are stored as int32, but often represent a 4-byte ASCII tag.
    # In vfx2obj, constants like 0x594D4D44 correspond to 'DMMY'.
    try:
        return struct.pack("<I", i & 0xFFFFFFFF).decode("ascii", errors="replace")
    except Exception:
        return "????"


def read_vfx(path: str) -> VfxFile:
    with open(path, "rb") as f:
        data = f.read()

    b = _Bin(data)
    header = _parse_header(b)
    data_len = len(data)

    chunks: list[VfxChunk] = []

    # Chunk layout (matching vfx2obj convention):
    #   int32 type
    #   int32 length
    #   body bytes of size (length - 4)
    # where 'length' includes the 4-byte type but does NOT include the 4-byte length field.
    while b.tell() < data_len:
        if b.tell() + 8 > data_len:
            break

        chunk_ofs = b.tell()
        type_int = b.s32()
        length = b.s32()

        if length < 8:
            raise VfxReadError(f"Bad section length {length} at offset {chunk_ofs}")

        body_len = length - 4
        if b.tell() + body_len > data_len:
            raise VfxReadError(
                f"Section overruns file: type={_tag_from_int(type_int)} len={length} at {chunk_ofs}"
            )

        body = b.read(body_len)

        chunks.append(
            VfxChunk(
                type_int=type_int,
                type_tag=_tag_from_int(type_int),
                length=length,
                body=body,
                offset=chunk_ofs,
            )
        )

    return VfxFile(header=header, chunks=chunks)
