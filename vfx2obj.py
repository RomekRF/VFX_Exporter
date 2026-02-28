#!/usr/bin/env python3
"""
vfx2obj_optionA.py — Red Faction 1 (.vfx) -> glTF 2.0 (single-scene) converter

Option A feature set (from our chat logs):
- Morphed meshes export as glTF morph targets + a real weights animation track
- DMMY nodes export as empty nodes with proper parenting
- Constant scale on NON-morph meshes can be baked into geometry (to avoid scaling children in Blender)

This script is intentionally "format-first": it follows the vfx.ksy (Rafalh) field order and avoids
guessy post-passes.

NOTE: This is a *reader/exporter* only (no VFX writer yet).
"""

from __future__ import annotations

import os
import sys
import json
import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# ----------------------------
# Bin reader
# ----------------------------

class Bin:
    __slots__ = ("data", "ofs")
    def __init__(self, data: bytes):
        self.data = data
        self.ofs = 0

    def tell(self) -> int:
        return self.ofs

    def seek(self, ofs: int) -> None:
        self.ofs = ofs

    def read(self, n: int) -> bytes:
        b = self.data[self.ofs:self.ofs+n]
        if len(b) != n:
            raise EOFError(f"Unexpected EOF at {self.ofs}, need {n}")
        self.ofs += n
        return b

    def u8(self) -> int:
        return self.read(1)[0]

    def s8(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def s16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def s32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def strz(self) -> str:
        start = self.ofs
        while self.ofs < len(self.data) and self.data[self.ofs] != 0:
            self.ofs += 1
        if self.ofs >= len(self.data):
            raise EOFError("Unterminated string")
        s = self.data[start:self.ofs].decode("ascii", errors="replace")
        self.ofs += 1
        return s

# ----------------------------
# RF stored basis -> Blender basis
# vfx.ksy vec3:
#   stored.x = -max.x
#   stored.y =  max.z
#   stored.z = -max.y
# so max/blender:
#   x = -stored.x
#   y = -stored.z
#   z =  stored.y
# ----------------------------

def rf_to_blender(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    sx, sy, sz = v
    return (-sx, -sz, sy)

def read_vec3_rf(b: Bin) -> Tuple[float, float, float]:
    return (b.f32(), b.f32(), b.f32())

def read_vec3(b: Bin) -> Tuple[float, float, float]:
    return rf_to_blender(read_vec3_rf(b))

def read_uv(b: Bin) -> Tuple[float, float]:
    u = b.f32()
    v_stored = b.f32()
    return (u, -v_stored)

def read_quat_raw(b: Bin) -> Tuple[float, float, float, float]:
    return (b.f32(), b.f32(), b.f32(), b.f32())

# ----------------------------
# Quaternion/matrix helpers for basis conversion
# We treat stored quats as operating in stored basis, and convert:
#   R_blender = M * R_stored * M^T
# where M maps stored vectors to blender vectors: v_b = M v_s
# M is orthonormal (with reflection), so inverse is transpose.
# ----------------------------

# M = [[-1,0,0],[0,0,-1],[0,1,0]]
_M = (
    (-1.0, 0.0, 0.0),
    ( 0.0, 0.0,-1.0),
    ( 0.0, 1.0, 0.0),
)

def mat3_mul(a, b):
    # a,b: 3x3 tuples
    return tuple(
        tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )

def mat3_T(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))

def quat_to_mat3(q: Tuple[float, float, float, float]) -> Tuple[Tuple[float,float,float],Tuple[float,float,float],Tuple[float,float,float]]:
    x,y,z,w = q
    # standard right-handed quaternion to matrix
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return (
        (1.0-2.0*(yy+zz), 2.0*(xy - wz),     2.0*(xz + wy)),
        (2.0*(xy + wz),     1.0-2.0*(xx+zz), 2.0*(yz - wx)),
        (2.0*(xz - wy),     2.0*(yz + wx),     1.0-2.0*(xx+yy)),
    )

def mat3_to_quat(m: Tuple[Tuple[float,float,float],Tuple[float,float,float],Tuple[float,float,float]]) -> Tuple[float,float,float,float]:
    # robust-ish matrix->quat
    m00,m01,m02 = m[0]
    m10,m11,m12 = m[1]
    m20,m21,m22 = m[2]
    tr = m00+m11+m22
    if tr > 0.0:
        S = math.sqrt(tr+1.0)*2.0
        w = 0.25*S
        x = (m21 - m12)/S
        y = (m02 - m20)/S
        z = (m10 - m01)/S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22)*2.0
        w = (m21 - m12)/S
        x = 0.25*S
        y = (m01 + m10)/S
        z = (m02 + m20)/S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22)*2.0
        w = (m02 - m20)/S
        x = (m01 + m10)/S
        y = 0.25*S
        z = (m12 + m21)/S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11)*2.0
        w = (m10 - m01)/S
        x = (m02 + m20)/S
        y = (m12 + m21)/S
        z = 0.25*S
    return quat_norm((x,y,z,w))

def quat_norm(q: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    x,y,z,w = q
    n = math.sqrt(x*x+y*y+z*z+w*w)
    if n <= 0.0:
        return (0.0,0.0,0.0,1.0)
    inv = 1.0/n
    return (x*inv,y*inv,z*inv,w*inv)

def quat_rf_to_blender(q: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    R = quat_to_mat3(q)
    Rt = mat3_mul(mat3_mul(_M, R), mat3_T(_M))
    return mat3_to_quat(Rt)

def quat_dot(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

def quat_neg(q: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    return (-q[0], -q[1], -q[2], -q[3])

def quat_fix_sign_seq(qs: List[Tuple[float,float,float,float]]) -> List[Tuple[float,float,float,float]]:
    # Quaternions q and -q represent the same rotation. Fix sign flips so we can measure variation sanely.
    if not qs:
        return []
    out = [qs[0]]
    for q in qs[1:]:
        if quat_dot(out[-1], q) < 0.0:
            q = quat_neg(q)
        out.append(q)
    return out

def quat_angle_diff_rad(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    # Small, stable rotation difference metric in radians (0..pi)
    d = abs(quat_dot(a, b))
    if d > 1.0: d = 1.0
    # angle = 2*acos(d); clamp numeric noise
    return 2.0 * math.acos(d)


# ----------------------------
# VFX structures (subset we need)
# ----------------------------

SEC_SFXO = 0x4F584653  # mesh
SEC_MATL = 0x4C54414D  # material
SEC_DMMY = 0x594D4D44  # dummy

@dataclass
class VfxHeader:
    version: int
    flags: Optional[int]
    end_frame: int

@dataclass
class VfxMaterial:
    # minimal: diffuse texture name (tex_0)
    index: int
    mat_type: int
    additive: bool
    tex0: str

@dataclass
class MeshFace:
    material_index: int     # as stored in VFX face (mesh-local index or -1)
    face_vertex_indices: Tuple[int,int,int]

@dataclass
class FaceVertex:
    vertex_index: int

@dataclass
class MeshFrame:
    # positions only present for morph or index==0
    center_rf: Optional[Tuple[float,float,float]] = None
    mult_rf: Optional[Tuple[float,float,float]] = None
    pos_s16: Optional[List[Tuple[int,int,int]]] = None

    # UVs present for index==0 (or dump_uvs)
    uvs: Optional[List[Tuple[float,float]]] = None  # length 3*num_faces

    # TRS per frame when not morph and not keyframed
    translation_rf: Optional[Tuple[float,float,float]] = None
    rotation_raw: Optional[Tuple[float,float,float,float]] = None
    scale_rf: Optional[Tuple[float,float,float]] = None

@dataclass
class MeshKeyframes:
    t_times: List[float]
    t_values: List[Tuple[float,float,float]]
    r_times: List[float]
    r_values: List[Tuple[float,float,float,float]]
    s_times: List[float]
    s_values: List[Tuple[float,float,float]]

@dataclass
class VfxMesh:
    name: str
    parent_name: str
    flags_raw: int
    morph: bool
    facing: bool
    facing_rod: bool
    dump_uvs: bool
    is_keyframed: bool
    num_vertices: int
    num_faces: int
    fps: int
    start_time: float
    end_time: float
    num_frames: int
    materials_indices: List[int]  # maps mesh-local material indices -> global material indices
    faces: List[MeshFace]
    face_vertices: List[FaceVertex]
    frames: List[MeshFrame]
    keyframes: Optional[MeshKeyframes]
    pivot_scale_rf: Optional[Tuple[float,float,float]]

@dataclass
class DummyFrame:
    pos_rf: Tuple[float,float,float]
    orient_raw: Tuple[float,float,float,float]

@dataclass
class VfxDummy:
    name: str
    parent_name: str
    pos_rf: Tuple[float,float,float]
    orient_raw: Tuple[float,float,float,float]
    frames: List[DummyFrame]

# ----------------------------
# Parsing (subset)
# ----------------------------

def parse_header(b: Bin) -> VfxHeader:
    if b.read(4) != b"VSFX":
        raise ValueError("Not a VSFX/VFX file")
    version = b.s32()
    flags = b.s32() if version >= 0x30008 else None
    end_frame = b.s32()
    # skip the rest of header counts (we don't need them, but must advance correctly)
    # See vfx.ksy file_header
    b.s32(); b.s32(); b.s32(); b.s32(); b.s32(); b.s32()  # num_meshes..num_cameras
    if version >= 0x3000F:
        b.s32()  # num_selsets
    if version >= 0x40000:
        b.s32()  # num_materials
    if version >= 0x40002:
        b.s32()  # num_mix_frames
    if version >= 0x40003:
        b.s32()  # num_self_illumination_frames
    if version >= 0x40005:
        b.s32()  # num_opacity_frames
    if version < 0x3000A:
        b.s32()  # unk_1
    # remaining fixed counts
    b.s32(); b.s32(); b.s32(); b.s32(); b.s32()  # num_faces..num_mesh_frames
    if version >= 0x3000D:
        b.s32()  # num_uv_frames
    if version >= 0x30009:
        b.s32(); b.s32(); b.s32(); b.s32(); b.s32()  # transform counts
    b.s32(); b.s32(); b.s32(); b.s32(); b.s32()  # light..camera frames
    if version >= 0x3000F:
        b.s32()  # num_selset_objects
    return VfxHeader(version=version, flags=flags, end_frame=end_frame)

def parse_material(b: Bin, version: int, index: int) -> VfxMaterial:
    mat_type = b.s32()
    fps = b.s32() if version >= 0x40003 else 15
    additive = False
    tex0 = ""
    if (mat_type in (0,1)) or (version >= 0x40006):
        additive = (b.s8() != 0)
    if mat_type in (0,1):
        tex0 = b.strz()
        if version >= 0x30012:
            _ = b.s32()  # start_frame
            _ = b.f32()  # playback_rate
            _ = b.s32()  # anim_type
        if mat_type == 1:
            _t1 = b.strz()
            if version >= 0x30012:
                _ = b.s32(); _ = b.f32(); _ = b.s32()
            num_mix = b.s32()
            if mat_type == 1 and version < 0x40003:
                _ = b.s32()
            for _ in range(num_mix):
                _ = b.f32()
        # spec/gloss/refl
        _ = b.f32(); _ = b.f32(); _ = b.f32()
        _ = b.strz()
    elif mat_type == 2:
        # solid_color rgb_s4
        _ = b.s32(); _ = b.s32(); _ = b.s32()
    if version >= 0x40003:
        n_si = b.s32()
        for _ in range(n_si):
            _ = b.f32()
    else:
        # older: one self illumination sample?
        _ = b.f32()
    if version >= 0x40005:
        n_op = b.s32()
        for _ in range(n_op):
            _ = b.f32()
    return VfxMaterial(index=index, mat_type=mat_type, additive=additive, tex0=tex0)

def parse_mesh(b: Bin, hdr: VfxHeader, section_end: int) -> VfxMesh:
    ver = hdr.version
    name = b.strz()
    parent_name = b.strz()
    _save_parent = b.s8()
    num_vertices = b.s32()
    if ver < 0x3000A:
        for _ in range(num_vertices):
            _ = read_vec3(b)
    num_faces = b.s32()

    # faces: we only keep material_index + face_vertex_indices
    faces: List[MeshFace] = []
    for _ in range(num_faces):
        # indices (ignored; topo is via face_vertex_indices + face_vertices table)
        _ = b.s32(); _ = b.s32(); _ = b.s32()
        if ver < 0x3000D:
            for __ in range(3):
                _ = read_uv(b)
        # 3 colors (rgb_f4) each has 3 floats
        for __ in range(3*3):
            _ = b.f32()
        _ = read_vec3_rf(b)  # face normal stored basis
        _ = read_vec3_rf(b)  # center
        _ = b.f32()          # radius
        material_index = b.s32()
        _ = b.s32()          # smoothing_group
        fvi0 = b.s32(); fvi1 = b.s32(); fvi2 = b.s32()
        faces.append(MeshFace(material_index=material_index, face_vertex_indices=(fvi0,fvi1,fvi2)))

    fps = b.s32() if ver >= 0x30009 else 15

    start_time = 0.0
    end_time = 0.0
    num_frames = 1
    if ver >= 0x40004:
        start_time = b.f32()
        end_time = b.f32()
        num_frames = b.s32()
    else:
        start_frame = b.s32()
        end_frame = b.s32()
        num_frames = max(1, (end_frame - start_frame + 1) if ver >= 0x3000C else (end_frame - start_frame))
        start_time = float(start_frame) / float(fps)
        end_time = float(end_frame) / float(fps)

    num_materials = b.s32()
    materials_indices: List[int] = []
    if ver >= 0x40000:
        for _ in range(num_materials):
            materials_indices.append(b.s32())
    else:
        # older: inline materials; skip
        for _ in range(num_materials):
            # mesh_material_old: too much; skip by seeking? can't easily without spec.
            pass

    _bounding_center = read_vec3_rf(b)
    _bounding_radius = b.f32()
    if ver < 0x30002:
        _ = b.s32()
    flags_raw = b.u32()
    facing = (flags_raw & 0x00000001) != 0
    dump_uvs = (flags_raw & 0x00000100) != 0
    facing_rod = (flags_raw & 0x00000800) != 0
    morph = (flags_raw & 0x00000004) != 0

    if facing and ver == 0x3000A:
        _ = b.f32(); _ = b.f32()

    num_face_vertices = b.s32()
    face_vertices: List[FaceVertex] = []
    for _ in range(num_face_vertices):
        _ = b.s32()  # smoothing_group
        vi = b.s32()
        _ = b.f32(); _ = b.f32()  # u,v garbage
        n_adj = b.s32()
        for __ in range(n_adj):
            _ = b.s32()
        face_vertices.append(FaceVertex(vertex_index=vi))

    is_keyframed = False
    if ver >= 0x30009:
        is_keyframed = (b.u8() != 0)

    frames: List[MeshFrame] = []
    for fi in range(num_frames):
        fr = MeshFrame()
        # positions
        if morph or fi == 0:
            fr.center_rf = read_vec3_rf(b)
            fr.mult_rf = read_vec3_rf(b)
            fr.pos_s16 = [(b.s16(), b.s16(), b.s16()) for _ in range(num_vertices)]
            if (facing or facing_rod) and ver >= 0x3000B:
                _ = b.f32(); _ = b.f32()
            if facing_rod and fi == 0 and ver >= 0x40001:
                _ = read_vec3_rf(b)  # up_vector
            # uvs
            if (dump_uvs or fi == 0) and ver >= 0x3000D:
                fr.uvs = [read_uv(b) for _ in range(3*num_faces)]
        # TRS for non-morph per-frame (when not keyframed)
        if (not morph) and (not is_keyframed):
            fr.translation_rf = read_vec3_rf(b)
            fr.rotation_raw = read_quat_raw(b)
            fr.scale_rf = read_vec3_rf(b)
        if ver < 0x30009:
            _ = b.read(1)
        if ver < 0x40005:
            _ = b.f32()  # opacity
        frames.append(fr)

    keyframes: Optional[MeshKeyframes] = None
    pivot_scale_rf: Optional[Tuple[float,float,float]] = None
    if is_keyframed:
        # pivots exist but we ignore for now (we can fold later)
        if ver >= 0x3000A:
            _ = read_vec3_rf(b)  # pivot_translation (ignored for now)
            _ = read_quat_raw(b) # pivot_rotation (ignored for now)
            pivot_scale_rf = read_vec3_rf(b)  # IMPORTANT: used for correct mesh scale
        # keyframe list
        num_t = b.s32()
        t_times: List[float] = []
        t_vals: List[Tuple[float,float,float]] = []
        for _ in range(num_t):
            t_raw = b.s32()
            v = read_vec3_rf(b)
            _ = read_vec3_rf(b); _ = read_vec3_rf(b)
            t_times.append((t_raw / 320.0) / float(fps))
            t_vals.append(rf_to_blender(v))
        num_r = b.s32()
        r_times: List[float] = []
        r_vals: List[Tuple[float,float,float,float]] = []
        for _ in range(num_r):
            t_raw = b.s32()
            q = read_quat_raw(b)
            _ = b.f32(); _ = b.f32(); _ = b.f32(); _ = b.f32(); _ = b.f32()
            r_times.append((t_raw / 320.0) / float(fps))
            r_vals.append(quat_rf_to_blender(q))
        num_s = b.s32()
        s_times: List[float] = []
        s_vals: List[Tuple[float,float,float]] = []
        for _ in range(num_s):
            t_raw = b.s32()
            v = read_vec3_rf(b)
            _ = read_vec3_rf(b); _ = read_vec3_rf(b)
            s_times.append((t_raw / 320.0) / float(fps))
            # scale: ignore sign flips
            sx,sy,sz = v
            s_vals.append((abs(sx), abs(sz), abs(sy)))
        keyframes = MeshKeyframes(
            t_times=t_times, t_values=t_vals,
            r_times=r_times, r_values=r_vals,
            s_times=s_times, s_values=s_vals
        )

    # be safe: jump to section_end
    if b.tell() < section_end:
        b.seek(section_end)

    return VfxMesh(
        name=name,
        parent_name=parent_name,
        flags_raw=flags_raw,
        morph=morph,
        facing=facing,
        facing_rod=facing_rod,
        dump_uvs=dump_uvs,
        is_keyframed=is_keyframed,
        num_vertices=num_vertices,
        num_faces=num_faces,
        fps=fps,
        start_time=start_time,
        end_time=end_time,
        num_frames=num_frames,
        materials_indices=materials_indices,
        faces=faces,
        face_vertices=face_vertices,
        frames=frames,
        keyframes=keyframes,
        pivot_scale_rf=pivot_scale_rf
    )

def parse_dummy(b: Bin, hdr: VfxHeader, section_end: int) -> VfxDummy:
    ver = hdr.version
    name = b.strz()
    parent_name = b.strz()
    _save_parent = b.u8()
    pos_rf = read_vec3_rf(b)
    orient_raw = read_quat_raw(b)
    num_frames = b.s32()
    frames: List[DummyFrame] = []
    for _ in range(num_frames):
        p = read_vec3_rf(b)
        q = read_quat_raw(b)
        frames.append(DummyFrame(pos_rf=p, orient_raw=q))
    if b.tell() < section_end:
        b.seek(section_end)
    return VfxDummy(name=name, parent_name=parent_name, pos_rf=pos_rf, orient_raw=orient_raw, frames=frames)

def parse_vfx(path: str) -> Tuple[VfxHeader, List[VfxMaterial], List[VfxMesh], List[VfxDummy]]:
    data = open(path, "rb").read()
    b = Bin(data)
    hdr = parse_header(b)

    materials: List[VfxMaterial] = []
    meshes: List[VfxMesh] = []
    dummies: List[VfxDummy] = []

    mat_index = 0
    while b.tell() < len(data):
        # section header: type(s4) + len(s4)
        sec_type = b.u32()
        sec_len = b.u32()
        # sec_len includes the len field itself, not the type field
        body_len = sec_len - 4
        section_end = b.tell() + body_len
        try:
            if sec_type == SEC_MATL:
                materials.append(parse_material(b, hdr.version, mat_index))
                mat_index += 1
            elif sec_type == SEC_SFXO:
                meshes.append(parse_mesh(b, hdr, section_end))
            elif sec_type == SEC_DMMY:
                dummies.append(parse_dummy(b, hdr, section_end))
            else:
                # skip unsupported sections
                b.seek(section_end)
        except Exception as e:
            # recover to next section
            raise RuntimeError(f"Failed parsing section type=0x{sec_type:08X} at ofs={b.tell()} in {os.path.basename(path)}: {e}") from e
        if b.tell() != section_end:
            b.seek(section_end)

    return hdr, materials, meshes, dummies

# ----------------------------
# glTF builder
# ----------------------------

def _align4(n: int) -> int:
    return (n + 3) & ~3

class GltfBuilder:
    def __init__(self):
        self.g: Dict[str, Any] = {
            "asset": {"version": "2.0", "generator": "vfx2obj_optionA.py"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [{"byteLength": 0, "uri": ""}],
            "materials": [],
            "images": [],
            "textures": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
            "animations": [],
        }
        self._bin = bytearray()
        self._mat_map: Dict[int,int] = {}  # global material idx -> gltf material idx
        self._img_map: Dict[str,int] = {}  # tex name -> image idx
        self._tex_map: Dict[str,int] = {}  # tex name -> texture idx

    def _add_bufferview(self, data: bytes, target: Optional[int]) -> Tuple[int, int]:
        # returns (bufferViewIndex, byteOffset)
        off = _align4(len(self._bin))
        if off > len(self._bin):
            self._bin += b"\x00" * (off - len(self._bin))
        self._bin += data
        bv = {
            "buffer": 0,
            "byteOffset": off,
            "byteLength": len(data),
        }
        if target is not None:
            bv["target"] = target
        self.g["bufferViews"].append(bv)
        return len(self.g["bufferViews"]) - 1, off

    def _add_accessor(self, bv_i: int, byte_offset: int, component_type: int, count: int, acc_type: str,
                      minv: Optional[List[float]] = None, maxv: Optional[List[float]] = None) -> int:
        acc = {
            "bufferView": bv_i,
            "byteOffset": byte_offset,
            "componentType": component_type,
            "count": count,
            "type": acc_type,
        }
        if minv is not None: acc["min"] = minv
        if maxv is not None: acc["max"] = maxv
        self.g["accessors"].append(acc)
        return len(self.g["accessors"]) - 1

    def _ensure_material(self, mat: Optional[VfxMaterial]) -> Optional[int]:
        if mat is None:
            return None
        if mat.index in self._mat_map:
            return self._mat_map[mat.index]
        # image+texture if tex0 exists
        pbr: Dict[str, Any] = {}
        if mat.tex0:
            img_i = self._img_map.get(mat.tex0)
            if img_i is None:
                self.g["images"].append({"uri": mat.tex0})
                img_i = len(self.g["images"]) - 1
                self._img_map[mat.tex0] = img_i
            tex_i = self._tex_map.get(mat.tex0)
            if tex_i is None:
                self.g["textures"].append({"sampler": 0, "source": img_i})
                tex_i = len(self.g["textures"]) - 1
                self._tex_map[mat.tex0] = tex_i
            pbr["baseColorTexture"] = {"index": tex_i}
        m = {"name": f"MATL_{mat.index}"}
        if pbr:
            m["pbrMetallicRoughness"] = pbr
        if mat.additive:
            m["alphaMode"] = "BLEND"
        self.g["materials"].append(m)
        gi = len(self.g["materials"]) - 1
        self._mat_map[mat.index] = gi
        return gi

    def add_node(self, name: str, mesh_index: Optional[int]=None) -> int:
        n: Dict[str, Any] = {"name": name}
        if mesh_index is not None:
            n["mesh"] = mesh_index
        self.g["nodes"].append(n)
        return len(self.g["nodes"]) - 1

    def set_node_trs(self, node_index: int, t: Optional[Tuple[float,float,float]]=None,
                     r: Optional[Tuple[float,float,float,float]]=None,
                     s: Optional[Tuple[float,float,float]]=None):
        n = self.g["nodes"][node_index]
        if t is not None: n["translation"] = [float(t[0]), float(t[1]), float(t[2])]
        if r is not None: n["rotation"] = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
        if s is not None: n["scale"] = [float(s[0]), float(s[1]), float(s[2])]

    def add_child(self, parent: int, child: int) -> None:
        n = self.g["nodes"][parent]
        ch = n.get("children")
        if ch is None:
            n["children"] = [child]
        else:
            ch.append(child)

    def add_mesh_from_faces(
        self,
        mesh_name: str,
        base_positions_bl: List[Tuple[float,float,float]],
        faces: List[MeshFace],
        face_vertices: List[FaceVertex],
        frame0_uvs: List[Tuple[float,float]],
        scale_geom: float,
        trs_scale: float,
        materials: List[VfxMaterial],
        mesh_local_mat_to_global: List[int],
        morph_frames_bl: Optional[List[List[Tuple[float,float,float]]]],  # includes frame0 positions in blender basis
        bake_constant_scale: bool = True,
        trs_frames_rf: Optional[List[Tuple[Tuple[float,float,float],Tuple[float,float,float,float],Tuple[float,float,float]]]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Returns (gltf_mesh_index, info dict used by animation builder)
        """
        # group faces by global material (or None)
        groups: Dict[Optional[int], List[int]] = {}
        for fi, f in enumerate(faces):
            mat_idx = f.material_index
            glb_mat: Optional[int] = None
            if mat_idx >= 0 and mat_idx < len(mesh_local_mat_to_global):
                glb = mesh_local_mat_to_global[mat_idx]
                if 0 <= glb < len(materials):
                    glb_mat = glb
            groups.setdefault(glb_mat, []).append(fi)

        primitives: List[Dict[str, Any]] = []
        prim_infos: List[Dict[str, Any]] = []  # for morph + anim mapping
        for glb_mat, face_ids in groups.items():
            # build vertex stream with uv splits
            out_pos: List[Tuple[float,float,float]] = []
            out_uv: List[Tuple[float,float]] = []
            out_src_pi: List[int] = []  # original position index for each out vert
            out_idx: List[int] = []
            vmap: Dict[Tuple[int,int,int], int] = {}  # (pos_index, u_q, v_q) -> out index

            def uv_key(uv: Tuple[float,float]) -> Tuple[int,int]:
                # quantize to stabilize keys
                return (int(round(uv[0]*100000.0)), int(round(uv[1]*100000.0)))

            for fi in face_ids:
                f = faces[fi]
                tri = []
                for corner in range(3):
                    fvi = f.face_vertex_indices[corner]
                    pi = face_vertices[fvi].vertex_index
                    uv = frame0_uvs[fi*3 + corner]
                    uq, vq = uv_key(uv)
                    key = (pi, uq, vq)
                    oi = vmap.get(key)
                    if oi is None:
                        oi = len(out_pos)
                        vmap[key] = oi
                        px,py,pz = base_positions_bl[pi]
                        out_pos.append((px*scale_geom, py*scale_geom, pz*scale_geom))
                        out_uv.append(uv)
                        out_src_pi.append(pi)
                    tri.append(oi)
                out_idx.extend(tri)

            # indices component type
            max_index = max(out_idx) if out_idx else 0
            if max_index <= 65535:
                idx_ct = 5123  # UNSIGNED_SHORT
                idx_pack = lambda arr: struct.pack("<" + "H"*len(arr), *arr)
                idx_blen = 2
            else:
                idx_ct = 5125  # UNSIGNED_INT
                idx_pack = lambda arr: struct.pack("<" + "I"*len(arr), *arr)
                idx_blen = 4

            # POSITION accessor
            pos_bytes = b"".join(struct.pack("<fff", *p) for p in out_pos)
            pos_bv, _ = self._add_bufferview(pos_bytes, target=34962)
            # compute min/max
            if out_pos:
                mn = [min(p[i] for p in out_pos) for i in range(3)]
                mx = [max(p[i] for p in out_pos) for i in range(3)]
            else:
                mn = [0.0,0.0,0.0]; mx = [0.0,0.0,0.0]
            pos_acc = self._add_accessor(pos_bv, 0, 5126, len(out_pos), "VEC3",
                                         minv=[float(mn[0]),float(mn[1]),float(mn[2])],
                                         maxv=[float(mx[0]),float(mx[1]),float(mx[2])])

            # TEXCOORD_0 accessor
            uv_bytes = b"".join(struct.pack("<ff", float(uv[0]), float(uv[1])) for uv in out_uv)
            uv_bv, _ = self._add_bufferview(uv_bytes, target=34962)
            uv_acc = self._add_accessor(uv_bv, 0, 5126, len(out_uv), "VEC2")

            # indices accessor
            idx_bytes = idx_pack(out_idx)
            idx_bv, _ = self._add_bufferview(idx_bytes, target=34963)
            idx_acc = self._add_accessor(idx_bv, 0, idx_ct, len(out_idx), "SCALAR")

            prim: Dict[str, Any] = {
                "attributes": {"POSITION": pos_acc, "TEXCOORD_0": uv_acc},
                "indices": idx_acc,
            }

            # material
            gltf_mat_i = None
            if glb_mat is not None and 0 <= glb_mat < len(materials):
                gltf_mat_i = self._ensure_material(materials[glb_mat])
            if gltf_mat_i is not None:
                prim["material"] = gltf_mat_i

            # morph targets
            morph_info = {}
            if morph_frames_bl is not None and len(morph_frames_bl) > 1:
                targets = []
                # frame0 in morph_frames_bl[0], deltas for each subsequent frame
                base0 = morph_frames_bl[0]
                for fi_m in range(1, len(morph_frames_bl)):
                    frp = morph_frames_bl[fi_m]
                    deltas: List[Tuple[float,float,float]] = []
                    for pi in out_src_pi:
                        bx,by,bz = base0[pi]
                        fx,fy,fz = frp[pi]
                        dx,dy,dz = (fx-bx)*scale_geom, (fy-by)*scale_geom, (fz-bz)*scale_geom
                        deltas.append((dx,dy,dz))
                    tgt_bytes = b"".join(struct.pack("<fff", *d) for d in deltas)
                    tgt_bv, _ = self._add_bufferview(tgt_bytes, target=34962)
                    if deltas:
                        mn = [min(d[i] for d in deltas) for i in range(3)]
                        mx = [max(d[i] for d in deltas) for i in range(3)]
                    else:
                        mn = [0.0,0.0,0.0]; mx=[0.0,0.0,0.0]
                    tgt_acc = self._add_accessor(tgt_bv, 0, 5126, len(deltas), "VEC3",
                                                 minv=[float(mn[0]),float(mn[1]),float(mn[2])],
                                                 maxv=[float(mx[0]),float(mx[1]),float(mx[2])])
                    targets.append({"POSITION": tgt_acc})
                prim["targets"] = targets
                # default weights = all 0
                prim["weights"] = [0.0 for _ in range(len(targets))]
                morph_info = {"targets_count": len(targets)}
            primitives.append(prim)
            prim_infos.append({"morph": morph_info, "out_vert_count": len(out_pos)})

        gltf_mesh = {"name": mesh_name, "primitives": primitives}
        self.g["meshes"].append(gltf_mesh)
        mesh_i = len(self.g["meshes"]) - 1
        return mesh_i, {"prim_infos": prim_infos}

    def add_animation(self, anim: Dict[str, Any]) -> int:
        self.g["animations"].append(anim)
        return len(self.g["animations"]) - 1

    def add_anim_sampler(self, anim: Dict[str, Any], input_acc: int, output_acc: int, interp: str="LINEAR") -> int:
        sam = {"input": input_acc, "output": output_acc, "interpolation": interp}
        anim.setdefault("samplers", []).append(sam)
        return len(anim["samplers"]) - 1

    def add_anim_channel(self, anim: Dict[str, Any], sampler_i: int, node_i: int, path: str) -> None:
        anim.setdefault("channels", []).append({"sampler": sampler_i, "target": {"node": node_i, "path": path}})

    def add_accessor_f32(self, values: List[float], acc_type: str, elem_n: int) -> int:
        # values length = count*elem_n
        data = struct.pack("<" + "f"*len(values), *values) if values else b""
        bv, _ = self._add_bufferview(data, target=None)
        count = (len(values) // elem_n) if elem_n else 0
        return self._add_accessor(bv, 0, 5126, count, acc_type)

    def finalize(self) -> Tuple[Dict[str, Any], bytes]:
        self.g["buffers"][0]["byteLength"] = len(self._bin)
        return self.g, bytes(self._bin)

# ----------------------------
# Export logic
# ----------------------------

def decompress_positions_bl(fr: MeshFrame, num_vertices: int) -> List[Tuple[float,float,float]]:
    assert fr.center_rf is not None and fr.mult_rf is not None and fr.pos_s16 is not None
    cx,cy,cz = fr.center_rf
    mx,my,mz = fr.mult_rf
    out: List[Tuple[float,float,float]] = []
    for (ix,iy,iz) in fr.pos_s16:
        # correct decompression: center + mult * raw_s16 (no /32767)
        vx = cx + (mx * float(ix))
        vy = cy + (my * float(iy))
        vz = cz + (mz * float(iz))
        out.append(rf_to_blender((vx, vy, vz)))
    return out

def should_bake_constant_scale(scales_rf: List[Tuple[float,float,float]]) -> Optional[Tuple[float,float,float]]:
    if not scales_rf:
        return None
    # compare all scales against first
    s0 = scales_rf[0]
    def close(a,b): return abs(a-b) <= 1e-6
    for s in scales_rf[1:]:
        if not (close(s[0],s0[0]) and close(s[1],s0[1]) and close(s[2],s0[2])):
            return None
    # bake only if meaningfully different from 1
    if abs(s0[0]-1.0) < 1e-6 and abs(s0[1]-1.0) < 1e-6 and abs(s0[2]-1.0) < 1e-6:
        return None
    return s0

def export_gltf(
    in_path: str,
    out_gltf: str,
    debug_frames: bool,
    scale: float,
    trs_scale: Optional[float],
    mesh_offsets: Optional[Dict[str, Tuple[float,float,float]]] = None,
) -> None:
    hdr, mats, meshes, dummies = parse_vfx(in_path)

    if trs_scale is None:
        trs_scale = scale

    mesh_offsets_l: Dict[str, Tuple[float,float,float]] = {}
    if mesh_offsets:
        for k,v in mesh_offsets.items():
            mesh_offsets_l[k.lower()] = v

    # --- Debug print ---
    if debug_frames:
        for m in meshes:
            print(f"[DEBUG_FRAMES] mesh='{m.name}' fps={m.fps} frames={m.num_frames} morph={m.morph} flags=0x{m.flags_raw:08X}")

    gb = GltfBuilder()

    # root node
    root_idx = gb.add_node("__VFX_ROOT__")
    gb.g["scenes"][0]["nodes"] = [root_idx]

    # Create glTF nodes for meshes and dummies, then parent them
    node_by_name: Dict[str, int] = {"__VFX_ROOT__": root_idx}
    pending_parent: List[Tuple[int,str]] = []

    # --- Mesh nodes ---
    mesh_node_infos: List[Dict[str, Any]] = []
    for m in meshes:
        # base geometry
        base_bl = decompress_positions_bl(m.frames[0], m.num_vertices)
        frame0_uvs = m.frames[0].uvs or [(0.0,0.0)]*(3*m.num_faces)

        morph_frames_bl: Optional[List[List[Tuple[float,float,float]]]] = None
        if m.morph:
            morph_frames_bl = []
            for fr in m.frames:
                morph_frames_bl.append(decompress_positions_bl(fr, m.num_vertices))

        # per-frame TRS (non-keyframed only)
        trs_frames = None
        if (not m.morph) and (not m.is_keyframed) and m.num_frames > 0:
            trs_frames = []
            for fr in m.frames:
                t_rf = fr.translation_rf or (0.0,0.0,0.0)
                r_raw = fr.rotation_raw or (0.0,0.0,0.0,1.0)
                s_rf = fr.scale_rf or (1.0,1.0,1.0)
                trs_frames.append((rf_to_blender(t_rf), quat_rf_to_blender(r_raw), (abs(s_rf[0]), abs(s_rf[2]), abs(s_rf[1]))))

        # Pivot scale (pre-keyframe) — this is the missing ingredient for correct scale on keyframed meshes (e.g. flagpole)
        # In CTFflag-blue, pivot_scale ~0.13495 and scale keyframe is 1.2, so effective scale is 0.161943 (what we debugged in Chat 04).
        baked_scale_from_pivot: Optional[Tuple[float,float,float]] = None
        if (not m.morph) and m.pivot_scale_rf is not None:
            psx, psy, psz = m.pivot_scale_rf
            baked_scale_from_pivot = (abs(psx), abs(psz), abs(psy))

        skip_scale_anim = False
        if (not m.morph) and m.is_keyframed and m.keyframes and baked_scale_from_pivot is not None:
            # if scale keys are constant, bake pivot_scale * scale into geometry and remove scale channel
            if m.keyframes.s_values:
                s0 = m.keyframes.s_values[0]
                if all(abs(s[0]-s0[0])<1e-6 and abs(s[1]-s0[1])<1e-6 and abs(s[2]-s0[2])<1e-6 for s in m.keyframes.s_values):
                    eff = (baked_scale_from_pivot[0]*s0[0], baked_scale_from_pivot[1]*s0[1], baked_scale_from_pivot[2]*s0[2])
                    if debug_frames:
                        print(f"[DEBUG_SCALE] mesh='{m.name}' baked geom_scale=({eff[0]:.6f},{eff[1]:.6f},{eff[2]:.6f})")
                    base_bl = [(p[0]*eff[0], p[1]*eff[1], p[2]*eff[2]) for p in base_bl]
                    # keyframed non-morph meshes won't have morph frames; but keep for completeness
                    if morph_frames_bl is not None:
                        morph_frames_bl = [[(p[0]*eff[0], p[1]*eff[1], p[2]*eff[2]) for p in fr] for fr in morph_frames_bl]
                    skip_scale_anim = True
        
        # bake constant scale for non-morph meshes (avoid scaling children)
        geom_scale_vec = None
        if trs_frames is not None:
            scales_rf = [fr.scale_rf for fr in m.frames if fr.scale_rf is not None]
            geom_scale_vec = should_bake_constant_scale(scales_rf)
            if geom_scale_vec is not None:
                # apply to vertices; remove scale from TRS frames
                sx,sy,sz = geom_scale_vec
                # convert scale axes to blender axis order (abs, swap)
                bake_s = (abs(sx), abs(sz), abs(sy))
                if debug_frames:
                    print(f"[DEBUG_SCALE] mesh='{m.name}' baked geom_scale=({bake_s[0]:.6f},{bake_s[1]:.6f},{bake_s[2]:.6f})")
                # fold into base + morph positions in blender basis, before global 'scale'
                base_bl = [(p[0]*bake_s[0], p[1]*bake_s[1], p[2]*bake_s[2]) for p in base_bl]
                if morph_frames_bl is not None:
                    morph_frames_bl = [[(p[0]*bake_s[0], p[1]*bake_s[1], p[2]*bake_s[2]) for p in fr] for fr in morph_frames_bl]
                # overwrite trs_frames scales to 1
                trs_frames2 = []
                for (t,r,_s) in trs_frames:
                    trs_frames2.append((t,r,(1.0,1.0,1.0)))
                trs_frames = trs_frames2


        # --- Optional per-mesh vertex offset (applied to base + ALL morph frames) ---
        # Offsets are specified in OUTPUT units (after --scale). We convert to pre-scale units here.
        off = mesh_offsets_l.get(m.name.lower())
        if off is not None:
            ox, oy, oz = off
            if abs(scale) > 1e-12:
                ox /= scale; oy /= scale; oz /= scale
            base_bl = [(p[0] + ox, p[1] + oy, p[2] + oz) for p in base_bl]
            if morph_frames_bl is not None:
                morph_frames_bl = [[(p[0] + ox, p[1] + oy, p[2] + oz) for p in fr] for fr in morph_frames_bl]
            if debug_frames:
                print(f"[DEBUG_OFFSET] mesh='{m.name}' applied mesh_offset=({off[0]:.6g},{off[1]:.6g},{off[2]:.6g})")

        mesh_i, info = gb.add_mesh_from_faces(
            mesh_name=m.name,
            base_positions_bl=base_bl,
            faces=m.faces,
            face_vertices=m.face_vertices,
            frame0_uvs=frame0_uvs,
            scale_geom=scale,
            trs_scale=trs_scale,
            materials=mats,
            mesh_local_mat_to_global=m.materials_indices,
            morph_frames_bl=morph_frames_bl,
        )
        node_i = gb.add_node(m.name, mesh_index=mesh_i)
        # set initial TRS (time 0)
        if m.morph:
            # morph meshes usually ride on parent
            gb.set_node_trs(node_i, t=(0.0,0.0,0.0), r=(0.0,0.0,0.0,1.0), s=(1.0,1.0,1.0))
        elif m.is_keyframed and m.keyframes:
            # set to first keyframe values if present
            t0 = m.keyframes.t_values[0] if m.keyframes.t_values else (0.0,0.0,0.0)
            r0 = m.keyframes.r_values[0] if m.keyframes.r_values else (0.0,0.0,0.0,1.0)
            s0 = m.keyframes.s_values[0] if m.keyframes.s_values else (1.0,1.0,1.0)
            if skip_scale_anim: s0 = (1.0,1.0,1.0)
            gb.set_node_trs(node_i, t=(t0[0]*trs_scale, t0[1]*trs_scale, t0[2]*trs_scale), r=r0, s=s0)
        elif trs_frames:
            t0,r0,s0 = trs_frames[0]
            gb.set_node_trs(node_i, t=(t0[0]*trs_scale, t0[1]*trs_scale, t0[2]*trs_scale), r=r0, s=s0)
        else:
            gb.set_node_trs(node_i, t=(0.0,0.0,0.0), r=(0.0,0.0,0.0,1.0), s=(1.0,1.0,1.0))

        node_by_name[m.name] = node_i
        pending_parent.append((node_i, m.parent_name))
        mesh_node_infos.append({
            "mesh": m,
            "node_i": node_i,
            "morph_targets_count": (len(morph_frames_bl)-1) if (morph_frames_bl is not None) else 0,
            "trs_frames": trs_frames,
            "skip_scale_anim": skip_scale_anim,
        })

    # --- Dummy nodes ---
    if debug_frames:
        print(f"[INFO] dummies found: {len(dummies)}")
    dummy_infos: List[Dict[str, Any]] = []
    for d in dummies:
        di = gb.add_node(d.name, mesh_index=None)
        node_by_name[d.name] = di
        pending_parent.append((di, d.parent_name))
        # set base TRS
        gb.set_node_trs(di, t=tuple(x*trs_scale for x in rf_to_blender(d.pos_rf)), r=quat_rf_to_blender(d.orient_raw), s=(1.0,1.0,1.0))
        if debug_frames:
            print(f"        {d.name} parent={d.parent_name} pos={rf_to_blender(d.pos_rf)} frames={len(d.frames)}")
        dummy_infos.append({"dummy": d, "node_i": di})

    # --- Parent hierarchy resolution ---
    for child_i, parent_name in pending_parent:
        p = parent_name.strip()
        if not p or p.lower() in ("scene root", "* scene root", "*scene root"):
            gb.add_child(root_idx, child_i)
            continue
        # prefer exact match; fallback case-insensitive
        parent_i = node_by_name.get(p)
        if parent_i is None:
            for k,v in node_by_name.items():
                if k.lower() == p.lower():
                    parent_i = v
                    break
        if parent_i is None:
            gb.add_child(root_idx, child_i)
        else:
            gb.add_child(parent_i, child_i)

    # --- Animations ---
    anim: Dict[str, Any] = {"name": "VFX"}
    # morph weights
    for mi in mesh_node_infos:
        m: VfxMesh = mi["mesh"]
        node_i = mi["node_i"]
        targets_count = mi["morph_targets_count"]
        if m.morph and targets_count > 0:
            fps = float(m.fps) if m.fps else 15.0
            times = [float(i)/fps for i in range(m.num_frames)]
            # weights: STEP, one-hot through targets
            weights: List[float] = []
            for fi in range(m.num_frames):
                w = [0.0]*targets_count
                if fi > 0:
                    w[fi-1] = 1.0
                weights.extend(w)
            inp_acc = gb.add_accessor_f32(times, "SCALAR", 1)
            out_acc = gb.add_accessor_f32(weights, "SCALAR", targets_count)  # type is SCALAR but elem_n = targets_count; accessor count = num_frames
            # BUT glTF expects output accessor type SCALAR with count=num_frames*targets_count? No: for weights, accessor type should be SCALAR with count=num_frames*targets_count? Actually spec wants VECn? There is no VECn>4.
            # Many tools use SCALAR with count=num_frames*targets_count and treat as array. We'll use SCALAR with count=num_frames*targets_count and set output stride? Not possible.
            # Safer approach: use accessor type "SCALAR" count = num_frames*targets_count, and sampler output interpreted with stride=targets_count (Blender handles).
            # We'll override accessor count to len(weights) and keep SCALAR; but sampler uses output accessor directly.
            # To keep builder simple, we re-add properly:
            # Rebuild output accessor as SCALAR with count=len(weights)
            # (Blender/glTF importer handles weights with count = keyframes * targets as packed array)
            gb.g["accessors"][out_acc]["count"] = len(weights)
            sam_i = gb.add_anim_sampler(anim, inp_acc, out_acc, interp="STEP")
            gb.add_anim_channel(anim, sam_i, node_i, "weights")

    # TRS for non-morph meshes (per-frame)
    for mi in mesh_node_infos:
        m: VfxMesh = mi["mesh"]
        node_i = mi["node_i"]
        if m.morph:
            continue
        if m.is_keyframed and m.keyframes:
            k = m.keyframes
            if k.t_times and k.t_values:
                inp_acc = gb.add_accessor_f32(k.t_times, "SCALAR", 1)
                out_vals = []
                for (x,y,z) in k.t_values:
                    out_vals.extend([x*trs_scale, y*trs_scale, z*trs_scale])
                out_acc = gb.add_accessor_f32(out_vals, "VEC3", 3)
                gb.g["accessors"][out_acc]["type"] = "VEC3"
                sam = gb.add_anim_sampler(anim, inp_acc, out_acc, interp="LINEAR")
                gb.add_anim_channel(anim, sam, node_i, "translation")
            if k.r_times and k.r_values:
                inp_acc = gb.add_accessor_f32(k.r_times, "SCALAR", 1)
                out_vals = []
                for (x,y,z,w) in k.r_values:
                    out_vals.extend([x,y,z,w])
                out_acc = gb.add_accessor_f32(out_vals, "VEC4", 4)
                gb.g["accessors"][out_acc]["type"] = "VEC4"
                sam = gb.add_anim_sampler(anim, inp_acc, out_acc, interp="LINEAR")
                gb.add_anim_channel(anim, sam, node_i, "rotation")
            if (not mi.get('skip_scale_anim')) and k.s_times and k.s_values:
                inp_acc = gb.add_accessor_f32(k.s_times, "SCALAR", 1)
                out_vals = []
                for (x,y,z) in k.s_values:
                    out_vals.extend([x,y,z])
                out_acc = gb.add_accessor_f32(out_vals, "VEC3", 3)
                gb.g["accessors"][out_acc]["type"] = "VEC3"
                sam = gb.add_anim_sampler(anim, inp_acc, out_acc, interp="LINEAR")
                gb.add_anim_channel(anim, sam, node_i, "scale")
        else:
            trs_frames = mi["trs_frames"]
            if trs_frames and len(trs_frames) > 1:
                fps = float(m.fps) if m.fps else 15.0
                times = [float(i)/fps for i in range(len(trs_frames))]
                inp_acc = gb.add_accessor_f32(times, "SCALAR", 1)
                # translation
                out_t = []
                out_r = []
                out_s = []
                for (t,r,s) in trs_frames:
                    out_t.extend([t[0]*trs_scale, t[1]*trs_scale, t[2]*trs_scale])
                    out_r.extend([r[0], r[1], r[2], r[3]])
                    out_s.extend([s[0], s[1], s[2]])
                out_t_acc = gb.add_accessor_f32(out_t, "VEC3", 3); gb.g["accessors"][out_t_acc]["type"]="VEC3"
                out_r_acc = gb.add_accessor_f32(out_r, "VEC4", 4); gb.g["accessors"][out_r_acc]["type"]="VEC4"
                out_s_acc = gb.add_accessor_f32(out_s, "VEC3", 3); gb.g["accessors"][out_s_acc]["type"]="VEC3"
                sam_t = gb.add_anim_sampler(anim, inp_acc, out_t_acc, interp="LINEAR")
                sam_r = gb.add_anim_sampler(anim, inp_acc, out_r_acc, interp="LINEAR")
                sam_s = gb.add_anim_sampler(anim, inp_acc, out_s_acc, interp="LINEAR")
                gb.add_anim_channel(anim, sam_t, node_i, "translation")
                gb.add_anim_channel(anim, sam_r, node_i, "rotation")
                gb.add_anim_channel(anim, sam_s, node_i, "scale")

    # dummies animation (only emit channels when something actually changes)
    # (For example: $prop_flag in CTFflag-blue has 46 frames but all values are identical,
    #  so exporting keys just adds noise and makes Blender show an unnecessary Action.)
    for di in dummy_infos:
        d: VfxDummy = di["dummy"]
        node_i = di["node_i"]
        if not d.frames or len(d.frames) <= 1:
            continue
        fps = 15.0
        times = [float(i)/fps for i in range(len(d.frames))]

        # Convert sequences to Blender space
        t_seq = []
        r_seq = []
        for fr in d.frames:
            t = rf_to_blender(fr.pos_rf)
            r = quat_rf_to_blender(fr.orient_raw)
            t_seq.append((t[0]*trs_scale, t[1]*trs_scale, t[2]*trs_scale))
            r_seq.append(r)
        r_seq = quat_fix_sign_seq(r_seq)

        # Decide whether translation / rotation actually vary
        eps_t = 1e-6
        eps_ang = 1e-6  # radians
        t0 = t_seq[0]
        r0 = r_seq[0]
        max_dt = 0.0
        for t in t_seq[1:]:
            dx = abs(t[0]-t0[0]); dy = abs(t[1]-t0[1]); dz = abs(t[2]-t0[2])
            if dx > max_dt: max_dt = dx
            if dy > max_dt: max_dt = dy
            if dz > max_dt: max_dt = dz
        max_ang = 0.0
        for r in r_seq[1:]:
            ang = quat_angle_diff_rad(r0, r)
            if ang > max_ang: max_ang = ang
        need_t = (max_dt > eps_t)
        need_r = (max_ang > eps_ang)
        if (not need_t) and (not need_r):
            continue

        inp_acc = gb.add_accessor_f32(times, "SCALAR", 1)
        if need_t:
            out_t = []
            for t in t_seq:
                out_t.extend([t[0], t[1], t[2]])
            out_t_acc = gb.add_accessor_f32(out_t, "VEC3", 3)
            gb.g["accessors"][out_t_acc]["type"] = "VEC3"
            sam_t = gb.add_anim_sampler(anim, inp_acc, out_t_acc, interp="LINEAR")
            gb.add_anim_channel(anim, sam_t, node_i, "translation")

        if need_r:
            out_r = []
            for r in r_seq:
                out_r.extend([r[0], r[1], r[2], r[3]])
            out_r_acc = gb.add_accessor_f32(out_r, "VEC4", 4)
            gb.g["accessors"][out_r_acc]["type"] = "VEC4"
            sam_r = gb.add_anim_sampler(anim, inp_acc, out_r_acc, interp="LINEAR")
            gb.add_anim_channel(anim, sam_r, node_i, "rotation")

    if anim.get("channels"):
        gb.add_animation(anim)

    g, blob = gb.finalize()
    # write files
    out_dir = os.path.dirname(os.path.abspath(out_gltf)) or "."
    base = os.path.splitext(os.path.basename(out_gltf))[0]
    out_bin = os.path.join(out_dir, base + ".bin")
    g["buffers"][0]["uri"] = os.path.basename(out_bin)

    with open(out_bin, "wb") as f:
        f.write(blob)
    with open(out_gltf, "w", encoding="utf-8", newline="") as f:
        json.dump(g, f, indent=2)

    print("Wrote:", out_gltf)
    print("Wrote:", out_bin)

def main(argv: List[str]) -> int:
    # Minimal CLI to match your workflow
    args = list(argv)
    debug_frames = False
    out_mode_gltf = False
    scale = 1.0
    trs_scale: Optional[float] = None

    mesh_offsets: Dict[str, Tuple[float,float,float]] = {}
    def _parse_mesh_offset(spec: str) -> Tuple[str, Tuple[float,float,float]]:
        # spec: Name=x,y,z  (floats)
        if "=" not in spec:
            raise ValueError("Expected Name=x,y,z")
        name, rhs = spec.split("=", 1)
        name = name.strip()
        parts = [p.strip() for p in rhs.split(",")]
        if len(parts) != 3:
            raise ValueError("Expected three comma-separated numbers: x,y,z")
        x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
        return name, (x,y,z)

    i = 0
    files: List[str] = []
    while i < len(args):
        a = args[i]
        if a == "--debug-frames":
            debug_frames = True
            i += 1
            continue
        if a == "--gltf":
            out_mode_gltf = True
            i += 1
            continue
        if a == "--scale":
            scale = float(args[i+1]); i += 2; continue
        if a == "--trs-scale":
            trs_scale = float(args[i+1]); i += 2; continue

        if a == "--mesh-offset":
            name,(x,y,z) = _parse_mesh_offset(args[i+1])
            mesh_offsets[name] = (x,y,z)
            i += 2
            continue
        if a == "--help" or a == "-h":
            print("Usage: vfx2obj_optionA.py [--gltf] [--debug-frames] [--scale N] [--trs-scale N] [--mesh-offset Name=x,y,z] file1.vfx [file2.vfx ...]")
            return 0
        files.append(a)
        i += 1

    if not files:
        print("No input files. Drag-drop .vfx onto the exe, or run with a .vfx path.")
        return 2

    for p in files:
        if not os.path.isfile(p):
            print(f"[SKIP] Not a file: {p}")
            continue
        in_path = p
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_dir = os.path.dirname(os.path.abspath(in_path)) or "."
        if out_mode_gltf:
            out_gltf = os.path.join(out_dir, base + ".gltf")
            export_gltf(in_path, out_gltf, debug_frames=debug_frames, scale=scale, trs_scale=trs_scale, mesh_offsets=mesh_offsets)
        else:
            # For now we keep the tool focused on glTF; OBJ path was in earlier builds.
            out_gltf = os.path.join(out_dir, base + ".gltf")
            export_gltf(in_path, out_gltf, debug_frames=debug_frames, scale=scale, trs_scale=trs_scale, mesh_offsets=mesh_offsets)

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))