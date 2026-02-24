#!/usr/bin/env python3
# vfx2obj.py - Drag/drop VFX -> OBJ for Blender
# Supports Red Faction VFX version 0x00040006 ("VFX V 4.6")

from __future__ import annotations
import os
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional

DEBUG_FRAMES = False  # set by --debug-frames

S16_MAX = 32767.0

# ---- binary helpers ----

class Bin:
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

    def s16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def s32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def cstr(self) -> str:
        start = self.ofs
        while self.ofs < len(self.data) and self.data[self.ofs] != 0:
            self.ofs += 1
        if self.ofs >= len(self.data):
            raise EOFError("Unterminated string")
        s = self.data[start:self.ofs].decode("ascii", errors="replace")
        self.ofs += 1
        return s

# ---- RF VFX coordinate conversion ----
# VFX stored vec3 is: x = -x_max, y = z_max, z = -y_max  (per spec)
# Convert back to Max/Blender-like coords: max_x = -sx, max_y = -sz, max_z = sy
def read_vec3_rf(b: Bin) -> Tuple[float, float, float]:
    # Raw vec3 as stored in VFX (RF space). Do NOT axis-swap here.
    return (b.f32(), b.f32(), b.f32())

def rf_to_blender(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # RF stored coord convention (from Max exporter docs/ksy):
    #   stored.x = -max.x
    #   stored.y =  max.z
    #   stored.z = -max.y
    # Convert to Blender coords (keep consistent & predictable):
    #   blender.x = -stored.x
    #   blender.y =  stored.y?  NO -> map to (x, y, z) = (-x, z, -y) from stored basis
    x, y, z = v
    return (-x, z, -y)

def read_vec3_to_blender(b: Bin) -> Tuple[float, float, float]:
    # Convenience wrapper: read raw RF vec3 then convert once.
    return rf_to_blender(read_vec3_rf(b))
def read_quat(b: Bin) -> Tuple[float, float, float, float]:
    # left as-is; not needed for OBJ export
    return (b.f32(), b.f32(), b.f32(), b.f32())

def read_uv_to_blender(b: Bin) -> Tuple[float, float]:
    u = b.f32()
    v = b.f32()
    # stored v is negated from Max; invert back
    return (u, -v)

# ---- model structures ----

@dataclass
class Face:
    vi: Tuple[int, int, int]      # vertex indices
    uvi: Tuple[int, int, int]     # per-corner uv indices (we'll build these)
    mat: int

@dataclass
class MeshOut:
    name: str
    verts: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]
    faces: List[Face]
    materials_used: List[int]

    # Animation / frame info (optional; used for glTF morph export)
    fps: int = 15
    start_time: float = 0.0
    end_time: float = 0.0
    num_frames: int = 1
    morph: bool = False
    flags: int = 0

    # For morph meshes: per-frame UNSPLIT positions in Blender coords
    frames_pos: Optional[List[List[Tuple[float, float, float]]]] = None
    # For split vertices: mapping split-vertex -> original position index
    split_pos_index: Optional[List[int]] = None

@dataclass
class MaterialOut:
    index: int
    tex0: str = ""
    tex1: str = ""
    additive: bool = False

# ---- VFX parsing ----

# section_type enum values (little-endian ints of tags), from spec
SEC_MESH   = 0x4F584653  # 'SFXO' in bytes
SEC_MATL   = 0x4C54414D  # 'MATL'
SEC_DMMY   = 0x594D4D44  # 'DMMY'
SEC_PART   = 0x54524150  # 'PART'
SEC_CHNE   = 0x454E4843  # 'CHNE'
SEC_WARP   = 0x50524157  # 'WARP'
SEC_ALGT   = 0x54474C41  # 'ALGT'
SEC_CMRA   = 0x41524D43  # 'CMRA'
SEC_MMOD   = 0x444F4D4D  # 'MMOD'

def parse_header(b: Bin) -> dict:
    magic = b.read(4)
    if magic != b"VSFX":
        raise ValueError("Not a VSFX/VFX file")
    version = b.s32()
    hdr = {"version": version}

    # version >= 0x30008 has flags
    if version >= 0x30008:
        hdr["flags"] = b.s32()

    # end_frame then a bunch of counts (we don't need them to extract geometry)
    hdr["end_frame"] = b.s32()
    hdr["num_meshes"] = b.s32()
    hdr["num_lights"] = b.s32()
    hdr["num_dummies"] = b.s32()
    hdr["num_particle_systems"] = b.s32()
    hdr["num_spacewarps"] = b.s32()
    hdr["num_cameras"] = b.s32()

    # version >= 0x3000F: selsets count
    if version >= 0x3000F:
        hdr["num_selsets"] = b.s32()

    # version >= 0x40000+: materials and frame counts
    if version >= 0x40000:
        hdr["num_materials"] = b.s32()
    if version >= 0x40002:
        hdr["num_mix_frames"] = b.s32()
    if version >= 0x40003:
        hdr["num_self_illumination_frames"] = b.s32()
    if version >= 0x40005:
        hdr["num_opacity_frames"] = b.s32()

    # faces + other totals
    hdr["num_faces"] = b.s32()
    hdr["num_mesh_material_indices"] = b.s32()
    hdr["num_vertex_normals"] = b.s32()
    hdr["num_adjacent_faces"] = b.s32()
    hdr["num_mesh_frames"] = b.s32()
    if version >= 0x3000D:
        hdr["num_uv_frames"] = b.s32()
    if version >= 0x30009:
        hdr["num_mesh_transform_frames"] = b.s32()
        hdr["num_mesh_transform_keyframe_lists"] = b.s32()
        hdr["num_mesh_translation_keys"] = b.s32()
        hdr["num_mesh_rotation_keys"] = b.s32()
        hdr["num_mesh_scale_keys"] = b.s32()

    hdr["num_light_frames"] = b.s32()
    hdr["num_dummy_frames"] = b.s32()
    hdr["num_part_sys_frames"] = b.s32()
    hdr["num_spacewarp_frames"] = b.s32()
    hdr["num_camera_frames"] = b.s32()

    if version >= 0x3000F:
        hdr["num_selset_objects"] = b.s32()

    return hdr


def parse_material_texture(b: Bin, version: int) -> str:
    name = b.cstr()
    if version >= 0x30012:
        _ = b.s32()   # start_frame
        _ = b.f32()   # playback_rate
        _ = b.s32()   # anim_type
    return name

def parse_matl_section(b: Bin, version: int, section_end: int, index: int) -> MaterialOut:
    # Based on rf-reversed vfx.ksy (rafalh). We keep texture name(s) + additive flag.
    mat_type = b.s32()
    if version >= 0x40003:
        _ = b.s32()  # frames_per_second

    additive = False
    if (mat_type in (0, 2)) or (version >= 0x40006):
        additive = (b.s8() != 0)

    tex0 = ""
    tex1 = ""

    if mat_type in (0, 2):
        tex0 = parse_material_texture(b, version)
    if mat_type == 2:
        tex1 = parse_material_texture(b, version)
        num_mix_frames = b.s32()
        if version < 0x40003:
            _ = b.s32()  # frames_per_second_legacy
        for _ in range(num_mix_frames):
            _ = b.f32()  # mix_frames

    if mat_type in (0, 2):
        _ = b.f32()  # specular_level
        _ = b.f32()  # glossiness
        _ = b.f32()  # reflection_amount
        _ = b.cstr() # refl_tex_name
    elif mat_type == 1:
        _ = b.s32(); _ = b.s32(); _ = b.s32()  # solid_color rgb_s4

    if version >= 0x40003:
        num_self = b.s32()
        for _ in range(max(0, num_self)):
            _ = b.f32()
    else:
        _ = b.f32()

    if version >= 0x40005:
        num_op = b.s32()
        for _ in range(max(0, num_op)):
            _ = b.f32()

    if b.tell() < section_end:
        b.seek(section_end)

    return MaterialOut(index=index, tex0=tex0, tex1=tex1, additive=additive)


def parse_mesh_section(b: Bin, version: int, section_end: int) -> Optional[MeshOut]:
    name = b.cstr()
    parent = b.cstr()
    _save_parent = b.s8()  # bool-ish

    num_vertices = b.s32()

    # version < 0x3000A has an unused vec3 array; not used for v4.6
    if version < 0x3000A:
        for _ in range(num_vertices):
            _ = read_vec3_to_blender(b)

    num_faces = b.s32()

    # faces: read indices (needed). Skip the rest.
    face_indices: List[Tuple[int, int, int]] = []
    face_mat: List[int] = []

    face_fv_indices: List[Tuple[int, int, int]] = []  # indices into face-vertex table

    for _ in range(num_faces):
        i0 = b.s32(); i1 = b.s32(); i2 = b.s32()  # vertex indices
        face_indices.append((i0, i1, i2))
        # version < 0x3000D had per-face UVs here; v4.6 does NOT.
        if version < 0x3000D:
            for __ in range(3):
                _ = read_uv_to_blender(b)

        # colors: 3 * rgb_f4 (9 floats)
        for __ in range(9):
            _ = b.f32()

        # normal + center vec3 + radius
        _ = read_vec3_to_blender(b)
        _ = read_vec3_to_blender(b)
        _ = b.f32()

        mat_index = b.s32()  # v4.6: 0-based material index OR -1
        face_mat.append(mat_index)

        _ = b.s32()  # smoothing_group

        # face_vertex_indices (3)  ✅ use these as the actual triangle indices
        fvi0 = b.s32(); fvi1 = b.s32(); fvi2 = b.s32()
        face_fv_indices.append((fvi0, fvi1, fvi2))
        # (no append here)
    # mesh timing / frame info
    frames_per_second = 15
    if version >= 0x30009:
        frames_per_second = b.s32()

    if version >= 0x40004:
        _start_time = b.f32()
        _end_time = b.f32()
        num_frames = b.s32()
    else:
        _start_frame = b.s32()
        _end_frame = b.s32()
        num_frames = (_end_frame - _start_frame + (1 if version >= 0x3000C else 0))




    # materials list
    num_materials = b.s32()
    if version >= 0x40000:
        for _ in range(num_materials):
            _ = b.s32()  # materials_indices
    else:
        # old material format; not implemented
        return None

    # bounding sphere
    _ = read_vec3_to_blender(b)
    _ = b.f32()

    # mesh flags
    flags_raw = b.u32()
    facing = (flags_raw & 0x00000001) != 0
    morph = (flags_raw & 0x00000004) != 0
    frames_pos = [] if (CAPTURE_FRAMES and morph) else None
    base_verts = None
    if DEBUG_FRAMES:
        print(f"[DEBUG_FRAMES] mesh='{name}' fps={frames_per_second} frames={num_frames} morph={morph} flags=0x{flags_raw:08X}")

    dump_uvs = (flags_raw & 0x00000100) != 0
    facing_rod = (flags_raw & 0x00000800) != 0

    # width/height special case in version 0x3000A only; skip

    # face-vertex (vertex normal) table
    num_face_vertices = b.s32()
    fv_vertex_index: List[int] = []  # maps face_vertex -> position vertex index
    for _ in range(num_face_vertices):
        _ = b.s32()  # smoothing_group
        v_idx = b.s32()  # vertex_index
        fv_vertex_index.append(v_idx)
        _ = b.f32()  # u (unused in newer versions)
        _ = b.f32()  # v (unused in newer versions)
        n_adj = b.s32()
        for __ in range(n_adj):
            _ = b.s32()

    is_keyframed = True
    if version >= 0x30009:
        is_keyframed = (b.u8() != 0)

    # ---- frames ----
    # We only need frame 0 positions + (frame 0 OR dump_uvs) UVs.
    verts: List[Tuple[float, float, float]] = []
    uvs_per_corner: List[Tuple[float, float]] = []

    for frame_idx in range(num_frames):
        # For morph or frame 0: has compressed positions
        if morph or frame_idx == 0:
            center_rf = read_vec3_rf(b)
            mult_rf = read_vec3_rf(b)
            # compressed vec3_s2
            raw = []
            for _ in range(num_vertices):
                rx = b.s16(); ry = b.s16(); rz = b.s16()
                raw.append((rx, ry, rz))

            if frame_idx == 0:
                # decompress
                for (rx, ry, rz) in raw:
                    nx = rx / S16_MAX
                    ny = ry / S16_MAX
                    nz = rz / S16_MAX
                    vx = center_rf[0] + mult_rf[0] * nx
                    vy = center_rf[1] + mult_rf[1] * ny
                    vz = center_rf[2] + mult_rf[2] * nz
                    verts.append(rf_to_blender((vx, vy, vz)))
            if frame_idx == 0:
                base_verts = verts
            if frames_pos is not None:
                frames_pos.append(verts)
            # billboard widths/heights can appear for facing/facing_rod in newer versions
            if (facing or facing_rod) and version >= 0x3000B:
                _ = b.f32(); _ = b.f32()
            if facing_rod and frame_idx == 0 and version >= 0x40001:
                _ = read_vec3_to_blender(b)  # up_vector

        if base_verts is not None:
            verts = base_verts

        if base_verts is not None:
            verts = base_verts

        # UV block: (dump_uvs OR frame 0) and version >= 0x3000D
        if (dump_uvs or frame_idx == 0) and version >= 0x3000D:
            # 3 * num_faces uvs (per face corner)
            if frame_idx == 0:
                for _ in range(3 * num_faces):
                    uvs_per_corner.append(read_uv_to_blender(b))
            else:
                # skip
                b.read(8 * 3 * num_faces)

        # optional transforms (skip safely)
        if not morph and (not is_keyframed or (version < 0x3000E and frame_idx == 0)):
            _ = read_vec3_to_blender(b)
            _ = read_quat(b)
            _ = read_vec3_to_blender(b)

        if version < 0x30009:
            b.read(1)

        if version < 0x40005:
            _ = b.f32()  # opacity

    # keyframed pivot + keyframes (skip)
    if is_keyframed and version >= 0x3000A:
        _ = read_vec3_to_blender(b)
        _ = read_quat(b)
        _ = read_vec3_to_blender(b)

    if is_keyframed:
        # translation keys
        n = b.s32()
        b.read(n * (4 + 12 + 12 + 12))  # time + vec3 + inTan + outTan

        # rotation keys
        n = b.s32()
        b.read(n * (4 + 16 + 4*5))  # time + quat + 5 floats

        # scale keys
        n = b.s32()
        b.read(n * (4 + 12 + 12 + 12))

    # If we didn't get what we need, bail
    if not verts or not uvs_per_corner:
        return None

    # Build OBJ-friendly indexing: per face corner vertex/uv pairs
    out_verts: List[Tuple[float, float, float]] = []
    out_uvs: List[Tuple[float, float]] = []
    out_faces: List[Face] = []
    split_pos_index: List[int] = []

    vert_uv_map = {}  # (v_idx, corner_uv) -> new_index

    def get_vt(v_idx: int, uv: Tuple[float, float]) -> Tuple[int, int]:
        key = (v_idx, uv)
        if key in vert_uv_map:
            return vert_uv_map[key]
        out_verts.append(verts[v_idx])
        split_pos_index.append(v_idx)
        out_uvs.append(uv)
        new_i = len(out_verts)  # OBJ is 1-based
        new_t = len(out_uvs)
        vert_uv_map[key] = (new_i, new_t)
        return (new_i, new_t)

    for f_i, (i0, i1, i2) in enumerate(face_indices):
        # In RF VFX, the real topology is usually defined by:
        # face_vertex_indices -> face_vertices[].vertex_index
        if face_fv_indices and fv_vertex_index and f_i < len(face_fv_indices):
            fvi0, fvi1, fvi2 = face_fv_indices[f_i]
            if (0 <= fvi0 < len(fv_vertex_index) and 0 <= fvi1 < len(fv_vertex_index) and 0 <= fvi2 < len(fv_vertex_index)):
                i0 = fv_vertex_index[fvi0]
                i1 = fv_vertex_index[fvi1]
                i2 = fv_vertex_index[fvi2]

        uv0 = uvs_per_corner[f_i*3 + 0]
        uv1 = uvs_per_corner[f_i*3 + 1]
        uv2 = uvs_per_corner[f_i*3 + 2]
        vi0, vi1, vi2 = i0, i1, i2

        v0, t0 = get_vt(vi0, uv0)
        v1, t1 = get_vt(vi1, uv1)
        v2, t2 = get_vt(vi2, uv2)
        out_faces.append(Face((v0, v1, v2), (t0, t1, t2), face_mat[f_i]))

    materials_used = sorted({f.mat for f in out_faces if f.mat >= 0})

    return MeshOut(
        name=name,
        verts=out_verts,
        uvs=out_uvs,
        faces=out_faces,
        materials_used=materials_used,
        fps=frames_per_second,
        start_time=0.0,
        end_time=0.0,
        num_frames=num_frames,
        morph=morph,
        flags=flags_raw,
        frames_pos=frames_pos,
        split_pos_index=split_pos_index,
    )


def write_gltf(path_gltf: str, mesh: MeshOut, mats: List[Material], scale: float) -> None:
    """
    Writes a minimal glTF 2.0 (.gltf + .bin) with:
      - POSITION, TEXCOORD_0
      - material groups (separate primitives)
      - morph-target animation if mesh.frames_pos exists and mesh.morph is True
    """
    import json

    base_no_ext, _ = os.path.splitext(path_gltf)
    path_bin = base_no_ext + ".bin"

    # Group faces by material index
    faces_by_mat: Dict[int, List[Face]] = {}
    for f in mesh.faces:
        faces_by_mat.setdefault(f.mat, []).append(f)

    # Flatten vertex attributes
    pos_f: List[float] = []
    for (x,y,z) in mesh.verts:
        pos_f.extend([x*scale, y*scale, z*scale])
    uv_f: List[float] = []
    for (u,v) in mesh.uvs:
        uv_f.extend([u, v])

    def pack_f32(arr: List[float]) -> bytes:
        return struct.pack("<" + "f"*len(arr), *arr)

    def pack_u32(arr: List[int]) -> bytes:
        return struct.pack("<" + "I"*len(arr), *arr)

    buffers = bytearray()

    def align4():
        while (len(buffers) % 4) != 0:
            buffers.extend(b"\\x00")

    gltf: Dict = {
        "asset": {"version": "2.0", "generator": "vfx2obj"},
        "buffers": [{"uri": os.path.basename(path_bin), "byteLength": 0}],
        "bufferViews": [],
        "accessors": [],
        "materials": [],
        "textures": [],
        "images": [],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "meshes": [],
        "nodes": [],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    def add_view(data: bytes, target: Optional[int]=None) -> int:
        align4()
        offset = len(buffers)
        buffers.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        gltf["bufferViews"].append(view)
        return len(gltf["bufferViews"]) - 1

    def add_accessor(view_idx: int, comp_type: int, count: int, type_str: str, mn=None, mx=None) -> int:
        acc = {"bufferView": view_idx, "componentType": comp_type, "count": count, "type": type_str}
        if mn is not None: acc["min"] = mn
        if mx is not None: acc["max"] = mx
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    # POSITION accessor
    pos_view = add_view(pack_f32(pos_f), target=34962)
    xs = pos_f[0::3]; ys = pos_f[1::3]; zs = pos_f[2::3]
    pos_acc = add_accessor(pos_view, 5126, len(mesh.verts), "VEC3",
                           [min(xs), min(ys), min(zs)],
                           [max(xs), max(ys), max(zs)])

    # UV accessor
    uv_view = add_view(pack_f32(uv_f), target=34962)
    us = uv_f[0::2]; vs = uv_f[1::2]
    uv_acc = add_accessor(uv_view, 5126, len(mesh.uvs), "VEC2",
                          [min(us) if us else 0.0, min(vs) if vs else 0.0],
                          [max(us) if us else 1.0, max(vs) if vs else 1.0])

    # Materials (very basic PBR)
    mat_map: Dict[int,int] = {}
    def ensure_material(mi: int) -> int:
        if mi in mat_map:
            return mat_map[mi]
        if mi < 0 or mi >= len(mats):
            gltf["materials"].append({"name":"default","pbrMetallicRoughness":{"metallicFactor":0.0,"roughnessFactor":1.0}})
            mat_map[mi] = len(gltf["materials"]) - 1
            return mat_map[mi]
        m = mats[mi]
        mat: Dict = {"name": f"mat_{mi}",
                     "pbrMetallicRoughness":{"metallicFactor":0.0,"roughnessFactor":1.0}}
        if m.tex0:
            gltf["images"].append({"uri": m.tex0})
            img_idx = len(gltf["images"]) - 1
            gltf["textures"].append({"sampler": 0, "source": img_idx})
            tex_idx = len(gltf["textures"]) - 1
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}
        gltf["materials"].append(mat)
        mat_map[mi] = len(gltf["materials"]) - 1
        return mat_map[mi]

    # Morph targets (POSITION deltas)
    targets_accessors: Optional[List[int]] = None
    anim: Optional[Dict] = None
    if mesh.frames_pos is not None and mesh.morph and mesh.split_pos_index:
        base_unsplit = mesh.frames_pos[0]
        targets_accessors = []
        for fi in range(1, len(mesh.frames_pos)):
            frame_unsplit = mesh.frames_pos[fi]
            deltas: List[float] = []
            for pos_idx in mesh.split_pos_index:
                bx,by,bz = base_unsplit[pos_idx]
                fx,fy,fz = frame_unsplit[pos_idx]
                deltas.extend([(fx-bx)*scale, (fy-by)*scale, (fz-bz)*scale])
            view = add_view(pack_f32(deltas), target=34962)
            dx = deltas[0::3]; dy = deltas[1::3]; dz = deltas[2::3]
            acc = add_accessor(view, 5126, len(mesh.verts), "VEC3",
                               [min(dx), min(dy), min(dz)],
                               [max(dx), max(dy), max(dz)])
            targets_accessors.append(acc)

        # Animate weights stepping through targets
        times = [i / float(mesh.fps if mesh.fps else 15) for i in range(len(mesh.frames_pos))]
        in_view = add_view(pack_f32(times))
        in_acc = add_accessor(in_view, 5126, len(times), "SCALAR", [min(times)], [max(times)])

        tcount = len(targets_accessors)
        out_weights: List[float] = []
        for i in range(len(times)):
            w = [0.0]*tcount
            if i > 0 and (i-1) < tcount:
                w[i-1] = 1.0
            out_weights.extend(w)

        out_view = add_view(pack_f32(out_weights))
        out_acc = add_accessor(out_view, 5126, len(times), f"VEC{tcount}")

        anim = {
            "samplers": [{"input": in_acc, "output": out_acc, "interpolation": "STEP"}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
        }

    # Build primitives
    prims: List[Dict] = []
    for mi, faces in faces_by_mat.items():
        indices: List[int] = []
        for f in faces:
            a,b,c = f.vi
            indices.extend([a,b,c])
        idx_view = add_view(pack_u32(indices), target=34963)
        idx_acc = add_accessor(idx_view, 5125, len(indices), "SCALAR", [min(indices)], [max(indices)])
        prim: Dict = {"attributes":{"POSITION":pos_acc,"TEXCOORD_0":uv_acc},
                      "indices": idx_acc,
                      "material": ensure_material(mi)}
        if targets_accessors:
            prim["targets"] = [{"POSITION": a} for a in targets_accessors]
        prims.append(prim)

    gltf["meshes"].append({"name": mesh.name, "primitives": prims})
    gltf["nodes"].append({"mesh": 0, "name": mesh.name})
    if anim:
        gltf["animations"] = [anim]

    gltf["buffers"][0]["byteLength"] = len(buffers)

    with open(path_bin, "wb") as f:
        f.write(buffers)
    with open(path_gltf, "w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)

    print(f"Wrote: {path_gltf}")
    print(f"Wrote: {path_bin}")

def write_obj(path_obj: str, mesh: MeshOut, scale: float = 10000.0, materials: Optional[List[MaterialOut]] = None) -> None:
    base = os.path.splitext(path_obj)[0]
    path_mtl = base + ".mtl"
    mtl_name = os.path.basename(path_mtl)

    # Write MTL (one entry per MATL section; fallback to a single default material)
    if not materials:
        materials = [MaterialOut(index=0, tex0="", tex1="", additive=False)]

    with open(path_mtl, "w", encoding="utf-8") as f:
        for mat in materials:
            mname = f"material{mat.index}"
            f.write(f"newmtl {mname}\n")
            f.write("Kd 1.0 1.0 1.0\n")
            tex = (mat.tex0 or "").strip()
            if tex:
                if "." not in os.path.basename(tex):
                    tex = tex + ".tga"
                f.write(f"map_Kd {tex}\n")
            f.write("\n")
    with open(path_obj, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"o {mesh.name}\n")

        for (x, y, z) in mesh.verts:
            f.write(f"v {x*scale:.6f} {y*scale:.6f} {z*scale:.6f}\n")

        for (u, v) in mesh.uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")

        current_mtl = None
        for face in mesh.faces:
            mi = face.mat
            if mi is None or mi < 0:
                mi = 0
            mtl = f"material{mi}"
            if mtl != current_mtl:
                f.write(f"usemtl {mtl}\n")
                current_mtl = mtl
            (v0, v1, v2) = face.vi
            (t0, t1, t2) = face.uvi
            f.write(f"f {v0}/{t0} {v1}/{t1} {v2}/{t2}\n")

def convert_file(path_vfx: str, scale: float = 10000.0) -> None:
    with open(path_vfx, "rb") as f:
        data = f.read()

    b = Bin(data)
    hdr = parse_header(b)
    version = hdr["version"]

    meshes: List[MeshOut] = []
    materials: List[MaterialOut] = []
    matl_index = 0

    data_len = len(data)

    # Walk sections until EOF
    while b.tell() < data_len:
        # Need at least 8 bytes for (type, len)
        if b.tell() + 8 > data_len:
            break

        sec_type = b.s32()
        sec_len  = b.s32()

        # Defensive checks (corrupt/odd files won't explode)
        if sec_len < 8:
            raise RuntimeError(f"Bad section length {sec_len} at offset {b.tell()-8}")

        section_start = b.tell()
        # Spec convention used by this script: length includes the 4-byte 'type'
        section_end = section_start + (sec_len - 4)

        if section_end > data_len or section_end < section_start:
            raise RuntimeError(f"Bad section end {section_end} (len={sec_len}) at offset {section_start}")

        if sec_type == SEC_MESH:
            m = parse_mesh_section(b, version, section_end)
            if m:
                meshes.append(m)
            b.seek(section_end)
        elif sec_type == SEC_MATL:
            mat = parse_matl_section(b, version, section_end, matl_index)
            materials.append(mat)
            matl_index += 1
            # parse_matl_section already seeks to section_end
        else:
            # skip unhandled sections
            b.seek(section_end)

    if not meshes:
        raise RuntimeError("No supported mesh sections found in this VFX.")

    base = os.path.splitext(path_vfx)[0]
    for m in meshes:
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in m.name).strip()
        if not safe:
            safe = "mesh"
        if EXPORT_OBJ:
            out_obj = f"{base}__{safe}.obj"
            write_obj(out_obj, m, scale=scale, materials=materials)
            print(f"Wrote: {out_obj}")
        if EXPORT_GLTF:
            out_gltf = f"{base}__{safe}.gltf"
            write_gltf(out_gltf, m, materials, scale=scale)

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Drag and drop .vfx onto this tool, or run:")
        print("  vfx2obj.exe [--scale 10000] [--gltf|--both] file1.vfx [file2.vfx ...]")
        return 2

    global EXPORT_OBJ, EXPORT_GLTF, CAPTURE_FRAMES

    scale = 10000.0
    args = list(argv[1:])

    EXPORT_OBJ = True
    EXPORT_GLTF = False
    CAPTURE_FRAMES = False

    # simple option parsing (no argparse to keep PyInstaller small)
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--':
            del args[i:i+1]
            break
        if not a.startswith('-'):
            break
        if a in ('--gltf',):
            EXPORT_GLTF = True
            CAPTURE_FRAMES = True
            EXPORT_OBJ = False
            del args[i:i+1]
            continue
        if a in ('--both',):
            EXPORT_OBJ = True
            EXPORT_GLTF = True
            CAPTURE_FRAMES = True
            del args[i:i+1]
            continue
        if a in ('--obj',):
            EXPORT_OBJ = True
            EXPORT_GLTF = False
            CAPTURE_FRAMES = False
            del args[i:i+1]
            continue
        if a in ('--debug-frames',):
            global DEBUG_FRAMES
            DEBUG_FRAMES = True
            del args[i:i+1]
            continue
        if a in ('--scale', '-s'):
            if i + 1 >= len(args):
                raise SystemExit('Missing value after --scale')
            scale = float(args[i + 1])
            del args[i:i+2]
            continue
        if a.startswith('--scale='):
            scale = float(a.split('=', 1)[1])
            del args[i:i+1]
            continue
        # Unknown flag: drop it so it doesn't get treated as a filename
        print(f'[WARN] Ignoring unknown option: {a}')
        del args[i:i+1]
        continue

    if not args:
        print("No .vfx files provided.")
        return 2

    for p in args:
        if os.path.isfile(p):
            try:
                convert_file(p, scale=scale)
            except Exception as e:
                print(f"[ERROR] {p}: {e}")
        else:
            print(f"[SKIP] Not a file: {p}")
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))



# Export options set by CLI
EXPORT_OBJ = True
EXPORT_GLTF = False
CAPTURE_FRAMES = False


