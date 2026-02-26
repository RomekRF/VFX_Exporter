#!/usr/bin/env python3
# vfx2obj.py - Drag/drop VFX -> OBJ for Blender
# Supports Red Faction VFX version 0x00040006 ("VFX V 4.6")

import os
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
    translation: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float, float]] = None
    scale_vec: Optional[Tuple[float, float, float]] = None
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
    base_translation = None  # captured placement (used for glTF vertex offset)
    base_rotation = None
    base_scale = None

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

            # decompress (for morph meshes: every frame has its own center/mult; for non-morph: only frame 0 has positions)
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

        # optional transforms (capture)
        if not morph and (not is_keyframed or (version < 0x3000E and frame_idx == 0)):
            t = read_vec3_to_blender(b)
            r = read_quat(b)
            s = read_vec3_to_blender(b)
            if frame_idx == 0 and base_translation is None:
                base_translation = t
                base_rotation = r
                base_scale = s

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
        # translation keys (capture first key pos)
        n = b.s32()
        if n > 0:
            _t = b.f32()  # time
            _p = read_vec3_to_blender(b)  # position
            b.read(12 + 12)  # inTan + outTan
            if base_translation is None:
                base_translation = _p
            if n > 1:
                b.read((n - 1) * (4 + 12 + 12 + 12))
        else:
            pass

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
            translation=base_translation,
        rotation=base_rotation,
        scale_vec=base_scale,
    )


def _gltf_pack_scene(meshes: List[MeshOut], materials: List[MaterialOut], out_dir: str, base_name: str, scale: float = 10000.0, trs_scale: float = 1.0, center_root: bool = False, center_geom: bool = False, bake_origin: bool = False) -> Tuple[str, str]:
    scene_nodes = [] 
    """
    Build a single .gltf + .bin containing all meshes as separate nodes.
    - Keeps meshes editable as separate objects in Blender.
    - Uses 0-based indices in glTF (RF VFX indices are 1-based).
    Returns (path_gltf, path_bin)
    """
    import json
    import struct
    import base64

    # Helper: align binary blob to 4 bytes
    bin_blob = bytearray()

    def align4() -> None:
        while (len(bin_blob) % 4) != 0:
            bin_blob.append(0)

    def push_bytes(b: bytes) -> Tuple[int, int]:
        off = len(bin_blob)
        bin_blob.extend(b)
        align4()
        return off, len(b)

    def push_f32(arr) -> Tuple[int, int]:
        return push_bytes(struct.pack("<%sf" % len(arr), *arr))

    def push_u16(arr) -> Tuple[int, int]:
        return push_bytes(struct.pack("<%sH" % len(arr), *arr))

    def push_u32(arr) -> Tuple[int, int]:
        return push_bytes(struct.pack("<%sI" % len(arr), *arr))

    gltf = {
        "asset": {"version": "2.0", "generator": "vfx2obj"},
        "scenes": [{"nodes": []}],
        "scene": 0,
        "nodes": [],
        "meshes": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "materials": [],
        "images": [],
        "textures": [],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
    }

    # Materials + textures
    tex_uri_to_index: Dict[str, int] = {}
    for mat in materials:
        # Ensure we have a PBR material even if no texture name exists
        pbr = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}
        texname = (mat.tex0 or "").strip()
        if texname:
            if texname not in tex_uri_to_index:
                img_i = len(gltf["images"])
                tex_i = len(gltf["textures"])
                gltf["images"].append({"uri": texname})
                gltf["textures"].append({"source": img_i, "sampler": 0})
                tex_uri_to_index[texname] = tex_i
            pbr["baseColorTexture"] = {"index": tex_uri_to_index[texname]}
        gltf["materials"].append({"name": f"mat_{mat.index}", "pbrMetallicRoughness": pbr, "doubleSided": True})

    # If there are no materials in the VFX, still provide a default one
    if not gltf["materials"]:
        gltf["materials"].append({"name": "default", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}, "doubleSided": True})

    # Mesh packing
    for mesh in meshes:
        tx = ty = tz = 0.0
        bt = getattr(mesh, "translation", None)
        if bt is not None:
            tx = (bt.x if hasattr(bt,"x") else bt[0])
            ty = (bt.y if hasattr(bt,"y") else bt[1])
            tz = (bt.z if hasattr(bt,"z") else bt[2])
        # Positions
        pos = []
        minx = miny = minz = float("inf")
        maxx = maxy = maxz = float("-inf")
        for v in mesh.verts:
            x = ((v.x if hasattr(v,'x') else v[0])) * scale
            y = ((v.y if hasattr(v,'y') else v[1])) * scale
            z = ((v.z if hasattr(v,'z') else v[2])) * scale
            pos.extend([x, y, z])
            minx = min(minx, x); miny = min(miny, y); minz = min(minz, z)
            maxx = max(maxx, x); maxy = max(maxy, y); maxz = max(maxz, z)

        pos_off, pos_len = push_f32(pos)
        bv_pos = len(gltf["bufferViews"])
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962})
        acc_pos = len(gltf["accessors"])
        gltf["accessors"].append({
            "bufferView": bv_pos,
            "componentType": 5126,
            "count": len(mesh.verts),
            "type": "VEC3",
            "min": [minx, miny, minz],
            "max": [maxx, maxy, maxz],
        })

        # UVs (if missing, write zeros)
        uvs = []
        if mesh.uvs and len(mesh.uvs) == len(mesh.verts):
            for uv in mesh.uvs:
                uvs.extend([(uv.u if hasattr(uv,'u') else uv[0]), 1.0 - (uv.v if hasattr(uv,'v') else uv[1])])
        else:
            uvs = [0.0, 0.0] * len(mesh.verts)

        uv_off, uv_len = push_f32(uvs)
        bv_uv = len(gltf["bufferViews"])
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": uv_off, "byteLength": uv_len, "target": 34962})
        acc_uv = len(gltf["accessors"])
        gltf["accessors"].append({"bufferView": bv_uv, "componentType": 5126, "count": len(mesh.verts), "type": "VEC2"})

        # Indices grouped by material
        faces_by_mat: Dict[int, List[int]] = {}
        for f in mesh.faces:
            # RF VFX face.vi is 1-based indices into mesh.verts
            i0, i1, i2 = f.vi
            i0 -= 1; i1 -= 1; i2 -= 1
            # Validate indices; skip broken triangles
            if (i0 < 0 or i1 < 0 or i2 < 0 or
                i0 >= len(mesh.verts) or i1 >= len(mesh.verts) or i2 >= len(mesh.verts)):
                continue
            mi = f.mat
            faces_by_mat.setdefault(mi, []).extend([i0, i1, i2])

        # If no faces survived validation, make an empty mesh
        primitives = []
        if faces_by_mat:
            for mi, idxs in sorted(faces_by_mat.items(), key=lambda kv: kv[0]):
                # choose 16-bit unless needed
                if idxs and max(idxs) <= 65535:
                    ind_off, ind_len = push_u16(idxs)
                    comp = 5123
                else:
                    ind_off, ind_len = push_u32(idxs)
                    comp = 5125
                bv_ind = len(gltf["bufferViews"])
                gltf["bufferViews"].append({"buffer": 0, "byteOffset": ind_off, "byteLength": ind_len, "target": 34963})
                acc_ind = len(gltf["accessors"])
                gltf["accessors"].append({"bufferView": bv_ind, "componentType": comp, "count": len(idxs), "type": "SCALAR"})

                mat_idx = mi if (0 <= mi < len(gltf["materials"])) else 0
                primitives.append({
                    "attributes": {"POSITION": acc_pos, "TEXCOORD_0": acc_uv},
                    "indices": acc_ind,
                    "material": mat_idx,
                })
        else:
            primitives.append({"attributes": {"POSITION": acc_pos, "TEXCOORD_0": acc_uv}, "material": 0})

        mesh_i = len(gltf["meshes"])
        gltf["meshes"].append({"name": mesh.name or "mesh", "primitives": primitives})

        node_i = len(gltf["nodes"])
        node = {"name": mesh.name or f"mesh_{mesh_i}", "mesh": mesh_i}
        t = getattr(mesh, "translation", None)
        r = getattr(mesh, "rotation", None)
        s = getattr(mesh, "scale_vec", None)
        if trs_scale is None:
            trs_scale = scale

        # Auto TRS scaling: if user scaled vertices but didn't set --trs-scale, follow --scale
        if (trs_scale is None) or ((trs_scale == 1.0) and (scale != 1.0)):
            trs_scale = scale

        if t is not None: node["translation"] = [float(t[0]) * trs_scale, float(t[1]) * trs_scale, float(t[2]) * trs_scale]
        if r is not None: node["rotation"] = list(r)
        if s is not None: node["scale"] = list(s)
        gltf["nodes"].append(node)
        scene_nodes.append(node_i)
    # Rootify: one clean scene root for Blender
    root_idx = len(gltf['nodes'])
    root_node = {'name': '__VFX_ROOT__', 'children': scene_nodes, 'rotation': [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]}

    def _qmul(a,b):
        ax,ay,az,aw=a; bx,by,bz,bw=b
        return (
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz
        )
    def _qconj(q):
        x,y,z,w=q; return (-x,-y,-z,w)
    def _qrot(q, v3):  # q is glTF [x,y,z,w]
        vx,vy,vz = v3
        p = (vx,vy,vz,0.0)
        return _qmul(_qmul(q,p), _qconj(q))[:3]

    if center_root:
        sx = sy = sz = 0.0
        nmesh = 0
        for n in gltf['nodes']:
            if isinstance(n, dict) and ('mesh' in n):
                t = n.get('translation')
                if isinstance(t, list) and len(t) == 3:
                    sx += float(t[0]); sy += float(t[1]); sz += float(t[2])
                    nmesh += 1
        if nmesh > 0:
                        # center_root: translate root by -(R * avg(mesh_translation)) so world lands near origin
            avg = (sx / nmesh, sy / nmesh, sz / nmesh)
            q = root_node.get('rotation', [0.0,0.0,0.0,1.0])  # glTF [x,y,z,w]
            vx,vy,vz = _qrot(tuple(q), avg)
            root_node['translation'] = [-vx, -vy, -vz]

    gltf['nodes'].append(root_node)
    gltf['scenes'][0]['nodes'] = [root_idx]
    # Buffer reference (external .bin)
    if center_geom:
        # Recenter POSITION accessor 0 (float32 VEC3) by editing bin_blob in-place
        import struct
        try:
            acc0 = gltf['accessors'][0]
            bv0  = gltf['bufferViews'][acc0['bufferView']]
            base0 = int(bv0.get('byteOffset', 0)) + int(acc0.get('byteOffset', 0))
            cnt0  = int(acc0['count'])
            minx=miny=minz=1e30; maxx=maxy=maxz=-1e30
            for ii in range(cnt0):
                o = base0 + ii*12
                x,y,z = struct.unpack_from('<fff', bin_blob, o)
                if x<minx: minx=x
                if y<miny: miny=y
                if z<minz: minz=z
                if x>maxx: maxx=x
                if y>maxy: maxy=y
                if z>maxz: maxz=z
            cx=(minx+maxx)*0.5; cy=(miny+maxy)*0.5; cz=(minz+maxz)*0.5
            for ii in range(cnt0):
                o = base0 + ii*12
                x,y,z = struct.unpack_from('<fff', bin_blob, o)
                struct.pack_into('<fff', bin_blob, o, x-cx, y-cy, z-cz)
            if 'min' in acc0 and 'max' in acc0 and acc0.get('type') == 'VEC3':
                acc0['min'] = [float(acc0['min'][0]) - cx, float(acc0['min'][1]) - cy, float(acc0['min'][2]) - cz]
                acc0['max'] = [float(acc0['max'][0]) - cx, float(acc0['max'][1]) - cy, float(acc0['max'][2]) - cz]
        except Exception as e:
            print('[WARN] center-geom failed:', e)

    if bake_origin:

        # Bake root rotation into geometry AND recenter geometry so Blender shows mesh at 0,0,0.

        import struct

        def _qmul(a,b):

            ax,ay,az,aw=a; bx,by,bz,bw=b

            return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)

        def _qconj(q):

            x,y,z,w=q; return (-x,-y,-z,w)

        def _qrot(q, v3):

            vx,vy,vz=v3

            p=(vx,vy,vz,0.0)

            x,y,z,_ = _qmul(_qmul(q,p), _qconj(q))

            return (x,y,z)

        try:

            rix = int(gltf['scenes'][0]['nodes'][0])

            rnode = gltf['nodes'][rix]

            rq = rnode.get('rotation', [0.0,0.0,0.0,1.0])  # glTF [x,y,z,w]

            if not isinstance(bin_blob, (bytearray,)):

                bin_blob = bytearray(bin_blob)

            accessors = gltf.get('accessors', [])

            bvs = gltf.get('bufferViews', [])

            visited = set()

            def _process_accessor_pos(acc_i):

                if acc_i in visited: return

                visited.add(acc_i)

                acc = accessors[acc_i]

                if acc.get('type') != 'VEC3' or acc.get('componentType') != 5126:

                    return

                bv = bvs[acc['bufferView']]

                base = int(bv.get('byteOffset', 0)) + int(acc.get('byteOffset', 0))

                stride = int(bv.get('byteStride', 12) or 12)

                cnt = int(acc['count'])

                # 1) rotate all verts by root rotation; gather bounds

                minx=miny=minz=1e30; maxx=maxy=maxz=-1e30

                for ii in range(cnt):

                    o = base + ii*stride

                    x,y,z = struct.unpack_from('<fff', bin_blob, o)

                    rx,ry,rz = _qrot(tuple(rq), (float(x),float(y),float(z)))

                    struct.pack_into('<fff', bin_blob, o, float(rx),float(ry),float(rz))

                    if rx<minx: minx=rx

                    if ry<miny: miny=ry

                    if rz<minz: minz=rz

                    if rx>maxx: maxx=rx

                    if ry>maxy: maxy=ry

                    if rz>maxz: maxz=rz

                cx=(minx+maxx)*0.5; cy=(miny+maxy)*0.5; cz=(minz+maxz)*0.5

                # 2) recenter by subtracting bounds center; recompute bounds

                minx=miny=minz=1e30; maxx=maxy=maxz=-1e30

                for ii in range(cnt):

                    o = base + ii*stride

                    x,y,z = struct.unpack_from('<fff', bin_blob, o)

                    x=float(x)-cx; y=float(y)-cy; z=float(z)-cz

                    struct.pack_into('<fff', bin_blob, o, x,y,z)

                    if x<minx: minx=x

                    if y<miny: miny=y

                    if z<minz: minz=z

                    if x>maxx: maxx=x

                    if y>maxy: maxy=y

                    if z>maxz: maxz=z

                acc['min'] = [float(minx), float(miny), float(minz)]

                acc['max'] = [float(maxx), float(maxy), float(maxz)]

            # apply to every POSITION accessor referenced by any mesh primitive

            for n in gltf.get('nodes', []):

                if not isinstance(n, dict) or ('mesh' not in n):

                    continue

                mesh = gltf['meshes'][n['mesh']]

                for prim in mesh.get('primitives', []):

                    attrs = prim.get('attributes', {})

                    if 'POSITION' in attrs:

                        _process_accessor_pos(int(attrs['POSITION']))

                n['translation'] = [0.0,0.0,0.0]

            rnode['translation'] = [0.0,0.0,0.0]

            rnode['rotation'] = [0.0,0.0,0.0,1.0]

            print('[INFO] Baking origin: baked root rotation + recentered geometry; node TRS zeroed')

        except Exception as e:

            print('[WARN] bake-origin failed:', e)


    out_bin = os.path.join(out_dir, base_name + ".bin")
    out_gltf = os.path.join(out_dir, base_name + ".gltf")
    with open(out_bin, "wb") as f:
        f.write(bin_blob)

    gltf["buffers"] = [{"uri": os.path.basename(out_bin), "byteLength": len(bin_blob)}]
    with open(out_gltf, "w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)

    return out_gltf, out_bin


def write_gltf_scene(path_gltf: str, meshes: List[MeshOut], materials: List[MaterialOut], scale: float = 10000.0, trs_scale: float = 1.0, center_root: bool = False, center_geom: bool = False, bake_origin: bool = False) -> None:
    out_dir = os.path.dirname(path_gltf) or "."
    base_name = os.path.splitext(os.path.basename(path_gltf))[0]
    out_gltf, out_bin = _gltf_pack_scene(meshes, materials, out_dir, base_name, scale=scale, trs_scale=trs_scale, center_root=center_root, center_geom=center_geom, bake_origin=bake_origin)
    print(f"Wrote: {out_gltf}")
    print(f"Wrote: {out_bin}")

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

def convert_file(path_vfx: str, scale: float = 10000.0, trs_scale: float = 1.0, center_root: bool = False, center_geom: bool = False, bake_origin: bool = False) -> None:
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
    # If requested, export a *single* glTF scene containing all meshes
    if EXPORT_GLTF:
        write_gltf_scene(base + '.gltf', meshes, materials, scale=scale, trs_scale=trs_scale, center_root=center_root, center_geom=center_geom, bake_origin=bake_origin)
    for m in meshes:
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in m.name).strip()
        if not safe:
            safe = "mesh"
        if EXPORT_OBJ:
            out_obj = f"{base}__{safe}.obj"
            write_obj(out_obj, m, scale=scale, materials=materials)
            print(f"Wrote: {out_obj}")

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Drag and drop .vfx onto this tool, or run:")
        print("  vfx2obj.exe [--scale 10000] [--gltf|--both] file1.vfx [file2.vfx ...]")
        return 2

    global EXPORT_OBJ, EXPORT_GLTF, CAPTURE_FRAMES
    scale = 10000.0
    trs_scale = 1.0

    center_geom = False  # --center-geom
    center_root = False
    bake_origin = False  # --bake-origin: bake root+mesh transforms into vertices so Blender shows 0,0,0

    center_geom = False  # --center-geom: recenter POSITION vertices around origin 
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
        if a in ('--center','--center-root'):
            center_root = True
            del args[i:i+1]
            continue
        if a in ('--debug-frames',):
            global DEBUG_FRAMES
            DEBUG_FRAMES = True
            del args[i:i+1]
            continue
        if a in ('--trs-scale',):
            if i + 1 >= len(args):
                raise SystemExit('Missing value after --trs-scale')
            trs_scale = float(args[i + 1])
            del args[i:i+2]
            continue
        if a.startswith('--trs-scale='):
            trs_scale = float(a.split('=', 1)[1])
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
        if a in ('--center-geom',):
            center_geom = True
            del args[i:i+1]
            continue

        if a in ('--bake-origin',):

            bake_origin = True

            del args[i:i+1]

            continue


        print(f'[WARN] Ignoring unknown option: {a}')
        del args[i:i+1]
        continue

    if not args:
        print("No .vfx files provided.")
        return 2

    for p in args:
        if os.path.isfile(p):
            try:
                convert_file(p, scale=scale, trs_scale=trs_scale, center_root=center_root, center_geom=center_geom, bake_origin=bake_origin)
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






