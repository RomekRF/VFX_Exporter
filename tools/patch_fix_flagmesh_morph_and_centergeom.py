import re, sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

def die(msg: str):
    raise SystemExit(msg)

# --------------------------
# Patch 1) FIX morph frame decode + frames_pos capture
# Current bug:
#   - only decompresses when frame_idx == 0
#   - frames_pos.append(verts) appends the same list repeatedly
# --------------------------
pat_frames = r"(?ms)^(?P<ind>[ \t]*)for frame_idx in range\(num_frames\):.*?^(?P=ind)# UV block:"
m = re.search(pat_frames, s)
if not m:
    die("Patch failed: could not locate the frame loop block (for frame_idx in range(num_frames) ... # UV block:).")

ind = m.group("ind")
blk = [
    ind + "for frame_idx in range(num_frames):",
    ind + "    # For morph meshes, every frame stores a full compressed position set.",
    ind + "    # For non-morph meshes, only frame 0 stores positions.",
    ind + "    if morph or frame_idx == 0:",
    ind + "        center_rf = read_vec3_rf(b)",
    ind + "        mult_rf = read_vec3_rf(b)",
    ind + "",
    ind + "        frame_unsplit: List[Tuple[float, float, float]] = []",
    ind + "        for _ in range(num_vertices):",
    ind + "            rx = b.s16(); ry = b.s16(); rz = b.s16()",
    ind + "            nx = rx / S16_MAX",
    ind + "            ny = ry / S16_MAX",
    ind + "            nz = rz / S16_MAX",
    ind + "            vx = center_rf[0] + mult_rf[0] * nx",
    ind + "            vy = center_rf[1] + mult_rf[1] * ny",
    ind + "            vz = center_rf[2] + mult_rf[2] * nz",
    ind + "            frame_unsplit.append(rf_to_blender((vx, vy, vz)))",
    ind + "",
    ind + "        if frame_idx == 0:",
    ind + "            verts = frame_unsplit",
    ind + "            base_verts = frame_unsplit",
    ind + "",
    ind + "        if frames_pos is not None:",
    ind + "            frames_pos.append(frame_unsplit)",
    ind + "",
    ind + "        # billboard widths/heights can appear for facing/facing_rod in newer versions",
    ind + "        if (facing or facing_rod) and version >= 0x3000B:",
    ind + "            _ = b.f32(); _ = b.f32()",
    ind + "        if facing_rod and frame_idx == 0 and version >= 0x40001:",
    ind + "            _ = read_vec3_to_blender(b)  # up_vector",
    ind + "",
    ind + "    # Always keep verts pointing at the base frame for topology/UV building",
    ind + "    if base_verts is not None:",
    ind + "        verts = base_verts",
    ind + "",
    ind + "# UV block:",
]
s = s[:m.start()] + "\n".join(blk) + s[m.end():]

# --------------------------
# Patch 2) FIX center-geom to recenter ALL meshes together (global center),
# not just POSITION accessor 0.
# --------------------------
pat_centergeom = r"(?ms)^(?P<ind>[ \t]*)if center_geom:\s*\n(?P<body>.*?)(?=^(?P=ind)\S)"
m2 = re.search(pat_centergeom, s)
if not m2:
    die("Patch failed: could not locate the 'if center_geom:' block in _gltf_pack_scene().")

ind = m2.group("ind")
blk2 = [
    ind + "if center_geom:",
    ind + "    # Global recenter: shift ALL POSITION accessors used by primitives by ONE shared center.",
    ind + "    import struct",
    ind + "    try:",
    ind + "        accessors = gltf.get('accessors', [])",
    ind + "        bvs = gltf.get('bufferViews', [])",
    ind + "        used = set()",
    ind + "        for mm in gltf.get('meshes', []):",
    ind + "            for prim in mm.get('primitives', []):",
    ind + "                attrs = prim.get('attributes', {})",
    ind + "                if isinstance(attrs, dict) and 'POSITION' in attrs:",
    ind + "                    try:",
    ind + "                        used.add(int(attrs['POSITION']))",
    ind + "                    except Exception:",
    ind + "                        pass",
    ind + "",
    ind + "        if used:",
    ind + "            gmin = [1e30, 1e30, 1e30]",
    ind + "            gmax = [-1e30, -1e30, -1e30]",
    ind + "",
    ind + "            # Pass 1: global bounds",
    ind + "            for ai in sorted(used):",
    ind + "                acc = accessors[ai]",
    ind + "                if acc.get('type') != 'VEC3' or acc.get('componentType') != 5126:",
    ind + "                    continue",
    ind + "                bv = bvs[acc['bufferView']]",
    ind + "                base = int(bv.get('byteOffset', 0)) + int(acc.get('byteOffset', 0))",
    ind + "                stride = int(bv.get('byteStride', 12) or 12)",
    ind + "                cnt = int(acc['count'])",
    ind + "                for ii in range(cnt):",
    ind + "                    o = base + ii * stride",
    ind + "                    x, y, z = struct.unpack_from('<fff', bin_blob, o)",
    ind + "                    if x < gmin[0]: gmin[0] = x",
    ind + "                    if y < gmin[1]: gmin[1] = y",
    ind + "                    if z < gmin[2]: gmin[2] = z",
    ind + "                    if x > gmax[0]: gmax[0] = x",
    ind + "                    if y > gmax[1]: gmax[1] = y",
    ind + "                    if z > gmax[2]: gmax[2] = z",
    ind + "",
    ind + "            cx = (gmin[0] + gmax[0]) * 0.5",
    ind + "            cy = (gmin[1] + gmax[1]) * 0.5",
    ind + "            cz = (gmin[2] + gmax[2]) * 0.5",
    ind + "",
    ind + "            # Pass 2: subtract center from every used POSITION accessor",
    ind + "            for ai in sorted(used):",
    ind + "                acc = accessors[ai]",
    ind + "                if acc.get('type') != 'VEC3' or acc.get('componentType') != 5126:",
    ind + "                    continue",
    ind + "                bv = bvs[acc['bufferView']]",
    ind + "                base = int(bv.get('byteOffset', 0)) + int(acc.get('byteOffset', 0))",
    ind + "                stride = int(bv.get('byteStride', 12) or 12)",
    ind + "                cnt = int(acc['count'])",
    ind + "                for ii in range(cnt):",
    ind + "                    o = base + ii * stride",
    ind + "                    x, y, z = struct.unpack_from('<fff', bin_blob, o)",
    ind + "                    struct.pack_into('<fff', bin_blob, o, x - cx, y - cy, z - cz)",
    ind + "                if 'min' in acc and 'max' in acc and acc.get('type') == 'VEC3':",
    ind + "                    acc['min'] = [float(acc['min'][0]) - cx, float(acc['min'][1]) - cy, float(acc['min'][2]) - cz]",
    ind + "                    acc['max'] = [float(acc['max'][0]) - cx, float(acc['max'][1]) - cy, float(acc['max'][2]) - cz]",
    ind + "",
    ind + "            print(f\"[INFO] center-geom: center=({cx:.6g},{cy:.6g},{cz:.6g}) accessors={len(used)}\")",
    ind + "    except Exception as e:",
    ind + "        print('[WARN] center-geom failed:', e)",
    ind + "",
]
s = s[:m2.start()] + "\n".join(blk2) + s[m2.end():]

p.write_text(s, encoding="utf-8")
print("[OK] patched:", str(p))
