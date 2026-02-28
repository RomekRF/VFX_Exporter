import json, os, struct, sys, shutil

SEC_MESH = 0x4F584653  # 'SFXO'

class Bin:
    def __init__(self, d: bytes): self.d=d; self.o=0
    def tell(self): return self.o
    def seek(self,o): self.o=o
    def read(self,n):
        b=self.d[self.o:self.o+n]
        if len(b)!=n: raise EOFError(f"need {n}, have {len(b)} at {self.o}")
        self.o+=n; return b
    def s8(self):  return struct.unpack("<b", self.read(1))[0]
    def u8(self):  return self.read(1)[0]
    def s16(self): return struct.unpack("<h", self.read(2))[0]
    def s32(self): return struct.unpack("<i", self.read(4))[0]
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def f32(self): return struct.unpack("<f", self.read(4))[0]
    def cstr(self):
        start=self.o
        while self.o < len(self.d) and self.d[self.o]!=0: self.o+=1
        if self.o>=len(self.d): raise EOFError("unterminated cstr")
        s=self.d[start:self.o].decode("ascii", errors="replace")
        self.o+=1
        return s

def vec3(b: Bin): return (b.f32(), b.f32(), b.f32())
def quat(b: Bin): return (b.f32(), b.f32(), b.f32(), b.f32())

# Must match your exporter mapping (positions): (-x, z, -y)
def rf_to_blender(v):
    x,y,z=v
    return (-x, z, -y)

# For SCALE: no sign, but same axis swap (x, y, z) -> (|x|, |z|, |y|)
def rf_scale_to_blender(s):
    sx,sy,sz=s
    return (abs(sx), abs(sz), abs(sy))

def scan_first_mesh(data: bytes, start: int):
    for ofs in range(start, len(data)-8, 4):
        t = struct.unpack("<i", data[ofs:ofs+4])[0]
        l = struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t == SEC_MESH and 8 <= l <= (len(data)-ofs):
            return ofs
    return None

def skip_faces(bb: Bin, version: int, nfaces: int):
    for _ in range(nfaces):
        bb.read(12)  # indices s32*3
        if version < 0x3000D:
            bb.read(24)  # old uvs
        bb.read(36)  # colors
        bb.read(12)  # normal
        bb.read(12)  # center
        bb.read(4)   # radius
        bb.read(4)   # mat
        bb.read(4)   # smoothing
        bb.read(12)  # face_vertex_indices

def skip_face_vertices(bb: Bin, nfv: int):
    for _ in range(nfv):
        bb.read(4); bb.read(4); bb.read(4); bb.read(4)
        nadj = bb.s32()
        if nadj > 0:
            bb.read(nadj * 4)

def skip_vec3_keyframes(bb: Bin, n: int):
    # time s32, value vec3, in_tangent vec3, out_tangent vec3
    bb.read(n * (4 + 12 + 12 + 12))

def skip_quat_keyframes(bb: Bin, n: int):
    # time s32, value quat, tension/cont/bias/easein/easeout (5 floats)
    bb.read(n * (4 + 16 + 4*5))

def extract_flagpole_scale(vfx_path: str):
    data = open(vfx_path, "rb").read()
    b = Bin(data)
    if b.read(4) != b"VSFX": raise SystemExit("Not VSFX")
    version = b.s32()
    if version >= 0x30008:
        _file_flags = b.s32()

    found = scan_first_mesh(data, b.tell())
    if found is None: raise SystemExit("No mesh sections found")
    b.seek(found)

    while b.tell() < len(data)-8:
        st = b.s32()
        sl = b.s32()
        if sl < 8: break
        body_start = b.tell()
        body_end = body_start + (sl - 4)
        if body_end > len(data): break

        if st != SEC_MESH:
            b.seek(body_end); continue

        bb = Bin(data[body_start:body_end])
        name = bb.cstr()
        parent = bb.cstr()
        bb.s8()  # save_parent

        nverts = bb.s32()
        if version < 0x3000A:
            bb.read(nverts * 12)
        nfaces = bb.s32()
        skip_faces(bb, version, nfaces)

        if version >= 0x30009:
            _fps = bb.s32()

        if version >= 0x40004:
            _start_t = bb.f32(); _end_t = bb.f32(); num_frames = bb.s32()
        else:
            _sf = bb.s32(); _ef = bb.s32()
            num_frames = (_ef - _sf + 1) if _ef >= _sf else 1

        nmat = bb.s32()
        if version >= 0x40000:
            bb.read(nmat * 4)

        _bound_center = vec3(bb)
        _bound_radius = bb.f32()

        flags = bb.u32()
        morph = (flags & 0x00000004) != 0
        dump_uvs = (flags & 0x00000100) != 0
        facing = (flags & 0x00000001) != 0
        facing_rod = (flags & 0x00000800) != 0

        nfv = bb.s32()
        skip_face_vertices(bb, nfv)

        is_keyframed = None
        if version >= 0x30009:
            is_keyframed = (bb.u8() != 0)

        if name.lower() != "flagpole":
            b.seek(body_end); continue

        # Read frames, capturing frame0 TRS scale if present
        frame0_scale_rf = None
        for fi in range(num_frames):
            has_positions = (morph or fi == 0)
            if has_positions:
                _c = vec3(bb); _m = vec3(bb)
                bb.read(nverts * 6)

            # skip facing extras if any (not expected for this asset, but safe)
            if has_positions and (facing or facing_rod) and version >= 0x3000B:
                bb.read(8)
            if has_positions and facing_rod and fi == 0 and version >= 0x40001:
                bb.read(12)

            if (dump_uvs or fi == 0) and version >= 0x3000D:
                bb.read(nfaces * 3 * 8)

            if (not morph) and (is_keyframed is False):
                _t = vec3(bb)
                _r = quat(bb)
                srf = vec3(bb)
                if fi == 0:
                    frame0_scale_rf = srf

            if version < 0x30009:
                bb.read(1)
            if version < 0x40005:
                bb.read(4)  # opacity

        # Keyframed pivot + keys
        pivot_pos_bl = (0.0, 0.0, 0.0)
        pivot_scale_rf = None
        scale_key0_rf = None

        if is_keyframed and version >= 0x3000A:
            ppos_rf = vec3(bb)
            _prot = quat(bb)
            pivot_scale_rf = vec3(bb)
            pivot_pos_bl = rf_to_blender(ppos_rf)

        if is_keyframed:
            nt = bb.s32(); skip_vec3_keyframes(bb, nt)
            nr = bb.s32(); skip_quat_keyframes(bb, nr)
            ns = bb.s32()
            if ns > 0:
                _time = bb.s32()
                scale_key0_rf = vec3(bb)
                bb.read(12 + 12)
                if ns > 1:
                    bb.read((ns - 1) * (4 + 12 + 12 + 12))

        # Decide scale source
        if is_keyframed:
            # multiply pivot_scale * first scale key if present
            sx = sy = sz = 1.0
            if pivot_scale_rf is not None:
                sx *= pivot_scale_rf[0]; sy *= pivot_scale_rf[1]; sz *= pivot_scale_rf[2]
            if scale_key0_rf is not None:
                sx *= scale_key0_rf[0]; sy *= scale_key0_rf[1]; sz *= scale_key0_rf[2]
            scale_bl = rf_scale_to_blender((sx, sy, sz))
            return {
                "name": name, "parent": parent,
                "is_keyframed": True,
                "pivot_pos_bl": pivot_pos_bl,
                "scale_bl": scale_bl,
                "pivot_scale_rf": pivot_scale_rf,
                "scale_key0_rf": scale_key0_rf,
                "frame0_scale_rf": frame0_scale_rf
            }
        else:
            if frame0_scale_rf is None:
                return {
                    "name": name, "parent": parent,
                    "is_keyframed": False,
                    "pivot_pos_bl": (0.0,0.0,0.0),
                    "scale_bl": (1.0,1.0,1.0),
                    "frame0_scale_rf": None
                }
            scale_bl = rf_scale_to_blender(frame0_scale_rf)
            return {
                "name": name, "parent": parent,
                "is_keyframed": False,
                "pivot_pos_bl": (0.0,0.0,0.0),
                "scale_bl": scale_bl,
                "frame0_scale_rf": frame0_scale_rf
            }

    raise SystemExit("flagpole mesh not found")

def find_node(nodes, name):
    nlow = name.strip().lower()
    for i,n in enumerate(nodes):
        if (n.get("name","") or "").strip().lower() == nlow:
            return i
    return None

def scale_accessor_positions(g, blob, acc_i, pivot, sxyz):
    acc = g["accessors"][acc_i]
    if acc["componentType"] != 5126 or acc["type"] != "VEC3":
        return 0
    bv = g["bufferViews"][acc["bufferView"]]
    stride = int(bv.get("byteStride") or 12)
    base = int(bv.get("byteOffset") or 0) + int(acc.get("byteOffset") or 0)
    count = int(acc["count"])
    px,py,pz = pivot
    sx,sy,sz = sxyz
    for i in range(count):
        off = base + i*stride
        x,y,z = struct.unpack_from("<fff", blob, off)
        x = px + (x - px) * sx
        y = py + (y - py) * sy
        z = pz + (z - pz) * sz
        struct.pack_into("<fff", blob, off, x,y,z)
    return 1

def main(vfx_path, gltf_path):
    info = extract_flagpole_scale(vfx_path)
    print("[VFX] flagpole:", info)

    g = json.load(open(gltf_path, "r", encoding="utf-8"))
    nodes = g.get("nodes", [])
    pole_i = find_node(nodes, "flagpole")
    if pole_i is None: raise SystemExit("glTF node 'flagpole' not found")

    pole_node = nodes[pole_i]
    if "mesh" not in pole_node: raise SystemExit("flagpole node has no mesh")
    mesh_i = pole_node["mesh"]
    mesh = g["meshes"][mesh_i]

    bin_uri = g["buffers"][0]["uri"]
    bin_path = os.path.join(os.path.dirname(gltf_path), bin_uri)
    blob = bytearray(open(bin_path, "rb").read())

    pivot = tuple(map(float, info["pivot_pos_bl"]))
    sxyz = tuple(map(float, info["scale_bl"]))

    if sxyz == (1.0,1.0,1.0):
        print("[INFO] VFX scale is (1,1,1) -> no change made.")
        out_gltf = os.path.splitext(gltf_path)[0] + "_poleScaled.gltf"
        out_bin  = os.path.splitext(gltf_path)[0] + "_poleScaled.bin"
        shutil.copyfile(bin_path, out_bin)
        g["buffers"][0]["uri"] = os.path.basename(out_bin)
        json.dump(g, open(out_gltf, "w", encoding="utf-8", newline=""), indent=2)
        print("[OK] Wrote:", out_gltf)
        print("[OK] Wrote:", out_bin)
        return

    touched = 0
    for prim in mesh.get("primitives", []):
        attrs = prim.get("attributes", {})
        if "POSITION" in attrs:
            touched += scale_accessor_positions(g, blob, attrs["POSITION"], pivot, sxyz)
        for tgt in prim.get("targets", []) or []:
            if "POSITION" in tgt:
                touched += scale_accessor_positions(g, blob, tgt["POSITION"], pivot, sxyz)

    if touched == 0:
        raise SystemExit("No POSITION accessors touched on flagpole mesh")

    out_gltf = os.path.splitext(gltf_path)[0] + "_poleScaled.gltf"
    out_bin  = os.path.splitext(gltf_path)[0] + "_poleScaled.bin"
    open(out_bin, "wb").write(blob)
    g["buffers"][0]["uri"] = os.path.basename(out_bin)
    json.dump(g, open(out_gltf, "w", encoding="utf-8", newline=""), indent=2)
    print("[OK] Wrote:", out_gltf)
    print("[OK] Wrote:", out_bin)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
