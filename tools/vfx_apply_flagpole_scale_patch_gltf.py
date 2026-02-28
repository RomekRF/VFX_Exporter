import json, struct, sys, os

SEC_MESH = 0x4F584653  # 'SFXO'
S16_MAX = 32767.0

class Bin:
    def __init__(self, data: bytes):
        self.d=data; self.o=0
    def tell(self): return self.o
    def seek(self,o): self.o=o
    def read(self,n):
        b=self.d[self.o:self.o+n]
        if len(b)!=n: raise EOFError
        self.o+=n; return b
    def s8(self): return struct.unpack("<b", self.read(1))[0]
    def u8(self): return self.read(1)[0]
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

def vec3(b): return (b.f32(), b.f32(), b.f32())
def quat(b): return (b.f32(), b.f32(), b.f32(), b.f32())

def scan_first_section(data: bytes, start: int):
    known={SEC_MESH}
    for ofs in range(start, len(data)-8, 4):
        t=struct.unpack("<i", data[ofs:ofs+4])[0]
        l=struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t in known and 8 <= l <= (len(data)-ofs):
            return ofs
    return None

def extract_flagpole_scale(vfx_path: str):
    data=open(vfx_path,"rb").read()
    b=Bin(data)
    if b.read(4)!=b"VSFX": raise SystemExit("Not VSFX")
    version=b.s32()
    if version >= 0x30008:
        b.s32()  # file flags

    # Skip the variable header by scanning for first mesh section (same strategy as your working dump/diag)
    found=scan_first_section(data, b.tell())
    if found is None: raise SystemExit("No mesh sections found")
    b.seek(found)

    pole_scale = None
    pole_is_keyframed = None

    while b.tell() < len(data)-8:
        sec_type=b.s32()
        sec_len=b.s32()
        if sec_len < 8: break
        body_start=b.tell()
        body_end=body_start + (sec_len-4)
        if body_end > len(data): break
        if sec_type != SEC_MESH:
            b.seek(body_end); continue

        bb=Bin(data[body_start:body_end])
        name=bb.cstr()
        _parent=bb.cstr()
        _save=bb.s8()
        nverts=bb.s32()
        if version < 0x3000A:
            bb.read(nverts*12)
        nfaces=bb.s32()

        # skip faces
        for _ in range(nfaces):
            bb.read(12)
            if version < 0x3000D:
                bb.read(24)
            bb.read(36)
            bb.read(12)
            bb.read(12)
            bb.read(4)
            bb.read(4)
            bb.read(4)
            bb.read(12)

        if version >= 0x30009:
            _fps=bb.s32()

        if version >= 0x40004:
            _start=bb.f32(); _end=bb.f32(); num_frames=bb.s32()
        else:
            bb.s32(); bb.s32(); num_frames=1

        nmat=bb.s32()
        if version >= 0x40000:
            bb.read(nmat*4)

        _bound_center=vec3(bb)
        _bound_radius=bb.f32()

        flags=bb.u32()
        morph = (flags & 0x00000004)!=0

        # face_vertices table
        n_face_verts=bb.s32()
        for _ in range(n_face_verts):
            bb.read(4); bb.read(4); bb.read(4); bb.read(4)
            nadj=bb.s32()
            if nadj>0: bb.read(nadj*4)

        is_keyframed = False
        if version >= 0x30009:
            is_keyframed = (bb.u8()!=0)

        if name.lower() != "flagpole":
            # skip frames quickly (not needed)
            # we don't fully skip every optional block—just jump to end of section
            continue

        pole_is_keyframed = is_keyframed

        # Read frame0 and capture scale if present in the per-frame TRS block
        # Per spec: for non-morph, if NOT keyframed, frames include translation/rotation/scale
        # (Your exporter guide says scale animation likely removed, but constant scale may still exist.)
        if num_frames <= 0:
            break

        # frame 0 positions block (always present for fi==0)
        _c=vec3(bb); _m=vec3(bb)
        bb.read(nverts*6)

        # optional extras we can conservatively skip based on flags are irrelevant here.
        # Now: read TRS if present
        if (not morph) and (not is_keyframed):
            _t=vec3(bb)
            _r=quat(bb)
            s=vec3(bb)
            pole_scale = s
        else:
            pole_scale = None

        break

    return pole_scale, pole_is_keyframed

def find_node(nodes, name):
    nlow=name.strip().lower()
    for i,n in enumerate(nodes):
        if (n.get("name","") or "").strip().lower()==nlow:
            return i
    return None

def remove_child(nodes, parent_i, child_i):
    ch = nodes[parent_i].get("children")
    if not ch: return
    nodes[parent_i]["children"] = [c for c in ch if c != child_i]

def main(vfx_path, gltf_path):
    pole_scale, pole_is_keyframed = extract_flagpole_scale(vfx_path)
    print(f"[INFO] flagpole is_keyframed={pole_is_keyframed} frame0_scale={pole_scale}")

    g=json.load(open(gltf_path,"r",encoding="utf-8"))
    nodes=g.get("nodes",[])
    if not nodes: raise SystemExit("No nodes in glTF")

    root_i = int(g.get("scenes",[{}])[0].get("nodes",[0])[0])

    pole_i = find_node(nodes, "flagpole")
    flag_i = find_node(nodes, "FlagMesh")
    if pole_i is None or flag_i is None:
        raise SystemExit("Could not find flagpole and/or FlagMesh nodes by name in glTF")

    # Flatten: ensure FlagMesh is NOT a child of flagpole (so pole scaling doesn't scale the cloth)
    remove_child(nodes, pole_i, flag_i)
    # Ensure FlagMesh is under root
    if flag_i not in nodes[root_i].get("children", []):
        nodes[root_i].setdefault("children", []).append(flag_i)

    # Apply pole scale if VFX provides it; otherwise do nothing (data-first)
    if pole_scale is not None:
        nodes[pole_i]["scale"] = [float(pole_scale[0]), float(pole_scale[1]), float(pole_scale[2])]
        print(f"[OK] Applied node scale to flagpole: {nodes[pole_i]['scale']}")
    else:
        print("[WARN] No per-frame scale found for flagpole (not keyframed or scale not stored). No scale applied.")

    out = os.path.splitext(gltf_path)[0] + "_poleScaled.gltf"
    json.dump(g, open(out,"w",encoding="utf-8",newline=""), indent=2)
    print("[OK] Wrote:", out)
    print("[NOTE] Bin is unchanged; this glTF still references the same .bin as the input file.")

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
