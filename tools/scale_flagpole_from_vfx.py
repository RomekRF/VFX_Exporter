import json, os, struct, sys

SEC_MESH = 0x4F584653  # SFXO

class Bin:
    def __init__(self, d: bytes): self.d=d; self.o=0
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

def scan_first_mesh(data, start):
    for ofs in range(start, len(data)-8, 4):
        t=struct.unpack("<i", data[ofs:ofs+4])[0]
        l=struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t==SEC_MESH and 8 <= l <= (len(data)-ofs):
            return ofs
    return None

# VFX->Blender axis mapping in your pipeline: (-x, z, -y)
# For SCALE (no sign), that means component reorder: (sx, sz, sy)
def scale_rf_to_blender(s):
    sx,sy,sz = s
    return (abs(sx), abs(sz), abs(sy))

def extract_flagpole_geom_scale(vfx_path: str):
    data=open(vfx_path,"rb").read()
    b=Bin(data)
    if b.read(4)!=b"VSFX": raise SystemExit("Not VSFX")
    version=b.s32()
    if version >= 0x30008:
        b.s32()  # file flags

    found=scan_first_mesh(data, b.tell())
    if found is None: return None
    b.seek(found)

    while b.tell() < len(data)-8:
        st=b.s32(); sl=b.s32()
        if sl < 8: break
        body_start=b.tell()
        body_end=body_start + (sl-4)
        if body_end > len(data): break

        if st != SEC_MESH:
            b.seek(body_end); continue

        bb=Bin(data[body_start:body_end])
        name=bb.cstr()
        _parent=bb.cstr()
        bb.s8()

        nverts=bb.s32()
        if version < 0x3000A:
            bb.read(nverts*12)
        nfaces=bb.s32()

        # skip faces
        for _ in range(nfaces):
            bb.read(12)
            if version < 0x3000D: bb.read(24)
            bb.read(36)
            bb.read(12); bb.read(12)
            bb.read(4); bb.read(4); bb.read(4)
            bb.read(12)

        if version >= 0x30009:
            bb.s32()  # fps

        if version >= 0x40004:
            bb.f32(); bb.f32(); num_frames=bb.s32()
        else:
            bb.s32(); bb.s32(); num_frames=1

        nmat=bb.s32()
        if version >= 0x40000:
            bb.read(nmat*4)

        vec3(bb)  # bound center
        bb.f32()  # bound radius

        flags=bb.u32()
        morph = (flags & 0x00000004)!=0

        # face-vertex table
        nfv=bb.s32()
        for _ in range(nfv):
            bb.read(4); bb.read(4); bb.read(4); bb.read(4)
            nadj=bb.s32()
            if nadj>0: bb.read(nadj*4)

        is_keyframed=None
        if version >= 0x30009:
            is_keyframed = (bb.u8()!=0)

        if name.lower() != "flagpole":
            b.seek(body_end)
            continue

        # frame0 positions (always present at fi==0) -> skip
        vec3(bb); vec3(bb); bb.read(nverts*6)

        # Option A: non-morph and NOT keyframed -> frame contains TRS scale
        if (not morph) and (is_keyframed is False):
            vec3(bb)       # translation
            quat(bb)       # rotation
            s = vec3(bb)   # scale (RF axis order)
            return ("frame0_trs_scale", scale_rf_to_blender(s))

        # Option B: keyframed -> pivot_scale + scale keys exist
        if is_keyframed:
            pivot_scale=None
            scale_key0=None

            if version >= 0x3000A:
                vec3(bb)      # pivot translation
                quat(bb)      # pivot rotation
                pivot_scale = vec3(bb)  # pivot scale (RF axis order)

            # t keys
            n=bb.s32()
            if n>0:
                bb.read(4+12+12+12)
                if n>1: bb.read((n-1)*(4+12+12+12))

            # r keys
            n=bb.s32()
            bb.read(n*(4+16+4*5))

            # s keys
            n=bb.s32()
            if n>0:
                bb.s32()           # time
                scale_key0 = vec3(bb)
                bb.read(12+12)
                if n>1: bb.read((n-1)*(4+12+12+12))

            # combine if present
            sx=sy=sz=1.0
            if pivot_scale:
                ps = scale_rf_to_blender(pivot_scale)
                sx*=ps[0]; sy*=ps[1]; sz*=ps[2]
            if scale_key0:
                sk = scale_rf_to_blender(scale_key0)
                sx*=sk[0]; sy*=sk[1]; sz*=sk[2]

            return ("keyframed_pivot_and_skeys", (sx,sy,sz))

        return None

    return None

def find_node(nodes, name):
    nlow=name.strip().lower()
    for i,n in enumerate(nodes):
        if (n.get("name","") or "").strip().lower()==nlow:
            return i
    return None

def accessor_info(g, acc_i):
    acc=g["accessors"][acc_i]
    bv=g["bufferViews"][acc["bufferView"]]
    byte_offset=bv.get("byteOffset",0)+acc.get("byteOffset",0)
    count=acc["count"]
    comp=acc["componentType"]
    typ=acc["type"]
    byte_length=bv["byteLength"]
    return acc, bv, byte_offset, count, comp, typ, byte_length

def main(vfx_path, gltf_path):
    kind_scale = extract_flagpole_geom_scale(vfx_path)
    if not kind_scale:
        print("[INFO] No scale data found in VFX for flagpole (TRS scale or keyframed scale). Leaving as-is.")
        return 0

    kind, s = kind_scale
    sx,sy,sz = map(float, s)
    print(f"[INFO] VFX flagpole scale source={kind} scale_blender=({sx:.6g},{sy:.6g},{sz:.6g})")

    g=json.load(open(gltf_path,"r",encoding="utf-8"))
    nodes=g.get("nodes",[])
    pole_i=find_node(nodes,"flagpole")
    if pole_i is None:
        raise SystemExit("Could not find node 'flagpole' in glTF")
    pole_node=nodes[pole_i]
    if "mesh" not in pole_node:
        raise SystemExit("flagpole node has no mesh")

    pole_mesh_i=pole_node["mesh"]
    pole_mesh=g["meshes"][pole_mesh_i]

    bin_uri=g["buffers"][0]["uri"]
    bin_path=os.path.join(os.path.dirname(gltf_path), bin_uri)
    blob=bytearray(open(bin_path,"rb").read())

    def scale_positions(acc_i):
        acc,bv,off,count,comp,typ,_ = accessor_info(g, acc_i)
        if comp != 5126 or typ != "VEC3":
            return
        # float32 xyz
        for vi in range(count):
            base = off + vi*12
            x,y,z = struct.unpack_from("<fff", blob, base)
            x*=sx; y*=sy; z*=sz
            struct.pack_into("<fff", blob, base, x,y,z)

    # Scale POSITION for all primitives in the pole mesh
    scaled = 0
    for prim in pole_mesh.get("primitives",[]):
        attrs=prim.get("attributes",{})
        if "POSITION" in attrs:
            scale_positions(attrs["POSITION"])
            scaled += 1
        # (no morph targets expected for pole, but keep safe)
        for tgt in prim.get("targets",[]) or []:
            if "POSITION" in tgt:
                scale_positions(tgt["POSITION"])

    if scaled == 0:
        print("[WARN] No POSITION accessors found on pole mesh; nothing scaled.")
        return 0

    out_gltf=os.path.splitext(gltf_path)[0] + "_poleGeomScaled.gltf"
    out_bin =os.path.splitext(gltf_path)[0] + "_poleGeomScaled.bin"

    # Write new bin and point glTF to it
    open(out_bin,"wb").write(blob)
    g["buffers"][0]["uri"]=os.path.basename(out_bin)
    json.dump(g, open(out_gltf,"w",encoding="utf-8",newline=""), indent=2)

    print("[OK] Wrote:", out_gltf)
    print("[OK] Wrote:", out_bin)
    return 0

if __name__=="__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
