import math, struct, sys

SEC_MESH = 0x4F584653  # 'SFXO'
S16_MAX = 32767.0

class Bin:
    def __init__(self, data: bytes): self.d=data; self.o=0
    def tell(self): return self.o
    def seek(self, o): self.o=o
    def read(self, n):
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

def dist(a,b):
    dx=a[0]-b[0]; dy=a[1]-b[1]; dz=a[2]-b[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def compute_radius(center, mult, s16blob, nverts, bound_center, norm: bool):
    r=0.0
    o=0
    for _ in range(nverts):
        sx,sy,sz = struct.unpack_from("<hhh", s16blob, o); o += 6
        if norm:
            px = center[0] + mult[0]*(sx/S16_MAX)
            py = center[1] + mult[1]*(sy/S16_MAX)
            pz = center[2] + mult[2]*(sz/S16_MAX)
        else:
            px = center[0] + mult[0]*float(sx)
            py = center[1] + mult[1]*float(sy)
            pz = center[2] + mult[2]*float(sz)
        r = max(r, dist((px,py,pz), bound_center))
    return r

def parse_mesh(body: bytes, version: int):
    b=Bin(body)
    name=b.cstr(); parent=b.cstr(); _=b.s8()
    nverts=b.s32()
    if version < 0x3000A:
        b.read(nverts*12)
    nfaces=b.s32()

    # skip faces
    for _ in range(nfaces):
        b.read(12)      # indices
        if version < 0x3000D:
            b.read(24)  # uvs old
        b.read(36)      # colors
        b.read(12)      # normal
        b.read(12)      # center
        b.read(4)       # radius
        b.read(4)       # mat
        b.read(4)       # smoothing
        b.read(12)      # face_vertex_indices

    if version >= 0x30009:
        _fps=b.s32()

    if version >= 0x40004:
        _start=b.f32(); _end=b.f32()
        num_frames=b.s32()
    else:
        b.s32(); b.s32()
        num_frames=1

    nmat=b.s32()
    if version >= 0x40000:
        b.read(nmat*4)

    bound_center=vec3(b)
    bound_radius=b.f32()

    flags=b.u32()
    morph = (flags & 0x00000004)!=0
    facing = (flags & 0x00000001)!=0
    facing_rod = (flags & 0x00000800)!=0
    dump_uvs = (flags & 0x00000100)!=0

    n_face_verts=b.s32()
    for _ in range(n_face_verts):
        b.read(4); b.read(4); b.read(4); b.read(4)
        nadj=b.s32()
        if nadj>0: b.read(nadj*4)

    is_keyframed=False
    if version >= 0x30009:
        is_keyframed = (b.u8()!=0)

    # read frame0 vertex block
    frame0_center=None; frame0_mult=None; frame0_s16=None
    for fi in range(num_frames):
        has_positions = (morph or fi==0)
        if has_positions:
            c=vec3(b); m=vec3(b)
            raw=b.read(nverts*6)
            if fi==0:
                frame0_center=c; frame0_mult=m; frame0_s16=raw

        if has_positions and (facing or facing_rod) and version >= 0x3000B:
            b.read(8)
        if has_positions and facing_rod and fi==0 and version >= 0x40001:
            b.read(12)
        if (dump_uvs or fi==0) and version >= 0x3000D:
            b.read(nfaces*3*8)

        if (not morph) and ((not is_keyframed) or (version < 0x3000E and fi==0)):
            b.read(12); b.read(16); b.read(12)

        if version < 0x30009:
            b.read(1)
        if version < 0x40005:
            b.read(4)

    rn = compute_radius(frame0_center, frame0_mult, frame0_s16, nverts, bound_center, True)
    rr = compute_radius(frame0_center, frame0_mult, frame0_s16, nverts, bound_center, False)

    return name, parent, morph, flags, bound_radius, rn, rr

def main(vfx_path):
    data=open(vfx_path,"rb").read()
    b=Bin(data)
    if b.read(4)!=b"VSFX": raise SystemExit("Not VSFX")
    version=b.s32()
    if version >= 0x30008: b.s32()

    # scan forward for first mesh section
    found=None
    for ofs in range(b.tell(), len(data)-8, 4):
        t=struct.unpack("<i", data[ofs:ofs+4])[0]
        l=struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t==SEC_MESH and 8 <= l <= (len(data)-ofs):
            found=ofs; break
    if found is None: raise SystemExit("No mesh sections found")
    b.seek(found)

    print(f"[HDR] version=0x{version:08X}")
    while b.tell() < len(data)-8:
        st=b.s32(); sl=b.s32()
        if sl < 8: break
        body_start=b.tell()
        body_end=body_start + (sl-4)
        if body_end > len(data): break
        if st==SEC_MESH:
            name,parent,morph,flags,br,rn,rr = parse_mesh(data[body_start:body_end], version)
            en=abs(rn-br); er=abs(rr-br)
            best="norm" if en<=er else "raw"
            print(f"\n[MESH] {name} parent='{parent}' morph={morph} flags=0x{flags:08X}")
            print(f"  stored_radius={br:.6g}")
            print(f"  computed_norm={rn:.6g}  err={en:.6g}  norm/stored={(rn/br):.6g}")
            print(f"  computed_raw ={rr:.6g}  err={er:.6g}  raw/stored ={(rr/br):.6g}")
            print(f"  -> best match: {best}")
        b.seek(body_end)

if __name__=="__main__":
    main(sys.argv[1])
