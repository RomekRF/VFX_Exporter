import struct, sys

SEC_MESH = 0x4F584653  # SFXO
SEC_DMMY = 0x594D4D44  # DMMY

def tag(i): return struct.pack("<I", i).decode("ascii", errors="replace")

class Bin:
    def __init__(self, data: bytes): self.d=data; self.o=0
    def tell(self): return self.o
    def seek(self, o): self.o=o
    def read(self,n):
        b=self.d[self.o:self.o+n]
        if len(b)!=n: raise EOFError
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

def rf_to_blender(v):
    # stored.x=-max.x, stored.y=max.z, stored.z=-max.y  => (-x, z, -y)
    x,y,z=v
    return (-x, z, -y)

def read_vec3_rf(b): return (b.f32(), b.f32(), b.f32())
def read_vec3_to_blender(b): return rf_to_blender(read_vec3_rf(b))
def read_quat(b): return (b.f32(), b.f32(), b.f32(), b.f32())  # keep raw order

def main(path):
    data=open(path,"rb").read()
    b=Bin(data)
    if b.read(4) != b"VSFX": raise SystemExit("Not VSFX")
    version=b.s32()
    if version >= 0x30008: _flags=b.s32()
    end_frame=b.s32()
    num_meshes=b.s32(); _=b.s32(); num_dummies=b.s32()

    print(f"[HDR] version=0x{version:08X} end_frame={end_frame} num_meshes={num_meshes} num_dummies={num_dummies}")

    # skip rest of header (match your exporter behavior: it reads a lot of counts)
    # easiest: reuse your exporter assumption by seeking to first section:
    # We can detect first section by scanning for known tags safely.
    # But simpler: trust current file layout by continuing to read the remaining header ints like exporter does.
    # We'll do a minimal safe approach: scan forward until we see a plausible section header.
    # (type in known set, len reasonable)
    known = {SEC_MESH, SEC_DMMY}
    # scan in 4-byte steps
    found = None
    for ofs in range(b.tell(), len(data)-8, 4):
        t = struct.unpack("<i", data[ofs:ofs+4])[0]
        l = struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t in known and 8 <= l <= (len(data)-ofs):
            found = ofs
            break
    if found is None:
        raise SystemExit("[FAIL] couldn't find first section")
    b.seek(found)

    dummies=[]
    meshes=[]
    while b.tell() < len(data)-8:
        sec_type = b.s32()
        sec_len  = b.s32()
        if sec_len < 8: break
        section_start = b.tell()
        section_end = section_start + (sec_len - 4)

        if section_end > len(data): break

        if sec_type == SEC_MESH:
            name = b.cstr()
            parent = b.cstr()
            _save_parent = b.s8()
            meshes.append((name, parent))
            b.seek(section_end)

        elif sec_type == SEC_DMMY:
            name = b.cstr()
            parent = b.cstr()
            _save_parent = b.u8()
            pos = read_vec3_to_blender(b)
            q   = read_quat(b)
            num_frames = b.s32()
            first = None
            if num_frames > 0:
                fpos = read_vec3_to_blender(b)
                fq   = read_quat(b)
                first = (fpos, fq)
            dummies.append((name, parent, pos, q, num_frames, first))
            b.seek(section_end)

        else:
            b.seek(section_end)

    print("\n[MESH parent links]")
    for n,p in meshes:
        print(f"  mesh='{n}'  parent='{p}'")

    print("\n[DMMY nodes]")
    for (n,p,pos,q,nf,first) in dummies:
        print(f"  dummy='{n}' parent='{p}' pos={pos} quat={q} frames={nf}")
        if first:
            print(f"        first_frame_pos={first[0]} first_frame_quat={first[1]}")

    # heuristic: likely pivot
    pivot = None
    for (n,p,_,_,_,_) in dummies:
        if n.strip().lower() == "scene root":
            pivot = n
            break
    if pivot is None and dummies:
        pivot = dummies[0][0]
    print(f"\n[GUESS] pivot/root dummy likely: {pivot}")

if __name__=="__main__":
    main(sys.argv[1])
