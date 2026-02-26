import json, os, struct, sys, shutil

SEC_MESH = 0x4F584653  # 'SFXO' (mesh section id used in your dump)
SEC_DMMY = 0x594D4D44  # 'DMMY'

class Bin:
    def __init__(self, data: bytes):
        self.d=data; self.o=0
    def tell(self): return self.o
    def seek(self,o): self.o=o
    def read(self,n):
        b=self.d[self.o:self.o+n]
        if len(b)!=n: raise EOFError
        self.o+=n; return b
    def s8(self):  return struct.unpack("<b", self.read(1))[0]
    def u8(self):  return self.read(1)[0]
    def s16(self): return struct.unpack("<h", self.read(2))[0]
    def s32(self): return struct.unpack("<i", self.read(4))[0]
    def f32(self): return struct.unpack("<f", self.read(4))[0]
    def cstr(self):
        start=self.o
        while self.o < len(self.d) and self.d[self.o]!=0: self.o+=1
        if self.o>=len(self.d): raise EOFError("unterminated cstr")
        s=self.d[start:self.o].decode("ascii", errors="replace")
        self.o+=1
        return s

def read_vec3_rf(b): return (b.f32(), b.f32(), b.f32())

# IMPORTANT: keep this mapping aligned with your current exporter output (the one that finally looks right).
# Your successful dump + current good-looking export are consistent with (-x, z, -y).
def rf_to_blender(v):
    x,y,z=v
    return (-x, z, -y)

def read_vec3_to_blender(b): return rf_to_blender(read_vec3_rf(b))
def read_quat_xyzw(b): return (b.f32(), b.f32(), b.f32(), b.f32())  # x,y,z,w

def parse_dummies(vfx_path: str):
    data=open(vfx_path,"rb").read()
    b=Bin(data)
    if b.read(4) != b"VSFX":
        raise SystemExit("Not VSFX")

    version=b.s32()
    if version >= 0x30008:
        _flags=b.s32()

    # Scan for the first plausible section (mesh or dummy), like your dump did
    known={SEC_MESH, SEC_DMMY}
    found=None
    for ofs in range(b.tell(), len(data)-8, 4):
        t = struct.unpack("<i", data[ofs:ofs+4])[0]
        l = struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t in known and 8 <= l <= (len(data)-ofs):
            found=ofs
            break
    if found is None:
        return []

    b.seek(found)
    out=[]

    while b.tell() < len(data)-8:
        sec_type=b.s32()
        sec_len=b.s32()
        if sec_len < 8: break
        section_start=b.tell()
        section_end=section_start + (sec_len - 4)
        if section_end > len(data): break

        if sec_type == SEC_DMMY:
            name=b.cstr()
            parent=b.cstr()
            _save_parent=b.u8()
            pos=read_vec3_to_blender(b)
            quat=read_quat_xyzw(b)
            num_frames=b.s32()

            first=None
            if num_frames > 0:
                fpos=read_vec3_to_blender(b)
                fquat=read_quat_xyzw(b)
                first=(fpos,fquat)
                # skip remaining frames
                remaining=num_frames-1
                if remaining>0:
                    b.read(remaining*(12+16))

            out.append((name,parent,pos,quat,num_frames,first))

        b.seek(section_end)

    return out

def find_node(nodes, name):
    nlow=(name or "").strip().lower()
    for i,n in enumerate(nodes):
        if (n.get("name","") or "").strip().lower() == nlow:
            return i
    return None

def ensure_child(nodes, parent_i, child_i):
    p=nodes[parent_i]
    ch=p.get("children")
    if ch is None:
        p["children"]=[child_i]
    else:
        if child_i not in ch: ch.append(child_i)

def main(vfx_path, gltf_path):
    dummies=parse_dummies(vfx_path)
    print(f"[INFO] dummies found: {len(dummies)}")
    for d in dummies:
        print("       ", d[0], "parent=", d[1], "pos=", d[2], "frames=", d[4])

    if not dummies:
        raise SystemExit("[FAIL] No DMMY nodes found (but your earlier dump shows 1).")

    g=json.load(open(gltf_path,"r",encoding="utf-8"))
    nodes=g.get("nodes",[])
    if not nodes: raise SystemExit("[FAIL] no nodes in glTF")

    root_i=int(g.get("scenes",[{}])[0].get("nodes",[0])[0])

    added=0
    for (name,parent,pos,quat,nf,first) in dummies:
        if find_node(nodes, name) is not None:
            continue

        ni=len(nodes)
        nodes.append({
            "name": name,
            "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation":    [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "extras": {
                "vfx_type": "DMMY",
                "vfx_parent": parent,
                "vfx_frames": int(nf),
                "vfx_first_frame_pos": list(first[0]) if first else None
            }
        })

        pi=find_node(nodes, parent) if parent else None
        if pi is None: pi=root_i
        ensure_child(nodes, pi, ni)
        added += 1

    g["nodes"]=nodes

    out_gltf=os.path.splitext(gltf_path)[0] + "_with_dummies.gltf"
    out_bin =os.path.splitext(gltf_path)[0] + "_with_dummies.bin"

    # Copy bin and update uri
    in_bin=os.path.splitext(gltf_path)[0] + ".bin"
    shutil.copyfile(in_bin, out_bin)
    g["buffers"][0]["uri"]=os.path.basename(out_bin)

    json.dump(g, open(out_gltf,"w",encoding="utf-8",newline=""), indent=2)

    print(f"[OK] Added {added} dummy node(s).")
    print("[OK] Wrote:", out_gltf)
    print("[OK] Wrote:", out_bin)

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
