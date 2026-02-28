import json, os, struct, sys, shutil

SEC_MESH = 0x4F584653  # 'SFXO'
SEC_DMMY = 0x594D4D44  # 'DMMY'

class Bin:
    def __init__(self, data: bytes): self.d=data; self.o=0
    def tell(self): return self.o
    def seek(self,o): self.o=o
    def read(self,n):
        b=self.d[self.o:self.o+n]
        if len(b)!=n: raise EOFError
        self.o+=n; return b
    def s32(self): return struct.unpack("<i", self.read(4))[0]
    def u8(self): return self.read(1)[0]
    def f32(self): return struct.unpack("<f", self.read(4))[0]
    def cstr(self):
        start=self.o
        while self.o < len(self.d) and self.d[self.o]!=0: self.o+=1
        if self.o>=len(self.d): raise EOFError("unterminated cstr")
        s=self.d[start:self.o].decode("ascii", errors="replace")
        self.o+=1
        return s

def read_vec3_rf(b): return (b.f32(), b.f32(), b.f32())
def read_quat(b):    return (b.f32(), b.f32(), b.f32(), b.f32())  # x,y,z,w

# Must match your exporter axis mapping: (-x, z, -y)
def rf_to_blender(v):
    x,y,z=v
    return (-x, z, -y)

def read_vec3_to_blender(b): return rf_to_blender(read_vec3_rf(b))

def scan_first_section(data, start):
    known={SEC_MESH, SEC_DMMY}
    for ofs in range(start, len(data)-8, 4):
        t=struct.unpack("<i", data[ofs:ofs+4])[0]
        l=struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t in known and 8 <= l <= (len(data)-ofs):
            return ofs
    return None

def parse_dummies(vfx_path: str):
    data=open(vfx_path,"rb").read()
    b=Bin(data)
    if b.read(4) != b"VSFX": raise SystemExit("Not VSFX")
    version=b.s32()
    if version >= 0x30008: b.s32()

    found = scan_first_section(data, b.tell())
    if found is None: return []
    b.seek(found)

    out=[]
    while b.tell() < len(data)-8:
        st=b.s32(); sl=b.s32()
        if sl < 8: break
        body_start=b.tell()
        body_end=body_start + (sl-4)
        if body_end > len(data): break

        if st == SEC_DMMY:
            name=b.cstr()
            parent=b.cstr()
            _save=b.u8()
            pos=read_vec3_to_blender(b)
            q=read_quat(b)
            nf=b.s32()
            # skip frames
            if nf > 0:
                b.read(nf*(12+16))
            out.append((name,parent,pos,q,nf))

        b.seek(body_end)
    return out

def find_node(nodes, name):
    nlow=(name or "").strip().lower()
    for i,n in enumerate(nodes):
        if (n.get("name","") or "").strip().lower()==nlow:
            return i
    return None

def ensure_child(nodes, parent_i, child_i):
    p=nodes[parent_i]
    ch=p.get("children")
    if ch is None: p["children"]=[child_i]
    elif child_i not in ch: ch.append(child_i)

def main(vfx_path, gltf_path):
    d=parse_dummies(vfx_path)
    print(f"[INFO] dummies found: {len(d)}")
    for x in d:
        print("       ", x[0], "parent=", x[1], "pos=", x[2], "frames=", x[4])

    if not d:
        print("[INFO] No DMMY nodes found; nothing to add.")
        return

    g=json.load(open(gltf_path,"r",encoding="utf-8"))
    nodes=g.get("nodes",[])
    if not nodes: raise SystemExit("No nodes in glTF")

    root_i=int(g.get("scenes",[{}])[0].get("nodes",[0])[0])

    added=0
    for (name,parent,pos,q,nf) in d:
        if find_node(nodes,name) is not None:
            continue
        ni=len(nodes)
        nodes.append({
            "name": name,
            "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            "extras": {"vfx_type":"DMMY","vfx_parent":parent,"vfx_frames":int(nf)}
        })
        pi=find_node(nodes,parent) if parent else None
        if pi is None: pi=root_i
        ensure_child(nodes, pi, ni)
        added += 1

    g["nodes"]=nodes

    out_gltf=os.path.splitext(gltf_path)[0] + "_with_dummies.gltf"
    out_bin =os.path.splitext(gltf_path)[0] + "_with_dummies.bin"
    in_bin=os.path.splitext(gltf_path)[0] + ".bin"

    shutil.copyfile(in_bin, out_bin)
    g["buffers"][0]["uri"]=os.path.basename(out_bin)
    json.dump(g, open(out_gltf,"w",encoding="utf-8",newline=""), indent=2)

    print(f"[OK] Added {added} dummy node(s).")
    print("[OK] Wrote:", out_gltf)
    print("[OK] Wrote:", out_bin)

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
