import json, os, struct, sys, shutil

SEC_DMMY = 0x594D4D44  # 'DMMY'

class Bin:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0
    def tell(self): return self.o
    def seek(self, o): self.o = o
    def read(self,n):
        b = self.d[self.o:self.o+n]
        if len(b)!=n: raise EOFError
        self.o += n
        return b
    def s32(self): return struct.unpack("<i", self.read(4))[0]
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u8(self): return self.read(1)[0]
    def f32(self): return struct.unpack("<f", self.read(4))[0]
    def cstr(self):
        start = self.o
        while self.o < len(self.d) and self.d[self.o] != 0:
            self.o += 1
        if self.o >= len(self.d): raise EOFError("unterminated cstr")
        s = self.d[start:self.o].decode("ascii", errors="replace")
        self.o += 1
        return s

def read_vec3_rf(b): return (b.f32(), b.f32(), b.f32())

# IMPORTANT: this must match the coordinate mapping you’re currently using in your working export.
# Your dump matched this mapping (and the asset looks correct now).
def rf_to_blender(v):
    x,y,z=v
    return (-x, z, -y)

def read_vec3_to_blender(b): return rf_to_blender(read_vec3_rf(b))
def read_quat(b): return (b.f32(), b.f32(), b.f32(), b.f32())  # x,y,z,w

def parse_dummies(vfx_path: str):
    data = open(vfx_path, "rb").read()
    b = Bin(data)

    if b.read(4) != b"VSFX":
        raise SystemExit("Not VSFX")
    version = b.s32()
    if version >= 0x30008:
        _flags = b.s32()

    # Skip header-ish ints by scanning forward for first plausible section
    known = {SEC_DMMY}
    found = None
    for ofs in range(b.tell(), len(data)-8, 4):
        t = struct.unpack("<i", data[ofs:ofs+4])[0]
        l = struct.unpack("<i", data[ofs+4:ofs+8])[0]
        if t in known and 8 <= l <= (len(data)-ofs):
            found = ofs
            break
    if found is None:
        # no dummies (fine)
        return []

    b.seek(found)
    out = []

    while b.tell() < len(data)-8:
        sec_type = b.s32()
        sec_len  = b.s32()
        if sec_len < 8: break
        section_start = b.tell()
        section_end = section_start + (sec_len - 4)
        if section_end > len(data): break

        if sec_type == SEC_DMMY:
            name = b.cstr()
            parent = b.cstr()
            _save_parent = b.u8()
            pos = read_vec3_to_blender(b)
            quat = read_quat(b)
            num_frames = b.s32()
            # skip frames (we only need rest pose now)
            if num_frames > 0:
                # each frame is vec3 + quat
                b.read(num_frames * (12 + 16))
            out.append((name, parent, pos, quat, num_frames))
        b.seek(section_end)

    return out

def find_node_by_name(nodes, name):
    nlow = name.strip().lower()
    for i,n in enumerate(nodes):
        if (n.get("name","") or "").strip().lower() == nlow:
            return i
    return None

def ensure_child(nodes, parent_i, child_i):
    p = nodes[parent_i]
    ch = p.get("children")
    if ch is None:
        p["children"] = [child_i]
    else:
        if child_i not in ch:
            ch.append(child_i)

def main(vfx_path, gltf_path):
    dummies = parse_dummies(vfx_path)
    if not dummies:
        print("[INFO] No DMMY nodes found; nothing to add.")
        return

    g = json.load(open(gltf_path, "r", encoding="utf-8"))
    nodes = g.get("nodes", [])
    if not nodes:
        raise SystemExit("No nodes in glTF")

    # Determine glTF scene root index
    root_i = int(g.get("scenes",[{}])[0].get("nodes",[0])[0])

    added = 0
    for (name, parent, pos, quat, nf) in dummies:
        if find_node_by_name(nodes, name) is not None:
            continue

        ni = len(nodes)
        nodes.append({
            "name": name,
            "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "extras": {"vfx_type": "dummy", "vfx_parent": parent, "vfx_frames": int(nf)}
        })

        # Parent it
        pi = find_node_by_name(nodes, parent) if parent else None
        if pi is None:
            pi = root_i  # fall back to root
        ensure_child(nodes, pi, ni)
        added += 1

    g["nodes"] = nodes

    out_gltf = os.path.splitext(gltf_path)[0] + "_with_dummies.gltf"
    out_bin  = os.path.splitext(gltf_path)[0] + "_with_dummies.bin"

    # copy bin & point buffer uri to the new file
    shutil.copyfile(os.path.splitext(gltf_path)[0] + ".bin", out_bin)
    g["buffers"][0]["uri"] = os.path.basename(out_bin)

    json.dump(g, open(out_gltf, "w", encoding="utf-8", newline=""), indent=2)

    print(f"[OK] Added {added} dummy node(s).")
    print("[OK] Wrote:", out_gltf)
    print("[OK] Wrote:", out_bin)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
