import json, os, struct, sys
SEC_DMMY = 0x594D4D44

def cstr(buf, off):
    end = buf.index(0, off)
    return buf[off:end].decode("latin1","replace"), end+1

# Our known mapping (from your confirmed VFX->glTF dump):
# glTF = (-fx, -fz, fy)
# file = (-tx, tz, -ty)
def file_to_gltf(fx,fy,fz): return (-fx, -fz, fy)
def gltf_to_file(tx,ty,tz): return (-tx, tz, -ty)

def dump_gltf(gltf_path, names=("$prop_flag","flagpole","__VFX_ROOT__")):
    g = json.load(open(gltf_path,"r",encoding="utf-8"))
    nodes = g.get("nodes",[])
    out = {}
    for n in nodes:
        nm = (n.get("name","") or "").strip()
        if nm in names:
            out[nm] = {
                "translation": n.get("translation"),
                "rotation": n.get("rotation"),
                "scale": n.get("scale"),
                "matrix": n.get("matrix"),
                "children": n.get("children"),
                "mesh": n.get("mesh"),
            }
    return out

def dump_vfx_dummies(vfx_path):
    b = open(vfx_path,"rb").read()
    if b[:4] != b"VSFX": raise RuntimeError("Not VSFX")
    ver, = struct.unpack_from("<I", b, 4)
    if ver != 0x00040006: raise RuntimeError(f"Not v4.6: 0x{ver:08X}")

    off = 128
    d = {}
    while off < len(b):
        stype, slen = struct.unpack_from("<II", b, off)
        body_len = slen - 4
        body_off = off + 8
        sec_end = body_off + body_len
        if stype == SEC_DMMY:
            body = b[body_off:sec_end]
            o = 0
            name,o = cstr(body,o)
            parent,o = cstr(body,o)
            save_parent = body[o]; o += 1
            fx,fy,fz = struct.unpack_from("<3f", body, o); o += 12
            qx,qy,qz,qw = struct.unpack_from("<4f", body, o); o += 16
            nfr, = struct.unpack_from("<i", body, o); o += 4

            # frame0 (if present)
            f0 = None
            if nfr > 0:
                ffx,ffy,ffz = struct.unpack_from("<3f", body, o); oo=o+12
                fqx,fqy,fqz,fqw = struct.unpack_from("<4f", body, oo)
                f0 = {"pos":(ffx,ffy,ffz), "quat":(fqx,fqy,fqz,fqw)}

            d[name] = {
                "parent": parent,
                "save_parent": int(save_parent),
                "base_pos": (fx,fy,fz),
                "base_quat": (qx,qy,qz,qw),
                "frames": int(nfr),
                "frame0": f0,
            }

        off = sec_end
    return d

def fmt3(v): return f"({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f})"
def fmt4(q): return f"({q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f})"

def main(orig_vfx, new_vfx, gltf_path):
    print("=== Files ===")
    print("ORIG:", orig_vfx)
    print("NEW :", new_vfx)
    print("GLTF:", gltf_path)

    gd = dump_gltf(gltf_path)
    od = dump_vfx_dummies(orig_vfx)
    nd = dump_vfx_dummies(new_vfx)

    print("\n=== glTF nodes (raw) ===")
    for k in ("__VFX_ROOT__","flagpole","$prop_flag"):
        if k in gd:
            print(f"{k}:")
            print("  t:", gd[k]["translation"])
            print("  r:", gd[k]["rotation"])
            print("  m:", "YES" if gd[k]["matrix"] else "no")
        else:
            print(f"{k}: (missing)")

    def show(tag, d):
        if "$prop_flag" not in d:
            print(f"\n[{tag}] $prop_flag not found in DMMY sections")
            return
        p = d["$prop_flag"]
        fx,fy,fz = p["base_pos"]
        tx,ty,tz = file_to_gltf(fx,fy,fz)
        print(f"\n[{tag}] $prop_flag")
        print("  parent      :", p["parent"])
        print("  save_parent :", p["save_parent"])
        print("  base file   :", fmt3((fx,fy,fz)))
        print("  base glTF   :", fmt3((tx,ty,tz)))
        print("  base quat   :", fmt4(p["base_quat"]))
        print("  frames      :", p["frames"])
        if p["frame0"]:
            f0 = p["frame0"]
            print("  frame0 file :", fmt3(f0["pos"]))
            print("  frame0 quat :", fmt4(f0["quat"]))

    show("ORIG VFX", od)
    show("NEW  VFX", nd)

    # Compare ORIG vs NEW in file coords
    if "$prop_flag" in od and "$prop_flag" in nd:
        o = od["$prop_flag"]; n = nd["$prop_flag"]
        op = o["base_pos"]; np = n["base_pos"]
        dq = tuple(n["base_quat"][i]-o["base_quat"][i] for i in range(4))
        dp = (np[0]-op[0], np[1]-op[1], np[2]-op[2])
        print("\n=== ORIG vs NEW delta ===")
        print("  pos delta (file):", fmt3(dp))
        print("  quat delta       :", fmt4(dq))
        print("  save_parent      :", o["save_parent"], "->", n["save_parent"])

        # Compare NEW expected from glTF (if present)
        if "$prop_flag" in gd and gd["$prop_flag"]["translation"]:
            gt = gd["$prop_flag"]["translation"]
            exp = gltf_to_file(gt[0],gt[1],gt[2])
            print("\n=== NEW expected from glTF ===")
            print("  glTF t:", fmt3((gt[0],gt[1],gt[2])))
            print("  expected file pos:", fmt3(exp))
            print("  actual   file pos:", fmt3(np))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: prop_compare.py orig.vfx new.vfx gltf.gltf")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
