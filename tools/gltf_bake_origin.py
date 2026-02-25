import json, os, struct, math, sys

gltf_path = sys.argv[1]
with open(gltf_path, "r", encoding="utf-8") as f:
    g = json.load(f)

# glTF rotation is [x,y,z,w]
def q_conj(q): x,y,z,w=q; return (-x,-y,-z,w)
def q_mul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    )
def q_rot(q, v):
    # v as quaternion (x,y,z,0)
    vx,vy,vz=v
    p=(vx,vy,vz,0.0)
    return q_mul(q_mul(q,p), q_conj(q))[:3]

nodes = g.get("nodes", [])
scene0 = g.get("scenes", [])[0]
root_idx = int(scene0.get("nodes",[0])[0])
root = nodes[root_idx]

root_t = root.get("translation",[0.0,0.0,0.0])
root_r = root.get("rotation",[0.0,0.0,0.0,1.0])

# Load BIN
bin_uri = g["buffers"][0]["uri"]
bin_path = os.path.join(os.path.dirname(gltf_path), bin_uri)
data = bytearray(open(bin_path, "rb").read())

accessors = g.get("accessors", [])
bufferViews = g.get("bufferViews", [])

def shift_accessor_pos(acc_i, dx,dy,dz):
    acc = accessors[acc_i]
    if acc.get("type") != "VEC3" or acc.get("componentType") != 5126:
        return
    bv = bufferViews[acc["bufferView"]]
    base = int(bv.get("byteOffset",0)) + int(acc.get("byteOffset",0))
    stride = int(bv.get("byteStride", 12) or 12)
    count = int(acc["count"])
    for i in range(count):
        o = base + i*stride
        x,y,z = struct.unpack_from("<fff", data, o)
        struct.pack_into("<fff", data, o, x+dx, y+dy, z+dz)
    if "min" in acc and "max" in acc:
        mn = acc["min"]; mx = acc["max"]
        acc["min"] = [float(mn[0])+dx, float(mn[1])+dy, float(mn[2])+dz]
        acc["max"] = [float(mx[0])+dx, float(mx[1])+dy, float(mx[2])+dz]

# For each mesh node, bake WORLD translation into geometry and zero node translations.
# WORLD translation = root_t + rotate(root_r, mesh_t)
for i,n in enumerate(nodes):
    if "mesh" not in n:
        continue
    mesh_t = n.get("translation",[0.0,0.0,0.0])
    rt = q_rot(root_r, mesh_t)
    world_t = (root_t[0]+rt[0], root_t[1]+rt[1], root_t[2]+rt[2])

    mesh = g["meshes"][n["mesh"]]
    for prim in mesh.get("primitives", []):
        attrs = prim.get("attributes", {})
        if "POSITION" in attrs:
            shift_accessor_pos(int(attrs["POSITION"]), -world_t[0], -world_t[1], -world_t[2])

    # zero node translation
    if "translation" in n:
        n["translation"] = [0.0,0.0,0.0]

# zero root translation
if "translation" in root:
    root["translation"] = [0.0,0.0,0.0]

# Write back
with open(bin_path, "wb") as f:
    f.write(data)
with open(gltf_path, "w", encoding="utf-8") as f:
    json.dump(g, f, indent=2)

print("Baked to origin OK:", gltf_path)
