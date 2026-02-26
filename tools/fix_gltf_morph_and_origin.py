import json, os, struct, sys

def qmul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)

def qconj(q):
    x,y,z,w=q
    return (-x,-y,-z,w)

def qrot(q, v):
    vx,vy,vz=v
    p=(vx,vy,vz,0.0)
    x,y,z,_ = qmul(qmul(q,p), qconj(q))
    return (x,y,z)

def acc_base_stride(g, acc_i):
    acc = g["accessors"][acc_i]
    if acc.get("componentType") != 5126 or acc.get("type") != "VEC3":
        return None
    bv = g["bufferViews"][acc["bufferView"]]
    base = int(bv.get("byteOffset",0)) + int(acc.get("byteOffset",0))
    stride = int(bv.get("byteStride",12) or 12)
    cnt = int(acc["count"])
    return base, stride, cnt

def read_vecs(blob, base, stride, cnt):
    out=[]
    for i in range(cnt):
        o = base + i*stride
        out.append(struct.unpack_from("<fff", blob, o))
    return out

def write_vecs(blob, base, stride, vecs):
    for i,(x,y,z) in enumerate(vecs):
        o = base + i*stride
        struct.pack_into("<fff", blob, o, float(x), float(y), float(z))

def main(gltf_path):
    with open(gltf_path,"r",encoding="utf-8") as f:
        g = json.load(f)

    uri = g["buffers"][0]["uri"]
    bin_path = os.path.join(os.path.dirname(gltf_path), uri)
    blob = bytearray(open(bin_path,"rb").read())

    # root node
    root_i = None
    for i,n in enumerate(g.get("nodes",[])):
        if isinstance(n,dict) and n.get("name") == "__VFX_ROOT__":
            root_i = i
            break
    if root_i is None:
        root_i = int(g["scenes"][0]["nodes"][0])

    root = g["nodes"][root_i]
    rq = root.get("rotation",[0.0,0.0,0.0,1.0])

    # For each node+primitive:
    # - bake node translation + root rotation into base positions
    # - convert morph targets ABSOLUTE -> DELTA (target -= base)
    # - rotate morph deltas by root rotation
    # - recenter base positions per-mesh to bbox center
    for n in g.get("nodes",[]):
        if not isinstance(n,dict) or ("mesh" not in n):
            continue

        tx,ty,tz = (n.get("translation") or [0.0,0.0,0.0])
        tx,ty,tz = float(tx), float(ty), float(tz)

        mesh = g["meshes"][n["mesh"]]
        for prim in mesh.get("primitives",[]):
            attrs = prim.get("attributes",{})
            if "POSITION" not in attrs:
                continue

            acc_pos = int(attrs["POSITION"])
            bas = acc_base_stride(g, acc_pos)
            if bas is None:
                continue
            base, stride, cnt = bas

            base_vecs = read_vecs(blob, base, stride, cnt)

            # bake translation + root rotation into base
            baked=[]
            minx=miny=minz=1e30
            maxx=maxy=maxz=-1e30
            for (x,y,z) in base_vecs:
                x=float(x)+tx; y=float(y)+ty; z=float(z)+tz
                x,y,z = qrot(tuple(rq),(x,y,z))
                baked.append((x,y,z))
                minx=min(minx,x); miny=min(miny,y); minz=min(minz,z)
                maxx=max(maxx,x); maxy=max(maxy,y); maxz=max(maxz,z)

            # recenter base per-mesh
            cx=(minx+maxx)*0.5
            cy=(miny+maxy)*0.5
            cz=(minz+maxz)*0.5
            baked_centered=[(x-cx,y-cy,z-cz) for (x,y,z) in baked]
            write_vecs(blob, base, stride, baked_centered)

            # update base accessor min/max
            g["accessors"][acc_pos]["min"] = [minx-cx, miny-cy, minz-cz]
            g["accessors"][acc_pos]["max"] = [maxx-cx, maxy-cy, maxz-cz]

            # morph targets: convert ABSOLUTE -> DELTA relative to (baked) base, then rotate by root
            if prim.get("targets"):
                # base for delta should be the *baked* (pre-center) base in same space as targets.
                # We use baked (not centered) so deltas remain correct; centering cancels out anyway.
                base_baked = baked

                for tgt in prim.get("targets") or []:
                    if "POSITION" not in tgt:
                        continue
                    acc_t = int(tgt["POSITION"])
                    bas_t = acc_base_stride(g, acc_t)
                    if bas_t is None:
                        continue
                    bt, st, ct = bas_t
                    tvecs = read_vecs(blob, bt, st, ct)

                    # If targets are absolute, convert to delta: (target_baked - base_baked)
                    # Also rotate target by root rotation so spaces match.
                    out=[]
                    for i,(x,y,z) in enumerate(tvecs):
                        x=float(x)+tx; y=float(y)+ty; z=float(z)+tz
                        x,y,z = qrot(tuple(rq),(x,y,z))
                        bx,by,bz = base_baked[i]
                        out.append((x-bx, y-by, z-bz))

                    write_vecs(blob, bt, st, out)

        # zero node TRS
        n["translation"] = [0.0,0.0,0.0]
        if "rotation" in n: n["rotation"] = [0.0,0.0,0.0,1.0]
        if "scale" in n: n["scale"] = [1.0,1.0,1.0]

    root["translation"] = [0.0,0.0,0.0]
    root["rotation"] = [0.0,0.0,0.0,1.0]
    root["scale"] = [1.0,1.0,1.0]

    open(bin_path,"wb").write(blob)
    with open(gltf_path,"w",encoding="utf-8",newline="") as f:
        json.dump(g,f,indent=2)

    print("[OK] Fixed morph targets (absolute->delta) + baked origin + centered meshes. Wrote in-place:", gltf_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python fix_gltf_morph_and_origin.py file.gltf")
    main(sys.argv[1])
