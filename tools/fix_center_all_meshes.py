import json, os, struct, sys, math

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
        o=base+i*stride
        out.append(struct.unpack_from("<fff", blob, o))
    return out

def write_vecs(blob, base, stride, vecs):
    for i,(x,y,z) in enumerate(vecs):
        o=base+i*stride
        struct.pack_into("<fff", blob, o, float(x), float(y), float(z))

def bbox_center(vecs):
    minx=miny=minz=1e30
    maxx=maxy=maxz=-1e30
    for x,y,z in vecs:
        minx=min(minx,x); miny=min(miny,y); minz=min(minz,z)
        maxx=max(maxx,x); maxy=max(maxy,y); maxz=max(maxz,z)
    return ((minx+maxx)*0.5, (miny+maxy)*0.5, (minz+maxz)*0.5)

def main(gltf_path):
    with open(gltf_path,"r",encoding="utf-8") as f:
        g=json.load(f)

    uri=g["buffers"][0]["uri"]
    bin_path=os.path.join(os.path.dirname(gltf_path), uri)
    blob=bytearray(open(bin_path,"rb").read())

    # Build node->mesh map
    nodes=g.get("nodes",[])
    meshes=g.get("meshes",[])

    # Ensure translations exist
    for n in nodes:
        if "translation" not in n:
            n["translation"]=[0.0,0.0,0.0]

    # For each node that has a mesh:
    # - center its POSITION accessors
    # - add the center to node.translation (keeps placement stable)
    node_centers={}
    for ni,n in enumerate(nodes):
        if "mesh" not in n:
            continue
        mi=int(n["mesh"])
        m=meshes[mi]
        total=[0.0,0.0,0.0]
        count=0

        for prim in m.get("primitives",[]):
            attrs=prim.get("attributes",{})
            if "POSITION" not in attrs:
                continue
            ai=int(attrs["POSITION"])
            bas=acc_base_stride(g, ai)
            if bas is None:
                continue
            base,stride,cnt=bas
            vecs=read_vecs(blob, base, stride, cnt)
            cx,cy,cz=bbox_center(vecs)

            # subtract center from positions
            vecs2=[(x-cx,y-cy,z-cz) for (x,y,z) in vecs]
            write_vecs(blob, base, stride, vecs2)

            # update accessor min/max
            mn=[min(v[i] for v in vecs2) for i in range(3)]
            mx=[max(v[i] for v in vecs2) for i in range(3)]
            g["accessors"][ai]["min"]=[float(mn[0]),float(mn[1]),float(mn[2])]
            g["accessors"][ai]["max"]=[float(mx[0]),float(mx[1]),float(mx[2])]

            total[0]+=cx; total[1]+=cy; total[2]+=cz
            count+=1

        if count>0:
            cx,cy,cz=(total[0]/count, total[1]/count, total[2]/count)
            node_centers[ni]=(cx,cy,cz)
            # add center to node translation so the object stays where it was
            n["translation"][0]+=cx
            n["translation"][1]+=cy
            n["translation"][2]+=cz

    # Choose anchor: flagpole if present, else first mesh node
    anchor=None
    for i,n in enumerate(nodes):
        if n.get("name","").lower()=="flagpole":
            anchor=i; break
    if anchor is None:
        anchor=next((i for i,n in enumerate(nodes) if "mesh" in n), None)

    if anchor is not None:
        ax,ay,az=nodes[anchor].get("translation",[0.0,0.0,0.0])

        # subtract anchor translation from all nodes (so pole goes to 0,0,0)
        for n in nodes:
            t=n.get("translation")
            if t is None: 
                continue
            n["translation"]=[t[0]-ax, t[1]-ay, t[2]-az]

        print(f"[INFO] Anchored scene to node {anchor} ({nodes[anchor].get('name','')}). Shift=({ax:.6g},{ay:.6g},{az:.6g})")

    open(bin_path,"wb").write(blob)
    with open(gltf_path,"w",encoding="utf-8",newline="") as f:
        json.dump(g,f,indent=2)

    print("[OK] Centered ALL meshes, stored offsets as node translations, anchored to flagpole at origin.")

if __name__=="__main__":
    if len(sys.argv)<2:
        raise SystemExit("Usage: python fix_center_all_meshes.py file.gltf")
    main(sys.argv[1])
