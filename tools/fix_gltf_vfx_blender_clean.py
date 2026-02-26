import json, os, struct, math, sys

def qmul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)

def qconj(q):
    x,y,z,w=q
    return (-x,-y,-z,w)

def q_to_mat3(q):
    x,y,z,w=q
    xx,yy,zz=x*x,y*y,z*z
    xy,xz,yz=x*y,x*z,y*z
    wx,wy,wz=w*x,w*y,w*z
    return (
        (1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)),
        (2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)),
        (2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)),
    )

def mat3_mul_v(m,v):
    return (m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
            m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
            m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2])

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

def compute_bbox_center(vecs):
    minx=miny=minz=1e30
    maxx=maxy=maxz=-1e30
    for x,y,z in vecs:
        minx=min(minx,x); miny=min(miny,y); minz=min(minz,z)
        maxx=max(maxx,x); maxy=max(maxy,y); maxz=max(maxz,z)
    return ((minx+maxx)*0.5, (miny+maxy)*0.5, (minz+maxz)*0.5)

def avg_mag(vecs):
    s=0.0
    for x,y,z in vecs:
        s += math.sqrt(x*x+y*y+z*z)
    return s / max(1,len(vecs))

def main(gltf_path):
    with open(gltf_path,"r",encoding="utf-8") as f:
        g = json.load(f)

    uri = g["buffers"][0]["uri"]
    bin_path = os.path.join(os.path.dirname(gltf_path), uri)
    blob = bytearray(open(bin_path,"rb").read())

    # Identify root (prefer __VFX_ROOT__)
    root_i = None
    for i,n in enumerate(g.get("nodes",[])):
        if isinstance(n,dict) and n.get("name") == "__VFX_ROOT__":
            root_i = i
            break
    if root_i is None:
        root_i = int(g["scenes"][0]["nodes"][0])

    nodes = g["nodes"]
    root = nodes[root_i]

    # Root TRS (we will bake these into verts)
    rq = tuple(root.get("rotation",[0.0,0.0,0.0,1.0]))
    rt = root.get("translation",[0.0,0.0,0.0])
    rs = root.get("scale",[1.0,1.0,1.0])
    Rroot = q_to_mat3(rq)

    def apply_root(p):
        # scale -> rotate -> translate
        x,y,z = p
        x*=float(rs[0]); y*=float(rs[1]); z*=float(rs[2])
        x,y,z = mat3_mul_v(Rroot,(x,y,z))
        return (x+float(rt[0]), y+float(rt[1]), z+float(rt[2]))

    def apply_root_linear(v):
        # for DELTAS: scale -> rotate only (no translate)
        x,y,z=v
        x*=float(rs[0]); y*=float(rs[1]); z*=float(rs[2])
        return mat3_mul_v(Rroot,(x,y,z))

    # Determine which meshes have morph targets
    mesh_has_morph = [False]*len(g.get("meshes",[]))
    for mi,mesh in enumerate(g.get("meshes",[])):
        for prim in mesh.get("primitives",[]):
            if prim.get("targets"):
                mesh_has_morph[mi]=True
                break

    # Per-node (mesh instance) bake + recenter
    for ni,n in enumerate(nodes):
        if not isinstance(n,dict) or ("mesh" not in n):
            continue

        mi = int(n["mesh"])
        mesh = g["meshes"][mi]
        nt = n.get("translation",[0.0,0.0,0.0])
        nq = tuple(n.get("rotation",[0.0,0.0,0.0,1.0]))
        ns = n.get("scale",[1.0,1.0,1.0])
        Rn = q_to_mat3(nq)

        def apply_node(p):
            # node local: scale -> rotate -> translate
            x,y,z=p
            x*=float(ns[0]); y*=float(ns[1]); z*=float(ns[2])
            x,y,z = mat3_mul_v(Rn,(x,y,z))
            return (x+float(nt[0]), y+float(nt[1]), z+float(nt[2]))

        def apply_node_linear(v):
            x,y,z=v
            x*=float(ns[0]); y*=float(ns[1]); z*=float(ns[2])
            return mat3_mul_v(Rn,(x,y,z))

        # bake world = root * node
        for prim in mesh.get("primitives",[]):
            attrs = prim.get("attributes",{})
            if "POSITION" not in attrs:
                continue

            acc_pos = int(attrs["POSITION"])
            bas = acc_base_stride(g, acc_pos)
            if bas is None:
                continue
            base_ofs, base_stride, base_cnt = bas
            base_raw = read_vecs(blob, base_ofs, base_stride, base_cnt)

            # Base in baked world (pre-center)
            base_baked = [apply_root(apply_node(v)) for v in base_raw]

            # Center this mesh instance so the visible mesh is at origin
            cx,cy,cz = compute_bbox_center(base_baked)
            base_centered = [(x-cx,y-cy,z-cz) for (x,y,z) in base_baked]
            write_vecs(blob, base_ofs, base_stride, base_centered)

            # Update min/max
            mn = [min(v[i] for v in base_centered) for i in range(3)]
            mx = [max(v[i] for v in base_centered) for i in range(3)]
            g["accessors"][acc_pos]["min"] = [float(mn[0]),float(mn[1]),float(mn[2])]
            g["accessors"][acc_pos]["max"] = [float(mx[0]),float(mx[1]),float(mx[2])]

            # Morph targets: enforce DELTAS, baked with linear part only
            if prim.get("targets"):
                for tgt in prim.get("targets") or []:
                    if "POSITION" not in tgt:
                        continue
                    acc_t = int(tgt["POSITION"])
                    bas_t = acc_base_stride(g, acc_t)
                    if bas_t is None:
                        continue
                    tofs, tstride, tcnt = bas_t
                    t_raw = read_vecs(blob, tofs, tstride, tcnt)

                    # Two interpretations:
                    # A) ABS positions (need full root+node, then delta = abs - base_baked)
                    t_abs_baked = [apply_root(apply_node(v)) for v in t_raw]
                    delta_A = [(t_abs_baked[i][0]-base_baked[i][0],
                                t_abs_baked[i][1]-base_baked[i][1],
                                t_abs_baked[i][2]-base_baked[i][2]) for i in range(base_cnt)]

                    # B) Already DELTAS (need linear root+node only)
                    # delta baked = rootLinear(nodeLinear(delta))
                    delta_B = [apply_root_linear(apply_node_linear(v)) for v in t_raw]

                    # pick whichever looks like "reasonable deltas"
                    # (smaller average magnitude wins)
                    if avg_mag(delta_A) <= avg_mag(delta_B):
                        chosen = delta_A
                    else:
                        chosen = delta_B

                    write_vecs(blob, tofs, tstride, chosen)

        # Zero the node TRS (Blender tools behave)
        n["translation"] = [0.0,0.0,0.0]
        n["rotation"] = [0.0,0.0,0.0,1.0]
        n["scale"] = [1.0,1.0,1.0]

    # Zero root TRS too (we baked it)
    root["translation"] = [0.0,0.0,0.0]
    root["rotation"] = [0.0,0.0,0.0,1.0]
    root["scale"] = [1.0,1.0,1.0]

    # Fix animations: remove TRS channels; keep only weights channels for morph meshes
    if "animations" in g:
        new_anims=[]
        for anim in g["animations"]:
            chans = anim.get("channels",[])
            samps = anim.get("samplers",[])
            kept_chans=[]
            kept_samp_idx=set()

            for ci,ch in enumerate(chans):
                tgt = ch.get("target",{})
                path = tgt.get("path","")
                node_i = tgt.get("node",None)

                if path in ("translation","rotation","scale"):
                    # drop all TRS anim; this fixes the "flagpole is animated" issue
                    continue

                if path == "weights":
                    # only keep if node's mesh actually has morph targets
                    if node_i is None:
                        continue
                    node = nodes[int(node_i)]
                    if not isinstance(node,dict) or ("mesh" not in node):
                        continue
                    if not mesh_has_morph[int(node["mesh"])]:
                        continue
                    kept_chans.append(ch)
                    kept_samp_idx.add(int(ch["sampler"]))
                    continue

                # drop anything else by default (safer)
                continue

            if not kept_chans:
                continue

            # rebuild samplers table compactly
            remap={}
            new_samps=[]
            for old_i in sorted(kept_samp_idx):
                remap[old_i]=len(new_samps)
                new_samps.append(samps[old_i])

            for ch in kept_chans:
                ch["sampler"]=remap[int(ch["sampler"])]

            anim["channels"]=kept_chans
            anim["samplers"]=new_samps
            new_anims.append(anim)

        g["animations"]=new_anims

    open(bin_path,"wb").write(blob)
    with open(gltf_path,"w",encoding="utf-8",newline="") as f:
        json.dump(g,f,indent=2)

    print("[OK] Blender-clean fix applied: per-mesh recenter to origin + morph targets as deltas + TRS anim stripped (weights kept).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python fix_gltf_vfx_blender_clean.py file.gltf")
    main(sys.argv[1])
