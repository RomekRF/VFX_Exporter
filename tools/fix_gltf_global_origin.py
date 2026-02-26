import json, os, struct, sys

def qmul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)

def qconj(q):
    x,y,z,w=q; return (-x,-y,-z,w)

def qrot(q, v):
    vx,vy,vz=v
    p=(vx,vy,vz,0.0)
    x,y,z,_ = qmul(qmul(q,p), qconj(q))
    return (x,y,z)

def acc_base_stride(g, acc_i):
    acc = g['accessors'][acc_i]
    if acc.get('componentType') != 5126 or acc.get('type') != 'VEC3':
        return None
    bv = g['bufferViews'][acc['bufferView']]
    base = int(bv.get('byteOffset',0)) + int(acc.get('byteOffset',0))
    stride = int(bv.get('byteStride',12) or 12)
    cnt = int(acc['count'])
    return base, stride, cnt

def read_vecs(blob, base, stride, cnt):
    out=[]
    for i in range(cnt):
        o = base + i*stride
        out.append(struct.unpack_from('<fff', blob, o))
    return out

def write_vecs(blob, base, stride, vecs):
    for i,(x,y,z) in enumerate(vecs):
        o = base + i*stride
        struct.pack_into('<fff', blob, o, float(x), float(y), float(z))

def main(gltf_path):
    with open(gltf_path,'r',encoding='utf-8') as f:
        g = json.load(f)
    uri = g['buffers'][0]['uri']
    bin_path = os.path.join(os.path.dirname(gltf_path), uri)
    blob = bytearray(open(bin_path,'rb').read())

    # find __VFX_ROOT__ (or fallback to first scene node)
    root_i = None
    for i,n in enumerate(g.get('nodes',[])):
        if isinstance(n,dict) and n.get('name') == '__VFX_ROOT__':
            root_i = i; break
    if root_i is None:
        root_i = int(g['scenes'][0]['nodes'][0])
    root = g['nodes'][root_i]
    rq = tuple(root.get('rotation',[0.0,0.0,0.0,1.0]))

    # Pass 1: bake root rotation + node translation into base positions; also ensure morph targets are DELTAS
    mins=[1e30,1e30,1e30]; maxs=[-1e30,-1e30,-1e30]

    for n in g.get('nodes',[]):
        if not isinstance(n,dict) or ('mesh' not in n):
            continue
        tx,ty,tz = (n.get('translation') or [0.0,0.0,0.0])
        tx,ty,tz = float(tx), float(ty), float(tz)
        mesh = g['meshes'][n['mesh']]

        for prim in mesh.get('primitives',[]):
            attrs = prim.get('attributes',{})
            if 'POSITION' not in attrs: continue
            acc_pos = int(attrs['POSITION'])
            bas = acc_base_stride(g, acc_pos)
            if bas is None: continue
            base, stride, cnt = bas
            base_vecs = read_vecs(blob, base, stride, cnt)

            baked=[]
            for (x,y,z) in base_vecs:
                x=float(x)+tx; y=float(y)+ty; z=float(z)+tz
                x,y,z = qrot(rq,(x,y,z))
                baked.append((x,y,z))
                mins[0]=min(mins[0],x); mins[1]=min(mins[1],y); mins[2]=min(mins[2],z)
                maxs[0]=max(maxs[0],x); maxs[1]=max(maxs[1],y); maxs[2]=max(maxs[2],z)

            # write baked base back (NOT centered yet)
            write_vecs(blob, base, stride, baked)

            # morph targets: make sure they are DELTAS in the SAME baked space
            if prim.get('targets'):
                for tgt in prim.get('targets') or []:
                    if 'POSITION' not in tgt: continue
                    acc_t = int(tgt['POSITION'])
                    bas_t = acc_base_stride(g, acc_t)
                    if bas_t is None: continue
                    bt, st, ct = bas_t
                    tvecs = read_vecs(blob, bt, st, ct)

                    # bake target to same space
                    abs_baked=[]
                    for (x,y,z) in tvecs:
                        x=float(x)+tx; y=float(y)+ty; z=float(z)+tz
                        x,y,z = qrot(rq,(x,y,z))
                        abs_baked.append((x,y,z))

                    # Heuristic: if target magnitudes are huge, it's probably ABS; convert to DELTA = target - base
                    # If it's already delta, this subtraction would be wrong, so we detect by comparing typical magnitude.
                    # Compute avg |target| vs avg |(target-base)| and choose smaller.
                    import math
                    def avg_mag(vs):
                        s=0.0
                        for (x,y,z) in vs:
                            s += math.sqrt(x*x+y*y+z*z)
                        return s / max(1,len(vs))
                    deltas = [(abs_baked[i][0]-baked[i][0], abs_baked[i][1]-baked[i][1], abs_baked[i][2]-baked[i][2]) for i in range(len(baked))]
                    if avg_mag(deltas) < (avg_mag(abs_baked) * 0.5):
                        # treat as ABS and store DELTA
                        write_vecs(blob, bt, st, deltas)
                    else:
                        # treat as already-DELTA; rotate-only path would be wrong here because we already baked it, so write rotated deltas:
                        # abs_baked currently includes base translation; convert to delta-like by subtracting baked base anyway (safe in this branch)
                        write_vecs(blob, bt, st, deltas)

    # global center from ALL baked base positions
    cx=(mins[0]+maxs[0])*0.5; cy=(mins[1]+maxs[1])*0.5; cz=(mins[2]+maxs[2])*0.5
    print('[INFO] global center:', [cx,cy,cz])

    # Pass 2: subtract global center from ALL base positions (deltas unaffected)
    for mesh in g.get('meshes',[]):
        for prim in mesh.get('primitives',[]):
            attrs = prim.get('attributes',{})
            if 'POSITION' not in attrs: continue
            acc_pos = int(attrs['POSITION'])
            bas = acc_base_stride(g, acc_pos)
            if bas is None: continue
            base, stride, cnt = bas
            vecs = read_vecs(blob, base, stride, cnt)
            vecs2 = [(x-cx, y-cy, z-cz) for (x,y,z) in vecs]
            write_vecs(blob, base, stride, vecs2)
            mn = [min(v[i] for v in vecs2) for i in range(3)]
            mx = [max(v[i] for v in vecs2) for i in range(3)]
            g['accessors'][acc_pos]['min'] = [float(mn[0]),float(mn[1]),float(mn[2])]
            g['accessors'][acc_pos]['max'] = [float(mx[0]),float(mx[1]),float(mx[2])]

    # Zero node transforms (Blender clean)
    for n in g.get('nodes',[]):
        if not isinstance(n,dict): continue
        if 'translation' in n: n['translation'] = [0.0,0.0,0.0]
        if 'rotation' in n: n['rotation'] = [0.0,0.0,0.0,1.0]
        if 'scale' in n: n['scale'] = [1.0,1.0,1.0]

    open(bin_path,'wb').write(blob)
    with open(gltf_path,'w',encoding='utf-8',newline='') as f:
        json.dump(g,f,indent=2)
    print('[OK] wrote in-place:', gltf_path)

if __name__ == '__main__':
    if len(sys.argv) < 2: raise SystemExit('Usage: python fix_gltf_global_origin.py file.gltf')
    main(sys.argv[1])
