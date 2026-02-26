import json, os, struct, math, sys

def q_to_mat(q):
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return [
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)],
    ]

def mat_mul_v(m,v):
    return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
            m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
            m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]

def read_f32_vec3(buf, ofs, count):
    out = []
    for i in range(count):
        o = ofs + i*12
        out.append(list(struct.unpack_from('<fff', buf, o)))
    return out

def write_f32_vec3(buf, ofs, vecs):
    for i,v in enumerate(vecs):
        struct.pack_into('<fff', buf, ofs + i*12, float(v[0]), float(v[1]), float(v[2]))

def accessor_byte_base(g, acc_i):
    acc = g['accessors'][acc_i]
    if acc.get('componentType') != 5126: return None
    if acc.get('type') != 'VEC3': return None
    bv = g['bufferViews'][acc['bufferView']]
    if bv.get('byteStride', 0) not in (0, 12): return None
    a_off = int(acc.get('byteOffset', 0))
    v_off = int(bv.get('byteOffset', 0))
    return v_off + a_off, int(acc['count'])

def gather_position_accessors(g):
    out = set()
    # base positions
    for mesh in g.get('meshes', []):
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes', {})
            if 'POSITION' in attrs: out.add(int(attrs['POSITION']))
            # morph target positions
            for tgt in prim.get('targets', []) or []:
                if 'POSITION' in tgt: out.add(int(tgt['POSITION']))
    return sorted(out)

def main(p):
    with open(p,'r',encoding='utf-8') as f:
        g = json.load(f)
    # load bin
    uri = g['buffers'][0]['uri']
    bin_path = os.path.join(os.path.dirname(p), uri)
    with open(bin_path,'rb') as f:
        blob = bytearray(f.read())

    # pick root node (prefer __VFX_ROOT__ if present)
    root_i = None
    for i,n in enumerate(g.get('nodes', [])):
        if n.get('name') == '__VFX_ROOT__':
            root_i = i
            break
    if root_i is None: root_i = 0
    rnode = g['nodes'][root_i]
    q = rnode.get('rotation', [0.0,0.0,0.0,1.0])
    R = q_to_mat(q)

    pos_accs = gather_position_accessors(g)
    if not pos_accs:
        print('[WARN] no POSITION accessors found'); return 0

    # First pass: rotate all positions by root rotation, collect global bounds
    mins = [1e30,1e30,1e30]
    maxs = [-1e30,-1e30,-1e30]
    rotated_cache = {}
    for ai in pos_accs:
        base = accessor_byte_base(g, ai)
        if base is None: continue
        ofs, cnt = base
        vecs = read_f32_vec3(blob, ofs, cnt)
        vecs = [mat_mul_v(R,v) for v in vecs]
        rotated_cache[ai] = (ofs, vecs)
        for v in vecs:
            mins[0]=min(mins[0],v[0]); mins[1]=min(mins[1],v[1]); mins[2]=min(mins[2],v[2])
            maxs[0]=max(maxs[0],v[0]); maxs[1]=max(maxs[1],v[1]); maxs[2]=max(maxs[2],v[2])

    cx = (mins[0]+maxs[0])*0.5
    cy = (mins[1]+maxs[1])*0.5
    cz = (mins[2]+maxs[2])*0.5
    center = [cx,cy,cz]
    print('[INFO] bake-origin center:', center)

    # Second pass: subtract center, write back, update accessor min/max
    for ai,(ofs, vecs) in rotated_cache.items():
        vecs2 = [[v[0]-cx, v[1]-cy, v[2]-cz] for v in vecs]
        write_f32_vec3(blob, ofs, vecs2)
        mn = [min(v[i] for v in vecs2) for i in range(3)]
        mx = [max(v[i] for v in vecs2) for i in range(3)]
        g['accessors'][ai]['min'] = [float(mn[0]),float(mn[1]),float(mn[2])]
        g['accessors'][ai]['max'] = [float(mx[0]),float(mx[1]),float(mx[2])]

    # Zero TRS everywhere (Blender-clean import)
    for n in g.get('nodes', []):
        if 'translation' in n: n['translation'] = [0.0,0.0,0.0]
        if 'rotation' in n: n['rotation'] = [0.0,0.0,0.0,1.0]
        if 'scale' in n: n['scale'] = [1.0,1.0,1.0]

    with open(bin_path,'wb') as f:
        f.write(blob)
    with open(p,'w',encoding='utf-8',newline='') as f:
        json.dump(g,f,ensure_ascii=False)
    print('[OK] wrote baked glTF/bin in-place')
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2: raise SystemExit('Usage: python bake_origin_gltf.py file.gltf')
    raise SystemExit(main(sys.argv[1]))
