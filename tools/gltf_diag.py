import json, os, struct, sys, math

COMPONENT_BYTE_SIZE = {5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4}
TYPE_COMPONENTS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
FMT = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}

def mat4_identity():
    return [1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0]

def mat4_mul(a,b):
    # column-major
    out = [0.0]*16
    for c in range(4):
        for r in range(4):
            out[c*4+r] = (a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3])
    return out

def quat_to_mat4(q):
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    # column-major 3x3 into 4x4
    return [
        1-2*(yy+zz), 2*(xy+wz),   2*(xz-wy),   0,
        2*(xy-wz),   1-2*(xx+zz), 2*(yz+wx),   0,
        2*(xz+wy),   2*(yz-wx),   1-2*(xx+yy), 0,
        0,0,0,1
    ]

def trs_to_mat4(t, r, s):
    T = mat4_identity()
    T[12], T[13], T[14] = t
    R = quat_to_mat4(r)
    S = mat4_identity()
    S[0], S[5], S[10] = s
    return mat4_mul(T, mat4_mul(R, S))

def transform_point(m, p):
    x,y,z = p
    # column-major: p' = M * [x,y,z,1]
    return (
        m[0]*x + m[4]*y + m[8]*z + m[12],
        m[1]*x + m[5]*y + m[9]*z + m[13],
        m[2]*x + m[6]*y + m[10]*z + m[14],
    )

def read_accessor(g, bin_blob, acc_idx):
    acc = g['accessors'][acc_idx]
    bv = g['bufferViews'][acc['bufferView']]
    comp = TYPE_COMPONENTS[acc['type']]
    ctype = acc['componentType']
    count = acc['count']
    stride = bv.get('byteStride', comp * COMPONENT_BYTE_SIZE[ctype])
    off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    fmt = '<' + (FMT[ctype] * comp)
    mn = [float('inf')] * comp
    mx = [float('-inf')] * comp
    for i in range(count):
        base = off + i * stride
        vals = struct.unpack_from(fmt, bin_blob, base)
        for j,v in enumerate(vals):
            if v < mn[j]: mn[j] = v
            if v > mx[j]: mx[j] = v
    return mn, mx

def main(p):
    with open(p, 'r', encoding='utf-8') as f: g = json.load(f)
    buf_uri = g.get('buffers',[{}])[0].get('uri')
    bin_blob = b''
    if buf_uri:
        bin_path = os.path.join(os.path.dirname(p), buf_uri)
        with open(bin_path, 'rb') as f: bin_blob = f.read()

    nodes = g.get('nodes', [])
    scenes = g.get('scenes', [])
    s0 = scenes[0] if scenes else {'nodes': list(range(len(nodes)))}
    roots = list(s0.get('nodes') or list(range(len(nodes))))

    # Mesh POSITION bounds (geometry scale)
    gmins = [float('inf')]*3
    gmaxs = [float('-inf')]*3
    pcnt = 0
    for mesh in g.get('meshes', []):
        for prim in mesh.get('primitives', []):
            pos = (prim.get('attributes') or {}).get('POSITION')
            if pos is None or not bin_blob: continue
            mn, mx = read_accessor(g, bin_blob, pos)
            if len(mn) >= 3:
                gmins[0]=min(gmins[0],mn[0]); gmins[1]=min(gmins[1],mn[1]); gmins[2]=min(gmins[2],mn[2])
                gmaxs[0]=max(gmaxs[0],mx[0]); gmaxs[1]=max(gmaxs[1],mx[1]); gmaxs[2]=max(gmaxs[2],mx[2])
                pcnt += 1
    gspans = [gmaxs[i]-gmins[i] for i in range(3)] if pcnt else [0,0,0]

    # Local TRS translation span
    tmins = [float('inf')]*3
    tmaxs = [float('-inf')]*3
    tcnt = 0
    rotCnt = 0
    sclCnt = 0
    childCnt = 0
    for n in nodes:
        if isinstance(n, dict) and 'children' in n: childCnt += 1
        if isinstance(n, dict) and 'rotation' in n: rotCnt += 1
        if isinstance(n, dict) and 'scale' in n: sclCnt += 1
        t = n.get('translation') if isinstance(n, dict) else None
        if isinstance(t, list) and len(t)==3:
            try:
                x,y,z = float(t[0]), float(t[1]), float(t[2])
            except Exception:
                continue
            tmins[0]=min(tmins[0],x); tmins[1]=min(tmins[1],y); tmins[2]=min(tmins[2],z)
            tmaxs[0]=max(tmaxs[0],x); tmaxs[1]=max(tmaxs[1],y); tmaxs[2]=max(tmaxs[2],z)
            tcnt += 1
    tspans = [tmaxs[i]-tmins[i] for i in range(3)] if tcnt else [0,0,0]

    # World-space translation span (using hierarchy + TRS)
    wmins = [float('inf')]*3
    wmaxs = [float('-inf')]*3
    wcnt = 0
    seen = set()

    def node_local_mat(i):
        n = nodes[i]
        if not isinstance(n, dict):
            return mat4_identity()
        t = n.get('translation') or [0.0,0.0,0.0]
        r = n.get('rotation') or [0.0,0.0,0.0,1.0]
        s = n.get('scale') or [1.0,1.0,1.0]
        try:
            t = [float(t[0]), float(t[1]), float(t[2])]
            r = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
            s = [float(s[0]), float(s[1]), float(s[2])]
        except Exception:
            return mat4_identity()
        return trs_to_mat4(t,r,s)

    sys.setrecursionlimit(10000)
    def walk(i, parent_m):
        nonlocal wcnt
        if i in seen: return
        seen.add(i)
        m = mat4_mul(parent_m, node_local_mat(i))
        x,y,z = transform_point(m, (0.0,0.0,0.0))
        wmins[0]=min(wmins[0],x); wmins[1]=min(wmins[1],y); wmins[2]=min(wmins[2],z)
        wmaxs[0]=max(wmaxs[0],x); wmaxs[1]=max(wmaxs[1],y); wmaxs[2]=max(wmaxs[2],z)
        wcnt += 1
        ch = nodes[i].get('children') if isinstance(nodes[i], dict) else None
        if isinstance(ch, list):
            for c in ch:
                if isinstance(c, int): walk(c, m)

    I = mat4_identity()
    for r in roots:
        if isinstance(r, int): walk(r, I)
    wspans = [wmaxs[i]-wmins[i] for i in range(3)] if wcnt else [0,0,0]

    print('--- glTF DIAG ---')
    print('file:', p)
    print('POSITION prims:', pcnt)
    print('nodes:', len(nodes), ' scene0.roots:', len(roots))
    print('nodesWithChildren:', childCnt, ' nodesWithRotation:', rotCnt, ' nodesWithScale:', sclCnt)
    print('mesh span  :', gspans)
    print('trs span   :', tspans)
    print('world span :', wspans)
    def safe_ratio(a,b):
        if not b: return None
        return a/b
    ratios = [safe_ratio(gspans[i], tspans[i]) for i in range(3)]
    ratios = [r for r in ratios if r is not None and r > 0]
    if ratios:
        ratios.sort()
        mid = ratios[len(ratios)//2]
        print('suggested TRS multiplier (median axis):', mid)
        print('Interpretation: if your TRS is tiny vs mesh, multiply --trs-scale by this.')

if __name__ == '__main__':
    if len(sys.argv) < 2: raise SystemExit('Usage: python tools/gltf_diag.py path/to/file.gltf')
    main(sys.argv[1])
