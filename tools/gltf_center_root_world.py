import json, os, struct, sys, math

COMPONENT_BYTE_SIZE = {5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4}
TYPE_COMPONENTS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
FMT = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}

def mat4_identity():
    return [1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,  0.0,0.0,0.0,1.0]

def mat4_mul(a,b):
    out=[0.0]*16
    for c in range(4):
        for r in range(4):
            out[c*4+r] = (a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3])
    return out

def quat_to_mat4(q):
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return [
        1-2*(yy+zz), 2*(xy+wz),   2*(xz-wy),   0,
        2*(xy-wz),   1-2*(xx+zz), 2*(yz+wx),   0,
        2*(xz+wy),   2*(yz-wx),   1-2*(xx+yy), 0,
        0,0,0,1
    ]

def trs_to_mat4(t,r,s):
    T=mat4_identity(); T[12],T[13],T[14]=t
    R=quat_to_mat4(r)
    S=mat4_identity(); S[0],S[5],S[10]=s
    return mat4_mul(T, mat4_mul(R,S))

def transform_point(m,p):
    x,y,z=p
    return (m[0]*x + m[4]*y + m[8]*z + m[12],
            m[1]*x + m[5]*y + m[9]*z + m[13],
            m[2]*x + m[6]*y + m[10]*z + m[14])

def read_accessor_bounds(g, bin_blob, acc_idx):
    acc=g['accessors'][acc_idx]
    if acc.get('type')=='VEC3' and 'min' in acc and 'max' in acc:
        mn=acc['min']; mx=acc['max']
        return [float(mn[0]),float(mn[1]),float(mn[2])],[float(mx[0]),float(mx[1]),float(mx[2])]
    bv=g['bufferViews'][acc['bufferView']]
    comp=TYPE_COMPONENTS[acc['type']]
    ctype=acc['componentType']
    count=acc['count']
    stride=bv.get('byteStride', comp*COMPONENT_BYTE_SIZE[ctype])
    off=bv.get('byteOffset',0)+acc.get('byteOffset',0)
    fmt='<'+(FMT[ctype]*comp)
    mn=[float('inf')]*comp; mx=[float('-inf')]*comp
    for i in range(count):
        base=off+i*stride
        vals=struct.unpack_from(fmt, bin_blob, base)
        for j,v in enumerate(vals):
            if v<mn[j]: mn[j]=v
            if v>mx[j]: mx[j]=v
    return mn[:3], mx[:3]

def mesh_local_bounds(g, bin_blob, mesh_idx):
    mesh=g['meshes'][mesh_idx]
    mn=[float('inf')]*3; mx=[float('-inf')]*3
    ok=False
    for prim in mesh.get('primitives', []):
        pos=(prim.get('attributes') or {}).get('POSITION')
        if pos is None: continue
        a,b = read_accessor_bounds(g, bin_blob, pos)
        mn[0]=min(mn[0],a[0]); mn[1]=min(mn[1],a[1]); mn[2]=min(mn[2],a[2])
        mx[0]=max(mx[0],b[0]); mx[1]=max(mx[1],b[1]); mx[2]=max(mx[2],b[2])
        ok=True
    return (mn,mx) if ok else None

def bbox_corners(mn,mx):
    x0,y0,z0=mn; x1,y1,z1=mx
    return [(x0,y0,z0),(x1,y0,z0),(x0,y1,z0),(x1,y1,z0),(x0,y0,z1),(x1,y0,z1),(x0,y1,z1),(x1,y1,z1)]

def main(path):
    with open(path,'r',encoding='utf-8') as f: g=json.load(f)
    nodes=g.get('nodes') or []
    scenes=g.get('scenes') or []
    if not scenes: raise SystemExit('No scenes')
    roots=list((scenes[0].get('nodes') or []))
    if not roots: raise SystemExit('scene0.nodes empty')
    root_idx=int(roots[0])
    if root_idx<0 or root_idx>=len(nodes): raise SystemExit('Invalid root index')

    buf_uri=(g.get('buffers') or [{}])[0].get('uri')
    if not buf_uri: raise SystemExit('No buffer uri')
    bin_path=os.path.join(os.path.dirname(path), buf_uri)
    with open(bin_path,'rb') as f: bin_blob=f.read()

    # World bounds from mesh nodes
    wmin=[float('inf')]*3; wmax=[float('-inf')]*3
    used=0
    for ni,n in enumerate(nodes):
        if not isinstance(n, dict): continue
        if 'mesh' not in n: continue
        mi=int(n['mesh'])
        b=mesh_local_bounds(g, bin_blob, mi)
        if not b: continue
        mn,mx=b
        t=n.get('translation') or [0.0,0.0,0.0]
        r=n.get('rotation') or [0.0,0.0,0.0,1.0]
        s=n.get('scale') or [1.0,1.0,1.0]
        try:
            t=[float(t[0]),float(t[1]),float(t[2])]
            r=[float(r[0]),float(r[1]),float(r[2]),float(r[3])]
            s=[float(s[0]),float(s[1]),float(s[2])]
        except Exception:
            t=[0.0,0.0,0.0]; r=[0.0,0.0,0.0,1.0]; s=[1.0,1.0,1.0]
        M=trs_to_mat4(t,r,s)
        for c in bbox_corners(mn,mx):
            x,y,z=transform_point(M,c)
            wmin[0]=min(wmin[0],x); wmin[1]=min(wmin[1],y); wmin[2]=min(wmin[2],z)
            wmax[0]=max(wmax[0],x); wmax[1]=max(wmax[1],y); wmax[2]=max(wmax[2],z)
        used += 1

    if used == 0: raise SystemExit('No mesh nodes found; cannot center')
    center=[(wmin[i]+wmax[i])*0.5 for i in range(3)]

    rt=nodes[root_idx].get('translation') or [0.0,0.0,0.0]
    try: rt=[float(rt[0]),float(rt[1]),float(rt[2])]
    except Exception: rt=[0.0,0.0,0.0]
    nodes[root_idx]['translation']=[rt[0]-center[0], rt[1]-center[1], rt[2]-center[2]]
    g['nodes']=nodes

    with open(path,'w',encoding='utf-8') as f: json.dump(g,f,ensure_ascii=False)
    print('Centered(root world): meshesUsed=',used,' worldCenter=',center,' newRootTranslation=',nodes[root_idx]['translation'])

if __name__=='__main__':
    if len(sys.argv)<2: raise SystemExit('Usage: python tools/gltf_center_root_world.py file.gltf')
    main(sys.argv[1])
