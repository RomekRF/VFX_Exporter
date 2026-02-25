import json, os, struct, sys

COMPONENT_BYTE_SIZE = {5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4}
TYPE_COMPONENTS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
FMT = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}

def read_accessor_minmax(g, bin_blob, acc_idx):
    acc = g['accessors'][acc_idx]
    if acc.get('type') == 'VEC3' and 'min' in acc and 'max' in acc:
        mn = acc['min']; mx = acc['max']
        return [float(mn[0]),float(mn[1]),float(mn[2])],[float(mx[0]),float(mx[1]),float(mx[2])]
    bv = g['bufferViews'][acc['bufferView']]
    comp = TYPE_COMPONENTS[acc['type']]
    ctype = acc['componentType']
    count = acc['count']
    stride = bv.get('byteStride', comp * COMPONENT_BYTE_SIZE[ctype])
    off = bv.get('byteOffset',0) + acc.get('byteOffset',0)
    fmt = '<' + (FMT[ctype] * comp)
    mn = [float('inf')]*comp
    mx = [float('-inf')]*comp
    for i in range(count):
        base = off + i*stride
        vals = struct.unpack_from(fmt, bin_blob, base)
        for j,v in enumerate(vals):
            if v < mn[j]: mn[j]=v
            if v > mx[j]: mx[j]=v
    return mn[:3], mx[:3]

def recenter_accessor_positions(g, bin_blob, acc_idx, center):
    acc = g['accessors'][acc_idx]
    if acc.get('componentType') != 5126 or acc.get('type') != 'VEC3':
        return 0
    bv = g['bufferViews'][acc['bufferView']]
    count = acc['count']
    stride = bv.get('byteStride', 12)
    off = bv.get('byteOffset',0) + acc.get('byteOffset',0)
    cx,cy,cz = center
    changed = 0
    for i in range(count):
        base = off + i*stride
        x,y,z = struct.unpack_from('<fff', bin_blob, base)
        x -= cx; y -= cy; z -= cz
        struct.pack_into('<fff', bin_blob, base, x,y,z)
        changed += 1
    # update min/max
    mn = acc.get('min')
    mx = acc.get('max')
    if isinstance(mn, list) and isinstance(mx, list) and len(mn)==3 and len(mx)==3:
        acc['min'] = [float(mn[0]) - cx, float(mn[1]) - cy, float(mn[2]) - cz]
        acc['max'] = [float(mx[0]) - cx, float(mx[1]) - cy, float(mx[2]) - cz]
    return changed

def main(gltf_path):
    with open(gltf_path,'r',encoding='utf-8') as f: g=json.load(f)
    buf_uri = (g.get('buffers') or [{}])[0].get('uri')
    if not buf_uri: raise SystemExit('No external buffer uri')
    bin_path = os.path.join(os.path.dirname(gltf_path), buf_uri)
    with open(bin_path,'rb') as f: blob=bytearray(f.read())

    meshes = g.get('meshes') or []
    if not meshes: raise SystemExit('No meshes in glTF')

    total_changed = 0
    for mi, mesh in enumerate(meshes):
        # gather all POSITION accessors used by this mesh
        pos_accessors = []
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes') or {}
            pos = attrs.get('POSITION')
            if isinstance(pos, int): pos_accessors.append(pos)
        if not pos_accessors:
            continue
        # compute bounds across all POSITION accessors
        gmin=[float('inf')]*3; gmax=[float('-inf')]*3
        for ai in pos_accessors:
            mn,mx = read_accessor_minmax(g, blob, ai)
            gmin[0]=min(gmin[0],mn[0]); gmin[1]=min(gmin[1],mn[1]); gmin[2]=min(gmin[2],mn[2])
            gmax[0]=max(gmax[0],mx[0]); gmax[1]=max(gmax[1],mx[1]); gmax[2]=max(gmax[2],mx[2])
        center=[(gmin[i]+gmax[i])*0.5 for i in range(3)]
        # recenter all accessors for this mesh
        for ai in pos_accessors:
            total_changed += recenter_accessor_positions(g, blob, ai, center)
        print(f'mesh[{mi}] recentered by {center} (pos_accessors={len(pos_accessors)})')

    if total_changed == 0:
        raise SystemExit('No POSITION vertices updated (unexpected)')

    # Write updated bin
    with open(bin_path,'wb') as f: f.write(blob)
    with open(gltf_path,'w',encoding='utf-8') as f: json.dump(g,f,ensure_ascii=False)
    print('DONE: updated POSITION vertices:', total_changed)

if __name__=='__main__':
    if len(sys.argv)<2: raise SystemExit('Usage: python tools/gltf_recenter_mesh_positions.py file.gltf')
    main(sys.argv[1])
