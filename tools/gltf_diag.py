import json, os, struct, sys

COMPONENT_BYTE_SIZE = {5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4}
TYPE_COMPONENTS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
FMT = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}

def read_accessor(g, bin_blob, acc_idx):
    acc = g['accessors'][acc_idx]
    bv = g['bufferViews'][acc['bufferView']]
    comp = TYPE_COMPONENTS[acc['type']]
    ctype = acc['componentType']
    count = acc['count']
    stride = bv.get('byteStride', comp * COMPONENT_BYTE_SIZE[ctype])
    off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    fmt = '<' + (FMT[ctype] * comp)
    size = struct.calcsize(fmt)
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
    # load first buffer (typical .gltf + .bin)
    buf_uri = g['buffers'][0].get('uri')
    if not buf_uri: raise SystemExit('No external buffer uri found.')
    bin_path = os.path.join(os.path.dirname(p), buf_uri)
    with open(bin_path, 'rb') as f: bin_blob = f.read()

    # Node translation stats
    tmins = [float('inf')]*3
    tmaxs = [float('-inf')]*3
    tcnt = 0
    for n in g.get('nodes', []):
        t = n.get('translation')
        if isinstance(t, list) and len(t)==3:
            try:
                x,y,z = float(t[0]), float(t[1]), float(t[2])
            except Exception:
                continue
            tmins[0]=min(tmins[0],x); tmins[1]=min(tmins[1],y); tmins[2]=min(tmins[2],z)
            tmaxs[0]=max(tmaxs[0],x); tmaxs[1]=max(tmaxs[1],y); tmaxs[2]=max(tmaxs[2],z)
            tcnt += 1
    tspans = [tmaxs[i]-tmins[i] for i in range(3)] if tcnt else [0,0,0]

    # Mesh POSITION bounds stats
    gmins = [float('inf')]*3
    gmaxs = [float('-inf')]*3
    pcnt = 0
    for mesh in g.get('meshes', []):
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes', {})
            pos = attrs.get('POSITION')
            if pos is None: continue
            mn, mx = read_accessor(g, bin_blob, pos)
            if len(mn) >= 3:
                gmins[0]=min(gmins[0],mn[0]); gmins[1]=min(gmins[1],mn[1]); gmins[2]=min(gmins[2],mn[2])
                gmaxs[0]=max(gmaxs[0],mx[0]); gmaxs[1]=max(gmaxs[1],mx[1]); gmaxs[2]=max(gmaxs[2],mx[2])
                pcnt += 1
    gspans = [gmaxs[i]-gmins[i] for i in range(3)] if pcnt else [0,0,0]

    print('--- glTF DIAG ---')
    print('file:', p)
    print('POSITION prims:', pcnt)
    print('nodes w/ translation:', tcnt)
    print('mesh span  :', gspans)
    print('trs span   :', tspans)
    # Suggest multiplier so TRS scale roughly matches mesh scale
    def safe_ratio(a,b):
        if b == 0: return None
        return a/b
    ratios = [safe_ratio(gspans[i], tspans[i]) for i in range(3)]
    ratios = [r for r in ratios if r is not None and r > 0]
    if ratios:
        ratios.sort()
        mid = ratios[len(ratios)//2]
        print('suggested TRS multiplier (median axis):', mid)
        print('Interpretation: if your TRS is tiny vs mesh, multiply --trs-scale by this.')
    else:
        print('No valid ratio could be computed (missing mesh or TRS).')

if __name__ == '__main__':
    if len(sys.argv) < 2: raise SystemExit('Usage: python tools/gltf_diag.py path/to/file.gltf')
    main(sys.argv[1])
