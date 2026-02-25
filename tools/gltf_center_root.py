import json, os, struct, sys

COMPONENT_BYTE_SIZE = {5120:1, 5121:1, 5122:2, 5123:2, 5125:4, 5126:4}
TYPE_COMPONENTS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
FMT = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}

def read_accessor_bounds(g, bin_blob, acc_idx):
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

def main(path):
    with open(path, 'r', encoding='utf-8') as f:
        g = json.load(f)

    nodes = g.get('nodes') or []
    scenes = g.get('scenes') or []
    if not scenes:
        raise SystemExit('No scenes in glTF')
    s0 = scenes[0]
    roots = list(s0.get('nodes') or [])
    if not roots:
        raise SystemExit('scene0.nodes is empty')
    root_idx = int(roots[0])
    if root_idx < 0 or root_idx >= len(nodes):
        raise SystemExit(f'Invalid root index {root_idx} for nodes={len(nodes)}')

    buf_uri = (g.get('buffers') or [{}])[0].get('uri')
    if not buf_uri:
        raise SystemExit('No external buffer uri found')
    bin_path = os.path.join(os.path.dirname(path), buf_uri)
    with open(bin_path, 'rb') as f:
        bin_blob = f.read()

    # Compute global POSITION bounds
    gmin = [float('inf')]*3
    gmax = [float('-inf')]*3
    pcnt = 0
    for mesh in g.get('meshes', []):
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes') or {}
            pos = attrs.get('POSITION')
            if pos is None:
                continue
            # Prefer accessor min/max if present
            acc = g['accessors'][pos]
            if 'min' in acc and 'max' in acc and acc.get('type') == 'VEC3':
                mn = acc['min']; mx = acc['max']
            else:
                mn, mx = read_accessor_bounds(g, bin_blob, pos)
            gmin[0] = min(gmin[0], float(mn[0])); gmin[1] = min(gmin[1], float(mn[1])); gmin[2] = min(gmin[2], float(mn[2]))
            gmax[0] = max(gmax[0], float(mx[0])); gmax[1] = max(gmax[1], float(mx[1])); gmax[2] = max(gmax[2], float(mx[2]))
            pcnt += 1

    if pcnt == 0:
        raise SystemExit('No POSITION primitives found; cannot compute center')

    center = [ (gmin[i] + gmax[i]) * 0.5 for i in range(3) ]
    rt = nodes[root_idx].get('translation') or [0.0, 0.0, 0.0]
    try:
        rt = [float(rt[0]), float(rt[1]), float(rt[2])]
    except Exception:
        rt = [0.0, 0.0, 0.0]

    # Move root so geometry center lands at origin
    nodes[root_idx]['translation'] = [rt[0] - center[0], rt[1] - center[1], rt[2] - center[2]]
    g['nodes'] = nodes

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(g, f, ensure_ascii=False)

    print('Centered root:', root_idx, 'center=', center, 'newRootTranslation=', nodes[root_idx]['translation'])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python tools/gltf_center_root.py file.gltf')
    main(sys.argv[1])
