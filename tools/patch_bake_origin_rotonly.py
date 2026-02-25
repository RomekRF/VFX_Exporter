import sys, re

p = sys.argv[1]
s = open(p, "r", encoding="utf-8").read()

# Replace the entire bake_origin block (we anchor on its INFO print)
pat = r"(?ms)^\s*if\s+bake_origin\s*:\s*.*?\[INFO\]\s+Baking\s+origin.*?\n(?=^\s*out_bin\s*=)"
m = re.search(pat, s)
if not m:
    raise SystemExit("Could not find bake_origin block to replace (anchor '[INFO] Baking origin').")

indent = re.match(r"^\s*", m.group(0)).group(0)

blk = "\n".join([
    indent + "if bake_origin:",
    indent + "    # Bake ONLY the root rotation into geometry so Blender shows nodes at 0,0,0 and mesh is oriented correctly.",
    indent + "    import struct",
    indent + "    def _qmul(a,b):",
    indent + "        ax,ay,az,aw=a; bx,by,bz,bw=b",
    indent + "        return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)",
    indent + "    def _qconj(q):",
    indent + "        x,y,z,w=q; return (-x,-y,-z,w)",
    indent + "    def _qrot(q, v3):",
    indent + "        vx,vy,vz=v3",
    indent + "        p=(vx,vy,vz,0.0)",
    indent + "        x,y,z,_ = _qmul(_qmul(q,p), _qconj(q))",
    indent + "        return (x,y,z)",
    indent + "    try:",
    indent + "        rix = int(gltf['scenes'][0]['nodes'][0])",
    indent + "        rnode = gltf['nodes'][rix]",
    indent + "        rq = rnode.get('rotation', [0.0,0.0,0.0,1.0])  # glTF [x,y,z,w]",
    indent + "        if not isinstance(bin_blob, (bytearray,)):",
    indent + "            bin_blob = bytearray(bin_blob)",
    indent + "        accessors = gltf.get('accessors', [])",
    indent + "        bvs = gltf.get('bufferViews', [])",
    indent + "        visited = set()  # accessor indices we've already rotated",
    indent + "        def _rot_accessor_pos(acc_i):",
    indent + "            if acc_i in visited: return",
    indent + "            visited.add(acc_i)",
    indent + "            acc = accessors[acc_i]",
    indent + "            if acc.get('type') != 'VEC3' or acc.get('componentType') != 5126:",
    indent + "                return",
    indent + "            bv = bvs[acc['bufferView']]",
    indent + "            base = int(bv.get('byteOffset', 0)) + int(acc.get('byteOffset', 0))",
    indent + "            stride = int(bv.get('byteStride', 12) or 12)",
    indent + "            cnt = int(acc['count'])",
    indent + "            minx=miny=minz=1e30; maxx=maxy=maxz=-1e30",
    indent + "            for ii in range(cnt):",
    indent + "                o = base + ii*stride",
    indent + "                x,y,z = struct.unpack_from('<fff', bin_blob, o)",
    indent + "                rx,ry,rz = _qrot(tuple(rq), (float(x),float(y),float(z)))",
    indent + "                struct.pack_into('<fff', bin_blob, o, float(rx),float(ry),float(rz))",
    indent + "                if rx<minx: minx=rx",
    indent + "                if ry<miny: miny=ry",
    indent + "                if rz<minz: minz=rz",
    indent + "                if rx>maxx: maxx=rx",
    indent + "                if ry>maxy: maxy=ry",
    indent + "                if rz>maxz: maxz=rz",
    indent + "            acc['min'] = [float(minx), float(miny), float(minz)]",
    indent + "            acc['max'] = [float(maxx), float(maxy), float(maxz)]",
    indent + "        # Rotate every POSITION accessor referenced by any mesh primitive",
    indent + "        for n in gltf.get('nodes', []):",
    indent + "            if not isinstance(n, dict) or ('mesh' not in n):",
    indent + "                continue",
    indent + "            mesh = gltf['meshes'][n['mesh']]",
    indent + "            for prim in mesh.get('primitives', []):",
    indent + "                attrs = prim.get('attributes', {})",
    indent + "                if 'POSITION' in attrs:",
    indent + "                    _rot_accessor_pos(int(attrs['POSITION']))",
    indent + "            # zero node translation so Blender shows 0,0,0",
    indent + "            n['translation'] = [0.0,0.0,0.0]",
    indent + "        # zero root translation+rotation so Blender shows 0,0,0",
    indent + "        rnode['translation'] = [0.0,0.0,0.0]",
    indent + "        rnode['rotation'] = [0.0,0.0,0.0,1.0]",
    indent + "        print('[INFO] Baking origin: baked root rotation into geometry; node TRS zeroed')",
    indent + "    except Exception as e:",
    indent + "        print('[WARN] bake-origin failed:', e)",
    ""
])

s = s[:m.start()] + blk + s[m.end():]
open(p, "w", encoding="utf-8", newline="").write(s)
print("Patched bake-origin block to rotation-only bake.")
