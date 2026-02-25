import sys, re

p = sys.argv[1]
s = open(p, "r", encoding="utf-8").read()

def ensure_var(name, anchor_pat, insert_line):
    global s
    if re.search(rf"(?m)^\s*{re.escape(name)}\s*=", s):
        return
    m = re.search(anchor_pat, s)
    if not m:
        raise SystemExit(f"Could not find anchor for {name}")
    # keep same indent as anchor line
    indent = re.match(r"^\s*", m.group(0)).group(0)
    ins = "\n" + indent + insert_line + "\n"
    s = s[:m.end()] + ins + s[m.end():]

def ensure_handler(flag, set_line):
    # Insert handler in the SAME args loop scope as the unknown-option warning.
    global s
    # already have real handler?
    if re.search(rf"if\s+a\s+in\s+\(\s*'{re.escape(flag)}'", s):
        return
    # anchor at the unknown-option warning print (f-string version)
    pat = r"(?m)^(?P<ind>\s*)print\(f'\[WARN\]\s+Ignoring\s+unknown\s+option:\s+\{a\}'\)\s*$"
    m = re.search(pat, s)
    if not m:
        raise SystemExit("Could not find unknown-option warning print to anchor flag handler insertion.")
    ind = m.group("ind")
    blk = "\n".join([
        ind + f"if a in ('{flag}',):",
        ind + f"    {set_line}",
        ind + "    del args[i:i+1]",
        ind + "    continue",
        ""
    ])
    s = s[:m.start()] + blk + "\n" + s[m.start():]

def add_param_to_def(defname, paramfrag):
    global s
    # only patch single-line defs
    pat = rf"(?m)^def\s+{re.escape(defname)}\((?P<args>[^\)]*)\)\s*->"
    m = re.search(pat, s)
    if not m:
        raise SystemExit(f"Could not find def {defname}(...)")
    args = m.group("args")
    if "bake_origin" in args:
        return
    new_args = args.rstrip() + ", " + paramfrag
    s = s[:m.start("args")] + new_args + s[m.end("args"):]

def add_kwarg_to_call(call_pat, kwfrag):
    global s
    m = re.search(call_pat, s)
    if not m:
        raise SystemExit("Could not find call site to add kwarg.")
    line = m.group(0)
    if "bake_origin=" in line:
        return
    # add before closing ')'
    line2 = re.sub(r"\)\s*$", f", {kwfrag})", line)
    s = s.replace(line, line2, 1)

# --- 1) Ensure bake_origin variable exists near center_root ---
ensure_var("bake_origin",
           r"(?m)^\s*center_root\s*=\s*(True|False).*$",
           "bake_origin = False  # --bake-origin: bake root+mesh transforms into vertices so Blender shows 0,0,0")

# --- 2) Ensure --bake-origin flag parsing (guaranteed in the real args loop) ---
ensure_handler("--bake-origin", "bake_origin = True")

# --- 3) Thread bake_origin through function signatures/calls ---
add_param_to_def("_gltf_pack_scene", "bake_origin: bool = False")
add_param_to_def("write_gltf_scene", "bake_origin: bool = False")
add_param_to_def("convert_file", "bake_origin: bool = False")

# write_gltf_scene -> _gltf_pack_scene call line
add_kwarg_to_call(r"(?m)^\s*out_gltf,\s*out_bin\s*=\s*_gltf_pack_scene\([^\n]+\)\s*$",
                  "bake_origin=bake_origin")

# convert_file -> write_gltf_scene call line (the one with center_root already)
add_kwarg_to_call(r"(?m)^\s*write_gltf_scene\([^\n]+center_root=center_root[^\n]*\)\s*$",
                  "bake_origin=bake_origin")

# main -> convert_file call line (the one with center_root already)
add_kwarg_to_call(r"(?m)^\s*convert_file\([^\n]+center_root=center_root[^\n]*\)\s*$",
                  "bake_origin=bake_origin")

# --- 4) Inject bake step inside _gltf_pack_scene right before writing out_bin ---
# Find "out_bin = os.path.join(...)" inside _gltf_pack_scene and insert block above it.
outbin_pat = r"(?m)^(?P<ind>\s*)out_bin\s*=\s*os\.path\.join\(out_dir,\s*base_name\s*\+\s*\"\.bin\"\)\s*$"
m = re.search(outbin_pat, s)
if not m:
    raise SystemExit("Could not find out_bin assignment to inject bake-origin block.")
ind = m.group("ind")

# Avoid double insert
if "bake_origin:" not in s or "Baking origin" not in s:
    bake_blk = "\n".join([
        ind + "if bake_origin:",
        ind + "    # Bake root rotation+translation and mesh-node translation into POSITION vertices, then zero node transforms.",
        ind + "    import struct",
        ind + "    def _qmul(a,b):",
        ind + "        ax,ay,az,aw=a; bx,by,bz,bw=b",
        ind + "        return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)",
        ind + "    def _qconj(q):",
        ind + "        x,y,z,w=q; return (-x,-y,-z,w)",
        ind + "    def _qrot(q, v3):",
        ind + "        vx,vy,vz=v3",
        ind + "        p=(vx,vy,vz,0.0)",
        ind + "        x,y,z,_ = _qmul(_qmul(q,p), _qconj(q))",
        ind + "        return (x,y,z)",
        ind + "    try:",
        ind + "        # root node is the only scene root",
        ind + "        rix = int(gltf['scenes'][0]['nodes'][0])",
        ind + "        rnode = gltf['nodes'][rix]",
        ind + "        rt = rnode.get('translation', [0.0,0.0,0.0])",
        ind + "        rq = rnode.get('rotation', [0.0,0.0,0.0,1.0])  # glTF [x,y,z,w]",
        ind + "        # editable bin",
        ind + "        if not isinstance(bin_blob, (bytearray,)):",
        ind + "            bin_blob = bytearray(bin_blob)",
        ind + "        def _apply_to_accessor(acc_i, dx,dy,dz):",
        ind + "            acc = gltf['accessors'][acc_i]",
        ind + "            if acc.get('type') != 'VEC3' or acc.get('componentType') != 5126:",
        ind + "                return",
        ind + "            bv = gltf['bufferViews'][acc['bufferView']]",
        ind + "            base = int(bv.get('byteOffset', 0)) + int(acc.get('byteOffset', 0))",
        ind + "            stride = int(bv.get('byteStride', 12) or 12)",
        ind + "            cnt = int(acc['count'])",
        ind + "            # recompute bounds",
        ind + "            minx=miny=minz=1e30; maxx=maxy=maxz=-1e30",
        ind + "            for ii in range(cnt):",
        ind + "                o = base + ii*stride",
        ind + "                x,y,z = struct.unpack_from('<fff', bin_blob, o)",
        ind + "                x = x + dx; y = y + dy; z = z + dz",
        ind + "                struct.pack_into('<fff', bin_blob, o, x,y,z)",
        ind + "                if x<minx: minx=x",
        ind + "                if y<miny: miny=y",
        ind + "                if z<minz: minz=z",
        ind + "                if x>maxx: maxx=x",
        ind + "                if y>maxy: maxy=y",
        ind + "                if z>maxz: maxz=z",
        ind + "            acc['min'] = [float(minx), float(miny), float(minz)]",
        ind + "            acc['max'] = [float(maxx), float(maxy), float(maxz)]",
        ind + "        # bake each mesh-node translation into geometry in WORLD space",
        ind + "        for n in gltf['nodes']:",
        ind + "            if not isinstance(n, dict) or ('mesh' not in n):",
        ind + "                continue",
        ind + "            mt = n.get('translation', [0.0,0.0,0.0])",
        ind + "            rmt = _qrot(tuple(rq), (float(mt[0]),float(mt[1]),float(mt[2])))",
        ind + "            # world offset to add to vertices so node translations can be zeroed",
        ind + "            dx = float(rt[0]) + float(rmt[0])",
        ind + "            dy = float(rt[1]) + float(rmt[1])",
        ind + "            dz = float(rt[2]) + float(rmt[2])",
        ind + "            mesh = gltf['meshes'][n['mesh']]",
        ind + "            for prim in mesh.get('primitives', []):",
        ind + "                attrs = prim.get('attributes', {})",
        ind + "                if 'POSITION' in attrs:",
        ind + "                    _apply_to_accessor(int(attrs['POSITION']), dx,dy,dz)",
        ind + "            n['translation'] = [0.0,0.0,0.0]",
        ind + "        # zero root",
        ind + "        rnode['translation'] = [0.0,0.0,0.0]",
        ind + "        rnode['rotation'] = [0.0,0.0,0.0,1.0]",
        ind + "        print('[INFO] Baking origin: baked transforms into geometry; node TRS zeroed')",
        ind + "    except Exception as e:",
        ind + "        print('[WARN] bake-origin failed:', e)",
        ""
    ])
    s = s[:m.start()] + bake_blk + "\n" + s[m.start():]

open(p, "w", encoding="utf-8", newline="").write(s)
print("Patched: added --bake-origin.")
