import json, os, sys

def lname(n): return (n.get("name","") or "").strip().lower()
def v3(v): return [float(v[0]), float(v[1]), float(v[2])] if v else [0.0,0.0,0.0]
def sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def main(path):
    g = json.load(open(path, "r", encoding="utf-8"))
    nodes = g.get("nodes", [])
    if not nodes: raise SystemExit("[FAIL] no nodes")

    # parent map
    parent = {}
    for pi,n in enumerate(nodes):
        for ci in n.get("children",[]) or []:
            parent[int(ci)] = pi

    # find indices
    idx_root = next((i for i,n in enumerate(nodes) if n.get("name")=="__VFX_ROOT__"), None)
    if idx_root is None:
        idx_root = int(g.get("scenes",[{}])[0].get("nodes",[0])[0])

    idx_pole = next((i for i,n in enumerate(nodes) if lname(n)=="flagpole"), None)
    idx_flag = next((i for i,n in enumerate(nodes) if lname(n)=="flagmesh"), None)
    if idx_pole is None: raise SystemExit("[FAIL] no flagpole node")
    if idx_flag is None: raise SystemExit("[FAIL] no FlagMesh node")

    # compute pole world translation by summing translations up the parent chain (ignoring rotation; your root is identity)
    pole_world = [0.0,0.0,0.0]
    cur = idx_pole
    chain = []
    while True:
        chain.append(cur)
        t = v3(nodes[cur].get("translation"))
        pole_world = [pole_world[0]+t[0], pole_world[1]+t[1], pole_world[2]+t[2]]
        if cur == idx_root or cur not in parent:
            break
        cur = parent[cur]

    print("[INFO] pole parent chain:", " -> ".join(nodes[i].get("name","?") for i in chain))
    print("[INFO] pole_world_translation:", pole_world)

    # 1) Anchor by shifting ONLY root children (root stays at 0,0,0)
    root_children = nodes[idx_root].get("children", []) or []
    for ci in root_children:
        c = int(ci)
        t = v3(nodes[c].get("translation"))
        nodes[c]["translation"] = sub(t, pole_world)

    # Ensure root is clean
    nodes[idx_root]["translation"] = [0.0,0.0,0.0]

    # 2) Retarget all weights animations to FlagMesh and strip TRS on pole
    anims = g.get("animations", []) or []
    for anim in anims:
        new_ch = []
        for ch in anim.get("channels", []) or []:
            tgt = ch.get("target", {})
            path2 = tgt.get("path", "")
            node_i = tgt.get("node", None)

            if path2 == "weights":
                tgt["node"] = idx_flag
                new_ch.append(ch)
                continue

            if node_i == idx_pole and path2 in ("translation","rotation","scale"):
                continue

            # keep other channels (if any) untouched
            new_ch.append(ch)

        anim["channels"] = new_ch

    # drop empty animations
    g["animations"] = [a for a in anims if (a.get("channels") or [])]

    # write new pair
    out_gltf = os.path.splitext(path)[0] + "_fixed2.gltf"
    out_bin  = os.path.splitext(path)[0] + "_fixed2.bin"

    # copy bin
    bin_uri = g["buffers"][0]["uri"]
    in_bin = os.path.join(os.path.dirname(path), bin_uri)
    blob = open(in_bin,"rb").read()
    open(out_bin,"wb").write(blob)

    g["buffers"][0]["uri"] = os.path.basename(out_bin)
    json.dump(g, open(out_gltf,"w",encoding="utf-8",newline=""), indent=2)

    print("[OK] wrote:", out_gltf)
    print("[OK] wrote:", out_bin)

if __name__=="__main__":
    if len(sys.argv)<2: raise SystemExit("Usage: python fix_flag_anchor_root_children.py file.gltf")
    main(sys.argv[1])
