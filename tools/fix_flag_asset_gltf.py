import json, os, sys, copy

def lower_name(n):
    return (n.get("name","") or "").strip().lower()

def vec3(v):
    if not v: return [0.0,0.0,0.0]
    return [float(v[0]), float(v[1]), float(v[2])]

def sub3(a,b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def main(gltf_path):
    with open(gltf_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    nodes = g.get("nodes", [])
    if not nodes:
        raise SystemExit("[FAIL] No nodes in glTF")

    # Find indices
    idx_root = None
    for i,n in enumerate(nodes):
        if n.get("name") == "__VFX_ROOT__":
            idx_root = i
            break
    if idx_root is None:
        # fallback to scene root
        idx_root = int(g.get("scenes",[{}])[0].get("nodes",[0])[0])

    idx_pole = None
    idx_flag = None
    for i,n in enumerate(nodes):
        if lower_name(n) == "flagpole":
            idx_pole = i
        if lower_name(n) == "flagmesh":
            idx_flag = i

    if idx_pole is None:
        raise SystemExit("[FAIL] Could not find node named 'flagpole'")
    if idx_flag is None:
        raise SystemExit("[FAIL] Could not find node named 'FlagMesh'")

    # Build parent map and pole subtree set
    parent_of = {}
    children_of = {i: [] for i in range(len(nodes))}
    for pi,n in enumerate(nodes):
        for ci in n.get("children", []) or []:
            parent_of[int(ci)] = pi
            children_of[pi].append(int(ci))

    # Confirm flag is in pole subtree (or warn)
    pole_subtree = set()
    stack = [idx_pole]
    while stack:
        x = stack.pop()
        if x in pole_subtree: continue
        pole_subtree.add(x)
        stack.extend(children_of.get(x, []))

    if idx_flag not in pole_subtree:
        print("[WARN] FlagMesh is not under flagpole subtree; parenting may still be wrong.")

    # 1) Retarget ALL weights channels to FlagMesh node
    anims = g.get("animations", []) or []
    moved = 0
    for anim in anims:
        for ch in anim.get("channels", []) or []:
            tgt = ch.get("target", {})
            if tgt.get("path") == "weights":
                if tgt.get("node") != idx_flag:
                    tgt["node"] = idx_flag
                    moved += 1
    if moved:
        print(f"[INFO] Retargeted {moved} weights channel(s) to FlagMesh node {idx_flag}.")

    # 2) Remove TRS animation channels on flagpole (CTF pole should be static)
    removed = 0
    for anim in anims:
        keep = []
        for ch in anim.get("channels", []) or []:
            tgt = ch.get("target", {})
            node_i = tgt.get("node", None)
            path = tgt.get("path", "")
            if node_i == idx_pole and path in ("translation","rotation","scale"):
                removed += 1
                continue
            keep.append(ch)
        anim["channels"] = keep
    if removed:
        print(f"[INFO] Removed {removed} TRS channel(s) from flagpole node {idx_pole}.")

    # 3) Anchor: shift so flagpole is at origin WITHOUT double-shifting children.
    # We do this by subtracting pole's translation from:
    # - flagpole itself
    # - all nodes NOT in the pole subtree
    # Children of pole are NOT changed (they inherit the moved parent).
    pole_t = vec3(nodes[idx_pole].get("translation"))
    if pole_t != [0.0,0.0,0.0]:
        for i,n in enumerate(nodes):
            t = vec3(n.get("translation"))
            if i == idx_pole or (i not in pole_subtree):
                n["translation"] = sub3(t, pole_t)
        print(f"[INFO] Anchored to flagpole: subtracted {pole_t} from pole + non-pole-subtree nodes.")

    # write new gltf (same bin)
    out_gltf = os.path.splitext(gltf_path)[0] + "_fixed.gltf"
    out_bin  = os.path.splitext(gltf_path)[0] + "_fixed.bin"

    # copy bin so the fixed file pair is self-contained
    in_bin_path = os.path.join(os.path.dirname(gltf_path), g["buffers"][0]["uri"])
    with open(in_bin_path, "rb") as f:
        bin_blob = f.read()
    with open(out_bin, "wb") as f:
        f.write(bin_blob)

    # point gltf to new bin name
    g["buffers"][0]["uri"] = os.path.basename(out_bin)

    with open(out_gltf, "w", encoding="utf-8", newline="") as f:
        json.dump(g, f, indent=2)

    print("[OK] Wrote:", out_gltf)
    print("[OK] Wrote:", out_bin)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python fix_flag_asset_gltf.py <file.gltf>")
    main(sys.argv[1])
