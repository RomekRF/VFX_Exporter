import importlib.util, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VFX2OBJ = os.path.join(ROOT, "vfx2obj.py")

def load_vfx2obj():
    # Python 3.13 dataclasses expects cls.__module__ to exist in sys.modules.
    spec = importlib.util.spec_from_file_location("vfx2obj_mod", VFX2OBJ)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # <-- critical fix
    spec.loader.exec_module(mod)
    return mod

def decode_frame0_positions_bl(mesh, mod):
    frames = getattr(mesh, "frames", None)
    if not frames: return []
    fr = frames[0]
    if getattr(fr, "center_rf", None) is None or getattr(fr, "mult_rf", None) is None or getattr(fr, "pos_s16", None) is None:
        return []
    cx,cy,cz = fr.center_rf
    mx,my,mz = fr.mult_rf
    out = []
    for (ix,iy,iz) in fr.pos_s16:
        sx = cx + mx * ix
        sy = cy + my * iy
        sz = cz + mz * iz
        out.append(mod.rf_to_blender((sx,sy,sz)))
    return out

def bbox(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    ctr = ((mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5)
    return mn, mx, ctr

def fmt3(v): return f"({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f})"

def mesh_report(vfx_path, want=("flagpole","FlagMesh")):
    mod = load_vfx2obj()
    hdr, mats, meshes, dums = mod.parse_vfx(vfx_path)
    byname = {m.name: m for m in meshes}

    print("\n" + "="*88)
    print(os.path.basename(vfx_path))

    for name in want:
        m = byname.get(name)
        if not m:
            print(f"\n[{name}] MISSING")
            continue

        print(f"\n[{name}] morph={m.morph} keyframed={m.is_keyframed} frames={m.num_frames} v={m.num_vertices} f={m.num_faces}")

        kf = getattr(m, "keyframes", None)
        if kf and getattr(kf, "t_values", None):
            t_bl = kf.t_values[0]
            t_rf = mod.blender_to_rf(t_bl)
            print("  t_key bl :", fmt3(t_bl))
            print("  t_key rf :", fmt3(t_rf))
        else:
            print("  t_key    : (none)")

        pts = decode_frame0_positions_bl(m, mod)
        if pts:
            mn, mx, ctr = bbox(pts)
            print("  bbox ctr :", fmt3(ctr))
            print("  bbox min :", fmt3(mn))
            print("  bbox max :", fmt3(mx))
        else:
            print("  bbox     : (no decoded verts)")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        mesh_report(p)
