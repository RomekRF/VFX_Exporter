import importlib.util, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VFX2OBJ = os.path.join(ROOT, "vfx2obj.py")

def load_vfx2obj():
    spec = importlib.util.spec_from_file_location("vfx2obj_mod", VFX2OBJ)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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
    out=[]
    for (ix,iy,iz) in fr.pos_s16:
        sx = cx + mx*ix
        sy = cy + my*iy
        sz = cz + mz*iz
        out.append(mod.rf_to_blender((sx,sy,sz)))
    return out

def bbox_ctr(points):
    xs=[p[0] for p in points]; ys=[p[1] for p in points]; zs=[p[2] for p in points]
    mn=(min(xs),min(ys),min(zs)); mx=(max(xs),max(ys),max(zs))
    return ((mn[0]+mx[0])*0.5,(mn[1]+mx[1])*0.5,(mn[2]+mx[2])*0.5)

def fmt3(v): return f"({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f})"

def report(vfx_path):
    mod=load_vfx2obj()
    hdr,mats,meshes,dums = mod.parse_vfx(vfx_path)
    by={m.name:m for m in meshes}
    fm=by.get("FlagMesh"); fp=by.get("flagpole")
    print("\n" + os.path.basename(vfx_path))
    for name,m in [("flagpole",fp),("FlagMesh",fm)]:
        if not m:
            print(f"  {name}: MISSING")
            continue
        pts=decode_frame0_positions_bl(m,mod)
        if pts:
            c=bbox_ctr(pts)
            print(f"  {name} bbox ctr: {fmt3(c)}")
        else:
            print(f"  {name} bbox ctr: (no verts decoded)")

if __name__=="__main__":
    for p in sys.argv[1:]:
        report(p)
