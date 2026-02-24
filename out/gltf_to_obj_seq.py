import bpy, sys, os

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if len(argv) < 6:
    print("Usage: blender --background --factory-startup --python gltf_to_obj_seq.py -- <in.gltf> <out_dir> <start> <end> <step> <basename>")
    sys.exit(2)

in_path  = os.path.abspath(argv[0])
out_dir  = os.path.abspath(argv[1])
start_f  = int(argv[2])
end_f    = int(argv[3])
step     = int(argv[4])
basename = argv[5]

os.makedirs(out_dir, exist_ok=True)

# Clean empty scene (factory-startup also prevents user addons from exploding in background mode)
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import glTF
bpy.ops.import_scene.gltf(filepath=in_path)

# Select mesh objects only
for o in bpy.data.objects:
    o.select_set(False)

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
for o in mesh_objs:
    o.select_set(True)

if not mesh_objs:
    print("[WARN] No mesh objects found in:", in_path)
    sys.exit(0)

scene = bpy.context.scene

def export_obj(path):
    # Blender 4.x exporter
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True)
        return
    # Older exporter (requires addon)
    if hasattr(bpy.ops.export_scene, "obj"):
        try:
            bpy.ops.preferences.addon_enable(module="io_scene_obj")
        except Exception:
            pass
        bpy.ops.export_scene.obj(filepath=path, use_selection=True, use_materials=True)
        return
    raise RuntimeError("No OBJ exporter found (neither wm.obj_export nor export_scene.obj).")

# Export one OBJ per sampled frame
count = 0
for f in range(start_f, end_f + 1, max(step,1)):
    scene.frame_set(f)
    out_path = os.path.join(out_dir, f"{basename}_f{f:04d}.obj")
    export_obj(out_path)
    count += 1

print(f"Exported {count} OBJ files to: {out_dir}")
