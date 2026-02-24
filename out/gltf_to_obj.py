import bpy, sys, os

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if len(argv) < 2:
    print("Usage: blender --background --factory-startup --python gltf_to_obj.py -- <in.gltf> <out.obj>")
    sys.exit(2)

in_path = os.path.abspath(argv[0])
out_obj = os.path.abspath(argv[1])
os.makedirs(os.path.dirname(out_obj), exist_ok=True)

# Fresh empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import glTF
bpy.ops.import_scene.gltf(filepath=in_path)

# Select meshes
for o in bpy.data.objects:
    o.select_set(False)

mesh_count = 0
for o in bpy.data.objects:
    if o.type == "MESH":
        o.select_set(True)
        mesh_count += 1

if mesh_count == 0:
    print("[WARN] No mesh objects found in:", in_path)
    sys.exit(0)

# Blender 4.x OBJ exporter
if hasattr(bpy.ops.wm, "obj_export"):
    bpy.ops.wm.obj_export(filepath=out_obj, export_selected_objects=True)
else:
    # Fallback for older Blenders (rare now)
    bpy.ops.export_scene.obj(filepath=out_obj, use_selection=True, use_materials=True)

print("Wrote:", out_obj)
