import bpy, sys, os

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if len(argv) < 2:
    print("Usage: blender --background --factory-startup --python gltf_to_glb.py -- <in.gltf> <out.glb>")
    sys.exit(2)

in_path = os.path.abspath(argv[0])
out_glb = os.path.abspath(argv[1])
os.makedirs(os.path.dirname(out_glb), exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import glTF
bpy.ops.import_scene.gltf(filepath=in_path)

# Export GLB (single file)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format='GLB',
    export_yup=True
)

print("Wrote:", out_glb)
