from __future__ import annotations
from pathlib import Path
import re, sys

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8", errors="replace")
orig = s

def die(msg): raise SystemExit(msg)

# 1) RAW decode: replace nx/ny/nz normalization block if present
if "nx = rx / S16_MAX" in s:
    s = s.replace(
        "                    nx = rx / S16_MAX\n"
        "                    ny = ry / S16_MAX\n"
        "                    nz = rz / S16_MAX\n"
        "                    vx = center_rf[0] + mult_rf[0] * nx\n"
        "                    vy = center_rf[1] + mult_rf[1] * ny\n"
        "                    vz = center_rf[2] + mult_rf[2] * nz\n",
        "                    # RAW decode verified by bounding_radius (center + mult * s16)\n"
        "                    vx = center_rf[0] + mult_rf[0] * float(rx)\n"
        "                    vy = center_rf[1] + mult_rf[1] * float(ry)\n"
        "                    vz = center_rf[2] + mult_rf[2] * float(rz)\n",
    )

# 2) Add debug print when base_scale is captured (so we can see flagpole's actual scale)
# Find the line "base_scale = s" inside parse_mesh_section
if "base_scale = s" in s and "[DEBUG_XFORM]" not in s:
    s = s.replace(
        "                base_scale = s\n",
        "                base_scale = s\n"
        "                if DEBUG_FRAMES:\n"
        "                    print(f\"[DEBUG_XFORM] mesh='{name}' is_keyframed={is_keyframed} base_scale={base_scale}\")\n",
    )

# 3) Bake base_scale into geometry for NON-morph meshes (prevents scaling children in Blender)
# Insert just before the bail check "if not verts or not uvs_per_corner:"
bail = "    # If we didn't get what we need, bail\n"
if bail not in s:
    die("Couldn't find bail marker to insert base_scale bake block.")

if "Bake base_scale into geometry" not in s:
    insert = (
        "    # Bake base_scale into geometry (non-morph meshes) so children don't inherit scale in Blender\n"
        "    # NOTE: base_scale was read through read_vec3_to_blender(), so abs() yields proper axis-mapped scale.\n"
        "    if (not morph) and (base_scale is not None):\n"
        "        sx = abs(base_scale[0] if isinstance(base_scale, (list, tuple)) else getattr(base_scale,'x',1.0))\n"
        "        sy = abs(base_scale[1] if isinstance(base_scale, (list, tuple)) else getattr(base_scale,'y',1.0))\n"
        "        sz = abs(base_scale[2] if isinstance(base_scale, (list, tuple)) else getattr(base_scale,'z',1.0))\n"
        "        if (sx,sy,sz) != (1.0,1.0,1.0):\n"
        "            verts = [(vx*sx, vy*sy, vz*sz) for (vx,vy,vz) in verts]\n"
        "            base_verts = verts\n"
        "\n"
    )
    s = s.replace(bail, insert + bail)

if s == orig:
    print("[INFO] No changes made (already patched?).")
else:
    p.write_text(s, encoding="utf-8")
    print("[OK] Patched vfx2obj.py: RAW decode + bake base_scale into geometry + DEBUG_XFORM print.")
