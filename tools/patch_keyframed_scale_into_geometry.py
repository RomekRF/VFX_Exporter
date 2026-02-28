from __future__ import annotations
from pathlib import Path
import re

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8", errors="replace")
orig = s

# 1) Ensure we have holders for pivot/scale-key capture (insert after base_scale = None)
m = re.search(r'(?m)^(\s*)base_scale\s*=\s*None\s*$', s)
if not m:
    raise SystemExit("Could not find 'base_scale = None' in parse_mesh_section()")

indent = m.group(1)
insert = (
    f"{indent}pivot_scale_vec = None  # keyframed pivot scale (raw vec3, no axis swap)\n"
    f"{indent}scale_key0_vec = None   # first scale key (raw vec3, no axis swap)\n"
)

# Only insert once
if "pivot_scale_vec = None" not in s:
    s = s[:m.end()] + "\n" + insert + s[m.end():]

# 2) Replace the pivot skip block to CAPTURE pivot_scale (do NOT axis swap scale!)
pivot_pat = r"""(?ms)
^(?P<i>\s*)#\s*keyframed\s*pivot\s*\+\s*keyframes\s*\(skip\)\s*\n
(?P=i)if\s+is_keyframed\s+and\s+version\s*>=\s*0x3000A:\s*\n
(?P=i)\s+_\s*=\s*read_vec3_to_blender\(b\)\s*\n
(?P=i)\s+_\s*=\s*read_quat\(b\)\s*\n
(?P=i)\s+_\s*=\s*read_vec3_to_blender\(b\)\s*
"""
m = re.search(pivot_pat, s)
if not m:
    # Some builds may have slightly different comment text, try a looser find
    if "# keyframed pivot" not in s:
        raise SystemExit("Could not locate keyframed pivot block to patch.")
else:
    i = m.group("i")
    repl = (
        f"{i}# keyframed pivot + keyframes (capture pivot scale)\n"
        f"{i}if is_keyframed and version >= 0x3000A:\n"
        f"{i}    _ = read_vec3_to_blender(b)  # pivot_translation\n"
        f"{i}    _ = read_quat(b)             # pivot_rotation\n"
        f"{i}    pivot_scale_vec = read_vec3_rf(b)  # pivot_scale (RAW!)\n"
    )
    s = s[:m.start()] + repl + s[m.end():]

# 3) Patch scale keys block to CAPTURE first scale key value (RAW vec3)
scale_pat = r"""(?ms)
^(?P<i>\s*)#\s*scale\s+keys\s*\n
(?P=i)n\s*=\s*b\.s32\(\)\s*\n
(?P=i)b\.read\(n\s*\*\s*\(4\s*\+\s*12\s*\+\s*12\s*\+\s*12\)\)\s*
"""
m = re.search(scale_pat, s)
if not m:
    raise SystemExit("Could not locate '# scale keys' skip block to patch.")
i = m.group("i")
repl = (
    f"{i}# scale keys (capture first)\n"
    f"{i}n = b.s32()\n"
    f"{i}if n > 0:\n"
    f"{i}    _ = b.s32()                # time (frame*320)\n"
    f"{i}    scale_key0_vec = read_vec3_rf(b)  # value (RAW!)\n"
    f"{i}    b.read(12 + 12)             # inTan + outTan\n"
    f"{i}    if n > 1:\n"
    f"{i}        b.read((n - 1) * (4 + 12 + 12 + 12))\n"
)
s = s[:m.start()] + repl + s[m.end():]

# 4) Insert geometry-only scale application before the bail check
bail_pat = r'(?m)^(?P<i>\s*)#\s*If\s+we\s+didn\'t\s+get\s+what\s+we\s+need,\s*bail\s*$'
m = re.search(bail_pat, s)
if not m:
    raise SystemExit("Could not find bail marker '# If we didn't get what we need, bail'")
i = m.group("i")

geom_block = (
    f"{i}# --- geometry-only scale for keyframed NON-morph meshes ---\n"
    f"{i}# VFX stores keyframed pivot_scale + scale keys; Blender parenting would scale children,\n"
    f"{i}# so we bake scale into THIS mesh's vertices only.\n"
    f"{i}geom_sx = geom_sy = geom_sz = 1.0\n"
    f"{i}if (not morph) and is_keyframed:\n"
    f"{i}    if pivot_scale_vec is not None:\n"
    f"{i}        geom_sx *= float(pivot_scale_vec[0]); geom_sy *= float(pivot_scale_vec[1]); geom_sz *= float(pivot_scale_vec[2])\n"
    f"{i}    if scale_key0_vec is not None:\n"
    f"{i}        geom_sx *= float(scale_key0_vec[0]); geom_sy *= float(scale_key0_vec[1]); geom_sz *= float(scale_key0_vec[2])\n"
    f"{i}if (geom_sx, geom_sy, geom_sz) != (1.0, 1.0, 1.0):\n"
    f"{i}    verts = [(vx*geom_sx, vy*geom_sy, vz*geom_sz) for (vx,vy,vz) in verts]\n"
    f"{i}    if base_verts is not None:\n"
    f"{i}        base_verts = verts\n"
    f"{i}\n"
)

# Insert only once
if "geometry-only scale for keyframed NON-morph meshes" not in s:
    s = s[:m.start()] + geom_block + s[m.start():]

if s == orig:
    raise SystemExit("Patch made no changes (already applied?)")

p.write_text(s, encoding="utf-8")
print("[OK] Patched vfx2obj.py: capture pivot_scale + scale_key0 and bake into non-morph keyframed mesh geometry.")
