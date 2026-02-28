from __future__ import annotations
from pathlib import Path
import re, sys

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8", errors="replace")
orig = s
changed = False

def must_find(substr: str):
    if substr not in s:
        raise SystemExit(f"Expected text not found: {substr!r}")

# 1) Keep RAW decode (remove any lingering /S16_MAX)
s2 = s.replace("rx / S16_MAX", "float(rx)").replace("ry / S16_MAX", "float(ry)").replace("rz / S16_MAX", "float(rz)")
if s2 != s:
    s = s2
    changed = True

# 2) Add holders for keyframed pivot/scale key near base_scale = None
m = re.search(r"(?m)^(\s*)base_scale\s*=\s*None\s*$", s)
if not m:
    raise SystemExit("Couldn't find 'base_scale = None' in parse_mesh_section().")
indent = m.group(1)
if "pivot_pos = None" not in s:
    s = s[:m.end()] + "\n" + indent + "pivot_pos = None\n" + indent + "pivot_scale = None\n" + indent + "scale_key0 = None\n" + s[m.end():]
    changed = True

# 3) Capture keyframed pivot scale (replace the exact skip block)
must_find("# keyframed pivot + keyframes (skip)")
pivot_pat = re.compile(
    r"(?s)#\s*keyframed pivot\s*\+\s*keyframes\s*\(skip\)\s*\n"
    r"(\s*)if\s+is_keyframed\s+and\s+version\s*>=\s*0x3000A\s*:\s*\n"
    r"\1\s*_\s*=\s*read_vec3_to_blender\(b\)\s*\n"
    r"\1\s*_\s*=\s*read_quat\(b\)\s*\n"
    r"\1\s*_\s*=\s*read_vec3_to_blender\(b\)\s*\n"
)
m = pivot_pat.search(s)
if m:
    ind = m.group(1)
    repl = (
        "# keyframed pivot + keyframes (capture)\n"
        f"{ind}if is_keyframed and version >= 0x3000A:\n"
        f"{ind}    pivot_pos = read_vec3_to_blender(b)\n"
        f"{ind}    _ = read_quat(b)  # pivot rotation (unused)\n"
        f"{ind}    pivot_scale = read_vec3_to_blender(b)\n"
    )
    s = s[:m.start()] + repl + s[m.end():]
    changed = True

# 4) Capture first scale key (replace the exact scale keys skip)
must_find("# scale keys")
scale_pat = re.compile(
    r"(?s)(\s*)#\s*scale keys\s*\n\1\s*n\s*=\s*b\.s32\(\)\s*\n\1\s*b\.read\(\s*n\s*\*\s*\(4\s*\+\s*12\s*\+\s*12\s*\+\s*12\)\s*\)\s*"
)
m = scale_pat.search(s)
if m:
    ind = m.group(1)
    repl = (
        f"{ind}# scale keys (capture first)\n"
        f"{ind}n = b.s32()\n"
        f"{ind}if n > 0:\n"
        f"{ind}    _ = b.s32()  # time\n"
        f"{ind}    scale_key0 = read_vec3_to_blender(b)\n"
        f"{ind}    b.read(12 + 12)  # inTan + outTan\n"
        f"{ind}    if n > 1:\n"
        f"{ind}        b.read((n - 1) * (4 + 12 + 12 + 12))\n"
    )
    s = s[:m.start()] + repl + s[m.end():]
    changed = True

# 5) Insert geometry-only bake before materials_used line (stable anchor)
must_find("materials_used = sorted")
if "[DEBUG_SCALE]" not in s:
    bake_pat = re.compile(r"(?m)^(\s*)materials_used\s*=\s*sorted\(.*\)\s*$")
    m = bake_pat.search(s)
    if not m:
        raise SystemExit("Couldn't find materials_used line to insert bake block.")
    ind = m.group(1)
    block = (
        f"{ind}# --- geometry-only scale bake for NON-morph meshes (prevents child inheritance in Blender) ---\n"
        f"{ind}geom_sx = geom_sy = geom_sz = 1.0\n"
        f"{ind}px = py = pz = 0.0\n"
        f"{ind}if not morph:\n"
        f"{ind}    if is_keyframed:\n"
        f"{ind}        if pivot_pos is not None:\n"
        f"{ind}            px, py, pz = pivot_pos\n"
        f"{ind}        if pivot_scale is not None:\n"
        f"{ind}            geom_sx *= abs(float(pivot_scale[0])); geom_sy *= abs(float(pivot_scale[1])); geom_sz *= abs(float(pivot_scale[2]))\n"
        f"{ind}        if scale_key0 is not None:\n"
        f"{ind}            geom_sx *= abs(float(scale_key0[0])); geom_sy *= abs(float(scale_key0[1])); geom_sz *= abs(float(scale_key0[2]))\n"
        f"{ind}    else:\n"
        f"{ind}        if base_scale is not None:\n"
        f"{ind}            geom_sx *= abs(float(base_scale[0])); geom_sy *= abs(float(base_scale[1])); geom_sz *= abs(float(base_scale[2]))\n"
        f"{ind}\n"
        f"{ind}if DEBUG_FRAMES:\n"
        f"{ind}    print(f\"[DEBUG_SCALE] mesh='{name}' is_keyframed={is_keyframed} morph={morph} base_scale={base_scale} pivot_pos={pivot_pos} pivot_scale={pivot_scale} scale_key0={scale_key0} geom_scale=({geom_sx},{geom_sy},{geom_sz})\")\n"
        f"{ind}\n"
        f"{ind}if (not morph) and (geom_sx, geom_sy, geom_sz) != (1.0, 1.0, 1.0):\n"
        f"{ind}    out_verts = [(px + (vx - px) * geom_sx, py + (vy - py) * geom_sy, pz + (vz - pz) * geom_sz) for (vx,vy,vz) in out_verts]\n"
        f"{ind}    # Clear node-scale so children don't inherit anything in Blender\n"
        f"{ind}    base_scale = None\n\n"
    )
    s = s[:m.start()] + block + s[m.start():]
    changed = True

if not changed:
    print("[INFO] No changes made (already patched).")
else:
    p.write_text(s, encoding="utf-8")
    print("[OK] Patched vfx2obj.py: capture pivot/scale key + bake non-morph scale into geometry + DEBUG_SCALE.")
