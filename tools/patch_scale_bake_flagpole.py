from __future__ import annotations
from pathlib import Path
import re, sys

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8", errors="replace")
orig = s

def subn(pattern, repl, s, flags=0):
    new_s, n = re.subn(pattern, repl, s, flags=flags)
    return new_s, n

changed = False

# 1) Ensure RAW decode for morph/non-morph (mult is already scaled; rx is raw s16)
s2, n = subn(r"\bnx\s*=\s*rx\s*/\s*S16_MAX\b", "nx = float(rx)", s)
s, changed = (s2, True) if n else (s, changed)
s2, n = subn(r"\bny\s*=\s*ry\s*/\s*S16_MAX\b", "ny = float(ry)", s)
s, changed = (s2, True) if n else (s, changed)
s2, n = subn(r"\bnz\s*=\s*rz\s*/\s*S16_MAX\b", "nz = float(rz)", s)
s, changed = (s2, True) if n else (s, changed)

# 2) Add pivot_scale + scale_key0 vars near base_scale
if "pivot_scale = None" not in s:
    s2, n = subn(r"(?m)^(\s*)base_scale\s*=\s*None\s*$",
                 r"\1base_scale = None\n\1pivot_scale = None\n\1scale_key0 = None",
                 s)
    if n:
        s, changed = s2, True

# 3) Capture keyframed pivot scale (was previously skipped)
pivot_pat = re.compile(
    r"(?s)(#\s*keyframed pivot\s*\+\s*keyframes\s*\(skip\)\s*\n\s*if\s+is_keyframed\s+and\s+version\s*>=\s*0x3000A\s*:\s*\n)"
    r"(\s*)_\s*=\s*read_vec3_to_blender\(b\)\s*\n"
    r"\2_\s*=\s*read_quat\(b\)\s*\n"
    r"\2_\s*=\s*read_vec3_to_blender\(b\)\s*\n"
)
m = pivot_pat.search(s)
if m:
    head, ind = m.group(1), m.group(2)
    repl = head + (
        f"{ind}pivot_pos = read_vec3_to_blender(b)\n"
        f"{ind}pivot_rot = read_quat(b)\n"
        f"{ind}pivot_scale = read_vec3_to_blender(b)\n"
    )
    s = s[:m.start()] + repl + s[m.end():]
    changed = True

# 4) Capture first scale key (was previously skipped with b.read(...))
scale_pat = re.compile(
    r"(?s)(#\s*scale keys\s*\n\s*n\s*=\s*b\.s32\(\)\s*\n)(\s*)b\.read\(\s*n\s*\*\s*\(4\s*\+\s*12\s*\+\s*12\s*\+\s*12\)\s*\)\s*\n"
)
m = scale_pat.search(s)
if m:
    head, ind = m.group(1), m.group(2)
    repl = head + f"""{ind}if n > 0:
{ind}    _t = b.f32()  # time (unused)
{ind}    _s = read_vec3_to_blender(b)
{ind}    b.read(12 + 12)  # inTan + outTan
{ind}    if scale_key0 is None:
{ind}        scale_key0 = _s
{ind}    if n > 1:
{ind}        b.read((n - 1) * (4 + 12 + 12 + 12))
{ind}else:
{ind}    pass
"""
    s = s[:m.start()] + repl + s[m.end():]
    changed = True

# 5) Bake mesh-local scale into NON-morph geometry and clear node-scale (prevents Blender child inheritance)
if "mesh-local scale bake" not in s:
    # best insertion point: right before "Build OBJ-friendly indexing"
    ins = re.search(r"(?m)^(\s*)#\s*Build OBJ-friendly indexing.*$", s)
    if ins:
        ind = ins.group(1)
        block = f"""{ind}# --- mesh-local scale bake (prevents parent scale inheritance in Blender) ---
{ind}geom_scale = None
{ind}if base_scale is not None:
{ind}    geom_scale = base_scale
{ind}elif (pivot_scale is not None) and (scale_key0 is not None):
{ind}    geom_scale = (pivot_scale[0] * scale_key0[0], pivot_scale[1] * scale_key0[1], pivot_scale[2] * scale_key0[2])
{ind}elif pivot_scale is not None:
{ind}    geom_scale = pivot_scale
{ind}elif scale_key0 is not None:
{ind}    geom_scale = scale_key0
{ind}
{ind}if (not morph) and (geom_scale is not None):
{ind}    sx, sy, sz = geom_scale
{ind}    sx = abs(float(sx)); sy = abs(float(sy)); sz = abs(float(sz))
{ind}    if (sx, sy, sz) != (1.0, 1.0, 1.0):
{ind}        verts = [(vx * sx, vy * sy, vz * sz) for (vx, vy, vz) in verts]
{ind}        base_verts = verts
{ind}        # IMPORTANT: do NOT export node-scale for non-morph meshes; we baked it
{ind}        base_scale = None
{ind}        if DEBUG_FRAMES:
{ind}            print(f"[DEBUG_SCALE] mesh='{{name}}' baked geom_scale=({{sx:.6g}},{{sy:.6g}},{{sz:.6g}})")
{ind}
"""
        s = s[:ins.start()] + block + s[ins.start():]
        changed = True

if not changed:
    print("[INFO] No changes needed (already patched).")
else:
    p.write_text(s, encoding="utf-8")
    print("[OK] Patched vfx2obj.py (capture pivot/scale keys + bake non-morph scale into verts).")
