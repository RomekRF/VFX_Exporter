from __future__ import annotations
from pathlib import Path
import re, sys

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8", errors="replace")
orig = s

# A) Ensure RAW decode (if any /S16_MAX left, remove it)
s = s.replace("rx / S16_MAX", "float(rx)")
s = s.replace("ry / S16_MAX", "float(ry)")
s = s.replace("rz / S16_MAX", "float(rz)")

# B) Add DEBUG_XFORM print right after base_scale assignment (handle multiple spacing variants)
if "[DEBUG_XFORM]" not in s:
    s = re.sub(
        r"(?m)^(\s*)base_scale\s*=\s*s\s*$",
        r"\1base_scale = s\n\1if DEBUG_FRAMES:\n\1    print(f\"[DEBUG_XFORM] mesh='{name}' is_keyframed={is_keyframed} base_scale={base_scale}\")",
        s,
        count=1
    )

# C) Bake base_scale into geometry for NON-morph meshes (avoid child inheritance in Blender)
# Insert once near the 'bail' comment (match loosely)
if "Bake base_scale into geometry" not in s:
    m = re.search(r"(?m)^\s*#\s*If\s+we\s+didn.?t\s+get\s+what\s+we\s+need,\s*bail", s)
    if not m:
        raise SystemExit("Could not find bail comment to insert base_scale bake block.")
    insert_at = m.start()
    block = """
    # Bake base_scale into geometry (non-morph meshes) so children don't inherit scale in Blender
    if (not morph) and (base_scale is not None):
        try:
            sx, sy, sz = base_scale  # tuple/list
        except Exception:
            sx, sy, sz = (base_scale.x, base_scale.y, base_scale.z)  # fallback
        sx = abs(float(sx)); sy = abs(float(sy)); sz = abs(float(sz))
        if (sx, sy, sz) != (1.0, 1.0, 1.0):
            verts = [(vx*sx, vy*sy, vz*sz) for (vx, vy, vz) in verts]
            base_verts = verts

"""
    s = s[:insert_at] + block + s[insert_at:]

if s == orig:
    print("[INFO] No changes made (already patched?)")
else:
    p.write_text(s, encoding="utf-8")
    print("[OK] Patched vfx2obj.py: RAW decode + bake base_scale into verts + DEBUG_XFORM print.")
