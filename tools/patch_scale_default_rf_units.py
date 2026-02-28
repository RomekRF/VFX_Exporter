from __future__ import annotations
from pathlib import Path
import re, sys

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8")

changed = False

# 1) Default scale: 10000.0 -> 1.0
s2 = re.sub(r"(?m)^(\s*)scale\s*=\s*10000\.0\s*$", r"\1scale = 1.0", s)
if s2 != s:
    s = s2
    changed = True
    print("[OK] default scale set to 1.0")
else:
    # If it was already not 10000, still report
    m = re.search(r"(?m)^\s*scale\s*=\s*([0-9.]+)\s*$", s)
    if m:
        print("[INFO] default scale already:", m.group(1))
    else:
        print("[WARN] could not locate default scale assignment")

# 2) Raw decode in the mesh frame loop (remove /32767 normalization)
# Replace the specific normalized-decode block if present.
pat = r"""(?ms)
^\s*#\s*decompress.*?\n
\s*for\s*\(rx,\s*ry,\s*rz\)\s*in\s*raw:\s*\n
\s*nx\s*=\s*rx\s*/\s*S16_MAX\s*\n
\s*ny\s*=\s*ry\s*/\s*S16_MAX\s*\n
\s*nz\s*=\s*rz\s*/\s*S16_MAX\s*\n
\s*vx\s*=\s*center_rf\[0\]\s*\+\s*mult_rf\[0\]\s*\*\s*nx\s*\n
\s*vy\s*=\s*center_rf\[1\]\s*\+\s*mult_rf\[1\]\s*\*\s*ny\s*\n
\s*vz\s*=\s*center_rf\[2\]\s*\+\s*mult_rf\[2\]\s*\*\s*nz\s*\n
\s*verts\.append\(rf_to_blender\(\(vx,\s*vy,\s*vz\)\)\)\s*
"""
m = re.search(pat, s)
if m:
    rep = """# decompress (RAW): verified by bounding_radius check -> center + mult * raw_s16
            for (rx, ry, rz) in raw:
                vx = center_rf[0] + mult_rf[0] * float(rx)
                vy = center_rf[1] + mult_rf[1] * float(ry)
                vz = center_rf[2] + mult_rf[2] * float(rz)
                verts.append(rf_to_blender((vx, vy, vz)))"""
    s = s[:m.start()] + rep + s[m.end():]
    changed = True
    print("[OK] switched position decode to RAW (no /32767)")
else:
    # Fallback: if someone already edited, just remove any remaining /S16_MAX lines
    if "/ S16_MAX" in s or "/S16_MAX" in s:
        s = s.replace("rx / S16_MAX", "float(rx)")
        s = s.replace("ry / S16_MAX", "float(ry)")
        s = s.replace("rz / S16_MAX", "float(rz)")
        changed = True
        print("[OK] removed remaining /S16_MAX normalizations (fallback)")

if not changed:
    print("[INFO] No changes needed.")
else:
    p.write_text(s, encoding="utf-8")
    print("[DONE] wrote patched vfx2obj.py")
