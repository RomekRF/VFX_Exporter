from __future__ import annotations
from pathlib import Path
import re

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8")

# 1) Remove normalization by S16_MAX in position decode (rx/ S16_MAX -> rx etc.)
s_new = s
s_new = re.sub(r"\brx\s*/\s*S16_MAX\b", "rx", s_new)
s_new = re.sub(r"\bry\s*/\s*S16_MAX\b", "ry", s_new)
s_new = re.sub(r"\brz\s*/\s*S16_MAX\b", "rz", s_new)

# 2) Update default scale from 10000.0 to 10000/32767 to keep roughly same real-world size after decode fix
# (This avoids the “everything became gigantic” effect after fixing decode.)
default_scale = "0.305185095"  # 10000 / 32767

# Replace argparse default if present
s_new = re.sub(r"(add_argument\(\s*['\"]--scale['\"].*?default\s*=\s*)10000\.0", r"\g<1>"+default_scale, s_new, flags=re.S)

# Replace any local default assignment 'scale = 10000.0' used when parsing args
s_new = re.sub(r"\bscale\s*=\s*10000\.0\b", "scale = "+default_scale, s_new)

# If it didn't change anything, fail loudly
if s_new == s:
    raise SystemExit("[FAIL] Patch made no changes (patterns not found). File layout differs.")

p.write_text(s_new, encoding="utf-8")
print("[OK] Patched decode: removed /S16_MAX and updated default scale to", default_scale)
