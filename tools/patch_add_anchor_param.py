from __future__ import annotations
from pathlib import Path

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8")

def add_anchor_to_def(def_name: str, multiline: bool) -> None:
    global s
    key = f"def {def_name}("
    i = s.find(key)
    if i < 0:
        raise SystemExit(f"[FAIL] Could not find {key}")
    j = s.find(") ->", i)
    if j < 0:
        raise SystemExit(f"[FAIL] Could not find end of signature for {def_name} (') ->')")
    sig = s[i:j]
    if "anchor" in sig:
        print(f"[OK] {def_name} already has anchor")
        return
    ins = ""
    if multiline or ("\n" in sig):
        ins = ",\n    anchor: Optional[str] = None"
    else:
        ins = ", anchor: Optional[str] = None"
    s = s[:j] + ins + s[j:]
    print(f"[OK] Added anchor param to {def_name} signature")

# Ensure Optional is imported
if "Optional" not in s.splitlines()[0:80].__str__():
    # (Usually already present; we won't force-edit imports unless missing)
    pass

add_anchor_to_def("write_gltf_scene", multiline=False)
add_anchor_to_def("_gltf_pack_scene", multiline=True)

p.write_text(s, encoding="utf-8")
print("[DONE] Patched signatures.")
