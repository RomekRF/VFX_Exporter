from __future__ import annotations
from pathlib import Path
import re, sys

p = Path("vfx2obj.py")
s = p.read_text(encoding="utf-8", errors="replace")
orig = s

def must(containing: str):
    if containing not in s:
        raise SystemExit(f"Expected snippet not found: {containing!r}")

# 0) Make sure key helpers exist
must("def read_vec3_rf")
must("is_keyframed = (b.u8() != 0)")
must("# keyframed pivot + keyframes (skip)")
must("# scale keys")

# 1) Ensure we have pivot_scale_rf + scale_key0_rf in scope (after base_scale)
marker = "base_scale = None"
if "pivot_scale_rf" not in s:
    s = s.replace(
        marker,
        marker + "\n    pivot_scale_rf = None  # keyframed pivot scale (RAW RF vec3)\n    scale_key0_rf = None   # first scale key value (RAW RF vec3)\n"
    )

# 2) Patch keyframed pivot block: capture pivot_scale as RAW (NOT read_vec3_to_blender)
old_pivot = """# keyframed pivot + keyframes (skip)
    if is_keyframed and version >= 0x3000A:
        _ = read_vec3_to_blender(b)
        _ = read_quat(b)
        _ = read_vec3_to_blender(b)
"""
if old_pivot in s:
    new_pivot = """# keyframed pivot + keyframes (capture pivot_scale)
    if is_keyframed and version >= 0x3000A:
        _ = read_vec3_to_blender(b)  # pivot translation
        _ = read_quat(b)             # pivot rotation
        pivot_scale_rf = read_vec3_rf(b)  # pivot scale (RAW RF axes)
"""
    s = s.replace(old_pivot, new_pivot)

# 3) Patch scale keys block: capture first scale key (RAW), then skip the rest correctly
old_scale = """        # scale keys
        n = b.s32()
        b.read(n * (4 + 12 + 12 + 12))
"""
if old_scale in s:
    new_scale = """        # scale keys (capture first)
        n = b.s32()
        if n > 0:
            _time = b.s32()              # time (int32; RF uses ticks)
            scale_key0_rf = read_vec3_rf(b)  # value (RAW RF axes)
            b.read(12 + 12)              # inTan + outTan
            if n > 1:
                b.read((n - 1) * (4 + 12 + 12 + 12))
"""
    s = s.replace(old_scale, new_scale)

# 4) Apply keyframed scale to THIS mesh's geometry only (to mimic "no scale inheritance")
# Insert right before the bail comment.
insert_at = "# If we didn't get what we need, bail"
must(insert_at)

if "Apply keyframed scale to geometry only" not in s:
    geom_block = """
    # Apply keyframed scale to geometry only (do NOT propagate to children)
    # VFX keyframed meshes carry pivot_scale + scale keys; RF likely doesn't inherit scale like glTF does.
    if (not morph) and is_keyframed and pivot_scale_rf is not None:
        sx, sy, sz = pivot_scale_rf
        if scale_key0_rf is not None:
            kx, ky, kz = scale_key0_rf
            sx *= kx; sy *= ky; sz *= kz

        # Convert RF-axis scale to Blender-axis scale (no sign flips, just axis mapping)
        # rf_to_blender for positions is (-x, z, -y) => scale maps (sx, sy, sz) -> (|sx|, |sz|, |sy|)
        sx_b = abs(float(sx))
        sy_b = abs(float(sz))
        sz_b = abs(float(sy))

        verts = [(vx * sx_b, vy * sy_b, vz * sz_b) for (vx, vy, vz) in verts]
        base_verts = verts
"""
    s = s.replace(insert_at, geom_block + "\n    " + insert_at)

# 5) OPTIONAL: if your file still uses /S16_MAX, switch to RAW decompress (matches your bounds proof)
# Only do this if the normalized block exists.
if "nx = rx / S16_MAX" in s:
    s = s.replace(
        """                    nx = rx / S16_MAX
                    ny = ry / S16_MAX
                    nz = rz / S16_MAX
                    vx = center_rf[0] + mult_rf[0] * nx
                    vy = center_rf[1] + mult_rf[1] * ny
                    vz = center_rf[2] + mult_rf[2] * nz
""",
        """                    vx = center_rf[0] + mult_rf[0] * float(rx)
                    vy = center_rf[1] + mult_rf[1] * float(ry)
                    vz = center_rf[2] + mult_rf[2] * float(rz)
"""
    )

if s == orig:
    print("[INFO] Patch already applied or nothing changed.")
else:
    p.write_text(s, encoding="utf-8")
    print("[OK] Patched vfx2obj.py to apply keyframed pole scale to pole geometry only.")
