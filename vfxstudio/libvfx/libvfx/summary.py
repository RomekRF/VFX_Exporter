from __future__ import annotations
import os
import re
import struct
import subprocess
from typing import List
from .model import VfxSummary, VfxMeshInfo, VfxMaterial

_DEBUG_RE = re.compile(
    r"\[DEBUG_FRAMES\]\s+mesh='(?P<name>[^']+)'\s+fps=(?P<fps>\d+)\s+frames=(?P<frames>\d+)\s+morph=(?P<morph>True|False)\s+flags=0x(?P<flags>[0-9A-Fa-f]+)"
)

def _read_header_version(data: bytes) -> str:
    # Expected: b'VSFX' + uint32 version LE
    if len(data) >= 8 and data[0:4] == b"VSFX":
        ver = struct.unpack_from("<I", data, 4)[0]
        return f"0x{ver:08X}"
    return "0x????????"

def _count_tag(data: bytes, tag: bytes) -> int:
    # Simple scan count of 4-byte tags; good enough for a quick summary.
    # (We will replace with full chunk parsing later.)
    return data.count(tag)

def _scan_textures(data: bytes) -> List[str]:
    found = []
    for ext in (b".tga", b".dds"):
        i = 0
        while True:
            j = data.find(ext, i)
            if j == -1:
                break
            # scan backward to likely start of ASCII token
            k = j
            while k > 0 and 32 <= data[k-1] <= 126 and data[k-1] != 0:
                k -= 1
            tex = data[k:j+len(ext)].decode("ascii", errors="ignore").strip()
            if tex and tex not in found:
                found.append(tex)
            i = j + 1
    return found

def _probe_mesh_debug(vfx_path: str) -> List[VfxMeshInfo]:
    # Use existing backend exe if present to get authoritative mesh facts.
    exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dist", "vfx2obj.exe"))
    exe = exe if os.path.exists(exe) else os.path.abspath(os.path.join(os.getcwd(), "dist", "vfx2obj.exe"))
    if not os.path.exists(exe):
        return []

    try:
        p = subprocess.run([exe, "--debug-frames", "--gltf", vfx_path], capture_output=True, text=True, check=True)
    except Exception:
        # Even if conversion fails, we may still get debug output
        try:
            p = subprocess.run([exe, "--debug-frames", vfx_path], capture_output=True, text=True)
        except Exception:
            return []

    meshes: List[VfxMeshInfo] = []
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    for m in _DEBUG_RE.finditer(out):
        meshes.append(
            VfxMeshInfo(
                name=m.group("name"),
                fps=int(m.group("fps")),
                frames=int(m.group("frames")),
                morph=(m.group("morph") == "True"),
                flags_hex="0x" + m.group("flags").upper(),
            )
        )
    return meshes

def summarize_vfx(path: str) -> VfxSummary:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read()

    version_hex = _read_header_version(data)

    # Tag counts (quick signals)
    dummy_count = _count_tag(data, b"DMMY")
    particle_count = _count_tag(data, b"PART")

    # Texture scan
    textures = _scan_textures(data)
    materials = [VfxMaterial(name=t, texture=t) for t in textures]

    # Mesh facts from backend debug (best source we have today)
    meshes = _probe_mesh_debug(path)

    # Reasonable approximation: end_frame = max(frames)-1 if we have mesh frame counts
    end_frame = max((mi.frames for mi in meshes), default=-1)
    if end_frame >= 0:
        end_frame -= 1

    return VfxSummary(
        path=path,
        file_size=size,
        version_hex=version_hex,
        end_frame=end_frame,
        meshes=meshes,
        materials=materials,
        dummy_count=dummy_count,
        particle_count=particle_count,
    )
