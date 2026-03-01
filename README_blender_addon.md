# RF VFX Blender Add-on (Phase 1)

This add-on is intentionally simple: it wraps the known-good `vfx2obj.py` CLI workflow.

## Install
1) Blender → Edit → Preferences → Add-ons → Install…
2) Select the folder: `blender_addon\rf_vfx_tools` as a ZIP (or zip the `rf_vfx_tools` folder).
3) Enable the add-on: **RF VFX Tools (vfx2obj wrapper)**

## Configure
Add-on Preferences:
- Repo Root: folder containing `vfx2obj.py`
- Python Executable: your system Python (ex: `python` or full path)
- glTF Scale: usually `1.0`

## Use
View3D → Sidebar → **RF VFX**
- **VFX -> glTF (and import)**: generates `<samebase>.gltf/.bin` next to the VFX, imports into Blender.
- **glTF -> VFX (patch template)**: runs patch-vfx-only (best for normal workflow).
- **glTF -> VFX (trueexport + pivot fix)**: runs trueexport then applies the proven pivot fix using the template VFX.

Notes:
- For patch mode: export glTF from Blender (same vertex counts), then patch back.
- For trueexport+fix: template VFX is used ONLY to borrow pivot/key0-scale behavior (your proven PIVOT_XKEY0 rule).
