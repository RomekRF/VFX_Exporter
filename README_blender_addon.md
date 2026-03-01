# RF VFX Blender Add-on

## Phase 2 features
- Settings live on the Scene (.blend): paths + flags persist.
- One-click Blender glTF export (for patch/trueexport workflows).
- Selection → `--only-mesh` (when Patch mode uses “Use selected meshes”).
- TrueExport runs the proven pivot fix automatically:
  - `pivot_translation_out = pivot_translation_template * key0_scale_template`

## Install
1) Zip `blender_addon\rf_vfx_tools\` (folder contains `__init__.py`)
2) Blender → Edit → Preferences → Add-ons → Install… (choose the zip)
3) Enable: **RF VFX Tools (vfx2obj wrapper)**
4) In add-on prefs:
   - Repo Root = folder containing `vfx2obj.py`
   - Python Executable optional (blank auto-uses RepoRoot\.venv\Scripts\python.exe if present)

## Use
View3D → Sidebar → **RF VFX**

### Import
- Set **Import VFX**
- (Optional) set **Anchor** + enable **--debug-frames**
- Click **VFX -> glTF (and import)**

### Export glTF
- Set **Export glTF** path (writes .gltf + .bin)
- Optional: export selected only / apply transforms
- Click **Export glTF**

### Patch (glTF -> VFX, topology must match)
- Template VFX + glTF In + VFX Out
- Enable “Use selected meshes” (auto builds `--only-mesh` from selected Mesh objects)
- Click **Patch VFX from glTF**

### TrueExport + PivotFix (brand-new VFX)
- Pivot Template VFX + glTF In + VFX Out
- Anchor optional
- Keep “Apply pivot×key0 fix” ON (this is the solved flagpole/child alignment fix)
- Click **TrueExport + PivotFix**
