# RF VFX Tools (Blender Add-on)

Standalone add-on for Red Faction 1 (2001) `.vfx` workflow.

## What it does
- **Import**: `.vfx` → `.gltf` → imports into Blender
- **Export**: Blender scene → `.gltf` → **TrueExport** → `.vfx`
  - If you provide a **Template VFX**, export applies the proven fix:
    - `pivot_translation_out = pivot_translation_template * key0_scale_template`
    - This preserves child alignment for keyframed parents (e.g. flagpole/flag cloth).

## Supported VFX versions
- Currently supported: **0x00040006 (v4.6)**

Older RF VFX formats (ex: `0x0003000E`) will show an “Unsupported version” message instead of crashing.

## Install
Zip the folder `blender_addon\rf_vfx_tools\` and install the zip in Blender:
Edit → Preferences → Add-ons → Install…

## Logs
On error (or success), output goes to a Blender text block named **RFVFX_Log**.
Open the Text Editor to view it.
