# RF VFX Tools (Blender Add-on)

## What this fixes
- Imports RF `.vfx` safely (pads header to 128 bytes, drops known-crashy `PART` blocks).
- Converts to **single-scene glTF** (multiple objects) via vendored `vfx2obj.py`.
- Preserves RF round-trip metadata in `node.extras.rf_vfx`:
  - keyframed pivot+keys are stored as a base64 blob (`keyframed_block_b64`)
  - any geometry scale-baking applied during VFX→glTF is recorded (`baked_geom_scale`)
- On export back to VFX, the writer uses that metadata to:
  - **un-bake** geometry scale before encoding
  - write the original keyframed pivot/key lists (so flags stop “floating high” in-game)

## Limitations (current)
- Particle emitter sections (PART/VParticle) are skipped (no particles imported yet).
- If a VFX contains no mesh sections, import will show a message and stop.

## Logs
All output (and full tracebacks) go to Blender Text Editor → **RFVFX_Log**.
