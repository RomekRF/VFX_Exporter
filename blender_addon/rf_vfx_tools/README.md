# RF VFX Tools (Blender Add-on)

## What this fixes
- Imports **any RF .vfx header version** without blocking.
- Normalizes input into a v4.6 **mesh-only** wrapper before conversion:
  - rewrites header to 0x00040006
  - pads to 128 bytes
  - copies only SFXO + MATL sections
  - skips PART/VParticle (so no crashes)
- Runs vendor scripts via **subprocess** (Blender Python) so Blender doesn't hard-crash on script exits.

## Limitations (current)
- Particle emitter sections (PART/VParticle) are skipped (no particles imported yet).
- If a VFX contains no mesh sections, import will show a message and stop.

## Logs
All output (and full tracebacks) go to Blender Text Editor → **RFVFX_Log**.
