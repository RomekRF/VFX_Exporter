bl_info = {
    "name": "RF VFX Tools",
    "author": "RomekRF",
    "version": (0, 3, 2),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > RF VFX",
    "description": "Standalone RF1 VFX (.vfx) <-> glTF workflow (Import + Export). Normalizes legacy headers; skips unsupported sections safely.",
    "category": "Import-Export",
}

import bpy
import os
import sys
import io
import shutil
import struct
import tempfile
import traceback
import subprocess
from bpy.props import StringProperty, BoolProperty, FloatProperty, PointerProperty

KEEP_SECTIONS = {b"SFXO", b"MATL"}  # mesh + materials (skip PART/VParticle etc for now)
V46 = 0x00040006

def _addon_dir():
    return os.path.dirname(__file__)

def _vendor_dir():
    return os.path.join(_addon_dir(), "vendor")

def _log_textblock():
    name = "RFVFX_Log"
    txt = bpy.data.texts.get(name)
    if txt is None:
        txt = bpy.data.texts.new(name)
    return txt

def _write_log(header: str, body: str):
    txt = _log_textblock()
    txt.clear()
    txt.write(header + "\n\n" + body)
    # try to show it
    scr = getattr(bpy.context, "screen", None)
    if scr:
        for area in scr.areas:
            if area.type == "TEXT_EDITOR":
                area.spaces.active.text = txt
                break

def _popup(msg: str, title="RF VFX"):
    def draw(self, _context):
        for line in msg.splitlines():
            self.layout.label(text=line)
    bpy.context.window_manager.popup_menu(draw, title=title, icon="INFO")

def _read_vfx_version(path: str):
    try:
        with open(path, "rb") as f:
            b = f.read(8)
        if len(b) < 8 or b[:4] != b"VSFX":
            return None
        return int.from_bytes(b[4:8], "little", signed=False)
    except Exception:
        return None

def _is_printable_4cc(b4: bytes) -> bool:
    if len(b4) != 4:
        return False
    for c in b4:
        if c < 0x20 or c > 0x7E:
            return False
    return True

def _scan_section_start(data: bytes, max_scan=65536):
    n = min(len(data) - 8, max_scan)
    best = None

    for off in range(8, n, 4):
        t = data[off:off+4]
        if not _is_printable_4cc(t):
            continue
        size = struct.unpack_from("<I", data, off+4)[0]
        if size < 8:
            continue
        end = off + 4 + size
        if end > len(data):
            continue

        # quick chain validation: walk a few sections
        ok = True
        cur = off
        steps = 0
        while cur + 8 <= len(data) and steps < 8:
            t2 = data[cur:cur+4]
            if not _is_printable_4cc(t2):
                ok = False
                break
            s2 = struct.unpack_from("<I", data, cur+4)[0]
            if s2 < 8 or cur + 4 + s2 > len(data):
                ok = False
                break
            cur = cur + 4 + s2
            steps += 1

        if ok:
            best = off
            break

    return best if best is not None else 128

def _normalize_to_v46_mesh_only(src_vfx: str, dst_vfx: str):
    """
    Creates a v4.6 wrapper VFX that the converter can digest consistently:
    - header rewritten to VSFX + version 0x40006 + zero padding to 128 bytes
    - copies only SFXO + MATL sections, in-order
    - skips everything else (PART, etc) so we never crash
    Returns: (kept_section_count, skipped_section_count, start_offset)
    """
    data = open(src_vfx, "rb").read()
    if len(data) < 8 or data[:4] != b"VSFX":
        raise RuntimeError("Not a VSFX file.")

    start = _scan_section_start(data)
    off = start
    kept = 0
    skipped = 0

    out = bytearray()
    out += b"VSFX"
    out += struct.pack("<I", V46)
    out += b"\x00" * (128 - 8)  # header to 128

    while off + 8 <= len(data):
        t = data[off:off+4]
        if not _is_printable_4cc(t):
            break
        size = struct.unpack_from("<I", data, off+4)[0]
        if size < 8 or off + 4 + size > len(data):
            break
        chunk = data[off:off + 4 + size]

        if t in KEEP_SECTIONS:
            out += chunk
            kept += 1
        else:
            skipped += 1

        off = off + 4 + size

    open(dst_vfx, "wb").write(out)
    return kept, skipped, start

def _run_vendor_subprocess(script_name: str, argv: list[str], cwd: str):
    vdir = _vendor_dir()
    script = os.path.join(vdir, script_name)
    if not os.path.exists(script):
        raise RuntimeError(f"Missing vendor script: {script_name}")

    env = os.environ.copy()
    env["PYTHONPATH"] = vdir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, script] + argv
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

class RFVFX_Settings(bpy.types.PropertyGroup):
    # Import
    import_vfx: StringProperty(name="VFX File", subtype="FILE_PATH", default="")
    import_scale: FloatProperty(name="Scale", default=1.0, min=0.0001, max=100000.0)
    import_anchor: StringProperty(name="Anchor (optional)", default="")
    import_debug: BoolProperty(name="Debug", default=False)

    # Export
    export_vfx_out: StringProperty(name="Output VFX", subtype="FILE_PATH", default="")
    export_template_vfx: StringProperty(name="Template VFX (optional)", subtype="FILE_PATH", default="")
    export_gltf_scale: FloatProperty(name="glTF Scale", default=1.0, min=0.0001, max=100000.0)
    export_anchor: StringProperty(name="Anchor (optional)", default="")
    export_selected_only: BoolProperty(name="Selected Only", default=False)
    export_apply_transforms: BoolProperty(name="Apply Transforms", default=True)

    # UX
    show_import_advanced: BoolProperty(name="Advanced", default=False)
    show_export_advanced: BoolProperty(name="Advanced", default=False)
    keep_temp: BoolProperty(name="Keep Temp Files", default=False)

class RFVFX_OT_ImportVFX(bpy.types.Operator):
    bl_idname = "rfvfx.import_vfx"
    bl_label = "Import VFX"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.rfvfx
        vfx = bpy.path.abspath(s.import_vfx).strip()
        if not vfx or not os.path.exists(vfx):
            self.report({"ERROR"}, "Pick a valid .vfx file.")
            return {"CANCELLED"}

        ver = _read_vfx_version(vfx)
        if ver is None:
            _popup("Not a VSFX file (bad header).", title="RF VFX: Import")
            return {"CANCELLED"}

        tmpdir = tempfile.mkdtemp(prefix="rfvfx_import_")
        norm_vfx = os.path.join(tmpdir, "input_norm.vfx")

        header = (
            "RF VFX Import\n"
            f"Input: {vfx}\n"
            f"Header version: 0x{ver:08X}\n"
            f"Temp: {tmpdir}\n"
        )

        try:
            kept, skipped, start = _normalize_to_v46_mesh_only(vfx, norm_vfx)
            header += f"Normalize: start_off={start} kept={kept} skipped={skipped}\n\n"
            if kept == 0:
                _write_log(header, "No mesh/material sections found to import (likely particle-only VFX).")
                _popup("This VFX contains no mesh sections to import (particle-only or unsupported).", title="RF VFX: Import")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            args = ["--gltf", "--scale", str(s.import_scale)]
            if s.import_debug:
                args.append("--debug-frames")
            if s.import_anchor.strip():
                args += ["--anchor", s.import_anchor.strip()]
            args.append(norm_vfx)

            header += "Run:\n  vfx2obj.py " + " ".join(args) + "\n"
            rc, out = _run_vendor_subprocess("vfx2obj.py", args, cwd=_vendor_dir())
            _write_log(header, out)

            if rc != 0:
                _popup("Import failed. Open Text Editor → RFVFX_Log.", title="RF VFX: Import")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            gltf = os.path.splitext(norm_vfx)[0] + ".gltf"
            if not os.path.exists(gltf):
                _popup("Conversion ran but glTF was not produced. See RFVFX_Log.", title="RF VFX: Import")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            bpy.ops.import_scene.gltf(filepath=gltf)

            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)

            self.report({"INFO"}, "Imported VFX (mesh-only) into Blender.")
            return {"FINISHED"}

        except BaseException:
            tb = traceback.format_exc()
            _write_log(header, "ERROR:\n" + tb)
            _popup("Import failed hard. Open Text Editor → RFVFX_Log.", title="RF VFX: Import")
            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return {"CANCELLED"}

class RFVFX_OT_ExportVFX(bpy.types.Operator):
    bl_idname = "rfvfx.export_vfx"
    bl_label = "Export VFX"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.rfvfx
        out_vfx = bpy.path.abspath(s.export_vfx_out).strip()
        tmpl_vfx = bpy.path.abspath(s.export_template_vfx).strip()

        if not out_vfx:
            self.report({"ERROR"}, "Set Output VFX path.")
            return {"CANCELLED"}

        out_dir = os.path.dirname(out_vfx)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        tmpl_ok = False
        if tmpl_vfx:
            if not os.path.exists(tmpl_vfx):
                self.report({"ERROR"}, "Template VFX does not exist.")
                return {"CANCELLED"}
            tv = _read_vfx_version(tmpl_vfx)
            # pivot_patch_xkey0 currently expects v4.6 template; keep strict to avoid corrupt pivots
            tmpl_ok = (tv == V46 and os.path.exists(os.path.join(_vendor_dir(), "pivot_patch_xkey0.py")))
            if not tmpl_ok:
                _popup(
                    "Template pivot-fix is only enabled when:\n"
                    "  - template is v4.6 (0x00040006)\n"
                    "  - vendor/pivot_patch_xkey0.py exists\n\n"
                    "Export will continue without pivot fix.",
                    title="RF VFX: Export"
                )

        tmpdir = tempfile.mkdtemp(prefix="rfvfx_export_")
        gltf_path = os.path.join(tmpdir, "scene.gltf")
        tmp_vfx = os.path.join(tmpdir, "trueexport_tmp.vfx")

        header = (
            "RF VFX Export\n"
            f"Output: {out_vfx}\n"
            f"Template: {tmpl_vfx if tmpl_vfx else '(none)'}\n"
            f"Temp: {tmpdir}\n\n"
        )

        try:
            bpy.ops.export_scene.gltf(
                filepath=gltf_path,
                export_format="GLTF_SEPARATE",
                use_selection=bool(s.export_selected_only),
                export_apply=bool(s.export_apply_transforms),
            )

            args = ["--new-vfx-from-gltf", gltf_path, "--vfx-out", tmp_vfx, "--gltf-scale", str(s.export_gltf_scale)]
            if s.export_anchor.strip():
                args += ["--anchor", s.export_anchor.strip()]

            header += "Run trueexport:\n  vfx2obj.py " + " ".join(args) + "\n"
            rc1, out1 = _run_vendor_subprocess("vfx2obj.py", args, cwd=_vendor_dir())
            if rc1 != 0:
                _write_log(header, out1)
                _popup("Export failed. Open Text Editor → RFVFX_Log.", title="RF VFX: Export")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            out2 = ""
            if tmpl_ok:
                args2 = ["--template", tmpl_vfx, "--in", tmp_vfx, "--out", out_vfx]
                header += "\nRun pivot fix:\n  pivot_patch_xkey0.py " + " ".join(args2) + "\n"
                rc2, out2 = _run_vendor_subprocess("pivot_patch_xkey0.py", args2, cwd=_vendor_dir())
                if rc2 != 0:
                    _write_log(header, out1 + "\n\n" + out2)
                    _popup("Pivot fix failed. Open Text Editor → RFVFX_Log.", title="RF VFX: Export")
                    if not s.keep_temp:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                    return {"CANCELLED"}
            else:
                shutil.copyfile(tmp_vfx, out_vfx)

            _write_log(header, out1 + ("\n\n" + out2 if out2 else ""))

            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                _popup(f"Temp kept at:\n{tmpdir}", title="RF VFX: Export")

            self.report({"INFO"}, "Exported VFX.")
            return {"FINISHED"}

        except BaseException:
            tb = traceback.format_exc()
            _write_log(header, "ERROR:\n" + tb)
            _popup("Export failed hard. Open Text Editor → RFVFX_Log.", title="RF VFX: Export")
            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return {"CANCELLED"}

class RFVFX_PT_Panel(bpy.types.Panel):
    bl_label = "RF VFX"
    bl_idname = "RFVFX_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RF VFX"

    def draw(self, context):
        s = context.scene.rfvfx
        layout = self.layout

        # Import
        box = layout.box()
        row = box.row()
        row.label(text="Import VFX (.vfx → .gltf)")
        row.prop(s, "show_import_advanced", text="Advanced", toggle=True)
        box.prop(s, "import_vfx")
        box.operator("rfvfx.import_vfx", icon="IMPORT")
        if s.show_import_advanced:
            col = box.column(align=True)
            col.prop(s, "import_scale")
            col.prop(s, "import_anchor")
            col.prop(s, "import_debug")
            col.prop(s, "keep_temp")

        # Export
        box = layout.box()
        row = box.row()
        row.label(text="Export VFX (Blender → .vfx)")
        row.prop(s, "show_export_advanced", text="Advanced", toggle=True)
        box.prop(s, "export_vfx_out")
        box.prop(s, "export_template_vfx")
        box.operator("rfvfx.export_vfx", icon="EXPORT")
        if s.show_export_advanced:
            col = box.column(align=True)
            col.prop(s, "export_selected_only")
            col.prop(s, "export_apply_transforms")
            col.prop(s, "export_gltf_scale")
            col.prop(s, "export_anchor")
            col.prop(s, "keep_temp")

_classes = (
    RFVFX_Settings,
    RFVFX_OT_ImportVFX,
    RFVFX_OT_ExportVFX,
    RFVFX_PT_Panel,
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.rfvfx = PointerProperty(type=RFVFX_Settings)

def unregister():
    try:
        del bpy.types.Scene.rfvfx
    except Exception:
        pass
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
