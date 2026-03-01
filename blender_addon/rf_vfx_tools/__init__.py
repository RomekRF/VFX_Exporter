bl_info = {
    "name": "RF VFX Tools",
    "author": "RomekRF",
    "version": (0, 3, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > RF VFX",
    "description": "Standalone RF1 VFX (.vfx) <-> glTF workflow (Import + Export). Ships with converter in the add-on.",
    "category": "Import-Export",
}

import bpy
import os
import sys
import io
import runpy
import shutil
import tempfile
import traceback
from contextlib import redirect_stdout, redirect_stderr
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, PointerProperty
)

SUPPORTED_VFX_VERSIONS = {0x00040006}  # v4.6 (validated pipeline)

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
    for area in bpy.context.screen.areas:
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

def _run_vendor_script(script_name: str, argv: list[str], cwd: str | None = None):
    """
    Runs vendor/<script_name> like a CLI (in-process), capturing stdout/stderr.
    Returns (output_text).
    """
    vdir = _vendor_dir()
    script_path = os.path.join(vdir, script_name)
    if not os.path.exists(script_path):
        raise RuntimeError(f"Missing vendor script: {script_name}")

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    old_syspath = sys.path[:]

    buf = io.StringIO()
    try:
        sys.path.insert(0, vdir)  # allow local imports from vendor/
        sys.argv = [script_name] + argv
        if cwd:
            os.chdir(cwd)
        with redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path = old_syspath
        try:
            os.chdir(old_cwd)
        except Exception:
            pass

    return buf.getvalue()

class RFVFX_Settings(bpy.types.PropertyGroup):
    # --- Import ---
    import_vfx: StringProperty(name="VFX File", subtype="FILE_PATH", default="")
    import_scale: FloatProperty(name="Scale", default=1.0, min=0.0001, max=100000.0)
    import_anchor: StringProperty(name="Anchor (optional)", default="")
    import_debug: BoolProperty(name="Debug", default=False)

    # --- Export ---
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

        if ver not in SUPPORTED_VFX_VERSIONS:
            _popup(
                f"Unsupported VFX version: 0x{ver:08X}\n\n"
                f"Currently supported: 0x00040006 (v4.6)\n\n"
                f"This file is an older RF VFX format. We can add support later,\n"
                f"but it won't import with the current converter.",
                title="RF VFX: Import"
            )
            return {"CANCELLED"}

        # run conversion: VFX -> glTF (writes next to input)
        args = ["--gltf", "--scale", str(s.import_scale)]
        if s.import_debug:
            args.append("--debug-frames")
        if s.import_anchor.strip():
            args += ["--anchor", s.import_anchor.strip()]
        args.append(vfx)

        header = f"RF VFX Import\nInput: {vfx}\nVersion: 0x{ver:08X}\n\nCommand:\n  vfx2obj.py " + " ".join(args)

        try:
            out = _run_vendor_script("vfx2obj.py", args, cwd=_vendor_dir())
            _write_log(header, out)

            gltf = os.path.splitext(vfx)[0] + ".gltf"
            if os.path.exists(gltf):
                bpy.ops.import_scene.gltf(filepath=gltf)
                self.report({"INFO"}, "Imported glTF into Blender.")
            else:
                _popup("Conversion finished but glTF was not found next to the VFX.\nCheck RFVFX_Log.", title="RF VFX: Import")
                return {"CANCELLED"}

        except Exception as e:
            tb = traceback.format_exc()
            _write_log(header, f"ERROR:\n{e}\n\n{tb}")
            _popup("Import failed. Open the Text Editor and read RFVFX_Log.", title="RF VFX: Import")
            return {"CANCELLED"}

        return {"FINISHED"}

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

        # Optional template (enables pivot×key0 fix for keyframed parents like flagpole)
        tmpl_ver = None
        if tmpl_vfx:
            if not os.path.exists(tmpl_vfx):
                self.report({"ERROR"}, "Template VFX does not exist.")
                return {"CANCELLED"}
            tmpl_ver = _read_vfx_version(tmpl_vfx)
            if tmpl_ver is None:
                self.report({"ERROR"}, "Template VFX is not a VSFX file.")
                return {"CANCELLED"}
            if tmpl_ver not in SUPPORTED_VFX_VERSIONS:
                _popup(
                    f"Template VFX version 0x{tmpl_ver:08X} is not supported for pivot-fix.\n"
                    f"Expected 0x00040006.\n\n"
                    f"Export will continue WITHOUT pivot fix.",
                    title="RF VFX: Export"
                )
                tmpl_vfx = ""  # disable pivot fix

        # Create temp working dir
        tmpdir = tempfile.mkdtemp(prefix="rfvfx_")
        gltf_path = os.path.join(tmpdir, "scene.gltf")
        tmp_vfx = os.path.join(tmpdir, "trueexport_tmp.vfx")

        header = (
            "RF VFX Export\n"
            f"Output: {out_vfx}\n"
            f"Template: {tmpl_vfx or '(none)'}\n\n"
        )

        try:
            # Export glTF from Blender
            bpy.ops.export_scene.gltf(
                filepath=gltf_path,
                export_format="GLTF_SEPARATE",
                use_selection=bool(s.export_selected_only),
                export_apply=bool(s.export_apply_transforms),
            )

            # True export glTF -> brand-new VFX
            args = ["--new-vfx-from-gltf", gltf_path, "--vfx-out", tmp_vfx, "--gltf-scale", str(s.export_gltf_scale)]
            if s.export_anchor.strip():
                args += ["--anchor", s.export_anchor.strip()]

            header += "Command (trueexport):\n  vfx2obj.py " + " ".join(args) + "\n"
            out1 = _run_vendor_script("vfx2obj.py", args, cwd=_vendor_dir())

            # Apply pivot fix if template provided
            out2 = ""
            if tmpl_vfx:
                args2 = ["--template", tmpl_vfx, "--in", tmp_vfx, "--out", out_vfx]
                header += "\nCommand (pivot fix):\n  pivot_patch_xkey0.py " + " ".join(args2) + "\n"
                out2 = _run_vendor_script("pivot_patch_xkey0.py", args2, cwd=_vendor_dir())
            else:
                shutil.copyfile(tmp_vfx, out_vfx)

            _write_log(header, out1 + ("\n\n" + out2 if out2 else ""))

            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                _popup(f"Temp kept at:\n{tmpdir}", title="RF VFX: Export")

            self.report({"INFO"}, "Exported VFX.")
            return {"FINISHED"}

        except Exception as e:
            tb = traceback.format_exc()
            _write_log(header, f"ERROR:\n{e}\n\n{tb}")
            _popup("Export failed. Open the Text Editor and read RFVFX_Log.", title="RF VFX: Export")
            try:
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
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

        # IMPORT
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

        # EXPORT
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
            col.separator()
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
