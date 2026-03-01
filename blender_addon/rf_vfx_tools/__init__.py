bl_info = {
    "name": "RF VFX Tools",
    "author": "RomekRF",
    "version": (0, 3, 4),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > RF VFX",
    "description": "Standalone RF1 VFX (.vfx) <-> glTF workflow. Fixes inside-out meshes by flipping winding symmetrically (import+export).",
    "category": "Import-Export",
}

import bpy
import os, sys, io, shutil, struct, tempfile, traceback, subprocess, json
from bpy.props import StringProperty, BoolProperty, FloatProperty, PointerProperty

# Drop only sections known to crash the mesh converter (keep everything else so transforms aren't lost)
DROP_SECTIONS = {b"PART"}  # VParticle blocks (we'll support later)

def _addon_dir(): return os.path.dirname(__file__)
def _vendor_dir(): return os.path.join(_addon_dir(), "vendor")

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

def _read_vfx_header(path: str):
    with open(path, "rb") as f:
        b = f.read(8)
    if len(b) < 8 or b[:4] != b"VSFX":
        return None
    ver = int.from_bytes(b[4:8], "little", signed=False)
    return ver

def _is_printable_4cc(b4: bytes) -> bool:
    if len(b4) != 4: return False
    return all(0x20 <= c <= 0x7E for c in b4)

def _scan_section_start(data: bytes, max_scan=65536):
    n = min(len(data) - 8, max_scan)
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
        # quick chain validation
        ok = True
        cur = off
        for _ in range(8):
            if cur + 8 > len(data): break
            t2 = data[cur:cur+4]
            if not _is_printable_4cc(t2): ok = False; break
            s2 = struct.unpack_from("<I", data, cur+4)[0]
            if s2 < 8 or cur + 4 + s2 > len(data): ok = False; break
            cur = cur + 4 + s2
        if ok:
            return off
    return 128

def _normalize_pad_header_keep_version(src_vfx: str, dst_vfx: str):
    """
    Fix 'header issue' safely:
    - Keep original version value
    - Ensure header is 128 bytes (pad with zeros)
    - Copy sections (optionally drop PART)
    This avoids lying about version and preserves transforms across files.
    """
    data = open(src_vfx, "rb").read()
    if len(data) < 8 or data[:4] != b"VSFX":
        raise RuntimeError("Not a VSFX file.")

    ver = struct.unpack_from("<I", data, 4)[0]
    start = _scan_section_start(data)
    off = start
    kept = skipped = 0

    out = bytearray()
    out += b"VSFX"
    out += struct.pack("<I", ver)
    out += b"\x00" * (128 - 8)

    while off + 8 <= len(data):
        t = data[off:off+4]
        if not _is_printable_4cc(t): break
        size = struct.unpack_from("<I", data, off+4)[0]
        if size < 8 or off + 4 + size > len(data): break

        chunk = data[off:off + 4 + size]
        if t in DROP_SECTIONS:
            skipped += 1
        else:
            out += chunk
            kept += 1

        off = off + 4 + size

    open(dst_vfx, "wb").write(out)
    return ver, kept, skipped, start

def _run_vendor(script_name: str, argv: list[str], cwd: str):
    vdir = _vendor_dir()
    script = os.path.join(vdir, script_name)
    if not os.path.exists(script):
        raise RuntimeError(f"Missing vendor script: {script_name}")

    env = os.environ.copy()
    env["PYTHONPATH"] = vdir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, script] + argv
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def _flip_gltf_winding_in_place(gltf_path: str) -> str:
    """
    Swaps i0 <-> i2 for every triangle in every indexed primitive.
    This fixes inside-out meshes for CCW glTF viewers (Blender).
    Returns a short status string for logging.
    """
    with open(gltf_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    # resolve .bin path
    buffers = g.get("buffers", [])
    if not buffers:
        return "flip_winding: no buffers (skipped)"
    uri = buffers[0].get("uri", "")
    if not uri:
        return "flip_winding: embedded/empty buffer uri (skipped)"
    bin_path = os.path.join(os.path.dirname(gltf_path), uri)
    if not os.path.exists(bin_path):
        return f"flip_winding: missing bin {bin_path} (skipped)"

    data = bytearray(open(bin_path, "rb").read())

    accessors = g.get("accessors", [])
    views = g.get("bufferViews", [])
    flipped = 0

    def comp_size(ct):
        return {5121:1, 5123:2, 5125:4}.get(ct, 0)

    for mesh in g.get("meshes", []) or []:
        for prim in mesh.get("primitives", []) or []:
            if "indices" not in prim:
                continue
            ai = prim["indices"]
            if ai is None or ai >= len(accessors):
                continue
            acc = accessors[ai]
            if acc.get("type") != "SCALAR":
                continue
            ct = acc.get("componentType")
            sz = comp_size(ct)
            if sz == 0:
                continue

            count = int(acc.get("count", 0))
            if count < 3 or (count % 3) != 0:
                continue

            vi = acc.get("bufferView")
            if vi is None or vi >= len(views):
                continue
            view = views[vi]

            bv_off = int(view.get("byteOffset", 0))
            a_off = int(acc.get("byteOffset", 0))
            off = bv_off + a_off

            # NOTE: indices are tightly packed in practice; if byteStride exists, we ignore (rare for indices)
            end = off + count * sz
            if end > len(data):
                continue

            # read indices
            fmt = {5121:"B", 5123:"H", 5125:"I"}[ct]
            # unpack -> list of ints
            vals = list(struct.unpack_from("<" + fmt*count, data, off))

            # flip triangles
            for t in range(0, count, 3):
                vals[t], vals[t+2] = vals[t+2], vals[t]
            struct.pack_into("<" + fmt*count, data, off, *vals)
            flipped += count // 3

    if flipped > 0:
        open(bin_path, "wb").write(data)
    return f"flip_winding: flipped_triangles={flipped}"

class RFVFX_Settings(bpy.types.PropertyGroup):
    import_vfx: StringProperty(name="VFX File", subtype="FILE_PATH", default="")
    import_scale: FloatProperty(name="Scale", default=1.0, min=0.0001, max=100000.0)
    import_anchor: StringProperty(name="Anchor (optional)", default="")
    import_debug: BoolProperty(name="Debug", default=False)

    export_vfx_out: StringProperty(name="Output VFX", subtype="FILE_PATH", default="")
    export_template_vfx: StringProperty(name="Template VFX", subtype="FILE_PATH", default="")
    use_last_import_as_template: BoolProperty(name="Use last imported VFX as Template", default=True)

    export_gltf_scale: FloatProperty(name="glTF Scale", default=1.0, min=0.0001, max=100000.0)
    export_anchor: StringProperty(name="Anchor (optional)", default="")
    export_selected_only: BoolProperty(name="Selected Only", default=False)
    export_apply_transforms: BoolProperty(name="Apply Transforms", default=True)

    # ✅ new: winding fix (default ON) applies both import and export (symmetrical)
    fix_winding: BoolProperty(name="Fix inside-out meshes (flip winding)", default=True)

    last_import_vfx: StringProperty(name="(internal) last import", default="")

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

        ver = _read_vfx_header(vfx)
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
            ver2, kept, skipped, start = _normalize_pad_header_keep_version(vfx, norm_vfx)
            header += f"Normalize: start_off={start} kept={kept} skipped={skipped} ver_preserved=0x{ver2:08X}\n\n"

            args = ["--gltf", "--scale", str(s.import_scale)]
            if s.import_debug: args.append("--debug-frames")
            if s.import_anchor.strip(): args += ["--anchor", s.import_anchor.strip()]
            args.append(norm_vfx)

            header += "Run:\n  vfx2obj.py " + " ".join(args) + "\n"
            rc, out = _run_vendor("vfx2obj.py", args, cwd=_vendor_dir())
            if rc != 0:
                _write_log(header, out)
                _popup("Import failed. Open Text Editor → RFVFX_Log.", title="RF VFX: Import")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            gltf = os.path.splitext(norm_vfx)[0] + ".gltf"
            if not os.path.exists(gltf):
                _write_log(header, out)
                _popup("Conversion ran but glTF was not produced. See RFVFX_Log.", title="RF VFX: Import")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            # ✅ fix inside-out (winding) before Blender imports it
            flip_msg = ""
            if s.fix_winding:
                flip_msg = _flip_gltf_winding_in_place(gltf)

            _write_log(header, out + ("\n\n" + flip_msg if flip_msg else ""))

            bpy.ops.import_scene.gltf(filepath=gltf)

            s.last_import_vfx = vfx
            if s.use_last_import_as_template and not s.export_template_vfx.strip():
                s.export_template_vfx = vfx

            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)

            self.report({"INFO"}, "Imported VFX into Blender.")
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

        if s.use_last_import_as_template and (not tmpl_vfx) and s.last_import_vfx.strip():
            tmpl_vfx = bpy.path.abspath(s.last_import_vfx).strip()

        out_dir = os.path.dirname(out_vfx)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        pivot_tool = os.path.join(_vendor_dir(), "pivot_patch_xkey0.py")
        tmpl_ok = False
        if tmpl_vfx and os.path.exists(tmpl_vfx) and os.path.exists(pivot_tool):
            tmpl_ok = (_read_vfx_header(tmpl_vfx) == 0x00040006)

        tmpdir = tempfile.mkdtemp(prefix="rfvfx_export_")
        gltf_path = os.path.join(tmpdir, "scene.gltf")
        tmp_vfx = os.path.join(tmpdir, "trueexport_tmp.vfx")

        header = (
            "RF VFX Export\n"
            f"Output: {out_vfx}\n"
            f"Template (used): {tmpl_vfx if tmpl_vfx else '(none)'}\n"
            f"PivotFix enabled: {tmpl_ok}\n"
            f"Winding fix enabled: {bool(s.fix_winding)}\n"
            f"Temp: {tmpdir}\n\n"
        )

        try:
            bpy.ops.export_scene.gltf(
                filepath=gltf_path,
                export_format="GLTF_SEPARATE",
                use_selection=bool(s.export_selected_only),
                export_apply=bool(s.export_apply_transforms),
            )

            # ✅ flip winding back before feeding glTF to vfx2obj (symmetry)
            flip_msg = ""
            if s.fix_winding:
                flip_msg = _flip_gltf_winding_in_place(gltf_path)

            args = ["--new-vfx-from-gltf", gltf_path, "--vfx-out", tmp_vfx, "--gltf-scale", str(s.export_gltf_scale)]
            if s.export_anchor.strip(): args += ["--anchor", s.export_anchor.strip()]

            header += "Run trueexport:\n  vfx2obj.py " + " ".join(args) + "\n"
            rc1, out1 = _run_vendor("vfx2obj.py", args, cwd=_vendor_dir())
            if rc1 != 0:
                _write_log(header, out1 + ("\n\n" + flip_msg if flip_msg else ""))
                _popup("Export failed. Open Text Editor → RFVFX_Log.", title="RF VFX: Export")
                if not s.keep_temp:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return {"CANCELLED"}

            out2 = ""
            if tmpl_ok:
                args2 = ["--template", tmpl_vfx, "--in", tmp_vfx, "--out", out_vfx]
                header += "\nRun pivot fix:\n  pivot_patch_xkey0.py " + " ".join(args2) + "\n"
                rc2, out2 = _run_vendor("pivot_patch_xkey0.py", args2, cwd=_vendor_dir())
                if rc2 != 0:
                    _write_log(header, out1 + "\n\n" + out2 + ("\n\n" + flip_msg if flip_msg else ""))
                    _popup("Pivot fix failed. Open Text Editor → RFVFX_Log.", title="RF VFX: Export")
                    if not s.keep_temp:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                    return {"CANCELLED"}
            else:
                shutil.copyfile(tmp_vfx, out_vfx)

            _write_log(header, out1 + ("\n\n" + out2 if out2 else "") + ("\n\n" + flip_msg if flip_msg else ""))

            if not s.keep_temp:
                shutil.rmtree(tmpdir, ignore_errors=True)

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

        box = layout.box()
        row = box.row()
        row.label(text="Import VFX")
        row.prop(s, "show_import_advanced", text="Advanced", toggle=True)
        box.prop(s, "import_vfx")
        box.operator("rfvfx.import_vfx", icon="IMPORT")
        if s.show_import_advanced:
            col = box.column(align=True)
            col.prop(s, "import_scale")
            col.prop(s, "import_anchor")
            col.prop(s, "import_debug")
            col.prop(s, "fix_winding")
            col.prop(s, "keep_temp")

        box = layout.box()
        row = box.row()
        row.label(text="Export VFX")
        row.prop(s, "show_export_advanced", text="Advanced", toggle=True)
        box.prop(s, "export_vfx_out")
        box.prop(s, "export_template_vfx")
        box.prop(s, "use_last_import_as_template")
        box.operator("rfvfx.export_vfx", icon="EXPORT")
        if s.show_export_advanced:
            col = box.column(align=True)
            col.prop(s, "export_selected_only")
            col.prop(s, "export_apply_transforms")
            col.prop(s, "export_gltf_scale")
            col.prop(s, "export_anchor")
            col.prop(s, "fix_winding")
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
