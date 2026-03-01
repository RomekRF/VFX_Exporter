bl_info = {
    "name": "RF VFX Tools (vfx2obj wrapper)",
    "author": "RomekRF + ChatGPT",
    "version": (0, 2, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > RF VFX",
    "description": "Phase 2: persistent settings, glTF export button, selection->--only-mesh, trueexport + pivot×key0 fix.",
    "category": "Import-Export",
}

import bpy
import os
import subprocess
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, PointerProperty
)

def _prefs():
    return bpy.context.preferences.addons[__name__].preferences

def _repo_root():
    p = bpy.path.abspath(_prefs().repo_root).strip()
    return p

def _python_exe():
    p = bpy.path.abspath(_prefs().python_exe).strip()
    if p:
        return p
    repo = _repo_root()
    venv = os.path.join(repo, ".venv", "Scripts", "python.exe")
    if repo and os.path.exists(venv):
        return venv
    return "python"

def _run(cmd, cwd=None):
    print("[RFVFX] RUN:", " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    )
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}). See Blender system console for output.")
    return p.stdout

def _selected_mesh_names():
    names = []
    for obj in bpy.context.selected_objects or []:
        if obj.type == "MESH":
            names.append(obj.name)
    # stable + unique
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return out

def _default_out_vfx_from_gltf(gltf_path: str, suffix: str) -> str:
    base, _ = os.path.splitext(gltf_path)
    return base + suffix + ".vfx"

class RFVFX_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__

    repo_root: StringProperty(
        name="Repo Root",
        description="Folder containing vfx2obj.py (e.g. C:\\Users\\Romek\\OneDrive\\Desktop\\vfx2obj_build)",
        default="",
        subtype="DIR_PATH",
    )

    python_exe: StringProperty(
        name="Python Executable (optional)",
        description="If blank, tries RepoRoot\\.venv\\Scripts\\python.exe, else uses 'python'.",
        default="",
        subtype="FILE_PATH",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "repo_root")
        layout.prop(self, "python_exe")

class RFVFX_Settings(bpy.types.PropertyGroup):
    # Import
    import_vfx: StringProperty(name="Import VFX", subtype="FILE_PATH", default="")
    import_anchor: StringProperty(name="Anchor (optional)", default="flagpole")
    import_debug: BoolProperty(name="--debug-frames", default=False)
    import_scale: FloatProperty(name="--scale", default=1.0, min=0.0001, max=100000.0)

    # Export glTF from Blender
    export_gltf: StringProperty(name="Export glTF", subtype="FILE_PATH", default="")
    export_selected_only: BoolProperty(name="Export Selected Only", default=False)
    export_apply: BoolProperty(name="Apply Transforms", default=True)

    # Patch VFX from glTF
    patch_template_vfx: StringProperty(name="Template VFX", subtype="FILE_PATH", default="")
    patch_gltf_in: StringProperty(name="glTF In", subtype="FILE_PATH", default="")
    patch_vfx_out: StringProperty(name="Patched VFX Out", subtype="FILE_PATH", default="")
    patch_gltf_scale: FloatProperty(name="--gltf-scale", default=1.0, min=0.0001, max=100000.0)
    patch_only_selected_meshes: BoolProperty(name="Use selected meshes as --only-mesh", default=True)
    patch_only_mesh_csv: StringProperty(name="--only-mesh (csv override)", default="")

    # True export + pivot fix
    true_template_vfx: StringProperty(name="Pivot Template VFX", subtype="FILE_PATH", default="")
    true_gltf_in: StringProperty(name="glTF In", subtype="FILE_PATH", default="")
    true_vfx_out: StringProperty(name="TrueExport VFX Out", subtype="FILE_PATH", default="")
    true_gltf_scale: FloatProperty(name="--gltf-scale", default=1.0, min=0.0001, max=100000.0)
    true_anchor: StringProperty(name="--anchor (optional)", default="flagpole")
    true_apply_pivot_fix: BoolProperty(name="Apply pivot×key0 fix (recommended)", default=True)

class RFVFX_OT_VfxToGltf(bpy.types.Operator):
    bl_idname = "rfvfx.vfx_to_gltf"
    bl_label = "VFX -> glTF (and import)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _prefs()
        repo = _repo_root()
        if not repo or not os.path.exists(repo):
            self.report({"ERROR"}, "Set Repo Root in add-on preferences.")
            return {"CANCELLED"}

        vfx2obj = os.path.join(repo, "vfx2obj.py")
        if not os.path.exists(vfx2obj):
            self.report({"ERROR"}, "vfx2obj.py not found under Repo Root.")
            return {"CANCELLED"}

        s = context.scene.rfvfx
        vfx = bpy.path.abspath(s.import_vfx)
        if not vfx or not os.path.exists(vfx):
            self.report({"ERROR"}, "Pick a valid VFX file.")
            return {"CANCELLED"}

        cmd = [_python_exe(), vfx2obj, "--gltf", "--scale", str(s.import_scale)]
        if s.import_debug:
            cmd.append("--debug-frames")
        if s.import_anchor.strip():
            cmd += ["--anchor", s.import_anchor.strip()]
        cmd.append(vfx)

        _run(cmd, cwd=repo)

        gltf = os.path.splitext(vfx)[0] + ".gltf"
        if os.path.exists(gltf):
            bpy.ops.import_scene.gltf(filepath=gltf)

        self.report({"INFO"}, "Done. Check system console for logs.")
        return {"FINISHED"}

class RFVFX_OT_ExportGltf(bpy.types.Operator):
    bl_idname = "rfvfx.export_gltf"
    bl_label = "Export glTF (for VFX workflow)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.rfvfx
        outp = bpy.path.abspath(s.export_gltf).strip()
        if not outp:
            self.report({"ERROR"}, "Set an Export glTF path.")
            return {"CANCELLED"}

        out_dir = os.path.dirname(outp)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        bpy.ops.export_scene.gltf(
            filepath=outp,
            export_format="GLTF_SEPARATE",
            use_selection=bool(s.export_selected_only),
            export_apply=bool(s.export_apply),
        )

        self.report({"INFO"}, "glTF exported.")
        return {"FINISHED"}

class RFVFX_OT_PatchFromGltf(bpy.types.Operator):
    bl_idname = "rfvfx.patch_from_gltf"
    bl_label = "Patch VFX from glTF"
    bl_options = {"REGISTER"}

    def execute(self, context):
        repo = _repo_root()
        if not repo or not os.path.exists(repo):
            self.report({"ERROR"}, "Set Repo Root in add-on preferences.")
            return {"CANCELLED"}

        vfx2obj = os.path.join(repo, "vfx2obj.py")
        if not os.path.exists(vfx2obj):
            self.report({"ERROR"}, "vfx2obj.py not found under Repo Root.")
            return {"CANCELLED"}

        s = context.scene.rfvfx
        tmpl = bpy.path.abspath(s.patch_template_vfx).strip()
        gltf = bpy.path.abspath(s.patch_gltf_in).strip()
        outv = bpy.path.abspath(s.patch_vfx_out).strip()

        if not (tmpl and os.path.exists(tmpl)):
            self.report({"ERROR"}, "Pick a valid Template VFX.")
            return {"CANCELLED"}
        if not (gltf and os.path.exists(gltf)):
            self.report({"ERROR"}, "Pick a valid glTF In.")
            return {"CANCELLED"}

        if not outv:
            outv = _default_out_vfx_from_gltf(gltf, ".patched")
            s.patch_vfx_out = outv

        out_dir = os.path.dirname(outv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        only_mesh = ""
        if s.patch_only_mesh_csv.strip():
            only_mesh = s.patch_only_mesh_csv.strip()
        elif s.patch_only_selected_meshes:
            sel = _selected_mesh_names()
            if sel:
                only_mesh = ",".join(sel)

        cmd = [
            _python_exe(), vfx2obj,
            "--vfx-out", outv,
            "--patch-vfx-only",
            "--gltf-in", gltf,
            "--gltf-scale", str(s.patch_gltf_scale),
        ]
        if only_mesh:
            cmd += ["--only-mesh", only_mesh]
        cmd.append(tmpl)

        _run(cmd, cwd=repo)
        self.report({"INFO"}, "Patched VFX written.")
        return {"FINISHED"}

class RFVFX_OT_TrueExportPivotFix(bpy.types.Operator):
    bl_idname = "rfvfx.trueexport_pivotfix"
    bl_label = "TrueExport + PivotFix"
    bl_options = {"REGISTER"}

    def execute(self, context):
        repo = _repo_root()
        if not repo or not os.path.exists(repo):
            self.report({"ERROR"}, "Set Repo Root in add-on preferences.")
            return {"CANCELLED"}

        vfx2obj = os.path.join(repo, "vfx2obj.py")
        pivot_tool = os.path.join(repo, "tools", "pivot_patch_xkey0.py")
        if not (os.path.exists(vfx2obj) and os.path.exists(pivot_tool)):
            self.report({"ERROR"}, "Missing vfx2obj.py or tools\\pivot_patch_xkey0.py")
            return {"CANCELLED"}

        s = context.scene.rfvfx
        tmpl = bpy.path.abspath(s.true_template_vfx).strip()
        gltf = bpy.path.abspath(s.true_gltf_in).strip()
        outv = bpy.path.abspath(s.true_vfx_out).strip()

        if not (tmpl and os.path.exists(tmpl)):
            self.report({"ERROR"}, "Pick a valid Pivot Template VFX.")
            return {"CANCELLED"}
        if not (gltf and os.path.exists(gltf)):
            self.report({"ERROR"}, "Pick a valid glTF In.")
            return {"CANCELLED"}

        if not outv:
            outv = _default_out_vfx_from_gltf(gltf, ".trueexport")
            s.true_vfx_out = outv

        out_dir = os.path.dirname(outv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        tmpv = os.path.splitext(outv)[0] + ".__tmp_trueexport.vfx"

        # True export
        cmd1 = [
            _python_exe(), vfx2obj,
            "--new-vfx-from-gltf", gltf,
            "--vfx-out", tmpv,
            "--gltf-scale", str(s.true_gltf_scale),
        ]
        if s.true_anchor.strip():
            cmd1 += ["--anchor", s.true_anchor.strip()]
        _run(cmd1, cwd=repo)

        if s.true_apply_pivot_fix:
            cmd2 = [
                _python_exe(), pivot_tool,
                "--template", tmpl,
                "--in", tmpv,
                "--out", outv,
            ]
            _run(cmd2, cwd=repo)
            try:
                os.remove(tmpv)
            except Exception:
                pass
        else:
            # no pivot fix requested
            try:
                if os.path.exists(outv):
                    os.remove(outv)
            except Exception:
                pass
            os.replace(tmpv, outv)

        self.report({"INFO"}, "TrueExport written (pivot fix applied if enabled).")
        return {"FINISHED"}

class RFVFX_PT_Panel(bpy.types.Panel):
    bl_label = "RF VFX"
    bl_idname = "RFVFX_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RF VFX"

    def draw(self, context):
        s = context.scene.rfvfx
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="Import")
        col.prop(s, "import_vfx")
        row = col.row(align=True)
        row.prop(s, "import_scale")
        row.prop(s, "import_debug")
        col.prop(s, "import_anchor")
        col.operator("rfvfx.vfx_to_gltf", icon="IMPORT")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Export glTF from Blender")
        col.prop(s, "export_gltf")
        row = col.row(align=True)
        row.prop(s, "export_selected_only")
        row.prop(s, "export_apply")
        col.operator("rfvfx.export_gltf", icon="EXPORT")

        layout.separator()

        box = layout.box()
        box.label(text="Patch VFX from glTF (topology must match)")
        box.prop(s, "patch_template_vfx")
        box.prop(s, "patch_gltf_in")
        box.prop(s, "patch_vfx_out")
        box.prop(s, "patch_gltf_scale")
        box.prop(s, "patch_only_selected_meshes")
        box.prop(s, "patch_only_mesh_csv")
        box.operator("rfvfx.patch_from_gltf", icon="FILE_TICK")

        layout.separator()

        box2 = layout.box()
        box2.label(text="TrueExport + PivotFix (recommended)")
        box2.prop(s, "true_template_vfx")
        box2.prop(s, "true_gltf_in")
        box2.prop(s, "true_vfx_out")
        box2.prop(s, "true_gltf_scale")
        box2.prop(s, "true_anchor")
        box2.prop(s, "true_apply_pivot_fix")
        box2.operator("rfvfx.trueexport_pivotfix", icon="FILE_TICK")

_classes = (
    RFVFX_Prefs,
    RFVFX_Settings,
    RFVFX_OT_VfxToGltf,
    RFVFX_OT_ExportGltf,
    RFVFX_OT_PatchFromGltf,
    RFVFX_OT_TrueExportPivotFix,
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
