bl_info = {
    "name": "RF VFX Tools (vfx2obj wrapper)",
    "author": "RomekRF + ChatGPT",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > RF VFX",
    "description": "Wraps vfx2obj.py workflow: VFX -> glTF, then glTF -> VFX (patch or trueexport+pivot fix).",
    "category": "Import-Export",
}

import bpy
import os
import subprocess
from bpy.props import StringProperty, BoolProperty, FloatProperty

def _run(cmd, cwd=None):
    # Blender console will show output; also prints into Info area
    print("[RF VFX] RUN:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}). See Blender system console for output.")
    return p.stdout

def _addon_prefs():
    return bpy.context.preferences.addons[__name__].preferences

class RFVFX_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__

    repo_root: StringProperty(
        name="Repo Root",
        description="Folder containing vfx2obj.py (e.g. C:\\Users\\Romek\\OneDrive\\Desktop\\vfx2obj_build)",
        default="",
        subtype="DIR_PATH",
    )

    python_exe: StringProperty(
        name="Python Executable",
        description="System Python to run vfx2obj.py (e.g. python or C:\\Python313\\python.exe)",
        default="python",
        subtype="FILE_PATH",
    )

    gltf_scale: FloatProperty(
        name="glTF Scale",
        description="Passed to vfx2obj.py where applicable",
        default=1.0,
        min=0.0001,
        max=1000.0,
    )

    auto_import_gltf: BoolProperty(
        name="Auto-import glTF after VFX->glTF",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "repo_root")
        layout.prop(self, "python_exe")
        layout.prop(self, "gltf_scale")
        layout.prop(self, "auto_import_gltf")

class RFVFX_OT_VfxToGltf(bpy.types.Operator):
    bl_idname = "rfvfx.vfx_to_gltf"
    bl_label = "VFX -> glTF (and import)"
    bl_options = {"REGISTER"}

    vfx_path: StringProperty(name="VFX File", subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        prefs = _addon_prefs()
        repo = bpy.path.abspath(prefs.repo_root)
        if not repo or not os.path.exists(repo):
            self.report({"ERROR"}, "Set Repo Root in add-on preferences.")
            return {"CANCELLED"}

        py = bpy.path.abspath(prefs.python_exe) or "python"
        vfx = bpy.path.abspath(self.vfx_path)
        if not os.path.exists(vfx):
            self.report({"ERROR"}, "VFX path does not exist.")
            return {"CANCELLED"}

        vfx2obj = os.path.join(repo, "vfx2obj.py")
        if not os.path.exists(vfx2obj):
            self.report({"ERROR"}, "vfx2obj.py not found under Repo Root.")
            return {"CANCELLED"}

        # vfx2obj writes <samebase>.gltf/.bin next to the VFX (per current workflow)
        cmd = [py, vfx2obj, "--gltf", "--scale", str(prefs.gltf_scale), vfx]
        _run(cmd, cwd=repo)

        gltf = os.path.splitext(vfx)[0] + ".gltf"
        if prefs.auto_import_gltf and os.path.exists(gltf):
            bpy.ops.import_scene.gltf(filepath=gltf)

        self.report({"INFO"}, "Done. Check system console for logs.")
        return {"FINISHED"}

class RFVFX_OT_PatchFromGltf(bpy.types.Operator):
    bl_idname = "rfvfx.patch_from_gltf"
    bl_label = "glTF -> VFX (patch template)"
    bl_options = {"REGISTER"}

    template_vfx: StringProperty(name="Template VFX", subtype="FILE_PATH")
    gltf_in: StringProperty(name="glTF In", subtype="FILE_PATH")
    vfx_out: StringProperty(name="VFX Out", subtype="FILE_PATH")
    only_mesh: StringProperty(name="Only Mesh (optional)", default="")

    def execute(self, context):
        prefs = _addon_prefs()
        repo = bpy.path.abspath(prefs.repo_root)
        py = bpy.path.abspath(prefs.python_exe) or "python"
        vfx2obj = os.path.join(repo, "vfx2obj.py")

        tmpl = bpy.path.abspath(self.template_vfx)
        gltf = bpy.path.abspath(self.gltf_in)
        outv = bpy.path.abspath(self.vfx_out)

        if not (os.path.exists(vfx2obj) and os.path.exists(tmpl) and os.path.exists(gltf)):
            self.report({"ERROR"}, "Missing vfx2obj.py / template VFX / input glTF.")
            return {"CANCELLED"}

        cmd = [py, vfx2obj, "--vfx-out", outv, "--patch-vfx-only", "--gltf-in", gltf]
        if self.only_mesh.strip():
            cmd += ["--only-mesh", self.only_mesh.strip()]
        cmd += [tmpl]
        _run(cmd, cwd=repo)

        self.report({"INFO"}, "Patched VFX written.")
        return {"FINISHED"}

class RFVFX_OT_TrueExportWithPivotFix(bpy.types.Operator):
    bl_idname = "rfvfx.trueexport_pivotfix"
    bl_label = "glTF -> VFX (trueexport + pivot fix)"
    bl_options = {"REGISTER"}

    template_vfx: StringProperty(name="Template VFX (for pivot)", subtype="FILE_PATH")
    gltf_in: StringProperty(name="glTF In", subtype="FILE_PATH")
    vfx_out: StringProperty(name="VFX Out", subtype="FILE_PATH")

    def execute(self, context):
        prefs = _addon_prefs()
        repo = bpy.path.abspath(prefs.repo_root)
        py = bpy.path.abspath(prefs.python_exe) or "python"
        vfx2obj = os.path.join(repo, "vfx2obj.py")
        pivot_tool = os.path.join(repo, "tools", "pivot_patch_xkey0.py")

        tmpl = bpy.path.abspath(self.template_vfx)
        gltf = bpy.path.abspath(self.gltf_in)
        outv = bpy.path.abspath(self.vfx_out)

        if not (os.path.exists(vfx2obj) and os.path.exists(pivot_tool) and os.path.exists(tmpl) and os.path.exists(gltf)):
            self.report({"ERROR"}, "Missing vfx2obj.py / pivot tool / template VFX / input glTF.")
            return {"CANCELLED"}

        tmpv = os.path.splitext(outv)[0] + ".__tmp_trueexport.vfx"

        # True export
        cmd1 = [py, vfx2obj, "--new-vfx-from-gltf", gltf, "--vfx-out", tmpv, "--gltf-scale", str(prefs.gltf_scale)]
        _run(cmd1, cwd=repo)

        # Pivot fix (proven rule: pivot_t * key0_scale)
        cmd2 = [py, pivot_tool, "--template", tmpl, "--in", tmpv, "--out", outv]
        _run(cmd2, cwd=repo)

        try:
            os.remove(tmpv)
        except Exception:
            pass

        self.report({"INFO"}, "Trueexport+PivotFix written.")
        return {"FINISHED"}

class RFVFX_PT_Panel(bpy.types.Panel):
    bl_label = "RF VFX"
    bl_idname = "RFVFX_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RF VFX"

    def draw(self, context):
        layout = self.layout
        layout.operator("rfvfx.vfx_to_gltf", icon="IMPORT")

        layout.separator()
        box = layout.box()
        box.label(text="Patch Template VFX from glTF")
        op = box.operator("rfvfx.patch_from_gltf", text="Run Patch", icon="FILE_TICK")
        # user fills args in redo panel (F9) for now

        layout.separator()
        box2 = layout.box()
        box2.label(text="Trueexport + PivotFix (needs template)")
        op2 = box2.operator("rfvfx.trueexport_pivotfix", text="Run Trueexport+Fix", icon="FILE_TICK")

_classes = (
    RFVFX_Prefs,
    RFVFX_OT_VfxToGltf,
    RFVFX_OT_PatchFromGltf,
    RFVFX_OT_TrueExportWithPivotFix,
    RFVFX_PT_Panel,
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
