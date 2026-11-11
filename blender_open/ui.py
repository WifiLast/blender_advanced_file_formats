import bpy


class IO_FH_nwd(bpy.types.FileHandler):
    bl_idname = "IO_FH_nwd"
    bl_label = "Navisworks"
    bl_import_operator = "import_scene.nwd"
    bl_file_extensions = ".nwd"

    @classmethod
    def poll_drop(cls, context):
        return True


class IO_FH_rvt(bpy.types.FileHandler):
    bl_idname = "IO_FH_rvt"
    bl_label = "Revit"
    bl_import_operator = "import_scene.rvt"
    bl_file_extensions = ".rvt"

    @classmethod
    def poll_drop(cls, context):
        return True


def menu_func_import(self, _context):
    self.layout.operator("import_scene.nwd", text="Navisworks (.nwd)")
    self.layout.operator("import_scene.rvt", text="Revit (.rvt)")
