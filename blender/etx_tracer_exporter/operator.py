import bpy
from bpy_extras.io_utils import ExportHelper
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
)
from bpy.types import Operator
from . import logic
import importlib


class ExportETXTracer(Operator, ExportHelper):
    """Operator to export ETX Tracer scenes."""

    bl_idname = "export_scene.etx_tracer"
    bl_label = "Export ETX Tracer Scene"

    filename_ext = ".json"

    filter_glob: StringProperty(
        default="*.json",
        options={"HIDDEN"},
        maxlen=255,
    )

    # === Export Options ===
    export_selected: BoolProperty(
        name="Export Selected Objects Only",
        description="Export only selected objects instead of the entire scene",
        default=False,
    )

    # === Render Settings ===
    samples: IntProperty(
        name="Samples",
        description="Number of samples for Monte Carlo integration",
        default=32,
        min=1,
        max=65536,
    )

    # === Texture Export ===
    export_textures: BoolProperty(
        name="Export Textures",
        description="Copy/save used images (including packed) next to OBJ",
        default=True,
    )

    texture_subdir: StringProperty(
        name="Texture Subfolder",
        description="Relative subfolder for textures (empty = alongside OBJ)",
        default="textures",
        maxlen=255,
    )

    # === Procedural Baking ===
    bake_procedural: BoolProperty(
        name="Bake Procedural Materials",
        description="Bake node-based materials to textures (Base Color, Normal)",
        default=False,
    )

    bake_resolution: EnumProperty(
        name="Bake Resolution",
        description="Output resolution for baked textures",
        items=[
            ("512", "512", "512x512"),
            ("1024", "1024", "1024x1024"),
            ("2048", "2048", "2048x2048"),
            ("4096", "4096", "4096x4096"),
        ],
        default="2048",
    )

    bake_margin: IntProperty(
        name="Bake Margin (px)",
        description="Bake dilation margin in pixels",
        default=4,
        min=0,
        max=64,
    )

    max_path_length: IntProperty(
        name="Max Path Length",
        description="Maximum path length for path tracing",
        default=65536,
        min=1,
        max=65536,
    )

    random_termination_start: IntProperty(
        name="Random Termination Start",
        description="Depth at which Russian roulette termination begins",
        default=6,
        min=1,
        max=65536,
    )

    spectral_rendering: BoolProperty(
        name="Spectral Rendering",
        description="Enable spectral rendering instead of RGB",
        default=False,
    )

    force_tangents: BoolProperty(
        name="Force Tangents Recalculation",
        description="Force recalculation of tangent vectors",
        default=False,
    )

    two_sided_materials: BoolProperty(
        name="Two-Sided Surfaces",
        description="Make opaque surfaces (diffuse/plastic/conductor/velvet) two-sided",
        default=True,
    )

    # === Camera Settings ===
    camera_class: EnumProperty(
        name="Camera Type",
        description="Camera projection type",
        items=[
            ("perspective", "Perspective", "Standard perspective camera"),
            ("eq", "Equirectangular", "360-degree equirectangular camera"),
        ],
        default="perspective",
    )

    use_active_camera: BoolProperty(
        name="Use Active Camera",
        description="Use active camera settings, otherwise use view settings",
        default=True,
    )

    # === Depth of Field ===
    lens_radius: FloatProperty(
        name="Lens Radius",
        description="Lens radius for depth of field (0 = no DOF)",
        default=0.0,
        min=0.0,
        max=10.0,
        precision=4,
    )

    focal_distance: FloatProperty(
        name="Focal Distance",
        description="Distance to focal plane",
        default=0.0,
        min=0.0,
        max=1000.0,
        precision=3,
    )

    def execute(self, context):
        """
        This is the main execution function.
        It reloads the exporter logic and then runs the main export function.
        """
        try:
            importlib.reload(logic)
            return logic.main_export(self, context)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to run exporter: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"CANCELLED"}


def menu_func_export(self, context):
    self.layout.operator(ExportETXTracer.bl_idname, text="ETX Tracer Scene (.json)")


classes = (ExportETXTracer,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
