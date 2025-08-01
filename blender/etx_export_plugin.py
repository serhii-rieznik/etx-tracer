bl_info = {
    "name": "ETX Tracer Scene Exporter",
    "author": "ETX Tracer Community",
    "version": (2, 0, 0),
    "blender": (4, 4, 0),
    "location": "File > Export",
    "description": "Export complete ETX Tracer scene (JSON + OBJ + Materials)",
    "doc_url": "",
    "tracker_url": "",
    "category": "Import-Export",
    "support": "COMMUNITY",
}

import bpy
import bmesh
import os
import json
import math
from mathutils import Vector, Matrix
from bpy_extras.io_utils import ExportHelper
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
)
from bpy.types import Operator


class ExportETXTracer(Operator, ExportHelper):
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
        default=1024,
        min=1,
        max=100000,
    )

    max_path_length: IntProperty(
        name="Max Path Length",
        description="Maximum path length for path tracing",
        default=16,
        min=1,
        max=100,
    )

    random_termination_start: IntProperty(
        name="Random Termination Start",
        description="Depth at which Russian roulette termination begins",
        default=8,
        min=1,
        max=50,
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
        """Main export function - creates JSON scene + OBJ geometry + ETX materials"""

        # === Generate file paths ===
        base_path = os.path.splitext(self.filepath)[0]
        json_path = self.filepath  # Main scene file
        obj_path = base_path + ".obj"
        materials_path = base_path + ".materials"

        try:
            # === Export geometry to OBJ ===
            self._export_obj(obj_path)

            # === Export materials to ETX format ===
            self._export_materials(materials_path)

            # === Export scene settings to JSON ===
            self._export_scene_json(json_path, obj_path, materials_path)

            self.report({"INFO"}, f"Successfully exported ETX scene to {json_path}")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Export failed: {str(e)}")
            return {"CANCELLED"}

    def _export_obj(self, obj_path):
        """Export geometry to OBJ file with material assignments"""
        if not hasattr(bpy.ops.wm, "obj_export"):
            raise Exception(
                "Wavefront OBJ export operator not found in this Blender version"
            )

        export_args = {
            "filepath": obj_path,
            "export_selected_objects": self.export_selected,
            "export_uv": True,
            "export_normals": True,
            "export_materials": True,  # Keep material assignments in OBJ
            "global_scale": 1.0,
            "forward_axis": "NEGATIVE_Z",  # -Z forward (into screen)
            "up_axis": "Y",  # Y-up orientation for renderer
        }

        bpy.ops.wm.obj_export(**export_args)

        # The MTL file will be created but we'll replace it with our custom format
        # in _export_materials(), so we don't remove it here

    def _export_materials(self, materials_path):
        """Export materials in ETX format, replacing the standard MTL file"""

        # Check if standard MTL file was created by OBJ export
        obj_path = materials_path.replace(".materials", ".obj")
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"

        # Get materials that were actually used in the OBJ export
        materials_to_export = self._get_materials_to_export()

        # If standard MTL exists, read it to get the material names that were actually exported
        exported_material_names = set()
        if os.path.exists(mtl_path):
            with open(mtl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("newmtl "):
                        mat_name = line.strip().split(" ", 1)[1]
                        exported_material_names.add(mat_name)

        # Filter materials to only those that were actually exported
        if exported_material_names:
            materials_to_export = [
                mat
                for mat in materials_to_export
                if mat and mat.name in exported_material_names
            ]

        materials_data = []
        for mat in materials_to_export:
            if mat is None:
                continue

            mat_data = self._convert_material_to_etx(mat)
            materials_data.append(mat_data)

        # Write ETX materials format
        def write_materials_file(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                for mat_data in materials_data:
                    f.write(f"newmtl {mat_data['name']}\n")

                    # Write ETX material properties
                    for key, value in mat_data["properties"].items():
                        if isinstance(value, (list, tuple)):
                            f.write(f"{key} {' '.join(map(str, value))}\n")
                        else:
                            f.write(f"{key} {value}\n")

                    f.write("\n")  # Blank line between materials

        # Replace the standard MTL file with ETX format (for OBJ compatibility)
        if os.path.exists(mtl_path):
            write_materials_file(mtl_path)

        # Also create the .materials file if it's different from MTL path
        if materials_path != mtl_path:
            write_materials_file(materials_path)

    def _export_scene_json(self, json_path, obj_path, materials_path):
        """Export scene settings to JSON file"""
        # Get relative paths for JSON references
        base_dir = os.path.dirname(json_path)
        rel_obj_path = os.path.relpath(obj_path, base_dir)

        # Reference the MTL file that matches the OBJ (standard format)
        # since we replace its contents with ETX format
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        rel_mtl_path = os.path.relpath(mtl_path, base_dir)

        # Build scene data
        scene_data = {
            "geometry": rel_obj_path.replace("\\", "/"),  # Use forward slashes
            "materials": rel_mtl_path.replace("\\", "/"),  # Reference MTL file
            "samples": self.samples,
            "max-path-length": self.max_path_length,
            "random-termination-start": self.random_termination_start,
            "spectral": self.spectral_rendering,
            "force-tangents": self.force_tangents,
            "camera": self._get_camera_data(),
        }

        # Write JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2)

    def _get_camera_data(self):
        """Extract camera data from Blender scene"""
        context = bpy.context
        scene = context.scene

        if self.use_active_camera and scene.camera:
            camera_obj = scene.camera
            camera_data = camera_obj.data

            # Get camera transform in Blender's coordinate system (Z-up)
            matrix_world = camera_obj.matrix_world
            location_blender = matrix_world.translation

            # Calculate target point (camera forward direction) in Blender coordinates
            forward_blender = matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
            target_blender = (
                location_blender + forward_blender * 10.0
            )  # 10 unit distance

            # Get up vector in Blender coordinates
            up_blender = matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))

            # Convert from Blender's Z-up to Y-up coordinate system (matching OBJ export)
            # Blender: X=right, Y=forward, Z=up
            # Y-up:    X=right, Y=up,      Z=back
            # Transformation: (x, y, z) -> (x, z, -y)
            location = Vector(
                (location_blender.x, location_blender.z, -location_blender.y)
            )
            target = Vector((target_blender.x, target_blender.z, -target_blender.y))
            up = Vector((up_blender.x, up_blender.z, -up_blender.y))

            # Get viewport from render settings
            render = scene.render
            viewport = [render.resolution_x, render.resolution_y]

            # Get field of view
            if camera_data.type == "PERSP":
                # Ensure we always export horizontal FOV, converting if necessary
                # We can use Blender's internal Camera 'angle_x' and 'angle_y' for direct conversion

                # Blender's angle_x and angle_y are calculated based on sensor_fit
                # This gives us a reliable way to get both horizontal and vertical FOV
                horizontal_fov = math.degrees(camera_data.angle_x)
                vertical_fov = math.degrees(camera_data.angle_y)

                fov = horizontal_fov

                focal_length = camera_data.lens
            else:
                # For orthographic cameras, default FOV
                fov = 50.0  # Default horizontal FOV
                focal_length = 50.0

        else:
            # Use viewport camera
            area = None
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    break

            if area and area.spaces[0].region_3d:
                rv3d = area.spaces[0].region_3d
                view_matrix = rv3d.view_matrix

                # Extract camera data from viewport (in Blender Z-up coordinates)
                location_blender = view_matrix.inverted().translation
                target_blender = (
                    location_blender
                    + (
                        view_matrix.inverted().to_quaternion()
                        @ Vector((0.0, 0.0, -1.0))
                    )
                    * 10.0
                )
                up_blender = view_matrix.inverted().to_quaternion() @ Vector(
                    (0.0, 1.0, 0.0)
                )

                # Convert to Y-up coordinate system
                location = Vector(
                    (location_blender.x, location_blender.z, -location_blender.y)
                )
                target = Vector((target_blender.x, target_blender.z, -target_blender.y))
                up = Vector((up_blender.x, up_blender.z, -up_blender.y))

                viewport = [1920, 1080]  # Default viewport
                # Estimate viewport FOV - Blender viewport typically uses vertical FOV
                estimated_vertical_fov = math.degrees(
                    rv3d.view_camera_zoom * 0.02
                )  # Approximate
                aspect_ratio = viewport[0] / viewport[1]
                estimated_horizontal_fov_rad = 2.0 * math.atan(
                    math.tan(math.radians(estimated_vertical_fov) / 2.0) * aspect_ratio
                )
                fov = math.degrees(estimated_horizontal_fov_rad)

                focal_length = 50.0
            else:
                # Fallback default camera (already in Y-up coordinates)
                location = Vector(
                    (7.4, 5.3, 6.5)
                )  # Adjusted for Y-up: (x, y_up, z_back)
                target = Vector((0.0, 0.0, 0.0))
                up = Vector((0.0, 1.0, 0.0))  # Y is up
                viewport = [1920, 1080]
                # Default FOV based on user choice
                fov = 50.0  # Always horizontal
                focal_length = 50.0

        return {
            "class": self.camera_class,
            "viewport": viewport,
            "origin": [location.x, location.y, location.z],
            "target": [target.x, target.y, target.z],
            "up": [up.x, up.y, up.z],
            "fov": fov,
            "focal-length": focal_length,
            "lens-radius": self.lens_radius,
            "focal-distance": self.focal_distance,
        }

    def _get_materials_to_export(self):
        """Get list of materials to export based on selection"""
        materials = set()

        if self.export_selected:
            # Get materials from selected objects
            for obj in bpy.context.selected_objects:
                if obj.type == "MESH" and obj.data.materials:
                    for mat in obj.data.materials:
                        if mat is not None:
                            materials.add(mat)
        else:
            # Get all materials in scene
            materials = set(bpy.data.materials)

        return list(materials)

    def _convert_material_to_etx(self, blender_mat):
        """Convert Blender material to ETX format"""
        mat_data = {"name": blender_mat.name, "properties": {}}

        # Get material type from custom properties or infer from nodes
        etx_class = self._get_etx_material_class(blender_mat)
        mat_data["properties"]["material"] = f"class {etx_class}"

        # Convert Principled BSDF or other shader nodes
        if blender_mat.use_nodes and blender_mat.node_tree:
            self._extract_node_properties(blender_mat.node_tree, mat_data["properties"])
        else:
            # Fallback to basic material properties
            self._extract_basic_properties(blender_mat, mat_data["properties"])

        # Add custom properties from Blender material
        for key, value in blender_mat.items():
            if not key.startswith("_"):
                mat_data["properties"][f"# ETXProperty {key}"] = str(value)

        return mat_data

    def _get_node_input_value(self, node, input_name, default_value=0.0):
        """Safely get input value from node, returns default if input doesn't exist"""
        try:
            if input_name in node.inputs:
                return node.inputs[input_name].default_value
            else:
                # Try common alternative names
                alt_names = {
                    "Transmission": ["Transmission Weight", "Transmission Factor"],
                    "Emission": ["Emission Color"],
                    "Emission Strength": ["Emission Weight"],
                    "Base Color": ["Albedo", "Diffuse Color"],
                }

                if input_name in alt_names:
                    for alt_name in alt_names[input_name]:
                        if alt_name in node.inputs:
                            return node.inputs[alt_name].default_value

                return default_value
        except (KeyError, AttributeError):
            return default_value

    def _get_etx_material_class(self, blender_mat):
        """Determine ETX material class from Blender material"""
        # Check custom property first
        if "etx_class" in blender_mat:
            return blender_mat["etx_class"]

        # Infer from material properties
        if blender_mat.use_nodes and blender_mat.node_tree:
            principled = None
            for node in blender_mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    principled = node
                    break

            if principled:
                # Check metallic value safely
                metallic = self._get_node_input_value(principled, "Metallic", 0.0)
                transmission = self._get_node_input_value(
                    principled, "Transmission", 0.0
                )

                if metallic > 0.8:
                    return "conductor"
                elif transmission > 0.1:
                    return "dielectric"
                else:
                    return "plastic"

        return "diffuse"  # Default fallback

    def _extract_node_properties(self, node_tree, properties):
        """Extract properties from Blender shader nodes"""
        principled = None
        for node in node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                principled = node
                break

        if not principled:
            return

        # Base color - safely get color value
        base_color = self._get_node_input_value(
            principled, "Base Color", [1.0, 1.0, 1.0, 1.0]
        )
        if hasattr(base_color, "__len__") and len(base_color) >= 3:
            properties["Kd"] = (
                f"{base_color[0]:.3f} {base_color[1]:.3f} {base_color[2]:.3f}"
            )
        else:
            properties["Kd"] = "0.800 0.800 0.800"  # Fallback

        # Roughness
        roughness = self._get_node_input_value(principled, "Roughness", 0.5)
        properties["Pr"] = f"{roughness:.3f}"

        # Metallic
        metallic = self._get_node_input_value(principled, "Metallic", 0.0)
        if metallic > 0.0:
            if hasattr(base_color, "__len__") and len(base_color) >= 3:
                properties["Ks"] = (
                    f"{base_color[0]:.3f} {base_color[1]:.3f} {base_color[2]:.3f}"
                )
        else:
            properties["Ks"] = f"{metallic:.3f} {metallic:.3f} {metallic:.3f}"

        # IOR
        ior = self._get_node_input_value(principled, "IOR", 1.45)
        properties["int_ior"] = f"{ior:.3f}"

        # Transmission
        transmission = self._get_node_input_value(principled, "Transmission", 0.0)
        if transmission > 0.0:
            properties["Kt"] = (
                f"{transmission:.3f} {transmission:.3f} {transmission:.3f}"
            )

        # Emission
        emission = self._get_node_input_value(
            principled, "Emission", [0.0, 0.0, 0.0, 1.0]
        )
        emission_strength = self._get_node_input_value(
            principled, "Emission Strength", 1.0
        )

        if (
            emission_strength > 0.0
            and hasattr(emission, "__len__")
            and len(emission) >= 3
        ):
            # Check if emission color has meaningful values
            if emission[0] > 0.0 or emission[1] > 0.0 or emission[2] > 0.0:
                properties["Ke"] = (
                    f"{emission[0] * emission_strength:.3f} {emission[1] * emission_strength:.3f} {emission[2] * emission_strength:.3f}"
                )

        # Check for connected texture nodes
        self._extract_texture_connections(principled, properties)

    def _extract_texture_connections(self, principled_node, properties):
        """Extract texture file connections from Principled BSDF"""
        # Base Color texture
        if self._input_is_linked(principled_node, "Base Color"):
            texture_node = self._find_texture_node(principled_node.inputs["Base Color"])
            if texture_node and texture_node.image:
                properties["map_Kd"] = texture_node.image.name

        # Normal map
        if self._input_is_linked(principled_node, "Normal"):
            normal_map = self._find_normal_map_node(principled_node.inputs["Normal"])
            if normal_map and normal_map.image:
                properties["normalmap"] = f"image {normal_map.image.name} scale 1.0"

        # Roughness texture
        if self._input_is_linked(principled_node, "Roughness"):
            texture_node = self._find_texture_node(principled_node.inputs["Roughness"])
            if texture_node and texture_node.image:
                properties["map_Pr"] = texture_node.image.name

    def _input_is_linked(self, node, input_name):
        """Safely check if a node input is linked"""
        try:
            if input_name in node.inputs:
                return node.inputs[input_name].is_linked
            return False
        except (KeyError, AttributeError):
            return False

    def _find_texture_node(self, socket):
        """Find connected Image Texture node"""
        if not socket.is_linked:
            return None

        from_node = socket.links[0].from_node
        if from_node.type == "TEX_IMAGE":
            return from_node

        # Handle ColorRamp or other intermediate nodes
        for input_socket in from_node.inputs:
            if input_socket.is_linked:
                result = self._find_texture_node(input_socket)
                if result:
                    return result

        return None

    def _find_normal_map_node(self, socket):
        """Find connected Normal Map node with Image Texture"""
        if not socket.is_linked:
            return None

        from_node = socket.links[0].from_node
        if from_node.type == "NORMAL_MAP":
            color_input = from_node.inputs["Color"]
            if color_input.is_linked:
                texture_node = color_input.links[0].from_node
                if texture_node.type == "TEX_IMAGE":
                    return texture_node

        return None

    def _extract_basic_properties(self, blender_mat, properties):
        """Extract basic material properties for non-node materials"""
        # Diffuse color
        if hasattr(blender_mat, "diffuse_color"):
            color = blender_mat.diffuse_color
            properties["Kd"] = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"

        # Roughness
        if hasattr(blender_mat, "roughness"):
            properties["Pr"] = f"{blender_mat.roughness:.3f}"

        # Metallic
        if hasattr(blender_mat, "metallic"):
            metallic = blender_mat.metallic
            if metallic > 0.0:
                properties["Ks"] = f"{metallic:.3f} {metallic:.3f} {metallic:.3f}"


def menu_func_export(self, context):
    self.layout.operator(ExportETXTracer.bl_idname, text="ETX Tracer Scene (.json)")


def register():
    bpy.utils.register_class(ExportETXTracer)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportETXTracer)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()
