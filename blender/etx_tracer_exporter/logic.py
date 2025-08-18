import bpy
import os
import json
import math
from mathutils import Vector


def main_export(operator, context):
    """Main export function - creates JSON scene + OBJ geometry + ETX materials"""

    # === Generate file paths ===
    base_path = os.path.splitext(operator.filepath)[0]
    json_path = operator.filepath  # Main scene file
    obj_path = base_path + ".obj"

    try:
        # === Export geometry to OBJ ===
        _export_obj(operator, obj_path)

        # === Export materials to ETX format ===
        _export_materials(operator, obj_path)

        # === Export scene settings to JSON ===
        _export_scene_json(operator, json_path, obj_path)

        operator.report({"INFO"}, f"Successfully exported ETX scene to {json_path}")
        return {"FINISHED"}

    except Exception as e:
        operator.report({"ERROR"}, f"Export failed: {str(e)}")
        return {"CANCELLED"}


def _export_obj(operator, obj_path):
    """Export geometry to OBJ file with material assignments"""
    if not hasattr(bpy.ops.wm, "obj_export"):
        raise Exception(
            "Wavefront OBJ export operator not found in this Blender version"
        )

    export_args = {
        "filepath": obj_path,
        "export_selected_objects": operator.export_selected,
        "export_uv": True,
        "export_normals": True,
        "export_materials": True,  # Keep material assignments in OBJ
        "global_scale": 1.0,
        "forward_axis": "NEGATIVE_Z",  # -Z forward (into screen)
        "up_axis": "Y",  # Y-up orientation for renderer
    }

    bpy.ops.wm.obj_export(**export_args)


def _export_materials(operator, obj_path):
    """Export materials in ETX format, replacing the standard MTL file"""
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"

    materials_data = []

    # Get light materials first to ensure they are at the top of the file
    env_light = _get_environment_light_material(operator)
    if env_light:
        materials_data.append(env_light)

    light_materials = _get_lights_as_materials(operator)
    materials_data.extend(light_materials)

    # Get mesh materials that were actually used in the OBJ export
    materials_to_export = _get_materials_to_export(operator)

    # If standard MTL exists, read it to get the material names that were actually exported
    # This ensures we only export materials assigned to the exported geometry
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

    for mat in materials_to_export:
        if mat is None:
            continue

        mat_data = _convert_material_to_etx(operator, mat)
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

    # Replace the standard MTL file with ETX format
    if os.path.exists(mtl_path):
        write_materials_file(mtl_path)


def _export_scene_json(operator, json_path, obj_path):
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
        "samples": operator.samples,
        "max-path-length": operator.max_path_length,
        "random-termination-start": operator.random_termination_start,
        "spectral": operator.spectral_rendering,
        "force-tangents": operator.force_tangents,
        "camera": _get_camera_data(operator),
    }

    # Write JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene_data, f, indent=2)


def _get_camera_data(operator):
    """Extract camera data from Blender scene"""
    context = bpy.context
    scene = context.scene

    if operator.use_active_camera and scene.camera:
        camera_obj = scene.camera
        camera_data = camera_obj.data

        # Get camera transform in Blender's coordinate system (Z-up)
        matrix_world = camera_obj.matrix_world
        location_blender = matrix_world.translation

        # Calculate target point (camera forward direction) in Blender coordinates
        forward_blender = matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        target_blender = location_blender + forward_blender * 10.0  # 10 unit distance

        # Get up vector in Blender coordinates
        up_blender = matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))

        # Convert from Blender's Z-up to Y-up coordinate system (matching OBJ export)
        # Blender: X=right, Y=forward, Z=up
        # Y-up:    X=right, Y=up,      Z=back
        # Transformation: (x, y, z) -> (x, z, -y)
        location = Vector((location_blender.x, location_blender.z, -location_blender.y))
        target = Vector((target_blender.x, target_blender.z, -target_blender.y))
        up = Vector((up_blender.x, up_blender.z, -up_blender.y))

        # Get viewport from render settings
        render = scene.render
        viewport = [render.resolution_x, render.resolution_y]

        # Get field of view
        if camera_data.type == "PERSP":
            horizontal_fov = math.degrees(camera_data.angle_x)
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
                + (view_matrix.inverted().to_quaternion() @ Vector((0.0, 0.0, -1.0)))
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
            location = Vector((7.4, 5.3, 6.5))  # Adjusted for Y-up: (x, y_up, z_back)
            target = Vector((0.0, 0.0, 0.0))
            up = Vector((0.0, 1.0, 0.0))  # Y is up
            viewport = [1920, 1080]
            # Default FOV based on user choice
            fov = 50.0  # Always horizontal
            focal_length = 50.0

    return {
        "class": operator.camera_class,
        "viewport": viewport,
        "origin": [location.x, location.y, location.z],
        "target": [target.x, target.y, target.z],
        "up": [up.x, up.y, up.z],
        "fov": fov,
        "focal-length": focal_length,
        "lens-radius": operator.lens_radius,
        "focal-distance": operator.focal_distance,
    }


def _get_environment_light_material(operator):
    """Checks for a world background and exports it as an ETX environment light."""
    world = bpy.context.scene.world
    if not (world and world.use_nodes and world.node_tree):
        return None

    node_tree = world.node_tree
    output_node = None
    for node in node_tree.nodes:
        if node.type == "OUTPUT_WORLD" and node.is_active_output:
            output_node = node
            break

    if not output_node or not output_node.inputs["Surface"].is_linked:
        return None

    background_node = output_node.inputs["Surface"].links[0].from_node
    if background_node.type != "BACKGROUND":
        return None

    strength = background_node.inputs["Strength"].default_value
    color_socket = background_node.inputs["Color"]
    base_color = color_socket.default_value

    env_material = {"name": "et::env", "properties": {}}

    texture_node = None
    if color_socket.is_linked:
        link_node = color_socket.links[0].from_node
        if link_node.type == "TEX_ENVIRONMENT":
            texture_node = link_node

    if texture_node and texture_node.image:
        tint_color = [c * strength for c in base_color[:3]]
        if sum(tint_color) < 1e-6:
            return None

        env_material["properties"]["image"] = texture_node.image.name
        env_material["properties"][
            "color"
        ] = f"{tint_color[0]:.4f} {tint_color[1]:.4f} {tint_color[2]:.4f}"

        mapping_node = None
        if texture_node.inputs["Vector"].is_linked:
            prev_node = texture_node.inputs["Vector"].links[0].from_node
            if prev_node.type == "MAPPING":
                mapping_node = prev_node

        if mapping_node:
            rotation_z_rad = mapping_node.inputs["Rotation"].default_value[2]
            rotation_z_deg = math.degrees(rotation_z_rad)
            env_material["properties"]["rotation"] = f"{rotation_z_deg:.2f}"

        if texture_node.image.source != "FILE":
            operator.report(
                {"WARNING"},
                f"Environment image '{texture_node.image.name}' is not saved to a file and may not be found by the renderer.",
            )

    else:
        final_color = [c * strength for c in base_color[:3]]
        if sum(final_color) < 1e-6:
            return None

        env_material["properties"][
            "color"
        ] = f"{final_color[0]:.4f} {final_color[1]:.4f} {final_color[2]:.4f}"

    return env_material


def _get_lights_as_materials(operator):
    """Finds all sun lights and converts them to ETX directional light materials"""
    light_materials = []

    for light_obj in bpy.context.scene.objects:
        if light_obj.type == "LIGHT" and light_obj.data.type == "SUN":
            if operator.export_selected and not light_obj.select_get():
                continue

            light_data = light_obj.data

            matrix_world = light_obj.matrix_world
            direction_blender = (
                matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
            ).normalized()

            direction = -Vector(
                (direction_blender.x, direction_blender.z, -direction_blender.y)
            )

            color = light_data.color
            energy = light_data.energy
            emission_color = [c * energy for c in color]

            angular_diameter_deg = math.degrees(light_data.angle)

            mat_data = {
                "name": "et::dir",
                "properties": {
                    "direction": f"{direction.x:.4f} {direction.y:.4f} {direction.z:.4f}",
                    "color": f"{emission_color[0]:.4f} {emission_color[1]:.4f} {emission_color[2]:.4f}",
                    "angular_diameter": f"{angular_diameter_deg:.4f}",
                },
            }
            light_materials.append(mat_data)

    return light_materials


def _get_materials_to_export(operator):
    """Get list of materials to export based on selection"""
    materials = set()

    if operator.export_selected:
        for obj in bpy.context.selected_objects:
            if obj.type == "MESH" and obj.data.materials:
                for mat in obj.data.materials:
                    if mat is not None:
                        materials.add(mat)
    else:
        materials = set(bpy.data.materials)

    return list(materials)


def _convert_material_to_etx(operator, blender_mat):
    """Convert Blender material to ETX format"""
    mat_data = {"name": blender_mat.name, "properties": {}}

    etx_class = _get_etx_material_class(operator, blender_mat)
    mat_data["properties"]["material"] = f"class {etx_class}"

    if blender_mat.use_nodes and blender_mat.node_tree:
        _extract_node_properties(
            operator, blender_mat.node_tree, mat_data["properties"]
        )
    else:
        _extract_basic_properties(operator, blender_mat, mat_data["properties"])

    for key, value in blender_mat.items():
        if not key.startswith("_"):
            mat_data["properties"][f"# ETXProperty {key}"] = str(value)

    return mat_data


def _get_node_input_value(node, input_name, default_value=0.0):
    """Safely get input value from node, returns default if input doesn't exist"""
    try:
        if input_name in node.inputs:
            return node.inputs[input_name].default_value
        else:
            alt_names = {
                "Transmission": ["Transmission Weight", "Transmission Factor"],
                "Emission": ["Emission Color"],
                "Emission Strength": ["Emission Weight"],
                "Base Color": ["Albedo", "Diffuse Color"],
                "Subsurface Weight": ["Subsurface"],
                "Metallic": ["Metallic"],
            }

            if input_name in alt_names:
                for alt_name in alt_names[input_name]:
                    if alt_name in node.inputs:
                        return node.inputs[alt_name].default_value

            return default_value
    except (KeyError, AttributeError):
        return default_value


def _find_shader_node(node_tree):
    """Finds the shader node connected to the material output."""
    output_node = None
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and node.is_active_output:
            output_node = node
            break

    if not (output_node and output_node.inputs["Surface"].is_linked):
        return None

    return output_node.inputs["Surface"].links[0].from_node


def _get_etx_material_class(operator, blender_mat):
    """Determine ETX material class from Blender material node graph."""
    if blender_mat.use_nodes and blender_mat.node_tree:
        shader_node = _find_shader_node(blender_mat.node_tree)

        if shader_node:
            node_type_to_class = {
                "BSDF_PRINCIPLED": "plastic",
                "BSDF_METALLIC": "conductor",
                "BSDF_GLASS": "dielectric",
                "BSDF_GLOSSY": "conductor",
                "BSDF_TRANSLUCENT": "translucent",
                "BSDF_DIFFUSE": "diffuse",
                "EMISSION": "diffuse",
            }
            return node_type_to_class.get(shader_node.type, "diffuse")

    return "diffuse"


def _extract_principled_properties(operator, principled, properties):
    base_color = _get_node_input_value(principled, "Base Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(base_color, "__len__") and len(base_color) >= 3:
        properties["Kd"] = (
            f"{base_color[0]:.3f} {base_color[1]:.3f} {base_color[2]:.3f}"
        )
    else:
        properties["Kd"] = "0.800 0.800 0.800"

    roughness = _get_node_input_value(principled, "Roughness", 0.5)
    properties["Pr"] = f"{roughness:.3f}"

    specular_tint = _get_node_input_value(
        principled, "Specular Tint", [1.0, 1.0, 1.0, 1.0]
    )
    if hasattr(specular_tint, "__len__") and len(specular_tint) >= 3:
        properties["Ks"] = (
            f"{specular_tint[0]:.3f} {specular_tint[1]:.3f} {specular_tint[2]:.3f}"
        )
    else:
        properties["Ks"] = "1.000 1.000 1.000"

    ior = _get_node_input_value(principled, "IOR", 1.45)
    properties["int_ior"] = f"{ior:.3f}"

    transmission = _get_node_input_value(principled, "Transmission", 0.0)
    if transmission > 0.0:
        if hasattr(base_color, "__len__") and len(base_color) >= 3:
            trans_color = [c * transmission for c in base_color[:3]]
            properties["Kt"] = (
                f"{trans_color[0]:.3f} {trans_color[1]:.3f} {trans_color[2]:.3f}"
            )
        else:
            properties["Kt"] = (
                f"{transmission:.3f} {transmission:.3f} {transmission:.3f}"
            )

    emission_socket = principled.inputs.get("Emission Color")
    emission_strength = _get_node_input_value(principled, "Emission Strength", 0.0)
    if emission_strength > 0.0 and emission_socket:
        use_color_emission = True
        if emission_socket.is_linked:
            linked_node = emission_socket.links[0].from_node
            if linked_node.type == "BLACKBODY":
                temperature = linked_node.inputs["Temperature"].default_value
                properties["emitter"] = (
                    f"nblackbody {temperature:.0f} scale {emission_strength:.4f}"
                )
                use_color_emission = False
        if use_color_emission:
            emission_color = _get_node_input_value(
                principled, "Emission Color", [0.0, 0.0, 0.0, 1.0]
            )
            if any(c > 0.0 for c in emission_color[:3]):
                final_emission = [c * emission_strength for c in emission_color[:3]]
                properties["Ke"] = (
                    f"{final_emission[0]:.4f} {final_emission[1]:.4f} {final_emission[2]:.4f}"
                )

    _extract_texture_connections(operator, principled, properties)


def _extract_glass_properties(operator, glass_node, properties):
    color = _get_node_input_value(glass_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Kt"] = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"

    properties["Ks"] = "1.000 1.000 1.000"
    roughness = _get_node_input_value(glass_node, "Roughness", 0.0)
    properties["Pr"] = f"{roughness:.3f}"
    ior = _get_node_input_value(glass_node, "IOR", 1.45)
    properties["int_ior"] = f"{ior:.3f}"


def _extract_glossy_properties(operator, glossy_node, properties):
    color = _get_node_input_value(glossy_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Ks"] = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"
    roughness = _get_node_input_value(glossy_node, "Roughness", 0.0)
    properties["Pr"] = f"{roughness:.3f}"


def _extract_metallic_properties(
    operator, metallic_node, properties, metallic_value="silver"
):
    properties["int_ior"] = metallic_value
    color = _get_node_input_value(metallic_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Ks"] = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"
    roughness = _get_node_input_value(metallic_node, "Roughness", 0.0)
    properties["Pr"] = f"{roughness:.3f}"


def _extract_emission_properties(operator, emission_node, properties):
    color = _get_node_input_value(emission_node, "Color", [0.0, 0.0, 0.0, 1.0])
    strength = _get_node_input_value(emission_node, "Strength", 0.0)
    # Make the surface purely emissive by default
    properties["Kd"] = "0.000 0.000 0.000"
    if hasattr(color, "__len__") and len(color) >= 3 and strength > 0.0:
        final_emission = [color[0] * strength, color[1] * strength, color[2] * strength]
        if any(c > 0.0 for c in final_emission):
            properties["Ke"] = (
                f"{final_emission[0]:.4f} {final_emission[1]:.4f} {final_emission[2]:.4f}"
            )


def _extract_translucent_properties(operator, translucent_node, properties):
    color = _get_node_input_value(translucent_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Kd"] = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"


def _extract_diffuse_properties(operator, diffuse_node, properties):
    color = _get_node_input_value(diffuse_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Kd"] = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f}"
    roughness = _get_node_input_value(diffuse_node, "Roughness", 0.0)
    properties["Pr"] = f"{roughness:.3f}"


def _extract_node_properties(operator, node_tree, properties):
    shader_node = _find_shader_node(node_tree)
    if not shader_node:
        return

    extractors = {
        "BSDF_PRINCIPLED": _extract_principled_properties,
        "BSDF_METALLIC": _extract_metallic_properties,
        "BSDF_GLASS": _extract_glass_properties,
        "BSDF_GLOSSY": _extract_glossy_properties,
        "BSDF_TRANSLUCENT": _extract_translucent_properties,
        "BSDF_DIFFUSE": _extract_diffuse_properties,
        "EMISSION": _extract_emission_properties,
    }

    extractor = extractors.get(shader_node.type)
    if extractor:
        extractor(operator, shader_node, properties)


def _extract_texture_connections(operator, principled_node, properties):
    """Extract texture file connections from Principled BSDF"""
    # Base Color texture
    if _input_is_linked(principled_node, "Base Color"):
        texture_node = _find_texture_node(principled_node.inputs["Base Color"])
        if texture_node and texture_node.image:
            properties["map_Kd"] = texture_node.image.name

    # Normal map
    if _input_is_linked(principled_node, "Normal"):
        normal_map = _find_normal_map_node(principled_node.inputs["Normal"])
        if normal_map and normal_map.image:
            properties["normalmap"] = f"image {normal_map.image.name} scale 1.0"

    # Roughness texture
    if _input_is_linked(principled_node, "Roughness"):
        texture_node = _find_texture_node(principled_node.inputs["Roughness"])
        if texture_node and texture_node.image:
            properties["map_Pr"] = texture_node.image.name


def _input_is_linked(node, input_name):
    """Safely check if a node input is linked"""
    try:
        if input_name in node.inputs:
            return node.inputs[input_name].is_linked
        return False
    except (KeyError, AttributeError):
        return False


def _find_texture_node(socket):
    """Find connected Image Texture node"""
    if not socket.is_linked:
        return None

    from_node = socket.links[0].from_node
    if from_node.type == "TEX_IMAGE":
        return from_node

    # Handle ColorRamp or other intermediate nodes
    for input_socket in from_node.inputs:
        if input_socket.is_linked:
            result = _find_texture_node(input_socket)
            if result:
                return result

    return None


def _find_normal_map_node(socket):
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


def _extract_basic_properties(operator, blender_mat, properties):
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
