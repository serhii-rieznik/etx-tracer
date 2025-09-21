import bpy
import os
import json
import math
import shutil
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

        # === Export materials to ETX format (handles textures/baking) ===
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

    # Sanitize material names in OBJ to ensure readability and valid characters
    try:
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        # Collect names from MTL if exists, else from OBJ usemtl
        names = set()
        if os.path.exists(mtl_path):
            with open(mtl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("newmtl "):
                        names.add(line.strip().split(" ", 1)[1])
        else:
            with open(obj_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("usemtl "):
                        names.add(line.strip().split(" ", 1)[1])

        name_map = _uniquify_material_names(list(names))

        # Rewrite MTL newmtl lines
        if os.path.exists(mtl_path):
            with open(mtl_path, "r", encoding="utf-8") as f:
                mtl_lines = f.readlines()
            changed = False
            out_lines = []
            for line in mtl_lines:
                if line.startswith("newmtl "):
                    orig = line.strip().split(" ", 1)[1]
                    if orig in name_map and name_map[orig] != orig:
                        line = f"newmtl {name_map[orig]}\n"
                        changed = True
                out_lines.append(line)
            if changed:
                with open(mtl_path, "w", encoding="utf-8") as f:
                    f.writelines(out_lines)

        # Rewrite OBJ usemtl occurrences
        if len(name_map) > 0:
            _rewrite_obj_material_names(obj_path, name_map)
    except Exception:
        pass


def _export_materials(operator, obj_path):
    """Export materials in ETX format, replacing the standard MTL file"""
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"

    materials_data = []

    # Get light materials first to ensure they are at the top of the file
    env_light = _get_environment_light_material(operator, obj_path)
    if env_light:
        materials_data.append(env_light)

    light_materials = _get_lights_as_materials(operator)
    materials_data.extend(light_materials)

    # Get mesh materials present in Blender
    materials_to_export = _get_materials_to_export(operator)

    # If standard MTL exists, read exact material names that OBJ references
    exported_material_names = []
    if os.path.exists(mtl_path):
        with open(mtl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("newmtl "):
                    mat_name = line.strip().split(" ", 1)[1]
                    exported_material_names.append(mat_name)

    # === Optional: bake procedural textures before material conversion ===
    if getattr(operator, "bake_procedural", False):
        try:
            _bake_procedural_textures(operator, materials_to_export, obj_path)
        except Exception as bake_err:
            operator.report({"WARNING"}, f"Bake failed: {str(bake_err)}")

    # Build quick lookup by name (also allow sanitized lookup)
    name_to_mat = {mat.name: mat for mat in materials_to_export if mat}
    sanitized_to_mat = {_sanitize_material_name(mat.name): mat for mat in materials_to_export if mat}

    # === Gather mediums for materials that have volume (only for mats we can resolve) ===
    used_medium_ids = set()
    material_to_medium_id = {}
    for name, mat in name_to_mat.items():
        medium_entry, medium_id = _extract_medium_from_material(operator, mat, used_medium_ids)
        if medium_entry is not None and medium_id is not None:
            materials_data.append(medium_entry)
            material_to_medium_id[name] = medium_id

    # === Export mesh materials ===
    if exported_material_names:
        # Honor the exact names OBJ references to avoid missing materials
        # Sanitize names to be valid and ensure consistency across OBJ/MTL/export
        name_map = _uniquify_material_names(exported_material_names)
        _rewrite_obj_material_names(obj_path, name_map)

        for raw_name in exported_material_names:
            name = name_map.get(raw_name, raw_name)
            mat = name_to_mat.get(raw_name, None)
            if mat is None:
                # Try sanitized lookup (Blender material with invalid chars)
                mat = sanitized_to_mat.get(_sanitize_material_name(raw_name), None)
            if mat is not None:
                mat_data = _convert_material_to_etx(operator, mat)
                # Force the material name to match OBJ/MTL exactly
                mat_data["name"] = name
                try:
                    _finalize_material_textures(operator, obj_path, mat, mat_data["properties"])
                except Exception as tex_err:
                    operator.report({"WARNING"}, f"Material '{name}' textures warning: {str(tex_err)}")
                if name in material_to_medium_id:
                    mat_data["properties"]["int_medium"] = material_to_medium_id[name]
            else:
                # Fallback default for unknown names referenced by OBJ
                mat_data = {"name": name, "properties": {"material": "class diffuse", "Kd": "1.000 1.000 1.000"}}
            materials_data.append(mat_data)
    else:
        # No reference MTL; export all Blender materials
        for mat in materials_to_export:
            if mat is None:
                continue
            mat_data = _convert_material_to_etx(operator, mat)
            try:
                _finalize_material_textures(operator, obj_path, mat, mat_data["properties"])
            except Exception as tex_err:
                operator.report({"WARNING"}, f"Material '{mat.name}' textures warning: {str(tex_err)}")
            if mat.name in material_to_medium_id:
                mat_data["properties"]["int_medium"] = material_to_medium_id[mat.name]
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

    # Write (or create) the ETX materials file unconditionally
    write_materials_file(mtl_path)


# =========================
# Texture export utilities
# =========================

def _get_textures_dir(operator, obj_path):
    base_dir = os.path.dirname(obj_path)
    sub = getattr(operator, "texture_subdir", "").strip() if getattr(operator, "export_textures", False) else ""
    out_dir = os.path.join(base_dir, sub) if len(sub) > 0 else base_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _mtl_relpath(obj_path, file_path):
    mtl_dir = os.path.dirname(os.path.splitext(obj_path)[0] + ".mtl")
    rel = os.path.relpath(file_path, mtl_dir)
    return rel.replace("\\", "/")


def _sanitize_filename(name):
    import re
    import unicodedata
    # Normalize to ASCII (strip diacritics), allow [A-Za-z0-9._-]
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    safe = "".join(ch if (("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9") or (ch in ("_", "-", "."))) else "_" for ch in ascii_name)
    safe = re.sub(r"_+", "_", safe).strip("._")
    if len(safe) == 0:
        safe = "image"
    return safe


def _sanitize_material_name(name):
    # Normalize to ASCII: remove diacritics and non-ASCII, allow [A-Za-z0-9_-]
    # Collapse multiple underscores and strip leading/trailing underscores
    import re
    import unicodedata
    if name.startswith("et::"):
        return name  # keep engine special names untouched
    # decompose accents and drop non-ascii
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    tmp = "".join(ch if (('A' <= ch <= 'Z') or ('a' <= ch <= 'z') or ('0' <= ch <= '9') or (ch in ("_", "-"))) else "_" for ch in ascii_name)
    tmp = re.sub(r"_+", "_", tmp).strip("_")
    if len(tmp) == 0:
        tmp = "material"
    return tmp


def _linear_to_srgb_component(x: float) -> float:
    try:
        if x <= 0.0031308:
            return 12.92 * x
        return 1.055 * (max(0.0, x)) ** (1.0 / 2.4) - 0.055
    except Exception:
        return x


def _format_rgb_linear_to_srgb(rgb) -> str:
    try:
        r = _linear_to_srgb_component(float(rgb[0]))
        g = _linear_to_srgb_component(float(rgb[1]))
        b = _linear_to_srgb_component(float(rgb[2]))
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        return f"{r:.3f} {g:.3f} {b:.3f}"
    except Exception:
        return "0.800 0.800 0.800"


def _uniquify_material_names(names):
    mapping = {}
    used = set()
    for orig in names:
        if orig.startswith("et::"):
            mapping[orig] = orig
            continue
        base = _sanitize_material_name(orig)
        cand = base
        idx = 1
        while cand in used:
            cand = f"{base}_{idx}"
            idx += 1
        mapping[orig] = cand
        used.add(cand)
    return mapping


def _rewrite_obj_material_names(obj_path, name_map):
    try:
        with open(obj_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        changed = False
        out_lines = []
        for line in lines:
            if line.startswith("usemtl "):
                orig = line.strip()[7:]
                if orig in name_map:
                    new_name = name_map[orig]
                    if new_name != orig:
                        line = f"usemtl {new_name}\n"
                        changed = True
            out_lines.append(line)
        if changed:
            with open(obj_path, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
    except Exception:
        pass


def _unique_name(operator, base_name):
    used = getattr(operator, "_etx_used_texture_names", None)
    if used is None:
        used = set()
        operator._etx_used_texture_names = used
    name = base_name
    idx = 1
    while name in used:
        root, ext = os.path.splitext(base_name)
        name = f"{root}_{idx}{ext}"
        idx += 1
    used.add(name)
    return name


def _abspath_image(image):
    try:
        fp = image.filepath
        if fp is None or len(fp) == 0:
            return None
        return bpy.path.abspath(fp, library=image.library)
    except Exception:
        return None


def _save_image_copy(image, target_path, prefer_format=None):
    # prefer_format: e.g., 'PNG', 'OPEN_EXR'
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # If source is a file and we are not forcing re-encode, copy
        src = _abspath_image(image)
        if src and os.path.isfile(src) and prefer_format is None:
            shutil.copy2(src, target_path)
            return True

        # Otherwise, save via Blender with chosen format
        old_path = image.filepath_raw
        old_fmt = image.file_format
        try:
            ext = os.path.splitext(target_path)[1].lower()
            if prefer_format is not None:
                image.file_format = prefer_format
            else:
                if ext in (".png", ".jpg", ".jpeg"):
                    image.file_format = "PNG" if ext == ".png" else "JPEG"
                elif ext in (".exr", ".hdr"):
                    image.file_format = "OPEN_EXR" if ext == ".exr" else old_fmt
                else:
                    image.file_format = "PNG"
                    target_path = os.path.splitext(target_path)[0] + ".png"

            image.filepath_raw = target_path
            image.save()
            return True
        finally:
            image.filepath_raw = old_path
            image.file_format = old_fmt
    except Exception:
        return False


def _ensure_image_export(operator, obj_path, image, *, is_environment=False, suggested_name=None):
    if image is None:
        return None
    mapping = getattr(operator, "_etx_exported_images", None)
    if mapping is None:
        mapping = {}
        operator._etx_exported_images = mapping
    if image in mapping:
        return mapping[image]

    textures_dir = _get_textures_dir(operator, obj_path)

    # Decide filename and format
    src_path = _abspath_image(image)
    src_ext = os.path.splitext(src_path)[1].lower() if src_path else ""
    # If image is already a file, just reference it relatively
    try:
        if getattr(image, "source", None) == "FILE" and src_path and os.path.isfile(src_path):
            rel = _mtl_relpath(obj_path, src_path)
            mapping[image] = rel
            return rel
    except Exception:
        pass

    # For environment maps preserve EXR/HDR when possible
    if is_environment and src_ext in (".exr", ".hdr"):
        ext = src_ext
        prefer_fmt = None  # we'll copy if possible
    else:
        # Default to PNG for regular textures; EXR if float env or float image and env
        if is_environment and getattr(image, "is_float", False):
            ext = ".exr"
            prefer_fmt = "OPEN_EXR"
        else:
            ext = ".png"
            prefer_fmt = "PNG"

    # Build sanitized base filename
    if suggested_name and len(suggested_name) > 0:
        root, sug_ext = os.path.splitext(suggested_name)
        base_name = _sanitize_filename(root) + (sug_ext if len(sug_ext) > 0 else ext)
    else:
        base_name = _sanitize_filename(os.path.splitext(image.name)[0]) + ext
    base_name = _unique_name(operator, base_name)
    out_path = os.path.join(textures_dir, base_name)

    # If source exists and ext matches and we don't need re-encode, copy
    copied = False
    if src_path and os.path.isfile(src_path) and os.path.splitext(src_path)[1].lower() == ext and prefer_fmt is None:
        try:
            shutil.copy2(src_path, out_path)
            copied = True
        except Exception:
            copied = False

    if copied == False:
        _save_image_copy(image, out_path, prefer_format=prefer_fmt)

    rel = _mtl_relpath(obj_path, out_path)
    mapping[image] = rel
    return rel


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


def _finalize_material_textures(operator, obj_path, blender_mat, properties):
    """Resolve texture properties to relative file paths and ensure files exist.

    Accepts temporary values placed by _extract_texture_connections:
      - map_Kd: bpy.types.Image or string
      - map_Pr: bpy.types.Image or string
      - normalmap: tuple(Image, scale) or string
    """
    if getattr(operator, "export_textures", False) == False:
        # Convert any Image objects to relative file paths if source is FILE
        kd = properties.get("map_Kd")
        if isinstance(kd, bpy.types.Image):
            sp = _abspath_image(kd)
            properties["map_Kd"] = _mtl_relpath(obj_path, sp) if (sp and os.path.isfile(sp)) else kd.name
        # If base color is textured, drop constant Kd to avoid double-application
        if properties.get("map_Kd"):
            if "Kd" in properties:
                properties.pop("Kd", None)

        pr = properties.get("map_Pr")
        if isinstance(pr, bpy.types.Image):
            sp = _abspath_image(pr)
            properties["map_Pr"] = _mtl_relpath(obj_path, sp) if (sp and os.path.isfile(sp)) else pr.name

        nm = properties.get("normalmap")
        if isinstance(nm, tuple) and isinstance(nm[0], bpy.types.Image):
            img, scale = nm[0], nm[1]
            sp = _abspath_image(img)
            rel = _mtl_relpath(obj_path, sp) if (sp and os.path.isfile(sp)) else img.name
            properties["normalmap"] = f"image {rel} scale {float(scale):.4f}"
        return

    # Export each referenced image and update to relative paths
    def resolve_image(img_or_str, *, suggested):
        if isinstance(img_or_str, bpy.types.Image):
            rel = _ensure_image_export(operator, obj_path, img_or_str, suggested_name=suggested)
            return rel
        return img_or_str

    if "map_Kd" in properties and properties["map_Kd"] is not None:
        rel = resolve_image(properties["map_Kd"], suggested=f"{blender_mat.name}_Kd.png")
        if rel:
            properties["map_Kd"] = rel
            # If base color is textured, drop constant Kd to avoid double-application
            if "Kd" in properties:
                properties.pop("Kd", None)

    if "map_Pr" in properties and properties["map_Pr"] is not None:
        rel = resolve_image(properties["map_Pr"], suggested=f"{blender_mat.name}_Pr.png")
        if rel:
            properties["map_Pr"] = rel

    # metallic texture
    if "map_Ml" in properties and properties["map_Ml"] is not None:
        rel = resolve_image(properties["map_Ml"], suggested=f"{blender_mat.name}_Ml.png")
        if rel:
            properties["map_Ml"] = rel

    # transmission texture
    if "map_Tm" in properties and properties["map_Tm"] is not None:
        rel = resolve_image(properties["map_Tm"], suggested=f"{blender_mat.name}_Tm.png")
        if rel:
            properties["map_Tm"] = rel

    nm = properties.get("normalmap")
    if isinstance(nm, tuple) and len(nm) >= 2:
        img, scale = nm[0], nm[1]
        rel = resolve_image(img, suggested=f"{blender_mat.name}_N.png")
        if rel:
            properties["normalmap"] = f"image {rel} scale {float(scale):.4f}"

    # Override with baked results if present
    baked = getattr(operator, '_etx_baked', None)
    if isinstance(baked, dict) and blender_mat.name in baked:
        info = baked[blender_mat.name]
        if 'Kd' in info:
            properties['map_Kd'] = info['Kd']
        if 'N' in info:
            properties['normalmap'] = f"image {info['N']} scale 1.0"


def _bake_procedural_textures(operator, materials, obj_path):
    """Bake base color and normals for node-based materials to images.

    Creates temporary images per material, assigns to a temporary image node,
    performs cycles bake for selected objects that use the material.
    """
    # Ensure Cycles is available
    scene = bpy.context.scene
    prev_engine = scene.render.engine
    try:
        # Switch to Cycles if available
        if bpy.app.build_options.cycles:
            scene.render.engine = 'CYCLES'
        else:
            return  # skip baking if no cycles

        # Common bake settings
        scene.cycles.bake_type = 'DIFFUSE'
        scene.render.bake.use_selected_to_active = False
        scene.render.bake.use_clear = True
        scene.render.bake.margin = int(getattr(operator, 'bake_margin', 4))
        res = int(getattr(operator, 'bake_resolution', '2048'))

        # Collect objects by material
        mat_to_objs = {}
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH' or obj.data is None:
                continue
            for slot in obj.material_slots:
                if slot and slot.material in materials:
                    mat_to_objs.setdefault(slot.material, []).append(obj)

        # Prepare export dir
        textures_dir = _get_textures_dir(operator, obj_path)

        for mat, objs in mat_to_objs.items():
            if mat is None or (not mat.use_nodes) or mat.node_tree is None:
                continue

            nt = mat.node_tree
            # Ensure active output/material
            out = None
            for n in nt.nodes:
                if n.type == 'OUTPUT_MATERIAL' and n.is_active_output:
                    out = n
                    break
            if out is None:
                continue

            # Create image node
            img_node = nt.nodes.new('ShaderNodeTexImage')
            img_node.interpolation = 'Smart'
            img_node.label = 'ETX_BAKE_TARGET'

            # Set bake targets we want
            bake_targets = [
                ('DIFFUSE', f"{_sanitize_filename(mat.name)}_baked_Kd.png"),
            ]

            # Try normal bake if there is a normal linkage
            has_normal = False
            try:
                shader = _find_shader_node(nt)
                if shader and _input_is_linked(shader, 'Normal'):
                    has_normal = True
            except Exception:
                has_normal = False
            if has_normal:
                bake_targets.append(('NORMAL', f"{_sanitize_filename(mat.name)}_baked_N.png"))

            # Ensure objects are selected for baking
            prev_selection = [o for o in bpy.context.selected_objects]
            prev_active = bpy.context.view_layer.objects.active
            try:
                for o in bpy.context.selected_objects:
                    o.select_set(False)
                for o in objs:
                    o.select_set(True)
                if len(objs) > 0:
                    bpy.context.view_layer.objects.active = objs[0]

                baked_paths = {}
                for bake_type, filename in bake_targets:
                    # Create/replace image
                    img = bpy.data.images.new(name=filename, width=res, height=res, alpha=True, float_buffer=False)
                    img.file_format = 'PNG'
                    img_node.image = img
                    nt.nodes.active = img_node

                    if bake_type == 'DIFFUSE':
                        scene.cycles.bake_type = 'DIFFUSE'
                        scene.render.bake.use_pass_direct = False
                        scene.render.bake.use_pass_indirect = False
                        scene.render.bake.use_pass_color = True
                    elif bake_type == 'NORMAL':
                        scene.cycles.bake_type = 'NORMAL'

                    # Perform bake
                    bpy.ops.object.bake(type=scene.cycles.bake_type)

                    # Save baked image
                    out_path = os.path.join(textures_dir, filename)
                    img.filepath_raw = out_path
                    img.file_format = 'PNG'
                    img.save()

                    rel = _mtl_relpath(obj_path, out_path)
                    # Register export mapping so later property resolution can use it
                    if getattr(operator, '_etx_exported_images', None) is None:
                        operator._etx_exported_images = {}
                    operator._etx_exported_images[img] = rel
                    if bake_type == 'DIFFUSE':
                        baked_paths['Kd'] = rel
                    elif bake_type == 'NORMAL':
                        baked_paths['N'] = rel

                # Store per-material baked results for later override
                if baked_paths:
                    if getattr(operator, '_etx_baked', None) is None:
                        operator._etx_baked = {}
                    operator._etx_baked[mat.name] = baked_paths

            finally:
                # Cleanup image node
                try:
                    nt.nodes.remove(img_node)
                except Exception:
                    pass
                # Restore selection
                for o in bpy.context.selected_objects:
                    o.select_set(False)
                for o in prev_selection:
                    try:
                        o.select_set(True)
                    except Exception:
                        pass
                try:
                    bpy.context.view_layer.objects.active = prev_active
                except Exception:
                    pass
    finally:
        try:
            scene.render.engine = prev_engine
        except Exception:
            pass


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

        # Get viewport from render settings taking percentage into account
        render = scene.render
        try:
            percent = float(getattr(render, "resolution_percentage", 100))
        except Exception:
            percent = 100.0
        scale = max(1.0, percent) / 100.0
        viewport = [int(max(1, render.resolution_x * scale)), int(max(1, render.resolution_y * scale))]

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

            # Use render settings with percentage if available, fallback to default
            try:
                render = scene.render
                percent = float(getattr(render, "resolution_percentage", 100))
                scale = max(1.0, percent) / 100.0
                viewport = [int(max(1, render.resolution_x * scale)), int(max(1, render.resolution_y * scale))]
            except Exception:
                viewport = [1920, 1080]
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
            try:
                render = scene.render
                percent = float(getattr(render, "resolution_percentage", 100))
                scale = max(1.0, percent) / 100.0
                viewport = [int(max(1, render.resolution_x * scale)), int(max(1, render.resolution_y * scale))]
            except Exception:
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


def _get_environment_light_material(operator, obj_path):
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

    # Blackbody color on background (supports simple chains)
    temp = _find_blackbody_temperature_from_socket(color_socket)
    if temp is not None:
        env_material["properties"][
            "color"
        ] = f"nblackbody {temp:.0f} scale {strength:.4f}"
        return env_material

    if texture_node and texture_node.image:
        # Export environment texture to disk and reference relatively
        if getattr(operator, "export_textures", False):
            rel = _ensure_image_export(operator, obj_path, texture_node.image, is_environment=True)
        else:
            rel = texture_node.image.name
        env_material["properties"]["image"] = rel
        # With a texture, use Background Strength as scalar (ignore base color)
        env_material["properties"]["color"] = f"{strength:.4f} {strength:.4f} {strength:.4f}"

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
            # Will be saved via _ensure_image_export
            pass

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

            color_value = None

            # If the sun light uses nodes, and the Emission color is driven by a Blackbody, export it as nblackbody
            try:
                if getattr(light_data, "use_nodes", False) and getattr(
                    light_data, "node_tree", None
                ):
                    ltree = light_data.node_tree
                    output_node = None
                    for n in ltree.nodes:
                        if n.type == "OUTPUT_LIGHT" and getattr(
                            n, "is_active_output", True
                        ):
                            output_node = n
                            break
                    if (
                        output_node
                        and output_node.inputs.get("Surface")
                        and output_node.inputs["Surface"].is_linked
                    ):
                        surf_from = output_node.inputs["Surface"].links[0].from_node
                        if surf_from.type == "EMISSION":
                            strength = _get_node_input_value(surf_from, "Strength", 1.0)
                            col_sock = surf_from.inputs.get("Color")
                            if col_sock:
                                temperature = _find_blackbody_temperature_from_socket(
                                    col_sock
                                )
                                if temperature is not None:
                                    color_value = f"nblackbody {temperature:.0f} scale {float(strength):.4f}"
            except Exception:
                color_value = None

            if color_value is None:
                # Prefer built-in temperature if present, regardless of enable flag
                try:
                    temperature = _get_light_temperature(light_data)
                    if (temperature is not None) and (temperature > 0.0):
                        energy = float(getattr(light_data, "energy", 1.0))
                        color_value = f"nblackbody {temperature:.0f} scale {energy:.4f}"
                except Exception:
                    color_value = None

            if color_value is None:
                color = light_data.color
                energy = light_data.energy
                emission_color = [c * energy for c in color]
                color_value = f"{emission_color[0]:.4f} {emission_color[1]:.4f} {emission_color[2]:.4f}"

            angular_diameter_deg = math.degrees(light_data.angle)

            mat_data = {
                "name": "et::dir",
                "properties": {
                    "direction": f"{direction.x:.4f} {direction.y:.4f} {direction.z:.4f}",
                    "color": color_value,
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

    # Two-sided toggle (only for opaque non-transmitting classes)
    try:
        etx_cls = mat_data["properties"].get("material", "class diffuse").split(" ")[-1]
        if getattr(operator, "two_sided_materials", False):
            if etx_cls in ("diffuse", "plastic", "conductor", "velvet", "principled"):
                mat_data["properties"]["two_sided"] = "1"
    except Exception:
        pass

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
                "BSDF_PRINCIPLED": "principled",
                "BSDF_METALLIC": "conductor",
                "BSDF_GLASS": "dielectric",
                "BSDF_GLOSSY": "conductor",
                "BSDF_TRANSLUCENT": "translucent",
                "BSDF_DIFFUSE": "diffuse",
                "BSDF_TRANSPARENT": "boundary",
                "EMISSION": "diffuse",
            }
            return node_type_to_class.get(shader_node.type, "diffuse")

    return "diffuse"


def _extract_principled_properties(operator, principled, properties):
    base_color = _get_node_input_value(principled, "Base Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(base_color, "__len__") and len(base_color) >= 3:
        properties["Kd"] = _format_rgb_linear_to_srgb(base_color)
    else:
        properties["Kd"] = "0.800 0.800 0.800"

    # Opacity: product of Base Color alpha and Principled 'Alpha' input
    try:
        base_alpha = float(base_color[3]) if (hasattr(base_color, "__len__") and len(base_color) >= 4) else 1.0
    except Exception:
        base_alpha = 1.0
    try:
        principled_alpha = float(_get_node_input_value(principled, "Alpha", 1.0))
    except Exception:
        principled_alpha = 1.0
    opacity_value = max(0.0, min(1.0, base_alpha * principled_alpha))
    properties["opacity"] = f"{opacity_value:.4f}"

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

    # Metallic and Transmission scalars for Principled BSDF
    metallic = _get_node_input_value(principled, "Metallic", 0.0)
    properties["metalness"] = f"{float(metallic):.3f}"

    transmission = _get_node_input_value(principled, "Transmission", 0.0)
    properties["transmission"] = f"{float(transmission):.3f}"
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
    # If color is (directly or indirectly) driven by a Blackbody node, export as nblackbody emitter
    try:
        color_input = emission_node.inputs.get("Color")
        if color_input and strength > 0.0:
            temp = _find_blackbody_temperature_from_socket(color_input)
            if temp is not None:
                properties["emitter"] = f"nblackbody {temp:.0f} scale {strength:.4f}"
                return
    except Exception:
        pass
    if hasattr(color, "__len__") and len(color) >= 3 and strength > 0.0:
        final_emission = [color[0] * strength, color[1] * strength, color[2] * strength]
        if any(c > 0.0 for c in final_emission):
            properties["Ke"] = (
                f"{final_emission[0]:.4f} {final_emission[1]:.4f} {final_emission[2]:.4f}"
            )


def _extract_translucent_properties(operator, translucent_node, properties):
    color = _get_node_input_value(translucent_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Kd"] = _format_rgb_linear_to_srgb(color)


def _extract_diffuse_properties(operator, diffuse_node, properties):
    color = _get_node_input_value(diffuse_node, "Color", [1.0, 1.0, 1.0, 1.0])
    if hasattr(color, "__len__") and len(color) >= 3:
        properties["Kd"] = _format_rgb_linear_to_srgb(color)
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
            properties["map_Kd"] = texture_node.image

    # Normal map
    if _input_is_linked(principled_node, "Normal"):
        normal_map = _find_normal_map_node(principled_node.inputs["Normal"])
        if normal_map and normal_map.image:
            properties["normalmap"] = (normal_map.image, 1.0)

    # Roughness texture
    if _input_is_linked(principled_node, "Roughness"):
        texture_node = _find_texture_node(principled_node.inputs["Roughness"])
        if texture_node and texture_node.image:
            properties["map_Pr"] = texture_node.image

    # Metallic texture
    if _input_is_linked(principled_node, "Metallic"):
        texture_node = _find_texture_node(principled_node.inputs["Metallic"])
        if texture_node and texture_node.image:
            properties["map_Ml"] = texture_node.image

    # Transmission texture
    if _input_is_linked(principled_node, "Transmission"):
        texture_node = _find_texture_node(principled_node.inputs["Transmission"])
        if texture_node and texture_node.image:
            properties["map_Tm"] = texture_node.image


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


def _find_blackbody_temperature_from_socket(socket, _visited=None):
    """Traverse upstream from a socket to find a Blackbody node temperature.
    Returns temperature in Kelvin as float if found, otherwise None.
    """
    try:
        if socket is None or getattr(socket, "is_linked", False) == False:
            return None
        if _visited is None:
            _visited = set()
        node = socket.links[0].from_node
        if node in _visited:
            return None
        _visited.add(node)
        if getattr(node, "type", None) == "BLACKBODY":
            return float(node.inputs["Temperature"].default_value)
        # DFS through all inputs of this node
        for inp in getattr(node, "inputs", []):
            if getattr(inp, "is_linked", False):
                t = _find_blackbody_temperature_from_socket(inp, _visited)
                if t is not None:
                    return t
    except Exception:
        return None
    return None


def _light_uses_temperature(light_data):
    """Return True if the light is configured to use color temperature.

    Blender versions differ: `use_temperature` (common) or `use_color_temperature`.
    """
    try:
        if hasattr(light_data, "use_temperature") and bool(light_data.use_temperature):
            return True
        if hasattr(light_data, "use_color_temperature") and bool(
            light_data.use_color_temperature
        ):
            return True
    except Exception:
        return False
    return False


def _get_light_temperature(light_data):
    """Return temperature in Kelvin from light data, or None if unavailable."""
    try:
        if hasattr(light_data, "temperature") and light_data.temperature is not None:
            return float(light_data.temperature)
        if (
            hasattr(light_data, "color_temperature")
            and light_data.color_temperature is not None
        ):
            return float(light_data.color_temperature)
    except Exception:
        return None
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


def _extract_medium_from_material(operator, blender_mat, used_ids):
    """Detect Blender volume nodes and build an et::medium entry.

    Returns (medium_entry_dict_or_None, medium_id_or_None).
    """
    if not (blender_mat and blender_mat.use_nodes and blender_mat.node_tree):
        return None, None

    node_tree = blender_mat.node_tree

    # Find active output and inspect its Volume input
    output_node = None
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and node.is_active_output:
            output_node = node
            break

    if output_node is None or ("Volume" not in output_node.inputs):
        return None, None

    if output_node.inputs["Volume"].is_linked == False:
        return None, None

    # Resolve upstream BSDF nodes connected into Volume
    from_node = output_node.inputs["Volume"].links[0].from_node

    # Helper to generate a unique ID
    def make_unique_id(base):
        base_clean = base if base else "medium"
        candidate = f"{base_clean}__vol"
        i = 1
        while candidate in used_ids:
            candidate = f"{base_clean}__vol_{i}"
            i += 1
        used_ids.add(candidate)
        return candidate

    # Accumulate absorption/scattering
    absorption_rgb = None
    scattering_rgb = None

    def get_color(node, socket_name, default_value):
        if socket_name in node.inputs:
            val = node.inputs[socket_name].default_value
            if hasattr(val, "__len__") and len(val) >= 3:
                return [float(val[0]), float(val[1]), float(val[2])]
        return list(default_value)

    # Transparent handling: for Volume input, relevant nodes are Volume Absorption and Volume Scatter
    # - Volume Absorption: output type "Shader", node.type == 'VOLUME_ABSORPTION', Color socket
    # - Volume Scatter: node.type == 'VOLUME_SCATTER', Color socket; (Anisotropy available but optional here)
    def traverse(node):
        nonlocal absorption_rgb, scattering_rgb
        if node is None:
            return
        if node.type == "VOLUME_ABSORPTION":
            absorption_rgb = get_color(node, "Color", [0.0, 0.0, 0.0])
            return
        if node.type == "VOLUME_SCATTER":
            scattering_rgb = get_color(node, "Color", [0.0, 0.0, 0.0])
            return
        # If a Mix Shader or similar is used, traverse any linked inputs to find volume nodes
        for inp in node.inputs:
            if getattr(inp, "is_linked", False):
                try:
                    src = inp.links[0].from_node
                except Exception:
                    src = None
                traverse(src)

    traverse(from_node)

    if (absorption_rgb is None) and (scattering_rgb is None):
        return None, None

    medium_id = make_unique_id(blender_mat.name)
    medium_entry = {"name": "et::medium", "properties": {}}
    medium_entry["properties"]["id"] = medium_id
    if absorption_rgb is not None:
        medium_entry["properties"][
            "absorption"
        ] = f"{absorption_rgb[0]:.4f} {absorption_rgb[1]:.4f} {absorption_rgb[2]:.4f}"
    if scattering_rgb is not None:
        medium_entry["properties"][
            "scattering"
        ] = f"{scattering_rgb[0]:.4f} {scattering_rgb[1]:.4f} {scattering_rgb[2]:.4f}"

    return medium_entry, medium_id
