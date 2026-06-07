import json
import shutil
import subprocess
from pathlib import Path

import imageio.v3 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
OUT = BIN / "validation" / "surface_bsdf_cases"
RAYTRACER = BIN / "raytracer.exe"


def read_text(path):
    return path.read_text(encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scene_json(base_path, geometry, materials, selected_integrator):
    data = json.loads(read_text(base_path))
    data["geometry"] = geometry
    data["materials"] = materials
    data["samples"] = 64
    data["spectral"] = False
    data["integrator"]["selected"] = selected_integrator
    return json.dumps(data, indent=2)


def replace_material_blocks(text, class_name, kd, pr, thinfilm, target_prefix, metalness=None, transmission=None):
    result = []
    blocks = text.split("\nnewmtl ")
    for block_index, block in enumerate(blocks):
        if block_index == 0:
            prefix = ""
            block_text = block
        else:
            prefix = "\nnewmtl "
            block_text = block

        lines = block_text.splitlines()
        material_name = lines[0] if len(lines) > 0 else ""
        if material_name.startswith(target_prefix) and any(line.startswith("material class ") for line in lines):
            next_lines = []
            for line in lines:
                if line.startswith("material class "):
                    next_lines.append(f"material class {class_name}")
                elif line.startswith("Kd "):
                    next_lines.append(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}")
                elif line.startswith("Kt "):
                    next_lines.append(f"Kt {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}")
                elif line.startswith("Ks "):
                    next_lines.append("Ks 1.000000 1.000000 1.000000")
                elif line.startswith("Pr "):
                    next_lines.append(f"Pr {pr:.6f}")
                elif line.startswith("int_ior "):
                    next_lines.append("int_ior plastic")
                elif line.startswith("ext_ior "):
                    next_lines.append("ext_ior air")
                elif line.startswith("metalness ") or line.startswith("transmission ") or line.startswith("thinfilm "):
                    continue
                else:
                    next_lines.append(line)

            if metalness is not None:
                next_lines.append(f"metalness {metalness:.6f}")
            if transmission is not None:
                next_lines.append(f"transmission {transmission:.6f}")
            if thinfilm is not None:
                if isinstance(thinfilm, tuple):
                    next_lines.append(f"thinfilm image {thinfilm[0]} range {thinfilm[1]:.6f} {thinfilm[2]:.6f} ior 1.500000")
                else:
                    next_lines.append(f"thinfilm range {thinfilm:.6f} {thinfilm:.6f} ior 1.500000")
            block_text = "\n".join(next_lines)

        result.append(prefix + block_text)
    return "".join(result)


def make_plastic_directional_scene():
    base_dir = BIN / "assets_testing" / "furnace"
    material_text = read_text(base_dir / "furnace-env.etx.materials")
    material_text = material_text.replace("color 0.500000 0.500000 0.500000", "color 0.000000 0.000000 0.000000")
    material_text = material_text.replace(
        "newmtl main",
        "newmtl et::dir\ncolor 8.000000 8.000000 8.000000\ndirection -0.500000 0.300000 0.812404\nangular_diameter 0.542200\n\nnewmtl main",
    )
    material_text = replace_material_blocks(material_text, "plastic", (1.0, 0.015, 0.01), 0.020, None, "main")
    write_text(OUT / "plastic-directional.etx.materials", material_text)
    write_text(
        OUT / "plastic-directional.etx.json",
        scene_json(base_dir / "furnace-env.etx.json", "../../assets_testing/furnace/furnace-env.etx", "plastic-directional.etx.materials", "pt"),
    )


def make_soap_scene():
    base_dir = BIN / "assets_testing" / "soap"
    shutil.copyfile(base_dir / "bubbles.etx.materials", OUT / "soap-bubble.etx.materials")
    write_text(
        OUT / "soap-bubble.etx.json",
        scene_json(base_dir / "bubbles.etx.json", "../../assets_testing/soap/bubbles.etx", "soap-bubble.etx.materials", "pt"),
    )


def make_openpbr_scene(tag, kd, pr, metalness, transmission, thinfilm):
    base_dir = BIN / "assets_testing" / "furnace"
    material_text = read_text(base_dir / "furnace-env.etx.materials")
    material_text = material_text.replace("color 0.500000 0.500000 0.500000", "color 0.000000 0.000000 0.000000")
    material_text = material_text.replace(
        "newmtl main",
        "newmtl et::dir\ncolor 6.000000 6.000000 6.000000\ndirection -0.500000 0.300000 0.812404\nangular_diameter 0.542200\n\nnewmtl main",
    )
    material_text = replace_material_blocks(material_text, "openpbr", kd, pr, thinfilm, "main", metalness, transmission)
    materials_name = f"openpbr-{tag}.etx.materials"
    scene_name = f"openpbr-{tag}.etx.json"
    write_text(OUT / materials_name, material_text)
    write_text(OUT / scene_name, scene_json(base_dir / "furnace-env.etx.json", "../../assets_testing/furnace/furnace-env.etx", materials_name, "pt"))


def make_openpbr_variable_thinfilm_scene():
    width = 256
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (width, 1))
    texture = np.dstack((gradient, gradient, gradient))
    texture_name = "openpbr-variable-thinfilm-thickness.png"
    imageio.imwrite(OUT / texture_name, texture)
    make_openpbr_scene("variable-thinfilm", (1.0, 0.18, 0.06), 0.30, 0.0, 0.0, (texture_name, 350.0, 750.0))


def run_render(scene_name, output_name, integrator, samples, resolution, seed, strategy_flags=None):
    cmd = [
        str(RAYTRACER),
        "--render",
        "--scene",
        str(OUT / scene_name),
        "--output",
        str(OUT / output_name),
        "--integrator",
        integrator,
        "--samples",
        str(samples),
        "--resolution",
        resolution,
        "--random-seed",
        str(seed),
        "--max-path-length",
        "16",
    ]
    if strategy_flags is not None:
        cmd.extend(["--strategy-flags", strategy_flags])
    completed = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = OUT / (Path(output_name).stem + ".log")
    write_text(log_path, completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"render failed: {output_name}\n{completed.stdout[-4000:]}")


def load_image_for_output(output_name):
    image_path = OUT / output_name
    image = imageio.imread(image_path)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    original_dtype = image.dtype
    image = image.astype(np.float32)
    if original_dtype == np.uint8:
        image /= 255.0
    elif image.max() > 1.0:
        image /= max(float(image.max()), 1.0)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise RuntimeError(f"unexpected image shape for {image_path}: {image.shape}")


def image_stats(output_name):
    image = load_image_for_output(output_name)
    preview_path = OUT / output_name
    preview_path = preview_path.with_suffix(".preview.png")
    imageio.imwrite(preview_path, np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))
    flat = image.reshape((-1, 3))
    return {
        "mean": flat.mean(axis=0).tolist(),
        "p95": np.percentile(flat, 95.0, axis=0).tolist(),
        "p99": np.percentile(flat, 99.0, axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
    }


def compare_outputs(a_name, b_name):
    a = load_image_for_output(a_name)
    b = load_image_for_output(b_name)
    diff = a - b
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def plastic_terminator_probe(output_name):
    image = load_image_for_output(output_name)
    red_signal = image[:, :, 0] - np.maximum(image[:, :, 1], image[:, :, 2])
    red_signal = np.maximum(red_signal, 0.0)
    row_scores = np.sum(red_signal > 0.08, axis=1)
    row = int(np.argmax(row_scores))
    profile = red_signal[row, :]
    kernel = np.ones(9, dtype=np.float32) / 9.0
    smooth = np.convolve(profile, kernel, mode="same")
    peak = float(np.max(smooth))
    if peak <= 1.0e-6:
        return {"row": row, "peak": peak, "problem": "no red substrate signal"}

    normalized = smooth / peak
    active = np.where(normalized > 0.08)[0]
    transition = np.where((normalized > 0.2) & (normalized < 0.8))[0]
    max_drop = float(np.max(np.maximum(0.0, normalized[:-1] - normalized[1:])))
    problem = ""
    if active.size == 0:
        problem = "no active red pixels"
    elif transition.size < 4:
        problem = "red substrate transition is too narrow"
    elif max_drop > 0.45:
        problem = "red substrate has an abrupt drop"

    return {
        "row": row,
        "peak": peak,
        "active_width": int(active[-1] - active[0] + 1) if active.size > 0 else 0,
        "transition_samples": int(transition.size),
        "max_normalized_drop": max_drop,
        "problem": problem,
    }


def variable_thinfilm_probe(output_name):
    image = load_image_for_output(output_name)
    luminance = image[:, :, 0] * 0.2126 + image[:, :, 1] * 0.7152 + image[:, :, 2] * 0.0722
    visible = luminance > 0.01
    if np.count_nonzero(visible) < 64:
        return {"visible_pixels": int(np.count_nonzero(visible)), "problem": "not enough visible pixels"}

    chroma = np.max(image, axis=2) - np.min(image, axis=2)
    visible_chroma = chroma[visible]
    x_coords = np.broadcast_to(np.arange(image.shape[1]), image.shape[:2])
    left = visible & (x_coords < image.shape[1] // 2)
    right = visible & (x_coords >= image.shape[1] // 2)
    left_mean = image[left].mean(axis=0) if np.count_nonzero(left) > 0 else np.zeros(3, dtype=np.float32)
    right_mean = image[right].mean(axis=0) if np.count_nonzero(right) > 0 else np.zeros(3, dtype=np.float32)
    side_delta = float(np.linalg.norm(left_mean - right_mean))
    chroma_std = float(np.std(visible_chroma))
    problem = ""
    if chroma_std < 0.002 and side_delta < 0.003:
        problem = "variable thinfilm render has too little color variation"

    return {
        "visible_pixels": int(np.count_nonzero(visible)),
        "chroma_std": chroma_std,
        "left_mean": left_mean.tolist(),
        "right_mean": right_mean.tolist(),
        "side_delta": side_delta,
        "problem": problem,
    }


def main():
    if RAYTRACER.exists() is False:
        raise RuntimeError(f"raytracer executable not found: {RAYTRACER}")

    OUT.mkdir(parents=True, exist_ok=True)
    make_plastic_directional_scene()
    make_soap_scene()
    make_openpbr_scene("plastic-thinfilm", (1.0, 0.08, 0.04), 0.35, 0.0, 0.0, 500.0)
    make_openpbr_scene("dielectric-thinfilm", (1.0, 1.0, 1.0), 0.20, 0.0, 1.0, 500.0)
    make_openpbr_scene("conductor-thinfilm", (1.0, 0.72, 0.33), 0.25, 1.0, 0.0, 500.0)
    make_openpbr_scene("mixed-thinfilm", (0.9, 0.12, 0.06), 0.30, 0.5, 0.25, 500.0)
    make_openpbr_variable_thinfilm_scene()

    cases = [
        ("plastic-directional.etx.json", "plastic-directional-pt.exr", "pt", 128, "320x320", 41001, None),
        ("plastic-directional.etx.json", "plastic-directional-bdpt.exr", "bdpt", 128, "320x320", 41001, "direct_hit,connect_to_light,connect_to_camera,connect_vertices"),
        ("soap-bubble.etx.json", "soap-bubble-pt.exr", "pt", 64, "256x256", 42001, None),
        ("soap-bubble.etx.json", "soap-bubble-bdpt.exr", "bdpt", 64, "256x256", 42001, "direct_hit,connect_to_light,connect_to_camera,connect_vertices"),
        ("openpbr-plastic-thinfilm.etx.json", "openpbr-plastic-thinfilm-pt.exr", "pt", 96, "320x320", 43001, None),
        ("openpbr-dielectric-thinfilm.etx.json", "openpbr-dielectric-thinfilm-pt.exr", "pt", 96, "320x320", 43002, None),
        ("openpbr-conductor-thinfilm.etx.json", "openpbr-conductor-thinfilm-pt.exr", "pt", 96, "320x320", 43003, None),
        ("openpbr-mixed-thinfilm.etx.json", "openpbr-mixed-thinfilm-pt.exr", "pt", 96, "320x320", 43004, None),
        ("openpbr-variable-thinfilm.etx.json", "openpbr-variable-thinfilm-pt.exr", "pt", 96, "320x320", 43005, None),
    ]

    for case in cases:
        run_render(*case)

    report = {
        "renders": {case[1]: image_stats(case[1]) for case in cases},
        "plastic_terminator": plastic_terminator_probe("plastic-directional-pt.exr"),
        "pt_bdpt": {
            "plastic_directional": compare_outputs("plastic-directional-pt.exr", "plastic-directional-bdpt.exr"),
            "soap_bubble": compare_outputs("soap-bubble-pt.exr", "soap-bubble-bdpt.exr"),
        },
        "variable_thinfilm": variable_thinfilm_probe("openpbr-variable-thinfilm-pt.exr"),
    }

    report["problems"] = []
    plastic_problem = report["plastic_terminator"]["problem"]
    if plastic_problem:
        report["problems"].append({"case": "plastic-directional-pt.exr", "problem": plastic_problem})
    variable_problem = report["variable_thinfilm"]["problem"]
    if variable_problem:
        report["problems"].append({"case": "openpbr-variable-thinfilm-pt.exr", "problem": variable_problem})

    write_text(OUT / "surface-bsdf-render-report.json", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
