"""Escreve no disco o modelo final, o report e o debug."""

import json
import os
import shutil
import cv2
import numpy as np
import trimesh
from PIL import Image

from export_glb import export_glb

from .config import (
    ALBEDO_FILE_NAME,
    DEBUG_FOLDER_NAME,
    NORMAL_FILE_NAME,
    OUTPUT_MODEL_NAME,
    OUTPUT_REPORT_NAME,
    REQUIRED_VIEW_FILES,
)


def ensure_output_folders(pipeline_paths):
    """Cria a árvore de pastas usada pelo pipeline."""
    os.makedirs(pipeline_paths.output_root_dir, exist_ok=True)
    os.makedirs(pipeline_paths.model_dir, exist_ok=True)
    os.makedirs(pipeline_paths.materials_dir, exist_ok=True)
    os.makedirs(pipeline_paths.reports_dir, exist_ok=True)
    os.makedirs(pipeline_paths.debug_root_dir, exist_ok=True)
    os.makedirs(pipeline_paths.debug_run_dir, exist_ok=True)


def save_pipeline_outputs(reconstruction_result, pipeline_paths):
    """Guarda o material e exporta o GLB final."""
    Image.fromarray(ensure_rgb_image(reconstruction_result.albedo)).save(pipeline_paths.albedo_path)
    if reconstruction_result.normal is not None:
        Image.fromarray(ensure_rgb_image(reconstruction_result.normal)).save(pipeline_paths.normal_path)
    elif os.path.exists(pipeline_paths.normal_path):
        os.remove(pipeline_paths.normal_path)

    mesh = reconstruction_result.mesh.copy()
    ensure_mesh_has_uv(mesh)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(ensure_rgb_image(reconstruction_result.albedo)),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    export_glb(mesh, pipeline_paths.output_model_path)


def ensure_mesh_has_uv(mesh):
    """Se a malha não tiver UV, cria um mapeamento simples."""
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        return

    uv = mesh.vertices[:, [0, 1]].astype(np.float64)
    uv -= uv.min(axis=0)
    uv_span = np.maximum(uv.max(axis=0), 1e-6)
    uv /= uv_span
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)


def write_debug_images(debug_dir, view_debug_images, reconstruction_debug_images):
    """Organiza o debug por vista e por reconstrução."""
    views_dir = os.path.join(debug_dir, "views")
    reconstruction_dir = os.path.join(debug_dir, "reconstruction")
    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)

    for role, images in view_debug_images.items():
        role_dir = os.path.join(views_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        for name, image in images.items():
            cv2.imwrite(os.path.join(role_dir, f"{name}.png"), image)

    for name, image in reconstruction_debug_images.items():
        cv2.imwrite(os.path.join(reconstruction_dir, f"{name}.png"), image[:, :, ::-1])


def remove_old_output_layout(output_root_dir):
    """Remove old flat output files so the folder stays easy to read."""
    old_files = [
        os.path.join(output_root_dir, ALBEDO_FILE_NAME),
        os.path.join(output_root_dir, NORMAL_FILE_NAME),
        os.path.join(output_root_dir, OUTPUT_MODEL_NAME),
        os.path.join(output_root_dir, OUTPUT_REPORT_NAME),
    ]
    for file_path in old_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    old_debug_dir = os.path.join(output_root_dir, "debug_test_obj")
    if os.path.isdir(old_debug_dir):
        shutil.rmtree(old_debug_dir)


def clear_run_debug_folder(debug_run_dir):
    """Clear old debug images for the current run before writing new ones."""
    if os.path.isdir(debug_run_dir):
        shutil.rmtree(debug_run_dir)
    os.makedirs(debug_run_dir, exist_ok=True)


def write_pipeline_report(report_path, report, reconstruction_result, output_model_path):
    """Escreve um ficheiro de texto com o resumo da execução."""
    mesh = reconstruction_result.mesh
    extents = mesh.extents.tolist()

    lines = [
        "Pipeline report: test_obj",
        f"output_glb: {output_model_path}",
        "",
        "warnings:",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "mesh:",
            f"- vertices: {len(mesh.vertices)}",
            f"- faces: {len(mesh.faces)}",
            f"- extents: {extents}",
            "",
            "views:",
        ]
    )

    for role in REQUIRED_VIEW_FILES:
        view_report = report["views"][role]
        lines.extend(
            [
                f"- {role}:",
                f"  file: {view_report['file_name']}",
                f"  source: {view_report['segmentation_source']}",
                f"  bbox: {view_report['record_bbox']}",
                f"  area_ratio: {view_report['area_ratio']:.4f}",
                f"  focus_score: {view_report['focus_score']:.2f}",
                f"  segmentation_metrics: {json.dumps(view_report['segmentation_metrics'], ensure_ascii=False)}",
            ]
        )
        if view_report["warnings"]:
            lines.append(f"  warnings: {json.dumps(view_report['warnings'], ensure_ascii=False)}")

    lines.extend(
        [
            "",
            "reconstruction:",
            json.dumps(report["reconstruction"], ensure_ascii=False, indent=2),
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))


def ensure_rgb_image(image):
    """Garante que a imagem está em RGB uint8 antes de guardar."""
    rgb_image = np.asarray(image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    return rgb_image
