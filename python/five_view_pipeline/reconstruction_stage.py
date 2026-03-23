"""Reconstrói uma malha simples a partir das 5 silhuetas preparadas."""

import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.measure import marching_cubes

from .config import (
    COMPACT_HEIGHT_RATIO_RANGE,
    LID_START_RATIO,
    MAX_FOOTPRINT_RESOLUTION,
    MAX_Z_SAMPLES,
    MIN_Z_SAMPLES,
    PAIR_MEAN_DELTA_WARNING,
    PAIR_PEAK_DELTA_WARNING,
    PROFILE_SAMPLES,
    TARGET_FACE_COUNT,
    VOLUME_LEVEL,
)
from .models import PipelineError, ReconstructionResult


def build_compact_case_reconstruction(prepared_views):
    """Reconstrói um objeto compacto usando sempre o mesmo contrato de input."""
    top_view = prepared_views["cima"]
    front_view = prepared_views["frente"]
    back_view = prepared_views["traz"]
    left_view = prepared_views["esquerda"]
    right_view = prepared_views["direita"]

    warnings = []

    # Cada par oposto é fundido num perfil só.
    front_profile, front_stats, front_mode, front_warning = combine_opposite_profiles(front_view, back_view, "frente", "traz")
    side_profile, side_stats, side_mode, side_warning = combine_opposite_profiles(left_view, right_view, "esquerda", "direita")
    warnings.extend(item for item in [front_warning, side_warning] if item)

    target_height, target_ratio, height_metrics, height_warning = estimate_case_height(
        top_view,
        [front_view, back_view, left_view, right_view],
    )
    if height_warning:
        warnings.append(height_warning)

    top_footprint = build_top_footprint(top_view)
    volume, volume_metrics = build_volume_from_silhouettes(
        top_footprint=top_footprint,
        front_profile=front_profile,
        side_profile=side_profile,
        front_views=[front_view, back_view],
        side_views=[left_view, right_view],
        target_ratio=target_ratio,
        front_mode=front_mode,
        side_mode=side_mode,
    )

    mesh = convert_volume_to_mesh(
        volume=volume,
        target_width=float(max(top_view.bbox[2], 1)),
        target_depth=float(max(top_view.bbox[3], 1)),
        target_height=target_height,
    )

    mesh.metadata["reconstruction_mode"] = "five_view_compact"
    mesh.metadata["profile_consistency"] = {
        "frente_traz": front_stats,
        "esquerda_direita": side_stats,
    }
    mesh.metadata["height_metrics"] = height_metrics
    mesh.metadata["volume_metrics"] = volume_metrics
    mesh.metadata["warnings"] = dedupe_messages(warnings)

    return ReconstructionResult(
        mesh=mesh,
        albedo=build_neutral_albedo(prepared_views),
        normal=None,
        metadata=mesh.metadata.copy(),
        warnings=mesh.metadata["warnings"],
        debug_images=build_reconstruction_debug_images(top_footprint, volume),
    )


def combine_opposite_profiles(view_a, view_b, role_a, role_b):
    """Combina duas vistas opostas num perfil vertical comum."""
    profile_a = extract_row_width_profile(view_a)
    profile_b = extract_row_width_profile(view_b)
    difference = np.abs(profile_a - profile_b)
    mean_delta = float(difference.mean())
    peak_delta = float(difference.max())

    warning = None
    mode = "balanced"
    if mean_delta > PAIR_MEAN_DELTA_WARNING or peak_delta > PAIR_PEAK_DELTA_WARNING:
        # Se as duas vistas discordam muito, usamos a interseção para ser conservador.
        warning = f"{role_a}/{role_b}: par inconsistente, a usar envelope conservador"
        mode = "conservative"
        combined_profile = np.minimum(profile_a, profile_b)
    else:
        combined_profile = (profile_a + profile_b) * 0.5

    combined_profile = gaussian_filter1d(combined_profile, sigma=1.2, mode="nearest")
    combined_profile = np.clip(combined_profile, 0.03, 1.0)
    return combined_profile, {"mean_delta": mean_delta, "peak_delta": peak_delta, "mode": mode}, mode, warning


def extract_row_width_profile(prepared_view):
    """Transforma a máscara numa lista de larguras ao longo da altura."""
    x, y, width, height = prepared_view.bbox
    crop = prepared_view.mask[y:y + height, x:x + width] > 0
    if crop.size == 0 or height <= 0 or width <= 0:
        raise PipelineError(f"Máscara inválida para a vista {prepared_view.role}.")

    row_widths = crop.sum(axis=1).astype(np.float32) / float(max(width, 1))
    if len(row_widths) == 1:
        profile = np.full(PROFILE_SAMPLES, row_widths[0], dtype=np.float32)
    else:
        source_positions = np.linspace(0.0, 1.0, num=len(row_widths), dtype=np.float32)
        target_positions = np.linspace(0.0, 1.0, num=PROFILE_SAMPLES, dtype=np.float32)
        profile = np.interp(target_positions, source_positions, row_widths).astype(np.float32)

    return np.clip(gaussian_filter1d(profile, sigma=1.5, mode="nearest"), 0.0, 1.0)


def build_top_footprint(top_view):
    """Cria a planta do objeto a partir da vista de cima."""
    x, y, width, height = top_view.bbox
    crop = (top_view.mask[y:y + height, x:x + width] > 0).astype(np.uint8)
    if crop.size == 0:
        raise PipelineError("A vista superior não gerou footprint válido.")

    if width >= height:
        footprint_width = MAX_FOOTPRINT_RESOLUTION
        footprint_height = max(96, int(round(MAX_FOOTPRINT_RESOLUTION * height / float(max(width, 1)))))
    else:
        footprint_height = MAX_FOOTPRINT_RESOLUTION
        footprint_width = max(96, int(round(MAX_FOOTPRINT_RESOLUTION * width / float(max(height, 1)))))

    footprint = cv2.resize(crop.astype(np.float32), (footprint_width, footprint_height), interpolation=cv2.INTER_LINEAR)
    footprint = np.clip(footprint, 0.0, 1.0)
    if np.count_nonzero(footprint > 0.20) == 0:
        raise PipelineError("Footprint superior vazio após reamostragem.")
    return footprint


def estimate_case_height(top_view, side_views):
    """Estima uma altura baixa e plausível para um estojo compacto."""
    base_span = float(max(min(top_view.bbox[2], top_view.bbox[3]), 1))
    raw_ratios = [float(view.bbox[3]) / float(max(view.bbox[2], 1)) for view in side_views]
    # Damos mais peso ao perfil mais baixo para evitar alturas exageradas.
    compact_ratio = min(min(raw_ratios) * 1.10, float(np.median(raw_ratios)))
    target_ratio = float(np.clip(compact_ratio, *COMPACT_HEIGHT_RATIO_RANGE))
    target_height = base_span * target_ratio

    warning = None
    if max(raw_ratios) > target_ratio * 1.25:
        warning = "altura aparente foi reduzida para um perfil compacto e baixo"

    return target_height, target_ratio, {"raw_ratios": raw_ratios, "target_ratio": target_ratio}, warning


def build_volume_from_silhouettes(
    top_footprint,
    front_profile,
    side_profile,
    front_views,
    side_views,
    target_ratio,
    front_mode="balanced",
    side_mode="balanced",
):
    """Monta um volume 3D simples cruzando topo, frente/trás e esquerda/direita."""
    max_footprint_side = max(top_footprint.shape[0], top_footprint.shape[1])
    z_samples = int(
        np.clip(
            round(max_footprint_side * target_ratio * 1.35),
            MIN_Z_SAMPLES,
            MAX_Z_SAMPLES,
        )
    )

    depth_resolution, width_resolution = top_footprint.shape
    front_view, back_view = front_views
    left_view, right_view = side_views

    # O topo define a base de cada fatia.
    top_support = gaussian_filter(top_footprint.astype(np.float32), sigma=0.6)
    front_support = combine_view_projections(
        resize_view_projection_to_floor(front_view, width_resolution, z_samples, target_ratio),
        resize_view_projection_to_floor(back_view, width_resolution, z_samples, target_ratio),
        mode=front_mode,
    )
    side_support = combine_view_projections(
        resize_view_projection_to_floor(left_view, depth_resolution, z_samples, target_ratio),
        resize_view_projection_to_floor(right_view, depth_resolution, z_samples, target_ratio),
        mode=side_mode,
    )

    front_profile = resample_profile(front_profile, z_samples)[:, None, None]
    side_profile = resample_profile(side_profile, z_samples)[:, None, None]

    # Esta curva simples achata a base e encolhe ligeiramente a tampa.
    height_progress = np.linspace(0.0, 1.0, num=z_samples, dtype=np.float32)
    lid_mix = smoothstep(LID_START_RATIO, 0.96, height_progress)
    base_mix = 1.0 - smoothstep(0.02, 0.16, height_progress)
    section_scale = (1.0 - (0.06 * base_mix)) * (1.0 - (0.11 * lid_mix))

    volume = np.zeros((z_samples, depth_resolution, width_resolution), dtype=np.float32)
    for index in range(z_samples):
        top_slice = scale_slice(top_support, float(section_scale[index]))
        slice_strength = float(np.sqrt(np.clip(front_profile[index, 0, 0] * side_profile[index, 0, 0], 0.0, 1.0)))
        slice_volume = top_slice * front_support[index][None, :] * side_support[index][:, None]
        volume[index] = slice_volume * slice_strength

    volume = np.clip((volume - 0.12) / 0.88, 0.0, 1.0)
    volume = gaussian_filter(volume, sigma=(0.7, 0.6, 0.6))
    if float(volume.max()) <= 0.05:
        raise PipelineError("Volume reconstruído sem ocupação suficiente.")

    return volume, {
        "z_samples": z_samples,
        "target_ratio": float(target_ratio),
        "lid_start_ratio": LID_START_RATIO,
    }


def resize_view_projection_to_floor(prepared_view, target_width, target_height, target_ratio):
    """Alinha a vista pela base para todas partilharem o mesmo "chão"."""
    x, y, width, height = prepared_view.bbox
    crop = (prepared_view.mask[y:y + height, x:x + width] > 0).astype(np.float32)
    if crop.size == 0:
        raise PipelineError(f"Máscara inválida para a vista {prepared_view.role}.")

    observed_ratio = float(height) / float(max(width, 1))
    vertical_scale = min(1.0, float(target_ratio) / float(max(observed_ratio, 1e-6)))
    effective_height = max(8, int(round(target_height * vertical_scale)))
    resized = cv2.resize(crop, (target_width, effective_height), interpolation=cv2.INTER_LINEAR)
    projection = np.zeros((target_height, target_width), dtype=np.float32)
    projection[target_height - effective_height:, :] = resized
    return np.clip(gaussian_filter(projection, sigma=0.5), 0.0, 1.0)


def combine_view_projections(primary_projection, opposite_projection, mode="balanced"):
    """Combina duas projeções já alinhadas."""
    if mode == "conservative":
        combined = np.minimum(primary_projection, opposite_projection)
    else:
        union = np.maximum(primary_projection, opposite_projection)
        overlap = np.minimum(primary_projection, opposite_projection)
        combined = (0.20 * union) + (0.80 * overlap)
    return np.clip(gaussian_filter(combined, sigma=0.7), 0.0, 1.0)


def convert_volume_to_mesh(volume, target_width, target_depth, target_height):
    """Converte o volume final numa malha e ajusta a escala real."""
    vertices, faces, _, _ = marching_cubes(volume, level=VOLUME_LEVEL)
    vertices = vertices[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    current_min, current_max = mesh.bounds
    current_size = np.maximum(current_max - current_min, 1e-6)
    mesh.apply_translation(-current_min)
    mesh.apply_scale([
        target_width / current_size[0],
        target_depth / current_size[1],
        target_height / current_size[2],
    ])
    mesh.apply_translation([-target_width / 2.0, -target_depth / 2.0, 0.0])

    # Esta limpeza reduz problemas normais antes da exportação.
    mesh.merge_vertices()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    trimesh.smoothing.filter_taubin(mesh, lamb=0.45, nu=-0.5, iterations=6)
    flatten_mesh_base(mesh)
    mesh.fill_holes()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh = simplify_mesh(mesh, TARGET_FACE_COUNT)
    flatten_mesh_base(mesh)
    mesh.fix_normals()
    return mesh


def flatten_mesh_base(mesh):
    """Aplana a base para o objeto pousar de forma estável."""
    bounds_min, bounds_max = mesh.bounds
    z_span = float(max(bounds_max[2] - bounds_min[2], 1e-6))
    cutoff = bounds_min[2] + (z_span * 0.04)
    base_vertices = mesh.vertices[:, 2] <= cutoff
    mesh.vertices[base_vertices, 2] = bounds_min[2]


def simplify_mesh(mesh, target_face_count):
    """Reduz a malha se ela ficar densa demais."""
    if len(mesh.faces) <= target_face_count:
        return mesh

    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_face_count, aggression=6)
        if simplified is not None and len(simplified.faces) > 0:
            simplified.remove_unreferenced_vertices()
            return simplified
    except Exception:
        pass

    return mesh


def build_neutral_albedo(prepared_views):
    """Cria um material simples com a cor média do objeto."""
    color_samples = []
    for prepared_view in prepared_views.values():
        visible_pixels = prepared_view.masked_rgb[prepared_view.mask > 0]
        if visible_pixels.size:
            color_samples.append(np.median(visible_pixels, axis=0))

    if color_samples:
        base_color = np.clip(np.median(np.asarray(color_samples), axis=0), 0, 255).astype(np.uint8)
    else:
        base_color = np.array([232, 232, 232], dtype=np.uint8)

    texture_size = 64
    albedo = np.full((texture_size, texture_size, 3), base_color, dtype=np.uint8)
    gradient = np.linspace(1.04, 0.96, num=texture_size, dtype=np.float32)[:, None]
    return np.clip(albedo.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)


def build_reconstruction_debug_images(top_footprint, volume):
    """Guarda projeções simples para perceber a forma reconstruída."""
    top_mask = ((top_footprint > 0.45).astype(np.uint8) * 255)
    top_projection = ((volume.max(axis=0) > VOLUME_LEVEL).astype(np.uint8) * 255)
    front_projection = ((volume.max(axis=1) > VOLUME_LEVEL).astype(np.uint8) * 255)
    side_projection = ((volume.max(axis=2) > VOLUME_LEVEL).astype(np.uint8) * 255)
    return {
        "footprint_top": convert_gray_to_rgb(top_mask),
        "projection_top": convert_gray_to_rgb(top_projection),
        "projection_front": convert_gray_to_rgb(front_projection),
        "projection_side": convert_gray_to_rgb(side_projection),
    }


def resample_profile(profile, sample_count):
    source_positions = np.linspace(0.0, 1.0, num=len(profile), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=sample_count, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, profile).astype(np.float32)
    return gaussian_filter1d(resampled, sigma=1.0, mode="nearest")


def scale_slice(base_slice, scale):
    """Encolhe ou alarga uma fatia sem mudar o centro."""
    scale = float(np.clip(scale, 0.72, 1.0))
    base_depth, base_width = base_slice.shape
    scaled_width = max(2, int(round(base_width * scale)))
    scaled_depth = max(2, int(round(base_depth * scale)))
    resized = cv2.resize(base_slice, (scaled_width, scaled_depth), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(base_slice, dtype=np.float32)
    start_x = (base_width - scaled_width) // 2
    start_y = (base_depth - scaled_depth) // 2
    canvas[start_y:start_y + scaled_depth, start_x:start_x + scaled_width] = resized
    return canvas


def smoothstep(start_value, end_value, values):
    if end_value <= start_value:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip((values - start_value) / float(end_value - start_value), 0.0, 1.0)
    return normalized * normalized * (3.0 - (2.0 * normalized))


def convert_gray_to_rgb(gray_image):
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def dedupe_messages(messages):
    seen = set()
    ordered = []
    for message in messages:
        if message and message not in seen:
            seen.add(message)
            ordered.append(message)
    return ordered
