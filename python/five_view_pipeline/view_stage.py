"""Prepara cada imagem para a etapa de reconstrução."""

import cv2
import numpy as np

from preprocessing import prepare_image_for_reconstruction
from segmentation_sam import segment_view_image

from .config import MAX_CENTER_OFFSET_RATIO, MIN_BBOX_SIDE_RATIO, MIN_FOREGROUND_RATIO
from .models import PipelineError, PreparedView, ViewPreparationResult


def prepare_views_for_reconstruction(loaded_views):
    """Segmenta, limpa e valida as 5 vistas antes da reconstrução."""
    prepared_views = {}
    report_views = {}
    warnings = []
    debug_images = {}

    for role, loaded_view in loaded_views.items():
        print(f"Processando vista {role}: {loaded_view.file_name}")

        segmentation = segment_view_image(loaded_view.image_bgr, role=role)
        if segmentation.segmented is None or segmentation.mask is None:
            raise PipelineError(f"Segmentação inválida para a vista '{role}' ({loaded_view.file_name}).")

        masked_gray, masked_rgb = prepare_image_for_reconstruction(segmentation.segmented)
        prepared_view = build_prepared_view(
            role=role,
            file_name=loaded_view.file_name,
            mask=segmentation.mask,
            masked_rgb=masked_rgb,
            masked_gray=masked_gray,
            segmentation_source=segmentation.source,
            segmentation_metrics=segmentation.metrics,
            segmentation_warnings=segmentation.warnings,
        )

        validation_error = validate_prepared_view(prepared_view)
        if validation_error:
            raise PipelineError(f"Vista '{role}' inválida ({loaded_view.file_name}): {validation_error}")

        prepared_views[role] = prepared_view
        report_views[role] = build_view_report(prepared_view)
        warnings.extend(prepared_view.segmentation_warnings)
        debug_images[role] = build_view_debug_images(loaded_view.image_bgr, prepared_view.mask)

    return ViewPreparationResult(
        prepared_views=prepared_views,
        report_views=report_views,
        warnings=_dedupe_preserve_order(warnings),
        debug_images=debug_images,
    )


def build_prepared_view(
    role,
    file_name,
    mask,
    masked_rgb,
    masked_gray,
    segmentation_source,
    segmentation_metrics,
    segmentation_warnings,
):
    """Cria um registo simples com tudo o que a reconstrução precisa."""
    clean_mask = normalize_binary_mask(mask)
    if role != "cima":
        # Nas vistas laterais e frontal tentamos cortar sombra de mesa.
        clean_mask = trim_shadow_near_table(clean_mask, masked_gray)
        clean_mask = normalize_binary_mask(clean_mask)

    bbox = find_mask_bbox(clean_mask)
    area_ratio = float(np.count_nonzero(clean_mask)) / float(clean_mask.size)
    centroid = find_mask_centroid(clean_mask)
    focus_score = measure_focus(masked_gray, bbox)

    masked_rgb = masked_rgb.copy()
    masked_rgb[clean_mask == 0] = 0
    masked_gray = masked_gray.copy()
    masked_gray[clean_mask == 0] = 0

    return PreparedView(
        role=role,
        file_name=file_name,
        mask=clean_mask,
        masked_rgb=masked_rgb,
        masked_gray=masked_gray,
        bbox=bbox,
        area_ratio=area_ratio,
        centroid=centroid,
        focus_score=focus_score,
        segmentation_source=segmentation_source,
        segmentation_metrics=segmentation_metrics,
        segmentation_warnings=segmentation_warnings,
    )


def validate_prepared_view(prepared_view):
    """Confirma que a vista tem qualidade mínima para seguir em frente."""
    mask_height, mask_width = prepared_view.mask.shape[:2]
    _, _, width, height = prepared_view.bbox

    if np.count_nonzero(prepared_view.mask) == 0:
        return "máscara vazia"

    if prepared_view.area_ratio < MIN_FOREGROUND_RATIO:
        return f"área segmentada demasiado pequena ({prepared_view.area_ratio:.3f})"

    min_width = max(16, int(mask_width * MIN_BBOX_SIDE_RATIO))
    min_height = max(16, int(mask_height * MIN_BBOX_SIDE_RATIO))
    if width < min_width or height < min_height:
        return f"bounding box degenerada ({width}x{height})"

    center_x, center_y = prepared_view.centroid
    offset_x = abs(center_x - (mask_width / 2.0)) / float(mask_width)
    offset_y = abs(center_y - (mask_height / 2.0)) / float(mask_height)
    if offset_x > MAX_CENTER_OFFSET_RATIO or offset_y > MAX_CENTER_OFFSET_RATIO:
        return f"objeto demasiado afastado do centro (offset_x={offset_x:.3f}, offset_y={offset_y:.3f})"

    return None


def build_view_report(prepared_view):
    return {
        "file_name": prepared_view.file_name,
        "segmentation_source": prepared_view.segmentation_source,
        "segmentation_metrics": prepared_view.segmentation_metrics,
        "record_bbox": {
            "x": prepared_view.bbox[0],
            "y": prepared_view.bbox[1],
            "w": prepared_view.bbox[2],
            "h": prepared_view.bbox[3],
        },
        "area_ratio": prepared_view.area_ratio,
        "focus_score": prepared_view.focus_score,
        "warnings": prepared_view.segmentation_warnings,
    }


def build_view_debug_images(original_bgr, clean_mask):
    """Cria imagens simples para perceber o que foi segmentado."""
    overlay = original_bgr.copy()
    color = np.zeros_like(overlay)
    color[:, :] = (32, 180, 64)
    blended = cv2.addWeighted(overlay, 1.0, cv2.bitwise_and(color, color, mask=clean_mask), 0.45, 0)
    return {
        "original": original_bgr,
        "mask": clean_mask,
        "overlay": blended,
    }


def normalize_binary_mask(mask):
    """Limpa ruído da máscara e mantém só a zona principal do objeto."""
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    binary_mask = keep_largest_component(binary_mask)
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask

    kernel_size = max(5, int(round(min(binary_mask.shape[:2]) * 0.01)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = keep_largest_component(binary_mask)
    binary_mask = fill_mask_holes(binary_mask)
    return keep_largest_component(binary_mask)


def trim_shadow_near_table(mask, masked_gray):
    """Tenta cortar a sombra escura que aparece por baixo do objeto."""
    x, y, width, height = find_mask_bbox(mask)
    if height <= 0 or width <= 0:
        return mask

    mask_crop = (mask[y:y + height, x:x + width] > 0).astype(np.uint8)
    gray_crop = masked_gray[y:y + height, x:x + width]
    aspect_ratio = float(height) / float(max(width, 1))

    if aspect_ratio < 0.50:
        return mask

    row_widths = mask_crop.sum(axis=1).astype(np.float32)
    visible_rows = np.flatnonzero(row_widths > 0)
    if len(visible_rows) < 20:
        return mask

    row_widths = row_widths[visible_rows]
    row_brightness = measure_row_brightness(gray_crop[visible_rows], mask_crop[visible_rows])

    # Esta zona costuma representar o "corpo" do objeto sem sombra.
    body_start = max(0, int(round(len(row_widths) * 0.25)))
    body_end = max(body_start + 1, int(round(len(row_widths) * 0.70)))
    body_width = float(np.median(row_widths[body_start:body_end]))
    body_brightness = float(np.median(row_brightness[body_start:body_end]))

    # Procuramos mais abaixo por uma mudança brusca de largura e brilho.
    scan_start = max(body_end, int(round(len(visible_rows) * 0.64)))
    window_size = max(12, int(round(len(visible_rows) * 0.035)))
    cut_row_index = None

    for row_index in range(scan_start, max(scan_start + 1, len(visible_rows) - window_size)):
        width_ratio = float(np.median(row_widths[row_index:row_index + window_size])) / float(max(body_width, 1.0))
        brightness_ratio = float(np.median(row_brightness[row_index:row_index + window_size])) / float(max(body_brightness, 1.0))
        if width_ratio > 0.82 and brightness_ratio < 0.76:
            cut_row_index = row_index
            break

    if cut_row_index is None:
        return mask

    cut_row = y + visible_rows[cut_row_index]
    trimmed_mask = mask.copy()
    trimmed_mask[cut_row:, :] = 0
    return trimmed_mask


def keep_largest_component(mask):
    """Se existirem várias manchas, mantém só a maior."""
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if label_count <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_mask_holes(mask):
    """Preenche pequenos buracos dentro da máscara."""
    filled = mask.copy()
    height, width = filled.shape[:2]
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, holes)


def find_mask_bbox(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return (0, 0, 0, 0)
    return tuple(int(value) for value in cv2.boundingRect(coords))


def find_mask_centroid(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        height, width = mask.shape[:2]
        return (width / 2.0, height / 2.0)
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def measure_focus(masked_gray, bbox):
    """Usa a variância do Laplaciano como medida simples de nitidez."""
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return 0.0
    crop = masked_gray[y:y + height, x:x + width]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_32F).var())


def measure_row_brightness(gray_rows, mask_rows):
    values = []
    for row_gray, row_mask in zip(gray_rows, mask_rows):
        pixels = row_gray[row_mask > 0]
        values.append(float(np.mean(pixels)) if pixels.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _dedupe_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
