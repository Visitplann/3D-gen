"""Segment one uploaded view using SAM, with GrabCut as fallback."""

from dataclasses import dataclass
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import SAM


DEBUG_VISUALS = os.environ.get("PIPELINE_VISUAL_DEBUG") == "1"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam2_t.pt"))
SAM_MODEL = SAM(MODEL_PATH)


@dataclass
class SegmentationResult:
    segmented: np.ndarray | None
    mask: np.ndarray | None
    source: str
    warnings: list[str]
    metrics: dict


def segment_view_image(image_bgr, role="vista"):
    """Try SAM first. If it fails, try GrabCut."""
    height, width = image_bgr.shape[:2]
    center_point = [[width // 2, height // 2]]

    try:
        results = SAM_MODEL.predict(
            source=image_bgr,
            points=center_point,
            labels=[1],
            verbose=False,
        )
    except Exception as error:
        print(f"Falhou no predict: {error}")
        return segment_with_grabcut(image_bgr, role)

    if not results or results[0].masks is None or results[0].masks.data is None or len(results[0].masks.data) == 0:
        print("Nenhuma máscara foi retornada.")
        return segment_with_grabcut(image_bgr, role)

    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = normalize_binary_mask(mask)
    if mask is None:
        print("Máscara do SAM inválida após normalização.")
        return segment_with_grabcut(image_bgr, role)

    segmented_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    warnings, metrics = collect_mask_warnings(image_bgr, mask, role, source="sam")

    if DEBUG_VISUALS:
        plotted_result = results[0].plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(plotted_result, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return SegmentationResult(
        segmented=segmented_image,
        mask=mask,
        source="sam",
        warnings=warnings,
        metrics=metrics,
    )


def segment_with_grabcut(image_bgr, role):
    """Fallback used when SAM misses the object."""
    original_height, original_width = image_bgr.shape[:2]
    resize_scale = min(1.0, 1024.0 / max(original_height, original_width))

    if resize_scale < 1.0:
        work_image = cv2.resize(
            image_bgr,
            (max(1, int(round(original_width * resize_scale))), max(1, int(round(original_height * resize_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_image = image_bgr

    work_height, work_width = work_image.shape[:2]
    mask = np.zeros((work_height, work_width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    start_rect = (
        max(1, int(work_width * 0.12)),
        max(1, int(work_height * 0.12)),
        max(2, int(work_width * 0.76)),
        max(2, int(work_height * 0.76)),
    )

    try:
        cv2.grabCut(work_image, mask, start_rect, background_model, foreground_model, 4, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return SegmentationResult(None, None, "grabcut", ["segmentação falhou"], {})

    fallback_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    if resize_scale < 1.0:
        fallback_mask = cv2.resize(fallback_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    fallback_mask = normalize_binary_mask(fallback_mask)
    if fallback_mask is None:
        return SegmentationResult(None, None, "grabcut", ["máscara inválida após fallback"], {})

    segmented_image = cv2.bitwise_and(image_bgr, image_bgr, mask=fallback_mask)
    warnings, metrics = collect_mask_warnings(image_bgr, fallback_mask, role, source="grabcut")
    warnings.insert(0, f"{role}: segmentação recorreu a GrabCut fallback")
    print("SAM falhou; a usar GrabCut fallback.")
    return SegmentationResult(
        segmented=segmented_image,
        mask=fallback_mask,
        source="grabcut",
        warnings=dedupe_messages(warnings),
        metrics=metrics,
    )


def normalize_binary_mask(mask):
    if mask is None:
        return None

    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = binary_mask.shape[:2]
    minimum_area = max(5000, int(height * width * 0.01))

    if np.count_nonzero(binary_mask) < minimum_area:
        return None

    kernel_size = max(5, int(round(min(height, width) * 0.01)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = keep_largest_component(binary_mask)
    binary_mask = fill_mask_holes(binary_mask)

    if np.count_nonzero(binary_mask) < minimum_area:
        return None

    return binary_mask


def collect_mask_warnings(image_bgr, mask, role, source):
    """Create simple warnings that explain weak input views."""
    warnings = []
    metrics = {
        "source": source,
        "area_ratio": float(np.count_nonzero(mask)) / float(mask.size),
    }

    x, y, width, height = find_mask_bbox(mask)
    metrics["bbox"] = {"x": x, "y": y, "w": width, "h": height}
    metrics["aspect_ratio"] = float(height) / float(max(width, 1))
    metrics["border_touch"] = measure_border_touch(mask)

    if role == "cima":
        return warnings, metrics

    mask_crop = (mask[y:y + height, x:x + width] > 0).astype(np.uint8)
    gray_crop = cv2.cvtColor(image_bgr[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    row_widths = mask_crop.sum(axis=1).astype(np.float32)
    occupied_rows = np.flatnonzero(row_widths > 0)
    if len(occupied_rows) == 0:
        warnings.append(f"{role}: máscara suspeita")
        return warnings, metrics

    row_widths = row_widths[occupied_rows]
    row_brightness = measure_row_brightness(gray_crop[occupied_rows], mask_crop[occupied_rows])

    body_start = max(0, int(round(len(row_widths) * 0.25)))
    body_end = max(body_start + 1, int(round(len(row_widths) * 0.75)))
    body_width = float(np.median(row_widths[body_start:body_end]))
    body_brightness = float(np.median(row_brightness[body_start:body_end]))

    tail_count = max(4, int(round(len(row_widths) * 0.15)))
    bottom_width = float(np.mean(row_widths[-tail_count:]))
    bottom_brightness = float(np.mean(row_brightness[-tail_count:]))

    bottom_spread_ratio = bottom_width / float(max(body_width, 1.0))
    bottom_brightness_ratio = bottom_brightness / float(max(body_brightness, 1.0))
    metrics["bottom_spread_ratio"] = bottom_spread_ratio
    metrics["bottom_brightness_ratio"] = bottom_brightness_ratio

    bottom_touch = metrics["border_touch"]["bottom"]
    if bottom_touch > 0.2 and bottom_spread_ratio > 0.95 and bottom_brightness_ratio < 0.82:
        warnings.append(f"{role}: possível sombra da mesa ou base espraiada")

    if metrics["aspect_ratio"] > 0.52:
        warnings.append(f"{role}: perspetiva baixa pode estar a inflacionar a altura")

    if any(metrics["border_touch"][side] > 0.18 for side in ("left", "right", "top")):
        warnings.append(f"{role}: máscara toca demasiado a moldura")

    return dedupe_messages(warnings), metrics


def keep_largest_component(mask):
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if label_count <= 1:
        return mask

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_mask_holes(mask):
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


def measure_border_touch(mask):
    height, width = mask.shape[:2]
    return {
        "top": float(np.count_nonzero(mask[0])) / float(max(width, 1)),
        "bottom": float(np.count_nonzero(mask[-1])) / float(max(width, 1)),
        "left": float(np.count_nonzero(mask[:, 0])) / float(max(height, 1)),
        "right": float(np.count_nonzero(mask[:, -1])) / float(max(height, 1)),
    }


def measure_row_brightness(gray_rows, mask_rows):
    values = []
    for row_gray, row_mask in zip(gray_rows, mask_rows):
        pixels = row_gray[row_mask > 0]
        values.append(float(np.mean(pixels)) if pixels.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def dedupe_messages(messages):
    seen = set()
    ordered = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            ordered.append(message)
    return ordered


# Backwards-compatible name used by older code.
segment_object = segment_view_image
