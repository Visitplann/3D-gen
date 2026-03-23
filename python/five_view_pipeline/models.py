from dataclasses import dataclass

import numpy as np
import trimesh


class PipelineError(RuntimeError):
    """Raised when the fixed input bundle cannot produce a valid model."""


@dataclass
class PipelinePaths:
    input_dir: str
    output_root_dir: str
    model_dir: str
    materials_dir: str
    reports_dir: str
    debug_root_dir: str
    debug_run_dir: str
    output_model_path: str
    report_path: str
    albedo_path: str
    normal_path: str


@dataclass
class LoadedViewImage:
    role: str
    file_name: str
    image_bgr: np.ndarray


@dataclass
class PreparedView:
    role: str
    file_name: str
    mask: np.ndarray
    masked_rgb: np.ndarray
    masked_gray: np.ndarray
    bbox: tuple[int, int, int, int]
    area_ratio: float
    centroid: tuple[float, float]
    focus_score: float
    segmentation_source: str
    segmentation_metrics: dict
    segmentation_warnings: list[str]


@dataclass
class ViewPreparationResult:
    prepared_views: dict[str, PreparedView]
    report_views: dict
    warnings: list[str]
    debug_images: dict[str, dict[str, np.ndarray]]


@dataclass
class ReconstructionResult:
    mesh: trimesh.Trimesh
    albedo: np.ndarray
    normal: np.ndarray | None
    metadata: dict
    warnings: list[str]
    debug_images: dict[str, np.ndarray]
