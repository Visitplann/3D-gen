"""Load and validate the fixed input folder."""

import os

import cv2

from .config import REQUIRED_VIEW_FILES, VALID_IMAGE_EXTENSIONS
from .models import LoadedViewImage, PipelineError


def load_required_view_images(input_dir):
    """Load the five required images and keep their role explicit."""
    image_files = sorted(
        file_name
        for file_name in os.listdir(input_dir)
        if file_name.lower().endswith(VALID_IMAGE_EXTENSIONS)
    )

    if not image_files:
        raise PipelineError(f"Nenhuma imagem encontrada em: {input_dir}")

    validate_input_folder_contract(image_files)

    loaded_views = {}
    lower_lookup = {file_name.lower(): file_name for file_name in image_files}

    for role, expected_name in REQUIRED_VIEW_FILES.items():
        actual_name = lower_lookup.get(expected_name)
        if actual_name is None:
            raise PipelineError(f"Falta a vista '{role}' ({expected_name}).")

        image_path = os.path.join(input_dir, actual_name)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise PipelineError(f"Não foi possível abrir a vista '{role}' ({actual_name}).")

        loaded_views[role] = LoadedViewImage(
            role=role,
            file_name=actual_name,
            image_bgr=image_bgr,
        )

    return loaded_views


def validate_input_folder_contract(image_files):
    """Fail early if the folder does not match the fixed bundle."""
    required_files = set(REQUIRED_VIEW_FILES.values())
    lower_lookup = {file_name.lower(): file_name for file_name in image_files}

    missing = sorted(required_files.difference(lower_lookup))
    unexpected = sorted(name for name in lower_lookup if name not in required_files)

    if not missing and not unexpected:
        return

    parts = []
    if missing:
        missing_roles = [
            role for role, file_name in REQUIRED_VIEW_FILES.items() if file_name in missing
        ]
        parts.append("faltam vistas: " + ", ".join(missing_roles))
    if unexpected:
        parts.append("ficheiros inesperados: " + ", ".join(unexpected))

    raise PipelineError("Contrato 5-view inválido: " + "; ".join(parts))

