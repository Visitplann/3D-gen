"""Shared constants for the fixed 5-view pipeline."""

REQUIRED_VIEW_FILES = {
    "frente": "frente.jpg",
    "esquerda": "esquerda.jpg",
    "direita": "direita.jpg",
    "traz": "traz.jpg",
    "cima": "cima.jpg",
}

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
INPUT_FOLDER_NAME = "test_obj"
OUTPUT_MODEL_NAME = "test_obj.glb"
OUTPUT_REPORT_NAME = "test_obj_report.txt"
MODEL_FOLDER_NAME = "model"
MATERIALS_FOLDER_NAME = "materials"
REPORTS_FOLDER_NAME = "reports"
DEBUG_FOLDER_NAME = "debug"
DEBUG_RUN_FOLDER_NAME = INPUT_FOLDER_NAME
ALBEDO_FILE_NAME = "albedo.png"
NORMAL_FILE_NAME = "normal.png"

PROFILE_SAMPLES = 192
MAX_FOOTPRINT_RESOLUTION = 192
MIN_FOREGROUND_RATIO = 0.02
MIN_BBOX_SIDE_RATIO = 0.12
MAX_CENTER_OFFSET_RATIO = 0.28
PAIR_MEAN_DELTA_WARNING = 0.19
PAIR_PEAK_DELTA_WARNING = 0.40
COMPACT_HEIGHT_RATIO_RANGE = (0.30, 0.48)
VOLUME_LEVEL = 0.45
MIN_Z_SAMPLES = 72
MAX_Z_SAMPLES = 128
TARGET_FACE_COUNT = 80000
LID_START_RATIO = 0.58
