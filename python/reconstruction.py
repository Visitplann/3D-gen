"""Compatibility layer for code that still imports `reconstruction.py`."""

from five_view_pipeline.config import REQUIRED_VIEW_FILES
from five_view_pipeline.models import PipelineError as ReconstructionError
from five_view_pipeline.models import PreparedView as ViewRecord
from five_view_pipeline.models import ReconstructionResult
from five_view_pipeline.reconstruction_stage import build_compact_case_reconstruction
from five_view_pipeline.view_stage import build_prepared_view, validate_prepared_view


def prepare_view_record(role, file_name, mask, clean, gray):
    return build_prepared_view(
        role=role,
        file_name=file_name,
        mask=mask,
        masked_rgb=clean,
        masked_gray=gray,
        segmentation_source="legacy",
        segmentation_metrics={},
        segmentation_warnings=[],
    )


def validate_view_record(view_record):
    return validate_prepared_view(view_record)


def build_five_view_reconstruction(view_records):
    return build_compact_case_reconstruction(view_records)
