"""Ponto de entrada do pipeline fixo de 5 vistas.

Ao correr este ficheiro, o código lê `input/test_obj`
e escreve o resultado final em `python/output`.
"""

import os
import traceback

from five_view_pipeline.config import (
    ALBEDO_FILE_NAME,
    DEBUG_FOLDER_NAME,
    DEBUG_RUN_FOLDER_NAME,
    INPUT_FOLDER_NAME,
    MATERIALS_FOLDER_NAME,
    MODEL_FOLDER_NAME,
    NORMAL_FILE_NAME,
    OUTPUT_MODEL_NAME,
    OUTPUT_REPORT_NAME,
    REPORTS_FOLDER_NAME,
)
from five_view_pipeline.input_stage import load_required_view_images
from five_view_pipeline.models import PipelineError, PipelinePaths
from five_view_pipeline.output_stage import (
    clear_run_debug_folder,
    ensure_output_folders,
    remove_old_output_layout,
    save_pipeline_outputs,
    write_debug_images,
    write_pipeline_report,
)
from five_view_pipeline.reconstruction_stage import build_compact_case_reconstruction
from five_view_pipeline.view_stage import prepare_views_for_reconstruction


def run_pipeline(input_dir, output_root_dir):
    """Executa todas as etapas do pipeline pela ordem certa."""
    pipeline_paths = build_pipeline_paths(input_dir, output_root_dir)
    ensure_output_folders(pipeline_paths)
    # Limpa restos do layout antigo para o output ficar sempre previsível.
    remove_old_output_layout(pipeline_paths.output_root_dir)
    # Limpa apenas o debug desta execução.
    clear_run_debug_folder(pipeline_paths.debug_run_dir)

    try:
        loaded_views = load_required_view_images(pipeline_paths.input_dir)
        print("Entrou em 5-view mode.")

        view_result = prepare_views_for_reconstruction(loaded_views)
        reconstruction_result = build_compact_case_reconstruction(view_result.prepared_views)
        report = build_pipeline_report(view_result, reconstruction_result)

        save_pipeline_outputs(reconstruction_result, pipeline_paths)
        write_pipeline_report(
            report_path=pipeline_paths.report_path,
            report=report,
            reconstruction_result=reconstruction_result,
            output_model_path=pipeline_paths.output_model_path,
        )
        write_debug_images(
            debug_dir=pipeline_paths.debug_run_dir,
            view_debug_images=view_result.debug_images,
            reconstruction_debug_images=reconstruction_result.debug_images,
        )

        for warning in report["warnings"]:
            print(f"WARNING: {warning}")

        print(
            "Reconstrução 5-view concluída:",
            f"{len(reconstruction_result.mesh.vertices)} vertices,",
            f"{len(reconstruction_result.mesh.faces)} faces",
        )
        print(f"Sucesso! Ficheiro exportado para: {pipeline_paths.output_model_path}")

    except PipelineError as error:
        print("Ocorreu um erro crítico no pipeline:")
        print(error)
        traceback.print_exc()
    except Exception as error:
        print("Ocorreu um erro inesperado no pipeline:")
        print(error)
        traceback.print_exc()


def build_pipeline_paths(input_dir, output_root_dir):
    """Centraliza todos os caminhos usados pelo pipeline."""
    model_dir = os.path.join(output_root_dir, MODEL_FOLDER_NAME)
    materials_dir = os.path.join(output_root_dir, MATERIALS_FOLDER_NAME)
    reports_dir = os.path.join(output_root_dir, REPORTS_FOLDER_NAME)
    debug_root_dir = os.path.join(output_root_dir, DEBUG_FOLDER_NAME)
    debug_run_dir = os.path.join(debug_root_dir, DEBUG_RUN_FOLDER_NAME)

    return PipelinePaths(
        input_dir=input_dir,
        output_root_dir=output_root_dir,
        model_dir=model_dir,
        materials_dir=materials_dir,
        reports_dir=reports_dir,
        debug_root_dir=debug_root_dir,
        debug_run_dir=debug_run_dir,
        output_model_path=os.path.join(model_dir, OUTPUT_MODEL_NAME),
        report_path=os.path.join(reports_dir, OUTPUT_REPORT_NAME),
        albedo_path=os.path.join(materials_dir, ALBEDO_FILE_NAME),
        normal_path=os.path.join(materials_dir, NORMAL_FILE_NAME),
    )


def build_pipeline_report(view_result, reconstruction_result):
    """Junta num só bloco a informação útil para o report final."""
    report_warnings = dedupe_messages(view_result.warnings + reconstruction_result.warnings)
    return {
        "views": view_result.report_views,
        "warnings": report_warnings,
        "reconstruction": reconstruction_result.metadata,
    }


def dedupe_messages(messages):
    """Remove mensagens repetidas sem perder a ordem original."""
    seen = set()
    ordered = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            ordered.append(message)
    return ordered


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "input", INPUT_FOLDER_NAME)
    output_root_dir = os.path.join(project_root, "python", "output")

    print(f"Resolved input path: {input_dir}")
    run_pipeline(input_dir, output_root_dir)
