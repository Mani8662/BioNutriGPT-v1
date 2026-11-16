#!/usr/bin/env python3
"""Stage 4 Clinical Compiler (Compatibility Shim).

All Stage 4 capabilities (along with Stages 1 to 3) have been consolidated
into the master unified file:
    infer_bionutrigpt_pipeline_Stage_1_to_4.py

This file re-exports all Stage 4 symbols for backwards compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from infer_bionutrigpt_pipeline_Stage_1_to_4 import (
    DEFAULT_DOSSIER_FILE as DEFAULT_INPUT_NAME,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_REPORT_FILE as DEFAULT_OUTPUT_NAME,
    DISEASE_LABELS,
    MEAL_SLOTS,
    BackendResult,
    CompilerConfig,
    Stage4Error,
    ValidatedDossier,
    _canonicalize_report,
    _clinical_rationale_for_food,
    _coerce_text,
    _default_meal_guidance,
    _expected_serving_totals,
    _integrity_report,
    _load_local_qwen_cached,
    _run_ollama_backend,
    _run_transformers_backend,
    _validate_and_canonicalize_schedule,
    _validate_no_nutrition_number_claims,
    build_compiler_prompt,
    build_deterministic_schedule,
    compile_patient_report,
    extract_json_object,
    load_dossier,
    main,
    print_clinical_report,
    run_backend,
    save_report,
    validate_dossier,
)

if __name__ == "__main__":
    raise SystemExit(main())
