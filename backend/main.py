"""BioNutriGPT FastAPI Clinical Decision Support Backend.

Exposes RESTful endpoints for:
  - Stage 1: Deep Cross-Attention Diagnostics (T2D, Obesity, CVD risks & drivers)
  - Stage 2: Gradient-Guided Counterfactual Optimization (4 Clinical Safety Filters)
  - Stage 3: Discrete Indian Meal Synthesis (HiGHS MILP & ICMR-NIN IFCT 2017)
  - Stage 4: Chrono-Nutrition Compiler & AST Safety Guardrails
  - Input parsing: Supports both typed numeric fields and TXT file uploads.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    ClinicalReport,
    DossierPayload,
    FeatureFieldInfo,
    FoodCatalogItem,
    FullPipelineRequest,
    FullPipelineResponse,
    HealthStatus,
    LLMCompileRequest,
    NutritionChatRequest,
    NutritionChatResponse,
    MealPlanResult,
    PatientInput,
    RawTextInput,
    ScreeningResult,
    TxtParseResponse,
)
import pipeline_service as ps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("BioNutriGPT.API")

app = FastAPI(
    title="BioNutriGPT Clinical Decision Support API",
    description="End-to-End Cardiometabolic Risk Diagnostics & Discrete Indian Meal Synthesis",
    version="1.0.0",
)


@app.post("/api/chat", response_model=NutritionChatResponse, tags=["Nutrition Chat"])
def nutrition_chat(payload: NutritionChatRequest) -> NutritionChatResponse:
    """Retrieve local food nutrient facts and explain them with the local LLM."""
    try:
        return NutritionChatResponse(**ps.run_nutrition_chat(payload.message, payload.history))
    except Exception as exc:
        LOGGER.exception("Nutrition chat failed")
        raise HTTPException(status_code=503, detail="Nutrition chat is temporarily unavailable.") from exc

# Configure CORS for local development and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & Metadata Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthStatus, tags=["System"])
def get_health() -> HealthStatus:
    """Return system readiness, CUDA status, and VRAM utilization."""
    return HealthStatus(**ps.get_hardware_status())


@app.get("/api/features", response_model=list[FeatureFieldInfo], tags=["Metadata"])
def get_feature_definitions() -> list[FeatureFieldInfo]:
    """Return clinical descriptions, units, labels, and defaults for all 87 patient input features."""
    return [FeatureFieldInfo(**f) for f in ps.FEATURE_DEFINITIONS]


@app.get("/api/sample-patient", tags=["Metadata"])
def get_sample_patient() -> dict[str, Any]:
    """Return pre-configured sample patient inputs and clinical metadata for one-click demo."""
    return {
        "features": ps.SAMPLE_PATIENT_VALUES,
        "metadata": ps.SAMPLE_METADATA,
        "feature_count": len(ps.SAMPLE_PATIENT_VALUES),
    }


@app.get("/api/foods", response_model=list[FoodCatalogItem], tags=["Database"])
def get_foods() -> list[FoodCatalogItem]:
    """Return list of validated cooked Indian dishes from ICMR-NIN IFCT 2017 & RecipeDB."""
    return [FoodCatalogItem(**item) for item in ps.get_food_catalog()]


# ---------------------------------------------------------------------------
# TXT Input Parsing Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/parse-txt", response_model=TxtParseResponse, tags=["Input Processing"])
def parse_text_payload(payload: RawTextInput) -> TxtParseResponse:
    """Parse raw text string (space/comma separated numbers + metadata comments)."""
    try:
        values, metadata = ps.parse_raw_text(payload.text)
        return TxtParseResponse(
            features=values,
            metadata=metadata,
            feature_count=len(values),
            is_valid=True,
            summary=f"Successfully parsed {len(values)} numeric features and {len(metadata)} clinical metadata attributes."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse text input: {exc}"
        ) from exc


@app.post("/api/upload-txt", response_model=TxtParseResponse, tags=["Input Processing"])
async def upload_txt_file(file: UploadFile = File(...)) -> TxtParseResponse:
    """Upload a .txt patient vector file and extract the 87 features and metadata."""
    if not file.filename.lower().endswith((".txt", ".csv")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .txt or .csv patient vector."
        )

    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8-sig", errors="replace")
        values, metadata = ps.parse_raw_text(content_str)
        return TxtParseResponse(
            features=values,
            metadata=metadata,
            feature_count=len(values),
            is_valid=True,
            summary=f"Successfully extracted {len(values)} features from '{file.filename}'."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error parsing file '{file.filename}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline Modular Stage Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/screen", response_model=ScreeningResult, tags=["Stages 1 & 2"])
def run_screening_stage(input_data: PatientInput) -> ScreeningResult:
    """Run Stage 1 (Diagnostics) and Stage 2 (Counterfactual Optimization)."""
    try:
        metadata_overrides = {
            "weight_kg": input_data.weight_kg,
            "height_cm": input_data.height_cm,
            "age_years": input_data.age_years,
            "is_female": input_data.is_female,
            "activity_level": input_data.activity_level,
        }
        res = ps.run_screening(input_data.features, metadata_overrides)
        return ScreeningResult(**res)
    except Exception as exc:
        LOGGER.error("Screening error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 1/2 screening failed: {exc}") from exc


@app.post("/api/optimize", response_model=MealPlanResult, tags=["Stage 3"])
def run_meal_optimization_stage(payload: DossierPayload) -> MealPlanResult:
    """Run Stage 3 (HiGHS Mixed-Integer Linear Programming Meal Synthesis)."""
    try:
        res = ps.run_optimization(payload.dossier)
        return MealPlanResult(**res)
    except Exception as exc:
        LOGGER.error("Meal optimization error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 3 optimization failed: {exc}") from exc


@app.post("/api/compile", response_model=ClinicalReport, tags=["Stage 4"])
def run_clinical_compiler(payload: DossierPayload) -> ClinicalReport:
    """Run Stage 4 using real on-device Local GPU LLM (Qwen 2.5 3B NF4)."""
    try:
        res = ps.run_compilation(payload.dossier, backend="transformers")
        return ClinicalReport(**res)
    except Exception as exc:
        LOGGER.error("Compiler error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 4 compilation failed: {exc}") from exc


@app.post("/api/compile/llm", response_model=ClinicalReport, tags=["Stage 4"])
def run_clinical_compiler_llm(payload: LLMCompileRequest) -> ClinicalReport:
    """Run Stage 4 using real on-device GPU LLM (Qwen 2.5 3B NF4)."""
    try:
        res = ps.run_compilation(payload.dossier, backend=payload.backend)
        return ClinicalReport(**res)
    except Exception as exc:
        LOGGER.error("LLM compiler error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 4 LLM compilation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# End-to-End Pipeline Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/full", response_model=FullPipelineResponse, tags=["Complete Pipeline"])
def run_full_pipeline_json(req: FullPipelineRequest) -> FullPipelineResponse:
    """Execute complete end-to-end 4-stage pipeline from typed features."""
    try:
        metadata_overrides = {
            "weight_kg": req.weight_kg,
            "height_cm": req.height_cm,
            "age_years": req.age_years,
            "is_female": req.is_female,
            "activity_level": req.activity_level,
        }
        res = ps.run_full_pipeline(req.features, metadata_overrides, backend=req.backend)
        return FullPipelineResponse(**res)
    except Exception as exc:
        LOGGER.error("Full pipeline error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Full pipeline execution failed: {exc}") from exc


@app.post("/api/pipeline/upload", response_model=FullPipelineResponse, tags=["Complete Pipeline"])
async def run_full_pipeline_file(
    file: UploadFile = File(...),
    backend: Literal["transformers", "ollama"] = Form("transformers"),
    weight_kg: float | None = Form(None),
    height_cm: float | None = Form(None),
    age_years: float | None = Form(None),
    is_female: bool | None = Form(None),
    activity_level: Literal["sedentary", "light", "moderate", "active"] | None = Form("sedentary"),
) -> FullPipelineResponse:
    """Execute complete end-to-end 4-stage pipeline directly from an uploaded .txt patient vector file."""
    try:
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8-sig", errors="replace")
        values, extracted_meta = ps.parse_raw_text(content_str)

        # Merge extracted metadata with explicit form overrides
        meta_overrides = dict(extracted_meta)
        if weight_kg is not None:
            meta_overrides["weight_kg"] = weight_kg
        if height_cm is not None:
            meta_overrides["height_cm"] = height_cm
        if age_years is not None:
            meta_overrides["age_years"] = age_years
        if is_female is not None:
            meta_overrides["is_female"] = is_female
        if activity_level is not None:
            meta_overrides["activity_level"] = activity_level

        res = ps.run_full_pipeline(values, meta_overrides, backend=backend)
        return FullPipelineResponse(**res)
    except Exception as exc:
        LOGGER.error("Pipeline file upload error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File pipeline execution failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
