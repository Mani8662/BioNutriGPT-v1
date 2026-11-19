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
import math
import os
import traceback
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from document_ingestion import DocumentExtractionError, extract_document_text

from schemas import (
    ChatAcceptRequest,
    ChatRequest,
    ChatResponse,
    ChatRollbackRequest,
    ClinicalReport,
    DossierPayload,
    DocumentPipelineResponse,
    FeatureFieldInfo,
    FoodCatalogItem,
    FullPipelineRequest,
    FullPipelineResponse,
    HealthStatus,
    IntakeChatRequest,
    IntakeChatResponse,
    LLMCompileRequest,
    NutritionChatRequest,
    NutritionChatResponse,
    MealPlanResult,
    PatientInput,
    RawTextInput,
    ScreeningResult,
    SessionCreateRequest,
    TxtParseResponse,
)
import pipeline_service as ps
import chat_service
import plan_store

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


@app.post("/api/pipeline/document", response_model=DocumentPipelineResponse, tags=["Complete Pipeline"])
async def run_document_pipeline(
    file: UploadFile = File(...),
    backend: Literal["transformers", "ollama"] | None = Form(None),
    answers: str | None = Form(None),
) -> DocumentPipelineResponse:
    """Extract explicit feature values, run the ML pipeline when the vector is complete, then compile with the LLM."""
    try:
        content = await file.read(8 * 1024 * 1024 + 1)
        extracted_text, truncated = extract_document_text(file.filename or "", content)
        extracted = None
        if (file.filename or "").lower().endswith((".txt", ".csv")):
            try:
                values, parsed_metadata = ps.parse_raw_text(extracted_text)
                parsed_features: dict[str, float] = {}
                rejected_values: list[dict[str, Any]] = []
                for field, value in zip(ps.FEATURE_DEFINITIONS, values):
                    if math.isfinite(value) and field["min_val"] <= value <= field["max_val"]:
                        parsed_features[field["name"]] = value
                    else:
                        rejected_values.append({
                            "name": field["name"],
                            "value": value,
                            "reason": "outside model input range",
                        })
                extracted = {
                    "features": parsed_features,
                    "metadata": parsed_metadata,
                    "rejected_values": rejected_values,
                }
            except ValueError:
                # Unstructured TXT/CSV content is passed to the LLM extractor below.
                pass
        if extracted is None:
            extracted = await run_in_threadpool(ps.extract_pipeline_features, extracted_text)
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Uploaded document extraction failed")
        raise HTTPException(status_code=503, detail="The local LLM could not extract values from this document.") from exc

    extracted_by_name = extracted["features"]
    missing_features = [
        {key: str(field[key]) for key in ("name", "label", "category", "unit")}
        for field in ps.FEATURE_DEFINITIONS
        if field["name"] not in extracted_by_name
    ]
    extracted_values = [
        {
            "name": field["name"],
            "label": field["label"],
            "value": extracted_by_name[field["name"]],
            "unit": field["unit"],
        }
        for field in ps.FEATURE_DEFINITIONS
        if field["name"] in extracted_by_name
    ]
    notices: list[str] = []
    if truncated:
        notices.append("Only the supported portion of the uploaded document was read.")
    if extracted["rejected_values"]:
        notices.append("Some extracted values were outside the model’s allowed ranges and were excluded.")
    questions = _document_questions(missing_features, extracted["metadata"])
    if answers is None and questions:
        notices.append("Answer the questions about the patient’s lifestyle and demographics. Other missing fields will use the model’s saved training-set imputation values.")
        return DocumentPipelineResponse(
            status="needs_more_input",
            extracted_values=extracted_values,
            extracted_metadata=extracted["metadata"],
            missing_features=missing_features,
            rejected_values=extracted["rejected_values"],
            truncated=truncated,
            notices=notices,
            questions=questions,
        )

    try:
        supplied_answers = json.loads(answers or "{}")
        if not isinstance(supplied_answers, dict):
            raise ValueError("Answers must be a JSON object.")
        _apply_document_answers(extracted_by_name, extracted["metadata"], questions, supplied_answers)
        _impute_document_features(extracted_by_name)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Please answer all required questions with valid values: {exc}") from exc

    feature_vector = [extracted_by_name[field["name"]] for field in ps.FEATURE_DEFINITIONS]
    metadata_overrides = {
        key: value for key, value in extracted["metadata"].items() if value is not None
    }
    selected_backend = backend or os.getenv("BIONUTRIGPT_STAGE4_BACKEND", "transformers")
    try:
        model_output = await run_in_threadpool(
            ps.run_full_pipeline, feature_vector, metadata_overrides, selected_backend
        )
    except Exception as exc:
        LOGGER.exception("Pipeline failed after document extraction")
        raise HTTPException(status_code=500, detail="The extracted values could not be processed by the project model.") from exc

    if model_output.get("clinical_report") is None:
        notices.append("The model returned screening output, but its safety gate did not certify a meal plan for LLM compilation.")
    return DocumentPipelineResponse(
        status="completed",
        extracted_values=extracted_values,
        extracted_metadata=extracted["metadata"],
        missing_features=[],
        rejected_values=extracted["rejected_values"],
        truncated=truncated,
        notices=notices,
        questions=[],
        model_output=model_output,
    )


def _document_questions(missing: list[dict[str, str]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    missing_names = {field["name"] for field in missing}
    categories = {field["category"] for field in missing}
    questions: list[dict[str, Any]] = []

    def number(key: str, label: str, unit: str, low: float, high: float, category: str) -> None:
        questions.append({"key": key, "label": label, "unit": unit, "min": low, "max": high, "kind": "number", "category": category})

    def select(key: str, label: str, options: list[str], category: str) -> None:
        questions.append({"key": key, "label": label, "unit": "", "kind": "select", "options": options, "category": category})

    if "RIDAGEYR" in missing_names: number("age_years", "Age", "years", 18, 100, "Demographics")
    if "SEX_FEMALE" in missing_names: select("sex", "Sex used by this model", ["female", "male"], "Demographics")
    if metadata.get("weight_kg") is None: number("weight_kg", "Weight", "kg", 1, 500, "Demographics")
    if metadata.get("height_cm") is None: number("height_cm", "Height", "cm", 30, 280, "Demographics")
    if metadata.get("activity_level") is None or "Lifestyle & Physical Activity" in categories:
        select("activity_level", "Usual physical activity", ["sedentary", "light", "moderate", "active"], "Lifestyle · Activity")
    if categories.intersection({"Lifestyle & Sleep"}):
        number("sleep_hours", "Typical sleep per night", "hours", 3, 14, "Lifestyle · Sleep")
        select("sleep_disorder", "Have you been told you have a sleep disorder?", ["yes", "no", "prefer not to say"], "Lifestyle · Sleep")
    if categories.intersection({"Lifestyle & Smoking"}):
        select("smoking_status", "Smoking status", ["never", "former", "some days", "every day", "prefer not to say"], "Lifestyle · Smoking")
    return questions


def _apply_document_answers(features: dict[str, float], metadata: dict[str, Any], questions: list[dict[str, Any]], answers: dict[str, Any]) -> None:
    for question in questions:
        key = question["key"]
        if key not in answers or answers[key] in (None, ""):
            raise ValueError(f"{question['label']} is required")
        value = answers[key]
        if question["kind"] == "select":
            if value not in question["options"]: raise ValueError(f"Invalid answer for {question['label']}")
            if key == "sex":
                features["SEX_FEMALE"] = 1.0 if value == "female" else 0.0
                metadata["is_female"] = value == "female"
            elif key == "activity_level":
                metadata[key] = value
                met_minutes = {"sedentary": 0.0, "light": 600.0, "moderate": 1440.0, "active": 2400.0}[value]
                features["PA_TOTAL_MET_MIN_WK"] = met_minutes
                for code in ("PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"):
                    features[code] = 0.0 if value == "sedentary" else 1.0
            elif key == "sleep_disorder":
                if value != "prefer not to say": features["SLQ050"] = 1.0 if value == "yes" else 0.0
            elif key == "smoking_status":
                smoking_keys = {"never": "SMOKE_NEVER", "former": "SMOKE_FORMER", "some days": "SMOKE_SOMEDAYS", "every day": "SMOKE_EVERYDAY"}
                if value != "prefer not to say":
                    for code in smoking_keys.values(): features[code] = 0.0
                    features[smoking_keys[value]] = 1.0
        else:
            number = float(value)
            if not math.isfinite(number) or not question["min"] <= number <= question["max"]:
                raise ValueError(f"{question['label']} must be between {question['min']} and {question['max']}")
            if key in {"weight_kg", "height_cm"}: metadata[key] = number
            elif key == "age_years": features["RIDAGEYR"] = number; metadata[key] = number
            elif key == "sleep_hours":
                features["SLEEP_HOURS"] = number
                features["SLEEP_DEV_FROM_7_5"] = number - 7.5


def _impute_document_features(features: dict[str, float]) -> None:
    artifacts = ps.p12.s2._load_preprocessing_artifacts(ps.p12.DEFAULT_PREPROCESSOR)
    allowed = {field["name"] for field in ps.FEATURE_DEFINITIONS}
    for name, value in artifacts.get("continuous_medians", {}).items():
        if name in allowed: features.setdefault(name, float(value))
    for name, value in artifacts.get("binary_modes", {}).items():
        if name in allowed: features.setdefault(name, 0.0 if float(value) == 2.0 else float(value))
    for field in ps.FEATURE_DEFINITIONS:
        name = field["name"]
        if name in features: continue
        if name.endswith("_MISS"):
            features[name] = 1.0
        elif name.startswith(("RACE_", "MARITAL_", "SMOKE_")):
            features[name] = 0.0
    derived = {
        "CHOL_HDL_RATIO": ("LBXTC", "LBDHDD", 1.0),
        "NON_HDL_TO_HDL_RATIO": ("NON_HDL_CHOL", "LBDHDD", 1.0),
        "PROT_PCT_KCAL": ("DR1TPROT", "DR1TKCAL", 4.0),
        "CARB_PCT_KCAL": ("DR1TCARB", "DR1TKCAL", 4.0),
        "FAT_PCT_KCAL": ("DR1TTFAT", "DR1TKCAL", 9.0),
        "FIBER_PER_1000_KCAL": ("DR1TFIBE", "DR1TKCAL", 1000.0),
        "SUGAR_TO_FIBER_RATIO": ("DR1TSUGR", "DR1TFIBE", 1.0),
        "SODIUM_POTASSIUM_RATIO": ("DR1TSODI", "DR1TPOTA", 1.0),
    }
    for name, (numerator, denominator, multiplier) in derived.items():
        if name not in features:
            divisor = features[denominator]
            features[name] = features[numerator] * multiplier / divisor if divisor else 0.0
    if "SAT_TO_UNSAT_FAT_RATIO" not in features:
        unsaturated = features.get("DR1TMFAT", 0.0) + features.get("DR1TPFAT", 0.0)
        features["SAT_TO_UNSAT_FAT_RATIO"] = features.get("DR1TSFAT", 0.0) / unsaturated if unsaturated else 0.0
    absent = [field["name"] for field in ps.FEATURE_DEFINITIONS if field["name"] not in features]
    if absent:
        raise ValueError("Training preprocessing data has no imputation value for: " + ", ".join(absent))

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
    """Return list of sourced cooked dishes with nutrient provenance and allergen info."""
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

@app.post("/api/sessions", tags=["Stage 4"])
def create_plan_session(payload: SessionCreateRequest) -> dict[str, Any]:
    """Create a persistent versioned meal-plan session separated from observed Stage 1/2 inputs."""
    try:
        record = plan_store.create_session(payload.dossier, session_id=payload.session_id)
        return {
            "session_id": record["session_id"],
            "current_version": record["current_version"],
            "versions_count": len(record["versions"]),
            "created_at": record["created_at"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}", tags=["Stage 4"])
def get_plan_session(session_id: str) -> dict[str, Any]:
    """Retrieve persistent session state, version history, preferences, and audit log."""
    try:
        return plan_store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat/intake-preferences", response_model=IntakeChatResponse, tags=["Stage 4"])
def chat_intake_preferences(payload: IntakeChatRequest) -> IntakeChatResponse:
    """Conversational free-text preference intake (dietary style, allergies, dislikes) -> triggers Stage 3."""
    try:
        res = chat_service.converse_intake_preferences(
            payload.message,
            payload.dossier,
            current_state=payload.current_state,
            use_llm=payload.use_llm,
        )
        return IntakeChatResponse(**res)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Intake chat error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Could not process intake preferences.") from exc


@app.post("/api/chat", response_model=ChatResponse, tags=["Stage 4"])
def chat_about_plan(payload: ChatRequest) -> ChatResponse:
    """Answer from catalog facts or return a validated, unaccepted controlled revision."""
    try:
        return ChatResponse(**chat_service.answer(
            payload.message,
            payload.dossier,
            session_id=payload.session_id,
            use_llm_explanation=payload.use_llm_explanation,
        ))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Chat error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Could not check this plan request.") from exc


@app.post("/api/chat/accept", tags=["Stage 4"])
def accept_chat_plan(payload: ChatAcceptRequest) -> dict[str, Any]:
    """Revalidate a proposal from catalog data before accepting it and saving a new version."""
    try:
        return chat_service.accept(payload.dossier, payload.proposal, session_id=payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Revision acceptance error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Could not accept this plan.") from exc


@app.post("/api/chat/rollback", tags=["Stage 4"])
def rollback_chat_plan(payload: ChatRollbackRequest) -> dict[str, Any]:
    """Rollback a persistent session to an earlier accepted plan version."""
    try:
        return chat_service.rollback(payload.session_id, payload.target_version)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Rollback error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Could not roll back plan version.") from exc

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
        res = ps.run_screening(
            input_data.features,
            metadata_overrides,
            named_features=input_data.named_features,
            schema_version=input_data.schema_version,
        )
        return ScreeningResult(**res)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Screening error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 1/2 screening failed: {exc}") from exc


@app.post("/api/optimize", response_model=MealPlanResult, tags=["Stage 3"])
def run_meal_optimization_stage(payload: DossierPayload) -> MealPlanResult:
    """Run Stage 3 (HiGHS MILP Meal Synthesis) incorporating dietary preferences, allergies, and dislikes."""
    try:
        res = ps.run_optimization(
            payload.dossier,
            dietary_preference=payload.dietary_preference,
            allergies=payload.allergies,
            excluded_food_ids=payload.excluded_food_ids,
            disliked_terms=payload.disliked_terms,
            max_active_dishes=payload.max_active_dishes,
        )
        return MealPlanResult(**res)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Meal optimization error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 3 optimization failed: {exc}") from exc


@app.post("/api/compile", response_model=ClinicalReport, tags=["Stage 4"])
def run_clinical_compiler(payload: DossierPayload) -> ClinicalReport:
    """Run Stage 4 using real on-device Local GPU LLM (Qwen 2.5 3B NF4)."""
    try:
        res = ps.run_compilation(payload.dossier, backend="transformers")
        return ClinicalReport(**res)
    except ps.p14.Stage4Error as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Compiler error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stage 4 compilation failed: {exc}") from exc


@app.post("/api/compile/llm", response_model=ClinicalReport, tags=["Stage 4"])
def run_clinical_compiler_llm(payload: LLMCompileRequest) -> ClinicalReport:
    """Run Stage 4 using real on-device GPU LLM (Qwen 2.5 3B NF4)."""
    try:
        res = ps.run_compilation(payload.dossier, backend=payload.backend)
        return ClinicalReport(**res)
    except ps.p14.Stage4Error as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        res = ps.run_full_pipeline(
            req.features,
            metadata_overrides,
            backend=req.backend,
            named_features=req.named_features,
            schema_version=req.schema_version,
        )
        return FullPipelineResponse(**res)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
