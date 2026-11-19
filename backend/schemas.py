"""BioNutriGPT FastAPI – Pydantic v2 request/response schemas."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class PatientInput(BaseModel):
    """87-element NHANES-format feature vector (or named feature map) plus patient metadata."""

    schema_version: str = Field("nhanes-87-v2", description="Explicit input schema version")
    features: list[float] | None = Field(None, description="Exactly 87 float32 NHANES features")
    named_features: dict[str, float] | None = Field(
        None, description="Optional explicit feature-name -> value map with documented NHANES units"
    )
    weight_kg: float | None = Field(None, gt=0, description="Patient weight in kg")
    height_cm: float | None = Field(None, gt=0, description="Patient height in cm")
    age_years: float | None = Field(None, ge=0, description="Patient age in years")
    is_female: bool | None = Field(None, description="Biological sex override (true=female, false=male)")
    activity_level: Literal["sedentary", "light", "moderate", "active"] | None = Field(
        "sedentary", description="Physical activity level"
    )

    @field_validator("features")
    @classmethod
    def _check_length(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) == 88:
            raise ValueError(
                "Ambiguous 88-value legacy vector rejected to prevent silent feature misalignment. "
                "Provide exactly 87 features matching schema nhanes-87-v2 (HbA1c excluded) or use named_features."
            )
        if len(v) != 87:
            raise ValueError(f"features must have exactly 87 values; got {len(v)}")
        return v


class RawTextInput(BaseModel):
    """Raw text content containing spaced or comma-separated numbers and optional # metadata comments."""
    text: str = Field(..., description="Raw text from a .txt file or paste box")


class DossierPayload(BaseModel):
    """Unified JSON dossier produced by Stages 1+2 (and optionally Stage 3), with optional meal preferences."""
    dossier: dict[str, Any]
    dietary_preference: Literal["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian", "vegetarian", "non_vegetarian"] = "veg_nonveg_combo"
    allergies: list[str] = Field(default_factory=list, description="Reported allergies/intolerances")
    excluded_food_ids: list[str] = Field(default_factory=list, description="Catalog food_ids to exclude")
    disliked_terms: list[str] = Field(default_factory=list, description="Free-text disliked dish/ingredient keywords")
    max_active_dishes: int | None = Field(None, ge=4, le=8, description="Optional maximum distinct dishes")


class IntakeChatRequest(BaseModel):
    """Free-text conversational preference intake turn before generating a meal plan."""
    message: str = Field(..., min_length=1, max_length=1000)
    dossier: dict[str, Any]
    current_state: dict[str, Any] = Field(default_factory=dict)
    use_llm: bool = Field(True, description="Use local Qwen 2.5 3B LLM to parse typed preferences")


class IntakeChatResponse(BaseModel):
    assistant_reply: str
    state: dict[str, Any]
    ready_to_generate: bool = False
    llm_used: bool = False
    meal_plan: dict[str, Any] | None = None


class LLMCompileRequest(BaseModel):
    """Request body for the LLM-backed Stage 4 compilation endpoint."""
    dossier: dict[str, Any]
    backend: Literal["transformers", "ollama"] = "transformers"


class ChatRequest(BaseModel):
    """A question or controlled revision request about a completed plan."""
    message: str = Field(..., min_length=1, max_length=1000)
    dossier: dict[str, Any]
    session_id: str | None = Field(None, description="Optional persistent plan session ID")
    use_llm_explanation: bool = Field(False, description="Optional local Qwen phrasing over backend facts")


class ChatResponse(BaseModel):
    answer: str
    intent: Literal["plan", "nutrient", "portion", "risk", "revision", "history", "rollback", "allergy", "unsupported"]
    sources: list[str] = Field(default_factory=list)
    tool_calls_used: list[str] = Field(default_factory=list)
    proposed_meal_plan: dict[str, Any] | None = None
    revision_diff: dict[str, Any] | None = None
    session_id: str | None = None
    plan_version: int | None = None
    plan_flagged_unsuitable: bool = False


class ChatAcceptRequest(BaseModel):
    dossier: dict[str, Any]
    proposal: dict[str, Any]
    session_id: str | None = None


class ChatRollbackRequest(BaseModel):
    session_id: str
    target_version: int = Field(..., ge=1)


class SessionCreateRequest(BaseModel):
    dossier: dict[str, Any]
    session_id: str | None = None


class FullPipelineRequest(BaseModel):
    """End-to-end pipeline request (Stages 1 → 4)."""
    schema_version: str = Field("nhanes-87-v2", description="Explicit input schema version")
    features: list[float] | None = Field(None, description="Exactly 87 float32 NHANES features")
    named_features: dict[str, float] | None = Field(None, description="Optional named NHANES feature dict")
    weight_kg: float | None = Field(None, gt=0)
    height_cm: float | None = Field(None, gt=0)
    age_years: float | None = Field(None, ge=0)
    is_female: bool | None = None
    activity_level: Literal["sedentary", "light", "moderate", "active"] | None = "sedentary"
    backend: Literal["transformers", "ollama"] = "transformers"

    @field_validator("features")
    @classmethod
    def _check_length(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) == 88:
            raise ValueError(
                "Ambiguous 88-value legacy vector rejected to prevent silent feature misalignment. "
                "Provide exactly 87 features matching schema nhanes-87-v2 (HbA1c excluded) or use named_features."
            )
        if len(v) != 87:
            raise ValueError(f"features must have exactly 87 values; got {len(v)}")
        return v


# ---------------------------------------------------------------------------
# Feature Metadata info for dynamic typing UI
# ---------------------------------------------------------------------------

class FeatureFieldInfo(BaseModel):
    name: str
    label: str
    category: str
    unit: str
    default_val: float
    description: str
    min_val: float = 0.0
    max_val: float = 10000.0
    step: float = 1.0


class TxtParseResponse(BaseModel):
    features: list[float]
    metadata: dict[str, Any]
    feature_count: int
    is_valid: bool
    summary: str


# ---------------------------------------------------------------------------
# Sub-models for ScreeningResult
# ---------------------------------------------------------------------------

class RiskProbabilities(BaseModel):
    diabetes: float
    obesity: float
    cvd: float


class RiskClassification(BaseModel):
    status: str
    threshold: float
    positive: bool


class FeatureDriver(BaseModel):
    feature: str
    weight_pct: float


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------

class ScreeningResult(BaseModel):
    patient_metadata: dict[str, Any]
    probabilities: RiskProbabilities
    classifications: dict[str, RiskClassification]
    top_feature_drivers: dict[str, list[FeatureDriver]]
    stage2_prescription: dict[str, Any]
    energy_prescription: dict[str, Any]
    certified_for_stage3: bool
    latency_ms: dict[str, float]
    dossier: dict[str, Any]


class FoodItem(BaseModel):
    food_id: str
    name: str
    category: str | None = "other"
    serving_unit: str = "serving"
    quantity: int = 1
    clinical_rationale: str | None = None


class MealSlot(BaseModel):
    meal: str
    time: str
    items: list[FoodItem] = Field(default_factory=list)
    guidance: str = ""


class MealPlanResult(BaseModel):
    accepted: bool = False
    failure: dict[str, Any] | None = None
    clinical_verification: dict[str, Any] = Field(default_factory=dict)
    solved_food_basket: list[dict[str, Any]]
    meal_schedule: list[MealSlot]
    delivered_nutrition: list[dict[str, Any]]
    solver_metadata: dict[str, Any]
    latency_ms: float
    dossier: dict[str, Any]


class ClinicalReport(BaseModel):
    patient_metadata: dict[str, Any]
    stage1_diagnostic_assessment: dict[str, Any]
    stage2_counterfactual_prescription: dict[str, Any]
    stage3_discrete_meal_synthesis: dict[str, Any]
    compiled_patient_report: dict[str, Any]
    integrity_checks: dict[str, Any]
    compiler_metadata: dict[str, Any]
    clinical_disclaimer: str


class FullPipelineResponse(BaseModel):
    pipeline_status: str = 'unknown'
    pipeline_failure: dict[str, Any] | None = None
    screening: ScreeningResult
    meal_plan: MealPlanResult | None
    clinical_report: ClinicalReport | None
    pipeline_latency: dict[str, float]
    mode_a_subsecond: bool


class HealthStatus(BaseModel):
    status: str
    cuda_available: bool
    cuda_device: str | None
    vram_total_gb: float | None
    vram_free_gb: float | None
    pipeline_loaded: bool


class FoodCatalogItem(BaseModel):
    """Slim summary of one food item from the sourced cooked catalog."""
    food_id: str
    name: str
    category: str
    serving_unit: str
    serving_weight_g: float
    data_source: str
    catalog_version: str = "sourced-cooked-v4-lactoveg-complete"
    unknown_nutrients: list[str] = Field(default_factory=list)
    allergen_info: dict[str, Any] = Field(default_factory=dict)


class NutritionChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)


class NutritionChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    status: Literal["answered", "general_response", "insufficient_evidence"]


class ExtractedModelValue(BaseModel):
    name: str
    label: str
    value: float
    unit: str


class DocumentPipelineResponse(BaseModel):
    status: Literal["needs_more_input", "completed"]
    extracted_values: list[ExtractedModelValue] = Field(default_factory=list)
    extracted_metadata: dict[str, Any] = Field(default_factory=dict)
    missing_features: list[dict[str, str]] = Field(default_factory=list)
    rejected_values: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    notices: list[str] = Field(default_factory=list)
    questions: list[dict[str, Any]] = Field(default_factory=list)
    model_output: FullPipelineResponse | None = None
