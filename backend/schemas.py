"""BioNutriGPT FastAPI – Pydantic v2 request/response schemas."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class PatientInput(BaseModel):
    """87-element NHANES-format feature vector plus optional patient metadata."""

    features: list[float] = Field(..., description="Exactly 87 float32 NHANES features (or 88 legacy)")
    weight_kg: float | None = Field(None, gt=0, description="Patient weight in kg")
    height_cm: float | None = Field(None, gt=0, description="Patient height in cm")
    age_years: float | None = Field(None, ge=0, description="Patient age in years")
    is_female: bool | None = Field(None, description="Biological sex override (true=female, false=male)")
    activity_level: Literal["sedentary", "light", "moderate", "active"] | None = Field(
        "sedentary", description="Physical activity level"
    )

    @field_validator("features")
    @classmethod
    def _check_length(cls, v: list[float]) -> list[float]:
        # Handle 87 features (standard) or 88 features (legacy with HbA1c at index 0 or elsewhere)
        if len(v) == 88:
            # If 88 values provided, we tolerate it and slice the last 87 or first 87 as appropriate
            # In our pipeline s1.NUM_FEATURES is 87
            return v[:87]
        if len(v) != 87:
            raise ValueError(f"features must have exactly 87 values; got {len(v)}")
        return v


class RawTextInput(BaseModel):
    """Raw text content containing spaced or comma-separated numbers and optional # metadata comments."""
    text: str = Field(..., description="Raw text from a .txt file or paste box")


class DossierPayload(BaseModel):
    """Unified JSON dossier produced by Stages 1+2 (and optionally Stage 3)."""
    dossier: dict[str, Any]


class LLMCompileRequest(BaseModel):
    """Request body for the LLM-backed Stage 4 compilation endpoint."""
    dossier: dict[str, Any]
    backend: Literal["transformers", "ollama"] = "transformers"


class NutritionChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)


class NutritionChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    status: Literal["answered", "general_response", "insufficient_evidence"]


class DocumentChatResponse(NutritionChatResponse):
    extracted_text: str
    truncated: bool = False


class FullPipelineRequest(BaseModel):
    """End-to-end pipeline request (Stages 1 → 4)."""
    features: list[float] = Field(..., description="87 float32 NHANES features")
    weight_kg: float | None = Field(None, gt=0)
    height_cm: float | None = Field(None, gt=0)
    age_years: float | None = Field(None, ge=0)
    is_female: bool | None = None
    activity_level: Literal["sedentary", "light", "moderate", "active"] | None = "sedentary"
    backend: Literal["transformers", "ollama"] = "transformers"

    @field_validator("features")
    @classmethod
    def _check_length(cls, v: list[float]) -> list[float]:
        if len(v) == 88:
            return v[:87]
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
    """Slim summary of one food item from the IFCT catalog."""
    food_id: str
    name: str
    category: str
    serving_unit: str
    serving_weight_g: float
    data_source: str
