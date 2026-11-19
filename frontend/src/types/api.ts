export interface FeatureFieldInfo {
  name: string
  label: string
  category: string
  unit: string
  default_val: number
  min_val: number
  max_val: number
  step: number
  description: string
}

export interface PatientMetadata {
  weight_kg: number
  height_cm: number
  age_years: number
  is_female: boolean
  activity_level: 'sedentary' | 'light' | 'moderate' | 'active'
}

export interface TxtParseResponse {
  features: number[]
  metadata: Partial<PatientMetadata>
  feature_count: number
  is_valid: boolean
  summary: string
}

export interface ExtractedModelValue {
  name: string
  label: string
  value: number
  unit: string
}

export interface DocumentPipelineResponse {
  status: 'needs_more_input' | 'completed'
  extracted_values: ExtractedModelValue[]
  extracted_metadata: Record<string, unknown>
  missing_features: Array<{ name: string; label: string; category: string; unit: string }>
  rejected_values: Array<{ name: string; value: unknown; reason: string }>
  truncated: boolean
  notices: string[]
  questions: Array<{ key: string; label: string; unit: string; category: string; kind: 'number' | 'select'; min?: number; max?: number; options?: string[] }>
  model_output: FullPipelineResponse | null
}

export interface RiskProbabilities {
  diabetes: number
  obesity: number
  cvd: number
}

export interface RiskClassification {
  status: string
  threshold: number
  positive: boolean
}

export interface FeatureDriver {
  feature: string
  weight_pct: number
}

export interface ScreeningResult {
  patient_metadata: PatientMetadata
  probabilities: RiskProbabilities
  classifications: Record<'diabetes' | 'obesity' | 'cvd', RiskClassification>
  top_feature_drivers: Record<'bio' | 'diet' | 'life' | 'demo', FeatureDriver[]>
  stage2_prescription: Record<string, number>
  energy_prescription: {
    bmr_kcal: number
    tdee_kcal: number
    target_calories: number
    deficit_kcal: number
  }
  certified_for_stage3: boolean
  latency_ms: {
    stage1_ms: number
    stage2_ms: number
    total_s1_s2_ms: number
  }
  dossier: Record<string, any>
}

export interface FoodItem {
  food_id: string
  name: string
  category: string
  serving_unit: string
  quantity: number
  clinical_rationale?: string
}

export interface MealSlot {
  meal: string
  time: string
  items: FoodItem[]
  guidance: string
}

export interface MealPlanResult {
  accepted: boolean
  failure?: { code: string; message: string; details?: Record<string, any> } | null
  clinical_verification?: Record<string, any>
  solved_food_basket: Array<{
    food_id: string
    name: string
    category: string
    serving_unit: string
    quantity: number
    serving_weight_g: number
    selection_cost: number
  }>
  meal_schedule: MealSlot[]
  delivered_nutrition: Array<{
    nutrient: string
    delivered?: number
    target?: number
    min_bound?: number
    max_bound?: number
    within_bounds?: boolean
    constraint_applied?: boolean
    unknown_food_count?: number
    value_status?: string
    rule_provenance?: Record<string, any>
    prescribed_target?: number
    delivered_value?: number
    lower_bound?: number
    upper_bound?: number
    delta?: number
    unit?: string
  }>
  solver_metadata: {
    solver: string
    status: string
    solve_time_sec: number
    num_variables: number
    num_constraints: number
    active_dishes: number
  }
  latency_ms: number
  dossier: Record<string, any>
}

export interface ClinicalReport {
  patient_metadata: PatientMetadata
  stage1_diagnostic_assessment: Record<string, any>
  stage2_counterfactual_prescription: Record<string, any>
  stage3_discrete_meal_synthesis: Record<string, any>
  compiled_patient_report: {
    health_narrative: string
    biomarker_driver_explanations: Array<{
      modality: string
      plain_language_explanation: string
    }>
    meal_schedule: MealSlot[]
    culinary_directives: string[]
    lifestyle_hydration_guidance: {
      hydration: string
      post_meal_activity: string
      sleep_hygiene: string
    }
    closing_message: string
  }
  integrity_checks: {
    stage3_basket_preserved: boolean
    no_new_food_ids: boolean
    all_servings_accounted_for: boolean
    llm_nutrient_calculation_authority: boolean
  }
  compiler_metadata: {
    stage: number
    backend_requested: string
    backend_used: string
    model: string
    generation_time_ms: number
    schedule_repaired_by_circuit_breaker: boolean
    zero_calculation_authority_enforced: boolean
  }
  clinical_disclaimer: string
}

export interface FullPipelineResponse {
  screening: ScreeningResult
  meal_plan: MealPlanResult | null
  clinical_report: ClinicalReport | null
  pipeline_latency: {
    stage1_diagnostics_ms: number
    stage2_counterfactual_ms: number
    stage3_milp_solver_ms: number
    stage4_compiler_ms: number
    total_pipeline_ms: number
  }
  mode_a_subsecond: boolean
}

export interface HealthStatus {
  status: string
  cuda_available: boolean
  cuda_device: string | null
  vram_total_gb: number | null
  vram_free_gb: number | null
  pipeline_loaded: boolean
}


export interface RevisionDiff {
  added: Array<{ food_id: string; name: string; quantity: number }>
  removed: Array<{ food_id: string; name: string; quantity: number }>
  quantity_changed: Array<{ food_id: string; name: string; previous_quantity: number; new_quantity: number }>
  unchanged: Array<{ food_id: string; name: string; quantity: number }>
  total_serving_delta: number
}

export interface ChatResponse {
  answer: string
  intent: 'plan' | 'nutrient' | 'portion' | 'risk' | 'revision' | 'history' | 'rollback' | 'allergy' | 'unsupported'
  sources: string[]
  tool_calls_used?: string[]
  revision_diff?: RevisionDiff | null
  session_id?: string | null
  plan_version?: number | null
  plan_flagged_unsuitable?: boolean
  proposed_meal_plan?: {
    solved_food_basket: FoodItem[]
    delivered_nutrition: Array<Record<string, any>>
    clinical_verification: Record<string, any>
    revision: Record<string, any>
  } | null
}

export interface PlanSessionVersion {
  version: number
  created_at: string
  reason: string
  diff_from_previous?: RevisionDiff | null
  flagged_unsuitable_for_allergies?: string[]
}

export interface PlanSessionRecord {
  session_id: string
  created_at: string
  updated_at: string
  current_version: number
  versions: PlanSessionVersion[]
  preferences: {
    excluded_food_ids: string[]
    preferred_food_ids: string[]
    locked_food_quantities: Record<string, number>
    max_active_dishes: number | null
    reported_allergies: string[]
  }
}

export interface ChatAcceptResponse {
  session_id?: string | null
  plan_version?: number
  revision_diff?: RevisionDiff
  dossier: Record<string, any>
  meal_plan: MealPlanResult
}

export interface FoodCatalogItem {
  food_id: string
  name: string
  category: string
  serving_unit: string
  serving_weight_g: number
  data_source: string
  catalog_version?: string
  unknown_nutrients?: string[]
  allergen_info?: {
    known_allergens?: string[]
    unknown_allergens_possible?: boolean
  }
}

export interface IntakeChatState {
  dietary_mode?: 'pure_vegetarian' | 'veg_nonveg_combo' | 'pure_non_vegetarian' | null
  allergies_answered?: boolean
  allergies?: string[]
  dislikes_answered?: boolean
  disliked_terms?: string[]
  excluded_food_ids?: string[]
}

export interface IntakeChatResponse {
  assistant_reply: string
  state: IntakeChatState
  ready_to_generate: boolean
  llm_used: boolean
  meal_plan?: MealPlanResult | null
}

export interface MealPlanPreferences {
  dietary_preference: 'pure_vegetarian' | 'veg_nonveg_combo' | 'pure_non_vegetarian'
  allergies: string[]
  excluded_food_ids: string[]
  disliked_terms: string[]
  max_active_dishes: number | null
}