import {
  ChatAcceptResponse,
  ChatResponse,
  ClinicalReport,
  FeatureFieldInfo,
  FoodCatalogItem,
  FullPipelineResponse,
  HealthStatus,
  IntakeChatResponse,
  IntakeChatState,
  MealPlanPreferences,
  MealPlanResult,
  PatientMetadata,
  PlanSessionRecord,
  ScreeningResult,
  TxtParseResponse,
} from '../types/api'

const API_BASE = '/api'

export async function createPlanSession(dossier: Record<string, any>): Promise<{ session_id: string; current_version: number }> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dossier }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not create plan session')
  }
  return res.json()
}

export async function getPlanSession(sessionId: string): Promise<PlanSessionRecord> {
  const res = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not load plan session')
  }
  return res.json()
}

export async function askPlanChat(
  message: string,
  dossier: Record<string, any>,
  sessionId?: string | null,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, dossier, session_id: sessionId || null }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not answer this plan question')
  }
  return res.json()
}

export async function acceptPlanProposal(
  dossier: Record<string, any>,
  proposal: NonNullable<ChatResponse['proposed_meal_plan']>,
  sessionId?: string | null,
): Promise<ChatAcceptResponse> {
  const res = await fetch(`${API_BASE}/chat/accept`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dossier, proposal, session_id: sessionId || null }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not accept the proposed plan')
  }
  return res.json()
}

export async function rollbackPlanVersion(
  sessionId: string,
  targetVersion: number,
): Promise<ChatAcceptResponse> {
  const res = await fetch(`${API_BASE}/chat/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, target_version: targetVersion }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not roll back plan version')
  }
  return res.json()
}

export async function getFoodCatalog(): Promise<FoodCatalogItem[]> {
  const res = await fetch(`${API_BASE}/foods`)
  if (!res.ok) throw new Error(`Failed to load food catalog: ${res.statusText}`)
  return res.json()
}

export async function screenPatient(payload: {
  features: number[]
  metadata: Partial<PatientMetadata>
}): Promise<ScreeningResult> {
  const body = {
    schema_version: 'nhanes-87-v2',
    features: payload.features,
    weight_kg: payload.metadata.weight_kg,
    height_cm: payload.metadata.height_cm,
    age_years: payload.metadata.age_years,
    is_female: payload.metadata.is_female,
    activity_level: payload.metadata.activity_level || 'sedentary',
  }
  const res = await fetch(`${API_BASE}/screen`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Stage 1/2 screening failed')
  }
  return res.json()
}

export async function sendIntakeChat(
  message: string,
  dossier: Record<string, any>,
  currentState: IntakeChatState,
  useLlm: boolean = true,
): Promise<IntakeChatResponse> {
  const res = await fetch(`${API_BASE}/chat/intake-preferences`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      dossier,
      current_state: currentState,
      use_llm: useLlm,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not process your dietary preferences')
  }
  return res.json()
}

export async function optimizeMealPlan(
  dossier: Record<string, any>,
  preferences: MealPlanPreferences,
): Promise<MealPlanResult> {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dossier,
      dietary_preference: preferences.dietary_preference,
      allergies: preferences.allergies,
      excluded_food_ids: preferences.excluded_food_ids,
      disliked_terms: preferences.disliked_terms,
      max_active_dishes: preferences.max_active_dishes,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Stage 3 meal optimization failed')
  }
  return res.json()
}

export async function compilePlan(dossier: Record<string, any>): Promise<ClinicalReport> {
  const res = await fetch(`${API_BASE}/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dossier }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not regenerate the report')
  }
  return res.json()
}

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return res.json()
}

export async function getFeatureDefinitions(): Promise<FeatureFieldInfo[]> {
  const res = await fetch(`${API_BASE}/features`)
  if (!res.ok) throw new Error(`Failed to load features: ${res.statusText}`)
  return res.json()
}

export async function getSamplePatient(): Promise<{
  features: number[]
  metadata: PatientMetadata
  feature_count: number
}> {
  const res = await fetch(`${API_BASE}/sample-patient`)
  if (!res.ok) throw new Error(`Failed to load sample patient: ${res.statusText}`)
  return res.json()
}

export async function parseRawText(text: string): Promise<TxtParseResponse> {
  const res = await fetch(`${API_BASE}/parse-txt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Failed to parse text input')
  }
  return res.json()
}

export async function uploadTxtFile(file: File): Promise<TxtParseResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/upload-txt`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Failed to parse uploaded file')
  }
  return res.json()
}

export async function runFullPipeline(payload: {
  features: number[]
  metadata: Partial<PatientMetadata>
  backend?: 'transformers' | 'ollama'
}): Promise<FullPipelineResponse> {
  const body = {
    features: payload.features,
    weight_kg: payload.metadata.weight_kg,
    height_cm: payload.metadata.height_cm,
    age_years: payload.metadata.age_years,
    is_female: payload.metadata.is_female,
    activity_level: payload.metadata.activity_level || 'sedentary',
    backend: payload.backend || 'transformers',
  }
  const res = await fetch(`${API_BASE}/pipeline/full`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Full pipeline execution failed')
  }
  return res.json()
}

export async function runFullPipelineFile(
  file: File,
  backend: 'transformers' | 'ollama' = 'transformers',
  metadataOverrides?: Partial<PatientMetadata>
): Promise<FullPipelineResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('backend', backend)
  if (metadataOverrides?.weight_kg) formData.append('weight_kg', String(metadataOverrides.weight_kg))
  if (metadataOverrides?.height_cm) formData.append('height_cm', String(metadataOverrides.height_cm))
  if (metadataOverrides?.age_years) formData.append('age_years', String(metadataOverrides.age_years))
  if (metadataOverrides?.is_female !== undefined) formData.append('is_female', String(metadataOverrides.is_female))
  if (metadataOverrides?.activity_level) formData.append('activity_level', metadataOverrides.activity_level)

  const res = await fetch(`${API_BASE}/pipeline/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'File pipeline execution failed')
  }
  return res.json()
}
