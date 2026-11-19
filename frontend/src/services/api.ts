import {
  ChatAcceptResponse,
  ChatResponse,
  AccountUser,
  AdminSummary,
  ClinicalReport,
  FoodCatalogItem,
  IntakeChatResponse,
  IntakeChatState,
  DocumentPipelineResponse,
  FeatureFieldInfo,
  FullPipelineResponse,
  HealthStatus,
  MealPlanPreferences,
  MealPlanResult,
  PlanSessionRecord,
  PatientMetadata,
  RiskHistoryEntry,
  ScreeningResult,
  TxtParseResponse,
} from '../types/api'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`${API_BASE}${path}`, { ...init, credentials: 'include' })
}

export async function uploadDocumentForPipeline(
  file: File | null,
  answers?: Record<string, string>,
  intakeId?: string,
): Promise<DocumentPipelineResponse> {
  const formData = new FormData()
  if (file) formData.append('file', file)
  if (answers) formData.append('answers', JSON.stringify(answers))
  if (intakeId) formData.append('intake_id', intakeId)
  const res = await apiFetch('/pipeline/document', { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not extract values from this file')
  }
  return res.json()
}

export async function checkHealth(): Promise<HealthStatus> {
  const res = await apiFetch('/health')
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return res.json()
}

export async function getFeatureDefinitions(): Promise<FeatureFieldInfo[]> {
  const res = await apiFetch('/features')
  if (!res.ok) throw new Error(`Failed to load features: ${res.statusText}`)
  return res.json()
}

export async function getSamplePatient(): Promise<{
  features: number[]
  metadata: PatientMetadata
  feature_count: number
}> {
  const res = await apiFetch('/sample-patient')
  if (!res.ok) throw new Error(`Failed to load sample patient: ${res.statusText}`)
  return res.json()
}

export async function parseRawText(text: string): Promise<TxtParseResponse> {
  const res = await apiFetch('/parse-txt', {
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
  const res = await apiFetch('/upload-txt', {
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
  const res = await apiFetch('/pipeline/full', {
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

  const res = await apiFetch('/pipeline/upload', {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'File pipeline execution failed')
  }
  return res.json()
}


export async function createPlanSession(dossier: Record<string, any>): Promise<{ session_id: string; current_version: number }> {
  const res = await apiFetch('/sessions', {
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
  const res = await apiFetch(`/sessions/${encodeURIComponent(sessionId)}`)
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
  const res = await apiFetch('/chat', {
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
  const res = await apiFetch('/chat/accept', {
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
  const res = await apiFetch('/chat/rollback', {
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
  const res = await apiFetch('/foods')
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
  const res = await apiFetch('/screen', {
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
  const res = await apiFetch('/chat/intake-preferences', {
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
  const res = await apiFetch('/optimize', {
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
  const res = await apiFetch('/compile', {
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

async function accountRequest(path: string, method: string, payload?: Record<string, unknown>): Promise<Response> {
  return apiFetch(path, {
    method,
    headers: payload ? { 'Content-Type': 'application/json' } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  })
}

export async function registerAccount(email: string, password: string): Promise<AccountUser> {
  const res = await accountRequest('/auth/register', 'POST', { email, password })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not create account')
  }
  return res.json()
}

export async function loginAccount(email: string, password: string): Promise<AccountUser> {
  const res = await accountRequest('/auth/login', 'POST', { email, password })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not sign in')
  }
  return res.json()
}

export async function getCurrentAccount(): Promise<AccountUser | null> {
  const res = await apiFetch('/auth/me')
  if (res.status === 401) return null
  if (!res.ok) throw new Error('Could not load account')
  return res.json()
}

export async function logoutAccount(): Promise<void> {
  await accountRequest('/auth/logout', 'POST')
}

export async function getRiskHistory(): Promise<RiskHistoryEntry[]> {
  const res = await apiFetch('/history')
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not load risk history')
  }
  return res.json()
}

export async function saveRiskHistory(
  screening: ScreeningResult,
  source: RiskHistoryEntry['source'],
): Promise<RiskHistoryEntry> {
  const res = await accountRequest('/history', 'POST', {
    source,
    diabetes: screening.probabilities.diabetes,
    obesity: screening.probabilities.obesity,
    cvd: screening.probabilities.cvd,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not save this assessment')
  }
  return res.json()
}

export async function deleteRiskHistoryEntry(assessmentId: string): Promise<void> {
  const res = await apiFetch(`/history/${encodeURIComponent(assessmentId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Could not delete this history entry')
}

export async function getAdminSummary(): Promise<AdminSummary> {
  const res = await apiFetch('/admin/summary')
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not load the admin dashboard')
  }
  return res.json()
}
