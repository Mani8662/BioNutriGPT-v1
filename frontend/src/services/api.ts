import {
  ClinicalReport,
  DocumentChatResponse,
  FeatureFieldInfo,
  FullPipelineResponse,
  HealthStatus,
  MealPlanResult,
  PatientMetadata,
  ScreeningResult,
  TxtParseResponse,
} from '../types/api'

const API_BASE = '/api'

export async function uploadDocumentForChat(file: File): Promise<DocumentChatResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/chat/upload`, { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Could not read this file')
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
