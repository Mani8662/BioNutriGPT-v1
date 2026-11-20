import type {
  HealthStatus,
  ScreeningResult,
  PatientMetadata,
  MealPlanPreferences,
  MealPlanResult,
  ClinicalReport,
  ChatResponse,
  ChatAcceptResponse,
  PlanSessionRecord,
  TxtParseResponse,
} from "./types/api";

async function request<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch(`/api${path}`, {
      method: method || (body ? "POST" : "GET"),
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      if (response.status === 422)
        throw new Error(
          "Some details could not be accepted. Review your entries and try again.",
        );
      if (response.status >= 500)
        throw new Error(
          "The service could not complete this request. Your details are still here. Please try again shortly.",
        );
      throw new Error(
        typeof data?.detail === "string" &&
        !/traceback|cuda|qwen|transformer|milp/i.test(data.detail)
          ? data.detail
          : "This action could not be completed. Please review your details and try again.",
      );
    }
    if (!data)
      throw new Error(
        "The nutrition service is not connected. Please start the service and try again.",
      );
    return data as T;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError")
      throw new Error(
        "This is taking longer than expected. Please wait a moment before trying again.",
      );
    if (error instanceof TypeError)
      throw new Error(
        "We cannot reach the nutrition service. Check your connection and try again.",
      );
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
export const api = {
  health: () => request<HealthStatus>("/health"),
  parse: (text: string) => request<TxtParseResponse>("/parse-txt", { text }),
  screen: (features: number[], metadata: PatientMetadata) =>
    request<ScreeningResult>("/screen", {
      schema_version: "nhanes-87-v2",
      features,
      ...metadata,
    }),
  plan: (dossier: Record<string, any>, preferences: MealPlanPreferences) =>
    request<MealPlanResult>("/optimize", { dossier, ...preferences }),
  report: (dossier: Record<string, any>) =>
    request<ClinicalReport>("/compile", { dossier }),
  session: (dossier: Record<string, any>) =>
    request<{ session_id: string; current_version: number }>("/sessions", {
      dossier,
    }),
  history: (id: string) =>
    request<PlanSessionRecord>(`/sessions/${encodeURIComponent(id)}`),
  chat: (
    message: string,
    dossier: Record<string, any>,
    session_id: string | null,
    use_llm_interpretation = false,
  ) => request<ChatResponse>("/chat", { message, dossier, session_id, use_llm_interpretation }),
  accept: (
    dossier: Record<string, any>,
    proposal: NonNullable<ChatResponse["proposed_meal_plan"]>,
    session_id: string | null,
  ) =>
    request<ChatAcceptResponse>("/chat/accept", {
      dossier,
      proposal,
      session_id,
    }),
  rollback: (session_id: string, target_version: number) =>
    request<ChatAcceptResponse>("/chat/rollback", {
      session_id,
      target_version,
    }),
};
