import React, { useState, useEffect, useRef } from 'react'
import {
  Play,
  RotateCcw,
  Layers,
  HeartPulse,
  Sliders,
  Utensils,
  FileText,
  Activity,
  AlertCircle,
  FileUp,
} from 'lucide-react'

import { Navbar } from './components/Navbar'
import { InputTypingForm } from './components/InputTypingForm'
import { TxtFileUpload } from './components/TxtFileUpload'
import { Stage1RiskCards } from './components/Stage1RiskCards'
import { Stage2Prescription } from './components/Stage2Prescription'
import { Stage3MealSchedule } from './components/Stage3MealSchedule'
import { Stage4ClinicalReport } from './components/Stage4ClinicalReport'
import { NutritionChat } from './components/NutritionChat'
import { MealPlanIntakeChat } from './components/MealPlanIntakeChat'
import { Stage4PlanChat } from './components/Stage4PlanChat'
import { AccountAccessModal } from './components/AccountAccessModal'
import { RiskHistoryDashboard } from './components/RiskHistoryDashboard'
import { AccountDashboard } from './components/AccountDashboard'
import { AboutPage } from './components/AboutPage'
import { AdminDashboard } from './components/AdminDashboard'
import { AppSection } from './components/Navbar'

import {
  CANONICAL_SAMPLE_FEATURES,
  CANONICAL_SAMPLE_METADATA,
} from './data/featureDefinitions'
import { checkHealth, compilePlan, deleteRiskHistoryEntry, getCurrentAccount, getRiskHistory, logoutAccount, runFullPipeline, runFullPipelineFile, saveRiskHistory } from './services/api'
import { AccountUser, ChatAcceptResponse, DocumentPipelineResponse, FullPipelineResponse, HealthStatus, MealPlanResult, PatientMetadata, RiskHistoryEntry, ScreeningResult } from './types/api'

export const App: React.FC = () => {
  // Input State
  const [features, setFeatures] = useState<number[]>([...CANONICAL_SAMPLE_FEATURES])
  const [metadata, setMetadata] = useState<PatientMetadata>({ ...CANONICAL_SAMPLE_METADATA })

  // Input modes: structured form or file upload.
  const [inputMode, setInputMode] = useState<'typing' | 'upload'>('typing')
  const [fileInputReady, setFileInputReady] = useState(false)

  // Ollama keeps local inference working on machines without a CUDA GPU.
  const stage4Backend = 'ollama' as const

  // UI Flow State
  const [activeTab, setActiveTab] = useState<AppSection>('input')
  const [resultsSection, setResultsSection] = useState<'overview' | 'risk' | 'prescription' | 'mealplan' | 'report'>('risk')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<FullPipelineResponse | null>(null)
  const [documentAnalysis, setDocumentAnalysis] = useState<DocumentPipelineResponse | null>(null)
  const [showPreferenceChat, setShowPreferenceChat] = useState(false)
  const [personalizedPlan, setPersonalizedPlan] = useState<MealPlanResult | null>(null)
  const [planChatKey, setPlanChatKey] = useState(0)
  const [personalizedReport, setPersonalizedReport] = useState<import('./types/api').ClinicalReport | null>(null)
  const [loadingPersonalizedReport, setLoadingPersonalizedReport] = useState(false)
  const [personalizedReportError, setPersonalizedReportError] = useState<string | null>(null)
  const [account, setAccount] = useState<AccountUser | null>(null)
  const [accountModalOpen, setAccountModalOpen] = useState(false)
  const [riskHistory, setRiskHistory] = useState<RiskHistoryEntry[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const stage4RunId = useRef(0)

  const compilePersonalizedReport = async (plan: MealPlanResult | null, openReport = false) => {
    const runId = ++stage4RunId.current
    if (!plan?.accepted || !plan.dossier) {
      setLoadingPersonalizedReport(false)
      return
    }
    setLoadingPersonalizedReport(true)
    setPersonalizedReport(null)
    setPersonalizedReportError(null)
    try {
      const report = await compilePlan(plan.dossier)
      if (stage4RunId.current === runId) {
        setPersonalizedReport(report)
        if (openReport) setResultsSection('report')
      }
    } catch (err) {
      if (stage4RunId.current === runId) {
        setPersonalizedReportError(err instanceof Error ? err.message : 'Stage 4 report generation failed.')
      }
    } finally {
      if (stage4RunId.current === runId) setLoadingPersonalizedReport(false)
    }
  }

  // Hardware Status State
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [chatOpen, setChatOpen] = useState(false)
  // Fetch health status on mount and interval
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const h = await checkHealth()
        setHealth(h)
      } catch (e) {
        console.warn('Backend server not reachable yet on :8000', e)
      }
    }
    fetchHealth()
    const timer = setInterval(fetchHealth, 10000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    void getCurrentAccount().then((current) => {
      setAccount(current)
      if (current) void getRiskHistory().then(setRiskHistory).catch((err) => console.warn('Could not load risk history', err))
    }).catch((err) => console.warn('Could not check account session', err))
  }, [])

  const recordRiskAssessment = (screening: ScreeningResult, source: RiskHistoryEntry['source']) => {
    if (!account) return
    void saveRiskHistory(screening, source).then((entry) => {
      setRiskHistory((previous) => [...previous, entry].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at)).slice(-1000))
    }).catch((err) => setError(err instanceof Error ? `Could not save assessment history: ${err.message}` : 'Could not save assessment history.'))
  }

  const handleAccountAuthenticated = async (authenticated: AccountUser) => {
    setAccount(authenticated)
    if (results?.screening) {
      try {
        await saveRiskHistory(results.screening, documentAnalysis ? 'document' : 'form')
        setRiskHistory(await getRiskHistory())
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not load account history.')
      }
    } else {
      setRiskHistory(await getRiskHistory().catch(() => []))
    }
  }

  const handleOpenHistory = async () => {
    setActiveTab('history')
    setHistoryLoading(true)
    try {
      setRiskHistory(await getRiskHistory())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load risk history.')
    } finally {
      setHistoryLoading(false)
    }
  }

  const handleDeleteRiskHistory = async (id: string) => {
    try {
      await deleteRiskHistoryEntry(id)
      setRiskHistory((previous) => previous.filter((entry) => entry.id !== id))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete this history entry.')
    }
  }

  const handleLogout = async () => {
    await logoutAccount().catch(() => undefined)
    setAccount(null)
    setRiskHistory([])
    setActiveTab('input')
  }

  // Input change handlers
  const handleFeatureChange = (index: number, value: number) => {
    setFeatures((prev) => {
      const copy = [...prev]
      copy[index] = value
      return copy
    })
  }

  const handleMetadataChange = (field: keyof PatientMetadata, value: any) => {
    setMetadata((prev) => ({ ...prev, [field]: value }))
  }

  const handleLoadSample = () => {
    setFeatures([...CANONICAL_SAMPLE_FEATURES])
    setMetadata({ ...CANONICAL_SAMPLE_METADATA })
    setDocumentAnalysis(null)
    setError(null)
  }

  const handleReset = () => {
    setFeatures(new Array(87).fill(0))
    setMetadata({
      weight_kg: 70.0,
      height_cm: 170.0,
      age_years: 40.0,
      is_female: false,
      activity_level: 'sedentary',
    })
    setResults(null)
    setPersonalizedPlan(null)
    setPersonalizedReport(null)
    setShowPreferenceChat(false)
    setDocumentAnalysis(null)
    setError(null)
    setActiveTab('input')
  }

  const handleParsedFromTxt = (newFeatures: number[], newMeta: Partial<PatientMetadata>) => {
    setFeatures(newFeatures)
    setMetadata((prev) => ({
      ...prev,
      ...newMeta,
    }))
    setError(null)
    setDocumentAnalysis(null)
    setFileInputReady(true)
  }

  // Execution: Run Full 4-Stage Pipeline
  const handleExecutePipeline = async () => {
    setLoading(true)
    setError(null)
    setDocumentAnalysis(null)
    try {
      const response = await runFullPipeline({
        features,
        metadata,
        backend: stage4Backend,
      })
      setResults(response)
      setPersonalizedPlan(response.meal_plan)
      setPersonalizedReport(null)
      setShowPreferenceChat(false)
      setActiveTab('results')
      setResultsSection('overview')
      recordRiskAssessment(response.screening, 'form')
      // Start the slow local-model report while the user reviews the plan.
      void compilePersonalizedReport(response.meal_plan)
    } catch (err: any) {
      setError(err.message || 'Execution failed. Ensure FastAPI backend is running on :8000.')
    } finally {
      setLoading(false)
    }
  }

  // Execution from direct file upload
  const handleRunFileDirectly = async (file: File) => {
    setLoading(true)
    setError(null)
    setDocumentAnalysis(null)
    try {
      const response = await runFullPipelineFile(file, stage4Backend, metadata)
      setResults(response)
      setPersonalizedPlan(response.meal_plan)
      setPersonalizedReport(null)
      setShowPreferenceChat(false)
      setActiveTab('results')
      setResultsSection('risk')
      recordRiskAssessment(response.screening, 'file')
      void compilePersonalizedReport(response.meal_plan)
    } catch (err: any) {
      setError(err.message || 'File execution failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleDocumentPipelineComplete = (response: DocumentPipelineResponse) => {
    if (!response.model_output) return
    setDocumentAnalysis(response)
    setResults(response.model_output)
    setPersonalizedPlan(response.model_output.meal_plan)
    setPersonalizedReport(null)
    setShowPreferenceChat(false)
    setActiveTab('results')
    setResultsSection('risk')
    recordRiskAssessment(response.model_output.screening, 'document')
    void compilePersonalizedReport(response.model_output.meal_plan)
  }

  const handlePreferencePlanGenerated = (plan: MealPlanResult) => {
    setPersonalizedPlan(plan)
    setPlanChatKey(value => value + 1)
    setResultsSection('mealplan')
    setShowPreferenceChat(false)
    void compilePersonalizedReport(plan)
  }

  const handlePlanAccepted = (accepted: ChatAcceptResponse) => {
    setPersonalizedPlan(accepted.meal_plan)
    void compilePersonalizedReport(accepted.meal_plan)
  }

  const handleCompilePersonalizedPlan = async () => {
    if (!personalizedPlan?.dossier || !personalizedPlan.accepted) return
    setError(null)
    await compilePersonalizedReport(personalizedPlan, true)
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-100 font-sans text-slate-900 md:flex-row">
      <Navbar
        health={health}
        activeTab={activeTab}
        hasResults={!!results}
        account={account}
        onTabChange={(section) => section === 'history' ? void handleOpenHistory() : setActiveTab(section)}
        onAccountClick={() => setAccountModalOpen(true)}
        onLogout={() => void handleLogout()}
        onReset={handleReset}
      />

      <div className="min-w-0 flex-1">
      <main className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
        {/* Error Notification Banner */}
        {error && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-start space-x-3 text-rose-800 text-xs shadow-sm">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-bold">Could not complete that action</h4>
              <p className="mt-0.5 text-rose-700 leading-relaxed">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-xs text-rose-500 hover:text-rose-700 font-semibold px-2 py-1"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 1: PATIENT INTAKE (TWO INPUT OPTIONS)                                  */}
        {/* ========================================================================= */}
        {activeTab === 'input' && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h1 className="text-xl font-bold text-slate-900">Start your assessment</h1>
              <p className="mt-1 text-sm text-slate-600">Choose how you want to provide patient information.</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <button type="button" aria-pressed={inputMode === 'typing'} onClick={() => { setInputMode('typing'); setFileInputReady(false) }} className={`rounded-xl border p-4 text-left transition ${inputMode === 'typing' ? 'border-emerald-600 bg-emerald-50 ring-2 ring-emerald-100' : 'border-slate-200 hover:border-emerald-300 hover:bg-slate-50'}`}>
                  <span className="flex items-center gap-2 text-sm font-bold text-slate-900"><Sliders className="h-4 w-4 text-emerald-700" />Enter manually</span>
                  <span className="mt-1 block text-xs text-slate-600">Fill in the patient details and measurements.</span>
                </button>
                <button type="button" aria-pressed={inputMode === 'upload'} onClick={() => { setInputMode('upload'); setFileInputReady(false) }} className={`rounded-xl border p-4 text-left transition ${inputMode === 'upload' ? 'border-blue-600 bg-blue-50 ring-2 ring-blue-100' : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50'}`}>
                  <span className="flex items-center gap-2 text-sm font-bold text-slate-900"><FileUp className="h-4 w-4 text-blue-700" />Upload a file</span>
                  <span className="mt-1 block text-xs text-slate-600">Use a health document or a patient data file.</span>
                </button>
              </div>
            </section>

            {/* Input Component Selection */}
            {inputMode === 'typing' ? (
              <InputTypingForm
                features={features}
                metadata={metadata}
                onFeatureChange={handleFeatureChange}
                onMetadataChange={handleMetadataChange}
                onLoadSample={handleLoadSample}
                onReset={handleReset}
              />
            ) : (
              <TxtFileUpload
                onParsedSuccess={handleParsedFromTxt}
                onRunFileDirectly={handleRunFileDirectly}
                onPipelineComplete={handleDocumentPipelineComplete}
              />
            )}
            {(inputMode === 'typing' || fileInputReady) && (
              <div className="flex justify-end rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <button type="button" disabled={loading} onClick={() => void handleExecutePipeline()} className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-700 to-teal-700 px-5 py-3 text-sm font-bold text-white shadow-md transition hover:from-emerald-600 hover:to-teal-600 disabled:cursor-wait disabled:opacity-60">
                  {loading ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />Preparing your stages…</> : <><Play className="h-4 w-4 fill-white" />Continue to stages</>}
                </button>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: CLINICAL RESULTS VIEWER                                             */}
        {/* ========================================================================= */}
        {activeTab === 'results' && results && (
          <div className="space-y-6">
            {/* Results Section Sub-Navigation Tabs */}
            {results.screening.dossier && (
              <section className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="font-bold text-emerald-950">Personalize your meal plan</h2>
                    <p className="mt-1 text-sm text-emerald-900">Set your diet style, allergies, and foods to avoid. Stage 3 will re-optimize the plan; Stage 4 lets you request safe changes.</p>
                  </div>
                  <button type="button" onClick={() => setShowPreferenceChat(value => !value)} className="rounded-xl bg-emerald-800 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700">
                    {showPreferenceChat ? 'Close preferences' : 'Set meal preferences'}
                  </button>
                </div>
                {showPreferenceChat && <div className="mt-4"><MealPlanIntakeChat dossier={personalizedPlan?.accepted ? personalizedPlan.dossier : results.screening.dossier} onPlanGenerated={handlePreferencePlanGenerated} /></div>}
              </section>
            )}

            <div className="bg-white rounded-2xl border border-slate-200 p-2 shadow-sm flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center space-x-1.5 overflow-x-auto">
                {documentAnalysis && (
                  <button
                    onClick={() => setResultsSection('overview')}
                    className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${resultsSection === 'overview' ? 'bg-slate-900 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}`}
                  >
                    <Layers className="w-4 h-4 text-emerald-400" />
                    <span>Uploaded Report: All Results</span>
                  </button>
                )}
                <button
                  onClick={() => setResultsSection('risk')}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${
                    resultsSection === 'risk'
                      ? 'bg-slate-900 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <HeartPulse className="w-4 h-4 text-rose-500" />
                  <span>Stage 1: Risk Gauges</span>
                </button>

                <button
                  onClick={() => setResultsSection('prescription')}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${
                    resultsSection === 'prescription'
                      ? 'bg-slate-900 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <Sliders className="w-4 h-4 text-emerald-400" />
                  <span>Stage 2: Nutrient Envelope</span>
                </button>

                <button
                  disabled={!personalizedPlan && !results.meal_plan}
                  onClick={() => setResultsSection('mealplan')}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${
                    !personalizedPlan && !results.meal_plan
                      ? 'text-slate-400 cursor-not-allowed opacity-50'
                      : resultsSection === 'mealplan'
                      ? 'bg-slate-900 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <Utensils className="w-4 h-4 text-amber-500" />
                  <span>Stage 3: Indian Meal Plan</span>
                </button>

                <button
                  disabled={!personalizedReport && !results.clinical_report && !personalizedPlan?.accepted}
                  onClick={() => setResultsSection('report')}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${
                    !personalizedReport && !results.clinical_report && !personalizedPlan?.accepted
                      ? 'text-slate-400 cursor-not-allowed opacity-50'
                      : resultsSection === 'report'
                      ? 'bg-slate-900 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <FileText className="w-4 h-4 text-blue-500" />
                  <span>Stage 4: Patient Report</span>
                </button>
              </div>

              {/* Back to Intake Button */}
              <button
                onClick={() => setActiveTab('input')}
                className="px-3 py-1.5 text-xs text-slate-600 hover:text-slate-900 font-semibold rounded-lg hover:bg-slate-100"
              >
                ← Back to Patient Intake
              </button>
            </div>

            {/* Results Section Content */}
            {resultsSection === 'overview' && documentAnalysis && (
              <div className="space-y-6">
                <Stage1RiskCards screening={results.screening} />
                <Stage2Prescription screening={results.screening} />
                {results.meal_plan && <Stage3MealSchedule mealPlan={results.meal_plan} />}
                {results.clinical_report && <Stage4ClinicalReport report={results.clinical_report} />}
                {!results.clinical_report && !results.meal_plan?.accepted && <p className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">The predictor returned its screening output, but its safety gate did not produce a personalized meal plan for this input.</p>}
                {!results.clinical_report && results.meal_plan?.accepted && <p className="rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm text-blue-900">Your checked meal plan is ready. The Stage 4 patient report is generating in the background.</p>}
              </div>
            )}
            {resultsSection === 'risk' && <Stage1RiskCards screening={results.screening} />}

            {resultsSection === 'prescription' && (
              <Stage2Prescription screening={results.screening} />
            )}

            {resultsSection === 'mealplan' && (personalizedPlan || results.meal_plan) && (
              <Stage3MealSchedule mealPlan={(personalizedPlan || results.meal_plan)!} />
            )}

            {personalizedPlan?.accepted && personalizedPlan.dossier && (
              <section className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-200 bg-blue-50 p-4">
                  <div>
                    <h2 className="font-bold text-blue-950">Stage 4: Personalized plan assistant</h2>
                    <p className="text-sm text-blue-900">The patient report generates in the background while you review this verified plan.</p>
                  </div>
                  <button type="button" disabled={loadingPersonalizedReport} onClick={handleCompilePersonalizedPlan} className="rounded-xl bg-blue-800 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">
                    {loadingPersonalizedReport ? 'Generating report…' : personalizedReport ? 'Regenerate Stage 4 report' : 'Generate Stage 4 report'}
                  </button>
                </div>
                <Stage4PlanChat key={planChatKey} dossier={personalizedPlan.dossier} onAccept={handlePlanAccepted} />
              </section>
            )}

            {resultsSection === 'report' && (personalizedReport || results.clinical_report) && (
              <Stage4ClinicalReport report={(personalizedReport || results.clinical_report)!} />
            )}
            {resultsSection === 'report' && !personalizedReport && !results.clinical_report && personalizedPlan?.accepted && (
              <div className="rounded-xl border border-blue-200 bg-blue-50 p-5 text-sm text-blue-950">
                <p className="font-semibold">{loadingPersonalizedReport ? 'Your meal plan is ready. The patient report is generating in the background.' : personalizedReportError ? 'The patient report could not be generated.' : 'Your meal plan is ready. The patient report has not been generated yet.'}</p>
                <p className="mt-1">{personalizedReportError || 'You can keep reviewing the plan; the report will appear here when it is ready.'}</p>
                <button type="button" disabled={loadingPersonalizedReport} onClick={handleCompilePersonalizedPlan} className="mt-3 rounded-xl bg-blue-800 px-4 py-2 font-semibold text-white hover:bg-blue-700 disabled:opacity-50">
                  {loadingPersonalizedReport ? 'Generating report…' : personalizedReportError ? 'Retry Stage 4 report' : 'Generate Stage 4 report'}
                </button>
              </div>
            )}
          </div>
        )}
        {activeTab === 'history' && account && (
          <RiskHistoryDashboard
            account={account}
            records={riskHistory}
            loading={historyLoading}
            onRefresh={() => void handleOpenHistory()}
            onDelete={(id) => void handleDeleteRiskHistory(id)}
          />
        )}
        {activeTab === 'account' && account && <AccountDashboard account={account} assessmentCount={riskHistory.length} onHistory={() => void handleOpenHistory()} onSignOut={() => void handleLogout()} />}
        {activeTab === 'about' && <AboutPage />}
        {activeTab === 'admin' && account?.role === 'admin' && <AdminDashboard health={health} />}
      </main>

      </div>

      {accountModalOpen && <AccountAccessModal onClose={() => setAccountModalOpen(false)} onAuthenticated={(authenticated) => void handleAccountAuthenticated(authenticated)} />}

      <button onClick={() => setChatOpen(value => !value)} className="fixed bottom-6 right-5 z-50 rounded-full bg-emerald-800 px-5 py-3 font-semibold text-white shadow-lg hover:bg-emerald-700" aria-expanded={chatOpen} aria-label="Open nutrition chat">{chatOpen ? 'Close chat' : 'Nutrition chat'}</button>
      {chatOpen && <NutritionChat onClose={() => setChatOpen(false)} />}
    </div>
  )
}

export default App
