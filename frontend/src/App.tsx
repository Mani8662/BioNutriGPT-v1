import React, { useState, useEffect, useRef } from 'react'
import {
  Play,
  Sliders,
  Utensils,
  FileText,
  AlertCircle,
  FileUp,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Settings2,
} from 'lucide-react'

import { Navbar } from './components/Navbar'
import { HardwareStatusBar } from './components/HardwareStatusBar'
import { InputTypingForm } from './components/InputTypingForm'
import { TxtFileUpload } from './components/TxtFileUpload'
import { Stage1RiskCards } from './components/Stage1RiskCards'
import { Stage2Prescription } from './components/Stage2Prescription'
import { MealPlanIntakeChat } from './components/MealPlanIntakeChat'
import { Stage3MealSchedule } from './components/Stage3MealSchedule'
import { Stage4ClinicalReport } from './components/Stage4ClinicalReport'
import { Stage4PlanChat } from './components/Stage4PlanChat'

import {
  CANONICAL_SAMPLE_FEATURES,
  CANONICAL_SAMPLE_METADATA,
} from './data/featureDefinitions'
import {
  checkHealth,
  compilePlan,
  screenPatient,
  uploadTxtFile,
} from './services/api'
import {
  ChatAcceptResponse,
  ClinicalReport,
  HealthStatus,
  MealPlanResult,
  PatientMetadata,
  ScreeningResult,
} from './types/api'

export const App: React.FC = () => {
  // Input State
  const [features, setFeatures] = useState<number[]>([...CANONICAL_SAMPLE_FEATURES])
  const [metadata, setMetadata] = useState<PatientMetadata>({ ...CANONICAL_SAMPLE_METADATA })
  const [inputMode, setInputMode] = useState<'typing' | 'upload'>('typing')

  // Progressive Workflow State
  const [activeTab, setActiveTab] = useState<'input' | 'results'>('input')
  const [loadingScreening, setLoadingScreening] = useState<boolean>(false)
  const [loadingReport, setLoadingReport] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const [screening, setScreening] = useState<ScreeningResult | null>(null)
  const [showStage2Details, setShowStage2Details] = useState<boolean>(false)
  const [showIntakeChat, setShowIntakeChat] = useState<boolean>(false)
  const [mealPlan, setMealPlan] = useState<MealPlanResult | null>(null)
  const [clinicalReport, setClinicalReport] = useState<ClinicalReport | null>(null)
  const [chatSessionKey, setChatSessionKey] = useState<number>(0)

  const mealPlanSectionRef = useRef<HTMLDivElement>(null)
  const preferencesRef = useRef<HTMLDivElement>(null)

  // Hardware Status State
  const [health, setHealth] = useState<HealthStatus | null>(null)

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

  // Input handlers
  const handleFeatureChange = (index: number, value: number) => {
    setFeatures(prev => {
      const copy = [...prev]
      copy[index] = value
      return copy
    })
  }

  const handleMetadataChange = (field: keyof PatientMetadata, value: any) => {
    setMetadata(prev => ({ ...prev, [field]: value }))
  }

  const handleLoadSample = () => {
    setFeatures([...CANONICAL_SAMPLE_FEATURES])
    setMetadata({ ...CANONICAL_SAMPLE_METADATA })
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
    setScreening(null)
    setMealPlan(null)
    setClinicalReport(null)
    setShowIntakeChat(false)
    setShowStage2Details(false)
    setChatSessionKey(prev => prev + 1)
    setError(null)
    setActiveTab('input')
  }

  const handleParsedFromTxt = (newFeatures: number[], newMeta: Partial<PatientMetadata>) => {
    setFeatures(newFeatures)
    setMetadata(prev => ({ ...prev, ...newMeta }))
    setError(null)
  }

  // Step 1: Run Disease Screening (Stages 1 + 2) -> Show disease scores first
  const handleRunScreening = async (overrideFeatures?: number[], overrideMeta?: Partial<PatientMetadata>) => {
    setLoadingScreening(true)
    setError(null)
    try {
      const scr = await screenPatient({
        features: overrideFeatures ?? features,
        metadata: overrideMeta ?? metadata,
      })
      setScreening(scr)
      setMealPlan(null)
      setClinicalReport(null)
      setShowIntakeChat(false)
      setShowStage2Details(false)
      setChatSessionKey(prev => prev + 1)
      setActiveTab('results')
    } catch (err: any) {
      setError(err.message || 'Screening failed. Ensure FastAPI backend is running on :8000.')
    } finally {
      setLoadingScreening(false)
    }
  }

  // File upload direct execution -> parse file then immediately run screening
  const handleRunFileDirectly = async (file: File) => {
    setLoadingScreening(true)
    setError(null)
    try {
      const parsed = await uploadTxtFile(file)
      const mergedMeta = { ...metadata, ...parsed.metadata }
      setFeatures(parsed.features)
      setMetadata(mergedMeta)
      await handleRunScreening(parsed.features, mergedMeta)
    } catch (err: any) {
      setError(err.message || 'File screening failed.')
      setLoadingScreening(false)
    }
  }

  // Step 2: Open conversational preference intake chat on the same page
  const handleOpenIntakeChat = () => {
    setShowIntakeChat(true)
    setTimeout(() => {
      preferencesRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 50)
  }

  // Step 3: Intake chat understood preferences and Stage 3 generated the meal plan
  const handlePlanGeneratedByIntake = (plan: MealPlanResult) => {
    setMealPlan(plan)
    setClinicalReport(null)
    setChatSessionKey(prev => prev + 1)
    setTimeout(() => {
      mealPlanSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 80)
  }

  // Step 4: Chatbot updated / rolled back meal plan -> Immediately refresh plan above
  const handlePlanUpdatedByChat = (accepted: ChatAcceptResponse) => {
    setMealPlan(accepted.meal_plan)
    setClinicalReport(null)
  }

  // Optional: Compile Stage 4 Qwen printable report on demand
  const handleGenerateReport = async () => {
    if (!mealPlan?.dossier) return
    setLoadingReport(true)
    setError(null)
    try {
      const report = await compilePlan(mealPlan.dossier)
      setClinicalReport(report)
    } catch (err: any) {
      setError(err.message || 'Could not generate Stage 4 printable report.')
    } finally {
      setLoadingReport(false)
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-slate-100 text-slate-900 font-sans">
      {/* Top Navigation */}
      <Navbar
        health={health}
        activeTab={activeTab}
        hasResults={!!screening}
        onTabChange={setActiveTab}
        onReset={handleReset}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Global Error Notification Banner */}
        {error && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-start space-x-3 text-rose-800 text-xs shadow-sm">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-bold">Execution Error</h4>
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
        {/* TAB 1: PATIENT INTAKE                                                     */}
        {/* ========================================================================= */}
        {activeTab === 'input' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider mr-1">
                  Input Mode:
                </span>
                <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200">
                  <button
                    type="button"
                    onClick={() => setInputMode('typing')}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                      inputMode === 'typing'
                        ? 'bg-white text-slate-900 shadow-sm border border-slate-200'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <Sliders className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Option 1: Form Typing</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setInputMode('upload')}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                      inputMode === 'upload'
                        ? 'bg-white text-slate-900 shadow-sm border border-slate-200'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <FileUp className="w-3.5 h-3.5 text-blue-600" />
                    <span>Option 2: TXT File Upload</span>
                  </button>
                </div>
              </div>

              <button
                type="button"
                disabled={loadingScreening}
                onClick={() => handleRunScreening()}
                className="px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl text-xs flex items-center space-x-2 shadow-md shadow-emerald-600/20 disabled:opacity-50 transition transform active:scale-95"
              >
                {loadingScreening ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Calculating Disease Risk Scores…</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" />
                    <span>Get Disease Risk Scores</span>
                  </>
                )}
              </button>
            </div>

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
              />
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: UNIFIED SINGLE-PAGE RESULTS FLOW                                   */}
        {/*   1) Disease Risk Scores                                                  */}
        {/*   2) "Generate Meal Plan" -> Preferences Questionnaire                    */}
        {/*   3) Generated Meal Plan                                                  */}
        {/*   4) Chatbot Below Meal Plan (Auto-Reconfigures Plan on Removal)          */}
        {/* ========================================================================= */}
        {activeTab === 'results' && screening && (
          <div className="space-y-6">
            {/* Top Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h1 className="text-lg font-bold text-slate-900">
                  Cardiometabolic Disease Screening Scores
                </h1>
                <p className="text-xs text-slate-500">
                  Review your screening scores below, then click Generate Meal Plan to personalize and build your daily plan.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setActiveTab('input')}
                className="px-3.5 py-2 text-xs text-slate-700 hover:text-slate-900 font-semibold rounded-xl bg-white border border-slate-200 hover:bg-slate-50 shadow-2xs"
              >
                ← Back to Patient Intake
              </button>
            </div>

            {/* STEP 1: DISEASE SCREENING SCORES */}
            <Stage1RiskCards screening={screening} />

            {/* Compact Collapsible Stage 2 Nutrient Targets & Constraints */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-2xs overflow-hidden">
              <button
                type="button"
                onClick={() => setShowStage2Details(prev => !prev)}
                className="w-full px-5 py-3.5 flex items-center justify-between text-xs font-bold text-slate-700 hover:bg-slate-50 transition"
              >
                <span className="flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-emerald-600" />
                  <span>
                    Stage 2 Nutrient Targets &amp; Explicit Rules (
                    {Math.round(screening.energy_prescription?.target_calories ?? 0)} kcal/day ·{' '}
                    {screening.certified_for_stage3 ? 'Checks Passed' : 'Checks Incomplete'})
                  </span>
                </span>
                {showStage2Details ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              {showStage2Details && (
                <div className="p-5 pt-2 border-t border-slate-100">
                  <Stage2Prescription screening={screening} />
                </div>
              )}
            </div>

            {/* STEP 2: GENERATE MEAL PLAN TRIGGER & CONVERSATIONAL FREE-TEXT INTAKE */}
            <div ref={preferencesRef} className="space-y-4">
              {!showIntakeChat && !mealPlan && (
                <div className="bg-white rounded-2xl border border-emerald-200 p-5 shadow-sm flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200">
                      <Utensils className="w-5 h-5" />
                    </div>
                    <div>
                      <h2 className="text-base font-bold text-slate-900">Ready for a personalized meal plan?</h2>
                      <p className="text-xs text-slate-600 mt-0.5">
                        Click below and the assistant will ask you about pure vegetarian / veg+non-veg combo / purely non-veg preferences, allergies, and dishes to remove.
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    disabled={!screening.certified_for_stage3}
                    onClick={handleOpenIntakeChat}
                    className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-bold flex items-center space-x-2 shadow-md shadow-emerald-600/20 disabled:opacity-50 transition"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>Generate Meal Plan</span>
                  </button>
                </div>
              )}

              {showIntakeChat && (
                <MealPlanIntakeChat
                  dossier={screening.dossier}
                  onPlanGenerated={handlePlanGeneratedByIntake}
                  onClose={mealPlan ? () => setShowIntakeChat(false) : undefined}
                />
              )}
            </div>

            {/* STEP 3: GENERATED MEAL PLAN (ON THE SAME PAGE) */}
            {mealPlan && (
              <div ref={mealPlanSectionRef} className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                    <Utensils className="w-4 h-4 text-emerald-600" />
                    <span>Your Generated Meal Plan</span>
                  </h2>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={handleOpenIntakeChat}
                      className="px-3 py-1.5 rounded-xl bg-white border border-slate-300 hover:bg-slate-50 text-xs font-semibold text-slate-700 flex items-center space-x-1.5"
                    >
                      <Settings2 className="w-3.5 h-3.5" />
                      <span>Change Preferences via Chat</span>
                    </button>
                    {mealPlan.accepted && (
                      <button
                        type="button"
                        disabled={loadingReport}
                        onClick={handleGenerateReport}
                        className="px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold flex items-center space-x-1.5 disabled:opacity-50"
                      >
                        <FileText className="w-3.5 h-3.5" />
                        <span>{loadingReport ? 'Generating Printable Report…' : 'Generate Printable Qwen Report'}</span>
                      </button>
                    )}
                  </div>
                </div>

                <Stage3MealSchedule mealPlan={mealPlan} />

                {/* STEP 4: CHATBOT DIRECTLY BELOW THE MEAL PLAN */}
                {mealPlan.accepted && mealPlan.dossier && (
                  <Stage4PlanChat
                    key={chatSessionKey}
                    dossier={mealPlan.dossier}
                    onAccept={handlePlanUpdatedByChat}
                  />
                )}

                {/* Optional Printable Stage 4 Qwen Report */}
                {clinicalReport && <Stage4ClinicalReport report={clinicalReport} />}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Sticky Hardware & Status Bar */}
      <HardwareStatusBar health={health} />
    </div>
  )
}

export default App
