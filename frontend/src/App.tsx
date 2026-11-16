import React, { useState, useEffect } from 'react'
import {
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  HeartPulse,
  Sliders,
  Utensils,
  FileText,
  Activity,
  AlertCircle,
  FileUp,
  Cpu,
  Zap,
} from 'lucide-react'

import { Navbar } from './components/Navbar'
import { HardwareStatusBar } from './components/HardwareStatusBar'
import { InputTypingForm } from './components/InputTypingForm'
import { TxtFileUpload } from './components/TxtFileUpload'
import { Stage1RiskCards } from './components/Stage1RiskCards'
import { Stage2Prescription } from './components/Stage2Prescription'
import { Stage3MealSchedule } from './components/Stage3MealSchedule'
import { Stage4ClinicalReport } from './components/Stage4ClinicalReport'

import {
  CANONICAL_SAMPLE_FEATURES,
  CANONICAL_SAMPLE_METADATA,
} from './data/featureDefinitions'
import { checkHealth, runFullPipeline, runFullPipelineFile } from './services/api'
import { FullPipelineResponse, HealthStatus, PatientMetadata } from './types/api'

export const App: React.FC = () => {
  // Input State
  const [features, setFeatures] = useState<number[]>([...CANONICAL_SAMPLE_FEATURES])
  const [metadata, setMetadata] = useState<PatientMetadata>({ ...CANONICAL_SAMPLE_METADATA })

  // Mode Selection: 'typing' (Option 1) vs 'upload' (Option 2)
  const [inputMode, setInputMode] = useState<'typing' | 'upload'>('typing')

  // Execution Backend for Stage 4: Local Qwen 2.5 3B NF4 GPU
  const stage4Backend = 'transformers' as const

  // UI Flow State
  const [activeTab, setActiveTab] = useState<'input' | 'results'>('input')
  const [resultsSection, setResultsSection] = useState<'risk' | 'prescription' | 'mealplan' | 'report'>('risk')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<FullPipelineResponse | null>(null)

  // Hardware Status State
  const [health, setHealth] = useState<HealthStatus | null>(null)

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
  }

  // Execution: Run Full 4-Stage Pipeline
  const handleExecutePipeline = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await runFullPipeline({
        features,
        metadata,
        backend: stage4Backend,
      })
      setResults(response)
      setActiveTab('results')
      setResultsSection('risk')
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
    try {
      const response = await runFullPipelineFile(file, stage4Backend, metadata)
      setResults(response)
      setActiveTab('results')
      setResultsSection('risk')
    } catch (err: any) {
      setError(err.message || 'File execution failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-slate-100 text-slate-900 font-sans">
      {/* Top Navigation */}
      <Navbar
        health={health}
        activeTab={activeTab}
        hasResults={!!results}
        onTabChange={setActiveTab}
        onReset={handleReset}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Error Notification Banner */}
        {error && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-start space-x-3 text-rose-800 text-xs shadow-sm">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-bold">Pipeline Execution Error</h4>
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
            {/* Input Mode Selector & Pipeline Configuration Bar */}
            <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm flex flex-wrap items-center justify-between gap-4">
              {/* Left: Input Mode Toggle (Typing vs TXT File) */}
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

              {/* Right: Stage 4 Backend Badge & Execute Button */}
              <div className="flex items-center space-x-3">
                {/* Stage 4 LLM Indicator */}
                <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-xl text-xs">
                  <Cpu className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-slate-500 font-medium">Stage 4:</span>
                  <span className="font-semibold text-slate-800">Qwen 2.5 3B NF4 (Local GPU)</span>
                </div>

                {/* Primary CTA: Run 4-Stage Clinical Pipeline */}
                <button
                  type="button"
                  disabled={loading}
                  onClick={handleExecutePipeline}
                  className="px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl text-xs flex items-center space-x-2 shadow-md shadow-emerald-600/20 disabled:opacity-50 transition transform active:scale-95"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Synthesizing Clinical Plan...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" />
                      <span>Execute 4-Stage Pipeline</span>
                    </>
                  )}
                </button>
              </div>
            </div>

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
              />
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: CLINICAL RESULTS VIEWER                                             */}
        {/* ========================================================================= */}
        {activeTab === 'results' && results && (
          <div className="space-y-6">
            {/* Results Section Sub-Navigation Tabs */}
            <div className="bg-white rounded-2xl border border-slate-200 p-2 shadow-sm flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center space-x-1.5 overflow-x-auto">
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
                  disabled={!results.meal_plan}
                  onClick={() => setResultsSection('mealplan')}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${
                    !results.meal_plan
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
                  disabled={!results.clinical_report}
                  onClick={() => setResultsSection('report')}
                  className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center space-x-2 transition ${
                    !results.clinical_report
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
            {resultsSection === 'risk' && <Stage1RiskCards screening={results.screening} />}

            {resultsSection === 'prescription' && (
              <Stage2Prescription screening={results.screening} />
            )}

            {resultsSection === 'mealplan' && results.meal_plan && (
              <Stage3MealSchedule mealPlan={results.meal_plan} />
            )}

            {resultsSection === 'report' && results.clinical_report && (
              <Stage4ClinicalReport report={results.clinical_report} />
            )}
          </div>
        )}
      </main>

      {/* Sticky Hardware & Status Bar */}
      <HardwareStatusBar
        health={health}
      />
    </div>
  )
}

export default App
